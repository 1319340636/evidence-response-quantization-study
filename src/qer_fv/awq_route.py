"""Frozen AWQ route configuration and fail-closed artifact auditing."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Mapping, Sequence


AWQ_ARTIFACT_PROTOCOL = "hf-awq-artifact-v1-20260721"
AWQ_ARTIFACT_PACKAGE_VERSIONS = {
    "llmcompressor": "0.12.0",
    "compressed-tensors": "0.17.1",
    "transformers": "5.10.1",
    "torch": "2.12.0",
}
AWQ_TEXT_BACKBONE_IGNORE = (
    "lm_head",
    r"re:^(?!model\.language_model(?:\.|$)).*",
)
MINISTRAL3_8B_REPO_ID = "mistralai/Ministral-3-8B-Instruct-2512-BF16"
MINISTRAL3_8B_REVISION = "06cc81bfd6e45321d8fc8f816576c5b6ac67ec22"
OLMO3_7B_REPO_ID = "allenai/OLMo-3-7B-Instruct"
OLMO3_7B_REVISION = "6e5971d9eba42665f5bd5a0fcf047f299ce1dccc"


class AWQRouteError(RuntimeError):
    """Raised when an AWQ route or artifact drifts from D5."""


@contextmanager
def align_floating_buffers_during_init(
    model_class: type,
    *,
    dtype: Any,
):
    """Align floating buffer metadata while a distributed model is built.

    compressed-tensors 0.17.1 reconstructs non-primary-rank buffers from
    locally initialized metadata but does not broadcast their dtype.  Some
    multimodal model classes initialize non-persistent floating buffers in
    float32 even when the frozen source load uses float16, so the receiving
    rank computes a storage size that cannot bind to rank zero's shared
    storage.  Restrict the compatibility shim to construction of the concrete
    model class and always restore its original initializer afterward.
    """
    original_init = model_class.__init__

    @wraps(original_init)
    def aligned_init(instance: Any, *args: Any, **kwargs: Any) -> None:
        original_init(instance, *args, **kwargs)
        for module in instance.modules():
            for name, buffer in module._buffers.items():
                if buffer is not None and buffer.is_floating_point():
                    module._buffers[name] = buffer.to(dtype=dtype)

    model_class.__init__ = aligned_init
    try:
        yield
    finally:
        model_class.__init__ = original_init


def _offload_with_source_dtype(self: Any, tensor: Any) -> Any:
    """Pinned compressed-tensors DistributedCPUCache.offload dtype repair."""
    import torch
    import torch.distributed as dist
    from compressed_tensors.distributed import (
        get_source_rank,
        is_source_process,
    )
    from compressed_tensors.offload.cache.cpu import CPUCache
    from compressed_tensors.offload.utils import to_tensor

    if tensor is None:
        return None
    tensor = tensor.contiguous()

    if is_source_process():
        tensor = CPUCache.offload(self, tensor).share_memory_()
        handle, filename, nbytes = tensor.untyped_storage()._share_filename_cpu_()
        source_size = tuple(tensor.size())
        source_stride = tuple(tensor.stride())
        source_storage_offset = tensor.storage_offset()
        source_dtype = None
        storage_numel = math.prod(source_size)
        for candidate in (tensor.dtype, tensor.data.dtype):
            canonical_itemsize = torch.empty((), dtype=candidate).element_size()
            if storage_numel * canonical_itemsize == nbytes:
                source_dtype = candidate
                break
        if source_dtype is None:
            raise AWQRouteError(
                "distributed source tensor metadata cannot describe its "
                f"storage: tensor={tensor.dtype}/{tensor.element_size()}, "
                f"data={tensor.data.dtype}/{tensor.data.element_size()}, "
                f"numel={storage_numel}, storage={nbytes}"
            )
        broadcast_obj = [
            handle,
            filename,
            nbytes,
            str(source_dtype),
            source_size,
            source_stride,
            source_storage_offset,
        ]
    else:
        broadcast_obj = [None, None, None, None, None, None, None]

    dist.broadcast_object_list(broadcast_obj, src=get_source_rank())
    if not is_source_process():
        dtype_name = broadcast_obj[3]
        if not isinstance(dtype_name, str) or not dtype_name.startswith("torch."):
            raise AWQRouteError("distributed offload source dtype is invalid")
        source_dtype = getattr(torch, dtype_name.removeprefix("torch."), None)
        if not isinstance(source_dtype, torch.dtype):
            raise AWQRouteError(
                f"distributed offload source dtype is unsupported: {dtype_name}"
            )
        source_size = tuple(broadcast_obj[4])
        source_stride = tuple(broadcast_obj[5])
        source_storage_offset = int(broadcast_obj[6])
        receiving_tensor = torch.empty_strided(
            source_size,
            source_stride,
            dtype=source_dtype,
            device=self.offload_device,
        )
        with torch.no_grad():
            receiving_tensor.set_(
                torch.UntypedStorage._new_shared_filename_cpu(*broadcast_obj[:3]),
                storage_offset=source_storage_offset,
                size=source_size,
                stride=source_stride,
            )
        tensor = to_tensor(receiving_tensor, tensor)

    dist.barrier()
    return tensor


@contextmanager
def broadcast_source_dtype_during_distributed_cpu_offload(
    *,
    cache_class: type | None = None,
):
    """Make pinned distributed CPU offload bind storage using source dtype.

    compressed-tensors 0.17.1 broadcasts the shared-memory handle and byte
    count but omits dtype.  A meta rank can therefore calculate a different
    byte extent for the same buffer.  Patch only the model-load scope, retain
    the package's CPU-memory diagnostic wrapper in production, and restore the
    original method even when loading fails.
    """
    production_class = cache_class is None
    if production_class:
        from compressed_tensors.offload.cache.dist_cpu import (
            DistributedCPUCache,
        )
        from compressed_tensors.offload.cache.utils import catch_cpu_mem_error

        cache_class = DistributedCPUCache
        patched_offload = catch_cpu_mem_error(_offload_with_source_dtype)
    else:
        patched_offload = _offload_with_source_dtype

    original_offload = cache_class.offload
    cache_class.offload = patched_offload
    try:
        yield
    finally:
        cache_class.offload = original_offload


def normalize_chat_template_token_ids(value: Any) -> list[int]:
    """Normalize Transformers 4.x lists and 5.x BatchEncoding outputs."""
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in value
        )
    ):
        raise AWQRouteError("chat template did not return valid token IDs")
    return value


def gemma4_awq_mapping_specs(
    layer_count: int,
    *,
    kv_shared_layers: int,
) -> list[dict[str, Any]]:
    """Expand Gemma AWQ semantics into exact per-layer module names."""
    if (
        isinstance(layer_count, bool)
        or not isinstance(layer_count, int)
        or layer_count <= 0
    ):
        raise ValueError("Gemma 4 layer_count must be a positive integer")
    if (
        isinstance(kv_shared_layers, bool)
        or not isinstance(kv_shared_layers, int)
        or kv_shared_layers < 0
        or kv_shared_layers >= layer_count
    ):
        raise ValueError(
            "Gemma 4 kv_shared_layers must be an integer in [0, layer_count)"
        )

    mappings: list[dict[str, Any]] = []
    non_shared_layers = layer_count - kv_shared_layers
    for index in range(layer_count):
        layer = f"model.language_model.layers.{index}"
        attention_balance_layers = [f"{layer}.self_attn.q_proj"]
        if index < non_shared_layers:
            attention_balance_layers.extend(
                [
                    f"{layer}.self_attn.k_proj",
                    f"{layer}.self_attn.v_proj",
                ]
            )
        mappings.append(
            {
                "smooth_layer": f"{layer}.input_layernorm",
                "balance_layers": attention_balance_layers,
            }
        )
        if index < non_shared_layers:
            mappings.append(
                {
                    "smooth_layer": f"{layer}.self_attn.v_proj",
                    "balance_layers": [f"{layer}.self_attn.o_proj"],
                }
            )
        mappings.extend(
            [
                {
                    "smooth_layer": f"{layer}.pre_feedforward_layernorm",
                    "balance_layers": [
                        f"{layer}.mlp.gate_proj",
                        f"{layer}.mlp.up_proj",
                    ],
                },
                {
                    "smooth_layer": f"{layer}.mlp.up_proj",
                    "balance_layers": [f"{layer}.mlp.down_proj"],
                },
            ]
        )
    return mappings


@dataclass(frozen=True)
class AWQRouteConfig:
    protocol_version: str
    model_key: str
    source_path: Path
    output_path: Path
    source_repo_id: str
    source_revision: str | None
    algorithm: str = "AWQ"
    scheme: str = "W4A16_ASYM"
    bits: int = 4
    activation_bits: int = 16
    group_size: int = 128
    zero_point: bool = True
    targets: tuple[str, ...] = ("Linear",)
    ignore: tuple[str, ...] = AWQ_TEXT_BACKBONE_IGNORE
    calibration_source_role: str = "vitaminc_dev"
    calibration_count: int = 256
    batch_size: int = 1
    ddp_world_size: int = 2
    shuffle_calibration_samples: bool = False
    allow_truncation: bool = False

    def __post_init__(self) -> None:
        source = Path(self.source_path)
        output = Path(self.output_path)
        if source.suffix.lower() == ".gguf" or (
            source.exists() and source.is_file()
        ):
            raise AWQRouteError(
                "AWQ source must be original Hugging Face weights, not GGUF"
            )
        source_resolved = source.resolve(strict=False)
        output_resolved = output.resolve(strict=False)
        if (
            source_resolved == output_resolved
            or output_resolved.is_relative_to(source_resolved)
            or source_resolved.is_relative_to(output_resolved)
        ):
            raise AWQRouteError("AWQ source and output paths must not overlap")
        if self.source_revision is not None:
            _validate_revision(self.source_revision)

    @classmethod
    def qwen35_9b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str | None = None,
    ) -> "AWQRouteConfig":
        return cls(
            protocol_version=AWQ_ARTIFACT_PROTOCOL,
            model_key="qwen35_9b",
            source_path=Path(source_path),
            output_path=Path(output_path),
            source_repo_id="Qwen/Qwen3.5-9B",
            source_revision=source_revision,
        )

    @classmethod
    def gemma4_e4b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str,
    ) -> "AWQRouteConfig":
        _validate_revision(source_revision)
        return cls(
            protocol_version=AWQ_ARTIFACT_PROTOCOL,
            model_key="gemma4_e4b",
            source_path=Path(source_path),
            output_path=Path(output_path),
            source_repo_id="google/gemma-4-E4B-it",
            source_revision=source_revision,
        )

    @classmethod
    def ministral3_8b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str,
    ) -> "AWQRouteConfig":
        if source_revision != MINISTRAL3_8B_REVISION:
            raise AWQRouteError(
                "Ministral source revision must match the frozen commit"
            )
        return cls(
            protocol_version=AWQ_ARTIFACT_PROTOCOL,
            model_key="ministral3_8b",
            source_path=Path(source_path),
            output_path=Path(output_path),
            source_repo_id=MINISTRAL3_8B_REPO_ID,
            source_revision=source_revision,
        )

    @classmethod
    def olmo3_7b(
        cls,
        *,
        source_path: str | Path,
        output_path: str | Path,
        source_revision: str,
    ) -> "AWQRouteConfig":
        if source_revision != OLMO3_7B_REVISION:
            raise AWQRouteError("OLMo source revision must match the frozen commit")
        return cls(
            protocol_version=AWQ_ARTIFACT_PROTOCOL,
            model_key="olmo3_7b",
            source_path=Path(source_path),
            output_path=Path(output_path),
            source_repo_id=OLMO3_7B_REPO_ID,
            source_revision=source_revision,
            ignore=("lm_head",),
        )

    def recipe_record(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "scheme": self.scheme,
            "bits": self.bits,
            "activation_bits": self.activation_bits,
            "group_size": self.group_size,
            "zero_point": self.zero_point,
            "targets": list(self.targets),
            "ignore": list(self.ignore),
            "batch_size": self.batch_size,
            "shuffle_calibration_samples": self.shuffle_calibration_samples,
            "allow_truncation": self.allow_truncation,
        }

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["source_path"] = self.source_path.as_posix()
        record["output_path"] = self.output_path.as_posix()
        record["targets"] = list(self.targets)
        record["ignore"] = list(self.ignore)
        if self.source_revision is None:
            record.pop("source_revision")
        return record


def build_awq_artifact_audit(
    config: AWQRouteConfig,
    *,
    calibration_manifest: Mapping[str, Any],
    calibration_records_path: str | Path,
    calibration_texts_path: str | Path,
    calibration_token_ids_path: str | Path,
    resolved_mappings: Sequence[Mapping[str, Any]],
    package_versions: Mapping[str, str],
    package_lock_path: str | Path,
    wheel_sha256: Mapping[str, str],
    runtime_environment: Mapping[str, str],
    gpu_inventory: Sequence[Mapping[str, Any]],
    source_audit_path: str | Path | None = None,
    source_audit_sha256_path: str | Path | None = None,
    quantized_target_modules: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Bind source, recipe, calibration, environment, and packed output."""
    if not isinstance(config, AWQRouteConfig):
        raise AWQRouteError("config must be an AWQRouteConfig")
    normalized_versions = {
        str(name): str(version) for name, version in package_versions.items()
    }
    if normalized_versions != AWQ_ARTIFACT_PACKAGE_VERSIONS:
        raise AWQRouteError("AWQ artifact package versions drifted")
    source_files = _hash_tree(config.source_path, "source")
    output_files = _hash_tree(config.output_path, "output")
    output_config = _load_json_object(
        config.output_path / "config.json", "AWQ output config"
    )
    quantization_config = output_config.get("quantization_config")
    if not isinstance(quantization_config, Mapping):
        raise AWQRouteError("AWQ output quantization_config is missing")

    calibration = _validate_calibration(
        calibration_manifest=calibration_manifest,
        records_path=Path(calibration_records_path),
        texts_path=Path(calibration_texts_path),
        token_ids_path=Path(calibration_token_ids_path),
        expected_role=config.calibration_source_role,
    )
    mappings = _normalize_mappings(resolved_mappings)
    lock_path = Path(package_lock_path)
    if not lock_path.is_file():
        raise AWQRouteError("AWQ package lock is missing")
    wheels = _validate_hash_mapping(wheel_sha256, "wheel")
    environment = _normalize_nonempty_mapping(
        runtime_environment, "runtime environment"
    )
    gpus = [dict(item) for item in gpu_inventory]
    if len(gpus) != config.ddp_world_size or any(not item for item in gpus):
        raise AWQRouteError("AWQ GPU inventory does not match DDP world size")

    record: dict[str, Any] = {
        "protocol_version": config.protocol_version,
        "model_key": config.model_key,
        "source_repo_id": config.source_repo_id,
        "source_path": str(config.source_path.resolve()),
        "output_path": str(config.output_path.resolve()),
        "recipe": config.recipe_record(),
        "calibration": calibration,
        "resolved_mappings": mappings,
        "package_versions": normalized_versions,
        "package_lock_sha256": _file_sha256(lock_path),
        "wheel_sha256": wheels,
        "runtime_environment": environment,
        "gpu_inventory": gpus,
        "ddp_world_size": config.ddp_world_size,
        "source_files": source_files,
        "output_files": output_files,
        "output_quantization_config": dict(quantization_config),
    }
    if config.source_revision is not None:
        record["source_revision"] = config.source_revision
    if config.model_key == "ministral3_8b":
        if source_audit_path is None or not quantized_target_modules:
            raise AWQRouteError(
                "Ministral AWQ requires source audit and resolved target modules"
            )
        from .ministral_source_audit import verify_ministral_source_against_audit

        source_audit = verify_ministral_source_against_audit(
            config.source_path, source_audit_path
        )
        expected_source = {
            name: item["sha256"] for name, item in source_audit["files"].items()
        }
        if source_files != expected_source:
            raise AWQRouteError("Ministral AWQ source audit inventory drifted")
        targets = sorted(set(str(name) for name in quantized_target_modules))
        if any(not name.startswith("model.language_model.") for name in targets):
            raise AWQRouteError("Ministral AWQ target escaped the text backbone")
        try:
            from .ministral_preflight import validate_awq_mapping_scope

            validate_awq_mapping_scope(mappings, targets)
        except RuntimeError as error:
            raise AWQRouteError(str(error)) from error
        record["source_audit_sha256"] = source_audit["audit_sha256"]
        record["source_audit_file_sha256"] = _file_sha256(Path(source_audit_path))
        record["recipe"]["targets"] = targets
        record["quantized_target_modules"] = targets
    elif config.model_key == "olmo3_7b":
        if source_audit_path is None or source_audit_sha256_path is None:
            raise AWQRouteError("OLMo AWQ requires source audit and SHA-256")
        from .olmo3_source_audit import verify_olmo3_source_against_audit

        source_audit = verify_olmo3_source_against_audit(
            config.source_path,
            source_audit_path,
            sha256_path=source_audit_sha256_path,
        )
        expected_source = {
            name: item["sha256"] for name, item in source_audit["files"].items()
        }
        if source_files != expected_source:
            raise AWQRouteError("OLMo AWQ source audit inventory drifted")
        record["source_audit_sha256"] = source_audit["audit_sha256"]
        record["source_audit_file_sha256"] = _file_sha256(
            Path(source_audit_path)
        )
    record["audit_sha256"] = _record_hash(record)
    return record


def write_awq_artifact_audit(
    audit: Mapping[str, Any], path: str | Path
) -> None:
    normalized = dict(audit)
    _validate_loaded_audit(normalized)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def load_awq_artifact_audit(path: str | Path) -> dict[str, Any]:
    value = _load_json_object(Path(path), "AWQ artifact audit")
    _validate_loaded_audit(value)
    return value


def _validate_loaded_audit(value: Mapping[str, Any]) -> None:
    if value.get("protocol_version") != AWQ_ARTIFACT_PROTOCOL:
        raise AWQRouteError("AWQ artifact audit protocol drifted")
    supplied = value.get("audit_sha256")
    if supplied != _record_hash(value):
        raise AWQRouteError("AWQ artifact audit hash mismatch")
    if value.get("model_key") not in {
        "qwen35_9b",
        "gemma4_e4b",
        "ministral3_8b",
        "olmo3_7b",
    }:
        raise AWQRouteError("AWQ artifact audit model key drifted")
    recipe = value.get("recipe")
    if not isinstance(recipe, Mapping) or {
        "algorithm": recipe.get("algorithm"),
        "scheme": recipe.get("scheme"),
        "bits": recipe.get("bits"),
        "activation_bits": recipe.get("activation_bits"),
        "group_size": recipe.get("group_size"),
        "zero_point": recipe.get("zero_point"),
    } != {
        "algorithm": "AWQ",
        "scheme": "W4A16_ASYM",
        "bits": 4,
        "activation_bits": 16,
        "group_size": 128,
        "zero_point": True,
    }:
        raise AWQRouteError("AWQ artifact audit recipe drifted")
    expected_ignore = (
        ("lm_head",)
        if value.get("model_key") == "olmo3_7b"
        else AWQ_TEXT_BACKBONE_IGNORE
    )
    if tuple(recipe.get("ignore", ())) != expected_ignore:
        raise AWQRouteError("AWQ artifact audit text-backbone scope drifted")
    if value.get("model_key") == "ministral3_8b":
        targets = value.get("quantized_target_modules")
        if (
            not isinstance(targets, list)
            or not targets
            or targets != sorted(set(targets))
            or recipe.get("targets") != targets
            or any(
                not isinstance(name, str)
                or not name.startswith("model.language_model.")
                for name in targets
            )
        ):
            raise AWQRouteError("Ministral AWQ exact target list drifted")
        try:
            from .ministral_preflight import validate_awq_mapping_scope

            validate_awq_mapping_scope(value.get("resolved_mappings", ()), targets)
        except RuntimeError as error:
            raise AWQRouteError(str(error)) from error


def _validate_calibration(
    *,
    calibration_manifest: Mapping[str, Any],
    records_path: Path,
    texts_path: Path,
    token_ids_path: Path,
    expected_role: str,
) -> dict[str, Any]:
    manifest = dict(calibration_manifest)
    if manifest.get("source_role") != expected_role:
        raise AWQRouteError("AWQ calibration source_role must be vitaminc_dev")
    if not records_path.is_file():
        raise AWQRouteError("AWQ calibration records are missing")
    if manifest.get("records_sha256") != _file_sha256(records_path):
        raise AWQRouteError("AWQ calibration records SHA-256 mismatch")
    rows = _load_jsonl(records_path, "calibration records")
    ids = [row.get("unique_id") for row in rows]
    if any(not isinstance(item, str) or not item for item in ids):
        raise AWQRouteError("AWQ calibration IDs are invalid")
    if len(set(ids)) != len(ids) or manifest.get("ids") != ids:
        raise AWQRouteError("AWQ calibration ID order drifted")
    if manifest.get("count") != len(ids):
        raise AWQRouteError("AWQ calibration count mismatch")
    ids_sha256 = hashlib.sha256(
        "".join(f"{item}\n" for item in ids).encode("utf-8")
    ).hexdigest()
    if manifest.get("ids_sha256") != ids_sha256:
        raise AWQRouteError("AWQ calibration ID SHA-256 mismatch")

    text_rows = _load_jsonl(texts_path, "rendered calibration texts")
    token_rows = _load_jsonl(token_ids_path, "calibration token IDs")
    if [row.get("unique_id") for row in text_rows] != ids:
        raise AWQRouteError("rendered calibration text IDs drifted")
    if [row.get("unique_id") for row in token_rows] != ids:
        raise AWQRouteError("calibration token ID membership drifted")
    if any(not isinstance(row.get("text"), str) or not row["text"] for row in text_rows):
        raise AWQRouteError("rendered calibration text is invalid")
    if any(
        not isinstance(row.get("input_ids"), list)
        or not row["input_ids"]
        or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in row["input_ids"]
        )
        for row in token_rows
    ):
        raise AWQRouteError("calibration token IDs are invalid")
    return {
        "source_role": expected_role,
        "count": len(ids),
        "ids_sha256": ids_sha256,
        "records_sha256": _file_sha256(records_path),
        "texts_sha256": _file_sha256(texts_path),
        "token_ids_sha256": _file_sha256(token_ids_path),
        "maximum_token_length": max(len(row["input_ids"]) for row in token_rows),
    }


def _normalize_mappings(
    values: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    mappings = [dict(value) for value in values]
    if not mappings or any(
        not isinstance(item.get("smooth_layer"), str)
        or not item["smooth_layer"]
        or not isinstance(item.get("balance_layers"), list)
        or not item["balance_layers"]
        for item in mappings
    ):
        raise AWQRouteError("resolved AWQ mappings are invalid")
    return mappings


def _normalize_nonempty_mapping(
    value: Mapping[str, str], label: str
) -> dict[str, str]:
    normalized = {str(key): str(child) for key, child in value.items()}
    if not normalized or any(not key or not child for key, child in normalized.items()):
        raise AWQRouteError(f"AWQ {label} is invalid")
    return normalized


def _validate_hash_mapping(
    value: Mapping[str, str], label: str
) -> dict[str, str]:
    normalized = {str(key): str(child) for key, child in value.items()}
    if not normalized:
        raise AWQRouteError(f"AWQ {label} hashes are missing")
    for name, digest in normalized.items():
        if not name or not _is_sha256(digest):
            raise AWQRouteError(f"AWQ {label} SHA-256 is invalid")
    return normalized


def _hash_tree(root: Path, label: str) -> dict[str, str]:
    root = Path(root)
    if not root.is_dir():
        raise AWQRouteError(f"AWQ {label} directory is missing")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise AWQRouteError(f"AWQ {label} directory is empty")
    return {
        path.relative_to(root).as_posix(): _file_sha256(path) for path in files
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AWQRouteError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise AWQRouteError(f"{label} must be a JSON object")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    try:
        values = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AWQRouteError(f"cannot read {label}") from error
    if not values or any(not isinstance(value, dict) for value in values):
        raise AWQRouteError(f"{label} must contain JSON objects")
    return values


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise AWQRouteError(f"cannot hash AWQ file: {path}") from error
    return digest.hexdigest()


def _record_hash(value: Mapping[str, Any]) -> str:
    clean = {key: child for key, child in value.items() if key != "audit_sha256"}
    try:
        payload = json.dumps(
            clean,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise AWQRouteError("AWQ artifact audit is not canonical JSON") from error
    return hashlib.sha256(payload).hexdigest()


def _validate_revision(value: object) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AWQRouteError("AWQ source revision must be an exact commit SHA")


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
