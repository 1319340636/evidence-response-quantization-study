"""Import-safe preflight contracts for Ministral artifact construction."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


GPTQ_DEPENDENCY_VERSIONS = {
    "gptqmodel": "7.1.0",
    "torch": "2.13.0",
    "transformers": "5.14.1",
}
AWQ_DEPENDENCY_VERSIONS = {
    "llmcompressor": "0.12.0",
    "compressed-tensors": "0.17.1",
    "transformers": "5.10.1",
    "torch": "2.12.0",
}
COMPATIBILITY_DEPENDENCY_VERSIONS = {
    **GPTQ_DEPENDENCY_VERSIONS,
    **AWQ_DEPENDENCY_VERSIONS,
}
COMPATIBILITY_GATE_PROTOCOL = (
    "ministral-quantizer-compatibility-gate-v1-20260822"
)
ROUTE_COMPATIBILITY_PROTOCOL = (
    "ministral-quantizer-route-compatibility-v1-20260822"
)
TEXT_BACKBONE_PREFIX = "model.language_model."
PUBLICATION_MARKER = ".ministral-publication.json"
PUBLICATION_PROTOCOL = "ministral-artifact-publication-v1-20260822"
DISPOSABLE_OWNER_PROTOCOL = "ministral-disposable-owner-v2-20260822"
DISPOSABLE_LEASE_PROTOCOL = "ministral-disposable-lease-v1-20260822"
DISPOSABLE_INITIALIZATION_GRACE_SECONDS = 30


class MinistralPreflightError(RuntimeError):
    """Raised before expensive model construction when a contract drifts."""


def validate_dependency_versions(
    expected: Mapping[str, str],
    *,
    version_getter: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, str]:
    """Require exact installed versions without importing any heavy package."""
    normalized = {str(name): str(version) for name, version in expected.items()}
    if not normalized:
        raise MinistralPreflightError("dependency version contract is empty")
    installed: dict[str, str] = {}
    for name, required in normalized.items():
        try:
            actual = str(version_getter(name))
        except Exception as error:
            raise MinistralPreflightError(
                f"required dependency is missing: {name}=={required}"
            ) from error
        if actual != required:
            raise MinistralPreflightError(
                f"required dependency mismatch: {name}=={required}; "
                f"installed {actual}"
            )
        installed[name] = actual
    return installed


def discover_text_linear_targets(model: Any) -> list[str]:
    """Return exact Linear module names strictly below the text backbone."""
    try:
        targets = sorted(
            name
            for name, module in model.named_modules()
            if isinstance(name, str)
            and name.startswith(TEXT_BACKBONE_PREFIX)
            and module.__class__.__name__ == "Linear"
        )
    except (AttributeError, TypeError) as error:
        raise MinistralPreflightError(
            "model does not expose a valid named_modules inventory"
        ) from error
    if not targets:
        raise MinistralPreflightError(
            "no Ministral text-backbone Linear modules were resolved"
        )
    if len(targets) != len(set(targets)):
        raise MinistralPreflightError("Ministral AWQ target names are duplicated")
    return targets


def validate_awq_mapping_scope(
    mappings: Sequence[Mapping[str, Any]],
    exact_targets: Sequence[str],
) -> list[dict[str, Any]]:
    """Reject every resolved AWQ mapping that escapes exact target scope."""
    targets = tuple(str(name) for name in exact_targets)
    target_set = set(targets)
    if not targets or len(target_set) != len(targets) or any(
        not name.startswith(TEXT_BACKBONE_PREFIX) for name in targets
    ):
        raise MinistralPreflightError("Ministral AWQ exact targets are invalid")
    if not mappings:
        raise MinistralPreflightError("resolved AWQ mappings must be nonempty")
    normalized: list[dict[str, Any]] = []
    for value in mappings:
        item = dict(value)
        if set(item) != {"smooth_layer", "balance_layers"}:
            raise MinistralPreflightError("resolved AWQ mapping schema drifted")
        smooth = item.get("smooth_layer")
        balances = item.get("balance_layers")
        if (
            not isinstance(smooth, str)
            or not smooth.startswith(TEXT_BACKBONE_PREFIX)
            or not isinstance(balances, list)
            or not balances
            or any(name not in target_set for name in balances)
        ):
            raise MinistralPreflightError(
                "resolved AWQ mapping escaped the Ministral text backbone"
            )
        normalized.append(
            {"smooth_layer": smooth, "balance_layers": list(balances)}
        )
    return normalized


def build_ministral_awq_mapping_records(
    exact_targets: Sequence[str],
) -> list[dict[str, Any]]:
    """Expand the default dense AWQ recipe into exact per-layer mappings."""
    targets = tuple(str(name) for name in exact_targets)
    target_set = set(targets)
    if not targets or len(target_set) != len(targets):
        raise MinistralPreflightError(
            "Ministral AWQ Linear inventory is invalid"
        )
    layer_prefix = f"{TEXT_BACKBONE_PREFIX}layers."
    suffixes = {
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.up_proj",
        "mlp.down_proj",
    }
    layer_indices: set[int] = set()
    for name in targets:
        if not name.startswith(layer_prefix):
            raise MinistralPreflightError(
                "Ministral AWQ Linear inventory escaped decoder layers"
            )
        remainder = name[len(layer_prefix) :]
        index_text, separator, suffix = remainder.partition(".")
        try:
            index = int(index_text)
        except ValueError as error:
            raise MinistralPreflightError(
                "Ministral AWQ Linear inventory has an invalid layer index"
            ) from error
        if (
            not separator
            or index < 0
            or str(index) != index_text
            or suffix not in suffixes
        ):
            raise MinistralPreflightError(
                "Ministral AWQ Linear inventory is not the frozen dense recipe"
            )
        layer_indices.add(index)
    ordered_indices = sorted(layer_indices)
    if ordered_indices != list(range(len(ordered_indices))):
        raise MinistralPreflightError(
            "Ministral AWQ Linear inventory has non-contiguous layers"
        )
    expected = {
        f"{layer_prefix}{index}.{suffix}"
        for index in ordered_indices
        for suffix in suffixes
    }
    if target_set != expected:
        raise MinistralPreflightError(
            "Ministral AWQ Linear inventory is incomplete"
        )

    records: list[dict[str, Any]] = []
    for index in ordered_indices:
        layer = f"{layer_prefix}{index}"
        records.extend(
            (
                {
                    "smooth_layer": f"{layer}.input_layernorm",
                    "balance_layers": [
                        f"{layer}.self_attn.q_proj",
                        f"{layer}.self_attn.k_proj",
                        f"{layer}.self_attn.v_proj",
                    ],
                },
                {
                    "smooth_layer": f"{layer}.self_attn.v_proj",
                    "balance_layers": [f"{layer}.self_attn.o_proj"],
                },
                {
                    "smooth_layer": f"{layer}.post_attention_layernorm",
                    "balance_layers": [
                        f"{layer}.mlp.gate_proj",
                        f"{layer}.mlp.up_proj",
                    ],
                },
                {
                    "smooth_layer": f"{layer}.mlp.up_proj",
                    "balance_layers": [f"{layer}.mlp.down_proj"],
                },
            )
        )
    return validate_awq_mapping_scope(records, targets)


_AWQ_COMPRESSED_TENSOR_SUFFIXES = (
    "weight_packed",
    "weight_scale",
    "weight_shape",
    "weight_zero_point",
)
_MINISTRAL_LEGACY_CHECKPOINT_PREFIXES = (
    "language_model.",
    "vision_tower.",
    "multi_modal_projector.",
)


def validate_ministral_awq_tensor_key_names(
    tensor_keys: Sequence[str], exact_targets: Sequence[str]
) -> list[str]:
    """Require canonical in-memory names for every compressed AWQ tensor."""
    keys = tuple(str(value) for value in tensor_keys)
    targets = tuple(str(value) for value in exact_targets)
    if not keys or len(keys) != len(set(keys)):
        raise MinistralPreflightError(
            "Ministral AWQ tensor-key inventory is empty or duplicated"
        )
    if not targets or len(targets) != len(set(targets)) or any(
        not target.startswith(TEXT_BACKBONE_PREFIX) for target in targets
    ):
        raise MinistralPreflightError("Ministral AWQ exact targets are invalid")
    if any(
        key.startswith(_MINISTRAL_LEGACY_CHECKPOINT_PREFIXES) for key in keys
    ):
        raise MinistralPreflightError(
            "Ministral AWQ checkpoint retained a legacy namespace"
        )

    key_set = set(keys)
    required = {
        f"{target}.{suffix}"
        for target in targets
        for suffix in _AWQ_COMPRESSED_TENSOR_SUFFIXES
    }
    if not required.issubset(key_set):
        raise MinistralPreflightError(
            "Ministral AWQ compressed tensor inventory is incomplete"
        )
    compressed_bases = {
        key[: -(len(suffix) + 1)]
        for key in keys
        for suffix in _AWQ_COMPRESSED_TENSOR_SUFFIXES
        if key.endswith(f".{suffix}")
    }
    if compressed_bases != set(targets):
        raise MinistralPreflightError(
            "Ministral AWQ compressed tensors escaped exact target scope"
        )
    return sorted(keys)


def validate_ministral_awq_checkpoint_directory(
    artifact_path: str | Path, exact_targets: Sequence[str]
) -> list[str]:
    """Read only safetensors metadata and validate canonical AWQ key names."""
    artifact = Path(artifact_path)
    files = sorted(artifact.glob("*.safetensors"))
    if not artifact.is_dir() or not files:
        raise MinistralPreflightError(
            "Ministral AWQ checkpoint has no safetensors files"
        )
    try:
        from safetensors import safe_open

        keys: list[str] = []
        for path in files:
            with safe_open(path, framework="pt", device="cpu") as stream:
                keys.extend(stream.keys())
    except Exception as error:
        raise MinistralPreflightError(
            "Ministral AWQ safetensors metadata cannot be read"
        ) from error
    return validate_ministral_awq_tensor_key_names(keys, exact_targets)


def write_detached_sha256(
    artifact_path: str | Path, sidecar_path: str | Path | None = None
) -> Path:
    """Atomically write a sha256sum-compatible detached digest."""
    artifact = Path(artifact_path)
    if not artifact.is_file():
        raise MinistralPreflightError(f"cannot hash missing artifact: {artifact}")
    sidecar = (
        Path(sidecar_path)
        if sidecar_path is not None
        else artifact.with_name(artifact.name + ".sha256")
    )
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    payload = f"{_file_sha256(artifact)}  {artifact.name}\n"
    handle, temporary = tempfile.mkstemp(
        prefix=f".{sidecar.name}.", suffix=".tmp", dir=sidecar.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="ascii", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, sidecar)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return sidecar


def verify_detached_sha256(
    artifact_path: str | Path, sidecar_path: str | Path | None = None
) -> str:
    """Validate the detached digest and its exact basename binding."""
    artifact = Path(artifact_path)
    sidecar = (
        Path(sidecar_path)
        if sidecar_path is not None
        else artifact.with_name(artifact.name + ".sha256")
    )
    try:
        line = sidecar.read_text(encoding="ascii")
    except (OSError, UnicodeError) as error:
        raise MinistralPreflightError(
            f"cannot read detached SHA-256: {sidecar}"
        ) from error
    pieces = line.rstrip("\n").split("  ", 1)
    if (
        len(pieces) != 2
        or not _is_sha256(pieces[0])
        or pieces[1] != artifact.name
        or line != f"{pieces[0]}  {pieces[1]}\n"
    ):
        raise MinistralPreflightError("detached SHA-256 sidecar is invalid")
    actual = _file_sha256(artifact)
    if actual != pieces[0]:
        raise MinistralPreflightError("detached SHA-256 mismatch")
    return actual


def _validate_invocation_token(owner_token: str) -> str:
    try:
        parsed = uuid.UUID(owner_token)
    except (ValueError, AttributeError) as error:
        raise MinistralPreflightError("disposable owner token is invalid") from error
    if str(parsed) != owner_token or parsed.version != 4:
        raise MinistralPreflightError("disposable owner token is invalid")
    return owner_token


def _process_start_time(pid: int) -> str | None:
    proc_stat = Path(f"/proc/{pid}/stat")
    if proc_stat.is_file():
        try:
            remainder = proc_stat.read_text(encoding="ascii").rpartition(")")[2]
            fields = remainder.split()
            return fields[19] if len(fields) > 19 else None
        except (OSError, UnicodeError):
            return None
    if os.name == "nt":
        try:
            import ctypes
            from ctypes import wintypes

            handle = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            creation = wintypes.FILETIME()
            exit_time = wintypes.FILETIME()
            kernel = wintypes.FILETIME()
            user = wintypes.FILETIME()
            try:
                if not ctypes.windll.kernel32.GetProcessTimes(
                    handle,
                    ctypes.byref(creation),
                    ctypes.byref(exit_time),
                    ctypes.byref(kernel),
                    ctypes.byref(user),
                ):
                    return None
                return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
            finally:
                ctypes.windll.kernel32.CloseHandle(handle)
        except (AttributeError, OSError, ValueError):
            return None
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return None
    return f"pid-{pid}"


def _current_process_owner(invocation_token: str) -> str:
    start_time = _process_start_time(os.getpid())
    if not start_time:
        raise MinistralPreflightError(
            "cannot construct disposable PID/start-time ownership token"
        )
    return f"{os.getpid()}:{start_time}:{invocation_token}"


def _parse_process_owner(value: object) -> tuple[int, str, str] | None:
    if not isinstance(value, str):
        return None
    pieces = value.split(":", 2)
    if len(pieces) != 3 or not pieces[0].isdigit() or not pieces[1]:
        return None
    try:
        invocation = _validate_invocation_token(pieces[2])
    except MinistralPreflightError:
        return None
    pid = int(pieces[0])
    if pid <= 0:
        return None
    return pid, pieces[1], invocation


def _process_owner_is_live(value: str) -> bool:
    parsed = _parse_process_owner(value)
    return bool(parsed and _process_start_time(parsed[0]) == parsed[1])


def _lease_record(process_owner: str, kind: str) -> dict[str, Any]:
    record = {
        "protocol_version": DISPOSABLE_LEASE_PROTOCOL,
        "kind": kind,
        "process_owner": process_owner,
    }
    record["record_sha256"] = _record_hash(record, excluded="record_sha256")
    return record


def _load_lease_record(path: Path, kind: str) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or set(value)
        != {"protocol_version", "kind", "process_owner", "record_sha256"}
        or value.get("protocol_version") != DISPOSABLE_LEASE_PROTOCOL
        or value.get("kind") != kind
        or _parse_process_owner(value.get("process_owner")) is None
        or value.get("record_sha256")
        != _record_hash(value, excluded="record_sha256")
    ):
        return None
    return value


def _path_is_abandoned(path: Path) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
    except OSError:
        return False
    return age >= DISPOSABLE_INITIALIZATION_GRACE_SECONDS


def _owner_snapshot(path: Path) -> tuple[tuple[str, bytes], ...] | None:
    try:
        if path.is_file():
            return (("__atomic_owner_record__", path.read_bytes()),)
        return tuple(
            (child.relative_to(path).as_posix(), child.read_bytes())
            for child in sorted(path.rglob("*"))
            if child.is_file()
        )
    except OSError:
        return None


def _owner_record(process_owner: str) -> dict[str, Any]:
    record = {
        "protocol_version": DISPOSABLE_OWNER_PROTOCOL,
        "process_owner": process_owner,
    }
    record["record_sha256"] = _record_hash(record, excluded="record_sha256")
    return record


def _load_owner_record(lock: Path) -> dict[str, Any] | None:
    try:
        record_path = lock if lock.is_file() else lock / "record.json"
        value = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if (
        not isinstance(value, dict)
        or set(value) != {"protocol_version", "process_owner", "record_sha256"}
        or value.get("protocol_version") != DISPOSABLE_OWNER_PROTOCOL
        or _parse_process_owner(value.get("process_owner")) is None
        or value.get("record_sha256")
        != _record_hash(value, excluded="record_sha256")
    ):
        return None
    return value


def _install_owner_lock(root: Path, process_owner: str) -> bool:
    prepared = root / f".owner.lock.prepare-{uuid.uuid4()}"
    payload = json.dumps(
        _owner_record(process_owner), sort_keys=True, separators=(",", ":")
    ) + "\n"
    try:
        with prepared.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(prepared, root / ".owner.lock")
            return True
        except FileExistsError:
            return False
        except OSError:
            if not (root / ".owner.lock").exists():
                raise
            return False
    finally:
        if prepared.exists():
            prepared.unlink()


def _claim_disposable_owner(root: Path, owner_token: str) -> Path:
    invocation_token = _validate_invocation_token(owner_token)
    process_owner = _current_process_owner(invocation_token)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".owner.lock"
    if not _install_owner_lock(root, process_owner):
        snapshot = _owner_snapshot(lock)
        record = _load_owner_record(lock)
        if record is not None:
            existing = record["process_owner"]
            if existing == process_owner:
                pass
            elif _process_owner_is_live(existing):
                raise MinistralPreflightError(
                    "disposable root is owned by another live owner"
                )
            else:
                _reclaim_owner_lock(root, lock, snapshot)
                if not _install_owner_lock(root, process_owner):
                    raise MinistralPreflightError(
                        "disposable owner changed during dead-owner reclaim"
                    )
        elif _path_is_abandoned(lock):
            _reclaim_owner_lock(root, lock, snapshot)
            if not _install_owner_lock(root, process_owner):
                raise MinistralPreflightError(
                    "disposable owner changed during abandoned-owner reclaim"
                )
        else:
            raise MinistralPreflightError(
                "disposable owner lock is still initializing"
            )
    invocation = root / "invocations" / invocation_token
    invocation.mkdir(parents=True, exist_ok=True)
    return invocation


def _reclaim_owner_lock(
    root: Path, lock: Path, expected_snapshot: tuple[tuple[str, bytes], ...] | None
) -> None:
    if expected_snapshot is None or _owner_snapshot(lock) != expected_snapshot:
        raise MinistralPreflightError("disposable owner changed during reclaim")
    quarantine = root / f".owner.lock.reclaimed-{uuid.uuid4()}"
    try:
        lock.rename(quarantine)
    except OSError as error:
        raise MinistralPreflightError(
            "disposable owner changed during reclaim"
        ) from error
    if quarantine.is_dir():
        shutil.rmtree(quarantine)
    else:
        quarantine.unlink()


def _lease_path(invocation: Path, kind: str) -> Path:
    leases = invocation / ".leases"
    leases.mkdir(parents=True, exist_ok=True)
    return leases / f"{kind}.json"


def _acquire_lease(invocation: Path, kind: str, owner_token: str) -> Path:
    path = _lease_path(invocation, kind)
    process_owner = _current_process_owner(owner_token)
    payload = json.dumps(
        _lease_record(process_owner, kind),
        sort_keys=True,
        separators=(",", ":"),
    ) + "\n"
    for _ in range(2):
        try:
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return path
        except FileExistsError:
            snapshot = path.read_bytes() if path.is_file() else None
            record = _load_lease_record(path, kind)
            if record is not None and _process_owner_is_live(
                record["process_owner"]
            ):
                raise MinistralPreflightError(f"{kind} route lease is active")
            if record is None and not _path_is_abandoned(path):
                raise MinistralPreflightError(
                    f"{kind} route lease is still initializing"
                )
            if snapshot is None:
                raise MinistralPreflightError(f"{kind} route lease is indeterminate")
            try:
                if path.read_bytes() != snapshot:
                    raise MinistralPreflightError(
                        f"{kind} route lease changed during reclaim"
                    )
                path.unlink()
            except OSError as error:
                raise MinistralPreflightError(
                    f"{kind} route lease changed during reclaim"
                ) from error
    raise MinistralPreflightError(f"cannot acquire {kind} route lease")


def _release_lease(invocation: Path, kind: str, owner_token: str) -> None:
    path = _lease_path(invocation, kind)
    record = _load_lease_record(path, kind)
    if record is None or record["process_owner"] != _current_process_owner(
        owner_token
    ):
        raise MinistralPreflightError(f"current process does not own {kind} lease")
    path.unlink()


def prepare_disposable_directory(
    disposable_root: str | Path, route: str, owner_token: str
) -> Path:
    """Reset only one route below the explicitly named disposable root."""
    root = Path(disposable_root).resolve(strict=False)
    if root.name != "ministral-quantizer-compatibility-disposable":
        raise MinistralPreflightError(
            "cleanup root is not the dedicated disposable directory"
        )
    if route not in {"gptq", "awq"}:
        raise MinistralPreflightError("disposable route must be gptq or awq")
    invocation = _claim_disposable_owner(root, owner_token)
    combine_lease = _lease_path(invocation, "combine")
    if combine_lease.exists():
        raise MinistralPreflightError("combine lease is active")
    _acquire_lease(invocation, route, owner_token)
    if combine_lease.exists():
        _release_lease(invocation, route, owner_token)
        raise MinistralPreflightError("combine lease is active")
    destination = invocation / route
    try:
        if destination.exists():
            if destination.is_symlink() or not destination.is_dir():
                raise MinistralPreflightError(
                    "disposable route path is not a real directory"
                )
            shutil.rmtree(destination)
        destination.mkdir(parents=True, exist_ok=False)
    except BaseException:
        _release_lease(invocation, route, owner_token)
        raise
    return destination


def complete_disposable_route(
    disposable_root: str | Path, route: str, owner_token: str
) -> None:
    """Mark a successfully certified route complete without deleting it."""
    root = _validated_disposable_root(disposable_root)
    if route not in {"gptq", "awq"}:
        raise MinistralPreflightError("disposable route must be gptq or awq")
    invocation = _claim_disposable_owner(root, owner_token)
    _release_lease(invocation, route, owner_token)


def cleanup_disposable_directory(
    disposable_root: str | Path, route: str, owner_token: str
) -> None:
    """Remove only one validated route below the dedicated disposable root."""
    root = _validated_disposable_root(disposable_root)
    if route not in {"gptq", "awq"}:
        raise MinistralPreflightError("disposable route must be gptq or awq")
    invocation = _claim_disposable_owner(root, owner_token)
    lease = _load_lease_record(_lease_path(invocation, route), route)
    if lease is None or lease["process_owner"] != _current_process_owner(
        owner_token
    ):
        raise MinistralPreflightError(
            "current process does not own disposable route lease"
        )
    destination = invocation / route
    if destination.exists():
        if destination.is_symlink() or not destination.is_dir():
            raise MinistralPreflightError(
                "disposable route path is not a real directory"
            )
        shutil.rmtree(destination)
    _release_lease(invocation, route, owner_token)


def acquire_disposable_combine_lease(
    disposable_root: str | Path, owner_token: str
) -> None:
    root = _validated_disposable_root(disposable_root)
    invocation = _claim_disposable_owner(root, owner_token)
    _acquire_lease(invocation, "combine", owner_token)
    active = [
        route
        for route in ("gptq", "awq")
        if _lease_path(invocation, route).exists()
    ]
    if active:
        _release_lease(invocation, "combine", owner_token)
        raise MinistralPreflightError(
            "route lease is active: " + ", ".join(active)
        )


def release_disposable_combine_lease(
    disposable_root: str | Path, owner_token: str
) -> None:
    root = _validated_disposable_root(disposable_root)
    invocation = _claim_disposable_owner(root, owner_token)
    _release_lease(invocation, "combine", owner_token)


def _validated_disposable_root(disposable_root: str | Path) -> Path:
    root = Path(disposable_root).resolve(strict=False)
    if root.name != "ministral-quantizer-compatibility-disposable":
        raise MinistralPreflightError(
            "cleanup root is not the dedicated disposable directory"
        )
    return root


def release_disposable_invocation(
    disposable_root: str | Path, owner_token: str
) -> None:
    root = _validated_disposable_root(disposable_root)
    invocation = _claim_disposable_owner(root, owner_token)
    leases = invocation / ".leases"
    if leases.is_dir() and any(leases.iterdir()):
        raise MinistralPreflightError(
            "cannot release disposable invocation with active leases"
        )
    if invocation.exists():
        shutil.rmtree(invocation)
    lock = root / ".owner.lock"
    record = _load_owner_record(lock)
    if record is None or record["process_owner"] != _current_process_owner(
        owner_token
    ):
        raise MinistralPreflightError("disposable owner token drifted")
    if lock.is_dir():
        shutil.rmtree(lock)
    else:
        lock.unlink()


@dataclass(frozen=True)
class ArtifactPublicationPlan:
    mode: str
    final_path: Path
    working_path: Path


def plan_artifact_publication(
    final_path: str | Path, *, provenance: Mapping[str, str]
) -> ArtifactPublicationPlan:
    """Plan a unique sibling build or validate an audit-only resume."""
    final = Path(final_path).resolve(strict=False)
    normalized = _validate_publication_provenance(provenance)
    if final.exists():
        if not final.is_dir():
            raise MinistralPreflightError("existing arbitrary final path is forbidden")
        marker_path = final / PUBLICATION_MARKER
        if not marker_path.is_file():
            raise MinistralPreflightError("existing arbitrary final directory is forbidden")
        marker = _load_json_object(marker_path, "publication marker")
        _require_exact_keys(
            marker,
            {
                "protocol_version",
                "final_path",
                "provenance",
                "artifact_files",
                "marker_sha256",
            },
            "publication marker",
        )
        if (
            marker["protocol_version"] != PUBLICATION_PROTOCOL
            or marker["final_path"] != str(final)
            or marker["provenance"] != normalized
            or not isinstance(marker["artifact_files"], Mapping)
            or not marker["artifact_files"]
            or marker["marker_sha256"]
            != _record_hash(marker, excluded="marker_sha256")
        ):
            raise MinistralPreflightError("publication provenance marker drifted")
        if _hash_directory(final, excluded_names={PUBLICATION_MARKER}) != dict(
            marker["artifact_files"]
        ):
            raise MinistralPreflightError("published artifact files drifted")
        return ArtifactPublicationPlan("resume_audit", final, final)
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent)
    ).resolve()
    return ArtifactPublicationPlan("build", final, staging)


def publish_staged_artifact(
    plan: ArtifactPublicationPlan, *, provenance: Mapping[str, str]
) -> None:
    if not isinstance(plan, ArtifactPublicationPlan) or plan.mode != "build":
        raise MinistralPreflightError("only a staged build can be published")
    normalized = _validate_publication_provenance(provenance)
    if plan.final_path.exists() or not plan.working_path.is_dir():
        raise MinistralPreflightError("atomic publication destination drifted")
    if not any(plan.working_path.iterdir()):
        raise MinistralPreflightError("staged artifact is empty")
    marker = {
        "protocol_version": PUBLICATION_PROTOCOL,
        "final_path": str(plan.final_path),
        "provenance": normalized,
        "artifact_files": _hash_directory(plan.working_path),
    }
    marker["marker_sha256"] = _record_hash(marker, excluded="marker_sha256")
    _write_json_atomic(marker, plan.working_path / PUBLICATION_MARKER)
    os.replace(plan.working_path, plan.final_path)


def discard_staged_artifact(plan: ArtifactPublicationPlan) -> None:
    """Delete only the unique unpublished sibling staging directory."""
    if not isinstance(plan, ArtifactPublicationPlan) or plan.mode != "build":
        return
    expected_prefix = f".{plan.final_path.name}.staging-"
    if (
        plan.working_path.parent != plan.final_path.parent
        or not plan.working_path.name.startswith(expected_prefix)
    ):
        raise MinistralPreflightError("refusing to delete an unowned staging path")
    if plan.working_path.exists():
        shutil.rmtree(plan.working_path)


def _validate_publication_provenance(
    value: Mapping[str, str],
) -> dict[str, str]:
    normalized = {str(key): str(child) for key, child in value.items()}
    if set(normalized) != {
        "route",
        "source_audit_sha256",
        "compatibility_gate_sha256",
    }:
        raise MinistralPreflightError("publication provenance schema drifted")
    if normalized["route"] not in {"gptq", "awq"} or not _is_sha256(
        normalized["source_audit_sha256"]
    ) or not _is_sha256(normalized["compatibility_gate_sha256"]):
        raise MinistralPreflightError("publication provenance is invalid")
    return normalized


def build_route_compatibility_certificate(
    *,
    route: str,
    source_path: str | Path,
    source_audit_sha256: str,
    source_audit_file_sha256: str,
    package_versions: Mapping[str, str],
    calibration_id: str,
    rendered_text_sha256: str,
    token_ids_sha256: str,
    artifact_path: str | Path,
    route_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind one real one-item disposable quantization route."""
    if route not in {"gptq", "awq"}:
        raise MinistralPreflightError("compatibility route must be gptq or awq")
    versions = {
        str(name): str(version) for name, version in package_versions.items()
    }
    expected_versions = (
        GPTQ_DEPENDENCY_VERSIONS if route == "gptq" else AWQ_DEPENDENCY_VERSIONS
    )
    if versions != expected_versions:
        raise MinistralPreflightError(
            f"{route} compatibility package versions drifted"
        )
    for value, label in (
        (source_audit_sha256, "source audit"),
        (source_audit_file_sha256, "source audit file"),
    ):
        if not _is_sha256(value):
            raise MinistralPreflightError(f"{label} SHA-256 is invalid")
    if not isinstance(calibration_id, str) or not calibration_id:
        raise MinistralPreflightError("compatibility calibration ID is invalid")
    for digest, label in (
        (rendered_text_sha256, "rendered text"),
        (token_ids_sha256, "token IDs"),
    ):
        if not _is_sha256(digest):
            raise MinistralPreflightError(f"compatibility {label} hash is invalid")
    contract = _validate_route_contract(route, dict(route_contract))
    artifact = Path(artifact_path).resolve()
    files = _hash_directory(artifact)
    config_path = artifact / "config.json"
    config = _load_json_object(config_path, "disposable artifact config")
    if not isinstance(config.get("quantization_config"), Mapping):
        raise MinistralPreflightError(
            "disposable artifact quantization_config is missing"
        )
    _validate_saved_quantization_config(
        route, config["quantization_config"], contract
    )
    record: dict[str, Any] = {
        "protocol_version": ROUTE_COMPATIBILITY_PROTOCOL,
        "route": route,
        "model_key": "ministral3_8b",
        "source": {
            "path": str(Path(source_path).resolve()),
            "audit_sha256": source_audit_sha256,
            "audit_file_sha256": source_audit_file_sha256,
        },
        "package_versions": versions,
        "calibration": {
            "count": 1,
            "ids": [calibration_id],
            "ids_sha256": hashlib.sha256(
                f"{calibration_id}\n".encode("utf-8")
            ).hexdigest(),
            "rendered_text_sha256": rendered_text_sha256,
            "token_ids_sha256": token_ids_sha256,
        },
        "artifact": {
            "path": str(artifact),
            "files": files,
            "config_file_sha256": _file_sha256(config_path),
        },
        "route_contract": contract,
        "gate_passed": True,
    }
    record["certificate_sha256"] = _record_hash(
        record, excluded="certificate_sha256"
    )
    _reject_outcome_keys(record)
    return record


def load_route_compatibility_certificate(
    certificate_path: str | Path,
    sha256_path: str | Path,
    *,
    verify_artifact: bool = True,
) -> dict[str, Any]:
    """Load one closed-schema disposable route certificate."""
    certificate_file = Path(certificate_path)
    verify_detached_sha256(certificate_file, sha256_path)
    try:
        value = json.loads(certificate_file.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MinistralPreflightError(
            "cannot load route compatibility certificate"
        ) from error
    if not isinstance(value, dict):
        raise MinistralPreflightError("route certificate must be an object")
    _reject_outcome_keys(value)
    _require_exact_keys(
        value,
        {
            "protocol_version",
            "route",
            "model_key",
            "source",
            "package_versions",
            "calibration",
            "artifact",
            "route_contract",
            "gate_passed",
            "certificate_sha256",
        },
        "route certificate",
    )
    route = value.get("route")
    if (
        value.get("protocol_version") != ROUTE_COMPATIBILITY_PROTOCOL
        or route not in {"gptq", "awq"}
        or value.get("model_key") != "ministral3_8b"
        or value.get("gate_passed") is not True
        or value.get("certificate_sha256")
        != _record_hash(value, excluded="certificate_sha256")
    ):
        raise MinistralPreflightError(
            "route compatibility certificate identity or hash drifted"
        )
    source = value.get("source")
    _require_exact_keys(
        source, {"path", "audit_sha256", "audit_file_sha256"}, "route source"
    )
    if (
        not isinstance(source["path"], str)
        or not source["path"]
        or not _is_sha256(source["audit_sha256"])
        or not _is_sha256(source["audit_file_sha256"])
    ):
        raise MinistralPreflightError("route source binding is invalid")
    expected_versions = (
        GPTQ_DEPENDENCY_VERSIONS if route == "gptq" else AWQ_DEPENDENCY_VERSIONS
    )
    if value.get("package_versions") != expected_versions:
        raise MinistralPreflightError("route package versions drifted")
    calibration = value.get("calibration")
    _require_exact_keys(
        calibration,
        {
            "count",
            "ids",
            "ids_sha256",
            "rendered_text_sha256",
            "token_ids_sha256",
        },
        "route calibration",
    )
    ids = calibration["ids"]
    if (
        isinstance(calibration["count"], bool)
        or calibration["count"] != 1
        or not isinstance(ids, list)
        or len(ids) != 1
        or not isinstance(ids[0], str)
        or not ids[0]
        or calibration["ids_sha256"]
        != hashlib.sha256(f"{ids[0]}\n".encode()).hexdigest()
        or not _is_sha256(calibration["rendered_text_sha256"])
        or not _is_sha256(calibration["token_ids_sha256"])
    ):
        raise MinistralPreflightError("route calibration binding is invalid")
    artifact = value.get("artifact")
    _require_exact_keys(
        artifact, {"path", "files", "config_file_sha256"}, "route artifact"
    )
    if not isinstance(artifact["path"], str) or not artifact["path"]:
        raise MinistralPreflightError("route artifact path is invalid")
    files = artifact["files"]
    if (
        not isinstance(files, dict)
        or not files
        or any(
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
            or not _is_sha256(digest)
            for name, digest in files.items()
        )
        or not _is_sha256(artifact["config_file_sha256"])
    ):
        raise MinistralPreflightError("route artifact file binding is invalid")
    _validate_route_contract(route, value.get("route_contract"))
    if verify_artifact:
        root = Path(artifact["path"])
        if _hash_directory(root) != files:
            raise MinistralPreflightError("disposable route artifact files drifted")
        if _file_sha256(root / "config.json") != artifact["config_file_sha256"]:
            raise MinistralPreflightError("disposable route config drifted")
        saved_config = _load_json_object(
            root / "config.json", "disposable artifact config"
        ).get("quantization_config")
        if not isinstance(saved_config, Mapping):
            raise MinistralPreflightError(
                "disposable artifact quantization_config is missing"
            )
        _validate_saved_quantization_config(
            route, saved_config, value["route_contract"]
        )
    return value


def build_compatibility_gate_record(
    *,
    gptq_certificate_path: str | Path,
    gptq_certificate_sha256_path: str | Path,
    awq_certificate_path: str | Path,
    awq_certificate_sha256_path: str | Path,
) -> dict[str, Any]:
    """Combine separately executed GPTQ and AWQ route certificates."""
    inputs = {
        "gptq": (
            Path(gptq_certificate_path),
            Path(gptq_certificate_sha256_path),
        ),
        "awq": (Path(awq_certificate_path), Path(awq_certificate_sha256_path)),
    }
    certificates = {
        route: load_route_compatibility_certificate(path, sidecar)
        for route, (path, sidecar) in inputs.items()
    }
    if any(certificates[route]["route"] != route for route in inputs):
        raise MinistralPreflightError("route certificate type drifted")
    if certificates["gptq"]["source"] != certificates["awq"]["source"]:
        raise MinistralPreflightError("route certificate source binding drifted")
    if certificates["gptq"]["calibration"] != certificates["awq"]["calibration"]:
        raise MinistralPreflightError(
            "route certificate calibration binding drifted"
        )
    routes = {
        route: {
            "certificate_path": str(path.resolve()),
            "certificate_sidecar_path": str(sidecar.resolve()),
            "certificate_sha256": certificates[route]["certificate_sha256"],
            "certificate_file_sha256": verify_detached_sha256(path, sidecar),
        }
        for route, (path, sidecar) in inputs.items()
    }
    record: dict[str, Any] = {
        "protocol_version": COMPATIBILITY_GATE_PROTOCOL,
        "model_key": "ministral3_8b",
        "source": certificates["gptq"]["source"],
        "calibration": certificates["gptq"]["calibration"],
        "routes": routes,
        "gate_passed": True,
    }
    record["gate_sha256"] = _record_hash(record, excluded="gate_sha256")
    _reject_outcome_keys(record)
    return record


def load_compatibility_gate(
    gate_path: str | Path,
    sha256_path: str | Path,
    *,
    source_path: str | Path,
    source_audit_sha256: str,
    source_audit_file_sha256: str,
) -> dict[str, Any]:
    """Require a closed, combined, source-bound compatibility gate."""
    gate_file = Path(gate_path)
    verify_detached_sha256(gate_file, sha256_path)
    value = _load_json_object(gate_file, "compatibility gate")
    _reject_outcome_keys(value)
    _require_exact_keys(
        value,
        {
            "protocol_version",
            "model_key",
            "source",
            "calibration",
            "routes",
            "gate_passed",
            "gate_sha256",
        },
        "compatibility gate",
    )
    if (
        value["protocol_version"] != COMPATIBILITY_GATE_PROTOCOL
        or value["model_key"] != "ministral3_8b"
        or value["gate_passed"] is not True
        or value["gate_sha256"] != _record_hash(value, excluded="gate_sha256")
    ):
        raise MinistralPreflightError("compatibility gate identity or hash drifted")
    expected_source = {
        "path": str(Path(source_path).resolve()),
        "audit_sha256": source_audit_sha256,
        "audit_file_sha256": source_audit_file_sha256,
    }
    if value["source"] != expected_source:
        raise MinistralPreflightError("compatibility gate source binding drifted")
    routes = value["routes"]
    _require_exact_keys(routes, {"gptq", "awq"}, "compatibility routes")
    loaded = {}
    for route, binding in routes.items():
        _require_exact_keys(
            binding,
            {
                "certificate_path",
                "certificate_sidecar_path",
                "certificate_sha256",
                "certificate_file_sha256",
            },
            f"{route} certificate binding",
        )
        if (
            not isinstance(binding["certificate_path"], str)
            or not binding["certificate_path"]
            or not isinstance(binding["certificate_sidecar_path"], str)
            or not binding["certificate_sidecar_path"]
            or not _is_sha256(binding["certificate_sha256"])
            or not _is_sha256(binding["certificate_file_sha256"])
        ):
            raise MinistralPreflightError(
                f"{route} certificate binding types drifted"
            )
        certificate = load_route_compatibility_certificate(
            binding["certificate_path"],
            binding["certificate_sidecar_path"],
            verify_artifact=False,
        )
        if (
            certificate["route"] != route
            or certificate["certificate_sha256"] != binding["certificate_sha256"]
            or verify_detached_sha256(
                binding["certificate_path"], binding["certificate_sidecar_path"]
            )
            != binding["certificate_file_sha256"]
        ):
            raise MinistralPreflightError(
                f"{route} route certificate binding drifted"
            )
        loaded[route] = certificate
    if (
        loaded["gptq"]["source"] != expected_source
        or loaded["awq"]["source"] != expected_source
        or loaded["gptq"]["calibration"] != value["calibration"]
        or loaded["awq"]["calibration"] != value["calibration"]
    ):
        raise MinistralPreflightError("combined route binding drifted")
    return value


def _reject_outcome_keys(value: object) -> None:
    forbidden = (
        "accuracy",
        "prediction",
        "probability",
        "label",
        "score",
        "effect",
        "metric",
        "logit",
        "payload",
    )
    if isinstance(value, Mapping):
        for key, child in value.items():
            folded = str(key).casefold()
            if any(term in folded for term in forbidden):
                raise MinistralPreflightError(
                    "compatibility gate contains an outcome field"
                )
            _reject_outcome_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_outcome_keys(child)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise MinistralPreflightError(f"cannot hash artifact: {path}") from error
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any], *, excluded: str = "gate_sha256") -> str:
    payload = json.dumps(
        {key: value for key, value in record.items() if key != excluded},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_route_contract(route: str, value: object) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise MinistralPreflightError("route contract must be an object")
    contract = dict(value)
    _reject_outcome_keys(contract)
    if route == "gptq":
        _require_exact_keys(contract, {"bits", "group_size", "sym"}, "GPTQ contract")
        if contract != {"bits": 4, "group_size": 128, "sym": True}:
            raise MinistralPreflightError("GPTQ route contract drifted")
        return contract
    _require_exact_keys(
        contract,
        {
            "scheme",
            "group_size",
            "zero_point",
            "exact_targets",
            "resolved_mappings",
        },
        "AWQ contract",
    )
    if (
        contract["scheme"] != "W4A16_ASYM"
        or contract["group_size"] != 128
        or contract["zero_point"] is not True
    ):
        raise MinistralPreflightError("AWQ scheme drifted")
    targets = contract["exact_targets"]
    if (
        not isinstance(targets, list)
        or not targets
        or targets != sorted(set(targets))
    ):
        raise MinistralPreflightError("AWQ exact targets drifted")
    mappings = validate_awq_mapping_scope(contract["resolved_mappings"], targets)
    return {
        "scheme": "W4A16_ASYM",
        "group_size": 128,
        "zero_point": True,
        "exact_targets": targets,
        "resolved_mappings": mappings,
    }


def _validate_saved_quantization_config(
    route: str,
    quantization_config: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    config = dict(quantization_config)
    if route == "gptq":
        observed = {
            "bits": config.get("bits"),
            "group_size": config.get("group_size"),
            "sym": config.get("sym"),
        }
        if observed != dict(contract):
            raise MinistralPreflightError(
                "saved GPTQ quantization_config drifted from route contract"
            )
        return
    groups = config.get("config_groups")
    if not isinstance(groups, Mapping) or not groups:
        raise MinistralPreflightError("saved AWQ config groups are missing")
    observed_targets: set[str] = set()
    for group in groups.values():
        if not isinstance(group, Mapping):
            raise MinistralPreflightError("saved AWQ config group is invalid")
        targets = group.get("targets")
        weights = group.get("weights")
        if (
            not isinstance(targets, list)
            or not targets
            or not isinstance(weights, Mapping)
            or weights.get("num_bits") != 4
            or weights.get("group_size") != contract["group_size"]
            or weights.get("symmetric") is not False
        ):
            raise MinistralPreflightError(
                "saved AWQ quantization_config drifted from scheme"
            )
        observed_targets.update(str(target) for target in targets)
    if observed_targets != set(contract["exact_targets"]):
        raise MinistralPreflightError(
            "saved AWQ target scope drifted from exact route contract"
        )


def _require_exact_keys(value: object, expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MinistralPreflightError(f"{label} schema drifted")


def _hash_directory(
    root: Path, *, excluded_names: set[str] | None = None
) -> dict[str, str]:
    if not root.is_dir():
        raise MinistralPreflightError("disposable artifact directory is missing")
    excluded = excluded_names or set()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.relative_to(root).as_posix() not in excluded
    )
    if not files:
        raise MinistralPreflightError("disposable artifact directory is empty")
    return {path.relative_to(root).as_posix(): _file_sha256(path) for path in files}


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise MinistralPreflightError(f"cannot load {label}") from error
    if not isinstance(value, dict):
        raise MinistralPreflightError(f"{label} must be an object")
    return value


def _write_json_atomic(value: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ) + "\n"
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


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("write", "check"):
        child = subparsers.add_parser(command)
        child.add_argument("artifact", type=Path)
        child.add_argument("sidecar", type=Path, nargs="?")
    args = parser.parse_args(argv)
    if args.command == "write":
        write_detached_sha256(args.artifact, args.sidecar)
    else:
        verify_detached_sha256(args.artifact, args.sidecar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
