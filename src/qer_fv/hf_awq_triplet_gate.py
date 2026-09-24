"""Blind structural Gate for frozen HF FP16/GPTQ/AWQ triplets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .awq_route import AWQRouteError, load_awq_artifact_audit
from .hf_model_identity import HFModelIdentityError, hf_model_identity
from .hf_runtime_audit import HFRuntimeAuditError, load_hf_runtime_audit
from .qwen_triplet_freeze import load_qwen_triplet_freeze


HF_AWQ_TRIPLET_GATE_PROTOCOL = "hf-awq-triplet-gate-v1-20260721"
QWEN_HF_TRIPLET_GATE_PROTOCOL = "qwen-hf-triplet-gate-v1-20260728"
QWEN_HF_TRIPLET_SMOKE_PROTOCOL = "qwen-hf-triplet-smoke-v1-20260728"
_D5_RUN_PROTOCOLS = {
    "smoke": "hf-awq-d5-smoke-v1-20260721",
    "formal": "hf-awq-d5-formal-v1-20260721",
}
_SPECS = {
    "smoke": {
        "split": "pilot",
        "full": 8,
        "baseline_controls": {
            "full": 8,
            "no_evidence": 4,
            "knowledge_only": 4,
        },
    },
    "formal": {
        "split": "dose_subset",
        "full": 4800,
        "baseline_controls": {
            "full": 4800,
            "no_evidence": 2400,
            "knowledge_only": 2400,
        },
    },
    "qwen_smoke16": {
        "split": "pilot",
        "full": 16,
        "baseline_controls": {
            "full": 16,
            "no_evidence": 0,
            "knowledge_only": 0,
        },
    },
    "qwen_formal4800": {
        "split": "dose_subset",
        "full": 4800,
        "baseline_controls": {
            "full": 4800,
            "no_evidence": 2400,
            "knowledge_only": 2400,
        },
    },
    "ministral_smoke16": {
        "split": "pilot",
        "full": 16,
        "baseline_controls": {
            "full": 16,
            "no_evidence": 0,
            "knowledge_only": 0,
        },
    },
    "ministral_formal4800": {
        "split": "dose_subset",
        "full": 4800,
        "baseline_controls": {
            "full": 4800,
            "no_evidence": 0,
            "knowledge_only": 0,
        },
    },
    "olmo3_smoke16": {
        "split": "pilot",
        "full": 16,
        "baseline_controls": {
            "full": 16,
            "no_evidence": 0,
            "knowledge_only": 0,
        },
    },
    "olmo3_formal4800": {
        "split": "dose_subset",
        "full": 4800,
        "baseline_controls": {
            "full": 4800,
            "no_evidence": 0,
            "knowledge_only": 0,
        },
    },
}
_COMMON_PAYLOAD_FIELDS = (
    "prompt_sha256",
    "generation_input_ids_sha256",
    "generation_prompt_sha256",
    "token_ids",
)
_REQUIRED_METADATA_FIELDS = (
    "case_id",
    "cell_index",
    "page",
    "gold_label",
    "negative_label",
)
_FORBIDDEN_AGGREGATES = {
    "accuracy",
    "balanced_accuracy",
    "brier",
    "ece",
    "effect",
    "f1",
    "mcc",
    "mcnemar",
}


class HFAWQTripletGateError(RuntimeError):
    """Raised when the frozen three-condition structure is not exact."""


def validate_hf_awq_triplet(
    *,
    fp16_export: str | Path,
    gptq_export: str | Path,
    awq_export: str | Path,
    fp16_audit: str | Path,
    gptq_audit: str | Path,
    awq_audit: str | Path,
    awq_artifact_audit: str | Path,
    model_key: str,
    mode: str = "smoke",
    output_path: str | Path | None = None,
    qwen_freeze_config: str | Path | None = None,
    qwen_freeze_sha256_path: str | Path | None = None,
) -> dict[str, Any]:
    """Prove aligned full-evidence inputs without inspecting outcomes."""
    if mode not in _SPECS:
        raise HFAWQTripletGateError(
            "triplet gate mode must be smoke, formal, qwen_smoke16, "
            "qwen_formal4800, ministral_smoke16, ministral_formal4800, "
            "olmo3_smoke16, or olmo3_formal4800"
        )
    qwen_freeze_sha256 = None
    if mode in {"qwen_smoke16", "qwen_formal4800"}:
        if model_key != "qwen35_9b":
            raise HFAWQTripletGateError(f"{mode} is Qwen-only")
        if qwen_freeze_config is None or qwen_freeze_sha256_path is None:
            raise HFAWQTripletGateError("Qwen-only freeze binding is required")
        try:
            load_qwen_triplet_freeze(
                qwen_freeze_config, sha256_path=qwen_freeze_sha256_path
            )
            qwen_freeze_sha256 = Path(qwen_freeze_sha256_path).read_text(
                encoding="ascii"
            ).strip()
        except (OSError, ValueError) as error:
            raise HFAWQTripletGateError(
                f"Qwen-only freeze binding is invalid: {error}"
            ) from error
    try:
        identity = hf_model_identity(model_key)
    except HFModelIdentityError as error:
        raise HFAWQTripletGateError(str(error)) from error
    paths = {
        "FP16 export": Path(fp16_export),
        "GPTQ export": Path(gptq_export),
        "AWQ export": Path(awq_export),
        "FP16 audit": Path(fp16_audit),
        "GPTQ audit": Path(gptq_audit),
        "AWQ audit": Path(awq_audit),
        "AWQ artifact audit": Path(awq_artifact_audit),
    }
    for label, path in paths.items():
        _reject_recovery_path(path, label)

    try:
        fp_certificate = load_hf_runtime_audit(fp16_audit)
        gptq_certificate = load_hf_runtime_audit(gptq_audit)
        awq_certificate = load_hf_runtime_audit(awq_audit)
    except HFRuntimeAuditError as error:
        raise HFAWQTripletGateError(f"runtime certificate invalid: {error}") from error
    try:
        artifact = load_awq_artifact_audit(awq_artifact_audit)
    except AWQRouteError as error:
        raise HFAWQTripletGateError(f"AWQ artifact invalid: {error}") from error

    certificates = {
        "FP16": fp_certificate,
        "GPTQ_INT4": gptq_certificate,
        "AWQ_INT4": awq_certificate,
    }
    for precision, certificate in certificates.items():
        if certificate.model_key != identity.model_key:
            raise HFAWQTripletGateError("runtime certificate model identity drifted")
        if certificate.precision != precision:
            raise HFAWQTripletGateError("runtime certificate precision drifted")
    for field in (
        "protocol_version",
        "model_key",
        "prompt_contract_sha256",
        "choice_token_ids",
        "generation_input_ids_sha256",
        "generation_prompt_sha256",
        "chat_template_sha256",
        "direct_logit_method",
    ):
        values = {
            json.dumps(
                dict(value) if isinstance(value, Mapping) else value,
                sort_keys=True,
            )
            for item in certificates.values()
            for value in (getattr(item, field),)
        }
        if len(values) != 1:
            raise HFAWQTripletGateError(
                f"runtime certificate alignment drifted at {field}"
            )
    base_packages = dict(fp_certificate.package_versions)
    if dict(gptq_certificate.package_versions) != base_packages:
        raise HFAWQTripletGateError("runtime certificate package alignment drifted")
    awq_packages = dict(awq_certificate.package_versions)
    if awq_packages.pop("compressed-tensors", None) != "0.17.1":
        raise HFAWQTripletGateError("AWQ runtime package binding drifted")
    if awq_packages != base_packages:
        raise HFAWQTripletGateError("AWQ runtime common packages drifted")
    artifact_file_sha256 = _file_sha256(Path(awq_artifact_audit))
    if artifact.get("model_key") != model_key:
        raise HFAWQTripletGateError("AWQ artifact model identity drifted")
    if (
        awq_certificate.awq_artifact_audit_sha256 != artifact.get("audit_sha256")
        or awq_certificate.awq_artifact_file_sha256 != artifact_file_sha256
    ):
        raise HFAWQTripletGateError("AWQ artifact certificate binding drifted")

    spec = _SPECS[mode]
    if mode in {"qwen_smoke16", "qwen_formal4800"}:
        if mode == "qwen_smoke16":
            baseline_protocol = QWEN_HF_TRIPLET_SMOKE_PROTOCOL
            awq_protocol = QWEN_HF_TRIPLET_SMOKE_PROTOCOL
        else:
            baseline_protocol = identity.formal_protocol
            awq_protocol = _D5_RUN_PROTOCOLS["formal"]
    elif mode in {
        "ministral_smoke16",
        "ministral_formal4800",
        "olmo3_smoke16",
        "olmo3_formal4800",
    }:
        expected_key = (
            "ministral3_8b" if mode.startswith("ministral_") else "olmo3_7b"
        )
        if model_key != expected_key:
            raise HFAWQTripletGateError(f"{mode} has the wrong model identity")
        baseline_protocol = (
            identity.smoke_protocol
            if mode.endswith("smoke16")
            else identity.formal_protocol
        )
        awq_protocol = baseline_protocol
    else:
        baseline_protocol = (
            identity.smoke_protocol if mode == "smoke" else identity.formal_protocol
        )
        awq_protocol = _D5_RUN_PROTOCOLS[mode]
    fp_manifest, fp_rows = _load_export(Path(fp16_export))
    gptq_manifest, gptq_rows = _load_export(Path(gptq_export))
    awq_manifest, awq_rows = _load_export(Path(awq_export))
    _validate_manifest(
        fp_manifest,
        certificate_sha256=fp_certificate.certificate_sha256,
        precision="FP16",
        protocol=baseline_protocol,
        model_key=model_key,
        split=spec["split"],
        controls=spec["baseline_controls"],
    )
    _validate_manifest(
        gptq_manifest,
        certificate_sha256=gptq_certificate.certificate_sha256,
        precision="GPTQ_INT4",
        protocol=baseline_protocol,
        model_key=model_key,
        split=spec["split"],
        controls=spec["baseline_controls"],
    )
    awq_controls = {
        "full": spec["full"],
        "no_evidence": 0,
        "knowledge_only": 0,
    }
    _validate_manifest(
        awq_manifest,
        certificate_sha256=awq_certificate.certificate_sha256,
        precision="AWQ_INT4",
        protocol=awq_protocol,
        model_key=model_key,
        split=spec["split"],
        controls=awq_controls,
    )
    identities = [
        fp_manifest["run_identity"],
        gptq_manifest["run_identity"],
        awq_manifest["run_identity"],
    ]
    for field in ("split", "split_sha256", "prompt_contract_sha256", "model_key"):
        if len({json.dumps(item.get(field), sort_keys=True) for item in identities}) != 1:
            raise HFAWQTripletGateError(f"run identity alignment drifted at {field}")

    baseline_total = sum(spec["baseline_controls"].values())
    fp_index = _index_rows(fp_rows, baseline_total)
    gptq_index = _index_rows(gptq_rows, baseline_total)
    awq_index = _index_rows(awq_rows, spec["full"])
    fp_full = {key: row for key, row in fp_index.items() if row.get("control") == "full"}
    gptq_full = {
        key: row for key, row in gptq_index.items() if row.get("control") == "full"
    }
    if len(fp_full) != spec["full"] or len(gptq_full) != spec["full"]:
        raise HFAWQTripletGateError("baseline full-owner count drifted")
    if set(fp_full) != set(gptq_full) or set(fp_full) != set(awq_index):
        raise HFAWQTripletGateError("full owner-set alignment drifted")

    for owner in sorted(fp_full):
        rows = (fp_full[owner], gptq_full[owner], awq_index[owner])
        if any(row.get("control") != "full" for row in rows):
            raise HFAWQTripletGateError(f"full control alignment drifted for {owner}")
        metadata = rows[0].get("metadata")
        if not isinstance(metadata, Mapping) or any(
            field not in metadata for field in _REQUIRED_METADATA_FIELDS
        ):
            raise HFAWQTripletGateError(f"metadata is incomplete for {owner}")
        normalized_metadata = _normalize_metadata(metadata)
        if any(
            not isinstance(row.get("metadata"), Mapping)
            or _normalize_metadata(row["metadata"]) != normalized_metadata
            for row in rows[1:]
        ):
            raise HFAWQTripletGateError(f"metadata alignment drifted for {owner}")
        for stage in ("hard", "probability"):
            payloads: list[Mapping[str, Any]] = []
            for row in rows:
                if row.get(f"{stage}_status") != "ok":
                    raise HFAWQTripletGateError(f"{stage} condition failed for {owner}")
                payload = row.get(f"{stage}_payload")
                if not isinstance(payload, Mapping):
                    raise HFAWQTripletGateError(f"{stage} payload missing for {owner}")
                payloads.append(payload)
            fields = list(_COMMON_PAYLOAD_FIELDS)
            if stage == "probability":
                fields.append("direct_logit_method")
            for field in fields:
                values = {json.dumps(payload.get(field), sort_keys=True) for payload in payloads}
                if len(values) != 1:
                    raise HFAWQTripletGateError(
                        f"{stage} alignment drifted at {field} for {owner}"
                    )
            if stage == "probability" and payloads[0].get(
                "direct_logit_method"
            ) != fp_certificate.direct_logit_method:
                raise HFAWQTripletGateError(
                    f"probability direct-logit method drifted for {owner}"
                )
            for payload in payloads:
                _validate_finite_scores(payload, stage=stage, owner=owner)
        for row in rows:
            for field in _COMMON_PAYLOAD_FIELDS:
                if row["hard_payload"].get(field) != row["probability_payload"].get(field):
                    raise HFAWQTripletGateError(
                        f"one-forward alignment drifted at {field} for {owner}"
                    )
            if row["hard_payload"].get("choice_logprobs") != row[
                "probability_payload"
            ].get("choice_logprobs"):
                raise HFAWQTripletGateError(
                    f"one-forward score alignment drifted for {owner}"
                )

    result: dict[str, Any] = {
        "protocol_version": (
            QWEN_HF_TRIPLET_GATE_PROTOCOL
            if mode in {"qwen_smoke16", "qwen_formal4800"}
            else HF_AWQ_TRIPLET_GATE_PROTOCOL
        ),
        "mode": mode,
        "model_key": model_key,
        "gate_passed": True,
        "aligned_full_owners": len(fp_full),
        "controls": {"full": len(fp_full)},
        "split_sha256": identities[0]["split_sha256"],
        "fp16_audit_certificate_sha256": fp_certificate.certificate_sha256,
        "gptq_audit_certificate_sha256": gptq_certificate.certificate_sha256,
        "awq_audit_certificate_sha256": awq_certificate.certificate_sha256,
        "awq_artifact_audit_sha256": artifact["audit_sha256"],
        "awq_artifact_file_sha256": artifact_file_sha256,
        "fp16_manifest_sha256": fp_manifest["manifest_sha256"],
        "gptq_manifest_sha256": gptq_manifest["manifest_sha256"],
        "awq_manifest_sha256": awq_manifest["manifest_sha256"],
    }
    if qwen_freeze_sha256 is not None:
        result["qwen_triplet_freeze_sha256"] = qwen_freeze_sha256
    result["gate_sha256"] = _record_hash(result, excluded="gate_sha256")
    _reject_aggregate_keys(result)
    if output_path is not None:
        _write_atomic_json(result, Path(output_path))
    return result


def _load_export(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json_object(root / "manifest.json", "manifest")
    if manifest.get("manifest_sha256") != _record_hash(
        manifest, excluded="manifest_sha256"
    ):
        raise HFAWQTripletGateError("manifest hash mismatch")
    records_path = root / "records.jsonl"
    try:
        records_bytes = records_path.read_bytes()
        rows = [
            json.loads(line, parse_constant=_reject_json_constant)
            for line in records_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFAWQTripletGateError("cannot read structural records") from error
    if manifest.get("records_jsonl_sha256") != hashlib.sha256(records_bytes).hexdigest():
        raise HFAWQTripletGateError("records hash mismatch")
    if any(not isinstance(row, dict) for row in rows):
        raise HFAWQTripletGateError("structural records must be JSON objects")
    _reject_aggregate_keys(manifest)
    _reject_aggregate_keys(rows)
    return manifest, rows


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _validate_finite_scores(
    payload: Mapping[str, Any], *, stage: str, owner: str
) -> None:
    logprobs = payload.get("choice_logprobs")
    if not isinstance(logprobs, Mapping) or set(logprobs) != {"A", "B", "C"} or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in logprobs.values()
    ):
        raise HFAWQTripletGateError(f"{stage} scores are non-finite for {owner}")
    if stage != "probability":
        return
    probabilities = payload.get("choice_probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != {"A", "B", "C"} or any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
        for value in probabilities.values()
    ):
        raise HFAWQTripletGateError(f"probability coverage is invalid for {owner}")
    mass = payload.get("label_probability_mass")
    if (
        isinstance(mass, bool)
        or not isinstance(mass, (int, float))
        or not math.isfinite(float(mass))
        or not 0.0 <= float(mass) <= 1.0
        or abs(sum(float(value) for value in probabilities.values()) - float(mass)) > 1e-8
    ):
        raise HFAWQTripletGateError(f"probability mass is invalid for {owner}")


def _validate_manifest(
    manifest: Mapping[str, Any],
    *,
    certificate_sha256: str,
    precision: str,
    protocol: str,
    model_key: str,
    split: str,
    controls: Mapping[str, int],
) -> None:
    total = sum(controls.values())
    expected_counts = {
        "units_registered": total,
        "full_registered": controls["full"],
        "no_evidence_registered": controls["no_evidence"],
        "knowledge_only_registered": controls["knowledge_only"],
        "hard_completed": total,
        "hard_failures": 0,
        "probability_completed": total,
        "probability_failures": 0,
    }
    for field, expected in expected_counts.items():
        if manifest.get(field) != expected:
            raise HFAWQTripletGateError(
                f"{precision} manifest count mismatch at {field}"
            )
    run_identity = manifest.get("run_identity")
    if not isinstance(run_identity, Mapping):
        raise HFAWQTripletGateError(f"{precision} run identity is missing")
    expected_identity = {
        "protocol_version": protocol,
        "split": split,
        "audit_certificate_sha256": certificate_sha256,
        "model_key": model_key,
        "quantization": precision,
        "expected_full": controls["full"],
        "expected_no_evidence": controls["no_evidence"],
        "expected_knowledge_only": controls["knowledge_only"],
    }
    for field, expected in expected_identity.items():
        if run_identity.get(field) != expected:
            raise HFAWQTripletGateError(
                f"{precision} run identity mismatch at {field}"
            )


def _index_rows(
    rows: list[dict[str, Any]], expected_total: int
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_total:
        raise HFAWQTripletGateError(
            f"structural record count must be {expected_total}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        owner = row.get("owner_key")
        if not isinstance(owner, str) or not owner or owner in indexed:
            raise HFAWQTripletGateError("structural owner keys must be unique")
        indexed[owner] = row
    return indexed


def _reject_recovery_path(path: Path, label: str) -> None:
    if any("recovery" in part.casefold() for part in path.parts):
        raise HFAWQTripletGateError(f"{label} must not use a recovery path")


def _reject_aggregate_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_AGGREGATES:
                raise HFAWQTripletGateError("aggregate metric key is forbidden")
            _reject_aggregate_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_aggregate_keys(child)


def _normalize_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize the sole backward-compatible metadata default."""
    normalized = dict(metadata)
    normalized.setdefault("mapping_variant", "original")
    return normalized


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFAWQTripletGateError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise HFAWQTripletGateError(f"{label} must be a JSON object")
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise HFAWQTripletGateError("cannot hash AWQ artifact audit") from error
    return digest.hexdigest()


def _record_hash(record: Mapping[str, Any], *, excluded: str) -> str:
    payload = json.dumps(
        {key: value for key, value in record.items() if key != excluded},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_atomic_json(record: Mapping[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(record),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "formal", "qwen_smoke16", "qwen_formal4800"),
        default="smoke",
    )
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--fp16-export", type=Path, required=True)
    parser.add_argument("--gptq-export", type=Path, required=True)
    parser.add_argument("--awq-export", type=Path, required=True)
    parser.add_argument("--fp16-audit", type=Path, required=True)
    parser.add_argument("--gptq-audit", type=Path, required=True)
    parser.add_argument("--awq-audit", type=Path, required=True)
    parser.add_argument("--awq-artifact-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-freeze-config", type=Path)
    parser.add_argument("--qwen-freeze-sha256", type=Path)
    args = parser.parse_args(argv)
    result = validate_hf_awq_triplet(
        fp16_export=args.fp16_export,
        gptq_export=args.gptq_export,
        awq_export=args.awq_export,
        fp16_audit=args.fp16_audit,
        gptq_audit=args.gptq_audit,
        awq_audit=args.awq_audit,
        awq_artifact_audit=args.awq_artifact_audit,
        model_key=args.model_key,
        mode=args.mode,
        output_path=args.output,
        qwen_freeze_config=args.qwen_freeze_config,
        qwen_freeze_sha256_path=args.qwen_freeze_sha256,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
