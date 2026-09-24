"""Structural-only gate for the Qwen FP16/GPTQ paired HF smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .hf_model_identity import HFModelIdentityError, hf_model_identity
from .hf_runtime_audit import load_hf_runtime_audit


HF_PAIR_GATE_PROTOCOL = "hf-pair-structural-gate-v2-20260718"
HF_FORMAL_GATE_PROTOCOL = "hf-pair-formal-gate-v1-20260718"
_MODE_SPECS = {
    "smoke": {
        "gate_protocol": HF_PAIR_GATE_PROTOCOL,
        "run_protocol": "hf-vitaminc-smoke-v2-20260718",
        "split": "pilot",
        "controls": {"full": 8, "no_evidence": 4, "knowledge_only": 4},
    },
    "formal": {
        "gate_protocol": HF_FORMAL_GATE_PROTOCOL,
        "run_protocol": "hf-vitaminc-formal-v1-20260718",
        "split": "dose_subset",
        "controls": {
            "full": 4800,
            "no_evidence": 2400,
            "knowledge_only": 2400,
        },
    },
}
_ALIGNMENT_FIELDS = (
    "prompt_sha256",
    "generation_input_ids_sha256",
    "generation_prompt_sha256",
    "token_ids",
)
_FORBIDDEN_AGGREGATES = {
    "accuracy",
    "balanced_accuracy",
    "brier",
    "ece",
    "effect",
    "f1",
    "mcnemar",
}


class HFPairGateError(RuntimeError):
    """Raised when the two smoke conditions are not an exact structural pair."""


def validate_hf_pair(
    *,
    fp16_export: str | Path,
    gptq_export: str | Path,
    fp16_audit: str | Path,
    gptq_audit: str | Path,
    output_path: str | Path | None = None,
    mode: str = "smoke",
    model_key: str = "qwen35_9b",
) -> dict[str, Any]:
    """Validate pairing without calculating or reporting outcome metrics."""
    if mode not in _MODE_SPECS:
        raise HFPairGateError("pair gate mode must be smoke or formal")
    try:
        identity = hf_model_identity(model_key)
    except HFModelIdentityError as error:
        raise HFPairGateError(str(error)) from error
    spec = dict(_MODE_SPECS[mode])
    if mode == "smoke":
        spec["gate_protocol"] = identity.smoke_gate_protocol
        spec["run_protocol"] = identity.smoke_protocol
    else:
        spec["gate_protocol"] = identity.formal_gate_protocol
        spec["run_protocol"] = identity.formal_protocol
    expected_controls = spec["controls"]
    expected_total = sum(expected_controls.values())
    fp_certificate = load_hf_runtime_audit(fp16_audit)
    gptq_certificate = load_hf_runtime_audit(gptq_audit)
    if (
        fp_certificate.model_key != identity.model_key
        or gptq_certificate.model_key != identity.model_key
    ):
        raise HFPairGateError("audit model identity does not match the gate")
    if (
        fp_certificate.precision != "FP16"
        or gptq_certificate.precision != "GPTQ_INT4"
    ):
        raise HFPairGateError("audit precisions do not form FP16/GPTQ_INT4 pair")
    for field in (
        "protocol_version",
        "model_key",
        "package_versions",
        "prompt_contract_sha256",
        "choice_token_ids",
        "generation_input_ids_sha256",
        "generation_prompt_sha256",
        "chat_template_sha256",
        "direct_logit_method",
    ):
        if getattr(fp_certificate, field) != getattr(gptq_certificate, field):
            raise HFPairGateError(f"audit pairing drifted at {field}")

    fp_manifest, fp_rows = _load_export(Path(fp16_export))
    gptq_manifest, gptq_rows = _load_export(Path(gptq_export))
    _validate_manifest(
        fp_manifest,
        fp_certificate.certificate_sha256,
        "FP16",
        spec,
        identity.model_key,
    )
    _validate_manifest(
        gptq_manifest,
        gptq_certificate.certificate_sha256,
        "GPTQ_INT4",
        spec,
        identity.model_key,
    )
    fp_identity = fp_manifest["run_identity"]
    gptq_identity = gptq_manifest["run_identity"]
    for field in (
        "protocol_version",
        "split",
        "split_sha256",
        "prompt_contract_sha256",
        "model_key",
        "expected_full",
        "expected_no_evidence",
        "expected_knowledge_only",
    ):
        if fp_identity.get(field) != gptq_identity.get(field):
            raise HFPairGateError(f"run identity pairing drifted at {field}")

    fp_by_owner = _index_rows(fp_rows, expected_total)
    gptq_by_owner = _index_rows(gptq_rows, expected_total)
    if set(fp_by_owner) != set(gptq_by_owner):
        raise HFPairGateError("owner-set alignment drifted")
    controls = {"full": 0, "no_evidence": 0, "knowledge_only": 0}
    for owner in sorted(fp_by_owner):
        fp_row = fp_by_owner[owner]
        gptq_row = gptq_by_owner[owner]
        if fp_row.get("metadata") != gptq_row.get("metadata"):
            raise HFPairGateError(f"metadata alignment drifted for {owner}")
        if fp_row.get("control") != gptq_row.get("control"):
            raise HFPairGateError(f"control alignment drifted for {owner}")
        control = fp_row.get("control")
        if control not in controls:
            raise HFPairGateError(f"invalid control for {owner}")
        controls[control] += 1
        for stage in ("hard", "probability"):
            if (
                fp_row.get(f"{stage}_status") != "ok"
                or gptq_row.get(f"{stage}_status") != "ok"
            ):
                raise HFPairGateError(f"{stage} condition failed for {owner}")
            fp_payload = fp_row.get(f"{stage}_payload")
            gptq_payload = gptq_row.get(f"{stage}_payload")
            if not isinstance(fp_payload, Mapping) or not isinstance(
                gptq_payload, Mapping
            ):
                raise HFPairGateError(f"{stage} payload missing for {owner}")
            for field in _ALIGNMENT_FIELDS:
                if fp_payload.get(field) != gptq_payload.get(field):
                    raise HFPairGateError(
                        f"{stage} alignment drifted at {field} for {owner}"
                    )
        for row in (fp_row, gptq_row):
            if (
                row["hard_payload"].get("choice_logprobs")
                != row["probability_payload"].get("choice_logprobs")
            ):
                raise HFPairGateError(
                    f"one-forward stage alignment drifted for {owner}"
                )
    if controls != expected_controls:
        raise HFPairGateError("control-count alignment drifted")

    result = {
        "protocol_version": spec["gate_protocol"],
        "gate_passed": True,
        "paired_owners": len(fp_by_owner),
        "controls": controls,
        "fp16_audit_certificate_sha256": fp_certificate.certificate_sha256,
        "gptq_audit_certificate_sha256": gptq_certificate.certificate_sha256,
        "split_sha256": fp_identity["split_sha256"],
    }
    if output_path is not None:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp")
        temporary.write_text(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        temporary.replace(destination)
    return result


def _load_export(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = _load_json_object(root / "manifest.json", "manifest")
    supplied_hash = manifest.get("manifest_sha256")
    if supplied_hash != _record_hash(manifest, excluded="manifest_sha256"):
        raise HFPairGateError("manifest hash mismatch")
    _reject_aggregate_keys(manifest)
    records_path = root / "records.jsonl"
    try:
        records_bytes = records_path.read_bytes()
        rows = [
            json.loads(line)
            for line in records_bytes.decode("utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFPairGateError("cannot read structural records") from error
    if manifest.get("records_jsonl_sha256") != hashlib.sha256(
        records_bytes
    ).hexdigest():
        raise HFPairGateError("records hash mismatch")
    if any(not isinstance(row, dict) for row in rows):
        raise HFPairGateError("structural records must be JSON objects")
    return manifest, rows


def _validate_manifest(
    manifest: Mapping[str, Any],
    certificate_sha256: str,
    precision: str,
    spec: Mapping[str, Any],
    model_key: str,
) -> None:
    controls = spec["controls"]
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
    for key, expected in expected_counts.items():
        if manifest.get(key) != expected:
            raise HFPairGateError(f"{precision} manifest count mismatch at {key}")
    identity = manifest.get("run_identity")
    if not isinstance(identity, Mapping):
        raise HFPairGateError(f"{precision} run identity is missing")
    expected = {
        "protocol_version": spec["run_protocol"],
        "split": spec["split"],
        "audit_certificate_sha256": certificate_sha256,
        "prompt_contract_sha256": None,
        "model_key": model_key,
        "quantization": precision,
        "expected_full": controls["full"],
        "expected_no_evidence": controls["no_evidence"],
        "expected_knowledge_only": controls["knowledge_only"],
    }
    for key, value in expected.items():
        if value is not None and identity.get(key) != value:
            raise HFPairGateError(
                f"{precision} run identity mismatch at {key}"
            )


def _index_rows(
    rows: list[dict[str, Any]], expected_total: int
) -> dict[str, dict[str, Any]]:
    if len(rows) != expected_total:
        raise HFPairGateError(
            f"structural record count must be {expected_total}"
        )
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        owner = row.get("owner_key")
        if not isinstance(owner, str) or not owner or owner in indexed:
            raise HFPairGateError("structural owner keys must be unique")
        indexed[owner] = row
    return indexed


def _reject_aggregate_keys(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if str(key).casefold() in _FORBIDDEN_AGGREGATES:
                raise HFPairGateError("aggregate metric key is forbidden")
            _reject_aggregate_keys(child)
    elif isinstance(value, list):
        for child in value:
            _reject_aggregate_keys(child)


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise HFPairGateError(f"cannot read {label}") from error
    if not isinstance(value, dict):
        raise HFPairGateError(f"{label} must be a JSON object")
    return value


def _record_hash(record: Mapping[str, Any], *, excluded: str) -> str:
    clean = {key: value for key, value in record.items() if key != excluded}
    payload = json.dumps(
        clean,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fp16-export", type=Path, required=True)
    parser.add_argument("--gptq-export", type=Path, required=True)
    parser.add_argument("--fp16-audit", type=Path, required=True)
    parser.add_argument("--gptq-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("smoke", "formal"), default="smoke"
    )
    parser.add_argument("--model-key", default="qwen35_9b")
    args = parser.parse_args(argv)
    result = validate_hf_pair(
        fp16_export=args.fp16_export,
        gptq_export=args.gptq_export,
        fp16_audit=args.fp16_audit,
        gptq_audit=args.gptq_audit,
        output_path=args.output,
        mode=args.mode,
        model_key=args.model_key,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
