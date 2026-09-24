"""Bounded fixed-artifact process-repeatability analysis; no significance tests.

The caller must verify run manifests, artifact/environment certificates, and the
frozen selection before calling this numerical API. This module independently
gates the complete matrix, process identities, and exported row identities.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import product
import math
import re
import statistics
import uuid


MODELS = ("qwen35_9b", "ministral3_8b", "olmo3_7b")
ROUTES = ("FP16", "GPTQ_INT4", "AWQ_INT4")
MAPPINGS = {
    "original": dict(zip("ABC", ("SUPPORTS", "REFUTES", "NOT ENOUGH INFO"))),
    "cycle_1": dict(zip("ABC", ("REFUTES", "NOT ENOUGH INFO", "SUPPORTS"))),
    "cycle_2": dict(zip("ABC", ("NOT ENOUGH INFO", "SUPPORTS", "REFUTES"))),
}
WEIGHTINGS = ("page_balanced", "quartet_weighted")
CONTRASTS = {"AWQ_minus_FP16": "FP16", "AWQ_minus_GPTQ": "GPTQ_INT4"}
LOG_SCORE_THRESHOLD = 1e-6
NEAR_ZERO_BASELINE = 1e-12
HASH_FIELDS = ("generation_input_ids_sha256", "prompt_sha256", "generation_prompt_sha256")
IDENTITY_FIELDS = ("case_id", "page", "cell_index", "negative_label")


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("choice log scores must be finite numbers")
    if not math.isfinite(value):
        raise ValueError("choice log scores must be finite numbers")
    return float(value)


def _validate(runs, expected_cases):
    if type(expected_cases) is not int or expected_cases < 1:
        raise ValueError("expected_cases must be a positive integer")
    if not isinstance(runs, list) or len(runs) != 81:
        raise ValueError("complete 27-condition by three-repeat matrix is required")
    indexed, invocation_ids = {}, set()
    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("run must be a dictionary")
        if type(run.get("repeat")) is not int or run["repeat"] not in (1, 2, 3):
            raise ValueError("repeat must be 1, 2 or 3")
        key = (run.get("mapping"), run.get("model"), run.get("route"), run["repeat"])
        if (key[0] not in MAPPINGS or key[1] not in MODELS or key[2] not in ROUTES
                or key in indexed):
            raise ValueError("invalid or duplicate matrix cell")
        try:
            invocation = str(uuid.UUID(run.get("invocation_id", "")))
        except (ValueError, AttributeError, TypeError) as exc:
            raise ValueError("invocation_id must be a UUID") from exc
        if invocation in invocation_ids:
            raise ValueError("invocation_id is reused; cached output is not an independent repeat")
        invocation_ids.add(invocation)
        if type(run.get("pid")) is not int or run["pid"] < 1:
            raise ValueError("pid must be a positive integer")
        rows = run.get("rows")
        if not isinstance(rows, list) or len(rows) != 4 * expected_cases:
            raise ValueError("each run requires exactly four inputs per expected case")
        owners, cells, cases = set(), set(), defaultdict(list)
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("row must be a dictionary")
            for field in ("owner_key", "case_id", "page"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"missing or invalid {field}")
            cell = row.get("cell_index")
            if type(cell) is not int or cell not in range(4):
                raise ValueError("quartet cell_index must be 0..3")
            identity = (row["case_id"], cell)
            if row["owner_key"] in owners or identity in cells:
                raise ValueError("duplicate owner or quartet cell")
            owners.add(row["owner_key"])
            cells.add(identity)
            cases[row["case_id"]].append(row)
            if row.get("negative_label") not in ("REFUTES", "NOT ENOUGH INFO"):
                raise ValueError("invalid negative_label")
            if row.get("choice_to_label") != MAPPINGS[key[0]]:
                raise ValueError("choice mapping differs from the frozen mapping")
            choice = row.get("scored_choice")
            if choice not in ("A", "B", "C") or row.get("scored_label") != MAPPINGS[key[0]][choice]:
                raise ValueError("invalid hard choice or inconsistent mapped label")
            scores = row.get("choice_logprobs")
            if not isinstance(scores, dict) or set(scores) != set("ABC"):
                raise ValueError("choice log scores must contain exactly A/B/C")
            for score in scores.values():
                _finite(score)
            for field in HASH_FIELDS:
                if not isinstance(row.get(field), str) or not re.fullmatch(r"[0-9a-f]{64}", row[field]):
                    raise ValueError(f"invalid {field}")
        if len(cases) != expected_cases or any(
            len(items) != 4 or len({(row["page"], row["negative_label"]) for row in items}) != 1
            for items in cases.values()
        ):
            raise ValueError("incomplete quartet or inconsistent quartet metadata")
        indexed[key] = run
    if set(indexed) != set(product(MAPPINGS, MODELS, ROUTES, (1, 2, 3))):
        raise ValueError("incomplete matrix")

    canonical = indexed[("original", MODELS[0], "FP16", 1)]["rows"]
    common_identity = [tuple(row[field] for field in IDENTITY_FIELDS) for row in canonical]
    for mapping, model, route in product(MAPPINGS, MODELS, ROUTES):
        repeats = [indexed[(mapping, model, route, repeat)] for repeat in (1, 2, 3)]
        if len({run["pid"] for run in repeats}) != 3:
            raise ValueError("condition repeats must have distinct process IDs")
        baseline = repeats[0]["rows"]
        for run in repeats:
            if [tuple(row[field] for field in IDENTITY_FIELDS) for row in run["rows"]] != common_identity:
                raise ValueError("case identities, metadata and row order must agree across all runs")
            for row, base in zip(run["rows"], baseline):
                # Scores and decisions are outcomes, not metadata. All other exported
                # fields must stay fixed between repeats of this condition.
                outcome_fields = {"choice_logprobs", "scored_choice", "scored_label"}
                if ({key: value for key, value in row.items() if key not in outcome_fields}
                        != {key: value for key, value in base.items() if key not in outcome_fields}):
                    raise ValueError("within-condition metadata, owners, tokens or prompts differ")
        reference_rows = indexed[(mapping, model, "FP16", 1)]["rows"]
        if any(any(row[field] != base[field] for field in HASH_FIELDS)
               for row, base in zip(baseline, reference_rows)):
            raise ValueError("tokens and prompts must match across routes of the same model/mapping")
    return indexed, {row["case_id"]: row["page"] for row in canonical}


def _span(values):
    return max(values) - min(values)


def _span_distribution(spans):
    ordered = sorted(spans)
    position = (len(ordered) - 1) * 0.99
    lower, upper = math.floor(position), math.ceil(position)
    weight = position - lower
    return {"max": ordered[-1], "p99": ordered[lower] * (1 - weight) + ordered[upper] * weight,
            "median": statistics.median(ordered)}


def _series(values, undefined_status="undefined_degenerate_reference_scale", center=0.0):
    if any(value is None for value in values):
        return {"values_by_repeat": values, "span": None, "status": undefined_status,
                "direction_stable": None, "span_to_baseline_ratio": None,
                "ratio_status": "undefined_endpoint"}
    if not all(math.isfinite(value) for value in values):
        raise ValueError("derived analysis values overflowed")
    span = _span(values)
    baseline = values[0]
    near_zero = abs(baseline) <= NEAR_ZERO_BASELINE
    return {"values_by_repeat": values, "span": span, "status": "defined",
            "direction_stable": len({(value > center) - (value < center) for value in values}) == 1,
            "span_to_baseline_ratio": None if near_zero else span / abs(baseline),
            "ratio_status": "undefined_near_zero_baseline" if near_zero else "defined"}


def _mean(values, pages, weighting):
    if weighting == "quartet_weighted":
        return statistics.fmean(values.values())
    grouped = defaultdict(list)
    for case, value in values.items():
        grouped[pages[case]].append(value)
    return statistics.fmean(statistics.fmean(group) for group in grouped.values())


def summarize_runs(runs: list[dict], expected_cases=120) -> dict:
    """Summarize all 81 fresh, complete runs without altering inputs or files.

    ``repeat`` is 1..3; ``model``/``route``/``mapping`` use the frozen names in
    MODELS, ROUTES and MAPPINGS. Flip denominators count unique condition-input
    combinations (any change across three runs), not repeat-pair comparisons.
    S is route-mean difference divided by reference quartet sample SD (ddof=1).
    D is the weighted mean of 1(delta>0)+0.5(delta==0); its direction is relative
    to 0.5. Cross-model endpoints are left-minus-right S/D, centered at zero.
    """
    indexed, pages = _validate(runs, expected_cases)
    conditions, interactions = {}, {}
    all_log_spans, all_margin_spans, all_interaction_spans = [], [], []
    flip_count = 0
    for mapping, model, route in product(MAPPINGS, MODELS, ROUTES):
        row_sets = [indexed[(mapping, model, route, repeat)]["rows"] for repeat in (1, 2, 3)]
        log_spans, margin_spans, flips = [], [], 0
        margins = [defaultdict(dict) for _ in range(3)]
        for parallel_rows in zip(*row_sets):
            flips += len({row["scored_label"] for row in parallel_rows}) > 1
            for choice in "ABC":
                log_spans.append(_span([row["choice_logprobs"][choice] for row in parallel_rows]))
            row_margins = []
            for repeat_index, row in enumerate(parallel_rows):
                by_label = {label: row["choice_logprobs"][choice] for choice, label in row["choice_to_label"].items()}
                margin = by_label["SUPPORTS"] - by_label[row["negative_label"]]
                if not math.isfinite(margin):
                    raise ValueError("derived margin overflowed")
                margins[repeat_index][row["case_id"]][row["cell_index"]] = margin
                row_margins.append(margin)
            margin_spans.append(_span(row_margins))
        case_series = {}
        for case in pages:
            values = []
            for repeat_index in range(3):
                m = margins[repeat_index][case]
                values.append(0.5 * math.fsum((m[0], m[3], -m[1], -m[2])))
            case_series[case] = _series(values)
        interactions[(mapping, model, route)] = case_series
        interaction_spans = [item["span"] for item in case_series.values()]
        conditions[f"{mapping}/{model}/{route}"] = {
            "log_score_span": _span_distribution(log_spans),
            "margin_span": _span_distribution(margin_spans),
            "interaction_span": _span_distribution(interaction_spans),
            "interaction_by_case": case_series,
            "hard_label_flips": {"count": flips, "denominator": expected_cases * 4, "rate": flips / (expected_cases * 4)},
        }
        all_log_spans.extend(log_spans)
        all_margin_spans.extend(margin_spans)
        all_interaction_spans.extend(interaction_spans)
        flip_count += flips

    route_effects, cross_effects, directions = {}, {}, []
    for mapping in MAPPINGS:
        route_effects[mapping], cross_effects[mapping] = {}, {}
        for model in MODELS:
            route_effects[mapping][model] = {}
            for contrast, reference_route in CONTRASTS.items():
                target = route_effects[mapping][model][contrast] = {}
                for weighting in WEIGHTINGS:
                    raw, standardized, directional, reference_sds = [], [], [], []
                    for repeat_index in range(3):
                        reference = {case: interactions[(mapping, model, reference_route)][case]["values_by_repeat"][repeat_index] for case in pages}
                        delta = {case: interactions[(mapping, model, "AWQ_INT4")][case]["values_by_repeat"][repeat_index] - reference[case] for case in pages}
                        mean = _mean(delta, pages, weighting)
                        sd = statistics.stdev(reference.values()) if len(reference) > 1 else 0.0
                        reference_sds.append(sd)
                        raw.append(mean)
                        standardized.append(mean / sd if sd > NEAR_ZERO_BASELINE else None)
                        directional.append(_mean({case: 1.0 if value > 0 else 0.5 if value == 0 else 0.0 for case, value in delta.items()}, pages, weighting))
                    target[weighting] = {"route_difference": _series(raw), "S": _series(standardized),
                                         "D": _series(directional, center=0.5), "reference_sd_by_repeat": reference_sds}
                    directions.extend(target[weighting][endpoint]["direction_stable"] for endpoint in ("route_difference", "S", "D"))
        for left, right in ((MODELS[1], MODELS[0]), (MODELS[2], MODELS[0]), (MODELS[2], MODELS[1])):
            pair = cross_effects[mapping][f"{left}_minus_{right}"] = {}
            for contrast in CONTRASTS:
                pair[contrast] = {}
                for weighting in WEIGHTINGS:
                    pair[contrast][weighting] = {}
                    for endpoint in ("S", "D"):
                        a = route_effects[mapping][left][contrast][weighting][endpoint]["values_by_repeat"]
                        b = route_effects[mapping][right][contrast][weighting][endpoint]["values_by_repeat"]
                        series = _series([x - y if x is not None and y is not None else None for x, y in zip(a, b)])
                        pair[contrast][weighting][endpoint] = series
                        directions.append(series["direction_stable"])
    log_summary = _span_distribution(all_log_spans)
    directions_stable = False if False in directions else None if None in directions else True
    passed = (False if log_summary["max"] > LOG_SCORE_THRESHOLD or flip_count or directions_stable is False
              else None if directions_stable is None else True)
    exact = log_summary["max"] == 0.0 and flip_count == 0
    classification = ("identical_outputs" if exact else "within_engineering_threshold" if passed
                      else "requires_investigation")
    return {
        "analysis_protocol": "hf-repeatability-analysis-v1-20260911",
        "population": {"runs": 81, "conditions": 27, "repeats": 3, "quartets": expected_cases,
                       "pages": len(set(pages.values())), "inputs_per_run": 4 * expected_cases},
        "conditions": conditions, "log_score_span": log_summary,
        "margin_span": _span_distribution(all_margin_spans),
        "interaction_span": _span_distribution(all_interaction_spans),
        "hard_label_flips": {"count": flip_count, "denominator": 27 * 4 * expected_cases,
                             "rate": flip_count / (27 * 4 * expected_cases)},
        "route_effects": route_effects, "cross_model_effects": cross_effects,
        "classification": classification,
        "engineering_gate": {"passed": passed, "absolute_log_score_span_threshold": LOG_SCORE_THRESHOLD,
                             "threshold_role": "engineering_screen_not_statistical_equivalence",
                             "effect_directions_stable": directions_stable,
                             "undefined_effects": sum(direction is None for direction in directions)},
        "definitions": {"interaction": "0.5*(m00+m11-m01-m10)", "margin": "SUPPORTS-minus-negative_label",
                        "S": "weighted_mean(AWQ-reference)/quartet_sample_sd(reference,ddof=1)",
                        "D": "weighted_mean(1(delta>0)+0.5(delta==0))",
                        "page_balanced": "equal page weights after within-page quartet means",
                        "quartet_weighted": "equal quartet weights",
                        "near_zero_ratio_baseline_threshold": NEAR_ZERO_BASELINE,
                        "span_ratio": "repeat_span/abs(repeat_1_value), undefined near zero"},
        "limitations": {"scope": "selected_subset_fixed_artifacts_fixed_environment_three_process_runs",
                        "full_population_significance": "not_assessed", "requantization": "not_assessed",
                        "hardware_generalization": "not_assessed", "input_generalization": "not_assessed",
                        "historical_comparison": "not_performed", "new_significance_tests": "not_performed"},
    }
