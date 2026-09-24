"""Render publication evidence into paper-ready LaTeX artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any


MODEL_LABELS = {
    "qwen35_4b": "Qwen3.5-4B",
    "qwen35_9b": "Qwen3.5-9B",
    "ministral3_3b": "Ministral-3-3B",
    "ministral3_8b": "Ministral-3-8B",
    "gemma4_e4b": "Gemma-4-E4B",
    "olmo3_7b": "OLMo-3-7B",
    "qwen35_9b_x_gemma4_e4b": "Gemma route minus Qwen route",
}


def latex_escape(value: object) -> str:
    """Escape LaTeX-reserved characters in plain-text values."""
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def render_tables(evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Render four deterministic LaTeX tables and cell-level provenance."""
    provenance: dict[str, dict[str, Any]] = {}
    files = {
        "tables/main_confirmatory.tex": _render_main_table(
            evidence["main"], provenance
        ),
        "tables/cross_family.tex": _render_cross_family_table(
            evidence["stage_a"], provenance
        ),
        "tables/route_robustness.tex": _render_route_table(
            evidence["hf_gptq"], evidence["route_interaction"],
            evidence["qwen_awq_formal"], provenance
        ),
        "tables/external_validation.tex": _render_external_table(
            evidence["tabfact"], evidence["cub"], provenance
        ),
    }
    return {"files": files, "provenance": provenance}


def render_paper_files(evidence: Mapping[str, Any]) -> dict[str, str]:
    """Render every importable manuscript artifact except the package manifest."""
    rendered_tables = render_tables(evidence)
    provenance = dict(rendered_tables["provenance"])
    provenance.update(
        {
            "methods.bootstrap_replicates": {
                "file": "sections/methods.tex",
                "protocol_source": "docs/experiment_design_v2.md#7",
                "protocol_field": "bootstrap_replicates",
                "rendered_value": "10,000",
            },
            "methods.bootstrap_seed": {
                "file": "sections/methods.tex",
                "protocol_source": "docs/experiment_design_v2.md#7",
                "protocol_field": "bootstrap_seed",
                "rendered_value": "20260711",
            },
            "methods.interval": {
                "file": "sections/methods.tex",
                "protocol_source": "docs/experiment_design_v2.md#7",
                "protocol_field": "confidence_interval",
                "rendered_value": "two-sided 95% percentile interval",
            },
        }
    )
    results = render_results(evidence, provenance)
    files = {
        "main.tex": render_main_tex(),
        "sections/methods.tex": render_methods(evidence, provenance),
        "sections/results.tex": results,
        **rendered_tables["files"],
        "claims/claim_boundary.md": render_claim_boundary(),
        "README_zh.md": render_readme_zh(),
    }
    files["provenance/table_source_map.json"] = json.dumps(
        provenance,
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return files


def render_main_tex() -> str:
    return r"""\documentclass[10pt]{article}
\usepackage[margin=1in]{geometry}
\usepackage{amsmath}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage[T1]{fontenc}
\usepackage[hidelinks]{hyperref}
\title{Paper-Ready Methods and Results Package}
\author{}
\date{}
\begin{document}
\maketitle
\input{sections/methods}
\input{sections/results}
\end{document}
"""


def render_methods(evidence, provenance) -> str:
    main = evidence["main"]
    stage = evidence["stage_a"][0]
    tabfact = evidence["tabfact"][0]
    main_quartets = _result_cell(
        provenance, main, "population.quartets",
        f"{main['population']['quartets']:,}", "methods.confirmatory_quartets",
        file_path="sections/methods.tex",
    )
    main_pages = _result_cell(
        provenance, main, "population.pages",
        f"{main['population']['pages']:,}", "methods.confirmatory_pages",
        file_path="sections/methods.tex",
    )
    stage_quartets = _result_cell(
        provenance, stage, "population.quartets",
        f"{stage['population']['quartets']:,}", "methods.stage_a_quartets",
        file_path="sections/methods.tex",
    )
    stage_pages = _result_cell(
        provenance, stage, "population.pages",
        f"{stage['population']['pages']:,}", "methods.stage_a_pages",
        file_path="sections/methods.tex",
    )
    tabfact_claims = _result_cell(
        provenance, tabfact, "population.claims",
        f"{tabfact['population']['claims']:,}", "methods.tabfact_claims",
        file_path="sections/methods.tex",
    )
    tabfact_tables = _result_cell(
        provenance, tabfact, "population.tables",
        f"{tabfact['population']['tables']:,}", "methods.tabfact_tables",
        file_path="sections/methods.tex",
    )
    qwen_awq = evidence["qwen_awq_formal"]
    qwen_analysis = qwen_awq["analysis"]
    qwen_quartets = _qwen_cell(
        provenance, qwen_awq, "/analysis/population/quartets",
        f"{qwen_analysis['population']['quartets']:,}",
        "qwen_awq.population.quartets", file_path="sections/methods.tex",
    )
    qwen_pages = _qwen_cell(
        provenance, qwen_awq, "/analysis/population/pages",
        f"{qwen_analysis['population']['pages']:,}",
        "qwen_awq.population.pages", file_path="sections/methods.tex",
    )
    qwen_draws = _qwen_cell(
        provenance, qwen_awq, "/analysis/bootstrap_contract/draws",
        f"{qwen_analysis['bootstrap_contract']['draws']:,}",
        "qwen_awq.bootstrap.draws", file_path="sections/methods.tex",
    )
    qwen_seed = _qwen_cell(
        provenance, qwen_awq, "/analysis/bootstrap_contract/seed",
        str(qwen_analysis["bootstrap_contract"]["seed"]),
        "qwen_awq.bootstrap.seed", file_path="sections/methods.tex",
    )

    return f"""\\section{{Methods}}
\\subsection{{Study design and evidence units}}
We evaluated post-training quantization in evidence-grounded fact verification
using frozen claim--evidence contrasts. The primary VitaminC analysis used
strict Cartesian quartets containing two claims and two evidence passages, with
the label pattern oriented so that supported pairs occupied the diagonal. For
each quartet, we computed the supported-versus-alternative log-odds contrast
and the difference-in-differences interaction
\\begin{{equation}}
I_q=(g_{{00}}-g_{{01}}-g_{{10}}+g_{{11}})/2.
\\end{{equation}}
The primary estimand was the page-balanced mean of the paired quantized-minus-
high-precision difference in $I_q$. Quartet-weighted estimates were treated as
sensitivity analyses.

The confirmatory set contained {main_quartets} quartets and {main_pages} pages.
The common-support Stage A subset contained {stage_quartets} quartets and {stage_pages} pages.
Fresh TabFact contained {tabfact_claims} claims from {tabfact_tables} tables.

\\subsection{{Models and deployment routes}}
The frozen registry covered Qwen3.5-4B and 9B, Ministral-3-3B and 8B,
Gemma-4-E4B, and OLMo-3-7B. GGUF F16 and Q4\\_K\\_M constituted the primary
deployment contrast. Q5\\_K\\_M and Q8\\_0 were exploratory deployment-format
extensions. Within the Hugging Face backend, FP16 and GPTQ INT4 formed a
separate paired route. For Qwen3.5-9B, the frozen \\texttt{{qwen\\_formal4800}}
Gate additionally aligned FP16, GPTQ INT4, and AWQ INT4 on {qwen_quartets}
quartets and {qwen_pages} page clusters. Cross-route contrasts were interpreted jointly because
quantizer, backend, packing, and kernel differed together.

\\subsection{{Outcomes and statistical analysis}}
The confirmatory continuous endpoint was $\\Delta I$. Hard-label accuracy and
balanced accuracy were behavioral outcomes. CUB/DRUID BCU and CCU were retained
as secondary context-utilisation diagnostics. VitaminC inference used page as
the clustering unit; Fresh TabFact used table as the clustering unit. Paired
cluster bootstrap inference used 10,000 replicates, seed 20260711, and
two-sided 95\\% percentile confidence intervals. Prespecified families used Holm
correction. Confirmatory, exploratory, robustness, and external-validation
analyses remained distinct throughout reporting.

The Qwen HF triplet used the same page-cluster estimand with {qwen_draws}
shared bootstrap draws and seed {qwen_seed}. Its two prespecified contrasts,
AWQ minus FP16 and AWQ minus GPTQ, formed one two-test Holm family. The
confidence intervals were unadjusted descriptive 95\\% percentile intervals.

Parsing or backend failures in hard-label analyses were counted as errors rather
than removed as complete cases; continuous logit outcomes retained explicit
coverage reporting and worst-case sensitivity analysis.

The behavioral design does not directly identify parametric memory or an
internal neural mechanism. Claim-only and context-utilisation measurements are
therefore described as behavioral diagnostics.
"""


def render_results(evidence, provenance) -> str:
    main = evidence["main"]
    main_delta = _result_cell(
        provenance, main, "metrics.delta_i", _signed(main["metrics"]["delta_i"], 3),
        "results.main.delta_i",
    )
    main_ci = _result_cell(
        provenance, main, "metrics.delta_i_ci95",
        _interval(main["metrics"]["delta_i_ci95"], digits=3),
        "results.main.delta_i_ci95",
    )
    main_ba = _result_cell(
        provenance, main, "metrics.balanced_accuracy_difference",
        _signed(100 * main["metrics"]["balanced_accuracy_difference"], 2),
        "results.main.ba_pp",
    )

    olmo = next(row for row in evidence["stage_a"] if row["model_key"] == "olmo3_7b")
    olmo_ba = _result_cell(
        provenance, olmo, "metrics.balanced_accuracy_difference",
        _signed(100 * olmo["metrics"]["balanced_accuracy_difference"], 2),
        "results.olmo.ba_pp",
    )
    olmo_p = _result_cell(
        provenance, olmo, "metrics.balanced_accuracy_holm_pvalue",
        _pvalue_number(olmo["metrics"]["balanced_accuracy_holm_pvalue"]),
        "results.olmo.holm_p",
    )

    qwen_hf = next(row for row in evidence["hf_gptq"] if row["model_key"] == "qwen35_9b")
    gemma_hf = next(row for row in evidence["hf_gptq"] if row["model_key"] == "gemma4_e4b")
    qwen_hf_delta = _result_cell(
        provenance, qwen_hf, "metrics.delta_i",
        _signed(qwen_hf["metrics"]["delta_i"], 3), "results.qwen_hf.delta_i",
    )
    gemma_hf_delta = _result_cell(
        provenance, gemma_hf, "metrics.delta_i",
        _signed(gemma_hf["metrics"]["delta_i"], 3), "results.gemma_hf.delta_i",
    )
    interaction = evidence["route_interaction"]
    interaction_value = _result_cell(
        provenance, interaction, "metrics.four_way_interaction",
        _signed(interaction["metrics"]["four_way_interaction"], 3),
        "results.route_interaction.value",
    )
    interaction_ci = _result_cell(
        provenance, interaction, "metrics.ci95",
        _interval(interaction["metrics"]["ci95"], digits=3),
        "results.route_interaction.ci95",
    )
    qwen_awq = evidence["qwen_awq_formal"]
    qwen_analysis = qwen_awq["analysis"]
    qwen_fp16 = qwen_analysis["contrasts"]["awq_minus_fp16"]
    qwen_gptq = qwen_analysis["contrasts"]["awq_minus_gptq"]
    awq_fp16_effect = _qwen_cell(
        provenance, qwen_awq,
        "/analysis/contrasts/awq_minus_fp16/page_bootstrap/estimate",
        _signed(qwen_fp16["page_bootstrap"]["estimate"], 3),
        "qwen_awq.results.fp16.estimate",
    )
    awq_fp16_ci = _qwen_cell(
        provenance, qwen_awq,
        "/analysis/contrasts/awq_minus_fp16/page_bootstrap",
        _interval(
            [qwen_fp16["page_bootstrap"]["lower"], qwen_fp16["page_bootstrap"]["upper"]],
            digits=3,
        ),
        "qwen_awq.results.fp16.ci",
    )
    awq_gptq_effect = _qwen_cell(
        provenance, qwen_awq,
        "/analysis/contrasts/awq_minus_gptq/page_bootstrap/estimate",
        _signed(qwen_gptq["page_bootstrap"]["estimate"], 3),
        "qwen_awq.results.gptq.estimate",
    )
    awq_gptq_ci = _qwen_cell(
        provenance, qwen_awq,
        "/analysis/contrasts/awq_minus_gptq/page_bootstrap",
        _interval(
            [qwen_gptq["page_bootstrap"]["lower"], qwen_gptq["page_bootstrap"]["upper"]],
            digits=3,
        ),
        "qwen_awq.results.gptq.ci",
    )
    awq_fp16_p = _qwen_cell(
        provenance, qwen_awq,
        "/analysis/contrasts/awq_minus_fp16/holm_adjusted_pvalue",
        "<.001", "qwen_awq.results.fp16.holm_p",
    )
    awq_gptq_p = _qwen_cell(
        provenance, qwen_awq,
        "/analysis/contrasts/awq_minus_gptq/holm_adjusted_pvalue",
        "<.001", "qwen_awq.results.gptq.holm_p",
    )
    hard_fp16 = qwen_analysis["hard_labels"]["awq_vs_fp16"]
    hard_gptq = qwen_analysis["hard_labels"]["awq_vs_gptq"]
    hard_values = {}
    for metric, prefix, digits, scale in (
        ("accuracy", "accuracy", 2, 100),
        ("balanced_accuracy", "ba", 3, 1),
        ("mcc", "mcc", 3, 1),
    ):
        for route, pair, role, pointer_pair in (
            ("fp16", hard_fp16, "reference", "awq_vs_fp16"),
            ("gptq", hard_gptq, "reference", "awq_vs_gptq"),
            ("awq", hard_fp16, "awq", "awq_vs_fp16"),
        ):
            rendered = f"{pair[metric][role] * scale:.{digits}f}"
            hard_values[(metric, route)] = _qwen_cell(
                provenance, qwen_awq,
                f"/analysis/hard_labels/{pointer_pair}/{metric}/{role}",
                rendered, f"qwen_awq.{prefix}.{route}",
            )

    tabfact = {
        row["model_key"]: row for row in evidence["tabfact"]
    }
    qwen_tab = tabfact["qwen35_9b"]
    gemma_tab = tabfact["gemma4_e4b"]
    qwen_tab_acc = _result_cell(
        provenance, qwen_tab, "metrics.accuracy_difference",
        _signed(100 * qwen_tab["metrics"]["accuracy_difference"], 2),
        "results.qwen_tabfact.accuracy_pp",
    )
    gemma_tab_acc = _result_cell(
        provenance, gemma_tab, "metrics.accuracy_difference",
        _signed(100 * gemma_tab["metrics"]["accuracy_difference"], 2),
        "results.gemma_tabfact.accuracy_pp",
    )

    return f"""\\section{{Results}}
\\subsection{{Primary confirmatory analysis}}
On the frozen VitaminC confirmatory set, Qwen3.5-9B Q4\\_K\\_M relative to F16
produced $\\Delta I={main_delta}$ (95\\% CI {main_ci}), whereas balanced
accuracy changed by {main_ba} percentage points (Table~\\ref{{tab:main-confirmatory}}).
The prespecified weakening hypothesis was therefore not supported: the primary
interaction estimate increased slightly while hard-label performance declined.

\\input{{tables/main_confirmatory}}

\\subsection{{Cross-family heterogeneity}}
The frozen common-support Stage A comparison showed heterogeneous interaction
changes across four model families (Table~\\ref{{tab:cross-family}}). OLMo-3-7B
had a negative balanced-accuracy point estimate ({olmo_ba} percentage points),
but it was not significant after Holm correction ($p={olmo_p}$). Stage A was
post-confirmatory and is not evidence that the same direction generalizes to all
six registered models.

\\input{{tables/cross_family}}

\\subsection{{Backend-constrained and deployment-route analyses}}
Within the same HF backend, GPTQ INT4 changed $\\Delta I$ by {qwen_hf_delta}
for Qwen3.5-9B and {gemma_hf_delta} for Gemma-4-E4B. The model-by-route contrast
was {interaction_value} (95\\% CI {interaction_ci}), indicating a joint deployment-route effect
(Table~\\ref{{tab:route-robustness}}). Because quantizer,
backend, packing, and kernel differ across GGUF and HF routes, this contrast
cannot be attributed to a quantization algorithm alone.

On the frozen Qwen3.5-9B HF triplet, AWQ INT4 minus FP16 yielded
${awq_fp16_effect}$ (95\\% CI {awq_fp16_ci}; Holm-adjusted $p{awq_fp16_p}$),
and AWQ INT4 minus GPTQ INT4 yielded ${awq_gptq_effect}$ (95\\% CI
{awq_gptq_ci}; Holm-adjusted $p{awq_gptq_p}$). Hard-label accuracy was
{hard_values[("accuracy", "fp16")]}\\%, {hard_values[("accuracy", "gptq")]}\\%, and {hard_values[("accuracy", "awq")]}\\%
for FP16, GPTQ, and AWQ, respectively; balanced accuracy was
{hard_values[("balanced_accuracy", "fp16")]}, {hard_values[("balanced_accuracy", "gptq")]}, and {hard_values[("balanced_accuracy", "awq")]},
and MCC was {hard_values[("mcc", "fp16")]}, {hard_values[("mcc", "gptq")]}, and {hard_values[("mcc", "awq")]}. These are single-model,
method-associated results; packing, loader, and quantized-kernel differences
remain.

\\input{{tables/route_robustness}}

\\subsection{{External validation and behavioral diagnostics}}
On Fresh TabFact, Q4\\_K\\_M relative to F16 changed accuracy by {qwen_tab_acc}
percentage points for Qwen3.5-9B and {gemma_tab_acc} percentage points for
Gemma-4-E4B. CUB/DRUID BCU and CCU changes varied by model and are reported as
secondary behavioral diagnostics rather than direct measures of an internal
mechanism (Table~\\ref{{tab:external-validation}}).

\\input{{tables/external_validation}}

\\subsection{{Robustness and incomplete routes}}
Frozen label-mapping sensitivity analyses retained the original direction for
the evaluated Qwen and Gemma routes. High-precision backend agreement was
treated as descriptive comparability, not equivalence. Gemma AWQ experiments were not formally completed
and consequently provide no positive, negative, or null
scientific result.
"""


def _render_claim_boundary_legacy() -> str:
    return """# Claim Boundary Contract

## Permitted central claim

Quantization is associated with changes in evidence-grounded fact-verification
behavior whose direction and magnitude vary across model families and deployment
routes.

## Required qualifications

- The main confirmatory result does not establish a universal accuracy loss.
- OLMo has a negative point estimate but is not significant after Holm correction.
- Cross-route evidence is a joint deployment-route effect, not a pure quantizer effect.
- AWQ is incomplete and cannot support a positive, negative, or null conclusion.
- Exploratory and sensitivity analyses remain non-confirmatory.

## Prohibited formulations

- “Quantization universally improves accuracy.”
- “Quantization universally reduces accuracy.”
- “GPTQ is intrinsically better or worse than GGUF Q4.”
- “AWQ failed scientifically.”
- “OLMo significantly declined.”
"""


def render_claim_boundary() -> str:
    return """# Claim Boundary Contract

## Permitted central claim

Quantization is associated with changes in evidence-grounded fact-verification
behavior whose direction and magnitude vary across model families, methods, and
deployment routes.

## Required qualifications

- The main confirmatory result does not establish a universal accuracy loss.
- OLMo has a negative point estimate but is not significant after Holm correction.
- Cross-route evidence is a joint deployment-route effect, not a pure quantizer effect.
- The Qwen3.5-9B AWQ result is formally completed but remains a single-model,
  method-associated result.
- Gemma AWQ remains not formally completed and cannot support a positive,
  negative, or null conclusion.
- The Qwen HF triplet is not a pure quantization-algorithm effect because
  packing, loader, and quantized-kernel differences remain.
- No monetary cost claim is permitted because deployment cost was not measured.
- Exploratory and sensitivity analyses remain non-confirmatory.

## Prohibited formulations

- “Quantization universally improves accuracy.”
- “Quantization universally reduces accuracy.”
- “GPTQ is intrinsically better or worse than GGUF Q4.”
- “AWQ is universally better or worse than GPTQ.”
- “The Qwen result generalizes across model families.”
- “The experiment identifies a pure AWQ algorithm effect.”
- “AWQ failed scientifically.”
- “OLMo significantly declined.”
- Any inference about monetary cost or cost savings.
"""


def render_readme_zh() -> str:
    return """# 论文 Results/Methods 包使用说明

本目录由冻结的 `publication_readiness_v1` 确定性生成。数字不得手工修改。

- `main.tex`：独立编译入口，不是目标期刊模板。
- `sections/`：可复制或 `\\input` 到后续论文模板。
- `tables/`：四张论文表格。
- `claims/claim_boundary.md`：允许、必须限定和禁止的结论。
- `provenance/table_source_map.json`：数字到冻结 inventory 字段的映射。

AWQ 未形成正式结果；Stage A 仅有四个 common-support 模型；六模型覆盖是
整个论文包层面的覆盖。换期刊模板时只能调整排版，不得改变统计口径。
"""


def _result_cell(
    provenance, row, field, rendered, cell_id,
    *, file_path="sections/results.tex",
):
    return _cell(
        provenance,
        file_path,
        row,
        field,
        rendered,
        cell_id,
    )


def _render_main_table(row, provenance):
    path = "tables/main_confirmatory.tex"
    quartets = _cell(
        provenance, path, row, "population.quartets", str(row["population"]["quartets"]),
        "main.quartets",
    )
    pages = _cell(
        provenance, path, row, "population.pages", str(row["population"]["pages"]),
        "main.pages",
    )
    delta_i = _cell(
        provenance, path, row, "metrics.delta_i", _signed(row["metrics"]["delta_i"], 3),
        "main.delta_i",
    )
    delta_i_ci = _cell(
        provenance, path, row, "metrics.delta_i_ci95",
        _interval(row["metrics"]["delta_i_ci95"], digits=3), "main.delta_i_ci95",
    )
    ba = _cell(
        provenance, path, row, "metrics.balanced_accuracy_difference",
        _signed(100 * row["metrics"]["balanced_accuracy_difference"], 2),
        "main.ba_pp",
    )
    body = " & ".join(
        [
            MODEL_LABELS[row["model_key"]],
            latex_escape(row["comparison"]),
            quartets,
            pages,
            delta_i,
            delta_i_ci,
            ba,
            "Confirmatory",
        ]
    ) + r" \\"
    return _table(
        caption="Primary confirmatory VitaminC result.",
        label="tab:main-confirmatory",
        columns="@{}lXrrrrrX@{}",
        header=(
            "Model & Contrast & Quartets & Pages & $\\Delta I$ & "
            "$95\\%$ CI & Balanced accuracy change (pp) & Scope"
        ),
        rows=[body],
        note=(
            "Positive $\\Delta I$ does not establish a universal accuracy "
            "gain; the prespecified weakening hypothesis was not supported."
        ),
    )


def _render_cross_family_table(rows, provenance):
    path = "tables/cross_family.tex"
    shared = rows[0]
    quartets = _cell(
        provenance, path, shared, "population.quartets",
        f"{shared['population']['quartets']:,}", "stage_a.shared_quartets",
    )
    pages = _cell(
        provenance, path, shared, "population.pages",
        f"{shared['population']['pages']:,}", "stage_a.shared_pages",
    )
    rendered_rows = []
    for row in sorted(rows, key=lambda item: item["model_key"]):
        prefix = f"stage_a.{row['model_key']}"
        delta_i = _cell(
            provenance, path, row, "metrics.delta_i",
            _signed(row["metrics"]["delta_i"], 3), f"{prefix}.delta_i",
        )
        ci = _cell(
            provenance, path, row, "metrics.delta_i_ci95",
            _interval(row["metrics"]["delta_i_ci95"], digits=3),
            f"{prefix}.delta_i_ci95",
        )
        ba = _cell(
            provenance, path, row, "metrics.balanced_accuracy_difference",
            _signed(100 * row["metrics"]["balanced_accuracy_difference"], 2),
            f"{prefix}.ba_pp",
        )
        holm = _cell(
            provenance, path, row, "metrics.balanced_accuracy_holm_pvalue",
            _pvalue(row["metrics"]["balanced_accuracy_holm_pvalue"]),
            f"{prefix}.holm_p",
        )
        if row["model_key"] == "olmo3_7b":
            interpretation = "not significant after Holm correction"
        else:
            interpretation = "exploratory; Holm-adjusted"
        rendered_rows.append(
            " & ".join(
                [
                    MODEL_LABELS[row["model_key"]],
                    delta_i,
                    ci,
                    ba,
                    holm,
                    interpretation,
                ]
            )
            + r" \\"
        )
    return _table(
        caption="Cross-family effects on the frozen Stage A common-support set.",
        label="tab:cross-family",
        columns="@{}lrrrrX@{}",
        header=(
            "Model & $\\Delta I$ & $95\\%$ CI & "
            "Balanced accuracy change (pp) & Holm-adjusted $p$ & Boundary"
        ),
        rows=rendered_rows,
        note=(
            f"All rows use {quartets} quartets on {pages} pages. Stage A is "
            "post-confirmatory and exploratory. Raw interaction "
            "magnitudes are scale- and calibration-sensitive."
        ),
    )


def _render_route_table(hf_rows, interaction, qwen_awq, provenance):
    path = "tables/route_robustness.tex"
    shared = hf_rows[0]
    quartets = _cell(
        provenance, path, shared, "population.quartets",
        f"{shared['population']['quartets']:,}", "hf.shared_quartets",
    )
    pages = _cell(
        provenance, path, shared, "population.pages",
        f"{shared['population']['pages']:,}", "hf.shared_pages",
    )
    rendered_rows = []
    for row in sorted(hf_rows, key=lambda item: item["model_key"]):
        prefix = f"hf.{row['model_key']}"
        delta_i = _cell(
            provenance, path, row, "metrics.delta_i",
            _signed(row["metrics"]["delta_i"], 3), f"{prefix}.delta_i",
        )
        ci = _cell(
            provenance, path, row, "metrics.delta_i_ci95",
            _interval(row["metrics"]["delta_i_ci95"], digits=3),
            f"{prefix}.delta_i_ci95",
        )
        ba = _cell(
            provenance, path, row, "metrics.balanced_accuracy_difference",
            _signed(100 * row["metrics"]["balanced_accuracy_difference"], 2),
            f"{prefix}.ba_pp",
        )
        rendered_rows.append(
            " & ".join(
                [
                    MODEL_LABELS[row["model_key"]],
                    "HF GPTQ INT4 minus HF FP16",
                    delta_i,
                    ci,
                    ba,
                    "same HF backend",
                ]
            )
            + r" \\"
        )
    qwen_analysis = qwen_awq["analysis"]
    for reference, label in (("fp16", "FP16"), ("gptq", "GPTQ INT4")):
        contrast = qwen_analysis["contrasts"][f"awq_minus_{reference}"]
        interval = contrast["page_bootstrap"]
        estimate = _qwen_cell(
            provenance, qwen_awq,
            f"/analysis/contrasts/awq_minus_{reference}/page_bootstrap/estimate",
            f"{interval['estimate']:+.3f}", f"qwen_awq.{reference}.estimate",
            file_path=path,
        )
        ci = _qwen_cell(
            provenance, qwen_awq,
            f"/analysis/contrasts/awq_minus_{reference}/page_bootstrap",
            _interval([interval["lower"], interval["upper"]], digits=3),
            f"qwen_awq.{reference}.ci", file_path=path,
        )
        rendered_rows.append(
            " & ".join(
                [
                    "Qwen3.5-9B",
                    f"AWQ INT4 minus {label}",
                    estimate,
                    ci,
                    "--",
                    "same HF backend; method-associated",
                ]
            )
            + r" \\"
        )
    interaction_value = _cell(
        provenance, path, interaction, "metrics.four_way_interaction",
        _signed(interaction["metrics"]["four_way_interaction"], 3),
        "route_interaction.value",
    )
    interaction_ci = _cell(
        provenance, path, interaction, "metrics.ci95",
        _interval(interaction["metrics"]["ci95"], digits=3),
        "route_interaction.ci95",
    )
    rendered_rows.append(
        " & ".join(
            [
                MODEL_LABELS[interaction["model_key"]],
                "GGUF-versus-HF interaction",
                interaction_value,
                interaction_ci,
                "--",
                "joint deployment-route effect",
            ]
        )
        + r" \\"
    )
    return _table(
        caption="Within-HF quantization and deployment-route robustness.",
        label="tab:route-robustness",
        columns=(
            "@{}>{\\raggedright\\arraybackslash}p{0.15\\textwidth}"
            ">{\\raggedright\\arraybackslash}p{0.21\\textwidth}"
            "rrr>{\\raggedright\\arraybackslash}p{0.18\\textwidth}@{}"
        ),
        header=(
            "Model/contrast & Comparison & "
            "\\shortstack{$\\Delta I$ /\\\\interaction} & "
            "$95\\%$ CI & \\shortstack{BA change\\\\(pp)} & Boundary"
        ),
        rows=rendered_rows,
        note=(
            f"All rows use {quartets} quartets on {pages} pages. The cross-route "
            "interaction jointly contains quantizer, backend, "
            "packing, and kernel differences and is not a pure quantizer effect. "
            "The Qwen AWQ contrasts hold the broad HF backend fixed, but residual "
            "packing, loader, and quantized-kernel differences remain."
        ),
    )


def _render_external_table(tabfact_rows, cub_rows, provenance):
    path = "tables/external_validation.tex"
    rendered_rows = []
    for row in sorted(tabfact_rows, key=lambda item: item["model_key"]):
        prefix = f"tabfact.{row['model_key']}"
        claims = _cell(
            provenance, path, row, "population.claims",
            str(row["population"]["claims"]), f"{prefix}.claims",
        )
        accuracy = _cell(
            provenance, path, row, "metrics.accuracy_difference",
            _signed(100 * row["metrics"]["accuracy_difference"], 2),
            f"{prefix}.accuracy_pp",
        )
        ci = _cell(
            provenance, path, row, "metrics.accuracy_ci95",
            _interval(row["metrics"]["accuracy_ci95"], digits=2, scale=100),
            f"{prefix}.accuracy_ci_pp",
        )
        ba = _cell(
            provenance, path, row, "metrics.balanced_accuracy_difference",
            _signed(100 * row["metrics"]["balanced_accuracy_difference"], 2),
            f"{prefix}.ba_pp",
        )
        rendered_rows.append(
            " & ".join(
                [
                    "Fresh TabFact",
                    MODEL_LABELS[row["model_key"]],
                    claims,
                    f"Accuracy change (pp): {accuracy} {ci}",
                    f"BA change (pp): {ba}",
                    "external convergent evidence",
                ]
            )
            + r" \\"
        )
    q4_rows = [row for row in cub_rows if row["route"] == "GGUF_Q4_K_M"]
    for row in sorted(q4_rows, key=lambda item: item["model_key"]):
        prefix = f"cub.{row['model_key']}"
        samples = _cell(
            provenance, path, row, "population.gold_samples",
            str(row["population"]["gold_samples"]), f"{prefix}.gold_samples",
        )
        bcu = _cell(
            provenance, path, row, "metrics.gold_bcu_effect",
            _signed(row["metrics"]["gold_bcu_effect"], 3),
            f"{prefix}.gold_bcu",
        )
        ccu = _cell(
            provenance, path, row, "metrics.gold_ccu_effect",
            _signed(row["metrics"]["gold_ccu_effect"], 3),
            f"{prefix}.gold_ccu",
        )
        rendered_rows.append(
            " & ".join(
                [
                    "CUB/DRUID",
                    MODEL_LABELS[row["model_key"]],
                    samples,
                    f"Gold BCU change (raw): {bcu}",
                    f"Gold CCU change (raw): {ccu}",
                    "secondary behavioral diagnostic",
                ]
            )
            + r" \\"
        )
    return _table(
        caption="External structured-evidence and context-utilisation diagnostics.",
        label="tab:external-validation",
        columns="@{}llrXXX@{}",
        header="Dataset & Model & $N$ & Primary effect & Secondary effect & Scope",
        rows=rendered_rows,
        note=(
            "TabFact effects are percentage-point changes. CUB/DRUID effects "
            "are raw behavioral diagnostic scales and are not directly comparable."
        ),
    )


def _cell(provenance, file_path, row, field, rendered, cell_id):
    provenance[cell_id] = {
        "file": file_path,
        "inventory_selector": {
            "source_role": row["source_role"],
            "model_key": row["model_key"],
            "route": row["route"],
        },
        "inventory_field": field,
        "rendered_value": rendered,
    }
    return rendered


def _qwen_cell(
    provenance, qwen_awq, json_pointer, rendered, cell_id,
    *, file_path="sections/results.tex",
):
    provenance[cell_id] = {
        "file": file_path,
        "source_path": qwen_awq["source_path"],
        "source_sha256": qwen_awq["source_sha256"],
        "json_pointer": json_pointer,
        "rendered_value": rendered,
    }
    return rendered


def _signed(value, digits):
    return f"{float(value):+.{digits}f}"


def _interval(values, *, digits, scale=1):
    low, high = values
    return f"[{float(low) * scale:+.{digits}f}, {float(high) * scale:+.{digits}f}]"


def _pvalue(value):
    return f"${_pvalue_number(value)}$"


def _pvalue_number(value):
    number = float(value)
    if number < 0.0001:
        return "<.0001"
    return f"{number:.4f}".removeprefix("0")


def _table(*, caption, label, columns, header, rows, note):
    body = "\n".join(rows)
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        "\\small\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\begin{{tabularx}}{{\\textwidth}}{{{columns}}}\n"
        "\\toprule\n"
        f"{header} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabularx}\n"
        f"\\par\\footnotesize\\emph{{Note.}} {note}\n"
        "\\end{table*}\n"
    )
