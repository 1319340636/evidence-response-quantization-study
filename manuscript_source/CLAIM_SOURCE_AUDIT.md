# Claim-to-source audit for the public candidate

Audit date: 2026-09-27. This is a scoped public-package audit, not an independent rerun of model inference or a declaration that the manuscript is submission-ready.

## Checks that can be rerun here

Run `uv run python scripts/verify_release.py --verify` after installing the package dependencies. The verifier checks the file-wide `SHA256SUMS.txt`, the relevant detached manifests, text-exclusion rules, and these links:

| Claim or display | Public evidence | Recomputed or checked |
| --- | --- | --- |
| Qwen GGUF confirmatory comparison on 2,483 quartets | `data/confirmatory/`, `results/confirmatory/` | Paired membership, 1,158 pages, page- and quartet-weighted point estimates, cell-level Balanced Accuracy |
| Four-model GGUF Stage A common support on 1,200 quartets | `data/stage_a/`, `data/manifests/vitaminc_dose_subset_ids.txt`, `reports/vitaminc_stage_a_v4/results.json` | Eight source-bound conditions, 1,078 pages, four page-weighted interaction point estimates, hard accuracy and Balanced Accuracy by route |
| Fresh TabFact point estimates | `data/tabfact/`, `results/tabfact/` | Four condition accuracies and Balanced Accuracies |
| CUB/DRUID diagnostic point estimates | `data/cub_druid/`, `results/cub_druid/` | Gold, conflicting, and irrelevant BCU/CCU point estimates |
| Three-model HF original-mapping input | `data/three_model/`, `results/three_model/` | Nine-condition membership and source-page structure; `scripts/reestimate_three_model_scale_heterogeneity.py` re-estimates the frozen scale analysis from released records |
| Balanced-mapping display | `data/mapping/`, `results/mapping/`, `manuscript_source/figures/fig_balanced_mapping.csv`, `manuscript_source/tables/tab_balanced_mapping.tex` | 27 export hashes and counts; all 36 figure rows and 18 table cells linked to the frozen sensitivity report |
| Fixed-artifact process repeats | `data/repeatability/`, `results/repeatability/` | 81 export hashes/counts and the frozen report hash binding |
| Numeric table/figure source maps | `manuscript_source/tables/*.json`, `manuscript_source/figures/*.json`, `reports/`, `configs/`, `src/` | 235 marked references, 12 distinct source files, 273 numeric value links, including source SHA-256 and resolved source keys |

The Stage A exporter validates each of the eight original export manifests and full-record SHA-256 hashes before selecting the frozen subset; Qwen's larger confirmatory export is filtered by the published 1,200-ID list. Only derived labels, selected-token scores, quartet/page identifiers, and source hashes are released. Original claim/evidence text, prompts, and raw responses are excluded. The source IDs are not anonymous.

## Boundaries and remaining work

- The verifier checks the Stage A **point estimates**, not its bootstrap intervals, permutation p values, or all cross-model tests from scratch. Those remain frozen-report evidence until an independently executable full analysis is demonstrated.
- The numeric-source check covers JSON maps that explicitly mark `source_path` and `source_key`; it is **not** a line-by-line audit of every prose statement, citation, or the three later TeX-only tables. The older figure/table manifests are partial indexes of the current manuscript assets.
- The mapping and repeatability archives are hash-bound and structurally checked; this does not by itself reproduce every inferential statistic, nor does process repetition test independent requantization.
- Hashes and matching public files demonstrate identity and traceability, not independent scientific replication. Upstream VitaminC, TabFact, CUB/DRUID texts and model weights must be obtained separately under their own terms.
- This commit is a candidate. A frozen GitHub Release/DOI, public-access check, final manuscript/data-code statements, author declarations, and submission preflight remain separate steps.
