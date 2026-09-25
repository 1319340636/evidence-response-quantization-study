# Evidence-response quantization study: public-release candidate

This public repository contains a **candidate commit**, not yet a frozen GitHub Release or a complete archive for every result in the manuscript. Its listed files can be inspected now, but it must not be described as a complete, final, independently reproducible release of all manuscript data.

## Included material

- `data/confirmatory/`: 4,966 text-free quartet-level records for the frozen Qwen GGUF F16/Q4_K_M comparison (2,483 paired VitaminC quartets, 1,158 pages), with source-bound and detached SHA-256 manifests. The recomputed page-balanced interaction contrast and quartet-weighted sensitivity equal `results/confirmatory/primary_results.json`; cell-level Balanced Accuracy equals `results/confirmatory/hard_metrics.json`.
- `data/tabfact/`: 8,000 text-free Fresh TabFact classification records for four Qwen/Gemma F16/Q4_K_M conditions (2,000 paired samples per condition). The public verifier recomputes each condition's accuracy and Balanced Accuracy against `results/tabfact/results.json`. Source table rows, claims, prompts, and raw model responses are excluded.
- `data/cub_druid/`: 25,812 paired, text-free CUB/DRUID diagnostic records for six F16-versus-quantized comparisons (4,302 context samples each). The public verifier recomputes the gold, conflicting, and irrelevant BCU/CCU point estimates against `results/cub_druid/`. Raw query claims, claimant strings, prompts, and original context text are excluded.
- `data/three_model/`: 10,800 text-free quartet-level records for the nine Qwen, Ministral, and OLMo FP16/GPTQ/AWQ conditions (1,200 quartets per condition), with a detached integrity manifest.
- `data/mapping/mapping_exports.tar.gz`: 27 original/cyclic-mapping condition exports, 4,800 full-evidence records per condition (129,600 total). Each export includes a manifest binding its `records.jsonl` SHA-256.
- `data/repeatability/repeatability_exports.tar.gz`: 81 fixed-artifact process-repeat exports, 480 full-evidence records per run (38,880 total), each with a SHA-256-bound manifest.
- `results/`: frozen three-model, mapping-sensitivity, and repeatability summaries and gates. The mapping gate's `gate_sha256` field is a canonical-content hash, not the byte hash of the JSON file.
- `src/`, `scripts/`, `configs/`, and `pyproject.toml`: analysis and provenance code, execution contracts, and the 1,200-case ID list.
- `manuscript_source/`: the five current manuscript figures as PDF/PNG/SVG plus their numeric JSON/CSV inputs, nine frozen table JSON inputs, and twelve manuscript table TeX files. See its README for the scope of the source manifests and omitted formats.

The exported records contain selected-token scores, labels, case IDs, and source-page identifiers. They do **not** contain claim/evidence passages, raw prompts, model weights, credentials, or host-specific paths. The page identifiers are not anonymized; they support page-clustered analysis. Automated screening checked every included mapping/repeatability row and found zero recorded errors or unexpected row counts. The three-model text-free loader verified all nine conditions and 1,078 source pages.

## Scope and boundaries

The 1,200-quartet three-model analysis is post-confirmatory, not an independent replication of the earlier 2,483-quartet comparison. The repeatability archive fixes model artifacts and runtime and does not test independent requantization. The original source datasets and model weights are not redistributed here. The present candidate now includes text-free row-level records for the confirmatory VitaminC comparison, nine-route VitaminC analysis, Fresh TabFact, and CUB/DRUID point estimates. It is still not a frozen Release or a demonstrated turnkey replay of every supplementary analysis, and the manuscript-wide source-data/figure audit remains to be completed before an unqualified Data Availability statement is used.

The original datasets are obtained from their maintainers: [VitaminC](https://github.com/TalSchuster/VitaminC), [TabFact](https://github.com/wenhuchen/Table-Fact-Checking), and [CUB/DRUID benchmark repository](https://github.com/copenlu/cmt-benchmark). VitaminC's source annotations incorporate Wikipedia and FEVER material; its [data-license notice](https://github.com/TalSchuster/VitaminC/blob/main/DATA_LICENSE) governs source data. Users must obtain source texts and model weights under their respective upstream terms.

## Integrity and reuse

The code in this candidate is offered under the MIT License (`LICENSE_CODE`). Original derived numeric/label records and report data are offered under CC BY 4.0 (`LICENSE_DATA.md`). These grants do not relicense third-party datasets or model artifacts. A frozen GitHub tag and any DOI remain to be assigned after final coverage and access verification; cite an exact commit if referring only to this candidate subset.

For the three-model original-mapping result, install the local package and run `scripts/reestimate_three_model_scale_heterogeneity.py` with `--analysis-records data/three_model` and `--reference-result results/three_model/results.json`, supplying an output path outside the release tree. This re-estimation completed successfully from the staged package on 2026-09-24. Full model execution requires separately acquired upstream datasets/models and the specified runtime. The mapping and repeatability archives are provided as per-run outputs; the included source scripts document the generating protocol, but an independent turnkey replay of those two addenda from this candidate has not yet been demonstrated.
