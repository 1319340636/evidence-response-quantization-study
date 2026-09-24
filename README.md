# Evidence-response quantization study: public-release candidate

This public repository contains a **candidate commit**, not yet a frozen GitHub Release or a complete archive for every result in the manuscript. Its listed files can be inspected now, but it must not be described as a complete, final, independently reproducible release of all manuscript data.

## Included material

- `data/three_model/`: 10,800 text-free quartet-level records for the nine Qwen, Ministral, and OLMo FP16/GPTQ/AWQ conditions (1,200 quartets per condition), with a detached integrity manifest.
- `data/mapping/mapping_exports.tar.gz`: 27 original/cyclic-mapping condition exports, 4,800 full-evidence records per condition (129,600 total). Each export includes a manifest binding its `records.jsonl` SHA-256.
- `data/repeatability/repeatability_exports.tar.gz`: 81 fixed-artifact process-repeat exports, 480 full-evidence records per run (38,880 total), each with a SHA-256-bound manifest.
- `results/`: frozen three-model, mapping-sensitivity, and repeatability summaries and gates. The mapping gate's `gate_sha256` field is a canonical-content hash, not the byte hash of the JSON file.
- `src/`, `scripts/`, `configs/`, and `pyproject.toml`: analysis and provenance code, execution contracts, and the 1,200-case ID list.

The exported records contain selected-token scores, labels, case IDs, and source-page identifiers. They do **not** contain claim/evidence passages, raw prompts, model weights, credentials, or host-specific paths. The page identifiers are not anonymized; they support page-clustered analysis. Automated screening checked every included mapping/repeatability row and found zero recorded errors or unexpected row counts. The three-model text-free loader verified all nine conditions and 1,078 source pages.

## Scope and boundaries

The 1,200-quartet three-model analysis is post-confirmatory, not an independent replication of the earlier 2,483-quartet comparison. The repeatability archive fixes model artifacts and runtime and does not test independent requantization. The original source datasets and model weights are not redistributed here. The present candidate does not yet provide text-free row-level exports for the earlier 2,483-quartet confirmatory comparison, Fresh TabFact, or CUB/DRUID diagnostics; their report-level coverage and access routes must be resolved before an unqualified manuscript Data Availability statement is used.

The original datasets are obtained from their maintainers: [VitaminC](https://github.com/TalSchuster/VitaminC), [TabFact](https://github.com/wenhuchen/Table-Fact-Checking), and [CUB/DRUID benchmark repository](https://github.com/copenlu/cmt-benchmark). VitaminC's source annotations incorporate Wikipedia and FEVER material; its [data-license notice](https://github.com/TalSchuster/VitaminC/blob/main/DATA_LICENSE) governs source data. Users must obtain source texts and model weights under their respective upstream terms.

## Integrity and reuse

The code in this candidate is offered under the MIT License (`LICENSE_CODE`). Original derived numeric/label records and report data are offered under CC BY 4.0 (`LICENSE_DATA.md`). These grants do not relicense third-party datasets or model artifacts. A frozen GitHub tag and any DOI remain to be assigned after final coverage and access verification; cite an exact commit if referring only to this candidate subset.

For the three-model original-mapping result, install the local package and run `scripts/reestimate_three_model_scale_heterogeneity.py` with `--analysis-records data/three_model` and `--reference-result results/three_model/results.json`, supplying an output path outside the release tree. This re-estimation completed successfully from the staged package on 2026-09-24. Full model execution requires separately acquired upstream datasets/models and the specified runtime. The mapping and repeatability archives are provided as per-run outputs; the included source scripts document the generating protocol, but an independent turnkey replay of those two addenda from this candidate has not yet been demonstrated.
