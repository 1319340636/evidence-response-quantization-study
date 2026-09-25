# Manuscript figure and table source subset

This folder contains the numeric source inputs and viewable PDF/PNG/SVG exports for the five figures in the current IJMLC manuscript draft. The copied assets match the SHA-256 values in `figures/figure_manifest.json`. That manifest describes the **full local manuscript export directory**: its TIFF entries are provenance references, not files distributed in this GitHub candidate.

`tables/` contains nine frozen numeric table JSON files, byte-identical to the entries in `tables/table_manifest.json`, plus twelve table TeX files copied from the current manuscript source directory. The table manifest is dated 2026-08-09 and is **not** a complete index of the three subsequently added TeX-only tables (`tab_balanced_mapping`, `tab_closest_work`, and `tab_label_mapping_sensitivity`). The current `SHA256SUMS.txt` separately binds all distributed files. This folder is a table/figure subset, not the complete manuscript source archive.

The figure and table hashes establish file identity, not independent verification of statistical estimates. The separate `data/`, `results/`, and `scripts/` folders provide the released text-free records, frozen reports, and the public checks currently implemented. No upstream claim/evidence text, source tables, model weights, or raw prompts are redistributed here.
