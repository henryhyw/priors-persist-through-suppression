# EMNLP 2026 submission package

> **Priors Persist Through Suppression:
> A Stroop Paradigm for Lexical Override**

This folder is a self-contained anonymous-review submission targeting EMNLP 2026
(long paper).

## Contents

| File / dir            | Purpose                                                                  |
|-----------------------|--------------------------------------------------------------------------|
| `main.tex`            | Main manuscript source.                                                  |
| `main.pdf`            | Compiled PDF (16 pages: 8 pages of main content + Limitations / Ethical Considerations + References + Appendices A–G). |
| `main.bbl`            | Pre-generated BibTeX output (so the PDF compiles with `pdflatex` alone). |
| `custom.bib`          | Bibliography (BibTeX).                                                   |
| `acl.sty`             | Official ACL/EMNLP style file (verbatim from `acl-style-files-master`).  |
| `acl_natbib.bst`      | ACL natbib bibliography style.                                           |
| `figures/`            | All six figures as vector PDFs, generated from the project CSVs.         |
| `make_figures.py`     | Matplotlib script that regenerates the figures from the data-package CSVs. |

## Submission profile

* **Track**: EMNLP main, long paper (8 pages of content + unlimited references / appendix).
* **Anonymity**: review version (`\usepackage[review]{acl}`); author block reads
  *Anonymous EMNLP submission*; no acknowledgements; no self-references.
* **Page count**: main content (Sections 1–6) fits within the 8-page body limit.
  Limitations, Ethical Considerations, References, and Appendices A–G add 8 more
  pages, none of which count toward the 8-page limit (16 pages total).
* **Mandatory section**: `Limitations` is present (desk-reject if missing).
* **Optional section**: `Ethical Considerations` is present.

## Build

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

`pdflatex` should be from a 2022+ TeX Live distribution. The Inconsolata font
(recommended by the ACL template for typewriter text) is *not* required;
the template falls back to Courier when Inconsolata is unavailable.

## Figures

Figures are produced from the data-package CSVs by `make_figures.py`. Each is a
vector PDF designed for two-column ACL pages:

| Figure | Width        | Source CSVs                                                                 |
|--------|--------------|------------------------------------------------------------------------------|
| 1      | column       | none (illustrative paradigm schematic)                                       |
| 2      | column       | `behavior_summary_by_model.csv`                                              |
| 3      | full width   | `behavior_summary_by_model_conflict_type.csv`, `behavior_summary_by_model_prompt_style.csv` |
| 4      | column       | `behavior_item_prompt_effects.csv`                                           |
| 5      | full width   | `mechanism_source_residual_patches.csv`, `mechanism_summary.csv`             |
| 6      | column       | `mechanism_random_controls.csv`, `mechanism_component_mediation_zero_ablation.csv` |

## Tables in the body

Three tables in the main text:

| Table | Content                                                | Source CSV                                |
|-------|--------------------------------------------------------|--------------------------------------------|
| 1     | Main controlled OLS regression                         | `behavior_control_regressions.csv`         |
| 2     | Source-triplet vs best source-position pair recovery   | `mechanism_summary.csv`                    |
| 3     | Logit decomposition under matched / swap / random      | `mechanism_logit_decomp_comparison.csv`    |

Full per-model and per-component tables live in Appendices A–F and reference the
project data package (`semantic_stroop_final_reproduction_data_package/csv/`).
