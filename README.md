# Priors Persist Through Suppression

A Stroop Paradigm for Lexical Override.

This repository contains the manuscript, code, and reproduction data package for the paper.

> **Status: anonymous review in progress.** This repository is shared via an anonymization service for ARR / EMNLP review. Author identity is suppressed; the corresponding non-anonymous repository will be linked after acceptance.

---

## Layout

```
.
├── README.md                  (this file)
├── LICENSE                    (MIT)
├── manuscript.pdf             (the paper)
├── code/
│   ├── semantic_stroop_final_end_to_end_reproduction.ipynb
│   │       — main reproduction notebook (behavior + mechanism)
│   ├── mechanism_random_controls_logit_decomp.ipynb
│   │       — supplementary notebook for the item-mismatched control
│   │         with per-perturbation logit decomposition
│   └── emnlp_submission/
│       ├── main.tex, main.pdf, main.bbl, custom.bib
│       ├── acl.sty, acl_natbib.bst   (ACL Rolling Review template)
│       ├── figures/                  (PDFs used by the paper)
│       ├── make_figures.py           (regenerates paper PDFs from CSVs)
│       └── README.md                 (build instructions)
└── data/
    ├── README.txt                    (package contents + CSV ↔ paper mapping)
    ├── csv/                          (65 CSV outputs)
    ├── figures/                      (PNGs)
    └── summaries/final_reproduction_summary.md
```

---

## Paper at a glance

- **Paradigm.** A prompt remaps a query word `w` to a contextual target `t` (e.g. *"A glossary for this document defines 'doctor' as 'forest'. Using only this glossary, a word related to 'doctor' is __"*). A matched neutral prompt substitutes a semantically weak word for `w` while keeping `t` and the lexical-prior distractor `d` (e.g. *hospital*) fixed. Δ = neutral S − conflict S, where S = log P(t) − log P(d). Positive Δ means the lexical prior reduced the target's preference under context.

- **Behavior (§4).** Across 11 open-weight models (Qwen, Gemma, OLMo, Mistral; 1B–9B) × 4 conflict families × 4 prompt styles, Δ is positive in every model. After item-level controls (answer prior, frequency, tokenization, prompt style, conflict family, model family, scale, instruction tuning), the distractor's lexical-prior strength remains a positive predictor of interference.

- **Mechanism (§5).** On five aligned tractable models (Qwen2.5-1.5B base/IT, Gemma-2-2B base/IT, OLMo-1B), residual-stream activation patching at a *source-position triplet* (definition subject, definition target, query word) recovers the override effect more than any pair (ΔR = +0.11 to +0.36, 5/5). A logit decomposition shows the binding-specific signature localizes to the **definition-target** position alone: distractor suppression occurs broadly under matched, swap, and item-mismatched controls, while target preservation occurs only under matched binding.

- **Convergence.** The behavioral predictor (lexical-prior strength of the distractor) and the mechanistic intervention (source-position triplet) land on the same target–distractor axis.

---

## Getting started

### Hardware

A single A100-class GPU is sufficient. Behavioral evaluation uses `bfloat16` where available; mechanistic experiments use `float32` for stable activation caching across model families.

### Software

The reproduction notebook installs its own dependencies in the first cell:
- `transformers`, `accelerate`, `safetensors`
- `transformer-lens` (TransformerLens — used for activation hooks)
- `pandas`, `numpy`, `scipy`, `statsmodels`, `matplotlib`

Some model checkpoints are gated on Hugging Face (Gemma family). Set `HF_TOKEN` before running, or accept the model card terms in your account first.

### Run

Open `code/semantic_stroop_final_end_to_end_reproduction.ipynb` in Jupyter or Colab and run all cells. The notebook is end-to-end deterministic given a fixed seed (recorded in the notebook). It saves:

- CSVs into a `csv/` subdirectory
- PNG figures into `figures/`
- Markdown summaries
- A final zip archive that mirrors `data/`

If you only need to regenerate the random-source control's per-item logit decomposition, run `code/mechanism_random_controls_logit_decomp.ipynb` separately — it is self-contained and reuses the same item-level seeds.

### Compiling the paper

```bash
cd code/emnlp_submission/
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

### Regenerating the paper's figures from the CSVs

```bash
cd code/emnlp_submission/
python make_figures.py
```

Reads from `../../data/csv/` and writes vector PDFs into `figures/`.

---

## Output → paper mapping

| Paper element | CSV / asset |
|---|---|
| Table 1 (main regression) | `behavior_control_regressions.csv` |
| Table 2 (triplet vs best pair) | `mechanism_summary.csv` |
| Table 3 (logit decomp M/S/R) | `mechanism_logit_decomp_comparison.csv` |
| Figure 2 (forest plot) | `behavior_summary_by_model.csv` |
| Figure 3 (heatmaps) | `behavior_summary_by_model_conflict_type.csv`, `behavior_summary_by_model_prompt_style.csv` |
| Figure 4 (lexical-prior scatter) | `behavior_item_prompt_effects.csv` |
| Figure 5 (mechanism core) | `mechanism_source_residual_patches.csv`, `mechanism_summary.csv` |
| Figure 6 (item-mismatched controls) | `mechanism_random_controls.csv` |
| App C Tables 7–11 | `behavior_aggregate_by_*`, `behavior_summary_by_*` CSVs |
| App D Tables 12–13 | `behavior_regression_robustness.csv`, `behavior_control_regressions.csv` |
| App F Tables 14–20 | per-table CSVs in `data/csv/` (see `data/README.txt`) |

The opening paragraph of the appendix maps CSV column names (`source_all`, `def_subject`, `def_target`, `query_word`, `hook_resid_pre`/`mid`/`post`) to paper-level terms (full source triplet, definition subject, definition target, query word, block input, mid-block, block output).

---

## Engineering notes

- **Tokenizer BOS handling.** Some open-weight tokenizers (notably Qwen) assume a BOS token that is not present in the vocabulary. The notebook disables automatic BOS insertion and uses `add_special_tokens=False` scoring across model families for consistent prompt likelihood computation.
- **Single-token mechanism subset.** Activation patching uses a single-token target/distractor subset so the final-position logit-difference readout is clean. Multi-token items are kept for the behavioral analysis with a token-count regression covariate.
- **Clean vs corrupted prompts.** In the mechanism pipeline, the "clean" run is the matched neutral prompt and the "corrupted" run is the lexical-conflict prompt; recovery R measures how much patching clean source activations into the corrupted run restores the neutral-prompt target–distractor margin.
- **Deterministic seeds.** Split validation and item-mismatched controls use a fixed seed recorded in the notebook; the CSVs in the data package are the exact outputs of that seeded run.

---

## Citation

```bibtex
@inproceedings{anon2026priors,
  title     = {Priors Persist Through Suppression: A Stroop Paradigm for Lexical Override},
  author    = {Anonymous},
  booktitle = {Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing},
  year      = {2026},
  note      = {Under review},
}
```

This entry will be updated once the paper is accepted.

---

## License

[MIT](LICENSE) — see the LICENSE file for the full text.
