Semantic Stroop — Reproduction Data Package
============================================

Companion artifact for:

  Priors Persist Through Suppression: A Stroop Paradigm for Lexical Override
  (EMNLP 2026 submission)

This package contains the CSV outputs, figures, and summary that reproduce
every numerical result in the manuscript. The notebook that generates
these outputs end-to-end is
semantic_stroop_final_end_to_end_reproduction.ipynb (in the project root),
together with the supplementary control notebook
mechanism_random_controls_logit_decomp.ipynb.

Headline findings reproduced by this package
--------------------------------------------
- Behavior (Section 4): aggregate Stroop interference is positive in every
  one of 11 open-weight model settings (Qwen, Gemma, OLMo, Mistral; 1B-9B),
  every conflict family (antonym, arbitrary, polysemy/entity,
  domain-definition), and every prompt style (game-rule, glossary,
  technical-document, scoped-definition). The distractor's lexical-prior
  strength remains a positive predictor of interference after item-level
  controls for answer prior, frequency, tokenization, prompt style,
  conflict family, model family, scale, and instruction tuning.

- Mechanism (Section 5): on five aligned tractable models
  (Qwen2.5-1.5B base/IT, Gemma-2-2B base/IT, OLMo-1B), residual-stream
  activation patching at the source-position triplet (definition subject,
  definition target, query word) recovers the override effect more than
  any source-position pair (Delta-R = +0.11 to +0.36, 5/5 models). A
  logit decomposition then shows the binding-specific signature
  localizes to the definition-target position alone: distractor
  suppression occurs broadly (under matched, swap, and item-mismatched
  controls), while target preservation occurs only under matched binding.

Directory layout
----------------
csv/        all numerical outputs
figures/    behavioral and mechanism PNGs
notebook/   copy of the end-to-end reproduction notebook
summaries/  final_reproduction_summary.md (run log + key tables)

Behavior CSVs (Section 4)
-------------------------
- behavior_model_run_log.csv
- behavior_external_validity_stimuli.csv
- behavior_item_prompt_effects.csv
- behavior_summary_by_model.csv
- behavior_summary_by_model_conflict_type.csv
- behavior_summary_by_model_prompt_style.csv
- behavior_aggregate_by_conflict_type.csv
- behavior_aggregate_by_prompt_style.csv
- behavior_control_regressions.csv
- behavior_regression_robustness.csv
- behavior_evidence_checklist.csv

Mechanism CSVs (Section 5)
--------------------------
- mechanism_model_run_log.csv
- mechanism_summary.csv
- mechanism_source_residual_patches.csv
- mechanism_component_scan.csv
- mechanism_component_scan_nondegenerate.csv
- mechanism_component_mediation_zero_ablation.csv
- mechanism_split_validation.csv
- mechanism_split_validation_summary.csv
- mechanism_random_controls.csv
- mechanism_random_controls_summary.csv
- mechanism_logit_decomposition.csv
- mechanism_evidence_checklist.csv
- mechanism_patch_items.csv
- mechanism_patch_prompts.csv
- mechanism_item_effects.csv
- mechanism_stimuli.csv

Definition-target swap control (Section 5.2 / Appendix F)
---------------------------------------------------------
- mechanism_definition_target_swap_summary.csv
- mechanism_definition_target_swap_controls.csv
- mechanism_definition_target_swap_item_summary.csv

Item-mismatched (random-source) control with logit decomposition
(per-perturbation per-item target_logit_delta and distractor_logit_delta;
schema mirrors mechanism_definition_target_swap_controls.csv):
- mechanism_random_controls_logit_decomp.csv
- mechanism_logit_decomp_comparison.csv  (5-row MATCHED vs SWAP vs RANDOM)

Manuscript table re-cuts
------------------------
Selected columns reshaped for direct paste into the manuscript:
- manuscript_Table_Behavior_Aggregate_Conflict_Type_Sum.csv
- manuscript_Table_Behavior_Aggregate_Prompt_Style_Sum.csv
- manuscript_Table_Behavior_Main_Control_Regression.csv
- manuscript_Table_Behavior_Model_Level_Sum.csv
- manuscript_Table_Mechanism_Logit_Decomposition.csv
- manuscript_Table_Mechanism_Random_Controls.csv
- manuscript_Table_Mechanism_Source_Triplet.csv
- manuscript_Table_Mechanism_Top_Mediation_Drops.csv

Column-name conventions (CSV vs paper)
--------------------------------------
The CSVs use the implementation-level identifiers from the notebook;
the paper uses paper-level terms (a one-time mapping is given in the
opening paragraph of the appendix in the manuscript). The most common
mappings are:

  CSV column                paper term
  ------------------------  -------------------------------
  source_all                full source triplet
  def_subject               definition subject
  def_target                definition target
  query_word                query word
  hook_resid_pre            block input
  hook_resid_mid            mid-block (post-attention)
  hook_resid_post           block output
  target_logit_delta        Delta-ell_t (target logit change)
  distractor_logit_delta    Delta-ell_d (distractor logit change)
  logit_diff_delta          Delta-m (target minus distractor)

Reproducibility notes
---------------------
- TransformerLens is used for activation hooks and patching.
- Behavioral evaluation uses bfloat16 where available; mechanistic
  experiments use float32 for stable activation caching across model
  families.
- All split validation and item-mismatched controls use a fixed seed
  recorded in the notebook; the CSVs in this package are the exact
  outputs of that seeded run.
- Some open-weight tokenizers (notably Qwen) need explicit BOS handling;
  the notebook disables automatic BOS insertion and uses
  add_special_tokens=False scoring across model families.

Hardware
--------
Reproduction is straightforward on a single A100-class GPU.
