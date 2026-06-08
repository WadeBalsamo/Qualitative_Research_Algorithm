# 09_harness_designs — ARMS.md

Complete arm inventory for the harness/voting-design battery.

**Design principle:** PromptSpec is FIXED at defaults for every arm.  Only
the HarnessSpec varies.  This means every accuracy difference measured in
this battery is attributable to the consensus/voting design, not the prompt.

---

## Arm Table

| name | what it changes | n_runs | merge | secondary_weight | presence_threshold | abstain_as_ballot | commit | runnable? |
|------|----------------|--------|-------|------------------|--------------------|-------------------|--------|-----------|
| single_pass | Baseline: n_runs=1, no consensus | 1 | first | 0.6 (default) | 0.5 (default) | True (default) | — | yes |
| majority_3 | **Method-of-record** (56dd301): 3 models, majority vote | 3 | majority | 0.6 | 0.5 | True | 56dd301 | yes |
| majority_5 | Expand panel: 5 models, majority vote | 5 | majority | 0.6 | 0.5 | True | — | yes |
| confidence_weighted_3 | Confidence tiebreak: 3 models, merge=confidence | 3 | confidence | 0.6 | 0.5 | True | — | yes |
| triplicate_flag_3 | Flag-on-split: 3 models, sets needs_review on 3-way divergence | 3 | triplicate_flag | 0.6 | 0.5 | True | 7dfa61c | yes |
| secondary_w_low | secondary_weight=0.3 (vs default 0.6) | 3 | majority | **0.3** | 0.5 | True | — | yes |
| secondary_w_high | secondary_weight=0.9 (vs default 0.6) | 3 | majority | **0.9** | 0.5 | True | — | yes |
| presence_low | presence_threshold=0.3 (vs default 0.5) | 3 | majority | 0.6 | **0.3** | True | — | yes |
| presence_high | presence_threshold=0.7 (vs default 0.5) | 3 | majority | 0.6 | **0.7** | True | — | yes |
| abstain_as_ballot | abstain_as_ballot=True (default; baseline for ablation pair) | 3 | majority | 0.6 | 0.5 | **True** | — | yes |
| abstain_excluded | abstain_as_ballot=False — ABSTAIN votes excluded from denominator | 3 | majority | 0.6 | 0.5 | **False** | — | yes |
| single_model_3runs | Stochastic 3 runs, ONE model repeated — tests run-variance vs model-diversity | 3 | majority | 0.6 | 0.5 | True | — | yes |
| hist_triplicate_and_flag__7dfa61c | Historical Mar-2026 original: custom consistency counter + flag | 3 | triplicate_flag | 0.6 | 0.5 | True | **7dfa61c** | yes |
| hist_unified_majority__56dd301 | Historical Apr-2026: introduction of vote_single_label + agreement tiers | 3 | majority | 0.6 | 0.5 | True | **56dd301** | yes |
| hist_model_first_sweep__c6a724f | Jun-2026 refactor: model-first sweep vs segment-first — **score unchanged** | 3 | majority | 0.6 | 0.5 | True | **c6a724f** | **no (documented_only)** |

---

## Historical Arms — Detail

### `hist_triplicate_and_flag__7dfa61c` — commit `7dfa61c` (Mar 2026)

**Commit:** `7dfa61cd400d0cb6841a6a2460360753e6ca3666`
**Message:** *"Replace mentalbert_sentence_aqua with vamr_labeling zero-shot data labeling pipeline"*

The original production design introduced in the first QRA rewrite.  Implemented in
`vamr_labeling/zero_shot_classifier.py` as `_compute_run_consistency()`:

- Ran each segment through the LLM 3 times ("triplicate").
- Used a raw `Counter` to count how many runs agreed on the primary stage
  (`majority_count = primary_counts.most_common(1)[0][1]`).
- The `consistency` field (integer 1–3) was stored in the output.
- Segments with `consistency < n_runs` were flagged for human review.
- Secondary stage was the most-common secondary across all valid runs.

This was **not** a proper majority-vote function — it was a bespoke consistency
counter that happened to select the plurality winner.  There was no `agreement_level`
enum, no evidence-pooled secondary, and no `vote_single_label`.

Captured in this battery as `merge='triplicate_flag'` (which preserves the
flag-on-split behavior) routed through the modern `vote_single_label` aggregator.

---

### `hist_unified_majority__56dd301` — commit `56dd301` (Apr 2026)

**Commit:** `56dd30118c52794433ca1b96c4c3485b123e6258`
**Message:** *"vamr -> vammr; decoupled avoidance and attention regulation to improve cue response analysis"*

Introduced `vote_single_label` as the canonical aggregator in
`classification_tools/majority_vote.py`.  Key changes:

- Replaced the per-segment consistency counter with a reusable, tested
  `vote_single_label()` function.
- Added `_evidence_secondary()` for evidence-pooled secondary-stage inference.
- Added `secondary_weight` (default 0.6) and `presence_threshold` (default 0.5)
  parameters.
- Introduced the `agreement_level` enum (`unanimous` / `majority` / `split` /
  `none`) and `agreement_fraction` fields.
- This became the **METHOD-OF-RECORD** for all subsequent experiments.

Functionally identical to the `majority_3` arm.

---

### `hist_model_first_sweep__c6a724f` — commit `c6a724f` (Jun 2026) — DOCUMENTED ONLY

**Commit:** `c6a724f78224a45784d7e47c56683ab5ca3ac3df`
**Message:** *"refactor: restructure project into src/ layout"*

This commit reorganised all first-party packages under `src/` and (as part of the
broader refactor) changed the segment-classification loop from **segment-first sweep**
(classify all n models for segment i, then move to i+1) to **model-first sweep**
(run all segments through model j, then switch to model j+1).

**This change affects RUNTIME and CHECKPOINTING GRANULARITY only.**  The final
consensus vote for every item is mathematically identical regardless of sweep order,
because `vote_single_label` is applied after all runs are collected.

`run.py` skips this arm with a printed note.  The arm is included in the table for
documentation completeness and so that future researchers understand why checkpoint
files from before and after this commit have different structures.

---

## Knob Reference

| HarnessSpec field | Default | What it controls |
|-------------------|---------|------------------|
| `n_runs` | 1 | How many independent LLM passes per item |
| `per_run_models` | None | Which model to use for each run (list or None = use `model`) |
| `merge` | 'majority' | Aggregation strategy: `first`, `majority`, `confidence`, `triplicate_flag` |
| `secondary_weight` | 0.6 | Weight applied to secondary-stage evidence in `_evidence_secondary` |
| `presence_threshold` | 0.5 | Minimum pooled evidence for a secondary stage to be surfaced |
| `abstain_as_ballot` | True | Whether ABSTAIN votes count in the majority denominator |
| `model` | 'nvidia/nemotron-3-super' | Default model when `per_run_models` is None |
| `base_url` | 'http://10.0.0.58:1234/v1' | LM Studio endpoint |
