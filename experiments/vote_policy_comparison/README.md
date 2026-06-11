# Vote-Policy Comparison Experiment

**Status:** completed 2026-06-10 — winner promoted to default.

## Purpose

Decide empirically the default consensus vote policy for **both** VAAMR and PURER
classifiers. Candidates:

| Policy | Description |
|---|---|
| `legacy` | Pre-M0 baseline (buggy denominator: threshold uses total raters incl. ERROR) |
| `majority` | Strict majority of *valid* ballots; sub-majority → unlabeled (FIXED denominator) |
| `majority_coded` | Like majority but sub-majority resolves by CODED-pref + confidence → labeled, flagged |
| `coded_plurality` | Among CODED ballots only; ABSTAIN only if zero CODED (monotone) |

## Data

Primary: `data/MMORE_Processed/` (SQLite `qra.db`)
- VAAMR: 66 usable human-consensus IRR items (testsets 1/2/3; source ≠ unresolved)
- Machine ballots: 3 raters per segment (qwen3-80b, gemma-4-31b, nemotron-3-nano-30b)
- PURER: 1 rater (nemotron-3-nano-4b), 221 labeled segments

Secondary: `data/MMORE_Processed_cohort2/` (pre-migration JSONL — no SQLite DB)
- No machine-readable human IRR codes (worksheets are .txt only) → VAAMR κ not computed
- PURER: 3 raters (nemotron-4b, gemma-4-4b, qwen3-8b), 39 labeled segments

## Design

No LLM calls. The experiment reads stored per-rater ballot dicts from `rater_votes`
JSON columns and re-votes each segment under each policy using the production
`vote_single_label()` function. Human consensus is loaded from `irr_human_codes`
(is_consensus=1, source ≠ 'unresolved') — exactly the filter used by
`analysis.irr_analysis._consensus_rows()`.

PURER comparison is coverage-only (no human PURER codes exist).

## Running

```bash
python experiments/vote_policy_comparison/run_experiment.py
```

Outputs `results.json` and `RESULTS.md` to this directory.

## Key findings

See `RESULTS.md` for full tables. Summary:

- `majority` and `legacy` tie at κ=0.597 on this testset because no `[CODED, ERROR, ERROR]`
  ballot patterns appear among the 66 human-coded items — the M0 bug does not manifest
  in this specific sample.
- `majority_coded` (κ=0.448) and `coded_plurality` (κ=0.378) score lower: forcing
  a label on split/sub-majority items introduces noise relative to leaving them unlabeled.
- PURER coverage is identical across all policies (MMORE_Processed 1-rater data has no
  ambiguous cases; cohort2 3-rater data also resolves fully under all policies).

## Decision

**Winner: `majority`** (highest κ among candidates; ties with legacy but legacy is not
a promotion candidate). Promoted to default for BOTH VAAMR and PURER.

Changes applied:
- `src/constructs/config.py` `ThemeClassificationConfig.vote_mode` default: `'majority'` (was `'majority'`, unchanged)
- `src/process/config.py` `purer_classification` default: `vote_mode='majority'` (was `'coded_plurality'`)
