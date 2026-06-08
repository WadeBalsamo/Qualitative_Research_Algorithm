# 09_harness_designs — Harness/Voting-Design Battery

## Purpose

This experiment isolates the **consensus/voting design** as the sole independent
variable in VAAMR zero-shot classification.

Every arm in this battery uses the **same fixed PromptSpec** (all defaults:
`context_window=2`, `randomize=True`, `zero_shot=False`, `n_exemplars=None`,
`include_subtle=True`, `include_adversarial=True`).  Only the `HarnessSpec`
differs between arms — specifically:

- **n_runs**: how many independent LLM passes per item
- **per_run_models**: which model(s) to use per run (distinct vs repeated)
- **merge**: aggregation strategy (`first` / `majority` / `confidence` / `triplicate_flag`)
- **secondary_weight**: weight on secondary-stage evidence (default 0.6)
- **presence_threshold**: minimum pooled evidence to surface a secondary stage (default 0.5)
- **abstain_as_ballot**: whether ABSTAIN votes count in the majority denominator

Because the prompt is held constant, any accuracy difference measured on the
`cv_vaamr_v1` content-validity test set is attributable to the consensus/voting
design, not prompt engineering.

## Relationship to Other Experiments

| Directory | What it varies | Reference |
|-----------|---------------|-----------|
| `03_single_model_zeroshot` | Prompt knobs (zero-shot vs exemplar, context window) | — |
| `04_multimodel_consensus` | Method-of-record (3-model majority) — production reference | 56dd301 |
| `07_llm_model_battery` | Which model in a single-pass baseline | — |
| **`09_harness_designs`** | **Harness/voting design (this battery)** | — |

## Run Command

```bash
# From repo root — activate the venv first:
source .venv/bin/activate

# Dry-run: print arm table (no LLM, no network)
python experiments/classification_methods/09_harness_designs/run.py --dry-run

# Live run — all arms
python experiments/classification_methods/09_harness_designs/run.py \
    --output-dir ./data/output/

# Single arm
python experiments/classification_methods/09_harness_designs/run.py \
    --arm majority_3 \
    --output-dir ./data/output/

# Override model pool (ordered; first 3 used for 3-model arms, first 5 for 5-model)
python experiments/classification_methods/09_harness_designs/run.py \
    --model-pool "nvidia/nemotron-3-super,qwen/qwen3-8b,google/gemma-2-9b,microsoft/phi-4-reasoning-plus,qwen/qwen3.6-27b" \
    --output-dir ./data/output/
```

Results are appended to `results.csv` and summarised in `results.md` in this
directory.  Each row records the arm name, accuracy metrics (overall, clear,
subtle, adversarial), and battery metadata.

## Arm Summary

See **`ARMS.md`** for the full arm table with per-arm descriptions, commit
anchors, and the historical arm design record.

- **15 arms total** (14 runnable + 1 documented-only)
- 3 historical commit-anchored arms (7dfa61c, 56dd301, c6a724f)
- `hist_model_first_sweep__c6a724f` is documented only — it changes execution
  order but not scores; `run.py` skips it with a printed explanation

## Status

NOT executed in this build.  Run-ready.  Requires a live LM Studio instance
at `http://10.0.0.58:1234/v1` (override with `--base-url`).
