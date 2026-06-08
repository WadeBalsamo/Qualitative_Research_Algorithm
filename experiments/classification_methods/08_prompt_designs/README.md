# 08_prompt_designs — Prompt-Design Battery

## Purpose

This battery evaluates how prompt construction choices affect VAAMR classification
accuracy on the `cv_vaamr_v1` content-validity test set.  Each arm changes
**exactly one** `PromptSpec` knob off the default (context window, exemplar count,
difficulty tiers, ordering, zero-shot vs few-shot, JSON strictness).  Two
historical arms anchor the comparison to known pipeline states from early 2026.

The harness is **fixed** (`HarnessSpec(n_runs=1, merge='first')`) across all
arms so that accuracy differences reflect the prompt design alone — not the
consensus strategy, model choice, or number of classification runs.

## Run command

```bash
# Dry run (no model load, no network — lists all arms and exits):
python experiments/classification_methods/08_prompt_designs/run.py --dry-run

# Live run — all arms (needs LM Studio at 10.0.0.58:1234):
python experiments/classification_methods/08_prompt_designs/run.py \
    --output-dir ./data/output

# Single arm:
python experiments/classification_methods/08_prompt_designs/run.py \
    --output-dir ./data/output --arm ctx6

# Override model:
python experiments/classification_methods/08_prompt_designs/run.py \
    --output-dir ./data/output --model qwen/qwen3-8b
```

## Arms

See [ARMS.md](ARMS.md) for the full table of arms — name, knob changed, git
commit (historical arms), and runnable vs documented-only status.

16 total arms: 15 runnable, 1 documented-only (`hist_4stage__7dfa61c` — the
original 4-stage VAAMR framework cannot be faithfully scored against the
5-stage `cv_vaamr_v1` test set; see ARMS.md for explanation).

## Outputs

Results are written to `08_prompt_designs/results.csv` (one row appended per
arm) and `08_prompt_designs/results.md` (markdown summary table, overwritten
after each arm).

## Status

NOT executed in this build.  Run-ready.  Needs live LM Studio backend at
`10.0.0.58:1234` (or override `--model` for an alternative endpoint).
