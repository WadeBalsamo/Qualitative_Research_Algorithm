# 07 — LLM Model Battery

**Purpose:** Evaluate every LM Studio chat model under 120B parameters against
`cv_vaamr_v1` using single-model zero-shot VAAMR classification.  One arm per
model; each arm runs independently so partial results accumulate in
`results.csv` even if a model fails or is skipped.

**Status:** NOT executed in this build.  Run-ready.  Requires a live LM Studio
instance.

---

## Model Filtering

Two rules applied by `filter_models()`:

1. **Embedding models excluded** — any model id containing `embed`
   (case-insensitive) is dropped.  These are not chat models and cannot
   produce JSON classification responses.

2. **>= 120B excluded** — the last path segment of the model id is scanned for
   a trailing size token (e.g. `480b`, `235b`, `120b`, `72b`, `27b`, `9b`,
   `0.5b`).  If the parsed value is >= 120, the model is dropped.  Models with
   **no detectable size token** are kept (assumed < 120B) and a warning is
   logged.

---

## Static Fallback Model List

Used by `--dry-run` (and when `--models` is not given and the live endpoint is
unreachable).  Intentionally excludes `qwen/qwen3-coder-480b` (>= 120B) and
the two `text-embedding-*` models.

```
qwen/qwen3.6-27b
google/gemma-4-31b
minimax-m2.7
deepseek-r1-finance-reasoning-14b
nvidia/nemotron-3-nano-omni
deepseek-v4-flash
unsloth/qwen3-coder-30b-a3b-instruct
qwen2.5-0.5b-instruct
nvidia/nemotron-3-nano-4b
google/gemma-4-e2b
gemma-4-26b-a4b-it
nvidia/nemotron-3-super
nvidia/nemotron-3-nano
qwen/qwen3-8b
business_consulting_finetune_llama_3.1_8b
qwen/qwen3-next-80b
microsoft/phi-4-reasoning-plus
google/gemma-2-9b
magnum-v4-72b
qwen/qwen3-coder-30b
```

---

## Run Commands

```bash
# Activate the project venv first
source .venv/bin/activate

# Dry-run: print planned arm table (offline, no classification)
python experiments/classification_methods/07_llm_model_battery/run.py --dry-run

# List models that LM Studio currently exposes (hits network)
python experiments/classification_methods/07_llm_model_battery/run.py --list-models

# Full battery against default LM Studio host
python experiments/classification_methods/07_llm_model_battery/run.py \
  --output-dir ./data/output/ \
  --base-url http://10.0.0.58:1234/v1

# If running LM Studio locally
python experiments/classification_methods/07_llm_model_battery/run.py \
  --output-dir ./data/output/ \
  --base-url http://127.0.0.1:1234/v1

# Override model list (subset or custom)
python experiments/classification_methods/07_llm_model_battery/run.py \
  --output-dir ./data/output/ \
  --models qwen/qwen3-8b,google/gemma-2-9b,nvidia/nemotron-3-super
```

---

## results.csv Columns

| Column | Description |
|---|---|
| `arm` | Model id (e.g. `qwen/qwen3-8b`) |
| `testset` | Always `cv_vaamr_v1` |
| `timestamp` | UTC ISO-8601 when the arm was scored |
| `n_items` | Total CV items evaluated |
| `n_abstain` | Items where the model returned no valid prediction |
| `acc_overall` | Primary accuracy (fraction correct, 0–1) |
| `acc_secondary` | Primary-or-secondary accuracy |
| `acc_clear` | Accuracy on `clear`-difficulty items |
| `acc_subtle` | Accuracy on `subtle`-difficulty items |
| `acc_adversarial` | Accuracy on `adversarial`-difficulty items |
| `acc_stage_0` … `acc_stage_4` | Per-VAAMR-stage accuracy |
| `n_clear`, `n_subtle`, `n_adversarial` | Item counts per difficulty tier |
| `n_stage_0` … `n_stage_4` | Item counts per expected stage |

Rows accumulate across runs (one row appended per arm per execution).  Re-run
a single model with `--models <id>` to add a fresh row without overwriting
prior results.

---

## Notes

- **No LLM calls happen** until the script is run in live mode (no `--dry-run`
  flag).  Import-time is clean; all heavy dependencies (`torch`,
  `sentence-transformers`, `urllib.request`) are lazy-loaded inside functions.
- Results are written incrementally — if a model errors mid-battery, earlier
  rows are already persisted.
- `results.md` is (re)written after each arm completes, providing a live
  markdown summary table.
- `--list-models` requires a live LM Studio instance; it does NOT run any
  classification.
