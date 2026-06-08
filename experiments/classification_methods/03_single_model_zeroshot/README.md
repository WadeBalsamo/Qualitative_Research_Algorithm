# 03 — Single-Model Zero-Shot

**What:** The foundation method — one LLM, one pass, no consensus voting.
Each CV item is classified in a single zero-shot call; the first (and only)
run's vote is taken as the final prediction.

**Why:** Establishes the simplest possible performance floor that all
multi-run and ensemble methods are measured against.  Any more-complex method
that fails to beat this baseline is not worth its added cost.

**Provenance:** Mar 2026, commit `7dfa61c` (first working LLM classification pass).

---

## Arms

| Arm name | Description |
|---|---|
| `single_zeroshot__<model>` | One model, `n_runs=1`, `merge='first'` |

The default model is `nvidia/nemotron-3-super`; override with `--model`.

---

## Run command

```bash
# Live run (needs LM Studio at 10.0.0.58:1234):
python experiments/classification_methods/03_single_model_zeroshot/run.py \
    --output-dir ./data/output

# Override model:
python experiments/classification_methods/03_single_model_zeroshot/run.py \
    --model qwen/qwen3-8b --output-dir ./data/output

# Dry run — no network, no model loads:
python experiments/classification_methods/03_single_model_zeroshot/run.py --dry-run
```

---

## Expected output

```
experiments/classification_methods/03_single_model_zeroshot/
├── results.csv   # one row appended per run; accumulates across invocations
└── results.md    # markdown table regenerated from full results.csv on each run
```

`results.csv` columns: `arm`, `testset`, `timestamp`, `n_items`, `n_abstain`,
`acc_overall`, `acc_secondary`, `acc_clear`, `acc_subtle`, `acc_adversarial`,
`acc_stage_{0..4}`, `n_clear`, `n_subtle`, `n_adversarial`, `n_stage_{0..4}`.

---

## Notes

> **NOT executed in this build; run-ready; needs live LM Studio backend at 10.0.0.58:1234.**

- This is the **reference baseline** — not a production method.
- Compare against `04_multimodel_consensus` (current production method) and
  `05_ensemble_embedding_llm` (embedding+LLM reconciliation).
- Single-run results will show higher variance; run multiple times across
  different models to form a stable estimate.
