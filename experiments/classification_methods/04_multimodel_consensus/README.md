# 04 — Multi-Model Consensus (METHOD-OF-RECORD)

**What:** The current production VAAMR classification method — multiple LLM
runs with distinct models, combined by majority vote.  Disagreement between
models acts as a natural confidence signal; segments where all models agree
receive a "High" confidence tier.

**Why:** Single-model zero-shot (03) shows high variance.  Rotating models
across runs mitigates idiosyncratic model failures and produces calibrated
confidence tiers from the agreement distribution.  This was validated as the
production method in Apr 2026 (commit `56dd301`) and is the **reference
baseline** all other experiments are compared against.

**Provenance:** Apr 2026, commit `56dd301`.

---

## Arms

| Arm name | n_runs | Models | Merge |
|---|---|---|---|
| `consensus_3model` | 3 | nemotron-3-super, qwen3-8b, gemma-2-9b | majority |
| `consensus_5model` | 5 | nemotron-3-super, qwen3-8b, gemma-2-9b, ministral-8b, llama-3.1-8b-instruct | majority |

Both arms run sequentially in a single invocation.

---

## Run command

```bash
# Live run — both arms (needs LM Studio at 10.0.0.58:1234):
python experiments/classification_methods/04_multimodel_consensus/run.py \
    --output-dir ./data/output

# Override 3-model arm models:
python experiments/classification_methods/04_multimodel_consensus/run.py \
    --models nvidia/nemotron-3-super,qwen/qwen3-8b,mistral/ministral-8b \
    --output-dir ./data/output

# Dry run — no network, no model loads:
python experiments/classification_methods/04_multimodel_consensus/run.py --dry-run
```

---

## Expected output

```
experiments/classification_methods/04_multimodel_consensus/
├── results.csv   # one row per arm per run; accumulates across invocations
└── results.md    # markdown table regenerated from full results.csv on each run
```

`results.csv` columns: `arm`, `testset`, `timestamp`, `n_items`, `n_abstain`,
`acc_overall`, `acc_secondary`, `acc_clear`, `acc_subtle`, `acc_adversarial`,
`acc_stage_{0..4}`, `n_clear`, `n_subtle`, `n_adversarial`, `n_stage_{0..4}`.

---

## Notes

> **NOT executed in this build; run-ready; needs live LM Studio backend at 10.0.0.58:1234.**

- This is the **METHOD-OF-RECORD** — the production system.  Results here are
  the target that all other experiments must beat to justify adoption.
- `consensus_5model` costs ~5× more than `03_single_model_zeroshot` but
  provides a tighter confidence distribution for downstream IRR.
- Use `--models` to substitute models when the default set is unavailable.
