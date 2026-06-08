# 05 — Ensemble: Embedding-Similarity + Single-LLM

**What:** Reconciles two independent signal sources — cosine similarity to
VAAMR theme anchors (all-MiniLM embeddings) and a single zero-shot LLM pass
— using five combination strategies.  Adapted from the VCE codebook ensemble
(commit `f84d38b`) to the VAAMR single-label task.

**Why:** The codebook ensemble showed that embedding and LLM signals catch
different failure modes.  Embedding is deterministic and fast; LLM captures
nuanced context.  Where both agree, confidence is high.  The disagreement
arms test whether deferring to one signal on disagreement improves or hurts
accuracy versus the pure single-model baseline (`03_single_model_zeroshot`).

**Provenance:** Adapted from codebook-era ensemble, commit `f84d38b`.

---

## Arms

| Arm name | Description |
|---|---|
| `llm_only` | Single LLM pass, no embedding (baseline within this experiment) |
| `embedding_only` | argmax cosine sim to theme anchors — no LLM |
| `agree_or_llm` | Where embedding == LLM: use that label.  On disagreement: keep LLM |
| `agree_or_embedding` | Where embedding == LLM: use that label.  On disagreement: keep embedding |
| `union_flag` | pred = LLM; records disagreement count in `meta` column for downstream review |

All five arms are scored and appended to `results.csv` in a single run.

---

## Embedding model

`sentence-transformers/all-MiniLM-L6-v2` — the same backbone used by the
VCE codebook classifier (`src/codebook/embedding_classifier.py`), already
cached in the repo environment.

Anchors: for each VAAMR theme, the anchor text is:
`<short_name> <description> <up to 3 exemplars>`

---

## Run command

```bash
# Live run (needs LM Studio at 10.0.0.58:1234 + sentence-transformers):
python experiments/classification_methods/05_ensemble_embedding_llm/run.py \
    --output-dir ./data/output

# Override LLM model:
python experiments/classification_methods/05_ensemble_embedding_llm/run.py \
    --model qwen/qwen3-8b --output-dir ./data/output

# Dry run — no network, no model loads:
python experiments/classification_methods/05_ensemble_embedding_llm/run.py --dry-run
```

---

## Expected output

```
experiments/classification_methods/05_ensemble_embedding_llm/
├── results.csv   # five rows appended per run (one per arm); accumulates
└── results.md    # markdown table regenerated from full results.csv on each run
```

`results.csv` columns: `arm`, `testset`, `timestamp`, `n_items`, `n_abstain`,
`acc_overall`, `acc_secondary`, `acc_clear`, `acc_subtle`, `acc_adversarial`,
`acc_stage_{0..4}`, `n_clear`, `n_subtle`, `n_adversarial`, `n_stage_{0..4}`.

The `union_flag` arm additionally records `disagree_n` and `disagree_frac`
in its `meta` dict (returned by `score_arm`, not in the fixed CSV schema).

---

## Notes

> **NOT executed in this build; run-ready; needs live LM Studio backend at 10.0.0.58:1234.**

- Compare `embedding_only` vs `llm_only` to quantify how much the LLM adds
  over pure semantic similarity for VAAMR.
- Compare `agree_or_llm` / `agree_or_embedding` against `04_multimodel_consensus`
  to decide whether a cheap single-LLM + embedding pipeline can match the
  multi-model consensus quality.
- `union_flag` is useful as a **triage signal**: high `disagree_frac` on a
  new dataset suggests domain shift or ambiguous segments warranting human review.
- Embedding is deterministic across runs; run-to-run variance comes entirely
  from the LLM pass.
