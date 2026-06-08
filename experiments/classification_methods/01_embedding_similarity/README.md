# 01_embedding_similarity — Pure Embedding-Similarity VAAMR Classifier

Classifies content-validity items by encoding both the CV item text and a per-theme **anchor text** with `all-MiniLM-L6-v2`, then taking the cosine-similarity argmax as the primary prediction and the 2nd-argmax as the secondary.  No training data is required — this is a fully zero-shot, framework-description–driven baseline that establishes how well the VAAMR stage definitions alone discriminate items in embedding space.

## Run command

```bash
.venv/bin/python experiments/classification_methods/01_embedding_similarity/run.py --output-dir data/Meta
```

Optional flags:

```
--model <sentence-transformers model>   (default: all-MiniLM-L6-v2)
--arm   <arm_name | all>                (default: all)
--dry-run                               (print arm list + anchor fields; no model load)
```

## Arms

| Arm | Anchor text composition |
|-----|------------------------|
| `def_only` | `theme.definition` |
| `def_exemplars` | definition + `exemplar_utterances` joined with ` | ` |
| `exemplars_only` | `exemplar_utterances` joined (falls back to definition if empty) |
| `def_criteria` | definition + `distinguishing_criteria` |
| `def_exemplars_qprefix` | same text as `def_exemplars` but CV items encoded with `"query: "` prefix (asymmetric bi-encoder style) while anchors are encoded as passages |

## Expected output

`results.csv` — one row appended per arm per run, with columns:

```
arm, testset, timestamp,
n_items, n_abstain,
acc_overall, acc_secondary,
acc_clear, acc_subtle, acc_adversarial,
acc_stage_0 .. acc_stage_4,
n_clear, n_subtle, n_adversarial,
n_stage_0 .. n_stage_4
```

`results.md` — markdown table regenerated from all rows in `results.csv` after each run (display columns: arm, timestamp, n_items, n_abstain, acc_overall, acc_secondary, acc_clear, acc_subtle, acc_adversarial).

**NOT executed in this build; run-ready.**
