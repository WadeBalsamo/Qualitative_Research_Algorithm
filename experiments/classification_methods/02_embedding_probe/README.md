# 02_embedding_probe — Trained Linear Probe VAAMR Classifier

Trains a logistic-regression probe on the labeled corpus segments (from `master_segments.csv`) embedded with `all-MiniLM-L6-v2`, then predicts CV items in the same embedding space.  Unlike the similarity classifier in `01_embedding_similarity`, this arm learns a decision boundary from the actual distribution of human/LLM-labeled therapy transcript segments, making it a stronger supervised baseline.  Corpus loading mirrors `experiments/gnn_reliability/harness.py::load_corpus` and the probe pattern mirrors `experiments/gnn_reliability/baselines.py::run_linear_probe`, except the embedding is all-MiniLM (not the cached Qwen NPZ) so CV items — which are not in the Qwen cache — share the same space.

## Run command

```bash
.venv/bin/python experiments/classification_methods/02_embedding_probe/run.py --output-dir data/Meta
```

Optional flags:

```
--model <sentence-transformers model>   (default: all-MiniLM-L6-v2)
--arm   <arm_name | all>                (default: all)
--dry-run                               (print corpus + CV item counts, arm list; no model load)
```

## Arms

| Arm | Description |
|-----|-------------|
| `probe_5class` | `LogisticRegression(max_iter=3000, C=1.0)` on L2-normalized all-MiniLM embeddings, VAAMR classes 0–4 |
| `probe_classweighted` | same but `class_weight='balanced'` to compensate for stage-frequency imbalance |

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

`results.md` — markdown table regenerated from all rows in `results.csv` after each run.

The `meta` columns returned by `score_arm` also include `n_corpus` (number of labeled training segments), `model`, and `class_weight`.

**NOT executed in this build; run-ready.**
