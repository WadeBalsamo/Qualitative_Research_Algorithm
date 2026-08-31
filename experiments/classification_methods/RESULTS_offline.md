# Offline Embedding Baseline Results — cv_vaamr_v1

These are the offline (no-LLM, no-network) embedding baseline arms scored against the
`cv_vaamr_v1` content-validity testset (109 items: 58 clear / 29 subtle / 22 adversarial,
5 VAAMR stages balanced at ~22 items each).  Accuracy here is content-validity accuracy
(primary match against the expected stage), NOT kappa-vs-consensus.  These establish the
LLM-free floor that LLM methods must beat.

_Date: 2026-06-10.  No participant text, utterances, or PHI appears in any output file._

---

## Experiment 01 — Pure Embedding Similarity (cosine argmax)

Model: `all-MiniLM-L6-v2`.  Anchor = VAAMR theme text in various forms; item = CV item text.
No training data used.

| arm | acc_overall | acc_clear | acc_subtle | acc_adversarial | acc_secondary | n_items |
|-----|-------------|-----------|------------|-----------------|---------------|---------|
| def_only | 0.3303 | 0.3276 | 0.3448 | 0.3182 | 0.6147 | 109 |
| def_exemplars | 0.4220 | 0.4310 | 0.4828 | 0.3182 | 0.6422 | 109 |
| exemplars_only | **0.4495** | **0.5862** | 0.4483 | **0.0909** | 0.6330 | 109 |
| def_criteria | 0.3028 | 0.3103 | 0.3103 | 0.2727 | 0.5596 | 109 |
| def_exemplars_qprefix | 0.4128 | 0.4483 | 0.4138 | 0.3182 | **0.6697** | 109 |

Per-stage accuracy (primary), arm = def_exemplars (best balanced arm):

| stage | acc_stage_0 | acc_stage_1 | acc_stage_2 | acc_stage_3 | acc_stage_4 |
|-------|-------------|-------------|-------------|-------------|-------------|
| def_exemplars | 0.4545 | 0.5909 | 0.5455 | 0.1000 | 0.3913 |
| exemplars_only | 0.2727 | 0.7727 | 0.4545 | 0.3000 | 0.4348 |

---

## Experiment 02 — Trained Linear Probe (LogReg on corpus embeddings)

Model: `all-MiniLM-L6-v2`.  Trained on 205 labeled participant segments from qra.db;
no CV items in training data.

| arm | acc_overall | acc_clear | acc_subtle | acc_adversarial | n_items | n_corpus |
|-----|-------------|-----------|------------|-----------------|---------|---------|
| probe_5class | 0.3211 | 0.3103 | 0.3793 | 0.2727 | 109 | 205 |
| probe_classweighted | **0.3394** | 0.3276 | 0.3448 | **0.3636** | 109 | 205 |

Per-stage accuracy (probe_5class vs probe_classweighted):

| stage | probe_5class_s0 | probe_5class_s1 | probe_5class_s2 | probe_5class_s3 | probe_5class_s4 |
|-------|-----------------|-----------------|-----------------|-----------------|-----------------|
| probe_5class | 0.0000 | 0.0000 | 0.7273 | 0.0000 | 0.8261 |
| probe_classweighted | 0.1818 | 0.5455 | 0.4545 | 0.1500 | 0.3478 |

---

## Interpretation

The best embedding-similarity arm (`exemplars_only`, acc=0.45) outperforms both trained
probes (0.32–0.34), confirming that for a 5-class VAAMR scheme with only ~200 labeled
examples the cosine-anchor signal from exemplar utterances is more useful than a
corpus-fitted boundary.  Critically, **all arms collapse on adversarial items**: even the
best cosine arm scores only 0.09–0.32 on adversarial items, and the unweighted probe
scores 0.00 on stages 0, 1, and 3 entirely (collapses to stages 2 and 4 which dominate
the corpus label distribution).  Class-weighting helps the probe on adversarial items
(0.36 vs 0.27) but at the cost of clear-item performance.  Secondary match rises to
0.67 for `def_exemplars_qprefix`, meaning the correct stage lands in the top-2 for 2 in 3
items — useful for narrowing LLM prompt candidates but not for standalone classification.
These numbers establish the LLM-free floor: any LLM method needs acc_overall > 0.45 and
especially needs to close the adversarial gap (currently ≤ 0.36) to be meaningful.

---

_Note (2026-06-10): No PHI, participant text, utterances, or session identifiers appear
in any output file under experiments/. All outputs contain only aggregate metrics, arm
labels, and model identifiers._
