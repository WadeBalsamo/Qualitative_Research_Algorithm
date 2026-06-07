# QRA Experiments Archive (`experiments/`)

> **What this is.** A self-contained, preserved record of **every** methodological experiment run for
> QRA's VAAMR classification work — the **GNN reliability battery**, the **classification /
> scaler distillation campaign**, and the **mechanism interaction campaign** — including the *failed*
> attempts. It exists so that no trialed method is lost and every reported number is reproducible.
>
> **Inert research code.** **Nothing in the pipeline** (`src/process`, `src/analysis`, `src/gnn_layer`,
> `qra.py`) imports `experiments/`. Everything is consolidated under `experiments/` at the repo root:
> `experiments/gnn_reliability/` is the live apparatus (where the unit tests and the GNN
> construct-validation work-stream import it from), `experiments/classification_scaler/` holds the
> unique scaler code and result files, `experiments/mechanism/` holds the mechanism interaction campaign,
> and `experiments/docs/` holds the narrative records for the first two campaigns.
>
> **Start here, then:** [`CATALOG.md`](CATALOG.md) (the master experiment table + promotion decisions) ·
> [`WORKFLOW.md`](WORKFLOW.md) (the systematic architectural-refinement process + the assessment rubric
> each arm is scored against) · the per-campaign `RESULTS.md` for full detail.

---

## The arc (why these three campaigns, in order)

1. **GNN reliability battery** (`gnn_reliability/`) asked: *can a content-similarity GraphSAGE GNN
   reproduce the multi-run LLM VAAMR consensus well enough to label new segments LLM-free?* (hypothesis
   **H5**). Under leak-free **participant-grouped** cross-validation the answer was **no** (grouped
   κ ≈ 0.05–0.14). The decisive by-product: a **plain linear probe on the same Qwen embeddings ties or
   beats the graph**, so the *probe*, not the graph, became the defensible cheap-classifier candidate.

2. **Classification / scaler distillation campaign** (`classification_scaler/`) then asked: *can that
   probe be bootstrapped — from the human- and LLM-labeled examples already in hand — to LLM-equivalent
   fidelity, so it can scale labeling?* The full ranked battery (context, capacity, soft-label,
   per-rater, structure, stacking, human-anchor) found a **better model than the single probe** — a
   **per-rater ensemble** — but **no configuration reaches the LLM-equivalence bar at n≈32**, and three
   independent methods converge on the same frontier, marking a **data ceiling**.

3. **Mechanism interaction campaign** (`mechanism/`) asked: *does the therapist PURER move's effect on the
   next participant VAAMR stage depend on the participant's FROM stage?* (H2, FROM×move interaction).
   The PURER-move main effect earns its place (held-out log-loss improves vs FROM-only); the interaction
   does not (overfits at n ≈ 32, LR p = 0.52, Gaussian model singular). Bayesian hierarchical ordinal
   (E1c) returns finite credible intervals but 0/16 exclude zero — honest under-identification. E2–E9
   corroboration arms provide confound E-values, cue-representation, trajectory, and H1 tests.

The headline of the classifier campaigns:

| | classifier↔LLM grouped κ | classifier↔human κ |
|---|---|---|
| Prior shipped probe (A1n) | 0.283 | 0.365 |
| **Winner — per-rater ensemble (`ens_softavg`, C=4)** | **0.361** [.28,.43] | **0.450** [.32,.60] |
| Success bar | ≥ 0.45 | ≥ 0.50 |
| GNN classifier (H5, dropped) | 0.21 | 0.36 (κ≈0.05–0.14 vs held-out participants) |

**Verdict:** the winner *dominates* the prior probe on both axes but is **not LLM-equivalent**; it ships
as an **assistive, gated, abstention-aware pre-labeler**, not an autonomous LLM replacement. The ceiling
is *data* (the rare stages Avoidance/Metacognition stay stuck at recall ≈.28–.35; n≈32 participants),
so fidelity is expected to rise as labeled participants accrue. Reproduced on the live corpus
(`data/Meta`): A1n = 0.283/0.365, `ens_softavg` C=1 = 0.325/0.389, and the headline **C=4 winner =
0.361/0.450** (`_csweep.py` → committed `classification_scaler/_csweep_results.json`).

---

## Directory map

```
experiments/
├── README.md                         ← this file (index — start here)
├── CATALOG.md                        ← master experiment table (all three campaigns) + promotion decisions
├── WORKFLOW.md                       ← the experimental-architectural-refinement process + assessment rubric
├── __init__.py
├── docs/                             ← narrative records for both campaigns
│   ├── graph_experiments.md          ← original battery narrative (why probe > GNN)
│   ├── design_decisions.md           ← battery decision record
│   ├── qra_gnn_trial_run_report.md   ← end-to-end GNN pipeline trial-run report (accuracy vs LLM)
│   ├── gnn_discovery_results.md      ← discovery/mechanism results summary + promotion guide
│   ├── gnn_reliability_results.md    ← reliability campaign narrative
│   └── method_application_review.md  ← methodology review + next-steps guide
├── gnn_reliability/                  ← Campaign 1: GNN reliability battery (probe ≥ graph)
│   ├── RESULTS.md                    ← detailed per-arm results + the CV-leakage correction
│   ├── harness.py                    ← THE validated apparatus: load_corpus, build_folds
│   │                                   (participant-grouped StratifiedGroupKFold, seed 42),
│   │                                   get_embeddings (cached Qwen 4096-d), score_arm
│   │                                   (dual-axis κ + participant-clustered bootstrap CIs)
│   ├── baselines.py                  ← linear probe (A1/A1w/A1n) + Correct-&-Smooth + probe helpers
│   ├── anchors_arm.py                ← CFiCS-style construct anchor-node arm (lowered reliability)
│   ├── capacity_scaler.py            ← GNN model-capacity / scale-mode arm
│   └── run_battery.py · run_mechanism.py
├── classification_scaler/            ← Campaign 2: distillation to a scalable classifier
│   ├── RESULTS.md                    ← detailed per-family results + the convergent-ceiling verdict
│   ├── CAMPAIGN_LOG.md               ← the append-only campaign log (every arm, as it ran)
│   ├── rater_distill.py              ← ★ WINNER: per-rater ensemble (ballot extraction + per-rater
│   │                                   probes + mean-proba reduction)
│   ├── _run_distill.py · _csweep.py · _csweep2.py · _paired_delta.py
│   │                                   ← winner drivers: C-sweep + paired cluster-bootstrap Δκ
│   ├── _distill_results.json          ← S6 battery results (C=1 default variants)
│   ├── _csweep_results.json           ← S6 winner: C-sweep + the tuned C=4 headline (0.361/0.450)
│   ├── run_softlabel.py · run_softlabel_grid.py · _confirm5.py · _selfcheck_softlabel.py
│   │                                   ← S2 soft-label distillation (MLP→ballot mixture) + grid
│   ├── _softlabel_results.jsonl       ← S2 raw grid results
│   ├── gate_sweep.py                  ← S5 No-code-gate threshold sweep (two-stage)
│   ├── ordinal_twostage.py            ← S5b ordinal-arc decoding arm (collapses)
│   ├── run_context.py · run_context_local.py · run_context_concat.py
│   │                                   ← S1 context arm (target⊕context; MiniLM + Qwen; all rejected)
│   ├── run_human_anchor.py            ← S7 human-anchor / calibration arm (66 human codes → probe)
│   ├── run_wave2_stack.py · run_wave2b_hybrid.py
│   │                                   ← S8/S8b stacking arms (per-rater × two-stage; both rejected)
│   └── ml2.py                         ← S4/S5 controlled model-lever runner (capacity/calibration/two-stage)
└── mechanism/                        ← Campaign 3: PURER×VAAMR interaction (FROM×move, H2)
    ├── RESULTS.md                    ← per-arm results + verdicts
    ├── run_interaction_model.py       ← E1a/b + E2: earns-its-place grouped CV, frequentist
    │                                   ordinal+mixed interaction, confound E-value sensitivity
    ├── run_bayesian_ordinal.py        ← E1c: Bayesian hierarchical ordinal (isolated; requires a
    │                                   dedicated .venv_bayes with numpy≥2 + bambi)
    ├── run_e3_cue_representation.py · run_e4_trajectory.py · run_e5_purer_noise.py
    │                                   ← E3–E5: cue representation, trajectory, PURER noise arms
    ├── run_e7_lift_controls.py · run_e8_transition_cf.py · run_e9_h1.py
    │                                   ← E7–E9: lift controls, transition counterfactual, H1 test
    ├── _common.py                     ← shared helpers (corpus load, cue-block builder, embeddings, κ CI)
    ├── _design.csv                    ← frozen FROM→CUE→TO design frame exported by run_interaction_model
    ├── _e1e2_results.json             ← E1a/b + E2 results (frequentist arms)
    └── _e1c_bayesian_results.json     ← E1c results (Bayesian; requires .venv_bayes)
```

**Relationship to the live tree (important).** `experiments/gnn_reliability/` is the **single canonical
copy** of the apparatus — it is what the **unit tests** and the discovery layer import directly. There is
no second copy. The hermetic apparatus tests are:
`tests/unit/test_gnn_reliability_harness.py` (6) · `test_gnn_baselines.py` (5) ·
`test_gnn_anchors_arm.py` (4) · `test_gnn_imbalance.py` (13) — GNN battery (28 tests); and
`test_scaler_rater_distill.py` (3) — the per-rater ensemble winner. All pass. The catalog's own scaler
scripts bootstrap `src/` first, so `import experiments.gnn_reliability` /
`experiments.classification_scaler` resolve correctly from the repo root; the reproduce commands run as
written.

---

## The shared apparatus (read before any results)

Every arm in Campaigns 1 and 2 is scored through `harness.py` so the numbers are comparable (Campaign 3
uses ordinal/Bayesian interaction models, not the classifier harness):

- **Folds — `build_folds(df, seed=42)`:** participant-grouped `StratifiedGroupKFold` (whole
  participants held out) — the *only* leak-free protocol here. Random k-fold leaks via same-participant
  temporal/kNN neighbours and **inflates κ** (this is the CV-leakage correction in Campaign 1).
- **Dual-axis scorer — `score_arm`:** reports both **classifier↔LLM-consensus** κ (205 labeled
  segments) and **classifier↔human** κ (66 human items), each with a **participant-clustered bootstrap
  95 % CI**. The human axis is load-bearing and read once per arm (never tuned on).
- **Features — `get_embeddings(df,'qwen')`:** cached 4096-d Qwen3-Embedding-8B vectors, L2-normalized.
- **Data:** n≈32 participants; 205 LLM-labeled + 134 "No code" participant segments; 66 human codes.
- **Success bar:** classifier↔LLM grouped κ ≥ **0.45** (human↔human ceiling) **or** classifier↔human
  κ ≥ **0.50** (LLM↔human is 0.537), CI-aware.

---

## Every trialed method, in detail

### Campaign 1 — GNN reliability (full numbers in `gnn_reliability/RESULTS.md`)

- **GNN soft-VAAMR distillation (H5).** GraphSAGE over kNN + temporal-chain edges, soft-KL to the
  ballot mixture. Held-out-**participant** grouped κ ≈ **0.05–0.14** — far below the κ≥0.70 gate.
  *Rejected as a classifier of record.*
- **CV-leakage correction.** An earlier κ ≈ **0.25** that looked near-passing was a random-fold
  **leakage artifact**; participant-grouped folds drop it to ≈0.05. Now the default protocol.
- **Linear probe (`baselines.py`).** Ties/beats the graph: human κ ≈ **0.37** / LLM **0.31** vs graph
  **0.36 / 0.21** → the probe is the scaler candidate that seeds Campaign 2.
- **Correct-&-Smooth.** Graph smoothing on the probe — the **worst** arm (≈0.16). Diagnostic: VAAMR is
  **not homophilous** in a content-similarity space.
- **Construct anchor nodes (`anchors_arm.py`).** CFiCS-style anchors **lowered** reliability further.
- **What actually helped — measurement, not graph machinery:** class-weighting (recovered rare stages
  from 0% recall) + an explicit **"No code" null** (~36% of participant segments).
- **Discriminant-validity corollary (provisional H6).** The graph's *inability* to recover VAAMR from
  content similarity is positive construct evidence (it is not a topic taxonomy).
- **`experiments/docs/qra_gnn_trial_run_report.md`.** An end-to-end pipeline run (legacy import →
  re-segment → GNN train → analyze) reporting GNN accuracy/validity against the LLM consensus — the
  operational counterpart to the controlled battery.

### Campaign 2 — Classification / scaler distillation (full numbers in `classification_scaler/RESULTS.md`)

- **★ Per-rater ensemble — `ens_softavg` (WINNER, `rater_distill.py`).** Distil **one class-weighted
  6-class LogReg probe per LLM rater** (gemma-4-31b / nemotron-3-nano-30b / qwen3-next-80b; ABSTAIN→No-code,
  ERROR→drop) and **ensemble by mean `predict_proba`**. → **LLM 0.361 / human 0.450**, paired Δ_LLM
  **+0.078 [+.036,+.132]** (reliable; C-sweep peaks at C=4, also reliable at C=1). **Dominates A1n.**
- **Soft-label distillation (`run_softlabel.py`).** MLP fit to the multi-run ballot mixture. 5-class
  MLP-KL = LLM **0.367** but human 0.228 (drops the No-code class); best human is a hard/regularized MLP
  ≈0.38. Lifts only the axis it optimizes; *no clear*.
- **Two-stage No-code gate (S5, `gate_sweep.py` + `ml2.py`).** No-code-vs-VAAMR gate@0.45 → 5-class
  stager. Human **0.447** (best human-axis single lever); the gate threshold trades LLM vs human axis.
- **Model capacity (S4, `ml2.py` / `capacity_scaler.py`).** MLP 0.068/0.167; calibrated 0.13–0.20;
  SVC-RBF 0.230; HGB ≤ linear; StandardScaler hurts. **All below linear A1n 0.283.**
- **Context (S1, `run_context*.py`) — the headline hypothesis, REJECTED.** Concatenating a 6-turn
  context embedding *lowered* κ in both MiniLM (0.158→0.125) and Qwen (0.283→0.227; self-check
  reproduces A1n exactly first); the combined-text variant collapsed (≈0.04). **The LLM's context edge
  is reasoning *over* context, not context in a feature vector.**
- **Ordinal-arc decoding (S5b, `ordinal_twostage.py`).** Cumulative-logit / `mord` / Frank-Hall over
  the VAAMR arc — **collapses** (0.01–0.22).
- **Human anchor (S7, `run_human_anchor.py`).** Can the scarce 66 human consensus codes pull the probe
  toward human-level VAAMR without overfitting (leave-one-participant-out calibration; human-weighted
  mixing)? **No reliable gain** — consistent with the per-rater finding that *weighting by human
  agreement hurts*, and n=66 is too small to calibrate on without overfit. Did not reach the bar.
- **Stacking (S8/S8b, `run_wave2_stack.py` / `run_wave2b_hybrid.py`).** Per-rater × two-stage (naive
  0.247/0.398; pooled-gate hybrid 0.189/0.292) — both **hurt** (per-rater No-code gate starves; the
  winner already subsumes the No-code decoupling).

**Convergent ceiling.** Three independent methods — per-rater ensemble, soft-label MLP, No-code
structure — converge on **LLM ≈ 0.36 / human ≈ 0.45**; every capacity/context/stacking/anchor lever
ties or hurts. That convergence is the signature of a **data ceiling, not a modelling gap**.

---

## Reproduce

All scripts are standalone (each sets its own `sys.path`), read the live corpus at `data/Meta`, and
reuse the harness. From the repository root (with `data/Meta` + the cached Qwen embeddings present):

```bash
# GNN reliability battery
python experiments/gnn_reliability/run_battery.py

# Scaler campaign — the winner (per-rater ensemble) + its paired-Δ and C-sweep
python experiments/classification_scaler/_run_distill.py
python experiments/classification_scaler/_paired_delta.py
python experiments/classification_scaler/_csweep.py
# Other arms: soft-label / context / gate-sweep / ordinal / human-anchor / stacking
python experiments/classification_scaler/run_softlabel.py
python experiments/classification_scaler/run_context_concat.py   # needs the LM Studio embedding endpoint
python experiments/classification_scaler/gate_sweep.py
python experiments/classification_scaler/ordinal_twostage.py     # needs `mord`
python experiments/classification_scaler/run_human_anchor.py
python experiments/classification_scaler/run_wave2_stack.py

# Hermetic apparatus tests (31 pass — no network, no model downloads)
python -m pytest tests/unit/test_gnn_reliability_harness.py tests/unit/test_gnn_baselines.py \
                 tests/unit/test_gnn_anchors_arm.py tests/unit/test_gnn_imbalance.py \
                 tests/unit/test_scaler_rater_distill.py
```

Required deps are all in the project venv (`scikit-learn`, `numpy`, `pandas`, `torch`,
`sentence_transformers`, `mord`, `statsmodels`, `krippendorff`). **Environment pin:**
`numpy==1.26.4` is required (`transformers==4.42.4` requires numpy<2.0 and will fail to import
otherwise). Only the **context** arm calls the LM Studio embedding endpoint (`10.0.0.58`) to build
context-augmented vectors; every other arm runs on the cached Qwen embeddings with no network.

---

## Promotions into the live pipeline (tracked elsewhere)

Recorded, not auto-applied — see `scalable_classification_master_plan.md`:

- **The winner** → `src/classification_tools/probe_classifier.py` as a `qra probe train/status/classify`
  tier: per-rater ensemble, gated on a per-project human-band check, abstention-aware, written to a
  `probe_consensus` provenance tier **below** the LLM consensus. Ships **assistive, not autonomous**.
- **The GNN** → keep as **mechanism/discovery only** (`docs/GNN_MASTER_PLAN.md`); remove its unused
  classifier-of-record surface.

Manuscript write-ups: **`docs/methodology.md` §8.5** (GNN reliability) and **§8.6** (the distillation
campaign).

## Scope note

This archive covers the **VAAMR classification** experiments only. A separate, concurrent **GNN
construct-validation** work-stream (H6 discriminant validity, Track-D dyadic routines, confound
localization) is tracked on its own branch, shares the root `experiments/gnn_reliability/` apparatus
(untouched by this archive), and is intentionally not duplicated here.
