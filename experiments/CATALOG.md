# Experiments Catalog — master table

> **One page, every experiment.** What we tried to make QRA's VAAMR classification cheaper/scalable, the
> quantified result on both reference axes, whether it worked, and whether it was promoted into the main
> codebase (and why / why not). Read [`WORKFLOW.md`](WORKFLOW.md) for *how* these were posed, run, and
> adjudicated; read each campaign's `RESULTS.md` for full per-arm detail.

**Three campaigns, one apparatus.** The two classifier campaigns (GNN reliability + scaler distillation)
score every arm through the same frozen harness (`gnn_reliability/{harness,baselines}.py`):
participant-grouped `StratifiedGroupKFold` (seed 42), a **dual-axis** Cohen-κ scorer
(classifier↔LLM-consensus over 205 labeled segments; classifier↔human over 66 consensus items), each
with a **participant-clustered bootstrap 95% CI**. Data: Move-MORE Cohorts 1–2, **n ≈ 32 participants**,
205 LLM-labeled + 134 "No code" segments, 66 human codes. Features: cached **Qwen3-Embedding-8B** 4096-d,
L2-normalized. A third campaign (`mechanism/`) tests the PURER×VAAMR FROM×move interaction (H2) using
frequentist ordinal/mixed models and confound sensitivity; it does not use the classifier harness.

**Success bar (the LLM-equivalence target):** classifier↔LLM grouped **κ ≥ 0.45** (the human↔human
ceiling) **OR** classifier↔human **κ ≥ 0.50** (LLM↔human is 0.537), CI-aware. **Reference bands:** trained
human coders agree at Krippendorff α ≈ 0.33–0.52; the multi-run LLM consensus is human-level (κ = 0.537
vs human). The legacy κ ≥ 0.70 gate is unreachable in principle at this scale and is *not* the target.

---

## Headline

| | classifier↔LLM (κ, 205) | classifier↔human (κ, 66) | promoted? |
|---|---|---|---|
| Prior shipped probe — **A1n** (Qwen class-wt 6-class LogReg) | 0.283 | 0.365 | baseline (already shipped as the consensus probe) |
| ⭐ **Winner — per-rater ensemble `ens_softavg`, C=4** | **0.361** [.28,.43] | **0.450** [.32,.60] | **spec'd, assistive-only** (not yet wired) |
| Success bar | ≥ 0.45 | ≥ 0.50 | — |
| GNN consensus-distillation classifier (H5) | 0.21 (grouped 0.05–0.14) | 0.36¹ | **dropped** as classifier of record |

¹ The GNN's 0.36 human κ (arm A4n) is carried by No-code abstention, not stage discrimination (its
LLM-axis stage agreement is 0.18). **Verdict for the whole program:** no LLM-free classifier clears the
LLM-equivalence bar at n ≈ 32; three independent methods converge on **LLM κ ≈ 0.36 / human κ ≈ 0.45** — a
**data ceiling, not a method gap**. The best shippable artifact is an **assistive, gated, human-reviewed
pre-labeler**, never an autonomous LLM replacement.

---

## Campaign 1 — GNN reliability battery (`gnn_reliability/`)

*Question (H5): can a content-similarity GraphSAGE GNN reproduce the LLM VAAMR consensus well enough to
label new segments LLM-free?* Full detail: [`gnn_reliability/RESULTS.md`](gnn_reliability/RESULTS.md).

| Arm | What was tried | Human κ (n) | LLM κ (205) | Worked? | Decision |
|---|---|---|---|---|---|
| A0 | GraphSAGE on MiniLM-384 | −0.02 (66) | 0.05 | ✗ ≈ chance | the leak-corrected floor |
| A1 | Linear probe, Qwen, 5-class, no imbalance | 0.21 (37) | 0.23 | ~ | features fix the floor; rare stages collapse |
| A1w | Linear probe, Qwen, 5-class, class-weighted | 0.30 (37) | **0.31** | ~ | best LLM-axis; class-weight recovers rare stages |
| **A1n** ⭐ | Linear probe, Qwen, **6-class** (No-code), class-weighted | **0.37 (66)** | 0.28 | ✓ best full-task | **battery winner → seeds Campaign 2 as A1n baseline** |
| A2 / A2n | Correct-&-Smooth (graph smoothing on the probe) | 0.16 / 0.20 | 0.16 / 0.07 | ✗ worst | propagation destroys signal (low homophily) |
| A3 | GraphSAGE, Qwen, 5-class | 0.14 (66) | 0.21 | ✗ < probe | plain GNN below the probe on both axes |
| A4 / A4n | GraphSAGE, Qwen, balanced+focal (5 / 6-class) | 0.10 / 0.36 | 0.16 / 0.18 | ✗ | rebalancing doesn't rescue the GNN; A4n human κ is No-code abstention |
| B1 | GraphSAGE **+ CFiCS construct anchors** | 0.29 (66) | 0.18 | ✗ hurts | anchors lower every axis vs A4n |
| capacity sweep | MLP / GBM / SVM-RBF / calibrated on Qwen features | — | 0.13–0.23 | ✗ < A1n | bridges into Campaign 2; capacity overfits at this n |

**Campaign-1 findings that became method-of-record:**
- **CV-leakage correction** — random k-fold leaks via same-participant temporal/kNN neighbours and inflated
  the gate κ 0.247 → 0.05 under participant-grouped folds. **Promoted:** the production reliability gate
  (`gnn_layer/classifier/validation.py`) now uses participant-grouped CV.
- **VAAMR is not homophilous in content-embedding space** — the probe ties/beats the graph and smoothing is
  the worst arm → the graph is a *liability* for this label. **Promoted (as a negative):** the GraphSAGE
  classifier is **default-OFF** (`gnn_classifier_enabled=False`); LLM consensus stays the label of record.
- **Class-weighting + an explicit "No code" null** were the only levers that helped (measurement, not graph
  machinery). **Promoted:** both are standard in the live classifier path.
- **Discriminant-validity corollary (H6)** — a label a content graph *cannot* recover is, by definition, not
  a topic taxonomy → positive construct evidence. **Promoted:** the discovery-layer H6 instrument
  (`gnn_layer/discriminant.py`, default-on at `qra analyze`).
- **Mechanism** — the influence-based per-segment counterfactual (`run_mechanism.py`, `gnn_layer/influence.py`)
  *inverted* the observed ranking (ρ = −0.13). **Retired and rebuilt** as the dyadic FROM→CUE→TO transition
  model (`gnn_layer/transition.py`, ρ ≈ +0.34) — see Campaign-1 RESULTS §10 + methodology §8.5.

---

## Campaign 2 — classification-scaler distillation (`classification_scaler/`)

*Question: can the A1n probe be distilled — from the human/LLM examples already in hand — to
LLM-equivalent fidelity, so it can scale labeling?* Full detail:
[`classification_scaler/RESULTS.md`](classification_scaler/RESULTS.md) · narrative:
[`classification_scaler/CAMPAIGN_LOG.md`](classification_scaler/CAMPAIGN_LOG.md).

| ID | Lever | What was tried | LLM κ (205) | Human κ (66) | Worked? | Decision |
|---|---|---|---|---|---|---|
| A1n | baseline | class-wt 6-class LogReg probe (Qwen) | 0.283 | 0.365 | — | the bar to beat |
| **S6** ⭐ | **per-rater ensemble** | one class-wt probe **per LLM rater**, mean `predict_proba`, argmax (`ens_softavg`, C=4) | **0.361** [.28,.43] | **0.450** [.32,.60] | ✓ **best**, both axes (paired Δ_LLM +0.078 ✓) | **WINNER → spec'd for promotion as an assistive probe tier (not yet wired)** |
| S6b | rater-weighting | weight raters by their human-subset κ | 0.318 | 0.372 | ✗ hurts | uniform averaging beats agreement-weighting |
| S2 | soft-label distillation | MLP w/ KL to the multi-run ballot mixture (5-/6-class) | 0.367 (5-cls) / 0.310 (6-cls) | 0.228 / 0.339 | ✗ no clear | lifts only the axis it optimizes; 5-class drops No-code |
| S5 | two-stage No-code gate | binary No-code-vs-VAAMR gate@0.45 → 5-class stager | 0.281 | **0.447** [.33,.60] | ~ grazes | best human-axis single lever; structural (subsumed by S6) |
| S5b | ordinal-arc decoding | mord / cumulative-logit / Frank-Hall over the VAAMR arc | 0.01–0.22 | 0.01–0.22 | ✗ collapses | ordinal assumption too strong; decoders degenerate |
| S4 | model capacity | MLP / HistGBM / SVM-RBF / calibration / StandardScaler | 0.07–0.23 | ≤ A1n | ✗ all < A1n | at n ≈ 32 the bottleneck is data, not capacity — stay linear |
| S1 | context embeddings | concat 6-turn context (MiniLM + Qwen, 8192-d) / combined text | 0.227 (Qwen) | 0.360 | ✗ HURTS | context-as-features doesn't transfer the LLM's *reasoning over* context |
| S7 | human anchor | LOPO human calibration / human-weighted mixing of the 66 codes | n/r | ≈ A1n band | ✗ overfits | n = 66 is a *validation* signal, not a *training* signal at this scale |
| S8 / S8b | stacking | per-rater × two-stage (naive / pooled-gate hybrid) | 0.247 / 0.189 | 0.398 / 0.292 | ✗ hurts | splitting 134 No-code rows 3 ways starves each per-rater gate |

**Convergent ceiling.** S6 (LLM 0.361), S2 5-class (LLM 0.367), and S5/S6 No-code structure (human ≈ 0.45)
converge on the same frontier by unrelated mechanisms; every capacity/context/ordinal/stacking/anchor lever
ties or hurts. The binding bottleneck is the **two rare stages** (Avoidance, Metacognition; recall ≈ .28–.35
across *every* arm) compounded by shallow VAAMR separability of a content-trained embedding and n ≈ 32.

---

## Promotion ledger — what graduated into the main codebase, and why

| Trial result | Where it lives now | Why promoted (or not) |
|---|---|---|
| Participant-grouped CV (CV-leakage correction) | `src/gnn_layer/classifier/validation.py` reliability gate | leak-free evaluation is non-negotiable; random folds inflate κ |
| Class-weighting + explicit "No code" null | live VAAMR classifier path | only levers that improved reliability; recover rare-stage recall |
| **GNN classifier is NOT a scaler** (H5 refuted) | `gnn_classifier_enabled=False` (default OFF) — `src/gnn_layer/classifier/` | probe ties/beats graph; graph is a liability for a non-homophilous label |
| Discriminant validity (H6) | `src/gnn_layer/discriminant.py` (discovery, default-on) | the negative classifier result is positive construct evidence |
| Dyadic transition mechanism (replaces influence counterfactual) | `src/gnn_layer/transition.py` + `confound.py` (discovery, default-on) | the per-segment counterfactual was mis-specified (ρ −0.13 → +0.34) |
| **Per-rater ensemble winner (S6 `ens_softavg`)** | **spec'd, NOT yet wired** — `scalable_classification_master_plan.md` §3 | best probe found, but **below the bar at n ≈ 32**: ships **assistive, gated, abstention-aware** below the LLM consensus, never autonomous. Re-evaluate as labeled participants accrue. |
| Context concat / nonlinearity / calibration / ordinal / stacking / anchors / rater-weighting | **NOT promoted** | flat-to-harmful at this n (documented above so they are not re-tried blind) |

**Standing rule:** the multi-run **LLM consensus remains the label of record**. No distilled classifier is
promoted *autonomously* until it clears the bar (LLM κ ≥ 0.45 or human κ ≥ 0.50, CI-aware) on the current
corpus. The whole apparatus is frozen at seed 42 and re-runnable, so every verdict here is re-adjudicable as
n grows.

---

## Campaign 3 — PURER×VAAMR mechanism interaction (`mechanism/`)

*Question (H2): does the therapist move's effect on the next VAAMR stage depend on the participant's FROM
stage (FROM×move interaction)?* Full results: `mechanism/RESULTS.md` · `mechanism/_e1e2_results.json` ·
`mechanism/_e1c_bayesian_results.json`.

| Arm | What was tried | Result | Decision |
|---|---|---|---|
| E1a earns-its-place | Participant-grouped CV held-out log-loss: FROM-only vs additive (+move) vs interaction (FROM×move) | additive: logloss 1.506 (acc 0.37); interaction: 1.531 — overfits; move main effect earns its place, **interaction does not** | PURER main effect keeps; interaction archived |
| E1b frequentist | Ordinal LR test additive vs interaction; Gaussian mixed FROM×move | LR p = 0.52 (ns); Gaussian model singular (un-fittable at n ≈ 32); 0/20 per-cell FDR-significant | no reliable interaction signal at this scale |
| E1c Bayesian ordinal | Hierarchical cumulative-logit FROM×move + (1\|participant) with partial pooling (bambi) | 0/16 interaction HDIs exclude 0 — honest under-identification; 0 divergences; requires isolated `.venv_bayes` | under-identified at n ≈ 32; estimated, not testable |
| E2 confound sensitivity | E-value (VanderWeele-Ding) per (from×move) cell | Avoidance×Education E=4.23, AttnReg×Reinforcement E=3.81, Metacog×Education E=3.49 — formal confound floor | hypothesis-generating only |
| E3–E9 corroboration | Cue representation (E3), trajectory (E4), PURER noise (E5), lift controls (E7), transition counterfactual (E8), H1 test (E9) | see `mechanism/RESULTS.md` | corroborative; none promoted at this n |

All arms are observational (186 FROM→CUE→TO triples, 20 participants / 160 with defined move);
hypothesis-generating, never causal. Nothing from this campaign is promoted to `src/` at this scale.
Confirmatory power awaits Cohorts 3–4.
