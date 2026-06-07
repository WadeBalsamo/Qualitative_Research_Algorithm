# Scalable VAAMR Classification — Distillation Campaign Log

> **STATUS: CAMPAIGN COMPLETE (historical append-only log).** This is the chronological record of how
> the distillation campaign ran (waves 1a → context → wave 2). The *consolidated final* results,
> verdict, and per-arm detail live in **`RESULTS.md`**; the cross-campaign index + promotion decisions
> live in **`../CATALOG.md`** and **`../WORKFLOW.md`**. A few arm rows below are marked "running" /
> "queued" — those are frozen historical states; S1 (context) completed NEGATIVE, S7 (human-anchor)
> completed (no reliable gain), and S3 (instruct-prefix) was not pursued after S1 fell. The winner
> (S6 `ens_softavg` C=4) is backed by the committed `_csweep_results.json`.
>
> This campaign committed nothing to the pipeline; the researcher decides what to promote. Goal: a
> computationally-efficient classifier that reproduces the multi-run LLM VAAMR consensus / human
> judgment well enough to scale labeling LLM-free, on `./data/Meta`. Apparatus:
> `experiments/gnn_reliability/{harness,baselines}.py` (participant-grouped CV + participant-clustered
> bootstrap CIs + both-axis scorer). Each experiment ran in its own isolated worktree via a parallel
> Opus subagent.

## Success bar
A cheap model (cached embeddings + a light classifier; no per-segment LLM at inference) reaching
**either** classifier↔LLM grouped κ ≥ ~0.45 (human↔human ceiling) **or** classifier↔human κ ≥ ~0.50
(approaching LLM↔human 0.537), **CI-aware**. Baseline to beat: **A1n** (Qwen class-weighted 6-class
linear probe) = **human κ 0.365 [0.23, 0.51] / LLM-axis grouped κ 0.31**.

## Results (filled as subagents return; κ = point [95% CI])

| Exp | Lever | best variant | LLM-axis κ (205) | human κ (66) | rare recall | beats A1n? | bar? |
|---|---|---|---|---|---|---|---|
| A1n | baseline (Qwen probe) | class-wt 6-class | **0.283** | 0.365 [.23,.51] | .36/.35/.31 | — | no |
| S1 | context embeddings (concat) | qwen3 target⊕context 8192-d | 0.227 | 0.360 | — | **no — HURTS** (MiniLM too) | no |
| S2 | soft-label distillation | MLP-KL 5-cls (LLM) · hard 6-cls (HUM) | 0.367 [.30,.44] | 0.388 [.26,.54] | .24/.25/.81/.31/.55 | LLM (axis it optimizes only) | no |
| S3 | instruct-prefix embedding | _endpoint back — queued after S1_ | | | | | |
| S4 | model capacity (MLP/GBM/calib) | linear best; rest hurt | 0.13–0.20 | <A1n | collapses | **no — all below A1n** | no |
| **S5** | **two-stage No-code gate** | gate@.45 → 5-cls stager | 0.281 | **0.447 [.33,.60]** | .28/.35/.31 | **hum +.08** | grazes |
| S5b | ordinal arc (mord/ridge/Frank-Hall) | — | 0.01–0.15 | 0.01–0.22 | collapsed | **no — hurts** | no |
| **S6** ⭐ | **per-rater ensemble (3 raters)** | **ens_softavg C=4** | **0.361 [.28,.43]** (Δ+.078 ✓) | **0.450 [.32,.60]** (Δ+.085 ns)¹ | .36/.35/.62/.28/.60 | **both axes ↑** | grazes (HUM CI incl .50) |
| S6b | rater-weight by human agreement | (vs uniform) | 0.318 | 0.372 | — | **no — hurts** | no |
| S7 | human anchor (calibration) | _re-run controlled_ | | | | | |
| S8 | combine winners — naive S5⊕S6 | per-rater two-stage softavg (C=4) | 0.247 | 0.398 | — | **no — hurts** (starves per-rater gate) | no |
| S8b | combine — pooled-gate × per-rater stager | C=4 | 0.189 | 0.292 | — | **no — hurts** | no |

¹ **Δ notation:** S2/S6 report *paired* cluster-bootstrap-by-participant Δκ vs A1n (n=3000) — the
honest within-fold contrast. `✓` = 95% CI excludes 0 (reliable); `ns` = CI overlaps 0. Absolute κ =
A1n + Δ (A1n: LLM 0.283 / human 0.365). S4/S5 LLM-axis points are grid-selected, so weight the bar by
the *human* axis (never tuned).

## The frontier (Wave 1a) — one strong single lever + one stackable structural lever
- **S6 `ens_softavg` (C=4) is the best single model** — distil one class-weighted probe *per LLM rater*
  (gemma-4-31b / nemotron-3-nano-30b / qwen3-next-80b), ensemble by mean `predict_proba`. It
  **dominates A1n on both axes**: LLM **0.361** (paired Δ+0.078 [+.036,+.132] ✓ reliable; still +0.042 ✓
  at default C=1) and human **0.450 [0.32, 0.60]**. The collapse-to-majority `final_label` throws away
  rater disagreement that is actually signal.
- **S5 two-stage No-code gate is a stackable structural lever** — human **0.447 [0.33, 0.60]** at no LLM
  cost; decouples "No code vs VAAMR" from the 5-stage decision. Worth nesting *inside* each per-rater
  probe (Wave 2).
- Both land at **human κ ≈ 0.45** (95% CI *includes* 0.50, point below) and **LLM κ ≈ 0.36 < 0.45 bar**.
  Closer, not cleared.

**Everything on isolated-segment features that adds capacity / arc-structure hurts or is flat:**
nonlinear (MLP/GBM/SVC-RBF all 0.13–0.23 < linear, S4), calibration (0.13–0.20, S4), ordinal decoding
(collapses, S5b), soft-KL distillation (trades LLM for human, ns, S2), rater-weighting by human
agreement (hurts, S6b). The one consistent bottleneck across every arm: the **two rare stages
(Avoidance, Metacognition) stay stuck** (recall ~.28–.35) — the residual ceiling is rare-class *data*
(n≈32 participants), not the model.

## Context lever (S1) — local test NEGATIVE; qwen3 concat running
Tested locally first (same MiniLM embedder, isolated vs +context, so the contrast is clean):
- isolated 0.158/0.264 → **concat (target⊕context) 0.125/0.269** (LLM *down*, human flat);
  **combined single-text 0.043 (collapses — dilution)**.
- i.e. **context as embedding-concatenation does not transfer the LLM's advantage** in the weak space.
- **Confirmed in the strong space:** qwen3 concat (8192-d) self-check reproduces A1n exactly
  (0.283/0.365 ✓), then **target⊕context = 0.227/0.360 — LLM down, human flat**; concat+two-stage
  worse still (0.192/0.355). Context as embedding-concatenation **hurts** at both 384-d and 4096-d.
- **Conclusion:** the LLM's context edge is *reasoning over* context, not context *presence* in the
  feature vector — a linear probe cannot cheaply distil it; concatenated context dims are noise the
  probe can't exploit at n≈32 participants.

---

# CAMPAIGN VERDICT — best shippable scaler + honest ceiling

**No cheap classifier clears the LLM-equivalence bar at this data scale.** Three *independent* methods
converge on the same frontier — per-rater ensembling (S6: LLM 0.361), soft-label MLP distillation
(S2: LLM 0.367), and structural No-code decoupling (S5/S6: human ≈0.45) — and **every** capacity,
calibration, ordinal, context, and stacking lever beyond that ties or hurts. Convergence of unrelated
methods on **LLM κ ≈ 0.36 / human κ ≈ 0.45** is the signature of a **data ceiling, not a method gap**.

| Bar | Target | Best achieved | Cleared? |
|---|---|---|---|
| classifier↔LLM (grouped κ) | ≥ 0.45 (human↔human) | **0.361** (S6 ens_softavg C=4; paired Δ+.078 ✓ vs A1n) | no |
| classifier↔human (κ) | ≥ 0.50 (→ LLM↔human .537) | **0.450** [0.32, 0.60] (S6; CI *includes* .50, point below) | no (grazes) |

### Best shippable model (the scaler to promote, if any)
**S6 `ens_softavg` (C=4)** — distil one 6-class **class-weighted L2-LogReg** probe **per LLM rater**
(gemma-4-31b / nemotron-3-nano-30b / qwen3-next-80b; ABSTAIN→No-code, ERROR→drop) on cached 4096-d
Qwen3-8B features, **ensemble by mean `predict_proba`**, argmax. Participant-grouped CV; C≈4 (LLM-axis
optimum, also wins at C=1). → **LLM 0.361 [.28,.43], human 0.450 [.32,.60]**, per-class recall
.36/.35/.62/.28/.60. **Dominates A1n on both axes** and is the single best model found; the LLM-axis
gain over the shipped consensus probe is statistically reliable (paired CI excludes 0). No LLM at
inference — cached embeddings + 3 tiny probes.

### What this means for scaling
- It is a **high-quality triage / pre-labeler, not an autonomous LLM replacement.** At human↔LLM = .537
  and human↔human ≈ .45, a scaler at human 0.45 reproduces the coding about as well as a *second human
  rater* on the majority stages — usable to pre-label + rank-by-confidence, with **human review of
  low-confidence and the two rare stages**.
- **The binding constraint is data, not modelling:** gains concentrate in the frequent stages; the two
  **rare stages (Avoidance, Metacognition) stay stuck** (recall ≈.28–.35) across every arm, and the
  Qwen embedding's VAAMR separability is shallow. Fidelity should rise as labeled participants accrue
  (n≈32 now) — the scaler is **re-runnable as the corpus grows** via the frozen harness (seed 42).
- **Do not** add context-concatenation, nonlinearity, calibration, ordinal decoding, or No-code/​
  per-rater stacking — all characterized here as flat-to-harmful at this n.

### Recommendation
Ship S6 `ens_softavg` as the **assistive** LLM-free scaler with a **confidence gate + mandatory human
review on abstentions/rare-class/low-margin**, and **re-evaluate the bar at larger n**. Promote the
spec into `scalable_classification_master_plan.md §3` only with the "assistive, not autonomous" caveat.
Reusable code (uncommitted, in subagent worktrees): `rater_distill.py` (S6), `run_softlabel.py` (S2).

## Notes / interpretation
- Wave 1a (model levers on cached Qwen *target* features) — **characterized & largely exhausted**:
  only the No-code decoupling (S5, human axis) and rater-ensembling (S6, LLM axis) help; both grouped.
- **S1 context is now RUNNING** (endpoint `10.0.0.58` recovered): embedding each participant segment
  *with its 6 preceding turns*, then probe + the S5 two-stage gate **on the context features** (stacks
  the two structural levers). This is the highest-leverage untested swing — the LLM's main edge is the
  preceding-turn context the isolated-segment probe lacks.
- **Wave 2** (after S1): stack the frontier — context features ⊕ S5 two-stage gate ⊕ S6 rater-ensemble
  — and read whether the combination clears the bar (LLM κ≥.45 or human κ≥.50, CI-aware), or report the
  honest n≈32-participant ceiling.
