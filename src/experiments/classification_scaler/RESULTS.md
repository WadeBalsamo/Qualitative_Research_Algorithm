# Classification Scaler Campaign — RESULTS

**Purpose:** Determine whether a cheap, LLM-free classifier — distilled from the human- and LLM-labeled
VAAMR examples already in hand — can reproduce the multi-run LLM VAAMR consensus (or human judgment)
well enough to scale participant-segment labeling without per-segment LLM calls.

> Archival note: this campaign committed nothing to the pipeline. It ran as a set of isolated worktrees
> driven by parallel subagents. The κ values below are the authoritative campaign results; the scripts in
> this directory are the salvaged, standalone reproductions. The canonical narrative is `CAMPAIGN_LOG.md`.

---

## Apparatus & success bar

**Task.** QRA classifies therapy-transcript *participant* segments into VAAMR — 5 developmental stages
(Vigilance, Avoidance, Attention-Regulation, Metacognition, Reappraisal) plus a 6th **"No code"** null
class. The scaler must reproduce that labeling cheaply.

**Evaluation harness** (`src/experiments/gnn_reliability/{harness,baselines}.py`, reused verbatim across
every arm so all results are on identical folds):

- **Cross-validation:** participant-grouped `StratifiedGroupKFold`, **seed 42**. No participant's segments
  ever straddle the train/test boundary (no speaker leakage). The "No-code" negatives are assigned
  by-participant inside the same fold logic.
- **Dual-axis scorer.** Every arm is scored on *both* reference axes, both Cohen's κ:
  - **LLM axis** — classifier ↔ multi-run LLM **consensus** (`final_label`) over **205** labeled
    participant segments.
  - **Human axis** — classifier ↔ **human consensus** over **66** validation items.
- **Uncertainty:** participant-clustered bootstrap **95% CIs** (resample participants, not segments).
- **Paired contrast:** for the headline arms, a *paired* cluster-bootstrap Δκ vs the A1n baseline on
  identical items (n_boot = 3000) — the honest within-fold comparison (`_paired_delta.py`, `_csweep2.py`).

**Data.** n ≈ 32 participants; **205** LLM-labeled + **134** "No code" participant segments; **66** human
consensus codes.

**Features.** Cached **4096-d Qwen3-Embedding-8B** vectors, **L2-normalized**. (The context arm additionally
re-embeds preceding turns; a local-MiniLM contrast is used to get a clean within-embedder signal.)

**Baseline to beat — A1n:** a single class-weighted 6-class L2-LogReg probe on the Qwen features (the prior
best / shipped consensus probe).

**Success bar (CI-aware), clear *either*:**

- classifier ↔ LLM grouped **κ ≥ 0.45** (the human↔human ceiling), **OR**
- classifier ↔ human **κ ≥ 0.50** (approaching LLM↔human = **0.537**).

---

## Master results table

κ reported as **point [95% CI]**; LLM axis over 205, human axis over 66. "—" = CI not part of the
authoritative result set for that sub-variant. Baseline A1n: LLM **0.283**, human **0.365**.

| Arm | Family | LLM κ (205) [95% CI] | Human κ (66) [95% CI] | Beats A1n? | Clears bar? |
|---|---|---|---|---|---|
| **A1n** — class-wt 6-class LogReg probe (Qwen) | baseline | **0.283** [0.203, 0.345] | **0.365** [0.228, 0.513] | — | no |
| ⭐ **ens_softavg, C=4 (WINNER)** | S6 per-rater ens | **0.361** [0.281, 0.432] | **0.450** [0.319, 0.599] | **both axes** | no — grazes (human CI incl. 0.50) |
| ens_softavg, C=1 | S6 per-rater ens | 0.325 [—] | 0.389 [—] | both axes | no |
| ens_softavg, human-weighted (S6b) | S6 per-rater ens | 0.318 [—] | 0.372 [—] | both (but < uniform) | no |
| ens_majority | S6 per-rater ens | 0.304 [—] | 0.354 [—] | LLM only | no |
| per-rater, qwen-only | S6 per-rater ens | 0.312 [—] | 0.346 [—] | LLM only | no |
| 5-class MLP-KL-soft | S2 soft-label | 0.362 [0.302, 0.439] | 0.229 [—] | LLM only (drops No-code) | no |
| 6-class MLP-KL-soft | S2 soft-label | 0.310 [—] | 0.339 [—] | LLM only | no |
| 6-class MLP, hard CE (best human) | S2 soft-label | 0.292 [—] | 0.377–0.388 [—] | both (LLM marginal) | no |
| confidence-weighted probe | S2 soft-label | 0.278 [—] | 0.370 [—] | human only | no |
| linear-KL control | S2 soft-label | ≈ A1n | ≈ A1n | tie | no |
| **two-stage No-code gate** (gate@0.45 → 5-class stager) | S5 structural | 0.281 [—] | **0.447** [0.33, 0.60] | human | no — grazes (human CI incl. 0.50) |
| ordinal arc (mord / cumulative-logit / Frank-Hall) | S5b ordinal | 0.01–0.22 | 0.01–0.22 | no — collapses | no |
| SVC-RBF (best nonlinear) | S4 capacity | 0.230 [—] | — | no | no |
| StandardScaler + probe | S4 capacity | 0.246 [—] | — | no | no |
| MLP (sklearn) | S4 capacity | 0.068 [—] | 0.167 [—] | no | no |
| calibrated probe (Platt / isotonic) | S4 capacity | 0.13–0.20 | — | no | no |
| HistGradientBoosting | S4 capacity | ≤ linear | — | no | no |
| Qwen self-check, isolated | S1 context | 0.283 [—] | 0.365 [—] | reproduces A1n ✓ | no |
| Qwen concat target⊕context (8192-d) | S1 context | 0.227 [—] | 0.360 [—] | no — LLM down | no |
| Qwen concat + two-stage | S1 context | 0.192 [—] | 0.355 [—] | no | no |
| MiniLM isolated (local control) | S1 context | 0.158 [—] | 0.264 [—] | (local baseline) | no |
| MiniLM concat target⊕context | S1 context | 0.125 [—] | 0.269 [—] | no — LLM down | no |
| MiniLM combined single-text | S1 context | 0.043 [—] | — | no — collapses | no |
| naive stack: per-rater × two-stage softavg, C=4 | S8 stacking | 0.247 [—] | 0.398 [—] | no — hurts | no |
| naive stack, C=1 | S8 stacking | 0.073 [—] | 0.119 [—] | no | no |
| hybrid stack: pooled-gate × per-rater stager, C=4 | S8b stacking | 0.189 [—] | 0.292 [—] | no — hurts | no |
| hybrid stack, C=1 | S8b stacking | 0.047 [—] | 0.063 [—] | no | no |
| human anchor (LOPO calib / human-mix) | S7 human-anchor | n/r | ≈ A1n band | no — overfits at n=66 | no |

**Headline:** the winner **dominates A1n on both axes**, but **no** configuration clears the bar
(best LLM **0.361 < 0.45**; best human **0.450 < 0.50**, though its CI includes 0.50).

---

## Per-rater ensemble — *the winner* (S6)

**Method.** The shipped consensus collapses the three LLM raters into a single `final_label` by stage-majority
vote, discarding their disagreement structure. Instead, fit **one class-weighted 6-class L2-LogReg probe per
LLM rater** — `google/gemma-4-31b`, `nvidia/nemotron-3-nano-30b`, `qwen/qwen3-next-80b` — on the cached 4096-d
Qwen features. Label mapping per rater: **ABSTAIN → No-code (class 5)**; **ERROR/unparseable → drop that row
for that rater** (never coerced to No-code). The three per-rater probes are **ensembled by mean
`predict_proba`, then argmax** (`ens_softavg`). LogReg regularization `C` is the one tuned knob.

**Scripts.** `rater_distill.py` (library: per-rater label extraction, the 3×5 OOF probe fits done once, and
the `softavg`/`majority` reducers) · `_run_distill.py` (battery runner → `_distill_results.json`) ·
`_csweep.py` / `_csweep2.py` (LogReg-C sweep + paired Δ for the tuned winner) · `_paired_delta.py`
(paired cluster-bootstrap Δκ vs A1n).

**Result — `ens_softavg`, C=4:** LLM **0.361 [0.281, 0.432]**, human **0.450 [0.319, 0.599]**; per-class
recall (Vig/Avo/AttReg/Meta/Reapp) **.36/.35/.62/.28/.60**. Paired cluster-bootstrap Δ vs A1n on the **LLM
axis = +0.078 [+0.036, +0.132]** (excludes 0 → **reliable**); still reliable at default **C=1**
(Δ **+0.042 [+0.004, +0.095]**). The human-axis paired Δ is positive but its CI overlaps 0 (not reliable;
the campaign log records ≈ +0.085, ns) — consistent with the absolute human gain 0.365 → 0.450. The C-sweep
peaks cleanly at **C=4** (no overfit/plateau pathology in the extended grid).

**Sub-variants** (all on the same OOF probas):

- per-rater, **qwen-only**: 0.312 / 0.346
- **ens_majority** (argmax vote instead of soft-average): 0.304 / 0.354
- **ens_softavg, C=1**: 0.325 / 0.389
- **ens_softavg, human-weighted** (S6b; raters weighted by their human-subset κ): 0.318 / 0.372 — **HURTS**;
  uniform averaging beats agreement-weighting.

**Interpretation.** Rater *disagreement is signal*: modeling each rater and soft-averaging recovers what the
majority-collapse throws away, lifting both axes. The gain concentrates in the frequent stages
(AttReg .52→.62, Reapp .43→.60); the two **rare stages stay stuck** (Avoidance .35, Metacognition .28). This
is the single best model the campaign found, and the LLM-axis improvement over the shipped consensus probe is
statistically reliable — but it is a *better probe*, not a bar-clearing one.

---

## Soft-label distillation (S2)

**Method.** Instead of training on the argmax `final_label`, train on the LLM's **full multi-run ballot
mixture** (`build_soft_targets`) — the per-segment distribution over stages. Variants on identical folds:
(a) small **MLP** with **KL / soft-CE** to the soft target; (b) the same MLP with **hard CE** on the argmax
(architecture ablation, isolating soft-signal from MLP-capacity); (c) **confidence-weighted probe** — A1n's
LogReg with per-sample weight = the LLM ballot's max-confidence; plus a **linear-KL** control (A1n capacity,
soft loss). Levers swept: temperature τ, label smoothing, hidden width, dropout, weight-decay, depth.

**Scripts.** `run_softlabel.py` (library: MLP soft/hard distillation + confidence-weighted probe) ·
`run_softlabel_grid.py` (full table + levers → `_softlabel_results.jsonl`) · `_confirm5.py` (multi-seed
confirmation of the 5-class champion) · `_selfcheck_softlabel.py` (reproduces A1n exactly + inspects the soft
targets).

**Results.**

- **6-class MLP-KL-soft:** 0.310 / 0.339
- **5-class MLP-KL-soft:** **0.362 [0.302, 0.439]** / 0.229 — the campaign's **best LLM-axis point**, but the
  5-class formulation *drops the No-code class* and the human axis collapses to 0.229.
- **hard 6-class MLP:** best **human ≈ 0.377–0.388** (LLM ≈ 0.292)
- **confidence-weighted probe:** 0.278 / 0.370
- **linear-KL control:** ≈ A1n on both axes

**Interpretation.** The soft signal lifts **only the axis it optimizes** — the LLM's own inter-run
disagreement — and slightly *hurts* the human axis. No variant clears the bar. Critically, the 5-class
MLP-KL's 0.362 LLM κ is an artifact of dropping No-code, not a real gain (its human κ craters). This arm and
S6 converge on the same LLM ≈ 0.36 frontier by an unrelated mechanism.

---

## Two-stage No-code gate (S5)

**Method.** Decouple the "Is this codable at all?" decision from the 5-stage decision. Stage 1: a **binary,
class-weighted No-code-vs-VAAMR gate** thresholded at **0.45**. Stage 2: a **class-weighted 5-class stager**
trained only on the VAAMR rows. At inference, route through the gate, then the stager.

**Script.** `ml2.py` (the `S5_twostage` block; same file also runs the S4 capacity probes and an exploratory
S7 human-upweight lever).

**Result.** LLM **0.281**, human **0.447 [0.33, 0.60]** — the **best human-axis single lever**, achieved at
**no LLM-axis cost**. The human CI includes 0.50 (point below); it does not clear the bar but grazes it.

**Interpretation.** The No-code/VAAMR boundary is the most learnable structural distinction in the data;
isolating it is the cheapest human-axis win. It is *structural* (orthogonal to the per-rater lever), which is
why Wave 2 tried to stack it with S6 — unsuccessfully (see Stacking).

---

## Model capacity (S4)

**Method.** Swap the linear probe for higher-capacity learners on the same 4096-d Qwen features: sklearn
**MLP**, **HistGradientBoosting**, **SVC-RBF** (kernel/C/γ sweep), probability **calibration**
(Platt / isotonic), and a **StandardScaler** front-end (vs L2-norm).

**Script.** `ml2.py` (S4 block: MLP / GBM / isotonic-calibrated probe). The SVC-RBF and StandardScaler sweeps
were run in the same family.

**Results.** MLP **0.068 / 0.167**; calibrated **0.13–0.20**; SVC-RBF best **0.230** (l2, C=10, γ=scale);
HistGradientBoosting **≤ linear**; StandardScaler **0.246** (worse than L2). **Every** capacity lever lands
**below the linear A1n 0.283.**

**Interpretation.** At n ≈ 32 participants the bottleneck is not model capacity — added nonlinearity overfits
and L2-normalization beats StandardScaler. Rejected. Stay linear.

---

## Context (S1) — the headline hypothesis

**Method.** The LLM sees the preceding turns; the isolated-segment probe does not. Test whether feeding
context *as embedding features* transfers that edge. A clean within-embedder contrast was run **locally with
MiniLM** first (isolated → concat target⊕context → combined single text), then confirmed in the **strong Qwen
space**: re-embed each participant segment's **6 preceding turns**, then (i) self-check the isolated probe,
(ii) **concat** target⊕context (8192-d), (iii) concat **+ the S5 two-stage gate** (stack both structural
levers on context features).

**Scripts.** `run_context_local.py` (MiniLM, endpoint-free) · `run_context_concat.py` (Qwen target⊕context,
8192-d) · `run_context.py` (Qwen context build + two-stage gate; uses the remote embedding endpoint).

**Results.**

- **Local MiniLM:** isolated 0.158 / 0.264 → concat target⊕context **0.125 / 0.269** (LLM *down*) → combined
  single-text **0.043** (collapses — dilution).
- **Qwen3:** self-check isolated **0.283 / 0.365** (reproduces A1n **exactly** ✓) → concat target⊕context
  (8192-d) **0.227 / 0.360** (LLM *down*) → concat + two-stage **0.192 / 0.355**.

**Interpretation.** Context as embedding-concatenation **does not transfer** the LLM's advantage — it *hurts*
at both 384-d and 4096-d. The LLM's edge is **reasoning over** context, not context *presence* in a feature
vector; concatenated context dimensions are noise a linear probe cannot exploit at this n. Rejected. (This
was the campaign's highest-leverage untested swing; its failure is a primary reason the verdict is a *data*
ceiling.)

---

## Ordinal-arc decoding (S5b)

**Method.** Exploit VAAMR's ordinal developmental arc with ordinal decoders: `mord`, cumulative-logit, and a
Frank-Hall reduction, instead of nominal multiclass.

**Script.** `ordinal_twostage.py` (the S5b arm; `gate_sweep.py` separately tunes the S5 No-code-gate
threshold across both axes).

**Result.** **0.01–0.22** on both axes — **collapses** to the endpoints / the mean.

**Interpretation.** The ordinal assumption is too strong for this label distribution; the decoders degenerate.
Rejected.

---

## Human anchor / calibration (S7)

**Method.** Can the scarce **66 human consensus codes** pull the Qwen probe toward human-level VAAMR
*without overfitting*? Variants vs A1n: **leave-one-participant-out** human calibration, and
**human-weighted mixing** of the LLM-trained probe with a human-fit adjustment.

**Script.** `run_human_anchor.py`.

**Result.** **No reliable gain.** Consistent with the per-rater finding that *weighting by human
agreement hurts* (S6b), and n = 66 is far too small to calibrate on without overfit — the human axis did
not move past its A1n band and the bar was not reached. (Exact per-variant κ were not cleanly logged; the
qualitative outcome is firm.)

**Interpretation.** Human supervision at n = 66 is a *validation* signal, not a *training* signal at this
scale — anchoring/calibrating in-loop overfits. The robust use of the human codes stays the held-out
gate, not in-loop calibration. Rejected.

---

## Stacking the winners (S8 / S8b)

**Method.** Combine the two independent levers — the S5 No-code structure and the S6 per-rater ensemble.
**S8 (naive):** one **two-stage** (No-code gate → 5-class stager) probe **per rater**, soft-averaged.
**S8b (hybrid):** a single **pooled No-code gate on the consensus** × a **per-rater 5-class ensemble stager**.

**Scripts.** `run_wave2_stack.py` (S8) · `run_wave2b_hybrid.py` (S8b).

**Results.** S8: C=1 **0.073 / 0.119**, C=4 **0.247 / 0.398**. S8b: C=1 **0.047 / 0.063**, C=4
**0.189 / 0.292**. Both **HURT** relative to either lever alone.

**Interpretation.** Splitting the 134 No-code examples three ways **starves each per-rater gate** — the
structural lever needs the pooled negatives it had in S5. The two levers are not additively stackable at
n ≈ 32. Rejected.

---

## Convergent ceiling & verdict

**No cheap classifier clears the LLM-equivalence bar at this data scale.**

| Bar | Target | Best achieved | Cleared? |
|---|---|---|---|
| classifier ↔ LLM (grouped κ) | ≥ 0.45 (human↔human) | **0.361** — S6 ens_softavg C=4 (paired Δ +0.078 ✓ vs A1n) | no |
| classifier ↔ human (κ) | ≥ 0.50 (→ LLM↔human 0.537) | **0.450** [0.319, 0.599] — S6 (CI includes 0.50, point below) | no (grazes) |

**Why this is a data ceiling, not a method gap.** **Three independent methods converge on the same frontier**
— per-rater ensembling (S6: LLM 0.361), soft-label MLP distillation (S2: LLM 0.362), and structural No-code
decoupling (S5/S6: human ≈ 0.45) — and **every** further capacity, calibration, ordinal, context, and
stacking lever ties or hurts. Convergence of unrelated methods on **LLM κ ≈ 0.36 / human κ ≈ 0.45** is the
signature of a data ceiling.

**The binding bottleneck** is the **two rare stages** — Avoidance and Metacognition, recall ≈ .28–.35 across
*every* arm — compounded by the **shallow VAAMR separability of a content-trained embedding** and **n ≈ 32
participants**. Gains concentrate in the frequent stages; the rare ones do not move regardless of method.

**Verdict — ship as assistive, not autonomous.** The best shippable scaler is **S6 `ens_softavg` (C=4)**: one
class-weighted 6-class L2-LogReg probe per LLM rater (gemma-4-31b / nemotron-3-nano-30b / qwen3-next-80b;
ABSTAIN→No-code, ERROR→drop) on cached 4096-d Qwen features, ensembled by mean `predict_proba`, argmax — no
LLM at inference. At human↔LLM = 0.537 and human↔human ≈ 0.45, a scaler at human 0.45 reproduces the coding
about as well as a *second human rater* on the majority stages. Deploy it as a **gated pre-labeler** —
confidence gate + **mandatory human review on abstentions, the two rare stages, and low-margin items** — and
**re-evaluate the bar as labeled participants accrue** (the harness is frozen at seed 42 and re-runnable).
**Do not** add context-concatenation, nonlinearity, calibration, ordinal decoding, or No-code/per-rater
stacking — all characterized here as flat-to-harmful at this n.

---

## Reproduce

Each script is **standalone**. From the repository root:

```bash
cd <repo-root>
python src/experiments/classification_scaler/<script>.py
```

- Reads the corpus from **`data/Meta`** via the evaluation harness
  (`src/experiments/gnn_reliability/{harness,baselines}.py`), which builds the seed-42 participant-grouped
  folds and the dual-axis scorer.
- Requires the **cached 4096-d Qwen3-Embedding-8B** vectors (`H.get_embeddings(df, 'qwen', ABS)`); all
  model-lever, soft-label, ensemble, and stacking arms run on those cached features (CPU-only is fine).
- The **LM Studio embedding endpoint at `http://10.0.0.58:1234/v1`** (model
  `text-embedding-qwen3-embedding-8b`) is **only** needed for the **context arm** (`run_context.py`,
  `run_context_concat.py`) to embed preceding turns; `run_context_local.py` is endpoint-free (local MiniLM).

**Script → experiment map.**

| Script | Produces / role |
|---|---|
| `rater_distill.py` | S6 library — per-rater labels, OOF probe fits, `softavg`/`majority` ensembles, soft/hard MLP |
| `_run_distill.py` | S6 battery runner → `_distill_results.json` |
| `_csweep.py`, `_csweep2.py` | LogReg-C sweep + paired Δκ for the tuned winner |
| `_paired_delta.py` | Paired cluster-bootstrap Δκ (ens_softavg, mlp_soft_kl) vs A1n |
| `run_softlabel.py` | S2 library — MLP soft/hard distillation + confidence-weighted probe |
| `run_softlabel_grid.py` | S2 full table + levers → `_softlabel_results.jsonl` |
| `_confirm5.py` | S2 multi-seed confirmation of the 5-class champion |
| `_selfcheck_softlabel.py` | A1n self-check + soft-target inspection |
| `ml2.py` | S4 capacity (MLP/GBM/calibrated) + S5 two-stage No-code gate + S7 human-mix |
| `run_context_local.py` | S1 MiniLM context contrast (endpoint-free) |
| `run_context_concat.py` | S1 Qwen target⊕context concat (8192-d) |
| `run_context.py` | S1 Qwen context + two-stage gate (remote endpoint) |
| `run_wave2_stack.py` | S8 naive per-rater × two-stage stack |
| `run_wave2b_hybrid.py` | S8b pooled-gate × per-rater stager stack |

Result artifacts in this directory: **`_distill_results.json`** (S6 battery) and
**`_softlabel_results.jsonl`** (S2 grid). The authoritative narrative is **`CAMPAIGN_LOG.md`**.
