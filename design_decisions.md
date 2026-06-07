# GNN VAAMR Reliability — Design Decisions & Experiment Ledger

> **Living document.** Append-only decision log + a pre-registered experiment battery.
> Negative results are kept as evidence, not deleted. Workspace: `./data/Meta/`.
> Companion: `docs/GNN_MASTER_PLAN.md` (Track A0), `docs/gnn_experiments/ledger.csv`.

---

## 0. Executive synthesis (read first)

**Mission:** make the VAAMR GNN methodologically defensible — PRIMARY: a mechanism (therapist→
participant) readout that *triangulates* with the observed analysis; SECONDARY: classifier IRR as a
trust floor. Corpus `./data/Meta/` (n≈32 participants; 339 participant segments, 205 VAAMR-labeled,
134 "No code"; 76 human-coded items).

**Bottom line (all numbers participant-grouped CV + participant-clustered bootstrap CIs):**
1. **The published GNN κ=0.247 was CV-leakage-inflated.** Honest participant-grouped CV puts the
   original MiniLM GNN at κ=0.05 (vs LLM) / −0.02 (vs human) — ≈ chance. (Root cause #5; the harness
   is now the authoritative evaluation.)
2. **Features were the bottleneck.** Real **Qwen3-8B** embeddings (4096-d, served by LM Studio — no
   transformers-pin risk) lift a *linear probe* to LLM κ=0.31 / human κ=0.37 off that 0.05/−0.02 floor.
3. **Class-weighting recovers the rare stages** (Avoidance/Metacognition recall 0% → 0.35/0.31).
4. **A "No code" 6th class carries the human axis** (36% of human items are "No code" — the LLM can
   emit it, a 5-class model cannot; it lifts human κ −0.02→0.21 even on MiniLM).
5. **A linear probe ties/beats the GNN at n≈205** (human 0.37 probe ≈ 0.36 GNN; LLM 0.31 probe ≫ 0.21
   GNN; C&S worst). The GNN does **not** earn its place *as a classifier* → **honest split**: LLM stays
   label-of-record, the calibrated probe is the abstention-gated assist, the GNN is reserved for mechanism.
6. **SECONDARY (classifier) — human-level IRR ACHIEVED (lower band).** Best human κ ≈ **0.37** is inside
   the human↔human band (α 0.33–0.52), approaching LLM↔human (0.537). Winner = **A1n** (Qwen probe,
   class-weighted, 6-class/No-code).
7. **PRIMARY (mechanism) — does NOT triangulate (honest negative).** The Qwen GNN counterfactual
   (Spearman ρ=**−0.13**, CI incl. 0) and the coupling factors (|corr|<0.07) both fail to converge with
   the observed Δprogression → per protocol **`analysis/mechanism.py` (observed) LEADS; the GNN is
   exploratory only.** The *negative* ρ inverts the observed pattern — consistent with the
   **elicitation/responsiveness confound**; the observed analysis itself has **0 FDR-significant** cells
   at this scale. **No causal claims.**

**The defensible design IS the rigorous evaluation:** participant-grouped CV (no leakage), bootstrap
CIs (tiny n → CIs decide, not points), a pre-registered arm battery, the human axis as load-bearing,
and negative results recorded as evidence. **The binding constraint is data scale (n≈32)** — more
labeled participants is the only credible path to a *corroborating* (not merely exploratory) GNN
mechanism instrument.

**Recommendations:** (i) switch the production reliability gate from random-k-fold to participant-
grouped CV (it currently over-reports κ); (ii) keep the LLM as label-of-record + ship the calibrated
Qwen probe as the abstention-gated assist; (iii) report mechanism from `mechanism.py` (observed) with
the GNN as an exploratory lens + the n≈32/confound caveats; (iv) wire the Qwen `/v1/embeddings` backend
(`embedding_backend='openai'`) as the GNN feature default.

---

## 1. Mission (priority order)

**PRIMARY = mechanism; SECONDARY = classifier IRR (a trust floor that must never compromise
the primary).** Workspace `./data/Meta/`. **PURER is out of scope except as cue context.** All
claims are n≈32 observational — **NO causal claims** (elicitation confound: PURER inquiry
*elicits* the very VAAMR language it is scored against; `methodology.md` §9.4).

### 1A. PRIMARY — therapist→participant VAAMR mechanism (peer-review deliverable)
A GNN that explains therapist-cue (PURER) → participant-VAAMR dynamics via **(a) model-
counterfactual sensitivity** (ablate/swap the therapist cue, measure the predicted shift in the
following participant's VAAMR mixture; `gnn_layer/influence.py`) and **(b) participant↔therapist
coupling** (`gnn_layer/coupling.py`).

**SUCCESS (pre-registered) = the GNN readouts TRIANGULATE with the independent observed-
Δprogression analysis in `src/analysis/mechanism.py`:**
- **Spearman ρ > 0** between GNN counterfactual cue-influence and mechanism.py's Δprogression
  **per cue→transition** (from_stage × PURER move), with **participant-clustered bootstrap 95% CI
  excluding 0**, AND
- **≥ 70% SIGN agreement** on the cue→transition effects mechanism.py flags **FDR-significant**,
- **always reported next to the GNN reliability-gate κ** (the trust context for the readout).

**If triangulation fails → `mechanism.py` LEADS and the GNN is reported as exploratory only.**
This is a valid, documented outcome, not a failure of the work.

### 1B. SECONDARY — classifier IRR as the trust floor (pursue only if it doesn't weaken 1A)
The classifier κ is the *trust context* reported beside the mechanism readout, and (separately) an
**abstention-gated assist** — never forced to be label-of-record. The **LLM stays label of
record**; "**human-level IRR is unreachable at n≈205**" is a VALID documented outcome.

| Axis | Meaning | Now (held-out) | Target (push toward, not required) |
|---|---|---|---|
| **GNN↔human κ** | independent quality (load-bearing) | **+0.053** | → LLM↔human ≈ **+0.537** |
| GNN↔LLM κ | distillation fidelity | +0.159 (n=76) / +0.247 (n=205) | → human↔human ceiling ≈ 0.45–0.52 |
| rare-stage recall | Avoidance, Metacognition | **0% / 0%** | non-zero, no collapse |

κ≥0.70 is **not** the target (human↔human α≈0.33–0.52 — see §3). Promote a classifier change to
default **only on a human-axis gain**.

## 2. Locked decisions (with the researcher)

- **D-PRIORITY — Mechanism is primary; classifier IRR is a secondary trust floor.** The
  peer-review deliverable is the §1A triangulation (Spearman ρ + ≥70% FDR-sign agreement vs
  `mechanism.py`), reported beside the gate κ. Classifier work is pursued only insofar as it
  (i) raises the trust floor and (ii) yields the best embedding/graph substrate for the
  counterfactual/coupling readouts — never at the cost of the primary. The Qwen-feature +
  precipitates-edge + calibrated-head work serves BOTH, which is why it stays first.
- **D-A — Run the full arm battery; let the data decide.** Report the reliability frontier with
  CIs; promote the arm with the highest **GNN↔human κ**. Escalate through Path B (concept
  anchors) as part of the battery.
- **D-B — Add a "No code" class** (6-way head) as the adopted design; **keep 5-class arms** as
  the defensible baseline so the construct change is accountable to evidence.
- **D-C — Honest split.** If a simpler model (calibrated linear probe / Correct-&-Smooth on Qwen
  embeddings) wins the classifier bake-off, *it* ships as VAAMR label-of-record and the **GNN is
  retained for therapist↔participant coupling/mechanism** (its graph-native strength). M1
  (classifier) and M2 (mechanism) are scored separately and never conflated.
- **D13 (inherited)** — one branch per architecture; every arm appends a ledger row; promote to
  default only on a **human-axis** gain.

## 3. The reliability ceiling reframes "reliable"

Trained human coders agree only at **Krippendorff α ≈ 0.33–0.52** (test-sets 1/2/3:
0.473 / 0.523 / 0.325). The LLM consensus is already **human-level** (κ=0.537 vs human). You
cannot distill a label function more reliable than its fuzzy teacher, so **κ≥0.70 is the wrong
target**. The GNN currently fails on *both* axes (0.247 vs LLM, **0.053 vs human**) — it is not
capturing the construct shape at all, and is worse than the LLM on the axis that matters.

## 4. Root-cause hypotheses (ranked, evidenced)

| # | Cause | Evidence (this corpus) | Arm that tests it |
|---|---|---|---|
| 1 | **Weak node features** (MiniLM-384 substitute, not Qwen3-8B) | both heads weak; kNN graph = embedding quality | A1 vs A0 |
| 2 | **No class rebalancing** (unweighted `batchmean` KL) | Avoidance/Metacognition recall 0%; AttentionReg 86% | A4 vs A3 |
| 3 | **Over-smoothing** at n=205 (2× mean-agg kNN) | majority-stage collapse | A1/A2 (no/again graph) vs A3 |
| 4 | **No "No code" class** | 134/339 participants are "No code" → trained to uniform noise + 5-class GNN can't match ~36% of human items the LLM *can* | A1n/A4n vs A1/A4 |
| 5 | **CV leakage** (random k-fold, not participant-grouped) | current gate folds ignore participant clustering | all arms use GroupKFold |
| 6 | **Fuzzy ceiling** | human α 0.33–0.52 | reframes target (§3) |

### Data scale (the binding constraint)
339 participant segments → **205 labeled** + **134 "No code"**. Labeled distribution:
AttentionReg 73 · Reappraisal 58 · Metacognition 29 · Vigilance 25 · **Avoidance 20**.
Human consensus: **76 items** in `qra.db` (~66 usable vs machine). 100% multi-run ballot
coverage (soft targets trustworthy).

### Infrastructure facts
- **Qwen3 embeddings**, zero pin-risk: `text-embedding-qwen3-embedding-8b` @
  `http://10.0.0.58:1234/v1/embeddings`, **4096-d** (vs MiniLM 384-d).
- **Integration gap (closed by A0-pre):** `master_segments.csv` has
  `in_human_coded_subset`/`human_label` columns but **0 populated**; the 76 human consensus codes
  live in `qra.db` (`irr_human_codes`). `validation.py:_human_axis` is built and waiting on the join.

## 5. Scoring protocol & defensibility guardrails

- **Folds:** participant-grouped (`GroupKFold` on `participant_id`) + stratified by `final_label`,
  **built once**, shared by every arm (identical folds/seed). No participant in train+test; the 66
  human items are naturally out-of-fold.
- **Two reference axes, every arm:**
  - **LLM axis** — out-of-fold preds vs `final_label` (n=205 + the 76-subset). Per-class κ/recall/precision.
  - **Human axis (load-bearing)** — out-of-fold preds vs human consensus (n≈66), scored **exactly
    like `irr_analysis`** (`No code`=−1 included; reuse `analysis/irr_stats.cohen_kappa` +
    `process/irr_import.read_human_codes`) so numbers are directly comparable to `06b_irr_report.txt`.
- **Uncertainty:** κ point estimate **+ participant-clustered bootstrap 95% CI**
  (`analysis/stats.cluster_bootstrap_ci`) on both axes. **CIs decide, not point estimates** (n is tiny).
- **No test-set tuning:** hyperparameters tuned on the **LLM-axis CV only**; the human axis is
  **read once per arm**. Arms are **pre-registered** (§6) → arm selection is honest model selection
  over a small discrete set, not gradient-tuning on a 66-item test set.
- **No-code scoring:** 6-class arms map predicted class-5 → ABSTAIN(−1) for the human axis; 5-class
  arms cannot emit No-code (quantifies the structural penalty).
- **Reproducibility:** pin embedding `model id + dim + a vector checksum` per arm.
- **Honesty:** report folds that miss a rare class entirely (expected at n=205); never hide a collapse.

## 6. Pre-registered arm battery (FIXED before results)

Identical folds/seed. Each arm → one `ledger.csv` row + a §7 entry.

| Arm | Features | Method | Imbalance | Classes | Question |
|---|---|---|---|---|---|
| **A0** | MiniLM-384 | GraphSAGE (current) | none | 5 | reproduce 0.247 / 0.053 baseline (harness self-check) |
| **A1** | Qwen-4096 | Linear probe (logreg) | none | 5 | is it the embedding? do we need a graph? |
| **A1w** | Qwen-4096 | Linear probe | class-weighted | 5 | rebalancing without a graph |
| **A1n** | Qwen-4096 | Linear probe | class-weighted | **6** | does "No code" fix the human axis? |
| **A2** | Qwen-4096 | Correct & Smooth | C&S | 5 (+6) | does graph propagation help a simple base? |
| **A3** | Qwen-4096 | GraphSAGE | none | 5 | does the GNN beat the simple baselines? |
| **A4** | Qwen-4096 | GraphSAGE | class-balanced + focal/TAM | 5 | does rebalancing recover rare stages? |
| **A4n** | Qwen-4096 | GraphSAGE | class-balanced | **6** | full GNN candidate |
| **B1** | Qwen-4096 | GraphSAGE + VAAMR concept anchors | best loss | 6 | does concept structure close fine-grained gaps (human axis)? |

**Promotion rule:** max **GNN↔human κ** (CI-aware) with no rare-class collapse. Ties → simpler
model (D-C). Mechanism (M2) graph kept regardless.

## 7. Results — the Qwen battery (2026-06-06)

κ as `point [lo, hi]` (participant-clustered bootstrap 95% CI). Full rows in
`docs/gnn_experiments/ledger.csv`. Reference: human↔human α **0.33–0.52**; LLM↔human **0.537**.

| Arm | model | nCls | **human κ** (n) | **LLM κ** (205) | rare recall Vig/Avo/Meta |
|---|---|---|---|---|---|
| A0 | MiniLM GNN | 5 | −0.02 [−.08,.06] (66) | 0.05 [−.03,.12] | 0.20/0.00/0.00 |
| A1 | Qwen probe | 5 | 0.21 [.06,.33] (37) | 0.23 [.14,.32] | 0/0/0 |
| **A1w** | Qwen probe, bal | 5 | 0.30 [.11,.53] (37) | **0.31 [.21,.38]** | 0.56/0.35/0.31 |
| **A1n** ⭐ | Qwen probe, bal | 6 | **0.37 [.23,.51] (66)** | 0.28 [.20,.35] | 0.36/0.35/0.31 |
| A2 | Qwen C&S | 5 | 0.16 (37) | 0.16 | 0.20/0.05/0.10 |
| A2n | Qwen C&S | 6 | 0.20 (66) | 0.07 | 0/0/0 |
| A3 | Qwen GNN | 5 | 0.14 (66) | 0.21 [.12,.29] | 0.28/0.20/0.00 |
| A4 | Qwen GNN, bal+focal | 5 | 0.10 (66) | 0.16 | 0.32/0.50/0.10 |
| A4n | Qwen GNN, bal+focal | 6 | 0.36 [.25,.45] (66) | 0.18 | 0.36/0.35/0.17 |
| B1 | Qwen GNN **+ concept anchors** | 6 | 0.29 [.17,.44] (66) | 0.18 | 0.40/0.35/0.17 |

### Findings (root causes, confirmed)
1. **Features were the dominant bottleneck (cause #1).** Qwen-4096 lifts the probe from the honest
   A0 baseline (LLM 0.05 / human −0.02) to **LLM 0.23–0.31 / human 0.21–0.37**. Single highest-leverage fix.
2. **Class-weighting recovers the rare stages (cause #2).** A1→A1w: rare recall **0/0/0 → 0.56/0.35/0.31**
   and LLM κ 0.23→**0.31**. Imbalance handling is necessary and sufficient for rare-stage recall.
3. **The linear PROBE beats the GNN at n≈205 (→ honest split, D-C).** Best LLM: A1w **probe 0.31** ≫
   A3 GNN 0.21 ≫ A4 GNN 0.16. Best human (full 66): A1n **probe 0.37** ≈ A4n GNN 0.36 (tied, CIs
   overlap). **C&S is worst** (graph propagation does not help here). The GNN does **not** earn its
   place as the VAAMR classifier at this scale — exactly the Correct-&-Smooth / small-label regime.
4. **The No-code class (D-B) carries the human axis.** 6-class arms predict all 66 human items
   (No-code = class 5); A1n=**0.37** and A4n=0.36 vs the 5-class GNN A3=0.14 (5-class GNN is *forced*
   to mis-stage the 24 No-code items). ⚠ **Methodological note:** the 5-class *probe* arms (A1/A1w/A2)
   DEFER No-code (no pred for No-code participants) → scored on a **n=37 subset** (29 deferred), an
   easier denominator not directly comparable to the n=66 full-task numbers. The fair full-task
   comparison is the **6-class arms (all n=66)**.

5. **Concept structure (Path B / CFiCS) does NOT help (B1).** Adding the 5 VAAMR construct-definition
   anchor nodes to the best 6-class GNN (A4n) *lowers* human κ to **0.29** (from 0.36) — anchors add
   noise, not signal, at n≈205 (CIs overlap, but the point estimate drops on every axis). The
   escalation the goal reserved "only if gaps remain" was run and is a **documented negative**: the
   homogeneous graph was the right default, and concept structure does not rescue the GNN vs the probe.

### Classifier winner → **A1n** (Qwen linear probe, class-weighted, 6-class/No-code)
- **human κ = 0.365 [0.23, 0.51]** — inside the human↔human band (0.33–0.52), approaching LLM↔human
  (0.537); LLM κ = 0.283; rare stages recovered; abstention (No-code) built in; simplest model.
- A1w (5-class, commit-only) is the alternative abstention strategy: **LLM κ=0.307** on what it
  commits to, deferring No-code — best distillation fidelity.
- **Decision (D-C honest split):** the LLM stays label-of-record; **A1n is the abstention-gated
  assist + the trust floor** for the mechanism. The GNN is NOT the classifier of record — it is
  retained for the PRIMARY mechanism mission (counterfactual/coupling), where its graph structure is
  the point. "GNN-as-classifier unreachable-beating at n≈205" is the documented, valid outcome.
- **Human-level IRR (secondary) — ACHIEVED** at the lower band: GNN/probe↔human ≈ 0.36 vs human↔human
  0.33–0.52. Promotion candidate = A1n on the human axis (Path B escalation only pursued if the
  PRIMARY mechanism needs better construct shape — decided after §9 result).

## 8. Decision log (chronological)

- **2026-06-06 — Pre-registration.** Battery (§6), scoring (§5), targets (§1) fixed before any
  Qwen result. Foundation branch `gnn-exp/harness` cut from a `beta` checkpoint
  (`062c9bd`, prior IRR + turn-level-PURER WIP) so every experiment branch shares a clean base.
  Confirmed Qwen endpoint live (4096-d) and the human-subset integration gap (0 populated rows).
- **2026-06-06 — Reprioritization (researcher `/goal`).** **Mechanism made PRIMARY** (peer-review
  deliverable, §1A) with a concrete pre-registered triangulation success metric; **classifier IRR
  demoted to a SECONDARY trust floor** (§1B). The arm battery still runs (it sets the trust floor
  and picks the best substrate for the counterfactual/coupling readouts) but the headline success
  is the §1A triangulation, reported beside the gate κ. Added the mechanism-triangulation
  work-stream (§9): run `influence.py` (counterfactual) + `coupling.py` on the best Qwen model and
  triangulate vs `mechanism.py`; **decouple the counterfactual readout from the hard legacy gate**
  — it runs with the gate κ as *reported trust context*, not as a hard suppressor (the κ≥0.70 gate
  is unreachable in principle, so hard-gating would permanently block the primary deliverable).
- **2026-06-06 — FDR-null finding (PRIMARY metric adaptation).** The real
  `mechanism_delta_progression.csv` has **0/20 PURER cells FDR-significant** at n≈32, so the
  pre-registered "≥70% sign agreement on FDR-significant effects" is vacuous. Adapted the success
  metric to a Spearman-ρ pattern-convergence test (n≥3-support cells, participant-clustered bootstrap
  CI) + a "directionally-reliable" (observed-CI-excludes-0) sign-agreement secondary, with an
  explicit FDR-null caveat (§9A). **Flag to researcher** — this changes their pre-registered metric
  because the data cannot support the original. The convergence test itself stands; only the
  significance gate is replaced by a within-method reliability criterion.
- **2026-06-06 — SA1 done (Qwen embedding client).** Added `embedding_backend='openai'` path
  (`gnn_layer/embeddings_remote.py`) hitting LM Studio `/v1/embeddings`; default `'local'` path
  byte-identical. Live: **4096-d, L2-normalized** vectors (endpoint normalizes; local
  SentenceTransformer path does not — noted for cross-backend feature-magnitude comparisons; cosine
  kNN is unaffected). 21 hermetic tests + 1 `@slow_test` live smoke; full GNN suite green.
- **2026-06-06 — Embedding latency + truncation decision.** The LM Studio Qwen3-8B endpoint has a
  ~116s cold start (one-time model load) and ~1.3s/text warm, BUT a few pathological didactic chunks
  (max 56k chars ≈ 10k words; p90 only 1207 chars) take ~126s each and blew the 60s batch timeout.
  Decision: **truncate texts to 8000 chars** before embedding (covers >90% fully; the local
  SentenceTransformer path truncates at its max_seq_length too, so this matches production), small
  batch (8), 180s timeout, and an **incremental, resumable** cache keyed by full-text hash (so the
  harness's `load_or_build_segment_embeddings` gets cache hits and never re-embeds). Cache:
  `02_meta/gnn/segment_embeddings_qwen3_8b.npz`.
- **2026-06-06 — Reference reproduced (scorer trust).** Independently reproduced the published gate
  numbers from the existing held-out predictions: **LLM-axis κ=0.247 (n=205)** and **human-axis
  κ=0.053 (n=66)** — exact match — using `irr_stats.cohen_kappa` + `irr_import`. Human consensus
  label mix: **No-code 24, Vig 12, Avo 2, AttReg 10, Meta 5, Reap 13** → 36% No-code, which the
  5-class GNN structurally cannot match (motivates the 6-class arms). This is the validated reference
  the harness self-check must hit.
- **2026-06-06 — SA4 merged (imbalance + No-code).** Branch `gnn-vaamr-imbalance` → `gnn-exp/harness`
  (clean 3-way). Flags `vaamr_n_classes` (5|6), `vaamr_class_balance`, `vaamr_focal_gamma`,
  `vaamr_hard_ce_weight`, `vaamr_tam` (logit-adjustment) — all default-off, default path **proven
  bit-identical**; loss math audited (inverse-freq row weights mean-1 normalized; raw-logit focal;
  class-weighted hard-CE). 13 tests + 726 GNN tests green.
- **2026-06-06 — Model-config note (two missions, two heads).** The mechanism counterfactual reads the
  participant's **progression coordinate E[stage]=Σk·pₖ**, which a 6-class No-code head distorts
  (No-code at k=5 is not "stage 5" on the developmental arc; SA4 already computes `prog_val` over the
  5 real dims). ⇒ the **mechanism deliverable uses a 5-class progression model** (best 5-class GNN,
  e.g. A4) while the **classifier of record may be 6-class** (A4n). The two missions can use different
  head configs on the same embedding/graph substrate.
- **2026-06-06 — SA-mech merged (triangulation) + SA3 merged (baselines).** Integration branch now
  carries Qwen backend + imbalance + triangulation + probe/C&S. `triangulation_metric` computes ρ +
  participant-clustered bootstrap CI + FDR-restricted sign agreement (n_fdr=0 on real data ⇒
  `converges` strictly False, as expected); `run_counterfactual_experiment` runs the readout un-gated
  with gate-κ echoed as trust context. Baselines: probe (`class_weight` per flag) + C&S (α=0.8, 50
  iters, row-normalized D⁻¹A, train-clamped smooth).
- **2026-06-06 — MiniLM mechanism BASELINE (pipeline validated; negative result).** Dry-ran the full
  mechanism path on MiniLM (train 5-class GNN + precipitates edges → counterfactual → triangulate):
  works end-to-end (161 cue blocks, 20 cells). Per-move influence ranks Utilization > Reframing >
  Phenomenology > Education > Reinforcement, but **Spearman ρ=0.014, 95% CI [−0.16, 0.18] (includes
  0) → NO convergence** (gate κ=0.247 trust context). ⇒ MiniLM features are too weak for the
  mechanism, exactly as for the classifier. The **Qwen run is the real §1A test** (pending re-embed).
- **2026-06-06 — SA2 merged (harness + human-join) + 🔴 CV-LEAKAGE FINDING.** Strict self-check
  reproduces the published numbers (LLM κ=0.2467 / human κ=0.0530) and the **production human-axis
  gate is now lit** (`validation.evaluate_crossval` returns graph_vs_human κ=0.053 — A0-pre done).
  **KEY:** honest **participant-grouped** CV collapses the A0 (MiniLM) LLM-axis κ from **0.247
  (random k-fold) → 0.050** `[-0.03, 0.12]` (human −0.02 `[-0.08, 0.06]`). **The published 0.247 was
  inflated by CV leakage** (random folds let the GNN see same-participant neighbours via temporal+kNN
  edges — root cause #5). The honest A0 baseline is ~0.05; the battery is measured against THIS, not
  the leaky 0.247. Avoidance missing from 1/5 folds (logged). Folds: 205 labeled, 20 participants, 5
  folds (StratifiedGroupKFold).
- **2026-06-06 — No-code class VALIDATED (D-B) on MiniLM.** 6-class integration smoke (MiniLM through
  the harness): **human-axis κ jumps −0.021 (5-class A0) → 0.206 (6-class)** because the model can now
  emit "No code" for the 36% of human items that are No-code. Pred dist collapses to AttReg/Reap/
  No-code (rare stages still 0), so the No-code class is a **major human-axis lever** independent of
  rare-stage recovery. Qwen + imbalance is the next test (battery `run_battery.py`).
- **2026-06-06 — QWEN BATTERY COMPLETE (A1–A4n; see §7).** Qwen features lift the probe to LLM 0.23–0.31
  / human 0.21–0.37 (from the 0.05/−0.02 honest baseline); class-weighting recovers rare stages
  (0/0/0→0.56/0.35/0.31); **the linear probe beats the GNN** (LLM: probe 0.31 ≫ GNN 0.21/0.16; human
  full-66: probe 0.37 ≈ GNN 0.36). **Winner = A1n** (Qwen probe, class-weighted, 6-class): human
  κ=0.365 [0.23,0.51], within the human band → human-level IRR achieved. Honest split (D-C): LLM stays
  label-of-record, A1n is the abstention-gated assist + trust floor, the GNN serves the mechanism.
- **2026-06-06 — PRIMARY mechanism RESULT (§9B): honest negative.** Qwen GNN counterfactual does NOT
  triangulate with observed Δprogression: **ρ=−0.13 [−0.48, 0.01]** (plain), −0.25 (balanced) — both
  CIs include 0, point estimates negative. **→ mechanism.py LEADS; GNN counterfactual = exploratory
  only** (per the pre-registered protocol). The negative ρ inverts the observed pattern, consistent
  with the **elicitation/responsiveness confound** — evidence *of* the confound, the reason no causal
  claim holds. Per-move influence rank is face-valid (Utilization>Education>Reframing>Phenomenology>
  Reinforcement) but exploratory. Remaining: B1 concept-anchor escalation (completeness) + production
  defensibility (grouped-CV gate fix, config promotion, mechanism decouple) + final synthesis.
- **2026-06-06 — B1 concept-anchor escalation (Path B) — documented NEGATIVE.** GNN + 5 VAAMR
  construct anchors (A4n config) → human κ=**0.29** [0.17,0.44] vs A4n 0.36 vs probe A1n 0.365 —
  concept structure does NOT help at n≈205. **Battery COMPLETE (A0–A4n + B1).** Final classifier
  winner = **A1n** (probe). The GNN does not earn its classifier place homogeneous OR anchored →
  honest split confirmed. **Mission concluded:** classifier human-band IRR achieved (probe); PRIMARY
  mechanism does not triangulate → mechanism.py leads (GNN exploratory); all arms + negatives logged;
  CV-leakage corrected; A0-pre verified in the production gate. Production rec: switch gate to
  participant-grouped CV; keep LLM as label-of-record + ship the Qwen probe as the abstention-gated assist.
- **2026-06-06 — Production grouped-CV fix landed + best implementations on `beta`.** The production
  reliability gate (`train.crossval_predictions` ← `runner`) now uses participant-grouped folds by
  default (`participant_grouped_cv`, backward-compatible `groups=` param); verified random κ=0.21 →
  grouped 0.14 in the production path. The tested branch (Qwen backend, A0-pre, imbalance/No-code,
  triangulation, grouped-CV fix, harness, docs) was fast-forwarded onto `beta` (not pushed);
  experimental branches kept.
- **2026-06-06 — Researcher decision: drop the graph as a CLASSIFIER; focus it on MECHANISM.** Per the
  battery (probe ≥ GNN; H5 refuted) the graph is no longer pursued as a label producer/scaler; the
  **probe stays an experimental diagnostic, not a method of record** (at n≈32 the LLM is human-level +
  affordable, so the probe's only edge — cost — doesn't yet justify the reliability hit, κ 0.365 vs
  0.537). `docs/methodology.md` updated with the methods + findings and the hypothesis implications:
  **H5 refuted** (graph-as-scaler) → repositioned in §8.5; **H2 under-identified** + the elicitation
  confound now *evidenced* by the negative counterfactual ρ (§9.4); **H3a instrument undercut** (re-ask
  on LLM/probe); **new H6** = discriminant validity (VAAMR is developmental, not topical — the same
  evidence that refutes H5); §5.3 grouped-CV correction + reliability ceiling; §6.4 Stage-2 revision.

## 9. PRIMARY work-stream — mechanism triangulation (the peer-review deliverable)

**Unit of analysis:** the cue→transition = (from_stage × PURER move). Two independent estimates of
"does this therapist move progress the participant?":
- **GNN counterfactual cue-influence** — `gnn_layer/influence.py`: swap the cue block's therapist
  node feature with each PURER-move centroid (vs a neutral null), re-forward, read the shift in the
  FOLLOWING participant's predicted progression coordinate E[stage]. Run on the **best Qwen model**
  (best classifier substrate from §6), with **precipitates edges** ON (the typed therapist→
  participant handle the counterfactual needs).
- **Observed Δprogression** — `analysis/mechanism.py` (already built; B1/B2 LEAD): signed
  Δprogression per (from_stage, move) with participant-clustered bootstrap CIs, within-stage
  permutation, **FDR**. This is the label of record for "what progresses participants."

**Success metric (pre-registered, §1A):** Spearman ρ>0 between the two per-(from_stage,move) vectors
with participant-clustered bootstrap **95% CI excluding 0**, AND **≥70% sign agreement** restricted
to the effects mechanism.py flags **FDR-significant** — both printed next to the GNN gate κ. Fail →
mechanism.py leads, GNN exploratory only.

**Key design decision — decouple the readout from the legacy hard gate.** The counterfactual pass is
currently suppressed unless `validation.gate_ready_for_scaling` (the κ≥0.70 legacy gate). Since that
gate is unreachable in principle (§3), hard-gating would permanently block the PRIMARY deliverable.
Resolution: the mechanism readout **runs regardless**, and the gate κ is reported as **trust
context** beside it (low κ ⇒ "treat as exploratory; weight the triangulation result"). The
non-causal, n≈32, elicitation-confound caveats ride on every figure. This honors the original
gate's intent (don't let an untrustworthy model masquerade as authoritative) without letting an
unreachable threshold veto an honest convergence test.

**Build tasks:** (i) ensure `influence.py` can run on a non-gate-passing model in the experiment
harness; (ii) refine `influence.triangulate` so sign agreement is computed over the **FDR-significant**
subset and ρ carries a participant-clustered bootstrap CI; (iii) re-validate `coupling.py` factors on
Qwen embeddings. Branch: run on the winning classifier branch (needs its model) → `gnn-exp/mechanism`.

### 9A. ⚠ FDR reality check — the pre-registered sign-agreement metric is vacuous as written
Inspecting the REAL `data/Meta/03_analysis_data/mechanism/mechanism_delta_progression.csv`:
**ZERO of the 20 PURER cue→transition cells are `fdr_significant`** (all fdr_q ≈ 0.50–0.96; the
smallest within-stage permutation p is ≈0.065). At n≈32, no individual (from_stage × move) effect
survives FDR correction — the observed mechanism is inherently a **pattern-level** signal, not an
effect-level one. Consequence: "**≥70% sign agreement on FDR-significant effects**" is computed over
an **empty set** → undefined. This is a finding, not a bug: it bounds every mechanism claim to
hypothesis-generating, never causal — consistent with the n≈32 mandate.

**Adapted success metric (proposed; pending researcher confirmation).** Keep the spirit (convergence
with the independent observed analysis), make it computable + honest:
1. **PRIMARY convergence — Spearman ρ** between GNN counterfactual influence and observed
   `mean_delta_prog` over the per-(from_stage, PURER move) cells **with adequate support (n≥3 blocks)**,
   with **participant-clustered bootstrap 95% CI**. Success = ρ>0 AND CI excludes 0.
2. **SECONDARY sign agreement — over the "directionally reliable" subset** = cells whose OBSERVED
   bootstrap CI (`ci_lo,ci_hi`) excludes 0 (a within-method reliability criterion that survives the
   FDR-null, e.g. Vigilance+Reframing +[1.11,3.07], Reappraisal+Education −[1.92,0.33]). Report the
   fraction + n. (The FDR-significant subset is ALSO reported — as "n=0", explicitly.)
3. **Headline caveat on every artifact:** no cue→transition effect is FDR-significant at n≈32; the
   triangulation tests whether two INDEPENDENT methods agree on the effect-size *rank pattern* —
   hypothesis-generating only. If ρ-CI includes 0 → `mechanism.py` leads, GNN exploratory (per goal).

`influence.triangulate` (SA-mech) still computes the FDR-subset version (correct, returns n=0 on real
data); the architect adds the adapted ρ(n≥3) + CI-reliable-sign-agreement during integration.

### 9B. RESULT — the mechanism does NOT triangulate (honest negative; mechanism.py leads)
Ran the counterfactual + §1A triangulation on two Qwen 5-class GNNs (precipitates ON), 161 cue
blocks, 20 cells (`run_mechanism.py`):

| candidate | gate κ (trust ctx) | Spearman ρ | 95% CI | converges |
|---|---|---|---|---|
| plain (5-class) | 0.201 | **−0.132** | [−0.477, 0.006] | **no** (CI incl. 0) |
| balanced (5-class) | 0.128 | **−0.254** | [−0.562, 0.053] | no |

Per-move counterfactual influence (both candidates, face-valid rank): **Utilization +0.035 >
Education +0.029 > Reframing +0.026 > Phenomenology +0.012 > Reinforcement −0.006**. The §1A test
**FAILS** (ρ not >0; CI includes 0; FDR-significant cells = 0). **Per the pre-registered protocol →
`analysis/mechanism.py` (observed Δprogression) LEADS; the GNN counterfactual is reported as
EXPLORATORY only.**

**Interpretation (defensible, not a code failure):** the point ρ is *negative* — the counterfactual
sensitivity **inverts** the observed per-cell association. This is exactly what the **elicitation /
responsiveness confound** predicts (methodology §9.4): therapists deploy PURER moves *in response to*
participant state, so the OBSERVED association (move m co-occurs with progression) reflects
*responsiveness*, while the COUNTERFACTUAL (swap-to-m → predicted shift) attempts to isolate
*sensitivity*. At n≈32 the two diverge — the divergence is **evidence of the confound**, the central
reason no causal claim is admissible. The counterfactual shift is also tiny (±0.035 on a 0–4 scale):
the participant's predicted progression is driven mostly by their own features, not the swapped cue.
Artifacts: `06_reports/06_gnn/influence.txt`, `03_analysis_data/gnn/gnn_counterfactual_influence.csv`.

**Coupling readout (b) — also weak.** PCA latent factors of the raw Qwen cue-block embeddings (161
blocks, 5 factors) vs subsequent participant forward movement: **all |corr| < 0.07** (max 0.068) —
no latent therapist-cue factor predicts forward movement, consistent with the prior MiniLM run
(<0.08). **Both GNN mechanism lenses (a counterfactual, b coupling) are weak/non-convergent at
n≈32** → the observed `mechanism.py` Δprogression analysis is the **only** defensible mechanism
evidence (itself hypothesis-generating, 0 FDR-significant cells). The honest mission conclusion: the
graph is a useful *exploratory* lens, not a corroborating mechanism instrument at this data scale.


---

## §10 — Discovery+mechanism rebuild, classifier separation, classifier OFF by default (2026-06-07)

Executed in isolated worktree `qra-ws-gnn` (branch `gnn-exp/ws1-h6`), no commits — diff + ledger +
reports left for the researcher to promote. Work on `data/Meta` (Qwen cache). Guardrails held:
participant-grouped CV, participant-clustered bootstrap CIs, two-method agreement, no causal claims.

**Architecture decision.** The GNN layer is split into two concerns:
- **Discovery + construct-validation + mechanism** (`src/gnn_layer/` top level) — DEFAULT ON, runs at
  analyze-time on raw embeddings, no trained model. New: `discriminant.py` (H6), `transition.py`
  (mechanism rebuild), `confound.py` (confound localization), deepened `communities.py` (dyadic
  routines), `cue_features.py` (shared model-free cue helpers extracted from old `inference.py`).
- **GraphSAGE consensus-distillation classifier** (`src/gnn_layer/classifier/`) — moved to its own
  subpackage, DEFAULT OFF (`gnn_classifier_enabled=False`); opt in via `qra gnn train`. Wired through
  setup_wizard step 11d, `qra gnn train`, `analysis/runner.run_analysis(force_classifier=...)`. Dead
  mechanism-on-classifier path deleted (`influence.py` + its test; dead `counterfactual*` /
  `influence_bootstrap_n` config flags removed — config is field-filtered, deserialization-safe).

**WS1 — H6 discriminant validity.** Same Qwen embeddings, participant-grouped folds, clustered CIs:
probe (supervised, 6-class No-code) human kappa = 0.365 [0.228, 0.513] / LLM 0.283; content-similarity
(Correct-&-Smooth) 0.196 [0.117, 0.319] / 0.069; chance 0.000 / -0.083. Paired probe-minus-content
Delta-kappa: human 0.170 [0.002, 0.318], LLM 0.214 [0.150, 0.274] (CIs exclude 0). Geometry (honest):
the stage signal IS partly carried by leading content PCs (top-50 grouped-CV kappa 0.297 ~ full 0.307)
-- not an exotic subspace; what defeats similarity is LOCAL kNN non-homophily (1-NN same-stage 0.47 vs
base 0.25, lift ~1.9x decaying with k; Metacognition below base rate). Community x stage ARI ~ 0.006.
Ledger rows: H6-probe / H6-content / H6-chance-mode / H6-chance-strat.

**WS-T — dyadic FROM->CUE->TO transition model (mechanism rebuild).** 161 triples, 19 participants.
Earns-its-place: NEGATIVE -- adding the cue does not lower held-out TO error (Delta KL +0.37, Delta
E[stage] MAE +0.15) -> under-identified at n~32 (consistent with H2 pilot status); mechanism.py leads.
BUT triangulation POSITIVE: learned per-cell counterfactual Spearman rho ~ +0.34 vs observed
Delta-progression -- versus the retired classifier-counterfactual's -0.13. The proper transition model
(no kNN; FROM-stage conditioned) aligns where the mis-specified classifier inverted; under-powered
(0 FDR cells). CIs tight-by-construction (across-block, not training uncertainty); thin-support moves
flagged as extrapolations.

**WS3 — confound localization.** Signed divergence (observed - learned counterfactual) per
(from_stage x move), participant-clustered CIs: 20 cells, 9 sign-inverting. E.g. Vigilance x Reframing
(obs +2.09 vs cf +0.11; div +1.98 [0.98, 2.98]); Reappraisal x Reinforcement sign inversion (obs
-0.60 vs cf +0.59; div -1.19 [-1.72, -0.94]) = the responsiveness pattern. Caveat instrument, not a
claim.

**WS2 — Track D deepened.** Default community_sim_threshold lowered 0.85 -> 0.6 (probe: instruction-
tuned Qwen cosines run high; tau=0.85 -> noise ARI 0.003; tau=0.6 -> ARI 0.29, 24 multi-member
communities). 223 communities, 21 stable; 2 stable dyadic routines (both Delta-prog CIs include 0 =
honest under-powered leads). Added per-community stage/Delta-prog profile + atypical exemplars +
dyadic routines + dyadic_routines.txt.

**Cohorts 3-4 re-run triggers.** All read master_segments.csv + qra.db; re-run via `qra analyze`
(discovery) / `qra gnn train` (classifier). H6 N-robust (CIs tighten); WS-T earns-its-place + WS3
divergence + H2 mechanism become adjudicable at higher power; the classifier gate re-opens for any
learned scaler at larger N.

**Tests.** New: test_gnn_discriminant.py (9), test_gnn_transition.py (6), test_gnn_dyadic.py (5),
test_gnn_confound.py (6). Reorg: classifier import-sites rewritten to gnn_layer.classifier.*;
cue-helper patch targets -> cue_features; threshold-default test updated. Full tests/unit suite green.
