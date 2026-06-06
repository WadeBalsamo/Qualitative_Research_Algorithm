# GNN VAAMR Reliability — Design Decisions & Experiment Ledger

> **Living document.** Append-only decision log + a pre-registered experiment battery.
> Negative results are kept as evidence, not deleted. Workspace: `./data/Meta/`.
> Companion: `docs/GNN_MASTER_PLAN.md` (Track A0), `docs/gnn_experiments/ledger.csv`.

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

## 7. Results ledger (append-only)

> Filled as arms complete. Format mirrors `docs/gnn_experiments/ledger.csv`.
> κ shown as `point [lo, hi]` (participant-clustered bootstrap 95% CI).

| Arm | branch | GNN↔human κ (n≈66) | GNN↔LLM κ (n=205) | rare recall (Avoid/Metacog) | decision |
|---|---|---|---|---|---|
| _pending_ | | | | | |

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
