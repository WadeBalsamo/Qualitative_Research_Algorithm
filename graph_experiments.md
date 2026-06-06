# Graph Experiments — VAAMR Reliability & Mechanism

**Date:** 2026-06-06 · **Corpus:** `./data/Meta/` (Move-MORE, n≈32 participants; 339 participant
segments, 205 VAAMR-labeled, 134 "No code"; 76 human-coded IRR items) · **Companion records:**
`design_decisions.md` (§0 synthesis + full decision log), `docs/gnn_experiments/ledger.csv` (per-arm
data), `docs/GNN_MASTER_PLAN.md` Track A0.

This document is the comprehensive account of *what we tried, what happened, **why** it happened,
what it means methodologically, and what to do next*. It is written to be read by a future
researcher (or agent) deciding where to spend the next unit of effort.

---

## 1. The question we set out to answer

Two missions, deliberately scored apart:

- **PRIMARY — mechanism.** Can a GNN explain *how therapist language (PURER) moves participant
  VAAMR expression*, via (a) model-counterfactual sensitivity and (b) participant↔therapist
  coupling, such that those readouts **triangulate** with the independent observed-Δprogression
  analysis (`analysis/mechanism.py`)? Success was pre-registered as Spearman ρ>0 (participant-
  clustered bootstrap 95% CI excluding 0) per cue→transition, plus ≥70% sign agreement on
  FDR-significant effects.
- **SECONDARY — classifier IRR (trust floor).** Push the graph's VAAMR labels toward human-level
  inter-rater reliability: GNN↔human κ → the LLM↔human level (≈0.54); GNN↔LLM κ → the human↔human
  ceiling (≈0.45–0.52). κ≥0.70 was explicitly *not* the target (human↔human α≈0.33–0.52).

The starting point was a prior trial reporting GNN↔LLM κ=0.247 / GNN↔human κ=0.053 on a substitute
MiniLM embedding, with Avoidance and Metacognition at 0% recall.

---

## 2. What we built (the apparatus)

All behind flags (default path byte-identical) and unit-tested (761 GNN/IRR tests pass):

| Piece | Where | What it is |
|---|---|---|
| **Qwen `/v1/embeddings` backend** | `gnn_layer/embeddings.py`, `embeddings_remote.py` | real Qwen3-Embedding-8B (4096-d) via LM Studio, sidestepping the transformers pin (`embedding_backend='openai'`). |
| **Human-subset → gate join (A0-pre)** | `src/analysis/irr_join.py` + `runner.py` | populates `in_human_coded_subset`/`human_label` from `qra.db`, lighting the gate's human axis. |
| **Imbalance + No-code losses** | `gnn_layer/{model,train,soft_labels,config}.py` | class-weighted/focal/TAM loss + an optional 6th "No code" class. |
| **Triangulation metric** | `gnn_layer/influence.py` | per-(from_stage×move) Spearman ρ + participant-clustered bootstrap CI + un-gated counterfactual entry point. |
| **Reliability harness** | `experiments/gnn_reliability/` | participant-grouped-CV engine + battery/mechanism/baseline/anchor runners + scorer (reuses the project's IRR libs, so numbers are comparable to `06b_irr_report.txt`). |

**Methodological backbone:** participant-grouped (`StratifiedGroupKFold`) folds built once and shared
by every arm; both axes (LLM `final_label` n=205; human consensus n≈66) scored with
participant-clustered bootstrap 95% CIs; arms pre-registered before any Qwen result; hyperparameters
tuned on the LLM axis only, the human axis read once per arm.

---

## 3. Results

### 3.1 Classifier battery (κ = point [95% CI]; reference: human↔human α 0.33–0.52, LLM↔human 0.537)

| Arm | model | nCls | **human κ** (n) | LLM κ (205) | rare recall Vig/Avo/Meta |
|---|---|---|---|---|---|
| A0 | MiniLM GNN | 5 | −0.02 [−.08,.06] (66) | 0.05 [−.03,.12] | 0.20/0.00/0.00 |
| A1 | Qwen probe | 5 | 0.21 (37) | 0.23 | 0/0/0 |
| A1w | Qwen probe, class-wt | 5 | 0.30 (37) | **0.31** | 0.56/0.35/0.31 |
| **A1n ⭐** | Qwen probe, class-wt | 6 | **0.37 [.23,.51] (66)** | 0.28 | 0.36/0.35/0.31 |
| A2 | Qwen C&S | 5 | 0.16 (37) | 0.16 | 0.20/0.05/0.10 |
| A2n | Qwen C&S | 6 | 0.20 (66) | 0.07 | 0/0/0 |
| A3 | Qwen GNN | 5 | 0.14 (66) | 0.21 | 0.28/0.20/0.00 |
| A4 | Qwen GNN, class-wt+focal | 5 | 0.10 (66) | 0.16 | 0.32/0.50/0.10 |
| A4n | Qwen GNN, class-wt+focal | 6 | 0.36 [.25,.45] (66) | 0.18 | 0.36/0.35/0.17 |
| B1 | Qwen GNN **+ concept anchors** | 6 | 0.29 [.17,.44] (66) | 0.18 | 0.40/0.35/0.17 |

> **Read note:** 5-class probe arms *defer* "No code" → scored on a 37-item subset (easier); the
> fair full-task comparison is the **6-class arms (all n=66)**. Best full-task human κ = **A1n 0.37**.

### 3.2 Mechanism (PRIMARY) — both lenses, on Qwen 5-class GNNs (precipitates edges ON)

| Lens | metric | result |
|---|---|---|
| (a) Counterfactual | Spearman ρ vs observed Δprogression (20 cells) | **−0.13** [−0.48, 0.01] (plain) / −0.25 (balanced) — **CI includes 0** |
| (a) sign agreement | over FDR-significant cells | **n/a — 0 FDR-significant cells** in the observed table |
| (b) Coupling | max factor↔forward-movement \|corr\| (5 factors, 161 blocks) | **0.068** — weak |

Per-move counterfactual rank is face-valid (Utilization > Education > Reframing > Phenomenology >
Reinforcement) but does not reproduce the observed per-cell pattern. **Verdict: the GNN mechanism
readouts do NOT triangulate → `analysis/mechanism.py` (observed) leads; the GNN is exploratory only.**

---

## 4. Why the results were this way (the analysis that matters)

### 4.1 Features were the binding constraint — but only up to a point
MiniLM-384 is a generic sentence encoder; it does not resolve the *subtle phenomenological*
distinctions VAAMR encodes (e.g., "I notice the pain" [Vigilance] vs "I notice my mind reacting to
the pain" [Metacognition] vs "the pain feels different now" [Reappraisal]). Swapping to
Qwen3-Embedding-8B (4096-d, instruction-tuned) made the stages **linearly separable enough** for a
probe to jump from κ≈0.05 to 0.31 (LLM) / 0.37 (human). *The original failure was a feature-
resolution problem, not a GNN-architecture problem.* This is the single highest-leverage fix and the
reason every downstream model improved.

### 4.2 Why the probe beats the GNN — **VAAMR is not homophilous in embedding space**
This is the load-bearing insight. A graph neural net's core inductive bias is **homophily**:
neighbours (here, cosine-kNN of segment embeddings) are assumed to share labels, so mean-aggregating
a node with its neighbours *denoises* the label. **VAAMR violates this.** VAAMR is a *developmental/
process* state, not a semantic-content state — and embedding similarity tracks *content* (topic,
words, body part, affect), not developmental stage. Concretely:
- Two utterances about "my lower back while sitting" can be at Vigilance vs Reappraisal; they are
  near-neighbours in Qwen space, but opposite ends of the VAAMR arc.
- So the kNN graph systematically wires **different-stage** nodes together, and two rounds of mean
  aggregation **blur exactly the distinction we are trying to classify** (over-smoothing toward the
  local content-cluster's majority stage, which is usually Attention-Regulation, the modal stage).
- The probe pays no such tax: it learns a clean linear boundary on the (already separable) Qwen
  features without averaging in mislabeled neighbours.

The evidence is consistent with this: the GNN's best LLM κ (A3 0.21) sits *below* the probe (A1w
0.31); **Correct-&-Smooth — which is *pure* graph smoothing on the probe — is the worst arm** (A2
0.16, A2n 0.07), i.e. *adding* graph propagation actively destroys signal. That is the fingerprint
of a low-homophily graph. At n≈205 the GNN also simply has more parameters to overfit, compounding it.

### 4.3 Why concept anchors (Path B / CFiCS) hurt (0.36 → 0.29)
Anchors are construct-*definition* embeddings ("Vigilance: attentional capture by pain…") connected
to segments by similarity. They fail for the *same* reason as 4.2, doubled: (i) the definitions live
in a different, more abstract region of the space than lived participant speech, so similarity edges
are noisy; and (ii) those edges are, again, **not label-homophilous** — they pull a segment's
representation toward a generic definition centroid rather than toward same-stage exemplars. CFiCS's
gains came from a *hand-authored, typed, definitional* graph (CF→IC→skill→example) in a 181-example
*synthetic few-shot* regime; QRA's empirical similarity-to-anchor edges are a categorically weaker
object. The trial's original instinct (homogeneous graph as default) was right; B1 is the measured
confirmation, on the human axis where anchors cannot inflate κ.

### 4.4 Why the "No code" class carries the human axis (label-space, not features)
36% of human-coded items are "No code" — logistics, greetings, off-topic chatter that is *not* a
VAAMR expression. A 5-class model is *forced* to assign a stage to these, so it is wrong on ~a third
of items by construction. The 6-class model can abstain ("No code" = class 5) and gets them right.
This is why the No-code class lifts human κ from −0.02→0.21 even on weak MiniLM, and why the LLM
(which can already abstain) reaches κ=0.54 vs human. **This was a construct-operationalization gap,
not a modeling gap:** VAAMR as specified lacked an explicit null category, and the human axis
punished its absence. The fix is a label-space fix and it is the second-biggest lever after features.

### 4.5 Why the published κ=0.247 was an illusion (CV leakage)
The prior gate used *random* k-fold. A participant's segments are (i) temporally chained, (ii)
mutually kNN-similar (same speaker, same recurring topics), and (iii) auto-correlated in stage (a
participant dwells near a developmental position across a session). Under random folds, a held-out
segment's neighbours include **its own and same-participant segments with visible labels** — the
model effectively *sees the answer*. Holding out **whole participants** (`StratifiedGroupKFold`)
removes this, and κ collapses **0.247 → 0.05**. The 0.247 was measuring leakage, not generalization.
This is the classic transductive-GNN evaluation trap and it is now corrected throughout the harness.

### 4.6 Why the mechanism doesn't triangulate — confounding *and* under-identification
Two independent reasons, both fundamental:
1. **The elicitation/responsiveness confound (identification).** The observed Δprogression measures
   "*when* move m was used, the participant moved by Δ." But therapists *choose* moves in response to
   participant state. If Phenomenological elicitation is deployed when a participant is *stuck*, its
   observed Δ is low/negative — not because it fails, but because of *when* it is used. The
   counterfactual ("swap any cue to m, read the model's predicted shift") tries to isolate m's effect
   *independent of when it is used*. So the two measures answer **different questions**, and the
   *negative* ρ (the observed pattern is inverted relative to the counterfactual) is the **signature
   of a strong responsiveness confound**, not a bug. This is the deepest reason "no causal claim" is
   admissible (methodology §9.4).
2. **Under-identification / low power (estimation).** The counterfactual shift is tiny (±0.035 on a
   0–4 scale) because a participant's predicted progression is dominated by *their own* features, not
   the swapped therapist cue; the mediating GNN is itself a weak classifier (gate κ≈0.13–0.20); and
   ρ is computed over **20 cells from n≈32 participants**, giving a CI of [−0.48, 0.01]. Even the
   *observed* analysis has **0 FDR-significant cells**. There is simply no statistical power to detect
   convergence at this scale, regardless of the confound.

The honest reading: the mechanism is *under-identified at n≈32*. The GNN adds a sensitivity lens that
**disagrees** with the confounded observational signal — which is itself informative (it flags the
confound) but is not corroboration.

---

## 5. What this means methodologically for QRA

1. **The GNN's two raisons d'être are both undercut *at this scale*, for principled reasons.** As a
   *classifier*, the graph's homophily assumption is wrong for a process label (4.2) → a probe is
   better. As a *mechanism* instrument, the readout is under-identified and confound-divergent (4.6).
   Neither is a coding bug; both are structural facts about *process labels on n≈32 observational
   data*.
2. **The real deliverable is the evaluation methodology, and it is now defensible.** Participant-
   grouped CV (no leakage), participant-clustered bootstrap CIs (tiny n → intervals, not points), a
   pre-registered arm battery, the human axis as load-bearing, and **negative results recorded as
   evidence**. Any future VAAMR-reliability claim can now be made (or refuted) honestly. The
   leakage catch alone changes how every prior κ in this project should be read.
3. **The LLM consensus remains the trustworthy label of record** (κ=0.54 vs human ≈ the human
   ceiling). The defensible production stance is: **LLM labels of record + a calibrated Qwen probe as
   an abstention-gated assist + `mechanism.py` (observed, caveated) for mechanism.** The GNN is an
   *exploratory* lens, explicitly so.
4. **Two of the three winning levers were not "AI" at all** — they were *measurement discipline*
   (grouped CV) and *construct operationalization* (the No-code null class). The one modeling lever
   that helped (class-weighting) is standard imbalance handling, not graph machinery.
5. **The binding constraint is data scale (n≈32), full stop.** No architecture move closes the gap;
   the human↔human ceiling (α 0.33–0.52) and the 20-cell mechanism table are the limits.

---

## 6. What to try next (ranked by expected value)

1. **Scale the labels with the LLM, not the GNN.** The GNN was meant to scale labeling so MindfulBERT
   could train — but it cannot be trusted to label autonomously (gate fails; non-homophilous graph).
   The LLM consensus *is* human-level and cheap. **Re-plan the bootstrap chain to: LLM labels the full
   corpus → observed Δprogression → MindfulBERT**, dropping the GNN-as-labeler step. Re-run the entire
   battery + mechanism *after* more participants/sessions land — most conclusions here are explicitly
   n-bound and may change with N.
2. **Ship the probe now.** Promote the **A1n** pipeline (Qwen embedding → class-weighted logistic probe
   → No-code-aware) as the abstention-gated VAAMR assist; keep the LLM as label of record. This is
   usable today and is the best classifier we have.
3. **Make the production gate honest.** Migrate the production reliability gate
   (`gnn_layer/validation.py` ← `train.crossval_predictions`) from random k-fold to **participant-
   grouped CV** (the harness `run_gnn_arm` is the reference implementation). Until then, treat any
   `validation.txt` κ as leakage-inflated.
4. **If the graph is kept for classification, fix the homophily problem — don't add more similarity
   edges.** The defensible graph structure for VAAMR is **process/temporal** (the PRECIPITATES
   therapist→participant edges, and the participant temporal chain), *not* content-similarity kNN.
   Worth testing: a graph with kNN **off** (temporal + precipitates only), or kNN built in a
   *stage-supervised* metric space (a learned projection where same-stage segments are close) — but
   both are speculative below N≈100.
5. **For mechanism, attack the confound, not the model.** The per-(from_stage, move) conditioning in
   `mechanism.py` already partially controls for participant state; report *those* cells as
   hypotheses with the responsiveness caveat. A cleaner claim needs either much more data (to power
   the conditioning) or a design lever (e.g., session-phase / dose variation as a quasi-instrument).
   The GNN counterfactual is worth keeping *only* as a confound-detector (its disagreement with the
   observed pattern is the signal).
6. **Cheap embedding ablations** (LLM-axis only, low effort): test the Qwen embedding *without* the
   retrieval query-prefix (it is a retrieval framing, possibly suboptimal for classification), and a
   mental-health-domain encoder (MentalBERT) as a sanity comparison. Unlikely to beat Qwen-8B but
   cheap to rule out.

---

## 7. Reproducing

```bash
# Qwen embeddings are cached at data/Meta/02_meta/gnn/segment_embeddings_qwen3_8b.npz
python experiments/gnn_reliability/run_battery.py            # A1..A4n  -> ledger.csv
python experiments/gnn_reliability/run_mechanism.py          # counterfactual + §1A triangulation
python -m pytest tests/unit/ -k "gnn or irr_join" -q         # 761 pass
```
Requires the LM Studio Qwen embedding endpoint (`http://10.0.0.58:1234/v1`) only if the cache is
cleared. Folds are deterministic (`StratifiedGroupKFold`, seed 42).

## 8. Honest limitations

- n≈32 participants / 205 labels / 66 human items / 20 mechanism cells — every number has a wide CI;
  CIs, not point estimates, carry the conclusions.
- The human axis (66 items, α 0.33–0.52 among raters) is itself fuzzy; "human-level" means the human
  *band*, not a sharp target.
- The Qwen embeddings truncate texts to 8000 chars (a handful of long didactic turns); the endpoint
  returns L2-normalized vectors (cosine-equivalent, but a magnitude asymmetry vs the local path).
- All mechanism statements are observational, n≈32, elicitation-confounded → **hypothesis-generating,
  never causal**.
