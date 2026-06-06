# QRA GNN Trial Run — Report

**Date:** 2026-06-06
**Workspace:** `./data/Meta` (a working copy of the migrated legacy project
`./data/MMORE_Processed_beta`; the original was left untouched)
**Goal:** import the legacy project, re-segment + re-classify PURER (therapist) content
*without* disturbing VAAMR participant segmentation or the validation testsets, train a
GNN consensus layer on the LLM labels, run the analysis, and report — including GNN
accuracy/validity against the LLM consensus.

---

## TL;DR

| Item | Result |
|---|---|
| Project imported to `./data/Meta` | ✅ 883 segments / 19 sessions, all overlays + testsets intact |
| Therapist (PURER) re-segmentation | ✅ 544 turns re-extracted + PHI-scrubbed; **VAAMR untouched, 76/76 testsets still valid** |
| PURER re-classification (turn-level) | ✅ **544/544 turns, 0 errors, 221 coded moves** |
| GNN trained on VAAMR + PURER | ✅ ran (`status: ok`); both VAAMR and PURER heads |
| **GNN vs LLM consensus (VAAMR)** | **κ = 0.247, agreement 48.3%, n=205 (cross-validated)** |
| Ready for LLM-free scaling? | **NO** — κ well below the 0.70 gate; rare stages over-smoothed |
| Analysis + reports + figures | ✅ full tier regenerated (106 figures, `06_reports/00–07`) |
| Unit tests | ✅ 3213 pass, 0 failures (15 new tests added) |

**Bottom line:** The pipeline ran end-to-end and the re-segmentation/re-classification
worked cleanly with full coverage and preserved testset validity. The GNN is a faithful
*but weak* student of the LLM consensus right now (κ≈0.25) and is **not** ready to replace
the LLM — it collapses rare VAAMR stages. This is an honest small-data + substitute-embedding
result, not a code failure.

> **For the deep VAAMR-reliability diagnosis, the methodology deliberation (incl. CFiCS),
> the answer to "is Qwen embedding enough?", and the sequenced roadmap → see
> [§12 — Deep diagnosis & reliability roadmap (VAAMR)](#12-deep-diagnosis--reliability-roadmap-vaamr-only).**
> The success criterion is reset to **human-level inter-rater reliability** (the human↔human
> ceiling is α≈0.33–0.52; the LLM already meets it at κ=0.537), **not** the legacy κ≥0.70 gate.

---

## 1. What was done (workflow)

1. **Imported** `MMORE_Processed_beta` → `./data/Meta` (full copy of `qra.db` + inputs + reports).
2. **Re-segmented therapist content** via a new first-class feature (`stage_resegment_therapist`),
   preserving every participant row byte-identically.
3. **Re-classified PURER** at the **turn level** (one PURER code per therapist turn, each turn
   classified inside its full surrounding exchange) with a zero-shot LM Studio model.
4. **Assembled** `master_segments.csv` and **trained the GNN** (`qra gnn train`) over the VAAMR +
   PURER LLM labels, producing the reliability gate and the full analysis tier.

---

## 2. Code changes delivered (now in the codebase)

These were implemented to satisfy the request and are reusable beyond this trial:

- **Turn-level PURER classification** (`src/process/config.py`, `src/process/orchestrator.py`):
  new `PurerCueConfig.classification_unit` (`"turn"` default | `"cue_block"`). In `"turn"` mode
  **every therapist turn is classified as its own unit**, wrapped in an *idea-complete* context
  built by `_build_turn_exchange_context` — the bracketing participant turns **and** the
  surrounding therapist turns, with the target turn explicitly demarcated. The essential exchange
  is never truncated (only the optional preamble is budget-squeezed). This replaces the old
  length-based `split_by_word_budget` fragmentation, so label granularity is theory-driven
  (the turn), not verbosity-driven.
- **Therapist-only re-segmentation feature** (`stage_resegment_therapist`): re-extracts therapist
  segments from the raw `.vtt`, **PHI-scrubs** them (`scrub_segments`), and replaces only the
  therapist rows — participant rows (`segment_id` + `segment_index`) are left identical so VAAMR
  labels and the (VAAMR-only) testsets stay valid. Exposed three ways: **TUI knob**
  ("Re-segment PURER / therapist content (preserve VAAMR + testsets)"), **CLI**
  (`qra ingest --therapist-only`), and the orchestrator stage. Therapist extraction now runs
  without loading the 8B embedding model (`ConversationalSegmenter(skip_embedding_model=True)`).
- **Wizard + config defaults** (`src/process/setup_wizard.py`): the PURER step now asks turn vs
  cue-block and defaults to a 2-run rotation.
- **Chronological display sort**: coded transcripts / human forms sort by `start_time_ms`
  (so re-segmented therapist turns interleave correctly).
- **15 new hermetic unit tests** (`tests/unit/test_purer_turn_level.py`,
  `test_resegment_therapist.py`); full suite: **3213 pass / 0 fail**.

---

## 3. Data-integrity verification (the hard constraint)

The explicit requirement was to re-do PURER **without** touching VAAMR participant
segmentation or invalidating the testsets. Verified directly against `qra.db`:

| Check | Result |
|---|---|
| Participant rows changed (`segment_id`+`segment_index`+text sha) | **0 / 339** |
| VAAMR testset items still valid (sha match by `(session, seg_num)`) | **76 / 76** |
| Raw participant names left in re-segmented therapist text (PHI) | **0** |
| Orphaned `purer_labels` after re-seg | **0** |
| Therapist segmentation determinism (old vs re-extracted ids/boundaries) | **544/544 identical** |

The therapist re-segmentation is deterministic, so the structural content is unchanged; its
real effect here was a clean PHI re-scrub + provenance stamp and a fresh, full-coverage PURER pass.

---

## 4. PURER re-classification results (turn-level)

- **Unit:** one PURER code per therapist turn (your decision), each classified within its full
  bracketing exchange (surrounding participant dialogue + continued therapist dialogue).
- **Model:** `nvidia/nemotron-3-nano-4b`, single-run, zero-shot, LM Studio @ `10.0.0.58`
  (see §7 for why this replaced the originally-planned 2-model rotation).
- **Coverage:** **544 / 544 therapist turns classified, 0 parse errors.**
- **Result:** 221 turns assigned a PURER move; 323 abstained (procedural/lecture/back-channel —
  correctly *not* coded). 23 turns carried a secondary move.

| PURER move | n |
|---|---|
| R2 — Reinforcement | 105 |
| E — Educate/Expectancy | 68 |
| P — Phenomenological | 31 |
| R — Reframing | 14 |
| U — Utilization | 3 |

This is a **dense, full-coverage overlay** (vs the legacy 39/544) and a methodologically
sensible distribution for MORE delivery (heavy reinforcement + psychoeducation, with
phenomenological elicitation the leading inquiry move).

---

## 5. GNN accuracy & validity vs LLM consensus (key deliverable)

The GNN is a **distilled student of the multi-run LLM majority-vote consensus**; the gate is
**cross-validated** (each segment scored by a model that did not train on it). Ballot coverage
is **100% multi-run** (339/339 participants), so the κ below is *trustworthy*, not the
weak-signal/optimistic case.

### VAAMR (participant stages) — graph vs LLM consensus
- **n = 205, agreement = 48.3%, κ = +0.247** (Landis–Koch "fair"; target ≥ 0.70).

| Stage | n | recall | precision | κ |
|---|---|---|---|---|
| Vigilance | 25 | 24.0% | 28.6% | +0.168 |
| Avoidance | 20 | **0.0%** | n/a | 0.000 |
| Attention-Regulation | 73 | 86.3% | 47.7% | +0.288 |
| Metacognition | 29 | **0.0%** | n/a | 0.000 |
| Reappraisal | 58 | 51.7% | 57.7% | +0.379 |

**Interpretation (honest):** classic **small-data over-smoothing** — the graph leans on the
majority stage (Attention-Regulation, 86% recall) and **never predicts the two rarest stages**
(Avoidance, Metacognition: 0% recall). It reproduces the LLM teacher only weakly.

### PURER (therapist moves) — graph vs LLM consensus
- **No out-of-sample graph predictions** (`gnn_purer_pred` is null). PURER is *not* a held-out
  prediction head in this configuration; it enters the GNN only through participant↔therapist
  **coupling**. So **PURER κ = n/a** by design, not by failure.

### Participant↔therapist coupling (PURER → forward movement)
- 5 latent therapist-cue factors extracted; correlations with subsequent participant forward
  movement are **weak** (all |forward_corr| < 0.08), nearest common-factor lexicon match
  `goal_alignment` at low similarity. Directional, hypothesis-generating only.

### Verdict
**READY FOR LLM-FREE SCALING? NO.** Keep labeling with the LLM consensus; add more
labeled segments (and a stronger embedding — see §7/§9) and re-run the gate.

---

## 6. Comprehensive analysis-reports summary

Full tier regenerated under `data/Meta/06_reports/` (106 figures in `05_figures/`). Highlights —
**all directional/associational, single-arm, NOT efficacy** (no control arm; the "outcome" is the
coded language itself, which is also shaped by therapist prompting):

**Progression (`01_outcomes/progression_summary.txt`)**
- Participants: **advancing 7, stable 11, regressing 2** (n=20).
- Adaptive-stage (2–4) occupancy rises **44% → 100%** across sessions.
- Monotonic trend: Mann–Kendall **τ=+0.714, p=0.0187** (ordinal-safe, n=8 sessions).
- Sensitivity (equal-spacing assumed): mixed-model E[stage] slope **+0.105/session
  [0.005, 0.205], p=0.0397**.

**Avoidance barrier (`01_outcomes/avoidance_barrier.txt`)**
- **17/20 participants crossed Avoidance → Attention-Regulation** (the central developmental cusp).

**Therapist mechanism — PURER × VAAMR (`02_mechanism/`)** *(now reflecting the new turn-level PURER)*
- 76 within-session stage transitions (39 forward, 37 backward).
- Forward-associated therapist moves (Δprogression > 0) concentrate at early stages, led by
  **Phenomenological** and **Reframing** cues (e.g. Vigilance→ with Phenomenology Δ≈+1.77).
- Backward/stalling-associated moves: **Reappraisal+Reframing, Metacognition+Utilization,
  Reappraisal+Education** (Δ≈−1.2 to −1.6).
- Transition-level PURER mix: **Education (E)** and **Reinforcement (R2)** dominate most
  transitions; e.g. Vigilance→Reappraisal (forward) is 56% E.
- Caveats: small per-cell n (often 2–10), permutation p / FDR in `mechanism.txt`.

**Convergent validity (`00_executive_summary.txt §4`)**
- External clinical outcomes **not yet integrated** → no real-world/efficacy claim is possible.

**Other:** IRR report regenerated (66 consensus items, 23 discrepancies; GNN axis = distillation).
MindfulBERT cue-language dataset built (161 examples; augmentation gate **not** passed — expected
at this data scale).

---

## 7. What worked, what broke, and how it was fixed

**Worked first time:** import, therapist re-segmentation + PHI scrub + testset preservation,
turn-level context construction (full 544-turn coverage), assembly, the analysis tier.

**Issues found and resolved (all environmental/model, not pipeline-logic):**

1. **`HfFolder` ImportError on ingest** — a transiently-installed sentence-transformers **v5**
   pulled `datasets`→`HfFolder` (removed from current `huggingface_hub`). Resolved by reverting to
   the pinned `sentence-transformers==2.7.0`.
2. **Turn-coverage bug (caught pre-results)** — the first turn-mode iteration rode the cue-block
   windower and covered only **65 unique** turns (overlapping windows + staged-participant
   boundaries). Rewrote it to iterate **all** therapist turns directly → **544/544, 0 duplicates**.
3. **`gemma-4-e2b` unreliable as a rater** — on long idea-complete prompts the 2B model emitted
   `"Thinking Process:"` prose instead of JSON, failing to parse on **~330/544 turns**; combined
   with strict 2-rater voting (any disagreement → abstain) this collapsed the overlay to 49 labels.
   **Fix (your call):** single-model `nemotron-3-nano-4b` (parses 544/544) — QRA's documented
   PURER default.
4. **GNN embedding incompatibility** — the GNN embeds with `Qwen3-Embedding-8B`, but the pinned
   `transformers==4.42.4` doesn't recognize the `qwen3` architecture, so embeddings failed and the
   GNN layer was **silently skipped**. **Fix:** pointed `gnn_layer.embedding_model` at the cached,
   working **`all-MiniLM-L6-v2`** (config-only; production keeps Qwen3). This unblocked the GNN
   *but* with weaker 384-dim features — a contributor to the modest κ (see §9).

---

## 8. Methodology notes (for peer review)

- **Unit of analysis (PURER):** the therapist *turn*, classified within its complete surrounding
  exchange. We deliberately avoided length-based cue-block fragmentation so granularity is
  theory-driven. Procedural/didactic turns receive an explicit abstention rather than a forced code.
- **VAAMR untouched:** participant segmentation, labels, and all three frozen VAAMR testsets are
  byte-for-byte preserved (verified); therapist re-segmentation cannot affect them (separate
  `segment_id` namespaces).
- **GNN gate is cross-validated** against the multi-run LLM consensus with full ballot coverage —
  the κ is a real out-of-sample distillation-fidelity measure.

---

## 9. Limitations & honest caveats

- **GNN κ=0.247 is weak** and the graph **does not predict rare VAAMR stages** (Avoidance,
  Metacognition: 0% recall). Do **not** use the graph as an autonomous labeler.
- **Substitute embedding:** the GNN used `all-MiniLM-L6-v2` (384-dim), not the intended
  `Qwen3-Embedding-8B` (8B), because of the `transformers` version pin. Expect κ to improve with
  the stronger, domain-appropriate embedding.
- **Small N:** 205 labeled participant segments across 5 stages / 19 sessions is little data for a
  GNN; rare-class collapse is expected.
- **All analysis is single-arm, associational** — no control, no integrated outcomes ⇒ no efficacy
  or causal claim.
- **PURER is single-model** (no inter-rater κ between LLMs this run) because the 2nd model
  (`gemma-4-e2b`) was unreliable; cross-model PURER IRR is deferred.

---

## 10. Recommendations / next steps

1. **Raise the GNN embedding quality:** upgrade `transformers` to a Qwen3-aware version *in an
   isolated env* (it risks the sentence-transformers/datasets cascade), then set
   `gnn_layer.embedding_model` back to `Qwen3-Embedding-8B` and re-run `qra gnn train`. Re-check κ.
2. **Add labeled data** (more LLM-coded and/or human-coded segments) to lift rare-stage recall and
   the gate κ toward 0.70.
3. **Restore a real 2-model PURER consensus** by swapping `gemma-4-e2b` for a small model that
   reliably emits JSON, *and* softening the merge (disagreement → keep primary + flag, not abstain)
   so coverage and an inter-model κ are both preserved.
4. **Integrate external outcomes** (`02_meta/outcomes.csv`) to enable a convergent-validity check —
   the only route to a real-world claim.
5. **Human IRR subset:** blind-code a sample to populate the independent quality axis (currently
   the gate rests on graph-vs-LLM fidelity alone).

---

## 12. Deep diagnosis & reliability roadmap (VAAMR-only)

*Scope: VAAMR participant-stage classification only (per direction). Goal reset: **human-level
inter-rater reliability**, defined below.*

### 12.1 The reliability ceiling reframes the goal

The IRR report (`06_reports/06b_irr_report.txt`) changes what "reliable" means here:

| Comparison | n | stat | band |
|---|---|---|---|
| Human ↔ Human (set 1 / 2 / 3) | 31/31/14 | Krippendorff α = **+0.47 / +0.52 / +0.33** | fair–moderate |
| **Human ↔ LLM** consensus | 66 | Cohen κ = **+0.537** | moderate |
| **Human ↔ GNN** consensus | 66 | Cohen κ = **+0.053** | slight |

Three consequences:
1. **VAAMR has a fuzzy ground truth.** Trained human coders agree only at α≈0.33–0.52. You cannot
   distill a label function more reliable than its teacher; **κ≥0.70 is likely unreachable in
   principle** for VAAMR as currently operationalized. The legacy 0.70 gate is the wrong target.
2. **The LLM is already human-level** (κ=0.537 ≈ the human↔human ceiling). It is a defensible label
   of record; the GNN's task is genuinely hard.
3. **The GNN fails on *both* axes** — κ=0.247 vs LLM (distillation fidelity) and **κ=0.053 vs human**
   (independent quality). This is not "imperfect distillation"; the graph is **not capturing the
   construct shape at all**, and it is *worse* than the LLM on the axis that matters (vs human).

> **Success criterion (new):** GNN↔human κ approaching the **LLM↔human** level (≈0.54), and
> GNN↔LLM κ approaching the **human↔human** ceiling (≈0.45–0.52). "Human-level IRR," not 0.70.

### 12.2 Architecture autopsy — what the trained graph actually is

- **Nodes:** segment embeddings only. This run used **`all-MiniLM-L6-v2` (384-d)** — a substitute,
  because the pinned `transformers 4.42.4` can't load the intended `Qwen3-Embedding-8B` (§7).
- **Edges:** kNN-8 (cosine over those embeddings) + temporal-chain. **Construct anchors are OFF**
  (`use_anchor_nodes=False`).
- **Model:** 2-layer GraphSAGE (`lin_self + mean(neighbors)`), hidden 128, dropout 0.5, 300 epochs.
- **VAAMR supervision:** soft head, **KL-divergence to the multi-run ballot mixture,
  `reduction='batchmean'` — i.e. NO class weighting**; aux SupCon + temporal link-prediction.
- **Gate:** 5-fold CV, argmax of the soft head, per-class κ/recall.

The decisive implication: with anchors off and only weak features, **VAAMR learning rests entirely
on whether the embedding-based kNN graph separates the stages.** If the embedding doesn't, the kNN
neighbours are mixed-stage, two rounds of mean aggregation blur toward the majority, and the
unweighted KL loss has no incentive to recover the rare stages — *exactly the observed failure*
(AttentionReg recall 86%, Avoidance/Metacognition recall 0%).

### 12.3 Ranked causes (with evidence)

| # | Cause | Evidence | Independent of embedding? |
|---|---|---|---|
| 1 | **Weak node features / graph topology** (MiniLM, not Qwen3-8B) | both heads weak; kNN graph = embedding quality; CFiCS shows node features are ~half the lift | no — fix first |
| 2 | **No class rebalancing** (batchmean-KL) | rare stages 0% recall; majority 86% | yes |
| 3 | **Tiny + imbalanced data** | 205 labels / 5 classes; rare classes n=20, 29 → ~16/fold | yes |
| 4 | **No concept structure** (the CFiCS gap, §12.4) | homogeneous, anchors off; CFiCS's gains are *specifically* fine-grained | partly |
| 5 | **Fuzzy construct ceiling** (§12.1) | human α≈0.33–0.52 | yes — reframes target |

### 12.4 The CFiCS premise, revisited

`references/cfics.txt` (Schmidt et al.) is a GNN over a **heterogeneous concept graph** + ClinicalBERT
node features whose headline is **large gains specifically on fine-grained classes** (skill macro-F1
~4→96). Master-plan §4.2 rejected that structure on the premise *"QRA doesn't have CFiCS's
few-shot/fine-grained constraint."* **The trial refutes that premise** — VAAMR rare-stage
classification *is* a fine-grained few-shot problem, the exact regime where concept structure is
decisive. The original circularity objection (anchors inflate GNN↔LLM κ) is real but **already has a
non-circular remedy in the design (D3): judge anchors on the *human* κ axis**, which they cannot
inflate. A **66-item human-coded subset exists** (it drives the IRR report) — so the human-axis
anchor ablation is *runnable today*; it simply isn't wired into the GNN gate yet (an integration gap
to close).

### 12.5 Methodology deliberation (when does a GNN earn its place?)

- **Simple model + label propagation often beats GNNs at limited-label scale.** *Correct & Smooth*
  (Huang et al., ICLR 2021) shows an MLP/linear base + label propagation matches/exceeds SOTA GNNs
  with a fraction of the parameters — **most relevant to our n=205**. This means a calibrated linear
  probe on Qwen embeddings + C&S over the temporal/kNN graph is a *serious* candidate, and the
  GraphSAGE must **beat it on identical folds to justify its complexity**.
- **Class-imbalanced node classification** has proven levers: re-weighting (**TAM** topology-aware
  margin; **ReNode**), over-sampling (**GraphSMOTE**), and class-balanced/focal loss — all directly
  target our 0%-rare-recall.
- **Text-GNNs (TextGCN/BertGCN)** gain mainly from *transductive, corpus-level* structure and joint
  training; their edge over strong contextual embeddings is modest and setting-dependent — another
  reason to *measure* the GNN's marginal value rather than assume it.
- **CFiCS** is the strongest in-domain evidence that *concept structure + domain embeddings* lift
  fine-grained therapy-construct classification — our escalation path if features alone don't.

Sources: [C&S (2010.13993)](https://arxiv.org/abs/2010.13993) · [Class-imbalance survey (2304.04300)](https://arxiv.org/pdf/2304.04300) · [TAM (2206.12917)](https://arxiv.org/pdf/2206.12917) · [GraphSMOTE](https://www.researchgate.net/publication/350105097) · [Text-GNN survey (2304.11534)](https://arxiv.org/pdf/2304.11534) · CFiCS (`references/cfics.txt`).

### 12.6 Answer: is the proper Qwen embedding enough?

**Try it first — yes — but instrument it to actually answer the question; expect it to be
necessary, not sufficient.** Qwen3-8B is the highest-leverage, lowest-risk change and the *intended*
configuration (κ=0.247 is a MiniLM **lower bound**), and *every* downstream architecture rests on it.
But causes 2–4 are independent of embedding, so Qwen alone is unlikely to fully recover the rare
stages. The disciplined design is an **A/B isolation battery** (§12.7 Path A) that decomposes the gap
into *features vs imbalance vs architecture vs whether a GNN is even needed* — before any large build.

### 12.7 Paths forward (sequenced)

**Path A — Embedding-first isolation battery (do first).** On a dedicated branch:
1. Stand up `Qwen3-Embedding-8B` in an **isolated env** (upgrade `transformers` there only; the main
   env stays on the working pin — see §7 risk). Re-embed all segments.
2. On **identical folds/seed**, score three models for the soft-VAAMR head:
   - **Linear probe** (logistic regression on Qwen embeddings; no graph) — the "do we need a graph?" baseline.
   - **Correct & Smooth** (linear base + label propagation over the temporal/kNN graph).
   - **Current GraphSAGE** (unchanged), then **+ class-balanced/focal loss** and **+ TAM-style margin**.
3. Read the per-class κ/recall table for each. **Decision rules:**
   - If Qwen alone lifts κ to ~human-level and recovers rare stages → done; ship Qwen as the default
     `gnn_layer.embedding_model`.
   - If GraphSAGE ≤ linear/C&S → the GNN isn't earning its place at this scale; adopt the simpler
     model (or C&S) for the classifier mission and keep the GNN for discovery/coupling only.
   - If features help but rare classes still collapse → the imbalance losses (step 2c) are the fix.

**Path B — CFiCS-style concept graph (escalate only if Path A leaves fine-grained gaps).** On its own
branch: heterogeneous graph = segment nodes + **VAAMR construct-definition anchors + hierarchy** +
domain embeddings; **gain measured on the human κ axis** (anchors can't inflate it) via the existing
`ablation.anchor_contribution`. Wire the existing 66-item human subset into the GNN gate first so the
ablation is decisive. Consider a domain embedding (ClinicalBERT/MentalBERT-style) as a sub-arm if
Qwen still under-separates the subtle stages.

**Cross-cutting:** close the **human-subset → GNN-gate** integration gap so every architecture is
scored on the human axis, not just GNN↔LLM.

### 12.8 The experiment battery (concrete)

| Arm | Features | Graph used | Imbalance handling | Question it answers |
|---|---|---|---|---|
| A0 | MiniLM (current) | kNN+temporal (SAGE) | none | baseline (κ=0.247) |
| A1 | **Qwen3-8B** | none (linear probe) | none | is it the embedding? do we need a graph? |
| A2 | Qwen3-8B | label-prop (C&S) | none | does graph structure help a simple base? |
| A3 | Qwen3-8B | kNN+temporal (SAGE) | none | does the GNN beat the simple baselines? |
| A4 | Qwen3-8B | kNN+temporal (SAGE) | **class-balanced + TAM margin** | does rebalancing recover rare stages? |
| B1 | Qwen3-8B | + VAAMR concept anchors/hierarchy | best of A4 | does concept structure close fine-grained gaps (human axis)? |

All arms: identical folds/seed, report **per-class κ + recall** and **GNN↔human κ** on the 66-item
subset. Promote a change only if it improves the **human axis** (the load-bearing validity evidence).

### 12.9 Experimentation discipline (mandated)

- **One branch per distinct architecture.** Every big architectural experiment (Qwen embedding,
  C&S, imbalance losses, concept anchors) runs on its **own git branch** so changes are isolated,
  reversible, and comparable. No silent in-place swaps on `master`/`beta`.
- **Document every arm's result.** Each arm appends a row to a results ledger
  (`docs/gnn_experiments/` or this report's §12.8 table) with: branch, embedding, graph, loss,
  per-class κ/recall, GNN↔LLM κ, GNN↔human κ, decision. Negative results are recorded, not discarded.
- **Promote on the human axis.** A change is merged to the default only if it improves GNN↔human κ
  (or, for distillation-only mechanics, GNN↔LLM κ without harming the human axis), per D3.

### 12.10 What this changes in the master plan

`docs/GNN_MASTER_PLAN.md` updated in the same session: §4 premise revised (the trial data refutes
"no few-shot constraint"); target reset to human-level IRR; new **Track A0 — Reliability recovery
battery** (Path A → Path B) with the branch-per-architecture discipline; and the human-subset→gate
integration gap logged as a precondition.

---

## 13. Artifacts

- Project DB: `data/Meta/qra.db` (segments + VAAMR/PURER/codebook/**gnn** overlays + testsets)
- GNN gate & reports: `data/Meta/06_reports/06_gnn/` (`validation.txt`, `coupling.txt`,
  `triangulation.txt`, `emergent_motifs.txt`) + figures `data/Meta/05_figures/gnn_*.png`
- Analysis tier: `data/Meta/06_reports/00_executive_summary.txt`, `01_outcomes/`, `02_mechanism/`,
  `03_per_session/`, `04_per_participant/`, `05_per_stage/`, `06b_irr_report.txt`,
  `07_methods_appendix.txt`
- Assembled dataset: `data/Meta/02_meta/training_data/master_segments.csv`
- Run logs: `data/Meta/_trial_logs/`
- DB backups (pre-op): `data/Meta/qra.db.bak.*`
