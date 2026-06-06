# QRA GNN — Master Implementation Plan & Decision Record

> **Status:** living document. Phases 0–2, Track 0 (baseline stabilization +
> gate-gated promotion), **all of Track A (A1–A5: typed cue edges, abstention, calibration +
> OOD, measured label propagation, scale-mode sim gate)**, and **all of Tracks B, C, and D
> (B: model-counterfactual influence + triangulation; C: the MindfulBERT training-set builder
> + augmentation-validation harness; D: subtext communities as routines with stability
> selection)** complete; a cross-cutting **GPU preference** mandate (D11/§6a) is audited and
> enforced. **All planned tracks are now implemented and unit-tested.** This is the single
> authoritative account of *why* the GNN exists in the form it does and *what* has been built.
> It supersedes the deleted `docs/GNN_LAYER_DESIGN.md` / `docs/GNN_IMPLEMENTATION.md`.
> Companion artifacts: `methodology.md` §8.5 (as-built prose spec),
> `gnn-influence-to-execution.md` (the original built-vs-designed reconciliation),
> `ROADMAP.md` Phase 3/6.

---

## Table of contents

1. [Mission & the end goal](#1-mission--the-end-goal)
2. [The epistemic chain (bootstrapping logic)](#2-the-epistemic-chain-bootstrapping-logic)
3. [Where the GNN sits in QRA](#3-where-the-gnn-sits-in-qra)
4. [The central deliberation: is the GNN the most defensible methodology?](#4-the-central-deliberation-is-the-gnn-the-most-defensible-methodology)
5. [Design decisions (the locked record + the why)](#5-design-decisions-the-locked-record--the-why)
6. [The defensibility spine](#6-the-defensibility-spine)
7. [What is already built](#7-what-is-already-built)
8. [What we did this session (Phases 0–2 + Track 0)](#8-what-we-did-this-session-phases-02--track-0)
9. [The tracks — exhaustive specification](#9-the-tracks--exhaustive-specification)
   - [Track 0 — Stabilize baseline + affirm the gate](#track-0--stabilize-baseline--affirm-the-gate)
   - [Track A — Scalable, trustworthy label engine](#track-a--scalable-trustworthy-label-engine)
   - [Track B — Therapist→participant progression analysis](#track-b--therapistparticipant-progression-analysis)
   - [Track C — MindfulBERT training-set builder](#track-c--mindfulbert-training-set-builder)
   - [Track D — Subtext communities as routines](#track-d--subtext-communities-as-routines)
10. [Sequencing, checkpoints, dependencies](#10-sequencing-checkpoints-dependencies)
11. [Module & file index](#11-module--file-index)
12. [Configuration reference](#12-configuration-reference)
13. [Testing strategy & conventions](#13-testing-strategy--conventions)
14. [Risk & honesty register](#14-risk--honesty-register)
15. [Glossary](#15-glossary)
16. [Status dashboard](#16-status-dashboard)

---

## 1. Mission & the end goal

QRA (Qualitative Research Algorithm) is a computational-phenomenology pipeline over therapy
transcripts from the **Move-MORE Feasibility Trial** (Mindfulness-Oriented Recovery
Enhancement for lumbosacral radicular pain; **n ≈ 32 participants, ~8 per cohort, four
sequential cohorts**). It applies two frameworks bilaterally:

- **VAAMR** — Vigilance · Avoidance · Attention-Regulation · Metacognition · Reappraisal —
  classifies **participant** segments along a five-stage developmental arc of
  mindfulness-skill development.
- **PURER** — Phenomenological · Utilization · Reframing · Educate/Expectancy ·
  Reinforcement — classifies **therapist** segments (guided-inquiry moves) at the
  cue-block level.

**The end goal of the GNN work is not the GNN.** It is to build a **training dataset to
fine-tune MindfulBERT** — a domain-adapted model that does *not merely classify VAAMR* but
**predicts which language patterns progress a participant across VAAMR stages.** The GNN's
job is twofold:

1. **Scale the labeling** so there is enough labeled corpus to train MindfulBERT (cheap,
   LLM-free VAAMR/PURER labels on the full corpus).
2. **Help identify** the progression-inducing language patterns that become the training
   signal (candidate generation + curation), and provide a model-counterfactual lens on
   *how* therapist language moves participant expression.

Everything below serves those two jobs, under hard methodological constraints (§6).

---

## 2. The epistemic chain (bootstrapping logic)

This is the backbone of every claim the GNN supports:

```
   human raters  ⟷ (validated at inter-rater reliability) ⟷  LLM multi-run consensus
                                                                      │
                                                       distillation (graph student)
                                                                      ▼
                                                                    GNN
                                                                      │
                       GATED:  may not label of record until graph↔LLM agreement is
                       consistent OUT-OF-SAMPLE (Cohen's κ ≥ target + rare-stage floor)
                                                                      │
                                                                      ▼
                          cheap, gated VAAMR/PURER labels on the FULL corpus
                                                                      │
                                                                      ▼
                  observed Δprogression of cue blocks  =  GROUND TRUTH for
                          "what language progresses participants"
                                                                      ▼
            MindfulBERT trained on observed (language → progression) pairs,
            augmented (separably, gated) by GNN model-counterfactual signals
```

**Reading the chain:**
- The LLM multi-run consensus is itself anchored to a human-validated 20% blind-coded
  subset — so the LLM labels are trustworthy at inter-rater reliability (IRR).
- The GNN is a **graph-distilled student** of that LLM consensus. It is allowed to label
  *new, unlabeled* segments on its own **only after** it reproduces the consensus
  out-of-sample to IRR (the reliability gate). This is the over-smoothing safeguard and the
  trigger for LLM-free scaling.
- Once gated, the GNN labels the full corpus cheaply, which yields enough labeled cue blocks
  to compute **observed Δprogression** — the participant's VAAMR state before vs after a
  therapist's language — which is the **ground truth** for "what progresses participants."
- MindfulBERT learns from those observed pairs.

---

## 3. Where the GNN sits in QRA

The GNN is an **analyze-time layer** (`gnn_layer/`) that runs after `master_segments`
assembly. It **augments, never replaces** the LLM/embedding classifiers, and never mutates
frozen segments or `master_segments`. It is **ON by default** (`config.gnn_layer.enabled =
True`), fully guarded (degrades to a logged warning if training can't run), and reuses the
same Qwen3 embedding substrate QRA already uses for segmentation and VCE coding (no second
model download).

It plays two roles:
1. **Discovery & triangulation** — surfaces continuous VAAMR positioning (superposition),
   cue motifs, participant↔therapist coupling, and an independent geometric measurement
   substrate to cross-check the LLM.
2. **Consensus-distillation classifier** — learns to reproduce the LLM majority-vote
   consensus from graph structure, gated for LLM-free scaling, optionally promotable to the
   authoritative label of record.

---

## 4. The central deliberation: is the GNN the most defensible methodology?

This question was examined against **CFiCS** (Schmidt et al., CLPsych 2025;
`references/cfics.txt`), the cited methodological influence. The conclusion is nuanced and
load-bearing.

### 4.1 CFiCS and QRA's GNN are different *kinds* of object

| | CFiCS | QRA GNN |
|---|---|---|
| Graph is a… | **knowledge taxonomy** — hand-authored CF→IC→skill→example ontology | **data graph** — real transcript segments |
| Edges | **definitional** (fosters/expresses/demonstrates), from theory | **empirical** (temporal order; embedding similarity) |
| Data | **181 synthetic/literature examples**, no real patients, no outcomes | real longitudinal trial corpus, n≈32 |
| Task | **classify into a fixed ontology** (multi-task CF/IC/skill) | **discover + triangulate + distill for scale** |
| Validation | **F1 on held-out synthetic examples** | **out-of-sample per-class κ vs LLM AND vs human** |
| Causal claims | **none** (pure classifier) | explicitly disclaimed |

### 4.2 Why CFiCS does **not** justify QRA heterogeneity (for the core mission)

CFiCS *needs* its heterogeneous taxonomy because (a) it classifies into a fixed ontology, so
the ontology *is* the label space, and (b) it has only 181 synthetic examples, so the
taxonomy is the structural prior that makes few-shot classification work at all (their
graph+ClinicalBERT lifts skill macro-F1 from ~4 to ~96). QRA has neither constraint: it has
real process/temporal structure, and its mission is discovery + triangulation + distillation,
not ontology classification. The three gaps the GNN exists to close — superposition discarded
at majority vote, cue collapsed to five labels, LLM-on-LLM convergence — are **all served by
the homogeneous segment graph**.

### 4.3 The circularity trap

Wiring construct "anchor" nodes "to strengthen independence" is **backwards**. Anchor nodes
would be seeded from the construct-definition text — the *same* text the LLM classifier
already consumes. Cross-framework "lift" edges are derived from the LLM's own output. Feeding
either into the graph makes the GNN's agreement with the LLM **less** independent, not more
— it would inflate GNN↔LLM agreement *by construction*. The homogeneous build is therefore
the **more defensible** substrate for triangulation, not a compromise.

### 4.4 The reframing that matters for the end goal

Once the end goal (MindfulBERT progression dataset) was explicit, the deliberation sharpened:

- **The ground truth for "what progresses participants" is the observed Δprogression of cue
  blocks**, not a learned GNN weight. `analysis/mechanism.py` already analyzes that observed
  signal with *more* rigor than a learned edge weight could (participant-clustered bootstrap
  CIs, within-stage permutation, FDR, mixed-effects).
- A bare **"learned causal influence matrix" is the wrong deliverable**: it overlaps ~85%
  with existing measures and its novel ~15% (learned causal weights) is the *least*
  defensible part at n≈32 observational.
- The GNN's genuinely-additive, defensible contributions to *influence understanding* are
  **(a) model-counterfactual sensitivity** (swap the cue, measure the predicted shift in the
  participant's VAAMR mixture — captures context-dependent/nonlinear influence the additive
  tables miss) and **(b) cross-method triangulation** (convergence with `mechanism.py` is
  stronger evidence than either alone).

**Bottom line:** the homogeneous graph is the most defensible *substrate*; the GNN is the
*scaling engine* and a *candidate-generation/curation/triangulation* instrument; **observed
Δprogression leads the progression claim**; heterogeneity is built only where it both (i)
demonstrably improves the classifier (measured on the gate) and (ii) yields a defensible
influence lens.

---

## 5. Design decisions (the locked record + the why)

Each decision below was made explicitly with the lead researcher.

| # | Decision | Why |
|---|----------|-----|
| D1 | **Homogeneous graph is the default substrate.** | Most defensible for triangulation/distillation; CFiCS does not argue otherwise; anchors risk circularity. |
| D2 | **Independence is reported from a model trained with LLM labels withheld** (G1). | Default `weak` mode trains on LLM ballots → GNN↔LLM agreement is *distillation fidelity*, not independence. Genuine corroboration requires LLM labels withheld. |
| D3 | **Anchors are opt-in and judged on the human κ axis** (G2). | Anchors inflate GNN↔LLM by construction; only a gain on the *human* out-of-sample axis justifies them. Default OFF. |
| D4 | **"Causal" is reframed to model-counterfactual sensitivity + triangulation** (G3). | n≈32 observational + elicitation confound → no causal claim is defensible; model-based sensitivity is. |
| D5 | **Subtext communities are built as routines/sequences with stability selection** (G4). | The genuinely-new ~60% (which language patterns *flow together*); fragile at n≈32 → stability-gated. |
| D6 | **Observed Δprogression leads; GNN scales + curates.** | The observed transition is a more direct, more controlled measure than a learned weight. |
| D7 | **GNN influence analysis is positioned primary — conditionally.** | "Primary" only if it (a) passes the gate and (b) converges with `mechanism.py`; otherwise `mechanism.py` leads. The user chose GNN-primary; the conditionality is the guardrail. |
| D8 | **MindfulBERT labels = observed Δprogression (primary) + GNN-counterfactual augmentation (secondary).** | User chose augmentation; it is provenance-tagged, gate-passing-only, and retained only if a held-out ablation shows it helps — never silently mixed with observed labels. |
| D9 | **Full classifier-hardening track** (abstention, calibration, label propagation, scale-mode sim gate). | This is what actually makes the graph a *trustworthy* LLM-free classifier at scale; the distillation backbone is ~80% built but lacks these trust mechanisms. |
| D10 | **Never train MindfulBERT on un-gated or unvalidated model labels.** | Distilling a model's guesses into another model compounds n≈32 fragility. |
| D11 | **GPU-preferred, CPU-safe compute.** All GNN compute uses CUDA when available (`config.device=None`→auto) and falls back to CPU cleanly; `config.device` governs BOTH the GNN model and the heavy embedding pass. | The 8B Qwen3 embedding + GraphSAGE training are the cost centres; the layer must exploit the GPU when present but never crash without one. Every new track inherits this (see §6a). |

---

## 6a. Compute & GPU preference (cross-cutting mandate)

The GNN layer is **GPU-preferred and CPU-safe**, audited and enforced as of this session:

- **Resolution.** `gnn_layer/train.py:_device(config)` returns `cuda` when
  `torch.cuda.is_available()` and `config.device` is unset, else the explicit `config.device`,
  else `cpu`. Every forward/backward (train, crossval, propagation A4, scale-sim A5) moves
  `x`/`edge_index`/`edge_weight`/`edge_type_ids` and all loss targets onto the model device;
  inference aligns via `inference._graph_tensors_on_model_device` (reads
  `next(model.parameters()).device`). **Audit (this session): no device-mismatch bugs.**
- **Embedding pass honors the device knob.** `EmbeddingClassifierConfig.device` was added and
  is forwarded to `SentenceTransformer(device=…)`; `gnn_layer/embeddings._make_embedder` passes
  `GnnLayerConfig.device` through, so the documented knob now governs the dominant compute (and
  can pin a specific GPU). fp16 on CUDA, fp32 on CPU; OOM falls back to CPU.
- **VRAM hygiene.** The 8B embedder is now built **once per run** (cached, reused for segments +
  anchors) and **freed before GNN training** via `embeddings.release_embedder()` (gc +
  `torch.cuda.empty_cache()`), so it no longer loads twice or coexists with training tensors.
- **Checkpoints + determinism.** `load_checkpoint` moves the scale-mode model to `_device(config)`
  (scale-mode inference now uses the GPU); `set_seed` also seeds `torch.cuda.manual_seed_all`.

**Mandate for the remaining tracks (B, C, D):** any new torch compute MUST move its tensors to
the model device (reuse `_graph_tensors_on_model_device`); any new heavy/embedding compute MUST
honor `config.device` and free GPU memory when done; numpy/sklearn post-processing (bootstraps,
community detection, ECE) stays on CPU by design. B3 counterfactual re-forwards and the C
MindfulBERT trainer are the GPU-relevant additions and must follow this. Tests assert
device resolution and checkpoint placement (`tests/unit/test_gnn_gpu_device.py`).

---

## 6. The defensibility spine

Every deliverable inherits these constraints:

1. **Observational reality.** n≈32, single-arm, unblinded, confounded (participant readiness,
   prior session content, alliance, group dynamics), plus the **elicitation confound**: PURER
   inquiry *elicits* the very language VAAMR scores (`methodology.md` §9.4). → All influence
   outputs are **model-based sensitivity / hypothesis-generating, NOT causal**
   (`methodology.md` §9.2 caveat on every figure).
2. **Gate-before-trust.** Nothing the graph produces — labels, curation, augmentation,
   influence — is used downstream unless the model passes the graph↔LLM IRR gate with the
   rare-stage recall floor (`gnn_layer/validation.py`).
3. **Observed outcome arbitrates.** The GNN generates/scores candidates; the observed
   Δprogression is the label of record. The GNN never overrides the observed label.
4. **Bootstrap CIs, participant-clustered.** Reuse `analysis/stats.py` cluster bootstrap and
   power-flag thin cells; **no silent truncation** (always `log()` what was dropped).
5. **Triangulate and report divergence.** Convergence across independent methods is the
   evidentiary standard; divergence is surfaced for human review, never hidden.

---

## 7. What is already built

The GNN layer (`gnn_layer/`) is a pure-PyTorch GraphSAGE analysis layer (no torch-geometric).
The homogeneous graph = segment nodes + temporal-chain edges + kNN-similarity edges. On it,
five capabilities run, each wrapped so one failure does not abort the rest:

| Capability | What it does | Module |
|---|---|---|
| **A — Continuous VAAMR positioning** | Soft-VAAMR head trained by KL to the multi-run ballot mixture + a scalar progression coordinate E[stage]=Σ k·pₖ; recovers superposition the majority vote discards. | `soft_labels.py`, `model.py`, `inference.py` |
| **B — Cue-motif discovery** | KMeans on cue-block embeddings; from-stage-conditioned logistic influence on forward transitions; emergent-motif flag (influential but low PURER purity). | `motifs.py` |
| **C — Triangulation** | GNN head predictions vs LLM (`final_label`/`purer_primary`) and vs the human blind subset, via Cohen's κ; the code itself labels GNN↔LLM "distillation fidelity" and GNN↔human "independent quality." | `triangulation.py`, `gnn_lift.py` |
| **D — Ablation** | Remove a construct head, retrain, report Δ; plus the VCE-on-VAAMR contribution test (gate κ with/without the VCE head). | `ablation.py` |
| **E — Coupling** | PCA latent factors of cue-block embeddings; per-factor correlation with forward movement; named against an inline CF/IC lexicon (discovered, not imposed). | `coupling.py` |

**Distillation & scale machinery (built):**
- Consensus-distillation training on LLM ballots; semi-supervised transductive training
  (unlabeled nodes still receive messages + train the link-prediction/contrastive heads).
- **Reliability gate** (`validation.py`): out-of-sample, label-masked k-fold, per-VAAMR-stage
  & per-PURER-move κ + rare-stage recall floor (Metacognition/Reappraisal) + an independent
  human-axis (κ(graph,human) vs κ(LLM,human)); explicit "ready for LLM-free scaling?" verdict.
- **Inductive scale mode** (`runner.run_gnn_classify`): loads the frozen training graph,
  attaches only unseen segments by kNN (`attach_new_segments` with correct node-type tagging),
  writes a `gnn_labels` overlay — no LLM calls, no retraining.
- **Label-of-record promotion** (`process/assembly/master_dataset.py`): priority
  adjudicated > human_consensus > **gnn_consensus** > llm_zero_shot; the `gnn_consensus` tier
  engages only when `gnn_authoritative=True`. Raw LLM ballots always preserved for audit.

**Gaps that motivate the tracks:** no abstention/deferral, no domain-shift calibration, no
explicit label propagation, the gate doesn't simulate inductive attachment of new sessions,
and `gnn_authoritative` promotes from the config flag alone without checking the gate verdict.

---

## 8. What we did this session (Phases 0–2 + Track 0)

### Phase 0 — Integrity & doc honesty (G5) ✅
- Corrected the `enabled` default to **ON** across all stale sites (config class docstring
  self-contradiction, `__init__.py`, `runner.py`, `methodology.md`, `USAGE.md`, `README.md`).
- Removed the `__init__.py` "SCAFFOLD / NotImplementedError" claim (every module is
  implemented).
- Marked `run_on_participants`/`run_on_therapists` as **reserved/not-enforced** (config theater).
- Found both GNN design docs deleted in the working tree; re-pointed 5 dangling references to
  `methodology.md` §8.5 / `gnn-influence-to-execution.md`; added **DESIGNED—NOT-BUILT** banners
  to ROADMAP §3.3 (subtext) and §3.7 (causal); rewrote the Phase 3 status note.
- (The `n_microskill` `build_model` crash and the `attach_new_segments` scale-mode bug the
  earlier memo flagged were already fixed on this branch — verified.)

### Phase 1 — Independence pass (G1) ✅
- Added a **second training pass with LLM labels withheld** that writes
  `triangulation_independence.txt`. Modes: `human` (heads supervised only by the blind subset
  → GNN↔LLM is genuine corroboration), `self_supervised` (geometry-only NULL control, low κ
  expected), `auto` (human when a usable subset exists, else self_supervised).
- New: `gnn_layer/runner._independence_mode`, mode-aware framing in
  `triangulation.write_triangulation_report(mode, filename)`, config fields
  `report_independence_pass`, `independence_label_mode`, `independence_min_human`.
- 8 unit tests + an end-to-end assertion; all green.

### Phase 2 — Anchors + human-axis ablation (G2) ✅
- New `gnn_layer/anchors.py`: build construct-anchor features from VAAMR/PURER/VCE definitions
  (via `theme_framework.registry.load`, avoiding the known `get_purer_framework` path bug);
  **label-free** anchor↔segment similarity edges; LLM-derived cross-framework anchor↔anchor
  lift edges (VCE only).
- New `ablation.anchor_contribution`: trains with/without anchors on identical folds and scores
  **Δκ on the GNN↔human axis** (decisive), reporting GNN↔LLM as an explicitly *inflated*
  secondary number. Verdict `inconclusive` when the human subset is too small → anchors stay OFF.
- Runner wiring: `use_anchor_nodes` (main graph; default OFF), `run_anchor_ablation` (default OFF).
- 15 tests (12 hermetic + 3 slow); broad GNN suite (135 tests) green.

### Track 0.1 — Baseline stabilization (in progress) ✅ (fixes applied; final verification running)
The authoritative unit baseline was **90 failing tests (27 failures + 63 errors)** on this
WIP branch. Triaged via **7 read-only diagnosis subagents** (one per cluster) and fixed via
**4 parallel implementation subagents** over disjoint file sets, plus hand-fixes for
ripple-risky items. Outcome:

**Real code bugs fixed:**
- `theme_framework/purer.py` — `parents[2]`→`parents[1]` path bug.
- `analysis/exemplars.py` — unguarded `llm_run_consistency` (the dominant cause of the
  ~40-test session/participant/theme cluster; these bypass the loader).
- `analysis/loader.py`, `analysis/stage_progression.py`, `analysis/figures.py`,
  `analysis/efficacy.py` (empty-frame guard + unconditional observational caveat),
  `analysis/stats.py` (Wilson-CI clamp).
- `process/config.py` — content-validity `vaamr` default aligned to `True` (verified intent
  against the parse path; updated the contradicting stale test).
- `codebook/embedding_classifier.py` — empty-codebook guard.
- `process/orchestrator.py` — `stage_ingest` now discovers files first and **short-circuits on
  an empty dir before constructing the embedding model** (a real efficiency improvement;
  fixes the empty-dir CLI test in any environment).

**Stale tests updated** (justified by tracing current code is correct): CLI `theme`→`vaamr`
rename + `_build_parser` 3-tuple; GNN assertion drift (`cv['purer']` now populated, 0.9 vs 0.8,
Greek `δloss`); asserts-outside-`with`-block; `assertIsNone` vs float64 `nan`; theme-schema
key-collision count; cls-validation dual-stratum invariant; `make_master_df` session-id
normalization (fixed locally in `test_longitudinal._part_df`).

**Re-tiered, not falsely fixed:** the `--what codebook` subprocess test genuinely needs the
Qwen3 embedding model (unavailable in this sandbox); its flag-bypass logic is already correct,
so it was marked `@slow_test`.

### Track 0.2 — Affirm the gate ✅
The gap (a config flag promoting GNN labels with no check that the reliability gate passed)
is closed: the gate now persists a machine-readable verdict (`03_analysis_data/gnn/gnn_gate.json`),
`assemble_master_dataset` gates the `gnn_consensus` tier on `gnn_authoritative AND gate_passed`
(default `gate_passed=False`), and all 3 orchestrator sites compute `gate_passed` from the
persisted verdict via `_gnn_promotion_flags`. See §16 "Track 0.2 — as built" for the full
file-level account.

---

## 9. The tracks — exhaustive specification

### Track 0 — Stabilize baseline + affirm the gate

**Goal:** a green hermetic unit baseline (so downstream Δκ deltas are measured cleanly) and a
gate that is the *hard precondition* for any graph label of record.

- **0.1 (done):** triage + fix the 90 failures (see §8).
- **0.2 (remaining) — gate-gated promotion.**
  - **Why:** the bootstrapping chain's whole safety rests on the gate; a config flag must not
    be able to promote an un-gated graph to label of record.
  - **What:**
    1. Persist the gate verdict machine-readably when `evaluate_crossval` runs — e.g.
       `gnn_layer/validation.py` writes `03_analysis_data/gnn/gnn_gate.json`
       `{ready_for_scaling: bool, vaamr_kappa, purer_kappa, rare_stage_ok, irr_target, timestamp}`.
    2. In `process/orchestrator.py` (the 3 sites that read
       `config.gnn_layer.gnn_authoritative`), compute the *effective* flag as
       `gnn_authoritative AND gate_verdict.ready_for_scaling`. If the flag is True but the
       verdict is missing/False, force False and log a clear warning.
    3. Optionally thread an explicit `gate_passed` arg into
       `assemble_master_dataset(..., gnn_authoritative=..., gate_passed=...)` so the safeguard
       is enforced at the assembly boundary, not just the caller.
  - **Verification:** a unit test that `assemble_master_dataset` with `gnn_authoritative=True`
    but a failing/absent gate verdict leaves `final_label_source == 'llm_zero_shot'` (graph
    cannot label of record un-gated). Update `test_gnn_consensus`/`test_methodology_boundary`
    as needed.
  - **Files:** `gnn_layer/validation.py`, `process/orchestrator.py`,
    `process/assembly/master_dataset.py`, tests under `tests/unit/`.

### Track A — Scalable, trustworthy label engine

**Goal:** make the GNN a *trustworthy* LLM-free classifier so it can label the full corpus
without poisoning the MindfulBERT training set. Precondition for "GNN-primary" influence.

- **A1 — Typed `precipitates` message passing (shared substrate with B/C). ✅ done.**
  - **Why:** lets the participant representation use the preceding therapist cue (potential
    classifier gain) and is the structural handle for B's influence readout.
  - **As built:**
    - `graph_builder.py`: canonical `EDGE_TYPE_VOCAB = ('temporal','knn','anchor','precipitates')`
      + `EDGE_TYPE_TO_ID`; `HeteroGraph` gained an `edge_type_ids` LongTensor (aligned with
      `edge_index`). When `config.precipitates_edges`, each cue block's therapist segments are
      connected to the FOLLOWING participant segment via `process.cue_blocks.cue_blocks_from_records`
      (`require_stage=False`); `meta['n_precipitates']` + `meta['edge_type_vocab']` recorded.
      `attach_new_segments` / `save_graph` / `load_graph` all preserve `edge_type_ids` (new
      inductive edges tagged `knn`).
    - `model.py`: `SAGEConv` gained a **learnable per-edge-type gate** (`edge_type_gate`,
      `nn.Parameter` of size `n_edge_types`), applied as `w *= softplus(gate)[edge_type_ids]`.
      Initialized at `log(e−1)` so `softplus(gate)=1` → **the OFF path is byte-identical** to the
      original fixed-weight aggregation. `MultiTaskGNN.encode/forward` thread `edge_type_ids`;
      `build_model` sets `n_edge_types = len(vocab)` only when `precipitates_edges` is on
      (otherwise 0 → zero added params). `train.py` / `inference.py` pass `graph.edge_type_ids`
      through every forward call.
  - **Decision rule (as built):** `ablation.precipitates_contribution` builds the graph
    with/without the family on identical folds/seed and compares out-of-sample κ on BOTH axes.
    Unlike anchors, precipitates edges are EMPIRICAL (not seeded from construct definitions), so
    the LLM axis is legitimate here. The family is recommended ON only if Δκ ≥ +0.02 on **both**
    the LLM and human axes AND the human subset is decisive (`n ≥ anchor_min_human`); otherwise
    `inconclusive`/`harm`/`neutral` → stays OFF (the homogeneous default). Gated behind
    `config.run_precipitates_ablation`; writes `06_reports/06_gnn/precipitates_contribution.txt`.
  - **Checkpoint:** run with `precipitates_edges=True, run_precipitates_ablation=True` on the
    real corpus; keep the family in the main graph only if the report says
    `RECOMMEND precipitates_edges ON: YES`.
  - **Tests:** `tests/unit/test_gnn_precipitates.py` (14 tests) — graph/edge-type-id alignment,
    save/load + attach roundtrip, gate presence/absence, neutral-init equivalence, OFF-path
    param-count invariance, ablation `inconclusive` without a human subset + report write.
  - **Files:** `graph_builder.py`, `model.py`, `train.py`, `inference.py`, `ablation.py`,
    `config.py`, `runner.py`.
- **A2 — Abstention / deferral. ✅ done.**
  - **Why:** a confident wrong label on a novel segment poisons the training set; the graph
    must be able to say "I don't know — keep the LLM label."
  - **As built:**
    - `inference.py`: `resolve_abstain_floors(config)` (per-VAAMR-stage max-prob floors) +
      `resolve_purer_abstain_floor(config)` (global PURER floor). Precedence: explicit
      `abstain_per_stage` > global `abstain_threshold` (with a higher
      `abstain_rare_stage_threshold` for the rare stages 3/4) > disabled (None → never abstain).
      `infer_head_predictions` emits `gnn_vaamr_abstain` / `gnn_purer_abstain` (bool per segment)
      ONLY when a floor is configured — the OFF path is unchanged.
    - `data_structures.py` + `classifications_io.py`: `gnn_vaamr_abstain` / `gnn_purer_abstain`
      added to the `Segment` and to `GNN_OVERLAY_FIELDS`, so abstention round-trips the overlay
      and every deferral is auditable.
    - `master_dataset.py`: the `gnn_consensus` tier is suppressed when the segment abstained
      (`getattr(seg,'gnn_*_abstain',False)`), so the LLM label is kept even when
      `gnn_authoritative AND gate_passed`. Abstain flags surfaced as output columns.
    - `runner.py`: threads abstain flags into both the analyze-time consensus overlay and the
      scale-mode (`run_gnn_classify`) overlay; logs the abstain count ("→ LLM label kept").
    - **Calibration (held-out, rare→higher floor):** `train.crossval_predictions(..., return_conf=True)`
      now also returns held-out confidences; `train.calibrate_abstain_floors` picks, per stage,
      the smallest floor whose KEPT held-out precision meets `abstain_target_precision`
      (hard/rare stages naturally get higher floors). Wired behind `config.abstain_calibrate`
      (writes the derived floors into `config.abstain_per_stage` before inference).
  - **Tests:** `tests/unit/test_gnn_abstention.py` (14) — floor precedence, inference flag
    presence + high/zero-floor extremes, overlay roundtrip, master_dataset deferral (VAAMR +
    PURER) with non-abstained promotion, calibration shape/clamping. Plus updated
    `test_gnn_overlay_io` field set.
  - **Files:** `inference.py`, `train.py`, `config.py`, `runner.py`, `classifications_io.py`,
    `data_structures.py`, `process/assembly/master_dataset.py`.
- **A3 — Confidence calibration for domain shift. ✅ done.**
  - **Why:** softmax confidence is trained in-distribution; on genuinely new transcripts it may
    be miscalibrated (over-confident).
  - **As built (new `gnn_layer/calibration.py`):**
    - **Temperature scaling** — `fit_temperature` learns a scalar T (log-space Adam, NLL on
      held-out logits vs the LLM consensus); `apply_temperature` divides logits by T;
      `expected_calibration_error` reports ECE; `temperature_from_cv` pairs the gate's held-out
      logits with `final_label`. `crossval_predictions(..., return_logits=True)` exposes the
      held-out logits so the gate's CV is reused (no extra retraining).
    - **OOD score** — `ood_scores` = mean cosine distance of a new segment to its k nearest
      TRAINING segments.
    - Wiring: `inference.py` applies T to the soft-VAAMR logits (affecting the mixture, the
      progression coordinate, and the A2 abstention confidences); `runner.py` fits T in the gate
      block when `config.calibrate` (persisted in `gnn_gate.json` as `calibration_temperature` +
      `ece_before/after`, reused by scale-mode); the scale-mode OOD gate forces ABSTAIN when a
      new segment's OOD score exceeds `config.ood_threshold`; `validation.py` reports T + ECE.
  - **Tests:** `tests/unit/test_gnn_calibration.py` (12) — T recovery + ECE drop on
    over-sharpened logits, no-op/degenerate guards, OOD ranking + empty-input guards,
    `return_logits` shape, `temperature_from_cv` structure, inference applies T (softens conf).
  - **Files:** `calibration.py` (new), `train.py`, `inference.py`, `runner.py`, `validation.py`,
    `config.py`.
- **A4 — Semi-supervised label propagation (optional, measured). ✅ done.**
  - **Why:** unlabeled nodes currently get classifier outputs but do not inherit neighbor
    soft-labels explicitly; diffusion may sharpen coverage near labeled regions.
  - **As built (new `gnn_layer/propagation.py`):** `propagate` diffuses the trained model's
    per-node soft predictions over the temporal/kNN edges
    (`F ← α·neighbour_weighted_mean(F) + (1−α)·P_model`, row-renormalized; numpy mirror of the
    SAGE aggregation). `propagation_contribution` mirrors the gate's k-fold and scores held-out
    κ raw-model vs diffused against `final_label`; **retained only if Δκ ≥ +0.02** (verdict
    `propagation_helps`/`harms`/`neutral`/`inconclusive`). Writes
    `06_gnn/label_propagation.txt`; gated behind `config.label_propagation` (`propagation_alpha`,
    `propagation_iters`).
  - **Tests:** `tests/unit/test_gnn_propagation.py` (6) — neighbour-mean on a hand graph,
    α=0 identity, row-sum-1 invariant, contribution verdict/Δκ + report, OFF default.
  - **Files:** `propagation.py` (new), `runner.py`, `config.py`.
- **A5 — Scale-mode simulation gate. ✅ done.**
  - **Why:** the k-fold gate trains on the same topology; it does not simulate attaching
    genuinely-new sessions inductively — the actual scale-mode condition.
  - **As built (`validation.py`):** `scale_mode_simulation` holds out whole sessions per fold,
    trains on the rest, attaches the held-out sessions via `attach_new_segments` (kNN-only, no
    temporal context — the real scale condition), scores held-out VAAMR κ vs `final_label`, and
    compares it to the full-graph in-sample CV κ. The gap (`κ_cv − κ_inductive`) above
    `scale_sim_max_gap` raises `domain_shift_risk`. Writes `06_gnn/scale_sim.txt`; gated behind
    `config.run_scale_sim` (`scale_sim_holdout_sessions`). Guards <2 sessions / missing
    `session_id`.
  - **Tests:** `tests/unit/test_gnn_scale_sim.py` (6) — runs + returns both κ + gap + flag,
    single-session & missing-column skips, report write, max-gap respected, OFF default.
  - **Files:** `validation.py`, `runner.py`, `config.py`.

> **Track A complete.** The graph is now a *trustworthy* LLM-free classifier: typed cue
> message passing (A1), abstention/deferral (A2), domain-shift calibration + OOD (A3), measured
> label propagation (A4), and an inductive scale-mode gate (A5) — each opt-in and measured, so
> nothing un-validated reaches the MindfulBERT training set.

### Track B — Therapist→participant progression analysis ✅ done

**Goal:** the deepest *defensible* understanding of how therapist language progresses
participant expression. Observed Δprogression leads; the GNN is the contextual/nonlinear lens
and an independent corroborating method.

- **B1 — Observed Δprogression on cue blocks = ground truth. ✅ (already built in `analysis/mechanism.py`).**
  - For every cue block the participant VAAMR transition before→after is computed as a signed Δ
    on the E[stage] progression coordinate (`_enrich_blocks`), with the continuous deadband
    classes (progress/stabilize/regress). Reuses `cue_blocks.py` + the `soft_labels`
    progression coordinate. This is the spine of both the analysis and the Track C training labels.
- **B2 — Rigorous association (LEAD). ✅ (already built in `analysis/mechanism.py`).**
  - `analysis/mechanism.py` already runs at analyze-time over the assembled corpus: signed
    Δprogression per (from_stage, PURER move) and (from_stage, motif), participant-clustered
    bootstrap CIs, within-stage permutation, FDR, and a mixed-effects model. This is the primary
    "which patterns progress" evidence and the label of record.
- **B3 — GNN model-counterfactual (candidate generation + curation; secondary, gated). ✅ done (new `gnn_layer/influence.py`).**
  - **As built:** `purer_centroids` builds each PURER move's centroid from therapist node
    *input features* + a neutral null baseline (mean therapist feature). For each mediated cue
    block, `counterfactual_influence` swaps the block's therapist node feature(s) with each move
    centroid (vs the null), re-forwards, and records the shift in the FOLLOWING participant's
    predicted progression coordinate. Aggregated per move and per (from_stage × move) with
    participant-clustered bootstrap CIs (reusing `analysis/stats.cluster_bootstrap_ci`).
    **Gated:** the runner invokes it only from a gate-passing model
    (`validation.gate_ready_for_scaling`); suppressed (logged) otherwise.
    **GPU (D11):** the per-block re-forwards run on `_device(config)` via
    `inference._graph_tensors_on_model_device`; block count is capped by
    `counterfactual_max_blocks` with a logged note (never silent).
- **B4 — Triangulation. ✅ done.**
  - `influence.triangulate` aligns the counterfactual influence ranking with `mechanism.py`'s
    observed signed-Δprogression per PURER move (parsed from `mechanism_delta_progression.csv`):
    Spearman rank correlation + per-move sign agreement, with a per-move `converges` flag so
    DIVERGENCES are surfaced for human review, never hidden. Report under
    `06_gnn/influence.txt` + `gnn_counterfactual_influence.csv`, with the non-causal caveat on
    every section.
- **B5 — Context/subgroup sensitivity (only if N supports). ✅ done.** `subgroup_influence`
  splits counterfactual influence by session-number tertile (early/mid/late) with explicit
  underpowered flags; thin cells (< 8 blocks) are dropped with a logged note. Gated behind
  `counterfactual_subgroups`. (Kinesiophobia-tertile split is deferred until the external
  outcome is wired into the corpus; the session-phase split ships now.)
- **Tests:** `tests/unit/test_gnn_influence.py` (10) — centroids/null, per-move + per-stage
  tables, cluster-bootstrap CIs, block cap, triangulation parse + structure, report/CSV writers,
  subgroup sidecar, OFF defaults.
- **Files:** `gnn_layer/influence.py` (new), `runner.py`, `config.py`
  (`counterfactual`, `counterfactual_max_blocks`, `influence_bootstrap_n`,
  `counterfactual_subgroups`).

### Track C — MindfulBERT training-set builder ✅ done

**Goal:** the end-goal artifact — a versioned dataset for fine-tuning MindfulBERT to predict
progression-inducing language. **As built in `process/assembly/mindfulbert_dataset.py`
(`build_mindfulbert_dataset`), wired into `analysis/runner.py` (§12b, behind
`config.build_mindfulbert_dataset`).**

- **C1 — Example assembly. ✅** Units = cue blocks (`inference.build_cue_blocks_with_segments`).
  Each example = (preceding participant `context_text` + `from_stage`, therapist `cue_text`,
  dominant PURER move, word/segment counts) → label. Output under `02_meta/training_data/`.
- **C2 — Primary labels = observed Δprogression. ✅** Signed `delta_progression` (E[stage] coord
  when present, else stage difference — recorded as `label_basis`) + categorical `direction`
  (advanced/stayed/regressed). **Per-example provenance**: the WEAKEST of the two endpoints'
  label-source tiers (adjudicated > human_consensus > gnn_consensus > llm_zero_shot), a GNN
  abstention flag, and the gate verdict.
- **C3 — GNN-counterfactual augmentation (secondary). ✅** When `augmentation_enabled` AND the
  gate passed, `_attach_augmentation` adds a `would_progress` value (from
  `gnn_counterfactual_influence.csv`) as a **separate, provenance-tagged** (`gnn_counterfactual`)
  channel; suppressed (logged) when the gate has not passed; never merged with the observed label.
- **C4 — Augmentation validation (the safeguard). ✅** `_augmentation_ablation` trains a
  lightweight logistic-regression proxy to predict the observed binary outcome from cue features
  **with vs without** the augmentation channel, using **participant-grouped** CV (`GroupKFold`,
  no participant leakage). Augmentation is **RETAINED only if the held-out gain exceeds
  `augmentation_min_gain`** — otherwise the channel is stripped from the exported examples.
- **C5 — Export + datasheet. ✅** Versioned `mindfulbert_dataset.jsonl` +
  `mindfulbert_datasheet.{json,txt}` recording dataset version, provenance mix, direction
  distribution, abstention count, gate status, the C4 ablation result, and the
  n≈32/observational/non-causal caveats.
- **Tests:** `tests/unit/test_mindfulbert_dataset.py` (8) — example/provenance build, coord vs
  stage-difference basis, datasheet write, augmentation suppression without a gate + attachment
  with one, the C4 ablation proxy (metrics + too-few guard), OFF defaults.
- **Files:** `process/assembly/mindfulbert_dataset.py` (new), `process/assembly/__init__.py`,
  `analysis/runner.py`, `config.py` (`build_mindfulbert_dataset`, `augmentation_enabled`,
  `augmentation_min_gain`).

> **Note on MindfulBERT itself.** Fine-tuning MindfulBERT is ROADMAP Phase 6 (downstream). The
> deliverable here is the *dataset builder* + the *augmentation-validation harness*; the C4
> ablation uses a lightweight sklearn proxy (CPU by design per D11) until Phase 6 wires in the
> real trainer.

### Track D — Subtext communities as routines ✅ done

**Goal:** the "deepest qualitative analysis" layer — which language *routines/sequences* flow
together and recur across sessions (the genuinely-new ~60% vs motifs/coupling). **As built in
`gnn_layer/communities.py` (`run_subtext_communities`), wired into `runner.py` behind
`config.subtext_communities` — independent of the gate (discovery, hypothesis-generating).**

- **D1 — Subtext similarity graph. ✅** `build_subtext_graph` builds a thresholded
  (cosine ≥ `community_sim_threshold`, default 0.85) similarity graph over the raw Qwen3
  segment embeddings (distinct from the trained kNN graph), recording the cross-session edge
  fraction; caps at `max_nodes` with a logged note.
- **D2 — Community detection, two algorithms. ✅** `detect_communities` partitions with
  **Louvain** (`networkx.community.louvain_communities` — no `python-louvain` dependency needed)
  and, as a different algorithmic family, **agglomerative hierarchical clustering** (sklearn,
  cosine/average linkage); their **adjusted Rand index** is reported so a community counts as
  structure, not an algorithm artifact.
- **D3 — Routine/sequence modeling (the novel part). ✅** `community_transitions` counts
  within-session community→community transitions ("X tends to precede Y") — language routines,
  not isolated moves.
- **D4 — Stability selection. ✅** `community_stability` runs participant-bootstrap resampling
  (rebuild graph → re-detect → measure co-membership), reports each community's co-membership
  stability, and **suppresses/flags communities below `community_stability_min`** (n≈32 →
  fragile). The report separates STABLE (findings) from UNSTABLE/SUPPRESSED (flagged, not
  dropped). Cites stability-selection / consensus-clustering literature (not CFiCS).
- **D5 — Semantic naming + drift. ✅** `name_communities` adds TF-IDF terms + exemplar quotes
  per community, per-session prevalence, and the cross-cohort distribution (drift), all
  hypothesis-generating.
- **Tests:** `tests/unit/test_gnn_communities.py` (9) — graph edges + cross-session count + cap,
  two-algorithm partition + ARI, routine transitions, bootstrap stability dict, TF-IDF naming,
  orchestrator report write, too-few-segments skip, OFF defaults.
- **Files:** `gnn_layer/communities.py` (new), `runner.py`, `config.py`
  (`subtext_communities`, `community_sim_threshold`, `community_min_size`,
  `community_stability_min`, `community_stability_boots`).

---

## 10. Sequencing, checkpoints, dependencies

```
Track 0  ─────────────▶ Track A ─────────▶ Track B ─────────▶ Track C
(green baseline +        (A1 first;          (needs a           (needs B1/B2
 gate-gated promotion)    gate κ checkpoint)  gate-passing        + gate-passing
        │                      │              model)              model;
        │                      │                                  C4 ablation
        ▼                      ▼                                  checkpoint)
   (precondition)         CHECKPOINT:                                  │
                          do typed edges                               ▼
                          raise gate κ?                          Track D (independent; last)
```

- **After Track 0 + A1:** checkpoint — do typed `precipitates` edges raise gate κ? Decides
  whether they stay in the main graph and whether B3 is worth running.
- **After Track B:** checkpoint — does the GNN converge with `mechanism.py` enough to be
  "primary"? If not, `mechanism.py` leads and the report says so.
- **After Track C:** checkpoint — does augmentation pass its retention ablation (C4)?
- **A2–A5** can proceed alongside B. **Track D** is independent and last.

---

## 11. Module & file index

**Existing (edit):**

| Path | Role |
|------|------|
| `gnn_layer/model.py` | `SAGEConv` (add learnable edge-type weights), multi-task heads, losses |
| `gnn_layer/graph_builder.py` | graph assembly; `edge_types`, `attach_new_segments`, anchors, lift; add `precipitates` |
| `gnn_layer/train.py` | `assemble_targets`, `train_model`, `crossval_predictions`; sim-gate, label-prop |
| `gnn_layer/inference.py` | head predictions, positions, cue blocks; abstention, counterfactual |
| `gnn_layer/validation.py` | reliability gate; sim-gate, calibration report, persisted verdict |
| `gnn_layer/soft_labels.py` | ballot→mixture, progression coordinate, label modes |
| `gnn_layer/triangulation.py` | GNN↔LLM↔human κ; independence-pass framing |
| `gnn_layer/ablation.py` | head ablation, VCE contribution, `anchor_contribution` |
| `gnn_layer/anchors.py` | construct-anchor features + similarity/lift edges (Phase 2) |
| `gnn_layer/runner.py` | orchestration; `run_gnn_analysis`, `run_gnn_classify`, independence pass |
| `gnn_layer/config.py` | `GnnLayerConfig` — all flags |
| `process/cue_blocks.py` | canonical FROM→CUE→TO builder (reuse for `precipitates`, B1, C1) |
| `analysis/mechanism.py` | signed Δprogression association (Track B LEAD) |
| `analysis/stats.py` | cluster bootstrap, power flag (reuse everywhere) |
| `process/assembly/master_dataset.py` | label-of-record promotion (gate-gate it) |
| `process/assembly/training_export.py` | training-data export path (reuse for Track C) |
| `process/orchestrator.py` | reads `gnn_authoritative` (gate-gate it); `stage_ingest` |

**New (all built):**

| Path | Track | Role | Status |
|------|------|------|--------|
| `gnn_layer/influence.py` | B3/B4/B5 | model-counterfactual sensitivity + triangulation + subgroup readout | ✅ built |
| `gnn_layer/calibration.py` | A3 | temperature/conformal + OOD score | ✅ built |
| `gnn_layer/propagation.py` | A4 (optional) | post-training soft-label diffusion | ✅ built |
| `gnn_layer/communities.py` | D | subtext graph + two-algorithm community detection + stability + naming | ✅ built |
| `process/assembly/mindfulbert_dataset.py` | C | training-set builder + augmentation-validation harness + datasheet | ✅ built |

**Reports/data:** `06_reports/06_gnn/` (validation, triangulation, triangulation_independence,
anchor_contribution, **influence**, **communities**, scale_sim, label_propagation,
precipitates_contribution); `02_meta/training_data/` (**mindfulbert_dataset.jsonl**,
**mindfulbert_datasheet.{json,txt}** — Track C); `03_analysis_data/gnn/`
(**gnn_counterfactual_influence.csv**, **gnn_counterfactual_influence_by_phase.csv**,
**subtext_communities.csv**, **subtext_community_transitions.csv**, CSVs, persisted gate verdict);
`03_analysis_data/mechanism/` (observed Δprogression — Track B1/B2 LEAD).

---

## 12. Configuration reference

`gnn_layer/config.py:GnnLayerConfig` (current + planned). Existing unless marked **[planned]**.

| Field | Default | Meaning |
|-------|---------|---------|
| `enabled` | `True` | master switch; layer runs at analyze-time, fully guarded |
| `embedding_model` | `Qwen/Qwen3-Embedding-8B` | reused embedding substrate |
| `knn_k` | `8` | kNN similarity edges per segment |
| `include_vaamr_nodes` / `include_purer_nodes` | `True` | anchor families (when anchors used) |
| `include_vce_nodes` | `False` | VCE anchor nodes + cross-framework lift edges |
| `cross_framework_min_lift` | `1.5` | threshold for VAAMR↔VCE anchor edges |
| `use_anchor_nodes` | `False` | include anchors in the MAIN graph (must earn it via ablation) |
| `run_anchor_ablation` | `False` | run the with/without-anchors human-axis Δκ test |
| `anchor_knn_m` / `anchor_min_human` | `8` / `10` | segments per anchor; min human rows for a decisive ablation |
| `label_mode` | `'weak'` | `weak` (LLM ballots) / `human` / `self_supervised` |
| `report_independence_pass` | `True` | run the LLM-labels-withheld triangulation pass (G1) |
| `independence_label_mode` | `'auto'` | `auto`/`human`/`self_supervised` |
| `independence_min_human` | `10` | min human rows for `auto` to pick `human` |
| `objectives` | soft_vaamr, progression, contrastive, link_prediction | trained heads/losses |
| `hidden_dim`/`n_layers`/`dropout`/`epochs`/`lr`/`patience`/`seed` | 128/2/0.5/300/1e-3/40/42 | model + training |
| `n_motif_clusters`/`min_motif_influence`/`motif_min_block_count` | 12/1.2/3 | Capability B |
| `n_latent_factors`/`interpret_against_cf_ic` | 5/`True` | Capability E |
| `run_gnn_ablation`/`test_vce_layer` | `False`/`False` | Capability D instruments |
| `produce_consensus_labels` | `True` | write per-segment graph labels to the overlay |
| `gnn_authoritative` | `False` | promote graph labels to label of record — **effective only when the persisted gate verdict reports `ready_for_scaling` (Track 0.2, done)** |
| `validation_folds`/`validation_holdout`/`irr_target` | 5/0.2/0.70 | reliability gate |
| `precipitates_edges` | `False` | **(A1, done)** typed therapist→participant edges + learnable per-edge-type gate |
| `run_precipitates_ablation` | `False` | **(A1, done)** run the with/without-precipitates Δκ checkpoint (both axes) |
| `abstain_threshold` | `None` | **(A2, done)** global max-prob floor; None disables abstention |
| `abstain_rare_stage_threshold` | `None` | **(A2, done)** higher floor for rare VAAMR stages (3,4) |
| `abstain_per_stage` | `None` | **(A2, done)** explicit per-stage floors (overrides; also where calibration writes) |
| `abstain_calibrate` / `abstain_target_precision` | `False` / `0.80` | **(A2, done)** derive per-stage floors from held-out CV at target precision |
| `calibrate` / `calibration_temperature` | `False` / `None` | **(A3, done)** fit + apply temperature scaling on the soft-VAAMR head |
| `ood_threshold` / `ood_knn_k` | `None` / `8` | **(A3, done)** scale-mode OOD deferral (mean kNN cosine distance to training) |
| `label_propagation` / `propagation_alpha` / `propagation_iters` | `False` / `0.5` / `20` | **(A4, done)** measured post-training soft-label diffusion (kept only if Δκ ≥ +0.02) |
| `run_scale_sim` / `scale_sim_holdout_sessions` / `scale_sim_max_gap` | `False` / `1` / `0.10` | **(A5, done)** inductive whole-session holdout vs CV κ; flags domain-shift risk |
| `device` (GnnLayerConfig + EmbeddingClassifierConfig) | `None` | **(D11, done)** compute device; None → auto-CUDA, else pin (`'cuda'`/`'cuda:1'`/`'cpu'`); governs GNN model AND embedding pass |
| `counterfactual` | `False` | **(B3, done)** run the model-counterfactual influence pass (GATED on the reliability gate) |
| `counterfactual_max_blocks` / `influence_bootstrap_n` | `None` / `1000` | **(B3, done)** cap on per-block re-forwards (logged when capped); participant-clustered bootstrap resamples |
| `counterfactual_subgroups` | `False` | **(B5, done)** split counterfactual influence by session-number tertile (underpowered-flagged) |
| `build_mindfulbert_dataset` | `False` | **(C, done)** build the versioned (cue language → observed Δprogression) MindfulBERT dataset |
| `augmentation_enabled` / `augmentation_min_gain` | `False` / `0.0` | **(C3/C4, done)** add the GNN-counterfactual channel (gate-passing only); retain only if the held-out C4 gain exceeds this |
| `subtext_communities` | `False` | **(D, done)** run the subtext-community / routine discovery layer (gate-independent) |
| `community_sim_threshold` / `community_min_size` | `0.85` / `3` | **(D1/D5, done)** cosine edge threshold; minimum community size to report |
| `community_stability_min` / `community_stability_boots` | `0.5` / `50` | **(D4, done)** suppress communities below this bootstrap co-membership stability; resamples |

---

## 13. Testing strategy & conventions

- **Two tiers.** `tests/unit/` is hermetic (no network, no model downloads, no Ollama) — the
  `@slow_test` gate (`tests/testhelpers/marks.py`) skips real-model tests unless `QRA_RUN_SLOW=1`.
  `tests/integration/` runs real tiny models. Run `python tests/run_unit_tests.py` and
  `python tests/run_integration_tests.py`.
- **Hermetic GNN tests** patch `gnn_layer.embeddings.embed_segment_texts` (and
  `embed_anchor_texts`) via `tests/testhelpers/fixtures.embedding_patch` so no 16GB model loads;
  fixtures via `synthetic_df` / `make_master_df`.
- **Per track:** unit tests for each new function + a `@slow_test` end-to-end through
  `run_gnn_analysis`; ablation/gate deltas recorded in the validation report; influence outputs
  asserted suppressed when the gate verdict is NO; community stability asserted to suppress
  unstable communities on resamples.
- **Sandbox caveat:** this environment's `transformers` cannot load Qwen3, so any test needing
  the real embedding model belongs in the slow/integration tier — never "fixed" by faking a pass.

---

## 14. Risk & honesty register

- **GNN-primary is conditional.** If the GNN fails the gate or diverges from `mechanism.py`, it
  cannot be the primary mechanistic evidence — `mechanism.py` leads and the report says so.
- **Augmentation is provisional.** Retained only if the C4 ablation shows a held-out gain;
  otherwise dropped. Never silently mixed with observed labels.
- **Counterfactual = model sensitivity, not causation.** Stated on every artifact; n≈32 caveats
  on every influence/community figure; subgroup analyses flagged underpowered.
- **Communities are fragile.** Nothing below the stability threshold is reported as a finding.
- **No model-distilling-model.** MindfulBERT trains on observed progression; GNN signals only
  curate/scale/augment-under-validation.
- **Gate-gated promotion.** Track 0.2 ensures an un-gated graph can never become the label of
  record, even if a config flag says so.

---

## 15. Glossary

- **VAAMR / PURER** — the participant / therapist classification frameworks (§1).
- **Cue block** — the run of therapist turns between two consecutive participant turns
  (FROM→CUE→TO); the unit of influence and of the MindfulBERT training examples.
- **Δprogression** — signed change in the participant's VAAMR progression coordinate
  (E[stage]) from the participant turn before a cue to the one after.
- **Reliability gate** — out-of-sample, per-class κ + rare-stage recall floor that licenses
  LLM-free scaling (`validation.py`).
- **Distillation fidelity** vs **independent quality** — GNN↔LLM agreement (expected high; the
  student echoing the teacher) vs GNN↔human agreement (the load-bearing validity axis).
- **Model-counterfactual** — swapping a node feature (the therapist cue) and measuring the
  model's predicted change; sensitivity analysis, not causation.
- **Superposition** — the soft VAAMR mixture over stages that the majority vote discards.

---

## 16. Status dashboard

| Item | Status |
|------|--------|
| Phase 0 — integrity & doc honesty (G5) | ✅ complete |
| Phase 1 — independence pass (G1) | ✅ complete |
| Phase 2 — anchors + human-axis ablation (G2) | ✅ complete |
| Track 0.1 — baseline triage & fixes | ✅ complete (green unit baseline: 3040 tests) |
| Track 0.2 — gate-gated promotion | ✅ complete |
| Track A — scalable label engine (A1–A5) | ✅ done |
| GPU preference (D11 / §6a) — audit + fixes | ✅ done (device knob → embeddings, VRAM hygiene, checkpoint/seed) |
| Track B — progression analysis (B1–B5) | ✅ done (B1/B2 in `mechanism.py`; B3/B4/B5 in `influence.py`, gate-gated) |
| Track C — MindfulBERT training-set builder (C1–C5) | ✅ done (`mindfulbert_dataset.py`; observed labels + gate-gated augmentation + C4 harness) |
| Track D — subtext communities (D1–D5) | ✅ done (`communities.py`; two-algorithm partition + stability selection) |

**All planned tracks complete.** Track B/C/D add 27 unit tests
(`test_gnn_influence.py` ×10, `test_mindfulbert_dataset.py` ×8, `test_gnn_communities.py` ×9).
**Unit baseline: 3136 tests, 0 failures, 0 errors (9 skipped)** via `tests/run_unit_tests.py`
(was 3109; +27). `run_gnn_analysis` was exercised end-to-end (embedding-patched): the gate
correctly SUPPRESSES counterfactual influence until it passes, and the influence + MindfulBERT
augmentation paths run cleanly when the gate is satisfied.

**Remaining (downstream / out of this plan's scope):** corpus checkpoints on real data — run the
GNN with `precipitates_edges` (A1), then once the reliability gate reports `ready_for_scaling`,
enable `counterfactual` (B) + `build_mindfulbert_dataset`/`augmentation_enabled` (C) and read the
triangulation (B4) + the C4 ablation verdict; ROADMAP Phase 6 fine-tunes MindfulBERT on the
exported dataset.

### Track 0.2 — as built

The reliability gate now persists a machine-readable verdict and promotion is gated on it:

- `gnn_layer/validation.py`: `write_gate_verdict()` writes
  `03_analysis_data/gnn/gnn_gate.json` (`ready_for_scaling`, `vaamr_ready`, `purer_ready`,
  `vaamr_kappa`, `purer_kappa`, `rare_stage_ok`, `rare_stage_notes`, `irr_target`,
  `timestamp`); `read_gate_verdict()` / `gate_ready_for_scaling()` read it (missing or
  unreadable ⇒ not ready). Wired into `gnn_layer/runner.py` right after the validation
  report/CSV.
- `process/assembly/master_dataset.py`: new `gate_passed` arg (**default False**). The
  `gnn_consensus` tier engages only when `gnn_authoritative AND gate_passed` — enforced at
  the assembly boundary, so a config flag alone can never promote an un-gated graph even if a
  caller forgets to thread the verdict.
- `process/orchestrator.py`: `_gnn_promotion_flags(config, output_dir)` computes
  `(gnn_authoritative, gate_passed)` by reading the persisted verdict; all 3 assembly sites
  use it and log a clear warning when the operator opted in but the gate has not passed.
- Tests: gate persistence/read-back + missing/failing verdict (test_gnn_consensus); un-gated
  promotion blocked at the assembly boundary (test_gnn_consensus, test_methodology_provenance);
  orchestrator forwards `gate_passed=False` with no verdict on disk + signature contract
  (test_orchestrator_gnn_wiring).

> **Note on timing.** The GNN runs at analyze-time (Stage 8), *after* assembly (Stage 6), so a
> gate verdict from a prior run is what licenses promotion on a subsequent run — promotion is
> inherently a cross-run safeguard, which is exactly the intended LLM-free-scaling workflow.

### Tracks B/C/D — as built (this session)

All three remaining tracks are now implemented, wired, and unit-tested:

- **Track B (`gnn_layer/influence.py`).** Model-counterfactual influence: per-PURER-move
  centroid swaps + a null baseline, re-forwarded per cue block to read the shift in the
  FOLLOWING participant's predicted progression coordinate; participant-clustered bootstrap CIs;
  Spearman/sign-agreement triangulation against `mechanism.py`'s observed Δprogression (B4) with
  per-move divergence flags; session-phase subgroup sidecar (B5). Runner-GATED on
  `validation.gate_ready_for_scaling`. B1/B2 (the observed-Δprogression LEAD) were already in
  `analysis/mechanism.py`.
- **Track C (`process/assembly/mindfulbert_dataset.py`).** The end-goal dataset builder: cue-block
  examples labelled by observed Δprogression with weakest-endpoint provenance tiers + abstention +
  gate verdict (C1/C2); a separate, gate-gated, provenance-tagged GNN-counterfactual augmentation
  channel (C3) retained only if a participant-grouped held-out proxy ablation clears
  `augmentation_min_gain` (C4); versioned JSONL + datasheet with the n≈32 caveats (C5). Wired into
  `analysis/runner.py` §12b.
- **Track D (`gnn_layer/communities.py`).** Thresholded subtext-similarity graph (D1), Louvain +
  agglomerative two-algorithm partition with ARI agreement (D2), within-session
  community→community routine transitions (D3), participant-bootstrap stability selection that
  suppresses/flags fragile communities (D4), and TF-IDF naming + per-session prevalence +
  cross-cohort drift (D5). Gate-independent discovery; wired into `runner.py`.

**Downstream (out of scope here):** run the corpus checkpoints on real data — A1 precipitates,
then (once the gate reports `ready_for_scaling`) `counterfactual` (B) and
`build_mindfulbert_dataset`/`augmentation_enabled` (C); ROADMAP Phase 6 fine-tunes MindfulBERT on
the exported dataset.
