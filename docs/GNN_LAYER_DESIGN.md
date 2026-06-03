# GNN Representation-and-Discovery Layer for QRA — Design

**Status:** Design proposal (no implementation in this change)
**Scope:** A new analysis layer for the Qualitative Research Algorithm that reuses *methods* from CFiCS (graph neural
networks, supervised contrastive learning, inductive node attachment) — and CFiCS's microcounseling-skills label set — to
close three structural limits of the current LLM-label pipeline. It **augments, never replaces** the existing VAAMR /
VCE / PURER classifiers.
**Substrate:** the `Qwen3-Embedding-8B` vectors QRA already computes for segmentation and VCE embedding classification.

---

## 1. Motivation: three gaps the LLM-label pipeline cannot close

QRA's methodology (`methodology.md`) is candid that VAAMR LLM-consensus classification already reaches **human-level
inter-rater reliability** (§5.4) and that a CFiCS-style GNN is "**not the bottleneck**" *for classification* (§8.5). We
agree. This layer is **not a better classifier**. It targets three things the LLM-label approach cannot do, each of which
the methodology itself flags as a limitation:

1. **Superposition is discarded.** `final_label` is a hard argmax, and `analysis/purer_analysis.py` builds every
   transition from single integer stages (`from_stage`/`to_stage`). Yet §9 and the §5.4 validation thresholds
   acknowledge participant utterances are "more mixed-stage than the framework anticipated." The soft signal *already
   exists and is thrown away* at majority vote: the multi-run ballots, `secondary_stage`, `agreement_fraction`, and
   `rater_votes` on the `Segment` dataclass (`classification_tools/data_structures.py`).

2. **The cue is collapsed to one of five labels before influence is measured.**
   `compute_purer_transition_influence` (`analysis/purer_analysis.py:232`) reduces each therapist cue block to
   `dominant_purer ∈ {P,U,R,E,R2}` (`:124`, `:286`, `:295`) and only then computes lift. Every bit of continuous signal
   in the therapist's actual language is gone before mechanism analysis begins. The "average cue" the cue-response report
   shows is LLM prose, not a measurable feature.

3. **Both convergence measures are LLM-on-LLM.** §5.2 names this the central construct-validity threat: because VAAMR
   and VCE are both LLM classifiers over the same text, their lift co-occurrence may reflect shared training-data
   patterns rather than phenomenological structure. The pipeline has no measurement substrate independent of the LLM.

The clinical mission these serve is unchanged: the mechanistic **FROM → CUE → TO** analysis that drives **between-cohort
protocol adaptation** for Move-MORE.

---

## 2. The symmetric construct design

We reuse CFiCS's *patterns* and its **microcounseling-skills axis only** — not its abstract Common-Factors (CF) or
Intervention-Concepts (IC) constructs. This produces a clean parallel structure across the therapeutic dyad:

| Side | Primary framework (validated / ordinal) | Sub-layer codebook (granular, multi-label) | Sub-layer role |
|---|---|---|---|
| **Participant** | VAAMR — 5 stages, ordered re-habituation arc | VCE — 59 phenomenology codes | content: *what* the stage looks like; cross-validate + subtype |
| **Therapist** | PURER — 5 rhetorical inquiry moves | **Microcounseling Skills — 8 (RL, G, V, A, RA, AP, OQ, Neutral)** | behavior: *how* the move is executed; cross-validate + subtype |

The 8 microskills are concrete, surface-observable therapist behaviors — unlike CF/IC, which are inferred relational
states. They therefore (a) **subtype PURER** (a Phenomenology move executed *via* Open-ended Question + Reflective
Listening; a Reinforcement move *via* Affirmation/Validation), and (b) give a **skills × PURER** cross-validation that
exactly mirrors the existing VAAMR × VCE lift.

This reframing also answers the standing "**is VCE superfluous?**" question: VCE gains a structural twin (Skills), and
both are recast as *behavioral/content sub-layers that subtype a primary framework* rather than flat parallel
classifiers. Whether VCE actually earns its place becomes an empirical question we can answer by ablation (§4, Capability
D).

**CF/IC are dropped as imposed labels.** They survive only as an *interpretive reference lexicon* for any latent
alliance-like factor the GNN discovers inductively (§4, Capability E). We do not impose Bordin's alliance triad on the
data; we let the graph tell us whether something like it emerges.

**Seed data.** CFiCS ships 182 skill-labeled counseling exemplars (`data/htc_examples_ids_with_pure_skills.csv`). We
**re-embed these with Qwen3** and use them as weak-supervision **anchors/exemplars** for the skills head — a small but
real labeled seed, never the final arbiter (domain shift from curated counseling text to Move-MORE therapist speech is
acknowledged and must be validated).

---

## 3. Architecture: one heterogeneous graph on Qwen3 embeddings

The layer builds a single heterogeneous graph that **represents all constructs but classifies none as primary**.

**Node types**
- `participant_segment`, `therapist_segment` — features = Qwen3-Embedding-8B vectors (reused from segmentation/VCE).
- Anchor nodes: 5 `vaamr_stage`, 59 `vce_code`, 5 `purer_move`, **8 `microskill`** (features = Qwen3 embeddings of their
  definitions/exemplars). Optional `session` / `participant` nodes for longitudinal structure.

**Edges**
- **Temporal chain** — `segment → next_segment` within a session. This encodes the FROM → CUE → TO structure *in the
  graph itself* and is the substrate for the self-supervised discovery objective.
- **Anchor/label edges** — `segment → assigned construct`, weighted by classification confidence / ballot fraction (soft).
- **Similarity edges** — kNN among segments in Qwen3 space; the basis for **inductive** attachment of new/unseen
  segments (new cohorts, new sessions) without retraining.
- **Cross-framework edges** — `vaamr_stage ↔ vce_code` and `purer_move ↔ microskill`, weighted by empirical lift; makes
  the lift structure learnable and ablatable.

**Backbone & features**
- GraphSAGE (inductive-friendly; CFiCS's default), reimplemented QRA-native. Node features are Qwen3 embeddings; no
  second embedding model is introduced.

**Training objectives (CFiCS multi-task + contrastive patterns)**
- **Soft-VAAMR head** — KL divergence to the multi-run **ballot distribution** (not cross-entropy to the argmax); a
  scalar **progression-coordinate** regression recovers the ordered 0→4 manifold.
- **Multi-label VCE head** (BCE, participant nodes).
- **PURER head** + **multi-label microskills head** (BCE, therapist nodes; seeded by the re-embedded CFiCS exemplars).
- **Supervised contrastive loss** — organizes the embedding space by stage/code/move/skill; recovers the ordered arc.
- **Self-supervised next-segment / link-prediction** on the temporal chain — the label-free engine behind Capabilities
  B, C, and E.

**What we deliberately leave in CFiCS:** the CF/IC codebook (kept only as interpretive lexicon), ClinicalBERT, the
transductive 182-node setup, the broken `example_data.py` / `predict_new_text` paths, and wandb. The patterns are
reimplemented fresh and QRA-native; only the **skills label set + its 182 exemplars** (re-embedded on Qwen3) are imported.

---

## 4. Capabilities (all centered; all run now on Cohort 1–2 weak labels, directional until validated)

### A — Continuous VAAMR positioning (model the superposition)
Train the soft-VAAMR head against the multi-run ballot distribution already stored per segment, upgraded to human labels
on the validated 20% sample. Output per participant segment: a **stage-mixture vector** over the 5 stages and a
**continuous progression coordinate** along the re-habituation arc. A FROM → TO transition becomes a **vector** in
embedding space (direction + magnitude), not a discrete jump; a segment that is "0.6 Avoidance / 0.4 Attention
Regulation — on the cusp" becomes a first-class, clinically meaningful object. Feeds the longitudinal trajectories and
the Avoidance-barrier analysis (§6.3) with partial-progression resolution the argmax cannot express.

### B — Cue granularization + emergent-motif discovery (flagship)
Embed each therapist **cue block** (reuse the cue-block construction in `compute_cue_block_purer_profiles:131`). Then:
1. **Cluster** cue embeddings → emergent therapist-language **motifs**, finer than (and orthogonal to) the 5 PURER moves.
2. **Regress the transition outcome on the cue embedding** — model P(forward transition | cue embedding, from_stage),
   conditioned on `from_stage` to preserve the §7.6 stage-moderation hypothesis — to **score each motif's influence**.
3. **Flag high-influence / low-PURER-purity / low-skill-purity motifs** — therapist language that reliably precedes
   participant progression but maps to *no* existing label. These are surfaced with exemplar cues as candidate new
   constructs for human review.

Output: per-cue-block embedding + motif id + influence score; a ranked "emergent influential motifs" report; exemplar
cues per (from → to) transition drawn by **geometry** rather than LLM averaging. This turns the cue-response analysis
from *descriptive* to *predictive/mechanistic* — the deepest realization of the FROM:CUE:TO mission, and the most
directly actionable input to the §6.3 curriculum-modification format (observation → mechanism → change → assessment).

### C — Independent (non-LLM) measurement substrate
Train on Qwen3 **geometry** + human-validated labels, **and** a **self-supervised variant** using only the temporal-chain
link-prediction objective (no LLM labels). Recompute the (VAAMR stage × VCE code), (PURER × VAAMR transition), and
**(PURER × microskill)** lift from GNN-derived assignments, and present them beside the LLM-derived tables.
**GNN ↔ LLM convergence is stronger evidence than LLM ↔ LLM**, directly complementing the planned shuffled-stage
permutation control (§8.2). For the independence claim to be clean, the human-only and self-supervised variants are
reported separately so convergence is not circular.

### D — Principled ablation & sub-typing (byproducts, not extra build)
The single graph makes prose-level design questions empirically testable:
- Ablate VCE nodes/edges → **is VCE superfluous?** (does superposition / transition-prediction degrade without it?)
- Ablate PURER-move distinctions or skills → do they carry signal beyond the emergent motifs?
- Cluster within each VAAMR stage and within each PURER move → **emergent sub-types**. Skills give PURER a ready
  behavioral sub-axis (e.g., Reframing-via-Reflective-Listening vs Reframing-via-Education).

### E — Inductive participant↔therapist coupling (CF/IC discovered, not imposed)
Learn the relationship between therapist-language representation (PURER + skills + chain position) and **subsequent
participant VAAMR movement** — the §7.6 context-dependent-therapist-effects / Varela mutual-constraints core. Test
whether **latent alliance-like factors emerge** (e.g., therapist Validation + Genuineness co-moving with participant
forward transitions). If they do, interpret them *post hoc against* the CFiCS CF/IC lexicon — rediscovering common
factors inductively rather than imposing them. This is the principled home for the dropped CF/IC constructs.

---

## 5. Honest tradeoffs and risks (Text-Psychometrics discipline)

- **Small N + run-now.** With ~32 participants, all outputs are **directional / hypothesis-generating** until human
  validation catches up (§5.4). The GNN never overrides LLM labels-of-record; curriculum use is framed as falsifiable
  hypotheses for the next cohort.
- **Weak-label circularity.** Training heads on LLM ballots inherits LLM bias. For the independence claim (Capability C),
  the human-only and self-supervised variants must be reported separately.
- **Discovered motifs / latent CF-IC factors are hypotheses.** Any motif or latent factor that informs curriculum must
  pass human review (the same content-validity / blind-coding discipline as VAAMR and PURER) before it becomes primary
  evidence (§5, §9.5).
- **Skills-seed domain shift.** CFiCS exemplars are curated counseling text; used only as Qwen3-re-embedded anchors, not
  arbiters. Skills labels must be validated on Move-MORE therapist speech.
- **No framework proliferation beyond the symmetric four.** Participant: VAAMR / VCE. Therapist: PURER / Skills. CF/IC are
  not added as labels.

---

## 6. Concrete QRA touchpoints (specification, not built here)

- **New analysis package** `analysis/gnn_layer/` — graph build, train/infer, motif discovery, GNN-lift, coupling —
  invoked from `analysis/runner.py` exactly as `purer_analysis.run_purer_analysis` is. It reads `master_segments` and
  writes to `03_analysis_data/` and `06_reports/`. Pure read of frozen segments + new output files preserves the
  frozen-segment / overlay invariants.
- **New `master_segments` fields** (additive, `getattr`-guarded in `process/assembly/master_dataset.py`, provenance
  `source='gnn_layer'`): `vaamr_mixture` (5-vector), `progression_coord` (float), `microskill_labels` (multi-label,
  therapist), `cue_motif_id`, `cue_influence_score`, `gnn_embedding_ref`.
- **New reports/CSVs** beside the PURER outputs (reuse the `export_purer_csvs` / `generate_purer_report` patterns):
  emergent-motif report, GNN-vs-LLM lift comparison, PURER × skills and VAAMR × VCE GNN-lift tables, continuous-trajectory
  exports, and the coupling / latent-factor report.
- **Microskills as a hot-reloadable framework** `MICROCOUNSELING_FRAMEWORK.md`, mirroring `VAAMR_FRAMEWORK.md` /
  `PURER_FRAMEWORK.md` and parsed by `theme_framework/markdown_loader.py`, seeded from the CFiCS skill definitions and
  exemplars; feeds the content-validity test sets like the other frameworks.
- **Dependency:** add `torch-geometric` to `requirements.txt` (or a minimal pure-PyTorch message-passing implementation
  to avoid the `torch-scatter`/`torch-sparse` compile fragility against torch 2.11). torch / transformers /
  sentence-transformers are already present.
- **Config:** a `GnnLayerConfig` and a `MicroskillClassifierConfig` nested dataclass in `process/config.py`, plus a
  `run_gnn_layer` flag. Exposed under the Stage 8 `analyze` command (`analyze --gnn`) — it is an analysis layer, not a
  Stage-3 classifier.

---

## 7. Positioning relative to the cohort schedule

Per the chosen direction, **all capabilities run now** on the Cohort 1–2 corpus using LLM consensus as weak labels, as
hypothesis-generation for between-cohort adaptation. Outputs are explicitly **directional** (per the §5.4 evidence tiers)
until human validation catches up; the independence claim (Capability C) and any primary-evidence use upgrade in strength
as the human-validated corpus grows through the final cohort. This is consistent with §8.5's principle that the GNN is
built on the labeled corpus the trial produces — we simply begin extracting exploratory value immediately rather than
waiting.

---

## 8. Relationship to CFiCS

See the companion document `docs/QRA_PATTERNS_REUSE.md` in the CFiCS repository, which records precisely which CFiCS
assets are reused (graph/SAGEConv model, supervised-contrastive loss, inductive node attachment, the 8-skill label set +
182 exemplars re-embedded on Qwen3) and which are not (CF/IC as imposed labels, ClinicalBERT, the transductive data
loader, the broken inference path), including the CF/IC-as-interpretive-lexicon note for Capability E.
