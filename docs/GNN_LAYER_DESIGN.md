# GNN Representation-and-Discovery Layer for QRA — Design

**Status:** Fully implemented and integrated (see `docs/GNN_IMPLEMENTATION.md` for module-by-module specification)
**Scope:** An analysis layer for the Qualitative Research Algorithm that reuses *methods* from CFiCS (graph neural
networks, supervised contrastive learning, inductive node attachment) — and CFiCS's microcounseling-skills label set — to
close three structural limits of the current LLM-label pipeline. It **augments, never replaces** the existing VAAMR /
VCE / PURER classifiers.
**Substrate:** the `Qwen3-Embedding-8B` vectors QRA already computes for segmentation and VCE embedding classification.
**Activation:** `config.gnn_layer.enabled=True` or `qra analyze --gnn`. Disabled by default; runs at analyze-time only.
All GNN artifacts go to `03_analysis_data/gnn/`, `06_reports/`, and `02_meta/gnn/model/` — frozen segments and
`master_segments` are never mutated.

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

## 4. Capabilities (all implemented; all directional until human validation catches up)

### A — Continuous VAAMR positioning (Implemented — `gnn_layer/inference.py`)
The soft-VAAMR head is trained against the multi-run ballot distribution stored per segment, upgraded to human labels
on the validated 20% sample. Output per participant segment: a **stage-mixture vector** over the 5 stages and a
**continuous progression coordinate** along the re-habituation arc. A FROM → TO transition becomes a **vector** in
embedding space (direction + magnitude), not a discrete jump; a segment that is "0.6 Avoidance / 0.4 Attention
Regulation — on the cusp" becomes a first-class, clinically meaningful object. When enabled, these GNN-derived mixtures
are the primary source for `analysis/superposition.py`, feeding all downstream mechanism and efficacy inference with
partial-progression resolution the argmax cannot express. Output: `03_analysis_data/gnn/segment_positions.csv`.

### B — Cue granularization + emergent-motif discovery / flagship (Implemented — `gnn_layer/motifs.py`)
Each therapist **cue block** is mean-pooled from its constituent GNN segment embeddings. Then:
1. **Cluster** cue embeddings → emergent therapist-language **motifs**, finer than (and orthogonal to) the 5 PURER moves.
2. **Regress the transition outcome on the cue embedding** — `P(forward | cue embedding, from_stage)` (from-stage-conditioned
   logistic regression preserving the §7.6 stage-moderation hypothesis) — to **score each motif's influence**.
3. **Flag high-influence / low-PURER-purity / low-microskill-purity motifs** — therapist language that reliably precedes
   participant progression but maps to *no* existing label. Surfaced with exemplar cues as candidate new constructs for human review.

The per-block motif assignment sidecar (`cue_block_assignments.csv`) feeds the `analysis/mechanism.py` Δprogression
aggregation by emergent motif. Outputs: `cue_motifs.csv`, `report_gnn_emergent_motifs.txt`.

### C — Independent (non-LLM) measurement substrate (Implemented — `gnn_layer/gnn_lift.py`, `gnn_layer/triangulation.py`)
GNN-derived VAAMR stage assignments are compared against LLM-ensemble and human-coded labels via Cohen's κ
(`report_gnn_triangulation.txt`). (VAAMR stage × VCE code), (PURER × VAAMR transition), and **(PURER × microskill)**
lift tables are recomputed from GNN geometry and placed beside the LLM-derived tables.
**GNN ↔ LLM convergence is stronger evidence than LLM ↔ LLM**, directly complementing the planned shuffled-stage
permutation control. For the independence claim, the `human` and `self_supervised` `label_mode` variants are reported
separately. Outputs: `gnn_vs_llm_lift.csv`, `purer_microskill_lift.csv`, `gnn_head_predictions.csv`.

### D — Principled ablation (Implemented — `gnn_layer/ablation.py`, opt-in via `run_gnn_ablation=True`)
The single graph makes prose-level design questions empirically testable by removing construct heads and measuring loss
delta:
- Ablate VCE head → **is VCE superfluous?** (does progression prediction degrade without it?)
- Ablate PURER head or microskill head → do they carry signal beyond the emergent motifs?

Output: `gnn_construct_signal.csv`, `report_gnn_construct_signal.txt`.

### E — Inductive participant↔therapist coupling (Implemented — `gnn_layer/coupling.py`)
NMF/PCA extracts latent factors of therapist cue language and their correlation with subsequent participant VAAMR forward
movement. Factors are named post-hoc against an inline CF/IC alliance-concept reference lexicon — rediscovering common
factors inductively rather than imposing them. The coupling-factor runner now passes exemplar therapist texts to
`interpret_factors()` so naming is grounded in actual session language. Outputs: `coupling_factors.csv`,
`report_gnn_coupling.txt`.

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

## 6. Concrete QRA touchpoints (implemented)

- **`gnn_layer/` package** (14 modules) — invoked from `analysis/runner.py` after PURER analysis, gated by
  `config.gnn_layer.enabled`. Reads `master_segments` DataFrame and writes to `03_analysis_data/gnn/`, `06_reports/`,
  and `02_meta/gnn/`. Frozen segments and `master_segments` are never mutated.
- **GNN artifacts** (no new `master_segments` fields — GNN outputs live in separate CSV files keyed by `segment_id`):
  `segment_positions.csv`, `cue_motifs.csv`, `cue_block_assignments.csv`, `gnn_head_predictions.csv`,
  `gnn_vs_llm_lift.csv`, `purer_microskill_lift.csv`, `coupling_factors.csv`, `gnn_construct_signal.csv` under
  `03_analysis_data/gnn/`; `report_gnn_emergent_motifs.txt`, `report_gnn_triangulation.txt`,
  `report_gnn_coupling.txt`, `report_gnn_construct_signal.txt` under `06_reports/`.
- **`MICROCOUNSELING_CODEBOOK.md`** — hot-reloadable 8-code microcounseling codebook, parsed by
  `codebook/microcounseling_codebook.py` (mirror of `phenomenology_codebook.py`). Seeds the microskill head via
  Qwen3-re-embedded exemplars.
- **No extra dependency** — GraphSAGE implemented in pure PyTorch (`gnn_layer/model.py`). torch-geometric is
  explicitly not required (avoids torch-scatter/torch-sparse compile fragility on torch 2.11).
- **Config:** `GnnLayerConfig` and `SuperpositionConfig` nested dataclasses in `process/config.py`, registered in
  `from_json.sub_config_map` and `setup_wizard.build_config_from_wizard_data`. `analyze --gnn` flag in `qra.py`
  overrides `config.gnn_layer.enabled` without modifying the config file.

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
