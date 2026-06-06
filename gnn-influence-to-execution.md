# GNN: Influence-to-Execution — Design Lineage vs. As-Built Reality

**Purpose.** This document is a decision aid, not a change. It lays out, honestly and with
`file:line` evidence, where the QRA Graph Neural Network layer's *design* (in
`docs/GNN_LAYER_DESIGN.md`, `docs/GNN_IMPLEMENTATION.md`, and the `ROADMAP.md` Phase 3/6
narrative) diverges from what the code in `gnn_layer/` *actually does today*, weighs the
pros and cons of the current implementation, and frames the choice of how much of the
drifted-from design to build. It deliberately does **not** make the architecture call —
that is the reader's decision. It is written as of the state **after** the microcounseling
("microskill") feature was fully removed from the codebase.

---

## 1. Executive summary

The QRA GNN was adapted from CFiCS (Schmidt, Hammerfald, Jahren & Vlassov, 2025, CLPsych).
The design documents describe a **heterogeneous** graph — construct *anchor* nodes
(VAAMR / PURER / VCE), cross-framework lift-weighted edges, a corpus-wide *subtext
similarity graph* with community detection, and participant / session / therapist nodes —
serving a multi-task model that ultimately supports causal-influence and counterfactual
manuscript outputs (ROADMAP Phase 3.7–3.8, 6.4).

The **as-built** layer is a **homogeneous** projection: the only materialized nodes are
participant/therapist *segments*, connected by within-session temporal-chain edges and
kNN-similarity edges over Qwen3 embeddings. Anchor nodes and cross-framework edges are
accepted as *parameters* by the builder but are **never supplied** by the runner; the
subtext similarity graph + community detection does not exist (cue-block KMeans is a
different, smaller thing); the 182 CFiCS skill exemplars are never loaded anywhere.

The homogeneous build **fully serves** the layer's live mission — continuous VAAMR
superposition, emergent cue-motif discovery, GNN↔LLM↔human triangulation, construct-signal
ablation over VCE/PURER, participant↔therapist coupling factors, and the
consensus-distillation classifier with its out-of-sample reliability gate. It **cannot**
produce the roadmap's heterogeneity-dependent deliverables (learned `precipitates` causal
edge weights, counterfactual simulation, the subtext community catalog) without new work.

The core decision: **stay homogeneous and make the docs honest**, **add heterogeneity
incrementally behind the existing flags**, or **build the full roadmap design**.

---

## 2. Lineage: what was taken from CFiCS

Reused as architectural *patterns*, reimplemented QRA-native in pure PyTorch (no
torch-geometric):

- **GraphSAGE message passing** — `gnn_layer/model.py` (`SAGEConv`, `MultiTaskGNN`,
  mean aggregation via `scatter_weighted_mean` / `index_add_`).
- **Supervised-contrastive loss** (InfoNCE over the ordered VAAMR arc) — `model.py:supervised_contrastive`.
- **Inductive node attachment** — `graph_builder.attach_new_segments` (the basis for
  LLM-free scale-mode classification).

Deliberately **not** taken: ClinicalBERT (QRA uses Qwen3-Embedding-8B, reused from
segmentation/VCE), the transductive 182-node data loader, CF/IC as *imposed* labels
(kept only as an interpretive lexicon in `coupling.py`), and wandb.

**Important post-removal note.** The one *concrete* asset borrowed from CFiCS was its
8-skill **microcounseling** label set (+ 182 exemplars). That feature has now been removed
in full. The GNN therefore no longer borrows any concrete CFiCS *data* — only the three
architecture patterns above. The CFiCS citation remains valid as the GraphSAGE lineage;
it should no longer be described as supplying a "microcounseling-skills axis."

---

## 3. As-designed vs. as-built

| Design element | Designed in | Built? | Evidence |
|---|---|---|---|
| Homogeneous segment graph (temporal + kNN) | DESIGN/IMPL | **Built** | `graph_builder.build_graph` materializes segment nodes + `temporal` + `knn` edges |
| Construct **anchor nodes** (VAAMR/PURER/VCE) | DESIGN §; methodology | **Design-only** | `build_graph(anchor_features=…, anchor_edges=…)` are params the runner never passes (`runner.run_gnn_analysis` calls `build_graph(df_all, seg_emb, config, framework=…)` with no anchors) |
| **Cross-framework** lift-weighted edges | DESIGN; ROADMAP 3.2 | **Design-only** | `graph_builder.compute_cross_framework_lift` exists but is **never called**; no anchor edges are ever added |
| **Subtext similarity graph** (corpus-wide) | ROADMAP 3.3 | **Not built** | kNN edges exist within `build_graph`, but there is no separate thresholded subtext graph |
| **Community detection** (Leiden/Louvain) | ROADMAP 3.3 | **Not built** | `motifs.cluster_cue_motifs` does KMeans over *cue-block* embeddings — a different, smaller construct |
| **182 CFiCS exemplars** re-embedded as anchors | DESIGN | **Not built** | No loader anywhere references the exemplar CSV |
| **Participant / session / therapist** nodes | ROADMAP 3.2 | **Not built** | Only `participant_segment` / `therapist_segment` node types exist |
| Multi-task heads (soft-VAAMR, progression, VCE, PURER) | DESIGN/IMPL | **Built** | `model.build_model` head set; VCE optional/off by default |
| Supervised-contrastive + link-prediction losses | DESIGN/IMPL | **Built** | `model.compute_losses` |
| Inductive attachment for new data | DESIGN; ROADMAP 6.2 | **Built (now wired)** | `graph_builder.attach_new_segments` + `runner.run_gnn_classify` (see §5) |
| Reliability gate (out-of-sample κ) | DESIGN/IMPL | **Built** | `train.crossval_predictions` → `validation.evaluate_crossval` |
| Construct-signal ablation | IMPL | **Built (VCE/PURER)** | `ablation.run_ablation`; microskill arm removed |
| Coupling / CF-IC factors | DESIGN/IMPL | **Built** | `coupling.extract_latent_factors` + interpret against inline CF/IC lexicon |

**Honest one-liner:** the runtime graph is homogeneous; most of the documented
heterogeneity is design-only.

---

## 4. Capabilities A–E, as they actually run

- **A — Continuous VAAMR positioning.** Live. Soft-label KL targets from multi-run ballots
  → per-segment stage mixture + progression coordinate (`inference.infer_segment_positions`).
- **B — Cue-motif discovery (flagship).** Live, but operates on KMeans of cue-block
  embeddings, not a corpus subtext community graph. Emergent-motif flagging now keys on
  **PURER purity only** (the microskill purity arm was removed) — i.e. "influential but not
  cleanly explained by a single PURER move." This is a deliberate, documented semantic
  change: motifs formerly rescued by high microskill purity will now flag.
- **C — Independent substrate / triangulation.** Live (`triangulation.py`, `gnn_lift.py`).
  Note the "independence" is embedding-geometry-based and, in the default `weak` label
  mode, the heads are trained on LLM ballots — the triangulation report says so. The
  `purer_microskill_lift` table was removed; VAAMR×VCE GNN-vs-LLM lift remains.
- **D — Construct ablation.** Live over **VCE and PURER** heads (microskill arm removed).
  `discover_subtypes` is defined but never called.
- **E — Coupling factors.** Live; PCA/NMF latent factors of therapist cue language,
  named post-hoc against an inline CF/IC lexicon (discovered, not imposed).

---

## 5. Documented drift & cleanup debt (catalog for the decision, not yet actioned)

These were left in place deliberately so the architecture call below can be made first.

1. **Stale "SCAFFOLD" docstring.** `gnn_layer/__init__.py` still says the package is a
   "SCAFFOLD" with "NotImplementedError bodies." Every module is fully implemented. False.
2. **`enabled` default conflict — now resolved in favor of ON.** The code default is
   `GnnLayerConfig.enabled = True` and `analysis/runner.py` honors it, matching
   `CLAUDE.md` ("ON by default"). The `config.py` module docstring and parts of the GNN
   docs still imply OFF. The test suite assertion was updated to match the ON default.
3. **Vestigial config flags never read by the engine:** `include_vce_nodes`,
   `include_purer_nodes`, `cross_framework_min_lift`, `run_on_participants`,
   `run_on_therapists`. (`run_on_*` are even prompted by the setup wizard but ignored.)
   They encode *intent* for the heterogeneous design but currently control nothing.
4. **Dead code:** `ablation.discover_subtypes` and `graph_builder.compute_cross_framework_lift`
   are defined and never called.
5. **`attach_new_segments` node-typing bug — FIXED in this pass.** It previously tagged
   attached nodes `'segment'` (not `participant_segment`/`therapist_segment`), so the
   head-prediction router would have skipped every newly attached node — scale-mode would
   have classified **zero** segments. It now accepts `node_type_of` and `run_gnn_classify`
   persists/loads the trained graph (`graph_builder.save_graph`/`load_graph`) and attaches
   inductively instead of rebuilding from scratch.

If the decision is to **stay homogeneous**, items 1–4 should be cleaned (delete the dead
code and vestigial flags, fix the docstrings). If the decision is to **build
heterogeneity**, items 3–4 are the scaffolding to build *onto*, and should be kept and
wired rather than deleted.

---

## 6. Pros and cons of the current homogeneous implementation

**Pros**
- **Simplicity & portability.** Pure PyTorch, no torch-geometric; trains on CPU/one GPU.
- **One embedding substrate.** Reuses Qwen3 already computed for segmentation/VCE — no
  second model, no duplicated cost.
- **Sufficient for the mission.** Delivers A–E and the consensus-distillation classifier +
  reliability gate, which is what the methodology paper and scale-mode goal actually need.
- **Low maintenance surface.** Few moving parts; easy to reason about and test.

**Cons**
- **No explicit relational/construct structure.** There are no typed `precipitates` edges,
  no construct anchors, no cross-framework edges — so the model cannot *learn* the
  therapist-move → participant-transition influence structure the roadmap's headline
  results assume; those are computed by separate lift statistics, not by the graph.
- **No cross-session / longitudinal graph structure.** Temporal edges are within-session
  only; participant trajectory across sessions is not in the graph.
- **No subtext community map.** The roadmap's "subtext community catalog" and cross-cohort
  novelty detection have no implementation to stand on.
- **Frozen embeddings.** No end-to-end fine-tuning of the encoder (intentional, but caps
  expressivity).

---

## 7. What building more would gain or lose (element by element)

| Increment | Unlocks (ROADMAP) | Scientific value | Effort | Added complexity / cost |
|---|---|---|---|---|
| **Construct anchor nodes** (VAAMR/PURER/VCE) wired via existing `anchor_features` param | 3.2; makes Capability-D ablation structural | Medium — lets the graph *represent* constructs, not just predict them | **S–M** (compute anchor embeddings from framework defs; pass to `build_graph`) | Modest; flags become live |
| **Cross-framework edges** via `compute_cross_framework_lift` (already written) | 3.2; "co-occurs_with" structure | Medium | **S** (call the existing fn; add anchor edges) | Low |
| **Typed `precipitates` edges** (therapist→next participant) + edge attention | 3.7 causal influence matrix; 3.8 counterfactual simulation | **High** — the headline mechanistic manuscript output | **L** (new edge family + attention readout + attribution) | High; new validation burden |
| **Subtext similarity graph + community detection** | 3.3 subtext community catalog; cross-cohort novelty | High (novel qualitative findings) | **M–L** (O(n²) similarity + Leiden/Louvain + reporting) | Medium; new artifacts + interpretation |
| **Participant/session nodes** | 3.2; longitudinal readout; 6.7 outcome prediction | Medium–High | **M** | Medium |

Across all increments, the recurring *loss* is maintenance surface and the risk of
out-running the validation the methodology demands: every structural claim the graph makes
needs its own reliability/triangulation check, or it becomes an unfalsifiable elaboration.

---

## 8. Recommendation framework (the reader chooses)

**Path A — Stay homogeneous, make the docs honest (lowest effort).**
Scope: clean items 1–4 of §5 (delete dead code + vestigial flags, fix the scaffold/`enabled`
docstrings, reconcile `docs/GNN_*` to describe the homogeneous build, move the heterogeneous/
subtext design under an explicit "future work" heading in ROADMAP Phase 3). Unlocks: nothing
new, but the code and the paper stop disagreeing, and the layer fully serves the methodology
paper + scale-mode. Risk: the roadmap's causal/counterfactual/subtext deliverables remain
unbuilt and must be acknowledged as such.

**Path B — Incremental heterogeneity behind the existing flags (measured).**
Scope: wire **anchor nodes** + **cross-framework edges** first (both are S–M and the code
is largely present), gate them on `include_vce_nodes`/`include_purer_nodes`, and use the
existing Capability-D ablation to *measure* whether the added structure earns its keep
before going further. Unlocks: a principled path toward 3.2 and a structural basis for D.
Risk: moderate; keeps complexity proportional to demonstrated value.

**Path C — Full roadmap build (highest value, highest cost).**
Scope: anchors + cross-framework edges + typed `precipitates` edges with attention +
subtext similarity graph/community detection + participant/session nodes. Unlocks: the
GNN manuscript's headline outputs (causal influence matrix, counterfactual simulation,
subtext community catalog — ROADMAP 3.7–3.8, 3.3). Risk: large effort, a real torch-
geometric-vs-pure-PyTorch decision at scale, and a substantial new validation burden.

**Suggested decision procedure (not a decision):** if the near-term goal is the
*methodology* paper and LLM-free scaling, Path A is sufficient and should be done now. If
the *GNN manuscript* (Phase 3 deliverables) is the target, Path B is the low-regret first
step — build anchors + cross-framework edges, run the ablation, and let the measured signal
decide whether Path C is warranted.

---

*Companion code references: `gnn_layer/{__init__,config,runner,graph_builder,model,train,
inference,motifs,gnn_lift,coupling,ablation,validation,triangulation,reports,embeddings,
soft_labels}.py`. Design references: `docs/GNN_LAYER_DESIGN.md`, `docs/GNN_IMPLEMENTATION.md`,
`ROADMAP.md` (Phase 3, Phase 6), `methodology.md` (GNN section).*
