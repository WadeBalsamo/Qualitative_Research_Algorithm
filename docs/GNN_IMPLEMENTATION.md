# GNN Layer — Implementation Guide for QRA

**Audience:** a programmer implementing the GNN representation-and-discovery layer and the therapist-side
microcounseling codebook in QRA, **without access to the CFiCS repository**. Everything needed is either in this
repo or specified here.

**Companion docs:** `docs/GNN_LAYER_DESIGN.md` (the *why*: capabilities, methodology fit) and, in the CFiCS repo,
`docs/QRA_PATTERNS_REUSE.md` (what was/ wasn't borrowed). This document is the *how*.

---

## 0. Status: fully implemented

This guide originally accompanied a scaffold; the layer is now **fully implemented and tested** (synthetic
unit tests; the 16 GB embedding model is the only piece exercised lazily/at runtime).

| Component | State |
|---|---|
| `MICROCOUNSELING_CODEBOOK.md` (8 codes, VCE format) | **Built** — parses via `codebook/markdown_loader.load_codebook_md` |
| `codebook/microcounseling_codebook.py` (`get_microcounseling_codebook()`) | **Built** — mirror of `phenomenology_codebook.py` |
| Microcounseling **classifier stage** (Segment fields, `microskill` overlay, orchestrator `stage_classify_microskill` + `_microskill_classify`, master row, `classify --what microskill`, gating) | **Built** (Part I) |
| `gnn_layer/` package (14 modules) | **Built** — pure-PyTorch GraphSAGE, multi-task heads, soft-label assembly, motif discovery, GNN/LLM lift, ablation, coupling, runner |
| `process/config.py` `gnn_layer` + `microskill_*` fields + `sub_config_map`; `setup_wizard` carries both | **Built** (`enabled=False` default) |
| `analysis/runner.py` guarded hook → `run_gnn_analysis`; `qra analyze --gnn/--no-gnn` | **Built** (only fires when enabled) |
| Tests | `tests/test_gnn_layer.py` (8) + `tests/test_microskill_classify.py` (5), both green |

Because the runner hook is gated `enabled=False` and wrapped in try/except, **the existing pipeline and all current
tests are unaffected** until the layer is explicitly turned on. The GNN runs at analyze-time and writes only to
`03_analysis_data/gnn/`, `06_reports/`, and `02_meta/gnn/` — frozen segments and `master_segments` are never mutated.

**Implementation note vs. the original design:** the heterogeneous graph is realized as a *homogeneous projection* —
all nodes share the Qwen3 embedding space, so one shared SAGE aggregation over the typed-edge union is faithful and
needs no torch-geometric. Construct-anchor nodes are an optional hook (`graph_builder.build_graph(anchor_features=…)`)
that the runner leaves off by default; segment + temporal-chain + kNN edges drive Capabilities A/B/C/E.

> Note on environment: `requirements.txt` pins `torch==2.11.0`, `numpy==2.4.4`, `pandas==3.0.2`,
> `scikit-learn==1.8.0`, `sentence-transformers==5.4.0`, `networkx==3.6.1`. **No new dependency is required** — the
> GraphSAGE is implemented in pure PyTorch (Part II, `model.py`). torch-geometric is an *optional* drop-in, not needed.

---

## 1. Architecture & the replace-vs-augment decision

The GNN **augments** QRA; it does not replace any classifier.

VAAMR LLM-consensus already reaches human-level inter-rater reliability (`methodology.md` §5.4), and the VCE pipeline
(`codebook/embedding_classifier.py` + LLM + `codebook/ensemble.py`) is a working, cheap, deterministic multi-label
coder. On a ~32-participant feasibility trial a from-scratch GNN classifier is statistically fragile, so:

- **Decision: AUGMENT.** The GNN is (a) a *representation/analysis layer* and (b) an optional *third rater* for the
  codebook classifiers. The embedding+LLM ensemble remains the **labeler-of-record** for both VCE and microcounseling.
- **Gated replacement only.** `codebook/ensemble.py:CodebookEnsemble.reconcile` (line 34) already supports
  `preferred_method ∈ {llm, embedding, both}` and `require_agreement`. Extend it to accept a third method (`gnn`) so
  the GNN can slot in as a rater; promote it to primary **only if** it beats the embedding classifier on the
  human-validated set. Do not remove the embedding/LLM coders.

The layer runs at **analyze-time** (after `master_segments` assembly), exactly like `analysis/purer_analysis.py`. Its
per-segment outputs are written as **separate `03_analysis_data/gnn/` artifacts keyed by `segment_id`**, not folded
into `master_segments` (assembly runs before analyze; re-assembly is out of scope). It reads the assembled DataFrame
and never mutates frozen segments.

The construct topology is the symmetric pair (see `docs/GNN_LAYER_DESIGN.md`): **participant = VAAMR (primary) + VCE
(sub-layer)**; **therapist = PURER (primary) + microcounseling (sub-layer)**. A single heterogeneous graph represents
all four plus segment nodes and the temporal chain; it *represents* every construct but *classifies* none as primary.

---

## 2. Part I — The microcounseling classifier stage (VCE mirror)

The microcounseling codebook is the therapist-side twin of VCE. The classifier is a **near-verbatim clone of the
codebook stage**, changing only (a) the codebook object and (b) the speaker filter (therapist instead of participant).
Reuse `EmbeddingCodebookClassifier`, the LLM codebook classifier, and `CodebookEnsemble` **unchanged**.

### 2.1 Built already
- `MICROCOUNSELING_CODEBOOK.md` — frontmatter keys `codebook`/`version`/`codebook_description`; one block per code with
  `## Code N — code_id — Category Name`, a ```yaml block (`code_id`,`category`,`domain`), and H3 sections
  `### Description` / `### Inclusive Criteria` / `### Exclusive Criteria` / `### Exemplar Utterances` (each exemplar a
  `>` blockquote separated by a blank line). 8 codes: `reflective_listening`, `validation`, `affirmation`,
  `genuineness`, `respect_for_autonomy`, `asking_permission`, `open_ended_question`, `neutral`.
- `codebook/microcounseling_codebook.py:get_microcounseling_codebook()` → `load_codebook_md(<root>/MICROCOUNSELING_CODEBOOK.md)`.

### 2.2 To build — Segment fields (`classification_tools/data_structures.py`)
Add after the codebook block (`data_structures.py:76-81`), exact analogs of `codebook_labels_*`:
```python
microskill_labels_embedding: Optional[List[str]] = None
microskill_labels_llm: Optional[List[str]] = None
microskill_labels_ensemble: Optional[List[str]] = None
microskill_disagreements: Optional[List[str]] = None
microskill_confidence: Optional[Dict[str, float]] = None
```

### 2.3 To build — config (`process/config.py`)
Add three fields (reuse the existing codebook dataclasses) + a flag, and register them:
```python
microskill_embedding: EmbeddingClassifierConfig = field(default_factory=EmbeddingClassifierConfig)
microskill_llm: LLMCodebookConfig = field(default_factory=LLMCodebookConfig)
microskill_ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
run_microskill_classifier: bool = False
```
Register all three in `from_json.sub_config_map` (`process/config.py:289`) and construct them in
`process/setup_wizard.build_config_from_wizard_data` (`setup_wizard.py:1417`) the same way `gnn_layer` now is.

### 2.4 To build — overlay (`process/classifications_io.py`)
Mirror the codebook overlay (`classifications_io.py:52`):
```python
MICROSKILL_OVERLAY_FIELDS = (
    'microskill_labels_embedding','microskill_labels_llm','microskill_labels_ensemble',
    'microskill_disagreements','microskill_confidence',
)
```
- add `'microskill'` to `OVERLAY_KEYS` (`:62`); `OVERLAY_FILENAMES['microskill']='microskill_labels.jsonl'` (`:64`);
  `_OVERLAY_FIELDS_MAP['microskill']=MICROSKILL_OVERLAY_FIELDS` (`:71`);
- add `write_microskill_overlay`/`merge_microskill_overlay`/`apply_microskill_overlay` (thin wrappers, mirror
  `:129`,`:191`,`:247`) and the `'microskill'` entry in `_APPLY_FUNCS` (`:260`).
- **Also** add the filename to the *second* copy in `process/output_paths.py:classification_overlay_path._filenames`
  (`output_paths.py:212`) — overlay filenames are duplicated in two places.

### 2.5 To build — orchestrator (`process/orchestrator.py`)
- `stage_classify_microskill(config, codebook, *, segments=None, output_dir=None, observer=None, only_session_ids=None)`
  — clone `stage_classify_codebook` (`:307`); call `_microskill_classify`; write/merge the `microskill` overlay;
  `update_classification_manifest(_od, key='microskill', entry=...)`.
- `_microskill_classify(config, codebook, segments, output_dir, observer)` — clone `_codebook_classify` (`:708`) but:
  - filter to therapist segments: `cb_segments = [s for s in segments if s.speaker == 'therapist']` (instead of
    `apply_speaker_filter(segments, config.speaker_filter)` at `:719-720`);
  - use `config.microskill_embedding` / `config.microskill_llm` / `config.microskill_ensemble`;
  - write the `seg.microskill_labels_*` fields (mirror `:749-760`).
- Extend `_build_classifier_manifest_entry` (`:164`) `sub_attr` map with `'microskill': 'microskill_llm'` (`:178`).
- Gate in `run_full_pipeline` after the codebook block (`:1461-1490`):
  ```python
  if config.run_microskill_classifier:
      from codebook.microcounseling_codebook import get_microcounseling_codebook
      micro_cb = get_microcounseling_codebook()
      # persist microcounseling_definitions.json (mirror :1466-1486)
      all_segments = stage_classify_microskill(config, micro_cb, segments=all_segments,
                                               output_dir=output_dir, observer=observer)
  ```

### 2.6 To build — assembly (`process/assembly/master_dataset.py`)
Add the 5 `microskill_*` fields to the `row` dict beside the codebook block (`master_dataset.py:96-100`), using plain
attribute access (they always exist on the Segment once 2.2 lands).

### 2.7 To build — CLI (`qra.py`)
- `cmd_classify` (`:1226`): add `'microskill'` to `valid` (`:1239`) and `to_run` (`:1268`); add a run block mirroring
  the codebook block (`:1287-1291`) calling `stage_classify_microskill(config, get_microcounseling_codebook(), ...)`;
  include `'microskill'` in the `apply_overlays` keys (`:1275`).
- argparse: add `'microskill'` to the `--what` `choices` (`:2044`).

### 2.8 Cross-validation (optional, recommended)
A PURER × microskill lift table is the therapist twin of the VAAMR × VCE cross-validation. It can be computed in the
GNN layer (Part II, `gnn_lift.py`) and/or as a small extension of `process/cross_validation.py`.

---

## 3. Part II — the `gnn_layer/` package (module-by-module)

All modules exist as scaffolds. Below is the contract each must satisfy. **Reuse, do not reinvent:** segment embeddings
come from the existing `EmbeddingCodebookClassifier`; cue blocks come from `analysis/purer_analysis.py`; the lift
formula is the one in `compute_purer_transition_influence`.

### 3.1 `gnn_layer/config.py` — `GnnLayerConfig` (BUILT)
Real dataclass; OFF by default. Key fields: `enabled`, `embedding_model='Qwen/Qwen3-Embedding-8B'`,
`hidden_dim`, `n_layers`, `dropout`, `knn_k`, `objectives` (subset of `soft_vaamr, progression, vce_multilabel, purer,
microskill_multilabel, contrastive, link_prediction`), `label_mode ∈ {weak, human, self_supervised}`,
`contrastive_temp`, `epochs`, `lr`, `seed`, `device`, `run_on_participants/therapists`, `n_motif_clusters`,
`min_motif_influence`, `n_latent_factors`, `interpret_against_cf_ic`. Registered as `PipelineConfig.gnn_layer`.

### 3.2 `gnn_layer/embeddings.py` — Qwen3 reuse
`load_or_build_segment_embeddings(df_all, config, cache_path)` → `{segment_id: np.ndarray}`.
- Build via `_make_embedder(config)` which constructs an `EmbeddingCodebookClassifier` with an
  `EmbeddingClassifierConfig(embedding_model=config.embedding_model, ...)` and calls `._embed_queries(texts)`
  (`codebook/embedding_classifier.py:235`) for segment texts and `._embed(texts)` (`:218`) for anchor/definition texts.
  This reuses the validated float16/CPU-fallback loader (`_get_model:138`) — **no second model**.
- Cache to `02_meta/gnn/segment_embeddings.npz` keyed by `segment_id` + a text hash; re-encode only changed rows when
  `config.cache_embeddings`. Optionally call `ensure_embedding_model_ready(model_id)` (`embedding_classifier.py:50`)
  first.

### 3.3 `gnn_layer/soft_labels.py` — Capability A targets
The superposition signal already exists per segment in `rater_votes` (schema `data_structures.py:60-65`:
`{rater, vote, stage, confidence, secondary_stage, secondary_confidence, justification}`); in the master CSV it is a
JSON string (parse with `analysis/loader._parse_list_column`-style `ast`/`json`).
- `ballots_to_mixture(rater_votes, n_stages=5, secondary_weight=0.5)` → normalized 5-vector (count primary stages
  weighted by `confidence`, add secondary at `secondary_weight`, L1-normalize; fallback = one-hot of `final_label`).
- `mixture_to_progression(mixture)` = `Σ_k k·p_k` (the continuous progression coordinate).
- `build_soft_targets(df_all, label_mode)` → `{segment_id: mixture}` for participant rows; `human` uses only
  `in_human_coded_subset` rows (one-hot of `human_label`); `self_supervised` returns `{}` (link-prediction only).

### 3.4 `gnn_layer/graph_builder.py` — the heterogeneous graph
`build_graph(df_all, segment_embeddings, config, framework)` → `HeteroGraph` (dataclass of torch tensors; **no
torch-geometric**). Node types: `participant_segment`, `therapist_segment`, anchors `vaamr_stage`(5), `purer_move`(5),
`vce_code`(N), `microskill`(8). Edge families:
1. **temporal_chain** seg→next within `session_id`, ordered by `start_time_ms` — reuse the timestamp-window pattern in
   `compute_cue_block_purer_profiles` (`purer_analysis.py:191-208`).
2. **anchor_label** seg→assigned construct, weight = confidence/`agreement_fraction`.
3. **knn_similarity** via `sklearn.neighbors.NearestNeighbors` on Qwen3 vectors (k=`config.knn_k`) — the inductive path.
4. **cross_framework** `vaamr_stage↔vce_code` and `purer_move↔microskill`, weight = empirical lift
   (`compute_cross_framework_lift`, gated by `config.cross_framework_min_lift`; ablatable in `ablation.py`).
`attach_new_segments(graph, new_embeddings, config)` adds nodes + kNN edges to the *frozen* anchor/labeled set only
(new nodes never link to each other → order-invariant, reproducible inductive inference).

### 3.5 `gnn_layer/model.py` — pure-PyTorch GraphSAGE
- `scatter_mean(src, index, dim_size)` via `index_add_` + degree normalization (the torch-geometric-free mean-agg).
- `SAGEConv(in,out).forward(x, edge_index)` = `ReLU(W_self·x + W_neigh·scatter_mean(x[src], dst, N))`.
- `build_model(graph, config)` stacks `config.n_layers` SAGEConv (ReLU+dropout) then the enabled heads: `soft_vaamr`
  (5), `progression` (1), `vce_multilabel`, `purer` (5), `microskill_multilabel`.
- `compute_losses(outputs, targets, config)` = KL(soft_vaamr‖ballot mixture) + MSE(progression) +
  BCE(vce/microskill) + CE(purer) + InfoNCE supervised-contrastive (rank-aware over the ordered VAAMR arc) +
  link-prediction BCE on the temporal chain. All stock torch.

### 3.6 `gnn_layer/train.py` — training + checkpoint
`train_model(graph, targets, config)` — full-batch transductive, Adam(`config.lr`), early stopping (`patience`),
seeded (`set_seed`). Supports all three `label_mode`s. `export_checkpoint(model, config, model_dir, metrics)` writes
`weights.pt` + `manifest.json` (config, seed, n_nodes, data hash, metrics) to `02_meta/gnn/model/`; `load_checkpoint`
rebuilds for inductive reuse.

### 3.7 `gnn_layer/inference.py` — per-segment outputs
`infer_segment_positions(model, graph, config)` → per segment: `vaamr_mixture` (softmax of soft_vaamr),
`progression_coord` (E[stage]), `gnn_embedding`, therapist `microskill_logits`. `cue_block_embeddings(df_all,
segment_gnn_embeddings)` mean-pools therapist-segment embeddings per cue block (cue blocks via
`compute_cue_block_purer_profiles`).

### 3.8 `gnn_layer/motifs.py` — Capability B (flagship)
The current pipeline collapses each cue block to `dominant_purer` (1 of 5) **before** measuring influence
(`purer_analysis.py:124,:286,:295`). This module keeps the continuous cue embedding:
- `cluster_cue_motifs(cue_embeddings, config)` — `sklearn` Agglomerative/KMeans (`n_motif_clusters`).
- `score_motif_influence(cue_embeddings, from_stages, forward_outcome, motif_ids, config)` — from_stage-conditioned
  `LogisticRegression` of forward-transition outcome on the cue embedding → per-motif influence (preserves the §7.6
  stage-moderation hypothesis).
- `flag_emergent_motifs(...)` — influential (`≥ min_motif_influence`) but low PURER/microskill purity = therapist
  language that fits no label. `select_motif_exemplars(...)` — nearest-to-centroid exemplar cues for human review.

### 3.9 `gnn_layer/gnn_lift.py` — Capability C (independent substrate)
Recompute (VAAMR×VCE), (PURER×transition), (PURER×microskill) lift from **GNN** assignments using the same formula as
`compute_purer_transition_influence` (`purer_analysis.py:347`, `lift = P(to|x)/P(to)`); `compare_gnn_vs_llm` joins them
to the LLM-derived tables. For a clean independence claim, run the `self_supervised` / `human` `label_mode` variant so
the GNN substrate is not trained on LLM labels.

### 3.10 `gnn_layer/ablation.py` — Capability D
`run_ablation(df_all, base_graph, config, ablate∈{vce,purer,microskill})` re-trains with a construct family removed and
reports metric deltas (settles "is VCE superfluous?"). `discover_subtypes(..., by∈{vaamr_stage,purer_move})` clusters
segment embeddings within each stage/move for emergent sub-types.

### 3.11 `gnn_layer/coupling.py` — Capability E
`fit_coupling_model`, `extract_latent_factors` (NMF/PCA on coupling-weighted therapist embeddings),
`interpret_factors` names factors against the **inline** `CF_IC_REFERENCE` lexicon (defined in the module, NOT imported
from CFiCS) — rediscovering common factors, never imposing them.

### 3.12 `gnn_layer/reports.py` — outputs
Writers mirroring `export_purer_csvs`/`generate_purer_report` (`purer_analysis.py:364,:385`): `segment_positions.csv`,
`cue_motifs.csv`, `gnn_vs_llm_lift.csv`, `coupling_factors.csv` under `03_analysis_data/gnn/`; and
`report_gnn_emergent_motifs.txt`, `report_gnn_coupling.txt` under `06_reports/`.

### 3.13 `gnn_layer/runner.py` — entry point (scaffold returns safely today)
`run_gnn_analysis(df_all, output_dir, framework=None, config=None, llm_client=None)` — signature mirrors
`run_purer_analysis`. Orchestrates 3.2→3.11 and returns `{'files_written', 'status', ...}`. The scaffold currently
logs and returns `status='scaffold'` so enabling the layer never breaks a run.

---

## 4. Part III — wiring touchpoints (summary table)

| File | Change | State |
|---|---|---|
| `process/config.py` | `gnn_layer` field + `sub_config_map`; (Part I) `microskill_*` + `run_microskill_classifier` | gnn: **built**; microskill: to do |
| `process/setup_wizard.py:1417` | carry `gnn_layer` (built); carry microskill configs (to do) | gnn: **built** |
| `analysis/runner.py` (after PURER block) | guarded `run_gnn_analysis` hook | **built** |
| `process/output_paths.py` | add `gnn_data_dir`→`03_analysis_data/gnn`, `gnn_model_dir`→`02_meta/gnn`; microskill overlay filename (`:212`) | to do |
| `process/classifications_io.py` | `microskill` overlay registration | to do (Part I) |
| `classification_tools/data_structures.py:76` | `microskill_*` fields | to do (Part I) |
| `process/orchestrator.py` | `stage_classify_microskill`+`_microskill_classify`+gating+manifest | to do (Part I) |
| `process/assembly/master_dataset.py:96` | microskill row fields | to do (Part I) |
| `qra.py` | `classify --what microskill`; optional `analyze --gnn` override threaded to `run_analysis(force_gnn=)` | to do |

`analyze --gnn`: add the arg at `qra.py:2112`, thread `force_gnn` through `cmd_analyze` (`:1078`) → `run_analysis`
(`analysis/runner.py:15`) to override `config.gnn_layer.enabled`.

---

## 5. Part IV — data contracts

- **Segment** gains only `microskill_*` (Part I). **No GNN per-segment fields** on Segment/master.
- **Overlay:** `02_meta/classifications/microskill_labels.jsonl` (+ manifest key `microskill`).
- **GNN artifacts (keyed by `segment_id`):** `03_analysis_data/gnn/{segment_positions.csv, cue_motifs.csv,
  gnn_vs_llm_lift.csv, coupling_factors.csv, segment_embeddings.npz}`; `06_reports/{report_gnn_emergent_motifs.txt,
  report_gnn_coupling.txt}`; model at `02_meta/gnn/model/`.
- **Definitions for `analyze`:** persist `02_meta/microcounseling_definitions.json` (mirror `codebook_definitions.json`,
  orchestrator `:1466`).

---

## 6. Part V — dependencies
None added. Uses installed `torch` (pure-PyTorch GraphSAGE), `scikit-learn` (clustering/logreg/NMF/kNN),
`numpy`/`pandas`, `sentence-transformers` (via the existing embedding classifier), `networkx` (optional graph utils).
torch-geometric is explicitly **not** required (avoids torch-scatter/torch-sparse compile fragility on torch 2.11).

---

## 7. Part VI — testing strategy
Mirror the existing patterns that avoid backends by passing pre-populated segments / synthetic frames:
- `tests/test_microskill_classify.py` — like `tests/test_zero_shot_classify.py` / `test_orchestrator_stages.py`:
  pre-set `microskill_labels_*` on therapist segments (config=None path), assert overlay written, manifest has
  `'microskill'`, master row carries the fields. No model/LLM needed.
- extend `tests/test_classifications_io.py` — round-trip the `microskill` overlay (write→read→apply), frozen segments
  untouched.
- `tests/test_gnn_layer.py` — synthetic `df_all` + random embeddings injected (monkeypatch `embeddings.*` so Qwen3 is
  never downloaded): assert `graph_builder` node/edge counts, `soft_labels.ballots_to_mixture` normalizes a synthetic
  `rater_votes`, `model` forward shapes on a ~6-node graph, and `run_gnn_analysis` writes the expected files.
- Keep `enabled=False` default so the full suite and pipeline are unchanged.

---

## 8. Part VII — verification (runnable checks)
```bash
# Codebook parses through the real loader (8 codes, 4 exemplars each)
python -c "from codebook.microcounseling_codebook import get_microcounseling_codebook as g; \
cb=g(); print(cb.name, cb.version, len(cb.codes), cb.domain_names); \
assert len(cb.codes)==8 and all(len(c.exemplar_utterances)==4 for c in cb.codes)"

# gnn_layer imports with no heavy deps loaded
python -c "import gnn_layer, gnn_layer.runner, gnn_layer.config; print('ok')"

# config carries the layer, OFF by default (requires pandas/numpy installed)
python -c "from process.config import PipelineConfig; print(PipelineConfig().gnn_layer.enabled)"  # -> False

# existing suite still green (layer disabled)
python -m pytest tests/test_orchestrator_stages.py tests/test_purer_analysis.py tests/test_classifications_io.py -q
```

---

## 9. Part VIII — risks & honest caveats (carry into reports)
Small N → all GNN outputs are **directional/hypothesis-generating** until human-validated (`methodology.md` §5.4); the
GNN never overrides labels-of-record. Weak-label circularity → report the `human`/`self_supervised` variants for the
independence claim. Discovered motifs / latent CF-IC factors are **hypotheses** requiring human review before curriculum
use (§5, §9.5). Augment, don't replace. No framework proliferation beyond VAAMR/VCE ‖ PURER/microskill.

---

## 10. Suggested implementation order
1. Part I microcounseling classifier (small, high-value, mirrors a proven stage) + its tests.
2. `gnn_layer/embeddings.py` + `soft_labels.py` + `graph_builder.py` (+ synthetic tests).
3. `model.py` + `train.py` + `inference.py` (Capability A: `segment_positions.csv`).
4. `motifs.py` + `reports.py` (Capability B — the flagship curriculum output).
5. `gnn_lift.py` (Capability C), then `ablation.py` (D), then `coupling.py` (E).
6. `qra.py analyze --gnn` + docs/README updates.
