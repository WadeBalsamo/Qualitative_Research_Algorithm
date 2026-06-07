# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

QRA (Qualitative Research Algorithm) is a computational phenomenology pipeline that applies two classification frameworks bilaterally to therapy transcripts from the Move-MORE Feasibility Trial (mindfulness for chronic pain):

- **VAAMR** (Vigilance-Avoidance-Attention Regulation-Metacognition-Reappraisal) — classifies **participant** segments across a five-stage developmental arc
- **PURER** (Phenomenological-Utilization-Reframing-Educate/Expectancy-Reinforcement) — classifies **therapist** segments across five guided-inquiry moves at the cue-block level (between consecutive participant turns)
- **VCE phenomenology codebook** — optional multi-label construct enrichment (54 codes, 6 domains) applied to participant segments

**Framework definitions are markdown-driven.** VAAMR/PURER are parsed from `frameworks/VAAMR_FRAMEWORK.md` / `frameworks/PURER_FRAMEWORK.md` via `src/theme_framework/registry.py` → `load('vaamr'|'purer')`; the VCE codebook is parsed from `frameworks/PHENOMENOLOGY_CODEBOOK.md` via `src/codebook/markdown_loader.py`. The `src/theme_framework/vaamr.py` / `purer.py` modules are thin wrappers over the markdown loader — **edit the `.md` files to change definitions, not the Python.** Each `.md` follows a parser contract (YAML frontmatter + structured headings); register a new framework by dropping a `.md` in `frameworks/` and adding it to `FRAMEWORKS` in `src/theme_framework/registry.py`.

See `docs/methodology.md` for the full neurophenomenological methodology and `README.md` for research context.

## Commands

```bash
# Install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Interactive setup wizard (creates qra_config.json)
python qra.py setup

# Run full pipeline with saved config (orchestrates ingest → classify → assemble phases)
python qra.py run --config ./data/output/02_meta/qra_config.json

# Run with inline overrides
python qra.py run --transcript-dir ./data/input --output-dir ./data/output \
  --backend openrouter --model qwen/qwen-3-70b

# Run pipeline + auto-generate analysis reports
python qra.py run --config ./qra_config.json --auto-analyze

# Post-hoc analysis on completed pipeline output
python qra.py analyze --output-dir ./data/output/

# Modular pipeline stages (Phase 3)
python qra.py ingest -o ./data/output/                       # segment only (--fresh = re-segment all)
python qra.py classify -o ./data/output/ --what vaamr        # classify VAAMR only
python qra.py classify -o ./data/output/ --what vaamr --fresh # re-classify FROM SCRATCH (clear ckpts+overlay)
python qra.py assemble -o ./data/output/                     # join frozen+overlays

# GNN discovery + mechanism layer runs automatically at analyze-time (default ON):
#   discriminant validity (H6), dyadic transition model + confound localization, subtext
#   communities + dyadic routines, cue motifs, coupling — all on raw embeddings, no trained model.
# The GraphSAGE consensus-distillation CLASSIFIER is a SEPARATE concern, DEFAULT OFF (H5-refuted):
python qra.py gnn train -o ./data/output/                    # opt IN: train the classifier + reliability gate (sets gnn_classifier_enabled)
python qra.py gnn status -o ./data/output/                   # κ(graph,LLM); ready for LLM-free scaling? (only after `gnn train`)
python qra.py gnn classify -o ./data/output/                 # LLM-free label new/unlabeled segments (needs a trained classifier)

# Import a legacy (pre-SQLite, JSONL) project into qra.db (preview, then --run)
python qra.py migrate -o ./data/output/                      # preview
python qra.py migrate -o ./data/output/ --run               # perform (originals → _legacy_files/)

# Incremental data addition (segment + classify only NEW transcripts; re-assembles + re-analyzes)
python qra.py add-data --config ./data/output/02_meta/qra_config.json

# Speaker anonymization management
python qra.py apply-anonymization -o ./data/output/       # scrub PHI names from frozen segment text
python qra.py edit-anonymization -o ./data/output/        # edit speaker key + cascade downstream

# Inter-rater reliability (Human↔Human, Human↔LLM, Human↔GNN held-out + distillation)
python qra.py irr import -o ./data/output/ --csv data/irr/human_coded_testsets.csv
python qra.py irr run -o ./data/output/                       # pull live LLM+GNN, compute, report
python qra.py irr list -o ./data/output/                      # bare `qra irr` → TUI

# Validation test set management (Phase 2)
python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1
python qra.py testset refresh -o ./data/output/ --all
python qra.py testset list -o ./data/output/

# Content-validity test set management (Phase 2)
python qra.py cv create -o ./data/output/ --framework purer --name cv_purer_v1
python qra.py cv refresh -o ./data/output/ --all
python qra.py cv list -o ./data/output/

# Zero-shot content validity test (skips full pipeline)
python qra.py run --test-zeroshot --preset small --output-dir ./data/output/

# Run test suite — two tiers
python tests/run_unit_tests.py          # hermetic: no network, no model downloads, no Ollama
python tests/run_integration_tests.py   # real tiny models (MiniLM embeddings + tiny Ollama LLM); skips LLM tests if Ollama absent

# Run a single unit test
python -m unittest tests.unit.test_purer_analysis -v
python -m unittest tests.unit.test_purer_analysis.TestClass.test_method
```

Unit tests live in `tests/unit/` (hermetic, always run), integration tests in `tests/integration/` (download model weights, write to gitignored `tests/testrun-outputs/`). Shared fakes/fixtures are in `tests/testhelpers/` (`fake_llm.py`, `fixtures.py`, `tiny_models.py`, `ollama_helper.py`). `tests/conftest.py` puts `src/` on `sys.path` so first-party packages import without `pip install -e .`. Integration tests are never collected by the unit runner.

## Pipeline Architecture (8 Stages + 2 Optional Sub-stages)

`src/process/orchestrator.py:run_full_pipeline()` sequences these stages:

### Stage 1 — Transcript Ingestion & Segmentation
(`src/process/transcript_ingestion.py`, `src/process/llm_segmentation.py`)
- Loads diarized JSON/VTT transcripts
- Semantic segmentation via embedding similarity (adaptive threshold + topic clustering)
- Optional LLM boundary refinement (`boundary_review`, `context_expansion`, `coherence_check`)
- Speaker normalization and anonymization with persistent speaker map
- **Frozen segments** written to the `segments` table of the project SQLite store (`qra.db`)
- Therapist segments extracted and interleaved with participant segments in chronological order

### Stage 2 — Construct Operationalization
- Builds framework definitions from `src/theme_framework/vaamr.py` / `purer.py`
- Exports theme definitions (JSON + txt)
- Creates and exports content validity test set from exemplar/subtle/adversarial utterances
- Writes content validity human worksheet, definition key, and AI answer key to `04_validation/`

### Stage 3 — Zero-Shot LLM Theme Classification (VAAMR)
(`src/classification_tools/classification_loop.py`)
- Multi-run zero-shot LLM classification with per-run model rotation (checker models)
- Context-aware prompting with preceding participant segment context
- Multi-run consensus voting produces confidence tiers (High/Medium/Low)
- Speaker filter applied: therapist segments excluded from VAAMR classification

### Stage 3c — PURER Cue-Unit Classification (optional)
- PURER classified at the **cue-block level**: one label per therapist response between consecutive participant turns
- Uses wider context window (6 preceding segments vs 2 for VAAMR)
- Can skip lesson-content chunks (long didactic stretches configured via `PurerCueConfig`)
- PURER label propagated back to all constituent therapist segments
- Single-run classification by default (no multi-model IRR needed for PURER)

### Stage 3b — Codebook Classification (optional)
(`src/codebook/embedding_classifier.py`, `src/codebook/ensemble.py`)
- VCE embedding similarity + LLM zero-shot multi-label coding on participant segments
- Ensemble reconciliation of embedding and LLM results
- GPU memory hand-off: reuses segmentation embedding model when model IDs match

### Stage 4 — Cross-Validation (optional, requires Stage 3b)
- VAAMR stage × VCE code co-occurrence lift statistics implementing Varela's mutual-constraints logic
- Summarizes top associations per theme

### Stage 5 — Human Validation Set
- Stratified balanced evaluation set for blind coding
- Exports `04_validation/human_coding_evaluation_set.csv`

### Stage 6 — Dataset Assembly
(`src/process/assembly/master_dataset.py`)
- Assembles `master_segments.csv` (generated export; the analysis layer reads it) with confidence tiering
- `run_full_pipeline` assembles here; standalone `qra assemble` joins frozen segments + classification overlays (both read from `qra.db`)

### Stage 7 — Report Generation
- Coded transcripts (`01_transcripts/coded/`)
- Human classification forms (`04_validation/human_classification_<session>.txt`)
- `flagged_for_review.txt`
- Cross-session validation test sets (VAAMR/PURER/codebook — Phase 2: `04_validation/testsets/`)
- Content-validity test sets (VAAMR/PURER — Phase 2: `04_validation/content_validity/`)
- Per-transcript stats, cumulative report, training data export

### Stage 8 — Results Analysis (optional, post-hoc)
(`src/analysis/runner.py`)
- Per-participant longitudinal trajectories, per-session summaries, per-theme analyses
- Therapist cue response analysis (PURER × VAAMR)
- Descriptive progression summary (single-arm, ordinal-safe — not efficacy) + mechanistic Δprogression (forward AND backward), avoidance barrier (bidirectional)
- GNN discovery + mechanism layer (ON by default; figures + reports under `06_gnn/`), all on raw embeddings and hypothesis-generating (never causal): **H6 discriminant validity** (`discriminant_validity`), the **dyadic FROM→CUE→TO transition model** + **confound localization** (`transition_model`/`confound_localization` — the mechanism rebuild that replaced the mis-specified classifier-counterfactual, `GNN_MASTER_PLAN.md` §4.7), **subtext communities + dyadic routines** (`subtext_communities`), cue motifs, coupling factors. The GraphSAGE **consensus-distillation classifier is a SEPARATE concern in `src/gnn_layer/classifier/`, DEFAULT OFF** (`gnn_classifier_enabled`; H5-refuted at n≈32 — κ≈0.05–0.14 < human band, a probe ties/beats it); enable with `qra gnn train` to re-adjudicate at Cohorts 3–4. Track C MindfulBERT dataset builder (`build_mindfulbert_dataset`). Full design record: `docs/GNN_MASTER_PLAN.md`, `graph_experiments.md`.
- Graph-ready CSVs, visualization figures (incl. GNN figures)
- Session and participant LLM summaries
- **Top-level synthesis written last**: `00_executive_summary.txt` (deterministic program-improvement brief), `00_READ_ME.txt`, `07_methods_appendix.txt`

## Framework Boundaries: Critical Design Rule

**VAAMR applies exclusively to participant segments. PURER applies exclusively to therapist segments.** These labels never cross. Therapist segments appear as read-only preceding context in classification prompts for participant segments, but are never themselves classified with VAAMR. This boundary is enforced by the speaker filter in each stage.

PURER moves frequently co-occur within a single therapist turn/cue-block. When a single label is required, an empirical precedence order is defined in `src/theme_framework/purer.py`:
- Reinforcement is often a wrapper (code the substantive inner move)
- Utilization > Reframing for forward-application prompts
- Reframing > Education when anchored to participant's specific story

## On-Disk Layout (SQLite store + human-facing/exported files)

The pipeline's **internal data store is a single SQLite file per project: `qra.db`** (schema in
`src/process/db.py`). Frozen segments, the five classification overlays, the provenance manifest,
and validation/content-validity testset metadata all live in `qra.db` tables — they are no longer
scattered JSONL/JSON files. Everything else on disk is either a human-facing artifact (worksheets,
reports, figures, coded transcripts), a generated export (`master_segments.csv`, BERT training JSONL),
or external config (`qra_config.json`, speaker key).

`qra.db` tables: `segments` (frozen raw segmentation + `params_hash`/`segmenter_version`/`ingest_timestamp`
columns), `theme_labels` / `purer_labels` / `codebook_labels` / `cv_labels` / `gnn_labels` (overlays),
`classification_manifest`, `testset_worksheets` + `testset_items`, `cv_testsets` + `cv_testset_items`,
`irr_testsets` + `irr_human_codes` (imported human IRR codes — ground truth, maintained across re-runs).

```
output_dir/
├── qra.db                   # SQLite store: segments, overlays, manifest, testset metadata
├── 01_transcripts_inputs/   # Raw diarized input copies (provenance)
├── 01_transcripts/          # (per-session segments are in qra.db, not files)
├── 02_meta/
│   ├── auditable_logs/      # LLM prompts/responses, checkpoints (kept as files)
│   ├── codebook_raw/        # Codebook embedding checkpoints
│   ├── training_data/       # master_segments.csv, BERT training JSONL exports
│   └── speaker_anonymization_key.json
├── 03_analysis_data/        # Session stats, graphing CSVs, per-{session,participant,theme} JSON
├── 04_validation/
│   ├── full_transcripts/    # Coded transcripts + human classification forms
│   ├── testsets/            # FROZEN — human worksheets + AI answer keys (.txt); item metadata in qra.db
│   ├── content_validity/<name>/  # FROZEN — worksheet/definition_key/AI_answer_key .txt; manifest+items in qra.db
│   ├── cross_validation/    # Lift statistics
│   ├── irr/                 # Inter-rater reliability: results.json, pairwise/discrepancy/item CSVs,
│   │                        #   per-testset dossiers (irr_items_testset_<n>.txt), figures
│   └── human_coding_evaluation_set.csv
├── _legacy_files/           # (only if migrated) original JSONL/JSON moved here, non-destructively
├── 05_figures/              # PNG visualization figures (incl. gnn_*.png)
├── 06_reports/              # Human-readable text reports (tiered, numbered)
│   ├── 00_READ_ME.txt           # guide to the tree + reading order
│   ├── 00_executive_summary.txt # deterministic program-improvement brief
│   ├── 01_outcomes/             # progression_summary.txt, longitudinal.txt, avoidance_barrier.txt
│   ├── 02_mechanism/            # transitions, cue_response, purer, mechanism, language_atlas, superposition
│   ├── 03_per_session/          # _overview.txt, session_<id>.txt, session_summaries.txt
│   ├── 04_per_participant/      # participant_<id>.txt
│   ├── 05_per_stage/            # stage_<name>.txt, codebook_exemplars.txt
│   ├── 06_gnn/                  # discriminant_validity, transition_model, confound_localization,
│   │                            #   communities, dyadic_routines, emergent_motifs, coupling
│   │                            #   (+ validation/triangulation only when the classifier is enabled)
│   └── 07_methods_appendix.txt  # how each metric is computed + caveats
└── 00_index.txt
```

**06_reports tier:** `00_*` files are the top-level synthesis (start here); `01_outcomes`
answers "did it work" (forward AND backward movement), `02_mechanism` answers "how";
`03–05` are drill-downs; `06_gnn` holds the independent graph-model discovery/validation.
All report paths are resolved through `src/process/output_paths.py` (e.g. `reports_outcomes_dir`,
`reports_mechanism_dir`, `reports_gnn_dir`, `themes_dir` → `05_per_stage`).

## Module Map

| Path | Role |
|------|------|
| `qra.py` | CLI entry point (setup / run / analyze / ingest / classify / assemble / gnn / migrate / add-data / reclassify-run / testset / cv / irr / *-anonymization) |
| `src/process/orchestrator.py` | Stage sequencing, `run_full_pipeline`, standalone `stage_*` functions |
| `src/process/config.py` | `PipelineConfig` dataclass — single source of truth for all settings |
| `src/process/db.py` | SQLite store: `qra.db` schema (12 tables), `open_db()` atomic-transaction context manager, WAL, JSON-column helpers |
| `src/process/segments_io.py` | Frozen segment I/O (SQLite `segments` table), `params_hash`, `read_master_segments` (frozen+overlays), `load_segments_for_stage` |
| `src/process/classifications_io.py` | Overlay tables read/write (SQLite UPSERT), `read_overlay`/`overlay_exists`/`clear_overlay`/`remap_overlay_segment_ids`, provenance manifest |
| `src/process/reclassify_ops.py` | From-scratch reset helper for `--fresh` (clears checkpoints + overlay per framework); shared by CLI + TUI |
| `src/process/_freeze.py` | Freeze primitives — `FrozenArtifactError`, SHA verification (still used for on-disk frozen .txt worksheets) |
| `src/process/legacy_migration.py` | Migration shim: v2.0→per-session, v2.5→v3 layout, **JSONL→SQLite (`migrate_jsonl_to_sqlite`, `preview_counts`; surfaced as `qra migrate`)**, and `qra_config.json` default-fill (`upgrade_config_file`) |
| `src/process/transcript_ingestion.py` | Semantic segmentation logic, `ConversationalSegmenter` |
| `src/process/llm_segmentation.py` | LLM-assisted segmentation boundary refinement |
| `src/process/speaker_anonymization.py` | Persistent speaker ID mapping across runs |
| `src/process/speaker_filter.py` | Speaker inclusion/exclusion rules per classifier |
| `src/process/output_paths.py` | Single source of truth for ALL output paths (incl. `db_path` → `qra.db`); overlay/manifest path helpers retained as legacy strings |
| `src/process/output_index.py` | `00_index.txt` generation |
| `src/process/cross_validation.py` | VAAMR × VCE code co-occurrence lift statistics |
| `src/process/cue_blocks.py` | Canonical cue-block builder (run of therapist turns between two participant turns); unifies orchestrator/analysis/gnn implementations |
| `src/process/setup_wizard.py` | Interactive configuration wizard (14 steps) |
| `src/theme_framework/registry.py` | `load(name)` → `ThemeFramework`; `FRAMEWORKS` name→`frameworks/*.md` dispatch (cached) |
| `src/theme_framework/markdown_loader.py` | Parses `frameworks/*_FRAMEWORK.md` → `ThemeFramework`/`ThemeDefinition` (parser contract) |
| `src/theme_framework/vaamr.py` | `get_vaamr_framework()` — thin wrapper parsing `frameworks/VAAMR_FRAMEWORK.md` |
| `src/theme_framework/purer.py` | `get_purer_framework()` — thin wrapper parsing `frameworks/PURER_FRAMEWORK.md` |
| `src/theme_framework/theme_schema.py` | `ThemeDefinition`, `ThemeFramework` dataclasses |
| `src/codebook/markdown_loader.py` | Parses `frameworks/PHENOMENOLOGY_CODEBOOK.md` → `Codebook` |
| `src/codebook/phenomenology_codebook.py` | `get_phenomenology_codebook()` — 54 VCE codes / 6 domains, parsed from markdown |
| `src/codebook/codebook_schema.py` | `CodeDefinition`, `Codebook` dataclasses |
| `src/codebook/embedding_classifier.py` | Sentence-transformer embedding-based classification |
| `src/codebook/ensemble.py` | Ensemble reconciliation of embedding + LLM results |
| `src/classification_tools/llm_classifier.py` | Zero-shot prompt construction and response parsing |
| `src/classification_tools/zeroshot_reporting.py` | `write_zeroshot_report` — graded `--test-zeroshot` content-validity report (extracted from `qra.py`) |
| `src/classification_tools/llm_client.py` | Backend abstraction (OpenRouter, Ollama, LM Studio, HuggingFace, Replicate) |
| `src/classification_tools/classification_loop.py` | Multi-run consensus voting with checkpointing |
| `src/classification_tools/data_structures.py` | `Segment` dataclass |
| `src/analysis/runner.py` | Post-hoc analysis entry point (auto-regenerates IRR when human codes present + models changed) |
| `src/process/irr_import.py` | Import human-coded IRR CSV → `irr_testsets`/`irr_human_codes`; label normalize, consensus validate, resolve worksheet item → `segment_id`, drift warnings |
| `src/process/irr_tui.py` | `qra irr` interactive menu (import / run / view / list) |
| `src/analysis/irr_analysis.py` | IRR orchestrator: Human↔Human, Human↔LLM (consensus + per-model), Human↔GNN (held-out + distillation); per-item details; `maybe_run_irr` change-gated regen |
| `src/analysis/irr_stats.py` | IRR statistics via proven libraries — Cohen κ (scikit-learn), Fleiss κ (statsmodels), Krippendorff α (`krippendorff`) |
| `src/analysis/reports/irr_report.py` | `06_reports/06b_irr_report.txt` (headline κ table, dual GNN axes + gate read, discrepancies) |
| `src/analysis/reports/irr_items.py` | Per-test-set line-by-line dossier (text + human/LLM reasonings + GNN held-out + LLM↔GNN) |
| `src/analysis/irr_figures.py` | IRR confusion matrices + rater-agreement heatmap |
| `src/analysis/purer_analysis.py` | PURER × VAAMR cue-block influence analysis |
| `src/analysis/purer_figures.py` | PURER × VAAMR lift heatmap and figures |
| `src/analysis/reports/` | Detailed text report generators (session, stage, transition, cue response, longitudinal, summaries) |
| `src/analysis/reports/executive_summary.py` | Deterministic `00_executive_summary.txt` program-improvement brief |
| `src/analysis/reports/reports_guide.py` | `00_READ_ME.txt` (report map) + `07_methods_appendix.txt` (how metrics are computed) |
| `src/analysis/efficacy.py` | Descriptive progression summary (ordinal-safe) + `efficacy_summary.json`; convergent-validity outcome linkage |
| `src/analysis/mechanism.py` | Signed Δprogression mechanism + bidirectional avoidance-barrier report |
| **`src/gnn_layer/` (top level)** | **Discovery + construct-validation + mechanism work-streams** — DEFAULT ON, run on raw embeddings (no trained model), hypothesis-generating, never causal. Orchestrated by `runner.run_gnn_analysis` (classifier gated OFF by default; discovery always). |
| `src/gnn_layer/discriminant.py` | **H6 discriminant validity** — supervised probe (above chance) vs content-similarity C&S (≈chance) on the *same* Qwen embeddings, both axes + paired Δκ CI; geometry (content-PC stage recovery, kNN stage homophily, community×stage ARI). `06_gnn/discriminant_validity.txt` |
| `src/gnn_layer/transition.py` | **Mechanism rebuild** — dyadic FROM→CUE→TO learned response model `TO_mixture≈f(FROM_mixture, FROM_stage, pooled raw cue)`, NO kNN; earns-its-place grouped-CV (cue vs FROM-only) + learned counterfactual + triangulation vs observed Δprog. `06_gnn/transition_model.txt` |
| `src/gnn_layer/confound.py` | **Confound localization** — signed divergence (observed − learned counterfactual) per (from_stage×move) with participant-clustered CIs; maps where the elicitation/responsiveness confound (§9.4) most distorts the observed table. `06_gnn/confound_localization.txt` |
| `src/gnn_layer/communities.py` | **Track D + dyadic routines** — subtext-similarity graph (τ≈0.6 for Qwen), Louvain+agglomerative ARI, participant-bootstrap stability selection, community↔stage/Δprog, atypical exemplars, therapist→participant dyadic routines. `communities.txt`, `dyadic_routines.txt` |
| `src/gnn_layer/{motifs,coupling}.py` | Cue-motif discovery + latent coupling factors (on raw pooled cue embeddings) |
| `src/gnn_layer/cue_features.py` | Shared, model-free cue-block builder + mean-pool (used by discovery AND `analysis/mechanism`, `language_atlas`, `mindfulbert`) |
| `src/gnn_layer/{embeddings,soft_labels,figures,reports,config}.py` | Shared substrate: Qwen3 embeddings (cached), soft VAAMR mixtures from ballots, figures, report writers, `GnnLayerConfig` |
| **`src/gnn_layer/classifier/`** | **GraphSAGE consensus-distillation CLASSIFIER — separate concern, DEFAULT OFF** (`gnn_classifier_enabled=False`). Pilot-refuted as a scaler (H5: grouped-CV κ≈0.05–0.14 < human↔human band; a probe ties/beats it; VAAMR not homophilous). Kept + re-adjudicable at Cohorts 3–4. Contains `model/train/graph_builder/validation`(gate)`/triangulation/inference/calibration/propagation/ablation/anchors/gnn_lift`. Enable via `qra gnn train`. |
| `src/process/assembly/__init__.py` | Assembly module exports |
| `src/process/assembly/master_dataset.py` | `assemble_master_dataset` (gate-gated `gnn_consensus` promotion) |
| `src/process/assembly/mindfulbert_dataset.py` | Track C — MindfulBERT (cue language → observed Δprogression) dataset builder + augmentation-validation harness + datasheet |
| `src/process/assembly/human_forms.py` | Human classification forms, test set freeze/refresh |
| `src/process/assembly/content_validity.py` | Content-validity test set freeze/refresh |
| `src/process/assembly/coded_transcripts.py` | Per-session coded transcript writer |
| `src/process/assembly/stats_reports.py` | Per-transcript stats, cumulative report |
| `src/process/assembly/training_export.py` | Training data, theme definitions, content validity item export |

## Configuration

`src/process/config.py:PipelineConfig` covers all settings: paths, trial metadata, framework/codebook selection, LLM backend and model, classification parameters (n_runs, temperature, confidence thresholds), embedding model, speaker filtering, feature flags, test set config (multi-kind: VAAMR/PURER/codebook), content validity config, PURER cue config (skip_lesson_content, max_lesson_words, therapist_max_gap_seconds, max_context_words), session/participant summary configs.

The setup wizard serializes it to `./data/output/02_meta/qra_config.json`; `--config` reproduces any run exactly.

LLM backends: LM Studio (local), OpenRouter (`OPENROUTER_API_KEY`), Replicate (`REPLICATE_API_TOKEN`), Ollama, HuggingFace.

## VAAMR Stage Reference

| theme_id | Name | Conceptual Stage | Description |
|----------|------|-----------------|-------------|
| 0 | Vigilance | 0 | Attentional capture by pain; body as dysappearing obstacle |
| 1 | Avoidance | 0.5 | Attentional skill deployed for experiential escape |
| 2 | Attention Regulation | 1 | Stable volitional presence with somatic experience |
| 3 | Metacognition | 2 | Reflexive observation of mental processes |
| 4 | Reappraisal | 3 | Transformation of pain's meaning/sensory structure |

## PURER Move Reference

| move_id | Code | Name | When to use |
|---------|------|------|-------------|
| 0 | P | Phenomenological | Step-by-step elicitation of practice experience |
| 1 | U | Utilization | Prompting forward application to everyday life |
| 2 | R | Reframing | Repositioning participant's report as a MORE concept |
| 3 | E | Educate/Expectancy | Psychoeducation about pain/mindfulness + expectation |
| 4 | R2 | Reinforcement | Selective affirmation of adaptive responses or insights |

## Key Design Invariants

1. **SQLite store**: The project's internal data lives in one `qra.db` per `output_dir` (`src/process/db.py`). Each `open_db()` `with` block is one atomic transaction (commit on clean exit, rollback on exception) — this replaces the old tmp-file+rename atomicity. `segment_id` is the join key across the `segments` table and every overlay table.
2. **Frozen segmentation**: Once segmented, a session's rows in the `segments` table are frozen — `write_session_segments` raises `FrozenArtifactError` on overwrite-without-force. Only raw-segmentation fields are persisted (no classification data); classification fields stay at dataclass defaults on read-back.
3. **Classification overlays**: Re-running a classifier replaces only its overlay table (`theme_labels` / `purer_labels` / …) — `write_*_overlay` is a full table replace, `merge_*_overlay` is an UPSERT by `segment_id`. Frozen segments are untouched. `read_overlay(run_dir, key)` returns records sorted by `segment_id`.
4. **Multi-kind test sets**: VAAMR testsets sample participant segments (stratified by cohort). PURER testsets sample therapist segments with non-null labels (stratified by PURER primary). Codebook testsets sample participant segments with non-empty ensemble labels. The human worksheet (`.txt`) stays frozen on disk; per-item metadata + content SHAs live in `testset_worksheets`/`testset_items`.
5. **Content-validity freeze**: Built from framework exemplar/subtle/adversarial utterances. Worksheet/definition-key `.txt` stay frozen on disk; manifest + items live in `cv_testsets`/`cv_testset_items` (the SHA there drives drift detection); AI answer keys are refreshable. Codebook CV is deferred (no exemplar utterances yet).
6. **Migration chain** (auto-run in `stage_ingest`, all idempotent): v2.0 monolithic `master_segments.jsonl` → per-session frozen segments; v2.5 → v3 directory layout; **v3-JSONL → SQLite** (`migrate_jsonl_to_sqlite` reads the old files, writes `qra.db` via temp-DB + atomic rename, relocates originals to `_legacy_files/`). `upgrade_config_file` additionally fills defaults for newly-added parameters in a legacy `qra_config.json` (non-destructive).
