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
python qra.py ingest -o ./data/output/                   # segment only
python qra.py classify -o ./data/output/ --what theme     # classify theme only
python qra.py assemble -o ./data/output/                  # join frozen+overlays

# Incremental data addition (segment + classify only NEW transcripts; re-assembles + re-analyzes)
python qra.py add-data --config ./data/output/02_meta/qra_config.json

# Speaker anonymization management
python qra.py apply-anonymization -o ./data/output/       # scrub PHI names from frozen segment text
python qra.py edit-anonymization -o ./data/output/        # edit speaker key + cascade downstream

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
- **Frozen segments** written to `01_transcripts/segmented/<sid>/segments.jsonl` (Phase 1)
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
- Assembles `master_segments.jsonl` with confidence tiering
- `run_full_pipeline` assembles here; standalone `qra assemble` joins frozen segments + classification overlays

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
- GNN representation/discovery layer (ON by default; figures + reports under `06_gnn/`): Capabilities A–E + the gate-gated LLM-free classifier (validation gate, abstention, calibration/OOD, propagation, scale-mode sim). Config-flag–driven advanced analyses: Track B model-counterfactual influence (`counterfactual`, gated), Track C MindfulBERT dataset builder (`build_mindfulbert_dataset`/`augmentation_enabled`), Track D subtext communities (`subtext_communities`). Full design record: `docs/GNN_MASTER_PLAN.md`
- Graph-ready CSVs, visualization figures (incl. GNN figures)
- Session and participant LLM summaries
- **Top-level synthesis written last**: `00_executive_summary.txt` (deterministic program-improvement brief), `00_READ_ME.txt`, `07_methods_appendix.txt`

## Framework Boundaries: Critical Design Rule

**VAAMR applies exclusively to participant segments. PURER applies exclusively to therapist segments.** These labels never cross. Therapist segments appear as read-only preceding context in classification prompts for participant segments, but are never themselves classified with VAAMR. This boundary is enforced by the speaker filter in each stage.

PURER moves frequently co-occur within a single therapist turn/cue-block. When a single label is required, an empirical precedence order is defined in `src/theme_framework/purer.py`:
- Reinforcement is often a wrapper (code the substantive inner move)
- Utilization > Reframing for forward-application prompts
- Reframing > Education when anchored to participant's specific story

## On-Disk Layout (Frozen Segments + Overlays)

Phase 1 froze per-session segments to `01_transcripts/segmented/<sid>/segments.jsonl`. Phase 3 separated classification results into overlay files at `02_meta/classifications/`:

```
output_dir/
├── 01_transcripts/
│   ├── diarized/            # Raw input copies (provenance)
│   ├── segmented/<sid>/     # FROZEN — raw segmentation only
│   │   ├── segments.jsonl
│   │   └── segmentation_meta.json
│   └── coded/               # Human-readable coded transcripts
├── 02_meta/
│   ├── classifications/     # Phase 3 overlays (refreshable)
│   │   ├── theme_labels.jsonl
│   │   ├── purer_labels.jsonl
│   │   ├── codebook_labels.jsonl
│   │   ├── cross_validation_labels.jsonl
│   │   └── classification_manifest.json
│   ├── auditable_logs/      # LLM prompts/responses, checkpoints
│   ├── codebook_raw/        # Codebook embedding checkpoints
│   ├── training_data/       # master_segments.jsonl/.csv, BERT training data
│   └── speaker_anonymization_key.json
├── 03_analysis_data/        # Session stats, graphing CSVs, per-{session,participant,theme} JSON
├── 04_validation/
│   ├── testsets/<name>/     # FROZEN — validation test sets (vaamr/purer/codebook)
│   ├── content_validity/<name>/  # FROZEN — content validity testsets
│   ├── cross_validation/    # Lift statistics
│   └── human_coding_evaluation_set.csv
├── 05_figures/              # PNG visualization figures (incl. gnn_*.png)
├── 06_reports/              # Human-readable text reports (tiered, numbered)
│   ├── 00_READ_ME.txt           # guide to the tree + reading order
│   ├── 00_executive_summary.txt # deterministic program-improvement brief
│   ├── 01_outcomes/             # progression_summary.txt, longitudinal.txt, avoidance_barrier.txt
│   ├── 02_mechanism/            # transitions, cue_response, purer, mechanism, language_atlas, superposition
│   ├── 03_per_session/          # _overview.txt, session_<id>.txt, session_summaries.txt
│   ├── 04_per_participant/      # participant_<id>.txt
│   ├── 05_per_stage/            # stage_<name>.txt, codebook_exemplars.txt
│   ├── 06_gnn/                  # validation, triangulation, emergent_motifs, coupling, construct_signal
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
| `qra.py` | CLI entry point (setup / run / analyze / ingest / classify / assemble / testset / cv) |
| `src/process/orchestrator.py` | Stage sequencing, `run_full_pipeline`, standalone `stage_*` functions |
| `src/process/config.py` | `PipelineConfig` dataclass — single source of truth for all settings |
| `src/process/segments_io.py` | Frozen per-session segment I/O, `params_hash`, `load_segments_for_stage` |
| `src/process/classifications_io.py` | Overlay read/write, provenance manifest (Phase 3) |
| `src/process/_freeze.py` | Freeze enforcement primitives (atomic write, SHA verification) |
| `src/process/legacy_migration.py` | Pre-modular project migration shim (legacy-only; safe to remove once no pre-v3 project dirs remain in use) |
| `src/process/transcript_ingestion.py` | Semantic segmentation logic, `ConversationalSegmenter` |
| `src/process/llm_segmentation.py` | LLM-assisted segmentation boundary refinement |
| `src/process/speaker_anonymization.py` | Persistent speaker ID mapping across runs |
| `src/process/speaker_filter.py` | Speaker inclusion/exclusion rules per classifier |
| `src/process/output_paths.py` | Single source of truth for ALL output directory paths |
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
| `src/classification_tools/llm_client.py` | Backend abstraction (OpenRouter, Ollama, LM Studio, HuggingFace, Replicate) |
| `src/classification_tools/classification_loop.py` | Multi-run consensus voting with checkpointing |
| `src/classification_tools/data_structures.py` | `Segment` dataclass |
| `src/analysis/runner.py` | Post-hoc analysis entry point |
| `src/analysis/purer_analysis.py` | PURER × VAAMR cue-block influence analysis |
| `src/analysis/purer_figures.py` | PURER × VAAMR lift heatmap and figures |
| `src/analysis/reports/` | Detailed text report generators (session, stage, transition, cue response, longitudinal, summaries) |
| `src/analysis/reports/executive_summary.py` | Deterministic `00_executive_summary.txt` program-improvement brief |
| `src/analysis/reports/reports_guide.py` | `00_READ_ME.txt` (report map) + `07_methods_appendix.txt` (how metrics are computed) |
| `src/analysis/efficacy.py` | Descriptive progression summary (ordinal-safe) + `efficacy_summary.json`; convergent-validity outcome linkage |
| `src/analysis/mechanism.py` | Signed Δprogression mechanism + bidirectional avoidance-barrier report |
| `src/gnn_layer/validation.py` | Graph reliability gate — κ(graph,LLM) / κ(graph,human), per-class breakdown, "ready for LLM-free scaling?" verdict |
| `src/gnn_layer/soft_labels.py` | Soft VAAMR supervision targets from multi-run ballots (mixture over 5 stages + E[stage] coordinate) |
| `src/gnn_layer/embeddings.py` | Qwen3 segment/anchor embeddings (reuses the VCE embedding path; cached) |
| `src/gnn_layer/coupling.py` | Inductive participant↔therapist coupling; latent cue-block factors vs forward movement |
| `src/gnn_layer/influence.py` | Track B — model-counterfactual influence (PURER-move swaps) + triangulation vs `mechanism.py` (gated on the reliability gate; sensitivity, not causal) |
| `src/gnn_layer/communities.py` | Track D — subtext-similarity graph, Louvain+hierarchical communities, within-session routine transitions, participant-bootstrap stability selection |
| `src/gnn_layer/calibration.py` | Track A3 — temperature scaling + ECE + OOD score (domain-shift confidence) |
| `src/gnn_layer/propagation.py` | Track A4 — measured post-training soft-label diffusion |
| `src/gnn_layer/figures.py` | GNN figures (validation κ, motif influence, coupling factors) |
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

1. **Frozen segmentation**: Once segmented, segments are frozen to `01_transcripts/segmented/<sid>/` and never rewritten (no `master_segments.jsonl` mutation on re-runs). Only raw-segmentation fields are persisted (no classification data).
2. **Classification overlays**: Re-running a classifier overwrites only its `02_meta/classifications/<key>_labels.jsonl` overlay. Frozen segments remain untouched.
3. **Multi-kind test sets**: VAAMR testsets sample participant segments (stratified by cohort). PURER testsets sample therapist segments with non-null labels (stratified by PURER primary). Codebook testsets sample participant segments with non-empty ensemble labels.
4. **Content-validity freeze**: Content-validity testsets are built from framework exemplar/subtle/adversarial utterances. Human worksheets and definition keys are frozen; AI answer keys are refreshable. Codebook CV is deferred (no exemplar utterances yet).
5. **Legacy migration**: Pre-modular project directories with `master_segments.jsonl` but no `01_transcripts/segmented/` are auto-migrated on first run via `src/process/legacy_migration.py`.
