# QRA: Qualitative Research Algorithm — Usage Guide

A comprehensive LLM-based classification pipeline for analyzing therapeutic dialogue transcripts. QRA applies bilateral classification frameworks: VAAMR (Vigilance, Avoidance, Attention Regulation, Metacognition, Reappraisal) classifies participant segments, PURER classifies therapist segments at the cue-block level, and the VCE phenomenology codebook provides optional multi-label construct enrichment.

## Overview

QRA is designed for qualitative research in psychotherapy and mindfulness-based interventions. It takes diarized transcripts (typically from speech-to-text pipelines like Whisper with speaker diarization) and produces structured, coded datasets suitable for statistical analysis and thematic interpretation.

### Key Capabilities

- **Semantic Segmentation**: Embedding-based segmentation with adaptive thresholds, topic clustering, and optional LLM-assisted boundary refinement; frozen to per-session files that are never rewritten
- **VAAMR Theme Classification**: Multi-run zero-shot LLM classification (participant segments) with per-run model rotation and consensus voting
- **PURER Cue-Block Classification**: Therapist dialogue classified at the cue-unit level (between participant turns), with configurable context window
- **Codebook Classification**: Multi-label coding via embedding similarity + LLM zero-shot prompting, reconciled by ensemble
- **Classification Overlays (Phase 3)**: Per-classifier results stored as independent overlay files at `02_meta/classifications/`; re-classification never touches frozen segments
- **Frozen Validation Test Sets (Phase 2)**: Multi-kind (VAAMR/PURER/codebook) stratified samples with frozen human worksheets and refreshable AI answer keys
- **Frozen Content-Validity Testsets (Phase 2)**: Built from framework exemplar/subtle/adversarial utterances (VAAMR and PURER); codebook deferred
- **Therapist Cue Analysis**: Surfaces therapist dialogue at stage transitions for dyadic interpretation
- **Automated Analysis**: Longitudinal reports, per-participant summaries, figures, graph-ready CSVs

## Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Transcript Ingestion & Segmentation                │
│  - Load diarized transcripts (JSON/VTT)                     │
│  - Semantic segmentation via sentence-transformer           │
│  - Adaptive threshold + topic clustering                    │
│  - Optional LLM-assisted boundary refinement                │
│  - Speaker normalization and anonymization                  │
│  - Therapist segments extracted and interleaved             │
│  - FROZEN to 01_transcripts/segmented/<sid>/segments.jsonl │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 2: Construct Operationalization                      │
│  - Build theme framework definitions                        │
│  - Export theme definitions (JSON + txt)                    │
│  - Create content validity test set, worksheets, keys       │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 3: Zero-Shot LLM Theme Classification (VAAMR)        │
│  - Multi-run classification with checker model rotation     │
│  - Context-aware prompting (preceding participant segs)     │
│  - Multi-run consensus voting (High/Medium/Low tiers)      │
│  - Speaker filter: therapist segments excluded              │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 3c: PURER Cue-Unit Classification (optional)         │
│  - One label per therapist response between participant     │
│    turns (cue-block level)                                  │
│  - Wider context window (6 preceding segments)              │
│  - Can skip lesson-content (configurable word threshold)    │
│  - Single-run classification (no multi-model IRR needed)    │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 3b: Codebook Classification (optional)               │
│  - Embedding-based similarity scoring                       │
│  - LLM zero-shot multi-label coding                         │
│  - Ensemble reconciliation of both methods                  │
│  - GPU memory hand-off from segmenter                       │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 4: Cross-Validation (optional, requires 3b)          │
│  - Theme ↔ codebook co-occurrence analysis                  │
│  - Lift and statistical validation of hypotheses            │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 5: Human Validation Set                              │
│  - Create balanced evaluation set for human coding          │
│  - Export evaluation set CSV                                │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 6: Dataset Assembly                                  │
│  - Assemble master segment dataset (JSONL) with confidence  │
│  - Standalone: joins frozen segments + overlays from disk   │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 7: Report Generation                                 │
│  - Coded transcripts (per-session)                          │
│  - Human blind-coding forms                                 │
│  - Validation test sets (VAAMR/PURER/codebook)              │
│  - Content-validity testsets (VAAMR/PURER)                  │
│  - Speaker anonymization key                                │
│  - Per-transcript stats + cumulative report                 │
│  - Training data export                                     │
│  - Output directory index (00_index.txt)                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 8: Results Analysis (optional, --auto-analyze)       │
│  - Per-participant longitudinal reports                     │
│  - Per-session and per-theme analyses                       │
│  - Graph-ready CSVs                                         │
│  - Stage progression + transition explanation               │
│  - Therapist cue response analysis                          │
│  - Session + participant LLM summaries                      │
│  - Visualization figures                                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Diarized transcripts (JSON or VTT)
2. **Ingest**: Segments frozen to `01_transcripts/segmented/<sid>/segments.jsonl` (once)
3. **Classify**: Each classifier writes its own overlay at `02_meta/classifications/<key>_labels.jsonl` (re-runnable)
4. **Assemble**: Joins frozen segments + overlays into `master_segments.jsonl`
5. **Output**: Master dataset, coded transcripts, analysis reports, testsets

## Installation & Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd Qualitative_Research_Algorithm

# Install dependencies (recommended in virtual environment)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

See [SETUP.md](SETUP.md) for detailed installation and configuration instructions.

## Usage

### Command-Line Interface

```bash
python qra.py <command> [options]
```

#### Full Pipeline Commands

| Command | Description |
|---------|-------------|
| `setup` | Interactive configuration wizard (14 steps) |
| `run` | Execute complete pipeline (orchestrates ingest → classify → assemble → analyze) |
| `analyze` | Post-hoc results analysis on existing pipeline output |

#### Modular Stage Commands (Phase 3)

| Command | Description |
|---------|-------------|
| `ingest` | Segment transcripts and freeze to disk only |
| `classify` | Run classifier(s) on frozen segments, write overlay files |
| `assemble` | Join frozen segments + overlays into `master_segments.jsonl` |

#### Validation Test Set Commands (Phase 2)

| Command | Description |
|---------|-------------|
| `testset create` | Create a new frozen validation testset |
| `testset refresh` | Refresh AI answer key(s) for existing testsets |
| `testset list` | List existing frozen testsets |
| `testsets` | **[DEPRECATED]** Use `testset refresh --all` instead |
| `cv create` | Create a content-validity testset |
| `cv refresh` | Refresh content-validity AI answer key(s) |
| `cv list` | List existing content-validity testsets |

### Setup Wizard

```bash
python qra.py setup
```

Runs an interactive configuration wizard (14 steps) that:

1. Configures input/output paths and trial metadata
2. Imports speaker anonymization key (optional)
3. Identifies therapist vs participant speakers
4. Sets segmentation parameters (including LLM refinement options)
5. Prompts for PURER therapist classification enablement + cue options
6. Selects LLM backend and model for VAAMR classification
7. Chooses theme framework and optional codebook
8. Configures classification parameters (n_runs, temperature)
9. Sets confidence thresholds
10. Configures validation test sets
11. Enables post-pipeline analysis
12. Configures therapist cue summarization
13. Configures session and participant summary reports
14. Saves configuration to `02_meta/qra_config.json`

### Running the Pipeline

```bash
# With saved config
python qra.py run --config ./data/output/02_meta/qra_config.json

# With inline configuration
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --transcript-dir ./data/input/ \
  --output-dir ./data/output/

# Run pipeline and automatically generate analysis reports
python qra.py run --config ./qra_config.json --auto-analyze
```

### Modular Stage Usage

```bash
# Segment only (freeze segments to disk)
python qra.py ingest -o ./data/output/

# Force re-segmentation of one session
python qra.py ingest -o ./data/output/ --reingest c1s1

# Force re-segmentation of all sessions
python qra.py ingest -o ./data/output/ --reingest-all

# Classify theme only (overwrites theme overlay)
python qra.py classify -o ./data/output/ --what theme

# Classify specific classifier
python qra.py classify -o ./data/output/ --what purer

# Join frozen segments + overlays into master_segments.jsonl
python qra.py assemble -o ./data/output/
```

### Test Set Management

```bash
# Create a PURER validation testset
python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1

# Create a VAAMR validation testset with custom parameters
python qra.py testset create -o ./data/output/ --kind vaamr --name vaamr_set_1 --fraction 0.15

# Refresh AI answer keys for all existing testsets
python qra.py testset refresh -o ./data/output/ --all

# Refresh a specific testset's AI answer key
python qra.py testset refresh -o ./data/output/ --name vaamr_testset_1

# List all existing testsets
python qra.py testset list -o ./data/output/

# Create a PURER content-validity testset
python qra.py cv create -o ./data/output/ --framework purer --name cv_purer_v1

# Refresh all content-validity AI answer keys
python qra.py cv refresh -o ./data/output/ --all

# List content-validity testsets
python qra.py cv list -o ./data/output/
```

### Post-Hoc Analysis

```bash
python qra.py analyze --output-dir ./data/output/
```

Generates:
- Per-participant longitudinal reports (VAAMR stage progression)
- Per-session summaries with prototypical exemplars
- Per-theme (stage + codebook) analyses
- Therapist cue response analysis
- Session and participant LLM summaries
- Graph-ready CSVs and visualization figures

## Configuration

### Common Arguments (`run` subcommand)

| Argument | Description |
|----------|-------------|
| `--config, -c` | Path to saved config JSON |
| `--transcript-dir` | Directory containing input transcripts |
| `--output-dir, -o` | Output directory for results |
| `--trial-id` | Trial identifier |
| `--resume-from` | Checkpoint resume path |
| `--auto-analyze` | Run analysis automatically after pipeline |

### Backend & Model Arguments

| Argument | Description |
|----------|-------------|
| `--backend` | LLM backend: openrouter, replicate, huggingface, ollama, lmstudio |
| `--lmstudio-url` | LM Studio server URL (default: http://127.0.0.1:1234/v1) |
| `--model` | Primary model ID for classification |
| `--models` | Multiple model IDs for cross-referencing (per_run_models) |
| `--api-key` | OpenRouter API key (or OPENROUTER_API_KEY env var) |
| `--replicate-api-token` | Replicate API token (or REPLICATE_API_TOKEN env var) |

### Feature Flags

| Argument | Description |
|----------|-------------|
| `--no-theme-labeler` | Skip theme classification |
| `--run-codebook-classifier` | Enable codebook classification |
| `--no-codebook-classifier` | Disable codebook classification |
| `--verbose-segmentation` | Write detailed process log |

### Classification Parameters

| Argument | Description |
|----------|-------------|
| `--n-runs` | Number of classification runs per segment |
| `--temperature` | LLM temperature |
| `--high-confidence-threshold` | High confidence tier threshold |
| `--medium-confidence-threshold` | Medium confidence tier threshold |

### Speaker Filtering

| Argument | Description |
|----------|-------------|
| `--speaker-filter-mode` | `none` (classify all) or `exclude` (drop listed speakers) |
| `--exclude-speakers` | Speaker labels to exclude |

## Output Directory Structure

After a complete pipeline run, the output directory:

```
output_dir/
├── 00_index.txt                              # Auto-generated file index
├── 01_transcripts/
│   ├── diarized/                             # Raw input copies (provenance)
│   ├── segmented/<sid>/                      # FROZEN raw segmentation
│   │   ├── segments.jsonl
│   │   └── segmentation_meta.json
│   └── coded/                                # Human-readable coded transcripts
│       └── coded_transcript_<session>.txt
├── 02_meta/
│   ├── classifications/                      # Phase 3 overlays (refreshable)
│   │   ├── theme_labels.jsonl
│   │   ├── purer_labels.jsonl
│   │   ├── codebook_labels.jsonl
│   │   ├── cross_validation_labels.jsonl
│   │   └── classification_manifest.json
│   ├── auditable_logs/                       # LLM prompts/responses, checkpoints
│   │   ├── llm_prompts.txt
│   │   ├── llm_classification_log.txt
│   │   ├── checkpoints/
│   │   ├── theme_definitions.json
│   │   └── segmentation_process_log.txt
│   ├── codebook_raw/                         # Codebook embedding checkpoints
│   ├── training_data/                        # master_segments.jsonl/.csv
│   │   ├── master_segments.jsonl
│   │   ├── theme_classification.jsonl
│   │   └── codebook_multilabel.jsonl
│   ├── speaker_anonymization_key.json
│   ├── theme_definitions.txt
│   └── anonymization_key.txt
├── 03_analysis_data/
│   ├── session_stats/stats_<session>.json
│   ├── graphing/*.csv                        # Graph-ready CSVs
│   ├── per_session/<session>.json
│   ├── per_participant/<participant>.json
│   ├── per_theme/<stage>.json
│   ├── cumulative_report.json
│   ├── longitudinal_summary.json
│   ├── session_stage_progression.csv
│   └── session_summaries.json
├── 04_validation/
│   ├── testsets/<name>/                      # FROZEN validation test sets
│   │   ├── manifest.json
│   │   ├── segments_snapshot.jsonl
│   │   ├── human_worksheet.txt               # Frozen
│   │   └── AI_answer_key.txt                 # Refreshable
│   ├── content_validity/<name>/              # FROZEN content-validity testsets
│   │   ├── manifest.json
│   │   ├── items.jsonl
│   │   ├── human_worksheet.txt               # Frozen
│   │   ├── definition_key.txt                # Frozen
│   │   └── AI_answer_key.txt                 # Refreshable
│   ├── cross_validation/
│   │   ├── cross_validation_results.json
│   │   └── top_theme_code_associations.json
│   ├── human_coding_evaluation_set.csv
│   ├── human_classification_<session>.txt
│   ├── content_validity_test_set.jsonl
│   └── flagged_for_review.txt
├── 05_figures/*.png                          # Visualization figures
├── 06_reports/
│   └── per_theme/                            # Theme/stage text reports
└── purer_classification_error.txt            # (if PURER fails)
```

## LLM Backend Configuration

### LM Studio (Local)
```bash
python qra.py run --backend lmstudio --model nvidia/nemotron-3-super
```

### OpenRouter
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python qra.py run --backend openrouter --model openai/gpt-4o
```

### HuggingFace (Local GPU)
```bash
python qra.py run --backend huggingface --model meta-llama/Llama-4-Maverick-17B-128E-Instruct
```

### Ollama (Local)
```bash
ollama pull llama3
python qra.py run --backend ollama --model llama3
```

## Confidence Tiers

| Tier | Consistency | Confidence | Description |
|------|-------------|------------|-------------|
| **High** | unanimous | >0.8 | All raters agree with high confidence |
| **Medium** | majority | >0.6 | Majority agreement or good single-run confidence |
| **Low** | minority | <0.6 | Split votes or low confidence |
| **Unclassified** | none | — | No consensus reached |

## Module Reference

### Classification Tools (`classification_tools/`)

| File | Description |
|------|-------------|
| `llm_client.py` | Unified LLM API client (OpenRouter, Replicate, Ollama, LM Studio, HuggingFace) |
| `data_structures.py` | `Segment` dataclass and core data structures |
| `llm_classifier.py` | Theme and codebook LLM classification logic |
| `majority_vote.py` | Interrater reliability voting aggregation |
| `response_parser.py` | Parse LLM outputs into structured format |
| `reliability.py` | Reliability metrics (Krippendorff's alpha, etc.) |
| `validation.py` | Evaluation set creation and consistency checking |
| `model_loader.py` | HuggingFace model downloading and loading |
| `classification_loop.py` | Classification loop with checkpointing |

### Process (`process/`)

| File | Description |
|------|-------------|
| `orchestrator.py` | Stage sequencing, `run_full_pipeline`, standalone `stage_*` functions |
| `config.py` | `PipelineConfig` and sub-config dataclasses |
| `setup_wizard.py` | Interactive 14-step configuration wizard |
| `segments_io.py` | Frozen segment I/O, `params_hash`, `load_segments_for_stage` |
| `classifications_io.py` | Overlay read/write, provenance manifest (Phase 3) |
| `_freeze.py` | Freeze enforcement (atomic write, SHA verification) |
| `legacy_migration.py` | Pre-modular project migration shim |
| `transcript_ingestion.py` | Load VTT/JSON, `ConversationalSegmenter` |
| `llm_segmentation.py` | LLM-assisted segmentation boundary refinement |
| `speaker_anonymization.py` | Persistent speaker ID mapping |
| `speaker_filter.py` | Speaker inclusion/exclusion rules |
| `output_paths.py` | ALL output directory paths (single source of truth) |
| `output_index.py` | `00_index.txt` generation |
| `cross_validation.py` | VAAMR × VCE lift statistics |
| `validation_exports.py` | Validation artifact export helpers |
| `process_logger.py` | Verbose LLM I/O logging |

### Analysis (`analysis/`)

| File | Description |
|------|-------------|
| `runner.py` | Post-hoc analysis orchestrator |
| `loader.py` | Load master JSONL and framework from output directory |
| `participant.py` | Per-participant report generation |
| `session.py` | Per-session analysis |
| `theme.py` | Per-theme (VAAMR stage + code) analyses |
| `stage_progression.py` | Session-level stage progression computation |
| `longitudinal.py` | Longitudinal summary generation |
| `figure_data.py` | Export graph-ready CSV datasets |
| `figures.py` | Matplotlib visualization figures |
| `exemplars.py` | Exemplar utterance extraction per stage |
| `text_reports.py` | Human-readable text report utilities |
| `purer_analysis.py` | PURER × VAAMR influence analysis |
| `purer_figures.py` | PURER × VAAMR lift heatmap and figures |
| `reports/` | Detail report generators (session, stage, transition, cue response, longitudinal, summaries) |

### Assembly (`process/assembly/`)

| File | Description |
|------|-------------|
| `__init__.py` | Module exports with Phase 1 back-compat aliases |
| `master_dataset.py` | `assemble_master_dataset` |
| `human_forms.py` | Human classification forms, test set freeze/refresh |
| `content_validity.py` | Content-validity test set freeze/refresh (VAAMR/PURER; codebook deferred) |
| `coded_transcripts.py` | Per-session coded transcript writer |
| `stats_reports.py` | Per-transcript stats, cumulative report |
| `training_export.py` | Training data, theme definitions, content validity item export |
| `_shared.py` | Shared helpers for assembly functions |

## Citation

```bibtex
@software{QRA2026,
  title  = {QRA: Qualitative Research Algorithm},
  author = {Balsamo, Wade and Wexler, Ryan S.},
  year   = {2026},
  note   = {Computational phenomenology pipeline for mindfulness-based intervention research}
}
```

## License

MIT License — see [LICENSE](LICENSE).
