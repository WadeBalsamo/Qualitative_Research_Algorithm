# QRA: Qualitative Research Algorithm — Usage Guide

A comprehensive LLM-based classification pipeline for analyzing therapeutic dialogue transcripts. QRA applies bilateral classification frameworks: VAAMR (Vigilance, Avoidance, Attention Regulation, Metacognition, Reappraisal) classifies participant segments, PURER classifies therapist segments at the cue-block level, and the VCE phenomenology codebook provides optional multi-label construct enrichment.

## Getting Started: Creating a New Project

### Step 1: Setup the Configuration Wizard
Run the interactive configuration wizard to set up your project:

```bash
python qra.py setup
```

This wizard guides you through:
- Setting input/output directories
- Configuring LLM backend and model selection
- Determining which frameworks to use (VAAMR, PURER, VCE)
- Setting classification parameters
- Configuring validation test sets
- Enabling automatic analysis

### Step 2: Prepare Input Data
Place your diarized transcripts in the input directory (configured during setup). 
Supported formats: JSON or VTT files from speech-to-text pipelines like Whisper with speaker diarization.

### Step 3: Run the Full Pipeline
With your config saved, run the complete pipeline:

```bash
# Using saved configuration
python qra.py run --config ./data/output/02_meta/qra_config.json

# Or with inline options
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --transcript-dir ./data/input/ \
  --output-dir ./data/output/
```

## Quick Commands for Common Tasks

| Command | Description |
|---------|-------------|
| `python qra.py setup` | Interactive configuration wizard |
| `python qra.py run --config config.json` | Execute full pipeline |
| `python qra.py run --config config.json --auto-analyze` | Run pipeline with automatic analysis reports |
| `python qra.py ingest -o ./output/` | Segment and freeze transcripts only |
| `python qra.py classify -o ./output/ --what theme` | Run VAAMR classification only |
| `python qra.py analyze -o ./output/` | Generate analysis reports on existing output |

## Workflow for Existing Projects

To modify an existing project, you can:
1. Run with existing configuration to update specific components
2. Re-run specific stages with modifiers to adjust or extend analysis
3. Add new classification layers (e.g., additional codebooks, different classifiers)

Example of re-running with updated parameters:
```bash
python qra.py run --config ./qra_config.json --n-runs 5
```

## Key Features and Capabilities

### 1. **Semantic Segmentation** 
- Embedding-based segmentation with adaptive thresholds
- Topic clustering and LLM-assisted boundary refinement
- Frozen to per-session files that are never rewritten

### 2. **VAAMR Theme Classification (Participant Segments)**
- Multi-run classification with per-run model rotation and consensus voting
- Standardizes process with a confidence tiering system (High/Medium/Low)
- Configurable with different number of runs (n_runs parameter)

### 3. **PURER Cue-Block Classification (Therapist Segments)**
- Therapist dialogue classified at the cue-unit level (between participant turns)
- Configurable context window (default: 6 preceding segments)
- Outputs PURER move type classification (Phenomenology, Utilization, Reframing, Education/Expectancy, Reinforcement)

### 4. **Codebook Classification (Multi-label Phenomenology)**
- Multi-label coding via embedding similarity + LLM zero-shot prompting
- Ensemble reconciliation of both methods for better accuracy
- Supports the 59-code Varieties of Contemplative Experience (VCE) codebook

### 5. **Classification Overlays**
- Per-classifier results stored as independent overlay files at `02_meta/classifications/`
- Re-classification never touches frozen segments
- Enables post-hoc re-analysis without reprocessing entire transcripts

### 6. **Frozen Validation Test Sets**
- Multi-kind (VAAMR/PURER/codebook) stratified samples with frozen human worksheets
- Refreshable AI answer keys for validation sets
- Supports inter-rater reliability testing

### 7. **Content-Validity Testsets**
- Built from framework exemplar/subtle/adversarial utterances
- VAAMR and PURER content-validity test sets currently supported
- Codebook implementation deferred for future development

### 8. **Therapist Cue Analysis**
- Surfaces therapist dialogue at stage transitions for dyadic interpretation
- Analyzes therapist language by transition type
- Produces PURER move distributions for different transition types

### 9. **Automated Analysis**
- Longitudinal reports and per-participant summaries
- Session and theme analyses
- Graph-ready CSVs and visualization figures

## Adding Another Layer of Classification

For extending your analysis with new classification layers:

### Option 1: Using the Configuration Wizard
Run `python qra.py setup` again to modify project settings and add new classifiers.

### Option 2: Manual Configuration Changes
Update your existing `qra_config.json` file:
- Add new classifier settings to the appropriate section
- Update the `run_codebook_classifier` flag to enable additional codebooks
- Modify framework specifications in the configuration

### Option 3: Modular Classification
Run specific classification stages:
```bash
# Add a new classification layer
python qra.py classify -o ./output/ --what codebook

# Re-run existing classifiers for consistency
python qra.py classify -o ./output/ --what theme
```

### Example: Adding a Custom Codebook or Additional Framework Analysis
To add any additional layer of classification beyond what's already supported:

1. **Via Configuration**: Update your `qra_config.json` to include new framework specifications
2. **Via Command Line**: Run modular classification stages to add new overlays to existing data
3. **Pipeline Extension**: Add new entries to the classification pipeline configuration by updating the relevant framework definitions

The system is designed to support:
- Multiple codebooks (VCE and custom)
- Additional classification frameworks
- Ensemble methods combining different classifier outputs
- Post-hoc re-analysis of any subset of classifications

All classification overlays are stored independently in `02_meta/classifications/`, allowing for flexible re-analysis without reprocessing frozen segments.

## Output Directory Structure

After a complete pipeline run, the output directory contains:

```
output_dir/
├── 00_index.txt                              # Auto-generated file index
├── 01_transcripts/
│   ├── diarized/                             # Raw input copies (provenance)
│   ├── segmented/<sid>/                      # FROZEN raw segmentation
│   │   ├── segments.jsonl
│   │   └── segmentation_meta.json
│   └── coded/                                # Human-readable coded transcripts
├── 02_meta/
│   ├── classifications/                      # Phase 3 overlays (refreshable)
│   │   ├── theme_labels.jsonl
│   │   ├── purer_labels.jsonl
│   │   ├── codebook_labels.jsonl
│   │   ├── cross_validation_labels.jsonl
│   │   └── classification_manifest.json
│   ├── auditable_logs/                       # LLM prompts/responses, checkpoints
│   ├── codebook_raw/                         # Codebook embedding checkpoints
│   ├── training_data/                        # master_segments.jsonl/.csv
│   └── speaker_anonymization_key.json
├── 03_analysis_data/
│   ├── session_stats/
│   ├── graphing/*.csv                        # Graph-ready CSVs
│   ├── per_session/<session>.json
│   ├── per_participant/<participant>.json
│   ├── per_theme/<stage>.json
│   └── cumulative_report.json
├── 04_validation/
│   ├── testsets/<name>/                      # FROZEN validation test sets
│   ├── content_validity/<name>/              # FROZEN content-validity testsets
│   ├── cross_validation/                     # Lift statistics
│   └── human_coding_evaluation_set.csv
├── 05_figures/*.png                          # Visualization figures
├── 06_reports/
└── 07_meta/                                  # Legacy directory
```

## Detailed Usage by Command

### Setup Wizard
```bash
python qra.py setup
```
Interactive 14-step configuration that:
1. Configures input/output paths 
2. Identifies therapist vs participant speakers
3. Sets segmentation parameters
4. Sets LLM backend and model for VAAMR classification
5. Selects frameworks (VAAMR, PURER, VCE)
6. Configures validation test sets
7. Enables post-pipeline analysis
8. Saves configuration to `02_meta/qra_config.json`

### Full Pipeline Execution
```bash
# With saved config
python qra.py run --config ./data/output/02_meta/qra_config.json

# With inline configuration
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --transcript-dir ./data/input/ \
  --output-dir ./data/output/
```

### Modular Stages

#### Ingest Only (Freeze Segments Only)
```bash
python qra.py ingest -o ./data/output/
```

#### Theme Classification Only
```bash
python qra.py classify -o ./data/output/ --what theme
```

#### PURER Classification Only 
```bash
python qra.py classify -o ./data/output/ --what purer
```

#### Codebook Classification Only
```bash
python qra.py classify -o ./data/output/ --what codebook
```

#### Dataset Assembly Only
```bash
python qra.py assemble -o ./data/output/
```

### Post-Hoc Analysis
```bash
python qra.py analyze --output-dir ./data/output/
```

Generates comprehensive analysis including:
- Per-participant longitudinal reports
- Per-session summaries with prototypical exemplars
- Per-theme (stage + codebook) analyses
- Therapist cue response analysis
- Session and participant LLM summaries
- Graph-ready CSVs and visualization figures

## Test Set Management

### Create a New Validation Test Set
```bash
# Create PURER validation testset
python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1

# Create VAAMR validation testset with custom parameters
python qra.py testset create -o ./data/output/ --kind vaamr --name vaamr_set_1 --fraction 0.15
```

### Refresh AI Answer Keys
```bash
# Refresh all existing testsets at once
python qra.py testset refresh -o ./data/output/ --all

# Refresh a specific testset
python qra.py testset refresh -o ./data/output/ --name vaamr_testset_1
```

### Create Content-Validity Test Sets
```bash
# Create PURER content-validity testset
python qra.py cv create -o ./data/output/ --framework purer --name cv_purer_v1
```

## Configuration Options

### Backend & Model Selection
```bash
# LM Studio (Local)
python qra.py run --backend lmstudio --model nvidia/nemotron-3-super

# OpenRouter
export OPENROUTER_API_KEY=sk-or-v1-...
python qra.py run --backend openrouter --model openai/gpt-4o

# Ollama (Local)
ollama pull llama3
python qra.py run --backend ollama --model llama3
```

### Feature Flags
```bash
# Skip theme classification
python qra.py run --no-theme-labeler

# Enable codebook classification (if configured)
python qra.py run --run-codebook-classifier

# Disable specific features
python qra.py run --no-codebook-classifier
```

### Classification Parameters
```bash
# Set number of classification runs (default: 1)
python qra.py run --n-runs 3

# Set LLM temperature (default: 0.0)
python qra.py run --temperature 0.7

# Enable automatic analysis
python qra.py run --auto-analyze

# Resume from a checkpoint
python qra.py run --resume-from ./checkpoints/last_run
```

### Confidence Tier System

| Tier | Consistency | Confidence | Description |
|------|-------------|------------|-------------|
| **High** | unanimous | >0.8 | All raters agree with high confidence |
| **Medium** | majority | >0.6 | Majority agreement or good single-run confidence |
| **Low** | minority | <0.6 | Split votes or low confidence |
| **Unclassified** | none | — | No consensus reached |

## Command-Line Interface Reference

### Main Commands
| Command | Description |
|---------|-------------|
| `setup` | Interactive configuration wizard |
| `run` | Execute complete pipeline |
| `analyze` | Post-hoc results analysis |
| `ingest` | Segment transcripts only |
| `classify` | Run classifiers only |
| `assemble` | Join segments and overlays |
| `testset create` | Create new validation set |
| `testset refresh` | Update AI answer keys |
| `testset list` | List existing test sets |
| `cv create` | Create content-validity testset |
| `cv refresh` | Refresh content-validity AI answer keys |
| `cv list` | List content-validity testsets |

### Advanced Options
```bash
# Re-segment specific session
python qra.py ingest -o ./output/ --reingest c1s1

# Re-segment all sessions
python qra.py ingest -o ./output/ --reingest-all

# Force re-classification
python qra.py classify -o ./output/ --what theme --force

# Verbose segmentation logging
python qra.py run --verbose-segmentation

# Zero-shot content-validity test (skips full pipeline)
python qra.py run --test-zeroshot --preset small --output-dir ./data/output/
```

---

## Architecture

### Pipeline Stages

QRA is designed for qualitative research in psychotherapy and mindfulness-based interventions. It takes diarized transcripts (typically from speech-to-text pipelines like Whisper with speaker diarization) and produces structured, coded datasets suitable for statistical analysis and thematic interpretation.

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Transcript Ingestion & Segmentation               │
│  - Load diarized transcripts (JSON/VTT)                     │
│  - Semantic segmentation via sentence-transformer           │
│  - Adaptive threshold + topic clustering                    │
│  - Optional LLM-assisted boundary refinement                │
│  - Speaker normalization and anonymization                  │
│  - Therapist segments extracted and interleaved             │
│  - FROZEN to 01_transcripts/segmented/<sid>/segments.jsonl  │
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
│  Stage 3:           LLM Theme Classification (VAAMR)        │
│  - Multi-run classification with checker model rotation     │
│  - Context-aware prompting (preceding participant segs)     │
│  - Multi-run consensus voting (High/Medium/Low tiers)       │
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
2. **Ingest** (Stage 1): Segments frozen to `01_transcripts/segmented/<sid>/segments.jsonl` (once; never rewritten)
3. **Classify** (Stage 3 / 3b / 3c): Each classifier writes its own overlay at `02_meta/classifications/<key>_labels.jsonl` (re-runnable independently)
4. **Validate** (Stage 2 / 4 / 5): Content-validity testsets, cross-validation, human evaluation sets
5. **Assemble** (Stage 6): Joins frozen segments + overlays into `master_segments.jsonl`
6. **Report** (Stage 7): Coded transcripts, test sets, stats, training data
7. **Analyze** (Stage 8): Longitudinal reports, figures, graph-ready CSVs, summaries

## Input Data Reference

### Supported Transcript Formats

| Format | Extension | Source | Description |
|--------|-----------|--------|-------------|
| JSON | `.json` | Custom diarization pipeline | Array of utterance objects with `speaker`, `text`, `start`, `end` fields |
| VTT | `.vtt` | Whisper + speaker diarization | WebVTT format with speaker labels in cues |

### Required Fields (JSON input)

Each utterance object should contain:

```json
{
  "speaker": "Participant_01",    // Speaker identifier
  "text": "Utterance text...",    // Spoken content
  "start": 0.0,                   // Start time in seconds
  "end": 5.2                      // End time in seconds
}
```

### Pipeline Configuration (qra_config.json)

The configuration JSON supports the following top-level fields. Sub-configs correspond to dataclass fields in `process/config.py`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `transcript_dir` | string | `./data/input/` | Input transcript directory |
| `output_dir` | string | `./data/output/` | Output root directory |
| `trial_id` | string | `standard` | Trial identifier |
| `run_theme_labeler` | bool | true | Enable VAAMR theme classification |
| `run_purer_labeler` | bool | true | Enable PURER therapist classification |
| `run_codebook_classifier` | bool | false | Enable VCE codebook classification |
| `auto_analyze` | bool | true | Run analysis automatically after pipeline |
| `resume_from` | string | null | Checkpoint path for resuming |
| `speaker_anonymization_key_path` | string | null | Path to pre-existing anonymization key |

#### Sub-config: `segmentation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `embedding_model` | string | `Qwen/Qwen3-Embedding-8B` | Sentence-transformer embedding model |
| `silence_threshold_ms` | int | 1500 | Silence gap threshold for utterance grouping |
| `semantic_shift_percentile` | int | 25 | Percentile for semantic shift boundary detection |
| `min_segment_words_conversational` | int | 60 | Minimum words per conversational segment |
| `max_segment_words_conversational` | int | 500 | Maximum words per conversational segment |
| `max_gap_seconds` | float | 15.0 | Max time gap (seconds) to group utterances |
| `min_words_per_sentence` | int | 20 | Sentences below this folded into adjacent same-speaker sentence |
| `max_segment_duration_seconds` | float | 60.0 | Max duration of a single segment in seconds |
| `use_adaptive_threshold` | bool | true | Use local-minima detection instead of static percentile |
| `min_prominence` | float | 0.05 | Minimum prominence for adaptive threshold peaks |
| `broad_window_size` | int | 7 | Window size for broad similarity curve |
| `use_topic_clustering` | bool | true | Use AgglomerativeClustering for topic boundaries |
| `use_llm_refinement` | bool | true | Enable LLM-assisted boundary refinement |
| `llm_refinement_mode` | string | `full` | Mode: `boundary_review`, `context_expansion`, `coherence_check`, `full` |
| `llm_ambiguity_threshold` | float | 0.15 | Similarity proximity for ambiguous boundaries |
| `llm_batch_size` | int | 5 | Boundaries/pairs per LLM call |
| `verbose_segmentation` | bool | true | Write detailed process log |

#### Sub-config: `speaker_filter`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | string | `none` | Filter mode: `none` (classify all) or `exclude` (drop listed speakers) |
| `speakers` | string[] | `[]` | Speaker labels to exclude when mode is `exclude` |

#### Sub-config: `theme_classification` / `purer_classification`

Both used for LLM classification. `purer_classification` defaults to `context_window_segments: 6` while `theme_classification` defaults to `2`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `backend` | string | `lmstudio` | LLM backend: `openrouter`, `ollama`, `lmstudio` |
| `model` | string | `nvidia/nemotron-3-super` | Primary model ID |
| `summarization_model` | string | `nvidia/nemotron-3-nano-4b` | Lighter model for summary generation |
| `models` | string[] | `[]` | Additional model IDs for cross-referencing |
| `per_run_models` | string[] | `[]` | Per-run model assignment (one per run when n_runs > 1) |
| `temperature` | float | 0.0 | LLM sampling temperature |
| `n_runs` | int | 1 | Number of classification runs per segment |
| `max_new_tokens` | int | 512 | Max tokens in LLM response |
| `context_window_segments` | int | 2 (theme) / 6 (purer) | Number of preceding segments included as context |
| `randomize_codebook` | bool | true | Randomize definition order in prompts |
| `zero_shot_prompt` | bool | false | Definitions only, no examples |
| `prompt_n_exemplars` | int | null | Number of exemplar utterances to include (null = all) |
| `prompt_include_subtle` | bool | true | Include subtle/difficult examples |
| `prompt_include_adversarial` | bool | true | Include adversarial counter-examples |
| `lmstudio_base_url` | string | `http://127.0.0.1:1234/v1` | LM Studio server URL |
| `ollama_host` | string | `0.0.0.0` | Ollama host address |
| `ollama_port` | int | 11434 | Ollama port |
| `save_interval` | int | 20 | Segments between checkpoint saves |
| `min_classifiable_words` | int | 10 | Minimum words to attempt classification (0 = disabled) |
| `evidence_secondary_weight` | float | 0.6 | Weight for secondary/dissenting vote reconciliation |
| `evidence_presence_threshold` | float | 0.5 | Minimum pooled evidence for secondary label |

#### Sub-config: `purer_cue`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `skip_lesson_content` | bool | true | Skip didactic/lesson segments exceeding word threshold |
| `max_lesson_words` | int | 400 | Word threshold for lesson-content detection |
| `therapist_max_gap_seconds` | float | 120.0 | Gap threshold for aggregating therapist sentences into cue blocks |
| `max_context_words` | int | 1000 | Word budget for conversational context preamble |

#### Sub-config: `confidence_tiers`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `high_consistency` | int | 3 | Raters agreeing for high tier |
| `high_confidence` | float | 0.8 | Confidence threshold for high tier |
| `medium_min_consistency` | int | 2 | Minimum raters agreeing for medium tier |
| `medium_min_confidence` | float | 0.6 | Confidence threshold for medium tier |

#### Sub-config: `validation`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `n_per_class` | int | 50 | Samples per class for evaluation set |
| `min_kappa` | float | 0.70 | Minimum Cohen's kappa for validation |
| `min_agreement` | float | 0.75 | Minimum agreement fraction for validation |

#### Sub-config: `test_sets`

| Key | Type | Description |
|-----|------|-------------|
| `vaamr` | object | VAAMR test set spec (see below) |
| `purer` | object | PURER test set spec |
| `codebook` | object | Codebook test set spec |

Each test set spec (`TestSetSpec`) accepts:

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | varies | Enable this test set kind |
| `name` | string | varies | Unique name identifier |
| `n_sets` | int | 1 | Number of stratified cross-validation sets |
| `fraction_per_set` | float | 0.10 | Fraction of segments per set |
| `random_seed` | int | 42 | Random seed for reproducibility |

#### Sub-config: `content_validity`

| Key | Type | Description |
|-----|------|-------------|
| `vaamr` | object | VAAMR content-validity spec (`enabled`, `name`) |
| `purer` | object | PURER content-validity spec (`enabled`, `name`) |

#### Sub-config: `therapist_cues`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | true | Surface therapist cues at stage transitions |
| `max_length_per_cue` | int | 250 | Max words per cue before LLM summarization |
| `max_length_of_average_cue_responses` | int | 500 | Cap per averaged block in cue response analysis |

#### Sub-config: `session_summaries` / `participant_summaries`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | true | Enable LLM summary generation |
| `max_words_per_session` | int | 500 (session) / 300 (participant) | Max words per summary |

## Output Data Reference

### Core Data Files

| File | Format | Description |
|------|--------|-------------|
| `01_transcripts/segmented/<sid>/segments.jsonl` | JSONL | Frozen segmentation: one `Segment` object per line |
| `01_transcripts/segmented/<sid>/segmentation_meta.json` | JSON | Segmentation parameters hash and metadata |
| `02_meta/classifications/theme_labels.jsonl` | JSONL | VAAMR classification overlay (refreshable) |
| `02_meta/classifications/purer_labels.jsonl` | JSONL | PURER classification overlay (refreshable) |
| `02_meta/classifications/codebook_labels.jsonl` | JSONL | VCE codebook overlay (refreshable) |
| `02_meta/classifications/cross_validation_labels.jsonl` | JSONL | Cross-validation overlay |
| `02_meta/classifications/classification_manifest.json` | JSON | Provenance record of all overlays |
| `02_meta/training_data/master_segments.jsonl` | JSONL | Full assembled dataset (segments + all overlays joined) |
| `02_meta/training_data/master_segments.csv` | CSV | Tabular version of master segments |

### Segment Object Fields

Each line in `master_segments.jsonl` is a serialized `Segment` dataclass (see `classification_tools/data_structures.py`):

| Field | Type | Description |
|-------|------|-------------|
| `segment_id` | string | Unique identifier |
| `trial_id` | string | Trial or project identifier |
| `participant_id` | string | Anonymized participant ID |
| `session_id` | string | Session identifier (e.g. `c1s1`) |
| `session_number` | int | Ordinal session number |
| `speaker` | string | Speaker role: `participant` or `therapist` |
| `text` | string | Segment text content |
| `word_count` | int | Word count of segment |
| `segment_index` | int | Position within session |
| `start_time_ms` | int | Start time in milliseconds |
| `end_time_ms` | int | End time in milliseconds |
| `primary_stage` | int | VAAMR primary stage (0-4, null if unclassified) |
| `secondary_stage` | int | VAAMR secondary stage |
| `llm_confidence_primary` | float | Confidence score for primary label |
| `llm_run_consistency` | int | Number of runs agreeing on primary |
| `agreement_level` | string | `unanimous`, `majority`, `split`, `none` |
| `agreement_fraction` | float | Fraction of raters in agreement |
| `needs_review` | bool | Flagged for human review |
| `consensus_vote` | int or string | Final consensus label or `ABSTAIN` |
| `label_confidence_tier` | string | `High`, `Medium`, `Low`, or `Unclassified` |
| `final_label` | int | Gold-standard label after human adjudication |
| `final_label_source` | string | Source of final label (`llm_only`, `human`, `adjudicated`) |
| `codebook_labels_ensemble` | string[] | Multi-label VCE codes from ensemble reconciliation |
| `codebook_confidence` | object | Per-code confidence scores |
| `purer_primary` | int | PURER primary move type (0-4) |
| `purer_confidence_primary` | float | PURER confidence score |
| `purer_agreement_level` | string | PURER agreement level |
| `human_label` | int | Human-coded label (when available) |
| `in_human_coded_subset` | bool | Included in human evaluation set |

### Analysis Output Files

| File/Directory | Description |
|----------------|-------------|
| `03_analysis_data/session_stats/stats_<session>.json` | Per-session classification statistics |
| `03_analysis_data/session_stats/stats_cumulative.json` | Cumulative statistics across all sessions |
| `03_analysis_data/per_session/<session>.json` | Full per-session analysis |
| `03_analysis_data/per_participant/<participant>.json` | Per-participant longitudinal analysis |
| `03_analysis_data/per_theme/<stage>.json` | Per-theme (VAAMR stage) analysis |
| `03_analysis_data/per_theme/<code>.json` | Per-code (VCE) analysis |
| `03_analysis_data/cumulative_report.json` | Aggregated cumulative report |
| `03_analysis_data/longitudinal_summary.json` | Longitudinal trajectory summary |
| `03_analysis_data/session_stage_progression.csv` | Stage progression per session (graph-ready) |
| `03_analysis_data/graphing/*.csv` | Graph-ready datasets |
| `03_analysis_data/session_summaries.json` | LLM-generated session summaries |

### Validation Output Files

| File/Directory | Description |
|----------------|-------------|
| `04_validation/testsets/<name>/manifest.json` | Test set metadata |
| `04_validation/testsets/<name>/segments_snapshot.jsonl` | Frozen segment snapshot |
| `04_validation/testsets/<name>/human_worksheet.txt` | Frozen human coding worksheet |
| `04_validation/testsets/<name>/AI_answer_key.txt` | Refreshable AI answer key |
| `04_validation/content_validity/<name>/manifest.json` | Content-validity metadata |
| `04_validation/content_validity/<name>/items.jsonl` | Content-validity items |
| `04_validation/content_validity/<name>/human_worksheet.txt` | Frozen human worksheet |
| `04_validation/content_validity/<name>/definition_key.txt` | Frozen definition key |
| `04_validation/content_validity/<name>/AI_answer_key.txt` | Refreshable AI answer key |
| `04_validation/cross_validation/cross_validation_results.json` | Lift statistics |
| `04_validation/cross_validation/top_theme_code_associations.json` | Top associations |
| `04_validation/human_coding_evaluation_set.csv` | Evaluation set for human coders |

### Report and Figure Output

| File/Directory | Description |
|----------------|-------------|
| `01_transcripts/coded/coded_transcript_<session>.txt` | Human-readable coded transcript |
| `04_validation/human_classification_<session>.txt` | Human blind-coding form |
| `04_validation/flagged_for_review.txt` | Segments needing human review |
| `05_figures/*.png` | Matplotlib visualization figures |
| `06_reports/per_theme/` | Per-theme text reports |

## Module Reference

### Classification Tools (`classification_tools/`)

| File | Description |
|------|-------------|
| `llm_client.py` | Unified LLM API client (OpenRouter, Ollama, LM Studio) |
| `data_structures.py` | `Segment` dataclass and core data structures |
| `llm_classifier.py` | Theme, PURER, and codebook LLM classification logic |
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

### Theme Framework (Markdown Source)

The VAAMR and PURER framework definitions are now maintained as Markdown files in the repository root (`VAAMR_FRAMEWORK.md` and `PURER.md`). These files are parsed at runtime by the theme framework loader (`src/theme_framework/markdown_loader.py`) to generate the framework objects used in classification. This allows non-technical users to edit framework definitions directly without touching Python code.

**Parsing Contract**: The Markdown files must follow a specific format (see `MODULARIZATION_UPDATE.md` for details). Do not alter the YAML frontmatter or heading patterns unless you intend to change the framework identity. Edits to definitions, exemplars, and criteria are safe and encouraged.

| File | Description |
|------|-------------|
| `theme_framework/markdown_loader.py` | Parses VAAMR/PURER Markdown → `ThemeFramework` |
| `theme_framework/theme_schema.py` | `ThemeDefinition` and `ThemeFramework` base classes |
| `theme_framework/registry.py` | Name-to-path dispatch for frameworks |
| `theme_framework/config.py` | `ThemeClassificationConfig` dataclass |

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
