# Qualitative Research Algorithm (QRA)

A modular LLM classification pipeline for analyzing therapeutic dialogue transcripts. QRA implements two composable classification approaches -- **single-label theme/stage classification** and **multi-label codebook classification** -- orchestrated through a 6-stage pipeline. Built for extensibility, it supports any theme framework or codebook definition and runs on locally-hosted Hugging Face models or cloud APIs.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Installation](#installation)
4. [Entry Points](#entry-points)
5. [Architecture](#architecture)
6. [Core Data Structures](#core-data-structures)
7. [Module Reference](#module-reference)
8. [Pipeline Stages](#pipeline-stages)
9. [Embedding Classifier: Three-Target Two-Pass System](#embedding-classifier-three-target-two-pass-system)
10. [Configuration Reference](#configuration-reference)
11. [CLI Reference](#cli-reference)
12. [Usage Examples](#usage-examples)
13. [Extending QRA](#extending-qra)
14. [Output Files](#output-files)
15. [Key Design Decisions](#key-design-decisions)
16. [Troubleshooting](#troubleshooting)

---

## Overview

QRA automatically classifies segments of therapeutic dialogue transcripts into meaningful categories. The system consists of four composable modules:

| Module | Purpose |
|---|---|
| `classification_tools/` | Common infrastructure: data structures, LLM client, model loader, validation |
| `constructs/` | Theme/stage framework definitions and presets |
| `codebook/` | Multi-label codebook classification via triple-LLM embedding and LLM ensemble |
| `process/` | 6-stage orchestration engine coordinating the full workflow |

### Key Capabilities

- **Framework-agnostic**: Works with any theme framework (VA-MR is the default preset)
- **Four LLM backends**: Hugging Face (local), OpenRouter (proprietary), Replicate (open-source), Ollama (local API)
- **Triple-LLM embedding classifier**: Three causal LLMs (Llama 4 Maverick, Mixtral 8x7B, Qwen 3 Next) provide independent embedding perspectives for multi-label classification
- **Two-pass exemplar accumulation**: Pass 1 discovers high-confidence exemplar segments that enrich pass 2 for improved codebook classification accuracy
- **Multi-model cross-referencing**: Runs classification across multiple models and measures agreement (unanimous, majority, split)
- **Triplicate consistency checking**: Classifies each segment 3 times and flags inconsistencies
- **Embedding-based segmentation**: Uses sentence-transformer semantic similarity to split long transcripts into coherent segments
- **Multi-label ensemble**: Combines embedding-based and LLM-based codebook results with disagreement flagging
- **Interactive setup wizard**: Guided configuration that saves reusable JSON configs
- **Human validation workflows**: Interactive validation of uncertain classifications with educational narration
- **Confidence tiering**: Assigns segments to high/medium/low confidence tiers based on consistency and scores
- **Cross-validation**: Empirically validates hypothesized theme-to-codebook code mappings via co-occurrence lift analysis
- **Checkpointing**: Saves intermediate results and supports resume from interruption

### Use Case: VA-MR Framework

By default, QRA uses the **VA-MR (Vigilance-Avoidance-Metacognition-Reappraisal)** framework to classify stages of contemplative transformation in therapeutic dialogue. The four stages represent progression in pain-related coping:

| Stage | ID | Description |
|---|---|---|
| **Vigilance** | 0 | Hypervigilance to pain, overwhelmed by sensation, reactive attention |
| **Avoidance** | 1 | Deliberate attentional control deployed to suppress pain |
| **Metacognition** | 2 | Observing one's own reactions without yet changing the experience |
| **Reappraisal** | 3 | Fundamental reinterpretation of pain sensations and their significance |

---

## Quick Start

### 1. Install

```bash
cd Qualitative_Research_Algorithm
pip install -r requirements.txt
```

### 2. Choose Your Workflow

QRA provides a unified CLI with three modes of operation:

#### Option A: Interactive Setup Wizard (Recommended for First-Time Users)

Walk through a guided 9-step configuration wizard that saves a reusable config file:

```bash
python qra.py setup
```

The wizard will prompt you for:
1. Input/output paths
2. LLM backend and model selection
3. Theme framework (VA-MR preset or custom)
4. Custom exemplar utterances (optional)
5. Codebook configuration (optional)
6. Classification parameters (n_runs, temperature)
7. Confidence thresholds
8. Run mode (auto/interactive/review)
9. Save location

After saving, the wizard offers to run the pipeline immediately.

#### Option B: Quick Run with Defaults

Run the full pipeline with sensible defaults (VA-MR framework, HuggingFace backend):

```bash
# Theme-only classification
python qra.py run \
    --transcript-dir ./data/input/diarized_sessions/ \
    --output-dir ./data/output/

# With codebook classification
python qra.py run \
    --run-codebook-classifier \
    --transcript-dir ./data/input/diarized_sessions/ \
    --output-dir ./data/output/
```

#### Option C: Guided Mode with Educational Narration

Step-by-step execution with explanations of each pipeline stage:

```bash
python qra.py guided \
    --backend openrouter \
    --model openai/gpt-4o \
    --transcript-dir ./data/diarized/
```

Guided mode pauses at each stage to explain what's happening, making it ideal for learning the pipeline or demonstrating to stakeholders.

#### Option D: Run from Saved Config

After using the setup wizard or creating a config file manually:

```bash
python qra.py run --config ./my_config.json
```

CLI arguments override config file values:

```bash
python qra.py run --config ./my_config.json --mode interactive --n-runs 5
```

---

## Installation

### Requirements

- **Python 3.8+**
- **GPU recommended** for embedding-based segmentation and the triple-LLM codebook classifier
- **~150 GB disk** for the three Hugging Face models (if using HuggingFace backend)

### Core Dependencies (`requirements.txt`)

| Package | Purpose |
|---|---|
| `pandas >= 2.0` | DataFrame operations, master dataset assembly |
| `numpy >= 1.24` | Matrix operations for embedding scoring |
| `scikit-learn >= 1.3` | Cosine similarity, balanced sampling |
| `sentence-transformers >= 2.2` | BERT embeddings for transcript segmentation |
| `requests >= 2.31` | HTTP client for OpenRouter API |
| `dataclasses-json >= 0.6` | Data structure serialization |
| `transformers >= 4.30` | Hugging Face model loading for causal LLMs |
| `torch >= 2.0` | PyTorch for model inference and embeddings |
| `scipy` | Euclidean distance computation |
| `srt >= 3.5` | Legacy SRT transcript parsing |

### Install

```bash
pip install -r requirements.txt
```

### API Keys (for cloud backends)

| Backend | Environment Variable | CLI Flag |
|---|---|---|
| OpenRouter | `OPENROUTER_API_KEY` | `--api-key` |
| Replicate | `REPLICATE_API_TOKEN` | `--replicate-api-token` |
| Ollama | N/A (local) | `--backend ollama` |
| Hugging Face | N/A (local) | `--backend huggingface` |

---

## Entry Points

### `qra.py` (Primary Entry Point)

**Purpose**: Unified CLI with three subcommands for all workflows.

**Subcommands**:
| Command | Description |
|---|---|
| `setup` | Interactive configuration wizard that saves reusable config JSON |
| `run` | Execute the classification pipeline (auto/interactive/review modes) |
| `guided` | Execute with step-by-step educational narration |

**Key features**:
- Config file support (`--config`)
- All backends supported (OpenRouter, Replicate, HuggingFace, Ollama)
- Three run modes: `auto` (fully automated), `interactive` (prompts for validation), `review` (batch validation)
- Human validation for uncertain classifications
- Cross-validation between theme and codebook labels
- Educational narration in guided mode

**Example**:
```bash
# Setup wizard
python qra.py setup

# Run with OpenRouter
python qra.py run --backend openrouter --model openai/gpt-4o

# Run with saved config
python qra.py run --config ./qra_config.json --mode interactive

# Guided mode
python qra.py guided --backend huggingface
```

### Backward-Compatible Entry Points

For compatibility with existing scripts, the old entry points are preserved as thin shims:

#### `run_pipeline.py`

Delegates to `qra.py run --backend huggingface`. Automatically downloads and manages three causal LLMs for multi-model cross-referencing.

```bash
python run_pipeline.py \
    --transcript-dir ./data/input/ \
    --run-codebook-classifier
```

#### `classify_and_label.py`

Delegates to `qra.py run` with OpenRouter backend (default). Maps `--run-mode` to `--mode`.

```bash
python classify_and_label.py \
    --transcript-dir ./data/diarized/ \
    --run-mode interactive
```

#### `run_vamr_pipeline.py`

Delegates to `qra.py run` with VA-MR framework (default).

```bash
python run_vamr_pipeline.py --backend openrouter --model openai/gpt-4o
```

---

## Architecture

### Directory Structure

```
Qualitative_Research_Algorithm/
├── classification_tools/         # Common infrastructure
│   ├── __init__.py
│   ├── data_structures.py        # Segment, SpeakerRun dataclasses
│   ├── llm_client.py             # Unified LLM API client (4 backends)
│   ├── model_loader.py           # Hugging Face model download/load/unload
│   ├── classification_loop.py    # Shared N-run classification with checkpointing
│   ├── majority_vote.py          # Single/multi-label voting logic
│   ├── validation.py             # Balanced evaluation set construction
│   ├── llm_classifier.py         # Zero-shot theme + LLM codebook classification
│   └── response_parser.py        # Multi-pass parsing, 8-category error taxonomy
│
├── constructs/                   # Theme/stage framework definitions
│   ├── __init__.py
│   ├── theme_schema.py           # ThemeFramework, ThemeDefinition dataclasses
│   ├── config.py                 # ThemeClassificationConfig
│   └── vamr.py                   # VA-MR preset (4 stages)
│
├── codebook/                     # Multi-label codebook classification
│   ├── __init__.py
│   ├── codebook_schema.py        # Codebook, CodeDefinition, CodeAssignment
│   ├── config.py                 # EmbeddingClassifierConfig, LLMCodebookConfig, EnsembleConfig
│   ├── embedding_classifier.py   # Triple-LLM two-pass embedding classifier
│   ├── ensemble.py               # Reconcile embedding vs LLM disagreements
│   └── phenomenology_codebook.py # 54-code VCE phenomenology codebook (6 domains)
│
├── process/                      # Orchestration & assembly
│   ├── __init__.py
│   ├── config.py                 # PipelineConfig (aggregates all sub-configs)
│   ├── orchestrator.py           # 6-stage pipeline engine with observer/validator hooks
│   ├── pipeline_hooks.py         # PipelineObserver, SilentObserver, GuidedModeObserver
│   ├── human_validator.py        # HumanValidator class for interactive validation
│   ├── setup_wizard.py           # Interactive 9-step SetupWizard
│   ├── transcript_ingestion.py   # Embedding-based semantic segmentation
│   ├── dataset_assembly.py       # Master dataset, adjacency index, progression
│   └── cross_validation.py       # Theme ↔ codebook co-occurrence analysis
│
├── qra.py                        # Unified CLI entry point (setup/run/guided)
├── run_pipeline.py               # Backward-compatible shim → qra run --backend huggingface
├── classify_and_label.py         # Backward-compatible shim → qra run
├── run_vamr_pipeline.py          # Backward-compatible shim → qra run
└── requirements.txt              # Python dependencies
```

### Module Dependency Graph

```
qra.py (setup / run / guided)
    |
    v
process.orchestrator (6-stage execution)
    +-> process.pipeline_hooks         (observer pattern for UI feedback)
    +-> process.human_validator        (interactive validation)
    +-> process.transcript_ingestion   (Stage 1: segmentation)
    +-> constructs.*                   (Stage 2: operationalization)
    +-> classification_tools.llm_classifier (Stage 3: theme classification)
    +-> classification_tools.response_parser (Stage 4: response parsing)
    +-> codebook.*                     (Stage 3b: optional)
    |   +-> embedding_classifier       (triple-LLM two-pass)
    |   +-> llm_classifier             (LLM multi-label)
    |   +-> ensemble                   (disagreement reconciliation)
    +-> process.cross_validation       (empirical theme-codebook validation)
    +-> classification_tools.validation (Stage 5: evaluation set)
    +-> process.dataset_assembly       (Stage 6: final output)

Shared infrastructure:
    classification_tools.data_structures (Segment, SpeakerRun)
    classification_tools.llm_client      (LLM API abstraction, 4 backends)
    classification_tools.model_loader    (HF model download/load/unload/cache)
    classification_tools.classification_loop (N-run loop with checkpointing)
    classification_tools.majority_vote   (single-label + multi-label voting)
```

---

## Core Data Structures

### Segment (`classification_tools/data_structures.py`)

The `Segment` dataclass is the **atomic unit** of the entire pipeline. Every operation produces or consumes `Segment` objects. Fields are populated progressively through pipeline stages.

```python
@dataclass
class Segment:
    # Identity
    segment_id: str          # Unique ID (e.g., "P001_S001_seg_003")
    trial_id: str            # Experiment/trial identifier
    participant_id: str      # Participant identifier
    session_id: str          # Session identifier
    session_number: int      # Ordinal session number

    # Temporal
    segment_index: int       # Position within session
    start_time_ms: int       # Audio start timestamp (milliseconds)
    end_time_ms: int         # Audio end timestamp (milliseconds)
    total_segments_in_session: int

    # Speaker and text
    speaker: str             # 'participant' or 'therapist'
    text: str                # Segment text content
    word_count: int

    # Theme labels (populated by theme classification, Stage 3-4)
    primary_stage: Optional[int]           # Theme ID (e.g., 0 = Vigilance)
    secondary_stage: Optional[int]         # Secondary theme ID or None
    llm_confidence_primary: Optional[float]
    llm_confidence_secondary: Optional[float]
    llm_justification: Optional[str]
    llm_run_consistency: Optional[int]     # How many of N runs agreed (0-3)

    # Multi-model cross-referencing (populated when using multiple models)
    model_agreement: Optional[str]         # 'unanimous', 'majority', 'split', 'none'
    model_predictions: Optional[Dict]      # Per-model predictions
    total_models: Optional[int]

    # Codebook labels (populated by codebook classification, Stage 3b)
    codebook_labels_embedding: Optional[List[str]]   # From embedding classifier
    codebook_labels_llm: Optional[List[str]]         # From LLM classifier
    codebook_labels_ensemble: Optional[List[str]]    # Reconciled final codes
    codebook_disagreements: Optional[List[str]]      # Codes where methods disagreed
    codebook_confidence: Optional[Dict[str, float]]  # Per-code confidence

    # Human validation (filled after manual coding)
    human_label: Optional[int]
    human_secondary_label: Optional[int]
    adjudicated_label: Optional[int]
    in_human_coded_subset: bool
    label_status: str        # 'llm_only', 'human_coded', 'adjudicated'

    # Final training label (populated by dataset assembly, Stage 6)
    final_label: Optional[int]
    final_label_source: Optional[str]
    label_confidence_tier: Optional[str]   # 'high', 'medium', 'low'
```

### ThemeFramework (`constructs/theme_schema.py`)

Defines a complete classification scheme. Key methods:
- `build_name_to_id_map()` -- Maps all name variants/aliases to integer IDs
- `to_prompt_string()` -- Formats themes for LLM prompting
- `to_json()` -- Exports as JSON for documentation
- `build_id_to_short_map()` -- Maps IDs to short display names

### Codebook (`codebook/codebook_schema.py`)

Defines a multi-label classification scheme. Key methods:
- `to_embedding_targets()` -- Returns list of dicts with `definition`, `criteria`, `exemplars` keys for embedding comparison
- `to_prompt_string(randomize=)` -- Formats codes for LLM prompting
- `get_codes_by_domain(domain)` -- Filters codes by domain
- `build_name_to_id_map()` -- Maps category names to code IDs

### PipelineConfig (`process/config.py`)

Top-level configuration with serialization support. Key methods:
- `to_json()` -- Serializes to JSON-safe dict (blanks API keys)
- `from_json(data)` -- Reconstructs from dict with nested dataclass conversion

---

## Module Reference

### `classification_tools/llm_client.py`

Unified LLM client supporting four backends:

| Backend | Config | Notes |
|---|---|---|
| `huggingface` | Local models via `model_loader.py` | GPU recommended |
| `openrouter` | `api_key` required | Proprietary models (GPT-4, Claude, etc.) |
| `replicate` | `replicate_api_token` required | Open-source models |
| `ollama` | `ollama_host`, `ollama_port` | Local Ollama server |

Key methods:
- `request(prompt)` -- Single-model request, returns `(text, metadata)`
- `multi_model_request(prompt, models=)` -- Cross-references across multiple models, returns `[(model_id, text, metadata), ...]`

Features: automatic retry with exponential backoff, JSON format enforcement, metadata tracking.

### `process/pipeline_hooks.py`

Observer pattern for pipeline events. Three implementations:
- `PipelineObserver` -- Base class with no-op methods
- `SilentObserver` -- Minimal output (stage headers + summaries)
- `GuidedModeObserver` -- Verbose educational narration with `Press Enter to continue` prompts

Methods: `on_stage_start()`, `on_stage_progress()`, `on_stage_complete()`, `on_pipeline_complete()`

### `process/human_validator.py`

Interactive CLI for human validation of uncertain classifications:
- `validate_theme_classification(segment, reason)` -- Prompts user to correct theme labels
- `validate_codebook_disagreement(segment, ensemble_result)` -- Reconciles embedding vs LLM method disagreements
- `validate_cooccurrence_anomaly(segment, anomaly_details)` -- Reviews theme ↔ codebook mapping violations

All methods respect `skip_confirmation` flag for automated mode. Maintains `validation_log` for audit trail.

### `process/setup_wizard.py`

Interactive 9-step configuration wizard:
1. Input/output paths
2. Backend & model selection
3. Framework selection (VA-MR or custom JSON)
4. Exemplar utterances (optional customization)
5. Codebook selection and two-pass settings
6. Classification parameters (n_runs, temperature)
7. Confidence thresholds
8. Run mode (auto/interactive/review)
9. Save config JSON and optionally run

### `process/cross_validation.py`

Theme-to-codebook co-occurrence analysis:
- `compute_theme_codebook_cooccurrence(segments_df, framework)` -- Computes per-theme code rates and lift values
- `validate_codebook_hypothesis(cooccurrence, framework, min_lift=1.5)` -- Tests hypothesized theme-to-code mappings against observed data, reports confirmed/unconfirmed/unexpected associations

---

## Pipeline Stages

The orchestrator (`process/orchestrator.py`) executes a 6-stage pipeline with optional observer and validator hooks:

### Stage 1: Transcript Ingestion and Segmentation

- Loads diarized JSON files from `transcript_dir`
- Segments using `TranscriptSegmenter` (BERT embeddings + silence gaps)
- Creates `Segment` objects with identity, temporal, speaker, and text fields

**Segmentation algorithm**:
1. Separate sentences by speaker
2. Compute sentence embeddings using sentence-transformers (default: `all-MiniLM-L6-v2`)
3. Compute rolling-window cosine similarity curve
4. Detect silence gaps in audio timestamps
5. Combine similarity dips and pauses to find segment boundaries
6. Group sentences into segments respecting word count constraints

**Input format** (diarized JSON):
```json
{
  "metadata": {
    "trial_id": "standard",
    "participant_id": "P001",
    "session_id": "S001",
    "session_number": 1
  },
  "sentences": [
    {"text": "I've been having a lot of pain lately.", "speaker": "participant", "start": 2.3, "end": 5.1},
    ...
  ]
}
```

### Stage 2: Construct Operationalization

- Exports theme definitions as JSON (`theme_definitions.json`)
- Constructs content validity test set from framework's exemplar/subtle/adversarial utterances
- Exports test set (`content_validity_test_set.jsonl`)

### Stage 3: Zero-Shot LLM Theme Classification

(If `run_theme_labeler=True`)

- For each segment, sends text to the LLM with framework definitions
- Runs N times per segment (default: 3) for consistency checking
- When multiple models are configured, cross-references predictions
- Saves raw results as JSON checkpoints (resumable)
- Optional: Interactive validation of uncertain classifications (low confidence or inconsistency)

### Stage 4: Response Parsing

- Parses JSON from each LLM run
- Resolves theme names to integer IDs using framework's `name_to_id` map
- Computes run consistency (how many of N runs agreed)
- Populates `primary_stage`, `secondary_stage`, `llm_confidence_*`, `llm_run_consistency`

### Stage 3b: Codebook Classification (Optional)

(If `run_codebook_classifier=True`)

Three sub-steps:

1. **Embedding classification**: Triple-LLM two-pass system (see [detailed section below](#embedding-classifier-three-target-two-pass-system))
2. **LLM classification**: Zero-shot multi-label prompting with majority voting
3. **Ensemble reconciliation**: Agrees/disagrees codes, flags for human review

Optional: Interactive validation of method disagreements

### Stage 4b: Cross-Validation (Optional)

(If both theme and codebook are enabled)

- Computes theme ↔ codebook co-occurrence statistics
- Validates hypothesized mappings (confirmed/unconfirmed/unexpected associations)
- Exports results and anomalies

### Stage 5: Human Validation Set Preparation

- Filters participant-labeled segments
- Samples balanced evaluation set (N segments per class, stratified by trial_id)
- Exports as CSV for human annotation

### Stage 6: Dataset Assembly

- Applies confidence tiering logic
- Exports master segment dataset as JSONL
- Builds session adjacency index
- Computes session stage progression (forward/backward transitions)

**Confidence tiers**:
| Tier | Criteria |
|---|---|
| **High** | 3/3 consistency AND confidence > 0.8 |
| **Medium** | 2+ consistency AND confidence > 0.6 |
| **Low** | Everything else |

Thresholds are configurable via `--high-confidence-threshold` and `--medium-confidence-threshold`.

---

## Embedding Classifier: Three-Target Two-Pass System

The embedding classifier (`codebook/embedding_classifier.py`) is the core of the multi-label codebook classification. It uses three causal LLMs to provide independent embedding perspectives via mean-pooled final hidden layer outputs.

### Three Models, Three Axes

| Phase | Model | Metric | What It Measures |
|---|---|---|---|
| 1 | Llama 4 Maverick 17B | Cosine similarity | Semantic alignment (higher = more similar) |
| 2 | Mixtral 8x7B | Euclidean distance | Fine-grained distance (lower = closer) |
| 3 | Qwen 3 Next 80B | Cosine distance (1 - cosine sim) | Complementary distance perspective |

### Three Comparison Targets

Each segment is compared against three target texts per code:

| Target | Source | Description |
|---|---|---|
| **Definition** | `code.category + subcodes + description` | The core meaning of the code |
| **Criteria** | `code.inclusive_criteria` | When to apply this code |
| **Exemplars** | `code.exemplar_utterances` + discovered exemplars | Prototypical matching text |

### Additive Scoring Formula

For each model phase, scores are computed additively (no subtraction):

```
score = sim(segment, definition)
      + criteria_weight * sim(segment, criteria)
      + exemplar_weight * sim(segment, exemplars)    [if exemplars exist]
```

- `criteria_weight` (default 0.5): How much weight the inclusive criteria get
- `exemplar_weight` (default 0.5): How much weight the exemplar text gets
- Phase 3 target is `definition + ' ' + criteria` (single combined embedding)

### Triple-Veto Logic

A code is assigned only when ALL three conditions are met:

```
passes_similarity: sim_matrix[i,j] > avg_sim * similarity_threshold
passes_distance:   dist_matrix[i,j] < avg_dist / distance_threshold
passes_tertiary:   tert_matrix[i,j] < avg_tert / tertiary_threshold
```

Default thresholds: similarity=1.375, distance=1.325, tertiary=1.35.

Candidate codes are selected as the union of top-K nearest codes across all three axes, where K = `max_codes // 3`.

### Two-Pass Exemplar Accumulation

When `two_pass=True` (default):

```
1. Load pre-populated exemplars from exemplar_import_path (if set)
2. Pass 1: Run triple-LLM classification with available exemplars
3. Accumulate: Collect segments with confidence >= exemplar_confidence_threshold
   - Per code: sort by confidence descending
   - Concatenate segment texts up to max_exemplar_tokens words
4. Save discovered exemplars to exemplar_export_path (if set)
5. Merge: Combine imported + discovered exemplars
6. Pass 2: Re-run classification with enriched exemplar targets
7. Return pass-2 results
```

When `two_pass=False`: Returns pass-1 results directly.

### Exemplar File Format

```json
{
  "fear_anxiety": [
    "I was so scared that the pain would never stop...",
    "Every time it flares up I just panic..."
  ],
  "meta_cognition": [
    "I noticed I was getting anxious and I could just watch that..."
  ]
}
```

---

## Configuration Reference

### `PipelineConfig` (`process/config.py`)

Top-level configuration aggregating all sub-configs:

```python
@dataclass
class PipelineConfig:
    transcript_dir: str = './data/input/diarized_sessions/'
    trial_id: str = 'standard'
    output_dir: str = './data/output/'
    run_mode: str = 'auto'  # 'auto', 'interactive', 'review'
    run_theme_labeler: bool = True
    run_codebook_classifier: bool = False

    segmentation: SegmentationConfig
    theme_classification: ThemeClassificationConfig
    codebook_embedding: EmbeddingClassifierConfig
    codebook_llm: LLMCodebookConfig
    codebook_ensemble: EnsembleConfig
    validation: ValidationConfig
    confidence_tiers: ConfidenceTierConfig

    resume_from: Optional[str] = None
```

Methods: `to_json()`, `from_json(data)`

### `SegmentationConfig` (`process/config.py`)

| Field | Default | Description |
|---|---|---|
| `embedding_model` | `'all-MiniLM-L6-v2'` | Sentence transformer for semantic similarity |
| `min_segment_words` | `30` | Minimum words per segment |
| `max_segment_words` | `200` | Maximum words per segment |
| `silence_threshold_ms` | `1500` | Pause duration to trigger boundary |
| `semantic_shift_percentile` | `25` | Percentile for similarity drop detection |

### `ThemeClassificationConfig` (`constructs/config.py`)

| Field | Default | Description |
|---|---|---|
| `model` | `'meta-llama/Llama-4-Maverick-17B-128E-Instruct'` | Primary LLM model |
| `models` | `[]` | List of models for cross-referencing |
| `temperature` | `0.0` | LLM sampling temperature |
| `n_runs` | `3` | Triplicate runs per segment |
| `backend` | `'huggingface'` | API backend |
| `randomize_codebook` | `True` | Shuffle theme order in prompt |
| `api_key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `replicate_api_token` | `$REPLICATE_API_TOKEN` | Replicate token |
| `max_new_tokens` | `512` | Max LLM output length |
| `output_dir` | `'./data/output/llm_labels/'` | Raw LLM output directory |
| `save_interval` | `20` | Checkpoint every N segments |

### `ConfidenceTierConfig` (`process/config.py`)

| Field | Default | Description |
|---|---|---|
| `high_consistency` | `3` | Required consistency for high tier |
| `high_confidence` | `0.8` | Required confidence for high tier |
| `medium_min_consistency` | `2` | Required consistency for medium tier |
| `medium_min_confidence` | `0.6` | Required confidence for medium tier |

---

## CLI Reference

### `qra.py`

```
usage: qra [-h] {setup,run,guided} ...

positional arguments:
  {setup,run,guided}
    setup             Interactive configuration wizard
    run               Execute the classification pipeline
    guided            Execute with step-by-step educational narration
```

#### `qra setup`

Runs interactive 9-step configuration wizard. No additional arguments.

#### `qra run`

```
usage: qra run [-h] [--mode {auto,interactive,review}] [--config CONFIG]
               [--transcript-dir DIR] [--output-dir DIR] [--trial-id ID]
               [--backend {openrouter,replicate,huggingface,ollama}]
               [--model MODEL] [--models MODEL [MODEL ...]]
               [--framework {vamr,PATH}] [--codebook {phenomenology,PATH}]
               [--n-runs N] [--temperature T]
               [--high-confidence-threshold F] [--medium-confidence-threshold F]
               [--no-theme-labeler] [--run-codebook-classifier]
               [--no-two-pass] [--exemplar-import-path PATH]
               [--criteria-weight F] [--exemplar-weight F]
               [--resume-from PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--mode` | `auto` | Run mode: `auto`, `interactive`, `review` |
| `--config, -c` | None | Load config from JSON file |
| `--transcript-dir` | `./data/input/diarized_sessions/` | Input directory |
| `--output-dir` | `./data/output/` | Output directory |
| `--trial-id` | `standard` | Trial identifier |
| `--backend` | `openrouter` | LLM backend |
| `--model` | `openai/gpt-4o` | Primary model |
| `--models` | `[]` | Multiple models for cross-referencing |
| `--framework` | `vamr` | Theme framework (vamr or JSON path) |
| `--codebook` | `phenomenology` | Codebook (phenomenology or JSON path) |
| `--n-runs` | `3` | Classification runs per segment |
| `--temperature` | `0.0` | LLM temperature |
| `--high-confidence-threshold` | `0.8` | High tier threshold |
| `--medium-confidence-threshold` | `0.6` | Medium tier threshold |
| `--no-theme-labeler` | off | Skip theme classification |
| `--run-codebook-classifier` | off | Enable codebook classification |
| `--no-two-pass` | off | Disable two-pass (single pass only) |
| `--exemplar-import-path` | None | Pre-populated exemplar JSON |
| `--criteria-weight` | `0.5` | Criteria similarity weight |
| `--exemplar-weight` | `0.5` | Exemplar similarity weight |
| `--resume-from` | None | Checkpoint to resume from |

#### `qra guided`

Same arguments as `qra run` except:
- No `--mode` flag (always uses interactive mode)
- Adds educational narration at each stage with `Press Enter to continue` prompts

---

## Usage Examples

### Example 1: Interactive Setup Wizard

```bash
python qra.py setup
```

Walks through 9 configuration steps, saves `qra_config.json`, optionally runs pipeline immediately.

### Example 2: Quick Run with Defaults

```bash
python qra.py run \
    --backend openrouter \
    --model openai/gpt-4o \
    --transcript-dir ./data/diarized/
```

### Example 3: Run from Saved Config

```bash
python qra.py run --config ./qra_config.json
```

### Example 4: Interactive Validation Mode

```bash
python qra.py run \
    --mode interactive \
    --backend openrouter \
    --model openai/gpt-4o \
    --transcript-dir ./data/diarized/
```

Prompts for validation of uncertain theme classifications and codebook disagreements.

### Example 5: Guided Mode with Educational Narration

```bash
python qra.py guided \
    --backend openrouter \
    --model openai/gpt-4o \
    --transcript-dir ./data/diarized/
```

Pauses at each stage to explain what's happening before proceeding.

### Example 6: Full Pipeline with Codebook + Two-Pass

```bash
python qra.py run \
    --run-codebook-classifier \
    --backend huggingface \
    --transcript-dir ./data/input/
```

Uses HuggingFace models for both theme and codebook classification with two-pass exemplar discovery.

### Example 7: Pre-Populated Exemplars

```bash
python qra.py run \
    --run-codebook-classifier \
    --exemplar-import-path ./curated_exemplars.json \
    --criteria-weight 0.6 \
    --exemplar-weight 0.8
```

### Example 8: Multi-Model Cross-Referencing

```bash
python qra.py run \
    --backend openrouter \
    --models openai/gpt-4o anthropic/claude-3-sonnet mistralai/mistral-large \
    --transcript-dir ./data/diarized/
```

Classifies each segment with all three models and tracks agreement.

### Example 9: Resume from Checkpoint

```bash
python qra.py run \
    --resume-from ./data/output/llm_raw/llm_results_*.json \
    --mode interactive
```

### Example 10: Override Config File Settings

```bash
python qra.py run \
    --config ./qra_config.json \
    --n-runs 5 \
    --mode interactive \
    --temperature 0.1
```

Loads base config from file, overrides specific settings via CLI.

---

## Extending QRA

### Adding a New Theme Framework

Create `constructs/my_framework.py`:

```python
from .theme_schema import ThemeFramework, ThemeDefinition

def get_my_framework() -> ThemeFramework:
    theme1 = ThemeDefinition(
        theme_id=0,
        key='theme_key_1',
        name='Full Name of Theme',
        short_name='Short Name',
        prompt_name='how to refer in prompts',
        definition='Detailed operational definition...',
        prototypical_features=['feature 1', 'feature 2'],
        distinguishing_criteria='Key distinction...',
        exemplar_utterances=['Example 1', 'Example 2'],
    )
    # Add more themes...

    return ThemeFramework(
        name='My Framework',
        version='1.0',
        description='Description',
        themes=[theme1, ...],
        codebook_hypothesis={            # Optional: expected code mapping
            'theme_key_1': ['Code A', 'Code B'],
        },
    )
```

Export as JSON for use with `--framework`:

```python
import json
fw = get_my_framework()
with open('my_framework.json', 'w') as f:
    json.dump(fw.to_json(), f, indent=2)
```

Use in pipeline:

```bash
python qra.py run --framework ./my_framework.json
```

### Adding a New Codebook

Create `codebook/my_codebook.py`:

```python
from .codebook_schema import Codebook, CodeDefinition

def get_my_codebook() -> Codebook:
    code1 = CodeDefinition(
        code_id='code_1',
        category='Code 1',
        domain='Domain A',
        description='What this code means...',
        subcodes=['subcode_1a', 'subcode_1b'],
        inclusive_criteria='When to apply this code...',
        exclusive_criteria='When NOT to apply...',
        exemplar_utterances=['Example 1', 'Example 2'],
    )
    # More codes...

    return Codebook(
        name='My Codebook',
        version='1.0',
        description='Full description',
        codes=[code1, ...],
        domains={'Domain A': ['code_1', ...], ...},
    )
```

Use in pipeline:

```bash
python qra.py run --codebook ./my_codebook.json --run-codebook-classifier
```

---

## Output Files

### `master_segments_[timestamp].jsonl`

Complete segment dataset with all fields. One JSON object per line.

```json
{
  "segment_id": "P001_S001_seg_003",
  "trial_id": "standard",
  "participant_id": "P001",
  "session_id": "S001",
  "speaker": "participant",
  "text": "I've been having a lot of pain...",
  "primary_stage": 0,
  "llm_confidence_primary": 0.92,
  "llm_run_consistency": 3,
  "codebook_labels_ensemble": ["fear_anxiety", "pain"],
  "codebook_confidence": {"fear_anxiety": 0.87, "pain": 0.91},
  "final_label": 0,
  "label_confidence_tier": "high"
}
```

### `codebook_raw/found_exemplar_utterances.json`

Discovered exemplar segments from the two-pass embedding classifier. Can be fed back as `--exemplar-import-path` for future runs.

### `human_coding_evaluation_set.csv`

Balanced sample of segments for human annotation. Stratified by trial_id and label.

### `theme_definitions.json`

Complete framework specification as JSON.

### `cross_validation_results_[timestamp].json`

Theme-to-codebook co-occurrence validation with confirmed/unconfirmed/unexpected associations.

### `human_validation_log_[timestamp].jsonl`

Log of all human validation decisions made during interactive/review mode.

### `session_adjacency_[timestamp].jsonl`

Adjacency relationships between segments within sessions.

### `session_stage_progression_[timestamp].csv`

Per-session analysis of stage transitions (forward/backward counts).

---

## Key Design Decisions

### 1. Unified CLI with Subcommands

All workflows accessible through `qra.py` with three subcommands (`setup`, `run`, `guided`). Old entry points preserved as backward-compatible shims.

### 2. Observer Pattern for UI Feedback

`PipelineObserver` interface enables silent (minimal), verbose (guided), or custom UI feedback without modifying the orchestrator core.

### 3. Human-in-the-Loop Validation

`HumanValidator` class supports interactive validation of uncertain classifications, codebook disagreements, and co-occurrence anomalies with full audit logging.

### 4. Config Serialization

`PipelineConfig.to_json()` / `from_json()` enable saving/loading complete pipeline configurations with nested dataclass reconstruction and API key blanking.

### 5. Framework-Agnostic Pipeline

The pipeline accepts any `ThemeFramework`, not hardcoded to VA-MR. Theme name resolution via `build_name_to_id_map()`, prompt generation parameterized by framework.

### 6. Triple-LLM Embedding

The embedding classifier uses three full causal LLMs (Llama 4 Maverick, Mixtral 8x7B, Qwen 3 Next) rather than traditional sentence transformers. Each model provides a different embedding perspective via mean-pooled hidden states, and all three must agree (triple-veto) for a code to be assigned.

### 7. Two-Pass Exemplar Discovery

Pass 1 identifies high-confidence segments. Pass 2 uses these as additional comparison targets, improving accuracy on codes that have few hand-written exemplars. 

---

## License

MIT 
---

## Contact

For questions or issues, please contact wade@wadebalsamo.com
