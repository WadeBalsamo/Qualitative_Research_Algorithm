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
| `shared/` | Common infrastructure: data structures, LLM client, model loader, validation |
| `theme_labeler/` | Single-label classification into a framework of stages/themes |
| `codebook_classifier/` | Multi-label classification via triple-LLM embedding and LLM ensemble |
| `pipeline/` | 6-stage orchestration engine coordinating the full workflow |

### Key Capabilities

- **Framework-agnostic**: Works with any theme framework (VA-MR is the default preset)
- **Four LLM backends**: Hugging Face (local), OpenRouter (proprietary), Replicate (open-source), Ollama (local API)
- **Triple-LLM embedding classifier**: Three causal LLMs (Llama 4 Maverick, Mixtral 8x7B, Qwen 3 Next) provide independent embedding perspectives for multi-label classification
- **Two-pass exemplar accumulation**: Pass 1 discovers high-confidence exemplar segments that enrich pass 2 for improved codebook classification accuracy
- **Multi-model cross-referencing**: Runs classification across multiple models and measures agreement (unanimous, majority, split)
- **Triplicate consistency checking**: Classifies each segment 3 times and flags inconsistencies
- **Embedding-based segmentation**: Uses sentence-transformer semantic similarity to split long transcripts into coherent segments
- **Multi-label ensemble**: Combines embedding-based and LLM-based codebook results with disagreement flagging
- **Human validation workflows**: Generates balanced evaluation sets and supports interactive validation of uncertain classifications
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
# For Hugging Face model support:
pip install -r requirements_huggingface.txt
```

### 2. Choose an Entry Point

QRA provides two pipeline entry points:

#### Option A: `run_pipeline.py` -- Hugging Face Multi-Model Pipeline

Uses three locally-hosted Hugging Face causal LLMs for theme classification with multi-model cross-referencing. Supports optional codebook classification.

```bash
# Theme-only classification (default)
python run_pipeline.py \
    --transcript-dir ./data/input/diarized_sessions/ \
    --output-dir ./data/output/

# With codebook classification + two-pass embedding
python run_pipeline.py \
    --run-codebook-classifier \
    --transcript-dir ./data/input/diarized_sessions/ \
    --output-dir ./data/output/

# With pre-populated exemplars, custom weights
python run_pipeline.py \
    --run-codebook-classifier \
    --exemplar-import-path ./my_exemplars.json \
    --criteria-weight 0.6 \
    --exemplar-weight 0.4
```

#### Option B: `classify_and_label.py` -- Integrated Pipeline with Human Validation

Full pipeline with theme + codebook + cross-validation + interactive human validation of uncertain results. Supports OpenRouter, Replicate, and Hugging Face backends.

```bash
# Fully automated
export OPENROUTER_API_KEY="sk-or-..."
python classify_and_label.py \
    --transcript-dir ./data/diarized/ \
    --output-dir ./data/output/ \
    --run-mode auto

# Interactive mode (prompts for validation of uncertain segments)
python classify_and_label.py \
    --transcript-dir ./data/diarized/ \
    --output-dir ./data/output/ \
    --model openai/gpt-4o \
    --run-mode interactive
```

---

## Installation

### Requirements

- **Python 3.8+**
- **GPU recommended** for embedding-based segmentation and the triple-LLM codebook classifier
- **~150 GB disk** for the three Hugging Face models (if using `run_pipeline.py`)

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
| Hugging Face | N/A (local) | `--backend huggingface` (default for `run_pipeline.py`) |

---

## Entry Points

### `run_pipeline.py`

**Purpose**: Multi-model Hugging Face pipeline with cross-referencing.

**Backend**: Hugging Face (local). Automatically downloads and manages three causal LLMs:
- Llama 4 Maverick 17B (`meta-llama/Llama-4-Maverick-17B-128E-Instruct`)
- Mixtral 8x7B Instruct (`mistralai/Mixtral-8x7B-Instruct-v0.1`)
- Qwen 3 Next 80B (`Qwen/Qwen3-Next-80B-A3B-Instruct`)

**Flow**: Downloads models -> Pre-loads for verification -> Configures pipeline -> Runs 6-stage pipeline -> Summary statistics.

**Key flags**: `--run-codebook-classifier`, `--no-two-pass`, `--exemplar-import-path`, `--criteria-weight`, `--exemplar-weight`. See [CLI Reference](#cli-reference) for full list.

### `classify_and_label.py`

**Purpose**: Integrated classification + labeling with human validation.

**Backends**: OpenRouter (default), Replicate, Ollama, Hugging Face.

**Three modes**:
| Mode | Behavior |
|---|---|
| `auto` | Fully automated, no prompts |
| `interactive` | Prompts for human validation of uncertain results as they occur |
| `review` | Batch validation of all uncertain results at the end |

**Additional features over `run_pipeline.py`**:
- Cross-validation between theme and codebook labels (co-occurrence lift analysis)
- Interactive human validation UI for theme classifications, codebook disagreements, and co-occurrence anomalies
- Exports human validation log (JSONL)

**Key flags**: Same embedding/exemplar flags plus `--run-mode`, `--no-codebook-classifier`. See [CLI Reference](#cli-reference).

---

## Architecture

### Directory Structure

```
Qualitative_Research_Algorithm/
+-- shared/                          # Common infrastructure
|   +-- __init__.py
|   +-- data_structures.py           # Segment, SpeakerRun dataclasses
|   +-- llm_client.py                # Unified LLM API client (4 backends)
|   +-- model_loader.py              # Hugging Face model download/load/unload
|   +-- classification_loop.py       # Shared N-run classification with checkpointing
|   +-- majority_vote.py             # Single/multi-label voting logic
|   +-- validation.py                # Balanced evaluation set construction
|
+-- theme_labeler/                   # Single-label theme classification
|   +-- __init__.py
|   +-- theme_schema.py              # ThemeFramework, ThemeDefinition dataclasses
|   +-- config.py                    # ThemeClassificationConfig
|   +-- zero_shot_classifier.py      # Zero-shot LLM classification + consistency
|   +-- response_parser.py           # Multi-pass parsing, 8-category error taxonomy
|   +-- frameworks/
|       +-- __init__.py
|       +-- vamr.py                  # VA-MR preset (4 stages)
|
+-- codebook_classifier/             # Multi-label codebook classification
|   +-- __init__.py
|   +-- codebook_schema.py           # Codebook, CodeDefinition, CodeAssignment
|   +-- config.py                    # EmbeddingClassifierConfig, LLMCodebookConfig, EnsembleConfig
|   +-- embedding_classifier.py      # Triple-LLM two-pass embedding classifier
|   +-- llm_classifier.py            # LLM-based multi-label classification
|   +-- ensemble.py                  # Reconcile embedding vs LLM disagreements
|   +-- codebooks/
|       +-- __init__.py
|       +-- phenomenology.py         # 54-code VCE phenomenology codebook (6 domains)
|
+-- pipeline/                        # Orchestration & assembly
|   +-- __init__.py
|   +-- config.py                    # PipelineConfig (aggregates all sub-configs)
|   +-- orchestrator.py              # 6-stage pipeline engine
|   +-- transcript_ingestion.py      # Embedding-based semantic segmentation
|   +-- dataset_assembly.py          # Master dataset, adjacency index, progression
|   +-- cross_validation.py          # Theme <-> codebook co-occurrence analysis
|
+-- run_pipeline.py                  # CLI: Hugging Face multi-model pipeline
+-- classify_and_label.py            # CLI: Integrated pipeline + human validation
+-- requirements.txt                 # Core Python dependencies
+-- requirements_huggingface.txt     # Additional HF dependencies
```

### Module Dependency Graph

```
run_pipeline.py / classify_and_label.py
    |
    v
pipeline.orchestrator (6-stage execution)
    +-> pipeline.transcript_ingestion    (Stage 1: segmentation)
    +-> theme_labeler.zero_shot_classifier (Stage 3: LLM classification)
    +-> theme_labeler.response_parser    (Stage 4: response parsing)
    +-> codebook_classifier.*            (Stage 3b: optional)
    |   +-> embedding_classifier         (triple-LLM two-pass)
    |   +-> llm_classifier               (LLM multi-label)
    |   +-> ensemble                     (disagreement reconciliation)
    +-> shared.validation                (Stage 5: evaluation set)
    +-> pipeline.dataset_assembly        (Stage 6: final output)
    +-> pipeline.cross_validation        (classify_and_label.py only)

Shared infrastructure:
    shared.data_structures  (Segment, SpeakerRun)
    shared.llm_client       (LLM API abstraction, 4 backends)
    shared.model_loader     (HF model download/load/unload/cache)
    shared.classification_loop (N-run loop with checkpointing)
    shared.majority_vote    (single-label + multi-label voting)
    shared.validation       (balanced sampling)
```

---

## Core Data Structures

### Segment (`shared/data_structures.py`)

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

    # Theme labels (populated by theme_labeler, Stage 3-4)
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

    # Codebook labels (populated by codebook_classifier, Stage 3b)
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

### ThemeFramework (`theme_labeler/theme_schema.py`)

Defines a complete classification scheme. Key methods:
- `build_name_to_id_map()` -- Maps all name variants/aliases to integer IDs
- `to_prompt_string()` -- Formats themes for LLM prompting
- `to_json()` -- Exports as JSON for documentation
- `build_id_to_short_map()` -- Maps IDs to short display names

### Codebook (`codebook_classifier/codebook_schema.py`)

Defines a multi-label classification scheme. Key methods:
- `to_embedding_targets()` -- Returns list of dicts with `definition`, `criteria`, `exemplars` keys for embedding comparison
- `to_prompt_string(randomize=)` -- Formats codes for LLM prompting
- `get_codes_by_domain(domain)` -- Filters codes by domain
- `build_name_to_id_map()` -- Maps category names to code IDs

### CodeAssignment (`codebook_classifier/codebook_schema.py`)

Result of applying a single code to a segment:

```python
@dataclass
class CodeAssignment:
    code_id: str          # Machine ID
    category: str         # Display name
    confidence: float     # 0.0 - 1.0
    justification: str    # Explanation (from LLM method)
    method: str           # 'embedding', 'llm', or 'ensemble'
```

---

## Module Reference

### `shared/llm_client.py`

Unified LLM client supporting four backends:

| Backend | Config | Notes |
|---|---|---|
| `huggingface` (default) | Local models via `model_loader.py` | GPU recommended |
| `openrouter` | `api_key` required | Proprietary models (GPT-4, Claude, etc.) |
| `replicate` | `replicate_api_token` required | Open-source models |
| `ollama` | `ollama_host`, `ollama_port` | Local Ollama server |

Key methods:
- `request(prompt)` -- Single-model request, returns `(text, metadata)`
- `multi_model_request(prompt, models=)` -- Cross-references across multiple models, returns `[(model_id, text, metadata), ...]`

Features: automatic retry with exponential backoff, JSON format enforcement, metadata tracking.

### `shared/model_loader.py`

Manages three Hugging Face causal LLMs:

| Model | ID | Use |
|---|---|---|
| Llama 4 Maverick 17B | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` | Cosine similarity axis + theme classification |
| Mixtral 8x7B Instruct | `mistralai/Mixtral-8x7B-Instruct-v0.1` | Euclidean distance axis + theme classification |
| Qwen 3 Next 80B | `Qwen/Qwen3-Next-80B-A3B-Instruct` | Cosine distance axis + theme classification |

Key functions:
- `ensure_models_ready(download_if_missing=True)` -- Download and verify all models
- `load_model(model_id)` -- Load into memory (cached), returns `(model, tokenizer)`
- `unload_model(model_id)` -- Free VRAM

Models are loaded with FP16 precision and `device_map="auto"` for automatic GPU/CPU distribution.

### `shared/classification_loop.py`

Shared N-run classification loop with periodic checkpointing:
- `classify_segments(segments, client, n_runs, build_prompt, parse_response, merge_runs, ...)` -- Runs each segment through the LLM `n_runs` times, saves checkpoints every `save_interval` segments, supports resume from checkpoint
- `filter_participant_segments(segments)` -- Filters to participant-only segments

### `shared/majority_vote.py`

- `single_label_majority_vote(parsed_runs)` -- Determines majority label across N runs, computes consistency count and average confidence
- `multi_label_majority_vote(all_assignments)` -- Merges multi-label assignments via majority voting (code appears in >= 50% of runs)

### `shared/validation.py`

- `create_balanced_evaluation_set(segments_df, n_per_class=50)` -- Samples balanced subsets stratified by trial_id and label for human annotation

### `theme_labeler/zero_shot_classifier.py`

Zero-shot LLM classification with triplicate consistency checking:
- `classify_segments_zero_shot(segments, framework, config, resume_from=)` -- Main classification function. Runs each segment N times, returns raw LLM responses keyed by segment_id. Supports multi-model cross-referencing when `config.models` is populated.
- `create_content_validity_test_set(framework)` -- Extracts exemplar/subtle/adversarial test utterances from framework definitions

### `theme_labeler/response_parser.py`

Multi-pass parsing of LLM outputs with 8-category error taxonomy:
1. Invalid JSON
2. Missing required fields
3. Invalid theme name
4. Confidence out of bounds
5. Logic errors (e.g., identical primary and secondary)
6. Parsing errors
7. Type mismatches
8. Unknown errors

Key function: `parse_all_results(results_all, all_segments, name_to_id)` -- Populates `primary_stage`, `secondary_stage`, `llm_confidence_*`, `llm_justification`, `llm_run_consistency` on each Segment.

### `codebook_classifier/llm_classifier.py`

LLM-based multi-label codebook classification:
- `LLMCodebookClassifier(llm_client, config)` -- Prompts the LLM with the full codebook for each segment
- `classify_segments(segments, codebook, output_dir=)` -- Runs N times per segment with majority voting, returns `Dict[str, List[CodeAssignment]]`

### `codebook_classifier/ensemble.py`

Reconciles embedding-based and LLM-based classifications:
- `CodebookEnsemble(config).reconcile(embedding_results, llm_results)` -- Returns `Dict[str, EnsembleResult]`

Reconciliation strategies (via `EnsembleConfig.preferred_method`):
| Strategy | Behavior |
|---|---|
| `'llm'` (default) | Use LLM results as final, flag disagreements |
| `'embedding'` | Use embedding results as final |
| `'both'` | Union of both methods |
| `require_agreement=True` | Only codes agreed by both methods |

### `pipeline/cross_validation.py`

Theme-to-codebook co-occurrence analysis:
- `compute_theme_codebook_cooccurrence(segments_df, framework)` -- Computes per-theme code rates and lift values
- `validate_codebook_hypothesis(cooccurrence, framework, min_lift=1.5)` -- Tests hypothesized theme-to-code mappings against observed data, reports confirmed/unconfirmed/unexpected associations

---

## Pipeline Stages

The orchestrator (`pipeline/orchestrator.py`) executes a 6-stage pipeline:

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

The orchestrator automatically sets up `codebook_raw/` output directory and configures `exemplar_export_path` for discovered exemplars.

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

The embedding classifier (`codebook_classifier/embedding_classifier.py`) is the core of the multi-label codebook classification. It uses three causal LLMs to provide independent embedding perspectives via mean-pooled final hidden layer outputs.

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

### VRAM Management

When `sequential_loading=True` (default), models are loaded and unloaded one at a time to manage GPU memory. Each phase:
1. Load model + tokenizer
2. Embed all texts
3. Unload model (free VRAM)
4. Compute scoring matrix

---

## Configuration Reference

### `PipelineConfig` (`pipeline/config.py`)

Top-level configuration aggregating all sub-configs:

```python
@dataclass
class PipelineConfig:
    transcript_dir: str = './data/input/diarized_sessions/'
    trial_id: str = 'standard'
    output_dir: str = './data/output/'
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

### `SegmentationConfig` (`pipeline/config.py`)

| Field | Default | Description |
|---|---|---|
| `embedding_model` | `'all-MiniLM-L6-v2'` | Sentence transformer for semantic similarity |
| `min_segment_words` | `30` | Minimum words per segment |
| `max_segment_words` | `200` | Maximum words per segment |
| `silence_threshold_ms` | `1500` | Pause duration to trigger boundary |
| `semantic_shift_percentile` | `25` | Percentile for similarity drop detection |

### `ThemeClassificationConfig` (`theme_labeler/config.py`)

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

### `EmbeddingClassifierConfig` (`codebook_classifier/config.py`)

| Field | Default | Description |
|---|---|---|
| `similarity_model` | `'meta-llama/Llama-4-Maverick-17B-128E-Instruct'` | Phase 1 model (cosine similarity) |
| `distance_model` | `'mistralai/Mixtral-8x7B-Instruct-v0.1'` | Phase 2 model (Euclidean distance) |
| `tertiary_model` | `'Qwen/Qwen3-Next-80B-A3B-Instruct'` | Phase 3 model (cosine distance) |
| `similarity_threshold` | `1.375` | Phase 1 veto threshold |
| `distance_threshold` | `1.325` | Phase 2 veto threshold |
| `tertiary_threshold` | `1.35` | Phase 3 veto threshold |
| `max_codes_per_sentence` | `None` | Max codes assigned (auto = 33% of codebook, max 6) |
| `criteria_weight` | `0.5` | Weight for criteria similarity in scoring |
| `exemplar_weight` | `0.5` | Weight for exemplar similarity in scoring |
| `sequential_loading` | `True` | Load/unload models one at a time for VRAM |
| `exemplar_import_path` | `None` | Path to JSON with pre-populated exemplars |
| `exemplar_export_path` | `None` | Path to save discovered exemplars (set by orchestrator) |
| `max_exemplar_tokens` | `512` | Max word count for combined exemplar text per code |
| `exemplar_confidence_threshold` | `0.8` | Min confidence for a segment to become an exemplar |
| `two_pass` | `True` | Enable two-pass classification |

### `LLMCodebookConfig` (`codebook_classifier/config.py`)

| Field | Default | Description |
|---|---|---|
| `n_runs` | `1` | Number of LLM runs per segment |
| `max_codes_per_segment` | `5` | Maximum codes to assign |
| `confidence_threshold` | `0.5` | Minimum confidence to keep a code |
| `randomize_codebook` | `True` | Shuffle code order in prompt |
| `save_interval` | `20` | Checkpoint every N segments |

### `EnsembleConfig` (`codebook_classifier/config.py`)

| Field | Default | Description |
|---|---|---|
| `require_agreement` | `False` | Only keep codes agreed by both methods |
| `flag_disagreements` | `True` | Flag segments with method disagreements |
| `preferred_method` | `'llm'` | Which method's codes to use as final (`'llm'`, `'embedding'`, `'both'`) |

### `ConfidenceTierConfig` (`pipeline/config.py`)

| Field | Default | Description |
|---|---|---|
| `high_consistency` | `3` | Required consistency for high tier |
| `high_confidence` | `0.8` | Required confidence for high tier |
| `medium_min_consistency` | `2` | Required consistency for medium tier |
| `medium_min_confidence` | `0.6` | Required confidence for medium tier |

---

## CLI Reference

### `run_pipeline.py`

```
usage: run_pipeline.py [-h] [--transcript-dir DIR] [--output-dir DIR]
                       [--trial-id ID] [--n-runs N] [--temperature T]
                       [--high-confidence-threshold F]
                       [--medium-confidence-threshold F]
                       [--no-theme-labeler] [--run-codebook-classifier]
                       [--no-two-pass] [--exemplar-import-path PATH]
                       [--criteria-weight F] [--exemplar-weight F]
                       [--exemplar-confidence-threshold F]
                       [--max-exemplar-tokens N] [--resume-from PATH]
```

| Flag | Default | Description |
|---|---|---|
| `--transcript-dir` | `./data/input/diarized_sessions/` | Input directory |
| `--output-dir` | `./data/output/` | Output directory |
| `--trial-id` | `multi_model_trial` | Trial identifier |
| `--n-runs` | `3` | Runs per model per segment |
| `--temperature` | `0.0` | LLM temperature |
| `--high-confidence-threshold` | `0.8` | High tier threshold |
| `--medium-confidence-threshold` | `0.6` | Medium tier threshold |
| `--no-theme-labeler` | off | Skip theme classification |
| `--run-codebook-classifier` | off | Enable codebook classification |
| `--no-two-pass` | off | Disable two-pass (single pass only) |
| `--exemplar-import-path` | None | Pre-populated exemplar JSON |
| `--criteria-weight` | `0.5` | Criteria similarity weight |
| `--exemplar-weight` | `0.5` | Exemplar similarity weight |
| `--exemplar-confidence-threshold` | `0.8` | Min confidence for exemplar discovery |
| `--max-exemplar-tokens` | `512` | Max words per code's exemplar text |
| `--resume-from` | None | Checkpoint JSON to resume from |

### `classify_and_label.py`

```
usage: classify_and_label.py [-h] [--transcript-dir DIR] [--output-dir DIR]
                             [--trial-id ID] [--model MODEL] [--n-runs N]
                             [--backend {openrouter,replicate}]
                             [--api-key KEY] [--replicate-api-token TOKEN]
                             [--high-confidence-threshold F]
                             [--medium-confidence-threshold F]
                             [--resume-from PATH]
                             [--run-mode {auto,interactive,review}]
                             [--no-theme-labeler] [--no-codebook-classifier]
                             [--no-two-pass] [--exemplar-import-path PATH]
                             [--criteria-weight F] [--exemplar-weight F]
                             [--exemplar-confidence-threshold F]
                             [--max-exemplar-tokens N]
```

All flags from `run_pipeline.py` plus:

| Flag | Default | Description |
|---|---|---|
| `--model` | `openai/gpt-4o` | LLM model for classification |
| `--backend` | `openrouter` | API backend (`openrouter` or `replicate`) |
| `--api-key` | `$OPENROUTER_API_KEY` | OpenRouter API key |
| `--replicate-api-token` | `$REPLICATE_API_TOKEN` | Replicate token |
| `--run-mode` | `auto` | Validation mode (`auto`, `interactive`, `review`) |
| `--no-codebook-classifier` | off | Skip codebook classification |

---

## Usage Examples

### Example 1: Theme-Only with Hugging Face Models

```bash
python run_pipeline.py \
    --transcript-dir ./data/input/diarized_sessions/ \
    --output-dir ./data/output/
```

### Example 2: Full Pipeline with Codebook + Two-Pass

```bash
python run_pipeline.py \
    --run-codebook-classifier \
    --transcript-dir ./data/input/diarized_sessions/ \
    --output-dir ./data/output/
```

Output includes `codebook_raw/found_exemplar_utterances.json` with discovered exemplars.

### Example 3: Pre-Populated Exemplars

```bash
python run_pipeline.py \
    --run-codebook-classifier \
    --exemplar-import-path ./curated_exemplars.json \
    --criteria-weight 0.6 \
    --exemplar-weight 0.8
```

### Example 4: Single-Pass Only (Skip Exemplar Discovery)

```bash
python run_pipeline.py \
    --run-codebook-classifier \
    --no-two-pass
```

### Example 5: Interactive Validation with OpenRouter

```bash
export OPENROUTER_API_KEY="sk-or-..."

python classify_and_label.py \
    --transcript-dir ./data/diarized/ \
    --output-dir ./data/output/ \
    --model openai/gpt-4o \
    --run-mode interactive
```

### Example 6: Replicate Backend

```bash
export REPLICATE_API_TOKEN="r8_..."

python classify_and_label.py \
    --backend replicate \
    --model google-deepmind/gemma-2b \
    --run-mode auto
```

### Example 7: Resume from Checkpoint

```bash
python classify_and_label.py \
    --resume-from ./data/output/llm_raw/llm_results_openai_gpt-4o_*.json \
    --run-mode interactive
```

### Example 8: Multi-Model Comparison

Run the pipeline multiple times with different models to compare:

```bash
# Run 1: GPT-4o
python classify_and_label.py --trial-id gpt4o --model openai/gpt-4o

# Run 2: Claude
python classify_and_label.py --trial-id claude --model anthropic/claude-3-sonnet

# Run 3: Gemma via Replicate
python classify_and_label.py --trial-id gemma --backend replicate --model google-deepmind/gemma-2b
```

Compare results across `trial_id` values in the master segments output.

### Example 9: Custom Confidence Thresholds

```bash
python run_pipeline.py \
    --high-confidence-threshold 0.7 \
    --medium-confidence-threshold 0.5
```

More segments classified as "high" and "medium" confidence.

---

## Extending QRA

### Adding a New Theme Framework

Create `theme_labeler/frameworks/my_framework.py`:

```python
from ..theme_schema import ThemeFramework, ThemeDefinition

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
        subtle_utterances=['Subtle example 1'],
        adversarial_utterances=['Could be multiple themes'],
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

Use in pipeline:

```python
from pipeline.orchestrator import run_full_pipeline
from theme_labeler.frameworks.my_framework import get_my_framework

framework = get_my_framework()
run_full_pipeline(config, framework)
```

### Adding a New Codebook

Create `codebook_classifier/codebooks/my_codebook.py`:

```python
from ..codebook_schema import Codebook, CodeDefinition

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

Pass to the orchestrator:

```python
from codebook_classifier.codebooks.my_codebook import get_my_codebook

codebook = get_my_codebook()
run_full_pipeline(config, framework, codebook=codebook)
```

### Custom Segmentation

Override the segmentation by creating `Segment` objects directly:

```python
from shared.data_structures import Segment

def my_custom_segmenter(sentences, metadata) -> List[Segment]:
    segments = []
    # Your segmentation logic...
    return segments
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

### `human_coding_evaluation_set.csv`

Balanced sample of segments for human annotation. Stratified by trial_id and label.

### `theme_definitions.json`

Complete framework specification as JSON.

### `content_validity_test_set.jsonl`

Test utterances extracted from framework definitions (exemplar/subtle/adversarial).

### `session_adjacency_[timestamp].jsonl`

Adjacency relationships between segments within sessions.

### `session_stage_progression_[timestamp].csv`

Per-session analysis of stage transitions (forward/backward counts).

### `llm_raw/llm_results_[model]_[timestamp].json`

Raw LLM responses before parsing. Acts as checkpoint for resumption.

### `codebook_raw/codebook_llm_results_[timestamp].json`

Raw LLM codebook classification results.

### `cross_validation_results_[timestamp].json` (classify_and_label.py only)

Theme-to-codebook co-occurrence validation with confirmed/unconfirmed/unexpected associations.

### `human_validation_log_[timestamp].jsonl` (classify_and_label.py only)

Log of all human validation decisions made during interactive/review mode.

---

## Key Design Decisions

### 1. Segment as Central Data Structure

All operations produce/consume `Segment` objects. Fields are populated progressively -- each pipeline stage adds to the Segment without overwriting prior fields, enabling multi-method ensemble approaches.

### 2. Framework-Agnostic Pipeline

The pipeline accepts any `ThemeFramework`, not hardcoded to VA-MR. Theme name resolution via `build_name_to_id_map()`, prompt generation parameterized by framework.

### 3. Triple-LLM Embedding (Not BERT)

The embedding classifier uses three full causal LLMs (Llama 4 Maverick, Mixtral 8x7B, Qwen 3 Next) rather than traditional sentence transformers. Each model provides a different embedding perspective via mean-pooled hidden states, and all three must agree (triple-veto) for a code to be assigned.

### 4. Two-Pass Exemplar Discovery

Pass 1 identifies high-confidence segments. Pass 2 uses these as additional comparison targets, improving accuracy on codes that have few hand-written exemplars. Discovered exemplars are saved for reuse across runs.

### 5. Additive Scoring (No Subtraction)

The scoring formula is purely additive: `definition + criteria_weight * criteria + exemplar_weight * exemplars`. The old subtractive exclusive_criteria term was removed because the triple-veto system already prevents false positives through its three independent thresholds.

### 6. Triplicate Consistency

Each segment is classified 3 times. Segments with 3/3 agreement are high confidence. This detects LLM instability even at temperature=0 (token randomness in long sequences).

### 7. Sequential Model Loading

Models are loaded and unloaded one at a time to manage VRAM. This trades speed for memory efficiency, enabling the three-model system to run on a single GPU.

### 8. Backward Compatibility

`primary_stage`/`secondary_stage` field names are preserved for backward compatibility with historical VA-MR analysis code.

### 9. Cross-Module Absolute Imports

`from shared.data_structures import Segment` -- clear dependency graph, avoids circular imports, supports CLI entry points.

### 10. Stateless LLM Client

Each segment classified independently. No session/conversation state. Enables parallelization and checkpoint resumption.

---

## Troubleshooting

### API Key Issues

```
Error: No API key provided
```

**Solution**: Set the environment variable or use the CLI flag:
```bash
export OPENROUTER_API_KEY="sk-or-..."
# or
python classify_and_label.py --api-key sk-or-...
```

### CUDA Out of Memory

```
RuntimeError: CUDA out of memory
```

**Solutions**:
1. Enable sequential model loading (default): `sequential_loading=True`
2. Use CPU: `export CUDA_VISIBLE_DEVICES=""`
3. Reduce `max_length` in tokenizer (edit `embedding_classifier.py`)
4. Use FP16 (already default)

### JSON Parsing Errors

```
extract_json() returned None
```

**Solutions**:
1. Check raw LLM responses in `llm_raw/` directory
2. Use `temperature=0.0` for deterministic output
3. Use a more capable model (GPT-4o > GPT-3.5)

### Segmentation Quality

If segments are too long or too short:
```bash
# Adjust via SegmentationConfig in code, or modify defaults in pipeline/config.py
# min_segment_words=50, max_segment_words=150, silence_threshold_ms=1000
```

### Model Download Issues

```
Error downloading model
```

**Solutions**:
1. Check internet connectivity
2. Accept model license on Hugging Face (some models require agreement)
3. Set `HF_HOME` environment variable for custom cache directory
4. Check disk space (~150 GB needed for all three models)

---

## License

[Include your license here]

---

## Contact

For questions or issues, please contact the development team.
