# QRA: Qualitative Research Algorithm

A comprehensive LLM-based classification pipeline for analyzing therapeutic dialogue transcripts. QRA classifies dialogue segments into theme/stage categories (e.g., the VA-MR framework: Vigilance, Avoidance, Metacognition, Reappraisal) and optionally applies multi-label phenomenology codebook classification.

## Overview

QRA is designed for qualitative research in psychotherapy and mindfulness-based interventions. It takes diarized transcripts (typically from speech-to-text pipelines like Whisper with speaker diarization) and produces structured, coded datasets suitable for statistical analysis and thematic interpretation.

### Key Capabilities

- **Semantic Segmentation**: Embedding-based segmentation with adaptive thresholds, topic clustering, and optional LLM-assisted boundary refinement
- **Theme Classification**: Zero-shot LLM classification into theoretical frameworks (e.g., VA-MR stages of contemplative transformation)
- **Codebook Classification**: Multi-label coding via embedding similarity + LLM zero-shot prompting, reconciled by ensemble
- **Interrater Reliability**: Multi-model runs with consensus voting for cross-rater reliability metrics
- **Validation Test Sets**: Stratified cross-session sampling for human blind-coding
- **Therapist Cue Analysis**: Surfaces therapist dialogue at stage transitions for interpretation
- **Automated Analysis**: Post-pipeline analysis generating longitudinal reports, per-participant summaries, figures, and graph-ready CSVs

## Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Transcript Ingestion & Segmentation               │
│  - Load diarized transcripts (JSON/VTT)                     │
│  - Semantic segmentation via sentence-transformer           │
│  - Adaptive threshold + topic clustering                    │
│  - Optional LLM-assisted boundary refinement                │
│  - Speaker normalization and anonymization                  │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 2: Construct Operationalization                      │
│  - Build theme framework definitions                        │
│  - Create content validity test set                         │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 3: Theme Classification (LLM)                        │
│  - Zero-shot LLM classification with framework prompting    │
│  - Multi-model consensus voting for interrater reliability  │
│  - Context-aware classification with preceding dialogue     │
└───────────────────────────▼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 3b: Codebook Classification (Optional)               │
│  - Embedding-based similarity scoring                       │
│  - LLM zero-shot multi-label coding                         │
│  - Ensemble reconciliation of both methods                  │
└───────────────────────────▼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 4: Cross-Validation (Optional, requires 3b)          │
│  - Theme ↔ codebook co-occurrence analysis                  │
│  - Lift and statistical validation of hypotheses            │
└───────────────────────────▼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 5: Human Validation Set                              │
│  - Create balanced evaluation set for human coding          │
│  - Export validation test set worksheets                    │
└───────────────────────────▼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 6: Dataset Assembly                                  │
│  - Assemble master segment dataset (JSONL)                  │
│  - Apply confidence tiering                                 │
│  - Generate training labels                                 │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 7: Report Generation                                 │
│  - Coded transcripts (per-session)                          │
│  - Human blind-coding forms                                 │
│  - Validation test set worksheets                           │
│  - Per-session statistics and cumulative report             │
│  - Training data export (JSONL)                             │
│  - Output directory index (00_index.txt)                    │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 8: Results Analysis (Optional, --auto-analyze)       │
│  - Per-participant longitudinal reports                     │
│  - Per-session and per-construct analyses                   │
│  - Graph-ready CSVs                                         │
│  - Stage progression and transition explanation             │
│  - Therapist cue response analysis                          │
│  - Visualization figures                                    │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Diarized transcripts from Whisper (JSON or VTT format)
2. **Processing**: Segments → LLM classification → Consensus voting
3. **Output**: Master dataset, coded transcripts, analysis reports

## Installation & Dependencies

### Requirements

- Python 3.9–3.11 (Python 3.12 may have compatibility issues with some dependencies)
- PyTorch (for HuggingFace models and sentence-transformers)
- sentence-transformers (for embedding-based segmentation and codebook classification)
- huggingface-hub (for model downloading)
- pandas, scikit-learn (for data processing)

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd Qualitative_Research_Algorithm

# Install dependencies (recommended in virtual environment)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### LLM Backend Options

QRA supports multiple LLM backends:

| Backend | Setup |
|---------|-------|
| **LM Studio** | Start LM Studio, load a model, start server (default: http://127.0.0.1:1234/v1) |
| **OpenRouter** | `export OPENROUTER_API_KEY=your_key` |
| **Replicate** | `export REPLICATE_API_TOKEN=your_token` |
| **HuggingFace** | Models downloaded automatically; GPU recommended for large models |
| **Ollama** | Install Ollama locally, pull desired model: `ollama pull llama3` |

## Usage

### Command-Line Interface

```bash
python qra.py <command> [options]
```

#### Commands

##### 1. Setup Wizard (Interactive Configuration)

```bash
python qra.py setup
```

Runs an interactive 12-step configuration wizard that:
- Configures input/output paths and speaker anonymization
- Identifies therapist vs participant speakers
- Sets segmentation parameters (including LLM refinement options)
- Selects LLM backend and model
- Configures multi-model interrater reliability
- Chooses theme framework and optional codebook
- Configures confidence thresholds and validation test sets
- Sets up therapist cue summarization

Saves configuration to `07_meta/qra_config.json` for reproducible runs.

##### 2. Run Pipeline

```bash
# With saved config
python qra.py run --config ./data/output/07_meta/qra_config.json

# With inline configuration
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --transcript-dir ./data/input/ \
  --output-dir ./data/output/

# Run pipeline and automatically generate analysis reports afterward
python qra.py run --config ./qra_config.json --auto-analyze
```

##### 3. Analyze (Post-Hoc Results Analysis)

```bash
python qra.py analyze --output-dir ./data/output/
```

Runs post-hoc analysis on existing pipeline output to generate:
- Per-participant longitudinal reports (VA-MR progression)
- Per-session summaries with prototypical exemplars
- Per-construct (stage + codebook) analyses
- Longitudinal text reports and transition explanations
- Therapist cue response analysis
- Graph-ready CSVs and visualization figures

##### 4. Testsets (Generate Validation Worksheets)

```bash
python qra.py testsets --output-dir ./data/output/

# With options
python qra.py testsets \
  --output-dir ./data/output/ \
  --config ./data/output/07_meta/qra_config.json \
  --n-sets 3 \
  --fraction 0.15
```

Generates cross-session human coding and AI classification worksheets from an existing pipeline run, for inter-rater reliability assessment.

### Configuration Options

#### Common Arguments (`run` subcommand)

| Argument | Description |
|----------|-------------|
| `--config, -c` | Path to saved config JSON |
| `--transcript-dir` | Directory containing input transcripts |
| `--output-dir` | Output directory for results |
| `--trial-id` | Trial identifier |
| `--resume-from` | Checkpoint path to resume an interrupted run |
| `--auto-analyze` | Run results analysis automatically after pipeline completes |

#### Backend & Model Arguments

| Argument | Description |
|----------|-------------|
| `--backend` | LLM backend: openrouter, replicate, huggingface, ollama, lmstudio |
| `--lmstudio-url` | LM Studio server URL (default: http://127.0.0.1:1234/v1) |
| `--model` | Primary model ID for classification |
| `--models` | Multiple model IDs for cross-referencing |
| `--api-key` | OpenRouter API key (or use `OPENROUTER_API_KEY` env var) |
| `--replicate-api-token` | Replicate API token (or use `REPLICATE_API_TOKEN` env var) |

#### Framework & Codebook Arguments

| Argument | Description |
|----------|-------------|
| `--framework` | Theme framework: `vamr` (default) or path to custom JSON |
| `--codebook` | Codebook: `phenomenology` (default) or path to custom JSON |

#### Classification Parameters

| Argument | Description |
|----------|-------------|
| `--n-runs` | Number of classification runs per segment (for IRR) |
| `--temperature` | LLM temperature |
| `--high-confidence-threshold` | High confidence tier threshold |
| `--medium-confidence-threshold` | Medium confidence tier threshold |

#### Feature Flags

| Argument | Description |
|----------|-------------|
| `--no-theme-labeler` | Skip theme classification |
| `--run-codebook-classifier` | Enable codebook classification |
| `--no-codebook-classifier` | Disable codebook classification |
| `--verbose-segmentation` | Write detailed process log to `07_meta/process_log.txt` |

#### Embedding Classifier Arguments

| Argument | Description |
|----------|-------------|
| `--embedding-model` | Sentence-transformer model (default: Qwen/Qwen3-Embedding-8B) |
| `--no-two-pass` | Disable two-pass embedding classification |
| `--exemplar-import-path` | Path to pre-built exemplar embeddings |
| `--criteria-weight` | Weight for criteria similarity in ensemble scoring |
| `--exemplar-weight` | Weight for exemplar similarity in ensemble scoring |

#### Speaker Filtering

| Argument | Description |
|----------|-------------|
| `--speaker-filter-mode` | `none` (classify all) or `exclude` (drop listed speakers) |
| `--exclude-speakers` | Speaker labels to exclude (use with `--speaker-filter-mode exclude`) |

### Example Workflows

#### Complete Pipeline with OpenRouter

```bash
# 1. Run setup wizard
python qra.py setup

# 2. Run the pipeline
python qra.py run --config ./data/output/07_meta/qra_config.json

# 3. Analyze results (or add --auto-analyze to step 2)
python qra.py analyze --output-dir ./data/output/
```

#### Multi-Model Interrater Reliability

```bash
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --models openai/gpt-4o anthropic/claude-3.5-sonnet google/gemini-2.0-flash-exp \
  --n-runs 3 \
  --run-codebook-classifier
```

#### Local LM Studio

```bash
# Start LM Studio server first, then:
python qra.py run \
  --backend lmstudio \
  --lmstudio-url http://localhost:1234/v1 \
  --model nvidia/nemotron-3-super \
  --auto-analyze
```

#### HuggingFace Models (No API Required)

```bash
export CUDA_VISIBLE_DEVICES=0

python qra.py run \
  --backend huggingface \
  --model meta-llama/Llama-4-Maverick-17B-128E-Instruct \
  --transcript-dir ./data/input/ \
  --output-dir ./data/output/
```

## Theme Frameworks

### VA-MR (Vigilance-Avoidance Metacognition-Reappraisal)

The default framework for mindfulness-based interventions:

| Stage | Key | Description |
|-------|-----|-------------|
| **0** | Vigilance | Pain hypervigilance and attention dysregulation |
| **1** | Avoidance | Attention control deployed for experiential avoidance |
| **2** | Metacognition | Observing mental processes without identification |
| **3** | Reappraisal | Fundamental reinterpretation of sensory experience |

### Custom Frameworks

Create a JSON file to define your own theme framework:

```json
{
  "framework": "Custom Framework",
  "version": "1.0",
  "description": "Your framework description",
  "themes": [
    {
      "theme_id": 0,
      "key": "category_a",
      "name": "Category A",
      "short_name": "A",
      "prompt_name": "category_a",
      "definition": "Description of category A...",
      "prototypical_features": ["feature1", "feature2"],
      "distinguishing_criteria": "What makes this unique",
      "exemplar_utterances": ["Example utterance 1", "Example utterance 2"]
    }
  ]
}
```

## Codebook Classification

The codebook classifier applies multi-label phenomenology codes to segments using two complementary methods:

### Embedding-Based Classification

- Uses sentence-transformer embeddings (default: `Qwen/Qwen3-Embedding-8B`)
- Asymmetric encoding: segments encoded as queries, codebook entries as passages
- Scoring based on definition + criteria + exemplar similarity
- Optional two-pass approach for higher precision

### LLM Zero-Shot Classification

- Reuses the configured theme classification backend
- Multi-label prompting with codebook definitions
- Majority voting across runs

### Ensemble Reconciliation

Combines both methods; disagreements are flagged for review and confidence scores from each method are preserved in the segment.

## Output Directory Structure

After a complete pipeline run, the output directory is organized as follows:

```
output_dir/
├── 00_index.txt                          # Auto-generated file index with sizes
│
├── 01_transcripts/
│   └── coded/
│       └── coded_transcript_<session>.txt  # Human-readable coded transcripts
│
├── 02_human_reports/
│   ├── per_session/
│   │   └── session_<id>.json             # Per-session analysis reports
│   ├── per_participant/
│   │   └── participant_<id>.json         # Per-participant longitudinal reports
│   └── per_construct/
│       ├── construct_<stage>.json        # Per-stage analysis
│       └── codebook_exemplars.txt        # Codebook exemplar report
│
├── 03_figures/
│   └── *.png                             # Visualization figures
│
├── 04_analysis_data/
│   ├── session_stats/
│   │   └── stats_<session>.json          # Per-session statistics
│   ├── graphing/
│   │   └── *.csv                         # Graph-ready CSVs for R/Python
│   ├── cumulative_report.json            # Dataset-wide summary
│   ├── longitudinal_summary.json         # Overall progression summary
│   └── session_stage_progression.csv    # Session-level stage progression
│
├── 05_validation/
│   ├── content_validity_test_set.jsonl   # Content validity items
│   ├── human_coding_evaluation_set.csv  # Balanced evaluation set
│   ├── human_classification_<session>.txt  # Blind-coding forms
│   ├── flagged_for_review.txt           # Segments needing review
│   ├── testsets/
│   │   ├── human_classification_testset_worksheet_1.txt
│   │   └── AI_classification_testset_worksheet_1.txt
│   └── cross_validation/
│       ├── cross_validation_results.json
│       └── top_theme_code_associations.json
│
├── 06_training_data/
│   ├── master_segments.jsonl             # Complete master dataset
│   ├── theme_classification.jsonl        # Training data for supervised models
│   └── codebook_multilabel.jsonl         # Multi-label codebook training data
│
└── 07_meta/
    ├── qra_config.json                   # Pipeline configuration
    ├── speaker_anonymization_key.json    # Speaker ID mapping
    ├── theme_definitions.json            # Framework definitions
    ├── codebook_definitions.json         # Codebook definitions (if enabled)
    └── process_log.txt                   # Verbose LLM I/O log (if --verbose-segmentation)
```

## Data Structures

### Segment

The atomic unit of classification:

```python
@dataclass
class Segment:
    # Identity
    segment_id: str
    trial_id: str
    participant_id: str
    session_id: str
    session_number: int
    cohort_id: Optional[int]
    session_variant: str           # '' for normal, 'a'/'b' for split sessions

    # Temporal
    segment_index: int
    start_time_ms: int
    end_time_ms: int
    total_segments_in_session: int

    # Content
    speaker: str                   # 'participant' | 'therapist'
    text: str
    word_count: int

    # Theme classification
    primary_stage: Optional[int]
    secondary_stage: Optional[int]
    llm_confidence_primary: Optional[float]
    llm_confidence_secondary: Optional[float]
    llm_justification: Optional[str]
    llm_run_consistency: Optional[int]

    # Interrater reliability
    rater_ids: Optional[List[str]]
    rater_votes: Optional[List[Dict]]
    agreement_level: Optional[str]     # 'unanimous'|'majority'|'split'|'none'
    agreement_fraction: Optional[float]
    needs_review: bool
    consensus_vote: Optional[object]
    tie_broken_by_confidence: bool

    # Codebook labels
    codebook_labels_embedding: Optional[List[str]]
    codebook_labels_llm: Optional[List[str]]
    codebook_labels_ensemble: Optional[List[str]]
    codebook_disagreements: Optional[List[str]]
    codebook_confidence: Optional[Dict[str, float]]

    # Human validation
    human_label: Optional[int]
    adjudicated_label: Optional[int]
    in_human_coded_subset: bool
    label_status: str              # 'llm_only' | 'human_coded' | 'adjudicated'

    # Final training label
    final_label: Optional[int]
    final_label_source: Optional[str]
    label_confidence_tier: Optional[str]  # 'high'|'medium'|'low'|'unclassified'
```

### PipelineConfig

Top-level configuration object:

```python
@dataclass
class PipelineConfig:
    transcript_dir: str
    output_dir: str
    trial_id: str
    run_theme_labeler: bool
    run_codebook_classifier: bool
    speaker_anonymization_key_path: Optional[str]
    auto_analyze: bool
    resume_from: Optional[str]

    segmentation: SegmentationConfig
    speaker_filter: SpeakerFilterConfig
    theme_classification: ThemeClassificationConfig
    codebook_embedding: EmbeddingClassifierConfig
    codebook_llm: LLMCodebookConfig
    codebook_ensemble: EnsembleConfig
    validation: ValidationConfig
    test_sets: TestSetConfig
    confidence_tiers: ConfidenceTierConfig
    therapist_cues: TherapistCueConfig
```

## Module Reference

### Classification Tools (`classification_tools/`)

| File | Description |
|------|-------------|
| `llm_client.py` | Unified LLM API client (OpenRouter, Replicate, Ollama, LM Studio, HuggingFace) |
| `data_structures.py` | Segment dataclass and core data structures |
| `llm_classifier.py` | Theme and codebook LLM classification logic |
| `majority_vote.py` | Interrater reliability voting aggregation |
| `response_parser.py` | Parse LLM outputs into structured format |
| `reliability.py` | Reliability metrics (Krippendorff's alpha, etc.) |
| `validation.py` | Create evaluation sets and consistency checking |
| `model_loader.py` | HuggingFace model downloading and loading |
| `classification_loop.py` | Classification loop with checkpointing |

### Constructs (`constructs/`)

| File | Description |
|------|-------------|
| `theme_schema.py` | `ThemeFramework` and `ThemeDefinition` dataclasses |
| `vamr.py` | VA-MR framework definitions (default) |
| `config.py` | `ThemeClassificationConfig` dataclass |

### Codebook (`codebook/`)

| File | Description |
|------|-------------|
| `codebook_schema.py` | `Codebook` and `CodeDefinition` dataclasses |
| `phenomenology_codebook.py` | Default phenomenology codebook |
| `embedding_classifier.py` | Sentence-transformer based classification |
| `ensemble.py` | Combine embedding + LLM results |
| `config.py` | Embedding and ensemble configuration dataclasses |

### Process (`process/`)

| File | Description |
|------|-------------|
| `orchestrator.py` | Main pipeline orchestration (7 stages + optional analysis) |
| `config.py` | `PipelineConfig` and sub-config dataclasses |
| `setup_wizard.py` | Interactive 12-step configuration wizard |
| `transcript_ingestion.py` | Load VTT/JSON, speaker normalization and anonymization |
| `llm_segmentation.py` | LLM-assisted segmentation boundary refinement |
| `dataset_assembly.py` | Export master dataset and reports |
| `speaker_filter.py` | Apply speaker inclusion/exclusion rules |
| `output_paths.py` | Single source of truth for all output directory paths |
| `output_index.py` | Generate `00_index.txt` at pipeline completion |
| `process_logger.py` | Verbose LLM I/O and segmentation logging |
| `cross_validation.py` | Theme ↔ codebook co-occurrence and lift analysis |
| `validation_exports.py` | Validation artifact export helpers |

### Analysis (`analysis/`)

| File | Description |
|------|-------------|
| `runner.py` | Post-hoc analysis orchestrator (8 sub-steps) |
| `loader.py` | Load master JSONL and framework from output directory |
| `participant.py` | Per-participant report generation |
| `session.py` | Per-session analysis |
| `construct.py` | Per-construct (stage + code) analyses |
| `stage_progression.py` | Session-level stage progression computation |
| `longitudinal.py` | Longitudinal summary generation |
| `figure_data.py` | Export graph-ready CSV datasets |
| `figures.py` | Generate visualization figures (matplotlib) |
| `exemplars.py` | Exemplar utterance extraction per stage |
| `text_reports.py` | Human-readable text report utilities |
| `reports/` | Detailed text report generators (session, stage, transition, cue response, longitudinal) |

## LLM Backend Configuration

### LM Studio (Local GUI)

1. Download and start LM Studio from lmstudio.ai
2. Load a model and start the local server (default port 1234)
3. Run: `python qra.py run --backend lmstudio --lmstudio-url http://127.0.0.1:1234/v1 --model <model-name>`

### OpenRouter

```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python qra.py run --backend openrouter --model openai/gpt-4o
```

### Replicate

```bash
export REPLICATE_API_TOKEN=r8_...
python qra.py run --backend replicate --model meta/llama-2-70b-chat
```

### Ollama (Local)

```bash
ollama pull llama3
python qra.py run --backend ollama --model llama3
```

### HuggingFace (Local GPU)

```bash
export CUDA_VISIBLE_DEVICES=0
python qra.py run --backend huggingface --model meta-llama/Llama-4-Maverick-17B-128E-Instruct
```

## Confidence Tiers

Segments are assigned to confidence tiers based on LLM agreement across runs:

| Tier | Consistency | Confidence | Description |
|------|-------------|------------|-------------|
| **High** | unanimous | >0.8 | All raters agree with high confidence |
| **Medium** | majority | >0.6 | Majority agreement or good single-run confidence |
| **Low** | minority | <0.6 | Split votes or low confidence |
| **Unclassified** | none | — | No consensus reached |

## Citation

If you use QRA in your research, please cite:

```
@software{QRA2024,
  title={QRA: Qualitative Research Algorithm},
  author={Wade Balsamo},
  year={2026},
  url={https://github.com/wadebalsamo/qra}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- VA-MR framework based on research by [Wexler, Balsamo et al.]
- Powered by LLMs from OpenRouter, Replicate, HuggingFace, Ollama, and LM Studio
- Embeddings powered by sentence-transformers and Qwen models
