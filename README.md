# QRA: Qualitative Research Algorithm

A comprehensive LLM-based classification pipeline for analyzing therapeutic dialogue transcripts. QRA classifies dialogue segments into theme/stage categories (e.g., VA-MR framework: Vigilance, Avoidance, Metacognition, Reappraisal) and optionally applies multi-label phenomenology codebook classification.

## Overview

QRA is designed for qualitative research in psychotherapy and mindfulness-based interventions. It takes diarized transcripts (typically from speech-to-text pipelines like Whisper with speaker diarization) and produces structured, coded datasets suitable for statistical analysis and thematic interpretation.

### Key Capabilities

- **Theme Classification**: Classify segments into theoretical frameworks (e.g., VA-MR stages of contemplative transformation)
- **Codebook Classification**: Multi-label coding using a phenomenology codebook via embedding similarity + LLM zero-shot prompting
- **Interrater Reliability**: Multiple LLM runs with consensus voting for reliability metrics
- **Human-in-the-Loop Validation**: Interactive and batch modes for validating uncertain classifications
- **Automated Analysis**: Post-pipeline analysis generating longitudinal reports, per-participant summaries, and visualizations

## Architecture

### Pipeline Stages

```
┌─────────────────────────────────────────────────────────────┐
│  Stage 1: Transcript Ingestion & Segmentation               │
│  - Load diarized transcripts (VTT/JSON)                     │
│  - Semantic segmentation using sentence-transformer         │
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
│  - Multi-run consensus voting for reliability               │
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
│  Stage 4: Cross-Validation (Optional)                       │
│  - Theme ↔ codebook co-occurrence analysis                  │
│  - Lift and statistical validation of hypotheses            │
└───────────────────────────▼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 5: Human Validation Set                              │
│  - Create balanced evaluation set for human coding          │
│  - Export forms for blind-coding validation                 │
└───────────────────────────▼─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 6: Dataset Assembly                                  │
│  - Assemble master segment dataset                          │
│  - Apply confidence tiering                                 │
│  - Generate final training labels                           │
└─────────────────────────────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│  Stage 7: Report Generation                                 │
│  - Coded transcripts                                        │
│  - Per-session statistics                                   │
│  - Human classification forms                               │
│  - Flagged-for-review reports                               │
│  - Training data export                                     │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Diarized transcripts from Whisper (JSON or VTT format)
2. **Processing**: Segments → LLM classification → Consensus voting
3. **Output**: Master dataset, coded transcripts, analysis reports

## Installation & Dependencies

### Requirements

- Python 3.9+
- PyTorch (for HuggingFace models and sentence-transformers)
- sentence-transformers (for embedding-based classification)
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
pip install -e .

# Or install required packages directly
pip install torch sentence-transformers huggingface-hub pandas scikit-learn replicate
```

### LLM Backend Options

QRA supports multiple LLM backends:

| Backend | Setup |
|---------|-------|
| **OpenRouter** | `export OPENROUTER_API_KEY=your_key` |
| **Replicate** | `export REPLICATE_API_TOKEN=your_token` |
| **HuggingFace** | Models downloaded automatically; GPU recommended for large models |
| **Ollama** | Install Ollama locally, pull desired model: `ollama pull llama3` |
| **LM Studio** | Download from lmstudio.ai, start local server |

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
- Configures input/output paths
- Identifies therapist vs participant speakers
- Sets segmentation parameters
- Selects LLM backend and model
- Chooses theme framework and codebook
- Configures classification parameters

Saves configuration to a JSON file for reproducible runs.

##### 2. Run Pipeline

```bash
# With saved config
python qra.py run --config ./data/output/meta/qra_config.json

# With inline configuration
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --transcript-dir ./data/input/ \
  --output-dir ./data/output/
```

**Run Modes:**
- `auto` (default): Fully automated, no human intervention
- `interactive`: Prompt for validation of uncertain results during pipeline
- `review`: Batch validation at end of pipeline

##### 3. Guided Mode (Educational)

```bash
python qra.py guided --config ./data/output/meta/qra_config.json
```

Executes the pipeline with step-by-step educational narration, ideal for learning the methodology or demonstrating to others.

##### 4. Analysis (Post-Hoc)

```bash
python qra.py analyze --output-dir ./data/output/
```

Runs post-hoc analysis on existing pipeline output to generate:
- Per-participant longitudinal reports (VA-MR progression)
- Per-session summaries with prototypical exemplars
- Per-construct (stage + codebook) analyses
- Graph-ready CSVs for visualization
- Longitudinal summary and transition explanations

### Configuration Options

#### Common Arguments

| Argument | Description |
|----------|-------------|
| `--config, -c` | Path to saved config JSON |
| `--transcript-dir` | Directory containing input transcripts |
| `--output-dir` | Output directory for results |
| `--trial-id` | Trial identifier (for multi-trial studies) |

#### Backend & Model Arguments

| Argument | Description |
|----------|-------------|
| `--backend` | LLM backend: openrouter, replicate, huggingface, ollama, lmstudio |
| `--lmstudio-url` | LM Studio server URL (default: http://127.0.0.1:1234/v1) |
| `--model` | Primary model ID for classification |
| `--models` | Multiple model IDs for cross-referencing |
| `--api-key` | OpenRouter API key |
| `--replicate-api-token` | Replicate API token |

#### Framework & Codebook Arguments

| Argument | Description |
|----------|-------------|
| `--framework` | Theme framework: "vamr" (default) or path to custom JSON |
| `--codebook` | Codebook: "phenomenology" (default) or path to custom JSON |

#### Classification Parameters

| Argument | Description |
|----------|-------------|
| `--n-runs` | Number of classification runs per segment (for IRR) |
| `--temperature` | LLM temperature for stochastic variation |
| `--high-confidence-threshold` | High confidence tier threshold |
| `--medium-confidence-threshold` | Medium confidence tier threshold |

#### Feature Flags

| Argument | Description |
|----------|-------------|
| `--no-theme-labeler` | Skip theme classification |
| `--run-codebook-classifier` | Enable codebook classification |
| `--no-codebook-classifier` | Disable codebook classification |
| `--verbose-segmentation` | Write detailed process log |

#### Speaker Filtering

| Argument | Description |
|----------|-------------|
| `--speaker-filter-mode` | none, exclude, or isolate |
| `--exclude-speakers` | Speaker labels to exclude from classification |
| `--isolate-speakers` | Speaker labels to keep (exclude others) |

### Example Workflows

#### Complete Pipeline with OpenRouter

```bash
python qra.py setup

# Edit config if needed
nano ./data/output/meta/qra_config.json

# Run the pipeline
python qra.py run --config ./data/output/meta/qra_config.json

# Analyze results
python qra.py analyze --output-dir ./data/output/
```

#### Multi-Model Interrater Reliability

```bash
python qra.py run \
  --backend openrouter \
  --model openai/gpt-4o \
  --models openai/gpt-4o anthropic/claude-3.5-sonco google/gemini-2.0-flash-exp \
  --n-runs 3 \
  --codebook phenomenology
```

#### Local LM Studio with VA-MR Framework

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
export CUDA_VISIBLE_DEVICES=0  # For GPU acceleration

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
      "definition": "Description of category A...",
      "prototypical_features": ["feature1", "feature2"],
      "distinguishing_criteria": "What makes this unique",
      "exemplar_utterances": ["Example utterance 1", "Example utterance 2"]
    }
  ]
}
```

## Codebook Classification

The codebook classifier applies multi-label phenomology codes to segments using two complementary methods:

### Embedding-Based Classification

- Uses sentence-transformer embeddings (default: Qwen/Qwen3-Embedding-8B)
- Asymmetric encoding for better construct retrieval
- Scoring based on definition + criteria + exemplar similarity

### LLM Zero-Shot Classification

- Reuses the configured theme classification backend
- Multi-label prompting with codebook definitions
- Strict majority voting across runs

### Ensemble Reconciliation

Combines both methods:
- Codes assigned by either method are considered
- Disagreements flagged for review
- Confidence scores from each method preserved

## Output Structure

```
output_dir/
├── meta/
│   ├── speaker_anonymization_key.json    # Speaker ID mapping
│   ├── theme_definitions.json            # Framework definitions
│   └── codebook_definitions.json         # Codebook definitions (if enabled)
├── reports/
│   ├── coded_transcript_<session>.txt    # Human-readable coded transcripts
│   ├── validation/
│   │   ├── human_classification_<session>.txt  # Blind-coding forms
│   │   └── flagged_for_review.txt        # Segments needing review
│   ├── per_transcript/
│   │   └── stats_<session>.json          # Per-session statistics
│   ├── cumulative_report_<timestamp>.json
│   └── analysis/                         # Post-hoc analysis outputs
│       ├── participants/                 # Per-participant reports
│       ├── sessions/                     # Per-session analyses
│       ├── constructs/                   # Per-construct analyses
│       │   ├── json/
│       │   └── codebook_exemplars.txt
│       ├── graphing/                     # CSVs for visualization
│       ├── longitudinal_summary.json     # Overall progression summary
│       └── figures/                      # Generated plots
├── trainingdata/
│   ├── theme_classification.jsonl        # Training data for supervised models
│   └── codebook_multilabel.jsonl         # Multi-label codebook training data
├── cross_validation_results_<timestamp>.json
├── cooccurrence_anomalies_<timestamp>.json
└── master_segments_<timestamp>.jsonl     # Complete master dataset
```

## Data Structures

### Segment

The atomic unit of classification:

```python
@dataclass
class Segment:
    segment_id: str                    # Unique identifier
    participant_id: str                # Anonymized participant ID
    session_id: str                    # Session identifier
    speaker: str                       # Speaker role (participant/therapist)
    text: str                          # Segment text
    primary_stage: int | None          # Primary theme classification
    secondary_stage: int | None        # Secondary theme (if dual-label)
    llm_confidence_primary: float      # Confidence score
    codebook_labels_ensemble: List[str]  # Applied codebook codes
    
    # Interrater reliability fields
    rater_votes: List[Dict]
    agreement_level: str               # unanimous/majority/split/none
    needs_review: bool
    
    # Final training label
    final_label: int | None
    label_confidence_tier: str         # high/medium/low/unclassified
```

### PipelineConfig

Serializable configuration object:

```python
@dataclass
class PipelineConfig:
    transcript_dir: str
    output_dir: str
    run_mode: str                      # auto/interactive/review
    
    segmentation: SegmentationConfig   # Semantic segmentation params
    speaker_filter: SpeakerFilterConfig  # Speaker filtering rules
    theme_classification: ThemeClassificationConfig
    codebook_embedding: EmbeddingClassifierConfig
    codebook_llm: LLMCodebookConfig
    codebook_ensemble: EnsembleConfig
    validation: ValidationConfig
    confidence_tiers: ConfidenceTierConfig
```

## Module Reference

### Classification Tools (`classification_tools/`)

| File | Description |
|------|-------------|
| `llm_client.py` | Unified LLM API client (OpenRouter, Replicate, Ollama, LM Studio, HuggingFace) |
| `data_structures.py` | Segment dataclass and core data structures |
| `llm_classifier.py` | Theme and codebook classification logic |
| `majority_vote.py` | Interrater reliability voting aggregation |
| `response_parser.py` | Parse LLM outputs into structured format |
| `validation.py` | Create evaluation sets, consistency checking |

### Constructs (`constructs/`)

| File | Description |
|------|-------------|
| `theme_schema.py` | ThemeFramework and ThemeDefinition dataclasses |
| `vamr.py` | VA-MR framework definitions (default) |
| `config.py` | Theme classification configuration |

### Codebook (`codebook/`)

| File | Description |
|------|-------------|
| `codebook_schema.py` | Codebook and CodeAssignment dataclasses |
| `phenomenology_codebook.py` | Default phenomenology codebook |
| `embedding_classifier.py` | Sentence-transformer based classification |
| `ensemble.py` | Combine embedding + LLM results |

### Process (`process/`)

| File | Description |
|------|-------------|
| `orchestrator.py` | Main pipeline orchestration (7 stages) |
| `config.py` | PipelineConfig dataclass and serialization |
| `setup_wizard.py` | Interactive configuration wizard |
| `transcript_ingestion.py` | Load VTT/JSON, speaker normalization |
| `llm_segmentation.py` | LLM-assisted segmentation refinement |
| `dataset_assembly.py` | Export master dataset and reports |

### Analysis (`analysis/`)

| File | Description |
|------|-------------|
| `runner.py` | Post-hoc analysis orchestrator |
| `participant.py` | Per-participant report generation |
| `session.py` | Per-session analysis |
| `construct.py` | Per-construct (stage + code) analyses |
| `graphing.py` | Export graph-ready CSVs |
| `figures.py` | Generate visualization figures |
| `longitudinal.py` | Longitudinal summary generation |

## API Integration

### LLM Backend Configuration

#### LM Studio (Local GUI)
1. Download and start LM Studio
2. Load a model server
3. Start the local server on port 1234
4. Run: `python qra.py run --backend lmstudio --model <model-name>`

#### OpenRouter
```bash
export OPENROUTER_API_KEY=sk-or-v1-...
python qra.py run --backend openrouter --model openai/gpt-4o
```

#### Replicate
```bash
export REPLICATE_API_TOKEN=r8_...
python qra.py run --backend replicate --model meta/llama-2-70b-chat
```

#### Ollama (Local)
```bash
# Pull a model first
ollama pull llama3

# Run with local backend
python qra.py run --backend ollama --model llama3
```

#### HuggingFace (Local GPU)
```bash
# For models that fit in your GPU VRAM
export CUDA_VISIBLE_DEVICES=0
python qra.py run --backend huggingface --model meta-llama/Llama-3.2-3B-Instruct
```

## Confidence Tiers

Segments are assigned to confidence tiers based on LLM agreement:

| Tier | Consistency | Confidence | Description |
|------|-------------|------------|-------------|
| **High** | 3/3 | >0.8 | Unanimous with high confidence |
| **Medium** | 2+/3 | >0.6 | Majority or good confidence |
| **Low** | <2 | - | Split votes or low confidence |
| **Unclassified** | 0 | - | No consensus |

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
- Powered by LLMs from OpenRouter, Replicate, HuggingFace
- Embeddings powered by sentence-transformers and Qwen models
