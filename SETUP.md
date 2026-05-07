# QRA Setup Guide

This guide walks you through setting up and running the QRA (Qualitative Research Algorithm) pipeline step by step.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Installation](#installation)
3. [LLM Backend Configuration](#llm-backend-configuration)
4. [Input Data Preparation](#input-data-preparation)
5. [Running the Setup Wizard](#running-the-setup-wizard)
6. [Configuration Reference](#configuration-reference)
7. [Troubleshooting](#troubleshooting)

---

## System Requirements

### Hardware Recommendations

| Task | Minimum | Recommended |
|------|---------|-------------|
| Cloud API backends (OpenRouter, Replicate) | 8 GB RAM | 16+ GB RAM |
| GPU acceleration (HuggingFace models) | RTX 3060 (12 GB VRAM) | RTX 4090 (24 GB+ VRAM) |
| Embedding model (Qwen3-Embedding-8B) | 16 GB RAM | 32+ GB RAM or GPU |
| LM Studio / Ollama local models | 16 GB RAM | 32+ GB RAM |

### Software Requirements

- **Python**: 3.9, 3.10, or 3.11 (Python 3.12 may have compatibility issues with some dependencies)
- **Operating System**: Linux (Ubuntu 20.04+), macOS 12+, Windows 11

### Required Packages

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch for deep learning models |
| `sentence-transformers` | Embedding-based segmentation and codebook classification |
| `huggingface-hub` | Download and cache transformer models |
| `pandas` | Data manipulation and analysis |
| `scikit-learn` | Similarity computations, evaluation metrics |
| `matplotlib` | Analysis figures |
| `krippendorff` | Inter-rater reliability metrics |

---

## Installation

### Step 1: Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# or
.venv\Scripts\activate      # Windows PowerShell

# Verify Python version
python --version  # Should be 3.9–3.11
```

### Step 2: Install Dependencies

```bash
# Clone the repository
git clone <repository-url>
cd Qualitative_Research_Algorithm

# Install from requirements.txt
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
```

---

## LLM Backend Configuration

QRA supports multiple LLM backends. Choose one that best fits your needs.

### Option A: LM Studio (Local GUI)

**Best for**: User-friendly local LLM interface, no API costs

1. Download from lmstudio.ai and install.
2. Open LM Studio, go to the **Models** tab, and download a model (e.g., `nvidia/nemotron-3-super`, `qwen/qwen3-30b-a3b`)
3. Start the local server (Developer tab → Start Server)
4. Note the server URL (default: `http://127.0.0.1:1234/v1`)

### Option B: OpenRouter (Cloud API)

**Best for**: Quick start, access to top models like GPT-4o, Claude

```bash
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

The key is resolved automatically from the environment at runtime; it is never saved to the config JSON.

### Option C: Replicate (Cloud API)

```bash
export REPLICATE_API_TOKEN=r8_your-token-here
```

### Option D: HuggingFace (Local GPU)

```bash
# Check your GPU
nvidia-smi

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# Verify GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Option E: Ollama (Local)

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull llama3
# or larger models
ollama pull qwen2.5:7b
```

---

## Input Data Preparation

### Supported Formats

QRA accepts diarized transcripts in two formats:

#### 1. JSON Format (Whisper Diarization Output)

```json
{
  "metadata": {
    "trial_id": "T001",
    "participant_id": "P001",
    "session_id": "c1s1",
    "session_number": 1,
    "cohort_id": 1,
    "source_file": "path/to/audio.wav"
  },
  "sentences": [
    {
      "text": "How are you feeling today?",
      "speaker": "Therapist",
      "start": 0.5,
      "end": 3.2
    },
    {
      "text": "I'm doing pretty well overall.",
      "speaker": "Participant_1",
      "start": 3.5,
      "end": 6.8
    }
  ]
}
```

**Session ID conventions**: The `session_id` field (e.g., `c1s1`, `c2s4a`) is parsed to extract `cohort_id`, `session_number`, and `session_variant` automatically.

#### 2. VTT Format (WebVTT)

```vtt
WEBVTT

00:00:00.500 --> 00:00:03.200
Therapist: How are you feeling today?

00:00:03.500 --> 00:00:06.800
Participant_1: I'm doing pretty well overall.
```

For VTT files, the filename stem (e.g., `c1s1.vtt`) is used as the session ID and parsed for cohort/session metadata.

### Input Directory Structure

```
data/input/
├── c1s1.json          # or .vtt
├── c1s2.json
├── c2s1.json
└── speaker_anonymization_key.json   # Optional — preserves participant IDs across runs
```

### Speaker Anonymization Key (Optional)

To maintain consistent participant IDs across pipeline runs, create a `speaker_anonymization_key.json`:

```json
{
  "John Smith": {"role": "participant", "anonymized_id": "participant_MM001"},
  "Jane Doe":   {"role": "participant", "anonymized_id": "participant_MM002"},
  "Dr. Therapist": {"role": "therapist", "anonymized_id": "therapist_1"}
}
```

Each entry must have exactly `role` (`"participant"` or `"therapist"`) and `anonymized_id`. New speakers not in the key are assigned `unknownparticipant_N` IDs.

---

## Running the Setup Wizard

```bash
python qra.py setup
```

The wizard walks through 14 steps and saves a config JSON to `02_meta/qra_config.json`.

---

### Setup Steps Overview

#### Step 1: Input/Output Paths

- **Transcript directory**: Where your JSON/VTT files are located (default: `./data/input/`)
- **Output directory**: Where results will be saved (default: `./data/output/`)
- **Trial ID**: Identifier for this analysis run (e.g., `baseline_study`)

#### Step 1b: Speaker Anonymization Key

Optionally import a pre-existing speaker ID mapping. The wizard looks for `speaker_anonymization_key.json` in the transcript directory automatically.

#### Step 2: Speaker Role Identification

The wizard scans transcripts and discovers speakers, then lets you designate which are therapists/facilitators vs participants.

- Therapist dialogue is **excluded from VAAMR theme classification** but preserved as conversational context
- Therapist dialogue is classified separately via **PURER** (see Step 4)

#### Step 3: Segmentation Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Max gap (seconds) | Time gap before starting new segment | 15.0 |
| Min words/sentence | Shorter sentences merged with neighbors | 20 |
| Max segment duration (seconds) | Longest single segment | 60.0 |
| Min words/segment | Minimum segment length | 60 |
| Max words/segment | Maximum segment length | 500 |
| Adaptive threshold | Local-minima boundary detection | enabled |
| Topic clustering | AgglomerativeClustering for boundary confidence | enabled |
| LLM refinement | LLM-assisted boundary review and context expansion | enabled |

**LLM refinement modes** (when enabled):

| Mode | Description |
|------|-------------|
| `boundary_review` | Re-evaluate ambiguous boundaries only |
| `context_expansion` | Expand segments with surrounding dialogue |
| `coherence_check` | Split oversized segments at natural breaks |
| `full` | All three passes (default) |

#### Step 4: Backend & Model (VAAMR)

Select your LLM backend and primary model for VAAMR classification. PURER classification uses a separate model (default: `nvidia/nemotron-3-nano-4b` for light preset, `nvidia/nemotron-3-super` for heavy).

| Backend | Example Models |
|---------|----------------|
| `lmstudio` | `nvidia/nemotron-3-super`, `qwen/qwen3-30b-a3b` |
| `openrouter` | `openai/gpt-4o`, `anthropic/claude-3.5-sonnet` |
| `replicate` | `meta/llama-2-70b-chat` |
| `huggingface` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |
| `ollama` | `llama3`, `qwen2.5:7b` |

#### Step 5: Theme Framework

Choose your classification framework:

- **`vammr`**: Default VAAMR framework (Vigilance, Avoidance, Attention Regulation, Metacognition, Reappraisal)
- **`custom`**: Load a custom JSON framework file

#### Step 6: Exemplar Utterances

Optionally customize exemplar utterances for each theme. Skip to use the framework defaults.

#### Step 7: Codebook Classification

Optionally enable multi-label codebook classification:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Enable? | Turn on/off codebook classification | No |
| Codebook preset | `phenomenology` or custom JSON path | `phenomenology` |
| Embedding model | `Qwen/Qwen3-Embedding-8B` or `all-MiniLM-L6-v2` | `Qwen/Qwen3-Embedding-8B` |
| Two-pass | Run embedding classification in two passes | Yes |

**Note**: The embedding model is also used for semantic segmentation.

#### Step 8: Classification Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Number of runs | Classification passes per segment | 3 |
| Temperature | Higher = more varied responses | 0.1 |

When `n_runs >= 2`, the wizard prompts for **checker models** for multi-model interrater reliability.

#### Step 9: Confidence Thresholds

| Tier | Threshold | Description |
|------|-----------|-------------|
| High | `high_confidence` (default 0.8) | Unanimous agreement + high confidence score |
| Medium | `medium_min_confidence` (default 0.6) | Majority agreement or moderate confidence |
| Low | below medium | Split votes or low confidence |

#### Step 10: Validation Test Sets

Cross-session test sets draw a stratified random sample of participant segments for human blind-coding. The wizard generates VAAMR participant testsets by default. PURER and codebook testsets are also supported but require the respective classifiers to be enabled.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Enable? | Generate validation test sets | Yes |
| Number of sets | How many independent test sets | 2 |
| Fraction per set | Proportion of segments per set | 0.10 (10%) |

Test sets can also be managed post-hoc:
```bash
# Create a PURER testset
python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1

# Refresh existing AI answer keys
python qra.py testset refresh -o ./data/output/ --all

# List existing testsets
python qra.py testset list -o ./data/output/
```

#### Step 11: Post-Pipeline Analysis

Choose whether to automatically run the analysis module after the pipeline completes. Analysis can also be run manually:
```bash
python qra.py analyze --output-dir ./data/output/
```

#### Step 11b: Therapist Cue Summarization

When enabled, therapist dialogue between two participant segments is surfaced as a **CUE** in transition analysis reports.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Enable? | Surface therapist cues in transition analysis | Yes |
| Max words per inline cue | Longer cues are LLM-summarized | 250 |
| Max words per averaged block | Cap per block in `cue_response.txt` | 500 |

#### Step 11c: Session & Participant Summaries

LLM-generated summaries of therapist language and participant language per session, shown in participant reports.

#### Step 12: Save Configuration

Saves the complete configuration to `02_meta/qra_config.json` in the output directory.

**Note**: API keys are never written to the config file. They are resolved from environment variables at runtime.

---

## Configuration Reference

### Key Config Sections

```json
{
  "pipeline": {
    "transcript_dir": "./data/input/",
    "output_dir": "./data/output/",
    "trial_id": "standard",
    "run_theme_labeler": true,
    "run_purer_labeler": true,
    "run_codebook_classifier": false,
    "auto_analyze": true
  },
  "segmentation": { ... },
  "speaker_filter": { ... },
  "theme_classification": {
    "backend": "lmstudio",
    "model": "nvidia/nemotron-3-super",
    "per_run_models": ["nvidia/nemotron-3-super", "google/gemma-4-31b", "qwen/qwen3-next-80b"],
    "n_runs": 3,
    "temperature": 0.1
  },
  "purer_classification": {
    "backend": "lmstudio",
    "model": "nvidia/nemotron-3-nano-4b",
    "n_runs": 1,
    "context_window_segments": 6
  },
  "purer_cue": {
    "skip_lesson_content": true,
    "max_lesson_words": 400,
    "therapist_max_gap_seconds": 120.0,
    "max_context_words": 1000
  },
  "test_sets": {
    "vaamr": {"enabled": true, "name": "vaamr_testset", "n_sets": 2, "fraction_per_set": 0.10},
    "purer": {"enabled": false, "name": "purer_testset"},
    "codebook": {"enabled": false, "name": "codebook_testset"}
  },
  "content_validity": {
    "vaamr": {"enabled": true, "name": "cv_vaamr_v1"},
    "purer": {"enabled": false, "name": "cv_purer_v1"}
  },
  "framework": {
    "preset": "vammr"
  }
}
```

### Programmatic Configuration

```python
from process.config import PipelineConfig, SegmentationConfig, SpeakerFilterConfig

config = PipelineConfig(
    transcript_dir='./data/input/',
    output_dir='./data/output/',
    auto_analyze=True,
    segmentation=SegmentationConfig(
        max_gap_seconds=15.0,
        min_words_per_sentence=20,
        use_llm_refinement=True,
    ),
    speaker_filter=SpeakerFilterConfig(
        mode='exclude',
        speakers=['Therapist'],
    ),
)
```

---

## Troubleshooting

### Common Issues

#### 1. OOM (Out of Memory) Errors

**Solutions**:
- Use the lightweight embedding model: set `embedding_model` to `all-MiniLM-L6-v2`
- Reduce `max_segment_words_conversational` to limit segment size
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512`

#### 2. LM Studio Connection Failed

**Solutions**:
- Verify LM Studio server is running: `http://127.0.0.1:1234/v1/models` should return JSON
- Ensure a model is loaded in LM Studio (not just downloaded)
- Check `lmstudio_base_url` in your config matches the actual server port

#### 3. HuggingFace Model Download Fails

```bash
# Pre-download the embedding model manually
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Qwen/Qwen3-Embedding-8B', trust_remote_code=True)"
```

#### 4. LLM Returns Empty/Invalid Responses

**Solutions**:
- Check API key environment variables are set
- Verify model name is correct for the chosen backend
- Try increasing temperature slightly (e.g., 0.1 → 0.3)
- Reduce `max_segment_words_conversational` if segments exceed the model's context window

#### 5. Speaker Names Not Normalizing Consistently

**Solution**: Create a `speaker_anonymization_key.json` file in your input directory with explicit mappings. The key in `02_meta/` always takes precedence on re-runs.

#### 6. PURER Classification Errors

If PURER classification fails during the pipeline, a `purer_classification_error.txt` is written to the output directory. The pipeline continues with VAAMR results only.

### Getting Help

1. Check the output index: `output_dir/00_index.txt`
2. Review the process log: `output_dir/02_meta/auditable_logs/segmentation_process_log.txt`
3. Check flagged segments: `output_dir/04_validation/flagged_for_review.txt`
4. Check PURER failures: `output_dir/purer_classification_error.txt`

---

## Example: Complete Setup from Scratch

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up API key (if using OpenRouter)
export OPENROUTER_API_KEY=sk-or-v1-your-key

# 4. Prepare input data
mkdir -p data/input
# Copy JSON/VTT files to data/input/

# 5. Run setup wizard
python qra.py setup

# 6. Review saved config
cat ./data/output/02_meta/qra_config.json

# 7. Run pipeline
python qra.py run --config ./data/output/02_meta/qra_config.json

# 8. Analyze results (or use --auto-analyze in step 7)
python qra.py analyze --output-dir ./data/output/

# 9. Manage test sets
python qra.py testset list -o ./data/output/
python qra.py cv list -o ./data/output/
```
