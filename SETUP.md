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
| CPU-only inference | 8 GB RAM | 16+ GB RAM |
| GPU acceleration (HuggingFace) | RTX 3060 (12GB VRAM) | RTX 4090 (24GB VRAM+) |
| LM Studio/Ollama local models | 16 GB RAM | 32+ GB RAM |

### Software Requirements

- **Python**: 3.9, 3.10, or 3.11 (Python 3.12 may have compatibility issues with some dependencies)
- **Operating System**: Linux (Ubuntu 20.04+, Debian 11+), macOS (12+), Windows 11

### Required Packages

QRA depends on the following core packages:

| Package | Purpose |
|---------|---------|
| `torch` | PyTorch for deep learning models |
| `sentence-transformers` | Embedding-based codebook classification |
| `huggingface-hub` | Download and cache transformer models |
| `pandas` | Data manipulation and analysis |
| `scikit-learn` | Similarity computations, evaluation metrics |

---

## Installation

### Step 1: Create a Virtual Environment (Recommended)

```bash
# Using venv
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or
.venv\Scripts\activate     # Windows PowerShell

# Verify Python version
python --version  # Should be 3.9-3.11
```

### Step 2: Install Dependencies

```bash
# Clone the repository (if not already done)
git clone <repository-url>
cd Qualitative_Research_Algorithm

# Install from requirements if available
pip install -r requirements.txt

# Or install manually
pip install torch sentence-transformers huggingface-hub pandas scikit-learn numpy
```

### Step 3: Verify Installation

```bash
python -c "import torch; print(f'PyTorch {torch.__version__}')"
python -c "from sentence_transformers import SentenceTransformer; print('sentence-transformers OK')"
```

---

## LLM Backend Configuration

QRA supports multiple LLM backends. Choose one that best fits your needs.

### Option A: OpenRouter (Cloud API)

**Best for**: Quick start, access to top models like GPT-4, Claude

```bash
# Get an API key from https://openrouter.ai/
export OPENROUTER_API_KEY=sk-or-v1-your-key-here

# Test the connection
python -c "
import requests
r = requests.get(
    'https://openrouter.ai/api/v1/models',
    headers={'Authorization': f'Bearer {\"your_key\"}'}
)
print(r.status_code, 'models available')
"
```

### Option B: Replicate (Cloud API)

**Best for**: Access to open-weight models like Llama

```bash
# Get an API key from https://replicate.com/
export REPLICATE_API_TOKEN=r8_your-token-here
```

### Option C: HuggingFace (Local GPU)

**Best for**: Full control, privacy, no API costs after model download

#### 1. Install PyTorch with CUDA (if using NVIDIA GPU)

```bash
# Check if you have an NVIDIA GPU
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

#### 2. Verify GPU Access

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Option D: Ollama (Local)

**Best for**: Easy local LLM running, no GPU required

#### 1. Install Ollama

```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows (PowerShell)
iwr -useb https://ollama.com/install.ps1 | iex
```

#### 2. Pull a Model

```bash
# Small model for testing
ollama pull llama3

# Or larger models for better quality
ollama pull qwen2.5:7b
ollama pull mistral:7b
```

### Option E: LM Studio (Local GUI)

**Best for**: User-friendly local LLM interface

#### 1. Install LM Studio

Download from [lmstudio.ai](https://lmstudio.ai/) and install.

#### 2. Load a Model

1. Open LM Studio
2. Click "Models" tab
3. Search for a model (e.g., `nvidia/nemotron-3-super`)
4. Click "Load" to start the local server
5. Note the server URL (default: http://127.0.0.1:1234/v1)

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

#### 2. VTT Format (WebVTT)

```vtt
WEBVTT

00:00:00.500 --> 00:00:03.200
Therapist: How are you feeling today?

00:00:03.500 --> 00:00:06.800
Participant_1: I'm doing pretty well overall.
```

### Input Directory Structure

Place your transcript files in the input directory:

```
data/input/
├── session_c1s1.json
├── session_c1s2.vtt
├── session_c1s3.json
└── speaker_anonymization_key.json  # Optional (for participant ID consistency)
```

### Speaker Anonymization Key (Optional)

To maintain consistent participant IDs across runs, create a `speaker_anonymization_key.json`:

```json
{
  "John Smith": {"role": "participant", "anonymized_id": "participant_1"},
  "Jane Doe": {"role": "participant", "anonymized_id": "participant_2"},
  "Dr. Therapist": {"role": "therapist", "anonymized_id": "therapist_1"}
}
```

---

## Running the Setup Wizard

The setup wizard guides you through all configuration options:

```bash
python qra.py setup
```

### Wizard Steps Explained

#### Step 1: Input/Output Paths

- **Transcript directory**: Where your input JSON/VTT files are located
- **Output directory**: Where results will be saved (auto-created)
- **Trial ID**: Identifier for this analysis run (e.g., "baseline_study")

#### Step 1b: Speaker Anonymization Key

Optionally import a speaker mapping file to preserve participant IDs across runs.

#### Step 2: Speaker Role Identification

Identify which speakers are therapists vs participants:
- The wizard scans transcripts and suggests default therapist speakers
- Select the correct speakers for each role
- Choose whether to exclude therapist utterances from classification

#### Step 3: Segmentation Parameters

Configure how transcripts are split into segments:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Max gap (seconds) | Time gap before starting new segment | 15.0 |
| Min words/sentence | Shorter sentences merged with neighbors | 20 |
| Max segment duration | Longest single segment in seconds | 60.0 |
| Min/max words/segment | Segment length constraints | 60-400 |

#### Step 4: Backend & Model

Select your LLM backend and model:

| Backend | Example Models |
|---------|----------------|
| openrouter | `openai/gpt-4o`, `anthropic/claude-3.5-sonnet` |
| replicate | `meta/llama-2-70b-chat`, `mistralai/mistral-7b-instruct:7b` |
| huggingface | `meta-llama/Llama-3.2-3B-Instruct`, `Qwen/Qwen3-Next-80B` |
| ollama | `llama3`, `qwen2.5:7b` |
| lmstudio | Your loaded local model |

#### Step 5: Theme Framework

Choose your classification framework:

- **vamr**: Default VA-MR framework (Vigilance, Avoidance, Metacognition, Reappraisal)
- **custom**: Load a custom JSON framework file

#### Step 6: Exemplar Utterances (Optional)

Customize the examples used for each theme if you have domain-specific guidance.

#### Step 7: Codebook Classification (Optional)

Enable multi-label codebook classification:

| Parameter | Description |
|-----------|-------------|
| Enable? | Turn on/off codebook classification |
| Codebook preset | "phenomenology" or custom JSON |
| Embedding model | `Qwen/Qwen3-Embedding-8B` (best) or `all-MiniLM-L6-v2` (lightweight) |

#### Step 8: Classification Parameters

Configure the LLM classification:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Number of runs | Classification passes per segment for IRR | 3 |
| Temperature | Higher = more creative, lower = more consistent | 0.1 |

For multi-model interrater reliability, specify additional checker models.

#### Step 9: Confidence Thresholds

Set thresholds for confidence tier assignment:

| Tier | Min Consistency | Min Confidence |
|------|-----------------|----------------|
| High | 3/3 (unanimous) | >0.8 |
| Medium | 2+/3 (majority) | >0.6 |

#### Step 10: Post-Pipeline Analysis

Choose whether to automatically run analysis after classification.

#### Step 11: Run Mode

Select how the pipeline handles uncertain results:

- **auto**: Fully automated, no human intervention
- **interactive**: Prompt for validation during pipeline execution
- **review**: Batch validation at end of pipeline

#### Step 12: Save Configuration

Review your settings and save to `meta/qra_config.json`.

---

## Configuration Reference

### Full Config File Structure

```json
{
  "pipeline": {
    "transcript_dir": "./data/input/",
    "output_dir": "./data/output/",
    "trial_id": "standard",
    "run_mode": "auto",
    "run_theme_labeler": true,
    "run_codebook_classifier": false,
    "speaker_anonymization_key_path": null,
    "auto_analyze": true
  },
  "segmentation": {
    "use_conversational_segmenter": true,
    "max_gap_seconds": 15.0,
    "min_words_per_sentence": 20,
    "max_segment_duration_seconds": 60.0,
    "min_segment_words_conversational": 60,
    "max_segment_words_conversational": 400,
    "use_adaptive_threshold": true,
    "min_prominence": 0.05,
    "broad_window_size": 7,
    "use_topic_clustering": false,
    "use_llm_refinement": false,
    "llm_refinement_mode": "boundary_review",
    "llm_ambiguity_threshold": 0.15,
    "llm_batch_size": 5
  },
  "speaker_filter": {
    "mode": "exclude",
    "speakers": ["Therapist", "Instructor"]
  },
  "theme_classification": {
    "backend": "openrouter",
    "model": "openai/gpt-4o",
    "models": [],
    "per_run_models": [],
    "n_runs": 3,
    "temperature": 0.1,
    "api_key": "",
    "replicate_api_token": "",
    "lmstudio_base_url": "http://127.0.0.1:1234/"
  },
  "codebook_embedding": {
    "two_pass": true,
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "exemplar_import_path": null
  },
  "validation": {
    "n_per_class": 50,
    "min_kappa": 0.70,
    "min_agreement": 0.75
  },
  "confidence_tiers": {
    "high_confidence": 0.8,
    "medium_min_confidence": 0.6
  }
}
```

### Programmatic Configuration

You can also modify configuration programmatically:

```python
from process.config import PipelineConfig, SegmentationConfig

config = PipelineConfig(
    transcript_dir='./data/input/',
    output_dir='./data/output/',
    run_mode='interactive',
    segmentation=SegmentationConfig(
        max_gap_seconds=20.0,
        min_words_per_sentence=15,
    ),
)
```

---

## Troubleshooting

### Common Issues

#### 1. OOM (Out of Memory) Errors

**Symptoms**: Process crashes with CUDA out of memory or memory allocation errors.

**Solutions**:
- Reduce batch size: `embedding_batch_size: 4`
- Use smaller model: `all-MiniLM-L6-v2` instead of `Qwen/Qwen3-Embedding-8B`
- Enable CPU offloading for HuggingFace models
- Close other GPU-intensive applications

#### 2. LM Studio Connection Failed

**Symptoms**: Error "Connection refused" or "Failed to connect to LM Studio"

**Solutions**:
- Verify LM Studio server is running: http://127.0.0.1:1234/v1/models should return JSON
- Check model is loaded in LM Studio
- Update `lmstudio_base_url` to match your server port

#### 3. HuggingFace Model Download Fails

**Symptoms**: `HFCacheDownloadError` or connection timeout during model download.

**Solutions**:
```bash
# Pre-download the model manually
python -c "from sentence_transformers import SentenceTransformer; \
SentenceTransformer('Qwen/Qwen3-Embedding-8B', trust_remote_code=True)"
```

#### 4. LLM Returns Empty/Invalid Responses

**Symptoms**: Classification results show `null` for all segments.

**Solutions**:
- Check API key is set correctly
- Verify model name is correct and accessible
- Increase temperature if using deterministic models
- Check context window isn't exceeded (shorten segments)

#### 5. Speaker Names Not Normalizing

**Symptoms**: Participant IDs keep changing between runs.

**Solution**: Create a `speaker_anonymization_key.json` file in your input directory with consistent mappings.

### Debug Mode

Enable verbose logging:

```bash
python qra.py run --config ./data/output/meta/qra_config.json --verbose-segmentation
```

This creates `meta/process_log.txt` with detailed processing information.

### Getting Help

1. Check the logs: `output_dir/meta/process_log.txt`
2. Review validation output: `output_dir/reports/validation/flagged_for_review.txt`
3. Verify input data format matches expected schema

---

## Next Steps

After setup is complete:

1. **Run a small test**: Use 2-3 short transcripts first
2. **Review initial results**: Check confidence tiers and flagged segments
3. **Adjust parameters**: Fine-tune thresholds and segmentation based on results
4. **Scale up**: Run full dataset after parameter tuning

---

## Example: Complete Setup from Scratch

```bash
# 1. Create environment
python -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install torch sentence-transformers huggingface-hub pandas scikit-learn replicate

# 3. Set up API key
export OPENROUTER_API_KEY=sk-or-v1-your-key

# 4. Prepare input data (place JSON/VTT files in data/input/)

# 5. Run setup wizard
python qra.py setup

# 6. Review saved config
cat ./data/output/meta/qra_config.json

# 7. Run pipeline
python qra.py run --config ./data/output/meta/qra_config.json

# 8. Analyze results
python qra.py analyze --output-dir ./data/output/
```
