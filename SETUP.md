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

#### 1. Install LM Studio

Download from lmstudio.ai and install.

#### 2. Load a Model and Start Server

1. Open LM Studio
2. Go to the **Models** tab and download a model (e.g., `nvidia/nemotron-3-super`, `qwen/qwen3-30b-a3b`)
3. Start the local server (Developer tab → Start Server)
4. Note the server URL (default: `http://127.0.0.1:1234/v1`)

The setup wizard defaults to LM Studio backend with URL `http://10.0.0.58:1234/v1` — change this to match your server.

### Option B: OpenRouter (Cloud API)

**Best for**: Quick start, access to top models like GPT-4o, Claude

```bash
# Get an API key from openrouter.ai
export OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

The key is resolved automatically from the environment at runtime; it is never saved to the config JSON.

### Option C: Replicate (Cloud API)

**Best for**: Access to open-weight models

```bash
export REPLICATE_API_TOKEN=r8_your-token-here
```

### Option D: HuggingFace (Local GPU)

**Best for**: Full control, privacy, no API costs after model download

#### 1. Install PyTorch with CUDA

```bash
# Check your GPU
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

#### 2. Verify GPU

```bash
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"
```

### Option E: Ollama (Local)

**Best for**: Easy local LLM with CPU support

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

The wizard walks through 12 steps and saves a config JSON to `07_meta/qra_config.json`.

---

### Step 1/12: Input/Output Paths

- **Transcript directory**: Where your JSON/VTT files are located (default: `./data/input/`)
- **Output directory**: Where results will be saved — organized into numbered subdirectories (default: `./data/output/`)
- **Trial ID**: Identifier for this analysis run (e.g., `baseline_study`)

---

### Step 1b/12: Speaker Anonymization Key

Optionally import a pre-existing speaker ID mapping to keep participant IDs consistent across pipeline runs.

- The wizard looks for `speaker_anonymization_key.json` in the transcript directory automatically
- If found, prompts to use it; otherwise offers a custom path
- The key is validated for correct format before use
- New speakers not in the key are assigned `unknownparticipant_N`

---

### Step 2/12: Speaker Role Identification

The wizard scans transcripts and discovers speakers, then lets you designate which are therapists/facilitators vs participants.

- Therapist dialogue is **excluded from theme classification** (to focus on participant-expressed content) but is preserved as read-only conversational context for adjacent participant segments
- You can confirm the default therapist list or manually select from discovered speakers

---

### Step 3/12: Segmentation Parameters

Configure how transcripts are split into segments. Accept defaults for most studies.

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

---

### Step 4/12: Backend & Model

Select your LLM backend and primary model. This model is used for all LLM calls: segmentation refinement, theme classification, codebook LLM classification, and rationale summarization.

| Backend | Example Models |
|---------|----------------|
| `lmstudio` | `nvidia/nemotron-3-super`, `qwen/qwen3-30b-a3b` |
| `openrouter` | `openai/gpt-4o`, `anthropic/claude-3.5-sonnet` |
| `replicate` | `meta/llama-2-70b-chat` |
| `huggingface` | `meta-llama/Llama-4-Maverick-17B-128E-Instruct` |
| `ollama` | `llama3`, `qwen2.5:7b` |

For multi-model interrater reliability, you can optionally specify checker models here (continued in Step 8).

---

### Step 5/12: Theme Framework

Choose your classification framework:

- **`vammr`**: Default VAMMR framework (Vigilance, Avoidance, Mindfulness, Metacognition, Reappraisal)
- **`custom`**: Load a custom JSON framework file

Custom framework JSON schema:

```json
{
  "framework": "My Framework",
  "version": "1.0",
  "description": "...",
  "themes": [
    {
      "theme_id": 0,
      "key": "stage_a",
      "name": "Stage A",
      "short_name": "A",
      "prompt_name": "stage_a",
      "definition": "...",
      "prototypical_features": ["feature 1", "feature 2"],
      "distinguishing_criteria": "...",
      "exemplar_utterances": ["Example 1", "Example 2"]
    }
  ]
}
```

---

### Step 6/12: Exemplar Utterances

Optionally customize exemplar utterances for each theme. These are included in the classification prompt to guide the LLM. Skip to use the framework defaults.

---

### Step 7/12: Codebook Classification

Optionally enable multi-label codebook classification:

| Parameter | Description | Default |
|-----------|-------------|---------|
| Enable? | Turn on/off codebook classification | No |
| Codebook preset | `phenomenology` or custom JSON path | `phenomenology` |
| Embedding model | `Qwen/Qwen3-Embedding-8B` (best) or `all-MiniLM-L6-v2` (lightweight) | `Qwen/Qwen3-Embedding-8B` |
| Two-pass | Run embedding classification in two passes | Yes |

**Note**: The embedding model is also used for semantic segmentation. `Qwen/Qwen3-Embedding-8B` is ~16 GB; `all-MiniLM-L6-v2` is 90 MB and needs no GPU.

---

### Step 8/12: Classification Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| Number of runs | Classification passes per segment | 3 |
| Temperature | Higher = more varied responses | 0.1 |

When `n_runs >= 2`, the wizard prompts for **checker models** for multi-model interrater reliability:

- Run 1 always uses the primary model
- Runs 2+ use independently specified checker models
- Checker models act as independent raters; majority-vote consistency reflects genuine cross-model agreement
- Default checker models for LM Studio: `google/gemma-4-31b`, `qwen/qwen3-next-80b`

---

### Step 9/12: Confidence Thresholds

| Tier | Threshold | Description |
|------|-----------|-------------|
| High | `high_confidence` (default 0.8) | Unanimous agreement + high confidence score |
| Medium | `medium_min_confidence` (default 0.6) | Majority agreement or moderate confidence |
| Low | below medium | Split votes or low confidence |
| Unclassified | — | No consensus reached |

---

### Step 10/12: Validation Test Sets

Cross-session test sets draw a stratified random sample of participant segments for human blind-coding, enabling inter-rater reliability comparison with AI classifications.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Enable? | Generate validation test sets | Yes |
| Number of sets | How many independent test sets | 2 |
| Fraction per set | Proportion of participant segments per set | 0.10 (10%) |

Two output worksheets are produced per set: one for human coders (no AI labels) and one for AI classification comparison.

You can also generate test sets from an existing run at any time:
```bash
python qra.py testsets --output-dir ./data/output/ --n-sets 3 --fraction 0.15
```

---

### Step 11/12: Post-Pipeline Analysis

Choose whether to automatically run the analysis module after the pipeline completes. When enabled, generates:

- Per-participant longitudinal reports
- Per-session summaries with prototypical exemplars
- Per-theme (stage + codebook) analyses
- Graph-ready CSVs for R/Python visualization
- Longitudinal summary and transition explanation

Analysis can also be run manually at any time:
```bash
python qra.py analyze --output-dir ./data/output/
```

---

### Step 11b/12: Therapist Cue Summarization

When enabled, therapist dialogue between two participant segments is surfaced as a **CUE** in `state_transition_explanation.txt`, and `cue_response.txt` is generated with averaged cues grouped by transition type.

| Parameter | Description | Default |
|-----------|-------------|---------|
| Enable? | Surface therapist cues in transition analysis | Yes |
| Max words per inline cue | Longer cues are LLM-summarized | 250 |
| Max words per averaged block | Cap per block in `cue_response.txt` | 1000 |

---

### Step 12/12: Save Configuration

Saves the complete configuration to `07_meta/qra_config.json` in the output directory. This file can be reloaded with `python qra.py run --config <path>` for fully reproducible runs.

**Note**: API keys are never written to the config file. They are resolved from environment variables at runtime.

---

## Configuration Reference

### Full Config File Structure

```json
{
  "pipeline": {
    "transcript_dir": "./data/input/",
    "output_dir": "./data/output/",
    "trial_id": "standard",
    "run_theme_labeler": true,
    "run_codebook_classifier": false,
    "speaker_anonymization_key_path": null,
    "auto_analyze": true
  },
  "segmentation": {
    "max_gap_seconds": 15.0,
    "min_words_per_sentence": 20,
    "max_segment_duration_seconds": 60.0,
    "min_segment_words_conversational": 60,
    "max_segment_words_conversational": 500,
    "use_adaptive_threshold": true,
    "min_prominence": 0.05,
    "broad_window_size": 7,
    "use_topic_clustering": true,
    "use_llm_refinement": true,
    "llm_refinement_mode": "full",
    "llm_ambiguity_threshold": 0.15,
    "llm_batch_size": 5
  },
  "speaker_filter": {
    "mode": "exclude",
    "speakers": ["Therapist", "Instructor"]
  },
  "theme_classification": {
    "backend": "lmstudio",
    "model": "nvidia/nemotron-3-super",
    "models": [],
    "per_run_models": ["nvidia/nemotron-3-super", "google/gemma-4-31b", "qwen/qwen3-next-80b"],
    "n_runs": 3,
    "temperature": 0.1,
    "lmstudio_base_url": "http://127.0.0.1:1234/v1"
  },
  "codebook_embedding": {
    "two_pass": true,
    "embedding_model": "Qwen/Qwen3-Embedding-8B",
    "exemplar_import_path": null
  },
  "confidence_tiers": {
    "high_confidence": 0.8,
    "medium_min_confidence": 0.6
  },
  "test_sets": {
    "enabled": true,
    "n_sets": 2,
    "fraction_per_set": 0.10,
    "random_seed": 42
  },
  "therapist_cues": {
    "enabled": true,
    "max_length_per_cue": 250,
    "max_length_of_average_cue_responses": 1000
  },
  "validation": {
    "n_per_class": 50,
    "min_kappa": 0.70,
    "min_agreement": 0.75
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

**Symptoms**: Process crashes with CUDA out of memory or system memory allocation errors.

**Solutions**:
- Use the lightweight embedding model: set `embedding_model` to `all-MiniLM-L6-v2`
- Reduce `max_segment_words_conversational` to limit segment size
- For HuggingFace backends: close other GPU-intensive applications
- Set `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512` environment variable

#### 2. LM Studio Connection Failed

**Symptoms**: Error "Connection refused" or pipeline hangs waiting for the server.

**Solutions**:
- Verify LM Studio server is running: `http://127.0.0.1:1234/v1/models` should return JSON
- Ensure a model is loaded in LM Studio (not just downloaded)
- Check `lmstudio_base_url` in your config matches the actual server port
- QRA will wait up to ~9 hours for LM Studio to start — you can interrupt with Ctrl+C

#### 3. HuggingFace Model Download Fails

**Symptoms**: `HFCacheDownloadError` or connection timeout during model download.

```bash
# Pre-download the embedding model manually
python -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('Qwen/Qwen3-Embedding-8B', trust_remote_code=True)
"
```

#### 4. LLM Returns Empty/Invalid Responses

**Symptoms**: Classification results show `null` for all segments, or `agreement_level` is `none`.

**Solutions**:
- Check API key environment variables are set
- Verify model name is correct for the chosen backend
- Try increasing temperature slightly (e.g., 0.1 → 0.3)
- Check context window — very long segments may exceed the model's limit; reduce `max_segment_words_conversational`

#### 5. Speaker Names Not Normalizing Consistently

**Symptoms**: Participant IDs change between runs (e.g., `participant_1` vs `participant_3`).

**Solution**: Create a `speaker_anonymization_key.json` file in your input directory with explicit mappings and provide it in Step 1b of the wizard. The key in `07_meta/` always takes precedence on re-runs.

#### 6. Segmentation Produces Too Many / Too Few Segments

**Solutions**:
- Adjust `min_segment_words_conversational` and `max_segment_words_conversational`
- Reduce `max_gap_seconds` to merge fewer utterances into each segment
- Enable `--verbose-segmentation` to inspect the process log: `07_meta/process_log.txt`

### Debug Mode

Enable verbose logging to capture every LLM prompt/response and segmentation decision:

```bash
python qra.py run --config ./data/output/07_meta/qra_config.json --verbose-segmentation
```

This writes `07_meta/process_log.txt` with detailed processing information.

### Getting Help

1. Check the output index: `output_dir/00_index.txt` (lists all files and sizes)
2. Review the process log: `output_dir/07_meta/process_log.txt`
3. Check flagged segments: `output_dir/05_validation/flagged_for_review.txt`
4. Verify input data format matches the expected schema above

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
cat ./data/output/07_meta/qra_config.json

# 7. Run pipeline
python qra.py run --config ./data/output/07_meta/qra_config.json

# 8. Analyze results (if not using --auto-analyze)
python qra.py analyze --output-dir ./data/output/

# 9. Check output index
cat ./data/output/00_index.txt
```

---

## Next Steps

After setup is complete:

1. **Run a small test**: Start with 2–3 short transcripts to validate segmentation and classification quality
2. **Review initial results**: Check `05_validation/flagged_for_review.txt` and coded transcript files in `01_transcripts/coded/`
3. **Tune parameters**: Adjust segmentation bounds and confidence thresholds based on the test run
4. **Scale up**: Run the full dataset after parameter tuning
5. **Human validation**: Use the test set worksheets in `05_validation/testsets/` for inter-rater reliability studies
