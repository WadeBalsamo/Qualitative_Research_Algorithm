# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Qualitative Research Algorithm (QRA)** is an LLM-based classification pipeline for analyzing therapeutic dialogue transcripts. It classifies dialogue segments into theme/stage categories (e.g., VA-MR framework: Vigilance, Avoidance, Metacognition, Reappraisal) and optionally applies multi-label phenomenology codebook classification. Your job is to focus on the Analysis module

## Architecture

### High-Level Flow

The 6-stage pipeline orchestrated in `process/orchestrator.py`:

1. **Transcript Ingestion** (`transcript_ingestion.py`): Load diarized sessions from VTT/JSON files
2. **Semantic Segmentation** (`llm_segmentation.py`): Split long transcripts into coherent segments using sentence-transformer embeddings
3. **Theme Classification** (`llm_classifier.py`): Classify segments into VA-MR stages with triplicate runs and consistency checking
4. **Codebook Classification** (optional, `codebook/embedding_classifier.py`): Multi-label codebook classification via triple-LLM ensemble
5. **Human Validation** (optional, `human_validator.py`): Interactive validation for uncertain classifications
6. **Dataset Assembly** (`dataset_assembly.py`): Export results in multiple formats (master dataset, coded transcripts, reports)

### Module Organization

```
classification_tools/          # Shared infrastructure
├── llm_client.py             # LLM backend abstraction (OpenRouter, HuggingFace, Replicate, Ollama, LM Studio)
├── model_loader.py           # Model downloading and caching
├── llm_classifier.py         # Theme classification logic
├── data_structures.py        # Segment, SegmentSet dataclasses
├── response_parser.py        # Parse LLM outputs into structured format
├── validation.py             # Evaluation set creation, consistency checking
└── majority_vote.py          # Multi-model agreement logic

constructs/                    # Theme frameworks
├── vamr.py                   # VA-MR preset (Vigilance, Avoidance, Metacognition, Reappraisal)
├── theme_schema.py           # ThemeFramework dataclass
└── config.py                 # Framework config loading

codebook/                      # Multi-label classification
├── embedding_classifier.py   # Triple-LLM embedding classifier (two-pass exemplar accumulation)
├── phenomenology_codebook.py # Default phenomenology codebook
├── codebook_schema.py        # Codebook dataclass
└── ensemble.py               # Combine embedding and LLM results

process/                       # Pipeline orchestration
├── orchestrator.py           # Main 6-stage pipeline
├── config.py                 # PipelineConfig dataclass (serializable to/from JSON)
├── transcript_ingestion.py   # Load and parse transcripts
├── llm_segmentation.py       # Embedding-based semantic segmentation
├── setup_wizard.py           # Interactive configuration prompts
├── human_validator.py        # Interactive validation UI
├── dataset_assembly.py       # Export results
├── cross_validation.py       # Empirical validation of theme-to-codebook mappings
└── pipeline_hooks.py         # Observer pattern for progress feedback

qra.py                         # Unified CLI entry point (setup/run/guided subcommands)
```

## Key Data Structures

### `Segment` (classification_tools/data_structures.py)
The core unit. Represents a single dialogue turn or coherent utterance with:
- `speaker`: Speaker label ("P" for participant, "T" for therapist)
- `text`: The segment content
- `theme_label`: Assigned VA-MR stage (0-3) or None
- `theme_confidence`: Confidence score
- `codes`: Multi-label codebook codes (if codebook enabled)
- `timestamp_ms`: Timing info

### `PipelineConfig` (process/config.py)
Serializable JSON config capturing all pipeline settings:
- Paths (transcript_dir, output_dir)
- Backend and model selection
- Framework and codebook specs
- Classification parameters (n_runs, temperature)
- Confidence thresholds
- Feature flags (theme_labeler, codebook_classifier)

### `ThemeFramework` (constructs/theme_schema.py)
Describes theme stages and optional exemplar utterances:
- `name`: Framework name (e.g., "VA-MR")
- `stages`: List of stage definitions with descriptions
- `exemplars`: Optional custom exemplar utterances for each stage

## Important Design Notes

### Segmentation Efficiency (★ Key Guidance)
**Do not waste LLM calls on therapist-only segments.** Therapist utterances should be used as read-only context for understanding participant segments, not classified themselves. The `--speaker-filter-mode` flag controls this:
- `exclude`: Drop therapist segments before classification (most efficient)
- `isolate`: Keep only participant segments (alternative)
- `none`: Classify everything

When modifying segmentation or speaker filtering logic, prioritize efficiency by minimizing LLM calls on context-only content.

### Triplicate Consistency Checking
Theme classification runs each segment 3 times with stochastic sampling. Results are categorized by agreement level (unanimous/majority/split) and confidence tier (high/medium/low). This pattern is in `classification_tools/llm_classifier.py`.

### Two-Pass Codebook Classification
The embedding classifier has two modes (controlled by `--no-two-pass` flag):
- **Pass 1**: Discovers high-confidence exemplar segments for each codebook code
- **Pass 2**: Uses accumulated exemplars to enrich context during full classification

If improving codebook accuracy, understand this two-pass logic in `codebook/embedding_classifier.py`.

## Important API and Configuration Details

### LLM Backend Abstraction (llm_client.py)
All LLM calls go through `LLMClient`, which abstracts:
- LM Studio (local API server)

When adding features that interact with LLMs, use `LLMClient.complete()` for consistency.

### Configuration Persistence
Configs are serialized to JSON via `PipelineConfig.to_json()` and deserialized via `PipelineConfig.from_json()`. This happens in `setup_wizard.py` (save) and `qra.py` (load). Secrets are redacted in saved configs via `_blank_secrets()`.

## Common Patterns and Gotchas

1. **Speaker filtering is critical for LLM efficiency**: Always consider whether segments should be classified or used as context only.
2. **Segmentation must produce coherent utterances**: Embedding-based segmentation uses semantic similarity; if it's creating mid-sentence breaks, the sentence-transformer embedding threshold may need tuning in `llm_segmentation.py`.
3. **Confidence tiers are computed post-hoc**: A segment's confidence level depends on agreement across runs, not the LLM's own confidence scores. See `classification_tools/validation.py`.

## Testing and Debugging

No formal test suite exists. 
