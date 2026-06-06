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
| `python qra.py classify -o ./output/ --what vaamr` | Run VAAMR classification only |
| `python qra.py analyze -o ./output/` | Generate analysis reports (superposition, mechanism, efficacy, language atlas) |
| `python qra.py analyze -o ./output/ --gnn` | Analysis + GNN layer (segment positioning, cue motifs, triangulation, coupling) |

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
python qra.py classify -o ./output/ --what vaamr
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
python qra.py classify -o ./data/output/ --what vaamr
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
- Stage superposition (mixture vectors, entropy, cusp density report)
- Per-participant longitudinal reports with continuous progression + volatility
- Per-session summaries with prototypical exemplars and superposition annotations
- Per-theme (stage + codebook) analyses
- PURER→VAAMR mechanism dossier with full statistical inference (CI/p/FDR)
- Program efficacy dossier with group trajectory and outcome linkage
- Therapeutic language atlas (ranked exemplar FROM→CUE→TO blocks)
- Therapist cue response analysis
- Session and participant LLM summaries
- Graph-ready CSVs and visualization figures (superposition, mechanism, efficacy)

GNN representation-and-discovery layer (ON by default; force with `--gnn`, skip with `--no-gnn`):

*Discovery & triangulation (Capabilities A–E):*
- Per-segment stage-mixture vectors from graph geometry (Capability A)
- Cue motif discovery: emergent therapist-language clusters + influence scoring (Capability B)
- GNN↔LLM↔human triangulation + lift comparison (Capability C)
- Construct signal ablation: which label families carry independent signal (Capability D, `run_gnn_ablation=true`); plus the VCE-on-VAAMR test (`test_vce_layer=true`)
- Latent participant↔therapist coupling factors with alliance naming (Capability E)

*Trustworthy LLM-free classifier:* out-of-sample reliability gate (`06_gnn/validation.txt`, persisted to `gnn_gate.json`) with a rare-stage recall floor and a YES/NO scaling verdict; gate-gated promotion to the `gnn_consensus` label tier; opt-in, individually-measured hardening — abstention/deferral, temperature + OOD calibration, label propagation, scale-mode simulation, typed therapist→participant edges.

*Config-flag–driven advanced analyses (set in the `gnn_layer` config section):*
- **Track B — model-counterfactual influence** (`counterfactual=true`, gated on the reliability gate): per-PURER-move sensitivity on the following participant's predicted progression, triangulated against the observed Δprogression (`06_gnn/influence.txt`). Sensitivity analysis, not causation.
- **Track C — MindfulBERT dataset builder** (`build_mindfulbert_dataset=true`; `augmentation_enabled=true` for the gate-gated, ablation-retained counterfactual channel): `02_meta/training_data/mindfulbert_dataset.jsonl` + datasheet.
- **Track D — subtext communities as routines** (`subtext_communities=true`): two-algorithm partition + ARI, within-session routine transitions, participant-bootstrap stability selection (`06_gnn/communities.txt`).

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

# Enable GNN analysis layer at analyze-time
python qra.py analyze -o ./output/ --gnn

# GNN with ablation (Capability D) — adds extra training time
# Set run_gnn_ablation: true in qra_config.json gnn_layer section
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

Running `python qra.py` with **no subcommand** launches the interactive TUI (see [Interactive TUI Reference](#interactive-tui-reference) below), which reaches every capability in this table through guided menus.

### Main Commands
| Command | Description |
|---------|-------------|
| *(none)* | Launch the interactive TUI |
| `setup` | Interactive configuration wizard (creates `qra_config.json`) |
| `run` | Execute the complete pipeline (ingest → classify → assemble → analyze) |
| `add-data` | Incrementally add new transcripts to an existing project (frozen segments/testsets untouched) |
| `ingest` | Segment + freeze transcripts only (Stage 1) |
| `classify` | Run classifiers only — `--what vaamr\|purer\|codebook\|cross-validation\|all` |
| `assemble` | Join frozen segments and classification overlays into `master_segments` |
| `validate` | Refresh human/AI validation artifacts without re-classifying |
| `analyze` | Post-hoc results analysis (`--gnn` / `--no-gnn` to force the GNN layer) |
| `reclassify-run` | Redo a single classification run (e.g. fix one run's model) |
| `testset create` / `refresh` / `list` | Manage frozen validation test sets (`--kind vaamr\|purer\|codebook`) |
| `cv create` / `refresh` / `list` | Manage content-validity test sets (`--framework vaamr\|purer`) |
| `apply-anonymization` | Retroactively scrub PHI names from already-frozen segment text |
| `edit-anonymization` | Edit the speaker anonymization key and cascade the change across all artifacts |

### Advanced Options
```bash
# Re-segment specific session
python qra.py ingest -o ./output/ --reingest c1s1

# Re-segment all sessions
python qra.py ingest -o ./output/ --reingest-all

# Classify one layer without the automatic downstream assemble/analyze
python qra.py classify -o ./output/ --what vaamr --no-downstream

# LLM-free graph "scale mode" (after the GNN reliability gate passes)
python qra.py classify -o ./output/ --backend gnn

# Force the GNN representation-and-discovery layer on/off at analyze-time
python qra.py analyze -o ./output/ --gnn
python qra.py analyze -o ./output/ --no-gnn

# Fix a single run's model, then auto re-assemble + re-analyze
python qra.py reclassify-run -o ./output/ --run 3 --model nvidia/nemotron-3-nano-30b

# Retroactively anonymize frozen text; edit the speaker key with full cascade
python qra.py apply-anonymization -o ./output/ --yes
python qra.py edit-anonymization -o ./output/ --merge "Jill,Bill=therapist_1" --yes

# Verbose segmentation logging
python qra.py run --verbose-segmentation

# Zero-shot content-validity test (skips full pipeline)
python qra.py run --test-zeroshot --preset small --output-dir ./data/output/
```

---

## Interactive TUI Reference

Running `python qra.py` with no subcommand launches a menu-driven Text User Interface. It is the recommended entry point for new users — it surfaces every pipeline stage, validation tool, and editor through guided prompts, and continuously displays project state (which stages have run, how many testsets exist, and the GNN's reliability status).

### Main Menu
| Option | Action |
|--------|--------|
| **1 — New Project** | Run the setup wizard, then optionally execute the full pipeline immediately |
| **2 — Open Project** | Load an existing (or legacy) project directory and access all stages, testset management, and editors |
| **3 — About / Help** | Framework reference (VAAMR/PURER/VCE), on-disk layout, and CLI command summary |
| **0 — Exit** | Quit |

### New Project — Setup Wizard
The wizard offers three modes — **Small/Test** (lightweight models for rapid iteration), **Production** (large research-grade models), and **Custom** (step-by-step control of every hyperparameter) — and walks through: input/output paths, speaker-key import, PHI text anonymization, therapist/speaker role identification, PURER cue options, segmentation parameters, LLM backend and per-run checker models, framework and exemplar selection, codebook (VCE) options, classification parameters and confidence thresholds, validation and content-validity test sets, post-pipeline analysis, therapist-cue summarization, session/participant summaries, the **GNN layer** (label mode, scope, reliability-gate target, authoritative toggle, ablation, motif/factor counts), and finally save-and-run.

### Open Project Menu
The menu adapts to detected project state. Available actions:

| Option | Action |
|--------|--------|
| **1 — Ingest & Freeze Segments** | Stage 1: segment transcripts, write frozen `segments.jsonl` (migrates legacy layouts) |
| **2 — Classify VAAMR** | Stage 3: participant-segment VAAMR classification (optional zero-shot mode) |
| **3 — Classify PURER** | Stage 3c: therapist cue-block PURER classification; sub-menu to re-run all / change model / add a run / redo one run |
| **4 — Assemble Master Dataset** | Stage 6: join overlays → `master_segments` + coded transcripts + validation artifacts |
| **5 — Analysis & Reports** | Stage 8: longitudinal/session/theme analyses, mechanism + efficacy dossiers, figures, summaries |
| **6 — Testset Management** | Create VAAMR/PURER testsets, refresh AI answer keys, list frozen testsets |
| **7 — Content-Validity Testsets** | Create/refresh/list content-validity testsets (exemplar/subtle/adversarial items) |
| **8 — Refresh Validation Artifacts** | Re-emit human forms, coded transcripts, and AI answer keys without re-classifying |
| **9 — Edit Configuration** | Change LM Studio URL, models, `n_runs`; toggle codebook / PURER / **GNN layer** / **GNN-authoritative labels**; open `qra_config.json` in `$EDITOR` |
| **10 — Edit Speaker Anonymization Key** | Launch the anonymization editor (rename/merge/relabel speakers, cascade across all artifacts) |
| **11 — Classify (Graph Consensus, LLM-Free)** | Label new data with the trained graph — recommended once `06_reports/06_gnn/validation.txt` reports the graph is ready; warns and asks for confirmation otherwise |

### Anonymization Editor (Open Project → option 10)
A roster TUI listing every speaker as `anonymized_id ⟵ original name (role)`. Per-speaker actions: rename anonymized ID, rename raw name, change role (participant/therapist/staff), merge into another speaker, remove. Roster-level actions: walk through all speakers in sequence (`r`), add a speaker (`a`), toggle NLP name re-removal (`n`), **preview cascade as a dry-run** (`p`), and **apply changes & cascade** (`w`) — which writes a timestamped backup, rewrites frozen segment fields/IDs/tokens, remaps overlays and checkpoints, updates validation worksheets, and regenerates the master dataset and analysis. The same operations are scriptable via `qra edit-anonymization` flags (`--rename`, `--rename-raw`, `--set-role`, `--merge`, `--remove-names`, `--dry-run`, `--yes`).

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
│  - Stage superposition: mixture vectors, entropy, cusp      │
│    density (analysis/superposition.py)                      │
│  - Per-participant longitudinal reports with continuous      │
│    progression coordinate + volatility                      │
│  - Per-session and per-theme analyses                       │
│  - Graph-ready CSVs                                         │
│  - Stage progression + transition explanation               │
│  - PURER→VAAMR mechanism dossier with CI/p/FDR             │
│    (analysis/mechanism.py + analysis/stats.py)              │
│  - Program efficacy dossier + optional external outcome      │
│    linkage (analysis/efficacy.py)                           │
│  - Therapist cue response analysis                          │
│  - Therapeutic language atlas (analysis/reports/           │
│    language_atlas.py)                                       │
│  - Session + participant LLM summaries                      │
│  - Visualization figures (superposition, mechanism,         │
│    efficacy, PURER heatmaps)                                │
│  - GNN analysis (optional, --gnn flag):                     │
│    Capability A: segment positioning + superposition        │
│    Capability B: cue motif discovery + influence            │
│    Capability C: GNN↔LLM triangulation + lift tables        │
│    Capability D: construct signal ablation (opt-in)         │
│    Capability E: participant↔therapist coupling factors     │
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

#### Sub-config: `superposition`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | true | Enable stage-superposition enrichment |
| `mixture_source` | string | `auto` | Mixture source: `gnn`, `ballots`, `secondary`, or `auto` (priority order: GNN → ballots → secondary) |
| `liminal_entropy_threshold` | float | 0.6 | Normalized entropy threshold for liminal classification |
| `liminal_gap_threshold` | float | 0.25 | Max gap between top-two mixture components for liminality |
| `active_stage_threshold` | float | 0.15 | Minimum mixture component to count as "active" |
| `run_mechanism_analysis` | bool | true | Run mechanism dossier after superposition enrichment |

#### Sub-config: `efficacy`

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | true | Enable efficacy dossier |
| `outcomes_path` | string | `02_meta/outcomes.csv` | Path to external outcomes CSV (graceful if absent) |
| `adaptive_stages` | int[] | `[2,3,4]` | Stages counted as adaptive occupancy |
| `maladaptive_stages` | int[] | `[0,1]` | Stages counted as maladaptive occupancy |
| `barrier_from` | int | 1 | Avoidance-barrier FROM stage index |
| `barrier_to` | int | 2 | Avoidance-barrier TO stage index |

#### Sub-config: `gnn_layer`

The GNN layer is ON by default and runs at analyze-time (set `enabled=False` to skip it). It plays two roles: discovery/triangulation (Capabilities A–E) and a **graph-distilled consensus classifier** with an out-of-sample reliability gate. See `methodology.md` §8.5 for the as-built specification.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `enabled` | bool | false | Enable the GNN layer (or use `qra analyze --gnn`) |
| `embedding_model` | string | `Qwen/Qwen3-Embedding-8B` | Segment embedding model (reused from segmentation/VCE — no extra model) |
| `use_query_prefix` | bool | true | Apply the embedding model's query prefix |
| `embedding_batch_size` | int | 8 | Embedding encode batch size |
| `cache_embeddings` | bool | true | Cache embeddings to `02_meta/gnn/segment_embeddings.npz` |
| `knn_k` | int | 8 | kNN similarity edges per segment node (enables inductive attachment of new data) |
| `include_vce_nodes` | bool | false | Add VCE-code anchor nodes (off by default; toggled on for the Capability-D ablation) |
| `include_purer_nodes` | bool | true | Add PURER-move anchor nodes |
| `cross_framework_min_lift` | float | 1.5 | Lift threshold for cross-framework (VAAMR↔VCE) edges |
| `hidden_dim` | int | 128 | GraphSAGE hidden dimension |
| `n_layers` | int | 2 | Number of SAGE aggregation layers |
| `dropout` | float | 0.5 | Dropout |
| `objectives` | string[] | `[soft_vaamr, progression, contrastive, link_prediction]` | Training objectives (also: `vce_multilabel`, `purer`) |
| `label_mode` | string | `weak` | Training signal: `weak` (LLM ballots), `human` (human-coded subset only), `self_supervised` (structure only) |
| `contrastive_temp` | float | 0.1 | InfoNCE temperature for the stage-separating contrastive head |
| `epochs` | int | 300 | Maximum training epochs |
| `lr` | float | 0.001 | Learning rate |
| `patience` | int | 40 | Early-stopping patience |
| `seed` | int | 42 | Random seed |
| `device` | string\|null | null | `cuda`/`cpu`, or null to auto-detect |
| `run_on_participants` | bool | true | Position participant segments (VAAMR mixture / progression) |
| `run_on_therapists` | bool | true | Position therapist segments (cue-block) |
| `n_motif_clusters` | int | 12 | Number of cue-motif clusters (Capability B) |
| `min_motif_influence` | float | 1.2 | Forward-transition lift above which a motif is flagged for review |
| `motif_min_block_count` | int | 3 | Ignore motifs with fewer than this many cue blocks |
| `n_latent_factors` | int | 5 | Latent coupling factors (Capability E) |
| `interpret_against_cf_ic` | bool | true | Name latent factors against the inline CF/IC alliance lexicon |
| `run_gnn_ablation` | bool | false | Construct-head ablation (Capability D; doubles training cost) |
| `produce_consensus_labels` | bool | true | Write per-segment graph labels to the `gnn_labels` overlay |
| `gnn_authoritative` | bool | false | **Promotion switch** — make graph labels the label of record. Effective **only** when the persisted gate verdict (`gnn_gate.json`) reports `ready_for_scaling` |
| `validation_folds` | int | 5 | k for k-fold held-out reliability evaluation |
| `validation_holdout` | float | 0.2 | Single-holdout fraction when `validation_folds ≤ 1` |
| `irr_target` | float | 0.70 | κ vs LLM consensus at which LLM-free graph scaling is recommended |

**Track A — classifier hardening (each opt-in + measured; kept only if it raises the gate κ):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `precipitates_edges` / `run_precipitates_ablation` | bool | false / false | Typed therapist→participant edges + learnable per-edge-type gate (A1); ablation tests Δκ on both axes |
| `abstain_threshold` / `abstain_rare_stage_threshold` | float\|null | null / null | Global max-prob abstention floor; higher floor for rare stages 3/4 (A2) |
| `abstain_per_stage` | object\|null | null | Explicit per-stage floors (overrides; also where calibration writes) |
| `abstain_calibrate` / `abstain_target_precision` | bool / float | false / 0.80 | Derive per-stage floors from held-out CV at target precision (A2) |
| `calibrate` / `calibration_temperature` | bool / float\|null | false / null | Temperature scaling on the soft-VAAMR head (A3) |
| `ood_threshold` / `ood_knn_k` | float\|null / int | null / 8 | Scale-mode OOD deferral by mean kNN cosine distance to training (A3) |
| `label_propagation` / `propagation_alpha` / `propagation_iters` | bool / float / int | false / 0.5 / 20 | Measured post-training soft-label diffusion; kept only if Δκ ≥ +0.02 (A4) |
| `run_scale_sim` / `scale_sim_holdout_sessions` / `scale_sim_max_gap` | bool / int / float | false / 1 / 0.10 | Inductive whole-session holdout vs CV κ; flags domain-shift risk (A5) |

**Track B — model-counterfactual influence (gated on the reliability gate; sensitivity, not causal):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `counterfactual` | bool | false | Run per-PURER-move counterfactual influence + triangulation vs observed Δprogression (B3/B4) |
| `counterfactual_max_blocks` / `influence_bootstrap_n` | int\|null / int | null / 1000 | Cap on per-block re-forwards (logged when capped); participant-clustered bootstrap resamples |
| `counterfactual_subgroups` | bool | false | Split influence by session-number tertile, underpowered-flagged (B5) |

**Track C — MindfulBERT training-set builder:**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `build_mindfulbert_dataset` | bool | false | Build the versioned (cue language → observed Δprogression) dataset + datasheet (C1/C2/C5) |
| `augmentation_enabled` / `augmentation_min_gain` | bool / float | false / 0.0 | Add the gate-gated GNN-counterfactual channel (C3); retain only if held-out gain exceeds this (C4) |

**Track D — subtext communities as routines (gate-independent discovery):**

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `subtext_communities` | bool | false | Run the subtext-community / routine-discovery layer (D) |
| `community_sim_threshold` / `community_min_size` | float / int | 0.85 / 3 | Cosine edge threshold; minimum community size to report |
| `community_stability_min` / `community_stability_boots` | float / int | 0.5 / 50 | Suppress communities below this bootstrap co-membership stability; resamples (D4) |

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
| `02_meta/classifications/gnn_labels.jsonl` | JSONL | Graph-distilled consensus overlay (`gnn_vaamr_pred`/`conf`, `gnn_purer_pred`/`conf`, `gnn_label_source`) |
| `02_meta/gnn/segment_embeddings.npz` | NPZ | Cached Qwen3 segment embeddings (incremental) |
| `02_meta/gnn/model/weights.pt` + `manifest.json` | PyTorch/JSON | Trained GraphSAGE checkpoint + config/seed/metrics |
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
| `final_label_source` | string | Source of final label — resolution order `adjudicated` > `human_consensus` > `gnn_consensus` > `llm_zero_shot` |
| `codebook_labels_ensemble` | string[] | Multi-label VCE codes from ensemble reconciliation |
| `codebook_confidence` | object | Per-code confidence scores |
| `purer_primary` | int | PURER primary move type (0-4) |
| `purer_confidence_primary` | float | PURER confidence score |
| `purer_agreement_level` | string | PURER agreement level |
| `gnn_vaamr_pred` / `gnn_vaamr_conf` | int / float | Graph-distilled VAAMR prediction + confidence (from `gnn_labels` overlay) |
| `gnn_purer_pred` / `gnn_purer_conf` | int / float | Graph-distilled PURER prediction + confidence |
| `gnn_label_source` | string | `gnn_trained` or `gnn_scale_mode` |
| `human_label` | int | Human-coded label (when available) |
| `in_human_coded_subset` | bool | Included in human evaluation set |

### Analysis Output Files

| File/Directory | Description |
|----------------|-------------|
| `03_analysis_data/session_stats/stats_<session>.json` | Per-session classification statistics |
| `03_analysis_data/session_stats/stats_cumulative.json` | Cumulative statistics across all sessions |
| `03_analysis_data/per_session/<session>.json` | Full per-session analysis (includes superposition summary) |
| `03_analysis_data/per_participant/<participant>.json` | Per-participant longitudinal analysis (includes continuous progression, volatility, entropy) |
| `03_analysis_data/per_theme/<stage>.json` | Per-theme (VAAMR stage) analysis |
| `03_analysis_data/per_theme/<code>.json` | Per-code (VCE) analysis |
| `03_analysis_data/cumulative_report.json` | Aggregated cumulative report |
| `03_analysis_data/longitudinal_summary.json` | Longitudinal trajectory summary |
| `03_analysis_data/session_stage_progression.csv` | Stage progression per session (graph-ready) |
| `03_analysis_data/graphing/*.csv` | Graph-ready datasets |
| `03_analysis_data/graphing/segment_superposition.csv` | Per-segment mixture vector + entropy export |
| `03_analysis_data/session_summaries.json` | LLM-generated session summaries |
| `03_analysis_data/mechanism/mechanism_delta_progression.csv` | Δprogression by PURER/from-stage with CI/p/effect/FDR columns |
| `03_analysis_data/mechanism/mechanism_liminality.csv` | Liminality leverage analysis |
| `03_analysis_data/mechanism/mechanism_avoidance_barrier.csv` | Avoidance-barrier crossing analysis |
| `03_analysis_data/mechanism/participant_trajectory_types.csv` | Trajectory typology per participant |
| `03_analysis_data/mechanism/mechanism_purer_mixed_effects.csv` | Mixed-effects Δprog ~ PURER model output |
| `03_analysis_data/efficacy/participant_session_outcomes.csv` | Per-(participant, session) progression + occupancy |
| `03_analysis_data/efficacy/barrier_crossing.csv` | Avoidance→AttnReg first-passage per participant |
| `03_analysis_data/efficacy/group_trajectory.csv` | Group mean progression with bootstrap CI |
| `03_analysis_data/efficacy/participant_slopes.csv` | Per-participant OLS progression slopes |
| `03_analysis_data/gnn/segment_positions.csv` | Per-segment GNN mixture, progression_coord, node_type (Capability A) |
| `03_analysis_data/gnn/cue_motifs.csv` | Motif stats: influence, purity, n_exemplars (Capability B) |
| `03_analysis_data/gnn/cue_block_assignments.csv` | Per-cue-block motif assignment sidecar (Capability B) |
| `03_analysis_data/gnn/gnn_head_predictions.csv` | Per-segment GNN head predictions for purer/vce (Capability C) |
| `03_analysis_data/gnn/gnn_vs_llm_lift.csv` | GNN-vs-LLM lift comparison (Capability C) |
| `03_analysis_data/gnn/coupling_factors.csv` | Latent coupling factors with forward correlation (Capability E) |
| `03_analysis_data/gnn/gnn_construct_signal.csv` | Ablation loss deltas per construct head (Capability D) |
| `03_analysis_data/gnn/gnn_validation.csv` | Per-class out-of-sample reliability-gate table (machine-readable) |
| `03_analysis_data/gnn/gnn_gate.json` | Persisted gate verdict (`ready_for_scaling`, per-framework κ, rare-stage notes, calibration T) — gates authoritative promotion |
| `03_analysis_data/gnn/gnn_counterfactual_influence.csv` | Per-PURER-move model-counterfactual influence + CIs (Track B) |
| `03_analysis_data/gnn/subtext_communities.csv` | Per-community size / stability / top TF-IDF terms (Track D) |
| `03_analysis_data/gnn/subtext_community_transitions.csv` | Within-session community→community routine transitions (Track D) |
| `02_meta/training_data/mindfulbert_dataset.jsonl` | MindfulBERT dataset: (cue language → observed Δprogression) + provenance (Track C) |
| `02_meta/training_data/mindfulbert_datasheet.{json,txt}` | Dataset datasheet: provenance mix, gate status, augmentation ablation, caveats (Track C) |
| `06_reports/01_outcomes/`, `02_mechanism/`, … | Tiered human-readable reports (superposition, mechanism, avoidance barrier, efficacy, language atlas, per-session/participant/stage) |
| `06_reports/06_gnn/validation.txt` | **Reliability gate** — per-stage/per-move out-of-sample κ vs LLM consensus + human, with the "ready for LLM-free scaling? YES/NO" verdict |
| `06_reports/06_gnn/emergent_motifs.txt` | Emergent cue motifs flagged for human review (Capability B) |
| `06_reports/06_gnn/triangulation.txt` (+ `triangulation_independence.txt`) | GNN↔LLM↔human agreement and lift comparison (Capability C) |
| `06_reports/06_gnn/construct_signal.txt` / `vce_contribution.txt` | Construct-signal ablation + VCE-on-VAAMR test (Capability D) |
| `06_reports/06_gnn/coupling.txt` | Latent coupling factors and alliance naming (Capability E) |
| `06_reports/06_gnn/influence.txt` | Model-counterfactual influence + triangulation vs observed Δprogression (Track B) |
| `06_reports/06_gnn/communities.txt` | Subtext communities as routines, with stability selection (Track D) |
| `06_reports/06_gnn/{scale_sim,label_propagation,precipitates_contribution}.txt` | Track A hardening reports (when those instruments are enabled) |

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
| `mindfulbert_dataset.py` | **Track C** — MindfulBERT (cue language → observed Δprogression) dataset builder + augmentation-validation harness + datasheet |
| `_shared.py` | Shared helpers for assembly functions |

### Analysis (`analysis/`)

| File | Description |
|------|-------------|
| `runner.py` | Post-hoc analysis orchestrator — sequences all analysis steps |
| `superposition.py` | Stage-mixture provider: GNN → LLM ballots → secondary_stage fallback; entropy, co-occurrence |
| `mechanism.py` | PURER→VAAMR mechanism dossier with full statistical inference (CI, permutation p, FDR) |
| `stats.py` | Inference toolkit: Wilson CI, cluster-bootstrap CI, permutation test, effect sizes, BH-FDR, mixed-effects |
| `efficacy.py` | Program efficacy: internal progression outcomes + external outcome linkage |
| `loader.py` | Load master JSONL and framework from output directory |
| `participant.py` | Per-participant report generation (includes continuous progression, volatility) |
| `session.py` | Per-session analysis (includes superposition summary) |
| `theme.py` | Per-theme (VAAMR stage + code) analyses |
| `stage_progression.py` | Session-level stage progression computation |
| `longitudinal.py` | Longitudinal summary generation |
| `figure_data.py` | Export graph-ready CSV datasets (includes segment_superposition.csv) |
| `figures.py` | Matplotlib figures: superposition timeline/entropy/co-occurrence, mechanism heatmap, GNN convergence, efficacy multipanel |
| `exemplars.py` | Exemplar utterance extraction per stage (includes mixture + entropy annotations) |
| `text_reports.py` | Human-readable text report utilities |
| `purer_analysis.py` | PURER × VAAMR conditional lift table, cue-response synthesis, Cramér's V association test |
| `purer_figures.py` | PURER × VAAMR lift heatmap and figures |
| `reports/superposition_report.py` | Superposition text report: corpus mixture, cusp matrix, liminal exemplars |
| `reports/language_atlas.py` | Therapeutic language atlas: ranked FROM→CUE→TO exemplars by PURER/motif/factor |
| `reports/transition_report.py` | Transition explanation + therapist cue reports (includes Δprogression by type) |
| `reports/` | Full suite: session, stage, transition, cue response, longitudinal, summaries |

### GNN Layer (`gnn_layer/`)

| File | Description |
|------|-------------|
| `runner.py` | GNN analysis entry point — orchestrates all five capabilities |
| `config.py` | `GnnLayerConfig` dataclass (enabled=True default) |
| `embeddings.py` | Qwen3 segment embedding reuse with NPZ cache |
| `graph_builder.py` | Heterogeneous graph: temporal chain, anchor/label, kNN, cross-framework edges |
| `model.py` | Pure-PyTorch GraphSAGE with multi-task heads (no torch-geometric needed) |
| `soft_labels.py` | Ballot-to-mixture conversion, progression coordinate, soft target assembly |
| `train.py` | Full-batch training, early stopping, checkpoint export |
| `inference.py` | Per-segment position inference, cue-block embedding assembly |
| `motifs.py` | Cue motif clustering, influence scoring, purity annotation, emergent-motif flagging |
| `gnn_lift.py` | GNN-derived lift tables vs LLM baseline (Capability C) |
| `triangulation.py` | GNN↔LLM↔human agreement (Cohen's κ), triangulation report |
| `ablation.py` | Construct-head ablation + VCE-on-VAAMR test + anchor/precipitates contribution ablations (Capability D / Track A1) |
| `coupling.py` | Latent participant↔therapist coupling factors, CF/IC alliance naming (Capability E) |
| `anchors.py` | Optional construct-anchor features + similarity/lift edges (opt-in, human-axis ablated) |
| `calibration.py` | Temperature scaling + ECE + OOD score for domain-shift confidence (Track A3) |
| `propagation.py` | Measured post-training soft-label diffusion (Track A4) |
| `influence.py` | **Track B** — model-counterfactual influence + triangulation vs `mechanism.py` (gated; sensitivity, not causal) |
| `communities.py` | **Track D** — subtext-similarity graph, two-algorithm communities, routine transitions, stability selection |
| `validation.py` | Out-of-sample reliability gate (per-stage/per-move κ, rare-stage floor) + scale-mode simulation + persisted gate verdict |
| `reports.py` | GNN artifact writers: CSVs and human-readable reports |

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
