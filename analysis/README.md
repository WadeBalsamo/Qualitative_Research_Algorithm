# QRA Analysis Module Documentation

This directory contains all components for analyzing the output of the QRA classification pipeline.

## Overview

The analysis module transforms classified segments into comprehensive reports that reveal patterns in participant and therapist behavior across therapy sessions. The system generates both machine-readable data files and human-readable text/visual reports.

## Core Components

### 1. Runner (runner.py)

The entry point for all analysis operations, called by `qra analyze --output-dir <path>` or automatically when `config.auto_analyze=True` in the pipeline.

**Input**: Pipeline output directory containing:
- `master_segments.jsonl` (integrated segments with classifications)
- `theme_definitions.json` (framework definitions)

**Output**: Comprehensive reports organized by type:

#### Data Files (03_analysis_data/)
| File | Description |
|------|-------------|
| `per_session/*.json` | JSON files with detailed session-level analytics including stage proportions, exemplars, and transition matrices |
| `per_participant/*.json` | JSON files with participant-level analytics including progression trends and dominant stages |
| `per_theme/*.json` | JSON files with theme-specific analytics including prevalence by participant/session and co-occurring codes |
| `graphing/*.csv` | CSV datasets for visualization (prevalence, transitions, etc.) |
| `longitudinal_summary.json` | Aggregated summary of longitudinal trends across participants |
| `session_stage_progression.csv` | CSV showing stage progression over time per session |

#### Visualizations (04_figures/)
| File | Description |
|------|-------------|
| `stage_*_prevalence.png` | Line charts showing theme prevalence by session number |
| `group_longitudinal_trajectory.png` | Overall group trajectory trend line |
| `participant_*_trajectory.png` | Individual participant trajectories over sessions |
| `stage_transition_heatmap.png` | Heatmap of within-session transitions (from → to) |
| `cross_session_transition_heatmap.png` | Heatmap of between-session transitions |
| `purer_vammr_lift.csv` | Statistical lift values showing PURER-to-VAAMR relationships |

#### Text Reports (06_reports/)

##### Individual Session Reports (`per_session/*.txt`)
- Overview with session number and participant count
- Stage distribution with prevalence percentages and visual bars
- Dominant stage and runner-up stage identification
- Transition counts (forward/backward/lateral) with examples
- PURER therapist move distribution when applicable
- Per-participant breakdown within the session
- Best expressions for dominant stages with confidence scores

##### Comprehensive Session Report (`stage_expressions.txt`)
- Summary of all sessions in one document
- Theme presence analysis: which themes appear in all, some, or only one session
- Consistency patterns across participants and time
- Comparison of theme prevalence across the entire cohort

##### Participant Reports (`per_participant/*.txt`)
- Overview with participant ID and cohort information
- Session-by-session progression trajectory
- Overall progression trend (advancing/stable/regressing) as a slope value
- Stage distribution across all sessions combined
- Longitudinal trajectory visualization summary
- Best expression for each stage identified in the participant's data

##### Comprehensive Participant Report (`longitudinal_analysis.txt`)
- Group-level analysis of VAAMR progression trends
- Regression patterns: when and how participants regress between sessions
- Per-participant trajectories with numerical scores
- PURER × VAAMR influence analysis (when PURER labels exist):
  - Dominant PURER move per session
  - Correlation between PURER moves and participant advancement
- Phenomenology code relationships by stage
- Illustrative journey quotes showing key advances in therapy

##### Transition Analysis (`stage_transitions.txt`)
- Detailed explanation of within-session transition heatmaps
- Cross-session transition patterns
- Purer × transition correlation analysis
- Exemplar quotes for the most common transitions

##### Therapist Cue Report (`report_cue_responses.txt`)
- Aggregated summary of therapist responses between participant segments
- Average FROM/CUE/TO blocks for each transition type
- PURER move distribution per transition type
- LLM-summarized representations when content exceeds word limits

##### Session Summaries (`session_summaries.json` and `session_summaries.txt`)
- LLM-generated summaries of therapist language in each session
- Captures therapeutic intent, key interventions, and dialogue flow
- Used to enrich participant reports with context about therapist behavior

## Workflow Summary

1. **Data Loading**: Loads master_segments.jsonl and framework definitions
2. **Per-Session Analysis**: Computes stage proportions, exemplars, and transitions for each session
3. **Participant-Level Analysis**: Tracks progression trends across sessions per participant
4. **Theme-Specific Analysis**: Identifies prevalence patterns, co-occurring codes, and longitudinal trends by theme
5. **Visualization Generation**: Creates charts and heatmaps to represent patterns visually
6. **Text Report Generation**: Produces comprehensive human-readable reports with aggregated insights
7. **Longitudinal Integration**: Combines all analyses into cohesive narratives of change over time

The system is designed for iterative refinement - researchers can run `qra analyze` on existing outputs after updating framework definitions or codebook labels to regenerate reports without re-running the full classification pipeline.