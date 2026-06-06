# QRA Analysis Module Documentation

This directory contains all components for analyzing the output of the QRA classification pipeline.

## Overview

The analysis module transforms classified segments into comprehensive reports that reveal patterns in participant and therapist behavior across therapy sessions. It produces both machine-readable data files and human-readable text/visual reports. The module is organized into five analytical layers, each building on the previous:

1. **Superposition** — enriches every segment with a stage-mixture vector and continuous progression coordinate
2. **Statistical foundation** — reusable inference toolkit for all downstream analyses
3. **Mechanism dossier** — how PURER moves drive VAAMR stage progression (with full statistical inference)
4. **Efficacy dossier** — does the program work? (internal + external outcome linkage)
5. **GNN layer** — independent graph-geometry substrate for measurement triangulation and emergent-construct discovery

---

## Core Components

### Runner (`runner.py`)

The entry point for all analysis operations, called by `qra analyze --output-dir <path>` or automatically when `config.auto_analyze=True` in the pipeline.

**Input**: Pipeline output directory containing:
- `master_segments.jsonl` (integrated segments with classifications)
- `theme_definitions.json` (framework definitions)

**Analysis sequence:**
1. Load segments and assemble corpus DataFrame
2. Attach superposition (`analysis/superposition.py`)
3. Generate per-session analyses
4. Generate per-participant longitudinal analyses
5. Generate per-theme analyses
6. Compute PURER × VAAMR lift table (`purer_analysis.py`)
7. Soft cross-validation lift (VCE + PURER mixture-weighted)
8. Generate superposition report and figures
9. Run mechanism analysis (`mechanism.py`)
10. Run efficacy analysis (`efficacy.py`)
11. Generate language atlas
12. Generate figures (superposition, mechanism, PURER heatmaps)
13. Generate all text reports
14. Run GNN layer (`gnn_layer/runner.py`) if enabled

---

## Analysis Layer 1: Stage Superposition (`superposition.py`)

Enriches every participant segment with:
- `mixture` — 5-vector probability distribution over VAAMR stages
- `progression_coord` — continuous E[stage] = Σk·p_k ∈ [0,4]
- `mixture_entropy` — normalized Shannon H ∈ [0,1] (liminality)
- `max_stage` / `second_stage` — top-two stage indices
- `n_active_stages` — number of stages with mixture > threshold
- `is_liminal` — boolean (high entropy or small gap between top stages)
- `mixture_source` — provenance: `gnn`, `llm_ballots`, or `secondary_stage`

**Source priority:** GNN `segment_positions.csv` (when GNN is enabled) → LLM multi-run ballot distributions → secondary_stage two-point mixture.

Also computes:
- `stage_cooccurrence_matrix(df, n_stages)` — symmetric n×n count matrix of co-active stage pairs
- `mixture_entropy(mixture, n_stages)` — standalone entropy computation

**Outputs:** `segment_superposition.csv` (via `figure_data.py`), `report_superposition.txt`

---

## Analysis Layer 2: Statistical Foundation (`stats.py`)

Reusable inference toolkit used by mechanism.py, efficacy.py, and purer_analysis.py. All functions accept plain Python lists/arrays and return dicts.

| Function | Purpose |
|----------|---------|
| `wilson_ci(k, n, alpha)` | Proportion confidence interval |
| `cluster_bootstrap_ci(values, clusters, statistic, n_boot, alpha)` | Resample whole participants → CI (respects nesting) |
| `permutation_test(values, group_mask, strata, statistic, n_perm)` | Within-stratum label shuffle → two-sided p |
| `cohens_h(p1, p2)` | Effect size for proportion difference |
| `cramers_v(contingency)` | Effect size for contingency table |
| `odds_ratio_ci(a, b, c, d)` | Odds ratio with CI |
| `cliffs_delta(x, y)` | Non-parametric effect size |
| `benjamini_hochberg(pvals, alpha)` | FDR correction → reject mask + q-values |
| `mixedlm_trend(df, outcome, time, group)` | Trajectory slope ≠ 0 with random participant intercept |
| `mixedlm_delta(df, outcome, fixed, group)` | Fixed-effect Δprog model with random intercept |
| `sign_test(n_positive, n_total)` | Exact binomial sign test |

Requires `statsmodels` for mixed-effects functions; falls back to OLS with a warning.

---

## Analysis Layer 3: Mechanism Dossier (`mechanism.py`)

Answers *how* PURER therapist moves drive VAAMR stage progression, with every estimate inferential.

### Key Functions

- `run_mechanism_analysis(df, df_all, output_dir, framework)` — top-level orchestrator
- `_enrich_blocks(blocks, lookup, block_motifs)` — adds `delta_prog`, `delta_direction` (forward/stabilize/regress with ±0.15 deadband), `participant_id`
- `_agg_delta(blocks, framework, key_field, grouping, min_n)` — aggregates Δprogression with: cluster-bootstrap CI, within-stratum permutation p, Cramér's V effect size, BH-FDR flag, n_progress/n_stabilize/n_regress counts
- `_mixed_effects_delta(enriched)` — fits `Δprog ~ C(dominant_purer) + (1|participant)` via statsmodels
- `_liminality_leverage(blocks)` — bins blocks by FROM entropy, tests leverage hypothesis
- `_avoidance_barrier(blocks, framework)` — Avoidance-barrier transition ranking
- `_cusp_density_by_session(df)` — Avoidance↔AttnReg cusp density longitudinal
- `_trajectory_typology(df, framework)` — rule-based participant trajectory types

### Outputs

| File | Description |
|------|-------------|
| `03_analysis_data/mechanism/mechanism_delta_progression.csv` | Δprogression by PURER/from-stage with CI/p/effect/FDR columns |
| `03_analysis_data/mechanism/mechanism_liminality.csv` | Liminality leverage analysis |
| `03_analysis_data/mechanism/mechanism_avoidance_barrier.csv` | Avoidance-barrier transition ranking |
| `03_analysis_data/mechanism/avoidance_cusp_density_by_session.csv` | Cusp density longitudinal |
| `03_analysis_data/mechanism/participant_trajectory_types.csv` | Trajectory typology per participant |
| `03_analysis_data/mechanism/mechanism_purer_mixed_effects.csv` | Mixed-effects model output |
| `06_reports/report_mechanism.txt` | PURER→VAAMR mechanism dossier |
| `06_reports/report_avoidance_barrier.txt` | Avoidance-barrier ranking report |

---

## Analysis Layer 4: Efficacy Dossier (`efficacy.py`)

Answers "does the program work?" with uncertainty quantification.

### Functions

- `load_external_outcomes(output_dir, config)` — auto-detects wide pre/post or long per-session CSV at `02_meta/outcomes.csv`; graceful if absent
- `compute_participant_session_outcomes(df, config)` — per (participant, session): progression coordinate, adaptive occupancy, barrier crossing, dwell time
- `compute_barrier_crossing(df, config)` — first-passage session per participant
- `compute_group_trajectory(ps_outcomes, outcome)` — group mean with cluster-bootstrap CI
- `compute_participant_slopes(ps_outcomes, outcome)` — per-participant OLS slopes
- `link_to_external(participant_slopes, outcomes)` — Pearson correlation with bootstrap CI
- `run_efficacy_analysis(df, framework, output_dir, config)` — orchestrator

### Outputs

| File | Description |
|------|-------------|
| `03_analysis_data/efficacy/participant_session_outcomes.csv` | Per-(participant, session) outcomes |
| `03_analysis_data/efficacy/barrier_crossing.csv` | Avoidance→AttnReg first-passage per participant |
| `03_analysis_data/efficacy/group_trajectory.csv` | Group mean with bootstrap CI |
| `03_analysis_data/efficacy/participant_slopes.csv` | Per-participant progression slopes |
| `06_reports/report_program_efficacy.txt` | 5-section efficacy dossier |
| `05_figures/program_efficacy.png` | Multipanel: group CI band, adaptive occupancy, barrier crossing, VAAMR-vs-outcome scatter |

---

## Figures (`figures.py`)

In addition to existing figures, the following have been added:

### Superposition Figures (`generate_superposition_figures`)
- `session_mixture_timeline_<session>.png` — per-session stacked area chart of stage mixture over time
- `participant_progression_trajectory_<participant>.png` — continuous progression coordinate across sessions
- `superposition_entropy_<session>.png` — mixture entropy by session (liminality over time)
- `stage_cooccurrence_matrix.png` — heatmap of co-active stage pairs across corpus

### Mechanism Figures (`generate_mechanism_figures`)
- `cue_motif_delta_progression.png` — Δprogression by PURER move × from-stage (FDR-significant cells starred)
- `gnn_vs_llm_convergence.png` — GNN-lift vs LLM-lift scatterplot by construct

---

## Reports Package (`reports/`)

| Module | Function | Output |
|--------|----------|--------|
| `superposition_report.py` | `generate_superposition_report` | `report_superposition.txt` |
| `language_atlas.py` | `generate_language_atlas` | `report_language_atlas.txt` |
| `session_report.py` | `generate_comprehensive_session_report` | Per-session JSON |
| `stage_report.py` | `generate_stage_text_report` | Per-stage text reports |
| `transition_report.py` | `generate_transition_explanation`, `generate_therapist_cues_report` | `stage_transitions.txt`, `report_cue_responses.txt` |
| `longitudinal_report.py` | `generate_longitudinal_text_report` | `longitudinal_analysis.txt` |
| `session_txt_report.py` | `generate_all_session_txt_reports` | Per-session `.txt` reports |
| `participant_txt_report.py` | `generate_all_participant_txt_reports` | Per-participant `.txt` reports |
| `session_summaries.py` | `generate_session_summaries` | `session_summaries.json` |

### Language Atlas (`reports/language_atlas.py`)

The atlas produces a curriculum-actionable guide to the key language patterns driving VAAMR stage progression. For each top-ranked (FDR-significant) PURER move, emergent GNN motif, and named coupling factor, it renders `FROM participant quote → therapist CUE text → TO participant quote` blocks with stage-mixture annotations and effect sizes.

Emergent motifs (high GNN influence + low PURER purity) are highlighted as candidates for new therapeutic constructs requiring human review. Alliance-named coupling factors (Capability E) appear if the GNN layer was run.

---

## PURER Analysis (`purer_analysis.py`)

Conditional lift table with omnibus Cramér's V + chi-square association test per transition type. The marginal lift table (collapsed across from_stage) is explicitly labelled "confounded" with explanation; the from-stage-conditioned table is the defensible headline. CI/p/effect columns are present in `purer_vaamr_lift.csv`.

---

## Configuration

All new analysis modules read from `PipelineConfig` sub-configs:
- `SuperpositionConfig` — mixture source, liminality thresholds, run_mechanism_analysis flag
- `EfficacyConfig` — outcomes path, adaptive/maladaptive stages, barrier stage indices
- `GnnLayerConfig` — all GNN hyperparameters and capability flags (see `gnn_layer/config.py`)

See `USAGE.md` for full sub-config reference tables.

---

## Data Files (03_analysis_data/)

| File | Description |
|------|-------------|
| `per_session/*.json` | JSON with stage proportions, superposition summary, exemplars, transition matrices |
| `per_participant/*.json` | JSON with progression trends, continuous_progression_by_session, volatility, entropy |
| `per_theme/*.json` | Theme-specific analytics with boundary-expressed codes |
| `graphing/*.csv` | Graph-ready datasets |
| `graphing/segment_superposition.csv` | Per-segment mixture + entropy export |
| `longitudinal_summary.json` | Aggregated longitudinal trends |
| `session_stage_progression.csv` | Stage progression per session |
| `mechanism/*.csv` | Mechanism dossier CSVs with CI/p/effect/FDR columns |
| `efficacy/*.csv` | Efficacy outcome CSVs |
| `gnn/*.csv` | GNN capability outputs (requires `gnn_layer.enabled=True`) |

---

## Workflow Summary

1. **Data Loading**: Loads `master_segments.jsonl` and framework definitions
2. **Superposition**: Attaches mixture vectors, entropy, and liminality flags to all segments
3. **Per-Session Analysis**: Stage proportions, exemplars, transitions, superposition summaries
4. **Participant-Level Analysis**: Continuous progression trajectories, volatility, trajectory typology
5. **Theme-Specific Analysis**: Prevalence patterns, co-occurring codes, boundary-expressed codes
6. **PURER × VAAMR Lift**: Conditional transition lift with association tests
7. **Mechanism Dossier**: Δprogression by PURER/from-stage with full statistical inference
8. **Efficacy Dossier**: Group trajectory, barrier crossing, external outcome linkage
9. **Language Atlas**: Ranked FROM→CUE→TO exemplar blocks by PURER/motif/factor
10. **Visualization Generation**: Superposition, mechanism, efficacy, and PURER heatmap figures
11. **Text Report Generation**: Comprehensive human-readable reports
12. **GNN Layer** (optional): Segment positioning, cue motif discovery, triangulation, coupling

The system supports iterative refinement — researchers can run `qra analyze` on existing outputs after updating framework definitions to regenerate reports without re-running the full classification pipeline.
