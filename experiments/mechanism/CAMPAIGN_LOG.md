# Mechanism Campaign — Log

**Purpose.** Re-center the central mechanism question (H2 / methodology §7.6 — *context-dependent therapist
effects = the FROM_stage × move interaction*) on the right estimator, after the review
(`methodology_assessment.md`) found the shipped `mechanism.py:_mixed_effects_delta` fits a move **main effect
only** (`delta_prog ~ C(dominant_purer)`, Gaussian, no interaction), and the interaction is tested only by an
underpowered per-cell FDR table. Companion: `masterplan.md`.

**Apparatus.** `./data/Meta`; canonical FROM→CUE→TO triples via `process/cue_blocks.py`
(`stage_key='final_label'`); participant-grouped `StratifiedGroupKFold` (seed 42); participant-cluster
bootstrap. Reuses `experiments/gnn_reliability/harness.py` patterns + `analysis/stats.py`. In-place on the
current branch, nothing committed.

## Timeline

- **Built E1+E2** (`run_interaction_model.py`): design-frame builder (direct-file-loads `cue_blocks.py` +
  `stats.py` to bypass the `transformers`/numpy-pin import chain); earns-its-place CV; frequentist ordinal LR
  + Gaussian-mixed interaction + per-cell FDR reproduction; E-values. 186 triples / 20 participants / 160
  with a CUE move.
- **Dependency obstacle** (logged because it shapes production): installing `bambi` upgraded `numpy` to 2.x,
  which broke the pinned `transformers==4.42.4` (needs numpy<2). pymc6/pytensor3 require numpy≥2 — *cannot
  coexist* with the pipeline. **Resolution:** main `.venv` restored to numpy 1.26.4 (pipeline healthy,
  frequentist arms run there); Bayesian arm isolated to a dedicated `.venv_bayes` (numpy≥2 + bambi) consuming
  the exported `_design.csv`. **Production implication:** frequentist estimator is the in-process default;
  Bayesian is opt-in + isolated. (bambi is NOT a `requirements.txt` dependency.)
- **Built E1c** (`run_bayesian_ordinal.py`, isolated): bambi cumulative-logit + partial pooling. Sampled
  clean (4 chains, 0 divergences). Fits the 16 interaction terms the frequentist Gaussian could not (singular);
  0/16 intervals exclude 0 — honest under-identification.
- **E3–E9** corroboration delegated to a sub-agent (writes per-experiment `run_*.py` + `_e*_results.json` in
  this dir).

## Key results (see RESULTS.md for the tables)
- Cue earns its place as a PURER-move **main effect** (held-out log-loss 1.553→1.506), **not** as the
  interaction (overfits; LR p=0.52; Gaussian singular; Bayesian 0/16 HDIs exclude 0; per-cell FDR 0/20).
- PURER **process** representation beats the GNN's **content** embedding cue (which never beat FROM-only).
- E-values computable now: Avoidance×Education 4.23, AttnReg×Reinforcement 3.81.

## Promotion guidance
The estimators are promoted to production by a separate worker (P3): `src/analysis/mechanism_model.py`
(frequentist default in-process; Bayesian opt-in/isolated) + `stats.py` helpers + `mechanism.py` report
re-lead + sensitivity + `expected_codes`/permutation controls. Nothing in this campaign dir is committed.
