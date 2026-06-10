# Mechanism Campaign — Log

**Purpose.** Re-center the central mechanism question (H2 / methodology §7.6 — *context-dependent therapist
effects = the FROM_stage × move interaction*) on the right estimator, after the methodology review
(2026-06; archived in git history) found the shipped `mechanism.py:_mixed_effects_delta` fits a move **main effect
only** (`delta_prog ~ C(dominant_purer)`, Gaussian, no interaction), and the interaction is tested only by an
underpowered per-cell FDR table. Companion: `docs/ROADMAP.md` (formerly `masterplan.md`).

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

---

## P0/P1 statistical-correctness review (2026-06-10)

Closing the Phase-A gates (now `docs/ROADMAP.md`; formerly masterplan §3 / §1) that stand between the wired hierarchical mechanism
estimator and promotion to the DEFAULT mechanism analysis. Reviewed `mechanism_model.py`,
`stats.py`, `mechanism.py`, `results_brief.py`, `stat_format.py`, `config.py`. End-to-end smoke on the
real `data/MMORE_Processed` qra.db (186 triples / 20 participants / 160 with a cue move) — READ-ONLY,
outputs to a temp dir. numpy stayed 1.26.4 throughout; nothing pip-installed; nothing committed.

### Gate 1 — E-value derivation (`smd_to_risk_ratio` / `sensitivity_bounds`)  ✅
- **Checked.** The continuous→SMD→`RR≈exp(0.91·SMD)`→E-value chain and the per-cell comparison group.
- **Was correct, under-documented.** The `0.91` constant and the implicit exposed/unexposed contrast
  were not spelled out.
- **Changed.** Documented in `smd_to_risk_ratio` that `0.91 ≈ π/(2√3)` (Chinn 2000 logistic↔normal
  scaling `logOR = d·π/√3`, then VanderWeele & Ding 2017 `logRR ≈ 0.5·logOR`), and stated the
  comparison group precisely: 'exposed' = cue block got THIS PURER move; 'unexposed' = it got some
  OTHER move at the SAME (held-fixed) FROM-stage. The point E-value AND the CI-limit E-value (E-value
  of the SMD-CI bound nearer the null; =1.0 when the CI spans 0) were already computed
  (`e_value_ci_limit`) and surfaced in `mechanism.txt` + `mechanism_sensitivity_evalues.csv` (additive
  columns `smd_ci_lo/hi`, `e_value_ci_limit`). Verified on real data: Avoidance×Education point
  **E=4.23, CI-limit E=1.71**; most other cells' CI-limit E-values collapse to **1.0** (honest
  under-identification). (Audit notes referenced ≈4.87/2.35; my values follow the principled
  derivation above and differ slightly — trusting the verified math.)
- **Tests.** `test_stats.TestEValueChain` (hand-computed RR + E + CI-limit, null-side-limit selection,
  floor ≤ point); pre-existing `test_mechanism_model.TestEValueCiLimit/TestSensitivityBounds`.

### Gate 2 — Ordinal-LR inference (`ordered_logit` LR p)  ✅
- **Checked.** The in-sample LR p built on `OrderedModel.llf` is NOT participant-cluster-robust
  (statsmodels exposes no cluster-robust covariance) — anti-conservative for nested data.
- **Was the real gap.** Only the naive p existed.
- **Changed.** Added `stats.cluster_bootstrap_lr_test` (no new deps): resample WHOLE participants with
  replacement, refit reduced+full ordered logits each resample, build the bootstrap ΔLR distribution,
  one-sided add-one-smoothed p. Degrades to `cluster_robust_p_unavailable` (NOT the naive p in disguise)
  when <20 refits succeed / <2 clusters / group column absent. Wired into
  `mechanism_model.fit_adjacency_interaction` (`lr_cluster_bootstrap_n_boot`, default 500; 0 disables);
  `ordinal_lr` now carries BOTH p's with explicit labels and rides them into
  `mechanism_interaction_cv.csv` as additive columns. Report (`mechanism.txt` lead (a) and
  `results_brief` §4.2/§6) presents BOTH, defensively worded. Real data: in-sample **p=.522**,
  cluster-bootstrap **p=.934** (n=500 refits) — the clustering correctly makes it MORE conservative.
- **Tests.** `test_stats.TestClusterBootstrapLR` (plumbing, single-cluster + missing-column degrade);
  `test_mechanism_model.test_cluster_bootstrap_lr_p_populated_when_enabled`.

### Gate 3 — Singular-fit handling + `rosenbaum_bounds` + `within_between_split`  ✅
- **`mixedlm_interaction` singular flag** was already returned; the REPORT didn't act on it.
  **Changed** `mechanism._append_interaction_lead` (b) to SUPPRESS the misleading "k/N CIs exclude 0"
  count and flag SINGULAR/under-identified when the fit is degenerate (≥1 non-finite interaction CI).
- **`rosenbaum_bounds`** — checked Γ grid, one-sided-in-majority-direction, ties handling. **Correct:**
  exact zeros are dropped (ties for a sign test), `binom.sf(successes-1,n,p₊)=P(X≥successes)` is right,
  Γ=1.0 on a balanced/unconfounded-NS set, capped at `max_gamma`. No bug; documented via tests.
- **`within_between_split`** — verified the Mundlak math: between = unweighted per-row group mean,
  within = deviation; within sums to 0 per group; within+between reconstructs x; input not mutated.
  **Correct** (the textbook Mundlak device; the per-row group-mean IS the correctly-entered between
  component). No bug; documented via tests.
- **Tests.** `test_stats.TestSingularFitFlag` (degenerate→singular True; clean 2×2→False),
  `TestRosenbaumBounds` (grid/flat/ties/weak-majority/small-n), `TestWithinBetweenSplit` (4 identities).

### Gate 4 — Production posture  ✅
- **Confirmed.** `grep` over `src/` finds the ONLY `import bambi`/`import arviz` inside
  `_bayesian_ordinal`, reachable only when `estimator in ('bayesian','both')` (default `'frequentist'`)
  — no numpy≥2 dependency on the default in-process path. `mechanism.enabled=True` is default-on in
  `config.py` (`MechanismModelConfig` / `PipelineConfig.mechanism`). **Recorded as intentional/shipped**
  in the `MechanismModelConfig` docstring (Gate 4 note). Bayesian arm stays isolated to `.venv_bayes`.

### Exit criterion — curriculum logic leads with the estimate + E-value bounds  ✅
- `mechanism.txt` lead now presents the stage-moderated verdict (earns-its-place CV + BOTH LR p's +
  honest under-identification statement + singular-aware (b)) BEFORE the per-cell FDR table; the cell
  table carries E-value point + CI-limit columns.
- `results_brief.py` §4.2 leads the mechanism section with the interaction-model verdict (CV + naive &
  cluster-bootstrap LR p + under-identification); §4.3 reports point + CI-limit E-values; §6 opens with
  a PRIMARY-ESTIMATOR VERDICT banner and each cell-naming recommendation cites its point + CI-limit
  E-value bounds. All numbers via `stat_format`.

**Verification.** `tests.unit.test_stats` (36), `test_stats_more`, `test_mechanism`, `test_mechanism_more`,
`test_mechanism_model`, `test_results_brief` (25) — all green (149 + 36 + 25). Real-data smoke read +
rendered the new lead with both E-value columns and both p-values at sane values.

**Sign-off:** Gate 1 ✅ · Gate 2 ✅ · Gate 3 ✅ · Gate 4 ✅ — Phase-A statistics signed off; the
estimator is the default mechanism analysis. Runtime note: the cluster-bootstrap adds the 500-refit cost
to the production analysis path; `lr_cluster_bootstrap_n_boot` tunes/disables it (tests run with 0–40).
