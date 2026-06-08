# Code-Review Request — Mechanism Estimator Re-centering

> **⚠️ SUPERSEDED IN PART (2026-06-07).** The §8.2 construct-validity rows in §5 (the `cross_validation.py` `compare_expected_codes` / `shuffled_stage_permutation_control` / `export_construct_validity_checks` additions, the `expected_codes`/`expected_transitions` schema + framework-markdown entries) and the P1 "verify the stage→code mappings" gate are **obsolete**: that VAAMR×VCE construct-validity work was **deferred to future research and removed**. The mechanism re-centering work described here is retained. Current framing: `docs/methodology.md` §3.3, §5.2, §8.2.

**To:** Wade (reviewer) · **From:** Claude (implementing agent) · **Date:** 2026-06-07
**Branch:** current working branch (in-place; **nothing committed** — review the working tree, then you promote)
**Companion docs:** `methodology_assessment.md` (the review), `masterplan.md` (the plan), `experiments/mechanism/RESULTS.md` (the evidence), `docs/methodology.md` (manuscript edits)

> **TL;DR.** I found that QRA's central scientific claim (H2/§7.6 — therapist effects *moderated by FROM
> stage*) is an **interaction**, but the shipped estimator fit a move **main effect only**. I built the
> correct estimator (a hierarchical ordinal **interaction** model + a formal confound-sensitivity analysis),
> validated it in an isolated experiment campaign, and wired it into production. **Full unit suite is green
> (3351 passed); the production estimator reproduces the experiment numbers exactly.** This request explains
> everything and lists, in priority order, **what must be reviewed before this goes to production** — the
> statistical correctness of the new estimators and one identification compromise are the load-bearing items.

---

## 1. The mission I was given

Evaluate, end-to-end, whether QRA built the right thing for its real research goal — *associate therapist
language patterns with participant developmental progression (and ultimately therapeutic benefit) in a
defensible, mechanistic, causal-if-possible way that survives peer review* — and, if there was a
methodological gap, write a plan to fix it and **implement the production-ready analysis**. Benefit-linkage
(H4) needs REDCap outcomes (not yet imported), so the binding question was: *with the present Cohorts 1–2
corpus (n≈32, single-arm, observational), what is the strongest defensible thing we can build?*

## 2. What I saw (the finding that drove everything)

Reading the manuscript, the existing self-review (`experiments/docs/method_application_review.md`), the two
settled experiment campaigns, and the primary instruments in code:

- **The classifier/scaler/discovery layer was built right and is honest.** H5 (graph-as-scaler) correctly
  refuted; the distillation ceiling is *data, not method*; **H6 (VAAMR is developmental, not topical) is a
  genuine, novel contribution.** No changes needed there.
- **The gap is the central mechanism question (H2 / §7.6).** The whole claim is an **interaction** — a
  therapist move's effect on the next participant stage is *moderated by the FROM stage*. But the shipped
  primary estimator `src/analysis/mechanism.py:_mixed_effects_delta` fits
  `delta_prog ~ C(dominant_purer)` — a move **main effect only, Gaussian, no `FROM_stage × move` interaction
  term.** The interaction was tested *only* by an additive per-cell table under Benjamini–Hochberg FDR over
  ~20 sparse cells (≈zero power → 0 significant). The GNN transition MLP built to carry mechanism doesn't
  earn its place. **The most important question was served with the wrong instrument**, and the
  causal-defensibility apparatus a methods venue expects (explicit identifying assumptions + a quantitative
  confound-sensitivity bound) was named in prose (§9.4) but not provided.

## 3. What I tried (the experiment campaign — `experiments/mechanism/`, not production)

Isolated campaign, reusing the validated harness (`experiments/gnn_reliability/harness.py` patterns +
`analysis/stats.py`), on `./data/Meta` (186 FROM→CUE→TO triples, 20 participants, 160 with a defined cue
move). Participant-grouped `StratifiedGroupKFold` (seed 42), participant-cluster bootstrap. **Nothing
committed; the researcher promotes.**

- **E1** — the interaction model three ways: earns-its-place CV (FROM-only vs +move vs FROM×move),
  frequentist (ordinal `OrderedModel` LR + Gaussian mixed interaction), and **Bayesian** cumulative-logit +
  partial pooling.
- **E2** — confound sensitivity (E-values).
- **E3–E9** — corroboration (cue representation, trajectory/within-between, PURER-noise robustness, lift
  permutation control, transition-CF honesty, H1 test, H6 robustness).

**One real obstacle, logged because it shapes production:** installing the Bayesian stack
(`bambi`/`pymc`/`pytensor`) forces `numpy>=2.0`, which **breaks the pipeline's pinned `transformers==4.42.4`
(needs `numpy<2.0`).** They cannot coexist. I restored the main `.venv` to `numpy==1.26.4` (pipeline
healthy), ran the frequentist arms there, and ran the **Bayesian arm in a dedicated, git-ignored
`.venv_bayes`** consuming the exported design frame. **This is now the production posture** (see §5).

## 4. What I found

**Right instruments, honestly under-identified at n≈32** — the masterplan's predicted signature:

- The cue earns its place as a PURER-move **main effect** (held-out log-loss FROM-only **1.553** → +move
  **1.51**, acc 0.28→0.37) — *better than the GNN's content-embedding cue, which never beat FROM-only*.
- The **`FROM × move` interaction does NOT earn its place** (out-of-sample 1.53, 25 params; ordinal LR
  **p=0.52**; the Gaussian interaction design is **singular**; per-cell FDR **0/20**).
- The **Bayesian hierarchical model fits the interaction** (16 terms, finite credible intervals, 0
  divergences) *where the frequentist MLE is singular*, and honestly reports **0/16 intervals exclude 0**.
- **E-values** (the bounded, non-causal statement): Avoidance×Education **4.23**, AttnReg×Reinforcement
  **3.81**, Metacog×Education **3.49**.
- **Two directional positives:** **E3** (process cue ≫ content cue, Δ −0.36) and **E9** (H1 group slope
  **+0.097, CI [0.007, 0.197] excludes 0**; barrier 17/20 cross; Mann–Kendall NS — honest).
- **Two honesty flags (consequential for what we can claim):** **E5** — the per-move ranking is *fragile*
  under PURER label noise (ρ=0.30) and **no PURER human-IRR exists yet**; **E6** — the **H6 contrast
  sign-flips on MiniLM** (plausibly a capacity/"No-code"-clustering artifact), so **H6 is defensible on
  Qwen but encoder-generality is unconfirmed**.

## 5. What I changed in the production codebase (and why)

> Implemented by a sub-agent under my direction, then independently verified by me. **Backward-compatible:
> with `mechanism.enabled=False` (or stats libs absent) the output is byte-identical to today.**

| File | Δ | Purpose |
|---|---|---|
| **`src/analysis/mechanism_model.py`** | **NEW (618)** | The re-centered estimator. `MechanismModelConfig` (default `estimator='frequentist'`); `fit_adjacency_interaction` (ordinal LR additive-vs-interaction; Gaussian-mixed `delta_prog ~ C(from)*C(move)+(1|participant)` with singular-safe CI counting; participant-grouped earns-its-place CV; opt-in lazy-bambi Bayesian arm that degrades gracefully); `sensitivity_bounds` (E-value + Rosenbaum Γ per cell); `purer_noise_robustness`; `fit_trajectory` (within/between Mundlak split); `run_mechanism_models` orchestrator → CSVs in `03_analysis_data/mechanism/`. |
| **`src/analysis/stats.py`** | +221 | New helpers (all lazy + graceful): `ordered_logit`, `likelihood_ratio_test`, `mixedlm_interaction`, `e_value`, `smd_to_risk_ratio`, `rosenbaum_bounds`, `within_between_split`. |
| **`src/analysis/mechanism.py`** | +148 / −6 | `run_mechanism_analysis(..., config=None)` calls `run_mechanism_models`; the report (`02_mechanism/mechanism.txt`) now **LEADS** with the interaction estimate (LR test, mixed-interaction CIs, earns-its-place CV) + the **E-value/Γ sensitivity table** + an **identifying-assumption statement**; the additive per-cell table is demoted to a labeled "DESCRIPTIVE COMPANION"; the GNN counterfactual is labeled "sensitivity lens, not the estimator". |
| **`src/analysis/runner.py`** | +2 / −1 | Passes `config.mechanism` into `run_mechanism_analysis`. |
| **`src/process/cross_validation.py`** | +171 | `compare_expected_codes` (predicted_and_confirmed/absent/unpredicted_but_elevated), `shuffled_stage_permutation_control` (real-vs-permuted lift), `export_construct_validity_checks` — the §8.2 *must-before-Cohort-3* controls. |
| **`src/process/orchestrator.py`** | +17 | Wires the two construct-validity checks (additive, guarded) → `construct_validity_checks.json`. |
| **`src/theme_framework/theme_schema.py`** | +9 | `expected_codes` / `expected_transitions` fields on `ThemeDefinition` (default empty). |
| **`src/theme_framework/markdown_loader.py`** | +22 | Parses optional `Expected Codes:` / `Expected Transitions:` lines (absent → empty; backward-compatible). |
| **`frameworks/VAAMR_FRAMEWORK.md`** | +15 | `Expected Codes:` per stage — the §3.3 predictions encoded as **pre-registration metadata** (stage→VCE code_ids). |
| **`frameworks/PURER_FRAMEWORK.md`** | +15 | `Expected Transitions:` per move (forward-target VAAMR stage). |
| **`src/process/config.py`** | +18 | `PipelineConfig.mechanism = MechanismModelConfig` (lazy default factory; serialized + reconstructed in `from_json`). |
| **`requirements.txt`** | +11 | **Commented** optional-Bayesian note (the numpy conflict; `bambi` is intentionally **NOT** a live dependency). |
| **`tests/unit/test_mechanism_model.py`** | **NEW (421)** | Hermetic: planted-interaction recovery vs null; e_value monotonicity/≥1; permutation control flags a planted co-dependency; expected_codes classification; **backward-compat** (disabled ⇒ legacy report unchanged); config round-trip. |

Also: manuscript edits in `docs/methodology.md` (§9.4 identifying assumption + E-value/Rosenbaum; §7.6/§3.4/§8.5
primary estimator; H1/H6 honest results; VanderWeele & Ding 2017 ref), and the new root docs
`methodology_assessment.md` + `masterplan.md`.

## 6. Verification I performed

- **Full unit suite GREEN — independently re-run:** `3351 passed / 10 skipped / 0 failures / 0 errors`
  (baseline 3321; the new test file adds the delta). Coexists cleanly with a concurrent agent's edits.
- **Reproduction guard:** `run_mechanism_models` on `./data/Meta` reproduces the experiment **exactly** —
  earns-its-place 1.553/1.514/1.528, ordinal **LR p=0.522**, E-values **4.23 / 3.81 / 3.49**.
- **Report contract:** `mechanism.txt` opens with `PRIMARY ESTIMATOR — STAGE-MODERATED THERAPIST EFFECT
  (FROM_stage × move)`, then earns-its-place, then the E-value sensitivity + identifying assumption; the
  additive table appears as "DESCRIPTIVE COMPANION"; the GNN counterfactual as "sensitivity lens".
- **Backward-compat:** `test_disabled_preserves_legacy_report` asserts the estimator-off path is unchanged.

---

## 7. ⚠️ WHAT MUST BE REVIEWED BEFORE PRODUCTION (priority order)

**P0 — statistical correctness (load-bearing; this is the science):**
1. **The E-value derivation in `sensitivity_bounds` / `stats.smd_to_risk_ratio`.** I convert a *continuous*
   Δprogression cell effect to a standardized mean difference, then `RR ≈ exp(0.91 · SMD)` (Chinn/VanderWeele),
   then the E-value. Review: (a) is the SMD→RR approximation appropriate for this outcome, (b) is the
   comparison group (same-FROM-stage *other moves*) the right contrast, (c) should the E-value be on the
   point estimate *and* the CI limit (the masterplan asks for the CI-limit E-value too). **This is the
   number a methods reviewer will scrutinize most.**
2. **The identification compromise (Agent's noted deviation):** I implemented `ordered_logit` (plain
   `OrderedModel`) **not** a participant-cluster-robust `ordered_logit_clustered` — statsmodels' `OrderedModel`
   doesn't expose cluster-robust covariance cleanly. So the **in-sample ordinal LR p-value does not account
   for within-participant correlation**; the leakage-free inference is carried by the *participant-grouped
   earns-its-place CV* instead. **Confirm this is acceptable**, or decide we need a cluster-bootstrap LR or a
   GEE/Bayesian alternative before the ordinal p-value is reported as inference.
3. **The Gaussian `mixedlm_interaction` singular-handling.** At n≈32 the `C(from)*C(move)` design is
   rank-deficient. Review how `mixedlm_interaction` detects/reports singular fits and counts "CI-excludes-0"
   interaction terms — it must not silently emit a misleading count.
4. **`rosenbaum_bounds`** correctness (Γ at which the rank test loses significance) and `within_between_split`
   (the Mundlak decomposition) — both new, both feed reported numbers.

**P1 — domain correctness of the pre-registration content:**
5. **`frameworks/VAAMR_FRAMEWORK.md` `Expected Codes:`** — I mapped the methodology §3.3 (stage → VCE code)
   table to valid VCE `code_id`s; the table's "Attention" row has **no VCE code** and was annotated/omitted.
   **A domain expert must verify these stage→code mappings** before they drive the Cohort-3 confirmatory
   comparison (this is pre-registration — it must be right *before* the data).
6. **`frameworks/PURER_FRAMEWORK.md` `Expected Transitions:`** — verify the move→forward-target-stage
   predictions match the §7.6 clinical hypotheses.

**P2 — production posture & wiring:**
7. **Dependency/isolation decision:** frequentist default **in-process**, Bayesian **opt-in + isolated
   `.venv_bayes`** (because `pytensor` needs numpy≥2, incompatible with `transformers`'s numpy<2 pin). Confirm
   this is the right call vs. the alternative of upgrading `transformers` to a numpy≥2-compatible version
   (larger blast radius). The Bayesian arm is **not exercised in-process** — only the isolated experiment ran it.
8. **`config.mechanism.enabled=True` default-on:** it adds report sections + writes CSVs to
   `03_analysis_data/mechanism/`. Confirm default-on is desired; confirm the disabled path is truly inert
   (test asserts it, but eyeball the guards in `mechanism.py`).
9. **`from_json` round-trip** for the new `mechanism` config block on a legacy `qra_config.json` that lacks
   the key (defaults must fill cleanly — the suite passes, but confirm with a real project config).

**P3 — findings the researcher must gate on (not code, but promotion-blocking):**
10. **Do not promote any therapist-*effect* claim until PURER is human-validated** (E5: ranking fragile; no
    PURER IRR exists yet).
11. **Claim H6 on Qwen embeddings only** until the faithful two-encoder `gnn_layer/discriminant.py` test runs
    (E6: the contrast sign-flipped on MiniLM; the Qwen embedding endpoint failed to load during the proxy run).

## 8. Out of scope / explicitly not done

- **PURER human validation** (needs the rater team) — gates the dyadic mechanism story.
- **The faithful two-encoder H6 test** — the Qwen embedding endpoint (`http://10.0.0.58:1234`) failed to
  load; needs a working endpoint or a local Qwen embed.
- **REDCap / H4 convergent validity** (the bridge to therapeutic *benefit*) — data pending; the scaffolding
  (`efficacy.link_to_external`) is ready and the masterplan says **pre-register the outcome directions now**.
- **`ordered_logit_clustered`** — not feasible cleanly in statsmodels; see P0-2.
- **Cohorts 3–4 confirmatory replication** — scale-gated; the `expected_codes` pre-registration makes it
  confirmatory.
- **No commits, no branch changes, no worktrees** — a second agent was concurrently consolidating
  `src/experiments → experiments/` and editing docs on this branch; my changes are additive and the suite is
  green with all of it. (The `.gitignore`/`CLAUDE.md`/some `docs/*` working-tree changes are that agent's,
  not mine.)

## 9. How to review (commands)

```bash
# 1. Full unit suite (must stay green)
.venv/bin/python tests/run_unit_tests.py

# 2. Reproduction guard — production estimator on real data should print
#    E-values 4.23/3.81/3.49 and ordinal LR p=0.522 (see experiments/mechanism/RESULTS.md for the targets)
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); import experiments... "   # or re-run:
.venv/bin/python experiments/mechanism/run_interaction_model.py        # E1a/E1b + E2 (frequentist)
.venv_bayes/bin/python experiments/mechanism/run_bayesian_ordinal.py   # E1c (isolated Bayesian)

# 3. Inspect the new report contract end-to-end
qra analyze -o ./data/Meta     # then read 06_reports/02_mechanism/mechanism.txt — it must LEAD with the
                               # interaction estimator + E-value sensitivity + identifying assumption

# 4. Read order for the science: methodology_assessment.md → masterplan.md → experiments/mechanism/RESULTS.md
#    → the docs/methodology.md diffs (§7.6, §9.4, §3.4 H1/H2/H6, §8.5)
```

**Bottom line for promotion:** the wiring is green, backward-compatible, and reproduces the experiment. The
gate to production is **P0 (statistical correctness of the E-value/ordinal-LR/singular-handling) + P1 (the
expected_codes domain mappings)**. P3 (PURER validation, H6 encoder-generality) gates the *claims*, not the
*code*. Once P0/P1 are signed off, this can ship as the default mechanism estimator; the Bayesian arm stays
isolated until the dependency posture is decided (P2-7).

---

## 10. ADDENDUM — review fixes applied (2026-06-07, post-review)

Three P0 statistical-correctness findings from the review were implemented in-place (still uncommitted;
**unit suite green**: the mechanism module went 27 → **35 passed**; the broader
mechanism/stats/cross-validation/config slice is **510 passed / 1 skipped / 0 failures**).

| Finding | Fix | Where |
|---|---|---|
| **P0-1 — point E-value only; no CI-limit E-value; cells selected post-hoc by \|SMD\| ⇒ point E-values optimistic** | Added a participant-cluster bootstrap 95% CI for each cell SMD (`_smd_cluster_ci`, resamples whole participants → respects nesting) and a new `stats.e_value_ci_limit(smd_lo, smd_hi)` = E-value of the CI bound nearest the null (=1.0 when the CI spans 0, per VanderWeele & Ding 2017). Cells now carry `smd_ci_lo/hi`, `e_value` (point) **and** `e_value_ci_limit` (the honest floor). Sort is now CI-limit E-value desc, then \|SMD\| as tiebreak (degrades to the old \|SMD\| order when every CI spans 0, which is the n≈32 reality). | `stats.py:e_value_ci_limit`; `mechanism_model.py:_pooled_smd`, `_smd_cluster_ci`, `sensitivity_bounds` |
| **P0-2 — in-sample ordinal LR p-value is not cluster-robust, reported in the lead without that caveat** | `ordered_logit` docstring now states `OrderedModel` exposes no cluster-robust covariance and the LR is anti-conservative for nested data; the report's (a) line now prints an explicit "in-sample, NOT cluster-robust — read it as descriptive; leakage-free inference is the participant-grouped CV in (c)" note. (No change to the estimator math — the grouped CV was already the leakage-free arm.) | `stats.py:ordered_logit` docstring; `mechanism.py:_append_interaction_lead` |
| **P0-3 — Rosenbaum Γ ran a one-sample sign test on the cell's raw deltas (a different null than the E-value)** | Γ is now computed on the **contrast** — cell deltas centered by the same-stage other-move mean — so Γ and the E-value bound the *same* "this move vs alternatives at this stage" association. | `mechanism_model.py:sensitivity_bounds` |

**Behavioral impact on the pilot read.** The point E-values are unchanged (Avoidance×Education ≈ 4.2,
etc.), but the new **CI-limit E-values collapse to ~1.0 at n≈32** (the SMD bootstraps span 0) — i.e. the
honest robustness floor shows the pilot can *bound* but not *establish* robustness, exactly the
under-identified status the rest of the writeup claims. `docs/methodology.md` §9.4 was updated to report
both E-values, the CI-limit collapse, and the ordinal-LR descriptive caveat. The
`mechanism_sensitivity_evalues.csv` export gains the `smd_ci_lo/hi` + `e_value_ci_limit` columns
automatically (it serializes the cell dicts).

**New tests** (`tests/unit/test_mechanism_model.py`): `TestEValueCiLimit` (spans-zero⇒1.0; same-sign uses
the null-side limit; CI-limit ≤ point; non-finite⇒NaN) and three additions to `TestSensitivityBounds`
(columns present + bounded; strong planted cell's CI excludes the null; **null data ⇒ all CI-limit
E-values collapse to 1.0**).

**Still open (unchanged by these fixes):** P1 (the `expected_codes`/`expected_transitions` domain mappings
need your sign-off), P2-7 (Bayesian dependency posture), P3 (PURER human validation; H6 encoder-generality).
The Gaussian-singular handling (original P0-3 item #3) was reviewed and judged correct/conservative — no
change needed.
