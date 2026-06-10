# Mechanism Campaign — Results

> **Question.** Does therapist language move participants through the VAAMR arc, and is that effect
> *moderated by the participant's FROM stage* (methodology §7.6 / H2 — an **interaction**)? The shipped
> estimator (`analysis/mechanism.py:_mixed_effects_delta` → `delta_prog ~ C(dominant_purer)`) fits a move
> **main effect only**, Gaussian, no interaction. This campaign fits the interaction with the right tools
> and asks whether it earns its place at n≈32.
>
> **Apparatus.** Move-MORE Cohorts 1–2 (`./data/Meta`), 186 FROM→CUE→TO triples, 20 participants, 160 with a
> defined CUE move. Canonical cue blocks (`process/cue_blocks.py`, `stage_key='final_label'`). Outcome =
> next participant VAAMR stage (ordinal 0–4) and signed Δprogression-coordinate. Participant-grouped
> `StratifiedGroupKFold` (seed 42); participant-cluster bootstrap CIs. **Observational, n≈32 —
> hypothesis-generating, never causal.**

---

## Headline

| Arm | Finding |
|---|---|
| **E1a earns-its-place** | The cue earns its place as a PURER-move **main effect** (held-out log-loss FROM-only **1.553** → +move **1.506**, acc 0.28→0.37) — but the **FROM×move interaction does NOT** (1.531, 25 params, overfits). |
| **E1b frequentist** | Ordinal LR additive-vs-interaction **p=0.52**; Gaussian FROM×move mixed design is **SINGULAR**; shipped per-cell FDR **0/20**. |
| **E1c Bayesian** | Cumulative-logit + partial pooling **fits all 16 interaction terms** (finite credible intervals, **0 divergences**) *where the frequentist MLE is singular*; **0/16 intervals exclude 0** — honest under-identification. |
| **E2 sensitivity** | Per-cell **E-values**: Avoidance×Education **4.23**, AttnReg×Reinforcement **3.81**, Metacog×Education **3.49** — a confounder would need an RR this large with *both* move-selection and Δprogression to explain the cell away. |

**Verdict.** The mechanism instruments are now *correct*. The §7.6 interaction is **under-identified at
n≈32** (not a positive effect), but it is now **estimated with honest bounds** (Bayesian) and a **formal
confound-sensitivity floor** (E-values), rather than tested with an underpowered FDR table, a singular
Gaussian fit, or a neural net that doesn't earn its place. The PURER-move *main effect* carries real
held-out signal — notably more than the GNN transition model's content-embedding cue, which never beat
FROM-only — confirming the cue should be represented by its **process** construct (PURER), not a content
embedding. Confirmatory power awaits Cohorts 3–4; the pre-registration (§8.2) makes that confirmatory.

---

## E1 — the interaction model (`run_interaction_model.py` + isolated `run_bayesian_ordinal.py`)

### E1a — earns-its-place (participant-grouped CV, held-out multiclass log-loss; lower = better)
| model | log-loss | acc | params | Δ vs FROM-only |
|---|---|---|---|---|
| FROM-only | 1.553 ± 0.10 | 0.277 | 5 | — |
| + move (additive) | **1.506 ± 0.14** | 0.367 | 9 | **−0.047 (better)** |
| FROM × move (interaction) | 1.531 ± 0.15 | 0.375 | 25 | −0.023 (better than FROM, **worse than additive**) |

The cue (as a PURER move) improves held-out prediction as a **main effect**; the **interaction overfits** and
generalizes worse than the additive model. Contrast: the GNN transition MLP (content-embedding cue) did
*not* beat FROM-only at all — evidence the **process** representation (PURER) beats the **content** one (Q16).

### E1b — frequentist inference
- **Ordinal `OrderedModel` LR test** (additive vs interaction): LR=15.0, df=16, **p=0.52** — interaction adds
  nothing in-sample.
- **Gaussian mixed** `delta_prog ~ C(from_stage)*C(move) + (1|participant)`: **singular matrix** — the
  interaction design is rank-deficient at n≈32 (cannot be fit by MLE). *This is the motivation for E1c.*
- **Shipped per-cell FDR** (`_agg_delta` reproduction): 20 cells, **0 FDR-significant** — reproduces the
  manuscript's "0 FDR-significant cells".

### E1c — Bayesian hierarchical ordinal (isolated `.venv_bayes`; bambi cumulative-logit)
`to_stage ~ from_stage * move + (1|participant_id)`, weakly-informative priors (interaction σ=1.0), partial
pooling. **16 interaction terms estimated** (the frequentist Gaussian could not fit *any*), max divergences
**0**, **0/16 interaction 95% intervals exclude 0**. The model *fits the question* and reports it honestly
under-identified — the defensible small-n instrument.

> **Why isolated:** bambi/pymc/pytensor need `numpy>=2.0`; the pipeline's `transformers==4.42.4` needs
> `numpy<2.0`. They cannot coexist. Production therefore defaults to the frequentist estimator in-process and
> runs the Bayesian arm in a dedicated `.venv_bayes` (see `docs/ROADMAP.md`, formerly `masterplan.md` §3/§5).

## E2 — confound sensitivity (E-values, VanderWeele–Ding)
Per (FROM stage × move) cell, SMD of Δprogression vs the other moves at the same FROM stage → approx RR →
E-value. Top cells (n≥4):

| cell | n | SMD | approx RR | E-value |
|---|---|---|---|---|
| Avoidance × Education | 13 | −0.96 | 0.42 | **4.23** |
| AttnReg × Reinforcement | 4 | +0.86 | 2.19 | **3.81** |
| Metacog × Education | 14 | −0.78 | 0.49 | 3.49 |
| Reappraisal × Reframing | 4 | −0.57 | 0.60 | 2.74 |
| Vigilance × Education | 16 | −0.46 | 0.66 | 2.41 |

These are the bounded, non-causal statements the manuscript §9.4 currently lacks. (Thin-support cells, e.g.
n=4, are flagged for the in-support restriction in E8.)

---

## E3–E9 (corroboration)

Shared `_common.py` (direct-file-loads `cue_blocks.py`/`stats.py`, MiniLM embeddings, cluster-bootstrap κ);
reuse `_design.csv`. n≈20 participants / 160 cue blocks. **Two genuinely directional positives (E3, E9); two
consequential honesty flags (E5, E6).**

| E | Result | Read |
|---|---|---|
| **E3** cue representation | Held-out CV log-loss: FROM-only 1.552, **PURER-move (process) 1.506 (−0.046)**, content-embed (MiniLM cue, PCA-8) 1.862 (**+0.31, content HURTS**). Process beats content by **Δ −0.356**. | **Positive.** The cue signal is *process* (PURER), not lexical content — corroborates §7.6 + H6. The only representation that beats FROM-only is the PURER move. |
| **E4** trajectory/within-between | Lagged forward-cue → next-session stage: coef −0.28, CI[−1.70,1.14], p=0.70 (under-identified). Sequence structure that beats a within-session shuffle null = **move persistence** (Education→Education p=.001), not cross-move directional routines (bigram p=.72). | Honest: robust dyadic *routines* are not established at this n; what's real is persistence. |
| **E5** PURER-noise robustness | No measured PURER IRR exists (shipped IRR is VAAMR-only). At 30% label noise the per-move ranking (Phenomenology>Reframing>Education>Reinforcement>Utilization) is **FRAGILE**: Spearman ρ vs base = 0.30 [−0.61, 0.90], frac(ρ≥0.8)=0.13 (Utilization n=6 drives instability). | **Honesty flag.** Gate every therapist-effect claim on PURER human-validation (α≥0.70). |
| **E7** lift controls | Shuffled-stage permutation: 3 cells exceed raw p<.05 (Metacog×rage_anger lift 3.25; Vigilance×pain 5.20; Reappraisal×somatic_energy 1.41) but **0 survive BH-FDR**. | Leads only; under-powered (39 co-labeled segments) — as expected. |
| **E8** transition-CF honesty | Participant-bootstrap vs naive across-block CI: median width ratio **0.88×** (clustering does *not* dominate at these cell sizes) — the real honesty problem is **thin support** (9 thin-support-large-effect flags; e.g. Vigilance×Reframing n=2 Δ+2.09, excluded). In-support ranking led by Vigilance×Phenomenology (n=4, Δ+1.77, [0.50,3.25]). | The tight CIs aren't a clustering artifact; they're thin-support extrapolation — restrict to in-support cells. |
| **E9** H1 tested | **Group slope +0.097 cluster-bootstrap CI [0.007, 0.197] (excludes 0); random-participant +0.107, p=0.037.** Mann–Kendall τ=0.5, p=0.108 (NS — honest). **Barrier: 17/20 (85%) cross Avoidance→AttnReg, median first-crossing session 2; endpoint progression crossed 2.74 vs never 0.86.** | **Positive.** H1's progression *and* the avoidance barrier as descriptively rate-limiting are supported (flagged n=20). |
| **E6** H6 robustness | On **MiniLM-384** the probe−content-similarity Δκ **SIGN-FLIPS**: 5-class −0.048 (CI incl. 0), 6-class **−0.091 (CI excludes 0)** — opposite the shipped **Qwen** result (probe≫content, +0.17/+0.21). | **Honesty flag, not refutation.** Attributable to (a) MiniLM carrying less linearly-separable stage signal (so nonparametric kNN beats a linear probe = capacity, not topicality) and (b) the 6-class "No code" being strongly content-clustered (kNN homophily 0.30→0.36 — answers Q27/Q28: "No code" loads the *content* model). **H6's embedding-generality is UNCONFIRMED:** the faithful test is `src/gnn_layer/discriminant.py` (probe vs Correct-&-Smooth) re-run on Qwen *and* MiniLM — the Qwen endpoint (`http://10.0.0.58:1234`) failed to load at runtime ("Operation canceled"), so Q8 remains open. |

**Cross-cutting verdict.** The campaign converges on "right instruments, honestly under-identified at n≈32" —
the planning doc's predicted signature (plan since merged into `docs/ROADMAP.md`). Directional positives: **E3** (process ≫ content cue) and **E9** (group
slope CI excludes 0 + barrier rate-limiting). Honesty flags the manuscript must carry: **E5** (PURER ranking
fragile → gate on validation) and **E6** (H6 not yet shown encoder-robust — run the faithful two-encoder
`discriminant.py` test before claiming generality).

## Files
- `run_interaction_model.py` — E1a/E1b (frequentist) + E2; exports `_design.csv`, writes `_e1e2_results.json`.
- `run_bayesian_ordinal.py` — E1c Bayesian arm; run under `.venv_bayes`; writes `_e1c_bayesian_results.json`.
- `_design.csv` — the FROM→CUE→TO design frame (reused by all arms).
