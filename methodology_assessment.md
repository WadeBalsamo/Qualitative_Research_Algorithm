# Methodology Assessment: Did QRA Build the Right Thing for the Mechanism Question?

> **⚠️ SUPERSEDED IN PART (2026-06-07).** This memo's §1/R6/R7/Q4/Q5/Q30 discussion of the §8.2 `expected_codes` pre-registration and the shuffled-stage permutation control (hypotheses **H3/H3a**) is superseded: that VAAMR×VCE **construct-validity** work was **deferred to future research and removed** from code and manuscript. The mechanism findings (the memo's substance) stand; the VCE classifier is retained as optional enrichment. Current framing: `docs/methodology.md` §3.3, §5.2, §8.2.

**An independent methodology-review memo.** *Date: 2026-06-07.* *Reviewer role: the methodology-review
agent charged in `experiments/docs/method_application_review.md` Part VIII-B.* *Corpus under review:
Move-MORE Cohorts 1–2 (`./data/Meta/`), n≈32 participants, 339 participant segments (205 VAAMR-labeled,
134 "No code"), 66–76 human IRR items.*

> **What this is.** A response to the self-review's explicit charge — *review the design against the
> stated intentions, adjudicate instrument choice, verify the verifiable, simulate peer review, and return
> a single recommendation: is the current GNN methodology the right vehicle or should it be re-scoped?* It
> is keyed to the falsifiable hypotheses H1–H6 / §7.6 of `docs/methodology.md` and to the Appendix-B
> question IDs of the self-review. It does **not** re-review code correctness (covered). Everything below is
> **observational, n≈32, hypothesis-generating**; the multi-run LLM consensus (κ=0.537 vs human) remains
> the label of record.

---

## 0. The one-paragraph verdict

**We built the right *discovery and construct-validation* layer, and we built the wrong *primary estimator*
for the one question the whole program is about.** The classifier/scaler work is settled, honest, and
correct (H5 refuted; the distillation ceiling is *data*, not method; H6 is a genuine, novel, N-robust
contribution). But the program's central scientific claim — H2 / §7.6, that a therapist move's effect on
the next participant stage is **moderated by the participant's FROM stage** — is an *interaction*, and **no
instrument in the codebase estimates that interaction with the right tool.** The additive per-cell table
tests it as ~20 independent comparisons under FDR (≈zero power at this n); the one formal mixed-effects
model omits the interaction term entirely; the GNN transition MLP that was built to carry mechanism does not
earn its place at n≈32. The fix is not more GNN — it is a **hierarchical Bayesian ordinal interaction
model** (partial pooling is the *correct* small-n tool, not a nicety), a **participant-level trajectory
model** (closer to "benefit"), and a **formal confound-sensitivity argument** (E-value / Rosenbaum) that
§9.4 currently gestures at in prose but does not provide. With those three additions the methodology is
defensible, novel, and peer-review-ready *as an honest under-identified-but-correctly-instrumented pilot*;
without them, a skeptical methods reviewer rejects the mechanism claim. The GNN's defensible footprint
shrinks to **H6 (construct validation) + lead-generation.**

### 0.1 Verified results from the mechanism campaign (run this session)

I did not just argue this — I built the estimators and ran them on `./data/Meta` (186 FROM→CUE→TO triples,
20 participants, 160 with a defined CUE move; campaign in `experiments/mechanism/`). The numbers confirm the
thesis *and* surface one optimistic finding:

- **The cue earns its place as a PURER-move *main effect*, but not as the *interaction*.** Under
  participant-grouped CV (multinomial-logistic, held-out multiclass log-loss; lower is better):
  FROM-only **1.553** → +move (additive) **1.506** (acc 0.28→0.37) → FROM×move (interaction) **1.531**
  (25 params, overfits). So the therapist move carries genuine held-out signal as a main effect — *notably
  better than the GNN transition model's content-embedding cue, which never beat FROM-only* (direct support
  for representing the cue by its PURER *process* construct, not a content embedding — Q16/E3).
- **The §7.6 interaction is under-identified — confirmed with the right instruments.** Frequentist ordinal
  LR additive-vs-interaction **p = 0.52**; the Gaussian FROM×move mixed design is **literally singular** at
  n≈32; the shipped per-cell FDR table reproduces at **0/20 significant**.
- **The Bayesian hierarchical ordinal model is the tool that can even *fit* this.** With weakly-informative
  priors + partial pooling it estimates all **16 interaction terms** (finite credible intervals, **0
  divergences**) *where the frequentist MLE is singular* — and honestly reports **0/16 interaction intervals
  exclude 0**: estimable-but-under-identified, not un-fittable and not over-claimed. *This is precisely the
  argument for the Bayesian estimator at small n.* (Run in an isolated `.venv_bayes` — see the dependency
  note below.)
- **Confound sensitivity is real and computable now.** Per-cell E-values (VanderWeele–Ding): the strongest
  associations are **Avoidance×Education** (SMD −0.96, **E-value 4.23**) and **AttnReg×Reinforcement** (SMD
  +0.86, **E-value 3.81**) — i.e. an unmeasured confounder would need an association of RR≈4.2 with *both*
  move-selection and Δprogression to explain the Avoidance×Education cell away. That is the defensible,
  bounded, non-causal statement the manuscript currently lacks.

**A consequential engineering finding for production:** the Bayesian stack (bambi/pymc/pytensor) requires
`numpy>=2.0`, but the pipeline's pinned `transformers==4.42.4` requires `numpy<2.0` — **they cannot coexist
in one environment.** The production recommendation therefore is: the **in-process default mechanism
estimator is the frequentist ordinal + Gaussian-mixed interaction model** (statsmodels/sklearn, numpy-
agnostic), and the **Bayesian estimator is an opt-in that runs in an isolated environment** (it degrades
gracefully when its libraries are absent). This is reflected in `masterplan.md` §3/§5.

---

## 1. Implementation → intention → question, verified against the code

For each hypothesis: *what it claims*, *what the code actually does* (verified, with file:line), the
**status**, and the **gap**. The self-review's Part IV is the starting map; this section verifies it and
adds the one finding the self-review did not state at the code level.

### H1 — Developmental progression (the VA→MR arc; avoidance barrier rate-limiting)
- **Code:** `src/analysis/efficacy.py` — `compute_group_trajectory` (efficacy.py:145) computes the group
  mean progression coordinate per session *with a participant-cluster bootstrap CI*; `compute_participant_slopes`
  (efficacy.py:165) gives per-participant OLS slopes; `compute_barrier_crossing` (efficacy.py:116) computes
  per-participant Avoidance barrier crossings; `stats.mixedlm_trend` provides the random-participant slope.
- **Status: now tested (E9), supported but underpowered.** The slope *and its CI* exist; the descriptive
  barrier exists; **and this session's E9 assembled the tested claim:** group slope **+0.097, participant-
  cluster bootstrap CI [0.007, 0.197]** (excludes zero; random-participant +0.107, p=0.037), and the barrier
  is descriptively rate-limiting — **17/20 (85%)** cross Avoidance→Attention-Regulation (median first-crossing
  session 2), endpoint progression **2.74** for crossers vs **0.86** for non-crossers.
- **Gap (now narrow):** the ordinal-safe Mann–Kendall monotonicity trend is *not* significant (τ=0.5,
  p=0.108) — reported honestly — and n=20 is small. So H1's progression and the barrier-as-rate-limiting are
  *supported by a tested CI/contrast*, but monotonicity and confirmatory power await Cohorts 3–4. *(Was
  "asserted more than tested"; E9 tested it.)*

### H2 / §7.6 — Context-dependent therapist effects (THE central question) — **the code-level finding**
- **What it claims:** the effect of a PURER move on the next participant stage **depends on the FROM
  stage** — "the same move helps more at some stages than others" (§7.6). This is, formally, a
  **FROM_stage × move interaction** on an ordinal outcome.
- **What the code does — three instruments, none of which estimates that interaction correctly:**
  1. **Additive per-cell table** — `mechanism.py:_agg_delta` (mechanism.py:151) buckets blocks by
     `(from_stage, behaviour)`, computes each cell's mean Δprogression with a participant-cluster bootstrap
     CI, a *within-from-stage* permutation p (this move vs other moves at the same FROM stage), and
     Benjamini–Hochberg FDR (mechanism.py:206–222). This *does* condition on FROM stage — but it tests each
     of ~20 cells **independently, with no pooling**, then applies FDR. At n≈32 the result is **0
     FDR-significant cells**: the design has essentially no power to detect an interaction this way.
  2. **The one formal model — and here is the finding the self-review did not state at the code level:**
     `mechanism.py:_mixed_effects_delta` (mechanism.py:386) calls
     `S.mixedlm_delta(mdf, outcome='delta_prog', fixed='C(dominant_purer)', group='participant_id')`
     (stats.py:370). The fixed effect is **`C(dominant_purer)` — the move *main effect only*.** There is
     **no `C(from_stage):C(dominant_purer)` interaction term**, and the outcome is treated as **Gaussian**,
     not ordinal. *The program's inferential backbone literally does not contain the moderation term that
     is the program's central hypothesis.* The interaction is therefore never estimated as a single
     parameter with a confidence/credible interval — only as the underpowered per-cell FDR table above.
  3. **GNN transition MLP** — `gnn_layer/transition.py` (`_make_net`, transition.py:159: a
     `Linear→ReLU→Dropout→Linear` net). Its participant-grouped earns-its-place check shows the cue **does
     not improve** held-out TO prediction over a FROM-only baseline (ΔKL +0.371, worse), so it is
     under-identified; its learned counterfactual triangulates *positively* with the observed ranking
     (Spearman ρ ≈ +0.34) but with 0 FDR-significant observed cells the convergence is under-powered.
- **Status: under-identified AND under-instrumented.** Under-identified is honest and partly unavoidable at
  n≈32. *Under-instrumented is the fixable gap:* the actual interaction hypothesis has never been fit with
  a model that (a) includes the interaction, (b) respects the ordinal outcome, and (c) borrows strength
  across sparse cells via partial pooling — which is exactly what a small-n interaction needs.
- **Gap (the headline of this memo):** **replace the mechanism lead with a hierarchical Bayesian ordinal
  interaction model** (§2). Even if it too reports the interaction credible intervals spanning zero, that
  is the *defensible* result; the current additive-FDR table is the wrong test, and the Gaussian
  main-effect-only mixed model answers a different question than H2 asks.

### H3 / H3a — Cross-framework construct validity (VAAMR × VCE lift; does VCE sharpen the arc?)
- **Code:** `src/process/cross_validation.py` computes the lift table; `gnn_layer/classifier/ablation.py:
  vce_vaamr_contribution` runs the held-out Δκ-with/without-VCE test.
- **Status: instruments exist; the controls do not.** The lift table is computed, but the
  **shuffled-stage permutation control** that §5.2/§8.2 commit to (the test against classifier
  co-dependency) **is not yet run**, and the `expected_codes` pre-specification that would make the test
  falsifiable-from-code is **not yet populated** (`theme_schema.py` has no such field). H3a's Δκ is
  currently measured on the *graph* classifier, which §3.4 itself flags as a low-power base.
- **Gap:** run the permutation control + populate `expected_codes` (both **runnable now**, computationally
  trivial; E7 / Q4, Q30); re-instrument H3a on the LLM/probe rather than the weak graph (E7 / Q5). These
  are the §8.2 *must-before-Cohort-3* items.

### H4 — Convergent validity vs clinical outcomes (the bridge to "benefit")
- **Code:** `efficacy.py:link_to_external` (efficacy.py:189) merges participant progression
  (slope/change/endpoint) with external change scores and reports Spearman with cluster-bootstrap CIs;
  `load_external_outcomes` (efficacy.py:34) reads `02_meta/outcomes.csv`.
- **Status: fully scaffolded, awaiting data.** The instrument is built and correct; it is inert until
  REDCap outcomes are exported to the `outcomes.csv` contract.
- **Gap:** **pre-register the correlation directions now** (per `docs/OUTCOME_INTEGRATION_ROADMAP.md`) so
  the linkage is confirmatory when data lands (Q6, REDCap-gated). This is the only path to the program's
  ultimate goal — *language → therapeutic benefit* — and it is convergent-validity, not efficacy (single
  arm).

### H5 — Distillation / scalability (graph reproduces consensus → LLM-free scaling)
- **Code:** `gnn_layer/classifier/validation.py` (the reliability gate), now **default-OFF**
  (`gnn_classifier_enabled=False`).
- **Status: settled and correctly closed.** Refuted at n≈32 under leakage-free participant-grouped CV
  (grouped κ ≈ 0.05–0.14); the distillation campaign (`experiments/classification_scaler/`) then proved the
  ceiling is *data* — three independent methods converge on classifier↔LLM κ≈0.36 / classifier↔human
  κ≈0.45, with the best being a per-rater ensemble (assistive, gated, not autonomous). **This was done
  right.** The only open thread is whether a learned scaler can earn the role at Cohorts 3–4 N (Q7,
  scale-gated).

### H6 — Discriminant validity (VAAMR is developmental, not topical) — **the strongest contribution**
- **Code:** `gnn_layer/discriminant.py` (746 lines, default-ON at `qra analyze`): on identical
  participant-grouped folds and the *same* Qwen embeddings, a supervised probe (human κ 0.365 [0.228,
  0.513]) is contrasted against a content-similarity Correct-&-Smooth model (human κ 0.196 [0.117, 0.319])
  and chance, with a paired Δκ (human +0.170 [0.002, 0.318], LLM +0.214 [0.150, 0.274], CIs exclude 0),
  plus geometry (content-PC recoverability, local kNN homophily, community×stage ARI ≈ 0).
- **Status: strong, N-robust, two-sided, falsifiable.** This is the rare claim that does not depend on n —
  it is a *contrast*. It converts the H5 negative into a positive construct-validity result and is, in my
  judgment, the **headline methodological finding** of the whole GNN effort.
- **Gap (generality — now an active flag, E6):** a lightweight robustness probe this session found the
  probe-vs-content-similarity Δκ **sign-flips on MiniLM-384** (5-class −0.048, 6-class −0.091 with CI
  excluding 0) — *opposite* the shipped Qwen result (probe ≫ content, +0.17/+0.21). This is **not a
  refutation** but an unresolved generality question: the flip is plausibly a *capacity* artifact (MiniLM
  carries less linearly-separable stage signal, so a nonparametric kNN beats a linear probe) compounded by
  the 6-class "No code" being strongly content-clustered (kNN homophily 0.30→0.36 — which *also answers
  Q27/Q28: the abstention class loads the content model*). **Consequence for the headline:** H6's
  *validity* on Qwen stands, but its *embedding-generality is unconfirmed* — the faithful test is the actual
  `gnn_layer/discriminant.py` probe-vs-Correct-&-Smooth instrument re-run on Qwen **and** a second encoder,
  which did **not** run because the Qwen embedding endpoint failed to load ("Operation canceled"). Until
  that runs, the manuscript should claim H6 *on Qwen embeddings*, not encoder-independently. Replication at
  Cohorts 3–4 (Q9) is scale-gated.

### §7.6 dyad — fully labeled, but on un-validated cue labels
- **Code:** `purer_analysis.py` (PURER × VAAMR cue-response) + `gnn_layer/communities.py` (dyadic routines).
- **Status: structurally complete, validity-pending.** Every FROM→CUE→TO triple is labeled; routines are
  computed. **But PURER human-validation (Krippendorff α ≥ 0.70) is pending (§8.1).**
- **Gap (the quiet load-bearing risk):** *the entire dyadic-mechanism story rests on therapist cue labels
  we have not yet validated.* Until PURER is validated, every therapist-effect claim is doubly provisional
  (under-identified *and* on unverified labels). A **PURER-label-noise robustness check** (perturb at the
  measured single-rater disagreement rate, re-fit, re-rank) is **runnable now** (E5 / Q17) and should gate
  every mechanism claim.

---

## 2. Instrument-choice adjudication — what should lead each job

The single most important framing for review (the self-review's Part III is right): **the GNN is three
instruments doing three jobs that are not equally defensible.** I adjudicate each and name the lead.

| Job | Current lead | My verdict | What should lead |
|---|---|---|---|
| **Construct validation** (is VAAMR developmental, not topical?) | `discriminant.py` (H6) | **Right tool.** A probe-vs-similarity *contrast* is exactly correct and N-robust. | **Keep — and make it the headline.** Harden across embeddings + 5/6-class. |
| **Mechanism** (does therapist language move participants, conditional on FROM stage?) | additive FDR table (lead) + GNN MLP (sensitivity) | **Wrong tool.** The interaction is the hypothesis; FDR-over-cells has no power and the formal model omits the interaction; the MLP doesn't earn its place. | **Hierarchical Bayesian ordinal interaction model** (`TO_stage ~ FROM_stage * move + (1|participant)`, cumulative-logit, weak priors → partial pooling). Frequentist `OrderedModel` + interaction-extended `mixedlm` as robustness. The MLP becomes an explicitly-labeled triangulation lens. |
| **Trajectory / consolidation** (does therapist language predict *learning*, not just the next utterance?) | — (not modeled) | **Missing.** Progression has a between-session consolidation component the adjacent triple ignores — and consolidation is *closer to benefit*. | **Add** a participant-level ordinal model with lagged cues + a within-vs-between-session split. Novelty lever and the natural bridge to H4. |
| **Discovery / leads** (recurring routines, motifs, coupling) | `communities.py`, `motifs.py`, `coupling.py` | **Right tool, narrow claim.** H6 says these clusters are *content*, so routines are content-adjacency leads, not developmental mechanism. | **Keep as human-filtered lead-generation only**; never primary evidence. |

**Why partial pooling is not optional at n≈32.** With ~20 (FROM × move) cells and a handful of
participants carrying the rare stages, independent per-cell estimates are noise and FDR rightly kills them
all — but that is a *property of the estimator*, not evidence of no effect. A hierarchical model with weak
priors shrinks unstable cells toward the grand mean and lets cells with more support speak, yielding a
**single interaction estimate with an honest interval**. This is the standard, expected small-n tool and
the one a methods reviewer will ask for by name. The Bayesian version additionally gives the *posterior
probability of direction* for each §7.6 directional prediction — the exact quantity §7.6 is about.

**Why the GNN MLP cannot be the mechanism estimator (independent of n).** Three structural mis-fits,
already diagnosed in `GNN_MASTER_PLAN.md` §4.7 and confirmed here: the cue reaches the outcome through a
content-similarity-shaped function (H6 says similarity ≠ stage), the net is not interpretable as a moderated
effect (no interaction coefficient to report), and it does not earn its place on held-out prediction. It is
a fine *sensitivity lens* (does a flexible learned response function agree in rank with the classical
estimate?) and should be labeled exactly that.

---

## 3. Peer-review simulation — the objections a methods reviewer will raise

Reading `docs/methodology.md` §3.4 / §7.6 / §8.5 / §9.4 cold, as a skeptical reviewer for a mixed-methods
methodology venue. For each: does the current text answer it, and the minimal addition.

| # | Likely objection | Answered now? | Minimal addition |
|---|---|---|---|
| R1 | "Your central hypothesis is an interaction, but you test it with per-cell FDR and a main-effects-only mixed model. Fit the interaction." | **No** | The hierarchical ordinal interaction model (§2). *This is the decisive one.* |
| R2 | "You name a confound (responsiveness) but don't bound it. How strong would it have to be to explain your associations?" | **No** (mapped, not bounded) | E-value / Rosenbaum Γ on the per-cell associations, reported beside the divergence map (§9.4). |
| R3 | "State the identifying assumption and show it violated." | **Partially** (prose) | A §9.4 paragraph + DAG: sequential ignorability given (FROM_mixture, FROM_stage, context), and why move-selection-on-difficulty violates it. |
| R4 | "Your therapist labels (PURER) aren't validated, yet the mechanism rests on them." | **Disclosed, not discharged** | Gate mechanism claims on PURER α≥0.70 *or* the label-noise robustness result (§E5); say which at each claim. |
| R5 | "Why a GNN/MLP for mechanism when a hierarchical model is standard and interpretable?" | **No** | Reposition: hierarchical model leads; MLP is a triangulation lens (§2, §8.5 rewrite). |
| R6 | "Lift co-occurrence between two LLM classifiers is shared-lexicon, not construct validity." | **Committed, not run** | Run the shuffled-stage permutation control (§8.2, E7); report real-vs-permuted lift. |
| R7 | "Your predictions are stated in prose, not pre-registered." | **Committed, not done** | Populate `expected_codes`/expected-transitions before Cohort 3; emit the mechanical comparison. |
| R8 | "Adjacency ≠ mechanism; and the next utterance ≠ therapeutic change." | **§9.2 yes; consolidation no** | Add the trajectory/within-vs-between model (§2-B) so 'learning' is distinguished from 'momentary nudge'. |
| R9 | "Is the GNN's large manuscript footprint proportionate to its evidence?" | **No** | Foreground H6 + the classical mechanism; scope §8.5 GNN tighter (discovery + H6). |

**Decision:** *as written, the mechanism sections would not pass a rigorous methods review* — R1, R2, R3
are load-bearing and unanswered. **With** the hierarchical interaction model + the sensitivity/identifying
argument + the §8.2 controls + the PURER gating, the package becomes defensible, and notably *novel*: I am
not aware of psychotherapy process research that estimates **stage-moderated therapist effects at
within-turn temporal adjacency with a hierarchical ordinal model and a formal confound-sensitivity bound**.
That combination — bilateral computational phenomenology + moderated-mechanism estimation + bounded
non-causal inference + validation-gated scaling + a pre-registered four-cohort replication architecture —
is the defensible novelty, and it is mostly *already here*; the missing piece is the right estimator and the
sensitivity argument.

---

## 4. What we can do *now*, with the present dataset (pre-REDCap)

The researcher's question — *we can't link to benefit until REDCap; what can we do now?* — has a concrete,
defensible answer. Everything here is runnable on `./data/Meta` today:

1. **The H6 construct-validity headline, hardened** (embeddings × 5/6-class). *Strongest, N-robust claim.*
2. **The re-centered mechanism estimate** — hierarchical ordinal interaction model + frequentist
   robustness, reporting interaction credible intervals and per-direction posterior probabilities for the
   §7.6 predictions. *Honest result expected: "instrument correct, effect under-identified at n≈32,
   bounded by sensitivity analysis."* That is publishable and defensible.
3. **The confound-sensitivity argument** — E-values / Rosenbaum bounds + the explicit identifying
   assumption, beside the existing divergence map. *Converts "we acknowledge a confound" into "the confound
   would have to be Γ this strong."*
4. **The trajectory / consolidation read** — within- vs between-session split; "does this therapist
   language precede *consolidated* advancement, not just the next utterance?" *Closest pre-REDCap proxy for
   benefit.*
5. **The §8.2 must-before-Cohort-3 items** — permutation-controlled lift (H3) + `expected_codes`
   pre-registration + the avoidance-barrier rate-limiting test (H1). *Turns Cohorts 3–4 into confirmatory
   replication.*
6. **The PURER-noise robustness result** — bounds how much the un-validated cue labels could be moving the
   mechanism ranking, pending the human validation.
7. **The dyadic descriptive structure** — PURER × VAAMR FROM→CUE→TO distributions (descriptive, gated on
   PURER validation before becoming primary).

What we **cannot** do now and should not claim: anything linking language to *clinical benefit* (H4 needs
REDCap), and any *confirmatory* mechanism claim (needs Cohorts 3–4 power). Pre-registering both now is the
move that makes them confirmatory later.

---

## 5. Adjudication of the self-review's open questions (Appendix B)

For each Appendix-B question: my disposition — **VERIFIED** (confirmed against code here), **SPEC'D**
(exact check defined, runnable now, scheduled as E1–E9), or **GATED** (needs Cohorts 3–4 / REDCap / human
PURER). This converts the self-review's "OPEN" register into an actionable disposition.

| Q | Disposition | Note |
|---|---|---|
| Q1 (H1 slope/rate-limiting) | SPEC'D → E9 | slope CI exists in `efficacy.py`; assemble the rate-limiting test |
| Q2 (any real cell) | GATED | needs Cohorts 3–4 power |
| Q3 (§7.6 per-cell directions) | GATED (needs Q30) | becomes posterior-prob-of-direction under the Bayesian model |
| Q4 (permutation lift) | SPEC'D → E7 | trivial; §8.2 |
| Q5 (H3a off-graph) | SPEC'D → E7 | re-instrument on probe/LLM |
| Q6 (H4 outcomes) | GATED (REDCap) | pre-register directions now |
| Q7 (scaler at higher N) | GATED | re-open at Cohorts 3–4 |
| Q8 (embedding robustness) | SPEC'D → E6 | ≥2 encoders |
| Q9 (H6 replication) | GATED | Cohorts 3–4 |
| Q10/Q17 (PURER validity/noise) | SPEC'D → E5 (noise) + human (validity) | gate mechanism on it |
| Q11 (routines: developmental?) | SPEC'D → part of E4/E6 | vs content-co-occurrence null |
| Q12 (geometry prose matches) | VERIFIED (numbers in discriminant.py outputs) / SPEC'D re-check | |
| **Q13 (hierarchical vs MLP)** | **SPEC'D → E1 — the decisive experiment** | and the code finding in §1 makes it sharper: even the classical model omits the interaction |
| Q14 (primary-vs-secondary framing) | SPEC'D (methodology.md edit) | |
| Q15 (nonlinear finds interaction additive misses) | SPEC'D → E1 | |
| Q16 (cue representation) | SPEC'D → E3 | |
| Q18/Q19 (trajectory; within/between) | SPEC'D → E4 | |
| Q20 (E-value/Rosenbaum) | SPEC'D → E2 | |
| Q21 (quasi-experimental lever) | GATED/SPEC'D | enumerate now; power at scale |
| Q22 (identifying assumption) | SPEC'D (methodology.md §9.4) | |
| Q23 (logistics contamination) | SPEC'D → E6-adjacent | re-run communities filtered |
| Q24 (routines beyond co-occurrence) | SPEC'D → E4 | null comparison |
| Q25/Q26 (training-uncertainty CIs; thin support) | SPEC'D → E8 | fixes misleadingly-tight CIs |
| Q27/Q28 (5 vs 6 class) | SPEC'D → E6 | |
| Q29 (GNN paper footprint) | ADJUDICATED — re-scope (this memo) | foreground H6 + classical mechanism |
| Q30 (pre-registration in code) | SPEC'D → E7/P3 | `expected_codes` |

---

## 6. The single recommendation (the charge's deliverable)

> **The current GNN methodology is the right vehicle for *construct validation* (H6) and *lead-generation*,
> and the wrong vehicle for *mechanism* (H2/§7.6). Re-scope it to those two roles, and re-center the
> mechanism question on a hierarchical Bayesian ordinal interaction model with a formal confound-sensitivity
> argument and a trajectory/consolidation companion.** The next experiment to fund is **not more GNN**; it
> is **E1 — the hierarchical ordinal interaction model — paired with E2 (E-value/Rosenbaum sensitivity)**,
> because together they convert the program's central claim from "tested with the wrong instrument and
> found null" into "tested with the right instrument, honestly bounded, under-identified at n≈32, and
> pre-registered for confirmatory replication at Cohorts 3–4." That is what passes peer review, and it is
> the strongest honest thing the present dataset can support on the road to linking therapist language to
> therapeutic benefit.

The production design that implements this recommendation — the estimator spec, the experiments, the
sequencing, and the MoveMORE analysis product — is in `masterplan.md`.
