# Methodology Assessment: Did QRA Build the Right Thing for the Mechanism Question?

*Updated 2026-06-08*

> **⚠️ SUPERSEDED IN PART (2026-06-07).** This memo's §1/R6/R7/Q4/Q5/Q30 discussion of the §8.2 `expected_codes` pre-registration and the shuffled-stage permutation control (hypotheses **H3/H3a**) is superseded: that VAAMR×VCE **construct-validity** work was **deferred to future research and removed** from code and manuscript. The mechanism findings (the memo's substance) stand; the VCE classifier is retained as optional enrichment. Current framing: `docs/methodology.md` §3.3, §5.2, §8.2.

**An independent methodology-review memo.** *Reviewer role: the methodology-review agent charged in
`experiments/docs/method_application_review.md` Part VIII-B.* *Corpus under review: Move-MORE Cohorts
1–2 (`./data/Meta/`), n≈32 participants, 339 participant segments (205 VAAMR-labeled, 134 "No code"),
66–76 human IRR items.*

> **What this is.** A response to the self-review's explicit charge — *review the design against the
> stated intentions, adjudicate instrument choice, verify the verifiable, simulate peer review, and
> return a single recommendation.* Keyed to the falsifiable hypotheses H1–H6 / §7.6 of
> `docs/methodology.md`. Everything below is **observational, n≈32, hypothesis-generating**; the
> multi-run LLM consensus (κ=0.537 vs human) remains the label of record.

---

## 0. Open Questions (resolve before / flag in the paper)

*These are genuinely unresolved. Each one either gates a manuscript claim or has a practical decision
attached. Resolve or explicitly flag before the Cohort-3 report.*

**OQ-A — PURER human IRR: when does blind-coding begin, and does it gate the Cohort-3 therapist
recommendations?**
The entire dyadic-mechanism story (§7.6, `purer_analysis.py`, `gnn_layer/communities.py`) rests on
therapist cue labels for which no human validation has started. The paper explicitly commits to
"Krippendorff α ≥ 0.70" before promoting the PURER mechanism from provisional to primary. That gate
has to be scheduled concretely — is the 20% blind-coding happening between Cohorts 2 and 3? If yes,
does a borderline α (say 0.55–0.69) trigger a label-noise robustness sensitivity run in lieu of full
validation? Until this is decided, every therapist-effect sentence in §7.3 and §7.6 carries an
undischarged hedge.

**OQ-B — Qwen3-8B embedder won't load under the pinned `transformers==4.42.4` — blocks two
things.**
(1) The faithful two-encoder H6 discriminant-validity test: until `gnn_layer/discriminant.py` runs on
Qwen3 embeddings *and* a second encoder with the same `discriminant.py` instrument, the claim is
encoder-generality-unconfirmed (the current segmentation-sensitivity and the probe both fall back to
MiniLM-L6-v2). (2) A clean Qwen-space segmentation-sensitivity run. Fix the embedding endpoint (pin
upgrade or swap to a loadable Qwen3 checkpoint) or accept MiniLM-space scope statements for both.
This is a concrete engineering decision, not a theoretical one.

**OQ-C — H4: run the descriptive convergent-validity correlations now, or explicitly state "no pilot
result"?**
`efficacy.py:link_to_external` is scaffolded and correct; it is inert until REDCap outcomes land in
`02_meta/outcomes.csv`. A skeptical reviewer (see §3, O6) will read the omission of any pilot result
as a gap. Two options: (a) export Cohorts 1–2 REDCap data, run Spearman progression-slope vs clinical
change scores, report descriptively; or (b) add a sentence to §8.4 explicitly stating "H4 yields no
pilot result from Cohorts 1–2 because REDCap outcomes were not imported; the correlation directions
are pre-registered for Cohort-3 confirmation." Option (b) is lower effort and fully defensible —
but it must be stated, not silently omitted.

**OQ-D — Segmentation-sensitivity non-convergence at n≈32: signal worth reporting?**
The opt-in `analysis/segmentation_sensitivity.py` check runs 6 OFAT arms; all 6 mixedlm fits are
"MLE on the boundary" (non-converged) at n≈32. The directional result — per-arm slope
+0.081…+0.093 in MiniLM space — is stable, but the CIs are not reliable. Two options: (a) document
as a directional signal in §9 with the non-convergence disclosed; (b) hold the result until Cohorts
3–4 power it. This memo's recommendation is (a), modest framing, because the result at least
establishes that boundary-placement perturbation does not flip the sign of the progression slope.
But decide explicitly before submission.

**OQ-E — The stronger re-classifying segmentation arm: build before submission?**
The current sensitivity check is partly structural (labels are overlap-projected, not re-derived).
A true robustness test would re-run the LLM classifier on each re-segmented corpus. Cost: N_arms ×
N_segments × N_classifier_runs. At n≈32 that is feasible but not trivial. If the paper claims
segmentation robustness, a reviewer may press on the projection artifact. Decide whether this is a
must-do before submission or a "future work" disclosure.

---

## 0.1 The one-paragraph verdict

**We built the right *discovery and construct-validation* layer, and we built the wrong *primary
estimator* for the one question the whole program is about.** The classifier/scaler work is settled,
honest, and correct (H5 refuted; the distillation ceiling is *data*, not method; H6 is a genuine,
novel, N-robust contribution). Two new trust instruments — LLM justification-grounding audit
(`src/analysis/reports/justification_grounding.py`, default-ON) and a segmentation-sensitivity check
(`src/analysis/segmentation_sensitivity.py`, opt-in) — were built, independently code-reviewed, fixed,
and the relevant caveats are incorporated below. The program's central scientific claim — H2 / §7.6,
that a therapist move's effect on the next participant stage is **moderated by the participant's FROM
stage** — is an *interaction*, and the Bayesian hierarchical ordinal interaction model that estimates it
correctly is now implemented (`src/analysis/mechanism_model.py`). Together with a formal confound-
sensitivity argument (E-values / Rosenbaum Γ, already computed) and a trajectory/consolidation
companion, the methodology is defensible, novel, and peer-review-ready *as an honest under-identified-
but-correctly-instrumented pilot*. The GNN's defensible footprint remains **H6 (construct validation) +
lead-generation**. A manuscript truthing pass was also applied to `docs/methodology.md`, surfacing five
must-fix items detailed in §3 that block submission.

---

## 0.2 Verified results from the mechanism campaign

I did not just argue this — I built the estimators and ran them on `./data/Meta` (186 FROM→CUE→TO
triples, 20 participants, 160 with a defined CUE move; campaign in `experiments/mechanism/`). The
numbers confirm the thesis *and* surface one optimistic finding:

- **The cue earns its place as a PURER-move *main effect*, but not as the *interaction*.** Under
  participant-grouped CV (multinomial-logistic, held-out multiclass log-loss; lower is better):
  FROM-only **1.553** → +move (additive) **1.506** (acc 0.28→0.37) → FROM×move (interaction)
  **1.531** (25 params, overfits). The therapist move carries genuine held-out signal as a main
  effect — *notably better than the GNN transition model's content-embedding cue, which never beat
  FROM-only* (direct support for representing the cue by its PURER *process* construct, not a
  content embedding).
- **The §7.6 interaction is under-identified — confirmed with the right instruments.** Frequentist
  ordinal LR additive-vs-interaction **p = 0.52**; the Gaussian FROM×move mixed design is **literally
  singular** at n≈32; the shipped per-cell FDR table reproduces at **0/20 significant**.
- **The Bayesian hierarchical ordinal model is the tool that can even *fit* this.** With weakly-
  informative priors + partial pooling it estimates all **16 interaction terms** (finite credible
  intervals, **0 divergences**) *where the frequentist MLE is singular* — and honestly reports **0/16
  interaction intervals exclude 0**: estimable-but-under-identified, not un-fittable and not
  over-claimed. This is precisely the argument for the Bayesian estimator at small n. The estimator
  lives in `src/analysis/mechanism_model.py` with the frequentist ordinal + Gaussian-mixed estimators
  as the in-process default (numpy-agnostic) and the Bayesian (bambi/pymc/pytensor, `numpy>=2.0`)
  as an isolated opt-in.
- **Confound sensitivity is real and computable.** Per-cell E-values (VanderWeele–Ding): strongest
  associations are **Avoidance×Education** (SMD −0.96, **E-value 4.23**) and
  **AttnReg×Reinforcement** (SMD +0.86, **E-value 3.81**) — i.e., an unmeasured confounder would
  need an association of RR≈4.2 with *both* move-selection and Δprogression to explain the
  Avoidance×Education cell away. That is the defensible, bounded, non-causal statement the manuscript
  currently lacks.

**A consequential engineering constraint:** the Bayesian stack (bambi/pymc/pytensor) requires
`numpy>=2.0`, but the pipeline's pinned `transformers==4.42.4` requires `numpy<2.0` — they cannot
coexist in one environment. The production recommendation: the **in-process default mechanism
estimator is the frequentist ordinal + Gaussian-mixed interaction model** (statsmodels/sklearn,
numpy-agnostic); the **Bayesian estimator is an opt-in that runs in an isolated environment** (it
degrades gracefully when its libraries are absent). This is reflected in `masterplan.md` §3/§5.

---

## 1. Implementation → intention → question, verified against the code

For each hypothesis: *what it claims*, *what the code actually does* (verified, with file:line), the
**status**, and the **gap**.

### H1 — Developmental progression (the VA→MR arc; avoidance barrier rate-limiting)

- **Code:** `src/analysis/efficacy.py` — `compute_group_trajectory` (efficacy.py:145) computes the
  group mean progression coordinate per session *with a participant-cluster bootstrap CI*;
  `compute_participant_slopes` (efficacy.py:165) gives per-participant OLS slopes;
  `compute_barrier_crossing` (efficacy.py:116) computes per-participant Avoidance barrier crossings;
  `stats.mixedlm_trend` provides the random-participant slope.
- **Status: supported but underpowered.** The slope *and its CI* exist: group slope **+0.097,
  participant-cluster bootstrap CI [0.007, 0.197]** (excludes zero; random-participant +0.107,
  p=0.037), barrier descriptively rate-limiting — **17/20 (85%)** cross Avoidance→Attention-
  Regulation (median first-crossing session 2), endpoint progression **2.74** for crossers vs
  **0.86** for non-crossers. Segmentation-sensitivity check (opt-in; `analysis/segmentation_sensitivity.py`)
  finds per-arm slope +0.081…+0.093 in MiniLM space — directionally stable, though (a) arms run in
  MiniLM not Qwen space (OQ-B), (b) labels are projected not re-classified (OQ-E), and (c) all six
  arms are mixedlm-non-converged at n≈32 (OQ-D). The honest claim is "slope direction stable to
  boundary-placement perturbation in MiniLM space."
- **Gap (narrow):** ordinal-safe Mann–Kendall monotonicity trend is not significant (τ=0.5, p=0.108);
  monotonicity and confirmatory power await Cohorts 3–4. Must-document: the segmentation-sensitivity
  scope, embedder, and non-convergence in §4.1/§9. See OQ-D.

### H2 / §7.6 — Context-dependent therapist effects (THE central question) — the code-level finding

- **What it claims:** the effect of a PURER move on the next participant stage **depends on the FROM
  stage** — "the same move helps more at some stages than others" (§7.6). This is, formally, a
  **FROM_stage × move interaction** on an ordinal outcome.
- **What the code does — now correctly instrumented:**
  1. **`src/analysis/mechanism_model.py`** (new) — hierarchical ordinal interaction model
     `TO_stage ~ FROM_stage * move + (1|participant)` (cumulative-logit, weak priors, partial
     pooling). Frequentist `OrderedModel` + interaction-extended `mixedlm` (statsmodels, default,
     numpy-agnostic) run first; Bayesian (bambi/pymc, opt-in isolated env) adds posterior probability
     of direction for each §7.6 prediction. This is the estimator that *should* lead the mechanism
     claim.
  2. **Additive per-cell table** — `mechanism.py:_agg_delta` (mechanism.py:151) + FDR
     (mechanism.py:206–222). Now correctly positioned as a descriptive supplement, not the
     inferential lead. Still reports **0 FDR-significant cells** at n≈32 — that is an honest result
     from an under-powered design, not a flaw.
  3. **GNN transition MLP** — `gnn_layer/transition.py`. Its participant-grouped earns-its-place check
     shows the cue does not improve held-out TO prediction over a FROM-only baseline (ΔKL +0.371,
     worse). Correctly repositioned as a triangulation lens, not the mechanism estimator.
- **Status: under-identified at n≈32, but now correctly instrumented.** The 0/16 Bayesian credible
  intervals excluding zero is the *honest, defensible result* — an interaction that is estimable but
  under-powered, not an interaction that was never tested. The prior version's omission (the shipped
  `mechanism.py` mixed model used `C(dominant_purer)` main effect only, no interaction, Gaussian
  outcome) is corrected.
- **Gap (not yet closed):** PURER labels still unvalidated (see §7.6 dyad entry and OQ-A); the
  mechanism ranking is provisional until human α ≥ 0.70 or a label-noise robustness result gates it.

### H3 / H3a — Cross-framework construct validity (VAAMR × VCE lift)

- **Status: deferred to future research.** The shuffled-stage permutation control and
  `expected_codes` pre-specification are not run / not populated. The instruments exist; the
  controls do not. H3/H3a are removed from the manuscript's primary claims. VCE is retained as
  optional enrichment. See `docs/methodology.md` §3.3, §5.2, §8.2.

### H4 — Convergent validity vs clinical outcomes (the bridge to "benefit")

- **Code:** `efficacy.py:link_to_external` (efficacy.py:189) merges participant progression with
  external change scores, reports Spearman with cluster-bootstrap CIs; `load_external_outcomes`
  (efficacy.py:34) reads `02_meta/outcomes.csv`.
- **Status: fully scaffolded, no pilot result.** Inert until REDCap outcomes are exported. A
  skeptical reviewer (see §3, O6) will note the conspicuous absence of any Cohort 1–2 descriptive
  result for a mixed-methods venue. See OQ-C.
- **Gap:** Pre-register the correlation directions now (per `docs/OUTCOME_INTEGRATION_ROADMAP.md`);
  add a sentence to §8.4 stating explicitly that H4 yields no pilot result and why. Do not leave
  the section absent without explanation.

### H5 — Distillation / scalability (graph reproduces consensus → LLM-free scaling)

- **Code:** `gnn_layer/classifier/validation.py` (reliability gate), **default-OFF**
  (`gnn_classifier_enabled=False`).
- **Status: settled and correctly closed.** Refuted at n≈32 under leakage-free participant-grouped
  CV (grouped κ ≈ 0.05–0.14); distillation campaign (`experiments/classification_scaler/`) proved
  the ceiling is *data* — three independent methods converge on classifier↔LLM κ≈0.36 / classifier↔
  human κ≈0.45, best being a per-rater ensemble (assistive-gated, not autonomous). Done right.
  Re-open at Cohorts 3–4 N.

### H6 — Discriminant validity (VAAMR is developmental, not topical) — the strongest contribution

- **Code:** `gnn_layer/discriminant.py` (746 lines, default-ON at `qra analyze`): on identical
  participant-grouped folds and the *same* Qwen embeddings, a supervised probe (human κ 0.365
  [0.228, 0.513]) is contrasted against a content-similarity Correct-&-Smooth model (human κ 0.196
  [0.117, 0.319]) and chance, with a paired Δκ (human +0.170 [0.002, 0.318], LLM +0.214 [0.150,
  0.274], CIs exclude 0), plus geometry (content-PC recoverability, local kNN homophily, community×
  stage ARI ≈ 0).
- **Status: strong, N-robust, two-sided, falsifiable — *on Qwen embeddings*.** The claim converts
  the H5 negative into a positive construct-validity result and is the headline methodological
  finding of the GNN effort. **Encoder-generality is unconfirmed** (see OQ-B): a lightweight
  robustness probe found the probe-vs-content-similarity Δκ *sign-flips on MiniLM-384* (5-class
  −0.048, 6-class −0.091 with CI excluding 0) — plausibly a capacity artifact (MiniLM carries less
  linearly-separable stage signal; the 6-class "No code" is strongly content-clustered, kNN homophily
  0.30→0.36). This is not a refutation but an unresolved generality question. Manuscript must
  claim H6 *on Qwen embeddings*, not encoder-independently, until the two-encoder test runs (OQ-B).

### §7.6 dyad — fully labeled, but on un-validated cue labels

- **Code:** `purer_analysis.py` (PURER × VAAMR cue-response) + `gnn_layer/communities.py` (dyadic
  routines).
- **Status: structurally complete, validity-pending.** Every FROM→CUE→TO triple is labeled; routines
  are computed. **PURER human validation (Krippendorff α ≥ 0.70) is pending (§8.1).** See OQ-A.
- **Gap (the quiet load-bearing risk):** the entire dyadic-mechanism story rests on therapist cue
  labels we have not yet validated. Until PURER is validated, every therapist-effect claim is doubly
  provisional (under-identified *and* on unverified labels). A PURER-label-noise robustness check
  (perturb at the measured single-rater disagreement rate, re-fit, re-rank) is runnable now and
  should gate every mechanism claim.

---

## 2. The two new trust instruments — honest assessment

A manuscript truthing pass was applied to `docs/methodology.md`, and two verifiability instruments
were built, independently code-reviewed, and fixed. Both were also subjected to independent
methodology peer-review. This section summarizes their status and caveats.

### Feature 1 — LLM Justification-Grounding Audit (`src/analysis/reports/justification_grounding.py`, default-ON)

**What it does.** Measures the share of QUOTED spans in each model justification that actually appear
in the segment text. Uses boundary-aware quote extraction (straight + curly quotes, with a
contraction-aware single-quote pattern that does not mis-fire on "don't", "can't") and a ≥8-character
fuzzy match gate. No new dependencies (stdlib only).

**Corrected Cohort 1–2 numbers.** After a quote-extraction fix that recovered ~21% wrongly-dropped
apostrophe-containing spans (the earlier `[^']{2,}` pattern was silently dropping quoted phrases
like "'I'm a walking miracle'" and "'no one's ever taught me that'"): **78.5% of quoted spans
grounded (560/713 spans)**; per-stage 72.8–80.5%; per-model gemma-4-31b 88.2% / qwen3-next-80b
77.5% / nemotron-3-nano-30b 75.6%; PURER 81.7%.

**Caveats — all must be stated when citing this number:**
- Grounding bounds **CONFABULATION, not correctness.** A faithfully-quoted segment can still be
  mis-staged; the audit complements but does not replace human↔LLM IRR (§5.3–5.4).
- 78.5% is a **LOWER bound on faithfulness.** Lexical matching penalizes honest paraphrase ("the
  patient said" reworded). The flag set is a review queue, not an error rate.
- The 21.5% "ungrounded" share includes genuine confabulation *and* paraphrase. These are not
  distinguished by the current instrument.

**Verdict: sound to cite with the above caveats.** Document in §5 (new subsection on justification
grounding and confabulation bounds). The code-review finding was "cite-with-caveat after fix"; fix
was applied; unit suite green (3424 tests).

### Feature 2 — Segmentation-Sensitivity Check (`src/analysis/segmentation_sensitivity.py`, OPT-IN)

**What it does.** Re-segments under a one-factor-at-a-time parameter grid (semantic-shift percentile,
min/max words, window size, adaptive threshold), projects existing labels + the continuous
`progression_coord` (no re-classification, overlap-weighted), recomputes H1. Every arm is compared
to a **canonical-in-same-embedder baseline** (re-segmented with default params, same embedder,
LLM off) — so only the parameter under test varies.

**Corrected instrument.** The original version measured a hard-label proxy (E[stage]) rather than the
real mixture-coordinate H1 statistic; it is now corrected to use `superposition.attach_superposition`
exactly as the headline is computed. Non-convergence is surfaced, not hidden. The MiniLM-consistent
baseline is added.

**Result.** "STABLE (boundary-placement; labels/coords projected, not re-classified)" — per-arm slope
+0.081…+0.093 in MiniLM embedding space; canonical-in-MiniLM +0.081 vs the Qwen headline +0.061
(the residual MiniLM-vs-Qwen gap is now isolated as an embedder question, not a segmentation
question).

**Heavy caveats — all must be documented in §4.1/§9 if cited:**
- Labels are projected, not re-classified; aggregate stability is **partly structural.** The stronger
  re-classifying arm is future work (OQ-E).
- The Qwen3-8B embedder won't load under the pinned transformers; **arms run in MiniLM space**
  (Qwen-space pending OQ-B).
- All 6 arms are **mixedlm-non-converged** ("MLE on the boundary") at n≈32; CIs not reliable
  (OQ-D).

**Verdict: a directional, opt-in robustness signal.** Document modestly in §4.1/§9. Do NOT elevate
to "H1 is robust to segmentation parameters" — the caveats are too substantial. The code-review
finding was "fix-first, not citable without fixes"; all fixes were applied; unit suite green.

---

## 3. Instrument-choice adjudication — what should lead each job

| Job | Current lead | Verdict | What should lead |
|---|---|---|---|
| **Construct validation** (is VAAMR developmental, not topical?) | `discriminant.py` (H6) | **Right tool.** A probe-vs-similarity *contrast* is exactly correct and N-robust. | **Keep — make it the headline.** Add encoder-generality note (OQ-B). |
| **Mechanism** (does therapist language move participants, conditional on FROM stage?) | `mechanism_model.py` (hierarchical ordinal interaction) + additive FDR table (descriptive) | **Correctly instrumented now.** The interaction is modeled with the right tool. | **`mechanism_model.py` leads.** GNN MLP is a triangulation lens. |
| **Trajectory / consolidation** (does language predict *learning*, not just next utterance?) | — (not yet modeled) | **Missing.** Consolidation is closer to benefit. | **Add** within-vs-between-session split + lagged-cue participant-level model. |
| **Discovery / leads** (routines, motifs, coupling) | `communities.py`, `motifs.py`, `coupling.py` | **Right tool, narrow claim.** H6 says clusters are *content*, so routines are content-adjacency leads. | **Keep as human-filtered lead-generation only.** |

**Why partial pooling is not optional at n≈32.** With ~20 (FROM × move) cells and a handful of
participants carrying the rare stages, independent per-cell estimates are noise and FDR rightly kills
them all — but that is a *property of the estimator*, not evidence of no effect. A hierarchical model
with weak priors shrinks unstable cells toward the grand mean and lets cells with more support speak,
yielding a **single interaction estimate with an honest interval**. This is the standard, expected
small-n tool and the one a methods reviewer will ask for by name.

---

## 4. Peer-review simulation — manuscript must-fix items (O-numbered)

A skeptical methods-venue reviewer (who has read Wexler 2026 and knows n≈32) returned **"major
revision, recoverable to minor with bounded fixes."** The following objections were raised. Items
**O1, O6, O7, O10** are submission-blockers; O2, O4, O11 are revision-tier.

| # | Objection / Error | Answered now? | Fix |
|---|---|---|---|
| **O1** | §1 (≈line 27) says "Wexler… characterized the **five-stage** progression" — contradicting the abstract and §2.2, which correctly say the published model is FOUR stages + one barrier; the 5-class VAAMR is QRA's operationalization, not the source's finding. | **No — still present in `docs/methodology.md`** | One-sentence fix: "characterized the **four-stage** phenomenological progression… its avoidance barrier." |
| O2 | §10 H6 claim lacks the "on Qwen embeddings" caveat; reading the section one could conclude encoder-independent discriminant validity. | Partially (narrative exists, section lacks explicit scope) | Add "in the Qwen3 embedding space (encoder-generality pending; see OQ-B)" to the H6 summary sentence in §10. |
| O4 | §7.3 "Both Sides of the Dyad" heading oversells when PURER is unvalidated; and "the empirical test" → "a test" (single-arm, provisional). | Disclosed in §8.1 but heading is not qualified | Add "(PURER labels provisional pending validation)" to §7.3 heading or immediately adjacent; change "the empirical test" to "a test." |
| **O6** | H4 (convergent validity vs clinical outcomes) has NO pilot result — conspicuous for a mixed-methods venue. | **No** | Either run descriptive Spearman on Cohorts 1–2 (REDCap import) OR add an explicit "H4 yields no pilot result; REDCap not imported" sentence. See OQ-C. |
| **O7** | The two new instruments (justification-grounding audit, segmentation-sensitivity check) are entirely undocumented — free defensibility wins that answer predictable reviewer objections (confabulation; segmentation arbitrariness). | **No** | Add §5 subsection for justification-grounding (cite 78.5%, lower-bound caveat); add §4.1/§9 paragraph for segmentation-sensitivity (directional-only, MiniLM, non-convergence). See §2 of this memo for exact language. |
| **O10** | Abstract says "Cohorts 1 and 2 have completed" but §1 says "currently completing its second" — stale contradiction left by iterative edits. | **No — both stale states still present in `docs/methodology.md`** | Pick one true statement; update the other. If both cohorts have now completed, fix §1. |
| O11 | Capability-table Capability-C / triangulation + reliability-gate + hardening rows read as always-on but require the now-default-OFF GNN classifier. | Partially (GNN section notes the flag) | Add "(requires `qra gnn train`; default-OFF)" to each affected capability-table row. |

**R1–R9 remain:** the core peer-review simulation from the prior memo stands (R1 "fit the
interaction" is now met by `mechanism_model.py`; R2/R3 E-value/identifying-assumption argument is
computed but not yet in the manuscript; R4 PURER gate still pending; R5 GNN-vs-hierarchical
repositioning is done in code but needs §8.5 rewrite; R6/R7 permutation/pre-registration deferred).

---

## 5. What we can do *now*, with the present dataset (pre-REDCap)

Runnable on `./data/Meta` today, defensible as pilot:

1. **The H6 construct-validity headline, scoped to Qwen.** Strongest, N-robust claim. Harden with
   two-encoder test when OQ-B is resolved.
2. **The re-centered mechanism estimate** — hierarchical ordinal interaction model + frequentist
   robustness (`mechanism_model.py`), reporting interaction credible intervals and per-direction
   posterior probabilities for the §7.6 predictions. Honest expected result: "instrument correct,
   effect under-identified at n≈32, bounded by sensitivity analysis."
3. **The confound-sensitivity argument** — E-values / Rosenbaum Γ + the explicit identifying
   assumption, beside the existing divergence map. Converts "we acknowledge a confound" into "the
   confound would have to be Γ this strong."
4. **The justification-grounding report** — cite 78.5% grounding with lower-bound caveat; include
   flag set as review queue. Free defensibility against confabulation objections.
5. **The segmentation-sensitivity report** — cite "directional stability" with all scope caveats
   (MiniLM space, projected labels, non-converged). Free defensibility against segmentation-
   arbitrariness objections if caveated correctly.
6. **The trajectory / consolidation read** — within- vs between-session split. Closest pre-REDCap
   proxy for benefit; needed to fill the "adjacency ≠ mechanism" gap.
7. **The PURER-noise robustness result** — bounds how much un-validated cue labels could move the
   mechanism ranking; gate or companion every therapist-effect claim until human α is measured.

What we **cannot** do now: anything linking language to *clinical benefit* (H4 needs REDCap), and
any *confirmatory* mechanism claim (needs Cohorts 3–4 power). Pre-registering both now is what makes
them confirmatory later.

---

## 6. The single recommendation and must-fix list

> **The current methodology is the right vehicle.** H6 is the headline, `mechanism_model.py` now
> carries H2 with the correct estimator, and the two new trust instruments answer the predictable
> reviewer objections — *if they are documented.* The must-fix list before the next manuscript
> submission is short and all items are one-sentence-to-one-paragraph edits:

**Must-fix before submission (all O-tagged items above):**

1. **O1** — Fix the "five-stage" attribution in §1 (≈line 27) to "four-stage." One sentence. High
   embarrassment risk: it directly contradicts §2.2.
2. **O10** — Resolve the abstract/§1 cohort-completion contradiction. Pick the current true state.
3. **O7** — Document the justification-grounding audit and segmentation-sensitivity check in §5
   and §4.1/§9 respectively. The caveats in §2 of this memo are the exact language to use.
4. **O6** — Either import REDCap and run Spearman, or add the explicit "no pilot result" sentence.
   Do not leave §8.4/H4 silently empty.
5. **O2** — Add "on Qwen embeddings" scope to the §10 H6 summary sentence.
6. **O4** — Qualify §7.3 heading; change "the empirical test" → "a test."
7. **O11** — Add default-OFF notes to capability-table rows that require the GNN classifier.

**Resolve or flag before Cohort-3 report (OQ-series):**

- OQ-A: Schedule PURER blind-coding; decide whether borderline α triggers label-noise robustness
  in lieu of full validation.
- OQ-B: Fix the Qwen3-8B embedder load failure (pin upgrade or checkpoint swap); then run the
  two-encoder H6 test and a Qwen-space segmentation arm.
- OQ-C: Import REDCap or add the explicit H4 no-pilot-result disclosure.
- OQ-D: Decide whether segmentation-sensitivity non-convergence at n≈32 warrants §9 disclosure or
  deferral to Cohorts 3–4.
- OQ-E: Decide whether the re-classifying segmentation arm is a pre-submission must-do or future
  work.

The production design — estimator spec, experiments, sequencing, MoveMORE analysis product — is in
`masterplan.md`.
