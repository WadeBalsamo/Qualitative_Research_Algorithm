# Method–Application Review: the GNN discovery & mechanism layer vs. QRA's primary research questions

**Status:** review draft for adjudication · **Branch:** `beta` · **Companion PR under review:** [#8](https://github.com/WadeBalsamo/Qualitative_Research_Algorithm/pull/8) (`gnn-exp/ws1-h6` → `beta`) · **Date:** 2026-06-07
**Pilot corpus:** Move-MORE Cohorts 1–2 (`./data/Meta/`), n≈32 participants; 339 participant segments (205 VAAMR-labeled, 134 "No code"); 66–76 human IRR items.

> **What this document is.** A systematic, self-critical review that holds the GNN *implementation* (PR #8) against QRA's *primary research questions* (`docs/methodology.md` §3.4) and asks: is this the **right** methodology for learning what we need to learn about therapeutic-language progression across the dialogue, is each claim **verifiable in the committed outputs**, did we **explain it well enough to survive rigorous peer review**, and **what is the better experiment next?** It is written for two readers:
> 1. the **PhD advisor** — adjudicate the science (Part VIII-A);
> 2. a **methodology-review agent** — review the design against the intentions and propose next steps (Part VIII-B).
>
> Everything here is **observational, n≈32, hypothesis-generating — never causal**. The multi-run LLM consensus remains the label of record (κ=0.537 vs human; §5.4).

---

## Table of contents
- [Part I — The primary research questions (the standard)](#part-i)
- [Part II — Intentions → implementation (the full arc)](#part-ii)
- [Part III — The defensibility spine: three instruments, three epistemic statuses](#part-iii)
- [Part IV — Systematic review: implementation vs each research question](#part-iv)
- [Part V — How the findings & this work change the methodology](#part-v)
- [Part VI — Deep questioning of the methodology (verifiable-but-unverified)](#part-vi)
- [Part VII — Did we do the right thing? The better next experiment](#part-vii)
- [Part VIII — Review charges (advisor; methodology agent)](#part-viii)
- [Appendix A — Repository context index](#appendix-a)
- [Appendix B — Register of open, verifiable questions](#appendix-b)

---

<a name="part-i"></a>
## Part I — The primary research questions (the standard we measure against)

From `docs/methodology.md` §3.4 (the falsifiable hypotheses) and §7 (what the method uniquely reveals). The project's scientific spine:

- **H1 — Developmental progression.** Over eight sessions participant language advances along VA→MR; the avoidance→attention-regulation crossing is rate-limiting. *Instrument:* `src/analysis/efficacy.py`, `src/analysis/mechanism.py` (avoidance barrier).
- **H2 — Context-dependent therapist effects (the central mechanism question; §7.6).** Specific PURER moves / cue language are associated with forward VAAMR transitions, and that association is **moderated by the participant's FROM stage** — *the same move helps more at some stages than others.* *Instrument:* `src/analysis/mechanism.py` (Δprogression × from-stage, mixed-effects, permutation, FDR).
- **H3 / H3a — Cross-framework construct validity** (VAAMR × VCE lift; does the codebook sharpen the arc). *Instrument:* `src/process/cross_validation.py`; `gnn_layer/classifier/ablation.py:vce_vaamr_contribution`.
- **H4 — Convergent validity** vs clinical outcomes. *Instrument:* `src/analysis/efficacy.py:link_to_external`.
- **H5 — Distillation/scalability** (graph reproduces LLM consensus to IRR → LLM-free scaling). *Instrument:* `gnn_layer/classifier/validation.py` gate. **Pilot result: refuted** at n≈32.
- **H6 — Discriminant validity** (VAAMR is developmental, not topical). *Surfaced by the pilot.* *Instrument (new):* `gnn_layer/discriminant.py`.

The **binding constraints** the methodology itself names: (i) n≈32, single-arm, observational; (ii) the **elicitation/responsiveness confound** (§9.4 — therapists select moves *in response to* participant state); (iii) linguistic expression ≠ phenomenological state (§9.1); (iv) PURER human-validation is **still pending** (§8.1, target Krippendorff α ≥ 0.70).

**The one-sentence research aim this review uses as its yardstick:** *Identify, defensibly and hypothesis-generatingly, the relationships between therapist language and participant developmental progression across the dialogue — at the resolution where mechanism actually operates — and say honestly how far each relationship can be trusted at this scale.*

---

<a name="part-ii"></a>
## Part II — Intentions → implementation (the full arc in this work)

The intent that drove PR #8, in order, with the implementing artifact and the **verified** pilot numbers.

### II.1 The repositioning that motivated everything
`graph_experiments.md` + `design_decisions.md` established three pilot facts under leakage-free **participant-grouped** CV: H5 refuted (graph reproduces consensus at grouped-κ≈0.05–0.14, below the human↔human band); a probe on the same Qwen features ties/beats the graph; **VAAMR is not homophilous** in embedding space. The §4.7 critique (`docs/GNN_MASTER_PLAN.md`) added that the as-built GraphSAGE was a per-segment *classifier* mis-specified for *mechanism*. **Intent:** stop using the graph as a classifier; rebuild the mechanism instrument; turn the negative (H5) into a positive (H6); separate the concerns in code.

### II.2 WS1 — H6 discriminant validity (the headline)
**Intent:** show VAAMR is recoverable by supervision yet orthogonal to content-similarity, and characterize the geometry. **Artifact:** `src/gnn_layer/discriminant.py` → `06_reports/06_gnn/discriminant_validity.txt`, `03_analysis_data/gnn/discriminant_validity.csv`, `gnn_discriminant_validity.png`.
**Verified (same Qwen embeddings, participant-grouped folds, clustered CIs):**

| arm | human κ [95% CI] (n=66) | LLM κ (n=205) |
|---|---|---|
| probe (supervised, 6-class No-code) | **0.365** [0.228, 0.513] | 0.283 |
| content-similarity (Correct-&-Smooth) | 0.196 [0.117, 0.319] | 0.069 |
| chance (modal / stratified) | 0.000 / −0.083 | 0.000 / −0.017 |

Paired **probe − content Δκ**: human **+0.170** [0.002, 0.318]; LLM **+0.214** [0.150, 0.274] (CIs exclude 0). **Geometry (honest):** the stage signal is *partly* recoverable from the leading content PCs (top-50 grouped-CV κ=0.297 ≈ full-embedding 0.307) — so it is **not** an exotic subspace; the operative property is **weak/uneven local kNN homophily** (1-NN same-stage 0.473 vs base rate 0.251, lift ≈1.88, decaying with k; Metacognition *below* base rate). **community × stage ARI ≈ 0.006** (Cramér's V 0.954 is inflated by many small communities; ARI is the chance-corrected read).

### II.3 WS-T — the dyadic FROM→CUE→TO transition model (the mechanism rebuild)
**Intent (researcher directive):** replace the classifier-counterfactual with a learned *response function* `TO_mixture ≈ f(FROM_mixture, FROM_stage, pooled raw-Qwen cue)` — directed FROM→CUE→TO only, **no kNN** (H6), FROM-stage conditioning baked in (the *partial* control for the confound: moves compared **within** a starting state). **Artifact:** `src/gnn_layer/transition.py` → `06_reports/06_gnn/transition_model.txt`, `transition_counterfactual.csv`, `transition_per_move.csv`.
**Verified (161 triples, 19 participants):**
- **Earns-its-place = NEGATIVE.** Adding the cue does **not** improve held-out TO prediction over a FROM-only baseline (Δ KL **+0.371**, Δ E[stage] MAE **+0.149** — both worse). ⇒ the transition is **under-identified at n≈32** (consistent with H2's pilot status); `mechanism.py` leads.
- **Triangulation = POSITIVE.** The learned per-cell counterfactual ranks with observed Δprogression at **Spearman ρ ≈ +0.337** (sign agreement 0.55) — versus the *retired* classifier-counterfactual's **−0.13**. **0 FDR-significant observed cells**, so convergence is under-powered.

### II.4 WS3 — confound localization
**Intent:** use the divergence between the (timing-independent) learned counterfactual and the (responsiveness-confounded) observed Δprogression to *locate* where the §9.4 confound most distorts the signal — a caveat instrument, not a claim. **Artifact:** `src/gnn_layer/confound.py` → `06_reports/06_gnn/confound_localization.txt`, `confound_localization.csv`, `gnn_confound_localization.png`.
**Verified:** 20 (from_stage × move) cells, **9 sign-inverting**, participant-clustered CIs. E.g. *Vigilance × Reframing* observed +2.089 vs counterfactual +0.107 (divergence +1.98 [0.98, 2.98]); the sign-inversion *Reappraisal × Reinforcement* observed −0.60 vs counterfactual +0.59 (divergence −1.19 [−1.72, −0.94]) — the classic "deployed at difficulty" responsiveness pattern.

### II.5 WS2 — Track-D discovery deepened (the dyad)
**Intent:** surface recurring therapist→participant language **routines** as hypothesis-generating leads, stability-selected. **Artifact:** `src/gnn_layer/communities.py` → `06_reports/06_gnn/communities.txt`, `dyadic_routines.txt`.
**Verified:** subtext-similarity threshold recalibrated **0.85→0.6** for instruction-tuned Qwen (a probe showed τ=0.85 ⇒ noise, Louvain↔hierarchical ARI≈0.003; τ=0.6 ⇒ ARI≈0.286, 24 multi-member communities); 223 communities, **21 stable**; **2 stable dyadic routines** (therapist "body/mindfulness/practice" community → participant communities, both Δprogression CIs include 0 = honest under-powered leads). Community naming surfaced **logistics** chatter (e.g. "can you see my body / do you get my text") alongside therapeutic content — honest, and a flag (see Q in Part VI).

### II.6 Architecture, cleanup, Track C, docs
- **Classifier → `src/gnn_layer/classifier/`** (model/train/graph_builder/validation/triangulation/inference/calibration/propagation/ablation/anchors/gnn_lift), **default OFF** (`gnn_classifier_enabled=False`); opt in via `qra gnn train`. `runner.run_gnn_analysis` split into `_run_classifier_layer` (gated) + `_run_discovery_layer` (always, raw embeddings). Wired through `setup_wizard` (step 11d), `analysis/runner` (`force_classifier`), TUI.
- **No tech debt:** deleted `influence.py` (mechanism-on-classifier) + its test; removed dead `counterfactual_*` config flags; extracted shared `cue_features.py`.
- **Track C repointed:** `process/assembly/mindfulbert_dataset.py` augmentation now sources `transition_per_move.csv` and is gated on the **transition** model (not the classifier gate); still OFF by default + retained only under the C4 ablation.
- **Manuscript updated** (`docs/methodology.md` §3.4 H2/H6, §8.5, §9.4, capability table) + `design_decisions.md` §10, ledger, `gnn_discovery_results.md`.
- **Tests:** full `tests/unit` suite green — **3311 passed, 10 skipped**.

---

<a name="part-iii"></a>
## Part III — The defensibility spine: three instruments, three epistemic statuses

The single most important framing for review: **the GNN is not one thing.** It is three instruments doing three different jobs, and they are *not equally defensible*. Conflating them is the chief risk to peer review.

| Instrument | Job | Defensible status | Right tool? |
|---|---|---|---|
| **H6 discriminant validity** (`discriminant.py`) | construct validation: stage is developmental, not topical | **Strong, N-robust, two-sided, falsifiable.** Rests on a *contrast* that holds regardless of n. | **Yes** — the probe-vs-similarity contrast is the correct tool. The headline contribution. |
| **Transition model** (`transition.py`) | mechanism: timing-independent cue influence | **Secondary, scale-gated.** Under-identified at n≈32 (does not beat FROM-only). Its value is a *sensitivity lens* + a *confound localizer*, NOT a primary estimator. | **Partly** — see Part VI; a hierarchical statistical model may be the more defensible estimator at this scale. |
| **Discovery** (`communities.py`, `motifs.py`, `coupling.py`) | hypothesis-generating leads | **Exploratory only.** H6 says these clusters are *content*, so routines are content-adjacency, weakly informative about *developmental* mechanism. | **Yes, for lead-generation** — provided it is framed as such and human-filtered. |

**The load-bearing claim the manuscript must make unmissable:** the **observed** Δprogression analysis (`mechanism.py` — from-stage-conditioned, mixed-effects, within-stratum permutation, FDR, participant-clustered CIs) is the **primary** mechanism instrument and **leads**; the GNN is **construct validation (H6) + a secondary sensitivity/discovery layer.** (PR #8's §8.5 says this; a reviewer skimming could still mis-read the transition model as the estimator — Q14.)

---

<a name="part-iv"></a>
## Part IV — Systematic review: implementation vs each primary research question

For each question: **what the implementation does**, what is **VERIFIED** (committed output exists), what is **NOT verified** (open + verifiable), and the **gap**.

### H1 — Developmental progression
- *Implements:* `efficacy.py` progression slope + `mechanism.py` avoidance-barrier (bidirectional). **VERIFIED:** descriptive trajectories exist. **NOT verified:** that the avoidance→attention crossing is *rate-limiting* in a tested sense (vs just descriptively prominent); a group slope CI excluding zero on this corpus (Q1). **Gap:** H1's "rate-limiting" clause is asserted more than tested. Untouched by PR #8.

### H2 — Context-dependent therapist effects (the central question)
- *Implements:* observed Δprogression × from-stage (`mechanism.py`, LEAD) + the transition-model counterfactual + confound-localization (PR #8, secondary). **VERIFIED:** 0 FDR-significant cells (observed); transition cue does not earn its place; counterfactual ρ=+0.34; 9/20 confound cells sign-invert. **NOT verified:** whether *any* (from_stage × move) effect is real (needs power) (Q2); whether a **hierarchical ordinal** model finds an interaction the additive table misses (Q15); whether the §7.6 *directional* predictions hold per-cell (Q3). **Gap:** H2 is honestly *under-identified*; the mechanism instrument set is in place but the data scale + confound are binding. **This is the question most affected by — and most limited in — this work.**

### H3 / H3a — Cross-framework construct validity
- *Implements:* `cross_validation.py` lift; `classifier/ablation.py:vce_vaamr_contribution`. **VERIFIED:** lift table exists. **NOT verified:** the **shuffled-stage permutation control** for lift (committed in §8.2 as a *must-before-Cohort-3* item) is not yet run (Q4); H3a Δκ is currently measured on the *graph* classifier, which §3.4 itself flags as a low-power test (Q5). **Gap:** the permutation control + re-instrumenting H3a on the LLM/probe (not the graph) are open. Untouched by PR #8.

### H4 — Convergent validity vs clinical outcomes
- *Implements:* `efficacy.py:link_to_external`. **VERIFIED:** scaffolding exists. **NOT verified:** the REDCap outcome linkage itself (pain/TSK-11/ODI/MAIA-2) — pending data integration (Q6). **Gap:** entirely pending. Untouched.

### H5 — Distillation / scalability
- *Implements:* `classifier/validation.py` gate. **VERIFIED & settled:** refuted at n≈32 (grouped κ≈0.05–0.14). PR #8 makes this structural (classifier default-OFF). **NOT verified:** whether *any* learned scaler (the §8.4 fine-tuned model, or the graph at Cohorts 3–4) can earn the role at higher N (Q7). **Gap:** re-opened only at scale.

### H6 — Discriminant validity
- *Implements:* `discriminant.py` (PR #8). **VERIFIED:** probe ≫ content ≈ chance; paired Δκ CI excludes 0; geometry. **NOT verified:** robustness to embedding choice (Qwen vs MentalBERT vs MiniLM — `graph_experiments.md` flagged this but it is not run) (Q8); replication at Cohorts 3–4 (Q9); that the *human axis* (not just the LLM axis) drives the contrast at larger N. **Gap:** H6 is the best-supported claim but its *generality* (embedding, cohort) is unverified.

### §7.6 — The dyad, fully labeled
- *Implements:* PURER × VAAMR cue-response (`purer_analysis.py`) + dyadic routines (PR #8). **VERIFIED:** the FROM→CUE→TO structure is fully labeled; routines computed. **NOT verified:** that PURER labels are reliable (human α≥0.70 pending, §8.1) — *the entire dyadic-mechanism story rests on cue labels we have not yet validated* (Q10); that dyadic routines carry developmental (not merely content) signal (Q11). **Gap:** the cue-label validity dependency is the quiet load-bearing risk.

---

<a name="part-v"></a>
## Part V — How the findings & this work change the methodology

What PR #8 *forces* the manuscript to say differently (done in PR #8; flagged here for the advisor to confirm):

1. **H2's pilot status (§3.4).** The mechanism reading changes: the *rebuilt* transition model triangulates **positively** (ρ≈+0.34) — a correction of the prior **−0.13** (which was the *mis-specified classifier* counterfactual, not a property of the mechanism). But the cue still does not earn its place, so H2 remains under-identified. *Implication:* the manuscript should no longer present ρ=−0.13 as "the mechanism doesn't triangulate"; it should present the *under-identification* (earns-its-place failure) as the honest status, with ρ flips as evidence the **instrument** is now correctly specified.
2. **§9.4 confound is now read at the cell level.** The old text used the global ρ=−0.13 inversion as *the* confound signature. With the proper model's ρ positive, the confound evidence is the **per-cell divergence map** (9/20 sign-inverting), not a single coefficient. (PR #8 rewrote this.)
3. **H6 becomes an operationalized instrument** (§8.5), not just a "standing observation," with an explicit honest nuance: stage *is* partly in the content PCs; the discriminant property is **local non-homophily**, not global subspace orthogonality. (A reviewer who runs the geometry must find the prose matches — Q12.)
4. **The classifier is repositioned in *code*, not just prose:** default-OFF, separate subpackage. The manuscript's "graph dropped as classifier of record, retained as mechanism/discovery" is now structurally true.

**Open manuscript-quality questions (Part VI):** is the *primary vs secondary* mechanism framing unmistakable? Is the confound *mapped vs solved* distinction explicit enough? Is the PURER-validity dependency disclosed where mechanism claims are made?

---

<a name="part-vi"></a>
## Part VI — Deep questioning of the methodology (verifiable-but-not-yet-verified)

This is the heart of the review. Each question is **answerable with a concrete check we have not run.** Status of all = **OPEN**. (Consolidated in Appendix B.) Grouped by theme.

### A. Is the mechanism *estimator* the right one?

- **Q13 — Learned MLP vs hierarchical model.** The transition model is a small neural net that **does not beat a FROM-only baseline** at n≈32. *Question:* would a **Bayesian/mixed-effects ordinal transition model** (`TO_stage ~ FROM_stage * move + (1 | participant)`, ordinal link, weak priors) be both more interpretable *and* more defensible at this scale? *Verify:* fit it on `master_segments.csv`; compare held-out predictive log-loss to the MLP and to FROM-only under the same participant-grouped folds; report interaction coefficients with credible intervals. *Why it matters:* a reviewer will ask why report a neural model that doesn't earn its place over a standard hierarchical model.
- **Q14 — Primary-vs-secondary framing.** *Question:* does the manuscript make it unmissable that `mechanism.py` (observed) is primary and the GNN counterfactual is a sensitivity lens? *Verify:* a methods reader reads §8.5/§3.4 cold and is asked "what is the primary mechanism estimator?" — do they answer `mechanism.py`?
- **Q15 — Does the nonlinear model find an interaction the additive table misses?** *Question:* the GNN's only claim to add value over `mechanism.py` is context-dependent/nonlinear influence. *Verify:* compare the transition model's per-(from_stage×move) counterfactual ranking to the additive `mechanism_delta_progression.csv` ranking; is there any cell where the conditioning materially changes the sign/order *and* survives a participant-clustered bootstrap? (Today: 9/20 sign-differ, but with 0 FDR-significant observed cells none is powered.)

### B. Is the *cue representation* right?

- **Q16 — Content embedding vs process features.** H6 says the embedding encodes content, not stage. The cue is currently a *content* pooled embedding. *Question:* if a cue's mechanism-relevant property is its **PURER move** (a process construct), should the cue be represented by PURER-move probabilities (or both)? *Verify:* re-fit the transition counterfactual with cue = PURER one-hot/soft vs cue = content embedding vs both; compare earns-its-place Δ and triangulation ρ.
- **Q17 — Does PURER reliability gate the whole story?** PURER human-validation (α≥0.70) is **pending** (§8.1). *Question:* how sensitive are the dyadic routines / per-move counterfactual to PURER label noise? *Verify:* perturb PURER labels at the measured single-rater disagreement rate and re-compute the per-move influence ranking stability.

### C. Is the *unit of analysis* right?

- **Q18 — Adjacent triple vs trajectory dynamics.** The model is per-adjacent-triple. Progression has a *consolidation* (between-session) component (§4.8) the triple ignores. *Question:* should mechanism be a **sequence/state-space** model over each participant's stage trajectory with cues as time-varying covariates? *Verify:* fit a simple participant-level state model (e.g., ordinal HMM or a mixed-effects model with lagged terms) and compare what it identifies vs the adjacent-triple model.
- **Q19 — Within- vs between-session.** *Question:* do the within-session momentary effects and between-session consolidation effects point the same way per move? *Verify:* split the Δprogression analysis by within-session vs between-session-dominant-stage transitions (the matrices already exist, §4.8) and compare.

### D. Identification & the confound

- **Q20 — Sensitivity to unmeasured confounding.** The confound is *mapped*, not *solved*. *Question:* how strong would the responsiveness selection have to be to explain away any apparent effect? *Verify:* compute **E-values / Rosenbaum bounds** on the observed per-cell associations; report alongside the divergence map. *Why:* this is the standard, expected way to defend "not causal but bounded."
- **Q21 — A quasi-experimental lever.** *Question:* is there naturalistic variation — session phase, dose, therapist identity, curriculum module — usable as an instrument or for a within-design contrast that partially breaks selection-on-state? *Verify:* enumerate candidate instruments in the data and test their relevance/exclusion plausibility.
- **Q22 — State the identifying assumptions explicitly.** *Question:* under exactly what assumptions (e.g., sequential ignorability given FROM_mixture+FROM_stage) would the adjacency association be causal, and are they stated and shown violated? *Verify:* a paragraph in §9.4 naming the assumption and why FROM-stage conditioning is insufficient.

### E. Discovery — meaningful or content/logistics?

- **Q23 — Logistics contamination.** Stable communities include logistics ("can you see my body", "do you get my text"). *Question:* should discovery run on a therapeutically-relevant subset, or is surface-then-human-filter the discipline? *Verify:* re-run communities excluding logistics/greeting turns; do the *therapeutic* routines survive stability selection?
- **Q24 — Do routines add anything beyond content co-occurrence?** Given H6 (communities are content), *Question:* are dyadic therapist→participant routines more than content adjacency? *Verify:* compare routine transition counts to a content-co-occurrence null; do the stable routines exceed it?
- **Q25 — Counterfactual CI honesty.** The per-move counterfactual CIs are tight because they are **across-block** sampling, not **model-training** uncertainty. *Question:* how much do the per-move influences move across training seeds / participant-bootstrap refits? *Verify:* refit the transition model on participant-resampled data K times; report the spread of per-move influence (a *training* CI), not just the across-block CI. (Reinforcement +0.60 on 18 thin-support blocks is a likely extrapolation artifact — Q26.)
- **Q26 — Thin-support extrapolation.** *Question:* are the large per-move shifts (Reinforcement +0.60, n=18) extrapolations of the response function beyond where those cues occur? *Verify:* restrict the counterfactual to in-support cells; recompute the ranking.

### F. Construct validity (H6) generality

- **Q8/Q27 — Embedding robustness.** *Question:* does H6 (probe ≫ similarity) hold for MentalBERT / a domain encoder, or is it Qwen-specific? *Verify:* re-run `discriminant.py` arms on ≥2 encoders.
- **Q28 — Is "No code" doing the work?** The probe needs 6 classes (No-code) to reach κ=0.365. *Question:* how much of the discriminant contrast is the abstention class vs the five stages? *Verify:* report the 5-class probe-vs-content contrast on the labeled-only subset alongside the 6-class.

### G. Whole-method coherence

- **Q29 — Does the GNN earn its place in the *paper*?** *Question:* if the transition model is under-identified and the discovery is content, is the GNN's manuscript footprint (a large §8.5) proportionate to its evidential contribution (H6 + leads)? Could the paper be *stronger* by foregrounding H6 + the classical mechanism and scoping the GNN tighter?
- **Q30 — Pre-registration.** *Question:* are the §7.6 directional predictions written as code-level `expected_codes`/expected-transitions before Cohort 3 (the §8.2 *must-before-Cohort-3* commitment)? *Verify:* check `vaamr.py`/`purer.py` for the populated structures; today they are not.

---

<a name="part-vii"></a>
## Part VII — Did we do the right thing? The better next experiment

**Honest verdict.** PR #8 did the *right structural* thing: it stopped over-claiming the classifier, made H6 a real instrument, rebuilt the mechanism tool correctly-specified, and kept everything honestly caveated and tested. **But** the *mechanism* aim (H2 — the project's center) is served only weakly by a learned model that doesn't earn its place at n≈32, and the discovery is content-level. The strongest, most defensible core is **H6 + the classical hierarchical mechanism + an explicit confound-sensitivity analysis**, with the GNN scoped to construct-validation and lead-generation.

**Ranked next experiments (each maps to OPEN questions above):**
1. **Hierarchical/Bayesian ordinal transition model** as the *primary* mechanism estimator (Q13/Q15) — more interpretable + defensible at small n than the MLP; the MLP becomes a triangulating sensitivity lens.
2. **E-value / Rosenbaum sensitivity analysis** in §9.4 (Q20) — the expected way to defend "bounded, not causal."
3. **Pre-register §7.6 directional predictions** in `vaamr.py`/`purer.py` before Cohort 3 (Q30) — turns Cohorts 3–4 into confirmatory replication (the §8.2 commitment).
4. **Human-validate PURER** to α≥0.70 (Q10/Q17) — the dyadic-mechanism story rests on cue labels not yet validated.
5. **Cue-representation experiment** (Q16) — process features vs content embedding.
6. **H6 robustness** across embeddings + 5-class vs 6-class (Q8/Q28) — generality of the headline finding.
7. **Shuffled-stage permutation control for lift** + re-instrument H3a off the graph (Q4/Q5) — the other §8.2 must-before-Cohort-3 item.
8. **Training-uncertainty CIs + in-support restriction** for the counterfactual (Q25/Q26) — fixes the misleadingly-tight CIs and the thin-support extrapolations.
9. **Trajectory/sequence model + within-vs-between-session split** (Q18/Q19) — engage consolidation, not just adjacency.

---

<a name="part-viii"></a>
## Part VIII — Review charges

### VIII-A — For the PhD advisor (adjudicate the science)
Please rule on:
1. **Scope of the GNN in the paper.** Is the three-instrument split (H6 / mechanism-sensitivity / discovery) the right framing, and is the GNN's manuscript footprint proportionate to its evidence (Q29)? Should mechanism be led by a hierarchical model (Q13)?
2. **H2 honesty.** Is "under-identified at n≈32; instrument now correctly specified (ρ flips −0.13→+0.34); confound localized not solved" the defensible statement, or an over- or under-claim?
3. **Confound treatment.** Is mapping (divergence) without an E-value/sensitivity analysis (Q20) sufficient for a methods venue?
4. **H6 as headline.** Is the discriminant-validity finding strong + general enough (pending Q8/Q9/Q28) to lead the contribution?
5. **Dependencies.** Should mechanism/dyadic claims be gated on PURER validation (Q10) and the §8.2 permutation control + pre-registration (Q4/Q30)?
6. **The next experiment.** Which of Part VII's nine to fund first.

### VIII-B — Prompt for the methodology-review agent (review design vs intentions → next steps)
> **You are a methodology-review agent for the QRA project.** Inputs: this file (`experiments/docs/method_application_review.md`), `docs/methodology.md`, `experiments/docs/graph_experiments.md`, `experiments/docs/design_decisions.md` (§10), `experiments/docs/gnn_discovery_results.md`, the `06_reports/06_gnn/*` outputs, and PR #8 (`gnn-exp/ws1-h6` → `beta`). **Do not review code correctness — that is covered. Review the *design against the stated intentions* and recommend next steps.**
>
> **Your charge:**
> 1. **Map implementation → intention → primary question.** For each of H1–H6 and §7.6, confirm the implementation actually tests the stated question, using the committed outputs. Flag every place the artifact does not substantiate the claim, or tests a *weaker* question than stated (Part IV is the starting map — verify it; do not trust it).
> 2. **Adjudicate instrument choice.** Is the GNN the *optimal logical process* for each job, or is a simpler/standard method superior at n≈32? Specifically rule on Q13 (hierarchical vs MLP mechanism), Q16 (cue representation), Q18 (unit of analysis), Q20 (confound sensitivity). Recommend the estimator the paper should lead with.
> 3. **Verify the verifiable.** Take the OPEN questions in Appendix B and, for each, either (a) run/spec the exact check and report the result, or (b) state precisely why it cannot be run yet (needs Cohort 3–4 / REDCap / PURER validation). Convert "we believe" into "we verified" wherever the data already allow it.
> 4. **Peer-review simulation.** Read the manuscript (`docs/methodology.md` §3.4/§7.6/§8.5/§9.4) as a skeptical methods reviewer. List the top objections and whether the current text answers them. Decide: would this pass rigorous peer review *as written*, and if not, the minimal additions required.
> 5. **Next-step plan.** Return a ranked, dependency-aware plan (Part VII is a candidate — improve it): for each step, the question it closes, the artifact it would add, the expected defensibility gain, and whether it is doable now or scale-gated.
>
> **Deliverable:** a memo keyed to H1–H6/§7.6 and to Appendix B's question IDs, ending with a single recommendation: *what is the next experiment, and is the current GNN methodology the right vehicle or should it be re-scoped?* Be adversarial; the authors want the strongest honest critique and the best next move.

---

<a name="appendix-a"></a>
## Appendix A — Repository context index

**Manuscript / design records**
- `docs/methodology.md` — the manuscript. Key sections: §2.2 (re-habituation), §3.3–3.4 (constructs + H1–H6), §4.7–4.8 (cue blocks; within/between session), §5.3–5.4 (reliability; human validation), §6.4 (four-cohort progression), §7.2/§7.6 (adjacency mechanism; context-dependent therapist effects), §8.1 (PURER validation pending), §8.2 (pre-registration + permutation control, *before Cohort 3*), §8.5 (GNN layer — updated in PR #8), §9.1/§9.2/§9.4/§9.5 (limits; the confound).
- `docs/GNN_MASTER_PLAN.md` — §4 deliberation (CFiCS; §4.4 GNN value; §4.7 architecture critique).
- `graph_experiments.md` — the executed reliability battery (the honest negatives, leakage correction).
- `design_decisions.md` — §10 (this work: rebuild + reorg + classifier-off, with numbers).
- `docs/gnn_experiments/ledger.csv` — per-arm κ ledger (incl. the H6 arms).
- `gnn_discovery_results.md` — promotion guide + results summary (on `gnn-exp/ws1-h6`).

**Code introduced/changed by PR #8** (`gnn-exp/ws1-h6`)
- `src/gnn_layer/discriminant.py` (H6), `transition.py` (mechanism), `confound.py` (WS3), `communities.py` (deepened), `cue_features.py` (shared), `runner.py` (split), `config.py` (flags), `figures.py`.
- `src/gnn_layer/classifier/` — the default-OFF classifier subpackage (`validation.py` = the H5 gate).
- `src/analysis/mechanism.py` — the **primary** mechanism estimator (observed Δprogression; unchanged by PR #8).
- `src/process/assembly/mindfulbert_dataset.py` — Track C (augmentation repointed to the transition counterfactual).
- Tests: `tests/unit/test_gnn_{discriminant,transition,dyadic,confound}.py`.

**Outputs to read** (`./data/Meta/06_reports/06_gnn/`)
- `discriminant_validity.txt`, `transition_model.txt`, `confound_localization.txt`, `communities.txt`, `dyadic_routines.txt`, `emergent_motifs.txt`, `coupling.txt`; CSVs in `03_analysis_data/gnn/`; observed mechanism in `03_analysis_data/mechanism/mechanism_delta_progression.csv`; human axis in `06_reports/06b_irr_report.txt`.

---

<a name="appendix-b"></a>
## Appendix B — Register of open, verifiable questions

All **OPEN** (not yet verified). ID → question → how to verify → cohort-gated?

| ID | Question | How to verify | Now or scale-gated |
|----|----------|---------------|--------------------|
| Q1 | Is the H1 group progression slope CI excluding 0; is the avoidance crossing rate-limiting in a tested sense? | `efficacy.py` slope + barrier-crossing rate with cluster-bootstrap CI | now |
| Q2 | Is any (from_stage × move) effect real? | power analysis on the observed table; needs FDR-significant cells | scale-gated |
| Q3 | Do §7.6 directional per-cell predictions hold (U helps Metacog not Vigilance, etc.)? | confirmatory test of pre-specified cells | scale-gated (needs Q30) |
| Q4 | Shuffled-stage permutation control for lift (H3) | re-run lift on permuted stage assignment (§8.2) | now |
| Q5 | Re-instrument H3a off the graph (LLM/probe), not the weak graph classifier | Δκ with/without VCE on the probe | now |
| Q6 | Convergent validity vs clinical outcomes (H4) | `efficacy.py:link_to_external` once REDCap integrated | data-gated |
| Q7 | Can any learned scaler earn the role at higher N? | re-run gate at Cohorts 3–4 | scale-gated |
| Q8 | Does H6 hold across embeddings (MentalBERT/MiniLM)? | re-run `discriminant.py` arms on ≥2 encoders | now |
| Q9 | Does H6 replicate at Cohorts 3–4? | re-run `discriminant.py` | scale-gated |
| Q10 | Are PURER labels reliable (α≥0.70)? | human IRR on PURER (§8.1) | now (human) |
| Q11 | Do dyadic routines carry developmental (not just content) signal? | routines vs content-co-occurrence null | now |
| Q12 | Does the geometry prose match the recomputed numbers? | re-run discriminant geometry; compare to §8.5 text | now |
| Q13 | Hierarchical ordinal model vs MLP for mechanism | fit both; compare held-out + interpretability | now |
| Q14 | Is primary(mechanism.py)-vs-secondary(GNN) framing unmissable? | cold methods read | now |
| Q15 | Does the nonlinear model find an interaction the additive table misses? | compare rankings + bootstrap | now (under-powered) |
| Q16 | Cue = PURER process features vs content embedding? | re-fit transition counterfactual by representation | now |
| Q17 | Sensitivity of routines/influence to PURER label noise | perturb labels at disagreement rate; re-rank | now |
| Q18 | Trajectory/sequence model vs adjacent triple | fit a participant-level state model; compare | now |
| Q19 | Within- vs between-session effect agreement per move | split Δprogression by transition type | now |
| Q20 | E-value / Rosenbaum bounds on observed associations | compute + report in §9.4 | now |
| Q21 | A quasi-experimental lever (session phase/dose/therapist) | enumerate + test instrument relevance | now |
| Q22 | State + falsify the identifying assumption | §9.4 paragraph | now |
| Q23 | Do therapeutic routines survive without logistics turns? | re-run communities on filtered corpus | now |
| Q24 | Routines beyond content co-occurrence? | null comparison | now |
| Q25 | Training-uncertainty CIs for the counterfactual | participant-bootstrap refit ×K | now |
| Q26 | Thin-support extrapolation (Reinforcement +0.60, n=18) | restrict counterfactual to in-support cells | now |
| Q27 | 5-class vs 6-class: how much is "No code" doing? | report both contrasts | now |
| Q28 | (= Q27, abstention contribution) | as Q27 | now |
| Q29 | Is the GNN's paper footprint proportionate to its evidence? | advisor judgment | now |
| Q30 | Are §7.6 predictions pre-registered in code before Cohort 3? | inspect `vaamr.py`/`purer.py` | now |

> **The honest one-line summary for the advisor:** *We built the discovery/construct-validation layer well and rebuilt the mechanism instrument correctly; H6 is a real, defensible contribution; the central mechanism question (H2) remains under-identified at n≈32 and is currently served best by the classical hierarchical analysis plus an explicit confound-sensitivity analysis we have not yet added — and the next, highest-value experiment is the hierarchical ordinal mechanism model + E-value sensitivity + pre-registration, not more GNN.*
