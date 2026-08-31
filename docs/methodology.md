# Phenomenology at Trial Speed: A Computational Mixed-Methods Pipeline for Iterative Refinement of Mindfulness-Movement Therapy in Chronic Pain

**Wade Balsamo¹, Ryan S. Wexler¹², and the Move-MORE Research Team**
¹ Helfgott Research Institute, National University of Natural Medicine, Portland, OR

---

## Abstract

**Background.** Iterative early-phase trials must complete qualitative synthesis within between-cohort windows measured in weeks, but careful phenomenological analysis takes months. The mismatch is most consequential for mindfulness-based interventions for chronic pain, whose active ingredients are first-person experiential transformations that outcome scales cannot capture.

**Method.** We describe the Qualitative Research Algorithm (QRA), an open-source computational phenomenology pipeline. QRA operationalizes a published four-stage model of therapeutic progression in Mindfulness-Oriented Recovery Enhancement (Vigilance–Avoidance Metacognition–Reappraisal; VA-MR) as a five-class labeling scheme (VAAMR), classifies every participant utterance by multi-run large-language-model (LLM) consensus with confidence tiering and abstention, classifies therapist dialogue against MORE's PURER inquiry framework at the cue-block level, and assembles the dyadic FROM→CUE→TO structure that makes therapist–participant interaction empirically interrogable at temporal adjacency. Validation follows a text-psychometrics framework: known-label content-validity test sets, blind human coding, chance-corrected agreement via canonical statistical libraries, a justification-grounding (confabulation) audit, and leakage-free participant-grouped cross-validation.

**Results (pilot deployment).** On the Move-MORE feasibility trial (Cohorts 1–2 complete, Cohort 3 in progress; 20 participants, 205 participant segments), trained human coders agree on VAAMR at Krippendorff's α = 0.33–0.52 — a measured reliability ceiling for a genuinely fuzzy construct. The LLM consensus reaches Cohen's κ = 0.537 [95% CI 0.39, 0.68] against the human consensus, i.e., human-level agreement at that ceiling. Adaptive-stage occupancy rises 44%→100% across the eight-session protocol (Mann-Kendall τ = +0.71, p = .019; single-arm, descriptive). A graph classifier distilled from the consensus failed held-out validation (κ ≈ 0.05–0.14) once participant-grouped cross-validation removed a leakage artifact (random-fold κ ≈ 0.25) — and that failure constitutes positive discriminant-validity evidence: stage is recoverable by direct supervision but not by content similarity (paired Δκ = +0.17 [0.00, 0.32]), so VAAMR indexes a developmental process, not a topic taxonomy.

**Conclusions.** Computational phenomenology at trial speed is feasible when the machine deploys, rather than replaces, expert attention; when reliability targets are set by the construct's measured human ceiling rather than convention; and when negative results are instruments. The pipeline, audit trail, and validation artifacts are open source.

**Keywords:** computational phenomenology; mixed methods; large language models; mindfulness; chronic pain; inter-rater reliability; text psychometrics; qualitative methods

---

## 1. Background

### 1.1 The methodological problem

Iterative clinical trial designs share a structural feature with significant practical consequences: each cohort's experience is meant to inform the next cohort's curriculum, and the analytic synthesis that should drive between-cohort modification must be completed within weeks. For interventions whose mechanisms are quantitatively measurable, modern outcome dashboards meet this demand. But for interventions whose active mechanisms are first-person experiential transformations, the temporal demand collides with the time qualitative analysis genuinely requires.

Mindfulness-based interventions (MBIs) for chronic pain are the case in point. The phenomenological tradition — Merleau-Ponty (1962) on the lived body, extended by Leder (1990) — describes chronic pain not as a persistent unpleasant sensation but as a structural disorder of embodied experience: *dys-appearance*, the body's pathological insistence on appearing as obstacle rather than receding as the transparent medium of engagement. On this account, mindfulness training is structured phenomenological re-habituation: it does not make pain disappear but alters the structural relationship between attention and the painful body. Evidence that Mindfulness-Oriented Recovery Enhancement (MORE; Garland, 2024) produces clinically meaningful effects is substantial (Garland et al., 2022). Wexler, Balsamo et al. (2026) characterized a four-stage phenomenological progression in MORE for lumbosacral radicular pain — Pain Vigilance, Attention Regulation applied to Experiential Avoidance, Metacognitive Awareness, Pain Reappraisal (VA-MR) — together with its central developmental obstacle, the *avoidance barrier*: participants who deploy newly acquired attentional skill to escape pain experience rather than engage it.

That characterization required months of thematic analysis on thirty session recordings. The Move-MORE Feasibility Trial that builds on it (NCT07125027) runs four cohorts in a learning-health-system design with between-cohort windows of weeks. Without a tractable methodology for systematic phenomenological analysis at trial speed, between-cohort refinement defaults to clinical intuition and aggregate outcome scores — important inputs, but ones that cannot answer the questions a phenomenologically grounded intervention most needs to answer about itself.

### 1.2 Can computational analysis preserve first-person validity?

A serious objection must be addressed before describing the pipeline. Chatzichristos (2025) argues that AI-assisted qualitative analysis risks an unacknowledged return to positivism: when the analytic focus becomes model output, the first-person perspective that defines qualitative inquiry is displaced rather than served. We take the objection seriously and build the answer into the architecture rather than the rhetoric.

QRA's position, in the neurophenomenological lineage of Varela (1996) and the computational phenomenology of Ramstead et al. (2022), is that the machine's role is to *deploy human qualitative attention efficiently*, never to replace phenomenological judgment. Concretely: every machine label carries a justification quoting the segment, and the rate at which those quotations are genuine is itself audited (§3.4); the classifier abstains rather than guesses; every human–machine disagreement is surfaced in a line-by-line dossier for expert adjudication; human-adjudicated labels permanently outrank machine labels in the dataset's provenance hierarchy; and the curriculum decisions the analysis informs remain deliberative acts by clinicians, patient partners, and researchers (§4). The pipeline's output is not "the analysis"; it is an organized, auditable evidence base that directs expert attention to where it is most consequential — the cases the machine cannot resolve, the patterns no human could enumerate at corpus scale, and the disagreements that reveal the construct's genuine boundaries. Section 5 returns to what this stance does and does not license.

### 1.3 The constructs

**VAAMR (participant side).** We operationalize the published four-stage VA-MR model as a **five-class labeling scheme**, promoting the avoidance barrier — which the source analysis identifies as MORE's central developmental obstacle — to a distinct labelable class, so that maladaptive attention deployment (Avoidance) is separated from adaptive sustained attention (Attention Regulation). The class boundary is our operational refinement; the developmental arc is unchanged from the source (Table 1). Each stage admits a phenomenological reading as a step in the progressive restoration of the body's transparency: from attentional colonization (Vigilance), through instrumental management (Avoidance), to sustained open attention (Attention Regulation), reflexive observation (Metacognition), and noematic transformation of pain itself (Reappraisal).

**PURER (therapist side).** Each MORE session follows a structured therapist inquiry format — **P**henomenology, **U**tilization, **R**eframing, **E**ducation/Expectancy, **R**einforcement — functioning as a structured method for eliciting, exploring, and consolidating first-person reports (cf. Giorgi, 1985). The methodological consequence is that therapist dialogue is not neutral context but a *systematic elicitor* of phenomenological description; capturing it empirically is required both for mechanism analysis and for honest interpretation of participant labels (§5).

**VCE (optional enrichment).** A 54-code adaptation of the Varieties of Contemplative Experience phenomenology codebook (Lindahl et al., 2017) can be applied to participant segments as multi-label content characterization. It is optional, is never a label of record, and no validity claim in this paper rests on it.

*Table 1 about here (VAAMR stage definitions with canonical expressions).*

### 1.4 Relation to existing computational approaches

Machine-assisted qualitative coding spans supervised assistants requiring substantial labeled corpora (e.g., AQUA; Lennon et al., 2021), graph-based classification of therapeutic elements over pretrained clinical embeddings (CFiCS; Schmidt et al., 2025), and, most recently, zero- and few-shot LLM coding against researcher-supplied codebooks. QRA differs in four commitments that jointly define its contribution. First, the constructs are operationalized from a *published, human-derived* qualitative model — the LLM is never asked to invent a classification scheme, only to apply an expert codebook whose every downstream label traces to human phenomenological coding of real therapy dialogue. Second, reliability is referenced to the construct's *measured human ceiling* rather than to a conventional threshold (§3.2): for genuinely fuzzy phenomenological categories, demanding κ ≥ 0.70 of a machine when expert humans agree at α ≈ 0.5 misstates both the machine and the construct. Third, the architecture is *bilateral* — participant and therapist frameworks classified from the same input in the same pass — which is what makes dyadic mechanism analysis possible at all; segment-level classifiers, however accurate, cannot provide it. Fourth, validation is adversarial toward the method's own architecture: leakage-free participant-grouped cross-validation, calibrated abstention, an automated confabulation audit, and published negative results — including the refutation of the graph-classifier design that CFiCS-style relational structure had motivated (§3.5), which we report with the same precision as the positives. The text-psychometrics framework (Low et al., 2024) supplies the validity vocabulary; QRA supplies a trial-embedded instantiation that ships with its psychometric evidence attached.

### 1.5 Aims

This paper has three aims: (1) to document what the pipeline computes, with the boundary between implemented and planned capability stated explicitly; (2) to report the pilot validation evidence — including the negative results — from the methodology's primary deployment on Move-MORE Cohorts 1–3; and (3) to state the epistemological limits that no engineering can dissolve, so that the method's claims are exactly as large as its warrant.

---

## 2. The QRA Pipeline

QRA operates on diarized transcripts of group session recordings (locally deployed Whisper with speaker diarization; research assistants verify transcripts; all identifying information is anonymized before analysis). The pipeline is open source; this section describes the analytical logic, and the repository documents the full technical specification. Eight stages run in sequence; we describe the five that carry methodological weight.

### 2.1 Semantic segmentation and speaker separation

The atomic unit of analysis is the *segment*: a semantically coherent multi-sentence unit, identified by boundary detection on sentence-embedding similarity curves with adaptive per-session thresholds, rather than by speaker turns. The choice embeds a theoretical commitment: the unit should be the stretch of discourse constituting a single identifiable expression of an experiential state — a participant may complete one coherent report across several short turns, or shift registers within one long turn. Because segmentation thresholds are a researcher degree of freedom, an opt-in sensitivity analysis re-segments the corpus across a parameter grid and tests whether the headline progression result survives (it does, directionally, within the honestly stated scope of that instrument; Supplementary S4).

Therapist speech is separated at segmentation and **never** classified against VAAMR: therapist dialogue has a systematically different register (questioning forms, second-person address, clinical hedging) and would generate systematic false positives. Participant segments go to VAAMR classification; therapist segments are retained, chronologically interleaved, and classified against PURER (§2.3). Segments, once written, are frozen and content-hashed; every classification is an overlay keyed to the frozen segment, so re-running a classifier can never silently move the text it claims to have coded.

### 2.2 Multi-run LLM consensus classification with abstention

Each participant segment is classified `n` times against the full VAAMR framework definitions (formal definition, prototypical features, distinguishing criteria for the confusable neighbors, and exemplar/subtle/adversarial example utterances per stage), with preceding-context preamble, returning structured output: primary stage, confidence, secondary stage, and a justification that must cite specific segment language. Two run modes carry different reliability semantics, and we are explicit about which is in force: *multi-model* runs (different LLMs per run) yield cross-model agreement — independent raters in the classical sense; *single-model stochastic* runs yield a stability measure only. The production configuration rotates three independent open-weights checker models, all served locally (no transcript text leaves institutional infrastructure).

A `null` primary stage is a **valid abstention ballot** ("no VAAMR stage expressed" — logistics, small talk, off-topic discussion), not an error. This matters more than it appears: roughly a third of human-coded participant items are "no code," so a classifier that cannot abstain is wrong on that share of the human axis by construction (§3.5). Ballots aggregate by majority vote into four agreement levels (unanimous/majority/split/none); split and none are automatically flagged for human review. A confidence tier (high/medium/low) integrates cross-run consistency with per-run confidence, because a confidently wrong model is stably confidently wrong — neither signal suffices alone.

### 2.3 PURER classification at the cue-block level

Therapist dialogue is classified at the **cue-block** level — the therapist's full contribution between two consecutive participant turns — using a wider context window, with empirical precedence rules resolving co-occurring moves (Reinforcement is often a wrapper around a substantive inner move; Utilization outranks Reframing for forward-application prompts). For every within-session VAAMR transition, the pipeline records the FROM → CUE → TO triple: the participant utterance before (stage X), the full intervening therapist contribution (PURER move), and the participant utterance after (stage Y). The triple is the atomic unit of mechanism analysis (§3.7): it asks *what does the therapist do, exactly, when participants make a particular kind of stage transition* — at the resolution where technique actually operates, not at session-aggregate correlation.

Human validation of PURER labels (target: Krippendorff's α ≥ 0.70 between human coders, appropriate for these lower-inference behavioral codes) is **in progress and not yet complete**; every therapist-side result in this paper is therefore directional, and the pipeline's own reports tier them as such.

### 2.4 Provenance hierarchy and the auditable store

Every segment's final label resolves through a strict provenance hierarchy:

> adjudicated > human-consensus > LLM-zero-shot > gated LLM-free fills (probe, graph)

Human adjudication permanently outranks machine labels. Two cheap LLM-free scalers (§3.6) sit *below* the LLM and can only fill segments the LLM has not labeled — each gated behind its own out-of-sample reliability check, each abstention-aware, and neither able to override an LLM or human label. Raw per-rater ballots, prompts, model identities and versions, and human codes persist in a single auditable store; reliability evidence regenerates automatically whenever machine labels change, so the agreement statistics never silently go stale relative to the labels they describe. The result is a corpus in which every claim about any segment can be traced to its source and its level of human verification.

### 2.5 What accumulates

Beyond per-cohort reports, the pipeline's byproduct is a durable research asset: a systematically labeled, human-anchored corpus of phenomenological stage expression — by trial's end, to our knowledge the largest such corpus in any mindfulness-based pain intervention — supporting replication, supervised fine-tuning, and cross-trial comparison (§6.3).

---

## 3. Validation Framework and Pilot Results

The application of LLM classification to phenomenological constructs requires validation appropriate to a distinctive epistemic situation: third-person computational tools applied to first-person reports, where ground truth is constitutively inaccessible to the classifier and only partially accessible to human coders. We implement the text-psychometrics framework of Low, Mair, Nock, and Ghosh (2024), mapping each classical psychometric requirement onto an operational instrument, and we report the pilot results for each. All chance-corrected statistics are computed with canonical third-party libraries (Cohen's κ via scikit-learn; Fleiss' κ via statsmodels; Krippendorff's α via the `krippendorff` package), so every number is auditable against its textbook definition.

**Pilot corpus.** Move-MORE Cohorts 1 and 2 complete (16 sessions; 14 unique participants — 5 in Cohort 1 and 10 in Cohort 2, one of whom attended sessions in both cohorts); Cohort 3 in progress (3 of 8 sessions ingested at the time of this analysis, contributing 6 further participants with classified segments; a seventh Cohort-3 participant appears in one short segment that does not yet carry a label of record and is excluded). Analyzed corpus: 20 participants, 19 sessions, **205 participant segments** (the VAAMR unit), 544 therapist segments. Consensus confidence tiers: 34% high, 66% medium, <1% low.

### 3.1 Content validity: known-label adversarial test sets

Each VAAMR stage definition includes exemplar (clear), subtle (harder), and adversarial (boundary-confusable) utterances whose correct labels are fixed by the framework design. Running the classifier against this known-label set *before* processing real transcripts measures whether the LLM has internalized the definitions, and systematic misclassification of specific adversarial items identifies the exact stage boundaries needing definitional sharpening — converting operational ambiguity from an implicit classifier property into an explicit, testable artifact. The same instrument doubles as the calibration exercise human raters complete before blind coding.

### 3.2 The human reliability ceiling

The ultimate validation standard is agreement with expert human judgment — but that standard must itself be measured, because no machine can meaningfully agree with a human consensus more than the humans agree with each other. Four qualitative researchers blind-coded stratified samples of the pilot corpus (three frozen test sets; n = 31, 31, 14 items; content-hash-verified against the segments the machine actually labeled).

**Result (Table 2).** Human↔human Krippendorff's α = 0.473, 0.523, and 0.325 across the three test sets (Fleiss' κ 0.467/–/0.308; unanimous agreement 45%/61%/29%; pairwise Cohen's κ from 0.171 to 0.820 depending on the pair). The band α ≈ 0.33–0.52 is *moderate* agreement, with the weakest set below the pre-registered α ≥ 0.40 floor.

We emphasize what this is and is not. It is not a measurement failure; it is the signature of a construct with genuinely fuzzy ground truth — experienced phenomenologists, coding the same segments blind, disagree at this rate because adjacent stages (Attention Regulation vs. Metacognition; Metacognition vs. Reappraisal) shade into one another in natural speech. It has two consequences the methodology takes seriously. First, it drives the framework-refinement loop: recurring disagreement sites become definitional clarifications before the next cohort. Second, it **sets the ceiling for every machine statistic that follows**: "human-level" for this construct denotes moderate agreement, and a conventional κ ≥ 0.70 or 0.80 target would demand the machine exceed the humans whose judgment defines the construct — a target we regard as conceptually incoherent for fuzzy phenomenological categories, and one we accordingly do not adopt.

*Table 2 about here (reliability summary: human↔human per test set; human↔LLM overall, per test set, per model; LLM-free scalers).*

Agreement statistics are interpreted against bands pre-registered before any coding was scored: agreement at or above 75% raw with α ≥ 0.60 licenses computational classifications as **primary evidence**; α between 0.40 and 0.59 licenses **directional evidence only**, with any conclusion requiring convergence from at least two independent sources; below 0.40 triggers **framework refinement** before classifications are used at all. The observed human band itself falls in the directional tier — which is why the pipeline tiers every downstream claim, why curriculum recommendations are framed as falsifiable hypotheses for the next cohort rather than conclusions (§4), and why the refinement loop (definitional clarification at the recurring disagreement boundaries) is a designed part of the method rather than a failure response.

### 3.3 Human ↔ LLM: the consensus performs at the ceiling

Against the human consensus on the same frozen items (n = 66), the multi-run LLM consensus reaches **Cohen's κ = 0.537 [95% CI 0.389, 0.681]** — within, indeed at the top of, the human↔human band. Per test set: κ = 0.641, 0.508, 0.319 (the weakest mirroring the test set on which the humans themselves agreed least). Per individual model: gemma-4-31b κ = 0.540 [0.383, 0.685]; qwen3-next-80b κ = 0.522 [0.379, 0.665]; nemotron-3-nano-30b κ = 0.379 [0.235, 0.514] — the consensus mechanism buffers the weakest rater. Per-stage diagnostics show the residual difficulty concentrated where human difficulty also concentrates (Metacognition recall 0.40 at n = 5 support; Vigilance precision 1.00 but recall 0.25), and rare-stage support is thin — exactly the cells the human-review queue prioritizes.

The claim we make is precise and bounded: the LLM consensus achieves *human-level* reliability in the only sense available — it matches a moderately agreeing expert panel, rather than exceeding it — and is therefore an acceptable label of record, with 20% human blind-coding ongoing and adjudication permanently senior. By the pipeline's own pre-registered evidence tiers, labels in this band support *directional* evidence: conclusions require convergence from independent sources, which is how every downstream analysis treats them.

### 3.4 Justification grounding: auditing confabulation

Section 2.2 requires every classification to cite segment language; we audit the requirement rather than trust it. The grounding instrument extracts quoted spans from each justification and measures the share that occur verbatim (with conservative fuzzy fallback) in the segment text. On the pilot corpus, **78% of quoted spans are grounded** (93% of segments carry at least one grounded quote; 12% flagged to the review queue). Two caveats fix the interpretation: grounding bounds *confabulation*, not correctness — a faithfully quoted segment can still be mis-staged — and lexical matching scores honest paraphrase as ungrounded, so the figure is a lower bound on faithfulness and the flag set is a review queue, not an error rate. With those bounds, the audit converts the pipeline's auditability claim from an assertion into a reported, falsifiable number.

### 3.5 The leakage lesson and the refuted scaler (H5)

The original design placed the scaling burden on a graph neural network classifier distilled from the LLM consensus: once it reproduced the consensus to reliability on held-out segments, it would label new cohorts with no LLM cost. The pilot **refuted this at current scale**, and the manner of the refutation is itself a methodological contribution.

First, the leakage. Random k-fold cross-validation over a transcript graph leaks: a held-out segment's temporal-chain and similarity neighbors include same-participant segments whose labels are visible in training, so the model effectively sees the answer. On the pilot this inflated the gate nearly fivefold — random-fold κ ≈ 0.25 versus **participant-grouped** (whole participants held out) κ ≈ 0.05–0.14. We report the artifact because it is the kind that silently validates scaling decisions; participant-grouped cross-validation with participant-clustered bootstrap CIs is now the only protocol the pipeline accepts for any learned classifier, and we recommend it as a floor requirement for the field.

Second, the diagnosis. A linear probe on the same embedding features ties or beats the graph; pure graph smoothing is the worst arm; and adding construct-definition anchor structure *lowered* reliability. The two levers that did help were not graph machinery but measurement discipline: class weighting (recovering rare stages from 0% held-out recall) and the explicit abstention class (≈36% of human-coded items are "no code" — a fixed-five-class model is wrong on a third of the human axis by construction). The consensus-distillation classifier accordingly ships **off by default**, and the LLM consensus — already at the human ceiling and affordable at trial scale — remains the label of record and scaling engine.

Third, the honest residual. A pre-registered distillation campaign then established the current best LLM-free scaler: a per-rater ensemble (one class-weighted logistic probe per checker model, ensembled by mean probability) reaching classifier↔human κ = 0.450 [0.319, 0.599] and classifier↔LLM κ = 0.361 [0.281, 0.432] — dominating the single probe on both axes, with three unrelated methods converging on the same figures. Convergence across independent methods at the same sub-ceiling fidelity is the signature of a **data-limited, not method-limited**, ceiling: the binding constraint is the small number of labeled participants carrying the rare stages. The ensemble ships as an assistive, gated, abstention-aware *pre-labeler* that fills below the LLM and never overrides it, re-gated on each project's own human subset before it is permitted to scale.

### 3.6 Discriminant validity: stage is not topic (H6)

The graph's failure converts into the pilot's strongest positive finding. If a content-similarity model could recover VAAMR, the construct would be, in effect, a topic taxonomy. It cannot: on identical participant-grouped folds and identical embedding features, a supervised probe recovers the stage well above chance (human-axis κ = 0.365 [0.228, 0.513]) while a content-similarity classifier (label propagation along embedding neighborhoods) performs far worse (κ = 0.196 [0.117, 0.319]) — a paired contrast of **Δκ = +0.170 [0.002, 0.318]** on the human axis (+0.214 [0.150, 0.274] on the LLM axis). The geometry agrees: local embedding neighborhoods are stage-mixed (5-NN same-stage fraction 0.401), and unsupervised content communities are statistically independent of stage (community×stage ARI ≈ 0).

VAAMR stage, then, is a direction in language that is *recoverable by direct supervision yet orthogonal to content similarity* — present in the features, but not aligned with what makes utterances similar (topic, body region, affect). This is precisely what the framework claims of itself: the stages index a developmental re-habituation trajectory, not what a participant happens to be talking about. The result directly addresses the standing concern that reliable language markers for psychological constructs are elusive: the marker exists, but only a supervised reader finds it — which is also why naive similarity-based scaling fails. One scope condition is declared: the contrast is established on one embedding family (a robustness proxy on a much smaller encoder behaved differently, plausibly a capacity artifact), so we claim H6 on the tested embedding space, with the two-encoder generality test registered as the immediate next experiment.

### 3.7 The substantive worked example: progression and mechanism, honestly tiered

**Developmental progression (H1; validated labels; single-arm).** The primary, ordinal-safe test: per-session adaptive-stage occupancy (share of a participant's segments in Attention Regulation/Metacognition/Reappraisal), averaged across participants, tested for monotone trend by Mann-Kendall. Occupancy rises **44% (first session) → 100% (last session)**; τ = +0.714, Theil-Sen slope +0.051/session, **p = .019**. Sensitivity analyses are subordinate and reported as such: the interval-scale E[stage] mixed-effects slope is +0.06/session [95% CI −0.01, +0.13], p = .072; 11 of 14 participants with ≥2 sessions have positive slopes (exact sign test p = .057). Descriptively, **17/20 participants crossed the avoidance barrier** after first expressing Avoidance. The single-arm caveat is stated once and governs everything: with no control arm, "progression" means progression in coded language over sessions — a computational replication of the arc the published model predicts, not a treatment effect.

**Stage-moderated therapist effects (H2; directional; unvalidated PURER).** The program's central mechanism hypothesis is an *interaction*: a PURER move's association with the next participant stage should depend on the FROM stage — the same informational content delivered as Education should land differently for a Vigilance-stage participant (who lacks a conceptual frame) than the same content delivered as Reframing lands for an Avoidance-stage participant (whose "failure" it recasts as practice). Of 186 cue blocks in the corpus, 120 contain therapist speech and 96 carry a PURER label — coverage that is itself reported, not assumed. The primary estimator is a hierarchical ordinal model (TO_stage ~ FROM_stage × move + random participant intercepts), adjudicated by participant-grouped "earns-its-place" cross-validation. The pilot verdict is exact: the therapist move earns its place as a **main effect** (held-out log-loss 1.553 FROM-only → 1.514 additive) but the interaction does not (1.528); the additive-vs-interaction likelihood ratio is non-significant in-sample (p = .52) and more so under participant-cluster bootstrap calibration (p = .93); no per-cell association survives false-discovery correction. H2 is therefore **under-identified at this n — estimable and bounded, not confirmable** — and is presented as such rather than mined for significance. Every reported cell carries an E-value sensitivity bound (VanderWeele & Ding, 2017): the strongest cell's point E-value is 4.33, but its confidence-limit E-value is 1.86 and most cells' collapse toward 1.0 — the pilot can *bound* robustness but cannot *establish* it.

The deepest caveat is structural and named: therapists choose moves in response to participant state (confounding by indication). We state the identifying assumption explicitly — sequential ignorability of move choice given the participant's pre-cue state — assert that it is *violated* in a specific, nameable way (therapists respond to within-state difficulty that the FROM label does not capture), and instrument the violation rather than dismissing it. A dyadic transition model trained directly on the FROM→CUE→TO triples (predicting the following participant's stage mixture from the preceding mixture, stage, and the raw therapist-cue representation, with no similarity edges) provides a learned counterfactual: what shift would each move predict, independent of when therapists chose to deploy it? On the pilot it confirms under-identification (the cue does not improve held-out prediction over a FROM-only baseline) while its counterfactual ranking triangulates *positively* with the observed ranking (Spearman ρ ≈ +0.34 — a marked correction from an earlier, mis-specified per-segment counterfactual that *inverted* the ranking at ρ = −0.13 and was replaced). Most informatively, a confound-localization map of the signed divergence between observed and counterfactual signals shows **nine of twenty (FROM-stage × move) cells inverting in sign** — exactly the pattern expected when genuinely helpful moves are deployed at moments of participant difficulty, depressing their observed association. This is sensitivity analysis of a model, not causal estimation; it is also the difference between acknowledging a confound and mapping where it bites (Supplementary S2).

**Convergent validity (H4; pre-registered; no pilot result).** Whether participants whose coded-language trajectory advances also improve on independently measured clinical outcomes (pain NRS, TSK-11, ODI, MRPS, MAIA-2) is the bridge from language to anything clinically real. The trial's outcome data had not been joined to the corpus at the time of this analysis, so **H4 yields no pilot result here** — a property of data availability, not a null finding. The correlation directions are pre-registered in the public repository before any outcome data is seen, so the eventual test is confirmatory; and even a positive result is convergent-validity evidence for the language index, **not** efficacy, which the single-arm design cannot support.

**Cross-framework convergence (H3; deferred by design).** Whether VAAMR's developmental stages co-occur with the VCE phenomenological content the theory predicts is, in principle, a construct-validity test — and we explicitly decline to run it as one at present scale. When both frameworks are coded by the same LLM substrate on the same text, part of any co-occurrence reflects shared classifier lexicon rather than independent measurement ("notice," "observe," and "aware" trigger both VAAMR Metacognition and VCE meta-cognition codes), and a label-permutation control does not break that confound — the same text still drives both classifiers in the real assignment. The (stage × code) lift table is therefore reported as exploratory phenomenological characterization only; a defensible test requires an independent measurement substrate (a human-coded VCE subset, or a structurally different model) at larger n, and is registered as future work. Deferring a hypothesis the design cannot yet test cleanly is, we suggest, the correct use of the falsifiability discipline the hypothesis table imposes.

*Table 3 about here (hypothesis scoreboard: statement, instrument, falsification condition, pilot status).*

---

## 4. Deployment: The Curriculum-Modification Workflow

The practical test of the methodology is whether its outputs convert into accountable curriculum decisions inside the between-cohort window. The pilot's Cohort-3 cycle demonstrates the full loop.

The pipeline's synthesis report leads with reliability (how far each label class can be trusted), then validated participant-side outcomes, then directional therapist-side mechanism — and every recommendation is generated in a fixed accountable format: *Observation → Mechanism Hypothesis → Proposed Change → How to Assess in the Next Cohort*, with each element carrying its evidence tier and, for mechanism cells, its E-value bound. For example: only 17/20 participants crossed the avoidance barrier (observation, validated labels); forward movement out of Avoidance is most associated with Phenomenology inquiry (Δprogression +1.37 [1.06, 2.01], permutation p = .037 — directional, unvalidated PURER, confound-bounded); script deliberate use of that move in Avoidance-dominant debriefs rather than leaving it to improvisation (proposed change); re-compute barrier-crossing against the 17/20 baseline in the next cohort (assessment). The format makes recommendations falsifiable and their epistemic status legible to the protocol team performing MoSCoW prioritization, as specified in the trial's person-based design.

Every modification report closes with a validation-caveats section that is not boilerplate: it attaches the current agreement statistics to specific recommendations, identifying which rest on validated high-confidence classifications and which on directional computational evidence requiring next-cohort confirmation, so the curriculum team can weight each accordingly. The same discipline governs how the research team is brought to trust the system at all: walkthroughs lead with the line-by-line reliability dossier (machine reasoning beside each human coder's reasoning, per item), the flagged-for-review queue demonstrates that the system knows its limits, and the qualitative team is standing-invited to find a wrong label among the high-confidence set — an invitation that has proven more persuasive than any aggregate κ.

Two further features of the deployment deserve note. First, instructor-facing artifacts — per-session briefings generated for the Cohort-3 instructors from Cohort-1–2 evidence — close the loop the trial design intends: pilot-derived patterns become explicit, testable adaptations rather than informal impressions. Second, patient and public involvement is preserved rather than bypassed: every report is human-readable (coded transcripts with visible justifications, per-participant trajectories), so patient partners and the Community Advisory Board can audit cases against their own experience and contribute to MoSCoW prioritization. The pipeline accelerates and structures deliberation; it does not replace it.

---

## 5. Epistemological Limits

Five structural limits follow from the kind of inference QRA performs. No engineering improvement dissolves them; the method's defensibility depends on stating them and instrumenting what can be instrumented.

**5.1 Linguistic expression is not phenomenological state.** QRA classifies what participants *say*, not what they experience. In-session language reflects pain, social context, impression management, and the struggle to articulate processes that may lack ready linguistic form. This limit applies equally to human coders; the machine's disadvantage is the absence of clinical-contextual judgment, met by human validation as an integral process rather than a post-hoc check. The segmentation degrees of freedom are bounded (not eliminated) by the sensitivity analysis of §2.1.

**5.2 Naturalistic adjacency is not causal mechanism.** FROM→CUE→TO analysis is closer to mechanism than session-level correlation — it controls participant- and session-level confounders by operating within participant, within session, within minutes — but it remains unblinded, uncontrolled observation. It yields systematic corpus-level description of associations for expert interpretation, not prescriptions.

**5.3 Frameworks derived elsewhere constrain application here.** VAAMR was derived from MORE without movement integration; movement introduces kinesthetic vocabulary the original exemplars do not cover. VCE derives from long-term contemplatives. Poor empirical fit is treated as informative evidence about contextual adaptation, not noise.

**5.4 Therapist inquiry elicits what it then measures.** PURER inquiry is *designed* to direct attention: a Phenomenology question invites metacognitive-sounding language regardless of underlying state. VAAMR labels therefore characterize the *dyadic expression* of the interaction, not pure participant states — the deepest reason therapist-side mechanism claims remain hypothesis-generating. The pilot quantifies rather than waves at this: the identifying assumption (sequential ignorability given FROM-state) is stated explicitly, shown violated in a specific way (therapists respond to within-state difficulty the FROM label does not capture), and every mechanism association is reported with point and confidence-limit E-values as its robustness floor (§3.7). Bilateral classification is also the *response* to this limit: with both sides labeled, participant-initiated stage expression is distinguishable from therapist-elicited expression — stronger and weaker evidence of attainment, respectively.

**5.5 Computational phenomenology cannot replace phenomenological judgment.** The pipeline's output is hypotheses about phenomenological expression, produced by a system with no access to first-person experience. Human validation is not methodological fastidiousness; it is the epistemological requirement that distinguishes computational phenomenology from pattern-matching (cf. §1.2). The pipeline directs expert attention; it cannot determine what patterns mean or what should be done about them.

---

## 6. Discussion

### 6.1 What the methodology uniquely provides

Four capabilities are, to our knowledge, jointly unavailable from conventional qualitative methods, conventional process research, or single-framework computational classification: (1) per-utterance phenomenological staging of every participant utterance in every session of a full trial, with per-label confidence and justification — making stage distributions, longitudinal trajectories, and transition structure tractable at a resolution at which they were previously invisible; (2) mechanism evidence at temporal adjacency rather than session-level correlation, via the FROM→CUE→TO structure; (3) both sides of the therapeutic dyad labeled from the same input in the same pass, making dyadic analysis structurally consistent rather than reconciled post hoc; and (4) a cumulative, human-anchored labeled corpus as a durable by-product. All four arrive inside the between-cohort window that motivated the work.

### 6.2 The pilot's central lesson

The pilot's most consequential findings are the ones that corrected us. The reliability ceiling of a genuinely fuzzy phenomenological construct is moderate, and pretending otherwise — by adopting conventional κ targets the construct's own experts cannot meet — would have misrepresented both the machine and the construct. The defensible scaling substrate is a human-validated LLM consensus, not a content-similarity graph; the participant-grouped yardstick that exposed this (and the leakage that had masked it) is the same instrument that makes the conclusion trustworthy. And the graph's failure was not waste: it produced the discriminant-validity finding that VAAMR indexes a developmental trajectory orthogonal to topic — the construct behaving as its theory says it should, demonstrated by the failure of the model that assumed otherwise. We commend the general pattern: in computational qualitative research, negative results about one's own architecture, reported with the same precision as positives, are the strongest available evidence that the positives mean something.

### 6.3 The replication architecture and future work

The four-cohort design makes pilot-derived patterns prospectively testable as the trial proceeds: Cohorts 3–4 provide the confirmatory frame for the progression arc, for the (currently under-identified) stage-moderated mechanism, and for re-adjudicating whether any learned scaler earns the LLM-free role as labeled data grows. Registered next steps, in order: completion of PURER human validation (gating all therapist-side claims), with external construct validation of the resulting therapist-fidelity profiles against the MORE Fidelity Measure (Hanley & Garland, 2021); the pre-registered convergent-validity test against clinical outcomes when the trial's outcome data are joined; the two-encoder discriminant-validity generality test; and cross-trial extension to contrasting MBI corpora, where the methodology's portability — and the VA-MR model's generality beyond one protocol — becomes an empirical question rather than an assumption.

### 6.4 Conclusion

Chronic pain is, in Leder's terms, a dys-appearance of the lived body; mindfulness-based therapy is most coherently understood as structured re-habituation of the relationship between attention and that body. What this paper contributes is a way of *measuring* that re-habituation, utterance by utterance, at the tempo iterative trials require — with reliability claims sized to a measured human ceiling, validity claims tested adversarially against the method's own architecture, mechanism claims bounded by explicit sensitivity analysis, and first-person judgment retaining authority at every consequential node. The contribution is deliberately modest and, for that reason, defensible: not a machine that understands experience, but an instrument that lets phenomenological seriousness and trial-speed evidence coexist.

---

## Statements

**Ethics.** The Move-MORE trial is approved by the National University of Natural Medicine IRB (RW6625) and registered at ClinicalTrials.gov (NCT07125027). All participants provided informed consent, including consent to recording of session content. Transcripts were de-identified before analysis; all LLM inference ran on locally served models within institutional infrastructure.

**Data protection and responsible AI use.** Session recordings are the most sensitive class of trial data, and the pipeline treats confidentiality as a design constraint. Transcripts are de-identified at the earliest practical stage via a persistent speaker-anonymization map that replaces all direct identifiers with coded tokens before analysis, with the token-to-identity linkage held separately from the analytic corpus. Confidential transcripts and the working store (`qra.db`) reside only on access-controlled institutional infrastructure and are excluded from version control; the public repository contains only code, framework definitions, synthetic fixtures, and de-identified derived data. All language-model and embedding computation runs on locally hosted, open-weight models on institutional hardware — no transcript text, identifiable or de-identified, is transmitted to any external or cloud AI service, and no participant data trains or is retained by any vendor. The external-API path is disabled by default and requires deliberate reconfiguration plus a credential the project does not set. Classifications are advisory: they carry visible justifications and provenance tiers, are benchmarked against blind human coders, and are adjudicated by human coders, so AI output is never the sole basis for a participant-related conclusion.

**Data and code availability.** The complete pipeline, validation instruments, prompts, and analysis code are open source at [repository URL]; an archival snapshot is deposited at [Zenodo DOI]. De-identified derived data (labels, reliability tables, per-item dossiers) are available in the repository; session transcripts cannot be shared.

**Use of AI.** Large-language-model classification is the object of study and the instrument described. The manuscript was written by the authors; [adjust per venue policy: AI writing-assistance disclosure].

**Author contributions (CRediT).** WB: conceptualization, methodology, software, validation, formal analysis, data curation, writing — original draft. RSW: conceptualization, investigation, resources, clinical supervision, writing — review & editing, funding acquisition. The Move-MORE Research Team: investigation, data collection, human coding, review.

**Competing interests.** [None declared / to complete.]

**Funding.** [K12 support for RSW; to complete.]

**Acknowledgments.** We thank the patient partners, Community Advisory Board, and Move-MORE participants, the qualitative coding team (whose blind-coded judgments constitute the human anchor of every reliability claim), and D. M. Low and colleagues for the Text Psychometrics framework adapted here.

---

## Figures and Tables (assembly plan)

- **Figure 1.** The re-habituation arc: per-session adaptive-stage occupancy with participant-clustered bootstrap CIs and Mann-Kendall trend. *(source: `06_reports/00_fig1_rehabituation_arc.png`)*
- **Figure 2.** The dyadic mechanism map: FROM-stage × PURER-move Δprogression with CIs, tier-flagged directional. *(source: `00_fig2_dyadic_mechanism.png`)*
- **Figure 3.** Results dashboard: reliability band, trajectory panel, barrier crossing, tier composition. *(source: `00_fig3_dashboard.png`)*
- **Figure 4.** Reliability forest plot: human↔human, human↔LLM (consensus and per model), LLM-free scalers, with CIs against the human band. *(source: `06_reports/01_reliability/reliability_forest.png`)*
- **Table 1.** VAAMR stage definitions, canonical expressions, distinguishing criteria (from the framework markdown).
- **Table 2.** Reliability summary (κ/α with CIs and n; the ceiling argument in one table).
- **Table 3.** Hypothesis scoreboard: H1–H6 with instrument, falsification condition, and pilot status (incl. H3 deferred, H4 no-pilot-result, H5 refuted, H6 supported-Qwen-scoped).

**Supplementary materials map.** S1: full pipeline technical specification and capability inventory (monograph §4). S2: graph layer, distillation campaign, transition model and confound localization (monograph §8.5–8.6, §9.4). S3: IRR dossiers, per-item audit trails, confusion matrices (repository `04_validation/irr/`). S4: justification-grounding and segmentation-sensitivity instruments with full caveats (monograph §5.6, §9.1). S5: curriculum-modification report exemplar (Cohort-3 instructor brief).

---

## References

Chatzichristos, G. (2025). Qualitative research in the era of AI: A return to positivism or a new paradigm? *International Journal of Qualitative Methods, 24*, 16094069251337583.

Garland, E. L. (2024). *Mindfulness-Oriented Recovery Enhancement: An evidence-based treatment for chronic pain and opioid use*. Guilford Press.

Garland, E. L., Hanley, A. W., Nakamura, Y., et al. (2022). Mindfulness-Oriented Recovery Enhancement vs supportive group therapy for co-occurring opioid misuse and chronic pain: A randomized clinical trial. *JAMA Internal Medicine, 182*, 407–417.

Giorgi, A. (1985). *Phenomenology and psychological research*. Duquesne University Press.

Hanley, A. W., & Garland, E. L. (2021). The Mindfulness-Oriented Recovery Enhancement Fidelity Measure (MORE-FM): Development and validation of a new tool to assess therapist adherence and competence. *Journal of Evidence-Based Social Work, 18*(3), 308–322.

Hayes, S. C., Wilson, K. G., Gifford, E. V., Follette, V. M., & Strosahl, K. (1996). Experiential avoidance and behavioral disorders. *Journal of Consulting and Clinical Psychology, 64*, 1152–1168.

Leder, D. (1990). *The absent body*. University of Chicago Press.

Lennon, R. P., Fraleigh, R., Van Scoy, L. J., et al. (2021). Developing and testing an automated qualitative assistant (AQUA) to support qualitative analysis. *Family Medicine and Community Health, 9*(Suppl 1), e001287.

Lindahl, J. R., Fisher, N. E., Cooper, D. J., Rosen, R. K., & Britton, W. B. (2017). The varieties of contemplative experience: A mixed-methods study of meditation-related challenges in Western Buddhists. *PLOS ONE, 12*, e0176239.

Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text psychometrics: Assessing psychological constructs in text using natural language processing. *Psychological Methods*.

Merleau-Ponty, M. (1962). *Phenomenology of perception* (C. Smith, Trans.). Routledge. (Original work published 1945)

Ramstead, M. J. D., Seth, A. K., Hesp, C., et al. (2022). From generative models to generative passages: A computational approach to (neuro)phenomenology. *Review of Philosophy and Psychology, 13*, 829–857.

Schmidt, F., Hammerfald, K., Jahren, H. H., & Vlassov, V. (2025). CFiCS: Graph-based classification of common factors and microcounseling skills. *arXiv preprint* arXiv:2503.22277.

VanderWeele, T. J., & Ding, P. (2017). Sensitivity analysis in observational research: Introducing the E-value. *Annals of Internal Medicine, 167*(4), 268–274.

Varela, F. J. (1996). Neurophenomenology: A methodological remedy for the hard problem. *Journal of Consciousness Studies, 3*, 330–349.

Wexler, R. S., Balsamo, W., Fox, D. J., et al. (2026). "Noticing the way that I'm noticing pain": A qualitative analysis of therapeutic progression in Mindfulness-Oriented Recovery Enhancement for patients with lumbosacral radicular pain. *Mindfulness, 17*, 819–833.

Wexler, R. S., Balsamo, W., Lendof, V., et al. (in review). Development and pilot feasibility testing of Move-MORE: A multicomponent mindfulness-and-movement intervention for lumbosacral radicular pain.

Wexler, R. S., Fox, D. J., ZuZero, D., et al. (2024). Virtually delivered MORE reduces daily pain intensity in patients with lumbosacral radiculopathy: A randomized controlled trial. *Pain Reports, 9*(2), e1132.

---

## Pre-submission checklist (remove before submission)

- [ ] Co-author pass (Wexler + team); confirm author list, affiliations, and the Move-MORE Research Team consortium listing.
- [ ] Insert repository URL + Zenodo DOI (Statements + ref placeholder); mint the Zenodo DOI from the submission tag.
- [ ] Build Tables 1–3 (sources: `frameworks/VAAMR_FRAMEWORK.md`; `04_validation/irr/irr_results.json`; methodology.md §3.4); export Figures 1–4 from `data/MMORE_Processed/06_reports/`.
- [ ] Re-verify every statistic against the frozen as-submitted run (`00_RESULTS.txt` generated 2026-06-10); if the corpus changes before submission, regenerate and re-sync. Sync the monograph's §3.4 H1 paragraph (older C1–C2 numbers) to the same run.
- [ ] Quoted-utterance de-identification check on Table 1 canonical expressions and any in-text quotes before public preprint.
- [ ] Venue formatting: IJQM (primary) or BMC Medical Research Methodology (alternate) — adjust abstract structure, reference style, AI-use statement wording per venue policy.
- [ ] Post PsyArXiv preprint on submission day; record DOI in `references/varela-2026.txt` (ref 7) and ROADMAP §3.2 checklist.
