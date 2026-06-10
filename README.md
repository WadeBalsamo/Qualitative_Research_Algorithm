# QRA: Qualitative Research Algorithm

**A clinical NLP system that discovers which language patterns predict therapeutic breakthrough moments. Built for iterative clinical trial designs**
 
---

QRA ingests diarized therapy session recordings, classifies every participant and therapist utterance against two independent phenomenological frameworks, then indexes the entire labeled corpus by stage transition to surface the exact language patterns present when participants cross clinically meaningful thresholds. Every instance of every transition type across the full trial corpus is extracted with the patient's own words on both sides and the therapist's contribution in between, creating a structured, searchable evidence base for curriculum refinement that conventional qualitative methods cannot produce at this scale or speed.

The pipeline runs locally on confidential clinical data (no cloud required), deploys in production on the Move-MORE Feasibility Trial at NUNM, and is generating results for two first-author publications in preparation.

**Why it matters clinically:** Between-cohort curriculum refinements in iterative feasibility trials are normally made on clinical intuition and aggregate outcome scores because full qualitative analysis takes months and the refinement window is weeks. QRA dissolves that constraint by producing per-utterance stage classifications, session-level transition matrices, per-participant longitudinal trajectories, and therapist cue-response language patterns within days of session completion.

---

## Table of Contents

- [Published Research That Made This Possible](#published-research-that-made-this-possible)
- [Engineering Design](#engineering-design)
- [The Two Classification Frameworks](#the-two-classification-frameworks)
- [What QRA Discovers: The Analysis Layer](#what-qra-discovers-the-analysis-layer)
- [Pipeline Architecture](#pipeline-architecture)
- [On-Disk Layout](#on-disk-layout)
- [Module Map](#module-map)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Validation Architecture](#validation-architecture)
- [Key Design Invariants](#key-design-invariants)
- [Installation & Quick Start](#installation--quick-start)
- [Further Reading](#further-reading)
- [Citations & Publications](#citations--publications)

---

## Published Research That Made This Possible

This pipeline operationalizes two frameworks from peer-reviewed research I co-authored:

### The VA-MR Framework (Published in *Mindfulness*)
**Wexler, R. S., Balsamo, W.,** Fox, D. J., ZuZero, D., Parikshak, A., Kwin, S., Ramirez, J., Thompson, A. R., Carlson, H. L., Kern, T., Mist, S. D., Bradley, R., Zwickey, H., Pickworth, C. K., & Garland, E. L. (2026). "Noticing the way that I'm noticing pain": A qualitative analysis of therapeutic progression in Mindfulness-Oriented Recovery Enhancement for patients with lumbosacral radicular pain. *Mindfulness*, 17, 819–833. DOI: [10.1007/s12671-026-02782-1](https://doi.org/10.1007/s12671-026-02782-1)

This paper derived the five-stage Vigilance–Avoidance–Attention Regulation–Metacognition–Reappraisal (VAAMR) model from thematic analysis of thirty MORE therapy sessions. Every VAAMR classification in this pipeline — every stage label, every operational definition, every exemplar and adversarial utterance — originates from empirical qualitative research published in a top-tier clinical psychology journal.

### The Move-MORE Feasibility Trial (Under Review)
**Wexler, R. S., Balsamo, W.,** Lendof, V., et al. (in review). Development and pilot feasibility testing of Move-MORE: A multicomponent mindfulness-and-movement intervention for lumbosacral radicular pain. *Research Square*. DOI: [10.21203/rs.3.rs-8682836/v1](https://doi.org/10.21203/rs.3.rs-8682836/v1)

This trial is the primary deployment context for QRA. The four-cohort iterative design — Cohorts 1–2 complete, human validation in progress — is the engineering problem the pipeline was built to solve.

### The Full Methodology Paper (in preparation)
[`methodology.md`](docs/methodology.md) — *Phenomenology at Trial Speed: A Computational Mixed-Methods Pipeline for Iterative Refinement of Mindfulness-Movement Therapy in Chronic Pain* — is a methodology paper (Balsamo, Wexler et al., *in preparation*) housed directly in this repository. It provides the neurophenomenological theoretical grounding, the engineering rationale for every design decision, and the Text Psychometrics validation framework (Low et al., 2024) that QRA implements.

**If you want to understand *why* this pipeline works the way it does, read that paper.**

---

## Built for Scalable Production

**TL;DR:** This is not a Jupyter notebook. Beta is in production as a CLI application with layered architecture, frozen data checkpoints, multi-model consensus voting, embedding ensembles, hot-reloadable configurations, and a full validation test suite, running with local models on a real clinical trial with published academic outputs. This is designed to scale with new datasets for an extensive post-hoc analysis of MBIs after validation of classification results and analysis of found language pattern cue:response relationships on the current trial.

### Pipeline Design

| Engineering Dimension | How QRA Implements It |
|---|---|
| **Language pattern discovery** | The labeled corpus is indexed chronologically by session and speaker. Every stage transition is extracted as a FROM → CUE → TO triple: participant utterance before, therapist response during, participant utterance after. The full text of each triple is retrievable, grouped by transition type, across the entire trial corpus. |
| **Corpus-level statistical analysis** | State transition matrices (within-session and between-session), PURER × VAAMR conditional lift tables, and per-participant longitudinal trajectories are computed in pure pandas/numpy on the assembled master dataset — no LLMs involved. |
| **Layered architecture** | 8-stage data pipeline: ingestion → operationalization → classification → cross-validation → assembly → reporting. Each stage is independently runnable and gated by a frozen data boundary. |
| **Immutable data checkpoints** | Segmentation results are written once and frozen (the `segments` table in `qra.db`). Classifiers write to independent overlay tables. Re-running any classifier never touches frozen segments — immutable staging → refreshable marts. |
| **Classification infrastructure** | Multi-run LLM consensus voting (unanimous / majority / split / none, confidence-tiered) for VAAMR/PURER; embedding + LLM ensemble with weighted reconciliation for 54-code VCE codebook. Split-vote segments auto-flagged for human review. |
| **Backend abstraction layer** | A single `LLMClient` interface wraps any OpenAI-compatible endpoint: LM Studio (local GPU), OpenRouter, Ollama. Swap backends with a CLI flag. Optimized for local deployment on confidential clinical data. |
| **Hot-reloadable definitions** | VAAMR, PURER, and VCE codebook definitions live in human-editable Markdown files parsed at runtime. Researchers refine operational definitions between cohorts without touching Python. |
| **Auditability by design** | Every LLM call logged with full prompt and response. Every segment's final label carries provenance source (`adjudicated` / `human_consensus` / `llm_zero_shot`). Full reproducibility via serialized `qra_config.json`. |
| **Human-in-the-loop validation** | Stratified 20% blind-coding evaluation set with frozen human worksheets and refreshable AI answer keys. Four qualitative researchers blind-coding VAAMR; Krippendorff's α targets ≥ 0.60 (VAAMR) and ≥ 0.70 (PURER). |

### What is actively deployed

- **Published empirical foundation:** VAAMR framework published in *Mindfulness* (2026) — [full text](https://doi.org/10.1007/s12671-026-02782-1)
- **Active clinical trial:** Move-MORE Feasibility Trial, Cohorts 1–2 analyzed, human validation underway
- **Target metrics:** Krippendorff's α ≥ 0.60 (VAAMR), α ≥ 0.70 (PURER)
- **Two first-author methodology papers in preparation** (see [Publications](#citations--publications))

---

## The Two Classification Frameworks

QRA applies two frameworks bilaterally to the same transcript corpus, addressing orthogonal analytical questions. **VAAMR applies exclusively to participant segments. PURER applies exclusively to therapist segments.** These labels never cross.

### VAAMR — Five-Stage Participant Developmental Arc

The Vigilance–Avoidance–Attention Regulation–Metacognition–Reappraisal model characterizes where participants are in their therapeutic progression. Derived from thematic analysis of MORE sessions for chronic pain (Wexler, Balsamo et al., 2026) — published in *Mindfulness* ([DOI: 10.1007/s12671-026-02782-1](https://doi.org/10.1007/s12671-026-02782-1)).

| theme_id | Stage | Core Phenomenology | Canonical Expression |
|----------|-------|--------------------|---------------------|
| 0 | Vigilance | Attentional capture by pain; body as dysappearing obstacle | *"I can't stop thinking about the pain, it's all I can focus on."* |
| 1 | Avoidance | Attentional skill deployed for experiential escape | *"When the pain comes, I focus really hard on my breathing to push it away."* |
| 2 | Attention Regulation | Stable volitional presence with somatic experience | *"I kept bringing my attention back to the sensations, just staying with them."* |
| 3 | Metacognition | Reflexive observation of one's own mental processes | *"I noticed I was getting anxious about the pain, and I could just watch that anxiety."* |
| 4 | Reappraisal | Noematic transformation — pain decomposed into constituent sensations | *"It's interesting, when I really look at it, the 'pain' is actually many different feelings."* |

Full operational definitions (prototypical features, distinguishing criteria, exemplar / subtle / adversarial utterances, word prototypes) are in [`VAAMR_FRAMEWORK.md`](frameworks/VAAMR_FRAMEWORK.md), parsed at runtime into `ThemeDefinition` objects by `constructs/markdown_loader.py`.

### PURER — Five-Move Therapist Guided-Inquiry Framework

PURER classifies therapist contributions at the **cue-block level** (one label per therapist response between consecutive participant turns). Operationalizes the MORE guided-inquiry structure as a formal classification target.

| move_id | Code | Move | When to apply |
|---------|------|------|---------------|
| 0 | P | Phenomenological | Step-by-step elicitation of practice experience |
| 1 | U | Utilization | Prompting forward application to everyday life |
| 2 | R | Reframing | Repositioning participant's report as a MORE concept |
| 3 | E | Educate/Expectancy | Psychoeducation about pain/mindfulness + expectation-setting |
| 4 | R2 | Reinforcement | Selective affirmation of adaptive responses or insights |

Precedence rule when moves co-occur: Reinforcement is often a wrapper — code the inner substantive move; Utilization > Reframing for forward-application prompts; Reframing > Education when anchored to participant's story.

Full definitions in [`PURER_FRAMEWORK.md`](frameworks/PURER_FRAMEWORK.md).

### VCE Phenomenology Codebook — Multi-Label Enrichment

The 54-code Varieties of Contemplative Experience codebook (Lindahl et al., 2017) is applied to participant segments via an embedding + LLM ensemble. Six domains: Affective (13), Cognitive (9), Perceptual (6), Sense of Self (6), Social (5), Somatic (15).

Full codebook in [`PHENOMENOLOGY_CODEBOOK.md`](frameworks/PHENOMENOLOGY_CODEBOOK.md).

For the complete theoretical neurophenomenological grounding and research methodology behind all three frameworks, see [`methodology.md`](docs/methodology.md).

---

## What QRA Discovers: The Analysis Layer

The classification stages are the *labeling mechanism*. The research findings live in the `analysis/` module. None of the following involves LLMs except where explicitly noted (session summaries).

### Stage Superposition and Continuous Progression

`analysis/superposition.py`, `analysis/reports/superposition_report.py`

Each participant segment is assigned not just a single hard-argmax stage label but a full **stage-mixture vector** — a probability distribution over all five VAAMR stages — and a **continuous progression coordinate** (E[stage] = Σk·p_k). This superposition representation is the keystone of all downstream mechanism and efficacy analysis.

Mixture sources are used in priority order: GNN geometry (when the GNN layer is enabled) → LLM multi-run ballot distributions → secondary_stage two-point mixture. The **mixture entropy** (normalized Shannon H ∈ [0,1]) captures liminality: segments near a stage boundary express high entropy and are clinically significant as cusp states. A co-occurrence matrix of stage pairs by segment counts boundary-expressed segments across the corpus.

Outputs include `report_superposition.txt` (corpus-level mixture summary, cusp density by session, most-liminal exemplars) and the `segment_superposition.csv` machine-readable export.

### Therapeutic Breakthrough Moments: FROM → CUE → TO Extraction

`analysis/reports/transition_report.py`, `analysis/stage_progression.py`

Every participant utterance in the corpus is indexed by its chronological position in the session. For every within-session VAAMR stage transition — segment at stage X immediately followed by segment at stage Y — the pipeline extracts the tripartite structure:

- **FROM** — the participant utterance at stage X, with text, confidence, mixture, and timestamp
- **CUE** — all therapist segments whose `segment_index` falls strictly between FROM and TO in the same session, with their PURER move labels
- **TO** — the participant utterance at stage Y, with text, confidence, mixture, and timestamp

The result is a searchable corpus of every observed instance of every transition type (e.g., all Avoidance → Attention Regulation crossings across both cohorts), with actual participant language on both sides and the therapist's contribution in between. Researchers read what the breakthrough moment looks like in the patient's own words and see what the therapist said to get there. One example per (cohort, session) pair is extracted and sorted; the full collection is aggregated into LLM-synthesized portraits of what each transition type characteristically looks like across the corpus.

The **Avoidance → Attention Regulation** crossing is the clinically critical case. It marks where emerging attentional skill stops being deployed for pain suppression and is redirected toward open, investigative presence — the central developmental barrier identified in Wexler, Balsamo et al. (2026). Every instance of this crossing is extracted with the full FROM/CUE/TO triple; the PURER move distribution in those cue blocks is the primary evidence for curriculum recommendations.

### PURER → VAAMR Mechanism Dossier

`analysis/mechanism.py`, `analysis/purer_analysis.py`

The mechanism dossier answers *how* PURER therapist moves drive VAAMR stage progression with statistical inference — not bare counts.

**Conditional lift with inference:** For each (from\_stage, to\_stage) transition type, PURER move lift is computed with cluster-bootstrap confidence intervals (resampling whole participants to respect nesting), within-stratum permutation p-values (shuffling PURER labels within `from_stage` to hold base rates), Cramér's V effect sizes, and Benjamini-Hochberg FDR correction across the 25-cell PURER×VAAMR family. The marginal lift table is explicitly labeled "confounded" (therapists deploy moves *in response to* participant state); the from-stage-conditioned view is the headline.

**Δprogression analysis:** Each cue block is enriched with Δprogression (continuous change in `progression_coord` from FROM to TO), classified as forward (Δ > +0.15), stabilize (|Δ| ≤ 0.15), or regress (Δ < −0.15). Every Δprogression estimate carries a cluster-bootstrap CI, within-stratum permutation p, effect size, and FDR flag. A mixed-effects model (`Δprog ~ C(purer) + (1|participant)`) estimates per-move marginal effects with participant-level random intercepts.

**Avoidance barrier analysis:** Dedicated ranking of which therapist moves precede the Avoidance → Attention Regulation crossing, the clinically critical threshold.

**Liminality leverage:** High-entropy (liminal) segments are tested for elevated Δprogression — the hypothesis that cusp states are the mechanistically pivotal moments for intervention.

**GNN motif integration:** When the GNN layer is enabled, emergent cue motifs (clusters in therapist-language embedding space that cut across PURER categories) are included with their own Δprogression estimates and FDR flags, extending the dossier with data-driven constructs not pre-specified in PURER.

Outputs: `report_mechanism.txt`, `report_avoidance_barrier.txt`, and CSVs including `mechanism_delta_progression.csv` (with CI/p/effect/FDR columns), `mechanism_liminality.csv`, `mechanism_avoidance_barrier.csv`, `participant_trajectory_types.csv`, `mechanism_purer_mixed_effects.csv`.

### Therapist Move × Stage Transition: Conditional Lift

`analysis/purer_analysis.py`

The `CueBlock` data structure pairs every therapist response with its surrounding FROM and TO participant stages. For each (from\_stage, to\_stage) transition type, the analysis computes the distribution of PURER moves across all cue blocks of that type, then computes lift against the corpus-wide base rate, with an omnibus Cramér's V and chi-square association test per transition type.

The clinically actionable question this answers: *is Reframing overrepresented before Reappraisal transitions? Is Phenomenology inquiry the dominant cue for Avoidance → Attention Regulation crossings, or does Reframing do more work there?* The marginal lift (collapsed across from\_stage) is explicitly labelled confounded; the from-stage-conditioned mechanism dossier is the defensible headline.

Outputs: `purer_transition_profiles.csv`, `purer_vaamr_lift.csv` (with CI/p/effect columns), `purer_empty_cue_rates.csv`.

### Program Progression Summary (descriptive — not efficacy)

`analysis/efficacy.py`

A **descriptive, single-arm** summary of how participants' LLM-coded VAAMR language moves across sessions. It is **not** an efficacy estimate: there is no control arm, and the "outcome" is the coded language itself (also shaped by therapist prompting). Read as hypothesis-generating for human validation and Cohort 3–4 replication.

**Primary (ordinal-safe):** adaptive-stage occupancy (stages 2–4) by session + a rank-based **Mann–Kendall** monotonic trend — no equal-spacing assumption (VAAMR is ordinal).

**Secondary (sensitivity, interval-scale):** the continuous E[stage] progression coordinate + `mixedlm_trend` linear slope, explicitly flagged as treating the stages as equally spaced. Per-participant slope direction + sign test. p-values/CIs are shown but **flagged when underpowered** (small single-arm n).

**Convergent validity (exploratory):** when an external-outcomes CSV is present at `02_meta/outcomes.csv` (auto-detects wide pre/post or long per-session), within-program progression is correlated with measured clinical change — *convergent-validity evidence that the language index tracks something real, still not efficacy*. Integration path: REDCap exports are mapped to `02_meta/outcomes.csv`.

Outputs: `06_reports/02_outcomes/progression_summary.txt`, `03_analysis_data/efficacy/*.csv` (incl. `efficacy_summary.json`), `05_figures/program_efficacy.png`.

### State Transition Matrices

`analysis/stage_progression.py`, `analysis/reports/transition_report.py`

Two levels of transition analysis, answering different questions:

**Within-session:** For each (participant, session), the segment sequence is sorted by `segment_index` and adjacent stage pairs are tabulated — forward, backward, and lateral transition counts, with mean continuous Δprogression per transition type. Within-session transition matrices aggregate across all participants. This answers: *how fluid is stage movement within a session? What is the ratio of forward to backward transitions in Session 5 vs Session 2? Does Session 3 (Mindful Reappraisal content) actually produce more Reappraisal-stage segments than Session 1?*

**Between-session:** The modal VAAMR stage per (participant, session) represents that participant's dominant stage for the session. Between-session transition matrices compare dominant stages across consecutive session numbers. This answers: *are participants consolidating gains between sessions or reverting? At what session number does the group's modal stage shift forward?* A participant who briefly reaches Reappraisal in Session 2 but then shows it as their dominant stage by Session 5 has demonstrated the consolidation VAAMR predicts.

### Per-Participant Longitudinal Trajectories

`analysis/longitudinal.py`, `analysis/participant.py`

Per-participant reports track the dominant stage, continuous progression coordinate, and progression volatility across all attended sessions. Participant-level output now includes `continuous_progression_by_session`, `continuous_progression_trend` (OLS slope), `mean_superposition_entropy_by_session`, and longitudinal trajectory typology (stable-advancer, late-mover, oscillator, non-responder). Group-level summaries compute mean stage proportion and mean progression coordinate with CI bands by session number.

### Therapeutic Language Atlas

`analysis/reports/language_atlas.py`

The language atlas produces a curriculum-actionable guide to *what therapists actually say* when participants advance through VAAMR stages. For each top-ranked (FDR-significant) PURER move, emergent motif, and named coupling factor, the atlas renders FROM → CUE → TO exemplar blocks with mixture annotations and stage contexts. Emergent motifs — high-influence, low-PURER-purity language patterns that predict progression but map to no existing PURER category — are highlighted as candidates for new therapeutic constructs.

Output: `06_reports/03_mechanism/language_atlas.txt` (top forward AND backward/stalling patterns).

### GNN Representation and Discovery Layer

`gnn_layer/`

The GNN layer is a pure-PyTorch GraphSAGE network (no torch-geometric) over a **homogeneous segment graph** — every node is a transcript segment in the shared Qwen3-Embedding-8B space (reused from segmentation/VCE, no second model), connected by temporal-chain edges (FROM→CUE→TO in graph form) and kNN-similarity edges. On that substrate it plays two roles: a **discovery & triangulation** instrument that augments (never replaces) the LLM/embedding classifiers, and a **consensus-distillation classifier** that learns to reproduce the multi-run LLM majority-vote consensus from graph structure and — once gated — can label new segments with no LLM calls. It is an **analysis-time layer only**: frozen segments and `master_segments` are never mutated. It is **ON by default** (`config.gnn_layer.enabled=True`), GPU-preferred / CPU-safe (`device=None` auto-detects CUDA), and wraps every capability in its own try/except so one failure never aborts the rest. Artifacts go to `03_analysis_data/gnn/` (CSVs), `06_reports/07_gnn/` (reports), `05_figures/gnn_*.png` (figures), and `02_meta/gnn/` (weights + embedding cache). The full design record is `experiments/docs/design_decisions.md` + `experiments/docs/graph_experiments.md`; the as-built prose spec is `methodology.md` §8.5.

**Discovery & triangulation — five capabilities (A–E):**

- **A — Continuous superposition:** soft-VAAMR head (KL to the ballot mixture) + a scalar progression coordinate `E[stage]=Σk·pₖ`; the primary mixture source when enabled. → `gnn/segment_positions.csv`.
- **B — Cue motif discovery:** cue blocks pooled into GNN embeddings, clustered into emergent *motifs* that cut across the five PURER categories, scored for from-stage-conditioned forward-transition influence, and flagged when influential but low-PURER-purity (candidate new constructs). → `cue_motifs.csv`, `cue_block_assignments.csv`, `07_gnn/emergent_motifs.txt`.
- **C — Triangulation:** GNN-vs-LLM and GNN-vs-human Cohen's κ + GNN-geometry lift tables beside LLM-derived ones; GNN↔LLM is labeled *distillation fidelity*, GNN↔human is the load-bearing validity axis. → `07_gnn/triangulation.txt`, `gnn_vs_llm_lift.csv`.
- **D — Construct-signal ablation (opt-in):** remove a head (VCE/PURER), retrain, report Δ; plus the decisive VCE-on-VAAMR test (held-out κ with/without the VCE head). → `07_gnn/construct_signal.txt`, `vce_contribution.txt`.
- **E — Participant↔therapist coupling:** latent factors of cue language correlated with subsequent forward movement, named post-hoc against an inline CF/IC alliance lexicon (discovered, not imposed). → `coupling_factors.csv`, `07_gnn/coupling.txt`.

**Trustworthy LLM-free classifier (the scaling engine).** Before the graph may label of record, it must reproduce the consensus *out-of-sample*. The reliability gate (`gnn_layer/validation.py`) reports per-VAAMR-stage and per-PURER-move κ with a rare-stage recall floor (the over-smoothing safeguard) and a YES/NO "ready for LLM-free scaling?" verdict (`07_gnn/validation.txt`), persisted machine-readably (`gnn_gate.json`). Promotion to the `gnn_consensus` provenance tier engages **only** when an analyst sets `gnn_authoritative=True` *and* the persisted gate verdict passes. Around that backbone the layer adds opt-in, individually-measured trust mechanisms: typed therapist→participant *precipitates* edges, per-stage **abstention/deferral**, temperature **calibration + OOD** deferral, measured label **propagation**, and an inductive **scale-mode simulation gate**. Each is kept only if it earns its place on the gate's κ.

**Track B — dyadic FROM→CUE→TO transition model** (`gnn_layer/transition.py` + `confound.py`; default-on discovery). The *observed* Δprogression (`analysis/mechanism.py`) is the lead "what language progresses participants" evidence; Track B adds a learned response function. The original per-segment model-counterfactual (`gnn_layer/influence.py`) was **retired** — it inverted the observed ranking (ρ = −0.13) and was mis-specified for a process question — and **rebuilt** as a small dyadic transition model `TO_mixture ≈ f(FROM_mixture, FROM_stage, pooled raw-Qwen cue)` (no kNN, FROM-stage conditioned). Its counterfactual swaps the cue to each PURER centroid and reads the predicted shift in the *following* participant's E[stage], per move and per from-stage×move, with participant-clustered bootstrap CIs, **triangulated** against `mechanism.py` (now ρ ≈ +0.34), plus a confound-localization map. Sensitivity analysis of a model, **not causation**; at n≈32 the cue is under-identified, so `mechanism.py` leads. → `07_gnn/{transition_model,confound_localization}.txt`, `03_analysis_data/gnn/{transition_counterfactual,transition_per_move,confound_localization}.csv`.

**Track C — MindfulBERT training-set builder** (`process/assembly/mindfulbert_dataset.py`). The end-goal artifact: a versioned `(cue language → observed Δprogression)` dataset for fine-tuning MindfulBERT. Units are cue blocks; **primary labels are the observed Δprogression** (signed + direction) with per-example provenance (weakest-endpoint label-source tier, abstention flag, gate verdict). An optional, gate-gated, provenance-tagged **GNN-counterfactual "would-progress" augmentation channel** is retained **only if** a participant-grouped held-out ablation shows it helps — otherwise dropped. Ships with a datasheet (provenance mix, gate status, ablation result, n≈32 caveats). → `02_meta/training_data/mindfulbert_dataset.jsonl` + `mindfulbert_datasheet.{json,txt}`.

**Track D — subtext communities as routines** (`gnn_layer/communities.py`). A thresholded (cosine ≥ τ) cross-session segment-similarity graph is partitioned by **two independent algorithms** (Louvain + agglomerative hierarchical; agreement reported as adjusted Rand index), within-session community→community **routine transitions** are modeled, and **participant-bootstrap stability selection** suppresses/flags communities too fragile at n≈32. Survivors are named with TF-IDF terms, exemplar quotes, per-session prevalence, and cross-cohort drift. Discovery / hypothesis-generating; independent of the gate. → `subtext_communities.csv`, `subtext_community_transitions.csv`, `06_gnn/communities.txt`.

> All GNN influence/community outputs are **directional, not causal** (n≈32, single-arm, unblinded, plus the elicitation confound — `methodology.md` §9.2/§9.4), and every discovery must pass human review before becoming primary evidence.

### Inter-Rater Reliability Validation (`qra irr`)

`process/irr_import.py`, `analysis/irr_analysis.py`

The validation counterpart to the discovery layer. Researchers' blind coding of the frozen
validation worksheets is imported once (`qra irr import`) and kept in `qra.db` as ground truth;
`qra irr run` then compares it against the project's **current** machine labels (pulled live) across
three families: **Human↔Human** (per test-set, primary + secondary; the reference band), **Human↔LLM**
(consensus *and* each individual model), and **Human↔GNN** along two clearly-separated axes — the
honest **held-out** prediction (out-of-fold, never trained on that segment's own LLM label → independent
construct validity + the reliability gate) and the in-sample **distillation** overlay (the operational
default; its LLM agreement is *distillation fidelity*, never reported as validity). All chance-corrected
statistics come from **proven libraries** — Cohen's κ (scikit-learn), Fleiss' κ (statsmodels),
Krippendorff's α (`krippendorff`, the headline statistic since it tolerates missing ballots). Outputs are
a single report (`06_reports/01_reliability/irr_report.txt`) plus, under `04_validation/irr/`, the stats
(`irr_results.json`, `irr_pairwise.csv`), a ranked discrepancy list, confusion/agreement figures, and a
**line-by-line per-test-set dossier** (`irr_items_testset_<n>.txt`) showing each item's text beside the
human codes + reasoning, the LLM codes + justifications, the GNN held-out prediction, and the LLM↔GNN
consensus. IRR regenerates automatically during `qra analyze` whenever the models or graph have changed.
Full methodology: `methodology.md` §5.5.

---

## Pipeline Architecture

`process/orchestrator.py:run_full_pipeline()` sequences these stages:

```
Stage 1  — Transcript ingestion & semantic segmentation
Stage 2  — Construct operationalization (serialize framework definitions + content-validity test sets)
Stage 3  — VAAMR multi-run LLM classification (participant segments)
Stage 3b — VCE codebook classification (embedding + LLM ensemble, optional)
Stage 3c — PURER cue-block classification (therapist segments, optional)
Stage 4  — Cross-framework lift statistics (VAAMR × VCE co-occurrence, requires 3b)
Stage 5  — Validation set generation (stratified blind-coding sample)
Stage 6  — Dataset assembly (frozen segments + overlays → master_segments.csv)
Stage 7  — Report generation (coded transcripts, human forms, stats)
Stage 8  — Post-hoc analysis (trajectories, cue-response, figures) — also via `qra analyze`
```

### Stage 1 — Semantic Segmentation

`process/transcript_ingestion.py`, `process/llm_segmentation.py`

Sentence-transformer embeddings → windowed cosine-similarity curve → adaptive threshold (25th percentile of session's own distribution) → segment boundaries. Also respects silence gaps (default 1500 ms) and enforces min/max word counts (30/200 words, recursive split at midpoint).

Therapist and participant segments are separated at this stage. Therapist segments flow to PURER (Stage 3c) and appear as read-only preceding context in VAAMR prompts. They are never VAAMR-classified.

Boundaries are calculated with embeddings, and ambiguous-case LLM-assisted boundary refinements: `boundary_review`, `context_expansion`, `coherence_check`.

**Output:** → `segments` table in `qra.db` (frozen) — one row per segment, with `params_hash`/`segmenter_version`/`ingest_timestamp` columns; never rewritten upon adding new data or changing framework definitions/exemplars. Raw input files are preserved at `01_transcripts_inputs/`.

### Stage 3 — VAAMR Classification

`classification_tools/classification_loop.py`, `classification_tools/theme_llm/llm_classifier.py`

Each participant segment gets a structured JSON prompt: segment text + preceding context (capped at 300 words) + full VAAMR `ThemeFramework` definitions. Required response fields: `primary_stage`, `primary_confidence`, `secondary_stage`, `secondary_confidence`, `justification` (must cite specific segment language). `null` primary_stage = valid ABSTAIN ballot.

Stage definition order is randomized per run (temperature > 0) to reduce position bias.

`n_runs` calls per segment → majority vote → confidence tier. Agreement levels: `unanimous` / `majority` / `split` / `none`. Split/none → `needs_review=True`.

**Output:** → `theme_labels` table in `qra.db` (refreshable overlay)

### Stage 3b — Codebook Ensemble

`classification_tools/codebook_multilabel/embedding_classifier.py`, `classification_tools/codebook_multilabel/ensemble.py`

Two independent classifiers:
- **Embedding**: query/passage cosine similarity against code definitions + inclusive criteria + exemplars; two-pass procedure for recall on subtle phenomenology
- **LLM**: multi-label zero-shot with strict majority rule per code

Ensemble reconciliation: union (default, recall-optimized) or intersection (precision-optimized). Both component outputs stored separately for post-hoc comparison. Embedding scores weighted 0.6, LLM 0.4 in composite confidence.

GPU memory hand-off: reuses segmentation embedding model when model IDs match.

**Output:** → `codebook_labels` table in `qra.db` (refreshable overlay)

### Stage 3c — PURER Cue-Block Classification

Therapist response between consecutive participant turns = one cue-block, one PURER label. Context window: 6 preceding segments (vs 2 for VAAMR). Long didactic stretches (lesson content) can be skipped via `PurerCueConfig.skip_lesson_content`. Label propagated back to all constituent therapist segments.

**Output:** → `purer_labels` table in `qra.db` (refreshable overlay)

### Stage 4 — Cross-Validation Lift Statistics

`process/cross_validation.py`

Lift(stage, code) = P(code | stage) / P(code). Default thresholds: lift ≥ 1.5 and minimum 3 segments. Currently exploratory — computes empirical co-occurrence patterns only; VCE is an optional enrichment layer and no construct-validity claim rests on the lift table. A rigorous cross-framework construct-validity test (controlled for the shared-LLM-lexicon confound, on a human-validated VCE subset at larger n) is deferred to future work.

**Output:** → `cv_labels` table in `qra.db` (refreshable overlay), lift report

### Stage 6 — Assembly

`process/assembly/master_dataset.py`

Joins frozen `segments` + all overlay tables. Final label priority: `adjudicated > human_consensus > llm_zero_shot`. Provenance fully auditable per segment.

**Output:** `02_meta/training_data/master_segments.csv` (generated export the analysis layer reads). Coded transcripts → `04_validation/full_transcripts/`. Human classification forms → `04_validation/full_transcripts/`.

---

## On-Disk Layout

```
output_dir/
├── 00_index.txt
├── qra.db                        # SQLite store: segments, overlays, manifest, testsets
├── 01_transcripts_inputs/        # Raw input VTT/JSON copies (provenance)
├── 02_meta/
│   ├── auditable_logs/           # LLM prompts/responses/checkpoints
│   ├── codebook_raw/             # Embedding checkpoints
│   ├── training_data/            # master_segments.csv (export), BERT training data,
│   │                             #   mindfulbert_dataset.jsonl + datasheet (Track C)
│   ├── gnn/                      # GNN model checkpoint + segment embedding cache
│   │   ├── model/                # weights.pt + manifest.json
│   │   └── segment_embeddings.npz
│   └── speaker_anonymization_key.json
├── 03_analysis_data/             # Session stats, graphing CSVs, mechanism/efficacy/GNN outputs
│   ├── mechanism/                # Δprogression CSVs with CI/p/FDR columns
│   ├── efficacy/                 # Efficacy outcome CSVs
│   └── gnn/                      # GNN artifacts: segment_positions.csv, cue_motifs.csv,
│                                 #   gnn_gate.json, gnn_counterfactual_influence.csv (B),
│                                 #   subtext_communities.csv (D), etc.
├── 04_validation/
│   ├── flagged_for_review.txt
│   ├── human_coding_evaluation_set.csv
│   ├── content_validity/         # Content-validity test sets
│   │   ├── content_validity_test_set.jsonl
│   │   ├── content_validity_human_worksheet.txt   # frozen
│   │   ├── content_validity_definition_key.txt    # frozen
│   │   ├── content_validity_answer_key.txt        # refreshable
│   │   └── <named_cv_testset>/   # FROZEN named CV test sets
│   ├── full_transcripts/         # Session-level human-readable forms
│   │   ├── coded_transcript_<sid>.txt
│   │   └── human_classification_<sid>.txt
│   └── testsets/                 # Flat numbered validation test sets
│       ├── human_classification_testset_worksheet_N.txt  # frozen
│       └── AI_classification_testset_worksheet_N.txt    # refreshable
├── 05_figures/                  # PNG figures (incl. gnn_*.png)
└── 06_reports/                  # tiered, numbered human-readable reports
    ├── 00_RESULTS.txt           # start here — publication-core results brief
    ├── 00_fig{1,2,3}_*.png     # flagship thesis figures
    ├── 01_reliability/          # irr_report.txt, probe_validation.txt
    ├── 02_outcomes/  03_mechanism/  04_per_session/
    ├── 05_per_participant/  06_per_stage/  07_gnn/
    └── 08_methods.txt           # [M#] registry + caveats
```

> Internal pipeline data (segments, classification overlays, provenance manifest, testset metadata) lives in `qra.db`; the files below it are human-facing or generated exports.

---

## Module Map

| Path | Role |
|------|------|
| `qra.py` | CLI entry point |
| `process/orchestrator.py` | Stage sequencing, `run_full_pipeline`, standalone `stage_*` functions |
| `process/config.py` | `PipelineConfig` dataclass — single source of truth for all settings |
| `process/segments_io.py` | Frozen per-session segment I/O, `params_hash`, `load_segments_for_stage` |
| `process/classifications_io.py` | Overlay read/write, provenance manifest, `clear_overlay` |
| `process/reclassify_ops.py` | From-scratch reset (`--fresh`): clears per-framework checkpoints + overlay; shared by CLI + TUI |
| `process/_freeze.py` | Freeze enforcement (atomic write, SHA verification) |
| `process/transcript_ingestion.py` | `ConversationalSegmenter` — embedding-based semantic segmentation |
| `process/llm_segmentation.py` | LLM-assisted boundary refinement |
| `process/speaker_anonymization.py` | Persistent speaker ID mapping across runs |
| `process/speaker_filter.py` | Speaker inclusion/exclusion rules per classifier |
| `process/output_paths.py` | Single source of truth for all output directory paths |
| `process/legacy_migration.py` | Auto-migration: v2.0 → per-session segments; v2.5 → v3 directory layout; `migrate_jsonl_to_sqlite()` folds per-session/overlay JSONL into `qra.db` on the next run (old files moved non-destructively to `<output_dir>/_legacy_files/`); `upgrade_config_file()` fills defaults for newly-added config parameters in place |
| `process/cross_validation.py` | VAAMR × VCE lift statistics (hard + mixture-weighted/soft) |
| `process/anonymization_editor.py` | Speaker-key editor with full downstream cascade (`edit-anonymization` / TUI) |
| `process/setup_wizard.py` | Interactive configuration wizard (presets + 17-step custom walkthrough) |
| `process/assembly/master_dataset.py` | `assemble_master_dataset` |
| `process/assembly/human_forms.py` | Human classification forms, test set freeze/refresh |
| `process/assembly/content_validity.py` | Content-validity test set freeze/refresh |
| `process/assembly/coded_transcripts.py` | Per-session coded transcript writer |
| `process/assembly/stats_reports.py` | Per-transcript stats, cumulative report |
| `process/assembly/training_export.py` | Training data and theme definition export |
| `constructs/vaamr.py` | `get_vaamr_framework()` — thin wrapper over `frameworks/VAAMR_FRAMEWORK.md` |
| `constructs/purer.py` | `get_purer_framework()` — thin wrapper over `frameworks/PURER_FRAMEWORK.md` |
| `constructs/markdown_loader.py` | Parser: `frameworks/*_FRAMEWORK.md` → `ThemeFramework` / `ThemeDefinition` objects |
| `constructs/theme_schema.py` | `ThemeDefinition`, `ThemeFramework` dataclasses |
| `constructs/registry.py` | `load(name)` → `ThemeFramework`; name-to-path dispatch (cached) |
| `constructs/codebook/phenomenology_codebook.py` | 54 VCE codes, loaded from `frameworks/PHENOMENOLOGY_CODEBOOK.md` |
| `constructs/codebook/codebook_schema.py` | `CodeDefinition`, `Codebook` dataclasses |
| `constructs/codebook/markdown_loader.py` | Parses `frameworks/PHENOMENOLOGY_CODEBOOK.md` → `Codebook` |
| `classification_tools/theme_llm/llm_classifier.py` | Zero-shot VAAMR/PURER prompt construction and response parsing |
| `classification_tools/codebook_multilabel/embedding_classifier.py` | Sentence-transformer embedding-based codebook classification |
| `classification_tools/codebook_multilabel/ensemble.py` | Embedding + LLM ensemble reconciliation |
| `classification_tools/probe/probe_classifier.py` | **LLM-free VAAMR scaler** — per-rater ensemble (one class-weighted L2-LogReg probe per LLM rater, mean proba; single-probe fallback) + calibration/abstention; `train_probe`/`evaluate_probe`(gate)/`classify_with_probe` |
| `classification_tools/zeroshot_reporting.py` | `write_zeroshot_report` — graded `--test-zeroshot` content-validity report |
| `classification_tools/llm_client.py` | Backend abstraction (OpenRouter, Ollama, LM Studio, HuggingFace, Replicate) |
| `classification_tools/classification_loop.py` | Multi-run consensus voting with checkpointing |
| `classification_tools/majority_vote.py` | Ballot aggregation (unanimous/majority/split/none) |
| `classification_tools/data_structures.py` | `Segment` dataclass |
| `analysis/runner.py` | Post-hoc analysis orchestrator — sequences all analysis steps |
| `analysis/superposition.py` | Stage-mixture provider: GNN geometry → LLM ballots → secondary_stage fallback; mixture entropy, co-occurrence matrix |
| `analysis/mechanism.py` | PURER→VAAMR mechanism dossier with full statistical inference (CI, permutation p, FDR) |
| `analysis/stats.py` | Reusable inference toolkit: Wilson CI, cluster-bootstrap CI, permutation test, effect sizes, BH-FDR, mixed-effects |
| `analysis/efficacy.py` | Descriptive progression summary (ordinal-safe, single-arm) + convergent-validity external outcome linkage |
| `analysis/purer_analysis.py` | PURER × VAAMR conditional lift table, cue-response synthesis |
| `analysis/purer_figures.py` | PURER × VAAMR lift heatmap and figures |
| `analysis/longitudinal.py` | Longitudinal summary generation |
| `analysis/stage_progression.py` | Session-level stage progression computation |
| `analysis/participant.py` | Per-participant report generation |
| `analysis/session.py` | Per-session analysis |
| `analysis/theme.py` | Per-theme (VAAMR stage + code) analyses |
| `analysis/figure_data.py` | Export graph-ready CSV datasets |
| `analysis/figures.py` | Matplotlib visualization figures (including superposition and mechanism figures) |
| `analysis/exemplars.py` | Exemplar utterance extraction per stage |
| `analysis/reports/superposition_report.py` | Superposition text report: corpus mixture, cusp matrix, liminal exemplars |
| `analysis/reports/language_atlas.py` | Therapeutic language atlas: ranked exemplar FROM→CUE→TO blocks by PURER/motif/factor |
| `analysis/reports/transition_report.py` | Transition explanation and therapist cue reports |
| `analysis/reports/` | Full suite of text report generators: session, stage, transition, cue-response, longitudinal, summaries |
| `gnn_layer/runner.py` | GNN layer entry point — orchestrates all five GNN capabilities |
| `gnn_layer/embeddings.py` | Qwen3 segment embedding reuse with NPZ cache |
| `gnn_layer/graph_builder.py` | Heterogeneous graph construction: temporal chain, anchor/label, kNN, cross-framework edges |
| `gnn_layer/model.py` | Pure-PyTorch GraphSAGE with multi-task heads (soft_vaamr, progression, vce, purer) |
| `gnn_layer/train.py` | Full-batch training, early stopping, checkpoint export |
| `gnn_layer/inference.py` | Per-segment position inference, cue-block embedding assembly |
| `gnn_layer/motifs.py` | Cue motif clustering, influence scoring, purity annotation, emergent-motif flagging |
| `gnn_layer/gnn_lift.py` | GNN-derived lift tables (VAAMR×VCE, PURER×transition) vs LLM baseline |
| `gnn_layer/triangulation.py` | GNN↔LLM↔human agreement (Cohen's κ), triangulation report |
| `gnn_layer/ablation.py` | Construct-head ablation: which families carry independent signal? |
| `gnn_layer/coupling.py` | Latent participant↔therapist coupling factors, CF/IC alliance naming (Capability E) |
| `gnn_layer/soft_labels.py` | Ballot-to-mixture conversion, progression coordinate, soft target assembly |
| `gnn_layer/anchors.py` | Optional construct-anchor features + similarity/lift edges (opt-in, human-axis ablated) |
| `gnn_layer/calibration.py` | Temperature scaling + ECE + OOD score for domain-shift confidence (Track A3) |
| `gnn_layer/propagation.py` | Measured post-training soft-label diffusion (Track A4) |
| `gnn_layer/transition.py` / `confound.py` | **Track B** — dyadic FROM→CUE→TO transition model + counterfactual triangulation vs `mechanism.py` + confound-localization map (default-on discovery; replaced the retired `influence.py`) |
| `gnn_layer/communities.py` | **Track D** — subtext-similarity graph, two-algorithm communities, routines, stability selection |
| `gnn_layer/reports.py` | GNN artifact writers: segment_positions.csv, cue_motifs.csv, gnn_vs_llm_lift.csv, coupling reports |
| `gnn_layer/validation.py` | Out-of-sample reliability gate (per-stage/per-move κ, rare-stage floor), scale-mode sim, persisted gate verdict |
| `gnn_layer/config.py` | `GnnLayerConfig` dataclass (enabled=True default; all Track A–D flags) |
| `process/assembly/mindfulbert_dataset.py` | **Track C** — MindfulBERT (cue language → observed Δprogression) dataset builder + datasheet |

---

## CLI Reference

QRA exposes a single entry point, `qra.py`, with the subcommands below. Running it with **no subcommand launches the interactive TUI** (`python qra.py`), which reaches every capability listed here through guided menus.

```bash
# Interactive TUI (no subcommand) and setup wizard
python qra.py                                              # menu-driven TUI
python qra.py setup                                        # config wizard → qra_config.json

# Full pipeline run
python qra.py run --config ./data/output/02_meta/qra_config.json

# Run with inline overrides
python qra.py run --transcript-dir ./data/input --output-dir ./data/output \
  --backend openrouter --model qwen/qwen-3-70b

# Pipeline + auto-generate analysis reports
python qra.py run --config ./qra_config.json --auto-analyze

# Incrementally add new transcripts to an existing project (frozen segments/testsets untouched)
python qra.py add-data --config ./qra_config.json
python qra.py add-data --config ./qra_config.json --classify-mode probe  # label new segments LLM-free (llm|probe|gnn)

# Post-hoc analysis on completed output
python qra.py analyze --output-dir ./data/output/          # full analysis suite
python qra.py analyze -o ./data/output/ --gnn              # force-enable the GNN layer
python qra.py analyze -o ./data/output/ --no-gnn           # force-disable the GNN layer

# Modular stage execution (Phase 3)
python qra.py ingest -o ./data/output/                     # segment + freeze only
python qra.py ingest -o ./data/output/ --fresh             # re-segment every session from scratch
python qra.py classify -o ./data/output/ --what vaamr      # VAAMR (participant)
python qra.py classify -o ./data/output/ --what purer      # PURER (therapist)
python qra.py classify -o ./data/output/ --what codebook   # VCE phenomenology
python qra.py classify -o ./data/output/ --what cross-validation  # VAAMR × VCE lift overlay
python qra.py classify -o ./data/output/ --what all        # every configured classifier
python qra.py classify -o ./data/output/ --what vaamr --fresh    # re-classify FROM SCRATCH
python qra.py assemble -o ./data/output/                   # join frozen + overlays
python qra.py validate -o ./data/output/                   # refresh human/AI validation artifacts

# Probe — RECOMMENDED LLM-free, gated, abstention-aware VAAMR scaler (per-rater ensemble; §8.6)
python qra.py probe train -o ./data/output/                # fit on LLM/human label of record + participant-grouped gate
python qra.py probe status -o ./data/output/               # gate verdict: probe↔human / probe↔LLM κ
python qra.py probe classify -o ./data/output/             # LLM-free FILL of unlabeled participant segments (gated; abstains; probe_consensus tier)

# GNN consensus layer (modular — add the GNN to an LLM-only project, then scale)
python qra.py gnn train -o ./data/output/                  # train graph + run the reliability gate
python qra.py gnn status -o ./data/output/                 # κ(graph,LLM); ready for LLM-free scaling?
python qra.py gnn classify -o ./data/output/               # LLM-free label new/unlabeled segments

# Import a legacy (pre-SQLite, JSONL) project into qra.db
python qra.py migrate -o ./data/output/                    # preview what would be imported
python qra.py migrate -o ./data/output/ --run             # perform (originals → _legacy_files/)

# Fix a single classification run without redoing the others
python qra.py reclassify-run -o ./data/output/ --run 3 --model nvidia/nemotron-3-nano-30b

# Zero-shot content-validity test (skips full pipeline)
python qra.py run --test-zeroshot --preset small --output-dir ./data/output/

# Validation test set management
python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1
python qra.py testset refresh -o ./data/output/ --all
python qra.py testset list -o ./data/output/

# Content-validity test set management
python qra.py cv create -o ./data/output/ --framework purer --name cv_purer_v1
python qra.py cv refresh -o ./data/output/ --all
python qra.py cv list -o ./data/output/

# Inter-rater reliability (Human↔Human, Human↔LLM, Human↔GNN) — bare `qra irr` = TUI
python qra.py irr import -o ./data/output/ --csv data/irr/human_coded_testsets.csv
python qra.py irr run -o ./data/output/                     # pull live LLM+GNN, compute, write report
python qra.py irr list -o ./data/output/

# PHI / speaker anonymization curation
python qra.py apply-anonymization -o ./data/output/        # retroactively scrub names from frozen text
python qra.py edit-anonymization -o ./data/output/         # interactive speaker-key editor + cascade
python qra.py edit-anonymization -o ./data/output/ --rename therapist_1=therapist_9 --yes
```

> **GNN scale mode** (`gnn classify`) and **GNN-authoritative labels** are recommended only after the graph passes its out-of-sample reliability gate — check it with `qra gnn status` (or `06_reports/07_gnn/validation.txt`). Until then the default flow remains LLM classification. A trustworthy gate needs multi-run LLM ballots (`n_runs >= 3`); `gnn train` warns when they are missing. See the [GNN Representation and Discovery Layer](#gnn-representation-and-discovery-layer) section above and `methodology.md` §8.5.

> **Track C/D outputs are config-flag driven** during `analyze` (no separate subcommand): set `gnn_layer.build_mindfulbert_dataset=true` (+`augmentation_enabled`) for the MindfulBERT dataset, and `gnn_layer.subtext_communities=true` for routine discovery. **Track B** (the dyadic FROM→CUE→TO transition model) runs **by default** — no flag (it replaced the retired `counterfactual`/`influence.py` path). See [Configuration](#configuration) and `experiments/docs/graph_experiments.md`.

### Interactive TUI

Running `python qra.py` with **no subcommand** launches a menu-driven TUI that surfaces every pipeline stage, validation tool, and editor through guided prompts, continuously displaying project state (which stages have run, testset counts, and the GNN's reliability status).

**Main menu:** `1` New Project (setup wizard → optional full run) · `2` Open Project · `3` About/Help · `0` Exit.

**Open Project menu** (actions adapt to detected state; ✓ marks completed stages; the GNN tag shows the live reliability-gate status). Items are grouped by workflow:

| Group | Actions |
|-------|---------|
| **Build** | *Ingest & freeze segments* (offers re-segment from scratch) · *Add new transcripts (incremental)* |
| **Classify** | *VAAMR* (zero-shot + re-run-from-scratch) · *PURER* (reclassify sub-menu) · *VCE codebook* · *Cross-validation* (VAAMR × VCE lift) |
| **GNN** | *Train / update GNN consensus layer* (also adds the GNN to an LLM-only project; warns on missing multi-run ballots) · *Classify — Graph consensus (LLM-free)* · *View GNN reliability / κ status* |
| **Assemble / Analyze** | *Assemble master dataset* → `master_segments.csv` + coded transcripts + validation artifacts · *Analysis & reports* (prompts to force the GNN on/off) |
| **Validation** | *Testset management* · *Content-validity testsets* · *Refresh validation artifacts* |
| **Maintenance** | *Edit configuration* · *Edit speaker anonymization key* · *Migrate legacy JSONL → qra.db* (shown only when a legacy project is detected) |

The **New Project** wizard offers three modes — *Small/Test*, *Production*, *Custom* — and walks through paths, speaker keys, PHI anonymization, segmentation, LLM backend + per-run checker models, frameworks, the VCE codebook, classification/confidence parameters, validation + content-validity testsets, summaries, and the **GNN layer** (label mode, reliability-gate target, authoritative toggle, ablation, motif/factor counts).

---

## Operating an Ongoing Project

QRA is built for longitudinal studies that accrue sessions over months. Every stage
is modular and runs from both the CLI and the TUI, so you can grow and re-derive a
project without ever re-segmenting frozen data by accident.

**Add a new batch of transcripts**
```bash
python qra.py add-data --config ./data/output/02_meta/qra_config.json
```
Segments + classifies only the *new* sessions (using the manifest-pinned config),
re-assembles, and re-analyzes. Frozen segments, frozen validation testsets, and
content-validity worksheets are never disturbed; newly-discovered speakers are
walked through interactively to extend the anonymization key.

**Re-derive a framework from scratch** (e.g. after changing models or `n_runs`)
```bash
python qra.py classify -o ./data/output/ --what vaamr --fresh   # clears VAAMR ckpts + overlay, then re-runs
python qra.py ingest   -o ./data/output/ --fresh                # rebuild frozen segmentation itself
```
Without `--fresh`, `classify` resumes from existing run checkpoints — fast for
finishing an interrupted run, but `--fresh` is the switch for a clean restart.

**Add the GNN to a project that has only run LLM consensus, then scale**
```bash
python qra.py gnn train  -o ./data/output/    # train the graph + run the out-of-sample reliability gate
python qra.py gnn status -o ./data/output/    # κ(graph,LLM) vs target — has it reached LLM-consensus IRR?
python qra.py gnn classify -o ./data/output/  # once READY: LLM-free label new/unlabeled segments
```
`gnn train` works even if the GNN was never enabled (it force-enables it for the run)
and warns if the project lacks the multi-run LLM ballots the consensus signal needs.
The gate verdict (`qra gnn status`) tells you when the graph agrees with the
LLM/human consensus closely enough to scale without LLM calls. The TUI surfaces the
same status inline and offers a "View GNN reliability / κ status" action.

**Upgrade a pre-SQLite project**
```bash
python qra.py migrate -o ./data/output/         # preview the import
python qra.py migrate -o ./data/output/ --run   # fold legacy JSONL into qra.db (originals → _legacy_files/)
```
This also happens automatically on the next `ingest`/`run`; `migrate` just makes it
explicit and previewable.

---

## Configuration

`process/config.py:PipelineConfig` covers all settings: paths, trial metadata, framework/codebook selection, LLM backend and model, classification parameters (`n_runs`, `temperature`, confidence thresholds), embedding model, speaker filtering, feature flags, test set config, content-validity config, and PURER cue config (`skip_lesson_content`, `max_lesson_words`, `therapist_max_gap_seconds`, `max_context_words`).

The setup wizard serializes to `qra_config.json`; `--config` reproduces any run exactly.

**LLM backends:** LM Studio (local), OpenRouter (`OPENROUTER_API_KEY`), Replicate (`REPLICATE_API_TOKEN`), Ollama, HuggingFace — any OpenAI-compatible endpoint.

---

## Validation Architecture

QRA implements a scalable pipeline for maintaining human-validated datasets across a scaling content database:

- **Content validity** — frozen content-validity test sets at `04_validation/content_validity/` with exemplar, subtle, and adversarial utterances from each framework definition. Run the classifier against known labels before touching real transcripts.
- **Construct validity** — empirical VAAMR × VCE lift statistics. Currently exploratory (Cohorts 1–2 characterization) — VCE is an optional enrichment layer and no construct-validity claim rests on it; a rigorous, independently-measured cross-framework test is deferred to future, larger-*n* work.
- **Reliability** — multi-run consensus (`llm_run_consistency`). Current production: single-model stochastic (stability measure). Planned: multi-model rotation (genuine cross-rater agreement).
- **Human validation** — stratified 20% evaluation set at `04_validation/human_coding_evaluation_set.csv`. Four qualitative researchers blind-coding VAAMR; PURER validation underway. Target: Krippendorff's α ≥ 0.60 (VAAMR) and α ≥ 0.70 (PURER).

---

## Key Design Invariants

1. **Frozen segmentation** — the `segments` table in `qra.db` is written once (overwrite-without-force raises `FrozenArtifactError`). No re-segmentation on re-runs. Only raw-segmentation fields are persisted; no classification data in the segments table.
2. **Overlay separation** — re-running any classifier overwrites only its overlay table (`<key>_labels`) in `qra.db`. Frozen segments are untouched.
3. **Framework boundary** — VAAMR classifies participants; PURER classifies therapists. Therapist segments appear as read-only context in participant classification prompts but are never VAAMR-classified.
4. **Auditable provenance** — every segment's final label carries its source. The resolution order is `adjudicated` > `human_consensus` > `gnn_consensus` > `llm_zero_shot`; the `gnn_consensus` tier is engaged only when `gnn_authoritative=True` (after the graph passes its reliability gate), and the raw multi-run LLM ballots remain visible per segment regardless. Every LLM call is logged with full prompt and response to `02_meta/auditable_logs/`.
5. **Hot markdown definitions** — VAAMR, PURER, and VCE codebook definitions live in human-editable `.md` files, parsed at runtime. Researchers can refine operational definitions between cohorts without touching Python.

---

## Installation & Quick Start

```bash
# Clone
git clone https://github.com/wadebalsamo/Qualitative_Research_Algorithm.git
cd qra

# Environment
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Interactive configuration wizard
python qra.py setup

# Run the full pipeline (with local model)
python qra.py run --config ./qra_config.json
```

**What happens on first run:** The setup wizard walks you through LLM backend selection, model choice, trial metadata, classification parameters, and output paths. It writes a reproducible `qra_config.json`. Then `python qra.py run` sequences all 8 pipeline stages automatically — or you can run stages modularly with `python qra.py ingest`, `python qra.py classify`, etc.

---

## Further Reading

- [`methodology.md`](docs/methodology.md) — Full neurophenomenological methodology paper (Balsamo, Wexler et al., *in preparation*). This is the canonical reference for *why* the pipeline is designed the way it is.
- [`VAAMR_FRAMEWORK.md`](frameworks/VAAMR_FRAMEWORK.md) — Complete operational VAAMR definitions with exemplar, subtle, and adversarial utterances
- [`PURER_FRAMEWORK.md`](frameworks/PURER_FRAMEWORK.md) — Complete operational PURER definitions
- [`PHENOMENOLOGY_CODEBOOK.md`](frameworks/PHENOMENOLOGY_CODEBOOK.md) — 54-code VCE codebook
- [`ROADMAP.md`](docs/ROADMAP.md) — Research and engineering trajectory

---

## Citations & Publications

### Peer-Reviewed Publications Using This Framework

> Wexler, R. S., **Balsamo, W.**, Fox, D. J., ZuZero, D., Parikshak, A., Kwin, S., Ramirez, J., Thompson, A. R., Carlson, H. L., Kern, T., Mist, S. D., Bradley, R., Zwickey, H., Pickworth, C. K., & Garland, E. L. (2026). "Noticing the way that I'm noticing pain": A qualitative analysis of therapeutic progression in Mindfulness-Oriented Recovery Enhancement for patients with lumbosacral radicular pain. *Mindfulness*, 17, 819–833. DOI: [10.1007/s12671-026-02782-1](https://doi.org/10.1007/s12671-026-02782-1)

> Wexler, R. S., **Balsamo, W.**, Lendof, V., et al. (in review). Development and pilot feasibility testing of Move-MORE: A multicomponent mindfulness-and-movement intervention for lumbosacral radicular pain. *Research Square*. DOI: [10.21203/rs.3.rs-8682836/v1](https://doi.org/10.21203/rs.3.rs-8682836/v1)

### Publications in Preparation

> **Balsamo, W.**, Wexler, R. S., et al. "Phenomenology at Trial Speed: A Computational Mixed-Methods Pipeline for Iterative Refinement of Mindfulness-Movement Therapy in Chronic Pain." *In Preparation*. — See [`methodology.md`](docs/methodology.md).

> **Balsamo, W.**, Wexler, R. S., et al. "From Vigilance to Reappraisal: Computational Neurophenomenological Results from Analyzing Contemplative Transformation in Mindfulness-Based Pain Therapy." *In Preparation*.

### Previous Work This Pipeline Builds Upon

> Lindahl, J. R., et al. (2017). The varieties of contemplative experience: A mixed-methods study of meditation-related challenges in Western Buddhists. *PLOS ONE*, 12(5), e0176239. DOI: [10.1371/journal.pone.0176239](https://doi.org/10.1371/journal.pone.0176239)

> Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text psychometrics: A framework for evaluating the validity of language models in clinical psychological assessment. *Psychological Methods*. DOI: [10.1037/met0000696](https://doi.org/10.1037/met0000696)

> Ramstead, M.J.D., et al. (2022). From Generative Models to Generative Passages: A Computational Approach to (Neuro) Phenomenology. *Review of Philosophy and Psychology*, 13, 829-857. 

> Varela, F. J. (1996). Neurophenomenology: A methodological remedy for the hard problem. *Journal of Consciousness Studies*, 3(4), 330–349.

### How to Cite This Repository

```bibtex
@software{QRA2026,
  title  = {QRA: Qualitative Research Algorithm},
  author = {Balsamo, Wade and Wexler, Ryan S.},
  year   = {2026},
  note   = {Computational phenomenology for mindfulness-based intervention research. DOI: 10.1007/s12671-026-02782-1},
  url    = {https://github.com/WadeBalsamo/Qualitative_Research_Algorithm}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).
