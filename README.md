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
[`methodology.md`](methodology.md) — *Phenomenology at Trial Speed: A Computational Mixed-Methods Pipeline for Iterative Refinement of Mindfulness-Movement Therapy in Chronic Pain* — is a complete, publication-ready methodology paper (Balsamo, Wexler et al., *in preparation*) housed directly in this repository. It provides the neurophenomenological theoretical grounding, the engineering rationale for every design decision, and the Text Psychometrics validation framework (Low et al., 2024) that QRA implements.

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
| **Immutable data checkpoints** | Segmentation results are written once and frozen (`01_transcripts/segmented/`). Classifiers write to independent overlay files. Re-running any classifier never touches frozen segments — immutable staging → refreshable marts. |
| **Classification infrastructure** | Multi-run LLM consensus voting (unanimous / majority / split / none, confidence-tiered) for VAAMR/PURER; embedding + LLM ensemble with weighted reconciliation for 59-code VCE codebook. Split-vote segments auto-flagged for human review. |
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

Full operational definitions (prototypical features, distinguishing criteria, exemplar / subtle / adversarial utterances, word prototypes) are in [`VAAMR_FRAMEWORK.md`](VAAMR_FRAMEWORK.md), parsed at runtime into `ThemeDefinition` objects by `theme_framework/markdown_loader.py`.

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

Full definitions in [`PURER_FRAMEWORK.md`](PURER_FRAMEWORK.md).

### VCE Phenomenology Codebook — Multi-Label Enrichment

The 59-code Varieties of Contemplative Experience codebook (Lindahl et al., 2017) is applied to participant segments via an embedding + LLM ensemble. Seven domains: Affective (13), Cognitive (10), Perceptual (7), Conative (3), Sense of Self (6), Social (5), Somatic (15).

Full codebook in [`PHENOMENOLOGY_CODEBOOK.md`](PHENOMENOLOGY_CODEBOOK.md).

For the complete theoretical neurophenomenological grounding and research methodology behind all three frameworks, see [`methodology.md`](methodology.md).

---

## What QRA Discovers: The Analysis Layer

The classification stages are the *labeling mechanism*. The research findings live in the `analysis/` module. None of the following involves LLMs.

### Therapeutic Breakthrough Moments: FROM → CUE → TO Extraction

`analysis/reports/transition_report.py`, `analysis/stage_progression.py`

Every participant utterance in the corpus is indexed by its chronological position in the session. For every within-session VAAMR stage transition — segment at stage X immediately followed by segment at stage Y — the pipeline extracts the tripartite structure:

- **FROM** — the participant utterance at stage X, with text, confidence, and timestamp
- **CUE** — all therapist segments whose `segment_index` falls strictly between FROM and TO in the same session, with their PURER move labels
- **TO** — the participant utterance at stage Y, with text, confidence, and timestamp

The result is a searchable corpus of every observed instance of every transition type (e.g., all Avoidance → Attention Regulation crossings across both cohorts), with actual participant language on both sides and the therapist's contribution in between. Researchers read what the breakthrough moment looks like in the patient's own words and see what the therapist said to get there. One example per (cohort, session) pair is extracted and sorted; the full collection is aggregated into LLM-synthesized portraits of what each transition type characteristically looks like across the corpus.

The **Avoidance → Attention Regulation** crossing is the clinically critical case. It marks where emerging attentional skill stops being deployed for pain suppression and is redirected toward open, investigative presence — the central developmental barrier identified in Wexler, Balsamo et al. (2026). Every instance of this crossing is extracted with the full FROM/CUE/TO triple; the PURER move distribution in those cue blocks is the primary evidence for curriculum recommendations.

### Therapist Move × Stage Transition: Conditional Lift

`analysis/purer_analysis.py`

The `CueBlock` data structure (`analysis/purer_analysis.py:CueBlock`) pairs every therapist response with its surrounding FROM and TO participant stages. For each (from\_stage, to\_stage) transition type, the analysis computes the distribution of PURER moves across all cue blocks of that type, then computes lift against the corpus-wide base rate:

> Lift(PURER move | transition type) = P(move | from\_stage, to\_stage) / P(move)

This generates a conditional probability table: *given that a participant just crossed the Avoidance barrier, which therapist inquiry moves were overrepresented in the preceding cue block?* The outputs are `purer_transition_profiles.csv`, `purer_vaamr_lift.csv`, and `purer_empty_cue_rates.csv` (transitions where no therapist speech occurred between participant segments — spontaneous, unmediated progressions tracked separately).

The clinically actionable question this answers: *is Reframing overrepresented before Reappraisal transitions? Is Phenomenology inquiry the dominant cue for Avoidance → Attention Regulation crossings, or does Reframing do more work there?* That distinction directly informs therapist training and session structure.

### State Transition Matrices

`analysis/stage_progression.py`, `analysis/reports/transition_report.py`

Two levels of transition analysis, answering different questions:

**Within-session:** For each (participant, session), the segment sequence is sorted by `segment_index` and adjacent stage pairs are tabulated — forward, backward, and lateral transition counts. Within-session transition matrices aggregate across all participants. This answers: *how fluid is stage movement within a session? What is the ratio of forward to backward transitions in Session 5 vs Session 2? Does Session 3 (Mindful Reappraisal content) actually produce more Reappraisal-stage segments than Session 1?*

**Between-session:** The modal VAAMR stage per (participant, session) represents that participant's dominant stage for the session. Between-session transition matrices compare dominant stages across consecutive session numbers. This answers: *are participants consolidating gains between sessions or reverting? At what session number does the group's modal stage shift forward?* A participant who briefly reaches Reappraisal in Session 2 but then shows it as their dominant stage by Session 5 has demonstrated the consolidation VAAMR predicts.

### Per-Participant Longitudinal Trajectories

`analysis/longitudinal.py`, `analysis/participant.py`

Per-participant reports track the dominant stage across all attended sessions — the developmental arc each participant traces. Group-level summaries compute mean stage proportion by session number across the cohort. A non-decreasing mean stage progression is the basic validity check for the VAAMR model; sessions where the mean stage dips despite content explicitly designed to produce advancement are primary curriculum modification targets.

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
Stage 6  — Dataset assembly (frozen segments + overlays → master_segments.jsonl)
Stage 7  — Report generation (coded transcripts, human forms, stats)
Stage 8  — Post-hoc analysis (trajectories, cue-response, figures) — also via `qra analyze`
```

### Stage 1 — Semantic Segmentation

`process/transcript_ingestion.py`, `process/llm_segmentation.py`

Sentence-transformer embeddings → windowed cosine-similarity curve → adaptive threshold (25th percentile of session's own distribution) → segment boundaries. Also respects silence gaps (default 1500 ms) and enforces min/max word counts (30/200 words, recursive split at midpoint).

Therapist and participant segments are separated at this stage. Therapist segments flow to PURER (Stage 3c) and appear as read-only preceding context in VAAMR prompts. They are never VAAMR-classified.

Boundaries are calculated with embeddings, and ambiguous-case LLM-assisted boundary refinements: `boundary_review`, `context_expansion`, `coherence_check`.

**Output:** `01_transcripts/segmented/<sid>/segments.jsonl` + `segmentation_meta.json` — frozen, never rewritten upon adding new data or changing framework definitions/exemplars. Raw input files are preserved at `01_transcripts_inputs/`.

### Stage 3 — VAAMR Classification

`classification_tools/classification_loop.py`, `classification_tools/llm_classifier.py`

Each participant segment gets a structured JSON prompt: segment text + preceding context (capped at 300 words) + full VAAMR `ThemeFramework` definitions. Required response fields: `primary_stage`, `primary_confidence`, `secondary_stage`, `secondary_confidence`, `justification` (must cite specific segment language). `null` primary_stage = valid ABSTAIN ballot.

Stage definition order is randomized per run (temperature > 0) to reduce position bias.

`n_runs` calls per segment → majority vote → confidence tier. Agreement levels: `unanimous` / `majority` / `split` / `none`. Split/none → `needs_review=True`.

**Output:** `02_meta/classifications/theme_labels.jsonl`

### Stage 3b — Codebook Ensemble

`codebook/embedding_classifier.py`, `codebook/ensemble.py`

Two independent classifiers:
- **Embedding**: query/passage cosine similarity against code definitions + inclusive criteria + exemplars; two-pass procedure for recall on subtle phenomenology
- **LLM**: multi-label zero-shot with strict majority rule per code

Ensemble reconciliation: union (default, recall-optimized) or intersection (precision-optimized). Both component outputs stored separately for post-hoc comparison. Embedding scores weighted 0.6, LLM 0.4 in composite confidence.

GPU memory hand-off: reuses segmentation embedding model when model IDs match.

**Output:** `02_meta/classifications/codebook_labels.jsonl`

### Stage 3c — PURER Cue-Block Classification

Therapist response between consecutive participant turns = one cue-block, one PURER label. Context window: 6 preceding segments (vs 2 for VAAMR). Long didactic stretches (lesson content) can be skipped via `PurerCueConfig.skip_lesson_content`. Label propagated back to all constituent therapist segments.

**Output:** `02_meta/classifications/purer_labels.jsonl`

### Stage 4 — Cross-Validation Lift Statistics

`process/cross_validation.py`

Lift(stage, code) = P(code | stage) / P(code). Default thresholds: lift ≥ 1.5 and minimum 3 segments. Currently exploratory — computes empirical co-occurrence patterns. An `expected_codes` pre-specification (encoding theoretical VCE predictions per VAAMR stage) is planned before Cohort 3, enabling mechanical expected-vs-observed comparison implementing Varela's (1996) neurophenomenological mutual-constraints logic.

**Output:** `02_meta/classifications/cross_validation_labels.jsonl`, lift report

### Stage 6 — Assembly

`process/assembly/master_dataset.py`

Joins frozen segments + all overlay files. Final label priority: `adjudicated > human_consensus > llm_zero_shot`. Provenance fully auditable per segment.

**Output:** `02_meta/training_data/master_segments.jsonl`, `master_segments.csv`. Coded transcripts → `04_validation/full_transcripts/`. Human classification forms → `04_validation/full_transcripts/`.

---

## On-Disk Layout

```
output_dir/
├── 00_index.txt
├── 01_transcripts/
│   └── segmented/<sid>/          # FROZEN — never rewritten
│       ├── segments.jsonl
│       └── segmentation_meta.json
├── 01_transcripts_inputs/        # Raw input VTT/JSON copies (provenance)
├── 02_meta/
│   ├── classifications/          # Refreshable overlays
│   │   ├── theme_labels.jsonl
│   │   ├── purer_labels.jsonl
│   │   ├── codebook_labels.jsonl
│   │   ├── cross_validation_labels.jsonl
│   │   └── classification_manifest.json
│   ├── auditable_logs/           # LLM prompts/responses/checkpoints
│   ├── codebook_raw/             # Embedding checkpoints
│   ├── training_data/            # master_segments.jsonl/.csv, BERT training data
│   └── speaker_anonymization_key.json
├── 03_analysis_data/             # Session stats, graphing CSVs
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
├── 05_figures/
└── 06_reports/
```

---

## Module Map

| Path | Role |
|------|------|
| `qra.py` | CLI entry point |
| `process/orchestrator.py` | Stage sequencing, `run_full_pipeline`, standalone `stage_*` functions |
| `process/config.py` | `PipelineConfig` dataclass — single source of truth for all settings |
| `process/segments_io.py` | Frozen per-session segment I/O, `params_hash`, `load_segments_for_stage` |
| `process/classifications_io.py` | Overlay read/write, provenance manifest |
| `process/_freeze.py` | Freeze enforcement (atomic write, SHA verification) |
| `process/transcript_ingestion.py` | `ConversationalSegmenter` — embedding-based semantic segmentation |
| `process/llm_segmentation.py` | LLM-assisted boundary refinement |
| `process/speaker_anonymization.py` | Persistent speaker ID mapping across runs |
| `process/speaker_filter.py` | Speaker inclusion/exclusion rules per classifier |
| `process/output_paths.py` | Single source of truth for all output directory paths |
| `process/legacy_migration.py` | Auto-migration: v2.0 → per-session segments; v2.5 → v3 directory layout |
| `process/cross_validation.py` | VAAMR × VCE lift statistics |
| `process/setup_wizard.py` | Interactive configuration wizard (14 steps) |
| `process/assembly/master_dataset.py` | `assemble_master_dataset` |
| `process/assembly/human_forms.py` | Human classification forms, test set freeze/refresh |
| `process/assembly/content_validity.py` | Content-validity test set freeze/refresh |
| `process/assembly/coded_transcripts.py` | Per-session coded transcript writer |
| `process/assembly/stats_reports.py` | Per-transcript stats, cumulative report |
| `process/assembly/training_export.py` | Training data and theme definition export |
| `theme_framework/vaamr.py` | `get_vaamr_framework()` — loads from `VAAMR_FRAMEWORK.md` |
| `theme_framework/purer.py` | `get_purer_framework()` — loads from `PURER_FRAMEWORK.md` |
| `theme_framework/markdown_loader.py` | Parser: `.md` → `ThemeFramework` / `ThemeDefinition` objects |
| `theme_framework/theme_schema.py` | `ThemeDefinition`, `ThemeFramework` dataclasses |
| `codebook/phenomenology_codebook.py` | 59 `CodeDefinition` objects (VCE), loaded from `PHENOMENOLOGY_CODEBOOK.md` |
| `codebook/embedding_classifier.py` | Sentence-transformer embedding-based codebook classification |
| `codebook/ensemble.py` | Embedding + LLM ensemble reconciliation |
| `classification_tools/llm_classifier.py` | Zero-shot prompt construction and response parsing |
| `classification_tools/llm_client.py` | Backend abstraction (OpenRouter, Ollama, LM Studio, HuggingFace, Replicate) |
| `classification_tools/classification_loop.py` | Multi-run consensus voting with checkpointing |
| `classification_tools/majority_vote.py` | Ballot aggregation (unanimous/majority/split/none) |
| `classification_tools/data_structures.py` | `Segment` dataclass |
| `analysis/runner.py` | Post-hoc analysis entry point |
| `analysis/purer_analysis.py` | PURER × VAAMR conditional lift table, cue-response synthesis |
| `analysis/purer_figures.py` | PURER × VAAMR heatmap and figures |
| `analysis/reports/` | Text report generators: session, stage, transition, cue-response, longitudinal, summaries |

---

## CLI Reference

```bash
# Interactive setup wizard — creates qra_config.json
python qra.py setup

# Full pipeline run
python qra.py run --config ./data/output/02_meta/qra_config.json

# Run with inline overrides
python qra.py run --transcript-dir ./data/input --output-dir ./data/output \
  --backend openrouter --model qwen/qwen-3-70b

# Pipeline + auto-generate analysis reports
python qra.py run --config ./qra_config.json --auto-analyze

# Post-hoc analysis on completed output
python qra.py analyze --output-dir ./data/output/

# Modular stage execution (Phase 3)
python qra.py ingest -o ./data/output/                    # segment only
python qra.py classify -o ./data/output/ --what theme     # VAAMR only
python qra.py classify -o ./data/output/ --what purer     # PURER only
python qra.py classify -o ./data/output/ --what codebook  # VCE only
python qra.py assemble -o ./data/output/                  # join frozen + overlays

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
```

---

## Configuration

`process/config.py:PipelineConfig` covers all settings: paths, trial metadata, framework/codebook selection, LLM backend and model, classification parameters (`n_runs`, `temperature`, confidence thresholds), embedding model, speaker filtering, feature flags, test set config, content-validity config, and PURER cue config (`skip_lesson_content`, `max_lesson_words`, `therapist_max_gap_seconds`, `max_context_words`).

The setup wizard serializes to `qra_config.json`; `--config` reproduces any run exactly.

**LLM backends:** LM Studio (local), OpenRouter (`OPENROUTER_API_KEY`), Replicate (`REPLICATE_API_TOKEN`), Ollama, HuggingFace — any OpenAI-compatible endpoint.

---

## Validation Architecture

QRA implements a scalable pipeline for maintaining human-validated datasets across a scaling content database:

- **Content validity** — frozen content-validity test sets at `04_validation/content_validity/` with exemplar, subtle, and adversarial utterances from each framework definition. Run the classifier against known labels before touching real transcripts.
- **Construct validity** — empirical VAAMR × VCE lift statistics. Currently exploratory (Cohorts 1–2 characterization). An `expected_codes` pre-specification (encoding theoretical VCE predictions per VAAMR stage) is a committed engineering item before Cohort 3 begins, which will enable mechanical expected-vs-observed comparison.
- **Reliability** — multi-run consensus (`llm_run_consistency`). Current production: single-model stochastic (stability measure). Planned: multi-model rotation (genuine cross-rater agreement).
- **Human validation** — stratified 20% evaluation set at `04_validation/human_coding_evaluation_set.csv`. Four qualitative researchers blind-coding VAAMR; PURER validation underway. Target: Krippendorff's α ≥ 0.60 (VAAMR) and α ≥ 0.70 (PURER).

---

## Key Design Invariants

1. **Frozen segmentation** — `01_transcripts/segmented/<sid>/segments.jsonl` is written once. No re-segmentation on re-runs. Only raw-segmentation fields are persisted; no classification data in segment files.
2. **Overlay separation** — re-running any classifier overwrites only its `02_meta/classifications/<key>_labels.jsonl`. Frozen segments are untouched.
3. **Framework boundary** — VAAMR classifies participants; PURER classifies therapists. Therapist segments appear as read-only context in participant classification prompts but are never VAAMR-classified.
4. **Auditable provenance** — every segment's final label carries its source (`adjudicated` / `human_consensus` / `llm_zero_shot`). Every LLM call is logged with full prompt and response to `02_meta/auditable_logs/`.
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

- [`methodology.md`](methodology.md) — Full neurophenomenological methodology paper (Balsamo, Wexler et al., *in preparation*). This is the canonical reference for *why* the pipeline is designed the way it is.
- [`VAAMR_FRAMEWORK.md`](VAAMR_FRAMEWORK.md) — Complete operational VAAMR definitions with exemplar, subtle, and adversarial utterances
- [`PURER_FRAMEWORK.md`](PURER_FRAMEWORK.md) — Complete operational PURER definitions
- [`PHENOMENOLOGY_CODEBOOK.md`](PHENOMENOLOGY_CODEBOOK.md) — 59-code VCE codebook
- [`ROADMAP.md`](ROADMAP.md) — Research and engineering trajectory

---

## Citations & Publications

### Peer-Reviewed Publications Using This Framework

> Wexler, R. S., **Balsamo, W.**, Fox, D. J., ZuZero, D., Parikshak, A., Kwin, S., Ramirez, J., Thompson, A. R., Carlson, H. L., Kern, T., Mist, S. D., Bradley, R., Zwickey, H., Pickworth, C. K., & Garland, E. L. (2026). "Noticing the way that I'm noticing pain": A qualitative analysis of therapeutic progression in Mindfulness-Oriented Recovery Enhancement for patients with lumbosacral radicular pain. *Mindfulness*, 17, 819–833. DOI: [10.1007/s12671-026-02782-1](https://doi.org/10.1007/s12671-026-02782-1)

> Wexler, R. S., **Balsamo, W.**, Lendof, V., et al. (in review). Development and pilot feasibility testing of Move-MORE: A multicomponent mindfulness-and-movement intervention for lumbosacral radicular pain. *Research Square*. DOI: [10.21203/rs.3.rs-8682836/v1](https://doi.org/10.21203/rs.3.rs-8682836/v1)

### Publications in Preparation

> **Balsamo, W.**, Wexler, R. S., et al. "Phenomenology at Trial Speed: A Computational Mixed-Methods Pipeline for Iterative Refinement of Mindfulness-Movement Therapy in Chronic Pain." *In Preparation*. — See [`methodology.md`](methodology.md).

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
