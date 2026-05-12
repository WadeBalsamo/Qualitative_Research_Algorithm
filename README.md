# QRA: Qualitative Research Algorithm

**Computational phenomenology for mindfulness-based intervention research.**

QRA is a machine-assisted qualitative analysis pipeline that applies two complementary classification frameworks bilaterally to therapy transcripts: the Vigilance-Avoidance-Attention Regulation-Metacognition-Reappraisal (VAAMR) model classifies participant speech across a five-stage developmental arc, PURER (Phenomenological, Utilization, Reframing, Educate/Expectancy, Reinforcement) classifies therapist speech across five guided-inquiry moves, and the Varieties of Contemplative Experience (VCE) phenomenology codebook enriches participant segments with multi-label phenomenological content codes — producing defensible cohort-level qualitative analysis in days rather than weeks.

---

## The Problem: Phenomenology at the Pace of Iterative Design

Chronic pain is not simply pain that persists. In Leder's (1990) phenomenological account, building on Merleau-Ponty's (1962) analysis of the lived body, chronic pain constitutes a *dys-appearance*: the body's forcible eruption into thematic attention as an alien obstacle rather than a transparent instrument of engagement. Where a healthy body recedes into the background of experience, functioning as the medium through which a person attends to the world, the chronically pained body insists on its own presence — continuously recruiting attentional resources that would otherwise support meaning, relationship, and action. This is not merely an unpleasant sensation; it is a structural disorder of experience.

Mindfulness-based interventions (MBIs) for chronic pain are, on this account, best understood as structured practices of phenomenological re-habituation: systematic attempts to restore functional structural relationships between attention and the lived body that chronic pain disorder disrupts. Mindfulness-Oriented Recovery Enhancement (MORE; Garland 2013, 2024) has demonstrated efficacy across multiple randomized controlled trials — reducing pain intensity, opioid misuse, and pain catastrophizing, with neural correlates including thalamic-default mode network decoupling that appears to drive mindfulness-based pain relief by supporting self-referential disengagement from pain (Riegner et al., 2023).

But tracking *how* this re-habituation unfolds — session by session, participant by participant, in the language patients actually use to describe their experience — requires qualitative analysis of therapy transcripts. And this creates a structural tension in iterative trial design.

The between-cohort refinement window in iterative feasibility trials is measured in weeks. Full qualitative analysis of a therapy transcript corpus takes months. The practical result is that curriculum modifications between cohorts are made on the basis of clinical intuition and aggregate outcome scores rather than systematic analysis of the phenomenological processes the intervention is designed to produce.

QRA was built to dissolve this tension.

---

## The Dual-Framework Architecture

QRA implements two complementary phenomenological frameworks concurrently on the same transcript corpus. They address orthogonal questions.

### VAAMR: Where is this participant in therapeutic progression?

The Vigilance-Avoidance-Attention Regulation-Metacognition-Reappraisal (VAAMR) framework was derived from thematic analysis of MORE sessions and characterizes five stages of mindfulness skill development, interpretable as progressive stages in the restoration of healthy structural relationships between attention and the lived body:

| Stage | Name | Phenomenological Character | Canonical Expression |
|-------|------|---------------------------|---------------------|
| 0 | **Vigilance** | Leder's dys-appearance at its most acute: the body occupying the total field of attentional engagement. Attention is reactive rather than directed, captured by pain signals that continuously reassert themselves. | *"I can't stop thinking about the pain, it's all I can focus on."* |
| 1 | **Avoidance** | The critical developmental barrier: attentional competence acquired but deployed in the service of experiential avoidance rather than investigation. Mindfulness techniques used instrumentally to push pain away rather than to inhabit it. | *"When the pain comes, I focus really hard on my breathing to push it away."* |
| 2 | **Attention Regulation** | Stable, sustained, volitional attention that stays *with* present somatic experience — anchored in sensation rather than fighting it or fleeing from it. Attention as deliberate investigative presence rather than defensive tool. | *"I kept bringing my attention back to the sensations, just staying with them."* |
| 3 | **Metacognition** | The emergence of reflexive distance — an observing standpoint from which the contents of experience, including pain-related cognitions and affects, can be witnessed without identification. The body begins to re-emerge as something observed rather than something constituting. | *"I noticed I was getting anxious about the pain, and I could just watch that anxiety."* |
| 4 | **Reappraisal** | A transformation of the noematic structure of pain experience: pain decomposed into constituent sensations, losing its character as monolithic threatening event and becoming a complex, textured, changing field. | *"It's interesting, when I really look at it, the 'pain' is actually many different feelings."* |

The arc from Stage 0 to Stage 4 describes the progressive recovery of the body as a transparent medium of experience rather than a dysappearing obstacle — the theoretical process MORE is designed to produce.

The full operational definitions, including prototypical features, distinguishing criteria, exemplar utterances, subtle utterances, and adversarial utterances (the boundary-case language that separates adjacent stages), are defined in `VAAMR_FRAMEWORK.md` and parsed at runtime by the theme framework.

### VCE: What is this participant phenomenologically experiencing?

The Varieties of Contemplative Experience codebook (Lindahl et al., 2017; DOI: [10.1371/journal.pone.0176239](https://doi.org/10.1371/journal.pone.0176239)) was derived from content-driven thematic analysis of 60 structured interviews with Western Buddhist meditators across Theravāda, Zen, and Tibetan traditions. It characterizes 59 categories of contemplative experience across seven higher-order domains, designed to be domain-descriptive rather than valence-prescriptive.

| Domain | Codes | What it captures |
|--------|-------|-----------------|
| Affective | 13 | Fear, anxiety, positive affect, agitation, emotional detachment, bliss |
| Cognitive | 10 | Meta-cognition, clarity, disintegration of meaning structures, change in narrative self |
| Conative | 3 | Changes in motivation, effort, desire |
| Perceptual | 7 | Sensory hypersensitivity, altered perception, synesthesia |
| Sense of Self | 6 | Change in self-other boundaries, changes in sense of agency and ownership |
| Social | 5 | Empathic changes, affiliative changes, relational shifts |
| Somatic | 15 | Pain, energy, pressure, temperature, internal sensations |

Defined in `PHENOMENOLOGY_CODEBOOK.md` and parsed at runtime into `CodeDefinition` objects with formal descriptions, subcodes, inclusive criteria, and exclusive criteria.

### Why Both: Orthogonal Analytical Dimensions

VAAMR is a *developmental stage model* — it answers *where in therapeutic progression*. VCE is a *content taxonomy* — it answers *what is being experienced at each location in that progression*. Their concurrent application generates evidence neither framework can produce alone.

The VAAMR framework encodes theoretically grounded expectations about which VCE codes should co-occur with each stage (to be pre-registered as an `expected_codes` field in [`theme_framework/vaamr.py`](theme_framework/vaamr.py) before Cohort 3):

- **Vigilance** is predicted to lift: *Fear/Anxiety/Panic/Paranoia*, *Agitation or Irritability*, *Pain (Somatic)* — consistent with dys-appearance as an affective-somatic phenomenon
- **Avoidance** is predicted to lift: *Affective Flattening*, *Emotional Detachment, or Alexithymia* — consistent with the emotional blunting documented in experiential avoidance strategies (Hayes et al., 1996)
- **Attention Regulation** is predicted to lift: *Attention*, *Change in Duration of Experience*, *Somatic Relaxation or Calming* — consistent with sustained volitional attention to somatic experience with reduced effortful struggle
- **Metacognition** is predicted to lift: *Meta-Cognition*, *Clarity*, *Change in Narrative Self* — consistent with the emergence of reflexive self-observation
- **Reappraisal** is predicted to lift: *Positive Affect*, *Change in Worldview*, *Change in Self-Other or Self-World Boundaries*, *Disintegration of Conceptual Meaning Structures* — consistent with a transformation in the noematic structure of pain experience

Pipeline Stage 4 tests these predictions empirically through **lift statistics** — co-occurrence ratios that measure how strongly each VCE code appears with each VAMMR stage relative to its corpus-wide base rate. This cross-validation implements Varela's (1996) neurophenomenological logic of *mutual constraints*: two independently-derived phenomenological frameworks constrain each other through empirical co-occurrence, and their convergence — or non-convergence — is informative in both directions. Confirmed predictions constitute convergent validity evidence; disconfirmed predictions reveal where theoretical frameworks require revision in this population.

---

## The PURER Framework: Bilateral Therapist Classification

QRA applies two classification frameworks bilaterally. VAAMR classifies every participant segment. PURER classifies every therapist segment at the **cue-block level** (one label per therapist response between consecutive participant turns). The two frameworks together constitute a complete phenomenological account of the therapeutic dyad.

PURER operationalizes five guided-inquiry moves drawn from the MORE Manual (Garland, 2018) and validated against the qualitative dataset analyzed by Wexler, Balsamo et al. (2026): **P**henomenological, **U**tilization, **R**eframing, **E**ducate/Expectancy, **R**einforcement. Implemented in [`theme_framework/purer.py`](theme_framework/purer.py).

In the context of this analysis, PURER functions as a structured phenomenological interview method — analogous to Giorgi's (1985) descriptive phenomenological interview — systematically eliciting and consolidating participants' first-person reports of their experience.

PURER moves frequently co-occur within a single therapist turn. When a single label is required, an empirical precedence order is specified in `theme_framework/purer.py`: Reinforcement is often a wrapper around a substantive move (code the inner move); Utilization takes precedence over Reframing for forward-application prompts; Reframing takes precedence over Education when the concept is anchored to the participant's specific story.

Therapist dialogue is not merely contextual background to participant phenomenological expression — it is a systematic *elicitor* of phenomenological description. The structure of PURER shapes which phenomenological reports participants offer; understanding therapeutic mechanism requires understanding that shaping.

QRA's **therapist cue-response analysis** makes this visible. Because the pipeline separates participant and therapist segments and indexes them chronologically, it can retrieve — for every observed within-session VAAMR stage transition — the therapist dialogue that immediately preceded it. This produces an empirical characterization of which therapist behaviors are associated with each type of stage change.

The Avoidance → Attention Regulation transition is the single most clinically important transition to monitor: it marks the crossing of the experiential avoidance barrier, where emerging attentional skill is redirected from suppression toward open, investigative presence.

---

## What the Pipeline Produces

Given diarized session transcripts (from Whisper + speaker diarization):

1. **Frozen semantically coherent segments** — embedding-based segmentation with adaptive thresholds, written to per-session frozen files (`01_transcripts/segmented/<sid>/segments.jsonl`) that are never rewritten
2. **Classification overlay files** — per-classifier JSONL files at `02_meta/classifications/` (theme, purer, codebook, cross-validation) that can be independently re-run without touching frozen segments
3. **VAAMR stage classifications** — multi-run LLM consensus voting with confidence tiering (High/Medium/Low)
4. **PURER cue-block classifications** — therapist dialogue classified at the cue-unit level between participant turns
5. **VCE phenomenology codes** — multi-label coding via embedding similarity + LLM zero-shot ensemble
6. **Cross-validation lift statistics** — empirical (VAAMR stage × VCE code) co-occurrence ratios
7. **Frozen validation test sets** — stratified evaluation sets for blind-coding (VAAMR, PURER, and codebook variants with frozen human worksheets and refreshable AI answer keys)
8. **Frozen content-validity test sets** — built from framework exemplar/subtle/adversarial utterances (VAAMR and PURER)
9. **Session-level stage progression summaries** — forward, backward, and lateral transition counts
10. **Longitudinal trajectory reports** — group-level mean stage proportions per session number
11. **Therapist cue-response reports** — therapist language grouped by transition type

---

## Output Directory Layout (Current)

```
output_dir/
├── 00_index.txt
├── 01_transcripts/
│   ├── diarized/            # Raw input copies (provenance)
│   ├── segmented/<sid>/     # FROZEN raw segments (Phase 1)
│   │   ├── segments.jsonl
│   │   └── segmentation_meta.json
│   └── coded/               # Human-readable coded transcripts
├── 02_meta/
│   ├── classifications/     # Phase 3 overlays
│   │   ├── theme_labels.jsonl, purer_labels.jsonl, codebook_labels.jsonl
│   │   ├── cross_validation_labels.jsonl
│   │   └── classification_manifest.json
│   ├── auditable_logs/      # LLM prompts/responses/checkpoints/configs
│   ├── codebook_raw/        # Codebook embedding checkpoints
│   ├── training_data/       # master_segments.jsonl/.csv, BERT training data
│   └── speaker_anonymization_key.json
├── 03_analysis_data/        # Session stats, graphing CSVs, per-{session,participant,theme} JSON
├── 04_validation/
│   ├── testsets/<name>/     # FROZEN validation test sets
│   │   ├── manifest.json, segments_snapshot.jsonl
│   │   ├── human_worksheet.txt (frozen), AI_answer_key.txt (refreshable)
│   ├── content_validity/<name>/  # FROZEN content-validity testsets
│   │   ├── manifest.json, items.jsonl
│   │   ├── human_worksheet.txt, definition_key.txt (frozen)
│   │   └── AI_answer_key.txt (refreshable)
│   ├── cross_validation/    # Lift statistics
│   └── human_coding_evaluation_set.csv
├── 05_figures/
├── 06_reports/
└── 07_meta/                 # (legacy — may be empty on new runs)
```

---

## What Can Be Learned

The outputs support three levels of analysis, each addressing a different research question:

### Session level: Which sessions are catalyzing stage progression?

Session stage distributions, compared against the theoretical arc implied by session content, reveal where the curriculum is working and where it is not. Sessions where Avoidance remains dominant in weeks 5–7 — despite content explicitly designed to catalyze the Avoidance → Attention Regulation transition — are primary curriculum targets.

### Dyadic level: What therapist behaviors facilitate contemplative transformation?

The cue-response analysis directly addresses a limitation of most MBI process research: participant phenomenological reports are typically analyzed in isolation from the therapist behaviors that elicited them. QRA's session adjacency index makes this dyad visible at the level of individual transitions — allowing researchers to identify which PURER components appear in the therapist dialogue preceding significant stage changes.

### Population level: What is the phenomenological texture of each stage?

The lift statistics answer a question that neither framework could address alone: what are participants *actually experiencing* at each stage, and does the phenomenological texture of therapeutic transformation in a chronic pain MBI population match theoretical predictions derived from Buddhist meditators?

### Longitudinal: Is there a coherent developmental trajectory?

A non-decreasing mean stage progression across session numbers is the basic validity check for the VAAMR model as a developmental framework. Departure from this pattern — particularly in movement-integrated sessions — is itself informative.

---

## Validation: Text Psychometrics

Computational phenomenological classification requires a validation methodology appropriate to the distinctive epistemic situation. Low, Mair, Nock, and Ghosh (2024) propose *Text Psychometrics* as a framework specifically designed for this purpose. QRA implements each component:

- **Content validity** — via frozen content-validity test sets at `04_validation/content_validity/`, comprising exemplar, subtle, and adversarial utterances from each framework definition.
- **Construct validity** — via Stage 4 lift statistics constituting convergent validity hypotheses.
- **Reliability** — via multi-run consensus (`llm_run_consistency`) with independent LLM calls as proxy raters.

Human validation of the stratified evaluation set by independent rater teams, with Krippendorff's alpha and raw agreement statistics, provides the ultimate validation standard.

---

## Getting Started

- **[SETUP.md](SETUP.md)** — System requirements, installation, LLM backend configuration, setup wizard walkthrough
- **[USAGE.md](USAGE.md)** — Full command reference, pipeline stages, output structure, configuration
- **[CLAUDE.md](CLAUDE.md)** — Developer reference with module map and design invariants

```bash
python qra.py setup                # interactive configuration wizard
python qra.py run --config ./data/output/02_meta/qra_config.json
python qra.py analyze --output-dir ./data/output/

# Modular stages
python qra.py ingest -o ./data/output/                   # segment only
python qra.py classify -o ./data/output/ --what theme     # classify theme only
python qra.py assemble -o ./data/output/                  # join frozen+overlays

# Validation test set management
python qra.py testset create -o ./data/output/ --kind purer --name purer_irr_1
python qra.py testset refresh -o ./data/output/ --all
python qra.py testset list -o ./data/output/

# Content validity management
python qra.py cv create -o ./data/output/ --framework purer --name cv_purer_v1
python qra.py cv refresh -o ./data/output/ --all
python qra.py cv list -o ./data/output/
```

**LLM backend options:** LM Studio (local), OpenRouter, Replicate, Ollama, HuggingFace — any OpenAI-compatible endpoint.

---

## Development Roadmap

The pipeline is validated on Move-MORE Cohorts 1 and 2. PURER classification is operational. Engineering priorities:

1. **PURER Validation and Refinement** — validate therapist segment classifications against human expert raters. Target: Krippendorff's α ≥ 0.70.
2. **Expected Codes Pre-Specification** — operationalize Varela-style mutual-constraints test by populating `expected_codes` fields before Cohort 3.
3. **Avoidance-Barrier Dedicated Report** — automated analysis of Avoidance prevalence and barrier-crossing timing.
4. **Outcome integration** — join session-level VAAMR stage distributions with quantitative outcomes.
5. **Supervised fine-tuning** — use the accumulated labeled corpus to train domain-adapted classifiers.

See **[ROADMAP.md](ROADMAP.md)** for the full research and engineering trajectory.

---

## Publications in Preparation

**Balsamo, W., Wexler, R. S., et al.** — "From Vigilance to Reappraisal: A Computational Neurophenomenological Method for Analyzing Contemplative Transformation in Mindfulness-Based Pain Therapy." *In preparation for Journal of Phenomenology and the Cognitive Sciences.*

**Balsamo, W., Wexler, R. S., Fox, D. J., Garland, E. L., et al.** — "Computational Phenomenology in Mindfulness-Based Interventions for Chronic Pain: A Machine-Assisted Methodology for Rapid Iterative Curriculum Refinement." *In preparation for Journal of Contemplative Studies.*

---

## Prior Work

This project is built directly upon the following bodies of work:

**VA-MR Framework**
- Wexler, R. S., Balsamo, W., et al. (2026). "Noticing the Way that I'm Noticing Pain": A qualitative analysis of therapeutic progression in mindfulness-oriented recovery enhancement for patients with lumbosacral radicular pain. *Mindfulness*, 17, 819–833.

**Move-MORE Feasibility Trial**
- Wexler, R. S., Balsamo, W., et al. (2026). Development and pilot feasibility testing of Move-MORE: A multicomponent mindfulness-and-movement intervention for lumbosacral radicular pain.

**MORE for LRP Randomized Controlled Trial**
- Wexler, R. S., Fox, D. J., et al. (2024). Virtually delivered MORE reduces daily pain intensity in patients with lumbosacral radiculopathy. *Pain Reports*, 9(2), e1132.

**VCE Phenomenology Codebook**
- Lindahl, J. R., et al. (2017). The varieties of contemplative experience. *PLOS ONE*, 12(5), e0176239.

**Text Psychometrics**
- Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text psychometrics. *Psychological Methods*.

---

## Citation

```bibtex
@software{QRA2026,
  title  = {QRA: Qualitative Research Algorithm},
  author = {Balsamo, Wade and Wexler, Ryan S.},
  year   = {2026},
  note   = {Computational phenomenology pipeline for mindfulness-based intervention research}
}
```

## License

MIT License — see [LICENSE](LICENSE).
