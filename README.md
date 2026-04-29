# QRA: Qualitative Research Algorithm

**Computational phenomenology for mindfulness-based intervention research.**

QRA is a machine-assisted qualitative analysis pipeline that classifies therapy transcript segments using two established phenomenological frameworks concurrently — the Vigilance-Avoidance-Mindfulness-Metacognition-Reappraisal (VAMMR) model of therapeutic progression and the Varieties of Contemplative Experience (VCE) phenomenology codebook — producing defensible cohort-level qualitative analysis in days rather than weeks.

---

## The Problem: Phenomenology at the Pace of Iterative Design

Chronic pain is not simply pain that persists. In Leder's (1990) phenomenological account, building on Merleau-Ponty's (1962) analysis of the lived body, chronic pain constitutes a *dys-appearance*: the body's forcible eruption into thematic attention as an alien obstacle rather than a transparent instrument of engagement. Where a healthy body recedes into the background of experience, functioning as the medium through which a person attends to the world, the chronically pained body insists on its own presence — continuously recruiting attentional resources that would otherwise support meaning, relationship, and action. This is not merely an unpleasant sensation; it is a structural disorder of experience.

Mindfulness-based interventions (MBIs) for chronic pain are, on this account, best understood as structured practices of phenomenological re-habituation: systematic attempts to restore functional structural relationships between attention and the lived body that chronic pain disorder disrupts. Mindfulness-Oriented Recovery Enhancement (MORE; Garland 2013, 2024) has demonstrated this efficacy across multiple randomized controlled trials — reducing pain intensity, opioid misuse, and pain catastrophizing, with neural correlates including thalamic-default mode network decoupling that appears to drive mindfulness-based pain relief by supporting self-referential disengagement from pain (Riegner et al., 2023).

But tracking *how* this re-habituation unfolds — session by session, participant by participant, in the language patients actually use to describe their experience — requires qualitative analysis of therapy transcripts. And this creates a structural tension in iterative trial design.

The between-cohort refinement window in iterative feasibility trials is measured in weeks. Full qualitative analysis of a therapy transcript corpus takes months. The practical result is that curriculum modifications between cohorts are made on the basis of clinical intuition and aggregate outcome scores rather than systematic analysis of the phenomenological processes the intervention is designed to produce.

QRA was built to dissolve this tension.

---

## The Dual-Framework Architecture

QRA implements two complementary phenomenological frameworks concurrently on the same transcript corpus. They address orthogonal questions.

### VAMMR: Where is this participant in therapeutic progression?

The Vigilance-Avoidance-Mindfulness-Metacognition-Reappraisal (VAMMR) framework was derived from thematic analysis of MORE sessions and characterizes five stages of mindfulness skill development, interpretable as progressive stages in the restoration of healthy structural relationships between attention and the lived body:

| Stage | Name | Phenomenological Character | Canonical Expression |
|-------|------|---------------------------|---------------------|
| 0 | **Vigilance** | Leder's dys-appearance at its most acute: the body occupying the total field of attentional engagement. Attention is reactive rather than directed, captured by pain signals that continuously reassert themselves. | *"I can't stop thinking about the pain, it's all I can focus on."* |
| 1 | **Avoidance** | The critical developmental barrier: attentional competence acquired but deployed in the service of experiential avoidance rather than investigation. Mindfulness techniques used instrumentally to push pain away rather than to inhabit it. | *"When the pain comes, I focus really hard on my breathing to push it away."* |
| 2 | **Mindfulness** | Stable, sustained, volitional attention that stays *with* present somatic experience — anchored in sensation rather than fighting it or fleeing from it. Attention as deliberate investigative presence rather than defensive tool. | *"I kept bringing my attention back to the sensations, just staying with them."* |
| 3 | **Metacognition** | The emergence of reflexive distance — an observing standpoint from which the contents of experience, including pain-related cognitions and affects, can be witnessed without identification. The body begins to re-emerge as something observed rather than something constituting. | *"I noticed I was getting anxious about the pain, and I could just watch that anxiety."* |
| 4 | **Reappraisal** | A transformation of the noematic structure of pain experience: pain decomposed into constituent sensations, losing its character as monolithic threatening event and becoming a complex, textured, changing field. | *"It's interesting, when I really look at it, the 'pain' is actually many different feelings."* |

The arc from Stage 0 to Stage 4 describes the progressive recovery of the body as a transparent medium of experience rather than a dysappearing obstacle — the theoretical process MORE is designed to produce.

The full operational definitions, including prototypical features, distinguishing criteria, exemplar utterances, subtle utterances, and adversarial utterances (the boundary-case language that separates adjacent stages), are implemented in [`theme_framework/vammr.py`](theme_framework/vammr.py).

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

Implemented in [`codebook/phenomenology_codebook.py`](codebook/phenomenology_codebook.py) as `CodeDefinition` objects with formal descriptions, subcodes, inclusive criteria, and exclusive criteria.

### Why Both: Orthogonal Analytical Dimensions

VAMMR is a *developmental stage model* — it answers *where in therapeutic progression*. VCE is a *content taxonomy* — it answers *what is being experienced at each location in that progression*. Their concurrent application generates evidence neither framework can produce alone.

The VAMMR framework encodes theoretically grounded expectations about which VCE codes should co-occur with each stage (to be pre-registered as an `expected_codes` field in [`theme_framework/vammr.py`](theme_framework/vammr.py) before Cohort 3):

- **Vigilance** is predicted to lift: *Fear/Anxiety/Panic/Paranoia*, *Agitation or Irritability*, *Pain (Somatic)* — consistent with dys-appearance as an affective-somatic phenomenon
- **Avoidance** is predicted to lift: *Affective Flattening*, *Emotional Detachment, or Alexithymia* — consistent with the emotional blunting documented in experiential avoidance strategies (Hayes et al., 1996)
- **Mindfulness** is predicted to lift: *Attention*, *Change in Duration of Experience*, *Somatic Relaxation or Calming* — consistent with sustained volitional attention to somatic experience with reduced effortful struggle
- **Metacognition** is predicted to lift: *Meta-Cognition*, *Clarity*, *Change in Narrative Self* — consistent with the emergence of reflexive self-observation
- **Reappraisal** is predicted to lift: *Positive Affect*, *Change in Worldview*, *Change in Self-Other or Self-World Boundaries*, *Disintegration of Conceptual Meaning Structures* — consistent with a transformation in the noematic structure of pain experience

Pipeline Stage 4 tests these predictions empirically through **lift statistics** — co-occurrence ratios that measure how strongly each VCE code appears with each VAMMR stage relative to its corpus-wide base rate. This cross-validation implements Varela's (1996) neurophenomenological logic of *mutual constraints*: two independently-derived phenomenological frameworks constrain each other through empirical co-occurrence, and their convergence — or non-convergence — is informative in both directions. Confirmed predictions constitute convergent validity evidence; disconfirmed predictions reveal where theoretical frameworks require revision in this population.

---

## The PURER Framework and Therapist Cue Analysis

Each MORE session follows a structured therapist inquiry format called PURER: **P**henomenology, **U**tilization, **R**eframing, **E**ducation/Expectancy, **R**einforcement. In the context of this analysis, PURER functions as a structured phenomenological interview method — analogous to Giorgi's (1985) descriptive phenomenological interview — systematically eliciting and consolidating participants' first-person reports of their experience.

The Phenomenology component (*"What did you notice during the practice?"*) is structurally similar to Husserlian epoché: the invitation to bracket naturalistic assumptions and describe experience as it presents itself. Reframing offers interpretive proposals that may shift participants' appraisive relationship to pain — what Giorgi calls free imaginative variation. Reinforcement consolidates participants' expressions of therapeutic insight.

This has a direct methodological implication. Therapist dialogue is not merely contextual background to participant phenomenological expression — it is a systematic *elicitor* of phenomenological description. The structure of PURER shapes which phenomenological reports participants offer; understanding therapeutic mechanism requires understanding that shaping.

QRA's **therapist cue-response analysis** makes this visible. Because the pipeline separates participant and therapist segments at Stage 1 and indexes them chronologically, it can retrieve — for every observed within-session VAMMR stage transition — the therapist dialogue that immediately preceded it. This produces an empirical characterization of which therapist behaviors are associated with each type of stage change.

The Avoidance → Mindfulness transition is the single most clinically important transition to monitor: it marks the crossing of the experiential avoidance barrier, where emerging attentional skill is redirected from suppression toward open, investigative presence. Examining the therapist dialogue at these transitions and mapping it onto its PURER component — Is the therapist using a Phenomenology inquiry? A Reframe? A Reinforcement? — answers what therapist behaviors precipitate the most therapeutically significant moments in the session. This directly informs therapist training and scripting for subsequent cohorts.

---

## What the Pipeline Produces

Given diarized session transcripts (from Whisper + speaker diarization):

1. **Semantically coherent segments** — embedding-based segmentation with adaptive thresholds that treats the semantically coherent utterance (not the speaker turn) as the unit of analysis, implemented in [`process/transcript_ingestion.py`](process/transcript_ingestion.py)
2. **VAMMR stage classifications** — multi-run LLM consensus voting with confidence tiering (High/Medium/Low) and auditable justifications citing specific participant language, via [`classification_tools/classification_loop.py`](classification_tools/classification_loop.py)
3. **VCE phenomenology codes** — multi-label coding via embedding similarity + LLM zero-shot ensemble, implemented in [`codebook/embedding_classifier.py`](codebook/embedding_classifier.py) and [`codebook/ensemble.py`](codebook/ensemble.py)
4. **Cross-validation lift statistics** — empirical (VAMMR stage × VCE code) co-occurrence ratios testing theoretically predicted phenomenological correlates, via [`process/cross_validation.py`](process/cross_validation.py)
5. **Human validation worksheets** — stratified evaluation sets for blind-coding and inter-rater reliability assessment
6. **Session-level stage progression summaries** — forward, backward, and lateral transition counts between adjacent participant segments
7. **Longitudinal trajectory reports** — group-level mean stage proportions per session number across participants
8. **Therapist cue-response reports** — therapist language grouped by transition type, characterizing the empirical PURER profile at each kind of stage change

---

## What Can Be Learned

The outputs support three levels of analysis, each addressing a different research question:

### Session level: Which sessions are catalyzing stage progression?

Session stage distributions, compared against the theoretical arc implied by session content, reveal where the curriculum is working and where it is not. Sessions where Avoidance remains dominant in weeks 5–7 — despite content explicitly designed to catalyze the Avoidance → Mindfulness transition (Sessions 3, 5) — are primary curriculum targets. Sessions where Reappraisal first emerges identify the intervention's active ingredients at the phenomenological level.

### Dyadic level: What therapist behaviors facilitate contemplative transformation?

The cue-response analysis directly addresses a limitation of most MBI process research: participant phenomenological reports are typically analyzed in isolation from the therapist behaviors that elicited them, obscuring the dyadic structure of the therapeutic interaction. QRA's session adjacency index makes this dyad visible at the level of individual transitions — allowing researchers to identify which PURER components appear in the therapist dialogue preceding the most significant stage changes and to directly translate these findings into therapist training and scripting recommendations.

### Population level: What is the phenomenological texture of each stage in this population?

The lift statistics answer a question that neither framework could address alone: what are participants *actually experiencing* at each stage of the developmental arc, and does the phenomenological texture of therapeutic transformation in a chronic pain MBI population match theoretical predictions derived from Buddhist meditators? High-lift associations that confirm the predicted (VAMMR stage, VCE code) co-occurrences constitute convergent validity evidence; unexpected associations constitute novel findings about the phenomenological character of MBI-induced transformation in this population.

### Longitudinal: Is there a coherent developmental trajectory?

A non-decreasing mean stage progression across session numbers is the basic validity check for the VAMMR model as a developmental framework in this population. If participants are, on average, expressing higher-stage language in later sessions, this is consistent with the model's theoretical structure. Departure from this pattern — particularly in movement-integrated sessions — is itself informative about where the intervention's phenomenological effects diverge from prediction.

---

## Validation: Text Psychometrics

Computational phenomenological classification requires a validation methodology appropriate to the distinctive epistemic situation: third-person computational tools classifying first-person experiential reports. Low, Mair, Nock, and Ghosh (2024) propose *Text Psychometrics* as a framework specifically designed for this purpose, adapting classical psychometric validity theory to text-based psychological measurement. QRA implements each component:

- **Content validity** — via the known-label test set exported at Stage 2 (`content_validity_test_set.jsonl`), comprising the exemplar, subtle, and adversarial utterances from each stage definition. Running the pipeline against these known-label utterances measures how well the classifier handles both clear instances and conceptual boundary cases.
- **Construct validity** — via the Stage 4 lift statistics, which constitute convergent validity hypotheses: if both frameworks are accurately tracking phenomenological states, the predicted (stage, code) co-occurrences should appear in the empirical data.
- **Reliability** — via multi-run consensus (`llm_run_consistency`), treating independent LLM calls as proxy raters. Full cross-run agreement is the computational analog of unanimous inter-rater agreement; confidence tiering integrates run consistency with per-run confidence into a single reliability indicator.

Human validation of the stratified evaluation set by two independent rater teams, with Krippendorff's alpha and raw agreement statistics, provides the ultimate validation standard. Agreement ≥ 75% (κ ≥ 0.60) supports using classifications as primary evidence for curriculum modification; lower agreement rates define tiered evidence standards described in the validation protocol.

The epistemological limit that applies throughout: QRA classifies *linguistic expressions of experience*, not experience itself. This limitation applies equally to human qualitative coders. The pipeline's value is in directing expert attention efficiently toward the cases, sessions, and transition moments where expert phenomenological judgment is most consequential — not in replacing it.

---

## Research Context

QRA was developed in service of the **Move-MORE Feasibility Trial** — an adaptation of Mindfulness-Oriented Recovery Enhancement (MORE) that integrates movement-based therapy for patients with lumbosacral radicular pain (LRP). Move-MORE uses an iterative convergent design across four cohorts; each cohort's session transcripts must be analyzed before the curriculum for the next cohort is finalized. Cohorts 1 and 2 have completed the eight-week intervention; Cohorts 3 and 4 are pending.

The eight-session MORE for Pain protocol proceeds from psychoeducation about pain and mindfulness (Sessions 1–2) through mindful reappraisal and savoring (Sessions 3–4), unhealthy coping and stress–pain relationships (Sessions 5–6), thought suppression and meaning-making (Session 7), to sustained practice planning (Session 8). Each session follows a standard structure anchored by PURER-guided practice debriefing. Move-MORE integrates movement practice within this structure, introducing a kinesthetic vocabulary — awareness of the body in motion, proprioceptive attention, the relationship between movement and pain sensation — that may express VAMMR stages through different linguistic surface forms than the verbal mindfulness exercises on which the framework was originally derived.

---

## Getting Started

- **[USAGE.md](USAGE.md)** — Installation, LLM backend configuration, pipeline commands, output structure, configuration reference
- **[methodology.txt](methodology.txt)** — Full neurophenomenological methodology paper (Journal of Phenomenology and the Cognitive Sciences)
- **[methodology_v2.txt](methodology_v2.txt)** — Applied methodology paper oriented toward clinical trial researchers (Journal of Contemplative Studies)

```bash
python qra.py setup    # interactive 12-step configuration wizard
python qra.py run --config ./data/output/07_meta/qra_config.json
python qra.py analyze --output-dir ./data/output/
```

**LLM backend options:** LM Studio (local), OpenRouter, Replicate, Ollama, HuggingFace — any OpenAI-compatible endpoint. The codebook embedding classifier uses Qwen/Qwen3-Embedding-8B by default and requires GPU for reasonable throughput.

---

## Development Roadmap

The pipeline is currently being validated on Move-MORE Cohorts 1 and 2. Engineering priorities before the final cohort:

1. **PURER classification of therapist dialogue** — extend classification to therapist segments, operationalizing the five PURER components as a therapist-side framework. Infrastructure already exists in Stage 1 speaker separation and the session adjacency index; requires a `theme_framework/purer.py` `ThemeFramework` definition and an aggregation layer in the cue-response report. This would transform the cue analysis from a content summary into a genuine PURER fidelity assessment.

2. **Context-window expansion** — add preceding-segment context to classification prompts to improve accuracy at stage boundaries, particularly the Avoidance–Mindfulness and Mindfulness–Metacognition transitions. Data structures already capture `segment_index` and `session_id`; the missing piece is a context-building function in [`classification_tools/llm_classifier.py`](classification_tools/llm_classifier.py).

3. **Movement language adaptation** — if human validation reveals systematic disagreement in movement-specific sessions, augment VAMMR stage definitions with kinesthetic exemplar utterances drawn from human-validated Cohorts 1–2 segments.

4. **Supervised fine-tuning** — the `master_segments.jsonl` files produced across all four cohorts will constitute the first systematically labeled corpus of VAMMR stage expression in mindfulness-based pain therapy, formatted for supervised fine-tuning. A domain-adapted classifier would enable therapeutic fidelity monitoring in future trials at a scale and cost that zero-shot LLM classification does not support.

---

## Publications in Preparation

Two methodology papers are in preparation, grounded in this pipeline:

**Balsamo, W., Wexler, R. S., et al.** — "From Vigilance to Reappraisal: A Computational Neurophenomenological Method for Analyzing Contemplative Transformation in Mindfulness-Based Pain Therapy." *In preparation for Journal of Phenomenology and the Cognitive Sciences.* [`methodology.txt`](methodology.txt)

**Balsamo, W., Wexler, R. S., Fox, D. J., Garland, E. L., et al.** — "Computational Phenomenology in Mindfulness-Based Interventions for Chronic Pain: A Machine-Assisted Methodology for Rapid Iterative Curriculum Refinement." *In preparation for Journal of Contemplative Studies.* [`methodology_v2.txt`](methodology_v2.txt)

---

## Prior Work

This project is built directly upon the following bodies of work:

**VA-MR Framework**
- Wexler, R. S., Balsamo, W., et al. (2026). "Noticing the Way that I'm Noticing Pain": A qualitative analysis of therapeutic progression in mindfulness-oriented recovery enhancement for patients with lumbosacral radicular pain. *Mindfulness*, 17, 819–833. Preprint: [https://doi.org/10.21203/rs.3.rs-7104279/v1](https://doi.org/10.21203/rs.3.rs-7104279/v1)

**Move-MORE Feasibility Trial**
- Wexler, R. S., Balsamo, W., et al. (2026). Development and pilot feasibility testing of Move-MORE: A multicomponent mindfulness-and-movement intervention for lumbosacral radicular pain. Preprint: [https://doi.org/10.21203/rs.3.rs-8682836/v1](https://doi.org/10.21203/rs.3.rs-8682836/v1)

**MORE for LRP Randomized Controlled Trial**
- Wexler, R. S., Fox, D. J., ZuZero, D., et al. (2024). Virtually delivered MORE reduces daily pain intensity in patients with lumbosacral radiculopathy: A randomized controlled trial. *Pain Reports*, 9(2), e1132.
- Wexler, R. S. (2022). Protocol for mindfulness-oriented recovery enhancement (MORE) in the management of lumbosacral radiculopathy/radiculitis symptoms: A randomized controlled trial. *Contemporary Clinical Trials Communications*.

**VCE Phenomenology Codebook**
- Lindahl, J. R., Fisher, N. E., Cooper, D. J., Rosen, R. K., & Britton, W. B. (2017). The varieties of contemplative experience: A mixed-methods study of meditation-related challenges in Western Buddhists. *PLOS ONE*, 12(5), e0176239. [https://doi.org/10.1371/journal.pone.0176239](https://doi.org/10.1371/journal.pone.0176239)

**Text Psychometrics Validation Framework**
- Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text psychometrics: Assessing psychological constructs in text using natural language processing. *Psychological Methods*.

**Foundational Phenomenology**
- Leder, D. (1990). *The Absent Body.* University of Chicago Press.
- Merleau-Ponty, M. (1962). *Phenomenology of Perception.* Routledge.
- Varela, F. J. (1996). Neurophenomenology: A methodological remedy for the hard problem. *Journal of Consciousness Studies*, 3(4), 330–349.
- Varela, F. J., Thompson, E., & Rosch, E. (1991). *The Embodied Mind.* MIT Press.

**MORE Efficacy and Mechanism**
- Garland, E. L. (2024). *Mindfulness-Oriented Recovery Enhancement: An Evidence-Based Treatment for Chronic Pain and Opioid Use.* The Guilford Press.
- Garland, E. L., Hanley, A. W., et al. (2022). Mindfulness-oriented recovery enhancement vs supportive group therapy for co-occurring opioid misuse and chronic pain in primary care. *JAMA Internal Medicine*, 182(4), 407–417.
- Riegner, G., Posey, G., Oliva, V., Jung, Y., Mobley, W., & Zeidan, F. (2023). Disentangling self from pain: Mindfulness meditation-induced pain relief is driven by thalamic-default mode network decoupling. *Pain*, 164(2), 280–291.
- Hanley, A. W., Bernstein, A., et al. (2020). The metacognitive processes of decentering scale. *Psychological Assessment*, 32(10), 956–971.
- Hanley, A. W., & Garland, E. L. (2021). The mindfulness-oriented recovery enhancement fidelity measure (MORE-FM). *Journal of Evidence-Based Social Work*, 18(3), 308–322.

**Iterative Trial Design**
- Alwashmi, M. F., Hawboldt, J., Davis, E., & Fetters, M. D. (2019). The iterative convergent design for mobile health usability testing. *JMIR Mhealth and Uhealth*, 7(4), e11656.

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
