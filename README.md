# QRA: Qualitative Research Algorithm

**Computational phenomenology for mindfulness-based intervention research.**

QRA is a machine-assisted qualitative analysis pipeline that classifies therapy transcript segments using two established phenomenological frameworks: (1) the Vigilance-Avoidance Metacognition-Reappraisal (VA-MR) model of therapeutic progression, and (2) the Varieties of Contemplative Experience (VCE) phenomenology codebook. This process is designed for producing defensible cohort-level qualitative analysis in days rather than weeks, enabling iterative refinement of program development.

---

## Why QRA Exists

Iterative clinical trial design creates an epistemological bind: the between-cohort refinement window is measured in weeks, but rigorous qualitative analysis of therapy transcripts takes months. The result is that curriculum modifications are routinely made on the basis of clinical intuition and aggregate outcome scores rather than systematic analysis of the phenomenological processes the intervention is designed to produce.

QRA was built to dissolve this tension. By operationalizing established phenomenological frameworks into an automated pipeline, it makes the qualitative evidence that should inform curriculum decisions available on the timeline that iterative trial design actually affords.

The methodology is described in full in [`methodology.txt`](methodology.txt).

---

## Research Context

QRA was developed in service of the **Move-MORE Feasibility Trial** — an adaptation of Mindfulness-Oriented Recovery Enhancement (MORE; Garland 2013) that integrates movement-based therapy for patients with lumbosacral radicular pain (LRP). Move-MORE uses an iterative convergent design across four cohorts, where each cohort's session transcripts must be analyzed before the curriculum for the next cohort is finalized.

**Prior work this builds on:**

- **Wexler, Balsamo et al. (2026)** — "Noticing the Way that I'm Noticing Pain": Qualitative derivation of the VA-MR framework from MORE session transcripts. *Mindfulness* (preprint: https://doi.org/10.21203/rs.3.rs-7104279/v1)
- **Wexler et al. (2026)** — Move-MORE Feasibility Trial protocol. (preprint: https://doi.org/10.21203/rs.3.rs-8682836/v1)
- **Wexler et al. (2024)** — MORE for LRP randomized controlled trial results. *Pain Reports*, 9(2):e1132.
- **Wexler et al. (2022)** — MORE for LRP RCT protocol. *Contemporary Clinical Trials Communications*.
- **Lindahl et al. (2017)** — Varieties of Contemplative Experience (VCE) phenomenology codebook. *PLOS ONE*, 12(5):e0176239.
- **Low et al. (2024)** — Text Psychometrics: Assessing Psychological Constructs in Text Using Natural Language Processing. *Psychological Methods*.

---

## Dual-Framework Design

QRA applies two frameworks concurrently to the same transcript corpus. They address categorically different questions:

### VA-MR: Where in therapeutic progression?

Derived from qualitative analysis of MORE sessions (Wexler, Balsamo et al., 2026), VA-MR characterizes four stages of mindfulness skill development:

| Stage | Key | Description |
|-------|-----|-------------|
| 0 | **Vigilance** | Pain hypervigilance and attention dysregulation |
| 1 | **Avoidance** | Attention control deployed for experiential avoidance |
| 2 | **Metacognition** | Observing mental processes without identification |
| 3 | **Reappraisal** | Fundamental reinterpretation of sensory experience |

### VCE: What are they phenomenologically experiencing?

The seven-domain Varieties of Contemplative Experience codebook (Lindahl et al., 2017) characterizes the texture of first-person experience, covering 59 codes across Affective, Cognitive, Conative, Perceptual, Sense of Self, Social, and Somatic domains. Implemented in [`codebook/phenomenology_codebook.py`](codebook/phenomenology_codebook.py).

### Cross-validation between frameworks

Stage 4 of the pipeline computes **lift statistics** — empirical measures of how strongly each VCE code co-occurs with each VA-MR stage — that serve simultaneously as a validity check on the pipeline and as a novel contribution to understanding the phenomenological correlates of mindfulness skill development in a chronic pain population.

---

## What the Pipeline Produces

Given diarized session transcripts (from Whisper + speaker diarization):

1. **Semantically coherent segments** of participant dialogue
2. **VA-MR stage classifications** via multi-run LLM consensus voting (with interrater reliability metrics)
3. **VCE phenomenology codes** per segment via embedding + LLM ensemble  
4. **Cross-validation lift statistics** between the two frameworks (optional, requires VCE)
5. **Human validation worksheets** for blind-coding and IRR comparison
6. **Longitudinal analysis reports** — per-participant progression, per-session summaries, stage transition explanations, therapist cue-response analysis, graph-ready CSVs

---

## Getting Started

- **[SETUP.md](SETUP.md)** — Installation, LLM backend configuration, input data format
- **[USAGE.md](USAGE.md)** — Pipeline commands, configuration reference, output structure
- **[methodology.txt](methodology.txt)** — Full methodology paper describing design decisions and validation approach

```bash
python qra.py setup    # interactive 12-step configuration wizard
python qra.py run --config ./data/output/07_meta/qra_config.json
python qra.py analyze --output-dir ./data/output/
```

---

## Continuing Research

### Move-MORE Cohorts 3 & 4

The pipeline is currently being validated on Move-MORE Cohorts 1 and 2. The remaining engineering priorities before the final cohort are:

1. **PURER fidelity assessment** — extend classification to therapist dialogue, operationalizing the five PURER guided inquiry elements (Phenomenology, Utilization, Reframing, Education/Expectancy, Reinforcement) as a therapist-side framework parallel to VA-MR
2. **VA-MR framework refinement** — incorporate findings from Cohorts 1–2 cross-validation lift analysis to sharpen stage boundary definitions
3. **Human validation expansion** — increase blind-coded test set coverage and compute final Krippendorff's alpha across all cohorts

### VA-MR Research

The VA-MR framework was derived from a single-site qualitative study (Wexler, Balsamo et al., 2026). QRA's longitudinal stage progression data from Move-MORE provides a quantitative test of the framework's theoretical structure and an opportunity to examine whether stage progression varies systematically with pain phenotype, movement capacity, or therapist technique.

### Extending to Other MBIs

The pipeline architecture is framework-agnostic. Any intervention whose therapeutic progression can be operationalized as a set of classifiable stages can substitute for VA-MR; any phenomenological codebook with defined inclusive and exclusive criteria can substitute for VCE. See [USAGE.md — Custom Frameworks](USAGE.md#custom-frameworks).

---

## Citation

```bibtex
@software{QRA2026,
  title  = {QRA: Qualitative Research Algorithm},
  author = {Balsamo, Wade and Wexler, Ryan S},
  year   = {2026},
  note   = {Computational phenomenology pipeline for mindfulness-based intervention research}
}
```

## License

MIT License — see [LICENSE](LICENSE).
