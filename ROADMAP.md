# QRA Research Roadmap

This document tracks the research and engineering trajectory for the Qualitative Research Algorithm across the remaining Move-MORE cohorts and beyond. It is organized into three phases: completing the dual-framework pipeline before Cohort 3, fine-tuning supervised classifiers during Cohorts 3–4, and building a graph-neural-network model for therapist-participant dynamics post-trial.

---

## Current State

The pipeline is validated on Move-MORE Cohorts 1 and 2:

- **Participant-side**: VA-MR stage classification (zero-shot LLM, multi-run consensus, confidence tiering) ✓
- **VCE codebook**: 59-code, 7-domain phenomenology classification (embedding + LLM ensemble) ✓
- **Cross-validation**: VA-MR × VCE lift statistics testing theoretical stage-phenomenology predictions ✓
- **Human validation infrastructure**: balanced evaluation set, blind-coding worksheets ✓
- **Analysis reports**: session summaries, longitudinal trajectories, transition matrices, therapist cue-response ✓

**Not yet built:**
- Therapist-side PURER classification
- Supervised/fine-tuned classifiers
- Graph-neural-network model of therapist-participant dynamics

---

## Phase 1 — Complete the Dual-Framework Pipeline

**Target: before Cohort 3 begins**

### 1.1 PURER Codebook and Therapist Classification

**Effort:** 3–5 days (codebook authorship is the binding constraint)
**Value:** unlocks the empirical test of Wexler, Balsamo et al.'s (2026) central finding — that PURER-guided phenomenological inquiry is the mechanism by which participants cross the Avoidance → Metacognition barrier

**What to build:**

Author `constructs/purer.py` as a `ThemeFramework` with five `ThemeDefinition` objects:

| Step | Definition focus | Key distinguishing criteria |
|------|------------------|-----------------------------|
| **Phenomenology** | Open-ended inquiry into moment-to-moment meditative experience; eliciting what the participant noticed | Distinguish from Education: Phenomenology asks, Education tells |
| **Utilization** | Applying what was learned in session to daily-life pain coping | Distinct from Reinforcement: Utilization bridges to outside the session |
| **Reframing** | Offering alternative interpretations of pain or practice difficulty as the practice itself | Distinct from Education: Reframing recontextualizes a specific experience rather than explaining a principle |
| **Education/Expectancy** | Explaining neurobiological or psychological mechanisms; building expectation of benefit | Distinct from Phenomenology: Education delivers content, does not solicit |
| **Reinforcement** | Positive acknowledgment of participant effort, progress, or insight | Distinct from Utilization: Reinforcement consolidates what happened in session, not its application outside |

Each definition needs: formal description, 6–8 prototypical features, distinguishing criteria vs. adjacent steps, 4 clear exemplar utterances, 3 subtle utterances, 3 adversarial utterances (especially Phenomenology/Education and Reframing/Education boundary cases). Source material: Table 1 and Table 2 of Wexler et al. (2026), existing Cohorts 1–2 therapist transcripts.

**Pipeline changes:**
- Add `retain_therapist_segments` flag to bypass speaker filter for therapist-role speakers while keeping the context-expansion pipeline intact
- Add a third classification stage in `process/orchestrator.py` pointing the existing LLM classification loop at therapist segments with the PURER framework
- Extend `process/dataset_assembly.py` to tag each row with `speaker_framework` (`vamr` for participants, `purer` for therapists)
- Add a PURER cross-framework influence table to `process/cross_validation.py`: for each therapist PURER step, the distribution of subsequent participant VA-MR stages — the direct empirical test of the PURER facilitation hypothesis

**Deliverable:** `master_segments.jsonl` with PURER-labeled therapist rows; `purer_to_vamr_influence_table.json` showing empirical PURER step → VA-MR transition distributions.

---

### 1.2 Windowed Context in Theme Classification

**Effort:** 1–2 days
**Value:** reduces false Reappraisal classifications in segments that echo session language without reflecting a genuine stage shift

The `theme_classification.context_window_segments` config option and `_build_context_block` helper in `classification_tools/llm_classifier.py` already exist but default to 2 without end-to-end verification. Surface this as a first-class config option in the setup wizard, verify the context block is formatted correctly in the prompt, and re-evaluate on `content_validity_test_set.jsonl` at window = 0, 1, 2, 3 before deploying to real transcripts.

The specific problem this solves: a participant utterance expressing apparent Reappraisal ("it's just sensation now") immediately following an Avoidance-dominant passage may reflect rote repetition of session vocabulary rather than a genuine stage shift. Explicit preceding context makes this distinction tractable.

**Deliverable:** working windowed context, wizard option, content validity test set comparison across window sizes.

---

### 1.3 Outcome Integration Layer

**Effort:** 1–2 days
**Value:** enables the convergent evidence section of the modification report to be auto-populated rather than assembled manually

Author `process/outcome_integration.py` that accepts a path to a trial outcomes CSV (weekly pain NRS, practice completion rate, ODI, any pre-specified change scores) and joins it to the master dataset on `session_id + participant_id`. Produce a session-level summary table with pain intensity, practice completion, and VA-MR stage distribution in the same row. This is the quantitative column in the joint display described in the Move-MORE protocol.

**Deliverable:** `04_analysis_data/session_outcomes_integrated.csv`; updated longitudinal report incorporating outcome columns.

---

### 1.4 Human Validation of Cohorts 1–2

**Effort:** 1–2 weeks (human rater scheduling is the constraint)
**Value:** without validated kappa ≥ 0.60 on the VA-MR evaluation set, no classification result should be presented as primary evidence

Two independent rater teams blind-code `05_validation/human_coding_evaluation_set.csv` following the protocol in `methodology.txt` Section 7. Before coding real segments, each team works through `content_validity_test_set.jsonl` to calibrate on VA-MR boundary cases. After coding, compute raw agreement and Krippendorff's alpha.

- **Agreement ≥ 75% and kappa ≥ 0.60**: classifications support curriculum modification recommendations
- **Agreement 50–74% or kappa 0.40–0.59**: directional evidence only; frame recommendations as hypotheses for Cohorts 3–4
- **Below 50% / kappa < 0.40**: investigate framework fit before proceeding

The validated Cohort 1–2 labels become the initial training signal for Phase 2 fine-tuning.

---

## Phase 2 — Supervised Fine-Tuning with Autoresearch

**Target: concurrent with Cohorts 3–4**

Once Cohort 1–2 validation is complete, the pipeline produces its first systematically labeled corpus of VA-MR and PURER expression in a mindfulness-based pain intervention. Phase 2 uses this corpus to fine-tune domain-adapted classifiers — replacing the zero-shot LLM with a supervised model that is faster, cheaper, and validated on real Move-MORE language.

### 2.1 Export the Fine-Tuning Corpus

Filter `master_segments.jsonl` to `label_confidence_tier in ["high", "medium"]` and rows with a confirmed `human_label`. Structure as labeled examples for sequence classification:

```
{"text": "<segment_text>", "label": <va_mr_stage_int>, "source": "cohort1_validated"}
{"text": "<therapist_segment_text>", "label": <purer_step_int>, "source": "cohort2_purer"}
```

Keep the validation split fixed (the same segments in `human_coding_evaluation_set.csv`) so every autoresearch experiment is scored against the same held-out benchmark.

---

### 2.2 Autoresearch: VA-MR Classifier Search

**What autoresearch does:** an LLM-driven search controller that repeatedly edits `train.py`, runs a fixed-duration training job, scores the result with a fixed metric, and uses `git reset` to retain only improvements. The human contributes the research agenda through `program.md`; the agent discovers the implementation through overnight search.

**Adapt for this task:**

- **Metric**: swap `val_bpb` (bits per byte, for language modeling) for `val_macro_f1` on the held-out VA-MR 4-class classification task
- **Fixed run length**: 5 minutes is appropriate — BERT fine-tuning for 4 classes on ~500–1000 examples converges in well under 5 minutes on a single GPU
- **`prepare.py`** (read-only): dataset loading, tokenization, and the held-out validation split
- **`train.py`** (agent edits): model choice, frozen layers, classification head depth, learning rate, batch size, label smoothing

**`program.md` search agenda:**

```
Goal: maximize val_macro_f1 on 4-class VA-MR stage classification
     from therapy session transcript segments (~50–300 words each)

Hypotheses to explore (in this order):
1. BERT-base vs. ClinicalBERT vs. BioBERT as the encoder backbone
2. Layer freezing depth: freeze all but last 2 layers (data-efficient baseline),
   then progressively unfreeze layers if val_macro_f1 plateaus
3. Classification head: single linear → [768, 256, 4] bottleneck → [768, 128, 4]
4. Learning rate: 2e-5 (BERT default) → 1e-5 → 3e-5
5. Label smoothing: 0.0 vs. 0.1 (boundary cases between adjacent VA-MR stages
   are genuinely ambiguous; smoothing may help)

Constraints:
- Do not modify prepare.py or the validation split
- Do not use the test set during search; val_macro_f1 on the fixed validation
  split is the only allowed metric
- Log every experiment result with the hypothesis being tested

Known challenge: Avoidance (Stage 1) and Vigilance (Stage 0) share pain-focused
language; Avoidance and Metacognition (Stage 2) share attentional vocabulary.
Prioritize improving per-class F1 on these boundary classes.
```

**Deliverable:** best `train.py` checkpoint with reproducible val_macro_f1; experiment log across the search.

---

### 2.3 Autoresearch: PURER Classifier Search

Same structure as 2.2 with a separate `program.md` targeting 5-class PURER step classification on therapist segments. The key boundary case is Phenomenology vs. Education (both involve discussing experience; Phenomenology asks, Education explains) — flag this explicitly in `program.md`.

**Deliverable:** best PURER classifier checkpoint; val_macro_f1 per PURER step.

---

### 2.4 Prospective Validation on Cohorts 3–4

Run both fine-tuned classifiers on Cohorts 3–4 as they complete, alongside the existing zero-shot LLM pipeline. Track agreement between the two systems per session:

- **High agreement**: both systems classify the same — use fine-tuned model output directly
- **Disagreement**: route to human review; disagreements between a validated fine-tuned model and a zero-shot LLM are the most informative cases for framework refinement

Each Cohort 3–4 session that clears human validation extends the labeled corpus. By the end of Cohort 4, the accumulated dataset spans four cohorts and two intervention variants, constituting the largest systematically labeled corpus of VA-MR stage expression in an MBI.

---

## Phase 3 — CFiCS-Style GNN for Therapist-Participant Dynamics

**Target: post-Cohort 4 analysis and manuscript**

### Rationale

CFiCS (Schmidt et al., 2024) demonstrated that combining domain-adapted BERT embeddings with GraphSAGE message-passing over a therapeutic taxonomy graph substantially outperforms either approach alone for psychotherapy content classification: ClinicalBERT alone achieved ~60% micro F1 on fine-grained skill classification; ClinicalBERT + GraphSAGE achieved ~96%. The key result is that graph structure encoding theoretical relationships between therapeutic concepts — which skills instantiate which higher-level factors — enables the model to learn distinctions that text embeddings alone cannot resolve.

The PURER → VA-MR relationship in Move-MORE is structurally identical to this problem: which therapist behaviors (PURER steps) precipitate which participant outcomes (VA-MR stage transitions)? This is a hierarchical, relational, multi-level classification task. A GNN over a VA-MR × PURER taxonomy graph, with node features from a fine-tuned MindfulBERT encoder, is the natural architecture.

The critical enabling condition — which CFiCS lacked — is a labeled corpus of real clinical interactions. The autoresearch fine-tuned classifiers from Phase 2 produce exactly this. CFiCS substituted synthetic examples; this project does not need to.

---

### 3.1 Build the VA-MR × PURER Taxonomy Graph

Construct a heterogeneous graph parallel to the CFiCS therapeutic taxonomy:

**Node types:**
- VA-MR stage nodes (4): Vigilance, Avoidance, Metacognition, Reappraisal
- PURER step nodes (5): Phenomenology, Utilization, Reframing, Education/Expectancy, Reinforcement
- Transition nodes (3): Forward (stage increase), Lateral (same stage), Backward (stage decrease)
- Example utterance nodes: participant segments (VA-MR labeled) and therapist segments (PURER labeled) from the full 4-cohort validated corpus

**Edge types** (parallel to CFiCS `fosters`, `expresses`, `demonstrates`):
- `precipitates`: therapist PURER step → participant VA-MR transition (weighted by empirical lift from Phase 1 cross-validation)
- `demonstrates`: example utterance node → VA-MR stage or PURER step node
- `co-occurs_with`: VA-MR stage × VA-MR stage temporal adjacency within sessions
- `supports`: PURER step → VA-MR stage sustained across session (not just at transition moments)

Populate edge weights from the `purer_to_vamr_influence_table.json` produced in Phase 1. The graph encodes what theory predicts (edge types) and what the data shows (edge weights) simultaneously.

---

### 3.2 Node Features from Fine-Tuned MindfulBERT

Use the autoresearch-optimized BERT checkpoint from Phase 2 as the text encoder. For each node, encode its text by average-pooling the last hidden state:

- Stage/step nodes: concatenation of name, definition, and prototypical features from the framework definition
- Example nodes: the raw segment text

This is the exact CFiCS approach, but the encoder has been fine-tuned on Move-MORE session transcripts with VA-MR and PURER labels. The MindfulBERT embeddings encode the Avoidance/Metacognition boundary and the Phenomenology/Education boundary in a way that ClinicalBERT — trained on clinical notes with no exposure to VA-MR or PURER — cannot.

CFiCS reported ~10–15% performance gain from ClinicalBERT vs. generic BERT. Expect a similar or larger gain from MindfulBERT vs. ClinicalBERT on this domain.

---

### 3.3 GraphSAGE Training

Multi-task objective over three prediction heads:

| Task | Classes | Loss |
|------|---------|------|
| VA-MR stage | 4 | Cross-entropy |
| PURER step | 5 | Cross-entropy |
| Transition type | 3 (forward/lateral/backward) | Cross-entropy |

Combined loss: `ℒ = λ₁ℒ_vamr + λ₂ℒ_purer + λ₃ℒ_transition`

GraphSAGE's inductive capability means the model trained on Cohorts 1–3 can classify Cohort 4 sessions without graph restructuring — only the new segments need to be embedded with the text encoder and passed through the learned aggregation weights. This is the property that makes the model deployable for future trials.

Autoresearch can be applied here as well: `train.py` controls GNN depth, aggregation function (mean vs. pool vs. LSTM), hidden dimension, and the task weight coefficients `λ₁, λ₂, λ₃`. The fixed metric is macro F1 on the held-out Cohort 4 validation set.

---

### 3.4 Interpretation for MBI Development and Publication

The t-SNE visualization of learned embeddings (analogous to CFiCS Figure 3) will reveal whether PURER steps cluster near the VA-MR transitions they theoretically precipitate. Specifically:

- Do **Reframing** and **Phenomenology** embeddings cluster near the Avoidance → Metacognition transition zone? This is the primary prediction from Wexler, Balsamo et al. (2026)
- Does **Reinforcement** cluster near Reappraisal → Reappraisal (lateral) transitions? This would confirm that Reinforcement consolidates Reappraisal gains rather than producing new transitions
- Do **Education/Expectancy** embeddings cluster near Stage 0 (Vigilance) transitions? This would show that psychoeducation is most active in early sessions, consistent with the MORE session structure

Confirming these predictions provides the first graph-structured, empirically validated account of the PURER facilitation mechanism — extending the qualitative finding of Wexler et al. (2026) into a quantitative, reproducible model.

**Target venue:** *Psychotherapy Research* or *Journal of Consulting and Clinical Psychology* for the GNN results; the VA-MR + PURER classification system as a standalone contribution to *npj Digital Medicine* or *JMIR Mental Health*.

---

## Summary

| Phase | Item | Deliverable | Effort | Status |
|-------|------|-------------|--------|--------|
| 1.1 | PURER codebook + therapist classification | `purer_to_vamr_influence_table.json` | 3–5 days | Not started |
| 1.2 | Windowed context classification | Wizard option, validated on test set | 1–2 days | Partial (code exists) |
| 1.3 | Outcome integration layer | `session_outcomes_integrated.csv` | 1–2 days | Not started |
| 1.4 | Human validation, Cohorts 1–2 | Kappa ≥ 0.60, agreement ≥ 75% | 1–2 weeks | Not started |
| 2.1 | Fine-tuning corpus export | Labeled JSONL for BERT training | 1 day | Blocked on 1.4 |
| 2.2 | Autoresearch VA-MR fine-tuning | Best VA-MR classifier checkpoint | 1 day setup + overnight | Blocked on 2.1 |
| 2.3 | Autoresearch PURER fine-tuning | Best PURER classifier checkpoint | 1 day setup + overnight | Blocked on 1.1, 2.1 |
| 2.4 | Cohorts 3–4 prospective validation | Extended labeled corpus | Concurrent with trial | Blocked on 2.2–2.3 |
| 3.1 | VA-MR × PURER taxonomy graph | Graph with empirical edge weights | 2–3 days | Blocked on 1.1, 2.4 |
| 3.2–3.3 | CFiCS-style GNN training | Trained GraphSAGE model | 1–2 weeks | Blocked on 3.1, 2.2 |
| 3.4 | Interpretation + manuscript | Published findings | Ongoing | Blocked on 3.3 |

**Critical path:** PURER codebook authorship (1.1) → validated Cohort 1–2 labels (1.4) → fine-tuning corpus (2.1) → autoresearch fine-tuning (2.2–2.3) → Cohorts 3–4 prospective data (2.4) → GNN training (3.1–3.3).

The PURER codebook is the binding constraint. Everything downstream depends on having operationally distinct, exemplar-grounded definitions for the five PURER steps before Cohort 3 begins.

---

## Key Files

| File | Purpose |
|------|---------|
| `constructs/vamr.py` | VA-MR framework definition (participant-side) |
| `constructs/purer.py` | PURER framework definition (therapist-side) — **to build** |
| `codebook/phenomenology_codebook.py` | VCE 59-code phenomenology codebook |
| `process/cross_validation.py` | VA-MR × VCE lift statistics; extend for PURER × VA-MR influence table |
| `process/outcome_integration.py` | Quantitative outcome join — **to build** |
| `methodology.txt` | Full methodology paper (Phenomenology and the Cognitive Sciences) |

## References

- Schmidt, F., Hammerfald, K., Jahren, H. H., & Vlassov, V. (2024). CFiCS: Graph-based classification of common factors and microcounseling skills. *KTH Royal Institute of Technology*.
- Wexler, R. S., Balsamo, W., et al. (2026). "Noticing the Way that I'm Noticing Pain." *Mindfulness*, 17, 819–833.
- Wexler, R. S., Balsamo, W., et al. (2026). Development and pilot feasibility testing of Move-MORE. Preprint. doi:10.21203/rs.3.rs-8682836/v1
- Lindahl, J. R., Fisher, N. E., Cooper, D. J., Rosen, R. K., & Britton, W. B. (2017). The Varieties of Contemplative Experience. *PLOS ONE*, 12(5), e0176239.
- Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text Psychometrics. *Psychological Methods*.
