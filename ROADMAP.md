# QRA Research Roadmap

This document tracks the research and engineering trajectory for the Qualitative Research Algorithm across the remaining Move-MORE cohorts and beyond. It is organized into three phases: completing the dual-framework pipeline before Cohort 3, fine-tuning supervised classifiers during Cohorts 3–4, and building a graph-neural-network model for therapist-participant dynamics post-trial.

---

## Current State

The pipeline is validated on Move-MORE Cohorts 1 and 2:

- **Participant-side**: VAAMR stage classification (zero-shot LLM, multi-run consensus, confidence tiering) ✓
- **VCE codebook**: 59-code, 7-domain phenomenology classification (embedding + LLM ensemble) ✓
- **Cross-validation**: VAAMR × VCE lift statistics testing theoretical stage-phenomenology predictions ✓
- **Human validation infrastructure**: balanced evaluation set, blind-coding worksheets ✓
- **Analysis reports**: session summaries, longitudinal trajectories, transition matrices, therapist cue-response ✓
- **Therapist-side PURER classification**: Five PURER constructs operationalized and applied to therapist dialogue; cue-response analysis now includes PURER move distribution ✓

**Not yet built:**
- Supervised/fine-tuned classifiers
- Graph-neural-network model of therapist-participant dynamics
 
---

### 1.3 Outcome Integration Layer

**Effort:** 1–2 days
**Value:** enables the convergent evidence section of the modification report to be auto-populated rather than assembled manually

Author `process/outcome_integration.py` that accepts a path to a trial outcomes CSV (weekly pain NRS, practice completion rate, ODI, any pre-specified change scores) and joins it to the master dataset on `session_id + participant_id`. Produce a session-level summary table with pain intensity, practice completion, and VAAMR stage distribution in the same row. This is the quantitative column in the joint display described in the Move-MORE protocol.

**Deliverable:** `04_analysis_data/session_outcomes_integrated.csv`; updated longitudinal report incorporating outcome columns.

---

### 1.4 Human Validation of Cohorts 1–2

**Effort:** 1–2 weeks (human rater scheduling is the constraint)
**Value:** without validated kappa ≥ 0.60 on the VAAMR evaluation set, no classification result should be presented as primary evidence

Two independent rater teams blind-code `05_validation/human_coding_evaluation_set.csv` following the protocol in `methodology.md` Section 7. Before coding real segments, each team works through `content_validity_test_set.jsonl` to calibrate on VAAMR boundary cases. After coding, compute raw agreement and Krippendorff's alpha.

- **Agreement ≥ 75% and kappa ≥ 0.60**: classifications support curriculum modification recommendations
- **Agreement 50–74% or kappa 0.40–0.59**: directional evidence only; frame recommendations as hypotheses for Cohorts 3–4
- **Below 50% / kappa < 0.40**: investigate framework fit before proceeding

The validated Cohort 1–2 labels become the initial training signal for Phase 2 fine-tuning.

---

## Phase 2 — Supervised Fine-Tuning with Autoresearch

**Target: concurrent with Cohorts 3–4**

Once Cohort 1–2 validation is complete, the pipeline produces its first systematically labeled corpus of VAAMR and PURER expression in a mindfulness-based pain intervention. Phase 2 uses this corpus to fine-tune domain-adapted classifiers — replacing the zero-shot LLM with a supervised model that is faster, cheaper, and validated on real Move-MORE language.

### 2.1 Export the Fine-Tuning Corpus

Filter `master_segments.jsonl` to `label_confidence_tier in ["high", "medium"]` and rows with a confirmed `human_label`. Structure as labeled examples for sequence classification:

```
{"text": "<segment_text>", "label": <va_mr_stage_int>, "source": "cohort1_validated"}
{"text": "<therapist_segment_text>", "label": <purer_step_int>, "source": "cohort2_purer"}
```

Keep the validation split fixed (the same segments in `human_coding_evaluation_set.csv`) so every autoresearch experiment is scored against the same held-out benchmark.

---

### 2.2 Autoresearch: VAAMR Classifier Search

**What autoresearch does:** an LLM-driven search controller that repeatedly edits `train.py`, runs a fixed-duration training job, scores the result with a fixed metric, and uses `git reset` to retain only improvements. The human contributes the research agenda through `program.md`; the agent discovers the implementation through overnight search.

**Adapt for this task:**

- **Metric**: swap `val_bpb` (bits per byte, for language modeling) for `val_macro_f1` on the held-out VAAMR 5-class classification task
- **Fixed run length**: 5 minutes is appropriate — BERT fine-tuning for 5 classes on ~500–1000 examples converges in well under 5 minutes on a single GPU
- **`prepare.py`** (read-only): dataset loading, tokenization, and the held-out validation split
- **`train.py`** (agent edits): model choice, frozen layers, classification head depth, learning rate, batch size, label smoothing

**`program.md` search agenda:**

```
Goal: maximize val_macro_f1 on 5-class VAAMR stage classification
     from therapy session transcript segments (~50–300 words each)

Hypotheses to explore (in this order):
1. BERT-base vs. ClinicalBERT vs. BioBERT as the encoder backbone
2. Layer freezing depth: freeze all but last 2 layers (data-efficient baseline),
   then progressively unfreeze layers if val_macro_f1 plateaus
3. Classification head: single linear → [768, 256, 5] bottleneck → [768, 128, 5]
4. Learning rate: 2e-5 (BERT default) → 1e-5 → 3e-5
5. Label smoothing: 0.0 vs. 0.1 (boundary cases between adjacent VAAMR stages
   are genuinely ambiguous; smoothing may help)

Constraints:
- Do not modify prepare.py or the validation split
- Do not use the test set during search; val_macro_f1 on the fixed validation
  split is the only allowed metric
- Log every experiment result with the hypothesis being tested

Known challenge: Avoidance (Stage 1) and Vigilance (Stage 0) share pain-focused
language; Avoidance and Attention Regulation (Stage 2) share attentional vocabulary;
Attention Regulation and Metacognition (Stage 3) share present-moment awareness language.
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

Each Cohort 3–4 session that clears human validation extends the labeled corpus. By the end of Cohort 4, the accumulated dataset spans four cohorts and two intervention variants, constituting the largest systematically labeled corpus of VAAMR stage expression in an MBI.

---

## Phase 3 — CFiCS-Style GNN for Therapist-Participant Dynamics

**Target: post-Cohort 4 analysis and manuscript**

### Rationale

CFiCS (Schmidt et al., 2024) demonstrated that combining domain-adapted BERT embeddings with GraphSAGE message-passing over a therapeutic taxonomy graph substantially outperforms either approach alone for psychotherapy content classification: ClinicalBERT alone achieved ~60% micro F1 on fine-grained skill classification; ClinicalBERT + GraphSAGE achieved ~96%. The key result is that graph structure encoding theoretical relationships between therapeutic concepts — which skills instantiate which higher-level factors — enables the model to learn distinctions that text embeddings alone cannot resolve.

The PURER → VAAMR relationship in Move-MORE is structurally identical to this problem: which therapist behaviors (PURER steps) precipitate which participant outcomes (VAAMR stage transitions)? This is a hierarchical, relational, multi-level classification task. A GNN over a VAAMR × PURER taxonomy graph, with node features from a fine-tuned MindfulBERT encoder, is the natural architecture.

The critical enabling condition — which CFiCS lacked — is a labeled corpus of real clinical interactions. The autoresearch fine-tuned classifiers from Phase 2 produce exactly this. CFiCS substituted synthetic examples; this project does not need to.

---

### 3.1 Build the VAAMR × PURER Taxonomy Graph

Construct a heterogeneous graph parallel to the CFiCS therapeutic taxonomy:

**Node types:**
- VAAMR stage nodes (5): Vigilance, Avoidance, Attention Regulation, Metacognition, Reappraisal
- PURER step nodes (5): Phenomenology, Utilization, Reframing, Education/Expectancy, Reinforcement
- Transition nodes (3): Forward (stage increase), Lateral (same stage), Backward (stage decrease)
- Example utterance nodes: participant segments (VAAMR labeled) and therapist segments (PURER labeled) from the full 4-cohort validated corpus

**Edge types** (parallel to CFiCS `fosters`, `expresses`, `demonstrates`):
- `precipitates`: therapist PURER step → participant VAAMR transition (weighted by empirical lift from Phase 1 cross-validation)
- `demonstrates`: example utterance node → VAAMR stage or PURER step node
- `co-occurs_with`: VAAMR stage × VAAMR stage temporal adjacency within sessions
- `supports`: PURER step → VAAMR stage sustained across session (not just at transition moments)

Populate edge weights from the `purer_to_vammr_influence_table.json` produced in Phase 1. The graph encodes what theory predicts (edge types) and what the data shows (edge weights) simultaneously.

---

### 3.2 Node Features from Fine-Tuned MindfulBERT

Use the autoresearch-optimized BERT checkpoint from Phase 2 as the text encoder. For each node, encode its text by average-pooling the last hidden state:

- Stage/step nodes: concatenation of name, definition, and prototypical features from the framework definition
- Example nodes: the raw segment text

This is the exact CFiCS approach, but the encoder has been fine-tuned on Move-MORE session transcripts with VAAMR and PURER labels. The MindfulBERT embeddings encode the Avoidance/Attention Regulation and Attention Regulation/Metacognition boundaries and the Phenomenology/Education boundary in a way that ClinicalBERT — trained on clinical notes with no exposure to VAAMR or PURER — cannot.

CFiCS reported ~10–15% performance gain from ClinicalBERT vs. generic BERT. Expect a similar or larger gain from MindfulBERT vs. ClinicalBERT on this domain.

---

### 3.3 GraphSAGE Training

Multi-task objective over three prediction heads:

| Task | Classes | Loss |
|------|---------|------|
| VAAMR stage | 5 | Cross-entropy |
| PURER step | 5 | Cross-entropy |
| Transition type | 3 (forward/lateral/backward) | Cross-entropy |

Combined loss: `ℒ = λ₁ℒ_vammr + λ₂ℒ_purer + λ₃ℒ_transition`

GraphSAGE's inductive capability means the model trained on Cohorts 1–3 can classify Cohort 4 sessions without graph restructuring — only the new segments need to be embedded with the text encoder and passed through the learned aggregation weights. This is the property that makes the model deployable for future trials.

Autoresearch can be applied here as well: `train.py` controls GNN depth, aggregation function (mean vs. pool vs. LSTM), hidden dimension, and the task weight coefficients `λ₁, λ₂, λ₃`. The fixed metric is macro F1 on the held-out Cohort 4 validation set.

---

### 3.4 Interpretation for MBI Development and Publication

The t-SNE visualization of learned embeddings (analogous to CFiCS Figure 3) will reveal whether PURER steps cluster near the VAAMR transitions they theoretically precipitate. Specifically:

- Do **Reframing** and **Phenomenology** embeddings cluster near the Avoidance → Attention Regulation transition zone? This is the primary prediction from Wexler, Balsamo et al. (2026)
- Does **Reinforcement** cluster near Reappraisal → Reappraisal (lateral) transitions? This would confirm that Reinforcement consolidates Reappraisal gains rather than producing new transitions
- Do **Education/Expectancy** embeddings cluster near Stage 0 (Vigilance) transitions? This would show that psychoeducation is most active in early sessions, consistent with the MORE session structure

Confirming these predictions provides the first graph-structured, empirically validated account of the PURER facilitation mechanism — extending the qualitative finding of Wexler et al. (2026) into a quantitative, reproducible model.

**Target venue:** *Psychotherapy Research* or *Journal of Consulting and Clinical Psychology* for the GNN results; the VAAMR + PURER classification system as a standalone contribution to *npj Digital Medicine* or *JMIR Mental Health*.

---

## Summary

| Phase | Item | Deliverable | Effort | Status |
|-------|------|-------------|--------|--------|
| 1.1 | PURER codebook + therapist classification | `purer_to_vammr_influence_table.json` | 3–5 days | ✓ Completed |
| 1.1b | Expected codes pre-spec, avoidance-barrier report, permutation control | `expected_codes` fields populated; `avoidance_barrier_analysis.json`; permutation-control results | 1–2 days | **HIGH PRIORITY** (before Cohort 3) |
| 1.2 | Windowed context classification | Wizard option, validated on test set | 1–2 days | Partial (code exists) |
| 1.3 | Outcome integration layer | `session_outcomes_integrated.csv` | 1–2 days | Not started |
| 1.4 | Human validation, Cohorts 1–2 | Kappa ≥ 0.60, agreement ≥ 75% | 1–2 weeks | In progress |
| 2.1 | Fine-tuning corpus export | Labeled JSONL for BERT training | 1 day | Ready (blocked on 1.4 completion) |
| 2.2 | Autoresearch VAAMR fine-tuning | Best VAAMR classifier checkpoint | 1 day setup + overnight | Blocked on 2.1 |
| 2.3 | Autoresearch PURER fine-tuning | Best PURER classifier checkpoint | 1 day setup + overnight | Blocked on 2.1 (1.1 now complete) |
| 2.4 | Cohorts 3–4 prospective validation | Extended labeled corpus | Concurrent with trial | Blocked on 2.2–2.3 |
| 3.1 | VAAMR × PURER taxonomy graph | Graph with empirical edge weights | 2–3 days | Blocked on 2.4 (1.1 now complete) |
| 3.2–3.3 | CFiCS-style GNN training | Trained GraphSAGE model | 1–2 weeks | Blocked on 3.1, 2.2 |
| 3.4 | Interpretation + manuscript | Published findings | Ongoing | Blocked on 3.3 |

**Critical path:** PURER codebook authorship (1.1) → validated Cohort 1–2 labels (1.4) → fine-tuning corpus (2.1) → autoresearch fine-tuning (2.2–2.3) → Cohorts 3–4 prospective data (2.4) → GNN training (3.1–3.3).

The PURER codebook is the binding constraint. Everything downstream depends on having operationally distinct, exemplar-grounded definitions for the five PURER steps before Cohort 3 begins.

---

## Key Files

| File | Purpose |
|------|---------|
| `theme_framework/vaamr.py` | VAAMR framework definition (participant-side) |
| `theme_framework/purer.py` | PURER framework definition (therapist-side) — **to build** |
| `codebook/phenomenology_codebook.py` | VCE 59-code phenomenology codebook |
| `process/cross_validation.py` | VAAMR × VCE lift statistics; extend for PURER × VAAMR influence table |
| `process/outcome_integration.py` | Quantitative outcome join — **to build** |
| `methodology.md` | Full methodology paper (Phenomenology and the Cognitive Sciences) |

## References

- Schmidt, F., Hammerfald, K., Jahren, H. H., & Vlassov, V. (2024). CFiCS: Graph-based classification of common factors and microcounseling skills. *KTH Royal Institute of Technology*.
- Wexler, R. S., Balsamo, W., et al. (2026). "Noticing the Way that I'm Noticing Pain." *Mindfulness*, 17, 819–833.
- Wexler, R. S., Balsamo, W., et al. (2026). Development and pilot feasibility testing of Move-MORE. Preprint. doi:10.21203/rs.3.rs-8682836/v1
- Lindahl, J. R., Fisher, N. E., Cooper, D. J., Rosen, R. K., & Britton, W. B. (2017). The Varieties of Contemplative Experience. *PLOS ONE*, 12(5), e0176239.
- Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text Psychometrics. *Psychological Methods*.
