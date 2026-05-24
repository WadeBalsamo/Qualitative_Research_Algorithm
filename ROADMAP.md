# QRA Research Roadmap

This document charts the forward trajectory for the Qualitative Research Algorithm (QRA) from its current unvalidated state through three major phases: (1) completion and submission of the computational phenomenology methodology paper, (2) supervised fine-tuning of domain-adapted classifiers via AutoResearch, and (3) construction of a heterogeneous Graph Neural Network (GNN) modeling therapist–participant dynamics across the full Move-MORE corpus. Each phase builds on the preceding one; the labeled corpus, validated constructs, and architectural patterns established in earlier phases become the inputs to later ones.

---

## Current State (Point of Departure)

 The pipeline can ingest diarized transcripts, semantically segment them, classify participant segments against VAAMR and therapist segments against PURER via zero-shot LLM consensus voting, optionally apply VCE codebook classification, assemble master datasets, generate cue-response reports, and prepare human validation sets. Cohorts 1 and 2 of the Move-MORE Feasibility Trial have been processed.

What remains is the research-facing work: validation, analysis, reporting, and the downstream ML pipeline.

---

## Phase 1 — Methodology Paper: "Phenomenology at Trial Speed: A Computational Pipeline for Iterative Refinement of Mindfulness-Movement Therapy in Chronic Pain"

**Target manuscript:** Journal of Contemplative Studies (or equivalent)

**Core argument:** QRA operationalizes two complementary phenomenological frameworks (VAAMR and VCE) into an automated transcript analysis pipeline capable of producing defensible cohort-level qualitative analysis within days rather than weeks. The pipeline's cross-validation between frameworks generates lift statistics that serve simultaneously as a methodological validity check and as a novel empirical contribution to understanding the phenomenological texture of MBI-induced therapeutic transformation.

### 1.1 Quantify Content Validity Using the Text Psychometrics Framework

**Reference:** Low et al. (2024), Text Psychometrics — content validity sensitivity metric

For each VAAMR stage, compile an expert-curated list of canonical tokens/phrases from the framework definitions in `theme_framework/vaamr.py`. Compute **content validity sensitivity** = proportion of expert list items that the LLM flags above a decision threshold across the corpus. Report per-stage sensitivity. Similarly compute for PURER components using therapist utterances.

**Deliverables for the paper:**
- Table: Per-stage content validity sensitivity (VAAMR: Vigilance, Avoidance, Metacognition, Reappraisal; plus PURER: P, U, R, E, R)
- Comparison of sensitivity across confidence tiers (High/Medium/Low)
- Identification of stages where sensitivity falls below 0.75, indicating definitional underspecification

### 1.2 Compute Inter-Rater Reliability Against Human Coders

Cohorts 1 and 2 validation set blind-coded by two independent rater teams following the protocol in `methodology.md`. Report raw agreement and Krippendorff's alpha per stage.

**Acceptance thresholds for paper:**
- **Agreement ≥ 75%, kappa ≥ 0.60:** Classifications support primary analyses
- **Agreement 50–74%, kappa 0.40–0.59:** Directional evidence only; explicit caveats required
- **Below threshold:** Investigate VAAMR framework fit for movement-integrated MBI context

**Deliverables:**
- Confusion matrix: human consensus label vs. LLM plurality label
- Per-stage agreement with confidence intervals
- Error analysis: characterize false positives/negatives by subtype (e.g., boundary confusion, movement-language mismatch, therapist-influence artifact)

### 1.3 Longitudinal Progression Analysis (Group-Level)

Compute per-participant, per-session **progression scores** (weighted average stage ID by segment proportion) across all eight sessions. Fit linear mixed-effects models with random intercepts for participant and cohort.

**Report types the paper can generate:**
- **Group trajectory plot:** Mean progression score ± SEM by session number, with Loess smoothing
- **Feasibility ribbon:** Proportion of segments at High/Medium/Low confidence per session — a high feasibility rating (≥60% at High/Medium) qualifies the trajectory for primary interpretation
- **Validity assessment:** Whether the group trajectory is non-decreasing across sessions (basic sanity check: VAAMR predicts mean stage advancement)
- **Cohort comparison:** Cohorts 1 vs. 2 trajectories overlaid with change-point detection
- **Progression score distribution histogram** aggregated across all sessions, showing the population spread

### 1.4 Between-Session Transition Analysis

For each participant, compute the **dominant stage** (modal VAAMR label) per session. Build the 4×4 cross-session transition matrix counting how participants move between dominant stages from week to week.

**Deliverables:**
- Cross-session transition heatmap (counts + row-normalized proportions)
- Per-participant dominant-stage trajectories as a parallel-coordinates or Sankey diagram
- Identification of "stuck" participants: those whose dominant stage does not advance across ≥4 consecutive sessions — these become clinical case-study targets
- Quantification of the **Avoidance persistence rate**: proportion of sessions where Avoidance is the dominant stage, disaggregated by session number

### 1.5 Within-Session Transition and Cue-Response Analysis

For each (from_stage → to_stage) transition within a session, extract the therapist dialogue that occurred between the two participant segments (the CUE). Aggregate by transition type.

**Report types the paper can generate:**
- **Within-session transition heatmap:** 4×4 matrix of (from, to) counts with forward/backward/lateral classification
- **PURER cue composition:** For each transition type, the proportion of therapist CUE text matching each PURER component (once PURER labels are computed on therapist segments) — this is the **PURER × VAAMR lift matrix**
- **Average CUE text per transition type:** LLM-synthesized aggregate of all therapist language preceding each transition type, preserving key therapeutic phrases (from `cue_response.txt`)
- **Exemplar (FROM, CUE, TO) triples:** One representative triple per (cohort, session) for the most frequent transition types
- **Forward transition yield:** Proportion of all transitions that are forward (Y > X), backward (Y < X), and lateral (Y = X) — a forward yield >0.40 indicates productive therapeutic engagement

### 1.6 The Avoidance Barrier: Dedicated Analysis

Targeted examination of the Avoidance stage as a developmental waypoint — the critical barrier where emerging attentional skill is misapplied toward suppression rather than investigation.

**Deliverables:**
- **Avoidance prevalence curve:** Proportion of segments classified as Avoidance by session number — does it peak mid-program and decline?
- **Avoidance → Metacognition transition rate:** The key clinical transition; report its frequency per session and the therapist cues most strongly associated with it
- **Duration analysis:** For participants who spend multiple consecutive sessions in Avoidance, characterize the linguistic register of their segments — do they show signs of metacognitive language mixed with avoidance language (the "stuck" boundary case)?
- **Movement-context effect:** Compare Avoidance prevalence in movement-integrated sessions vs. standard meditation-only sessions within Move-MORE

### 1.7 VCE × VAAMR Cross-Validation: Exploratory Lift Discovery

Compute co-occurrence lift statistics for all (VAAMR stage, VCE code) pairs to discover emergent empirical associations between construct frameworks. This is an **exploratory, data-driven analysis** — no directional hypotheses are encoded in the codebase. The VCE codebook (59 codes, 7 domains) and the VAAMR framework are independent constructs whose co-occurrence structure is unknown a priori.

**Theoretical framing in text** (from `methodology.md`): Varela's mutual-constraints logic predicts that participant therapeutic-stage development (VAAMR) and the specific experiential qualities they describe (VCE) should show systematic empirical associations. For example, participants articulating pain-related somatic language may be more likely to be in an early vigilance/avoidance stage, while those describing metacognitive clarity may be in a later metacognitive/reappraisal stage. These are text-level expectations in the methodology document — **not hardcoded predictions in the code**.

**Deliverables:**
- Lift table: rows = VCE codes, columns = VAAMR stages, cell = lift(VAAMR stage, VCE code)
- Top-N empirical associations: the (VAAMR stage, VCE code) pairs with highest lift, annotated with representative segment quotes
- Unexpected discovery reporting: VCE codes with lift > 2.0 in an unexpected VAAMR stage — these become targets for qualitative re-analysis
- Permutation control analysis: compare empirical lift to null distribution from shuffled labels, reporting empirical p-values

### 1.8 Outcome Integration: Mixed-Methods Joint Display

Author `process/outcome_integration.py` to join trial outcomes CSV (pain NRS, ODI, TSK-11, FFMQ, MAIA-2, actigraphy, session attendance, home practice minutes) with the master segment dataset on `session_id + participant_id`.

**Deliverables for the paper:**
- **Joint display matrix:** Rows = session numbers, columns = [mean progression score, % High confidence, pain NRS, home practice minutes, attendance] with color-coded directional arrows
- **Correlation table:** Pearson ρ between progression score change and outcome change scores (pre-post), with bootstrapped 95% CIs
- **Trajectory clustering:** k-means on (progression slope, pain change, attendance rate) to identify participant archetypes — "fast progressors," "high-pain avoiders," "low-engagement lateral," etc.
- **Scatterplot:** Within-participant mean progression score vs. pain intensity change, with regression line and participant-level labels

### 1.9 Therapist Fidelity: PURER Component Distribution

For each therapist segment classified against PURER, compute the distribution of PURER components across sessions and therapists.

**Deliverables:**
- **PURER profile per therapist:** Proportion of P/U/R/E/R in their speech — reveals individual stylistic differences in implementation
- **Session-level PURER variation:** Does PURER composition shift as the program progresses? (e.g., more Reframing in later sessions, more Phenomenology in early sessions)
- **High-yield PURER patterns:** For sessions with above-median forward transition counts, what PURER profile does the therapist show? Compare to low-yield sessions.

### 1.10 PURER × VAAMR Influence Lift Matrix (The Central Mechanistic Finding)

Once PURER labels exist for all therapist segments, compute for each (PURER component, VAAMR transition type) pair: given that a therapist segment classified as PURER component P appears in the CUE position, what is the lift of the subsequent participant segment being at VAAMR stage Y?

This is the paper's primary mechanistic evidence: **which therapist behaviors precipitate which participant outcomes at scale**.

**Deliverables:**
- **5×4 lift matrix:** PURER components (rows) × VAAMR stages (columns), cell = lift(VAAMR stage | PURER component in preceding CUE)
- **PURER efficacy ranking:** For each VAAMR transition (e.g., Avoidance → Metacognition), which PURER component shows the highest positive lift?
- **Conditional lift by session phase:** Lift(PURER → VAAMR) in early sessions (1-4) vs. late sessions (5-8) — does the same therapist move produce different outcomes at different points in the program?
- **Negative lift flagging:** PURER components that systematically precede backward transitions (Reappraisal → Avoidance) — potential iatrogenic patterns

### 1.11 Content Validity Test Set Performance

Run the classification pipeline against the exported content validity test sets (exemplar, subtle, adversarial tiers per VAAMR stage). Report per-stage accuracy on each tier.

**Deliverables:**
- **Confusion matrix on clear-tier items:** Tests whether the LLM has internalized core framework definitions
- **Adversarial-tier boundary accuracy:** Reveals which stage boundaries (e.g., Vigilance⇄Avoidance, Avoidance⇄Metacognition) are most confusable — these are precisely the boundaries needing human review
- **PURER test set performance:** Same analysis for therapist segments against PURER framework definitions

---

## Phase 2 — AutoResearch: Supervised Fine-Tuning of Domain-Adapted Classifiers

**Target:** Concurrent with Cohorts 3–4 data collection; trained model ready for prospective validation on Cohort 4.

**Prerequisite:** Phase 1 human validation complete with agreement ≥ 75% on Cohorts 1–2.

### 2.1 Fine-Tuning Corpus Assembly

Filter `master_segments.jsonl` to rows where `label_confidence_tier in ["high", "medium"]` and a confirmed `human_label` exists (from adjudicated human validation). Structure as labeled examples for sequence classification.

**Corpus specifications:**
- **VAAMR classifier:** Text → 1 of 4 stages (+ unclassified)
- **PURER classifier:** Therapist text → 1 of 5 components (+ unclassified)
- **Minimum viable size:** ~1,000–5,000 labeled examples per class (autoresearch.txt recommendation for BERT-style classifiers)
- **Split strategy:** Hold out entire sessions for validation — never split utterances from the same session across train/test to prevent session-level leakage

### 2.2 Model Selection and Architecture

Depending on observed transcript length distribution:
- **Segments ≤ 512 tokens:** BERT-base-uncased or RoBERTa-base with sequence classification head
- **Segments > 512 tokens:** Longformer-base-4096 or Longformer-base-2048 for extended-context encoding

**Parameter-efficient fine-tuning (for 3090 Ti):**
- LoRA adapters (rank = 8, alpha = 16, dropout = 0.1)
- Mixed-precision (fp16) training
- Gradient accumulation steps = 2–4 to manage effective batch size
- Weight decay = 0.01, warmup ratio = 0.06

### 2.3 AutoResearch Hyperparameter Search

AutoResearch orchestrates a search over:
- **Model backbone:** BERT-base vs. RoBERTa-base vs. ClinicalBERT vs. Longformer
- **Max sequence length:** 256, 512, 1024 (based on transcript length percentile)
- **Learning rate:** log-uniform [1e-5, 5e-5]
- **Batch size:** {8, 16, 32} with gradient accumulation
- **LoRA rank:** {4, 8, 16}
- **Label smoothing:** {0.0, 0.1, 0.2}

**Optimization target:** Validation macro-F1 on held-out sessions

### 2.4 Prospective Validation on Cohorts 3–4

Run both fine-tuned classifiers alongside the existing zero-shot LLM pipeline on all incoming Cohort 3 and 4 transcripts. Track per-session agreement rate between the two systems.

**Validation metrics:**
- **Convergent agreement:** Macro-F1 between fine-tuned and LLM labels
- **Confidence calibration:** Does the fine-tuned classifier produce better-calibrated probabilities than the LLM?
- **Speed benchmark:** Inference time per segment (fine-tuned: milliseconds; LLM: seconds with API latency)
- **Edge case detection:** Sessions where fine-tuned and LLM disagree — these are flagged for human review as potential novel linguistic patterns in later cohorts

### 2.5 Ablation: Impact of Training Set Size

Train the top-performing model configuration on progressively larger subsets of the labeled corpus (25%, 50%, 75%, 100%). Plot validation macro-F1 against training set size to estimate the learning curve and predict whether Cohorts 3–4 will meaningfully improve performance.

---

## Phase 3 — Graph Neural Network for Therapist–Participant Dynamics

**Target:** Post-Cohort 4 — full corpus of 4 cohorts × 8 sessions available for graph construction and training.

**Reference:** CFiCS (Schmidt et al., 2025, CLPsych 2025) — graph-based classification integrating ClinicalBERT embeddings with GraphSAGE message-passing over a heterogeneous psychotherapy taxonomy.

### 3.1 Rationale: Why a GNN for This Problem?

The central empirical question of the Move-MORE analysis is **causal-structural**: which therapist behaviors (PURER components) precipitate which participant outcomes (VAAMR stage transitions)? This is fundamentally a graph problem because:

1. **The relationships are relational, not independent.** A PURER Phenomenology move does not exist in isolation — its effect depends on the current VAAMR stage of the participant, the session number, the preceding therapist–participant exchange, and the participant's trajectory across prior sessions. A GNN naturally encodes these dependencies as edges.

2. **The taxonomy is hierarchical and multi-relational.** PURER components are not flat categories. They have theoretical relationships to each other (e.g., Utilization builds on Phenomenology; Reinforcement consolidates Reappraisal). A heterogeneous graph with typed edges (`precedes`, `precipitates`, `co-occurs_with`, `demonstrates`, `supports`) captures these relationships explicitly.

3. **The graph enables inductive generalization.** Once trained on Cohorts 1–3, the GNN can generalize to unseen Cohort 4 segments without retraining — predicting the likely participant response to a therapist move before the response occurs. This is the path toward real-time therapeutic feedback.

### 3.2 Taxonomy Graph Construction

Build a heterogeneous graph with the following node types:

| Node Type | Description | Count (approx.) |
|-----------|-------------|-----------------|
| **VAAMR Stage** | 4 participant developmental stages | 4 |
| **PURER Component** | 5 therapist move types | 5 |
| **Transition Type** | Forward, Backward, Lateral | 3 |
| **Session** | Session number (1–8) with curriculum topic metadata | 8 |
| **Participant** | Per-participant node with demographic/outcome covariates | ~32 (full trial) |
| **Therapist** | Per-therapist node (multiple therapists across cohorts) | 2–4 |
| **Segment Node** | Individual classified utterance — the atomic unit; initially, all segments from Cohorts 1–4 (~10,000–20,000) | 10k–20k |

**Edge types:**

| Edge | From | To | Meaning |
|------|------|----|---------|
| `classified_as` | Segment Node | VAAMR Stage | Participant segment expresses this stage |
| `classified_as` | Segment Node | PURER Component | Therapist segment implements this component |
| `precedes` | Segment Node | Segment Node | Temporal adjacency in session timeline (chronological ordering) |
| `precipitates` | Therapist Segment | Participant Segment | Therapist move immediately precedes participant response (the CUE→TO link) |
| `co-occurs_with` | VAAMR Stage | PURER Component | Empirical lift > 1.5 in the corpus (from Phase 1 cross-validation) |
| `belongs_to` | Segment Node | Session Node | Session membership |
| `belongs_to` | Segment Node | Participant Node | Speaker identity |
| `transitions_to` | Participant Segment | Participant Segment | Stage change between consecutive participant segments |
| `semantically_similar` | Segment Node | Segment Node | Embedding cosine similarity > 0.85 between segments from different sessions/participants — **the subtext bridge** |

### 3.3 Subtext Graph: Semantic Bridges Across the Corpus

Beyond the taxonomy edges, build a **subtext similarity graph** that captures latent thematic connections across the entire dialogue corpus:

1. **Encode all segments** using ClinicalBERT (or the fine-tuned encoder from Phase 2) to produce dense embedding vectors (768-d).
2. **Compute pairwise cosine similarity** across all segment embeddings — this is O(n²) but tractable for 10k–20k segments.
3. **Threshold edges** at similarity ≥ 0.85 to create the subtext graph: edges connect segments that express semantically similar content even if they come from different sessions, participants, and cohorts.
4. **Apply community detection** (Leiden or Louvain algorithm) to identify latent thematic communities in the subtext graph — these communities represent **recurrent phenomenological motifs** that may not correspond to any single VAAMR stage or VCE code but emerge organically from the dialogue.

**What the subtext graph reveals that other analyses cannot:**

- **Latent thematic communities:** Clusters of semantically similar utterances that cut across session boundaries, revealing thematic preoccupations that span the entire program (e.g., a "fear of movement" motif that appears in sessions 2, 5, and 7 but maps to different VAAMR stages each time)
- **Subtext bridges between stages:** Segments classified as Avoidance that are semantically closer to Metacognition segments than to other Avoidance segments — these are "boundary" utterances where the participant is on the cusp of transition, identifiable only through embedding proximity
- **Therapist–participant echo patterns:** Participant segments that are semantically similar to the therapist segment that preceded them, quantified as the mean embedding cosine similarity between CUE and TO — high echo indicates strong therapist influence; low echo indicates participant autonomy or resistance
- **Cross-cohort novelty detection:** Segments in Cohort 3 or 4 that have low maximum similarity to any Cohort 1–2 segment — these are linguistically novel expressions that may reflect curriculum changes between cohorts
- **Motif persistence curves:** For each detected community, plot its prevalence across session numbers — a motif that peaks early and decays suggests an initial concern that resolves with treatment; a motif that persists or grows suggests an unresolved experiential domain

### 3.4 Node Features

Each node receives a feature vector constructed from:

- **Segment nodes:** The ClinicalBERT or Phase-2 fine-tuned encoder embedding (768-d), plus scalar features: word count, confidence tier (one-hot), session number (normalized), timestamp within session
- **VAAMR Stage nodes:** The average embedding of all segments classified at that stage (centroid), plus the framework definition embedding
- **PURER Component nodes:** The average embedding of all therapist segments classified as that component, plus the framework definition embedding
- **Participant nodes:** Baseline outcome vector (pain NRS, ODI, TSK-11, FFMQ, MAIA-2), age, sex, pain duration — encoded as a dense vector
- **Session nodes:** Session number (one-hot), curriculum topic (one-hot), cohort ID

### 3.5 GraphSAGE Training

**Architecture:**
- 2–3 layer GraphSAGE with mean or LSTM aggregator (following CFiCS findings that GraphSAGE produces the best-clustered embeddings)
- Hidden dimension: 256
- Dropout: 0.3
- Activation: ReLU

**Multi-task objective** (following CFiCS multi-class pattern):

| Task | Type | Classes | Loss Weight |
|------|------|---------|-------------|
| VAAMR stage | Multi-class | 4 stages + neutral | λ₁ = 1.0 |
| PURER component | Multi-class | 5 components + neutral | λ₂ = 1.0 |
| Transition type | Multi-class | Forward / Backward / Lateral | λ₃ = 0.5 |
| Next-stage prediction | Multi-class | 4 stages | λ₄ = 0.5 |
| Subtext community | Contrastive | N communities (variable) | λ₅ = 0.3 |

**Training details:**
- Optimizer: Adam (lr = 1e-3, weight decay = 1e-4)
- Early stopping: 50 epochs without validation loss improvement
- Split: Train on Cohorts 1–2 segments, validate on Cohort 3 segments, test on Cohort 4 segments (session-level split)
- Hardware: Single 3090 Ti (24 GB VRAM) sufficient for graphs of 10k–20k nodes with GraphSAGE

### 3.6 Graph Interpretation and Visualization

**t-SNE or UMAP projection** of learned node embeddings, colored by:
- VAAMR stage (do PURER nodes cluster near the VAAMR transitions they theoretically precipitate?)
- PURER component (do Reframing and Phenomenology embeddings cluster near the Avoidance → Metacognition transition zone?)
- Subtext community (do communities map onto coherent phenomenological themes?)
- Cohort (to detect distribution shift across cohorts)

**Deliverable figures for the manuscript:**
- **2D embedding projection** with node types color-coded, showing the geometric arrangement of the therapeutic ontology
- **Transition-zone heatmap:** For each PURER component, compute the average embedding proximity to each VAAMR transition type — a proximity matrix that can be visualized as a heatmap
- **Subtext community word clouds:** Top TF-IDF terms for each detected community, labeled with a human-interpretable theme name

### 3.7 Causal Structure Discovery via the Graph

The GNN's attention weights on `precipitates` edges provide an empirical estimate of **which therapist moves most strongly influence which participant transitions** — a data-driven causal structure learned from the full corpus.

**Analysis types:**
- **Edge weight distribution for `precipitates` edges:** Across all (PURER_P, VAAMR_Y) pairs, do certain pairs receive consistently higher learned edge weights? (e.g., Phenomenology → Metacognition weight > Reframing → Metacognition weight)
- **Counterfactual prediction:** Hold all node features constant except the PURER component of a therapist segment node; predict the resulting VAAMR stage distribution — this is a within-graph causal estimate
- **Attribution by participant subgroup:** Do the `precipitates` edge weights differ for participants with high vs. low baseline kinesiophobia? (i.e., does the same therapist move produce different effects depending on participant characteristics?)
- **Session-level modulation:** Do `precipitates` edge weights change as a function of session number? (i.e., does Phenomenology become more or less effective at precipitating Metacognition in later sessions?)

### 3.8 GNN-Based Report Types (Novel Manuscript Outputs)

The trained GNN enables a new class of reports unavailable from the current pipeline:

| Report | Description | Manuscript Section |
|--------|-------------|-------------------|
| **PURER efficacy topology** | 2D embedding projection showing which PURER components cluster near which VAAMR transitions | Results: GNN Embedding Space |
| **Subtext community catalog** | Catalogue of N detected communities with: dominant VAAMR stage, top TF-IDF terms, prevalence across sessions, and representative quotes | Results: Subtext Motifs |
| **Causal influence matrix** | Learned `precipitates` edge weights aggregated as a 5×4 matrix (PURER × VAAMR), with bootstrapped confidence intervals | Results: Causal Structure |
| **Participant archetype trajectories** | GNN-based clustering of participant node embeddings into archetypes, with each archetype's characteristic progression trajectory and outcome profile | Results: Participant Archetypes |
| **Counterfactual simulation** | "What if this therapist had used Reframing instead of Education at this moment?" — GNN prediction of alternative VAAMR outcome distribution | Discussion: Therapeutic Implications |
| **Therapist fingerprint** | Per-therapist distribution of learned `precipitates` edge weights — each therapist's characteristic pattern of influence on participant stage transitions | Results: Therapist Fidelity |
| **Cross-cohort drift** | Distribution shift in subtext community prevalence between Cohorts 1–2 and Cohorts 3–4, attributable to curriculum modifications | Discussion: Iterative Refinement |
| **Embedded subtext network** | Full similarity graph with community overlay — a navigable map of latent phenomenological motifs across the entire Move-MORE dialogue corpus | Figures |

### 3.9 Comparison: GNN vs. Zero-Shot LLM vs. Fine-Tuned Classifier

Ablation experiment comparing three classification approaches on the same held-out test set:

| Method | VAAMR Macro-F1 | PURER Macro-F1 | Transition Prediction Accuracy | Inference Time per Segment |
|--------|----------------|----------------|-------------------------------|---------------------------|
| Zero-shot LLM (current) | Baseline | Baseline | — | ~2–5 s (API) |
| Fine-tuned BERT/Longformer (Phase 2) | — | — | — | ~10–50 ms (local GPU) |
| GraphSAGE (Phase 3) | — | — | — | ~5–20 ms (local GPU) |

The GNN is expected to outperform the fine-tuned classifier on transition prediction (because it has access to relational structure) while performing comparably or slightly below on isolated segment classification (because it trades some per-node expressivity for relational generalization).

---

## Phase 4 — Future Research Directions

### 4.1 Real-Time Therapeutic Feedback System

Deploy the Phase 3 GNN as a real-time inference engine: ingest a partial session transcript as it is produced, classify therapist moves as they occur, and predict the likely participant response distribution. Surface a dashboard to the therapist showing:
- Current participant VAAMR stage estimate
- PURER move just executed
- Predicted transition likelihood given alternative next moves
- Historical comparison: "When you used Reframing at this point in Session 5 with Participant X, the forward transition rate was Y%"

### 4.2 Cross-Trial Generalization

Apply the trained GNN (fine-tuned on Move-MORE) to transcripts from other MBI trials (e.g., standard MORE for chronic pain, MBSR for chronic pain, MBCT for depression). Measure the domain shift in embedding space and the degradation in classification performance. If the graph structure generalizes, this provides evidence for a universal phenomenological taxonomy of mindfulness-based therapeutic processes.

### 4.3 Multimodal Integration

Incorporate non-textual features into the graph:
- **Paralinguistic:** Prosodic features (pitch, rate, pause duration) extracted from audio — attached to segment nodes as additional feature dimensions
- **Kinematic:** Movement data from wearable sensors during sessions — attached to participant nodes as time-series features
- **Physiological:** Heart rate variability, skin conductance during sessions (if collected) — attached to segment or session nodes

This extends the analysis from what is said to how it is said and what the body is doing while saying it.

### 4.4 Prospective Clinical Prediction

Use the graph to predict clinical outcomes at trial endpoint from early-session data alone:
- **Task:** Given Session 1–3 segment embeddings and graph structure, predict 8-week change in pain NRS, ODI, or TSK-11
- **Method:** Add a regression head to the GNN that reads out from participant nodes after message-passing
- **Clinical utility:** Identify participants at risk of non-response by Session 3 for targeted intervention augmentation

---

## Summary Timeline

| Phase | Item | Deliverable | Timeline |
|-------|------|-------------|----------|
| **1** | Content validity (Text Psychometrics) | Sensitivity table per VAAMR stage | Q2 2026 |
| **1** | Inter-rater reliability | Kappa, agreement, confusion matrix | Q2 2026 |
| **1** | Longitudinal progression | Group trajectory, feasibility ribbon | Q2 2026 |
| **1** | Between-session transitions | Cross-session heatmap, participant trajectories | Q2 2026 |
| **1** | Within-session cues | Cue-response report, PURER × VAAMR lift matrix | Q2 2026 |
| **1** | Avoidance barrier report | Avoidance prevalence, transition rate, movement effect | Q2 2026 |
| **1** | VCE × VAAMR cross-validation | Lift table, permutation controls | Q2 2026 |
| **1** | Outcome integration | Joint displays, correlation table, archetype clustering | Q2–Q3 2026 |
| **1** | Therapist PURER fidelity | PURER profile per therapist, session-level variation | Q2–Q3 2026 |
| **1** | **Manuscript submission** | Computational phenomenology paper | Q3 2026 |
| **2** | Fine-tuning corpus assembly | Filtered, session-split labeled corpus | Q3 2026 |
| **2** | AutoResearch hyperparameter search | Optimal model config for 3090 Ti | Q3–Q4 2026 |
| **2** | Supervised fine-tuning | Domain-adapted VAAMR + PURER classifiers | Q4 2026 |
| **2** | Prospective validation on Cohorts 3–4 | Agreement rates, calibration, speed benchmarks | Q4 2026–Q1 2027 |
| **3** | Subtext similarity graph | Pairwise embedding matrix, community detection | Q1 2027 |
| **3** | Taxonomy graph construction | Heterogeneous graph with all node/edge types | Q1 2027 |
| **3** | GraphSAGE training | Trained model with multi-task objective | Q1–Q2 2027 |
| **3** | GNN interpretation | Embedding projections, causal structure, archetypes | Q2 2027 |
| **3** | **GNN manuscript submission** | Therapist–participant dynamics paper | Q3 2027 |
| **4** | Real-time feedback prototype | Dashboard design + offline simulation | 2027–2028 |
| **4** | Cross-trial generalization | Application to external MBI corpora | 2028 |
| **4** | Multimodal integration | Audio + kinematic + physiological fusion | 2028+ |

---

## References

- Low, D. M., Mair, P., Nock, M. K., & Ghosh, S. S. (2024). Text psychometrics: Assessing psychological constructs in text using natural language processing. *Psychological Methods*.
- Schmidt, F., Hammerfald, K., Jahren, H. H., & Vlassov, V. (2025). CFiCS: Graph-based classification of common factors and microcounseling skills. *Proceedings of CLPsych 2025*, 106–115.
- Wexler, R. S., Balsamo, W., et al. (2026). "Noticing the way that I'm noticing pain": A qualitative analysis of therapeutic progression in MORE for LRP. *Mindfulness*, 17, 819–833.
- Lindahl, J. R., et al. (2017). The varieties of contemplative experience. *PLOS ONE*, 12(5), e0176239.
- Garland, E. L. (2024). *Mindfulness-oriented recovery enhancement*. Guilford Press.
- Wexler, R. S., Balsamo, W., et al. (2026). Computational phenomenology in mindfulness-based interventions for chronic pain: A machine-assisted methodology for rapid iterative curriculum refinement. (In preparation.)
- Wexler, R. S., et al. (2025). Development and pilot feasibility testing of Move-MORE. *Protocol summary*.
