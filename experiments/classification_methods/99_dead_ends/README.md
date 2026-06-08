# Dead-End Classification Methods — R&D Trail

This document records the classification approaches that were tried, abandoned, or repositioned
during QRA development. It exists so those directions are not re-explored blindly and so the full
provenance of the current pipeline is traceable from git.

---

## 1. Lit-Trained NSP Classifier

**What it was.**
A custom Transformer trained from scratch on arXiv papers to score qualitative transcript segments
using Next-Sentence Prediction (NSP). The model built a two-class NSP head on top of a 6-layer,
8-head self-attention stack (d_model=512, max_seq=100) and was intended to assign VAMR construct
categories via sentence-pair scoring. Training data was pulled from an arXiv query matching the
drafted paper; the scoring logic then applied the NSP logit to therapy transcript sentence pairs.

**When it lived.**
Introduced 2024-01-31. Single-file implementation: `lit_trained_nsp_classification.py`.

| Event | Date | Commit |
|-------|------|--------|
| Created | 2024-01-31 | `32040ca` ("Create lit_trained_nsp_classification.py") |
| Deleted | 2025-06-22 | `c33741a` ("architecting graph of sentence embeddings for GNN qualitative coding methodology") |

**Why abandoned.**
The approach required custom fine-tuning on arXiv papers, which produced a label-agnostic sentence-
pair scorer rather than a VAMR-aware classifier. It needed training data, a compatible tokenizer,
and a domain-matched corpus; none of those were available in the form required, and the NSP
objective does not directly map to phenomenological stage assignment. When the project pivoted to
embedding-based graph methods (c33741a), this file was removed along with the script and test
scaffolding of that era.

**What replaced it.**
Zero-shot LLM classification via multi-run consensus voting — no training data required, directly
prompted with VAMR stage definitions. The successor path is documented in `../03_single_model_zeroshot/`
and is the current label-of-record pipeline (`src/classification_tools/classification_loop.py`).

---

## 2. MentalBERT Graph Classifier (mentalbert_sentence_aqua)

**What it was.**
A graph-based classification pipeline using MentalBERT (a mental-health domain BERT variant) as the
sentence embedding backbone, Louvain community detection on a semantic+temporal similarity graph, and
a "triple-veto" decision combining (1) MentalBERT construct probabilities, (2) cosine-similarity
graph structure, and (3) community-level majority aggregation. The pipeline had ten modules:
`embedding.py`, `graph_construction.py`, `clustering.py`, `classification.py`, `preprocessing.py`,
`evaluation.py`, `visualization.py`, `audit_trail.py`, `config.py`, `main.py`.

The graph was built by thresholding cosine similarity (default 0.75) and adding temporal-adjacency
decay edges; Louvain partitioned the graph into communities; the dominant construct label per
community was assigned to constituent sentences. Classification confidence combined the MentalBERT
softmax, the graph edge weight, and the community consensus — hence "triple-veto."

**When it lived.**
Introduced 2025-06-22. Root-level package `mentalbert_sentence_aqua/`.

| Event | Date | Commit |
|-------|------|--------|
| Created | 2025-06-22 | `c33741a` ("architecting graph of sentence embeddings for GNN qualitative coding methodology") |
| Deleted | 2026-03-14 | `7dfa61c` ("Replace mentalbert_sentence_aqua with vamr_labeling zero-shot data labeling pipeline") |

**Why abandoned.**
Two independent reasons, either of which was sufficient:

1. **H2 not validated.** The core premise — that graph structure over content embeddings would
   improve construct classification — was later formally tested across multiple arms (GraphSAGE A0,
   A3, A4, A4n; Correct-&-Smooth A2/A2n) in the GNN reliability battery. VAAMR is not homophilous
   in content-embedding space: graph smoothing and graph-structure-based methods consistently tied
   or hurt versus a simple linear probe. The triple-veto design presupposed homophily that does not
   exist for this label.

2. **Infeasible dependency stack.** MentalBERT requires a HuggingFace transformers version
   incompatible with the pinned environment (transformers 4.42.4 / numpy 1.26.4 / torch-geometric
   compatibility constraints). Loading MentalBERT weights silently fails or produces garbage under
   the current pin. Recreating the weights+deps environment is not feasible without breaking the
   existing pipeline.

**What replaced it.**
The zero-shot LLM pipeline (commit `7dfa61c`) replaced the entire mentalbert package, ultimately
landing in `src/classification_tools/`. The *graph* ideas were later refined and formally tested in
the GNN reliability campaign (`experiments/gnn_reliability/`), which is where the hypothesis-testing
against this approach is documented.

---

## 3. GNN GraphSAGE Consensus-Distillation Classifier (H5)

**What it was.**
A heterogeneous speaker-pair graph fed into GraphSAGE with learned per-edge-type gates ("precipitates"
edges from therapist to participant turns, weighted by edge type). The GNN was trained to reproduce the
multi-run LLM consensus labels (soft-label distillation) so that new segments could be classified
LLM-free at inference time. The architecture included: temperature scaling (A3), OOD confidence gating
(A3), semi-supervised label propagation (A4), and an inductive whole-session scale-mode simulation gate
(A5). It was fully wired into the pipeline as `gnn_classifier_enabled=True` and surfaced as `qra gnn
train` / `qra gnn classify`.

**When it lived.**
Built June 2026. Package: `src/gnn_layer/` (moved to `src/gnn_layer/classifier/` on separation).

| Event | Date | Commit |
|-------|------|--------|
| Full Track A implementation | 2026-06-05 | `628e71a` ("GNN: complete Track 0 + Track A (A1–A5) + GPU-preference hardening") |
| Repositioned default-OFF, classifier isolated | 2026-06-07 | `8619066` ("gnn: separate classifier (default OFF) + rebuild mechanism as a dyadic transition model + H6/discovery layer") |

**Why demoted.**
H5 was formally refuted on the current corpus (n ≈ 32 participants, 205 LLM-labeled + 66 human-coded
segments). Measured outcomes from the GNN reliability battery:

- GraphSAGE classifier↔human κ ≈ 0.36 (arm A4n), but this is carried entirely by No-code abstention
  — its stage-discriminative agreement (LLM-axis κ) is only 0.18–0.21.
- Linear probe (A1n) ties or beats the GNN on both axes with no graph machinery at all.
- Graph smoothing (Correct-&-Smooth, A2/A2n) was the *worst* arm — confirming the label is not
  recoverable from content similarity alone.
- A CV-leakage correction (random folds → participant-grouped folds) collapsed the gate κ from 0.247
  to 0.05 — the reported gate was invalid before the correction.

Conclusion: this is a **data ceiling, not a method gap**. Three independent methods (GNN, probe,
per-rater ensemble) converge at LLM κ ≈ 0.36 / human κ ≈ 0.45 — the binding bottleneck is two rare
VAAMR stages (Avoidance, Metacognition) and shallow separability of a content-trained embedding at this
corpus size, not the classifier architecture. The GraphSAGE classifier is default-OFF
(`gnn_classifier_enabled=False`) and remains in the codebase at `src/gnn_layer/classifier/` for future
re-evaluation as labeled participants accrue.

**NOT recreated here.** This method is fully studied in the sibling campaign
`experiments/gnn_reliability/` — see its `RESULTS.md` for per-arm numbers and `experiments/CATALOG.md`
for the full promotion ledger. All harness code is at `experiments/gnn_reliability/harness.py`.

---

## Note on the GNN Discovery Layer

The GNN *discovery* layer — `src/gnn_layer/discriminant.py` (H6 discriminant validity),
`src/gnn_layer/transition.py` (dyadic FROM→CUE→TO transitions), `src/gnn_layer/communities.py`
(subtext communities), `src/gnn_layer/confound.py` (confound sensitivity) — is **not a classifier**
and is out of scope for this campaign. It is hypothesis-generating mechanism and construct-validation
work that runs default-on at `qra analyze`. Its embedding-based geometry findings (e.g., VAAMR is not
topic-structured, H6: probe κ 0.365 vs content-similarity 0.196) are what give the failed classifier
result its positive interpretation as construct evidence. The discovery layer is documented in
`experiments/docs/gnn_discovery_results.md` and `experiments/CATALOG.md §Campaign 1`.

---

## Summary Table

| Method | Era | Key commits | Status | Where it went |
|--------|-----|-------------|--------|---------------|
| Lit-trained NSP classifier | Jan 2024 – Jun 2025 | `32040ca` (created), `c33741a` (deleted) | Abandoned — wrong objective; needed fine-tuning data | Replaced by zero-shot LLM pipeline (`src/classification_tools/`) |
| MentalBERT graph classifier (`mentalbert_sentence_aqua`) | Jun 2025 – Mar 2026 | `c33741a` (created), `7dfa61c` (deleted) | Abandoned — H2 not validated; infeasible deps under env pin | Replaced by zero-shot LLM pipeline; graph hypothesis formally tested in `experiments/gnn_reliability/` |
| GNN GraphSAGE consensus-distillation (H5) | Jun 2026 – present | `628e71a` (built), `8619066` (demoted default-OFF) | Demoted — H5 refuted at n ≈ 32; data ceiling | Remains in `src/gnn_layer/classifier/` default-OFF; fully studied in `experiments/gnn_reliability/` |
