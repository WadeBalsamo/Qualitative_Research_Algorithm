# GNN Discovery + Mechanism Rebuild — Results & Promotion Guide

**Date:** 2026-06-07 · **Worktree:** `qra-ws-gnn` (branch `gnn-exp/ws1-h6`), **no commits** — review
the diff and promote. **Corpus:** `data/Meta` (Move-MORE Cohorts 1–2, n≈32; Qwen3-8B cache). Every
result is **hypothesis-generating, never causal** (n≈32 observational + elicitation confound §9.4).

## What changed (architecture)

The GNN layer is now two separate concerns:

1. **Discovery + construct-validation + mechanism** (`src/gnn_layer/` top level) — **DEFAULT ON**,
   runs at `qra analyze` on raw embeddings, no trained model. The new build.
2. **GraphSAGE consensus-distillation classifier** (`src/gnn_layer/classifier/`) — **DEFAULT OFF**
   (`gnn_classifier_enabled=False`). Pilot-refuted as a scaler (H5); kept as the documented
   gate/distillation instrument. Opt in with `qra gnn train`. The LLM consensus / Qwen probe remain
   the labels of record.

Wired as default through `setup_wizard` (step 11d), `analysis/runner.run_analysis`, `qra gnn`, and
`GnnLayerConfig`. The mis-specified mechanism-on-classifier path (`influence.py`) was deleted.

## Headline results

**WS1 — H6 discriminant validity** (`06_gnn/discriminant_validity.txt`): VAAMR is recoverable by
supervision yet **not** by content similarity — the positive reading of the H5 refutation.

| arm (same Qwen embeddings, grouped CV) | human κ [95% CI] | LLM κ |
|---|---|---|
| probe (supervised) | **0.365 [0.228, 0.513]** | 0.283 |
| content-similarity (Correct-&-Smooth) | 0.196 [0.117, 0.319] | 0.069 |
| chance (modal / stratified) | 0.000 / −0.083 | 0.000 / −0.017 |

Paired **probe − content Δκ**: human **+0.170 [0.002, 0.318]**, LLM **+0.214 [0.150, 0.274]**.
Geometry (honest): stage signal is partly in the top content PCs (so not an exotic subspace); the
operative property is **weak/uneven local kNN homophily** (1-NN same-stage 0.47 vs base 0.25;
Metacognition below base rate) → similarity neighbours are stage-mixed. Community×stage **ARI ≈ 0.006**.

**WS-T — dyadic FROM→CUE→TO transition model** (`06_gnn/transition_model.txt`): the mechanism rebuild.
- **Earns-its-place: NO** — the cue does not improve held-out TO prediction over a FROM-only baseline
  (Δ KL +0.37) → the transition is under-identified at n≈32 (consistent with H2). `mechanism.py` leads.
- **Triangulation: POSITIVE** — the learned counterfactual ranks with observed Δprogression at
  Spearman **ρ ≈ +0.34**, versus the retired classifier-counterfactual's **−0.13**. The properly
  specified model (no kNN, FROM-stage-conditioned) aligns where the old one inverted. Under-powered
  (0 FDR cells) — a Cohorts 3–4 question.

**WS3 — confound localization** (`06_gnn/confound_localization.txt`): signed divergence (observed −
learned counterfactual) per (from_stage × move), 20 cells, **9 sign-inverting**. Maps where
responsiveness most distorts the observed table (e.g. Reappraisal×Reinforcement: obs −0.60 vs cf
+0.59). A caveat instrument for `mechanism.py`, not a claim.

**WS2 — Track D deepened** (`06_gnn/communities.txt`, `dyadic_routines.txt`): default similarity
threshold recalibrated **0.85→0.6** for Qwen (τ=0.85 gave noise, ARI 0.003; τ=0.6 gives ARI 0.29).
223 communities, 21 stable; 2 stable dyadic routines (Δprog CIs include 0 = honest under-powered
leads). Added community↔stage/Δprog profiles, atypical exemplars, and therapist→participant routines.

## New / changed files

- New: `src/gnn_layer/{discriminant,transition,confound,cue_features}.py`; deepened `communities.py`;
  `classifier/` subpackage (moved: model, train, graph_builder, validation, triangulation, inference,
  calibration, propagation, ablation, anchors, gnn_lift); figures + config + runner wiring.
- Deleted: `src/gnn_layer/influence.py`, `tests/unit/test_gnn_influence.py`.
- New tests: `test_gnn_{discriminant,transition,dyadic,confound}.py`. Reorg rewrote classifier
  import-sites to `gnn_layer.classifier.*`.
- Docs synced: `CLAUDE.md`, `docs/methodology.md` §8.5, `design_decisions.md` §10,
  `docs/gnn_experiments/ledger.csv` (H6 arms).

## Track C — revisited (done)

`process/assembly/mindfulbert_dataset.py`'s optional augmentation channel now sources the
transition model's `transition_per_move.csv` (replacing the retired `influence.py` CSV) and is
**gated on the transition instrument having run, not on the classifier gate** (`gate_passed` now
governs only the `gnn_consensus` provenance tier). It is still OFF by default
(`augmentation_enabled=False`) and **retained only if the C4 held-out ablation clears
`augmentation_min_gain`** — and because the transition cue is under-identified at n≈32, it honestly
tends to be stripped. Tests updated (`test_mindfulbert_dataset.py`).

## How to promote

1. Review `git -C $(pwd) diff 2040766` in this worktree.
2. Reproduce: `python -m pytest tests/unit -q`; `qra analyze -o ./data/Meta` (discovery default-on);
   `qra gnn train -o ./data/Meta` (opt-in classifier).
3. Merge `gnn-exp/ws1-h6` (or cherry-pick) into the working branch; carry `design_decisions.md` §10
   and the `06_gnn/*` artifacts.

## Cohorts 3–4 re-run triggers

H6 is N-robust (CIs tighten). WS-T earns-its-place, WS3 divergence, and H2 mechanism become
adjudicable at higher power (today 0 FDR cells). The classifier gate re-opens for any learned scaler.
All re-run unchanged via `qra analyze` / `qra gnn train`.
