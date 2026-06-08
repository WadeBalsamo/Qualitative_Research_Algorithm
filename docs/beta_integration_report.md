# Beta Integration Report — GNN Repositioning + Experiments Catalog

**Scope.** This report records the review, fixes, and integration that landed two pull requests on `beta`
together with a reorganized, comprehensively-documented experiments catalog. It is the narrative companion
to the per-campaign records in `experiments/` (start at `experiments/CATALOG.md` and
`experiments/WORKFLOW.md`) and the manuscript sections `docs/methodology.md` §8.5 (GNN reliability /
repositioning) and §8.6 (the distillation campaign).

**Corpus throughout:** Move-MORE Cohorts 1–2, `data/Meta`, **n ≈ 32 participants**; 205 LLM-labeled + 134
"No code" participant segments; 66 human-consensus codes. **Evaluation apparatus (frozen):**
participant-grouped `StratifiedGroupKFold` (seed 42), dual-axis Cohen-κ scorer (classifier↔LLM-consensus
over 205; classifier↔human over 66), participant-clustered bootstrap 95% CIs. **Features:** cached
Qwen3-Embedding-8B (4096-d), L2-normalized. **Success bar (LLM-equivalence):** classifier↔LLM κ ≥ 0.45 (the
human↔human band) **or** classifier↔human κ ≥ 0.50 (LLM↔human is 0.537), CI-aware. The legacy κ ≥ 0.70 gate
is unreachable in principle at this scale and is not the target.

---

## 1. What landed on `beta`

1. **PR #8 — GNN repositioning:** the GraphSAGE consensus-distillation **classifier** becomes a separate,
   **default-OFF** subpackage (`src/gnn_layer/classifier/`, `gnn_classifier_enabled=False`); the
   **discovery + mechanism work-streams** (H6 discriminant validity, the rebuilt dyadic transition model,
   confound localization, deepened subtext communities) become the **default** analyze-time build on raw
   embeddings. The multi-run **LLM consensus remains the label of record**; everything is
   hypothesis-generating, never causal (n ≈ 32 observational + the elicitation confound).
2. **PR #9 — experiments archive:** a self-contained, documented archive of **every** VAAMR-classification
   experiment under `experiments/` (the GNN reliability battery + the classification-scaler campaign,
   including the failed arms).
3. **Review fixes** for both PRs, and a **reorganized catalog** (`experiments/CATALOG.md` master table +
   promotion ledger; `WORKFLOW.md` refinement process + assessment rubric; README index; per-campaign
   `RESULTS.md`/`CAMPAIGN_LOG.md`).

Full unit suite green throughout: **`pytest tests/unit -q` → 3312 passed, 10 skipped, 0 failed**.

---

## 2. The two PRs — what they did and how they were reviewed

### 2.1 PR #8 — classifier default-OFF + mechanism rebuild + H6/discovery (`gnn-exp/ws1-h6`)

**What it did.** Moved 11 classifier modules (`model, train, validation, graph_builder, triangulation,
inference, calibration, propagation, ablation, anchors, gnn_lift`) into `src/gnn_layer/classifier/` (git
renames); deleted the old `influence.py` mechanism-on-classifier and its test; split
`runner.run_gnn_analysis` into `_run_classifier_layer` (gated on `gnn_classifier_enabled`) and
`_run_discovery_layer` (always, raw embeddings); added the discovery/mechanism modules `discriminant.py`,
`transition.py`, `confound.py` and deepened `communities.py`.

**Review verdict — sound, with doc-debt fixed in integration.**
- **Reorg integrity:** all classifier imports resolve via `.classifier`; `run_gnn_classify` and the anchors
  sub-path use `.classifier` (a lazy-import bug was already fixed). No stale top-level imports in `src/`,
  `tests/`, `qra.py`.
- **Off-by-default verified:** `runner.py:94` gates `_run_classifier_layer` on `gnn_classifier_enabled`
  (default `False`); `_run_discovery_layer` runs unconditionally; the only enabling path is `qra gnn train`
  (`force_classifier=True`). `master_dataset` gate-promotion semantics unchanged (empty diff). Config
  `from_json` is field-filtered, so removed `counterfactual_*` flags can't crash a legacy config.
- **Scientific honesty:** every new `06_gnn/*` report carries the n ≈ 32 / observational / no-causal
  caveats; participant-grouped CV + participant-clustered CIs throughout; geometry framed as **local kNN
  non-homophily**, not global subspace orthogonality; no causal verbs about discovery outputs.
- **Test reproducibility caveat we hit:** the `qra-ws-gnn` worktree was being flipped between `beta` and the
  PR branch by a concurrent process; a first full-suite run there produced 37 phantom import failures (it
  collected `beta`'s old-path test files mid-checkout). Re-running in a stable, pinned worktree gave a clean
  **3311 passed**.

### 2.2 PR #9 — experiments archive (`experiments-archive`)

**What it did.** Added `experiments/` — the consolidated experiments tree: the live reliability apparatus
(`gnn_reliability/`), the unique scaler campaign (`classification_scaler/`), the narrative docs
(`docs/`), and the result files. Purely additive.

**Review verdict — safe to merge; reproduce-scripts fixed in integration.**
- **Non-disruption (the blocking gate) passes entirely:** `git diff --name-only beta...experiments-archive`
  lists only `experiments/**`; the protected paths (`experiments`, `tests`, `src/process|analysis|
  gnn_layer`, `qra.py`) are byte-identical to beta; nothing in the pipeline imports `experiments/`.
- **Experiments run:** all 27 scripts compile; the 15 preserved GNN tests pass; the smoke-run reproduces the
  committed numbers (A1n 0.283/0.365, ens_softavg C=1 0.325/0.389) and the C=4 winner **0.361/0.450** — all
  reproduced live on `data/Meta`.
- **Numbers faithful:** every documented κ traces to a committed artifact (`ledger.csv`,
  `_distill_results.json`, `_softlabel_results.jsonl`, and the newly-committed `_csweep_results.json`).

---

## 3. Workstream results

### 3.1 Campaign 1 — GNN reliability battery: **H5 refuted at this scale**

*Question (H5): can a content-similarity GraphSAGE GNN reproduce the LLM VAAMR consensus well enough to
label new segments LLM-free?* Detail: `experiments/gnn_reliability/RESULTS.md`.

| Arm | What was tried | Human κ (n) | LLM κ (205) | Verdict |
|---|---|---|---|---|
| A0 | GraphSAGE on MiniLM-384 | −0.02 (66) | 0.05 | ≈ chance — the leak-corrected floor |
| A1w | Linear probe, Qwen, 5-class, class-weighted | 0.30 (37) | **0.31** | best LLM-axis; class-weight recovers rare stages |
| **A1n** ⭐ | Linear probe, Qwen, **6-class** (No-code), class-wt | **0.37 (66)** | 0.28 | **battery winner** → seeds Campaign 2 |
| A2 | Correct-&-Smooth (graph smoothing on the probe) | 0.16 | 0.16 | **worst** — propagation destroys signal |
| A3 / A4n | GraphSAGE, Qwen (5 / 6-class) | 0.14 / 0.36 | 0.21 / 0.18 | GNN < probe; A4n human κ is No-code abstention, not stage discrimination |
| B1 | GraphSAGE **+ CFiCS anchors** | 0.29 | 0.18 | anchors **lower** every axis |

**Findings.** Under leak-free participant-grouped CV the graph reproduces the consensus at grouped
**κ ≈ 0.05–0.14** — below the human↔human band. A **linear probe on the same Qwen features ties or beats the
graph**, and pure graph smoothing is the *worst* arm: the signature that **VAAMR is not homophilous in
content-embedding space** (cosine similarity tracks topic/affect; VAAMR is a developmental state, so
similarity edges wire different-stage segments together). The decisive methodological by-product was the
**CV-leakage correction**: an earlier κ ≈ 0.247 that looked near-passing was a random-fold leakage artifact
(temporal/kNN neighbours leak same-participant labels); participant-grouped folds drop it to ≈ 0.05. The only
levers that helped were **class-weighting** and an explicit **"No code" null** — measurement discipline, not
graph machinery. **Verdict:** the graph is **dropped as a classifier of record**; the LLM consensus stays the
label of record. The same negative is positive construct evidence (provisional **H6**): a label a content
graph cannot recover is not a topic taxonomy.

### 3.2 Campaign 2 — classification-scaler distillation: **a better probe, but a data ceiling**

*Question: can the A1n probe be distilled to LLM-equivalent fidelity?* Detail:
`experiments/classification_scaler/RESULTS.md` + `CAMPAIGN_LOG.md`.

| ID | Lever | LLM κ (205) | Human κ (66) | Verdict |
|---|---|---|---|---|
| A1n | baseline probe | 0.283 | 0.365 | the bar to beat |
| **S6** ⭐ | **per-rater ensemble** `ens_softavg` (C=4) | **0.361** [.28,.43] | **0.450** [.32,.60] | **WINNER**, both axes (paired Δ_LLM +0.078 [+.036,+.132] ✓) |
| S2 | soft-label MLP-KL (5-cls / 6-cls) | 0.367 / 0.310 | 0.228 / 0.339 | lifts only the optimized axis; no clear |
| S5 | two-stage No-code gate | 0.281 | 0.447 [.33,.60] | best human-axis single lever; grazes |
| S4 | model capacity (MLP/GBM/SVM-RBF/calib) | 0.07–0.23 | ≤ A1n | all below the linear baseline |
| S1 | context embeddings (concat) | 0.227 | 0.360 | **HURTS** — context-as-features doesn't transfer |
| S5b | ordinal-arc decoding | 0.01–0.22 | 0.01–0.22 | collapses |
| S7 | human anchor (LOPO calib) | n/r | ≈ A1n | overfits at n=66 |
| S8/S8b | stacking (per-rater × two-stage) | 0.247 / 0.189 | 0.398 / 0.292 | hurts — starves the per-rater gate |

**Findings.** The **per-rater ensemble** — one class-weighted probe per LLM rater (gemma-4-31b /
nemotron-3-nano-30b / qwen3-next-80b), mean-`predict_proba` ensembled — **dominates A1n on both axes**
(LLM 0.283→0.361, human 0.365→0.450; the LLM-axis paired Δκ excludes 0). But **no configuration clears the
bar**, and **three independent methods converge** on LLM ≈ 0.36 / human ≈ 0.45 — the signature of a **data
ceiling, not a method gap**. The binding bottleneck is the two rare stages (Avoidance, Metacognition; recall
≈ 0.28–0.35 across every arm) at n ≈ 32. Most diagnostically, **context-as-features failed to transfer**
(concatenating the 6-turn context *lowered* κ in both MiniLM and Qwen spaces) — the LLM's edge is *reasoning
over* context, not context in a feature vector. **Verdict:** ship the winner (if at all) as an **assistive,
gated, abstention-aware pre-labeler** below the LLM consensus — never autonomous; re-adjudicable as n grows.

### 3.3 Rebuilding the GNN analysis (discovery + mechanism)

The H5 refutation repositioned the GNN from a *classifier* to a *discovery/mechanism* instrument. Five
rebuilds, all default-on at `qra analyze`, all hypothesis-generating:

- **Mechanism, rebuilt (Track B).** The original mechanism read a *per-segment classifier's* counterfactual
  sensitivity (`gnn_layer/influence.py`); on the pilot it **inverted** the observed Δprogression ranking
  (Spearman **ρ = −0.13**) and was mis-specified for a process question (kNN content-noise; never trained on
  transitions; a single diluted cue→participant edge). It was **retired and rebuilt** as the dyadic
  **FROM→CUE→TO transition model** (`gnn_layer/transition.py`): `TO_mixture ≈ f(FROM_mixture, FROM_stage,
  pooled raw-Qwen cue)`, no kNN, FROM-stage conditioned. Its learned counterfactual now triangulates
  **positively** (Spearman **ρ ≈ +0.34**). Honest at n ≈ 32: the cue does *not* earn its place under
  participant-grouped CV (the transition is under-identified), so the observed `analysis/mechanism.py`
  **leads** and the counterfactual is exploratory. (The retired `run_mechanism.py` harness is kept as a
  documented RETIRED stub.)
- **Confound localization (`gnn_layer/confound.py`).** A signed-divergence map per (from_stage × move):
  **9 of 20 cells invert in sign** between the observed Δprogression and the learned counterfactual — the
  elicitation/responsiveness confound read at the cell level rather than from a single inverted coefficient.
- **H6 discriminant validity (`gnn_layer/discriminant.py`).** On identical folds and the *same* Qwen
  embeddings, a supervised probe (human **κ = 0.365** [.23,.51]) vs a content-similarity model (**0.196**
  [.12,.32]); the paired Δκ CI **excludes 0** on both axes (human +0.170 [.002,.318], LLM +0.214
  [.150,.274]). Geometry is framed as **weak/uneven local kNN non-homophily** (stage *is* partly in the
  content PCs — not an exotic subspace; community×stage ARI ≈ 0.006), bounding every similarity method.
- **Subtext communities + dyadic routines (`gnn_layer/communities.py`).** Two-algorithm partition (Louvain +
  hierarchical) with their ARI, participant-bootstrap **stability selection** (fragile communities flagged,
  not dropped), and therapist→participant dyadic routines. The Qwen cosine threshold was recalibrated
  **0.85 → 0.6** (probe: τ = 0.85 → ARI ≈ 0.003 noise; τ = 0.6 → ARI ≈ 0.29).
- **Provenance discipline.** Gate-gated promotion is unchanged: `gate_passed` governs only the
  `gnn_consensus` provenance tier; Track C (MindfulBERT dataset) augmentation reads `transition_per_move.csv`
  and is gated on the transition model + its C4 ablation, not the classifier gate.

---

## 4. Fixes applied during integration

**PR #9 reproducibility & numbers.**
- Fixed every broken reproduce-script import so the documented commands run (the archival relocation of
  `rater_distill`/`run_softlabel` into `classification_scaler/`, the `experiments.scaler` typo in
  `gate_sweep.py`, and a hard-coded foreign-worktree path in `run_human_anchor.py`); the archive scripts now
  bootstrap `src/` first so `experiments.*` resolves to the self-contained archive.
- Committed `_csweep_results.json` (the raw artifact backing the C=4 winner; `_csweep.py` regenerates it).
  Verified end-to-end: `_csweep.py` reproduces **C=4 → LLM 0.361 / human 0.450**.
- Corrected the soft-label point estimate `0.362 → 0.367` to match the committed `_softlabel_results.jsonl`.

**PR #8 code & manuscript.**
- Retired `run_mechanism.py` (it still imported the deleted `gnn_layer.influence`) → an honest catalog stub
  pointing to `transition.py`.
- **Synced the catalog `gnn_reliability` mirror to the live `classifier/`-rewired apparatus** (the stale copy
  broke test collection under the merge) and added `tests/unit/test_experiments_catalog_sync.py` to enforce
  the mirror byte-identically so it can never silently re-drift.
- Doc-synced README/USAGE/methodology/CLAUDE: removed the deleted `counterfactual_*` flag instructions,
  redirected Track B to the transition model, fixed the `GNN_MASTER_PLAN.md` §4.7 cross-refs, softened §8.5's
  "Status: Fully implemented", added methodology §8.6, local-homophily docstring in `discriminant.py`,
  mindfulbert datasheet gate/provenance strings, and `community_sim_threshold` fallbacks `0.85 → 0.6`.

---

## 5. The reorganized experiments catalog (`experiments/`)

- **`CATALOG.md`** — one page, every experiment across both campaigns (what was tried, dual-axis κ, worked?)
  + a **promotion ledger**: what graduated into the codebase (participant-grouped CV; class-weighting +
  No-code null; the transition mechanism; the H6 instrument; the classifier shipped default-OFF) and what was
  archived as a negative (context, capacity, ordinal, stacking, anchors, rater-weighting).
- **`WORKFLOW.md`** — the systematic architectural-refinement process: the frozen apparatus, the
  **quantifiable assessment rubric** (leak-free → clears-bar → beats-incumbent paired-Δ → rare-stage recall →
  parsimony), the **promote / assistive / archive** decision gate, and how to add an arm or a campaign.
- **`README.md`** — the index; per-campaign `RESULTS.md` (detailed per-arm) and `CAMPAIGN_LOG.md`
  (chronological).

---

## 6. Verification

- **`pytest tests/unit -q` → 3312 passed, 10 skipped, 0 failed**.
- **Reproduce verified live** on `data/Meta`: A1n 0.2831/0.3652; ens_softavg C=1 0.3254/0.3893; C=4 winner
  0.3608/0.4502 — matching the documented headlines exactly (paired Δ = +0.078).
- Classifier confirmed **default-OFF**; **LLM consensus remains the label of record**; no causal claims.

---

## 7. Bottom line

The throughline across both campaigns is that **the binding constraint is data scale (n ≈ 32), not the model
architecture.** The graph is the right *discovery/mechanism* instrument and the wrong *classifier* at this
scale; the per-rater ensemble is the best cheap classifier found but is assistive, not autonomous. The whole
apparatus is frozen at seed 42 and re-runnable, so every verdict here is re-adjudicable as labeled
participants accrue in Cohorts 3–4.
