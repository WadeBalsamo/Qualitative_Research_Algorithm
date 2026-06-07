# Experimental Workflow — how QRA refines its classification methodology

> The systematic process behind this catalog: how a candidate architecture is posed, run, **quantitatively
> assessed**, and either **promoted into the pipeline** or **archived as a documented negative**. It exists
> so the refinement loop is *reproducible and scalable* — a new campaign drops in without re-litigating the
> ground rules, and no method is ever re-tried blind.
>
> Companion docs: [`CATALOG.md`](CATALOG.md) (every experiment + the promotion ledger) · each campaign's
> `RESULTS.md` (full per-arm detail).

---

## 0. Why a fixed workflow

QRA's labels (VAAMR developmental stages, PURER therapist moves) are **measurement instruments**, not
just ML targets. Changing how a segment is classified changes what the study *finds*. So every proposed
change to the classification architecture is treated as a **measurement experiment** with a pre-registered
protocol, a leak-free evaluation, and an explicit promotion decision — not an ad-hoc model swap. Two
properties make this scale:

1. **One frozen apparatus, many arms.** Every arm is scored by the same harness on the same folds and
   axes, so results across campaigns and across months are directly comparable.
2. **Negatives are first-class.** A rejected architecture is a *result* — it bounds the search and stops
   the team re-trying it. The catalog documents *what didn't work and why* as carefully as what did.

---

## 1. The apparatus (the fixed measuring stick)

Code: `gnn_reliability/harness.py` + `gnn_reliability/baselines.py` (mirrored from the live
`experiments/gnn_reliability/`; kept byte-identical by `tests/unit/test_experiments_catalog_sync.py`).

- **Corpus.** `data/Meta` — Move-MORE Cohorts 1–2, n ≈ 32 participants; 205 LLM-labeled + 134 "No code"
  participant segments; 66 human-consensus codes.
- **Features.** Cached **Qwen3-Embedding-8B** (4096-d), L2-normalized, text-hash-keyed cache (a warm cache
  never re-hits the endpoint). MiniLM-384 is kept only as a *within-embedder control*.
- **Folds — the methodological backbone.** `build_folds(df, seed=42)` = participant-grouped
  `StratifiedGroupKFold`. **Whole participants are held out**, so no segment's temporal-chain or kNN
  neighbour leaks its label into training. This is non-negotiable: random k-fold inflated a GNN gate from
  κ 0.05 → 0.247 purely through leakage (Campaign 1 §5). The fold map is built **once** and shared
  byte-for-byte by every arm.
- **Dual-axis scorer — `score_arm`.** Two Cohen-κ axes, each with a **participant-clustered bootstrap 95%
  CI** (resample whole participants, 2000×):
  - **LLM axis** — preds vs the multi-run LLM consensus (`final_label`) over 205, + per-class recall.
  - **Human axis (load-bearing)** — preds vs human consensus over 66, scored *exactly* like
    `analysis/irr_analysis.py` so κ is comparable to the shipped `06b_irr_report.txt`.
- **Ledger.** Each arm appends a row to `docs/gnn_experiments/ledger.csv` (arm, embedding, method, both-axis
  κ + CI, per-class recall, seed, decision).

---

## 2. The quantifiable assessment rubric

Every arm is judged on the **same five quantities**, in this order:

| # | Criterion | How it's measured | Pass condition |
|---|---|---|---|
| 1 | **Leak-free** | participant-grouped folds, seed 42 | mandatory — no random-fold numbers admitted |
| 2 | **Clears the bar** | LLM-axis κ over 205 **or** human-axis κ over 66 | **LLM κ ≥ 0.45** OR **human κ ≥ 0.50**, CI-aware |
| 3 | **Beats the incumbent** | **paired** cluster-bootstrap Δκ vs A1n on identical items (n_boot = 3000) | Δ CI **excludes 0** (point gain alone is not enough) |
| 4 | **Recovers the rare stages** | held-out recall on Avoidance + Metacognition | non-trivial recall (the binding bottleneck) |
| 5 | **Parsimony / cost** | model complexity + inference cost (LLM-free?) | prefer the simplest model at equal κ |

**Reference bands** (so a κ is interpretable, not a bare number): trained human coders agree at
Krippendorff α ≈ 0.33–0.52; the LLM consensus is human-level at κ = 0.537 vs human. The bar (0.45 / 0.50)
is set at the **human↔human ceiling**, not the unreachable legacy κ ≥ 0.70.

**Honesty discipline (anti-overfitting on a 66-item test set):**
- Arms are **pre-registered** as a small discrete set *before* any Qwen result — honest model selection,
  not gradient-tuning on the test set.
- Hyperparameters are tuned on the **LLM axis only**; the **human axis is read once per arm** and never
  tuned on.
- **CIs decide, not point estimates** — at n ≈ 32 a 0.02 point gain inside overlapping CIs is noise.
- Folds, seed, and scorer are frozen, so any arm is re-runnable and any verdict re-adjudicable as n grows.

---

## 3. The lifecycle of an experiment

```
  pose hypothesis ─► pre-register arm(s) ─► run in an isolated worktree ─► score (dual-axis + CI)
        ▲                                                                          │
        │                                                                          ▼
        └──────────────  archive as documented negative  ◄──── adjudicate ────► promote
```

1. **Pose** a falsifiable hypothesis about the *architecture* ("context as embedding features transfers the
   LLM's edge"; "rater disagreement is signal"). State the expected pass condition up front.
2. **Pre-register** the concrete arm(s) — model, features, imbalance handling, class count — as an entry
   before running, so model selection is over a fixed discrete set.
3. **Run in an isolated worktree** driven by a parallel subagent (no commits to the main tree; see
   `feedback_parallel_worktree_experiments`). Reuse the harness verbatim — never re-derive the scorer.
4. **Score** on both axes with participant-clustered CIs; append the ledger row.
5. **Adjudicate** against the rubric (§2): clears bar? beats incumbent (paired Δ)? rare-stage recall? cost?
6. **Decide** — promote, archive-as-negative, or escalate to a follow-up arm. Record the decision in
   `CATALOG.md` (the promotion ledger) regardless of outcome.

---

## 4. The decision gate — promote, assistive, or archive

A trialed architecture resolves to exactly one of:

- **PROMOTE (method of record).** It is leak-free, clears the bar (or is a measurement fix like the
  grouped-CV correction / the No-code null that is correct independent of κ), and is parsimonious. → wired
  into `src/` and recorded in the promotion ledger. *Examples: participant-grouped gate; class-weighting +
  No-code null; the dyadic transition mechanism; the H6 discriminant instrument.*
- **PROMOTE AS ASSISTIVE (gated, not autonomous).** It is the best model found and reliably beats the
  incumbent (paired Δ excludes 0), but does **not** clear the LLM-equivalence bar. → may ship **below** the
  LLM consensus as a gated, abstention-aware **pre-labeler** with mandatory human review on
  abstentions/rare-stages/low-margin items; it **never** becomes the label of record. Re-evaluate as n
  grows. *Example: the per-rater ensemble winner S6 `ens_softavg` (spec'd, not yet wired).*
- **ARCHIVE AS NEGATIVE.** It ties or hurts, or only wins by overfitting the test set. → documented in the
  catalog with the *mechanism* of failure, so it is not re-tried blind. *Examples: context-as-features;
  nonlinear capacity; ordinal decoding; anchors; rater-weighting; stacking.*

**Standing invariant:** the multi-run **LLM consensus remains the label of record** until a classifier
clears the bar on the current corpus. Promotion writes any distilled label to a provenance tier *below* the
LLM consensus, never over it.

---

## 5. Adding a new experiment or campaign (the scalable part)

To add an **arm** to an existing campaign:
1. Add a standalone script under the campaign dir; bootstrap `src/` onto `sys.path` (so
   `experiments.<campaign>` resolves to this archive) and `import experiments.gnn_reliability.harness`.
2. Build folds with `H.build_folds(df, seed=42)` and score with `H.score_arm(...)` — **do not** re-derive
   folds or κ.
3. Run it; add a row to the campaign `RESULTS.md` table and to `CATALOG.md`; append the ledger row.

To add a **campaign** (a new question, e.g. a different framework or a fine-tuned encoder):
1. Create `src/experiments/<campaign>/` with its own `RESULTS.md` (+ `CAMPAIGN_LOG.md` if run as a wave of
   subagents). Reuse the harness; if a new corpus is needed, extend `harness.load_corpus`, keeping
   participant-grouped folds.
2. Add the campaign to this `WORKFLOW.md` overview and to `CATALOG.md` (master table + promotion ledger).
3. Keep the apparatus mirror in sync: if you change the live `experiments/gnn_reliability/*.py`, re-copy it
   into `src/experiments/gnn_reliability/` (the drift-guard test enforces this).

**Reproducibility checklist for any committed result:** seed 42 · participant-grouped folds · dual-axis κ +
participant-clustered CI · the raw result artifact committed next to the script (e.g. `_distill_results.json`,
`_csweep_results.json`, `_softlabel_results.jsonl`) · the number quoted in prose traceable to that artifact.

---

## 6. What this workflow has produced so far

- **Campaign 1 (GNN reliability):** refuted H5 (graph-as-scaler) leak-free; promoted the grouped-CV
  correction, class-weighting, and the No-code null; turned the negative into the H6 discriminant
  instrument and the rebuilt transition mechanism.
- **Campaign 2 (scaler distillation):** ranked seven lever families; found the per-rater ensemble winner,
  established the n ≈ 32 **data ceiling** via three converging methods, and bounded the search (context,
  capacity, ordinal, stacking, anchors all archived as negatives).

The throughline: **the binding constraint is data scale, not model architecture.** The workflow is built so
that when the corpus grows, every arm re-runs unchanged and every verdict is re-adjudicated against the same
frozen yardstick.
