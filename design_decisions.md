# GNN VAAMR Reliability — Design Decisions & Experiment Ledger

> **Living document.** Append-only decision log + a pre-registered experiment battery.
> Negative results are kept as evidence, not deleted. Workspace: `./data/Meta/`.
> Companion: `docs/GNN_MASTER_PLAN.md` (Track A0), `docs/gnn_experiments/ledger.csv`.

---

## 1. Mission & target

Make the VAAMR GNN reach **inter-rater reliability comparable to the LLM↔human level** on
`./data/Meta/`, in a methodologically defensible design, and keep the graph useful for
understanding therapist→participant VAAMR mechanism. **PURER is out of scope this phase.**

| Axis | Meaning | Now (held-out, honest) | Target |
|---|---|---|---|
| **GNN↔human κ** | independent quality (load-bearing) | **+0.053** | → **LLM↔human ≈ +0.537**; success floor = the human band (α 0.33–0.52) |
| GNN↔LLM κ | distillation fidelity | +0.159 (n=76) / +0.247 (n=205) | → human↔human ceiling ≈ 0.45–0.52 |
| rare-stage recall | Avoidance, Metacognition | **0% / 0%** | non-zero, no collapse |

The success criterion is **human-level IRR**, not the legacy κ≥0.70 gate (unreachable in
principle — see §3).

## 2. Locked decisions (with the researcher)

- **D-A — Run the full arm battery; let the data decide.** Report the reliability frontier with
  CIs; promote the arm with the highest **GNN↔human κ**. Escalate through Path B (concept
  anchors) as part of the battery.
- **D-B — Add a "No code" class** (6-way head) as the adopted design; **keep 5-class arms** as
  the defensible baseline so the construct change is accountable to evidence.
- **D-C — Honest split.** If a simpler model (calibrated linear probe / Correct-&-Smooth on Qwen
  embeddings) wins the classifier bake-off, *it* ships as VAAMR label-of-record and the **GNN is
  retained for therapist↔participant coupling/mechanism** (its graph-native strength). M1
  (classifier) and M2 (mechanism) are scored separately and never conflated.
- **D13 (inherited)** — one branch per architecture; every arm appends a ledger row; promote to
  default only on a **human-axis** gain.

## 3. The reliability ceiling reframes "reliable"

Trained human coders agree only at **Krippendorff α ≈ 0.33–0.52** (test-sets 1/2/3:
0.473 / 0.523 / 0.325). The LLM consensus is already **human-level** (κ=0.537 vs human). You
cannot distill a label function more reliable than its fuzzy teacher, so **κ≥0.70 is the wrong
target**. The GNN currently fails on *both* axes (0.247 vs LLM, **0.053 vs human**) — it is not
capturing the construct shape at all, and is worse than the LLM on the axis that matters.

## 4. Root-cause hypotheses (ranked, evidenced)

| # | Cause | Evidence (this corpus) | Arm that tests it |
|---|---|---|---|
| 1 | **Weak node features** (MiniLM-384 substitute, not Qwen3-8B) | both heads weak; kNN graph = embedding quality | A1 vs A0 |
| 2 | **No class rebalancing** (unweighted `batchmean` KL) | Avoidance/Metacognition recall 0%; AttentionReg 86% | A4 vs A3 |
| 3 | **Over-smoothing** at n=205 (2× mean-agg kNN) | majority-stage collapse | A1/A2 (no/again graph) vs A3 |
| 4 | **No "No code" class** | 134/339 participants are "No code" → trained to uniform noise + 5-class GNN can't match ~36% of human items the LLM *can* | A1n/A4n vs A1/A4 |
| 5 | **CV leakage** (random k-fold, not participant-grouped) | current gate folds ignore participant clustering | all arms use GroupKFold |
| 6 | **Fuzzy ceiling** | human α 0.33–0.52 | reframes target (§3) |

### Data scale (the binding constraint)
339 participant segments → **205 labeled** + **134 "No code"**. Labeled distribution:
AttentionReg 73 · Reappraisal 58 · Metacognition 29 · Vigilance 25 · **Avoidance 20**.
Human consensus: **76 items** in `qra.db` (~66 usable vs machine). 100% multi-run ballot
coverage (soft targets trustworthy).

### Infrastructure facts
- **Qwen3 embeddings**, zero pin-risk: `text-embedding-qwen3-embedding-8b` @
  `http://10.0.0.58:1234/v1/embeddings`, **4096-d** (vs MiniLM 384-d).
- **Integration gap (closed by A0-pre):** `master_segments.csv` has
  `in_human_coded_subset`/`human_label` columns but **0 populated**; the 76 human consensus codes
  live in `qra.db` (`irr_human_codes`). `validation.py:_human_axis` is built and waiting on the join.

## 5. Scoring protocol & defensibility guardrails

- **Folds:** participant-grouped (`GroupKFold` on `participant_id`) + stratified by `final_label`,
  **built once**, shared by every arm (identical folds/seed). No participant in train+test; the 66
  human items are naturally out-of-fold.
- **Two reference axes, every arm:**
  - **LLM axis** — out-of-fold preds vs `final_label` (n=205 + the 76-subset). Per-class κ/recall/precision.
  - **Human axis (load-bearing)** — out-of-fold preds vs human consensus (n≈66), scored **exactly
    like `irr_analysis`** (`No code`=−1 included; reuse `analysis/irr_stats.cohen_kappa` +
    `process/irr_import.read_human_codes`) so numbers are directly comparable to `06b_irr_report.txt`.
- **Uncertainty:** κ point estimate **+ participant-clustered bootstrap 95% CI**
  (`analysis/stats.cluster_bootstrap_ci`) on both axes. **CIs decide, not point estimates** (n is tiny).
- **No test-set tuning:** hyperparameters tuned on the **LLM-axis CV only**; the human axis is
  **read once per arm**. Arms are **pre-registered** (§6) → arm selection is honest model selection
  over a small discrete set, not gradient-tuning on a 66-item test set.
- **No-code scoring:** 6-class arms map predicted class-5 → ABSTAIN(−1) for the human axis; 5-class
  arms cannot emit No-code (quantifies the structural penalty).
- **Reproducibility:** pin embedding `model id + dim + a vector checksum` per arm.
- **Honesty:** report folds that miss a rare class entirely (expected at n=205); never hide a collapse.

## 6. Pre-registered arm battery (FIXED before results)

Identical folds/seed. Each arm → one `ledger.csv` row + a §7 entry.

| Arm | Features | Method | Imbalance | Classes | Question |
|---|---|---|---|---|---|
| **A0** | MiniLM-384 | GraphSAGE (current) | none | 5 | reproduce 0.247 / 0.053 baseline (harness self-check) |
| **A1** | Qwen-4096 | Linear probe (logreg) | none | 5 | is it the embedding? do we need a graph? |
| **A1w** | Qwen-4096 | Linear probe | class-weighted | 5 | rebalancing without a graph |
| **A1n** | Qwen-4096 | Linear probe | class-weighted | **6** | does "No code" fix the human axis? |
| **A2** | Qwen-4096 | Correct & Smooth | C&S | 5 (+6) | does graph propagation help a simple base? |
| **A3** | Qwen-4096 | GraphSAGE | none | 5 | does the GNN beat the simple baselines? |
| **A4** | Qwen-4096 | GraphSAGE | class-balanced + focal/TAM | 5 | does rebalancing recover rare stages? |
| **A4n** | Qwen-4096 | GraphSAGE | class-balanced | **6** | full GNN candidate |
| **B1** | Qwen-4096 | GraphSAGE + VAAMR concept anchors | best loss | 6 | does concept structure close fine-grained gaps (human axis)? |

**Promotion rule:** max **GNN↔human κ** (CI-aware) with no rare-class collapse. Ties → simpler
model (D-C). Mechanism (M2) graph kept regardless.

## 7. Results ledger (append-only)

> Filled as arms complete. Format mirrors `docs/gnn_experiments/ledger.csv`.
> κ shown as `point [lo, hi]` (participant-clustered bootstrap 95% CI).

| Arm | branch | GNN↔human κ (n≈66) | GNN↔LLM κ (n=205) | rare recall (Avoid/Metacog) | decision |
|---|---|---|---|---|---|
| _pending_ | | | | | |

## 8. Decision log (chronological)

- **2026-06-06 — Pre-registration.** Battery (§6), scoring (§5), targets (§1) fixed before any
  Qwen result. Foundation branch `gnn-exp/harness` cut from a `beta` checkpoint
  (`062c9bd`, prior IRR + turn-level-PURER WIP) so every experiment branch shares a clean base.
  Confirmed Qwen endpoint live (4096-d) and the human-subset integration gap (0 populated rows).
