# GNN VAAMR Reliability Battery — Archived Results

> **Purpose:** the leak-free test of whether QRA's GraphSAGE GNN can act as a consensus-distillation classifier (hypothesis **H5** — reproduce the multi-run LLM VAAMR consensus from graph structure well enough to label new segments LLM-free). **Verdict: H5 is NOT supported at this scale; the graph is dropped as a classifier of record and retained as a mechanism/discovery instrument.**

**Status:** archived experiment battery (uncommitted working-tree research; nothing here is a method of record). **Accuracy notice:** every κ in this file is a recorded value from the archived ledger (`docs/gnn_experiments/ledger.csv`) and the salvaged design records (`design_decisions.md` §7, `graph_experiments.md` §3.1). No κ is invented.

---

## 1. The claim under test and the corpus

The QRA GNN layer (`src/gnn_layer/`) was built as a *scaler*: learn the multi-run LLM VAAMR consensus over a graph of therapy segments (kNN-cosine-similarity edges + temporal-chain edges over Qwen3-Embedding-8B vectors, GraphSAGE message passing, soft-VAAMR supervision from the LLM ballots) so the graph could then label *new* transcript segments without paying for the LLM. That is hypothesis **H5** — the graph as a consensus-distillation classifier / labeling scaler.

This battery tests H5 on the real Cohort 1–2 corpus under the **only leak-free protocol**: participant-grouped cross-validation (whole participants held out).

**Corpus** (`data/Meta/`, Move-MORE, n ≈ 32 participants):
- 339 participant segments → **205 carry an integer VAAMR label**, **134 are "No code"** (~36% — no VAAMR stage expressed).
- Labeled-stage distribution: Attention-Regulation 73 · Reappraisal 58 · Metacognition 29 · Vigilance 25 · **Avoidance 20** (the two binding rare classes are Avoidance and Metacognition).
- **76 human-coded IRR items** (~66 usable vs. the model); human consensus mix is No-code 24, Vigilance 12, Avoidance 2, Attention-Regulation 10, Metacognition 5, Reappraisal 13.

**Reference bands** (what "reliable" can even mean here): trained human coders agree only at **Krippendorff α ≈ 0.33–0.52** (test-sets 1/2/3: 0.473 / 0.523 / 0.325); the multi-run LLM consensus is already human-level at **κ = 0.537 vs. human**. The legacy promotion gate of **κ ≥ 0.70 is unreachable in principle** at this scale and is *not* the target — the honest yardstick is the human↔human band.

---

## 2. The apparatus

All arms plug into one shared, pre-registered measurement harness so every number is comparable to the shipping IRR report (`06b_irr_report.txt`). κ math, the human-consensus read path, and the bootstrap are **reused from the production modules** (`analysis.irr_stats`, `process.irr_import`, `analysis.irr_analysis`, `analysis.stats`, `gnn_layer.*`) — nothing is re-derived in the harness.

### Participant-grouped CV (the methodological backbone)
`harness.build_folds` builds **one** fold map with `sklearn.model_selection.StratifiedGroupKFold` (`groups = participant_id`, `y = final_label`, 5 folds, **seed 42**), shared byte-for-byte by every arm. No participant appears in both train and test; the rare-stage mix is preserved fold-to-fold; the 66 human items are naturally out-of-fold. Because folds are participant-pure, *all* of a held-out participant's segments — labeled and "No code" — are held out together, so No-code rows inherit their participant's fold (pure-No-code participants are assigned by deterministic round-robin). Folds that miss a rare class entirely are logged, never hidden.

### Dual-axis scorer
`harness.score_arm` scores every arm's out-of-fold predictions on **two axes**, each with a **participant-clustered bootstrap 95% CI** (`analysis.stats.cluster_bootstrap_ci`, 2000 resamples, whole participants resampled so the (pred, ref) pairing is preserved — CIs decide, not point estimates, at this n):
- **LLM axis** — OOF preds vs. `final_label` over the 205 (+ per-class recall/precision, + a κ over the 76-item human-coded subset).
- **Human axis (load-bearing)** — OOF preds vs. the human consensus, scored *exactly* like `analysis.irr_analysis` ("No code" = −1 as a 6th category; 6-class arms map predicted class 5 → −1) so the κ is directly comparable to `06b_irr_report.txt`.

Discipline: hyperparameters tuned on the **LLM axis only**; the human axis is **read once per arm**; arms are **pre-registered** before any Qwen result (honest model selection over a small discrete set, not gradient-tuning on a 66-item test set). Each arm appends one row to `docs/gnn_experiments/ledger.csv`.

### Files in this directory

| File | Role |
|---|---|
| `harness.py` | The validated apparatus. `load_corpus` (master_segments.csv + the human-subset join), `build_folds` (participant-grouped `StratifiedGroupKFold`, shared by all arms), `get_embeddings` (MiniLM local cache / Qwen3-8B remote `/v1/embeddings`, text-hash-keyed cache), `run_gnn_arm` (the GNN measurement engine: build graph once → per grouped fold, mask that fold's participant VAAMR soft targets, train fresh, read held-out argmax — **NOT** `train.crossval_predictions`, which is the leaky random-k-fold path), `score_arm` (dual-axis κ + participant-clustered bootstrap CI + ledger row). |
| `baselines.py` | Non-GNN arms. `run_linear_probe` (A1/A1w/A1n — class-weighted logistic-regression probe on L2-normalized Qwen features, **no graph**) and `run_correct_smooth` (A2/A2n — Correct-&-Smooth, Huang et al. ICLR 2021: probe soft base → residual diffusion (Correct) → train-clamped label diffusion (Smooth) over the QRA kNN+temporal graph; the simplest "does graph propagation help a feature-only base?" test). |
| `anchors_arm.py` | The pre-registered Path-B escalation (B1) — CFiCS-style **construct-anchor** GNN. The anchored twin of `run_gnn_arm`: identical grouped-CV loop, single departure being that the graph is built **with the 5 VAAMR construct-definition anchor nodes** and their label-free anchor↔segment cosine edges. Measured on the GNN↔human axis (where anchors cannot inflate κ by construction). |
| `capacity_scaler.py` | The capacity sweep that *bridges into the successor campaign*: on the same Qwen target features / grouped folds, sweep MLP (torch class-weighted + sklearn unweighted), HistGradientBoosting, SVM-RBF, and calibrated linear probes against the A1n bar (LLM 0.283 / human 0.365). Capacity beyond a linear probe overfits — all families land ≈ 0.13–0.20 LLM-axis, below A1n. |
| `run_battery.py` | Orchestrates the full pre-registered battery (A1–A4n): builds corpus + folds + Qwen embeddings once, runs every arm on the same folds, scores both axes, appends the ledger. **The entry point to reproduce the classifier results.** |
| `run_mechanism.py` | **RETIRED stub** (superseded). It originally ran the model-counterfactual cue-influence readout on a per-segment GNN classifier (the former `gnn_layer/influence.py`); on the pilot that counterfactual *inverted* the observed ranking (Spearman ρ = −0.13, §10). It was **rebuilt** as the dyadic FROM→CUE→TO transition model `src/gnn_layer/transition.py` (ρ ≈ +0.34) — see §10. The file is kept as a catalog record and no longer imports the deleted module; run `qra analyze` for the current mechanism read. |
| `graph_experiments.md`, `design_decisions.md` | The salvaged narrative records (method, decision log, full battery + mechanism analysis). |

---

## 3. Results — the pre-registered battery

κ as `point [95% CI]`; human-axis n in parentheses. Recall is held-out per-class **Vig / Avo / Meta** (the three load-bearing rare stages). Reference: human↔human α **0.33–0.52**; LLM↔human **0.537**; promotion gate κ ≥ 0.70 (unreachable).

| Arm | Features | Method | Imbal. | Cls | **Human κ** (n) | **LLM κ** (205) | Vig/Avo/Meta recall | Verdict |
|---|---|---|---|---|---|---|---|---|
| A0 | MiniLM-384 | GraphSAGE | none | 5 | −0.02 [−.08, .06] (66) | 0.05 [−.03, .12] | 0.20/0.00/0.00 | honest baseline (≈ chance); the leak-corrected floor |
| A1 | Qwen-4096 | Linear probe | none | 5 | 0.21 [.06, .33] (37) | 0.23 [.14, .32] | 0/0/0 | features fix the floor; rare stages collapse |
| A1w | Qwen-4096 | Linear probe | bal | 5 | 0.30 [.11, .53] (37) | **0.31 [.21, .38]** | 0.56/0.35/0.31 | best LLM-axis; class-weight recovers rare stages |
| **A1n** ⭐ | Qwen-4096 | Linear probe | bal | **6** | **0.37 [.23, .51] (66)** | 0.28 [.20, .35] | 0.36/0.35/0.31 | **battery winner** — best full-task human κ; simplest model |
| A2 | Qwen-4096 | Correct&Smooth | bal | 5 | 0.16 [.04, .29] (37) | 0.16 [.08, .28] | 0.20/0.05/0.10 | pure graph smoothing = **worst**; propagation destroys signal |
| A2n | Qwen-4096 | Correct&Smooth | bal | 6 | 0.20 [.12, .32] (66) | 0.07 [.01, .14] | 0/0/0 | propagation collapses the LLM axis |
| A3 | Qwen-4096 | GraphSAGE | none | 5 | 0.14 [.05, .24] (66) | 0.21 [.12, .29] | 0.28/0.20/0.00 | plain GNN < probe on both axes |
| A4 | Qwen-4096 | GraphSAGE | bal+focal | 5 | 0.10 [.01, .20] (66) | 0.16 [.09, .23] | 0.32/0.50/0.10 | rebalancing doesn't rescue the GNN |
| A4n | Qwen-4096 | GraphSAGE | bal+focal | **6** | 0.36 [.25, .45] (66) | 0.18 [.09, .24] | 0.36/0.35/0.17 | human κ *carried by No-code abstention*, not stage discrimination (LLM 0.18) |
| B1 | Qwen-4096 | GraphSAGE **+ anchors** | bal+focal | 6 | 0.29 [.17, .44] (66) | 0.18 [.11, .25] | 0.40/0.35/0.17 | concept anchors **lower** every axis vs. A4n |

> **Read note (comparability):** the 5-class probe arms (A1/A1w/A2) *defer* "No code" → they are scored on an easier **n = 37** subset (29 No-code items dropped). The fair full-task comparison is the **6-class arms at n = 66**. The headline winner is therefore **A1n (human 0.37, n = 66)**, and "graph ≈ probe on the human axis" (A4n 0.36 vs. A1n 0.37) holds only because A4n's human κ is the No-code abstention lever, not VAAMR-stage discrimination — A4n's LLM-axis stage agreement is just 0.18.

---

## 4. The H5 classifier verdict

**H5 is not supported at n ≈ 32.** With real Qwen features the graph reproduces the LLM consensus on held-out **participants** at grouped **κ ≈ 0.05–0.14** — far below the κ ≥ 0.70 promotion gate and **below even the human↔human band**. (The MiniLM floor is 0.05; the production Qwen GNN grouped gate lands at 0.14; the Qwen GNN battery arms reach LLM-axis 0.16–0.21 / human-axis 0.10–0.14 for the five real stages, with A4n's higher human 0.36 attributable to No-code abstention, not stage discrimination.) The graph never earns its place as the VAAMR classifier of record: a **linear probe on the same features ties or beats it**, and adding graph machinery (Correct-&-Smooth, anchors) only *lowers* reliability. The multi-run LLM consensus (κ = 0.537 vs. human, already human-level and affordable at trial scale) remains the **label of record**.

---

## 5. The CV-leakage correction (the central methodological result)

A prior trial reported the GNN at **κ ≈ 0.247** (LLM axis), which looked within striking distance of "passing." That number was an **artifact of random k-fold leakage**, not generalization.

**Mechanism.** In a transcript graph, a held-out segment's neighbours — via both the **temporal chain** (its adjacent turns) and the **kNN-cosine** edges (same speaker, same recurring topics) — include **its own and same-participant segments whose labels are visible to the model**. A participant also dwells near one developmental position across a session (stage auto-correlation). So under *random* folds the model effectively **sees the answer** for held-out items through their leaked neighbours. This is the classic transductive-GNN evaluation trap.

**Correction.** Holding out **whole participants** (`StratifiedGroupKFold`) severs every leaked edge, and the same MiniLM GNN collapses from **κ ≈ 0.247 (random k-fold) → ≈ 0.05 (participant-grouped)** (human axis −0.02). Participant-grouped CV is now the default throughout the harness, and the production reliability gate (`gnn_layer.validation` ← `train.crossval_predictions`) was migrated to grouped folds as a result (random κ 0.21 → grouped 0.14 in the production path). **Every prior κ in this project must be read through this correction** — random-fold gate numbers are leakage-inflated.

---

## 6. Probe ≥ graph, and why — VAAMR is not homophilous in embedding space

The load-bearing diagnostic. A GNN's core inductive bias is **homophily**: neighbours are assumed to share labels, so mean-aggregating a node with its neighbours *denoises* the label. **VAAMR violates this.**

- A linear probe on the Qwen features **ties or beats the graph**: probe **human κ ≈ 0.37 / LLM-axis κ 0.31** (A1n / A1w) vs. graph **0.36 / 0.21** (A4n / A3). And **pure graph smoothing — Correct-&-Smooth on the probe — is the *worst* arm (≈ 0.16, A2)**: *adding* graph propagation actively destroys signal. That is the fingerprint of a low-homophily graph.
- **Why:** cosine similarity in the embedding space tracks **content** (topic, words, body region, affect), while VAAMR is a **developmental / process state**. Two utterances about "my lower back while sitting" can sit at Vigilance vs. Reappraisal — near-neighbours in Qwen space but opposite ends of the VAAMR arc. The kNN graph therefore systematically wires **different-stage** nodes together, and message passing **blurs the very distinction being classified** (over-smoothing toward the local content cluster's modal stage, usually Attention-Regulation). The probe pays no such tax — it learns a clean linear boundary on the already-separable features without averaging in mislabeled neighbours.

The features were the original binding constraint (MiniLM-384 → Qwen3-8B-4096 lifts the probe off the 0.05 floor to 0.31/0.37) — but once features are adequate, **the graph structure is a liability for this label, not an asset.**

---

## 7. Anchors hurt

The pre-registered Path-B escalation (B1, `anchors_arm.py`) added the 5 VAAMR **construct-definition anchor nodes** (CFiCS-style) to the best 6-class GNN. On the load-bearing human axis it **lowers** reliability: **0.36 (A4n) → 0.29 (B1)**, with the point estimate dropping on every axis. Same homophily reason, doubled: (i) the construct definitions live in a more abstract region of the space than lived participant speech, so the similarity edges are noisy; and (ii) those edges are again **not label-homophilous** — they pull a segment's representation toward a generic definition centroid rather than toward same-stage exemplars. CFiCS's gains came from a hand-authored, typed, definitional graph in a synthetic few-shot regime; QRA's empirical similarity-to-anchor edges are a categorically weaker object. **A content-similarity graph cannot recover a process label**, and bolting on definitional anchors does not change that.

---

## 8. What actually helped — class-weight + a "No code" null category

The two levers that improved reliability are **measurement / operationalization, not graph machinery**:

1. **Class-weighting** (`class_weight='balanced'`) recovered the rare stages from **0% held-out recall** to non-zero: A1 → A1w takes Vigilance/Avoidance/Metacognition recall from **0/0/0 → 0.56/0.35/0.31** and LLM-axis κ 0.23 → 0.31. Standard imbalance handling — necessary and sufficient for rare-stage recall — not a graph contribution.
2. **An explicit "No code" 6th class.** ~36% of participant segments (and 36% of human-coded items) express **no VAAMR stage**. A 5-class model is *forced* to assign a stage and is wrong on ~a third of items by construction; only a model that can **abstain** matches human coders there. The No-code class lifts human κ even on weak MiniLM (−0.02 → 0.21) and is what carries the 6-class human axis (A1n 0.37, A4n 0.36). This was a **construct-operationalization gap** — VAAMR as specified lacked a null category, and the human axis punished its absence — not a modeling gap.

Neither lever is "AI"; both are measurement discipline (plus the grouped-CV correction in §5). The one modeling lever that helped is standard imbalance handling.

---

## 9. Discriminant-validity corollary (provisional H6)

The graph's **inability** to classify VAAMR from semantic similarity is itself **positive evidence about the construct**. A label that a content-similarity graph could recover would be, by definition, a *topic taxonomy*. VAAMR is **not recoverable that way** — message passing over cosine-similar neighbours blurs rather than sharpens it — which is consistent with VAAMR indexing a **developmental re-habituation trajectory orthogonal to surface content**, not a semantic category. The same evidence that **refutes H5** (graph-as-scaler) **supports H6** (discriminant validity: VAAMR is developmental/process, not topical). Provisional, n-bound, but a genuine construct-validity signal extracted from a negative classifier result.

---

## 10. Repositioning — mechanism & discovery, not classifier of record

**Position adopted:** the multi-run LLM consensus (human-level at κ = 0.537, affordable at trial scale) stays the **label of record**. The GNN is **dropped as a classifier / scaler of record** and **retained as a mechanism & discovery instrument** — counterfactual cue-influence, participant↔therapist coupling, subtext communities, emergent motifs — where its graph-native structure is the point and it is used as an *exploratory* lens, explicitly so. The reliability gate is **kept** as the honest, grouped, per-stage yardstick (now leak-free).

Even in the mechanism role the readout is honest about its limits: the *first* mechanism instrument (the
model-counterfactual on a per-segment GNN classifier, the former `gnn_layer/influence.py`, run by the
now-retired `run_mechanism.py`) found the Qwen GNN counterfactual cue-influence does **not triangulate**
with the observed `analysis/mechanism.py` Δprogression (Spearman ρ = **−0.13** [−0.48, 0.01] plain / −0.25
balanced — both CIs include 0; coupling factor |corr| < 0.07), with **0 of 20 cells FDR-significant** in the
observed table at this scale.

**Update (GNN repositioning).** That per-segment counterfactual was diagnosed as **mis-specified for a
*process* question** (kNN content-noise; never trained on transitions; a single diluted cue→participant
edge) and was **retired**. It has been **rebuilt** as the dyadic FROM→CUE→TO transition model
`src/gnn_layer/transition.py` (`TO_mixture ≈ f(FROM_mixture, FROM_stage, pooled raw-Qwen cue)`, no kNN,
FROM-stage conditioned), whose learned counterfactual now triangulates **positively** with the observed
ranking (Spearman ρ ≈ **+0.34**, versus the retired −0.13), shipped with a confound-localization map
(`src/gnn_layer/confound.py`). It remains hypothesis-generating, not causal: at n ≈ 32 the cue does not
*earn its place* under participant-grouped CV (the transition is under-identified), so **`mechanism.py`
(observed) still leads**. Full record: `docs/methodology.md` §8.5 (Track B). Per the pre-registered
protocol the binding constraint throughout is **data scale (n ≈ 32)**; most conclusions here are explicitly
n-bound and worth re-running as participants accrue.

---

## 11. Reproduce

```bash
# Classifier battery (A1–A4n) → docs/gnn_experiments/ledger.csv + stdout
python src/experiments/gnn_reliability/run_battery.py            # full battery
python src/experiments/gnn_reliability/run_battery.py A3 A4 A4n  # subset

# Anchors arm (B1) and the capacity sweep (bridges to the scaler campaign)
python src/experiments/gnn_reliability/capacity_scaler.py

# PRIMARY mechanism readout (counterfactual + §1A triangulation)
python src/experiments/gnn_reliability/run_mechanism.py
```

- Reads the project corpus at **`data/Meta`** (`master_segments.csv` + the human consensus join from `qra.db`).
- Folds are **participant-grouped `StratifiedGroupKFold`, seed 42** — deterministic and shared by every arm.
- Qwen embeddings are served by LM Studio (`text-embedding-qwen3-embedding-8b`, 4096-d, `/v1/embeddings`) and cached at `data/Meta/02_meta/gnn/segment_embeddings_qwen3_8b.npz` (text-hash-keyed; a warm cache never re-hits the endpoint).

---

## 12. Successor campaign

This battery is the **predecessor that motivated the classification-scaler campaign** documented in [`../classification_scaler/CAMPAIGN_LOG.md`](../classification_scaler/CAMPAIGN_LOG.md). The winning arm here — **A1n, the Qwen class-weighted 6-class linear probe (human κ 0.365 / LLM-axis grouped κ 0.31)** — became the **scaler candidate and the baseline to beat** in that campaign, which reused this directory's `harness.py` / `baselines.py` apparatus (same participant-grouped CV, same dual-axis scorer, seed 42) to sweep further levers (context embeddings, soft-label distillation, two-stage No-code gating, per-rater ensembling, capacity). That campaign's honest conclusion mirrors this one: three independent methods converge on **LLM κ ≈ 0.36 / human κ ≈ 0.45**, neither clearing the LLM-equivalence bar (LLM κ ≥ 0.45 or human κ ≥ 0.50) — the signature of a **data ceiling at n ≈ 32, not a method gap** — yielding at best an *assistive, human-reviewed* pre-labeler, never an autonomous LLM replacement.
