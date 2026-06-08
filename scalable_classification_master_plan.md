# Scalable VAAMR Classification — Probe Master Plan

> **One-line purpose.** Ship a **calibrated linear probe on Qwen embeddings** as QRA's
> **LLM-free, gated, abstention-aware scalable VAAMR classifier** — the working realization of the
> "validation-gated acceleration" the methodology specifies (§6.4) and the role the GNN failed at
> (H5). The LLM consensus stays the label of record; the probe fills gaps *below* it, cheaply, at
> the larger scale where running the LLM on every segment is impractical.
>
> **Companion records:** `experiments/docs/graph_experiments.md` (why probe > GNN), `experiments/docs/design_decisions.md` (the battery),
> `docs/methodology.md` §6.4/§8.5 (manuscript), `docs/GNN_MASTER_PLAN.md` §1.1/§4.7 (GNN repositioning).

---

## 1. Decision & honest assessment — should the probe scale the LLM?

**Yes, with guardrails.** The pilot battery (`experiments/docs/design_decisions.md` §7) settled the architecture
question: on identical participant-grouped folds, a class-weighted logistic probe on Qwen3-8B
embeddings reaches **human κ = 0.365 [0.23, 0.51]** and recovers the rare stages, **beating the GNN**
(0.36/collapse) and Correct-&-Smooth (0.16). So the probe — not the graph — is the defensible cheap
classifier. But two honest facts bound how it ships:

- **It is lossy vs the LLM.** Probe human κ = 0.365 (the *lower* human↔human band, α 0.33–0.52) vs the
  LLM's 0.537 (the human *level*). Probe-labeled segments are therefore noisier than LLM-labeled ones
  and must be **provenance-tagged and ranked below the LLM**, never allowed to override it.
- **Its payoff is prospective.** At pilot scale (n≈32, ~hundreds of participant segments) the
  multi-run LLM is affordable, so the probe is not *needed* yet. It earns its place when (a) new
  cohorts must be labeled without per-segment frontier-LLM cost, (b) the MindfulBERT corpus needs bulk
  labels beyond what the LLM budget covers, or (c) a downstream consumer wants instant re-labeling.
  **Build it now** (cheap, and it is the defensible scaler), **gate it per-project**, and **use it only
  where the LLM is impractical**.

**Net recommendation:** ship the probe as an **optional, gated scaling tier** — `qra probe train/
classify` — that labels segments the LLM has *not* labeled, defers (abstains) where it is unsure, and
writes a `probe_consensus` provenance tier below `llm_zero_shot`. Remove the GNN's failed classifier/
scaler surface (CLI `gnn classify`, the `gnn_authoritative`/`gnn_consensus` promotion, the scale-mode
machinery) and route those responsibilities to the probe. **Keep the GNN for mechanism/discovery only**
(GNN_MASTER_PLAN §1.1).

---

## 2. The revised epistemic chain

```
   human raters  ⟷ (validated at IRR) ⟷  multi-run LLM consensus   ← LABEL OF RECORD (human-level, κ=0.537)
                                                  │
                                   distillation (cheap linear student on Qwen embeddings)
                                                  ▼
                                          CALIBRATED PROBE
                                                  │
                       GATED: may fill UNLABELED segments only after probe↔human κ
                       reaches the human band on the held-out human subset (participant-grouped),
                       and only where it does not abstain
                                                  │
                                                  ▼
                cheap, gated, abstention-aware VAAMR labels on segments the LLM did NOT label
                                                  │
                                                  ▼
                          full-corpus labels (LLM/human where validated; probe where scaled,
                          provenance-tagged lower-confidence)  →  analysis + MindfulBERT
```

H5 (distillation/scalability) was wrong about the *architecture* (a content-similarity graph), not the
*principle*. The probe distills the LLM to the human band with a model whose inductive bias fits the
data (direct supervision finds the VAAMR direction in the embeddings; it does not assume homophily —
H6). This chain is identical to the GNN's intended one (`GNN_MASTER_PLAN.md` §2) with the probe
substituted for the graph and the promotion direction corrected (probe fills *below* the LLM).

---

## 3. The probe specification (exact)

### 3.1 New module — `src/classification_tools/probe_classifier.py`
Promotes the proven experimental code (`experiments/gnn_reliability/baselines.run_linear_probe`,
`harness.build_folds`/`score_arm`) into production. Pure sklearn + the existing embedding path; no
torch, no graph.

```python
@dataclass
class ProbeConfig:                       # serialized into PipelineConfig.probe
    enabled: bool = False                # opt-in scaling tier (default off; LLM stays primary)
    # ---- features: reuse the GNN/codebook embedding substrate ----
    embedding_backend: str = 'openai'    # Qwen via LM Studio /v1/embeddings (graph_experiments §2)
    embedding_base_url: Optional[str] = 'http://10.0.0.58:1234/v1'
    embedding_model: str = 'text-embedding-qwen3-embedding-8b'
    use_query_prefix: bool = True
    # ---- model ----
    n_classes: int = 6                   # 5 VAAMR stages + "No code" (the human-axis lever, H/§8.5)
    class_weight: str = 'balanced'       # recovers rare stages (Avoidance/Metacognition)
    C: float = 1.0
    max_iter: int = 3000
    # ---- abstention / calibration (don't poison the corpus) ----
    calibrate: bool = True               # temperature-scale held-out logits → honest confidence
    abstain_threshold: Optional[float] = None        # global max-prob floor below which → defer
    abstain_per_stage: Optional[Dict[int, float]] = None   # per-stage floors (rare stages higher)
    abstain_target_precision: float = 0.80           # calibrate floors to this held-out precision
    # ---- gate (per-project trust) ----
    irr_human_band_floor: float = 0.33   # probe↔human κ must reach the human band to scale
    rare_stage_recall_floor: float = 0.20
    validation_folds: int = 5
    seed: int = 42
```

### 3.2 Functions
- `train_probe(df_all, output_dir, config) -> ProbeModel`
  1. Qwen embeddings via `gnn_layer.embeddings.load_or_build_segment_embeddings` (cached npz).
  2. Training rows = participant segments with an LLM `final_label` (0–4); plus, for the 6-class head,
     participant rows the LLM marked "No code" → class 5. (Provenance: train ONLY on the LLM/human
     label of record — never on prior probe labels; no model-distilling-model.)
  3. Fit `LogisticRegression(class_weight, C, max_iter)` on L2-normalized embeddings.
  4. If `calibrate`: fit temperature on held-out logits (reuse `gnn_layer.calibration.fit_temperature`).
  5. Persist to `02_meta/probe/probe_model.joblib` + `probe_manifest.json` (embedding model+dim+hash,
     classes, class counts, temperature, training-label provenance mix, timestamp, code version).
- `ProbeModel.predict(texts_or_emb) -> (pred:int[0..5], conf:float, stage_mixture:float[5])`
  — calibrated max-softmax confidence; class 5 → "No code"/abstain.
- `evaluate_probe(df_all, output_dir, config) -> verdict`
  — **participant-grouped** CV (`StratifiedGroupKFold` — the only leak-free protocol, §`train.py`
  fix): probe↔LLM κ (held-out, n=labeled) **and** probe↔human κ (on the 76-item subset via
  `analysis.irr_join.populate_human_columns` + `analysis.irr_stats.cohen_kappa`), per-class
  recall/precision, participant-clustered bootstrap CIs. Writes `06_reports/06_classifier/
  probe_validation.txt` + `03_analysis_data/probe/probe_gate.json`
  (`{ready_for_scaling, probe_human_kappa, probe_llm_kappa, rare_ok, ...}`). **Verdict `ready` iff
  probe↔human κ ≥ `irr_human_band_floor` AND no rare-stage recall < floor.**
- `classify_with_probe(df_all, output_dir, config) -> n_written`
  — load the persisted probe; predict on participant segments **without** an LLM label of record (the
  scaling target); apply abstention (class 5, or calibrated per-stage floor → defer); write a
  `probe_labels` overlay (segment_id, probe_pred, probe_conf, probe_abstain). LLM-labeled segments are
  untouched.

### 3.3 Reuse, not reinvent
- Embeddings: `gnn_layer/embeddings.py` (Qwen backend already shipped).
- Human-axis join + κ: `analysis/irr_join.py`, `analysis/irr_stats.py`.
- Grouped folds + bootstrap CI: lift `experiments/gnn_reliability/harness.build_folds` /
  `_kappa_cluster_ci` into `probe_classifier` (or a shared `analysis/grouped_cv.py`).
- Calibration + abstention floors: `gnn_layer/calibration.py`, `gnn_layer/train.calibrate_abstain_floors`
  (port the logic; they are model-agnostic).

---

## 4. Label of record & provenance

The probe is **lossier than the LLM**, so it ranks **below** it (the inverse of the old GNN tier):

```
adjudicated  >  human_consensus  >  llm_zero_shot  >  PROBE_CONSENSUS
```

In `src/process/assembly/master_dataset.py`:
- Add a `probe_consensus` source, engaged **only** when (a) `config.probe.enabled`, (b) the persisted
  `probe_gate.json` says `ready_for_scaling`, (c) the segment has **no** higher-tier label, and (d) the
  probe did **not** abstain on it. Otherwise the segment keeps its LLM/human label or stays unlabeled.
- Emit a `label_confidence_tier` of `probe` (distinct from the LLM tiers) so downstream consumers
  (MindfulBERT datasheet, analysis) can weight or exclude probe labels.
- The probe **never** overrides an existing LLM/human label (the opposite of `gnn_authoritative`).

`MindfulBERT` (Track C) already tags each example with the *weakest* endpoint provenance; add
`probe_consensus` to that tier ladder (below `llm_zero_shot`) so probe-scaled cue blocks are
down-weighted, never silently mixed with human/LLM-grade labels.

---

## 5. CLI changes (`qra.py`)

**Remove / deprecate (GNN classifier-scaler surface):**
- `cmd_gnn_classify` (`qra gnn classify`, qra.py:1921) — delete the subcommand (or keep a stub that
  prints "deprecated → use `qra probe classify`"). It calls `run_gnn_classify`, which is removed (§8).
- The `qra gnn` help line (qra.py:1462) drops `classify`; `qra gnn` becomes **train / status** only
  (the mechanism gate, not a scaler).

**Add (`qra probe …`, mirroring the old gnn verbs):**
```
qra probe train    -o ./data/output/     # fit on LLM-labeled segments + run the participant-grouped gate
qra probe status   -o ./data/output/     # probe↔human / probe↔LLM κ; "ready to scale?" verdict
qra probe classify -o ./data/output/     # LLM-free label UNLABELED participant segments (+ abstain)
                   [--fresh]             #   re-fit from scratch
```
- `qra probe classify` is the scaler; it refuses to run unless `probe_gate.json` says `ready` (or
  `--force` with a logged warning), mirroring the gate-before-trust discipline.
- Wire into `qra run` as an optional post-classification step behind `config.probe.enabled` (train +
  gate + scale unlabeled), so a full pipeline run can produce probe-scaled labels in one pass.

---

## 6. TUI changes (`src/process/interactive_tui.py`)

**Remove (GNN classifier-scaler actions):**
- The menu action `'Classify — Graph consensus (LLM-free)'` (tui:1310) and its handler
  (tui:401–433, the `run_gnn_classify` block + the "Classify with the graph anyway?" confirm).
- The `gnn_authoritative` config toggle (tui:1027–1032) — the GNN never becomes label of record now.
- The `qra gnn … classify` line in the help blurb (tui:1462).

**Add (Probe scaling actions):**
- A menu group **"Probe — LLM-free scalable classification"** with three actions mirroring the GNN's
  old train/status/classify, each showing the gate verdict tag (`_probe_tag(status)` analogous to
  `_gnn_tag`): *Train + gate*, *View probe reliability (probe↔human/LLM κ)*, *Classify unlabeled
  segments (LLM-free)* — the last gated on the persisted verdict with the same "scale anyway?" confirm.
- A `probe.enabled` + `irr_human_band_floor` knob in the config editor (replacing the removed
  `gnn_authoritative` knob).

---

## 7. Pipeline & store changes

- **New overlay table `probe_labels`** in `qra.db` (`src/process/db.py` + `classifications_io.py`):
  columns `segment_id, probe_pred, probe_conf, probe_abstain` (+ `PROBE_OVERLAY_FIELDS`, the
  `'probe' -> probe_labels` route, read/write/clear/merge mirroring the `gnn` overlay). The
  `master_segments.csv` export gains `probe_pred/probe_conf/probe_abstain` columns.
- **Orchestrator hook** (`src/process/orchestrator.py`): after the LLM classification stage, if
  `config.probe.enabled`, call `probe_classifier.train_probe` → `evaluate_probe` → (if ready)
  `classify_with_probe`, then `assemble_master_dataset` consumes the `probe_consensus` tier. Reuse the
  existing `_gnn_promotion_flags` pattern for a `_probe_promotion_flags(config, output_dir)` that reads
  `probe_gate.json`.
- **Setup wizard** (`src/process/setup_wizard.py`): replace the `gnn_authoritative` /
  `produce_consensus_labels` prompts (178–179, 1497) with a single "Enable the probe scaling tier?
  (LLM-free, gated)" prompt that sets `probe.enabled`.
- **Reports**: a new `06_reports/06_classifier/probe_validation.txt` (the probe gate, mirroring the old
  `06_gnn/validation.txt`) — note this lives in a *classifier* report dir, separate from `06_gnn/`
  (which is now mechanism-only).

---

## 8. What to take AWAY from the GNN (and what to keep)

**Remove / repurpose (the failed classifier-scaler responsibilities):**
| Item | File | Action |
|---|---|---|
| `run_gnn_classify` (inductive LLM-free labeler) | `gnn_layer/runner.py:553` | **delete** — the probe is the scaler; the graph never labels of record |
| `attach_new_segments` scale-mode + A5 scale-sim | `graph_builder.py`, `validation.py` | **delete/retire** — probe predicts on embeddings; no graph attachment needed |
| `gnn_consensus` provenance tier + `gnn_authoritative AND gate_passed` promotion | `assembly/master_dataset.py:47–108`, `orchestrator._gnn_promotion_flags` | **delete** — replaced by §4's `probe_consensus` (ranked *below* the LLM) |
| `produce_consensus_labels` / the `gnn_labels` overlay as a *label of record* | `config.py`, `classifications_io.py`, `master_dataset.py` | **demote** — keep `gnn_labels` only as a held-out diagnostic for the IRR triangulation, never promoted |
| `gnn_authoritative` knob | `setup_wizard.py:178/1497`, `interactive_tui.py:1027` | **delete** — there is no "graph as label of record" path |
| The reliability gate as a **scaling trigger** | `gnn_layer/validation.py`, `runner.py` | **repurpose** — the GNN gate κ stays only as the *trust-context* number reported beside the counterfactual (GNN_MASTER_PLAN §4.7), not a license to scale |
| `qra gnn classify` CLI + TUI action | `qra.py:1921`, `interactive_tui.py:401/1310` | **delete** (→ `qra probe classify`) |

**Keep (the mechanism/discovery role — GNN_MASTER_PLAN §1.1/§4.7):**
- `qra gnn train` for the counterfactual influence (`influence.py`), coupling, communities, motifs.
- The GNN gate κ as a reported *trust context* for the mechanism readout (not a scaling gate).
- The Qwen embedding backend (shared by the probe and the GNN — built once).

Net: the GNN's surface shrinks to **train (mechanism) + status (trust context)**; the probe owns
**train (gate) + status + classify (scale)**.

---

## 9. Testing strategy

- **Hermetic unit tests** (`tests/unit/test_probe_classifier.py`): synthetic df + patched embeddings
  (reuse `tests/testhelpers/fixtures.embedding_patch`). Assert: 5- vs 6-class head; `class_weight` lifts
  rare-class recall; calibration softens confidence; abstention defers below the floor; the gate verdict
  is `ready` only above the human-band floor + rare-recall floor; the overlay round-trips; the
  `probe_consensus` tier in `master_dataset` engages only when `enabled AND gate ready AND no
  higher-tier label AND not abstained` (4-way guard) and **never** overrides an LLM label.
- **Reproduction guard:** the probe gate on the existing `data/Meta` corpus reproduces the battery's
  A1n numbers (human κ ≈ 0.365), proving the production module matches the experiment.
- **Backward-compat:** with `probe.enabled=False` (default) the pipeline is byte-identical to today.
- Slow/real-embedding tests gated `@slow_test`.

---

## 10. Sequencing (phases)

1. **P1 — Probe module + gate (no pipeline wiring).** `probe_classifier.py` + tests; `qra probe
   train/status` only. Verify the gate reproduces A1n on `data/Meta`. *(Self-contained; low risk.)*
2. **P2 — Overlay + classify.** `probe_labels` table, `classify_with_probe`, `qra probe classify` with
   the gate guard + abstention. Unlabeled-segment scaling works end to end.
3. **P3 — Provenance + assembly.** `probe_consensus` tier in `master_dataset` (below LLM, 4-way
   guarded) + MindfulBERT tier-ladder update + the export columns.
4. **P4 — Remove the GNN classifier-scaler surface** (§8) + setup-wizard/TUI swap. One commit per
   removed responsibility (reversible).
5. **P5 — Orchestrator hook + `qra run` integration** behind `probe.enabled`; docs (methodology §6.4
   gains the probe as the realized cheap-scaling tier; CLAUDE.md command list).
6. **P6 (future, when N grows) — re-gate per cohort**; if probe↔human reaches the LLM level, consider
   raising its tier. The gate is the standing decision rule.

---

## 11. Honest caveats & the decision rule for *using* it

- **Lower-band quality is the price of cheapness.** Probe-scaled labels are ~lower-human-band; they
  are tagged `probe_consensus` and should be **excluded or down-weighted** in any analysis where the
  reliability ceiling matters, and **never** mixed un-tagged into the human/LLM-grade corpus.
- **The gate is per-project, not once-and-done.** `evaluate_probe` must pass on *that project's* human
  subset before `classify` scales; a project without a human subset cannot trust the probe (the gate
  returns `not ready`).
- **Abstention is mandatory, not optional.** A confident-wrong probe label on a novel segment poisons
  the MindfulBERT corpus the next cohort trains on; the probe must be able to say "No code"/defer.
- **It does not dissolve the construct ceiling.** Human↔human α ≈ 0.33–0.52 bounds *any* VAAMR labeler;
  the probe at 0.365 is inside that band, not beyond it.
- **Decision rule:** *use the LLM where you can afford it (it is human-level); use the gated probe to
  fill what the LLM budget cannot reach, tagged as such.* At the pilot, that is "rarely"; at trial
  scale or for the bulk MindfulBERT corpus, that is "often" — which is exactly when this plan pays off.

---

## 12. Module & file index (anticipated)

| Path | Role | Status |
|------|------|--------|
| `src/classification_tools/probe_classifier.py` | probe train/predict/evaluate/classify + `ProbeConfig` | **new** |
| `src/process/probe_io.py` *(or extend `classifications_io.py`)* | `probe_labels` overlay r/w | **new/edit** |
| `analysis/grouped_cv.py` *(optional)* | shared participant-grouped folds + κ-CI (lifted from the harness) | **new** |
| `src/process/config.py` | `PipelineConfig.probe = ProbeConfig` | edit |
| `src/process/assembly/master_dataset.py` | `probe_consensus` tier (below LLM) | edit |
| `src/process/orchestrator.py` | probe train→gate→scale hook; `_probe_promotion_flags` | edit |
| `src/process/db.py` | `probe_labels` table | edit |
| `qra.py` | `qra probe {train,status,classify}`; remove `gnn classify` | edit |
| `src/process/interactive_tui.py` | probe actions; remove GNN classify + `gnn_authoritative` | edit |
| `src/process/setup_wizard.py` | `probe.enabled` prompt; remove `gnn_authoritative` | edit |
| `gnn_layer/runner.py`, `graph_builder.py`, `validation.py` | **remove** `run_gnn_classify` + scale-mode; gate → trust-context only | edit |
| `tests/unit/test_probe_classifier.py` | hermetic coverage | **new** |
| `docs/methodology.md` §6.4 / `docs/GNN_MASTER_PLAN.md` §1.1 | note the probe as the realized cheap-scaling tier | edit |
