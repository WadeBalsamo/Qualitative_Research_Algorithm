# Classification Methods Campaign (`experiments/classification_methods/`)

> **What this is.** A systematic R&D history of every distinct VAAMR classification method tried on
> the QRA master branch — from the earliest dead-end approaches through the current multi-model LLM
> consensus — reconstructed as standalone, run-ready scripts and scored against a single shared axis:
> the **`cv_vaamr_v1` content-validity testset**.

---

## The R&D arc

QRA's VAAMR classifier did not arrive fully formed. The earliest attempt (Jan 2024) was a
lit-trained Next-Sentence Prediction transformer custom-built on arXiv papers — a training-data
problem with no training data. It was deleted when the project pivoted to graph methods (Jun 2025)
and a MentalBERT semantic graph took its place: ten modules of Louvain community detection and
triple-veto scoring, discarded in Mar 2026 when the dep stack became unloadable under the current
numpy/transformers pin and the graph-homophily assumption it relied on was formally refuted. Both
of those dead ends are documented in `99_dead_ends/`.

The live pipeline started in Mar 2026 with **single-model zero-shot** classification (commit
`7dfa61c`): one LLM, one pass, no consensus. That became the reference floor. Apr 2026 (commit
`56dd301`) layered in **multi-model majority voting** — the current method-of-record — where N
models are rotated across runs and per-item confidence tiers emerge from the agreement
distribution. Around the same time the VCE **codebook embedding classifier** (`f84d38b`) and a
**single-LLM + embedding ensemble** explored whether a deterministic cosine-similarity signal
could complement or replace expensive LLM calls.

The subsequent sibling campaigns (`experiments/gnn_reliability/`,
`experiments/classification_scaler/`) asked whether a trained **linear probe** or **GNN** could
scale labeling cheaply. Both campaigns score on a κ-vs-consensus axis (classifier ↔ LLM,
classifier ↔ human). **This campaign scores the same arc on a different axis: content validity —
does a method assign the right VAAMR stage to curated exemplar, subtle, and adversarial utterances
whose gold labels are known?** That distinction is the reason this campaign exists separately from
its siblings.

---

## How it is scored

**Testset:** `cv_vaamr_v1` — 109 items, 5 VAAMR stages × {clear, subtle, adversarial} difficulty
tiers. Gold label = `expected_stage` (int 0–4). Items are loaded from the project SQLite database
(`data/Meta/qra.db`); the testset was frozen there by `qra cv create --framework vaamr --name
cv_vaamr_v1`.

**Scoring surface:** `common/cv_scoring.py::score_arm(arm_name, predictions, results_csv=...)`.
Appends one row per arm to `results.csv`; optionally (re)writes a markdown table to `results.md`.
Returns a metrics dict.

**`results.csv` columns** (fixed schema — rows from different arms align):

| Column | Meaning |
|--------|---------|
| `arm` | Short arm identifier (e.g. `consensus_3model`, `probe_5class`) |
| `testset` | Always `cv_vaamr_v1` |
| `timestamp` | UTC ISO-8601 when the arm was scored |
| `n_items` | Total CV items evaluated (109 when full) |
| `n_abstain` | Items where the method returned no valid prediction |
| `acc_overall` | Primary accuracy — fraction of items correct (0–1) |
| `acc_secondary` | Primary-or-secondary accuracy (secondary = 2nd-argmax / 2nd-run vote) |
| `acc_clear` | Accuracy on `clear`-difficulty items |
| `acc_subtle` | Accuracy on `subtle`-difficulty items |
| `acc_adversarial` | Accuracy on `adversarial`-difficulty items |
| `acc_stage_0` … `acc_stage_4` | Per-VAAMR-stage primary accuracy |
| `n_clear`, `n_subtle`, `n_adversarial` | Item counts per difficulty tier |
| `n_stage_0` … `n_stage_4` | Item counts per expected stage |

---

## Status: built, not run

**Everything in this campaign is run-ready. Nothing has been executed.**

- `results.csv` / `results.md` do not yet exist in any subdir — they are created on first run.
- **LLM arms** (03, 04, 05, 07, 08, 09) require a live LM Studio backend at
  `http://10.0.0.58:1234/v1` (override with `--base-url` or `--model` where supported).
- **Embedding arms** (01, 02, 05 embedding-only) run fully offline with
  `sentence-transformers/all-MiniLM-L6-v2`. The project env pins `transformers==4.42.4` /
  `numpy==1.26.4`, which cannot load Qwen3 architecture weights (see memory note
  `project_gnn_embedding_transformers_pin.md`); all-MiniLM is used throughout.
- **Experiment 06** (VCE codebook embedding) is run-ready but will produce `n_items=0` until
  exemplar utterances are populated in `frameworks/PHENOMENOLOGY_CODEBOOK.md` — the parser
  contract is implemented, exemplars are not.

---

## How to run

All commands run from the **repo root** with the project venv active.

Add `--dry-run` to any script to list its arms (and for embedding/LLM arms, skip model loads and
network calls entirely).

```bash
# Activate venv
source .venv/bin/activate

# 01 — embedding similarity (offline, all-MiniLM)
.venv/bin/python experiments/classification_methods/01_embedding_similarity/run.py \
    --output-dir data/Meta
# Options: --model <st-model>  --arm <name|all>  --dry-run

# 02 — embedding probe (offline, all-MiniLM + corpus labels from qra.db)
.venv/bin/python experiments/classification_methods/02_embedding_probe/run.py \
    --output-dir data/Meta
# Options: --model <st-model>  --arm <name|all>  --dry-run

# 03 — single-model zero-shot (needs LM Studio)
.venv/bin/python experiments/classification_methods/03_single_model_zeroshot/run.py \
    --output-dir ./data/output
# Options: --model <id>  --dry-run

# 04 — multi-model consensus, METHOD-OF-RECORD (needs LM Studio)
.venv/bin/python experiments/classification_methods/04_multimodel_consensus/run.py \
    --output-dir ./data/output
# Options: --models <comma-list>  --dry-run

# 05 — ensemble embedding+LLM (needs LM Studio + all-MiniLM)
.venv/bin/python experiments/classification_methods/05_ensemble_embedding_llm/run.py \
    --output-dir ./data/output
# Options: --model <id>  --dry-run

# 06 — VCE codebook embedding self-retrieval check
#       (no exemplars yet → n_items=0; --dry-run confirms this cleanly)
.venv/bin/python experiments/classification_methods/06_vce_codebook_embedding/run.py \
    --output-dir experiments/classification_methods/06_vce_codebook_embedding/results
# Options: --arm <name>  --dry-run

# 07 — LLM model battery (needs LM Studio; ~20 models from static list)
.venv/bin/python experiments/classification_methods/07_llm_model_battery/run.py \
    --output-dir ./data/output/ --base-url http://10.0.0.58:1234/v1
# Options: --models <comma-list>  --list-models  --dry-run

# 08 — prompt-design grid (needs LM Studio; 15 runnable + 1 documented-only arm)
.venv/bin/python experiments/classification_methods/08_prompt_designs/run.py \
    --output-dir ./data/output
# Options: --arm <name>  --model <id>  --dry-run

# 09 — harness/voting-design grid (needs LM Studio; 14 runnable + 1 documented-only arm)
.venv/bin/python experiments/classification_methods/09_harness_designs/run.py \
    --output-dir ./data/output/
# Options: --arm <name>  --model-pool <comma-list>  --dry-run
```

---

## Directory map

```
experiments/classification_methods/
├── README.md                  ← this file
├── CATALOG.md                 ← master per-arm table (every arm, one page)
├── __init__.py
├── common/
│   ├── data.py                ← load_cv_items / load_cv_segments / load_vaamr / stage_names
│   ├── cv_scoring.py          ← score_arm → results.csv + results.md
│   └── prompt_harness.py      ← PromptSpec / HarnessSpec / run_llm_arm
├── 01_embedding_similarity/   ← pure cosine-sim vs VAAMR anchors (5 arms)         README.md + run.py
├── 02_embedding_probe/        ← trained LogReg probe on MiniLM corpus (2 arms)     README.md + run.py
├── 03_single_model_zeroshot/  ← foundation method, Mar 2026, commit 7dfa61c (1 arm) README.md + run.py
├── 04_multimodel_consensus/   ← METHOD-OF-RECORD, Apr 2026, commit 56dd301 (2 arms) README.md + run.py
├── 05_ensemble_embedding_llm/ ← embedding+LLM reconciliation (5 arms)             README.md + run.py
├── 06_vce_codebook_embedding/ ← VCE triple-veto self-retrieval check (3 arms)     README.md + run.py
├── 07_llm_model_battery/      ← per-model single-pass battery (~20 arms)          README.md + run.py
├── 08_prompt_designs/         ← prompt-knob grid + historical anchors (16 arms)   README.md + run.py + arms.py + ARMS.md
├── 09_harness_designs/        ← voting-knob grid + historical anchors (15 arms)   README.md + run.py + arms.py + ARMS.md
└── 99_dead_ends/              ← document-only: NSP classifier, MentalBERT graph, GNN GraphSAGE  README.md
```

**Relationship to sibling campaigns.** `experiments/gnn_reliability/` and
`experiments/classification_scaler/` score classifiers on a **κ-vs-consensus** axis (participant-
grouped StratifiedGroupKFold, dual-axis Cohen κ). This campaign scores on a **content-validity**
axis (known-exemplar accuracy, difficulty-tier breakdown). The two axes are complementary: κ
measures production reliability on real transcript segments; content validity measures definitional
fidelity on curated test cases. Neither subsumes the other.
