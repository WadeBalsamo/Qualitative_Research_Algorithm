# Classification Methods — Master Arm Catalog

> **Shipped method:** `04_multimodel_consensus` (multi-model majority vote, commit `56dd301`) is the
> VAAMR label-of-record for the QRA pipeline. All other arms are baselines, ablations, historical
> anchors, or alternative signals measured against it.
>
> **Scoring axis:** content validity — accuracy on `cv_vaamr_v1` (109 curated exemplar/subtle/
> adversarial utterances; gold = `expected_stage`). This is **not** the κ-vs-consensus axis used by
> the sibling campaigns (`gnn_reliability/`, `classification_scaler/`).

---

## Arm Table

| Experiment | Method / era | Git commits | Arms (count) | Scored against | Runnable? | CV result |
|---|---|---|---|---|---|---|
| **01 — embedding_similarity** | Pure cosine-sim to VAAMR theme anchors (all-MiniLM-L6-v2). No training data; zero-shot framework-description baseline. | — | 5 (`def_only`, `def_exemplars`, `exemplars_only`, `def_criteria`, `def_exemplars_qprefix`) | `cv_vaamr_v1` | yes (offline) | not run |
| **02 — embedding_probe** | Trained LogReg probe on MiniLM corpus embeddings (205 labeled segments), predict CV items in same space. Mirrors `gnn_reliability/baselines.py` probe pattern. | — | 2 (`probe_5class`, `probe_classweighted`) | `cv_vaamr_v1` | yes (offline) | not run |
| **03 — single_model_zeroshot** | Foundation method: one model, one pass, `n_runs=1`, `merge='first'`. Establishes the simplest performance floor for all consensus experiments. | `7dfa61c` (Mar 2026) | 1 (`single_zeroshot__<model>`) | `cv_vaamr_v1` | yes — needs LM Studio | not run |
| **04 — multimodel_consensus** ⭐ | **METHOD-OF-RECORD.** N-model majority vote; per-item confidence tiers from agreement distribution. Production system as of Apr 2026. | `56dd301` (Apr 2026) | 2 (`consensus_3model`, `consensus_5model`) | `cv_vaamr_v1` | yes — needs LM Studio | not run |
| **05 — ensemble_embedding_llm** | Reconcile cosine-similarity (all-MiniLM anchors) + single LLM pass using five combination strategies. Adapted from VCE codebook ensemble. | `f84d38b` (Mar 2026 codebook era) | 5 (`llm_only`, `embedding_only`, `agree_or_llm`, `agree_or_embedding`, `union_flag`) | `cv_vaamr_v1` | yes — needs LM Studio + all-MiniLM | not run |
| **06 — vce_codebook_embedding** | Self-retrieval validity check: does the VCE codebook embedding classifier recover each code from its own exemplar utterances (top-1 / top-3)? Uses all-MiniLM (Qwen3 blocked by env pin). | `f84d38b`, `cdbfc87`, `002757e` (Mar–Apr 2026) | 3 (`triple_veto_default`, `relaxed_threshold`, `no_two_pass`) | codebook own exemplars (self-retrieval; not `cv_vaamr_v1`) | yes — needs all-MiniLM | n/a — no exemplars yet (`n_items=0` until `PHENOMENOLOGY_CODEBOOK.md` is populated) |
| **07 — llm_model_battery** | Single-pass zero-shot on every LM Studio chat model under 120B. One arm per model; 20 models in static fallback list. | — | ~20 (one per enumerated model) | `cv_vaamr_v1` | yes — needs LM Studio | not run |
| **08 — prompt_designs** | Prompt-knob grid: vary exactly one `PromptSpec` field off default per arm (context window, exemplars, difficulty tiers, zero-shot flag, JSON strictness). Harness fixed at `n_runs=1, merge='first'`. Includes 2 historical commit-anchored arms. | `7dfa61c` (hist_no_context), `216bb4f` (hist_3tier_exemplars) | 16 (15 runnable + 1 documented-only: `hist_4stage__7dfa61c`) | `cv_vaamr_v1` | yes — needs LM Studio (15 arms); 1 skip | not run |
| **09 — harness_designs** | Voting/merge-knob grid: vary exactly one `HarnessSpec` field off default per arm (n_runs, model pool, merge strategy, secondary_weight, presence_threshold, abstain_as_ballot). Prompt fixed at defaults. Includes 3 historical commit-anchored arms. | `7dfa61c` (hist_triplicate_and_flag), `56dd301` (hist_unified_majority), `c6a724f` (hist_model_first_sweep) | 15 (14 runnable + 1 documented-only: `hist_model_first_sweep__c6a724f`) | `cv_vaamr_v1` | yes — needs LM Studio (14 arms); 1 skip | not run |

---

## Dead Ends (document-only, not run)

All three are documented in `99_dead_ends/README.md`. No run.py exists; no scores are available or
meaningful.

| Method | Era | Key commits | Status | Canonical reference |
|---|---|---|---|---|
| Lit-trained NSP classifier | Jan 2024 – Jun 2025 | `32040ca` (created), `c33741a` (deleted) | Abandoned — wrong objective (NSP ≠ stage assignment); required training data that did not exist | `99_dead_ends/README.md §1` |
| MentalBERT graph classifier (`mentalbert_sentence_aqua`) | Jun 2025 – Mar 2026 | `c33741a` (created), `7dfa61c` (deleted) | Abandoned — graph homophily assumption refuted; infeasible dep stack under env pin | `99_dead_ends/README.md §2` |
| GNN GraphSAGE consensus-distillation (H5) | Jun 2026 | `628e71a` (built), `8619066` (demoted default-OFF) | Demoted — H5 refuted at n ≈ 32; data ceiling; fully studied in sibling campaign | `99_dead_ends/README.md §3`; full battery in `experiments/gnn_reliability/` |

---

## Notes

- **All CV results are `not run`** — the campaign was built and verified but not executed. Run any
  `run.py` to produce results; each script appends one row per arm to its local `results.csv`.
- **`--dry-run`** on any script lists its arms without loading models or hitting the network.
- **Experiment 06** uses a different scoring surface than the others (not `cv_vaamr_v1`; no `common/cv_scoring`).
  Its result column will remain `n/a` until exemplar utterances are added to `PHENOMENOLOGY_CODEBOOK.md`.
- **07** has no static `arms.py` / `ARMS.md` — the model list is enumerated dynamically from the
  LM Studio endpoint at runtime (static fallback list of 20 models in `README.md`).
- **κ-vs-consensus numbers** (not content-validity scores) are reported in the sibling campaign
  catalogs at `experiments/CATALOG.md` (Campaigns 1–2).
