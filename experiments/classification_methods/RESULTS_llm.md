# LLM Classification Results — cv_vaamr_v1

LLM-based VAAMR classification arms scored against the `cv_vaamr_v1` content-validity
testset (109 items: 58 clear / 29 subtle / 22 adversarial, 5 VAAMR stages balanced at
~22 items each). Accuracy here is content-validity accuracy (primary match against the
expected stage), NOT kappa-vs-consensus. These results establish which model the
production pipeline should use, by empirical comparison against the same fixed testset
used for the offline floor (`RESULTS_offline.md`).

These experiments back the methodology claim that the production model is selected by
**empirical content-validity comparison against `cv_vaamr_v1`**, not by reputation or
size. Abstentions (empty / unparseable responses) count against `acc_overall` — the
denominator is always all 109 items — so a model that abstains is penalized honestly.

_Date: 2026-06-11. No participant text, utterances, session identifiers, or PHI appears
in any output file under `experiments/`. The `cv_vaamr_v1` items are synthetic VAAMR
exemplar/subtle/adversarial utterances (item ids `cv_0000`–`cv_0108`,
`participant_id='cv_participant'`, `trial_id='cv_test'`). All outputs contain only
aggregate metrics, difficulty/stage counts, and model identifiers._

Backend: LM Studio (single shared instance at `http://10.0.0.58:1234/v1`). The VAAMR
classification prompt (full framework + all exemplar/subtle/adversarial examples,
`context_window=2`) measures **7,311 tokens** (29,246 chars) — this is load-bearing for
the context-window findings below.

---

## Experiment A — LLM Model-Type Battery (single-pass zero-shot)

`07_llm_model_battery/run.py` — one model, one pass (`n_runs=1`, `merge='first'`),
`PromptSpec(context_window=2, zero_shot=False, include_subtle=True,
include_adversarial=True)`. The offline floor any LLM must beat is **acc_overall ≈ 0.45**
with **adversarial ≤ 0.36** (`RESULTS_offline.md`).

### Completed arms

| model | quant | reasoning? | acc_overall | acc_clear | acc_subtle | acc_adversarial | acc_secondary | n_abstain | n_items |
|-------|-------|-----------|-------------|-----------|------------|-----------------|---------------|-----------|---------|
| **nvidia/nemotron-3-nano-4b** | Q4_K_M | no | **0.789** | 0.845 | 0.862 | **0.545** | 0.789 | 0 | 109 |
| qwen/qwen3-8b | Q3_K_L | yes (`--no-reasoning`) | 0.762 | 0.810 | 0.931 | 0.409 | 0.762 | 1 | 109 |
| google/gemma-4-e2b | Q4_K_M | yes (`--no-reasoning`) | 0.688 | 0.672 | 0.862 | 0.500 | 0.688 | 1 | 109 |
| qwen2.5-0.5b-instruct | Q8_0 | no | 0.202 | 0.241 | 0.172 | 0.136 | 0.202 | 23 | 109 |

Per-stage accuracy (primary), completed arms:

| model | stage_0 (Vig) | stage_1 (Avoid) | stage_2 (AttReg) | stage_3 (Meta) | stage_4 (Reap) |
|-------|---------------|-----------------|------------------|----------------|----------------|
| nvidia/nemotron-3-nano-4b | 0.545 | 0.909 | 0.955 | 0.650 | 0.870 |
| qwen/qwen3-8b | 0.591 | 0.909 | 0.909 | 0.550 | 0.826 |
| google/gemma-4-e2b | 0.545 | 0.727 | 0.909 | 0.500 | 0.739 |
| qwen2.5-0.5b-instruct | 0.364 | 0.136 | 0.136 | 0.150 | 0.217 |

**Read:** `nvidia/nemotron-3-nano-4b` (4B, Q4_K_M, non-reasoning) is the winner among
completable models — 0.789 overall, zero abstentions, and the only model to materially
close the adversarial gap (0.545 vs the offline ceiling of 0.36). It beats the embedding
floor (0.45) by +0.34 overall. `qwen/qwen3-8b` (8B Qwen3 thinking model, `--no-reasoning`)
is a close second at 0.762 — strongest of all on clear/subtle items (0.81 / 0.93) but the
weakest of the top three on adversarial items (0.409, below the 0.45 floor), i.e. it reads
the obvious cases best but is most easily fooled by deliberately misleading ones.
`google/gemma-4-e2b` (2B Gemma4 thinking model, `--no-reasoning`) follows at 0.688. The
tiny `qwen2.5-0.5b-instruct` (0.5B) sits *below* the offline embedding floor (0.20 < 0.45)
and abstains on 23/109 items — too small to follow the 5-class instruction reliably;
included only as a small-model floor.

The `--no-reasoning` flag is the enabling trick for the two thinking models that completed
(`qwen3-8b` 8B and `gemma-4-e2b` 2B): with chain-of-thought suppressed both return direct
JSON in seconds and finished all 109 items with ≤1 abstention. It does **not**, however,
tame the heavier reasoning models on this host (`nemotron-3-super`, Qwen3-27B, Gemma4-12B+,
Phi-4-reasoning), which keep emitting long chains in the content field and stall — so the
split is model/runtime-specific, not a clean size cutoff (an 8B Qwen3 cooperates while a
12B+ Gemma4 does not). Empirically: try `--no-reasoning`; keep the model if it completes.

### Attempted but not completable in this LM Studio setup

The remaining models the campaign targets (across the qwen / gemma / nemotron / gemma-QAT /
phi / minimax families the researcher asked to compare) could not be scored here due to
two host constraints — not model quality. They remain in `STATIC_FALLBACK_MODELS` and are
re-runnable once the host is reconfigured (see "Host constraints" below).

| model | quant | arch | ctx (max) | family | blocker |
|-------|-------|------|-----------|--------|---------|
| nvidia/nemotron-3-super | Q4_K_M | nemotron_h_moe | 1,048,576 | nemotron | **stall (confirmed)**: reasons in content even with `--no-reasoning`; live log silent >25 min, killed with 0 items. Not battery-able on this host |
| nvidia/nemotron-3-nano | Q4_K_M | nemotron_h_moe | 1,048,576 | nemotron | **stall (confirmed)**: patient run (single-model eviction, 25-min stall tolerance, 3h ceiling) → log silent 1513s, killed at 0 items. Reasons in content despite `--no-reasoning` |
| nvidia/nemotron-3-nano-omni | Q4_K_M | nemotron_h_moe | 262,144 | nemotron | reasoning model (omni MoE) — same stall class as the other nemotron MoEs |
| qwen/qwen3.6-27b | Q4_K_M | qwen35 | 262,144 | qwen | **stall (confirmed)**: patient run loaded successfully and coded item 1 (stage=Vigilance) then went silent >25 min at item ~3/109 → killed, no row. Unlike its 8B sibling, the 27B reasons in content even with `--no-reasoning` |
| qwen/qwen3-coder-30b | Q4_K_M | qwen3moe | 262,144 | qwen | 30B — fails to load (VRAM): "Operation canceled" |
| unsloth/qwen3-coder-30b-a3b-instruct | (n/a) | qwen3moe | 262,144 | qwen | 30B MoE — VRAM |
| google/gemma-2-9b | Q4_K_M | gemma2 | 8,192 | gemma | **context (confirmed empirically)**: ran the full battery → 109/109 abstain, acc 0.0. LM Studio loaded it at n_ctx=4096 < 7,311-token prompt; every item errors (`n_keep 6101 >= n_ctx 4096`). Non-reasoning; would be viable if reloaded at ≥8192 ctx |
| google/gemma-4-12b-qat | Q4_0 | gemma4 | 262,144 | gemma-QAT | 12B QAT — fails to load (VRAM): "Operation canceled" |
| google/gemma-4-26b-a4b-qat | Q4_0 | gemma4 | 262,144 | gemma-QAT | 26B QAT MoE — VRAM |
| google/gemma-4-31b-qat | Q4_0 | gemma4 | 262,144 | gemma-QAT | 31B QAT — VRAM |
| google/gemma-4-31b | Q4_K_M | gemma4 | 262,144 | gemma | 31B — VRAM |
| microsoft/phi-4-reasoning-plus | Q8_0 | phi3 | 32,768 | phi | reasoning model — reasons in content; same slowness/stall class |
| minimax-m2.7 | Q3_K_S | minimax-m2 | 196,608 | minimax | repeatedly "Failed to load … Operation canceled" / gets stuck in `loading` state (large MoE + huge ctx) |
| magnum-v4-72b | Q4_K_M | qwen2 | 32,768 | (qwen2 ft) | 72B — VRAM ("Failed to load") |
| deepseek-v4-flash | Q4_K_M | deepseek4 | 1,048,576 | deepseek | VRAM ("Failed to load") |
| qwen/qwen3-next-80b | Q4_K_M | qwen3next | 262,144 | qwen | 80B — VRAM |

---

## Experiment B — Multi-Model Consensus (method of record)

`04_multimodel_consensus/run.py` — N-model majority vote (the production VAAMR method,
commit `56dd301`). **Not run live this session** (the host's VRAM-stacking prevents a
3-model per-item rotation: it can hold ~1–2 small models at once and wedges on the second/
third load). However, the battery now yields a **ready, complementary 3-model pool** for it:
`nvidia/nemotron-3-nano-4b` (best overall + most adversarial-robust), `qwen/qwen3-8b
--no-reasoning` (best clear/subtle), and `google/gemma-4-e2b --no-reasoning` (small
fallback). Their error profiles differ on exactly the axis consensus exploits (adversarial
vs clear/subtle), so a majority vote should pick up where any single model is weak. To run:
set LM Studio to single-model JIT eviction, then
`04_multimodel_consensus/run.py --models nvidia/nemotron-3-nano-4b,qwen/qwen3-8b,google/gemma-4-e2b`
(the consensus harness does not yet thread `--no-reasoning`; add it the same way the battery
does, or pre-warm the thinking models, before running).

## Experiment C — Single-Model Zero-Shot Floor

`03_single_model_zeroshot/run.py` — identical method to a single Experiment-A arm
(`n_runs=1`, `merge='first'`). The Experiment-A row for `nvidia/nemotron-3-nano-4b`
(acc_overall **0.789**, 0 abstentions) is exactly the single-model zero-shot floor for the
recommended model; the `qwen2.5-0.5b-instruct` row (0.202) is the small-model floor. No
separate run is needed to characterize the single-model baseline for the completable
models — Experiment A already produced it.

---

## Host constraints (why the battery is partial, and how to widen it)

Two LM Studio (10.0.0.58) behaviors — independent of model quality — capped the battery:

1. **VRAM stacking without auto-eviction.** LM Studio JIT-loads each requested model and
   keeps it resident. After ~2 small models are loaded it reports `Failed to load … Error:
   Operation canceled` for every subsequent model, including 4B ones. There is no
   server-side unload REST endpoint (`POST /api/v0/models/unload` and `DELETE` both return
   "Unexpected endpoint or method") and no `lms` CLI on this orchestration host, so models
   cannot be evicted programmatically from here. **Fix:** in LM Studio set *Max loaded
   models = 1* (auto-evict LRU), or raise the JIT TTL eviction so each new battery arm
   evicts the previous one. This alone unblocks every dense model ≤ ~13B and the QAT
   gemmas at Q4_0.

2. **Reasoning-in-content.** Every Qwen3 / Gemma4 / Nemotron-Super/Nano(MoE) / Phi-4-
   reasoning model emits its chain of thought in the *visible content* field and takes
   ~20s–15+ min per item (often stalling on adversarial items). LM Studio's
   `include_reasoning=false` (wired through this session — see below) does **not** suppress
   it for these models; they reason in content regardless. A 109-item battery is therefore
   impractical for reasoning models on this host. **Fix:** prefer *instruct/non-reasoning*
   variants for the battery, or cap generation with a reasoning-budget knob if/when the
   model+runtime support one.

### Infrastructure added this session

To make reasoning models testable, `no_reasoning` (LM Studio `include_reasoning=false`,
Ollama `think=false`) was threaded end-to-end:

- `ThemeClassificationConfig.no_reasoning` (`src/constructs/config.py`)
- forwarded into `LLMClientConfig` in `src/classification_tools/theme_llm/llm_classifier.py`
- `HarnessSpec.no_reasoning` + `--no-reasoning` battery CLI flag
  (`experiments/classification_methods/common/prompt_harness.py`,
  `07_llm_model_battery/run.py`)

This is correct and verified to flow through to the request body, and it demonstrably
works: `qwen/qwen3-8b` (8B) and `google/gemma-4-e2b` (2B) — both thinking models — completed
the full 109-item battery only because of `--no-reasoning` (0.762 and 0.688 overall, 1
abstention each). The flag does *not*, however, tame the heavier reasoning models hosted
here (Nemotron-Super, Qwen3-27B, Gemma4-12B+, Phi-4-reasoning), which reason in the content
field regardless and stall — a model/runtime-specific split (not a clean size cutoff) that
is itself a documented finding.

---

## Synthesis / recommendation

- **Production recommendation: `nvidia/nemotron-3-nano-4b` (Q4_K_M).** Of the four models
  that completed the full battery it is the clear top — 0.789 overall vs the offline floor
  of 0.45, with 0.545 on adversarial items (vs the embedding ceiling of 0.36) and zero
  abstentions. Per-stage it is strong on stages 1/2/4 (0.87–0.96) and weaker but usable on
  stage 0 Vigilance (0.55) and stage 3 Metacognition (0.65).
- **`qwen/qwen3-8b` (8B, `--no-reasoning`)** is a close runner-up at 0.762 — best of all on
  clear/subtle items (0.81 / 0.93) but weakest of the top three on adversarial items
  (0.409, just under the floor). A good multi-model-consensus partner *because* its error
  profile differs from nemotron's (nemotron is the more robust adversarial coder; qwen3-8b
  the stronger clear/subtle coder).
- **`google/gemma-4-e2b` (2B, `--no-reasoning`)** is a credible third at 0.688 (1
  abstention), also clearing the 0.45 floor — the smallest-footprint fallback that still
  works. Together qwen3-8b and gemma-4-e2b are the existence proof that `--no-reasoning`
  makes thinking models battery-able when the model cooperates with the flag.
- The **0.5B floor** (`qwen2.5-0.5b-instruct`, 0.202) confirms model capacity matters: a
  sub-1B model underperforms even the no-LLM embedding floor and abstains heavily.
- The battery is **partial by host limitation, not by design.** The two fixes above
  (single-model eviction; prefer instruct variants) are the prerequisite to extend it
  across the full qwen / gemma / gemma-QAT / nemotron / phi / minimax matrix and to run the
  multi-model consensus (Experiment B). The arm list and `--no-reasoning` plumbing are
  ready; only the host configuration blocks completion.

_All numbers are content-validity accuracy on `cv_vaamr_v1`, directly comparable to
`RESULTS_offline.md`. No PHI, participant text, utterances, or session identifiers appear
in any `experiments/` output file; all outputs contain only aggregate metrics, arm labels,
and model identifiers._
