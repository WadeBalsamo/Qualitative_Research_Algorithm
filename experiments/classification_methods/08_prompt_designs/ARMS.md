# 08_prompt_designs — Arm Reference

Prompt-design battery: each arm varies **exactly one** PromptSpec knob off the
default (or documents a historical variant).  The harness is fixed at
`HarnessSpec(n_runs=1, merge='first')` so accuracy differences isolate the
prompt design.

## Default PromptSpec values

| Field | Default |
|-------|---------|
| `context_window` | `2` |
| `randomize` | `True` |
| `zero_shot` | `False` |
| `n_exemplars` | `None` (all) |
| `include_subtle` | `True` |
| `include_adversarial` | `True` |
| `with_criteria` | `True` |
| `strict_json` | `True` |

---

## Arms

| Arm Name | Knob Changed vs Default | Git Commit | Status |
|---|---|---|---|
| `baseline_default` | *(all defaults)* | — | runnable |
| `ctx0` | `context_window=0` | — | runnable |
| `ctx2` | `context_window=2` *(same as default; explicit reference)* | — | runnable |
| `ctx6` | `context_window=6` | — | runnable |
| `randomize_off` | `randomize=False` | — | runnable |
| `zero_shot` | `zero_shot=True` | — | runnable |
| `exemplars_1` | `n_exemplars=1` | — | runnable |
| `exemplars_3` | `n_exemplars=3` | — | runnable |
| `exemplars_all` | `n_exemplars=None` *(same as default; explicit reference)* | — | runnable |
| `no_subtle` | `include_subtle=False` | — | runnable |
| `no_adversarial` | `include_adversarial=False` | — | runnable |
| `no_criteria` | `with_criteria=False` *(reserved field)* | — | runnable |
| `lenient_json` | `strict_json=False` *(reserved field)* | — | runnable |
| `hist_no_context__7dfa61c` | `context_window=0` | `7dfa61c` | runnable |
| `hist_3tier_exemplars__216bb4f` | `include_subtle=True, include_adversarial=True` | `216bb4f` | runnable |
| `hist_4stage__7dfa61c` | *(4-stage framework — see note)* | `7dfa61c` | **documented-only** |

---

## Historical Arms

### `hist_no_context__7dfa61c` (runnable)

Commit `7dfa61c` (Mar 14 2026) introduced the original zero-shot pipeline.
Segments were classified independently — no preceding-context window was built
into the prompt.  Represented here as `context_window=0`.  The 5-stage VAAMR
framework is current; only the context-window characteristic of that era is
reproduced.

### `hist_3tier_exemplars__216bb4f` (runnable)

Commit `216bb4f` (Mar 15 2026) refactored the zero-shot classifier and introduced
the 3-difficulty-tier structure: `clear` / `subtle` / `adversarial` exemplar
utterances were added to the content-validity set and the prompt.  Represented as
`include_subtle=True, include_adversarial=True` (the configuration that became
active at that commit).

### `hist_4stage__7dfa61c` (documented-only — NOT RUN)

Commit `7dfa61c` used a **4-stage** VAAMR framework: Vigilance / Avoidance /
Metacognition / Reappraisal (`NUM_STAGES = 4`).  The current framework has 5
stages — Attention Regulation was decoupled from Avoidance, and stage IDs were
reassigned (`id=2` is now Attention Regulation, not Metacognition).

The `cv_vaamr_v1` test set was built against the 5-stage schema.  Running a
4-stage prompt against it would misalign `expected_stage` integers, producing
meaningless accuracy figures.  There is no faithful way to reproduce this arm
with the current PromptSpec.

Therefore this arm is **documented here for historical completeness** but is
**skipped by `run.py`** rather than producing misleading results.  `run.py`
prints a note when it encounters this arm name.
