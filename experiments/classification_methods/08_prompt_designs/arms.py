"""
experiments/classification_methods/08_prompt_designs/arms.py
-------------------------------------------------------------
Prompt-design battery arm definitions.

Each entry in PROMPT_ARMS is a dict with:
    name               : str    — short unique identifier
    prompt_spec_kwargs : dict   — keyword args passed to PromptSpec(**kwargs)
    note               : str    — human-readable description of what knob is varied
    commit             : str    — git commit hash this arm reflects (historical arms only)
    documented_only    : bool   — if True, arm is SKIPPED at runtime (see run.py)

DOCUMENTED-ONLY ARMS
---------------------
hist_4stage__7dfa61c is marked documented_only=True.

Rationale: commit 7dfa61c (Mar-2026 initial pipeline) used a 4-stage VAAMR
framework — Vigilance / Avoidance / Metacognition / Reappraisal.  The current
5-stage framework (Vigilance / Avoidance / Attention Regulation / Metacognition /
Reappraisal) decoupled Attention Regulation from Avoidance, reassigning stage IDs
so that id=2 now means Attention Regulation (not Metacognition).  The cv_vaamr_v1
test set was built against the current 5-stage definitions.

Consequence: there is NO faithful way to run a 4-stage prompt against the 5-stage
test set — the expected_stage integers would be misaligned.  Faking the comparison
would produce misleading accuracy numbers.  Therefore this arm is documented here
for historical completeness but is excluded from live runs.  The PromptSpec kwargs
shown (context_window=0) reflect the other confirmed characteristic of that era
(no context window); the 4-stage-specific knob cannot be expressed in PromptSpec.
"""

PROMPT_ARMS = [
    # ------------------------------------------------------------------
    # BASELINE
    # ------------------------------------------------------------------
    {
        'name': 'baseline_default',
        'prompt_spec_kwargs': {},
        'note': (
            'All PromptSpec defaults: context_window=2, randomize=True, '
            'zero_shot=False, n_exemplars=None, include_subtle=True, '
            'include_adversarial=True, with_criteria=True, strict_json=True.'
        ),
    },

    # ------------------------------------------------------------------
    # CONTEXT WINDOW SWEEP
    # ------------------------------------------------------------------
    {
        'name': 'ctx0',
        'prompt_spec_kwargs': {'context_window': 0},
        'note': 'No preceding context — each segment classified in isolation.',
    },
    {
        'name': 'ctx2',
        'prompt_spec_kwargs': {'context_window': 2},
        'note': 'context_window=2 (same as baseline_default; explicit reference point).',
    },
    {
        'name': 'ctx6',
        'prompt_spec_kwargs': {'context_window': 6},
        'note': 'Wider context window (6 preceding segments).',
    },

    # ------------------------------------------------------------------
    # RANDOMIZE ORDER
    # ------------------------------------------------------------------
    {
        'name': 'randomize_off',
        'prompt_spec_kwargs': {'randomize': False},
        'note': 'Present VAAMR stages in fixed order (no shuffle) — tests order-bias sensitivity.',
    },

    # ------------------------------------------------------------------
    # ZERO-SHOT (definitions only, no exemplars)
    # ------------------------------------------------------------------
    {
        'name': 'zero_shot',
        'prompt_spec_kwargs': {'zero_shot': True},
        'note': 'Definitions only — all exemplars stripped from the prompt.',
    },

    # ------------------------------------------------------------------
    # EXEMPLAR COUNT SWEEP
    # ------------------------------------------------------------------
    {
        'name': 'exemplars_1',
        'prompt_spec_kwargs': {'n_exemplars': 1},
        'note': 'At most 1 exemplar per stage.',
    },
    {
        'name': 'exemplars_3',
        'prompt_spec_kwargs': {'n_exemplars': 3},
        'note': 'At most 3 exemplars per stage.',
    },
    {
        'name': 'exemplars_all',
        'prompt_spec_kwargs': {'n_exemplars': None},
        'note': 'All available exemplars per stage (n_exemplars=None, same as baseline).',
    },

    # ------------------------------------------------------------------
    # DIFFICULTY TIER INCLUSION
    # ------------------------------------------------------------------
    {
        'name': 'no_subtle',
        'prompt_spec_kwargs': {'include_subtle': False},
        'note': 'Exclude subtle-difficulty exemplars from the prompt.',
    },
    {
        'name': 'no_adversarial',
        'prompt_spec_kwargs': {'include_adversarial': False},
        'note': 'Exclude adversarial-difficulty exemplars from the prompt.',
    },

    # ------------------------------------------------------------------
    # CRITERIA / JSON FORMAT FLAGS
    # ------------------------------------------------------------------
    {
        'name': 'no_criteria',
        'prompt_spec_kwargs': {'with_criteria': False},
        'note': (
            'with_criteria=False (reserved flag — current template always includes '
            'criteria; this arm exercises the field pathway).'
        ),
    },
    {
        'name': 'lenient_json',
        'prompt_spec_kwargs': {'strict_json': False},
        'note': (
            'strict_json=False (reserved flag — current template always requests '
            'strict JSON; this arm exercises the field pathway).'
        ),
    },

    # ------------------------------------------------------------------
    # HISTORICAL: no context window (Mar-2026 initial pipeline)
    # ------------------------------------------------------------------
    {
        'name': 'hist_no_context__7dfa61c',
        'prompt_spec_kwargs': {'context_window': 0},
        'note': (
            'Historical: Mar-2026 initial pipeline (commit 7dfa61c) classified '
            'segments independently with no preceding-context window. '
            'The 5-stage VAAMR framework is current; only the context_window=0 '
            'characteristic of that era is reproduced here.'
        ),
        'commit': '7dfa61c',
    },

    # ------------------------------------------------------------------
    # HISTORICAL: 3-difficulty-tier exemplars introduced (Mar-2026 refactor)
    # ------------------------------------------------------------------
    {
        'name': 'hist_3tier_exemplars__216bb4f',
        'prompt_spec_kwargs': {
            'include_subtle': True,
            'include_adversarial': True,
        },
        'note': (
            'Historical: commit 216bb4f introduced the 3-difficulty tier structure '
            '(clear / subtle / adversarial exemplars) into the content-validity set '
            'and prompt construction. include_subtle=True, include_adversarial=True '
            'matches the defaults enabled at that refactor point.'
        ),
        'commit': '216bb4f',
    },

    # ------------------------------------------------------------------
    # HISTORICAL: 4-stage framework — DOCUMENTED ONLY, NOT RUNNABLE
    # ------------------------------------------------------------------
    {
        'name': 'hist_4stage__7dfa61c',
        'prompt_spec_kwargs': {'context_window': 0},
        'note': (
            'Historical: commit 7dfa61c used a 4-stage VAAMR framework '
            '(Vigilance / Avoidance / Metacognition / Reappraisal — no decoupled '
            'Attention Regulation). Cannot be faithfully reproduced with the current '
            '5-stage framework: expected_stage integers in cv_vaamr_v1 are aligned '
            'to the 5-stage schema, so running a 4-stage prompt would produce '
            'misaligned comparisons. Documented here for historical completeness; '
            'SKIPPED at runtime.'
        ),
        'commit': '7dfa61c',
        'documented_only': True,
    },
]
