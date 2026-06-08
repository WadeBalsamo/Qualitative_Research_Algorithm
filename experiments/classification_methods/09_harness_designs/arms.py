"""
experiments/classification_methods/09_harness_designs/arms.py
-------------------------------------------------------------
Arm definitions for the HARNESS/VOTING-DESIGN battery.

Every arm holds a fixed PromptSpec (defaults) and varies ONLY the
HarnessSpec (consensus/voting design).  This isolates harness differences
as the sole independent variable across the battery.

Each entry in HARNESS_ARMS is a dict:
    name              str   — unique arm identifier
    harness_spec_kwargs dict — passed as **kwargs to HarnessSpec(...)
    note              str   — human-readable description
    commit            str|None — git commit if historically anchored
    documented_only   bool  — if True, run.py prints a note and skips this arm

MODEL POOLS
-----------
DEFAULT_MODEL_POOL_3: the 3 models used in 3-run multi-model arms.
DEFAULT_MODEL_POOL_5: the 5 models used in 5-run multi-model arms.
REPEATED_MODEL:       a single model repeated for the stochastic-only arms.

These can be overridden at runtime via --model-pool.
"""

# ---------------------------------------------------------------------------
# Default model pools
# ---------------------------------------------------------------------------

DEFAULT_MODEL_POOL_3 = [
    'nvidia/nemotron-3-super',
    'qwen/qwen3-8b',
    'google/gemma-2-9b',
]

DEFAULT_MODEL_POOL_5 = [
    'nvidia/nemotron-3-super',
    'qwen/qwen3-8b',
    'google/gemma-2-9b',
    'microsoft/phi-4-reasoning-plus',
    'qwen/qwen3.6-27b',
]

# Single model repeated for stochastic-multi-run experiments.
REPEATED_MODEL = 'nvidia/nemotron-3-super'


# ---------------------------------------------------------------------------
# Arm definitions
# ---------------------------------------------------------------------------

HARNESS_ARMS = [
    # ------------------------------------------------------------------
    # Single-pass baseline — equals the single-model baseline from
    # experiments/07_llm_model_battery
    # ------------------------------------------------------------------
    {
        'name': 'single_pass',
        'harness_spec_kwargs': {
            'n_runs': 1,
            'per_run_models': None,   # uses HarnessSpec.model default
            'merge': 'first',
        },
        'note': (
            'n_runs=1, merge=first.  Equivalent to a single-model zero-shot '
            'baseline; no consensus, no voting.  Reference floor for all '
            'multi-run arms.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Method-of-record (Apr 2026, introduced in commit 56dd301):
    # 3-model majority vote
    # ------------------------------------------------------------------
    {
        'name': 'majority_3',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
        },
        'note': (
            'n_runs=3, 3 distinct models, merge=majority.  This is the '
            'METHOD-OF-RECORD introduced in commit 56dd301 (Apr 2026: '
            'vote_single_label majority + agreement tiers).  Reference '
            'arm for the battery.'
        ),
        'commit': '56dd301',
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # 5-model majority — does adding 2 more models improve over 3?
    # ------------------------------------------------------------------
    {
        'name': 'majority_5',
        'harness_spec_kwargs': {
            'n_runs': 5,
            'per_run_models': DEFAULT_MODEL_POOL_5,
            'merge': 'majority',
        },
        'note': (
            'n_runs=5, 5 distinct models, merge=majority.  Tests whether '
            'expanding the panel from 3 to 5 diverse models improves '
            'accuracy over majority_3.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Confidence-weighted merge (3 models)
    # ------------------------------------------------------------------
    {
        'name': 'confidence_weighted_3',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'confidence',
        },
        'note': (
            'n_runs=3, 3 models, merge=confidence.  Uses vote_single_label '
            'with confidence-based tiebreaking active.  Compares to '
            'majority_3 to assess whether confidence scores add signal.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Triplicate-flag design — the original Mar-2026 design
    # (historical, now superseded by unified majority_vote)
    # ------------------------------------------------------------------
    {
        'name': 'triplicate_flag_3',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'triplicate_flag',
        },
        'note': (
            'n_runs=3, 3 models, merge=triplicate_flag.  Sets needs_review=True '
            'when all 3 raters diverge (agreement_level==split, n_ballots==3).  '
            'This was the original Mar-2026 production design (commit 7dfa61c) '
            'before vote_single_label majority was introduced.  Useful to measure '
            'the review-flag rate vs majority_3 accuracy.'
        ),
        'commit': '7dfa61c',
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Secondary-weight ablations (3-model majority baseline)
    # ------------------------------------------------------------------
    {
        'name': 'secondary_w_low',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
            'secondary_weight': 0.3,
        },
        'note': (
            'secondary_weight=0.3 vs default 0.6.  Tests whether down-weighting '
            'secondary-stage evidence improves primary classification accuracy.'
        ),
        'commit': None,
        'documented_only': False,
    },
    {
        'name': 'secondary_w_high',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
            'secondary_weight': 0.9,
        },
        'note': (
            'secondary_weight=0.9 vs default 0.6.  Tests whether up-weighting '
            'secondary-stage evidence (nearly equal to primary) shifts '
            'consensus quality.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Presence-threshold ablations (3-model majority baseline)
    # ------------------------------------------------------------------
    {
        'name': 'presence_low',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
            'presence_threshold': 0.3,
        },
        'note': (
            'presence_threshold=0.3 vs default 0.5.  Lower threshold means '
            'secondary stages are surfaced with less pooled evidence.'
        ),
        'commit': None,
        'documented_only': False,
    },
    {
        'name': 'presence_high',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
            'presence_threshold': 0.7,
        },
        'note': (
            'presence_threshold=0.7 vs default 0.5.  Higher threshold suppresses '
            'weak secondary signals; tests if fewer false-positive secondaries '
            'improve overall accuracy.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Abstain-ballot ablations
    # ------------------------------------------------------------------
    {
        'name': 'abstain_as_ballot',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
            'abstain_as_ballot': True,
        },
        'note': (
            'abstain_as_ballot=True (default).  ABSTAIN votes count as a ballot '
            'in the denominator, reducing the majority fraction.  Baseline '
            'for the abstain ablation pair.'
        ),
        'commit': None,
        'documented_only': False,
    },
    {
        'name': 'abstain_excluded',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
            'abstain_as_ballot': False,
        },
        'note': (
            'abstain_as_ballot=False.  ABSTAIN votes are dropped before '
            'majority aggregation; the denominator only counts coded raters.  '
            'Compares to abstain_as_ballot to see if abstention dilutes '
            'consensus on CV items.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # Single-model 3-run: stochastic diversity vs true multi-model IRR
    # ------------------------------------------------------------------
    {
        'name': 'single_model_3runs',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': [REPEATED_MODEL, REPEATED_MODEL, REPEATED_MODEL],
            'merge': 'majority',
        },
        'note': (
            'n_runs=3 with ONE model repeated (nvidia/nemotron-3-super x3), '
            'merge=majority.  Isolates stochastic run-to-run variance from '
            'true inter-model diversity.  Expected to be worse than majority_3 '
            'if model diversity adds independent signal beyond temperature noise.'
        ),
        'commit': None,
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # HISTORICAL: original triplicate-and-flag consistency counter
    # (commit 7dfa61c, Mar 2026)
    # ------------------------------------------------------------------
    {
        'name': 'hist_triplicate_and_flag__7dfa61c',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'triplicate_flag',
        },
        'note': (
            'HISTORICAL — commit 7dfa61c (Mar 2026, "Replace mentalbert_sentence_aqua '
            'with vamr_labeling zero-shot pipeline").  Original production design: '
            '_compute_run_consistency() used a custom Counter-based consistency '
            'score (majority_count / n_runs) rather than vote_single_label agreement '
            'tiers.  Segments with consistency < n_runs were flagged for human review.  '
            'Captured here as merge=triplicate_flag which preserves the flag-on-split '
            'behavior; the scoring/agreement-tier logic now routes through '
            'vote_single_label (introduced in 56dd301).'
        ),
        'commit': '7dfa61c',
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # HISTORICAL: introduction of vote_single_label majority + agreement tiers
    # (commit 56dd301, Apr 2026)
    # ------------------------------------------------------------------
    {
        'name': 'hist_unified_majority__56dd301',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
        },
        'note': (
            'HISTORICAL — commit 56dd301 (Apr 2026, "vamr -> vammr; decoupled '
            'avoidance and attention regulation").  Introduced vote_single_label '
            'as the canonical aggregator, replacing the custom consistency counter.  '
            'Added secondary-stage evidence pooling (_evidence_secondary), '
            'secondary_weight and presence_threshold parameters to vote_single_label.  '
            'This is the METHOD-OF-RECORD and is functionally identical to majority_3.'
        ),
        'commit': '56dd301',
        'documented_only': False,
    },

    # ------------------------------------------------------------------
    # HISTORICAL: model-first sweep execution order change
    # (commit c6a724f, Jun 2026) — DOCUMENTED ONLY, not runnable
    # ------------------------------------------------------------------
    {
        'name': 'hist_model_first_sweep__c6a724f',
        'harness_spec_kwargs': {
            'n_runs': 3,
            'per_run_models': DEFAULT_MODEL_POOL_3,
            'merge': 'majority',
        },
        'note': (
            'DOCUMENTED ONLY — commit c6a724f (Jun 2026, "refactor: restructure '
            'project into src/ layout").  Changed the execution order from '
            'segment-first sweep (all models for segment i before moving to i+1) '
            'to model-first sweep (all segments for model j before switching to '
            'j+1).  This change affects RUNTIME and CHECKPOINTING GRANULARITY only '
            '-- it does NOT change the final consensus score for any item.  '
            'Scores from this arm are therefore IDENTICAL to hist_unified_majority__56dd301 '
            'and majority_3.  Included for documentation completeness; skipped by run.py.'
        ),
        'commit': 'c6a724f',
        'documented_only': True,
    },
]
