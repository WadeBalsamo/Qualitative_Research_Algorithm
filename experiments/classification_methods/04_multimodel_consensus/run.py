"""
experiments/classification_methods/04_multimodel_consensus/run.py
-----------------------------------------------------------------
Method-of-record (Apr 2026, commit 56dd301): multi-model majority vote.
This is the CURRENT production method; all other classification-methods
experiments are compared against this arm.

Arms
----
  consensus_3model  — 3 runs, 3 distinct models, majority vote  (default)
  consensus_5model  — 5 runs, 5 distinct models, majority vote

Usage
-----
  # Live run (needs LM Studio at 10.0.0.58:1234):
  python experiments/classification_methods/04_multimodel_consensus/run.py \\
      --output-dir ./data/output

  # Override models for the 3-model arm (comma-separated):
  python experiments/classification_methods/04_multimodel_consensus/run.py \\
      --models nvidia/nemotron-3-super,qwen/qwen3-8b,mistral/ministral-8b \\
      --output-dir ./data/output

  # Dry run (no model/network — prints config and exits):
  python experiments/classification_methods/04_multimodel_consensus/run.py --dry-run

NOT executed in this build; run-ready; needs live LM Studio backend at 10.0.0.58:1234.
"""

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# sys.path: make common/ importable from any cwd
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent
_METHODS_ROOT = _HERE.parent
if str(_METHODS_ROOT) not in sys.path:
    sys.path.insert(0, str(_METHODS_ROOT))

from common import data, cv_scoring, prompt_harness  # noqa: E402


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_3_MODELS = [
    'nvidia/nemotron-3-super',
    'qwen/qwen3-8b',
    'google/gemma-2-9b',
]

_DEFAULT_5_MODELS = [
    'nvidia/nemotron-3-super',
    'qwen/qwen3-8b',
    'google/gemma-2-9b',
    'mistral/ministral-8b',
    'meta-llama/llama-3.1-8b-instruct',
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='04_multimodel_consensus',
        description=(
            'METHOD-OF-RECORD: multi-model majority-vote VAAMR classifier '
            '(Apr 2026, 56dd301).  Reference for all other experiments.'
        ),
    )
    p.add_argument(
        '--output-dir', default=None,
        help='Pipeline output dir containing qra.db.  Default: data/Meta/qra.db.',
    )
    p.add_argument(
        '--models', default=None,
        help=(
            'Comma-separated list of LM Studio model IDs for the 3-model arm. '
            'Overrides the default three models.  Must supply exactly 3 models '
            '(or omit to use defaults).'
        ),
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Print config and exit — no LLM calls, no model loads, no network.',
    )
    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    # Override 3-model list if --models supplied
    models_3 = _DEFAULT_3_MODELS
    if args.models:
        models_3 = [m.strip() for m in args.models.split(',') if m.strip()]

    prompt = prompt_harness.PromptSpec()          # all defaults

    harness_3 = prompt_harness.HarnessSpec(
        n_runs=3,
        per_run_models=models_3,
        merge='majority',
        model=models_3[0],
    )
    harness_5 = prompt_harness.HarnessSpec(
        n_runs=5,
        per_run_models=_DEFAULT_5_MODELS,
        merge='majority',
        model=_DEFAULT_5_MODELS[0],
    )

    _results_dir = _HERE
    results_csv = _results_dir / 'results.csv'
    results_md  = _results_dir / 'results.md'

    if args.dry_run:
        print('=== 04_multimodel_consensus — DRY RUN ===')
        print()
        print('  ARM: consensus_3model')
        print(f'    n_runs          = {harness_3.n_runs}')
        print(f'    merge           = {harness_3.merge!r}')
        print(f'    per_run_models  = {harness_3.per_run_models}')
        print()
        print('  ARM: consensus_5model')
        print(f'    n_runs          = {harness_5.n_runs}')
        print(f'    merge           = {harness_5.merge!r}')
        print(f'    per_run_models  = {harness_5.per_run_models}')
        print()
        print('  PromptSpec (shared):')
        print(f'    context_window     = {prompt.context_window}')
        print(f'    randomize          = {prompt.randomize}')
        print(f'    zero_shot          = {prompt.zero_shot}')
        print(f'    n_exemplars        = {prompt.n_exemplars}')
        print(f'    include_subtle     = {prompt.include_subtle}')
        print(f'    include_adversarial= {prompt.include_adversarial}')
        print(f'    strict_json        = {prompt.strict_json}')
        print()
        print(f'  output_dir  : {args.output_dir or "(default) data/Meta/qra.db"}')
        print(f'  results_csv : {results_csv}')
        print(f'  results_md  : {results_md}')
        print()
        print('  [DRY RUN] No LLM calls made.')
        return

    # -----------------------------------------------------------------------
    # Live run
    # -----------------------------------------------------------------------
    print(f'Loading CV items from {args.output_dir or "default DB"}...')
    items = data.load_cv_items(output_dir=args.output_dir)
    print(f'  {len(items)} items loaded.')

    # --- consensus_3model arm ---
    print('Running arm: consensus_3model')
    preds_3 = prompt_harness.run_llm_arm(items, prompt, harness_3)
    metrics_3 = cv_scoring.score_arm(
        'consensus_3model',
        preds_3,
        output_dir=args.output_dir,
        results_csv=results_csv,
        results_md=results_md,
        meta={'per_run_models': str(models_3)},
    )

    # --- consensus_5model arm ---
    print('Running arm: consensus_5model')
    preds_5 = prompt_harness.run_llm_arm(items, prompt, harness_5)
    metrics_5 = cv_scoring.score_arm(
        'consensus_5model',
        preds_5,
        output_dir=args.output_dir,
        results_csv=results_csv,
        results_md=results_md,
        meta={'per_run_models': str(_DEFAULT_5_MODELS)},
    )

    print()
    print('=== Results ===')
    for metrics, arm_label in [(metrics_3, 'consensus_3model'), (metrics_5, 'consensus_5model')]:
        print(f'  [{arm_label}]')
        print(f'    n_items      : {metrics["n_items"]}')
        print(f'    n_abstain    : {metrics["n_abstain"]}')
        print(f'    acc_overall  : {metrics["acc_overall"]}')
        print(f'    acc_secondary: {metrics["acc_secondary"]}')
        print(f'    acc_clear    : {metrics["acc_clear"]}')
        print(f'    acc_subtle   : {metrics["acc_subtle"]}')
        print(f'    acc_adversarial: {metrics["acc_adversarial"]}')
    print(f'  results_csv  : {results_csv}')
    print(f'  results_md   : {results_md}')


if __name__ == '__main__':
    main()
