"""
experiments/classification_methods/03_single_model_zeroshot/run.py
------------------------------------------------------------------
Foundation method (Mar 2026, commit 7dfa61c): one model, one pass, no
consensus.  The simplest possible zero-shot LLM arm — establishes the
baseline that all other methods are compared against.

Arms
----
  single_zeroshot__<model>  — one model, one run, merge='first'

Usage
-----
  # Live run (needs LM Studio at 10.0.0.58:1234):
  python experiments/classification_methods/03_single_model_zeroshot/run.py \\
      --output-dir ./data/output

  # Dry run (no model/network — prints config and exits):
  python experiments/classification_methods/03_single_model_zeroshot/run.py --dry-run

  # Override model:
  python experiments/classification_methods/03_single_model_zeroshot/run.py \\
      --model qwen/qwen3-8b --output-dir ./data/output

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
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='03_single_model_zeroshot',
        description='Foundation single-model zero-shot VAAMR classifier (Mar 2026, 7dfa61c).',
    )
    p.add_argument(
        '--output-dir', default=None,
        help='Pipeline output dir containing qra.db.  Default: data/Meta/qra.db.',
    )
    p.add_argument(
        '--model', default='nvidia/nemotron-3-super',
        help='LM Studio model identifier.  Default: nvidia/nemotron-3-super.',
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

    prompt = prompt_harness.PromptSpec()          # all defaults
    harness = prompt_harness.HarnessSpec(
        n_runs=1,
        merge='first',
        model=args.model,
    )

    arm_name = f'single_zeroshot__{args.model.replace("/", "_")}'

    _results_dir = _HERE
    results_csv = _results_dir / 'results.csv'
    results_md  = _results_dir / 'results.md'

    if args.dry_run:
        print('=== 03_single_model_zeroshot — DRY RUN ===')
        print(f'  arm         : {arm_name}')
        print(f'  model       : {args.model}')
        print(f'  output_dir  : {args.output_dir or "(default) data/Meta/qra.db"}')
        print()
        print('  PromptSpec:')
        print(f'    context_window     = {prompt.context_window}')
        print(f'    randomize          = {prompt.randomize}')
        print(f'    zero_shot          = {prompt.zero_shot}')
        print(f'    n_exemplars        = {prompt.n_exemplars}')
        print(f'    include_subtle     = {prompt.include_subtle}')
        print(f'    include_adversarial= {prompt.include_adversarial}')
        print(f'    strict_json        = {prompt.strict_json}')
        print()
        print('  HarnessSpec:')
        print(f'    n_runs             = {harness.n_runs}')
        print(f'    merge              = {harness.merge!r}')
        print(f'    backend            = {harness.backend!r}')
        print(f'    base_url           = {harness.base_url}')
        print(f'    model              = {harness.model}')
        print()
        print('  results_csv :', results_csv)
        print('  results_md  :', results_md)
        print()
        print('  [DRY RUN] No LLM calls made.')
        return

    # -----------------------------------------------------------------------
    # Live run
    # -----------------------------------------------------------------------
    print(f'Loading CV items from {args.output_dir or "default DB"}...')
    items = data.load_cv_items(output_dir=args.output_dir)
    print(f'  {len(items)} items loaded.')

    print(f'Running arm: {arm_name}')
    preds = prompt_harness.run_llm_arm(items, prompt, harness)

    print('Scoring...')
    metrics = cv_scoring.score_arm(
        arm_name,
        preds,
        output_dir=args.output_dir,
        results_csv=results_csv,
        results_md=results_md,
    )

    print()
    print('=== Results ===')
    print(f'  arm          : {metrics["arm"]}')
    print(f'  n_items      : {metrics["n_items"]}')
    print(f'  n_abstain    : {metrics["n_abstain"]}')
    print(f'  acc_overall  : {metrics["acc_overall"]}')
    print(f'  acc_secondary: {metrics["acc_secondary"]}')
    print(f'  acc_clear    : {metrics["acc_clear"]}')
    print(f'  acc_subtle   : {metrics["acc_subtle"]}')
    print(f'  acc_adversarial: {metrics["acc_adversarial"]}')
    print(f'  results_csv  : {results_csv}')
    print(f'  results_md   : {results_md}')


if __name__ == '__main__':
    main()
