"""
experiments/classification_methods/08_prompt_designs/run.py
------------------------------------------------------------
Prompt-design battery: score many PromptSpec variants against cv_vaamr_v1 with
a FIXED HarnessSpec(n_runs=1, merge='first') so that differences in accuracy
isolate the prompt knob rather than consensus strategy or model choice.

Usage
-----
  # Dry run — print arm table, no model load, no network:
  python experiments/classification_methods/08_prompt_designs/run.py --dry-run

  # Live run (needs LM Studio):
  python experiments/classification_methods/08_prompt_designs/run.py \\
      --output-dir ./data/output

  # Run a single arm:
  python experiments/classification_methods/08_prompt_designs/run.py \\
      --output-dir ./data/output --arm ctx0

  # Override model:
  python experiments/classification_methods/08_prompt_designs/run.py \\
      --output-dir ./data/output --model qwen/qwen3-8b

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

# arms.py is a plain data module (no heavy deps) — safe to import at top level.
from arms import PROMPT_ARMS  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='08_prompt_designs',
        description=(
            'Prompt-design battery: many PromptSpec arms, fixed HarnessSpec, '
            'scored against cv_vaamr_v1.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  --dry-run                         Print arm table, no model load.\n'
            '  --output-dir ./data/output        Run all arms (live LM Studio).\n'
            '  --arm ctx0                        Run a single named arm.\n'
            '  --model qwen/qwen3-8b             Override the model.\n'
        ),
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
        '--arm', default=None,
        help='Run only the named arm (default: run all runnable arms).',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help='Print full arm table and exit — no LLM calls, no model load, no network.',
    )
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_runnable(arm: dict) -> bool:
    """Return True if the arm should be executed (not documented_only)."""
    return not arm.get('documented_only', False)


def _knob_diff(arm: dict) -> str:
    """Summarize the one knob changed from defaults (or 'all defaults')."""
    kwargs = arm.get('prompt_spec_kwargs', {})
    if not kwargs:
        return 'all defaults'
    return ', '.join(f'{k}={v!r}' for k, v in kwargs.items())


def _print_dry_run_table(arms: list, selected_arm: str | None) -> None:
    """Print a formatted arm summary table (--dry-run mode)."""
    print('=' * 80)
    print('08_prompt_designs — DRY RUN: prompt-design battery arm table')
    print('=' * 80)
    print()

    col_w = {
        'name': 36,
        'knob_diff': 48,
        'commit': 9,
        'status': 15,
    }
    hdr = (
        f"{'ARM NAME':<{col_w['name']}} "
        f"{'KNOB DIFF FROM DEFAULT':<{col_w['knob_diff']}} "
        f"{'COMMIT':<{col_w['commit']}} "
        f"STATUS"
    )
    print(hdr)
    print('-' * (sum(col_w.values()) + 3))

    for arm in arms:
        if selected_arm and arm['name'] != selected_arm:
            continue
        name = arm['name']
        knob = _knob_diff(arm)
        commit = arm.get('commit', '—')
        runnable = _is_runnable(arm)
        status = 'runnable' if runnable else 'documented-only (SKIP)'

        # Wrap knob text at col width
        knob_lines = []
        while len(knob) > col_w['knob_diff']:
            split_at = knob.rfind(',', 0, col_w['knob_diff'])
            if split_at == -1:
                split_at = col_w['knob_diff']
            knob_lines.append(knob[:split_at])
            knob = knob[split_at:].lstrip(', ')
        knob_lines.append(knob)

        first = True
        for kline in knob_lines:
            if first:
                row = (
                    f"{name:<{col_w['name']}} "
                    f"{kline:<{col_w['knob_diff']}} "
                    f"{commit:<{col_w['commit']}} "
                    f"{status}"
                )
                first = False
            else:
                row = (
                    f"{'':<{col_w['name']}} "
                    f"{kline:<{col_w['knob_diff']}} "
                    f"{'':<{col_w['commit']}} "
                    f"{''}"
                )
            print(row)

    print()
    runnable_count = sum(1 for a in arms if _is_runnable(a))
    doc_count = len(arms) - runnable_count
    if selected_arm:
        print(f'(Showing arm: {selected_arm!r})')
    else:
        print(f'Total arms: {len(arms)}  |  Runnable: {runnable_count}  |  Documented-only: {doc_count}')
    print()
    print('Fixed harness: HarnessSpec(n_runs=1, merge="first").')
    print('[DRY RUN] No LLM calls made.')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _build_parser().parse_args()

    results_csv = _HERE / 'results.csv'
    results_md  = _HERE / 'results.md'

    if args.dry_run:
        _print_dry_run_table(PROMPT_ARMS, args.arm)
        return

    # Heavy imports are deferred until a live run is actually requested.
    from common import data, cv_scoring, prompt_harness  # noqa: F401

    print(f'Loading CV items from {args.output_dir or "default DB (data/Meta/qra.db)"}...')
    items = data.load_cv_items(output_dir=args.output_dir)
    print(f'  {len(items)} items loaded.')
    print()

    fixed_harness = prompt_harness.HarnessSpec(
        n_runs=1,
        merge='first',
        model=args.model,
    )

    arms_to_run = [
        arm for arm in PROMPT_ARMS
        if _is_runnable(arm) and (args.arm is None or arm['name'] == args.arm)
    ]

    if not arms_to_run:
        if args.arm:
            # Check if the named arm is documented-only.
            matches = [a for a in PROMPT_ARMS if a['name'] == args.arm]
            if matches and not _is_runnable(matches[0]):
                print(
                    f'NOTE: arm {args.arm!r} is documented-only and cannot be run.\n'
                    f'  {matches[0]["note"]}'
                )
            else:
                print(f'ERROR: arm {args.arm!r} not found in PROMPT_ARMS.')
            sys.exit(1)
        print('No runnable arms found.')
        sys.exit(0)

    # Warn about skipped documented-only arms (only when running all).
    if args.arm is None:
        skipped = [a for a in PROMPT_ARMS if not _is_runnable(a)]
        if skipped:
            print(f'Skipping {len(skipped)} documented-only arm(s):')
            for arm in skipped:
                print(f'  [{arm["name"]}] {arm["note"][:100]}...')
            print()

    total = len(arms_to_run)
    for idx, arm in enumerate(arms_to_run, 1):
        name = arm['name']
        kwargs = arm.get('prompt_spec_kwargs', {})
        commit = arm.get('commit')
        meta = {'battery': 'prompt'}
        if commit:
            meta['commit'] = commit

        print(f'[{idx}/{total}] Running arm: {name}')
        print(f'  knob : {_knob_diff(arm)}')
        if commit:
            print(f'  commit: {commit}')

        prompt_spec = prompt_harness.PromptSpec(**kwargs)
        preds = prompt_harness.run_llm_arm(items, prompt_spec, fixed_harness)

        metrics = cv_scoring.score_arm(
            name,
            preds,
            output_dir=args.output_dir,
            results_csv=results_csv,
            results_md=results_md,
            meta=meta,
        )

        print(
            f'  acc_overall={metrics["acc_overall"]}  '
            f'acc_clear={metrics["acc_clear"]}  '
            f'acc_subtle={metrics["acc_subtle"]}  '
            f'acc_adversarial={metrics["acc_adversarial"]}  '
            f'n_abstain={metrics["n_abstain"]}'
        )
        print()

    print(f'All arms complete.  Results: {results_csv}')
    if results_md.exists():
        print(f'Markdown summary: {results_md}')


if __name__ == '__main__':
    main()
