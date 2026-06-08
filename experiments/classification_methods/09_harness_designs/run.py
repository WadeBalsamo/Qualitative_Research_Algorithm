"""
experiments/classification_methods/09_harness_designs/run.py
------------------------------------------------------------
HARNESS/VOTING-DESIGN BATTERY — run all arms defined in arms.py against
the cv_vaamr_v1 content-validity test set.

The PromptSpec is FIXED at defaults for all arms.  Only the HarnessSpec
(n_runs, per_run_models, merge, secondary_weight, presence_threshold,
abstain_as_ballot) varies.  This isolates the consensus/voting design as
the sole independent variable.

Usage
-----
  # Dry-run: print arm table (no LLM calls, no model load, no network)
  python run.py --dry-run

  # Run all arms (requires LM Studio at 10.0.0.58:1234)
  python run.py --output-dir /path/to/data/output/

  # Run a specific arm (or comma-separated list)
  python run.py --arm majority_3 --output-dir /path/to/data/output/

  # Override the model pool (comma-separated; first 3 = 3-model arms,
  # first 5 = 5-model arms, first 1 repeated = single_model_3runs)
  python run.py --model-pool "nvidia/nemotron-3-super,qwen/qwen3-8b,google/gemma-2-9b,microsoft/phi-4-reasoning-plus,qwen/qwen3.6-27b" \\
      --output-dir /path/to/data/output/

NOT executed in this build; run-ready; needs live LM Studio at 10.0.0.58:1234.
"""

import argparse
import pathlib
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# sys.path bootstrap — make common/ importable from any cwd
# ---------------------------------------------------------------------------

_HERE = pathlib.Path(__file__).resolve().parent            # .../09_harness_designs/
_METHODS = _HERE.parent                                    # .../classification_methods/
_ROOT = _METHODS.parent.parent                             # repo root

for _p in (str(_ROOT / 'src'), str(_ROOT), str(_METHODS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Argparse-first: common/ is imported lazily inside functions so that
# --help and --dry-run work with zero model loads.

from arms import (                                          # noqa: E402
    HARNESS_ARMS,
    DEFAULT_MODEL_POOL_3,
    DEFAULT_MODEL_POOL_5,
    REPEATED_MODEL,
)


# ---------------------------------------------------------------------------
# Pool expansion helpers
# ---------------------------------------------------------------------------

def _expand_pool(kwargs: dict, pool: List[str]) -> dict:
    """
    Return a copy of kwargs with per_run_models resolved from pool.

    Arms store per_run_models as either:
      - A concrete list (used as-is, subject to pool-override expansion)
      - None (single_pass / single model arms)

    If the arm already has per_run_models set to a list, we replace it
    with the correct slice from the caller-supplied pool.  The slice
    length is determined by n_runs (capped at len(pool)).

    Special case — single_model_3runs: all three slots are the SAME
    model (pool[0] repeated), not distinct models.
    """
    out = dict(kwargs)
    existing = out.get('per_run_models')
    n = out.get('n_runs', 1)

    if existing is None:
        # single_pass or explicit single-model arm — leave as-is
        return out

    # Detect repeated-model arm: all entries in the stored list are identical
    if len(set(existing)) == 1 and n > 1:
        # stochastic repeat arm — use pool[0] repeated n times
        out['per_run_models'] = [pool[0]] * n
        return out

    # Multi-model arm — take first n distinct entries from pool
    distinct = list(dict.fromkeys(pool))  # deduplicate, preserve order
    out['per_run_models'] = distinct[:n]
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='run.py',
        description=(
            'Harness/voting-design battery for VAAMR zero-shot classification.\n\n'
            'All arms share a fixed PromptSpec; only HarnessSpec varies.\n'
            'Use --dry-run to inspect the arm table without any model load.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--output-dir', default=None,
        help=(
            'Pipeline output directory containing qra.db. '
            'Default: data/Meta/qra.db in repo.'
        ),
    )
    p.add_argument(
        '--model-pool', default=None,
        help=(
            'Comma-separated ordered model pool used to fill per_run_models '
            'for multi-run arms. First 3 models fill 3-model arms; first 5 '
            'fill 5-model arms; first model is repeated for single_model_3runs. '
            'Default: arms.DEFAULT_MODEL_POOL_5 (superset of DEFAULT_MODEL_POOL_3).'
        ),
    )
    p.add_argument(
        '--arm', default=None,
        help=(
            'Comma-separated arm name(s) to run. Default: all runnable arms. '
            'Documented-only arms are always skipped regardless of this flag.'
        ),
    )
    p.add_argument(
        '--base-url', default='http://10.0.0.58:1234/v1',
        help='LM Studio base URL (default: http://10.0.0.58:1234/v1).',
    )
    p.add_argument(
        '--dry-run', action='store_true',
        help=(
            'Print the arm table (name, n_runs, per_run_models, merge, key knobs, '
            'commit, runnable/documented) and exit. No model load, no network call.'
        ),
    )
    return p


# ---------------------------------------------------------------------------
# Dry-run arm table printer
# ---------------------------------------------------------------------------

def _print_dry_run_table(arms: list, pool: List[str]) -> None:
    """Print a full arm table showing resolved configuration."""
    col_name = max(len(a['name']) for a in arms) + 2

    header = (
        f"{'name':<{col_name}}  "
        f"{'n_runs':>6}  "
        f"{'merge':<16}  "
        f"{'sw':>5}  "
        f"{'pt':>5}  "
        f"{'abstain':>7}  "
        f"{'commit':<12}  "
        f"{'status':<12}  "
        f"per_run_models"
    )
    print(header)
    print('-' * (len(header) + 20))

    runnable_n = 0
    documented_n = 0

    for arm in arms:
        name = arm['name']
        is_doc = arm.get('documented_only', False)
        commit = arm.get('commit') or '—'
        status = 'documented' if is_doc else 'runnable'

        kwargs = _expand_pool(arm['harness_spec_kwargs'], pool)
        n = kwargs.get('n_runs', 1)
        merge = kwargs.get('merge', 'majority')
        sw = kwargs.get('secondary_weight', 0.6)
        pt = kwargs.get('presence_threshold', 0.5)
        abstain = kwargs.get('abstain_as_ballot', True)
        prm = kwargs.get('per_run_models')

        if prm is None:
            prm_str = f'[HarnessSpec.model default]'
        else:
            prm_str = str(prm)

        print(
            f"{name:<{col_name}}  "
            f"{n:>6}  "
            f"{merge:<16}  "
            f"{sw:>5.2f}  "
            f"{pt:>5.2f}  "
            f"{str(abstain):>7}  "
            f"{commit:<12}  "
            f"{status:<12}  "
            f"{prm_str}"
        )

        if is_doc:
            documented_n += 1
        else:
            runnable_n += 1

    print()
    print(f"Total arms   : {len(arms)}")
    print(f"  runnable   : {runnable_n}")
    print(f"  documented : {documented_n}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Resolve model pool
    # ------------------------------------------------------------------
    if args.model_pool:
        pool = [m.strip() for m in args.model_pool.split(',') if m.strip()]
        if len(pool) < 5:
            print(
                f"WARNING: --model-pool has {len(pool)} model(s); "
                f"5-model arms will use fewer models than requested.",
                file=sys.stderr,
            )
    else:
        pool = DEFAULT_MODEL_POOL_5

    # ------------------------------------------------------------------
    # Filter arms by --arm flag
    # ------------------------------------------------------------------
    requested: Optional[List[str]] = None
    if args.arm:
        requested = [a.strip() for a in args.arm.split(',') if a.strip()]

    arms_to_process = []
    for arm in HARNESS_ARMS:
        if requested is not None and arm['name'] not in requested:
            continue
        arms_to_process.append(arm)

    if not arms_to_process:
        print('No matching arms. Check --arm flag against ARMS.md.', file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # --dry-run: print table and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print('\n=== DRY RUN — harness/voting-design battery (no model load) ===\n')
        _print_dry_run_table(arms_to_process, pool)
        return

    # ------------------------------------------------------------------
    # Live run — lazy imports (no model load at argparse time)
    # ------------------------------------------------------------------
    from common import data, cv_scoring, prompt_harness     # noqa: E402

    results_dir = _HERE
    results_csv = results_dir / 'results.csv'
    results_md = results_dir / 'results.md'

    print(f'Loading CV items (testset: cv_vaamr_v1) …', file=sys.stderr)
    items = data.load_cv_items(output_dir=args.output_dir)
    print(f'  {len(items)} items loaded.', file=sys.stderr)

    # Fixed prompt for all arms
    prompt = prompt_harness.PromptSpec()   # all defaults

    results_summary = []

    for arm in arms_to_process:
        name = arm['name']
        is_doc = arm.get('documented_only', False)
        commit = arm.get('commit')

        if is_doc:
            print(
                f'\n[SKIP] {name}  (documented_only=True — '
                f'see ARMS.md for explanation)',
                file=sys.stderr,
            )
            continue

        kwargs = _expand_pool(arm['harness_spec_kwargs'], pool)
        # Inject base_url from CLI
        kwargs['base_url'] = args.base_url

        print(f'\n[RUN] {name}', file=sys.stderr)
        print(
            f'  n_runs={kwargs.get("n_runs")}  '
            f'merge={kwargs.get("merge")}  '
            f'per_run_models={kwargs.get("per_run_models")}',
            file=sys.stderr,
        )

        harness = prompt_harness.HarnessSpec(**kwargs)

        # Real LLM calls happen here — not invoked during --dry-run
        preds = prompt_harness.run_llm_arm(items, prompt, harness)

        meta = {'battery': 'harness'}
        if commit:
            meta['commit'] = commit

        metrics = cv_scoring.score_arm(
            arm_name=name,
            predictions=preds,
            output_dir=args.output_dir,
            results_csv=results_csv,
            results_md=results_md,
            meta=meta,
        )
        results_summary.append(metrics)
        print(
            f'  acc_overall={metrics.get("acc_overall"):.4f}  '
            f'acc_clear={metrics.get("acc_clear")}  '
            f'acc_subtle={metrics.get("acc_subtle")}  '
            f'acc_adversarial={metrics.get("acc_adversarial")}',
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Final sorted summary
    # ------------------------------------------------------------------
    if results_summary:
        print('\n=== Results sorted by acc_overall ===\n')
        sorted_results = sorted(
            results_summary,
            key=lambda r: (r.get('acc_overall') or 0.0),
            reverse=True,
        )
        col_w = max(len(r['arm']) for r in sorted_results) + 2
        print(
            f"{'arm':<{col_w}}  {'acc_overall':>12}  "
            f"{'acc_clear':>10}  {'acc_subtle':>11}  {'acc_adversarial':>15}"
        )
        print('-' * (col_w + 56))
        for r in sorted_results:
            print(
                f"{r['arm']:<{col_w}}  "
                f"{str(r.get('acc_overall', '')):>12}  "
                f"{str(r.get('acc_clear', '')):>10}  "
                f"{str(r.get('acc_subtle', '')):>11}  "
                f"{str(r.get('acc_adversarial', '')):>15}"
            )
        print(f'\nResults written to: {results_csv}')
        if results_md.exists():
            print(f'Markdown table:     {results_md}')


if __name__ == '__main__':
    main()
