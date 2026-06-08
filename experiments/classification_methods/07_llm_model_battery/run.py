"""
experiments/classification_methods/07_llm_model_battery/run.py
---------------------------------------------------------------
LLM Model-Type Battery — evaluate every LM Studio CHAT model < 120B
against cv_vaamr_v1 via single-model zero-shot.

Usage
-----
  # Dry-run: print planned arm table (no network, no classification)
  python run.py --dry-run

  # List available + filtered models (hits LM Studio network)
  python run.py --list-models

  # Live run (requires LM Studio at --base-url)
  python run.py --output-dir /path/to/data/output/

  # Override model list
  python run.py --models qwen/qwen3-8b,google/gemma-2-9b --output-dir /path/to/...

See README.md for full documentation.
"""

import argparse
import re
import sys
import pathlib

# ---------------------------------------------------------------------------
# sys.path bootstrap — insert common/ parent so ``from common import …`` works
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent          # .../07_llm_model_battery/
_METHODS = _HERE.parent                                  # .../classification_methods/
_ROOT = _METHODS.parent.parent                           # repo root
for _p in (str(_ROOT / 'src'), str(_ROOT), str(_METHODS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common import data, cv_scoring, prompt_harness      # noqa: E402

# ---------------------------------------------------------------------------
# Static fallback model list
# (Excludes: text-embedding-*, qwen/qwen3-coder-480b)
# ---------------------------------------------------------------------------
STATIC_FALLBACK_MODELS: list[str] = [
    'qwen/qwen3.6-27b',
    'google/gemma-4-31b',
    'minimax-m2.7',
    'deepseek-r1-finance-reasoning-14b',
    'nvidia/nemotron-3-nano-omni',
    'deepseek-v4-flash',
    'unsloth/qwen3-coder-30b-a3b-instruct',
    'qwen2.5-0.5b-instruct',
    'nvidia/nemotron-3-nano-4b',
    'google/gemma-4-e2b',
    'gemma-4-26b-a4b-it',
    'nvidia/nemotron-3-super',
    'nvidia/nemotron-3-nano',
    'qwen/qwen3-8b',
    'business_consulting_finetune_llama_3.1_8b',
    'qwen/qwen3-next-80b',
    'microsoft/phi-4-reasoning-plus',
    'google/gemma-2-9b',
    'magnum-v4-72b',
    'qwen/qwen3-coder-30b',
]

# ---------------------------------------------------------------------------
# Size parsing helpers
# ---------------------------------------------------------------------------

# Matches a trailing size token like 480b, 235b, 120b, 80b, 72b, 30b, 27b,
# 9b, 8b, 4b, 0.5b, 3b, etc.  The token must appear at the end of the
# last path component (after the final '/') or the whole id.
_SIZE_RE = re.compile(r'(\d+(?:\.\d+)?)b(?:[^a-z]|$)', re.IGNORECASE)

_EMBED_SUBSTRINGS = ('embed',)


def _parse_size_gb(model_id: str) -> float | None:
    """
    Return the billion-parameter count parsed from model_id, or None if
    no size token is detectable.

    Parsing strategy: search the model id (everything after the last '/')
    for a pattern like '72b', '27b', '480b', '0.5b'.  Returns the first
    match (leftmost) as a float.
    """
    # Look at the last segment after the final '/' to avoid matching
    # organization names that happen to contain digits.
    tail = model_id.split('/')[-1]
    match = _SIZE_RE.search(tail)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def filter_models(ids: list[str]) -> list[str]:
    """
    Filter a list of model ids to chat models < 120B.

    Rules (in order):
      1. Drop any id containing 'embed' (case-insensitive) — embedding models.
      2. Parse trailing size token (e.g. '480b', '72b', '0.5b').
         - If detectable AND >= 120: drop.
         - If not detectable: keep but log a warning (assumed < 120B).
      3. All remaining ids are kept.

    Returns a new list of kept model ids (order preserved).
    """
    kept = []
    for mid in ids:
        low = mid.lower()

        # Rule 1: drop embeddings
        if any(sub in low for sub in _EMBED_SUBSTRINGS):
            print(f"  [filter] DROP (embedding): {mid}", file=sys.stderr)
            continue

        # Rule 2: size filter
        size = _parse_size_gb(mid)
        if size is not None:
            if size >= 120:
                print(f"  [filter] DROP (>= 120B, {size}B): {mid}", file=sys.stderr)
                continue
            # else: size is detectable and < 120B — keep silently
        else:
            print(f"  [filter] KEEP (no size detected, assumed <120B): {mid}", file=sys.stderr)

        kept.append(mid)

    return kept


# ---------------------------------------------------------------------------
# Model discovery (real run only — hits network)
# ---------------------------------------------------------------------------

def discover_models(base_url: str) -> list[str]:
    """
    Query LM Studio's /v1/models endpoint and return all model ids.

    Lazy-imports urllib.request and json so this module stays import-clean.
    Only called in live mode or --list-models; never invoked during --dry-run.
    """
    import urllib.request
    import json

    url = base_url.rstrip('/') + '/models'
    req = urllib.request.Request(url, headers={'Accept': 'application/json'})
    with urllib.request.urlopen(req, timeout=10) as resp:
        payload = json.loads(resp.read().decode('utf-8'))

    # OpenAI-compatible response: {"data": [{"id": "..."}, ...]}
    models = payload.get('data', [])
    return [m['id'] for m in models if 'id' in m]


# ---------------------------------------------------------------------------
# Arm table helpers
# ---------------------------------------------------------------------------

def _print_arm_table(model_ids: list[str], mode: str = 'planned') -> None:
    """Print a one-row-per-model arm table to stdout."""
    col_w = max(len(m) for m in model_ids) + 2 if model_ids else 40
    header = f"{'model':<{col_w}}  {'status'}"
    print(header)
    print('-' * len(header))
    for mid in model_ids:
        print(f"{mid:<{col_w}}  {mode}")
    print()
    print(f"Total arms: {len(model_ids)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='run.py',
        description=(
            'LLM Model-Type Battery: zero-shot VAAMR classification on cv_vaamr_v1 '
            'across all chat models < 120B available in LM Studio.\n\n'
            'Default base URL: http://10.0.0.58:1234/v1\n'
            'Fallback (local):  http://127.0.0.1:1234/v1'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--base-url',
        default='http://10.0.0.58:1234/v1',
        help=(
            'LM Studio base URL (default: http://10.0.0.58:1234/v1). '
            'Use http://127.0.0.1:1234/v1 if running on the same machine.'
        ),
    )
    p.add_argument(
        '--output-dir',
        default=None,
        help='Project output directory containing qra.db (default: data/Meta/qra.db in repo).',
    )
    p.add_argument(
        '--list-models',
        action='store_true',
        help='Query LM Studio, print discovered + filtered models, then exit. Hits network.',
    )
    p.add_argument(
        '--dry-run',
        action='store_true',
        help=(
            'Use STATIC_FALLBACK_MODELS; print planned arm table. '
            'No network calls, no classification.'
        ),
    )
    p.add_argument(
        '--models',
        default=None,
        help='Comma-separated model ids to use instead of discovered/fallback list.',
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    out_dir = _HERE
    results_csv = out_dir / 'results.csv'
    results_md = out_dir / 'results.md'

    # ------------------------------------------------------------------
    # --list-models: discover, filter, print, exit
    # ------------------------------------------------------------------
    if args.list_models:
        print(f"Querying {args.base_url}/models …", file=sys.stderr)
        raw = discover_models(args.base_url)
        print(f"Discovered {len(raw)} model(s). Filtering …", file=sys.stderr)
        kept = filter_models(raw)
        print("\nKept models after filtering:")
        for mid in kept:
            print(f"  {mid}")
        print(f"\nTotal kept: {len(kept)}")
        return

    # ------------------------------------------------------------------
    # Resolve model list
    # ------------------------------------------------------------------
    if args.models:
        model_ids_raw = [m.strip() for m in args.models.split(',') if m.strip()]
        print("Using --models override; applying filter …", file=sys.stderr)
        model_ids = filter_models(model_ids_raw)
    elif args.dry_run:
        # Use static fallback, still apply filter (for consistency / testing)
        model_ids = filter_models(STATIC_FALLBACK_MODELS)
    else:
        # Live run: discover from LM Studio
        print(f"Discovering models from {args.base_url} …", file=sys.stderr)
        raw = discover_models(args.base_url)
        model_ids = filter_models(raw)

    if not model_ids:
        print("No models remaining after filtering. Exiting.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------
    # --dry-run: print planned arm table and exit
    # ------------------------------------------------------------------
    if args.dry_run:
        print("\n=== DRY RUN — planned arms (no classification will run) ===\n")
        _print_arm_table(model_ids, mode='planned')
        return

    # ------------------------------------------------------------------
    # Live run: load items, iterate models, score each arm
    # ------------------------------------------------------------------
    print(f"Loading CV items (testset: cv_vaamr_v1) …", file=sys.stderr)
    items = data.load_cv_items(output_dir=args.output_dir)
    print(f"  {len(items)} items loaded.", file=sys.stderr)

    prompt_spec = prompt_harness.PromptSpec(
        context_window=2,
        randomize=True,
        zero_shot=False,
        n_exemplars=None,
        include_subtle=True,
        include_adversarial=True,
    )

    results_summary: list[dict] = []
    for i, model_id in enumerate(model_ids, 1):
        print(f"\n[{i}/{len(model_ids)}] Running arm: {model_id}", file=sys.stderr)
        harness_spec = prompt_harness.HarnessSpec(
            n_runs=1,
            merge='first',
            model=model_id,
            base_url=args.base_url,
            backend='lmstudio',
        )

        # Real LLM calls happen here — not invoked during --dry-run
        preds = prompt_harness.run_llm_arm(items, prompt_spec, harness_spec)

        metrics = cv_scoring.score_arm(
            arm_name=model_id,
            predictions=preds,
            output_dir=args.output_dir,
            results_csv=results_csv,
            results_md=results_md,
            meta={'battery': 'llm_models'},
        )
        results_summary.append(metrics)
        print(
            f"  acc_overall={metrics.get('acc_overall'):.4f}  "
            f"acc_clear={metrics.get('acc_clear')}  "
            f"acc_subtle={metrics.get('acc_subtle')}  "
            f"acc_adversarial={metrics.get('acc_adversarial')}",
            file=sys.stderr,
        )

    # ------------------------------------------------------------------
    # Final sorted accuracy table
    # ------------------------------------------------------------------
    print("\n=== Results sorted by acc_overall ===\n")
    sorted_results = sorted(
        results_summary,
        key=lambda r: (r.get('acc_overall') or 0.0),
        reverse=True,
    )
    col_w = max(len(r['arm']) for r in sorted_results) + 2 if sorted_results else 40
    print(f"{'model':<{col_w}}  {'acc_overall':>12}  {'acc_clear':>10}  {'acc_subtle':>11}  {'acc_adversarial':>15}")
    print('-' * (col_w + 56))
    for r in sorted_results:
        print(
            f"{r['arm']:<{col_w}}  "
            f"{str(r.get('acc_overall', '')):>12}  "
            f"{str(r.get('acc_clear', '')):>10}  "
            f"{str(r.get('acc_subtle', '')):>11}  "
            f"{str(r.get('acc_adversarial', '')):>15}"
        )
    print(f"\nResults written to: {results_csv}")
    if results_md.exists():
        print(f"Markdown table:     {results_md}")


if __name__ == '__main__':
    main()
