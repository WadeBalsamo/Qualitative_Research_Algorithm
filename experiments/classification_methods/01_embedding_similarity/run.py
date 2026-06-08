"""
experiments/classification_methods/01_embedding_similarity/run.py
-----------------------------------------------------------------
Pure embedding-similarity VAAMR classifier.

For each arm, each CV item text and each VAAMR theme anchor text are
encoded with all-MiniLM-L6-v2 (or --model override).  Cosine similarity
argmax gives the primary prediction; 2nd-argmax gives the secondary.
Five arms differ in what text represents each theme:

  def_only           theme.definition
  def_exemplars      definition + exemplar_utterances joined
  exemplars_only     exemplar_utterances joined
  def_criteria       definition + distinguishing_criteria
  def_exemplars_qprefix  same as def_exemplars but with asymmetric
                         query/passage prefix encoding

Results are appended to 01_embedding_similarity/results.csv
(and optionally results.md) via cv_scoring.score_arm.

NOT executed in this build; run-ready.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# ---------------------------------------------------------------------------
# sys.path bootstrap so `from common import ...` resolves when run as a script
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent          # .../01_embedding_similarity/
_METHODS = _HERE.parent                                  # .../classification_methods/
_ROOT = _METHODS.parent.parent                           # repo root
for _p in (str(_ROOT / 'src'), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Re-root for relative imports from the methods package
if str(_METHODS) not in sys.path:
    sys.path.insert(0, str(_METHODS))

from common import data as _data          # noqa: E402
from common import cv_scoring             # noqa: E402

# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------

# Each entry: (arm_name, description, anchor_builder_key)
# anchor_builder_key is resolved in _build_anchor_text.
_ARMS = [
    ('def_only',              'theme.definition'),
    ('def_exemplars',         'definition + exemplar_utterances joined'),
    ('exemplars_only',        'exemplar_utterances joined'),
    ('def_criteria',          'definition + distinguishing_criteria'),
    ('def_exemplars_qprefix', 'def_exemplars with asymmetric query/passage prefix'),
]

ARM_NAMES = [a for a, _ in _ARMS]


def _build_anchor_text(theme, arm: str) -> str:
    """Return the anchor text for one theme under the given arm."""
    definition = (theme.definition or '').strip()
    distinguishing = (theme.distinguishing_criteria or '').strip()
    exemplars = ' | '.join(e.strip() for e in (theme.exemplar_utterances or []) if e.strip())

    if arm == 'def_only':
        return definition
    if arm == 'def_exemplars':
        parts = [definition]
        if exemplars:
            parts.append(exemplars)
        return ' '.join(parts)
    if arm == 'exemplars_only':
        return exemplars or definition  # fall back to definition if no exemplars
    if arm == 'def_criteria':
        parts = [definition]
        if distinguishing:
            parts.append(distinguishing)
        return ' '.join(parts)
    if arm == 'def_exemplars_qprefix':
        # anchor text same as def_exemplars; prefix applied at encode time
        parts = [definition]
        if exemplars:
            parts.append(exemplars)
        return ' '.join(parts)
    raise ValueError(f"Unknown arm: {arm!r}")


# ---------------------------------------------------------------------------
# Embedding + similarity (lazy imports — only reached in the real run path)
# ---------------------------------------------------------------------------

def _encode(model, texts: list[str], *, is_query: bool = False) -> 'np.ndarray':
    """Encode texts.  For asymmetric arms, queries get a 'query: ' prefix."""
    # numpy imported lazily inside this function (always available, but keep
    # the pattern consistent with the sentinel guard below)
    import numpy as np  # noqa: F401
    if is_query:
        texts = [f"query: {t}" for t in texts]
    return model.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _cosine_sim(a: 'np.ndarray', b: 'np.ndarray') -> 'np.ndarray':
    """[n, d] x [m, d] -> [n, m] cosine similarity (vectors are already unit-norm)."""
    import numpy as np
    return np.dot(a, b.T)


def run_arm(arm: str, items, framework, model_name: str,
            results_csv: pathlib.Path, results_md: pathlib.Path | None) -> dict:
    """Run one arm end-to-end: embed anchors + items, argmax, score."""
    # Heavy imports inside the function — not at module top level.
    from sentence_transformers import SentenceTransformer
    import numpy as np  # noqa: F401

    model = SentenceTransformer(model_name)

    use_qprefix = arm == 'def_exemplars_qprefix'
    anchor_texts = [_build_anchor_text(t, arm) for t in framework.themes]
    item_texts = [item.text for item in items]

    # Anchors are "passages"; items are "queries" for asymmetric arm
    anchor_vecs = _encode(model, anchor_texts, is_query=False)
    item_vecs = _encode(model, item_texts, is_query=use_qprefix)

    sims = _cosine_sim(item_vecs, anchor_vecs)  # [n_items, n_themes]

    import numpy as np
    # Build primary and secondary predictions
    predictions: dict[str, int] = {}
    secondary: dict[str, int] = {}
    for idx, item in enumerate(items):
        row = sims[idx]
        ranked = np.argsort(row)[::-1]
        primary_theme = framework.themes[int(ranked[0])]
        secondary_theme = framework.themes[int(ranked[1])]
        predictions[item.item_id] = primary_theme.theme_id
        secondary[item.item_id] = secondary_theme.theme_id

    return cv_scoring.score_arm(
        arm,
        predictions,
        results_csv=results_csv,
        results_md=results_md,
        secondary=secondary,
        meta={'model': model_name, 'method': 'embedding_similarity'},
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Pure embedding-similarity VAAMR classifier.  Encodes CV items and "
            "VAAMR theme anchors with all-MiniLM-L6-v2; cosine-argmax → predicted "
            "stage.  Arms vary the anchor text composition."
        )
    )
    ap.add_argument(
        '--output-dir', default=None,
        help='Path to the QRA project output directory containing qra.db '
             '(default: data/Meta relative to repo root).',
    )
    ap.add_argument(
        '--model', default='all-MiniLM-L6-v2',
        help='Sentence-transformers model name or local path (default: all-MiniLM-L6-v2).',
    )
    ap.add_argument(
        '--arm', default='all',
        help=f'Which arm to run: {ARM_NAMES} or "all" (default: all).',
    )
    ap.add_argument(
        '--dry-run', action='store_true',
        help='Print the arm list and anchor fields each uses, then exit '
             '(no model load, no encoding).',
    )
    return ap


def _dry_run() -> None:
    print("01_embedding_similarity — DRY RUN")
    print(f"  Arms ({len(_ARMS)}):")
    for name, desc in _ARMS:
        print(f"    {name:<30} {desc}")
    print()
    print("  No model is loaded in --dry-run mode.")


def main(argv: list[str] | None = None) -> None:
    ap = _make_parser()
    args = ap.parse_args(argv)

    if args.dry_run:
        _dry_run()
        return

    # Resolve output paths relative to this file's location
    results_csv = _HERE / 'results.csv'
    results_md = _HERE / 'results.md'

    # Load framework + CV items (lightweight — no model load)
    framework = _data.load_vaamr()
    items = _data.load_cv_items(output_dir=args.output_dir)

    arms_to_run = ARM_NAMES if args.arm == 'all' else [args.arm]
    for arm in arms_to_run:
        if arm not in ARM_NAMES:
            print(f"[WARN] Unknown arm {arm!r}; skipping. Known arms: {ARM_NAMES}")
            continue
        print(f"[{arm}] running ...")
        metrics = run_arm(arm, items, framework, args.model, results_csv, results_md)
        print(f"[{arm}] acc_overall={metrics.get('acc_overall')}  "
              f"acc_secondary={metrics.get('acc_secondary')}  "
              f"n_items={metrics.get('n_items')}")

    print(f"\nResults appended to: {results_csv}")
    if results_md.exists():
        print(f"Markdown table:      {results_md}")


if __name__ == '__main__':
    main()
