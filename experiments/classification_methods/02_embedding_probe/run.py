"""
experiments/classification_methods/02_embedding_probe/run.py
-------------------------------------------------------------
Trained linear probe VAAMR classifier.

Embeds the labeled corpus segments (all-MiniLM-L6-v2) and trains a
logistic-regression probe on their VAAMR labels; then predicts the CV
items using the same embedding model so corpus and CV items share one
embedding space.

Arms:
  probe_5class         — LogisticRegression, VAAMR classes 0-4
  probe_classweighted  — same but class_weight='balanced'

Corpus loading mirrors experiments/gnn_reliability/harness.py::load_corpus
(master_segments.csv + qra.db) and the probe pattern mirrors
experiments/gnn_reliability/baselines.py::run_linear_probe, but embeds
with all-MiniLM instead of the Qwen cache.

Results are appended to 02_embedding_probe/results.csv
(and optionally results.md) via cv_scoring.score_arm.

NOT executed in this build; run-ready.
"""

from __future__ import annotations

import argparse
import pathlib
import sqlite3
import sys

# ---------------------------------------------------------------------------
# sys.path bootstrap so `from common import ...` resolves when run as a script
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve().parent          # .../02_embedding_probe/
_METHODS = _HERE.parent                                  # .../classification_methods/
_ROOT = _METHODS.parent.parent                           # repo root
for _p in (str(_ROOT / 'src'), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

if str(_METHODS) not in sys.path:
    sys.path.insert(0, str(_METHODS))

from common import data as _data          # noqa: E402
from common import cv_scoring             # noqa: E402

# ---------------------------------------------------------------------------
# Arm registry
# ---------------------------------------------------------------------------

_ARMS = [
    ('probe_5class',        'LogisticRegression on all-MiniLM embeddings, VAAMR 0-4'),
    ('probe_classweighted', 'same but class_weight="balanced"'),
]

ARM_NAMES = [a for a, _ in _ARMS]


# ---------------------------------------------------------------------------
# Corpus loading (sqlite-only path; no model load)
# ---------------------------------------------------------------------------

def _corpus_counts(output_dir: str | None) -> dict:
    """Count labeled participant segments in qra.db + master_segments.csv without
    loading any model.  Used by --dry-run to report n corpus segments."""
    from process import output_paths as _paths
    import os

    db_path = _data.resolve_db(output_dir)
    csv_path = pathlib.Path(
        _paths.master_segments_dir(output_dir or str(_ROOT / 'data' / 'Meta'))
    ) / 'master_segments.csv'

    n_labeled = 0
    n_total_csv = 0

    # Try qra.db first (segments + theme_labels join)
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            try:
                # Count participant segments with a non-null final_label via
                # the theme_labels overlay (consensus_label) joined to segments.
                row = conn.execute(
                    """
                    SELECT COUNT(*) FROM segments s
                    JOIN theme_labels tl ON tl.segment_id = s.segment_id
                    WHERE s.speaker = 'participant'
                      AND tl.primary_stage IS NOT NULL
                    """
                ).fetchone()
                n_labeled = row[0] if row else 0
            except Exception:
                n_labeled = -1  # table may not exist yet
            finally:
                conn.close()
        except Exception:
            n_labeled = -1

    # Count total rows in master_segments.csv if it exists
    if csv_path.exists():
        try:
            with open(csv_path, encoding='utf-8') as fh:
                # Count non-header lines
                n_total_csv = sum(1 for _ in fh) - 1
        except Exception:
            n_total_csv = -1

    return {
        'db_path': str(db_path),
        'csv_path': str(csv_path),
        'n_labeled_db': n_labeled,
        'n_csv_rows': n_total_csv,
    }


# ---------------------------------------------------------------------------
# Corpus loading for the real run (lazy imports)
# ---------------------------------------------------------------------------

def _load_corpus(output_dir: str | None):
    """Load master_segments.csv and filter to labeled participant segments.

    Returns (seg_ids: list[str], labels: list[int], texts: list[str]).
    Mirrors harness.load_corpus() + _labeled_participants() pattern but reads
    the CSV directly to avoid the heavy irr_join dependency.
    """
    import pandas as pd
    from process import output_paths as _paths

    csv_path = pathlib.Path(
        _paths.master_segments_dir(output_dir or str(_ROOT / 'data' / 'Meta'))
    ) / 'master_segments.csv'

    if not csv_path.exists():
        raise FileNotFoundError(
            f"master_segments.csv not found at {csv_path}.\n"
            "Run `qra assemble` or `qra run` first to produce the assembled dataset."
        )

    df = pd.read_csv(csv_path)

    # Filter to participant segments with a non-NaN final_label
    mask = (
        (df['speaker'] == 'participant') &
        df['final_label'].notna()
    )
    labeled = df[mask].copy()
    labeled['final_label'] = labeled['final_label'].astype(float).astype(int)

    seg_ids = labeled['segment_id'].astype(str).tolist()
    labels = labeled['final_label'].tolist()
    texts = labeled['text'].tolist()
    return seg_ids, labels, texts


# ---------------------------------------------------------------------------
# Embedding (lazy imports — only reached in the real run path)
# ---------------------------------------------------------------------------

def _embed(model, texts: list[str]) -> 'np.ndarray':
    """Encode texts and L2-normalize."""
    import numpy as np
    from sklearn.preprocessing import normalize
    vecs = model.encode(texts, normalize_embeddings=False, show_progress_bar=False)
    return normalize(np.asarray(vecs, dtype=np.float64), norm='l2', axis=1)


# ---------------------------------------------------------------------------
# Probe fitting (lazy imports)
# ---------------------------------------------------------------------------

def _fit_predict_probe(
    X_train: 'np.ndarray',
    y_train: list[int],
    X_test: 'np.ndarray',
    *,
    class_weight: str | None,
) -> 'np.ndarray':
    """Fit a logistic-regression probe and return predicted class for each test row."""
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    if len(np.unique(y_train)) < 2:
        # Degenerate fold: predict the only class present
        sole = int(np.unique(y_train)[0])
        return np.full(len(X_test), sole, dtype=int)

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight=class_weight)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)


# ---------------------------------------------------------------------------
# Main arm runner
# ---------------------------------------------------------------------------

def run_arm(
    arm: str,
    output_dir: str | None,
    model_name: str,
    results_csv: pathlib.Path,
    results_md: pathlib.Path | None,
) -> dict:
    """Embed corpus + CV items, train probe, predict, score."""
    from sentence_transformers import SentenceTransformer

    class_weight = 'balanced' if arm == 'probe_classweighted' else None

    # Load corpus
    seg_ids, labels, corpus_texts = _load_corpus(output_dir)
    if not seg_ids:
        raise RuntimeError(
            "No labeled participant segments found in master_segments.csv. "
            "Run the pipeline to classify segments before running this probe."
        )

    # Load CV items
    items = _data.load_cv_items(output_dir=output_dir)
    cv_texts = [item.text for item in items]

    print(f"  [{arm}] corpus n={len(seg_ids)}, cv items n={len(items)}")

    # Embed everything in one model load
    model = SentenceTransformer(model_name)
    X_corpus = _embed(model, corpus_texts)
    X_cv = _embed(model, cv_texts)

    # Fit probe on full corpus, predict CV items (no CV items in training data
    # — CV items are content-validity probes, not part of the labeled corpus)
    preds_arr = _fit_predict_probe(
        X_corpus, labels, X_cv, class_weight=class_weight
    )

    predictions: dict[str, int] = {
        item.item_id: int(preds_arr[i])
        for i, item in enumerate(items)
    }

    return cv_scoring.score_arm(
        arm,
        predictions,
        output_dir=output_dir,
        results_csv=results_csv,
        results_md=results_md,
        meta={
            'model': model_name,
            'method': 'embedding_probe',
            'n_corpus': len(seg_ids),
            'class_weight': class_weight or 'none',
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _make_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Trained linear probe VAAMR classifier on all-MiniLM embeddings.  "
            "Fits logistic regression on labeled corpus segments; predicts CV items "
            "in the same embedding space."
        )
    )
    ap.add_argument(
        '--output-dir', default=None,
        help='Path to the QRA project output directory containing qra.db and '
             'master_segments.csv (default: data/Meta relative to repo root).',
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
        help='Print corpus row counts, CV item count, and arm list — '
             'no model load, no embedding, no fitting.',
    )
    return ap


def _dry_run(output_dir: str | None) -> None:
    print("02_embedding_probe — DRY RUN")

    counts = _corpus_counts(output_dir)
    print(f"  DB path:              {counts['db_path']}")
    print(f"  master_segments.csv:  {counts['csv_path']}")
    print(f"  n labeled segments (db join):   {counts['n_labeled_db']}")
    print(f"  n total CSV rows:               {counts['n_csv_rows']}")

    # CV items count via sqlite (no model needed)
    try:
        items = _data.load_cv_items(output_dir=output_dir)
        print(f"  n CV items (cv_vaamr_v1):       {len(items)}")
    except Exception as exc:
        print(f"  n CV items: unavailable ({exc})")

    print()
    print(f"  Arms ({len(_ARMS)}):")
    for name, desc in _ARMS:
        print(f"    {name:<30} {desc}")
    print()
    print("  No model is loaded in --dry-run mode.")


def main(argv: list[str] | None = None) -> None:
    ap = _make_parser()
    args = ap.parse_args(argv)

    if args.dry_run:
        _dry_run(args.output_dir)
        return

    results_csv = _HERE / 'results.csv'
    results_md = _HERE / 'results.md'

    arms_to_run = ARM_NAMES if args.arm == 'all' else [args.arm]
    for arm in arms_to_run:
        if arm not in ARM_NAMES:
            print(f"[WARN] Unknown arm {arm!r}; skipping. Known arms: {ARM_NAMES}")
            continue
        print(f"[{arm}] running ...")
        metrics = run_arm(arm, args.output_dir, args.model, results_csv, results_md)
        print(f"[{arm}] acc_overall={metrics.get('acc_overall')}  "
              f"n_items={metrics.get('n_items')}  "
              f"n_corpus={metrics.get('n_corpus')}")

    print(f"\nResults appended to: {results_csv}")
    if results_md.exists():
        print(f"Markdown table:      {results_md}")


if __name__ == '__main__':
    main()
