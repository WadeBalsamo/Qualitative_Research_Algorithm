"""
experiments/classification_methods/common/cv_scoring.py
--------------------------------------------------------
Content-validity scoring harness for classification-method experiments.

Mirrors the pass/fail logic from
``process.assembly.content_validity._grade_cv_items`` (primary match)
and adds secondary-match, per-difficulty, and per-theme breakdowns.

Public API
----------
score_arm(
    arm_name       : str,
    predictions    : dict[item_id -> int | None],
    *,
    output_dir     : str | None = None,
    results_csv    : Path,
    results_md     : Path | None = None,
    secondary      : dict[item_id -> int | None] | None = None,
    meta           : dict | None = None,
    testset        : str = 'cv_vaamr_v1',
) -> dict
    Appends one row to results_csv (creates with header if missing).
    Optionally (re)writes a markdown table to results_md.
    Returns the full metrics dict.

No heavy dependencies (torch, sentence-transformers) are imported at
module level — this file is import-safe in offline/CI environments.
"""

import csv
import datetime
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# sys.path bootstrap — same pattern as data.py
# ---------------------------------------------------------------------------

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
for _p in (os.path.join(_ROOT, 'src'), _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.data import load_cv_items, stage_names   # noqa: E402


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_DIFFICULTY_TIERS = ('clear', 'subtle', 'adversarial')

# CSV columns — fixed order so rows from different arms line up.
_CSV_COLUMNS = [
    'arm', 'testset', 'timestamp',
    'n_items', 'n_abstain',
    'acc_overall', 'acc_secondary',
    'acc_clear', 'acc_subtle', 'acc_adversarial',
    'acc_stage_0', 'acc_stage_1', 'acc_stage_2', 'acc_stage_3', 'acc_stage_4',
    'n_clear', 'n_subtle', 'n_adversarial',
    'n_stage_0', 'n_stage_1', 'n_stage_2', 'n_stage_3', 'n_stage_4',
]


def score_arm(
    arm_name: str,
    predictions: Dict,
    *,
    output_dir: Optional[str] = None,
    results_csv: Path,
    results_md: Optional[Path] = None,
    secondary: Optional[Dict] = None,
    meta: Optional[Dict] = None,
    testset: str = 'cv_vaamr_v1',
) -> Dict:
    """
    Score one classifier arm against the content-validity testset.

    Parameters
    ----------
    arm_name : str
        Short label for this arm (e.g. 'zeroshot_nemotron', 'probe_v1').
    predictions : dict
        Maps item_id (str) -> predicted primary stage (int) or None (abstain).
    output_dir : str or None
        Passed to load_cv_items; None → default DB at data/Meta/qra.db.
    results_csv : Path
        CSV ledger file.  Created with header row if absent; one row APPENDED
        per call so incremental runs accumulate.
    results_md : Path or None
        If provided, a markdown table summarising ALL rows in results_csv is
        (re)written here after appending the new row.
    secondary : dict or None
        Optional secondary predictions dict (item_id -> int | None).
        A miss on primary that matches secondary counts as a pass for the
        acc_secondary metric — mirrors _grade_cv_items behaviour.
    meta : dict or None
        Arbitrary key-value metadata written as extra JSON-encoded columns.
        Not reflected in the fixed CSV schema but returned in the metrics dict.
    testset : str
        Name of the cv_testset to load from the DB.

    Returns
    -------
    dict with keys matching _CSV_COLUMNS (plus any meta keys).
    """
    items = load_cv_items(output_dir=output_dir, testset=testset)
    snames = stage_names()
    secondary = secondary or {}

    # -----------------------------------------------------------------------
    # Accumulate counts
    # -----------------------------------------------------------------------
    n_items = len(items)
    n_abstain = 0

    # Per difficulty: correct counts (primary + secondary)
    diff_correct_primary: Dict[str, int] = {t: 0 for t in _DIFFICULTY_TIERS}
    diff_correct_secondary: Dict[str, int] = {t: 0 for t in _DIFFICULTY_TIERS}
    diff_total: Dict[str, int] = {t: 0 for t in _DIFFICULTY_TIERS}

    # Per expected stage: correct primary only
    stage_correct: Dict[int, int] = {s: 0 for s in range(5)}
    stage_total: Dict[int, int] = {s: 0 for s in range(5)}

    total_correct_primary = 0
    total_correct_secondary = 0   # primary OR secondary matches expected

    for item in items:
        iid = item.item_id
        pred = predictions.get(iid)
        sec_pred = secondary.get(iid)
        expected = item.expected_stage
        diff = (item.difficulty or 'clear').lower()
        if diff not in _DIFFICULTY_TIERS:
            diff = 'clear'

        if pred is None:
            n_abstain += 1

        primary_pass = (pred == expected)
        secondary_pass = primary_pass or (sec_pred == expected)

        if primary_pass:
            total_correct_primary += 1
            diff_correct_primary[diff] += 1
        if secondary_pass:
            total_correct_secondary += 1
            diff_correct_secondary[diff] += 1

        diff_total[diff] += 1

        if expected in stage_correct:
            stage_total[expected] += 1
            if primary_pass:
                stage_correct[expected] += 1

    # -----------------------------------------------------------------------
    # Metrics
    # -----------------------------------------------------------------------
    def _pct(num: int, den: int) -> Optional[float]:
        return round(num / den, 4) if den > 0 else None

    ts = datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    metrics: Dict = {
        'arm': arm_name,
        'testset': testset,
        'timestamp': ts,
        'n_items': n_items,
        'n_abstain': n_abstain,
        'acc_overall': _pct(total_correct_primary, n_items),
        'acc_secondary': _pct(total_correct_secondary, n_items),
        'acc_clear': _pct(diff_correct_primary['clear'], diff_total['clear']),
        'acc_subtle': _pct(diff_correct_primary['subtle'], diff_total['subtle']),
        'acc_adversarial': _pct(diff_correct_primary['adversarial'], diff_total['adversarial']),
    }
    for s in range(5):
        metrics[f'acc_stage_{s}'] = _pct(stage_correct[s], stage_total[s])
    for tier in _DIFFICULTY_TIERS:
        metrics[f'n_{tier}'] = diff_total[tier]
    for s in range(5):
        metrics[f'n_stage_{s}'] = stage_total[s]

    if meta:
        metrics.update(meta)

    # -----------------------------------------------------------------------
    # Write / append CSV
    # -----------------------------------------------------------------------
    results_csv = Path(results_csv)
    results_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not results_csv.exists() or results_csv.stat().st_size == 0

    with open(results_csv, 'a', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction='ignore')
        if write_header:
            writer.writeheader()
        writer.writerow({k: metrics.get(k, '') for k in _CSV_COLUMNS})

    # -----------------------------------------------------------------------
    # Optionally (re)write markdown table
    # -----------------------------------------------------------------------
    if results_md is not None:
        _write_markdown_table(results_csv, Path(results_md))

    return metrics


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

def _write_markdown_table(csv_path: Path, md_path: Path) -> None:
    """Read all rows from csv_path and render a markdown summary table."""
    if not csv_path.exists():
        return

    with open(csv_path, newline='', encoding='utf-8') as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return

    # Display columns (subset of full CSV, in a readable order)
    display_cols = [
        'arm', 'timestamp', 'n_items', 'n_abstain',
        'acc_overall', 'acc_secondary',
        'acc_clear', 'acc_subtle', 'acc_adversarial',
    ]
    # Keep only columns that exist in every row
    display_cols = [c for c in display_cols if all(c in r for r in rows)]

    header = '| ' + ' | '.join(display_cols) + ' |'
    sep = '| ' + ' | '.join('---' for _ in display_cols) + ' |'
    data_rows = []
    for row in rows:
        data_rows.append('| ' + ' | '.join(str(row.get(c, '')) for c in display_cols) + ' |')

    md_path.parent.mkdir(parents=True, exist_ok=True)
    with open(md_path, 'w', encoding='utf-8') as fh:
        fh.write('# CV Scoring Results\n\n')
        fh.write(header + '\n')
        fh.write(sep + '\n')
        for dr in data_rows:
            fh.write(dr + '\n')
        fh.write('\n')
