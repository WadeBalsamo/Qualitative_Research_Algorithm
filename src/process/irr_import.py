"""
process/irr_import.py
---------------------
Importer for human-coded VAAMR inter-rater-reliability (IRR) test-sets.

The reviewable source of truth is a single wide CSV (``data/irr/human_coded_testsets.csv``)
carrying, per worksheet item, each researcher's primary/secondary VAAMR code plus a
pre-filled human consensus. This module:

  1. parses the CSV (pandas, blank-preserving),
  2. normalizes free-text labels to VAAMR theme_ids via
     ``ThemeFramework.build_name_to_id_map`` (``No code`` -> ABSTAIN, empty cell -> missing),
  3. recomputes majority/unanimous consensus from the rater columns
     (reusing ``majority_vote.vote_single_label``) and WARNS on any mismatch with the
     pre-filled ``consensus_source`` / ``human_consensus_*`` cells,
  4. resolves each ``(worksheet_n, item)`` to a frozen ``segment_id`` via the
     ``testset_items`` -> ``segments`` join, validating item counts + content SHAs
     against the frozen worksheets and warning on drift,
  5. persists the per-rater ballots + the consensus row to ``irr_testsets`` /
     ``irr_human_codes``.

Re-import is idempotent: only the IRR tables for the affected worksheets are
replaced; frozen segments and validation worksheets are never touched.

Label encoding in ``irr_human_codes``::

    prim / secondary INTEGER
        >= 0  -> VAAMR theme_id
        -1    -> ABSTAIN ("No code", an active abstain ballot)
        NULL  -> rater did not code that item / no secondary

(The analysis layer maps -1 back to the ``'ABSTAIN'`` sentinel that
``classification_tools.reliability`` expects.)
"""

import datetime
import hashlib
from typing import Dict, List, Optional, Tuple

import pandas as pd

from theme_framework.registry import load as load_framework
from classification_tools.majority_vote import vote_single_label
from . import db

# Stable rater roster (CSV column prefixes).  Absent raters simply have blank
# cells for a given worksheet.
RATER_COLUMNS: Tuple[str, ...] = ('becca', 'ryan', 'wade', 'adam')

# Sentinel stored for an active "No code" abstain ballot.
ABSTAIN_CODE = -1

# rater value used for the consensus-of-record row.
CONSENSUS_RATER = '__consensus__'


class TestsetDriftError(RuntimeError):
    """A frozen test-set item's segment text no longer matches the SHA the humans
    coded against (or its segment can no longer be resolved).

    Carries the ``drift`` list (see :func:`check_testset_drift`) so callers can report
    exactly which items diverged. Raised by ``irr_analysis.run_irr_analysis`` when
    ``strict_drift`` is set (the ``qra irr run`` default — pass ``--allow-drift`` to
    downgrade to a warning)."""

    def __init__(self, message: str, drift: List[dict]):
        super().__init__(message)
        self.drift = drift


# ---------------------------------------------------------------------------
# Label normalization
# ---------------------------------------------------------------------------

def _normalize_label(
    raw: Optional[str],
    name_to_id: Dict[str, int],
    warnings: List[str],
    where: str,
) -> Tuple[str, Optional[int]]:
    """Map a free-text cell to ``(kind, value)``.

    kind is one of ``'missing'`` (blank cell -> value None), ``'abstain'``
    ("No code" -> value ABSTAIN_CODE), or ``'coded'`` (recognised label -> theme_id).
    Unrecognised non-blank text is treated as missing and recorded as a warning.
    """
    text = (raw or '').strip()
    if text == '':
        return 'missing', None
    low = text.lower()
    if low in ('no code', 'nocode', 'none', 'abstain'):
        return 'abstain', ABSTAIN_CODE
    if low in name_to_id:
        return 'coded', name_to_id[low]
    warnings.append(f"unrecognised label {text!r} at {where}; treated as no-ballot")
    return 'missing', None


def _ballot_for_vote(kind: str, value: Optional[int]) -> Optional[dict]:
    """Build a ``vote_single_label`` parsed-run dict for one rater ballot, or
    None if the rater cast no ballot (missing cell)."""
    if kind == 'missing':
        return None
    if kind == 'abstain':
        return {'vote': 'ABSTAIN', 'primary_stage': None}
    return {'vote': 'CODED', 'primary_stage': value}


# ---------------------------------------------------------------------------
# Consensus recompute / validation
# ---------------------------------------------------------------------------

_SOURCE_FROM_LEVEL = {
    'unanimous': 'unanimous',
    'majority': 'majority',
    'split': 'unresolved',
    'none': 'unresolved',
}


def _recompute_consensus(
    rater_ballots: List[Tuple[str, str, Optional[int]]],
) -> Tuple[str, Optional[int]]:
    """Return ``(computed_source, computed_value)`` from the coding raters.

    ``computed_value`` is a theme_id, ABSTAIN_CODE, or None (unresolved).
    """
    raters = [rid for rid, _, _ in rater_ballots]
    parsed = [_ballot_for_vote(kind, val) for _, kind, val in rater_ballots]
    if not any(p is not None for p in parsed):
        return 'unresolved', None
    res = vote_single_label(parsed, rater_ids=raters)
    level = res.get('agreement_level', 'split')
    source = _SOURCE_FROM_LEVEL.get(level, 'unresolved')
    if source == 'unresolved':
        return 'unresolved', None
    cv = res.get('consensus_vote')
    if cv == 'ABSTAIN':
        return source, ABSTAIN_CODE
    return source, cv


# ---------------------------------------------------------------------------
# Segment resolution / drift validation
# ---------------------------------------------------------------------------

def _resolve_worksheet_segments(
    conn, worksheet_n: int,
) -> Dict[int, Tuple[Optional[str], Optional[str], Optional[str]]]:
    """Map ``item_num -> (segment_id, frozen_sha256, live_text)`` for a worksheet.

    Joins ``testset_items`` to the frozen ``segments`` table on
    ``(session_id, segment_index = seg_num - 1)``.  Returns ``{}`` if the
    worksheet has no items (e.g. the target project lacks it).
    """
    rows = conn.execute(
        """
        SELECT ti.item_num   AS item_num,
               ti.sha256     AS frozen_sha,
               s.segment_id  AS segment_id,
               s.text        AS text
        FROM testset_items ti
        LEFT JOIN segments s
               ON s.session_id = ti.session_id
              AND s.segment_index = ti.seg_num - 1
        WHERE ti.worksheet_n = ?
        ORDER BY ti.item_num
        """,
        (worksheet_n,),
    ).fetchall()
    return {
        r['item_num']: (r['segment_id'], r['frozen_sha'], r['text'])
        for r in rows
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def import_irr_csv(
    run_dir: str,
    csv_path: str,
    *,
    framework: str = 'vaamr',
    verbose: bool = True,
) -> dict:
    """Import a human-coded IRR CSV into ``run_dir``'s ``qra.db``.

    Returns a summary dict::

        {'worksheets': [n, ...], 'n_items': int, 'n_codes': int,
         'warnings': [str, ...]}

    Warnings (consensus mismatches, content drift, unresolved segment ids,
    unrecognised labels) are collected and, when ``verbose``, printed.
    """
    fw = load_framework(framework)
    name_to_id = fw.build_name_to_id_map()

    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    df['worksheet_n'] = df['worksheet_n'].astype(int)
    df['item'] = df['item'].astype(int)

    warnings: List[str] = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    # Accumulate everything in memory, then write in one transaction.
    testset_rows: List[tuple] = []          # (worksheet_n, name, raters_json, n_items, created_at)
    code_rows: List[tuple] = []             # irr_human_codes tuples
    total_items = 0

    with db.open_db(run_dir) as conn:
        for worksheet_n, wdf in df.groupby('worksheet_n'):
            wdf = wdf.sort_values('item')
            seg_map = _resolve_worksheet_segments(conn, int(worksheet_n))

            # Item-count drift vs the frozen worksheet metadata.
            meta = conn.execute(
                "SELECT n_items FROM testset_worksheets WHERE worksheet_n = ?",
                (int(worksheet_n),),
            ).fetchone()
            if meta is None:
                warnings.append(
                    f"worksheet {worksheet_n}: no frozen worksheet in qra.db; "
                    f"machine comparisons will be unavailable for these items"
                )
            elif meta['n_items'] != len(wdf):
                warnings.append(
                    f"worksheet {worksheet_n}: CSV has {len(wdf)} items but the "
                    f"frozen worksheet has {meta['n_items']}"
                )

            raters_present: List[str] = []

            for _, row in wdf.iterrows():
                item = int(row['item'])
                total_items += 1
                where = f"ws{worksheet_n} item{item}"

                segment_id, frozen_sha, live_text = seg_map.get(
                    item, (None, None, None)
                )
                if item not in seg_map and meta is not None:
                    warnings.append(f"{where}: no testset item -> segment mapping found")
                # Content-drift check: frozen sha vs current segment text.
                if segment_id is not None and frozen_sha and live_text is not None:
                    live_sha = hashlib.sha256(live_text.encode()).hexdigest()
                    if live_sha != frozen_sha:
                        warnings.append(
                            f"{where}: segment text changed since the worksheet was "
                            f"frozen (SHA drift) — machine label may not match what "
                            f"humans coded"
                        )

                # Per-rater ballots.
                rater_ballots: List[Tuple[str, str, Optional[int]]] = []
                for rater in RATER_COLUMNS:
                    pcol, scol = f'{rater}_p', f'{rater}_s'
                    if pcol not in row:
                        continue
                    pkind, pval = _normalize_label(
                        row.get(pcol), name_to_id, warnings, f"{where} {pcol}"
                    )
                    if pkind == 'missing':
                        continue  # rater did not code this item
                    skind, sval = _normalize_label(
                        row.get(scol), name_to_id, warnings, f"{where} {scol}"
                    )
                    rater_ballots.append((rater, pkind, pval))
                    if rater not in raters_present:
                        raters_present.append(rater)
                    code_rows.append((
                        int(worksheet_n), item, segment_id, rater,
                        pval, sval if skind != 'missing' else None,
                        0, None, None,
                    ))

                # Consensus of record (pre-filled, human-reviewed) + validation.
                csv_source = (row.get('consensus_source') or '').strip().lower()
                ckind, cval = _normalize_label(
                    row.get('human_consensus_p'), name_to_id, warnings,
                    f"{where} human_consensus_p",
                )
                cskind, csval = _normalize_label(
                    row.get('human_consensus_s'), name_to_id, warnings,
                    f"{where} human_consensus_s",
                )
                cons_value = None if (csv_source == 'unresolved' or ckind == 'missing') else cval

                _validate_consensus(
                    rater_ballots, csv_source, cons_value, warnings, where
                )

                notes = (row.get('notes') or '').strip() or None
                code_rows.append((
                    int(worksheet_n), item, segment_id, CONSENSUS_RATER,
                    cons_value, csval if cskind != 'missing' else None,
                    1, csv_source or 'unresolved', notes,
                ))

            testset_rows.append((
                int(worksheet_n),
                f'testset_{int(worksheet_n)}',
                db.dumps(raters_present),
                len(wdf),
                now,
            ))

        # Idempotent replace: drop IRR rows for the affected worksheets, re-insert.
        affected = [r[0] for r in testset_rows]
        for n in affected:
            conn.execute("DELETE FROM irr_human_codes WHERE worksheet_n = ?", (n,))
            conn.execute("DELETE FROM irr_testsets WHERE worksheet_n = ?", (n,))
        conn.executemany(
            "INSERT INTO irr_testsets (worksheet_n, name, raters, n_items, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            testset_rows,
        )
        conn.executemany(
            "INSERT INTO irr_human_codes "
            "(worksheet_n, item_num, segment_id, rater, prim, secondary, is_consensus, source, notes) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            code_rows,
        )

    summary = {
        'worksheets': sorted(r[0] for r in testset_rows),
        'n_items': total_items,
        'n_codes': len(code_rows),
        'warnings': warnings,
    }
    if verbose:
        _print_summary(csv_path, summary)
    return summary


def _validate_consensus(
    rater_ballots: List[Tuple[str, str, Optional[int]]],
    csv_source: str,
    csv_value: Optional[int],
    warnings: List[str],
    where: str,
) -> None:
    """Recompute consensus from rater columns and warn on mismatch with the
    pre-filled CSV cells (explicit cells may legitimately override the majority,
    so only a contradicted unanimous vote warns there)."""
    computed_source, computed_value = _recompute_consensus(rater_ballots)

    if csv_source == 'explicit':
        if computed_source == 'unanimous' and computed_value != csv_value:
            warnings.append(
                f"{where}: explicit consensus overrides a UNANIMOUS rater vote "
                f"(rater_value={computed_value}, explicit_value={csv_value})"
            )
    elif csv_source in ('unanimous', 'majority'):
        if computed_source != csv_source or computed_value != csv_value:
            warnings.append(
                f"{where}: consensus_source={csv_source} (rater_value={csv_value}) but "
                f"raters recompute as {computed_source} (rater_value={computed_value})"
            )
    elif csv_source == 'unresolved':
        if computed_source in ('unanimous', 'majority'):
            warnings.append(
                f"{where}: marked unresolved but raters recompute a "
                f"{computed_source} (rater_value={computed_value})"
            )


def _print_summary(csv_path: str, summary: dict) -> None:
    print(f"Imported IRR codes from {csv_path}")
    print(f"  worksheets: {summary['worksheets']}")
    print(f"  items: {summary['n_items']}   code rows: {summary['n_codes']}")
    if summary['warnings']:
        print(f"  warnings ({len(summary['warnings'])}):")
        for w in summary['warnings']:
            print(f"    - {w}")
    else:
        print("  no warnings.")


# ---------------------------------------------------------------------------
# Read-back helpers (used by irr_analysis + the TUI list view)
# ---------------------------------------------------------------------------

def list_imported_testsets(run_dir: str) -> List[dict]:
    """Return imported IRR test-sets as dicts (worksheet_n, name, raters, n_items)."""
    if not db.db_exists(run_dir):
        return []
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT worksheet_n, name, raters, n_items, created_at "
            "FROM irr_testsets ORDER BY worksheet_n"
        ).fetchall()
    out = []
    for r in rows:
        out.append({
            'worksheet_n': r['worksheet_n'],
            'name': r['name'],
            'raters': db.loads(r['raters']) or [],
            'n_items': r['n_items'],
            'created_at': r['created_at'],
        })
    return out


def read_human_codes(run_dir: str) -> List[dict]:
    """Return every ``irr_human_codes`` row as a dict, ordered by (ws, item, rater)."""
    if not db.db_exists(run_dir):
        return []
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT worksheet_n, item_num, segment_id, rater, prim, secondary, "
            "is_consensus, source, notes FROM irr_human_codes "
            "ORDER BY worksheet_n, item_num, rater"
        ).fetchall()
    return [
        {
            'worksheet_n': r['worksheet_n'],
            'item_num': r['item_num'],
            'segment_id': r['segment_id'],
            'rater': r['rater'],
            'primary': r['prim'],
            'secondary': r['secondary'],
            'is_consensus': bool(r['is_consensus']),
            'source': r['source'],
            'notes': r['notes'],
        }
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Run-time content guard (the IRR comparison must score the EXACT text humans coded)
# ---------------------------------------------------------------------------

def _worksheet_numbers(conn) -> List[int]:
    rows = conn.execute(
        "SELECT worksheet_n FROM testset_worksheets ORDER BY worksheet_n"
    ).fetchall()
    return [r['worksheet_n'] for r in rows]


def check_testset_drift(run_dir: str) -> List[dict]:
    """Re-validate every frozen test-set item against the CURRENT segment text.

    Returns the list of items whose content no longer matches what the humans coded —
    each ``{worksheet_n, item_num, segment_id, kind, frozen_sha, live_sha}`` with
    ``kind`` in ``{'drift', 'unresolved'}``. An empty list means the test-sets are
    byte-consistent with the frozen worksheets (the IRR comparison is sound).

    Reuses the same ``testset_items`` -> ``segments`` join and SHA comparison the
    importer applies at :func:`import_irr_csv` time, so the run-time guard and the
    import-time warning agree by construction.
    """
    if not db.db_exists(run_dir):
        return []
    out: List[dict] = []
    with db.open_db(run_dir) as conn:
        for ws in _worksheet_numbers(conn):
            seg_map = _resolve_worksheet_segments(conn, ws)
            for item_num, (segment_id, frozen_sha, live_text) in sorted(seg_map.items()):
                if segment_id is None or live_text is None or not frozen_sha:
                    out.append({'worksheet_n': ws, 'item_num': item_num,
                                'segment_id': segment_id, 'kind': 'unresolved',
                                'frozen_sha': frozen_sha, 'live_sha': None})
                    continue
                live_sha = hashlib.sha256(live_text.encode()).hexdigest()
                if live_sha != frozen_sha:
                    out.append({'worksheet_n': ws, 'item_num': item_num,
                                'segment_id': segment_id, 'kind': 'drift',
                                'frozen_sha': frozen_sha, 'live_sha': live_sha})
    return out


def testset_content_signature(run_dir: str) -> str:
    """Short hash of the CURRENT resolved test-set segment content.

    Folded into ``irr_analysis.machine_signature`` so any change to a coded segment's
    text flips the change-gate and forces IRR regeneration (surfacing the drift banner)
    on the next ``qra analyze``. Empty string when the project has no ``qra.db``."""
    if not db.db_exists(run_dir):
        return ''
    parts: List[str] = []
    with db.open_db(run_dir) as conn:
        for ws in _worksheet_numbers(conn):
            seg_map = _resolve_worksheet_segments(conn, ws)
            for item_num, (segment_id, frozen_sha, live_text) in sorted(seg_map.items()):
                live_sha = (hashlib.sha256(live_text.encode()).hexdigest()
                            if live_text is not None else '')
                parts.append(f"{ws}:{item_num}:{live_sha}")
    return hashlib.sha256('||'.join(parts).encode()).hexdigest()[:16]


def format_drift_banner(drift: List[dict], *, limit: int = 12) -> str:
    """Human-readable multi-line banner describing test-set content drift (shared by
    the analysis warning/raise path, the report header, and the CLI refusal)."""
    n = len(drift)
    lines = [f"{n} test-set item(s) no longer match the human-coded segment text:"]
    for d in drift[:limit]:
        if d.get('kind') == 'unresolved':
            lines.append(f"  ws{d['worksheet_n']} item{d['item_num']}: segment unresolved "
                         f"({d.get('segment_id')})")
        else:
            lines.append(f"  ws{d['worksheet_n']} item{d['item_num']} ({d.get('segment_id')}): "
                         f"SHA {(d.get('frozen_sha') or '')[:10]} -> {(d.get('live_sha') or '')[:10]}")
    if n > limit:
        lines.append(f"  … and {n - limit} more")
    return "\n".join(lines)
