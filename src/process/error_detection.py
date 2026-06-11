"""
process/error_detection.py
--------------------------
M4 — detect classification errors and identify repair targets.

Three public entry points:

``overlay_row_status(rec, overlay)``
    Classify a single overlay record as one of:
    - ``'ok'``                 — has a primary label (including ``plurality_coded``)
    - ``'legitimate_abstain'`` — consensus ABSTAIN (all raters genuinely abstained /
                                 abstained or the whole cue was skipped)
    - ``'repairable_error'``   — agreement is ``'none'``, rater_votes is missing/empty,
                                 or primary is NULL without a consensus ABSTAIN marker
    - ``'review'``             — agreement level is ``'split'`` (genuine disagreement
                                 among valid CODED ballots — NEVER auto-repaired)

``detect_overlay_errors(run_dir, overlay)``
    Read the overlay table and classify every row.  Returns
    ``{'by_segment': {seg_id: status}, 'summary': {status: count}}``.

``detect_run_error_cells(run_dir, run_id)``
    Return the segment_ids whose ``label_ballots`` row for ``run_id`` has
    ``vote='ERROR'`` (hard parse failure, raw_json NULL).

``repair_targets(run_dir, overlay, run_ids=None)``
    For each non-archived, non-failed run (or the given subset), return the
    segment_ids with at least one ERROR ballot.
    Returns ``{run_id: [segment_id, ...]}``.
"""

from typing import Dict, List, Optional

from . import db
from . import classifications_io as _cio
from . import run_registry as _rr

# ---------------------------------------------------------------------------
# overlay_row_status — per-row classification
# ---------------------------------------------------------------------------

# Agreement levels that indicate a labeled (ok) row when a primary is present.
_LABELED_LEVELS = frozenset({'unanimous', 'majority', 'plurality_coded'})
# Agreement levels that signal genuine rater disagreement (NOT auto-repairable).
_SPLIT_LEVELS = frozenset({'split'})
# Agreement levels that mean no consensus formed (error-ish).
_NONE_LEVELS = frozenset({'none'})


def overlay_row_status(rec: dict, overlay: str) -> str:
    """Classify one overlay record as ``'ok'`` | ``'legitimate_abstain'`` |
    ``'repairable_error'`` | ``'review'``.

    Truth table (per overlay):

    **theme** (uses primary_stage / consensus_vote / agreement_level / rater_votes):
    - primary_stage is not None → ``'ok'``  (covers unanimous/majority/plurality_coded)
    - consensus_vote == 'ABSTAIN' (or int-coded ABSTAIN sentinel None+level=none but
      explicit ABSTAIN vote) → ``'legitimate_abstain'``
    - agreement_level in ('split',) AND primary_stage is None → ``'review'``
    - agreement_level == 'none' OR rater_votes missing/empty → ``'repairable_error'``
    - fallback null primary → ``'repairable_error'``

    **purer** (uses purer_primary / purer_agreement_level / purer_rater_votes):
    same logic with purer_* field names; ``purer_run_consistency`` or ABSTAIN on
    ``purer_rater_votes`` entries signals legitimate_abstain.

    **codebook** (no abstain concept — ensemble is a list):
    - codebook_labels_ensemble is non-empty list → ``'ok'``
    - otherwise → ``'repairable_error'``
    """
    if overlay == 'theme':
        return _theme_row_status(rec)
    if overlay == 'purer':
        return _purer_row_status(rec)
    if overlay == 'codebook':
        return _codebook_row_status(rec)
    # Unknown overlay: treat as ok (don't error).
    return 'ok'


def _theme_row_status(rec: dict) -> str:
    """Status for a theme_labels row."""
    primary = rec.get('primary_stage')
    agreement = (rec.get('agreement_level') or '').lower()
    consensus_vote = rec.get('consensus_vote')
    rater_votes = rec.get('rater_votes')

    # primary present → labeled (ok) regardless of agreement level.
    if primary is not None:
        return 'ok'

    # Explicit consensus ABSTAIN → legitimate_abstain (all raters abstained/agreed).
    if consensus_vote == 'ABSTAIN':
        return 'legitimate_abstain'

    # Genuine split among valid CODED ballots → review (NOT auto-repairable).
    if agreement in _SPLIT_LEVELS:
        return 'review'

    # agreement='none' is the sentinel for no valid ballots / all errors → repairable.
    if agreement in _NONE_LEVELS:
        return 'repairable_error'

    # Missing or empty rater_votes with no primary → repairable (no ballot data).
    if not _has_rater_votes(rater_votes):
        return 'repairable_error'

    # Fallthrough: null primary with votes present but no ABSTAIN consensus → repairable.
    return 'repairable_error'


def _purer_row_status(rec: dict) -> str:
    """Status for a purer_labels row."""
    primary = rec.get('purer_primary')
    agreement = (rec.get('purer_agreement_level') or '').lower()
    rater_votes = rec.get('purer_rater_votes')

    # primary present → labeled (ok).
    if primary is not None:
        return 'ok'

    # Detect consensus ABSTAIN: either the level is 'none' but all votes are ABSTAIN
    # (the PURER single-run path leaves agreement='none' on a clean abstain row), or
    # an explicit 'ABSTAIN' appears in rater_votes values.
    if _purer_is_unanimous_abstain(rater_votes):
        return 'legitimate_abstain'

    # Split among valid CODED ballots → review.
    if agreement in _SPLIT_LEVELS:
        return 'review'

    # agreement='none' with no ABSTAIN → repairable.
    if agreement in _NONE_LEVELS:
        return 'repairable_error'

    # Missing or empty rater_votes → repairable.
    if not _has_rater_votes(rater_votes):
        return 'repairable_error'

    # Fallthrough: null primary with votes but no ABSTAIN consensus → repairable.
    return 'repairable_error'


def _codebook_row_status(rec: dict) -> str:
    """Status for a codebook_labels row.  No abstain concept in the codebook."""
    ensemble = rec.get('codebook_labels_ensemble')
    if _is_nonempty_list(ensemble):
        return 'ok'
    return 'repairable_error'


# ---------------------------------------------------------------------------
# Overlay-level detection
# ---------------------------------------------------------------------------

def detect_overlay_errors(run_dir: str, overlay: str) -> dict:
    """Return ``{'by_segment': {seg_id: status}, 'summary': {status: count}}``
    for every row in the overlay table.

    Status values: ``'ok'`` | ``'legitimate_abstain'`` | ``'repairable_error'`` |
    ``'review'``.  Returns empty by_segment when the store/table is absent.
    """
    records = _cio.read_overlay(run_dir, overlay)
    by_segment: Dict[str, str] = {}
    summary: Dict[str, int] = {'ok': 0, 'legitimate_abstain': 0,
                                'repairable_error': 0, 'review': 0}
    for rec in records:
        status = overlay_row_status(rec, overlay)
        by_segment[rec['segment_id']] = status
        summary[status] = summary.get(status, 0) + 1
    return {'by_segment': by_segment, 'summary': summary}


# ---------------------------------------------------------------------------
# Per-run error-cell detection
# ---------------------------------------------------------------------------

def detect_run_error_cells(run_dir: str, run_id: int) -> List[str]:
    """Return segment_ids whose ``label_ballots.vote = 'ERROR'`` for ``run_id``.

    An ERROR ballot is a hard parse failure (NULL raw_json, NULL stage/confidence).
    Returns ``[]`` when the store is absent or the run has no ERROR ballots.
    """
    if not db.db_exists(run_dir):
        return []
    # Determine the overlay for this run.
    run = _rr.get_run(run_dir, run_id)
    if run is None:
        return []
    overlay = run.get('overlay', '')
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT segment_id FROM label_ballots "
            "WHERE overlay = ? AND run_id = ? AND vote = 'ERROR' "
            "ORDER BY segment_id",
            (overlay, run_id),
        ).fetchall()
    return [r['segment_id'] for r in rows]


# ---------------------------------------------------------------------------
# repair_targets
# ---------------------------------------------------------------------------

def repair_targets(
    run_dir: str,
    overlay: str,
    run_ids: Optional[List[int]] = None,
) -> Dict[int, List[str]]:
    """Return ``{run_id: [segment_id, ...]}`` for runs with ≥1 ERROR ballot.

    Only non-archived, non-failed runs are candidates unless ``run_ids`` is
    given explicitly.  Runs with an empty error list are omitted.
    """
    if run_ids is not None:
        candidates = run_ids
    else:
        runs = _rr.list_runs(run_dir, overlay=overlay)
        _terminal = frozenset({'archived', 'failed'})
        candidates = [r['run_id'] for r in runs if r['status'] not in _terminal]

    result: Dict[int, List[str]] = {}
    for rid in candidates:
        errors = detect_run_error_cells(run_dir, rid)
        if errors:
            result[int(rid)] = errors
    return result


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _has_rater_votes(rater_votes) -> bool:
    """True when rater_votes is a non-empty list or non-empty dict."""
    if rater_votes is None:
        return False
    if isinstance(rater_votes, (list, dict)):
        return bool(rater_votes)
    return False


def _purer_is_unanimous_abstain(rater_votes) -> bool:
    """True when all entries in rater_votes are ABSTAIN (or the list is ABSTAIN-only).

    Handles both list-of-dicts and dict-of-dicts shapes that the PURER overlay
    may carry.
    """
    if not _has_rater_votes(rater_votes):
        return False
    if isinstance(rater_votes, list):
        entries = rater_votes
    elif isinstance(rater_votes, dict):
        entries = list(rater_votes.values())
    else:
        return False
    if not entries:
        return False
    for entry in entries:
        if not isinstance(entry, dict):
            return False
        # A vote of 'ABSTAIN' or a None primary_stage (with no 'CODED' vote) counts.
        if entry.get('vote') == 'ABSTAIN':
            continue
        if entry.get('vote') is None and entry.get('primary_stage') is None:
            continue
        return False
    return True


def _is_nonempty_list(value) -> bool:
    """True when value is a non-empty list (or JSON-decoded list)."""
    if isinstance(value, list):
        return bool(value)
    if isinstance(value, str):
        v = value.strip()
        return v not in ('', '[]', 'null')
    return False
