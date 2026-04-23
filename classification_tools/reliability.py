"""
reliability.py
--------------
Interrater reliability (IRR) metrics for multi-rater categorical coding.

Given a list of per-segment ballot lists (one ballot per rater, in a stable
rater order), computes:

  - percent_agreement : raw (all-rater) agreement rate
  - fleiss_kappa      : Fleiss' kappa, appropriate for k raters on k categories
                        with missing ballots tolerated via per-subject n
  - krippendorff_alpha : Krippendorff's α (nominal) — handles missing ballots
                         and unequal rater counts per subject natively

Ballot encoding
---------------
Each segment is represented as a ``List[Optional[Union[int, str]]]`` whose
length equals the roster size. Entry ``i`` is the ballot from rater ``i``:

    int         -> coded theme id
    'ABSTAIN'   -> rater judged utterance irrelevant (real ballot)
    None        -> rater errored / no parseable response (excluded)

ABSTAIN is treated as a distinct nominal category (``-1``) so that two
raters who both abstain count as agreement.

These helpers operate purely on ballot matrices; converting Segment objects
into a ballot matrix is the caller's job — see ``ballots_from_segments``.
"""

from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np
import krippendorff

ABSTAIN_CATEGORY = -1


def _encode_ballot(b: Any) -> Optional[int]:
    """Encode one ballot into a nominal integer or None (missing)."""
    if b is None:
        return None
    if b == 'ABSTAIN':
        return ABSTAIN_CATEGORY
    if isinstance(b, (int, np.integer)):
        return int(b)
    try:
        return int(b)
    except (TypeError, ValueError):
        return None


def _encode_matrix(ballot_matrix: List[List[Any]]) -> np.ndarray:
    """Convert a [n_segments x n_raters] list-of-lists into an int array with
    np.nan for missing. Returns a float array so np.nan can coexist."""
    if not ballot_matrix:
        return np.zeros((0, 0), dtype=float)
    n_raters = max(len(row) for row in ballot_matrix)
    out = np.full((len(ballot_matrix), n_raters), np.nan, dtype=float)
    for i, row in enumerate(ballot_matrix):
        for j, b in enumerate(row):
            enc = _encode_ballot(b)
            if enc is not None:
                out[i, j] = enc
    return out


def percent_agreement(ballot_matrix: List[List[Any]]) -> Dict[str, float]:
    """
    Raw agreement rate.

    Returns dict with:
        all_agree  : fraction of segments where every non-missing rater cast
                     the same ballot (unanimous).
        pairwise   : mean fraction of rater pairs that agree, per segment,
                     averaged across segments.
        n_segments : number of segments contributing (at least 2 ballots).
    """
    unanimous = 0
    pairwise_sum = 0.0
    n_valid = 0
    for row in ballot_matrix:
        encoded = [_encode_ballot(b) for b in row if _encode_ballot(b) is not None]
        if len(encoded) < 2:
            continue
        n_valid += 1
        counts = Counter(encoded)
        top = counts.most_common(1)[0][1]
        if top == len(encoded):
            unanimous += 1
        # Pairwise agreement on this segment: sum_{c} C(count_c, 2) / C(n, 2)
        n = len(encoded)
        pairs_total = n * (n - 1) / 2
        if pairs_total > 0:
            agreeing_pairs = sum(c * (c - 1) / 2 for c in counts.values())
            pairwise_sum += agreeing_pairs / pairs_total

    return {
        'all_agree': unanimous / n_valid if n_valid else 0.0,
        'pairwise': pairwise_sum / n_valid if n_valid else 0.0,
        'n_segments': n_valid,
    }


def fleiss_kappa(ballot_matrix: List[List[Any]]) -> Optional[float]:
    """
    Fleiss' kappa for a fixed number of raters per subject.

    When ballots are missing (rater errors), each subject can have a
    different effective n. This implementation uses the generalized Fleiss
    κ (Conger 1980 extension) which tolerates unequal n per subject.

    Returns None if fewer than 2 segments have at least 2 ballots.
    """
    category_counts: List[Counter] = []
    for row in ballot_matrix:
        encoded = [_encode_ballot(b) for b in row if _encode_ballot(b) is not None]
        if len(encoded) < 2:
            continue
        category_counts.append(Counter(encoded))

    if len(category_counts) < 2:
        return None

    all_categories = set()
    for c in category_counts:
        all_categories.update(c.keys())
    categories = sorted(all_categories)
    if not categories:
        return None

    # Per-subject agreement P_i = (sum n_ij^2 - n_i) / (n_i (n_i - 1))
    P_is = []
    subject_ns = []
    category_totals = {c: 0.0 for c in categories}
    total_n = 0.0
    for counts in category_counts:
        n_i = sum(counts.values())
        if n_i < 2:
            continue
        subject_ns.append(n_i)
        total_n += n_i
        for c in categories:
            category_totals[c] += counts.get(c, 0)
        sum_sq = sum(counts.get(c, 0) ** 2 for c in categories)
        P_is.append((sum_sq - n_i) / (n_i * (n_i - 1)))

    if not P_is or total_n == 0:
        return None

    P_bar = sum(P_is) / len(P_is)
    p_j = {c: category_totals[c] / total_n for c in categories}
    P_e = sum(v ** 2 for v in p_j.values())

    if P_e >= 1.0:
        return 1.0 if P_bar >= 1.0 else 0.0
    return (P_bar - P_e) / (1.0 - P_e)


def krippendorff_alpha(
    ballot_matrix: List[List[Any]],
    level: str = 'nominal',
) -> Optional[float]:
    """
    Krippendorff's α (default: nominal level).

    Uses the reliability-data orientation of the ``krippendorff`` package:
    a [n_raters x n_segments] array with np.nan for missing ballots.
    """
    encoded = _encode_matrix(ballot_matrix)
    if encoded.size == 0:
        return None
    # Package expects raters-by-subjects.
    reliability_data = encoded.T
    if reliability_data.shape[1] < 2:
        return None
    try:
        return float(krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement=level,
        ))
    except Exception:
        return None


def ballots_from_segments(segments) -> List[List[Any]]:
    """Extract a ballot matrix from a list of Segments.

    Each segment contributes one row; each row is ordered by
    ``seg.rater_ids``. Missing/ERROR ballots become ``None``; abstentions
    become the string ``'ABSTAIN'``; coded ballots are integer stage IDs.

    Segments with no ``rater_votes`` are skipped.
    """
    matrix: List[List[Any]] = []
    for seg in segments:
        votes = getattr(seg, 'rater_votes', None)
        if not votes:
            continue
        row: List[Any] = []
        for rv in votes:
            v = rv.get('vote')
            if v == 'CODED':
                row.append(rv.get('stage'))
            elif v == 'ABSTAIN':
                row.append('ABSTAIN')
            else:
                row.append(None)
        matrix.append(row)
    return matrix


def compute_reliability(segments) -> Dict[str, Any]:
    """Convenience wrapper: one call returns all three metrics + metadata."""
    matrix = ballots_from_segments(segments)
    n_segments = len(matrix)
    rater_ids: List[str] = []
    for seg in segments:
        rids = getattr(seg, 'rater_ids', None)
        if rids and len(rids) > len(rater_ids):
            rater_ids = list(rids)

    agreement = percent_agreement(matrix)
    return {
        'n_segments': n_segments,
        'rater_ids': rater_ids,
        'n_raters': len(rater_ids),
        'percent_agreement_unanimous': agreement['all_agree'],
        'percent_agreement_pairwise': agreement['pairwise'],
        'fleiss_kappa': fleiss_kappa(matrix),
        'krippendorff_alpha_nominal': krippendorff_alpha(matrix, 'nominal'),
    }
