"""
analysis/irr_stats.py
---------------------
Inter-rater-reliability statistics — thin wrappers over PROVEN, third-party
libraries (no hand-rolled chance-corrected κ):

  * Cohen's κ        -> ``sklearn.metrics.cohen_kappa_score``       (scikit-learn)
  * Fleiss' κ        -> ``statsmodels.stats.inter_rater.fleiss_kappa`` (statsmodels)
  * Krippendorff's α -> ``krippendorff.alpha``                       (krippendorff)
  * confusion / P,R  -> ``sklearn.metrics``                          (scikit-learn)

These are the statistics cited in ``docs/methodology.md``.

Ballot encoding (matches ``process.irr_import``):
    int   -> VAAMR theme_id (0–4)
    -1    -> ABSTAIN ("No code"), treated as a 6th nominal category
    None  -> no ballot (missing); excluded by every estimator below.

Fleiss' κ assumes a fixed number of raters per subject, so it is computed only
over COMPLETE-CASE items (every roster rater cast a ballot); ``n_complete`` is
reported alongside it. Krippendorff's α natively tolerates missing ballots and
unequal rater counts, so it is the headline multi-rater statistic for these
test-sets (T2 in particular has only two raters per item).
"""

from typing import Any, Dict, List, Optional

ABSTAIN_CODE = -1


def _encode(b: Any) -> Optional[int]:
    """Ballot -> nominal int (ABSTAIN stays -1) or None (missing)."""
    if b is None:
        return None
    if b == 'ABSTAIN':
        return ABSTAIN_CODE
    try:
        return int(b)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Pairwise
# ---------------------------------------------------------------------------

def cohen_kappa(a: List[int], b: List[int]) -> Optional[float]:
    """Cohen's κ (scikit-learn) for two aligned integer label lists."""
    if len(a) < 2:
        return None
    try:
        from sklearn.metrics import cohen_kappa_score
        if len(set(a)) < 2 and len(set(b)) < 2 and set(a) == set(b):
            return 1.0  # both constant and identical -> perfect agreement
        return float(cohen_kappa_score(a, b))
    except Exception:
        return None


def observed_agreement(a: List[int], b: List[int]) -> Optional[float]:
    """Raw proportion of matching labels (descriptive, not chance-corrected)."""
    if not a:
        return None
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


# ---------------------------------------------------------------------------
# Multi-rater (matrix = list of rows, each row a list over raters)
# ---------------------------------------------------------------------------

def krippendorff_alpha(matrix: List[List[Any]]) -> Optional[float]:
    """Krippendorff's α (nominal) via the ``krippendorff`` package.

    Robust to missing ballots / unequal rater counts. Expects the package's
    reliability-data orientation: [n_raters x n_units] with np.nan for missing.
    """
    import numpy as np
    if not matrix:
        return None
    n_raters = max(len(r) for r in matrix)
    data = np.full((len(matrix), n_raters), np.nan, dtype=float)
    for i, row in enumerate(matrix):
        for j, b in enumerate(row):
            enc = _encode(b)
            if enc is not None:
                data[i, j] = enc
    rel = data.T  # raters x units
    if rel.shape[1] < 2:
        return None
    try:
        import krippendorff
        return float(krippendorff.alpha(reliability_data=rel,
                                        level_of_measurement='nominal'))
    except Exception:
        return None


def fleiss_kappa(matrix: List[List[Any]]) -> Dict[str, Any]:
    """Fleiss' κ (statsmodels) over COMPLETE-CASE items.

    Returns ``{'kappa': float|None, 'n_complete': int, 'n_raters': int}``.
    Only items where every rater in the row cast a ballot contribute (statsmodels'
    estimator assumes a fixed rater count). Use ``krippendorff_alpha`` for the
    missing-data-robust estimate.
    """
    if not matrix:
        return {'kappa': None, 'n_complete': 0, 'n_raters': 0}
    n_raters = max(len(r) for r in matrix)
    # Complete-case rows: every rater slot has a (non-missing) ballot.
    complete = []
    for row in matrix:
        enc = [_encode(b) for b in row]
        if len(enc) == n_raters and all(e is not None for e in enc):
            complete.append(enc)
    if len(complete) < 2:
        return {'kappa': None, 'n_complete': len(complete), 'n_raters': n_raters}

    categories = sorted({e for row in complete for e in row})
    cat_idx = {c: i for i, c in enumerate(categories)}
    import numpy as np
    table = np.zeros((len(complete), len(categories)), dtype=int)
    for i, row in enumerate(complete):
        for e in row:
            table[i, cat_idx[e]] += 1
    try:
        from statsmodels.stats.inter_rater import fleiss_kappa as _fk
        return {'kappa': float(_fk(table)), 'n_complete': len(complete),
                'n_raters': n_raters}
    except Exception:
        return {'kappa': None, 'n_complete': len(complete), 'n_raters': n_raters}


def unanimous_agreement(matrix: List[List[Any]]) -> Dict[str, float]:
    """Descriptive raw agreement across items with ≥2 ballots."""
    from itertools import combinations
    unanimous = 0
    pairwise_sum = 0.0
    n_valid = 0
    for row in matrix:
        enc = [_encode(b) for b in row if _encode(b) is not None]
        if len(enc) < 2:
            continue
        n_valid += 1
        if len(set(enc)) == 1:
            unanimous += 1
        pairs = list(combinations(enc, 2))
        if pairs:
            pairwise_sum += sum(1 for x, y in pairs if x == y) / len(pairs)
    return {
        'unanimous': unanimous / n_valid if n_valid else 0.0,
        'pairwise': pairwise_sum / n_valid if n_valid else 0.0,
        'n_items': n_valid,
    }


# ---------------------------------------------------------------------------
# Confusion / per-class
# ---------------------------------------------------------------------------

def confusion(human: List[int], machine: List[int], labels: List[int],
              label_names: List[str]) -> dict:
    """Confusion matrix + per-class precision/recall (scikit-learn)."""
    if not human:
        return {}
    try:
        from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
    except Exception:
        return {}
    cm = confusion_matrix(human, machine, labels=labels)
    prec, rec, _f, sup = precision_recall_fscore_support(
        human, machine, labels=labels, zero_division=0
    )
    per_class = {
        label_names[i]: {
            'precision': float(prec[i]),
            'recall': float(rec[i]),
            'support': int(sup[i]),
        }
        for i in range(len(labels))
    }
    return {
        'labels': labels,
        'label_names': label_names,
        'matrix': cm.tolist(),
        'per_class': per_class,
    }
