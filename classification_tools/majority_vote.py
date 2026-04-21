"""
majority_vote.py
----------------
Shared majority-voting logic for both single-label (theme) and
multi-label (codebook) classification runs.
"""

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple


def single_label_majority_vote(
    parsed_runs: List[Optional[Dict]],
    label_key: str = 'primary_stage',
    confidence_key: str = 'primary_confidence',
) -> Dict:
    """
    Determine how many of the *n* runs agree on the primary label.

    Replaces ``_compute_run_consistency()`` from
    ``theme_labeler/zero_shot_classifier.py``.

    Parameters
    ----------
    parsed_runs : list of dict or None
        Each dict must contain at least *label_key* and *confidence_key*.
    label_key : str
        Key holding the primary label value.
    confidence_key : str
        Key holding the primary confidence value.

    Returns
    -------
    dict
        primary_stage, consistency, confidence, secondary_stage,
        secondary_confidence, justification.
    """
    valid_runs = [
        r for r in parsed_runs
        if r is not None and r.get(label_key) is not None
    ]

    if not valid_runs:
        return {
            'primary_stage': None,
            'consistency': 0,
            'confidence': 0.0,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
        }

    primary_counts = Counter(r[label_key] for r in valid_runs)
    max_count = primary_counts.most_common(1)[0][1]
    tied_stages = [s for s, c in primary_counts.items() if c == max_count]

    tie_broken_by_confidence = False
    if len(tied_stages) > 1:
        majority_stage = max(
            tied_stages,
            key=lambda s: sum(r[confidence_key] for r in valid_runs if r[label_key] == s) / max_count,
        )
        tie_broken_by_confidence = True
    else:
        majority_stage = tied_stages[0]
    majority_count = max_count

    agreeing_runs = [r for r in valid_runs if r[label_key] == majority_stage]
    avg_confidence = sum(r[confidence_key] for r in agreeing_runs) / len(agreeing_runs)

    # Secondary stage: most common non-None secondary
    secondary_stages = [
        r['secondary_stage'] for r in valid_runs if r.get('secondary_stage') is not None
    ]
    secondary_stage = None
    secondary_confidence = None
    if secondary_stages:
        sec_counts = Counter(secondary_stages)
        secondary_stage = sec_counts.most_common(1)[0][0]
        sec_runs = [r for r in valid_runs if r.get('secondary_stage') == secondary_stage]
        sec_confs = [
            r['secondary_confidence'] for r in sec_runs
            if r.get('secondary_confidence') is not None
        ]
        if sec_confs:
            secondary_confidence = sum(sec_confs) / len(sec_confs)

    justification = agreeing_runs[0].get('justification', '') if agreeing_runs else ''

    return {
        'primary_stage': majority_stage,
        'consistency': majority_count,
        'confidence': avg_confidence,
        'secondary_stage': secondary_stage,
        'secondary_confidence': secondary_confidence,
        'justification': justification,
        'tie_broken_by_confidence': tie_broken_by_confidence,
    }


def multi_label_majority_vote(
    all_assignments: List[List[Any]],
    get_id: Callable[[Any], str] = lambda a: a.code_id,
    get_confidence: Callable[[Any], float] = lambda a: a.confidence,
) -> List[Tuple[Any, float]]:
    """
    Merge multi-label assignments across runs via majority voting.

    Replaces ``_merge_runs()`` from
    ``codebook_classifier/llm_classifier.py``.

    Parameters
    ----------
    all_assignments : list of list
        Each inner list holds assignment objects from one run.
    get_id : callable
        Extracts the identifier from an assignment object.
    get_confidence : callable
        Extracts the confidence from an assignment object.

    Returns
    -------
    list of (exemplar_assignment, avg_confidence) tuples
        One entry per code that appeared in a majority of runs.
        The exemplar is the first assignment seen for that code.
    """
    if not all_assignments:
        return []
    if len(all_assignments) == 1:
        return [(a, get_confidence(a)) for a in all_assignments[0]]

    code_counts: Dict[str, int] = {}
    code_confidences: Dict[str, List[float]] = {}
    code_exemplars: Dict[str, Any] = {}

    for assignments in all_assignments:
        for a in assignments:
            cid = get_id(a)
            code_counts[cid] = code_counts.get(cid, 0) + 1
            code_confidences.setdefault(cid, []).append(get_confidence(a))
            if cid not in code_exemplars:
                code_exemplars[cid] = a

    threshold = len(all_assignments) / 2
    merged = []
    for cid, count in code_counts.items():
        if count >= threshold:
            avg_conf = sum(code_confidences[cid]) / len(code_confidences[cid])
            merged.append((code_exemplars[cid], avg_conf))

    return merged
