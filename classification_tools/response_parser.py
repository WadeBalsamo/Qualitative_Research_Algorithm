"""
response_parser.py
------------------
Translate raw LLM classification results into updated ``Segment`` objects.

Input shape (``results_all``): ``segment_id -> merge_result`` where each
``merge_result`` is the unified dict produced by
``llm_classifier.classify_segments_zero_shot``::

    {
      'rater_ids':   ['model_a', 'model_b', 'model_c'],
      'rater_votes': [ {rater, vote, stage, confidence, ...}, ... ],
      'consensus': {
        'primary_stage':      int | None,
        'primary_confidence': float,
        'secondary_stage':    int | None,
        'secondary_confidence': float | None,
        'justification':      str,
        'consensus_vote':     int | 'ABSTAIN' | None,
        'agreement_level':    'unanimous' | 'majority' | 'split' | 'none',
        'n_agree':            int,
        'n_ballots':          int,
        'n_raters':           int,
        'tie_broken_by_confidence': bool,
        'needs_review':       bool,
        'rater_votes':        [... same list as above ...],
      },
    }
"""

from typing import List, Dict, Any, Tuple

from .data_structures import Segment


class ErrorCategory:
    """Error taxonomy for LLM response classification."""
    API_ERROR = 'api_error'
    MALFORMED_JSON = 'malformed_json'
    ALL_RATERS_FAILED = 'all_raters_failed'
    ABSTAIN = 'abstain'
    SPLIT = 'split'


def parse_all_results(
    results_all: Dict[str, Any],
    segments: List[Segment],
    name_to_id: Dict[str, int],
) -> Tuple[List[Segment], Dict[str, Any]]:
    """
    Copy consensus + per-rater ballots onto the corresponding Segment objects.

    Parameters
    ----------
    results_all : dict
        Raw LLM merge results keyed by ``segment_id``.
    segments : list[Segment]
        Segment objects to update in place.
    name_to_id : dict
        Unused; retained for API compatibility with the previous parser
        (the new consensus already has integer stage IDs).
    """
    segment_lookup = {s.segment_id: s for s in segments}

    counts = {
        'classified': 0,
        'abstained': 0,
        'split': 0,
        'all_raters_failed': 0,
        'legacy_or_malformed': 0,
    }
    error_ids: Dict[str, List[str]] = {
        ErrorCategory.ABSTAIN: [],
        ErrorCategory.SPLIT: [],
        ErrorCategory.ALL_RATERS_FAILED: [],
        ErrorCategory.MALFORMED_JSON: [],
    }

    for seg_id, result in results_all.items():
        seg = segment_lookup.get(seg_id)
        if seg is None:
            continue

        if not isinstance(result, dict):
            error_ids[ErrorCategory.MALFORMED_JSON].append(seg_id)
            counts['legacy_or_malformed'] += 1
            continue

        consensus = result.get('consensus')
        if not isinstance(consensus, dict):
            error_ids[ErrorCategory.MALFORMED_JSON].append(seg_id)
            counts['legacy_or_malformed'] += 1
            continue

        rater_ids = result.get('rater_ids') or []
        rater_votes = result.get('rater_votes') or consensus.get('rater_votes') or []

        seg.rater_ids = list(rater_ids)
        seg.rater_votes = list(rater_votes)

        agreement_level = consensus.get('agreement_level')
        n_agree = consensus.get('n_agree', 0)
        n_raters = consensus.get('n_raters', len(rater_ids) or 1) or 1
        seg.agreement_level = agreement_level
        seg.agreement_fraction = (n_agree / n_raters) if n_raters else 0.0
        seg.needs_review = bool(consensus.get('needs_review', False))
        seg.consensus_vote = consensus.get('consensus_vote')
        seg.tie_broken_by_confidence = bool(
            consensus.get('tie_broken_by_confidence', False)
        )

        # llm_run_consistency is kept as an integer count of agreeing raters
        # for backwards-compat with downstream analysis code.
        seg.llm_run_consistency = n_agree

        primary_stage = consensus.get('primary_stage')
        consensus_vote = consensus.get('consensus_vote')

        if primary_stage is not None:
            seg.primary_stage = primary_stage
            seg.secondary_stage = consensus.get('secondary_stage')
            seg.llm_confidence_primary = consensus.get('primary_confidence', 0.0)
            seg.llm_confidence_secondary = consensus.get('secondary_confidence')
            seg.llm_justification = consensus.get('justification', '')
            seg.secondary_agreement_level = consensus.get('secondary_agreement_level')
            seg.secondary_agreement_fraction = consensus.get('secondary_agreement_fraction')
            counts['classified'] += 1
            continue

        # Unclassified — explain why.
        seg.primary_stage = None
        seg.secondary_stage = None
        seg.llm_confidence_primary = consensus.get('primary_confidence', 0.0)
        seg.llm_confidence_secondary = None
        seg.llm_justification = consensus.get('justification', '')

        if consensus_vote == 'ABSTAIN':
            error_ids[ErrorCategory.ABSTAIN].append(seg_id)
            counts['abstained'] += 1
        elif agreement_level == 'split':
            error_ids[ErrorCategory.SPLIT].append(seg_id)
            counts['split'] += 1
        elif agreement_level == 'none':
            error_ids[ErrorCategory.ALL_RATERS_FAILED].append(seg_id)
            counts['all_raters_failed'] += 1
        else:
            error_ids[ErrorCategory.MALFORMED_JSON].append(seg_id)
            counts['legacy_or_malformed'] += 1

    total = len(results_all)
    stats = {
        'total': total,
        'parsed': counts['classified'],
        'abstained': counts['abstained'],
        'split': counts['split'],
        'all_raters_failed': counts['all_raters_failed'],
        'malformed': counts['legacy_or_malformed'],
        'error_breakdown': {
            category: {'count': len(ids), 'segment_ids': ids}
            for category, ids in error_ids.items() if ids
        },
    }

    print(f"Parsing results: {counts['classified']}/{total} classified")
    if counts['abstained']:
        print(f"  Abstained (consensus=null): {counts['abstained']}")
    if counts['split']:
        print(f"  Split vote (needs review):  {counts['split']}")
    if counts['all_raters_failed']:
        print(f"  All raters failed:          {counts['all_raters_failed']}")
    if counts['legacy_or_malformed']:
        print(f"  Malformed / legacy format:  {counts['legacy_or_malformed']}")

    return segments, stats
