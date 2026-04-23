"""
majority_vote.py
----------------
Interrater-reliability vote aggregation for categorical (theme) and
multi-label (codebook) classification.

Design
------
Each rater returns one of three outcomes, encoded in the ``vote`` field
of the parsed run dict:

    'CODED'   — rater assigned a concrete theme ID
    'ABSTAIN' — rater judged the utterance irrelevant to the framework
                (JSON had primary_stage=null)
    'ERROR'   — the response could not be parsed (dict itself is None,
                or a sentinel dict with vote='ERROR')

ABSTAIN is a real ballot and is counted alongside the coded theme IDs
when determining the majority. ERROR ballots are excluded from the
denominator.

Unified voting
--------------
``vote_single_label`` is the single source of truth for both per-run
agreement (3 stochastic runs of one model) and per-model agreement
(3 distinct models, one run each). The caller provides rater identities
so the returned ``agreement_profile`` can be rendered in reports.
"""

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

ABSTAIN = 'ABSTAIN'   # Sentinel stage value for "irrelevant to study"

AGREEMENT_UNANIMOUS = 'unanimous'
AGREEMENT_MAJORITY = 'majority'
AGREEMENT_SPLIT = 'split'
AGREEMENT_NONE = 'none'       # No ballots (all raters errored)


def _vote_value(run: Optional[Dict]) -> Any:
    """Return the ballot for one run: stage id, ABSTAIN, or None (error)."""
    if run is None:
        return None
    v = run.get('vote')
    if v == 'ERROR':
        return None
    if v == 'ABSTAIN':
        return ABSTAIN
    if v == 'CODED':
        return run.get('primary_stage')
    # Legacy dicts without explicit vote field:
    if run.get('primary_stage') is None:
        return ABSTAIN
    return run.get('primary_stage')


def vote_single_label(
    parsed_runs: List[Optional[Dict]],
    rater_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Aggregate N raters' single-label ballots into one consensus result.

    Parameters
    ----------
    parsed_runs : list
        One entry per rater, in rater order. Each entry is either a
        parsed run dict (see module docstring) or None (hard parse
        failure / no response).
    rater_ids : list of str, optional
        Stable rater identifiers, same length as ``parsed_runs``. In the
        multi-model case these are model names; in the single-model
        case they are synthetic ``run_1``/``run_2``/...

    Returns
    -------
    dict
        A consensus result with the following keys:

            primary_stage        : int | None  (None when consensus is ABSTAIN or split)
            primary_confidence   : float (mean of agreeing raters, 0 when no majority)
            secondary_stage      : int | None
            secondary_confidence : float | None
            justification        : str (first agreeing rater's rationale)
            consensus_vote       : int | 'ABSTAIN' | None (None = split/no majority)
            agreement_level      : 'unanimous' | 'majority' | 'split' | 'none'
            n_agree              : int
            n_ballots            : int   (raters that produced a ballot)
            n_raters             : int   (len(parsed_runs))
            tie_broken_by_confidence : bool
            needs_review         : bool  (True when split or no ballots)
            rater_votes          : list of per-rater dicts (see below)

        rater_votes entries::

            {
              'rater': <rater_id>,
              'vote': 'CODED' | 'ABSTAIN' | 'ERROR',
              'stage': int | None,
              'confidence': float | None,
              'secondary_stage': int | None,
              'secondary_confidence': float | None,
              'justification': str,
            }
    """
    n_raters = len(parsed_runs)
    rater_ids = rater_ids or [f'run_{i + 1}' for i in range(n_raters)]

    # Build per-rater transparent records regardless of vote type.
    rater_votes: List[Dict[str, Any]] = []
    for rid, run in zip(rater_ids, parsed_runs):
        if run is None:
            rater_votes.append({
                'rater': rid,
                'vote': 'ERROR',
                'stage': None,
                'confidence': None,
                'secondary_stage': None,
                'secondary_confidence': None,
                'justification': '',
            })
            continue
        v = run.get('vote')
        if v is None:
            # Legacy dict — infer.
            v = 'ABSTAIN' if run.get('primary_stage') is None else 'CODED'
        rater_votes.append({
            'rater': rid,
            'vote': v,
            'stage': run.get('primary_stage'),
            'confidence': run.get('primary_confidence'),
            'secondary_stage': run.get('secondary_stage'),
            'secondary_confidence': run.get('secondary_confidence'),
            'justification': run.get('justification', '') or '',
        })

    # Ballots that count toward the vote: CODED + ABSTAIN.
    ballots: List[Tuple[str, Any, Optional[float], Dict]] = []
    for rv, run in zip(rater_votes, parsed_runs):
        if rv['vote'] == 'ERROR' or run is None:
            continue
        ballot_value = ABSTAIN if rv['vote'] == 'ABSTAIN' else rv['stage']
        if ballot_value is None:
            continue
        ballots.append((rv['rater'], ballot_value, rv['confidence'], run))

    n_ballots = len(ballots)

    if n_ballots == 0:
        return {
            'primary_stage': None,
            'primary_confidence': 0.0,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'consensus_vote': None,
            'agreement_level': AGREEMENT_NONE,
            'n_agree': 0,
            'n_ballots': 0,
            'n_raters': n_raters,
            'tie_broken_by_confidence': False,
            'needs_review': True,
            'rater_votes': rater_votes,
        }

    counts = Counter(b[1] for b in ballots)
    max_count = counts.most_common(1)[0][1]
    tied_values = [v for v, c in counts.items() if c == max_count]

    tie_broken_by_confidence = False
    if len(tied_values) > 1:
        # Prefer CODED stages over ABSTAIN when tied (qualitative coding
        # bias: we'd rather assign a label than drop the segment).
        coded_tied = [v for v in tied_values if v != ABSTAIN]
        candidates = coded_tied if coded_tied else tied_values

        if len(candidates) > 1:
            def avg_conf(val):
                confs = [c for _, v, c, _ in ballots
                         if v == val and c is not None]
                return sum(confs) / len(confs) if confs else 0.0
            winner = max(candidates, key=avg_conf)
            tie_broken_by_confidence = True
        else:
            winner = candidates[0]
            tie_broken_by_confidence = coded_tied != tied_values
    else:
        winner = tied_values[0]

    # Agreement level is defined over *raters*, not over ballots, so a
    # unanimous result requires every rater (incl. errors) to have cast
    # the winning ballot.
    if max_count == n_raters and n_ballots == n_raters:
        agreement_level = AGREEMENT_UNANIMOUS
    elif max_count > n_raters / 2:
        agreement_level = AGREEMENT_MAJORITY
    else:
        # No strict majority of raters — always split (even if the vote
        # technically resolved by tie-break above, it isn't a majority).
        agreement_level = AGREEMENT_SPLIT
        winner = None
        tie_broken_by_confidence = False

    # Confidence & justification from agreeing ballots.
    primary_confidence = 0.0
    justification = ''
    secondary_stage: Optional[int] = None
    secondary_confidence: Optional[float] = None

    if winner is not None:
        agreeing = [b for b in ballots if b[1] == winner]
        confs = [c for _, _, c, _ in agreeing if c is not None]
        primary_confidence = sum(confs) / len(confs) if confs else 0.0
        for _, _, _, run in agreeing:
            j = (run.get('justification') or '').strip()
            if j:
                justification = j
                break

        # Secondary: most common non-null secondary among agreeing raters.
        sec_vals = [run.get('secondary_stage') for _, _, _, run in agreeing
                    if run.get('secondary_stage') is not None]
        if sec_vals:
            sec_counts = Counter(sec_vals)
            top_sec_count = sec_counts.most_common(1)[0][1]
            # If multiple secondaries tied, pick the one with higher mean conf.
            sec_candidates = [s for s, c in sec_counts.items() if c == top_sec_count]
            if len(sec_candidates) > 1:
                def sec_avg_conf(s):
                    cs = [run.get('secondary_confidence') or 0.0
                          for _, _, _, run in agreeing
                          if run.get('secondary_stage') == s]
                    return sum(cs) / len(cs) if cs else 0.0
                secondary_stage = max(sec_candidates, key=sec_avg_conf)
            else:
                secondary_stage = sec_candidates[0]
            sec_confs = [run.get('secondary_confidence') for _, _, _, run in agreeing
                         if run.get('secondary_stage') == secondary_stage
                         and run.get('secondary_confidence') is not None]
            if sec_confs:
                secondary_confidence = sum(sec_confs) / len(sec_confs)

    # primary_stage in the returned dict is the int theme id, or None
    # when consensus is ABSTAIN or there is no majority. Downstream
    # (dataset_assembly) treats None as "unclassified".
    if winner == ABSTAIN:
        primary_stage_out = None
        consensus_vote = ABSTAIN
    else:
        primary_stage_out = winner
        consensus_vote = winner

    needs_review = (agreement_level == AGREEMENT_SPLIT
                    or agreement_level == AGREEMENT_NONE)

    return {
        'primary_stage': primary_stage_out,
        'primary_confidence': primary_confidence,
        'secondary_stage': secondary_stage,
        'secondary_confidence': secondary_confidence,
        'justification': justification,
        'consensus_vote': consensus_vote,
        'agreement_level': agreement_level,
        'n_agree': max_count,
        'n_ballots': n_ballots,
        'n_raters': n_raters,
        'tie_broken_by_confidence': tie_broken_by_confidence,
        'needs_review': needs_review,
        'rater_votes': rater_votes,
    }


def vote_multi_label(
    all_assignments: List[List[Any]],
    rater_ids: Optional[List[str]] = None,
    get_id: Callable[[Any], str] = lambda a: a.code_id,
    get_confidence: Callable[[Any], float] = lambda a: a.confidence,
    get_justification: Callable[[Any], str] = lambda a: getattr(a, 'justification', '') or '',
) -> Dict[str, Any]:
    """
    Aggregate multi-label codebook assignments across raters.

    A code is included in the consensus when at least a strict majority
    of raters assigned it (``count > n_raters / 2``). This is stricter
    than the previous ``>= n/2`` threshold, which let 1 of 2 raters win.

    Returns
    -------
    dict
        {
          'assignments': [(exemplar_assignment, mean_confidence), ...],
          'code_rater_votes': {code_id: [{rater, applied, confidence,
                                         justification}, ...], ...},
        }
    """
    n_raters = len(all_assignments)
    rater_ids = rater_ids or [f'run_{i + 1}' for i in range(n_raters)]

    if n_raters == 0:
        return {'assignments': [], 'code_rater_votes': {}}

    # Per-code, per-rater tracking.
    code_rater_votes: Dict[str, List[Dict[str, Any]]] = {}
    code_counts: Dict[str, int] = {}
    code_confidences: Dict[str, List[float]] = {}
    code_exemplars: Dict[str, Any] = {}

    for rid, assignments in zip(rater_ids, all_assignments):
        seen_codes = set()
        for a in assignments:
            cid = get_id(a)
            seen_codes.add(cid)
            code_counts[cid] = code_counts.get(cid, 0) + 1
            code_confidences.setdefault(cid, []).append(get_confidence(a))
            code_exemplars.setdefault(cid, a)
            code_rater_votes.setdefault(cid, []).append({
                'rater': rid,
                'applied': True,
                'confidence': get_confidence(a),
                'justification': get_justification(a),
            })
        # Note raters that did NOT apply each code-of-interest (filled in
        # after the loop, once we know which codes at least one rater
        # applied).

    # Backfill "applied: False" entries so consumers can display who
    # *rejected* each code.
    for cid, rater_list in code_rater_votes.items():
        seen_raters = {entry['rater'] for entry in rater_list}
        for rid in rater_ids:
            if rid not in seen_raters:
                rater_list.append({
                    'rater': rid,
                    'applied': False,
                    'confidence': None,
                    'justification': '',
                })

    threshold = n_raters / 2.0
    assignments_out: List[Tuple[Any, float]] = []
    for cid, count in code_counts.items():
        if count > threshold:
            confs = code_confidences[cid]
            assignments_out.append((code_exemplars[cid], sum(confs) / len(confs)))

    return {
        'assignments': assignments_out,
        'code_rater_votes': code_rater_votes,
    }
