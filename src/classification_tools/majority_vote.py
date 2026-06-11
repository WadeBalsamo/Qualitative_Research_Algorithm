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
denominator (``n_ballots``) entirely — a rater that failed to parse never
dilutes the vote of the raters that succeeded.

Vote modes
----------
``vote_single_label`` supports three ``vote_mode`` policies. All three
share the corrected denominator (``n_ballots`` = valid CODED+ABSTAIN
ballots, ERROR excluded) and the same secondary-evidence pooling; they
differ only in how a sub-majority / tie among the valid ballots resolves:

    'majority'        (default, conservative)
        A label is assigned only when one value holds a strict majority
        of the valid ballots (``max_count > n_ballots / 2``). Anything
        short of that — including ties broken by the CODED-preference /
        confidence chain — collapses to ``split`` with ``winner=None``.
        ``[CODED-P, ERROR, ERROR]`` → P (majority, 1/1); ``[P, ABSTAIN]``
        → split / unlabeled.

    'majority_coded'
        Identical to 'majority' when a strict majority exists, but when
        it does not the CODED-preference + mean-confidence tie-break
        resolves a winner instead of nullifying it. That winner is
        reported at agreement level ``plurality_coded`` with
        ``needs_review=True``. ``[P, ABSTAIN]`` → P (plurality_coded).

    'coded_plurality'  (PURER's monotone mode)
        The primary is decided among the CODED ballots ONLY. ABSTAIN is
        consensus only when there are zero valid CODED ballots and at
        least one ABSTAIN; level ``none`` only when there are zero valid
        ballots at all. The coded winner is the plurality count, broken
        in order by mean confidence → ``tie_break_order`` (an explicit
        stage-precedence list) → lowest stage id. Agreement level is
        ``unanimous`` (every rater coded the winner), ``majority``
        (> ``n_ballots`` / 2 of ALL valid ballots, abstains included),
        else ``plurality_coded``.

        **Hard invariant**: ``primary_stage is not None`` ⟺ at least one
        valid CODED ballot exists. Adding a rater therefore never turns a
        labeled segment unlabeled (monotonicity).

Unified voting
--------------
``vote_single_label`` is the single source of truth for both per-run
agreement (3 stochastic runs of one model) and per-model agreement
(3 distinct models, one run each). The caller provides rater identities
so the returned ``agreement_profile`` can be rendered in reports.
"""

from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

ABSTAIN = 'ABSTAIN'   # Sentinel stage value for "irrelevant to study"

AGREEMENT_UNANIMOUS = 'unanimous'
AGREEMENT_MAJORITY = 'majority'
AGREEMENT_SPLIT = 'split'
AGREEMENT_NONE = 'none'       # No ballots (all raters errored)
AGREEMENT_PLURALITY = 'plurality_coded'   # Resolved among coded ballots without a strict majority

VOTE_MODE_MAJORITY = 'majority'
VOTE_MODE_MAJORITY_CODED = 'majority_coded'
VOTE_MODE_CODED_PLURALITY = 'coded_plurality'


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


def _evidence_secondary(
    ballots: List[Tuple[str, Any, Optional[float], Dict]],
    winner: Any,
    secondary_weight: float,
    presence_threshold: float,
) -> Tuple[Optional[int], Optional[float]]:
    """
    Evidence-pooled secondary stage from all rater ballots.

    For each non-winner stage, sums contributions from:
      - Any rater's secondary vote for that stage (secondary_conf * weight)
      - Any dissenting rater's primary vote for that stage (primary_conf * weight)

    Returns (secondary_stage_id, mean_confidence) or (None, None) if no
    stage clears presence_threshold.
    """
    stage_evidence: Dict[Any, float] = defaultdict(float)
    stage_conf_sum: Dict[Any, float] = defaultdict(float)
    stage_conf_n: Dict[Any, int] = defaultdict(int)

    for _, val, conf, run in ballots:
        if val == ABSTAIN:
            continue
        primary_conf = conf or 0.0
        secondary_s = run.get('secondary_stage')
        secondary_c = run.get('secondary_confidence') or 0.0

        if val != winner and isinstance(val, int):
            stage_evidence[val] += primary_conf * secondary_weight
            stage_conf_sum[val] += primary_conf
            stage_conf_n[val] += 1

        if secondary_s is not None and secondary_s != winner and isinstance(secondary_s, int):
            stage_evidence[secondary_s] += secondary_c * secondary_weight
            stage_conf_sum[secondary_s] += secondary_c
            stage_conf_n[secondary_s] += 1

    if not stage_evidence:
        return None, None

    best_stage, best_evidence = max(stage_evidence.items(), key=lambda x: x[1])
    if best_evidence < presence_threshold:
        return None, None

    n = stage_conf_n[best_stage]
    s_conf = stage_conf_sum[best_stage] / n if n > 0 else None
    return best_stage, s_conf


def vote_single_label(
    parsed_runs: List[Optional[Dict]],
    rater_ids: Optional[List[str]] = None,
    secondary_weight: float = 0.6,
    presence_threshold: float = 0.5,
    vote_mode: str = VOTE_MODE_MAJORITY,
    tie_break_order: Optional[List[int]] = None,
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
    secondary_weight, presence_threshold : float
        Secondary-evidence pooling controls (see ``_evidence_secondary``).
    vote_mode : str
        ``'majority'`` (default, conservative), ``'majority_coded'``, or
        ``'coded_plurality'`` (PURER's monotone mode). See module
        docstring for the precise resolution of sub-majority ballots.
    tie_break_order : list of int, optional
        Stage-precedence list (highest precedence first) used only by
        ``'coded_plurality'`` after the confidence tie-break is itself
        tied. Stages not listed sort after listed ones; the final
        fallback is the lowest stage id.

    Returns
    -------
    dict
        A consensus result with the following keys:

            primary_stage        : int | None  (None when consensus is ABSTAIN or no label)
            primary_confidence   : float (mean of agreeing raters, 0 when no label)
            secondary_stage      : int | None
            secondary_confidence : float | None
            justification        : str (first agreeing rater's rationale)
            consensus_vote       : int | 'ABSTAIN' | None (None = split/no label)
            agreement_level      : 'unanimous' | 'majority' | 'plurality_coded' | 'split' | 'none'
            n_agree              : int
            n_ballots            : int   (raters that produced a CODED/ABSTAIN ballot)
            n_raters             : int   (len(parsed_runs))
            tie_broken_by_confidence : bool
            tie_broken_by_precedence : bool  (coded_plurality only; tie_break_order used)
            needs_review         : bool  (True when split, none, or plurality_coded)
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
            'secondary_agreement_level': None,
            'secondary_agreement_fraction': None,
            'justification': '',
            'consensus_vote': None,
            'agreement_level': AGREEMENT_NONE,
            'n_agree': 0,
            'n_ballots': 0,
            'n_raters': n_raters,
            'tie_broken_by_confidence': False,
            'tie_broken_by_precedence': False,
            'needs_review': True,
            'rater_votes': rater_votes,
        }

    def _avg_conf(val: Any) -> float:
        confs = [c for _, v, c, _ in ballots if v == val and c is not None]
        return sum(confs) / len(confs) if confs else 0.0

    tie_broken_by_confidence = False
    tie_broken_by_precedence = False

    if vote_mode == VOTE_MODE_CODED_PLURALITY:
        # Monotone mode: the primary is decided among CODED ballots only.
        # ABSTAIN is consensus only when no rater coded anything.
        coded_ballots = [b for b in ballots if b[1] != ABSTAIN]
        n_coded = len(coded_ballots)

        if n_coded == 0:
            # Every valid ballot is an ABSTAIN → ABSTAIN consensus.
            winner = ABSTAIN
            counts = Counter(b[1] for b in ballots)
            max_count = counts[ABSTAIN]
        else:
            counts = Counter(b[1] for b in coded_ballots)
            max_count = counts.most_common(1)[0][1]
            tied = sorted(v for v, c in counts.items() if c == max_count)
            if len(tied) == 1:
                winner = tied[0]
            else:
                # Plurality tie → mean confidence → explicit precedence → lowest id.
                best_conf = max(_avg_conf(v) for v in tied)
                conf_tied = [v for v in tied if _avg_conf(v) == best_conf]
                if len(conf_tied) == 1:
                    winner = conf_tied[0]
                    tie_broken_by_confidence = True
                else:
                    order = tie_break_order or []
                    rank = {sid: i for i, sid in enumerate(order)}
                    winner = min(
                        conf_tied,
                        key=lambda v: (rank.get(v, len(order)), v),
                    )
                    if any(v in rank for v in conf_tied):
                        tie_broken_by_precedence = True

        # Agreement level over ALL raters (errors included for unanimity;
        # abstains included in the majority denominator).
        if winner != ABSTAIN and max_count == n_raters and n_ballots == n_raters:
            agreement_level = AGREEMENT_UNANIMOUS
        elif max_count > n_ballots / 2:
            agreement_level = AGREEMENT_MAJORITY
        elif winner == ABSTAIN:
            # All-abstain that is not a strict majority of raters (errors
            # present) is still a clean ABSTAIN consensus, not a split.
            agreement_level = (AGREEMENT_UNANIMOUS
                               if (max_count == n_raters and n_ballots == n_raters)
                               else AGREEMENT_MAJORITY)
        else:
            agreement_level = AGREEMENT_PLURALITY
    else:
        counts = Counter(b[1] for b in ballots)
        max_count = counts.most_common(1)[0][1]
        tied_values = [v for v, c in counts.items() if c == max_count]

        if len(tied_values) > 1:
            # Prefer CODED stages over ABSTAIN when tied (qualitative coding
            # bias: we'd rather assign a label than drop the segment).
            coded_tied = [v for v in tied_values if v != ABSTAIN]
            candidates = coded_tied if coded_tied else tied_values

            if len(candidates) > 1:
                winner = max(candidates, key=_avg_conf)
                tie_broken_by_confidence = True
            else:
                winner = candidates[0]
                tie_broken_by_confidence = coded_tied != tied_values
        else:
            winner = tied_values[0]

        # Agreement level is defined over *raters*, not over ballots, so a
        # unanimous result requires every rater (incl. errors) to have cast
        # the winning ballot. The MAJORITY threshold uses ``n_ballots`` (valid
        # CODED+ABSTAIN ballots) as the denominator — ERROR ballots no longer
        # dilute the vote of the raters that succeeded.
        if max_count == n_raters and n_ballots == n_raters:
            agreement_level = AGREEMENT_UNANIMOUS
        elif max_count > n_ballots / 2:
            agreement_level = AGREEMENT_MAJORITY
        elif vote_mode == VOTE_MODE_MAJORITY_CODED and winner is not None and winner != ABSTAIN:
            # majority_coded: keep the CODED-preference / confidence winner
            # instead of nullifying it, flagged for review as a plurality.
            agreement_level = AGREEMENT_PLURALITY
        else:
            # No strict majority of valid ballots — split (the conservative
            # 'majority' mode nullifies any tie-break winner here).
            agreement_level = AGREEMENT_SPLIT
            winner = None
            tie_broken_by_confidence = False

    # Confidence & justification from agreeing ballots.
    primary_confidence = 0.0
    justification = ''
    secondary_stage: Optional[int] = None
    secondary_confidence: Optional[float] = None
    secondary_agreement_level: Optional[str] = None
    secondary_agreement_fraction: Optional[float] = None

    # Count raters (across all ballots) that assigned any secondary.
    n_with_secondary = sum(
        1 for _, _, _, run in ballots
        if run.get('vote') == 'CODED' and run.get('secondary_stage') is not None
    )

    if winner is not None:
        agreeing = [b for b in ballots if b[1] == winner]
        confs = [c for _, _, c, _ in agreeing if c is not None]
        primary_confidence = sum(confs) / len(confs) if confs else 0.0
        for _, _, _, run in agreeing:
            j = (run.get('justification') or '').strip()
            if j:
                justification = j
                break

        # Secondary: evidence pooling across ALL raters (agreeing + dissenting).
        if winner != ABSTAIN:
            secondary_stage, secondary_confidence = _evidence_secondary(
                ballots, winner, secondary_weight, presence_threshold
            )

        # Secondary agreement level: how many agreeing raters also assigned a secondary?
        if n_with_secondary > 0:
            n_agreeing_with_secondary = sum(
                1 for _, _, _, run in agreeing
                if run.get('secondary_stage') is not None
            )
            secondary_agreement_fraction = (
                n_agreeing_with_secondary / len(agreeing) if agreeing else 0.0
            )
            if n_agreeing_with_secondary == len(agreeing):
                secondary_agreement_level = 'unanimous'
            elif n_agreeing_with_secondary > len(agreeing) / 2:
                secondary_agreement_level = 'majority'
            else:
                secondary_agreement_level = 'partial'
    elif n_with_secondary > 0:
        # Split vote but some raters still assigned secondaries.
        secondary_agreement_level = 'split'
        secondary_agreement_fraction = n_with_secondary / n_ballots if n_ballots else 0.0

    # primary_stage in the returned dict is the int theme id, or None
    # when consensus is ABSTAIN or there is no majority. Downstream
    # (dataset_assembly) treats None as "unclassified".
    if winner == ABSTAIN:
        primary_stage_out = None
        consensus_vote = ABSTAIN
    else:
        primary_stage_out = winner
        consensus_vote = winner

    needs_review = agreement_level in (
        AGREEMENT_SPLIT, AGREEMENT_NONE, AGREEMENT_PLURALITY,
    )

    return {
        'primary_stage': primary_stage_out,
        'primary_confidence': primary_confidence,
        'secondary_stage': secondary_stage,
        'secondary_confidence': secondary_confidence,
        'secondary_agreement_level': secondary_agreement_level,
        'secondary_agreement_fraction': secondary_agreement_fraction,
        'justification': justification,
        'consensus_vote': consensus_vote,
        'agreement_level': agreement_level,
        'n_agree': max_count,
        'n_ballots': n_ballots,
        'n_raters': n_raters,
        'tie_broken_by_confidence': tie_broken_by_confidence,
        'tie_broken_by_precedence': tie_broken_by_precedence,
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
        for a in (assignments or []):
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
