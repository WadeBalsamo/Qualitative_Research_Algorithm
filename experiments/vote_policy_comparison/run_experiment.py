"""
experiments/vote_policy_comparison/run_experiment.py
------------------------------------------------------
Vote-policy comparison experiment (M0, plan §"Vote-policy experiment").

Re-votes stored ballots from MMORE_Processed (primary) and
MMORE_Processed_cohort2 (secondary) under four policies — 'legacy'
(buggy baseline), 'majority', 'majority_coded', 'coded_plurality' — and
scores each against human-consensus IRR codes.

**No LLM calls.** All data is read from SQLite / JSONL on disk (read-only).

Decision rule (from plan): highest κ; tie (within 0.005) → higher coverage
(fewer unlabeled).

Outputs (written to experiments/vote_policy_comparison/):
  results.json   — raw numbers for downstream reference
  RESULTS.md     — human-readable tables + DECISION section

Usage:
  python experiments/vote_policy_comparison/run_experiment.py
  python experiments/vote_policy_comparison/run_experiment.py --data ./data/MMORE_Processed
"""

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

# --- bootstrap src/ onto sys.path so imports work when run from repo root -----
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_SRC = os.path.join(_ROOT, 'src')
for _p in (_SRC, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
from analysis import irr_stats
from classification_tools.majority_vote import (
    ABSTAIN,
    AGREEMENT_NONE,
    AGREEMENT_SPLIT,
    AGREEMENT_PLURALITY,
    VOTE_MODE_MAJORITY,
    VOTE_MODE_MAJORITY_CODED,
    VOTE_MODE_CODED_PLURALITY,
    vote_single_label,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ABSTAIN_CODE = -1   # irr_stats sentinel for "No code" (mirrors irr_stats.ABSTAIN_CODE)
POLICIES = ['legacy', 'majority', 'majority_coded', 'coded_plurality']

BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 20260610   # mirrors irr_analysis.BOOTSTRAP_SEED

# VAAMR stage names (for per-class recall table)
STAGE_NAMES = {
    0: 'Vigilance',
    1: 'Avoidance',
    2: 'AttentionReg',
    3: 'Metacognition',
    4: 'Reappraisal',
    ABSTAIN_CODE: 'No-code',
}

# ---------------------------------------------------------------------------
# Legacy vote (pre-M0, buggy denominator: n_raters includes ERROR ballots)
# ---------------------------------------------------------------------------

def _vote_legacy(parsed_runs: List[Optional[Dict]]) -> Optional[int]:
    """
    Replicate the OLD majority_vote decision core (pre-M0).

    Key bug: threshold uses ``len(parsed_runs)`` (n_raters, including ERROR)
    as the denominator.  ``[CODED, ERROR, ERROR]`` → max_count=1, threshold=1.5,
    1 > 1.5 is False → split → None.

    Returns the consensus stage int, ABSTAIN_CODE, or None (split / no label).
    """
    n_raters = len(parsed_runs)

    # Build ballots (CODED + ABSTAIN only — same as new code).
    ballots = []
    for run in parsed_runs:
        if run is None:
            continue
        v = run.get('vote')
        if v == 'ERROR':
            continue
        if v == 'ABSTAIN' or (v is None and run.get('primary_stage') is None):
            ballots.append(ABSTAIN)
        elif v == 'CODED' or (v is None and run.get('primary_stage') is not None):
            stage = run.get('primary_stage')
            if stage is not None:
                ballots.append(stage)

    if not ballots:
        return None  # all errors

    counts = Counter(ballots)
    max_count = counts.most_common(1)[0][1]
    tied_values = [v for v, c in counts.items() if c == max_count]

    # Tie-break: prefer CODED over ABSTAIN; break by mean confidence.
    def avg_conf(val):
        confs = []
        for run in parsed_runs:
            if run is None:
                continue
            if run.get('vote') == 'ERROR':
                continue
            vote_val = (ABSTAIN if (run.get('vote') == 'ABSTAIN' or
                        (run.get('vote') is None and run.get('primary_stage') is None))
                        else run.get('primary_stage'))
            if vote_val == val:
                c = run.get('primary_confidence')
                if c is not None:
                    confs.append(c)
        return sum(confs) / len(confs) if confs else 0.0

    if len(tied_values) > 1:
        coded_tied = [v for v in tied_values if v != ABSTAIN]
        candidates = coded_tied if coded_tied else tied_values
        if len(candidates) > 1:
            winner = max(candidates, key=avg_conf)
        else:
            winner = candidates[0]
    else:
        winner = tied_values[0]

    # BUG: denominator is n_raters (includes ERROR), not n_ballots.
    if max_count == n_raters:
        pass  # unanimous
    elif max_count > n_raters / 2:
        pass  # majority
    else:
        # Sub-majority → split → no label.
        return None

    if winner == ABSTAIN:
        return ABSTAIN_CODE
    return winner


# ---------------------------------------------------------------------------
# Policy wrapper — new modes use vote_single_label
# ---------------------------------------------------------------------------

def _vote_policy(policy: str, parsed_runs: List[Optional[Dict]]) -> Optional[int]:
    """
    Apply a vote policy to a list of parsed rater dicts.

    Returns the winning stage (int ≥ 0), ABSTAIN_CODE (-1), or None (unlabeled).
    """
    if policy == 'legacy':
        return _vote_legacy(parsed_runs)

    mode_map = {
        'majority': VOTE_MODE_MAJORITY,
        'majority_coded': VOTE_MODE_MAJORITY_CODED,
        'coded_plurality': VOTE_MODE_CODED_PLURALITY,
    }
    result = vote_single_label(parsed_runs, vote_mode=mode_map[policy])
    cv = result['consensus_vote']
    if cv == ABSTAIN:
        return ABSTAIN_CODE
    if cv is None:
        return None
    return int(cv)


# ---------------------------------------------------------------------------
# Bootstrap CI for Cohen κ (mirrors irr_analysis._bootstrap_kappa_ci)
# ---------------------------------------------------------------------------

def _bootstrap_kappa_ci(h: List[int], m: List[int]) -> Optional[Dict]:
    if not h or len(h) < 2:
        return None
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    h_arr = np.asarray(h)
    m_arr = np.asarray(m)
    n = len(h)
    stats_arr = []
    for _ in range(BOOTSTRAP_REPS):
        idx = rng.integers(0, n, size=n)
        k = irr_stats.cohen_kappa(h_arr[idx].tolist(), m_arr[idx].tolist())
        if k is not None:
            stats_arr.append(k)
    if len(stats_arr) < 2:
        return None
    arr = np.asarray(stats_arr)
    return {
        'point': irr_stats.cohen_kappa(h, m),
        'lo': float(np.percentile(arr, 2.5)),
        'hi': float(np.percentile(arr, 97.5)),
        'n_boot': len(stats_arr),
    }


# ---------------------------------------------------------------------------
# A. VAAMR κ comparison
# ---------------------------------------------------------------------------

def _load_human_consensus_db(db_path: str) -> Dict[str, int]:
    """
    Load human-consensus IRR codes from a SQLite qra.db (read-only).

    Mirrors _consensus_rows() + source filter from irr_analysis.py.
    Returns {segment_id: primary_code (int or ABSTAIN_CODE)}.
    """
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT segment_id, prim
            FROM irr_human_codes
            WHERE is_consensus = 1
              AND segment_id IS NOT NULL
              AND lower(source) != 'unresolved'
            """
        ).fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        code = r['prim']
        if code is None:
            code = ABSTAIN_CODE
        out[r['segment_id']] = int(code)
    return out


def _load_theme_ballots_db(db_path: str) -> Dict[str, List[Optional[Dict]]]:
    """
    Load per-segment rater_votes from theme_labels in SQLite (read-only).

    Returns {segment_id: [parsed_run_dict, ...]}.
    """
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            'SELECT segment_id, rater_votes FROM theme_labels WHERE rater_votes IS NOT NULL'
        ).fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        rv = json.loads(r['rater_votes'])
        # Convert rater_votes list to parsed_run dicts (vote_single_label format).
        parsed = []
        for entry in rv:
            vote = entry.get('vote')
            if vote == 'ERROR' or entry.get('stage') is None and vote not in ('ABSTAIN', 'CODED'):
                # Treat as None (hard parse failure)
                parsed.append(None if vote == 'ERROR' else {
                    'vote': 'ABSTAIN',
                    'primary_stage': None,
                    'primary_confidence': entry.get('confidence'),
                    'secondary_stage': entry.get('secondary_stage'),
                    'secondary_confidence': entry.get('secondary_confidence'),
                    'justification': entry.get('justification', ''),
                })
            else:
                parsed.append({
                    'vote': vote,
                    'primary_stage': entry.get('stage'),
                    'primary_confidence': entry.get('confidence'),
                    'secondary_stage': entry.get('secondary_stage'),
                    'secondary_confidence': entry.get('secondary_confidence'),
                    'justification': entry.get('justification', ''),
                })
        out[r['segment_id']] = parsed
    return out


def _score_policy_vs_human(
    policy: str,
    human: Dict[str, int],
    ballots: Dict[str, List],
) -> Dict[str, Any]:
    """
    Re-vote each segment under policy; score against human consensus.

    Returns a dict with κ, CI, %agree, n_items, n_unlabeled, coverage,
    per_class_recall.
    """
    h_list = []
    m_list = []
    n_unlabeled = 0
    n_scored = 0

    for seg_id, human_code in human.items():
        seg_ballots = ballots.get(seg_id)
        if seg_ballots is None:
            # No machine ballots for this segment — skip (excluded from κ).
            n_unlabeled += 1
            continue

        pred = _vote_policy(policy, seg_ballots)
        n_scored += 1
        if pred is None:
            n_unlabeled += 1
            continue
        h_list.append(human_code)
        m_list.append(pred)

    n_human = len(human)
    n_with_ballots = sum(1 for s in human if s in ballots)
    n_policy_labeled = len(h_list)
    coverage = n_policy_labeled / n_with_ballots if n_with_ballots > 0 else 0.0

    kappa = irr_stats.cohen_kappa(h_list, m_list)
    ci = _bootstrap_kappa_ci(h_list, m_list)
    pct_agree = irr_stats.observed_agreement(h_list, m_list)

    # Per-class recall (human is ground truth).
    present_labels = sorted(set(h_list) | set(m_list)) if h_list else []
    per_class: Dict[str, Dict] = {}
    if h_list:
        for label in present_labels:
            n_true = sum(1 for x in h_list if x == label)
            n_correct = sum(1 for x, y in zip(h_list, m_list) if x == label and y == label)
            recall = n_correct / n_true if n_true > 0 else 0.0
            per_class[STAGE_NAMES.get(label, str(label))] = {
                'recall': recall,
                'n_support': n_true,
                'n_correct': n_correct,
            }

    return {
        'policy': policy,
        'n_human_items': n_human,
        'n_with_ballots': n_with_ballots,
        'n_scored': n_scored,
        'n_labeled': n_policy_labeled,
        'n_unlabeled': n_unlabeled,
        'coverage': coverage,
        'kappa': kappa,
        'kappa_ci': ci,
        'pct_agree': pct_agree,
        'per_class': per_class,
    }


def run_vaamr_comparison(db_path: str, label: str) -> Dict[str, Any]:
    """Run the full VAAMR κ comparison for one dataset. Returns results dict."""
    print(f'\n=== VAAMR κ comparison: {label} ===')

    human = _load_human_consensus_db(db_path)
    print(f'  Human consensus items (non-unresolved): {len(human)}')
    if not human:
        return {'dataset': label, 'note': 'no human IRR codes', 'policies': {}}

    ballots = _load_theme_ballots_db(db_path)
    print(f'  Segments with machine ballots: {len(ballots)}')

    results = {}
    for policy in POLICIES:
        r = _score_policy_vs_human(policy, human, ballots)
        results[policy] = r
        kappa_str = f'{r["kappa"]:.3f}' if r['kappa'] is not None else 'N/A'
        ci = r['kappa_ci']
        ci_str = f'[{ci["lo"]:.3f}, {ci["hi"]:.3f}]' if ci else '(no CI)'
        print(f'  {policy:20s}  κ={kappa_str} {ci_str}  coverage={r["coverage"]:.3f}'
              f'  n={r["n_labeled"]}  unlabeled={r["n_unlabeled"]}')

    return {'dataset': label, 'db_path': db_path, 'policies': results}


# ---------------------------------------------------------------------------
# B. PURER coverage comparison (no human codes; coverage-only)
# ---------------------------------------------------------------------------

def _load_purer_ballots_db(db_path: str) -> Dict[str, List[Optional[Dict]]]:
    """Load purer rater_votes from SQLite (read-only)."""
    con = sqlite3.connect(f'file:{db_path}?mode=ro', uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            'SELECT segment_id, purer_rater_votes FROM purer_labels WHERE purer_rater_votes IS NOT NULL'
        ).fetchall()
    finally:
        con.close()
    out = {}
    for r in rows:
        rv = json.loads(r['purer_rater_votes'])
        parsed = []
        for entry in rv:
            vote = entry.get('vote')
            parsed.append({
                'vote': vote,
                'primary_stage': entry.get('stage'),
                'primary_confidence': entry.get('confidence'),
                'secondary_stage': entry.get('secondary_stage'),
                'secondary_confidence': entry.get('secondary_confidence'),
                'justification': entry.get('justification', ''),
            } if vote != 'ERROR' else None)
        out[r['segment_id']] = parsed
    return out


def _load_purer_ballots_jsonl(jsonl_path: str) -> Dict[str, List[Optional[Dict]]]:
    """Load purer rater_votes from a legacy JSONL file (cohort2)."""
    out = {}
    with open(jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rv = row.get('purer_rater_votes')
            if not rv:
                continue
            parsed = []
            for entry in rv:
                vote = entry.get('vote')
                parsed.append({
                    'vote': vote,
                    'primary_stage': entry.get('stage'),
                    'primary_confidence': entry.get('confidence'),
                    'secondary_stage': entry.get('secondary_stage'),
                    'secondary_confidence': entry.get('secondary_confidence'),
                    'justification': entry.get('justification', ''),
                } if vote != 'ERROR' else None)
            out[row['segment_id']] = parsed
    return out


def _purer_coverage_for_policy(
    policy: str,
    ballots: Dict[str, List],
) -> Dict[str, Any]:
    """Re-vote all PURER segments under policy; report coverage breakdown."""
    n_labeled = 0
    n_abstain = 0
    n_unlabeled = 0
    n_with_secondary = 0

    for seg_id, segs_ballots in ballots.items():
        pred = _vote_policy(policy, segs_ballots)
        if pred is None:
            n_unlabeled += 1
        elif pred == ABSTAIN_CODE:
            n_abstain += 1
        else:
            n_labeled += 1

        # Count secondary in the new-mode result (only for non-legacy).
        if policy != 'legacy':
            mode_map = {
                'majority': VOTE_MODE_MAJORITY,
                'majority_coded': VOTE_MODE_MAJORITY_CODED,
                'coded_plurality': VOTE_MODE_CODED_PLURALITY,
            }
            result = vote_single_label(segs_ballots, vote_mode=mode_map[policy])
            if result.get('secondary_stage') is not None:
                n_with_secondary += 1

    total = len(ballots)
    return {
        'policy': policy,
        'n_total': total,
        'n_labeled': n_labeled,
        'n_abstain_consensus': n_abstain,
        'n_unlabeled': n_unlabeled,
        'n_with_secondary': n_with_secondary if policy != 'legacy' else None,
        'coverage': (n_labeled + n_abstain) / total if total > 0 else 0.0,
    }


def _count_label_flips(
    ballots: Dict[str, List],
    policy_a: str,
    policy_b: str,
) -> Dict[str, int]:
    """Count segments that flip labeled↔unlabeled between two policies."""
    flip_to_labeled = 0
    flip_to_unlabeled = 0
    unchanged = 0
    for seg_id, seg_ballots in ballots.items():
        pred_a = _vote_policy(policy_a, seg_ballots)
        pred_b = _vote_policy(policy_b, seg_ballots)
        a_labeled = pred_a is not None
        b_labeled = pred_b is not None
        if a_labeled == b_labeled:
            unchanged += 1
        elif b_labeled:
            flip_to_labeled += 1
        else:
            flip_to_unlabeled += 1
    return {
        'unchanged': unchanged,
        'flip_to_labeled': flip_to_labeled,
        'flip_to_unlabeled': flip_to_unlabeled,
    }


def run_purer_comparison(
    db_path_or_jsonl: str,
    label: str,
    source_type: str = 'sqlite',
    n_raters_note: str = '',
) -> Dict[str, Any]:
    """Run PURER coverage comparison for one dataset."""
    print(f'\n=== PURER coverage comparison: {label} ===')

    if source_type == 'sqlite':
        ballots = _load_purer_ballots_db(db_path_or_jsonl)
    else:
        ballots = _load_purer_ballots_jsonl(db_path_or_jsonl)

    print(f'  PURER segments with rater_votes: {len(ballots)}'
          + (f'  ({n_raters_note})' if n_raters_note else ''))

    coverage = {}
    for policy in POLICIES:
        r = _purer_coverage_for_policy(policy, ballots)
        coverage[policy] = r
        print(f'  {policy:20s}  labeled={r["n_labeled"]:3d}  abstain={r["n_abstain_consensus"]:3d}'
              f'  unlabeled={r["n_unlabeled"]:3d}  coverage={r["coverage"]:.3f}')

    # Flip counts: legacy vs each new policy (for the 2-rater cohort2 data this
    # is the most informative; for 1-rater DB1 there are no flips possible).
    flips = {}
    for policy in ['majority', 'majority_coded', 'coded_plurality']:
        flips[policy] = _count_label_flips(ballots, 'legacy', policy)

    return {
        'dataset': label,
        'source_type': source_type,
        'n_raters_note': n_raters_note,
        'policies': coverage,
        'flips_vs_legacy': flips,
    }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

def _decide(vaamr_results: List[Dict]) -> Tuple[str, str]:
    """
    Apply the decision rule: highest κ across datasets; tie (within 0.005)
    → higher coverage.  Only VAAMR κ is used (PURER is coverage-only).

    Returns (winner_policy, rationale_str).
    """
    # Gather mean κ per policy across datasets that have human codes.
    datasets_with_kappa = [r for r in vaamr_results if r.get('policies')]
    if not datasets_with_kappa:
        return 'majority', 'no human codes available; defaulting to majority'

    # Candidate policies: exclude 'legacy' (baseline; not eligible as default).
    CANDIDATE_POLICIES = [p for p in POLICIES if p != 'legacy']

    # For each policy, collect κ values from all datasets that have them.
    policy_kappas: Dict[str, List[float]] = {p: [] for p in CANDIDATE_POLICIES}
    policy_coverages: Dict[str, List[float]] = {p: [] for p in CANDIDATE_POLICIES}
    for res in datasets_with_kappa:
        for p in CANDIDATE_POLICIES:
            pdata = res['policies'].get(p, {})
            k = pdata.get('kappa')
            cov = pdata.get('coverage')
            if k is not None:
                policy_kappas[p].append(k)
            if cov is not None:
                policy_coverages[p].append(cov)

    mean_kappa = {p: (sum(ks) / len(ks) if ks else None) for p, ks in policy_kappas.items()}
    mean_cov = {p: (sum(cs) / len(cs) if cs else None) for p, cs in policy_coverages.items()}

    best_k = max((v for v in mean_kappa.values() if v is not None), default=None)
    if best_k is None:
        return 'majority', 'all κ values are None'

    candidates = [p for p in CANDIDATE_POLICIES if mean_kappa.get(p) is not None
                  and abs(mean_kappa[p] - best_k) <= 0.005]
    if len(candidates) == 1:
        winner = candidates[0]
        rationale = (f'highest mean κ={mean_kappa[winner]:.3f} '
                     f'(coverage={mean_cov.get(winner, 0):.3f})')
    else:
        # Tie → pick highest coverage.
        winner = max(candidates, key=lambda p: mean_cov.get(p) or 0.0)
        kappas_str = ', '.join(f'{p}={mean_kappa[p]:.3f}' for p in candidates)
        rationale = (f'tie within 0.005 among [{kappas_str}]; '
                     f'winner by coverage={mean_cov.get(winner, 0):.3f}')

    return winner, rationale


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _fmt_kappa(r: Dict) -> str:
    k = r.get('kappa')
    ci = r.get('kappa_ci')
    if k is None:
        return '    N/A      '
    ci_str = f'[{ci["lo"]:.3f}, {ci["hi"]:.3f}]' if ci else '[    N/A    ]'
    return f'{k:.3f} {ci_str}'


def _fmt_cov(r: Dict) -> str:
    return f'{r["coverage"]:.3f} ({r["n_labeled"]}/{r["n_with_ballots"]})'


def _per_class_table(vaamr_result: Dict) -> str:
    """Format per-class recall table for one dataset."""
    # Collect all labels across all policies.
    all_labels: set = set()
    for p in POLICIES:
        pdata = vaamr_result['policies'].get(p, {})
        all_labels.update(pdata.get('per_class', {}).keys())

    if not all_labels:
        return '  (no per-class data)\n'

    label_order = [STAGE_NAMES[k] for k in sorted(STAGE_NAMES.keys()) if STAGE_NAMES[k] in all_labels]
    remaining = sorted(all_labels - set(label_order))
    label_order += remaining

    header = f'  {"Label":15s}' + ''.join(f'  {p:18s}' for p in POLICIES)
    lines = [header, '  ' + '-' * (15 + 20 * len(POLICIES))]
    for label in label_order:
        row = f'  {label:15s}'
        for p in POLICIES:
            pdata = vaamr_result['policies'].get(p, {})
            pc = pdata.get('per_class', {}).get(label)
            if pc:
                row += f'  {pc["recall"]:5.3f} (n={pc["n_support"]:2d}){" ":4s}'
            else:
                row += f'  {"—":>18s}'
        lines.append(row)
    return '\n'.join(lines) + '\n'


def _write_results_md(
    vaamr_results: List[Dict],
    purer_results: List[Dict],
    winner: str,
    rationale: str,
    out_dir: str,
) -> None:
    lines = []
    lines.append('# Vote-Policy Comparison — Results')
    lines.append('')
    lines.append('Experiment: re-vote stored ballots under four policies and score against')
    lines.append('human-consensus IRR codes. No LLM calls; pure re-voting of stored ballots.')
    lines.append('')
    lines.append('Policies:')
    lines.append('  legacy          — pre-M0 buggy baseline (denominator includes ERROR ballots)')
    lines.append('  majority        — strict majority of valid ballots; sub-majority → unlabeled')
    lines.append('  majority_coded  — like majority but sub-majority resolves by CODED-preference')
    lines.append('                    + confidence tie-break (labeled, flagged for review)')
    lines.append('  coded_plurality — among CODED ballots only; ABSTAIN only if no CODED ballot')
    lines.append('                    (monotone: adding a rater never unlabels a segment)')
    lines.append('')

    # ------------------------------------------------------------------
    # Section A: VAAMR κ per dataset
    # ------------------------------------------------------------------
    lines.append('---')
    lines.append('## A. VAAMR κ comparison (human-consensus IRR)')
    lines.append('')

    for res in vaamr_results:
        dset = res['dataset']
        if not res.get('policies'):
            lines.append(f'### {dset}')
            lines.append(f'  Note: {res.get("note", "no data")}')
            lines.append('')
            continue

        lines.append(f'### {dset}')
        lines.append('')
        # Table header
        lines.append(f'  {"Policy":20s}  {"κ (95% CI)":35s}  {"% agree":8s}'
                     f'  {"n_scored":8s}  {"n_unlabeled":11s}  {"coverage":8s}')
        lines.append('  ' + '-' * 100)
        for p in POLICIES:
            r = res['policies'][p]
            kappa_str = _fmt_kappa(r)
            pct_str = f'{r["pct_agree"]:.3f}' if r['pct_agree'] is not None else 'N/A'
            lines.append(f'  {p:20s}  {kappa_str}  {pct_str:8s}'
                         f'  {r["n_labeled"]:8d}  {r["n_unlabeled"]:11d}'
                         f'  {_fmt_cov(r):20s}')
        lines.append('')

        # Per-class recall table.
        lines.append('  Per-class recall (human ground truth):')
        lines.append(_per_class_table(res))

    # ------------------------------------------------------------------
    # Section B: PURER coverage
    # ------------------------------------------------------------------
    lines.append('---')
    lines.append('## B. PURER coverage comparison (no human codes; coverage-only)')
    lines.append('')

    for res in purer_results:
        dset = res['dataset']
        n_note = res.get('n_raters_note', '')
        lines.append(f'### {dset}' + (f'  ({n_note})' if n_note else ''))
        lines.append('')
        lines.append(f'  {"Policy":20s}  {"n_labeled":9s}  {"n_abstain":9s}'
                     f'  {"n_unlabeled":11s}  {"coverage":8s}  {"n_with_secondary":16s}')
        lines.append('  ' + '-' * 90)
        for p in POLICIES:
            r = res['policies'][p]
            sec_str = str(r['n_with_secondary']) if r['n_with_secondary'] is not None else 'N/A'
            lines.append(f'  {p:20s}  {r["n_labeled"]:9d}  {r["n_abstain_consensus"]:9d}'
                         f'  {r["n_unlabeled"]:11d}  {r["coverage"]:8.3f}  {sec_str:>16s}')
        lines.append('')

        # Flip table vs legacy.
        flips = res.get('flips_vs_legacy', {})
        if flips:
            lines.append('  Labeled↔unlabeled flips vs legacy:')
            lines.append(f'  {"Policy":20s}  {"→labeled":10s}  {"→unlabeled":12s}  {"unchanged":10s}')
            lines.append('  ' + '-' * 60)
            for p in ['majority', 'majority_coded', 'coded_plurality']:
                f = flips.get(p, {})
                lines.append(f'  {p:20s}  {f.get("flip_to_labeled", 0):10d}'
                              f'  {f.get("flip_to_unlabeled", 0):12d}'
                              f'  {f.get("unchanged", 0):10d}')
        lines.append('')

    # ------------------------------------------------------------------
    # Section C: Decision
    # ------------------------------------------------------------------
    lines.append('---')
    lines.append('## DECISION')
    lines.append('')
    lines.append(f'**Winner: `{winner}`**')
    lines.append('')
    lines.append(f'Rationale: {rationale}')
    lines.append('')
    lines.append('Default changes applied:')
    lines.append(f'  - `src/constructs/config.py` `ThemeClassificationConfig.vote_mode` → `{winner}`')
    lines.append(f'  - `src/process/config.py` `purer_classification` default: `vote_mode={winner}`')
    lines.append('')

    # ------------------------------------------------------------------
    # Limitations
    # ------------------------------------------------------------------
    lines.append('---')
    lines.append('## Limitations')
    lines.append('')
    lines.append('1. κ is measured on the VAAMR IRR testset items only (MMORE_Processed, n=66')
    lines.append('   usable consensus items); the testset is also the reporting sample, so there')
    lines.append('   is selection overlap — κ values are descriptive, not held-out.')
    lines.append('2. PURER vote policy is unmeasurable by κ (no human PURER codes); coverage')
    lines.append('   is the only available proxy. The winner is inherited from VAAMR.')
    lines.append('3. MMORE_Processed_cohort2 has no SQLite qra.db (pre-migration JSONL format);')
    lines.append('   human IRR codes are in .txt worksheets only (not machine-readable). VAAMR')
    lines.append('   κ for cohort2 is therefore not computed. PURER coverage uses the JSONL file.')
    lines.append('4. At n≈20 participants, κ CIs are wide — the decision is a best-available')
    lines.append('   signal, not a high-powered experiment.')
    lines.append('')

    out_path = os.path.join(out_dir, 'RESULTS.md')
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'\nResults written to {out_path}')


def _print_headline_table(vaamr_results: List[Dict]) -> None:
    """Print the headline κ / coverage table to console."""
    print('\n' + '=' * 90)
    print('HEADLINE TABLE — VAAMR κ / coverage per policy per dataset')
    print('=' * 90)
    header = f'{"Policy":20s}  {"kappa":6s}  {"95% CI":20s}  {"%agree":7s}  {"n":5s}  {"coverage":8s}'
    for res in vaamr_results:
        dset = res['dataset']
        if not res.get('policies'):
            print(f'\n{dset}: {res.get("note", "no data")}')
            continue
        print(f'\n{dset}:')
        print('  ' + header)
        print('  ' + '-' * len(header))
        for p in POLICIES:
            r = res['policies'][p]
            k = r.get('kappa')
            ci = r.get('kappa_ci')
            k_str = f'{k:.3f}' if k is not None else '  N/A '
            ci_str = f'[{ci["lo"]:.3f},{ci["hi"]:.3f}]' if ci else '[   N/A    ]'
            pct_str = f'{r["pct_agree"]:.3f}' if r.get("pct_agree") is not None else '  N/A '
            print(f'  {p:20s}  {k_str}  {ci_str:20s}  {pct_str:7s}  {r["n_labeled"]:5d}'
                  f'  {r["coverage"]:.3f}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Vote-policy comparison: re-vote stored ballots under 4 policies vs human IRR'
    )
    parser.add_argument(
        '--data', default=None,
        help='Primary data directory (default: auto-detected relative to repo root)'
    )
    args = parser.parse_args()

    out_dir = _HERE  # write results into the experiment directory itself

    # Locate data directories.
    if args.data:
        primary_dir = os.path.abspath(args.data)
    else:
        primary_dir = os.path.join(_ROOT, 'data', 'MMORE_Processed')
    cohort2_dir = os.path.join(_ROOT, 'data', 'MMORE_Processed_cohort2')

    primary_db = os.path.join(primary_dir, 'qra.db')
    cohort2_db = os.path.join(cohort2_dir, 'qra.db')
    cohort2_purer_jsonl = os.path.join(
        cohort2_dir, '02_meta', 'classifications', 'purer_labels.jsonl'
    )
    cohort2_theme_jsonl = os.path.join(
        cohort2_dir, '02_meta', 'classifications', 'theme_labels.jsonl'
    )

    # ---- A. VAAMR κ comparison ----------------------------------------
    vaamr_results = []

    if os.path.isfile(primary_db):
        vaamr_results.append(run_vaamr_comparison(primary_db, 'MMORE_Processed'))
    else:
        print(f'WARNING: primary DB not found: {primary_db}')
        vaamr_results.append({'dataset': 'MMORE_Processed', 'note': 'DB not found', 'policies': {}})

    if os.path.isfile(cohort2_db):
        vaamr_results.append(run_vaamr_comparison(cohort2_db, 'MMORE_Processed_cohort2'))
    else:
        # cohort2 has no SQLite DB (pre-migration JSONL format).
        # Human IRR codes are only in human-readable .txt worksheets; skip.
        print(f'\n=== VAAMR κ comparison: MMORE_Processed_cohort2 ===')
        print('  No SQLite qra.db found (pre-migration JSONL format).')
        print('  Human IRR codes are in .txt worksheets only — not machine-readable.')
        print('  VAAMR κ for cohort2 not computed.')
        vaamr_results.append({
            'dataset': 'MMORE_Processed_cohort2',
            'note': 'no qra.db (pre-migration JSONL); human IRR codes not machine-readable',
            'policies': {},
        })

    # ---- B. PURER coverage comparison ---------------------------------
    purer_results = []

    if os.path.isfile(primary_db):
        purer_results.append(run_purer_comparison(
            primary_db, 'MMORE_Processed', source_type='sqlite',
            n_raters_note='1 rater (nvidia/nemotron-3-nano-4b)'
        ))

    if os.path.isfile(cohort2_purer_jsonl):
        purer_results.append(run_purer_comparison(
            cohort2_purer_jsonl, 'MMORE_Processed_cohort2', source_type='jsonl',
            n_raters_note='3 raters (nemotron-4b, gemma-4-4b, qwen3-8b)'
        ))
    else:
        print(f'\n=== PURER coverage: MMORE_Processed_cohort2 ===')
        print(f'  File not found: {cohort2_purer_jsonl}')

    # ---- Decision -------------------------------------------------------
    winner, rationale = _decide(vaamr_results)
    print(f'\n{"="*60}')
    print(f'DECISION: winner = {winner}')
    print(f'Rationale: {rationale}')
    print('=' * 60)

    # ---- Console headline table -----------------------------------------
    _print_headline_table(vaamr_results)

    # ---- PURER console table --------------------------------------------
    print('\n' + '=' * 70)
    print('PURER coverage per policy')
    print('=' * 70)
    for res in purer_results:
        dset = res['dataset']
        n_note = res.get('n_raters_note', '')
        print(f'\n{dset}' + (f'  ({n_note})' if n_note else '') + ':')
        print(f'  {"Policy":20s}  {"labeled":7s}  {"abstain":7s}  {"unlabeled":9s}  {"coverage":8s}')
        for p in POLICIES:
            r = res['policies'][p]
            print(f'  {p:20s}  {r["n_labeled"]:7d}  {r["n_abstain_consensus"]:7d}'
                  f'  {r["n_unlabeled"]:9d}  {r["coverage"]:.3f}')

    # ---- Save results.json ----------------------------------------------
    results_json = {
        'winner': winner,
        'rationale': rationale,
        'vaamr': [
            {
                'dataset': r['dataset'],
                'note': r.get('note'),
                'policies': {
                    p: {
                        k: v for k, v in r['policies'].get(p, {}).items()
                        if k != 'per_class'  # keep JSON compact
                    }
                    for p in POLICIES
                } if r.get('policies') else {},
                'per_class_by_policy': {
                    p: r['policies'][p].get('per_class', {})
                    for p in POLICIES
                    if r.get('policies')
                },
            }
            for r in vaamr_results
        ],
        'purer': purer_results,
    }
    json_path = os.path.join(out_dir, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    print(f'\nRaw results written to {json_path}')

    # ---- Write RESULTS.md -----------------------------------------------
    _write_results_md(vaamr_results, purer_results, winner, rationale, out_dir)

    return winner


if __name__ == '__main__':
    main()
