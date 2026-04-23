"""
dataset_assembly.py
-------------------
Dataset assembly: produces output files from classified segments.
"""

import json
import os
import textwrap
from collections import Counter, defaultdict
from typing import List, Dict, Optional

import pandas as pd

from classification_tools.data_structures import Segment
from constructs.theme_schema import ThemeFramework


# ---------------------------------------------------------------------------
# Shared formatting helpers (used by multiple export functions)
# ---------------------------------------------------------------------------

def _ms_to_hms(ms: int) -> str:
    s = ms // 1000
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _fmt_conf(c) -> str:
    return f"{c:.2f}" if isinstance(c, (int, float)) else '?'


def _theme_name_from(stage, id_to_name: dict) -> str:
    if stage is None:
        return '—'
    return id_to_name.get(stage, str(stage))


def _summarize_rationales(justifications: List[str], llm_client) -> str:
    """Ask the primary LLM to produce a ≤50-word summary of multiple rater rationales."""
    combined = ' | '.join(f'R{i+1}: {j}' for i, j in enumerate(justifications) if j)
    prompt = (
        "You are a qualitative research assistant. Given the following rationales "
        "from multiple raters classifying the same therapeutic dialogue segment, "
        "write a concise summary (50 words or fewer) that captures the key reasoning.\n\n"
        f"{combined}\n\n"
        "Summary (50 words max, plain text only, no bullet points):"
    )
    try:
        text, _ = llm_client.request(prompt)
        if text:
            words = text.strip().split()
            return ' '.join(words[:50])
    except Exception:
        pass
    return justifications[0][:300] if justifications else ''

try:
    from classification_tools.reliability import compute_reliability as _compute_reliability
    _RELIABILITY_AVAILABLE = True
except ImportError:
    _compute_reliability = None
    _RELIABILITY_AVAILABLE = False


# ---------------------------------------------------------------------------
# Output 1: Master Segment Dataset
# ---------------------------------------------------------------------------

def assemble_master_dataset(
    segments: List[Segment],
    output_path: str,
    confidence_tiers: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Produce the master segment dataset.

    Computes final_label and label_confidence_tier following the priority:
    adjudicated > human_consensus > llm_zero_shot.
    """
    ct = confidence_tiers or {}
    high_consistency = ct.get('high_consistency', 3)
    high_confidence = ct.get('high_confidence', 0.8)
    medium_min_consistency = ct.get('medium_min_consistency', 2)
    medium_min_confidence = ct.get('medium_min_confidence', 0.6)

    rows = []
    for seg in segments:
        # Compute final_label
        if seg.adjudicated_label is not None:
            final_label = seg.adjudicated_label
            final_label_source = 'adjudicated'
        elif seg.human_label is not None and seg.human_label == seg.primary_stage:
            final_label = seg.human_label
            final_label_source = 'human_consensus'
        elif seg.primary_stage is not None:
            final_label = seg.primary_stage
            final_label_source = 'llm_zero_shot'
        else:
            final_label = None
            final_label_source = None

        # Compute confidence tier
        if (
            seg.llm_run_consistency == high_consistency
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > high_confidence
        ):
            confidence_tier = 'high'
        elif (
            seg.llm_run_consistency is not None
            and seg.llm_run_consistency >= medium_min_consistency
            and seg.llm_confidence_primary is not None
            and seg.llm_confidence_primary > medium_min_confidence
        ):
            confidence_tier = 'medium'
        else:
            confidence_tier = 'low'

        row = {
            'segment_id': seg.segment_id,
            'trial_id': seg.trial_id,
            'participant_id': seg.participant_id,
            'session_id': seg.session_id,
            'session_number': seg.session_number,
            'cohort_id': seg.cohort_id,
            'session_variant': seg.session_variant,
            'segment_index': seg.segment_index,
            'start_time_ms': seg.start_time_ms,
            'end_time_ms': seg.end_time_ms,
            'total_segments_in_session': seg.total_segments_in_session,
            'speaker': seg.speaker,
            'text': seg.text,
            'word_count': seg.word_count,
            # Theme labels
            'primary_stage': seg.primary_stage,
            'secondary_stage': seg.secondary_stage,
            'llm_confidence_primary': seg.llm_confidence_primary,
            'llm_confidence_secondary': seg.llm_confidence_secondary,
            'llm_justification': seg.llm_justification,
            'llm_run_consistency': seg.llm_run_consistency,
            # Interrater-reliability fields (getattr guards against older Segment schemas)
            'rater_ids': (json.dumps(v) if (v := getattr(seg, 'rater_ids', None)) else None),
            'rater_votes': (json.dumps(v) if (v := getattr(seg, 'rater_votes', None)) else None),
            'agreement_level': getattr(seg, 'agreement_level', None),
            'agreement_fraction': getattr(seg, 'agreement_fraction', None),
            'needs_review': getattr(seg, 'needs_review', False),
            'consensus_vote': (
                json.dumps(cv) if isinstance(cv := getattr(seg, 'consensus_vote', None), str)
                else cv
            ),
            'tie_broken_by_confidence': getattr(seg, 'tie_broken_by_confidence', False),
            # Codebook labels (if populated)
            'codebook_labels_embedding': seg.codebook_labels_embedding,
            'codebook_labels_llm': seg.codebook_labels_llm,
            'codebook_labels_ensemble': seg.codebook_labels_ensemble,
            'codebook_disagreements': seg.codebook_disagreements,
            # Validation
            'human_label': seg.human_label,
            'human_secondary_label': seg.human_secondary_label,
            'adjudicated_label': seg.adjudicated_label,
            'in_human_coded_subset': seg.in_human_coded_subset,
            'label_status': seg.label_status,
            # Final
            'final_label': final_label,
            'final_label_source': final_label_source,
            'label_confidence_tier': confidence_tier,
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Save as both CSV and JSONL
    csv_path = output_path.replace('.jsonl', '.csv')
    df.to_csv(csv_path, index=False)
    df.to_json(output_path, orient='records', lines=True)

    # Print class distribution
    participant_labeled = df[
        (df['speaker'] == 'participant') & (df['final_label'].notna())
    ]
    print("\nFinal label distribution:")
    if len(participant_labeled) > 0:
        print(participant_labeled['final_label'].value_counts().sort_index())
    print(f"\nTotal segments: {len(df)}")
    print(f"Participant segments with labels: {len(participant_labeled)}")

    return df


# ---------------------------------------------------------------------------
# Output 3: Theme Definition File
# ---------------------------------------------------------------------------

def export_theme_definitions(
    framework: ThemeFramework,
    output_path: str,
) -> None:
    """Export theme/stage definitions as JSON."""
    with open(output_path, 'w') as f:
        json.dump(framework.to_json(), f, indent=2)


# ---------------------------------------------------------------------------
# Output 5: Content Validity Test Set
# ---------------------------------------------------------------------------

def export_content_validity_test_set(
    test_items: List[Dict],
    output_path: str,
) -> None:
    """Export content validity test set as JSONL."""
    with open(output_path, 'w') as f:
        for item in test_items:
            f.write(json.dumps(item) + '\n')


# ---------------------------------------------------------------------------
# Output: Coded Transcript (human-readable, per session)
# ---------------------------------------------------------------------------

def export_coded_transcript(
    segments: List[Segment],
    framework,
    codebook,
    output_path: str,
    llm_client=None,
) -> None:
    """
    Write a human-readable coded transcript for one session.

    Layout::

        SESSION HEADER
          - session id, duration, segment count
          - raters used (rater roster)
          - theme counts + mean confidence per theme
          - code counts + mean confidence per code (if codebook active)
          - IRR: percent agreement, Fleiss κ, Krippendorff α

        For each segment:
          segment metadata + text
          RATER BALLOTS (one line per rater: vote, stage, conf, justification)
          CONSENSUS: CLASSIFIED / UNCLASSIFIED / NEEDS REVIEW
          CODES (if any)
    """
    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    code_id_to_desc: dict = {}
    if codebook is not None:
        code_id_to_desc = {c.code_id: c.description for c in codebook.codes}

    def _theme_name(stage) -> str:
        return _theme_name_from(stage, id_to_name)

    sorted_segs = sorted(segments, key=lambda s: (s.session_id, s.segment_index))

    # ---------------- Session header aggregates ----------------
    theme_counts: Counter = Counter()
    theme_confs: Dict[int, List[float]] = defaultdict(list)
    code_counts: Counter = Counter()
    code_confs: Dict[str, List[float]] = defaultdict(list)
    unclassified = 0
    needs_review = 0
    abstentions = 0

    for seg in sorted_segs:
        if seg.primary_stage is not None:
            theme_counts[seg.primary_stage] += 1
            if seg.llm_confidence_primary is not None:
                theme_confs[seg.primary_stage].append(seg.llm_confidence_primary)
        else:
            unclassified += 1
            if getattr(seg, 'consensus_vote', None) == 'ABSTAIN':
                abstentions += 1
        if getattr(seg, 'needs_review', False):
            needs_review += 1

        codes = seg.codebook_labels_ensemble or seg.codebook_labels_llm or []
        cb_conf = seg.codebook_confidence or {}
        for c in codes:
            code_counts[str(c)] += 1
            conf_val = cb_conf.get(str(c)) if isinstance(cb_conf, dict) else None
            if conf_val is not None:
                code_confs[str(c)].append(conf_val)

    rater_roster: List[str] = []
    for seg in sorted_segs:
        rids = getattr(seg, 'rater_ids', None)
        if rids and len(rids) > len(rater_roster):
            rater_roster = list(rids)

    irr_stats = (
        _compute_reliability(sorted_segs)
        if sorted_segs and _RELIABILITY_AVAILABLE
        else {}
    )

    with open(output_path, 'w', encoding='utf-8') as fh:
        if sorted_segs:
            first = sorted_segs[0]
            last = sorted_segs[-1]
            duration_s = max(0, (last.end_time_ms - first.start_time_ms) // 1000)
            dur_h, dur_rem = divmod(duration_s, 3600)
            dur_m, dur_s = divmod(dur_rem, 60)
            all_pids = sorted({s.participant_id for s in sorted_segs if s.participant_id})
            participants_str = ", ".join(all_pids) if all_pids else first.participant_id
            fh.write("=" * 78 + "\n")
            fh.write(f"SESSION: {first.session_id}\n")
            fh.write(f"Participants: {participants_str}   "
                     f"Trial: {first.trial_id}\n")
            fh.write(f"Duration: {dur_h}:{dur_m:02d}:{dur_s:02d}   "
                     f"Segments: {len(sorted_segs)}\n")
            fh.write("=" * 78 + "\n\n")

            # Rater roster
            if rater_roster:
                fh.write("RATERS\n")
                fh.write("-" * 78 + "\n")
                for i, rid in enumerate(rater_roster, 1):
                    fh.write(f"  R{i}: {rid}\n")
                fh.write("\n")

            # Theme distribution
            if theme_counts:
                total_classified = sum(theme_counts.values())
                fh.write(f"THEME DISTRIBUTION  ({total_classified} classified, "
                         f"{unclassified} unclassified)\n")
                fh.write("-" * 78 + "\n")
                fh.write(f"  {'Theme':<28} {'Count':>6}  {'%':>5}  "
                         f"{'Mean Conf':>9}\n")
                for stage, cnt in sorted(theme_counts.items(),
                                         key=lambda x: -x[1]):
                    name = _theme_name(stage)
                    pct = 100.0 * cnt / total_classified
                    confs = theme_confs[stage]
                    mean_c = sum(confs) / len(confs) if confs else 0.0
                    fh.write(f"  {name:<28} {cnt:>6}  {pct:>4.1f}%  "
                             f"{mean_c:>9.3f}\n")
                fh.write("\n")

            # Codebook distribution
            if code_counts:
                fh.write(f"CODEBOOK DISTRIBUTION  ({sum(code_counts.values())} "
                         f"code applications across {len(code_counts)} codes)\n")
                fh.write("-" * 78 + "\n")
                fh.write(f"  {'Code':<40} {'Count':>6}  {'Mean Conf':>9}\n")
                for code, cnt in sorted(code_counts.items(), key=lambda x: -x[1]):
                    confs = code_confs[code]
                    mean_c = sum(confs) / len(confs) if confs else 0.0
                    fh.write(f"  {code:<40} {cnt:>6}  {mean_c:>9.3f}\n")
                fh.write("\n")

            # IRR
            if irr_stats and irr_stats.get('n_segments'):
                fh.write("INTERRATER RELIABILITY\n")
                fh.write("-" * 78 + "\n")
                fh.write(f"  Raters: {irr_stats.get('n_raters', 0)}   "
                         f"Segments w/ ballots: {irr_stats['n_segments']}\n")
                fh.write(f"  Unanimous agreement:  "
                         f"{irr_stats['percent_agreement_unanimous']:.3f}\n")
                fh.write(f"  Pairwise agreement:   "
                         f"{irr_stats['percent_agreement_pairwise']:.3f}\n")
                fk = irr_stats.get('fleiss_kappa')
                ka = irr_stats.get('krippendorff_alpha_nominal')
                fk_s = f"{fk:.3f}" if isinstance(fk, (int, float)) else "n/a"
                ka_s = f"{ka:.3f}" if isinstance(ka, (int, float)) else "n/a"
                fh.write(f"  Fleiss' kappa:        {fk_s}\n")
                fh.write(f"  Krippendorff's alpha: {ka_s}\n")
                fh.write(f"  Flagged for review:   {needs_review}\n")
                if abstentions:
                    fh.write(f"  Consensus abstentions: {abstentions}\n")
                fh.write("\n")

        # ---------------- Per-segment detail ----------------
        for seg in sorted_segs:
            speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker])
            fh.write("=" * 78 + "\n")
            fh.write(
                f"[SEGMENT {seg.segment_index + 1:03d}]  "
                f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                f"{seg.word_count}w   Speakers: {speakers_str}\n"
            )
            fh.write("-" * 78 + "\n")
            for line in textwrap.wrap(seg.text, width=76,
                                      initial_indent="  ",
                                      subsequent_indent="  ") or ["  "]:
                fh.write(line + "\n")
            fh.write("\n")

            # Rater ballots
            rater_votes = getattr(seg, 'rater_votes', None) or []
            if rater_votes:
                fh.write("RATER BALLOTS\n")
                for rv in rater_votes:
                    rid = rv.get('rater', '?')
                    vote = rv.get('vote', '?')
                    stage = rv.get('stage')
                    conf = rv.get('confidence')
                    just = (rv.get('justification') or '').strip()
                    if vote == 'CODED':
                        sec = rv.get('secondary_stage')
                        sec_conf = rv.get('secondary_confidence')
                        if sec is not None:
                            sec_conf_str = f", conf={_fmt_conf(sec_conf)}" if sec_conf is not None else ""
                            sec_part = f"  (2nd: {_theme_name(sec)}{sec_conf_str})"
                        else:
                            sec_part = ""
                        fh.write(f"  [{rid}]  {_theme_name(stage)}  "
                                 f"conf={_fmt_conf(conf)}{sec_part}\n")
                    elif vote == 'ABSTAIN':
                        fh.write(f"  [{rid}]  ABSTAIN (irrelevant to framework)\n")
                    else:
                        fh.write(f"  [{rid}]  ERROR (no parseable response)\n")
                    if just:
                        for line in textwrap.wrap(just, width=70,
                                                  initial_indent="      → ",
                                                  subsequent_indent="        "):
                            fh.write(line + "\n")
                fh.write("\n")

            # Consensus
            fh.write("CONSENSUS: ")
            agreement = getattr(seg, 'agreement_level', None) or '?'
            _rids = getattr(seg, 'rater_ids', None) or []
            n_agree = int(round((getattr(seg, 'agreement_fraction', None) or 0.0)
                                * max(len(_rids), 1)))
            n_raters = len(_rids)
            if seg.primary_stage is not None:
                fh.write(f"CLASSIFIED as {_theme_name(seg.primary_stage)}  "
                         f"({agreement}, {n_agree}/{n_raters})\n")
                fh.write(f"  Mean confidence: "
                         f"{_fmt_conf(seg.llm_confidence_primary)}\n")
                if seg.secondary_stage is not None:
                    fh.write(f"  Secondary: {_theme_name(seg.secondary_stage)} "
                             f"(conf {_fmt_conf(seg.llm_confidence_secondary)})\n")
                all_justs = [
                    rv.get('justification', '') or ''
                    for rv in (getattr(seg, 'rater_votes', None) or [])
                    if rv.get('vote') == 'CODED'
                ]
                unique_justs = list(dict.fromkeys(j for j in all_justs if j.strip()))
                if unique_justs:
                    if llm_client is not None and len(unique_justs) > 1:
                        rationale = _summarize_rationales(unique_justs, llm_client)
                    else:
                        rationale = unique_justs[0]
                elif seg.llm_justification:
                    rationale = seg.llm_justification
                else:
                    rationale = ''
                if rationale:
                    for line in textwrap.wrap(rationale, width=70,
                                              initial_indent="  Rationale: ",
                                              subsequent_indent="    "):
                        fh.write(line + "\n")
            else:
                if getattr(seg, 'consensus_vote', None) == 'ABSTAIN':
                    fh.write(f"UNCLASSIFIED — consensus ABSTAIN "
                             f"({agreement}, {n_agree}/{n_raters})\n")
                elif agreement == 'split':
                    fh.write(f"UNCLASSIFIED — SPLIT VOTE (no majority); "
                             f"flagged for review\n")
                elif agreement == 'none':
                    fh.write(f"UNCLASSIFIED — all raters failed to respond\n")
                else:
                    fh.write(f"UNCLASSIFIED ({agreement})\n")
            if getattr(seg, 'tie_broken_by_confidence', False):
                fh.write("  ↳ Tie broken by confidence\n")
            if getattr(seg, 'needs_review', False):
                fh.write("  ↳ FLAGGED FOR HUMAN REVIEW\n")

            # Codebook labels
            codes = seg.codebook_labels_ensemble or seg.codebook_labels_llm or []
            if codes:
                label = "CODES" if seg.codebook_labels_ensemble else "CODES (LLM)"
                fh.write(f"{label}: [{', '.join(str(c) for c in codes)}]\n")

            fh.write("\n")


# ---------------------------------------------------------------------------
# Output: Per-transcript statistics report
# ---------------------------------------------------------------------------

def export_per_transcript_stats(
    master_df: pd.DataFrame,
    framework,
    codebook,
    output_dir: str,
) -> None:
    """
    Write one JSON stats file per session_id into output_dir.

    Each file contains: segment count, theme distribution + scores,
    code frequencies, confidence tier breakdown, and top exemplar segments.
    """
    import datetime as _dt

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    os.makedirs(output_dir, exist_ok=True)

    for session_id, sdf in master_df.groupby('session_id'):
        n_segs = len(sdf)
        duration_ms = int(sdf['end_time_ms'].max() - sdf['start_time_ms'].min())

        # Theme distribution
        theme_dist: dict = {}
        theme_scores: dict = {}
        if 'primary_stage' in sdf.columns:
            labeled = sdf[sdf['primary_stage'].notna()]
            for tid, count in labeled['primary_stage'].value_counts().items():
                name = id_to_name.get(int(tid), str(tid))
                theme_dist[name] = int(count)
                theme_scores[name] = round(count / n_segs, 4) if n_segs else 0.0

        # Code frequencies
        code_freq: dict = {}
        if 'codebook_labels_ensemble' in sdf.columns:
            for codes in sdf['codebook_labels_ensemble'].dropna():
                if isinstance(codes, list):
                    for c in codes:
                        code_freq[str(c)] = code_freq.get(str(c), 0) + 1
        code_freq = dict(sorted(code_freq.items(), key=lambda x: -x[1]))

        # Confidence tiers
        tier_counts: dict = {}
        if 'label_confidence_tier' in sdf.columns:
            tier_counts = sdf['label_confidence_tier'].value_counts().to_dict()

        # Top exemplars per theme (highest confidence)
        exemplars: dict = {}
        if 'primary_stage' in sdf.columns and 'llm_confidence_primary' in sdf.columns:
            for tid in sdf['primary_stage'].dropna().unique():
                name = id_to_name.get(int(tid), str(tid))
                top = (
                    sdf[sdf['primary_stage'] == tid]
                    .sort_values('llm_confidence_primary', ascending=False)
                    .head(2)
                )
                exemplars[name] = [
                    {
                        'segment_id': row['segment_id'],
                        'text': row['text'][:200],
                        'confidence': round(float(row['llm_confidence_primary'] or 0), 3),
                    }
                    for _, row in top.iterrows()
                ]

        report = {
            'session_id': str(session_id),
            'generated': _dt.datetime.utcnow().strftime('%Y-%m-%d'),
            'n_segments': n_segs,
            'duration_ms': duration_ms,
            'theme_distribution': theme_dist,
            'theme_scores': theme_scores,
            'code_frequencies': code_freq,
            'confidence_tiers': {str(k): int(v) for k, v in tier_counts.items()},
            'exemplar_segments_by_theme': exemplars,
        }

        fname = os.path.join(output_dir, f"stats_{session_id}.json")
        with open(fname, 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Output: Cumulative report across all transcripts
# ---------------------------------------------------------------------------

def export_cumulative_report(
    master_df: pd.DataFrame,
    framework,
    codebook,
    output_path: str,
) -> None:
    """
    Write a single JSON report aggregating results across all sessions.

    Includes: per-transcript construct scores, aggregate distributions,
    cumulative code frequencies, and top exemplar utterances per theme
    and per code.
    """
    import datetime as _dt
    from collections import defaultdict

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.name for t in framework.themes}

    sessions = sorted(master_df['session_id'].unique().tolist(), key=str)
    n_segs_total = len(master_df)

    # Per-transcript construct scores
    construct_scores_by_session: dict = {}
    for session_id, sdf in master_df.groupby('session_id'):
        n = len(sdf)
        scores: dict = {}
        if 'primary_stage' in sdf.columns:
            for tid, cnt in sdf['primary_stage'].dropna().value_counts().items():
                name = id_to_name.get(int(tid), str(tid))
                scores[name] = round(cnt / n, 4) if n else 0.0
        construct_scores_by_session[str(session_id)] = scores

    # Aggregate theme distribution
    agg_theme_dist: dict = {}
    if 'primary_stage' in master_df.columns:
        for tid, cnt in master_df['primary_stage'].dropna().value_counts().items():
            name = id_to_name.get(int(tid), str(tid))
            agg_theme_dist[name] = int(cnt)

    agg_theme_scores: dict = {
        name: round(cnt / n_segs_total, 4)
        for name, cnt in agg_theme_dist.items()
    }

    # Aggregate code frequencies
    agg_code_freq: dict = {}
    if 'codebook_labels_ensemble' in master_df.columns:
        for codes in master_df['codebook_labels_ensemble'].dropna():
            if isinstance(codes, list):
                for c in codes:
                    agg_code_freq[str(c)] = agg_code_freq.get(str(c), 0) + 1
    agg_code_freq = dict(sorted(agg_code_freq.items(), key=lambda x: -x[1]))

    # Top exemplars per theme across all sessions
    exemplars_by_theme: dict = {}
    if 'primary_stage' in master_df.columns and 'llm_confidence_primary' in master_df.columns:
        for tid in master_df['primary_stage'].dropna().unique():
            name = id_to_name.get(int(tid), str(tid))
            top = (
                master_df[master_df['primary_stage'] == tid]
                .sort_values('llm_confidence_primary', ascending=False)
                .head(3)
            )
            exemplars_by_theme[name] = [
                {
                    'segment_id': row['segment_id'],
                    'session_id': str(row['session_id']),
                    'text': row['text'][:300],
                    'confidence': round(float(row['llm_confidence_primary'] or 0), 3),
                }
                for _, row in top.iterrows()
            ]

    # Top exemplars per code
    exemplars_by_code: dict = {}
    if 'codebook_labels_ensemble' in master_df.columns:
        code_to_segs: dict = defaultdict(list)
        for _, row in master_df.iterrows():
            codes = row.get('codebook_labels_ensemble')
            if isinstance(codes, list):
                for c in codes:
                    code_to_segs[str(c)].append({
                        'segment_id': row['segment_id'],
                        'session_id': str(row['session_id']),
                        'text': str(row['text'])[:300],
                        'confidence': round(
                            float((row.get('codebook_confidence') or {}).get(c, 0.0)), 3
                        ) if isinstance(row.get('codebook_confidence'), dict) else 0.0,
                    })
        for code, segs in sorted(code_to_segs.items()):
            top3 = sorted(segs, key=lambda x: -x['confidence'])[:3]
            exemplars_by_code[code] = top3

    report = {
        'generated': _dt.datetime.utcnow().strftime('%Y-%m-%d'),
        'n_transcripts': len(sessions),
        'transcripts': [str(s) for s in sessions],
        'total_segments': n_segs_total,
        'theme_distribution_aggregate': agg_theme_dist,
        'theme_scores_aggregate': agg_theme_scores,
        'code_frequencies_aggregate': agg_code_freq,
        'construct_scores_by_transcript': construct_scores_by_session,
        'exemplar_segments_by_theme': exemplars_by_theme,
        'exemplar_segments_by_code': exemplars_by_code,
    }

    with open(output_path, 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Output: Human classification forms (no results, for blind coding)
# ---------------------------------------------------------------------------

def export_human_classification_forms(
    segments: List[Segment],
    framework,
    output_dir: str,
) -> List[str]:
    """
    Write one blank-form .txt per session into <output_dir>/validation/.

    Each file shows segments in order with empty rating fields so human
    coders can classify blind, without any LLM results present.
    Only participant segments are included (therapist segments omitted).
    Returns list of written file paths.
    """
    import datetime as _dt

    id_to_name: dict = {}
    stage_labels: str = '?'
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}
        stage_labels = ', '.join(
            f"{t.theme_id}={t.short_name}" for t in framework.themes
        )

    def _theme_name(stage) -> str:
        return _theme_name_from(stage, id_to_name)

    validation_dir = os.path.join(output_dir, 'validation')
    os.makedirs(validation_dir, exist_ok=True)

    # Group by session
    by_session: Dict[str, List[Segment]] = defaultdict(list)
    for seg in segments:
        by_session[seg.session_id].append(seg)

    written: List[str] = []

    for session_id, segs in sorted(by_session.items()):
        sorted_segs = sorted(segs, key=lambda s: s.segment_index)

        # Only participant segments (skip therapist-only context turns)
        participant_segs = [
            s for s in sorted_segs
            if s.speaker and s.speaker.lower() not in ('therapist', 't', 'interviewer')
        ]
        if not participant_segs:
            participant_segs = sorted_segs  # fallback: include all if no match

        if not participant_segs:
            continue

        first = participant_segs[0]
        last = participant_segs[-1]
        duration_s = max(0, (last.end_time_ms - first.start_time_ms) // 1000)
        dur_h, dur_rem = divmod(duration_s, 3600)
        dur_m, dur_s = divmod(dur_rem, 60)

        all_pids = sorted({s.participant_id for s in participant_segs if s.participant_id})
        participants_str = ", ".join(all_pids) if all_pids else first.participant_id

        out_path = os.path.join(validation_dir, f'human_classification_{session_id}.txt')
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write("=" * 78 + "\n")
            fh.write(f"SESSION: {session_id}\n")
            fh.write(f"Participants: {participants_str}   Trial: {first.trial_id}\n")
            fh.write(f"Duration: {dur_h}:{dur_m:02d}:{dur_s:02d}   "
                     f"Segments: {len(participant_segs)}\n")
            fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}\n")
            fh.write("=" * 78 + "\n\n")
            fh.write("INSTRUCTIONS\n")
            fh.write("-" * 78 + "\n")
            fh.write(f"  Stage labels: {stage_labels}\n")
            fh.write("  For each segment, record the primary stage, an optional\n")
            fh.write("  secondary stage, and a brief rationale.\n\n")

            for seg in participant_segs:
                speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker or '?'])
                fh.write("=" * 78 + "\n")
                fh.write(
                    f"[SEGMENT {seg.segment_index + 1:03d}]  "
                    f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                    f"{seg.word_count}w   Speaker(s): {speakers_str}\n"
                )
                fh.write("-" * 78 + "\n")
                for line in textwrap.wrap(seg.text, width=76,
                                          initial_indent="  ",
                                          subsequent_indent="  ") or ["  "]:
                    fh.write(line + "\n")
                fh.write("\n")
                fh.write("  Stage: ___   Secondary (optional): ___\n")
                fh.write("  Rationale: " + "_" * 60 + "\n")
                fh.write("  " + "_" * 72 + "\n")
                fh.write("\n")

        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# Output: Dataset-wide flagged-for-review report
# ---------------------------------------------------------------------------

def export_flagged_for_review(
    segments: List[Segment],
    framework,
    output_path: str,
) -> None:
    """
    Write a single .txt listing every needs_review segment across all sessions.

    Includes full rater ballot detail, combined justifications, coded result
    and confidence (primary and secondary), and agreement metadata.
    """
    import datetime as _dt

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    def _theme_name(stage) -> str:
        return _theme_name_from(stage, id_to_name)

    flagged = [s for s in segments if getattr(s, 'needs_review', False)]
    flagged.sort(key=lambda s: (s.session_id, s.segment_index))

    session_ids = sorted({s.session_id for s in flagged})

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(
            f"FLAGGED FOR REVIEW — {len(flagged)} segment(s) "
            f"across {len(session_ids)} session(s)\n"
        )
        fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}\n")
        fh.write("=" * 78 + "\n\n")

        for seg in flagged:
            speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker or '?'])
            fh.write(
                f"[Session {seg.session_id}  Segment {seg.segment_index + 1:03d}]  "
                f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                f"{seg.word_count}w   Speaker(s): {speakers_str}\n"
            )
            fh.write("-" * 78 + "\n")
            for line in textwrap.wrap(seg.text, width=76,
                                      initial_indent="  ",
                                      subsequent_indent="  ") or ["  "]:
                fh.write(line + "\n")
            fh.write("\n")

            # Rater ballots
            rater_votes = getattr(seg, 'rater_votes', None) or []
            if rater_votes:
                fh.write("  RATER BALLOTS\n")
                for rv in rater_votes:
                    rid = rv.get('rater', '?')
                    vote = rv.get('vote', '?')
                    stage = rv.get('stage')
                    conf = rv.get('confidence')
                    just = (rv.get('justification') or '').strip()
                    sec = rv.get('secondary_stage')
                    sec_conf = rv.get('secondary_confidence')
                    if vote == 'CODED':
                        if sec is not None:
                            sec_conf_str = f", conf={_fmt_conf(sec_conf)}" if sec_conf is not None else ""
                            sec_part = f"  (2nd: {_theme_name(sec)}{sec_conf_str})"
                        else:
                            sec_part = ""
                        fh.write(f"    [{rid}]  {_theme_name(stage)}  "
                                 f"conf={_fmt_conf(conf)}{sec_part}\n")
                    elif vote == 'ABSTAIN':
                        fh.write(f"    [{rid}]  ABSTAIN\n")
                    else:
                        fh.write(f"    [{rid}]  ERROR\n")
                    if just:
                        for line in textwrap.wrap(just, width=68,
                                                  initial_indent="        → ",
                                                  subsequent_indent="          "):
                            fh.write(line + "\n")
                fh.write("\n")

            # Coded result summary
            agreement = getattr(seg, 'agreement_level', None) or '?'
            _rids = getattr(seg, 'rater_ids', None) or []
            n_agree = int(round(
                (getattr(seg, 'agreement_fraction', None) or 0.0) * max(len(_rids), 1)
            ))
            n_raters = len(_rids)

            if seg.primary_stage is not None:
                primary_str = (
                    f"{_theme_name(seg.primary_stage)} (conf {_fmt_conf(seg.llm_confidence_primary)})"
                )
                if seg.secondary_stage is not None:
                    sec_str = (
                        f"  |  Secondary: {_theme_name(seg.secondary_stage)} "
                        f"(conf {_fmt_conf(seg.llm_confidence_secondary)})"
                    )
                else:
                    sec_str = ""
                fh.write(f"  CODED RESULT: {primary_str}{sec_str}\n")
            else:
                if getattr(seg, 'consensus_vote', None) == 'ABSTAIN':
                    fh.write("  CODED RESULT: UNCLASSIFIED — consensus ABSTAIN\n")
                elif agreement == 'split':
                    fh.write("  CODED RESULT: UNCLASSIFIED — split vote\n")
                elif agreement == 'none':
                    fh.write("  CODED RESULT: UNCLASSIFIED — all raters failed\n")
                else:
                    fh.write("  CODED RESULT: UNCLASSIFIED\n")

            fh.write(f"  AGREEMENT: {agreement}  ({n_agree}/{n_raters})\n")
            if getattr(seg, 'tie_broken_by_confidence', False):
                fh.write("  Tie-broken by confidence: Yes\n")

            fh.write("-" * 78 + "\n\n")


# ---------------------------------------------------------------------------
# Output: BERT training data
# ---------------------------------------------------------------------------

def export_training_data(
    segments: List[Segment],
    framework,
    codebook,
    output_dir: str,
) -> List[str]:
    """
    Write BERT-ready JSONL files into <output_dir>/trainingdata/.

    Produces:
      theme_classification.jsonl  — one record per segment with a final theme label
      codebook_multilabel.jsonl   — one record per segment with codebook labels
      label_map.json              — label-ID ↔ name mappings for both tasks
    """
    import datetime as _dt

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name.lower() for t in framework.themes}

    # Build ordered codebook code index from codebook object; fall back to
    # alphabetical order of codes seen in the data.
    code_to_idx: dict = {}
    if codebook is not None:
        for idx, c in enumerate(codebook.codes):
            code_to_idx[c.code_id] = idx
    else:
        seen: list = []
        for seg in segments:
            for cid in (seg.codebook_labels_ensemble or []):
                if cid and cid not in code_to_idx:
                    code_to_idx[cid] = len(seen)
                    seen.append(cid)

    training_dir = os.path.join(output_dir, 'trainingdata')
    os.makedirs(training_dir, exist_ok=True)

    theme_path = os.path.join(training_dir, 'theme_classification.jsonl')
    codebook_path = os.path.join(training_dir, 'codebook_multilabel.jsonl')
    map_path = os.path.join(training_dir, 'label_map.json')

    theme_count = 0
    code_count = 0

    with open(theme_path, 'w', encoding='utf-8') as tf, \
         open(codebook_path, 'w', encoding='utf-8') as cf:

        for seg in segments:
            # Theme training record
            label_id = getattr(seg, 'final_label', None)
            if label_id is None:
                label_id = seg.primary_stage
            if label_id is not None:
                record = {
                    'text': seg.text,
                    'label': id_to_name.get(label_id, str(label_id)),
                    'label_id': int(label_id),
                    'label_confidence_tier': getattr(seg, 'label_confidence_tier', None),
                    'confidence': seg.llm_confidence_primary,
                    'consistency': seg.llm_run_consistency,
                    'label_source': getattr(seg, 'final_label_source', None) or 'llm_zero_shot',
                    'segment_id': seg.segment_id,
                    'participant_id': seg.participant_id,
                    'session_id': seg.session_id,
                    'session_number': seg.session_number,
                }
                tf.write(json.dumps(record, ensure_ascii=False) + '\n')
                theme_count += 1

            # Codebook multi-label record
            codes = seg.codebook_labels_ensemble or []
            if codes:
                cb_conf = seg.codebook_confidence or {}
                record = {
                    'text': seg.text,
                    'labels': list(codes),
                    'label_ids': [code_to_idx.get(c, -1) for c in codes],
                    'confidences': {c: cb_conf.get(c) for c in codes
                                   if isinstance(cb_conf, dict)},
                    'theme_label': id_to_name.get(label_id, None) if label_id is not None else None,
                    'theme_label_id': int(label_id) if label_id is not None else None,
                    'segment_id': seg.segment_id,
                    'participant_id': seg.participant_id,
                    'session_id': seg.session_id,
                    'session_number': seg.session_number,
                }
                cf.write(json.dumps(record, ensure_ascii=False) + '\n')
                code_count += 1

    # Label map
    label_map = {
        'generated': _dt.datetime.utcnow().strftime('%Y-%m-%d'),
        'theme_labels': {str(k): v for k, v in id_to_name.items()},
        'codebook_codes': list(code_to_idx.keys()),
        'codebook_code_to_index': code_to_idx,
    }
    with open(map_path, 'w', encoding='utf-8') as f:
        json.dump(label_map, f, indent=2, ensure_ascii=False)

    print(f"Training data: {theme_count} theme records, {code_count} codebook records")
    return [theme_path, codebook_path, map_path]
