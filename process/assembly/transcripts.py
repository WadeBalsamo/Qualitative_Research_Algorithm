"""Per-session coded transcript and statistics export functions."""

import json
import os
import textwrap
from collections import Counter, defaultdict
from typing import List, Dict

import pandas as pd

from classification_tools.data_structures import Segment
from .. import output_paths as _paths
from ._common import _ms_to_hms, _fmt_conf, _theme_name_from, _summarize_rationales

try:
    from classification_tools.reliability import compute_reliability as _compute_reliability
    _RELIABILITY_AVAILABLE = True
except ImportError:
    _compute_reliability = None
    _RELIABILITY_AVAILABLE = False


def export_coded_transcript(
    segments: List[Segment],
    framework,
    codebook,
    run_dir: str,
    session_id: str,
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

    coded_dir = _paths.transcripts_coded_dir(run_dir)
    os.makedirs(coded_dir, exist_ok=True)
    output_path = os.path.join(coded_dir, f'coded_transcript_{session_id}.txt')

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


def export_per_transcript_stats(
    master_df: pd.DataFrame,
    framework,
    codebook,
    run_dir: str,
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

    output_dir = _paths.session_stats_dir(run_dir)
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


def export_cumulative_report(
    master_df: pd.DataFrame,
    framework,
    codebook,
    run_dir: str,
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

    _cumdir = _paths.cumulative_report_dir(run_dir)
    os.makedirs(_cumdir, exist_ok=True)
    with open(os.path.join(_cumdir, 'cumulative_report.json'), 'w', encoding='utf-8') as fh:
        json.dump(report, fh, indent=2, default=str)
