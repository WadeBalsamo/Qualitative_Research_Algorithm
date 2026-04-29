"""Per-transcript and cumulative statistics report export functions."""

import json
import os
from typing import List

import pandas as pd

from .. import output_paths as _paths
from ._shared import _ms_to_hms, _fmt_conf, _theme_name_from


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

        # Dual-code co-occurrence: primary → secondary → count
        dual_cooccurrence: dict = {}
        if 'primary_stage' in sdf.columns and 'secondary_stage' in sdf.columns:
            dual = sdf[sdf['secondary_stage'].notna() & sdf['primary_stage'].notna()]
            for _, row in dual.iterrows():
                p_name = id_to_name.get(int(row['primary_stage']), str(int(row['primary_stage'])))
                s_name = id_to_name.get(int(row['secondary_stage']), str(int(row['secondary_stage'])))
                dual_cooccurrence.setdefault(p_name, {})[s_name] = (
                    dual_cooccurrence.get(p_name, {}).get(s_name, 0) + 1
                )

        report = {
            'session_id': str(session_id),
            'generated': _dt.datetime.utcnow().strftime('%Y-%m-%d'),
            'n_segments': n_segs,
            'duration_ms': duration_ms,
            'theme_distribution': theme_dist,
            'theme_scores': theme_scores,
            'dual_code_cooccurrence': dual_cooccurrence,
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
