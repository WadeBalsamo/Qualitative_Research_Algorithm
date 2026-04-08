"""
dataset_assembly.py
-------------------
Dataset assembly: produces output files from classified segments.
"""

import json
import os
from typing import List, Dict, Optional

import pandas as pd

from classification_tools.data_structures import Segment
from constructs.theme_schema import ThemeFramework


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
# Output 2: Session Adjacency Index
# ---------------------------------------------------------------------------

def build_session_adjacency_index(
    segments_df: pd.DataFrame,
    output_path: str,
) -> List[Dict]:
    """
    Produce the session adjacency index for CFiCS graph construction.
    """
    sessions = []

    for session_id, session_df in segments_df.groupby('session_id'):
        session_df = session_df.sort_values('segment_index')

        all_ids = session_df['segment_id'].tolist()
        participant_ids = session_df[
            session_df['speaker'] == 'participant'
        ]['segment_id'].tolist()
        therapist_ids = session_df[
            session_df['speaker'] == 'therapist'
        ]['segment_id'].tolist()

        # Therapist-to-participant pairs
        t_to_p_pairs = []
        for i in range(1, len(session_df)):
            current = session_df.iloc[i]
            previous = session_df.iloc[i - 1]
            if (
                current['speaker'] == 'participant'
                and previous['speaker'] == 'therapist'
            ):
                t_to_p_pairs.append([
                    previous['segment_id'],
                    current['segment_id'],
                ])

        # Participant sequential pairs
        p_sequential = []
        for i in range(1, len(participant_ids)):
            p_sequential.append([participant_ids[i - 1], participant_ids[i]])

        sessions.append({
            'session_id': session_id,
            'segment_sequence': all_ids,
            'participant_segments': participant_ids,
            'therapist_segments': therapist_ids,
            'therapist_to_participant_pairs': t_to_p_pairs,
            'participant_sequential_pairs': p_sequential,
        })

    with open(output_path, 'w') as f:
        for session in sessions:
            f.write(json.dumps(session) + '\n')

    return sessions


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
# Longitudinal Stage Tracking
# ---------------------------------------------------------------------------

def compute_session_stage_progression(
    segments_df: pd.DataFrame,
    id_to_short: Optional[Dict[int, str]] = None,
) -> pd.DataFrame:
    """
    Track theme/stage progression within each session.

    Parameters
    ----------
    segments_df : pd.DataFrame
        Master segment dataset.
    id_to_short : dict, optional
        Mapping from theme_id to short name. If None, uses string
        representation of the IDs.
    """
    if id_to_short is None:
        id_to_short = {}

    session_progressions = []

    for session_id, session_df in segments_df.groupby('session_id'):
        participant_df = session_df[
            (session_df['speaker'] == 'participant')
            & (session_df['primary_stage'].notna())
        ].sort_values('segment_index')

        if len(participant_df) == 0:
            continue

        stages = participant_df['primary_stage'].astype(int).tolist()

        transitions = []
        forward = 0
        backward = 0
        for i in range(1, len(stages)):
            from_stage = stages[i - 1]
            to_stage = stages[i]
            transitions.append((from_stage, to_stage))
            if to_stage > from_stage:
                forward += 1
            elif to_stage < from_stage:
                backward += 1

        stage_counts = {}
        for s in stages:
            stage_counts[s] = stage_counts.get(s, 0) + 1

        dominant_stage = max(stage_counts, key=stage_counts.get)

        session_progressions.append({
            'session_id': session_id,
            'trial_id': participant_df['trial_id'].iloc[0],
            'participant_id': participant_df['participant_id'].iloc[0],
            'n_segments': len(stages),
            'stage_sequence': stages,
            'stage_transitions': transitions,
            'forward_transitions': forward,
            'backward_transitions': backward,
            'lateral_transitions': len(transitions) - forward - backward,
            'max_stage_reached': max(stages),
            'dominant_stage': dominant_stage,
            'dominant_stage_name': id_to_short.get(dominant_stage, str(dominant_stage)),
            'stage_distribution': {
                id_to_short.get(k, str(k)): v
                for k, v in sorted(stage_counts.items())
            },
        })

    return pd.DataFrame(session_progressions)


# ---------------------------------------------------------------------------
# Output: Coded Transcript (human-readable, per session)
# ---------------------------------------------------------------------------

def export_coded_transcript(
    segments: List[Segment],
    framework,
    codebook,
    output_path: str,
) -> None:
    """
    Write a human-readable coded transcript for one session.

    Format per segment:
        [SEGMENT 001] HH:MM:SS - HH:MM:SS | N words | Speakers: ...
        Speaker A: text...
        Speaker B: text...
        ---
        THEME: Name (id=N) | Confidence: 0.XX | Consistency: N/3
        CODES: [code_a, code_b]
        ---
    """
    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.name for t in framework.themes}

    code_id_to_desc: dict = {}
    if codebook is not None:
        code_id_to_desc = {c.code_id: c.description for c in codebook.codes}

    def _ms_to_hms(ms: int) -> str:
        s = ms // 1000
        h, rem = divmod(s, 3600)
        m, sec = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{sec:02d}"

    sorted_segs = sorted(segments, key=lambda s: (s.session_id, s.segment_index))

    with open(output_path, 'w', encoding='utf-8') as fh:
        if sorted_segs:
            first = sorted_segs[0]
            last = sorted_segs[-1]
            duration_s = (last.end_time_ms - first.start_time_ms) // 1000
            dur_h, dur_rem = divmod(duration_s, 3600)
            dur_m, dur_s = divmod(dur_rem, 60)
            fh.write(f"SESSION: {first.session_id}\n")
            fh.write(f"Duration: {dur_h}:{dur_m:02d}:{dur_s:02d} | Segments: {len(sorted_segs)}\n")
            fh.write("=" * 72 + "\n\n")

        for seg in sorted_segs:
            speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker])
            fh.write(
                f"[SEGMENT {seg.segment_index + 1:03d}] "
                f"{_ms_to_hms(seg.start_time_ms)} - {_ms_to_hms(seg.end_time_ms)} "
                f"| {seg.word_count} words | Speakers: {speakers_str}\n"
            )
            fh.write(seg.text + "\n")
            fh.write("-" * 48 + "\n")

            # Theme label
            if seg.primary_stage is not None:
                theme_name = id_to_name.get(seg.primary_stage, str(seg.primary_stage))
                conf = seg.llm_confidence_primary or 0.0
                cons = seg.llm_run_consistency or 0
                fh.write(
                    f"THEME: {theme_name} (id={seg.primary_stage}) | "
                    f"Confidence: {conf:.2f} | Consistency: {cons}/3\n"
                )
                if seg.secondary_stage is not None:
                    sec_name = id_to_name.get(seg.secondary_stage, str(seg.secondary_stage))
                    fh.write(f"SECONDARY: {sec_name} (id={seg.secondary_stage})\n")
                if seg.llm_justification:
                    fh.write(f"Justification: {seg.llm_justification}\n")

            # Codebook labels
            if seg.codebook_labels_ensemble:
                fh.write(f"CODES: [{', '.join(str(c) for c in seg.codebook_labels_ensemble)}]\n")
            elif seg.codebook_labels_llm:
                fh.write(f"CODES (LLM): [{', '.join(str(c) for c in seg.codebook_labels_llm)}]\n")

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
        id_to_name = {t.theme_id: t.name for t in framework.themes}

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
