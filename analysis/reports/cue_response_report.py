"""cue response aggregate report generator."""

import os
from collections import defaultdict
from datetime import date

import pandas as pd

from process import output_paths as _paths
from ._common import _collect_therapist_cue, _summarize_cue, _summarize_participant_text, _wrap_quote


def generate_therapist_cues_report(
    df: pd.DataFrame,
    framework: dict,
    output_dir: str,
    therapist_cue_config,
    llm_client,
) -> str:
    """Generate cue_response.txt in the output root.

    Iterates all within-session transitions, groups by (from_stage, to_stage),
    and produces averaged FROM / CUE / TO blocks for each transition type.
    Returns path to written file.
    """
    stage_ids = sorted(framework.keys())
    stage_names = {sid: framework[sid].get('short_name', f'Stage {sid}') for sid in stage_ids}
    max_agg = therapist_cue_config.max_length_of_average_cue_responses

    # Collect (from_text, cue_text, to_text) per (from_stage, to_stage)
    transitions: dict = defaultdict(list)
    total = total_forward = total_backward = total_lateral = 0

    for (pid, sid), group in df.groupby(['participant_id', 'session_id']):
        if 'speaker' in group.columns:
            pgroup = group[group['speaker'] != 'therapist']
        else:
            pgroup = group
        pgroup = pgroup[pgroup['final_label'].notna()].sort_values('segment_index')
        if len(pgroup) < 2:
            continue

        labels = pgroup['final_label'].astype(int).tolist()
        texts = pgroup['text'].tolist()
        seg_idxs = pgroup['segment_index'].tolist()

        for i in range(len(labels) - 1):
            fr, to = labels[i], labels[i + 1]
            cue = _collect_therapist_cue(df, sid, seg_idxs[i], seg_idxs[i + 1])
            transitions[(fr, to)].append((
                str(texts[i]).strip(),
                cue,
                str(texts[i + 1]).strip(),
            ))
            total += 1
            if to > fr:
                total_forward += 1
            elif to < fr:
                total_backward += 1
            else:
                total_lateral += 1

    sorted_pairs = sorted(transitions.items(), key=lambda x: -len(x[1]))

    lines = []
    lines.append('THERAPIST CUE ANALYSIS')
    lines.append('=' * 60)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')
    lines.append(f'Total within-session transitions: {total}')
    lines.append(f'  Forward:  {total_forward}')
    lines.append(f'  Backward: {total_backward}')
    lines.append(f'  Lateral:  {total_lateral}')
    lines.append('')
    lines.append(
        'For each transition type the "average" blocks show a representative\n'
        'synthesis of all observed examples (LLM-summarized when over the word cap).\n'
    )

    for (fr, to), entries in sorted_pairs:
        n = len(entries)
        direction = 'forward' if to > fr else ('backward' if to < fr else 'lateral/stay')
        fr_name = stage_names.get(fr, str(fr))
        to_name = stage_names.get(to, str(to))
        n_empty_cues = sum(1 for e in entries if not e[1])

        lines.append(f'── {fr_name} → {to_name}  (n={n}, [{direction}])')
        lines.append('─' * 60)
        lines.append(
            f'  n = {n}  |  empty cues (no therapist speech between segments): {n_empty_cues}'
        )
        lines.append('')

        # average FROM
        from_texts = [e[0] for e in entries if e[0]]
        if from_texts:
            agg_from = ' || '.join(from_texts)
            agg_from, _ = _summarize_participant_text(agg_from, llm_client, max_agg)
            lines.append(f'  average FROM [{fr_name}]:')
            lines.append(_wrap_quote(agg_from, indent=4))
            lines.append('')

        # average CUE (skip empty-cue entries)
        cue_texts = [e[1] for e in entries if e[1]]
        if cue_texts:
            agg_cue = ' || '.join(cue_texts)
            agg_cue, _ = _summarize_cue(agg_cue, llm_client, max_agg)
            lines.append('  average CUE:')
            lines.append(_wrap_quote(agg_cue, indent=4))
            lines.append('')
        else:
            lines.append(
                '  average CUE: [none — all transitions had no therapist speech between segments]'
            )
            lines.append('')

        # average TO
        to_texts = [e[2] for e in entries if e[2]]
        if to_texts:
            agg_to = ' || '.join(to_texts)
            agg_to, _ = _summarize_participant_text(agg_to, llm_client, max_agg)
            lines.append(f'  average TO [{to_name}]:')
            lines.append(_wrap_quote(agg_to, indent=4))
            lines.append('')

        lines.append('')

    content = '\n'.join(lines)
    os.makedirs(_paths.human_reports_dir(output_dir), exist_ok=True)
    path = os.path.join(_paths.human_reports_dir(output_dir), 'cue_response.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    return path
