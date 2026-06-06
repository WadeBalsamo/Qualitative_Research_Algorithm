"""Export functions for theme definitions, content validity test sets, and training data."""

import json
import os
from typing import List, Dict

from classification_tools.data_structures import Segment
from theme_framework.theme_schema import ThemeFramework
from .. import output_paths as _paths


def export_theme_definitions(
    framework: ThemeFramework,
    output_path: str,
) -> None:
    """Export theme/stage definitions as JSON."""
    with open(output_path, 'w') as f:
        json.dump(framework.to_json(), f, indent=2)


def export_theme_definitions_txt(
    framework: ThemeFramework,
    theme_config,
    output_path: str,
) -> None:
    """Export a human-readable theme definitions reference document."""
    import datetime as _dt
    import textwrap as _tw

    zero_shot = getattr(theme_config, 'zero_shot_prompt', False)
    n_exemplars = getattr(theme_config, 'prompt_n_exemplars', None)
    include_subtle = getattr(theme_config, 'prompt_include_subtle', True)
    n_subtle = getattr(theme_config, 'prompt_n_subtle', None)
    include_adversarial = getattr(theme_config, 'prompt_include_adversarial', True)
    n_adversarial = getattr(theme_config, 'prompt_n_adversarial', None)

    W = 76
    lines = []
    lines.append('═' * W)
    lines.append(f'THEME DEFINITIONS — {framework.name}  v{framework.version}')
    lines.append('═' * W)
    lines.append(f'Generated   : {_dt.datetime.utcnow().strftime("%Y-%m-%d")}')
    lines.append(f'Themes      : {len(framework.themes)}')
    lines.append(f'Prompt mode : {"zero-shot (no examples)" if zero_shot else "few-shot"}')
    if not zero_shot:
        lines.append(f'Exemplars   : {n_exemplars if n_exemplars is not None else "all"}')
        subtle_str = 'yes' if include_subtle else 'no'
        if include_subtle and n_subtle is not None:
            subtle_str += f'  (max {n_subtle})'
        lines.append(f'Subtle      : {subtle_str}')
        adv_str = 'yes' if include_adversarial else 'no'
        if include_adversarial and n_adversarial is not None:
            adv_str += f'  (max {n_adversarial})'
        lines.append(f'Adversarial : {adv_str}')
    lines.append('')

    for t in sorted(framework.themes, key=lambda x: x.theme_id):
        lines.append('═' * W)
        lines.append(f'  [{t.theme_id}]  {t.name}  ({t.short_name})')
        lines.append('─' * W)
        lines.append('  DEFINITION')
        for line in _tw.wrap(t.definition, width=W - 4,
                             initial_indent='    ', subsequent_indent='    '):
            lines.append(line)
        if t.prototypical_features:
            lines.append('')
            lines.append('  PROTOTYPICAL FEATURES')
            for feat in t.prototypical_features:
                for line in _tw.wrap(feat, width=W - 6,
                                     initial_indent='    • ', subsequent_indent='      '):
                    lines.append(line)
        lines.append('')
        lines.append('  KEY DISTINCTION')
        for line in _tw.wrap(t.distinguishing_criteria, width=W - 4,
                             initial_indent='    ', subsequent_indent='    '):
            lines.append(line)

        if not zero_shot:
            ex = (t.exemplar_utterances if n_exemplars is None
                  else t.exemplar_utterances[:n_exemplars])
            if ex:
                lines.append('')
                lines.append('  EXAMPLES')
                for e in ex:
                    for line in _tw.wrap(f'"{e}"', width=W - 6,
                                         initial_indent='    • ', subsequent_indent='      '):
                        lines.append(line)

            if include_subtle and t.subtle_utterances:
                sub = (t.subtle_utterances if n_subtle is None
                       else t.subtle_utterances[:n_subtle])
                if sub:
                    lines.append('')
                    lines.append('  EDGE CASES (SUBTLE)')
                    for e in sub:
                        for line in _tw.wrap(f'"{e}"', width=W - 6,
                                             initial_indent='    • ', subsequent_indent='      '):
                            lines.append(line)

            if include_adversarial and t.adversarial_utterances:
                adv = (t.adversarial_utterances if n_adversarial is None
                       else t.adversarial_utterances[:n_adversarial])
                if adv:
                    lines.append('')
                    lines.append('  WATCH-OUTS (BOUNDARY CASES)')
                    for e in adv:
                        for line in _tw.wrap(f'"{e}"', width=W - 6,
                                             initial_indent='    • ', subsequent_indent='      '):
                            lines.append(line)

        lines.append('')

    lines.append('═' * W)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')


def export_content_validity_test_set(
    test_items: List[Dict],
    output_path: str,
) -> None:
    """Export content validity test set as JSONL."""
    with open(output_path, 'w') as f:
        for item in test_items:
            f.write(json.dumps(item) + '\n')


def export_training_data(
    segments: List[Segment],
    framework,
    codebook,
    run_dir: str,
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

    training_dir = _paths.training_data_dir(run_dir)
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
