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
