"""
process/legacy_migration.py
---------------------------

# ============================================================================
# LEGACY MIGRATION SHIM — REMOVE IN NEXT VERSION
#
# This module exists solely to upgrade pre-modular project directories
# (those that have a populated 02_meta/training_data/master_segments.jsonl
# but no 01_transcripts/segmented/ tree) to the frozen-segments + frozen-
# testsets layout introduced in Phase 1.
#
# Once the in-flight projects have all been migrated, delete this file
# and the two call sites in process/orchestrator.py marked
# `# LEGACY-MIGRATION CALL SITE` (search the repo).
# ============================================================================
"""
import datetime
import json
import os
import re
import shutil
from typing import List, Tuple

from . import output_paths as _paths


def is_legacy_project(run_dir: str) -> bool:
    """True iff there is master_segments.jsonl but no 01_transcripts/segmented/."""
    ms_dir = _paths.master_segments_dir(run_dir)
    has_master = any(
        f.startswith('master_segments') and f.endswith('.jsonl')
        for f in _list_dir_safe(ms_dir)
    )
    segmented_dir = _paths.segmented_sessions_dir(run_dir)
    has_segmented = os.path.isdir(segmented_dir) and bool(os.listdir(segmented_dir))
    return has_master and not has_segmented


def migrate_legacy_segments(run_dir: str) -> int:
    """
    Read master_segments.jsonl, group by session_id, write per-session
    segments.jsonl files containing only raw-segmentation fields.

    Marks segmentation_meta.json with params_hash='legacy-pre-modular' so
    subsequent runs don't trigger re-segmentation just because params drifted.
    Returns number of sessions migrated.
    """
    from .segments_io import write_session_segments

    segments = _load_master_segments_raw(run_dir)
    if not segments:
        return 0

    by_session: dict = {}
    for seg in segments:
        by_session.setdefault(seg.session_id, []).append(seg)

    n = 0
    for sid, segs in by_session.items():
        segs.sort(key=lambda s: s.segment_index)
        segs_path = _paths.session_segments_path(run_dir, sid)
        if os.path.exists(segs_path):
            continue
        write_session_segments(run_dir, sid, segs, 'legacy-pre-modular')
        n += 1

    return n


def migrate_legacy_testsets(run_dir: str) -> int:
    """
    For every legacy 04_validation/testsets/{human,AI}_classification_testset_worksheet_<n>.txt
    pair, create 04_validation/testsets/<name>/ and move the legacy files in.

    Writes manifest.json + segments_snapshot.jsonl alongside the moved worksheets.
    Returns number of testsets migrated.
    """
    ts_dir = _paths.testsets_dir(run_dir)
    if not os.path.isdir(ts_dir):
        return 0

    segments = _load_master_segments_raw(run_dir)
    seg_lookup = {(s.session_id, s.segment_index): s for s in segments}

    pattern = re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    indices = []
    for fname in os.listdir(ts_dir):
        m = pattern.match(fname)
        if m:
            indices.append(int(m.group(1)))

    n = 0
    for idx in sorted(indices):
        human_src = os.path.join(ts_dir, f'human_classification_testset_worksheet_{idx}.txt')
        ai_src = os.path.join(ts_dir, f'AI_classification_testset_worksheet_{idx}.txt')
        if not os.path.isfile(human_src):
            continue

        name = f'vaamr_testset_{idx}'
        new_dir = _paths.testset_dir(run_dir, name)
        human_dst = _paths.testset_human_worksheet_path(run_dir, name)

        if os.path.exists(human_dst):
            continue

        os.makedirs(new_dir, exist_ok=True)

        items = _parse_worksheet_items(human_src)
        segs_in_testset = []
        for session_id, seg_num_1based in items:
            seg_index = seg_num_1based - 1
            seg = seg_lookup.get((session_id, seg_index))
            if seg is not None:
                segs_in_testset.append(seg)

        from ._freeze import sha256_text

        manifest = {
            'kind': 'vaamr',
            'name': name,
            'set_index': idx,
            'n_sets': max(indices),
            'seed': None,
            'fraction': None,
            'framework': {'name': 'vaamr', 'version': 'legacy'},
            'segment_ids': [s.segment_id for s in segs_in_testset],
            'content_sha256': {s.segment_id: sha256_text(s.text) for s in segs_in_testset},
            'created_at': datetime.datetime.utcnow().isoformat() + 'Z',
            'migrated_from_legacy': True,
        }
        with open(os.path.join(new_dir, 'manifest.json'), 'w', encoding='utf-8') as fh:
            json.dump(manifest, fh, indent=2)

        with open(os.path.join(new_dir, 'segments_snapshot.jsonl'), 'w', encoding='utf-8') as fh:
            for seg in segs_in_testset:
                rec = {
                    'segment_id': seg.segment_id,
                    'session_id': seg.session_id,
                    'segment_index': seg.segment_index,
                    'participant_id': seg.participant_id,
                    'speaker': seg.speaker,
                    'start_time_ms': seg.start_time_ms,
                    'end_time_ms': seg.end_time_ms,
                    'word_count': seg.word_count,
                    'speakers_in_segment': seg.speakers_in_segment,
                    'text': seg.text,
                    'content_sha256': sha256_text(seg.text),
                }
                fh.write(json.dumps(rec) + '\n')

        shutil.move(human_src, human_dst)
        if os.path.isfile(ai_src):
            shutil.move(ai_src, _paths.testset_answer_key_path(run_dir, name))

        n += 1

    return n


def _load_master_segments_raw(run_dir: str):
    """Load Segment objects from master_segments.jsonl for migration purposes."""
    import glob
    try:
        from .segments_io import _load_segments_from_jsonl
        ms_dir = _paths.master_segments_dir(run_dir)
        jsonl_files = sorted(glob.glob(os.path.join(ms_dir, 'master_segments*.jsonl')))
        if not jsonl_files:
            jsonl_files = sorted(glob.glob(os.path.join(run_dir, 'master_segments_*.jsonl')))
        if not jsonl_files:
            return []
        return _load_segments_from_jsonl(jsonl_files[-1])
    except Exception:
        return []


def _parse_worksheet_items(worksheet_path: str) -> List[Tuple[str, int]]:
    """
    Parse (session_id, segment_number_1based) pairs from an item-header line:
      [ITEM NNN]  Session: <session_id>   Segment NNN
    """
    pattern = re.compile(r'\[ITEM\s+\d+\]\s+Session:\s+(\S+)\s+Segment\s+(\d+)')
    items = []
    with open(worksheet_path, encoding='utf-8') as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                items.append((m.group(1), int(m.group(2))))
    return items


def _list_dir_safe(path: str) -> List[str]:
    try:
        return os.listdir(path)
    except OSError:
        return []
