"""
process/legacy_migration.py
---------------------------

# ============================================================================
# LEGACY MIGRATION SHIM — REMOVE IN NEXT VERSION
#
# This module upgrades older project layouts to the current v3 layout:
#
# v2.0 (pre-modular): has master_segments.jsonl but no 01_transcripts/segmented/
#   → migrate_legacy_segments() extracts per-session frozen segments
#
# v2.5: has 01_transcripts/diarized/ or 01_transcripts/coded/ (old path layout)
#   → migrate_v25_to_v3() moves files to their v3 locations
#
# Both are called automatically from stage_ingest() on first encounter.
# Once all in-flight projects have migrated, delete this file and the call
# site in process/orchestrator.py marked `# LEGACY-MIGRATION CALL SITE`.
# ============================================================================
"""
import os
import re
import shutil
from typing import Dict, List, Tuple

from . import output_paths as _paths


# ---------------------------------------------------------------------------
# v2.0 detection and migration (pre-modular: no per-session frozen segments)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# v2.5 detection and migration (old path layout)
# ---------------------------------------------------------------------------

def is_v25_layout(run_dir: str) -> bool:
    """True if the project uses the v2.5 path layout (pre-v3 directory structure).

    Detects by presence of the old diarized or coded transcript subdirectories
    inside 01_transcripts/, or flat content validity files in 04_validation/.
    """
    old_diarized = os.path.isdir(os.path.join(run_dir, '01_transcripts', 'diarized'))
    old_coded = os.path.isdir(os.path.join(run_dir, '01_transcripts', 'coded'))
    cv_flat = os.path.isfile(
        os.path.join(run_dir, '04_validation', 'content_validity_test_set.jsonl')
    )
    human_class_flat = any(
        f.startswith('human_classification_') and f.endswith('.txt')
        for f in _list_dir_safe(os.path.join(run_dir, '04_validation'))
    )
    return old_diarized or old_coded or cv_flat or human_class_flat


def migrate_v25_to_v3(run_dir: str) -> Dict[str, int]:
    """
    Migrate a v2.5 project layout to v3 in-place. Idempotent: skips files
    whose destination already exists. Never modifies frozen testset worksheets.

    Returns a dict with move/deletion counts for each migration step.
    """
    results: Dict[str, int] = {}

    # 1. Move 01_transcripts/diarized/* → 01_transcripts_inputs/
    old_diarized = os.path.join(run_dir, '01_transcripts', 'diarized')
    new_inputs = _paths.transcripts_diarized_dir(run_dir)  # 01_transcripts_inputs/
    if os.path.isdir(old_diarized):
        os.makedirs(new_inputs, exist_ok=True)
        n = 0
        for f in os.listdir(old_diarized):
            src = os.path.join(old_diarized, f)
            dst = os.path.join(new_inputs, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                n += 1
        if not os.listdir(old_diarized):
            os.rmdir(old_diarized)
        results['diarized_moved'] = n

    # 2. Move 01_transcripts/coded/* → 04_validation/full_transcripts/
    old_coded = os.path.join(run_dir, '01_transcripts', 'coded')
    full_transcripts = _paths.full_transcripts_dir(run_dir)
    if os.path.isdir(old_coded):
        os.makedirs(full_transcripts, exist_ok=True)
        n = 0
        for f in os.listdir(old_coded):
            src = os.path.join(old_coded, f)
            dst = os.path.join(full_transcripts, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                n += 1
        if not os.listdir(old_coded):
            os.rmdir(old_coded)
        results['coded_moved'] = n

    # 3. Move 04_validation/human_classification_*.txt → 04_validation/full_transcripts/
    val_dir = _paths.validation_dir(run_dir)
    if os.path.isdir(val_dir):
        os.makedirs(full_transcripts, exist_ok=True)
        n = 0
        for f in _list_dir_safe(val_dir):
            if f.startswith('human_classification_') and f.endswith('.txt'):
                src = os.path.join(val_dir, f)
                dst = os.path.join(full_transcripts, f)
                if os.path.isfile(src) and not os.path.exists(dst):
                    shutil.move(src, dst)
                    n += 1
        results['human_classification_moved'] = n

    # 4. Move flat content validity files → 04_validation/content_validity/
    cv_dir = _paths.content_validity_dir(run_dir)  # 04_validation/content_validity/
    _cv_flat_files = [
        'content_validity_test_set.jsonl',
        'content_validity_human_worksheet.txt',
        'content_validity_definition_key.txt',
        'content_validity_answer_key.txt',
    ]
    if os.path.isdir(val_dir):
        os.makedirs(cv_dir, exist_ok=True)
        n = 0
        for f in _cv_flat_files:
            src = os.path.join(val_dir, f)
            dst = os.path.join(cv_dir, f)
            if os.path.isfile(src) and not os.path.exists(dst):
                shutil.move(src, dst)
                n += 1
        results['cv_files_moved'] = n

    # 5. Delete legacy worksheetN_base.txt files from testsets dir
    ts_dir = _paths.testsets_dir(run_dir)
    n = 0
    _base_pattern = re.compile(r'^worksheet\d+_base\.txt$')
    for f in _list_dir_safe(ts_dir):
        if _base_pattern.match(f):
            os.remove(os.path.join(ts_dir, f))
            n += 1
    results['legacy_base_removed'] = n

    # 6. Delete folder-based testset dirs (contain manifest.json or segments_snapshot.jsonl)
    n = 0
    for name in _list_dir_safe(ts_dir):
        d = os.path.join(ts_dir, name)
        if not os.path.isdir(d):
            continue
        has_manifest = os.path.isfile(os.path.join(d, 'manifest.json'))
        has_snapshot = os.path.isfile(os.path.join(d, 'segments_snapshot.jsonl'))
        if has_manifest or has_snapshot:
            shutil.rmtree(d)
            n += 1
    results['legacy_testset_dirs_removed'] = n

    return results


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

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
