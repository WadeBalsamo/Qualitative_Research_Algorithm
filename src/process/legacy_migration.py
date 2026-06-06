"""
process/legacy_migration.py
---------------------------

# ============================================================================
# LEGACY MIGRATION SHIM — legacy-only; safe to remove once no pre-v3 projects remain
#
# This module upgrades older project layouts to the current layout:
#
# v2.0 (pre-modular): has master_segments.jsonl but no frozen per-session segments
#   → migrate_legacy_segments() extracts per-session frozen segments (into SQLite)
#
# v2.5: has 01_transcripts/diarized/ or 01_transcripts/coded/ (old path layout)
#   → migrate_v25_to_v3() moves files to their v3 locations
#
# v3-jsonl: per-session segments.jsonl + 02_meta/classifications/*_labels.jsonl
#           + testset/CV JSON, but no qra.db (the pre-SQLite main-branch layout)
#   → migrate_jsonl_to_sqlite() imports it all into the SQLite store
#
# All three are called automatically from stage_ingest() on first encounter, in
# order, and are idempotent.  Once all in-flight projects have migrated, delete
# this file and the call site in process/orchestrator.py marked
# `# LEGACY-MIGRATION CALL SITE`.
# ============================================================================
"""
import copy
import glob
import json
import os
import re
import shutil
from typing import Dict, List, Optional, Tuple

from . import db as _db
from . import output_paths as _paths

_TS_META_RE = re.compile(r'^human_classification_testset_worksheet_(\d+)\.meta\.json$')


# ---------------------------------------------------------------------------
# v2.0 detection and migration (pre-modular: no per-session frozen segments)
# ---------------------------------------------------------------------------

def is_legacy_project(run_dir: str) -> bool:
    """True iff there is master_segments.jsonl but no frozen segments yet.

    "No frozen segments" now means: the SQLite ``segments`` table is empty AND
    there is no populated 01_transcripts/segmented/ directory.  This keeps the
    detector from re-firing after migrate_legacy_segments() has written rows to
    the database (under SQLite no per-session segments.jsonl is created).
    """
    from . import segments_io as _sio
    ms_dir = _paths.master_segments_dir(run_dir)
    has_master = any(
        f.startswith('master_segments') and f.endswith('.jsonl')
        for f in _list_dir_safe(ms_dir)
    )
    if not has_master:
        return False
    segmented_dir = _paths.segmented_sessions_dir(run_dir)
    has_segmented = bool(_sio.list_segmented_sessions(run_dir)) or (
        os.path.isdir(segmented_dir) and bool(os.listdir(segmented_dir))
    )
    return not has_segmented


def migrate_legacy_segments(run_dir: str) -> int:
    """
    Read master_segments.jsonl, group by session_id, write per-session frozen
    segments (now into the SQLite ``segments`` table) containing only
    raw-segmentation fields.

    Marks params_hash='legacy-pre-modular' so subsequent runs don't trigger
    re-segmentation just because params drifted.  Idempotent: sessions already
    present in the store are skipped.  Returns number of sessions migrated.
    """
    from .segments_io import write_session_segments, list_segmented_sessions

    segments = _load_master_segments_raw(run_dir)
    if not segments:
        return 0

    by_session: dict = {}
    for seg in segments:
        by_session.setdefault(seg.session_id, []).append(seg)

    existing = set(list_segmented_sessions(run_dir))
    n = 0
    for sid, segs in by_session.items():
        if sid in existing:
            continue
        segs.sort(key=lambda s: s.segment_index)
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
# v3-jsonl → SQLite migration (the pre-SQLite main-branch layout)
# ---------------------------------------------------------------------------

def is_jsonl_project(run_dir: str) -> bool:
    """True if v3 JSONL pipeline files exist but qra.db does not.

    Detects projects created on main branch before the SQLite migration.
    Returns False once qra.db exists (so the migration never re-runs).
    """
    if _db.db_exists(run_dir):
        return False
    seg_dir = _paths.segmented_sessions_dir(run_dir)
    has_segments = os.path.isdir(seg_dir) and any(
        os.path.isfile(os.path.join(seg_dir, sid, 'segments.jsonl'))
        for sid in _list_dir_safe(seg_dir)
    )
    cls_dir = _paths.classifications_dir(run_dir)
    has_overlays = any(f.endswith('.jsonl') for f in _list_dir_safe(cls_dir))
    return has_segments or has_overlays


def migrate_jsonl_to_sqlite(run_dir: str) -> Dict:
    """
    Import all JSONL/JSON pipeline data into the SQLite store in one transaction.

    Idempotent: guarded by ``is_jsonl_project`` (qra.db absent).  Reads the OLD
    files directly (NOT via the now-SQLite-backed io modules, which would read
    the empty new DB), writes everything to ``qra.db.tmp``, then atomically
    renames it onto ``qra.db`` — so a crash mid-migration leaves no half-written
    store.  Finally relocates the migrated originals to ``<run_dir>/_legacy_files/``
    (non-destructive; never deletes).

    Returns counts: {'sessions', 'segments', 'overlays': {key: n}, 'manifest_keys',
                     'testset_worksheets', 'cv_testsets'}.
    """
    from classification_tools.data_structures import Segment
    from . import classifications_io as _cio
    from . import segments_io as _sio

    counts: Dict = {
        'sessions': 0, 'segments': 0, 'overlays': {}, 'manifest_keys': 0,
        'testset_worksheets': 0, 'cv_testsets': 0,
    }
    if _db.db_exists(run_dir):
        return counts

    final_db = _db.db_path(run_dir)
    tmp_db = final_db + '.tmp'
    if os.path.exists(tmp_db):
        os.remove(tmp_db)

    conn = _db.connect(tmp_db)
    try:
        _db.ensure_schema(conn)

        # 1. Frozen segments (+ segmentation provenance).
        seg_dir = _paths.segmented_sessions_dir(run_dir)
        for sid in sorted(_list_dir_safe(seg_dir)):
            segs_path = os.path.join(seg_dir, sid, 'segments.jsonl')
            if not os.path.isfile(segs_path):
                continue
            meta = _read_json(os.path.join(seg_dir, sid, 'segmentation_meta.json')) or {}
            phash = str(meta.get('params_hash', '') or '')
            ts = str(meta.get('ingest_timestamp', '') or '')
            rows = [
                _sio._seg_insert_row(_segment_from_raw_record(rec), phash, ts)
                for rec in _read_jsonl(segs_path)
            ]
            if rows:
                conn.executemany(_sio._INSERT_SEGMENT_SQL, rows)
                counts['segments'] += len(rows)
                counts['sessions'] += 1

        # 2. Classification overlays.
        cls_dir = _paths.classifications_dir(run_dir)
        for key in _cio.OVERLAY_KEYS:
            path = os.path.join(cls_dir, _cio.OVERLAY_FILENAMES[key])
            if not os.path.isfile(path):
                continue
            table = _cio._OVERLAY_TABLES[key]
            fields = _cio._OVERLAY_FIELDS_MAP[key]
            json_fields = _cio._OVERLAY_JSON_FIELDS[key]
            cols = ('segment_id',) + fields
            sql = (
                f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
                f"VALUES ({', '.join('?' for _ in cols)})"
            )
            n = 0
            for rec in _read_jsonl(path):
                vals = [rec.get('segment_id')]
                for f in fields:
                    v = rec.get(f)
                    if f in json_fields:
                        v = _db.dumps(v)
                    vals.append(v)
                conn.execute(sql, tuple(vals))
                n += 1
            if n:
                counts['overlays'][key] = n

        # 3. Classification manifest.
        manifest = _read_json(_paths.classification_manifest_path(run_dir))
        if isinstance(manifest, dict):
            for k, entry in manifest.items():
                conn.execute(
                    "INSERT OR REPLACE INTO classification_manifest (key, entry_json) "
                    "VALUES (?, ?)",
                    (k, _db.dumps(entry)),
                )
                counts['manifest_keys'] += 1

        # 4. Validation testset worksheets (testset_meta/*.meta.json + .txt header).
        ts_meta_dir = os.path.join(_paths.meta_dir(run_dir), 'testset_meta')
        for mp in sorted(glob.glob(os.path.join(ts_meta_dir, '*.meta.json'))):
            m = _TS_META_RE.match(os.path.basename(mp))
            if not m:
                continue
            wn = int(m.group(1))
            meta = _read_json(mp) or {}
            segs = meta.get('segments', []) or []
            legacy_import = 1 if meta.get('legacy_import') else 0
            conn.execute(
                "INSERT OR REPLACE INTO testset_worksheets "
                "(worksheet_n, kind, name, created_at, n_items, params_hash, frozen, legacy_import) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (wn, _sniff_testset_kind(run_dir, wn), None, '', len(segs), None, 1, legacy_import),
            )
            for i, e in enumerate(segs, 1):
                conn.execute(
                    "INSERT OR REPLACE INTO testset_items "
                    "(worksheet_n, item_num, session_id, seg_num, sha256) VALUES (?, ?, ?, ?, ?)",
                    (wn, i, e.get('session_id', ''), int(e.get('seg_num') or 0), e.get('sha256')),
                )
            counts['testset_worksheets'] += 1

        # 5. Content-validity testsets (manifest.json + items.jsonl).
        cv_root = _paths.content_validity_dir(run_dir)
        for name in sorted(_list_dir_safe(cv_root)):
            man = _read_json(os.path.join(cv_root, name, 'manifest.json'))
            if not isinstance(man, dict):
                continue
            ts_name = man.get('name') or name
            fw = man.get('framework', {}) or {}
            conn.execute(
                "INSERT OR REPLACE INTO cv_testsets "
                "(name, kind, framework_name, framework_version, created_at) VALUES (?, ?, ?, ?, ?)",
                (ts_name, man.get('kind', 'vaamr'), fw.get('name', ''),
                 str(fw.get('version', '1')), man.get('created_at', '')),
            )
            for i, rec in enumerate(_read_jsonl(os.path.join(cv_root, name, 'items.jsonl'))):
                conn.execute(
                    "INSERT OR REPLACE INTO cv_testset_items "
                    "(testset_name, item_id, ord, text, expected_stage, difficulty, source_field, content_sha256) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    (ts_name, rec.get('id', ''), i, rec.get('text', ''),
                     rec.get('expected_stage'), rec.get('difficulty'),
                     rec.get('source_field'), rec.get('content_sha256')),
                )
            counts['cv_testsets'] += 1

        _db.set_meta(conn, 'migrated_from', 'jsonl_v3')
        conn.commit()
    except Exception:
        conn.close()
        if os.path.exists(tmp_db):
            try:
                os.remove(tmp_db)
            except OSError:
                pass
        raise
    conn.close()

    # Commit point: the DB is now authoritative and complete.
    os.replace(tmp_db, final_db)

    # Best-effort relocation of migrated originals (never fail the migration).
    try:
        _relocate_legacy_jsonl(run_dir)
    except OSError:
        pass

    return counts


# ---------------------------------------------------------------------------
# Config-file upgrade (fill defaults for parameters added since the file was written)
# ---------------------------------------------------------------------------

def _config_file_path(run_dir: str) -> Optional[str]:
    """Locate the project's qra_config.json (02_meta first, then run_dir root)."""
    for candidate in (
        os.path.join(_paths.meta_dir(run_dir), 'qra_config.json'),
        os.path.join(run_dir, 'qra_config.json'),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def upgrade_config_file(run_dir: str) -> bool:
    """Fill in every parameter added since a legacy qra_config.json was written.

    The file stays in its wizard layout (``pipeline`` nesting + ``framework`` /
    ``codebook`` selection keys preserved).  Missing sub-config blocks
    (e.g. gnn_layer / superposition / efficacy / validation / codebook_*) are
    added in full from PipelineConfig() defaults; missing fields inside existing
    blocks are filled; missing scalar pipeline flags are added under ``pipeline``.
    Non-destructive: no existing value or unknown key is ever overwritten.

    Returns True iff the file was changed.  Idempotent (a no-op once complete).
    """
    path = _config_file_path(run_dir)
    if path is None:
        return False
    data = _read_json(path)
    if not isinstance(data, dict):
        return False

    try:
        from .config import PipelineConfig
        defaults = PipelineConfig().to_json()
    except Exception:
        return False

    before = json.dumps(data, sort_keys=True, default=str)
    _merge_config_defaults(data, defaults)
    after = json.dumps(data, sort_keys=True, default=str)
    if before == after:
        return False

    tmp = path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp, path)
    except OSError:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        return False
    return True


def _merge_config_defaults(data: dict, defaults: dict) -> None:
    """Add missing keys from the flat PipelineConfig().to_json() ``defaults`` into
    the wizard-format ``data``.  Sub-config dicts merge into their top-level block;
    scalar PipelineConfig fields are nested under ``data['pipeline']`` (wizard
    convention) unless already present at top level or under pipeline."""
    pipeline = data.setdefault('pipeline', {})
    for key, dval in defaults.items():
        if isinstance(dval, dict):
            existing = data.get(key)
            if isinstance(existing, dict):
                _deep_fill(existing, dval)
            elif key not in data:
                data[key] = copy.deepcopy(dval)
        else:
            if key in data or key in pipeline:
                continue
            pipeline[key] = copy.deepcopy(dval)


def _deep_fill(target: dict, default: dict) -> None:
    """Recursively add keys present in ``default`` but missing in ``target``.
    Never overwrites an existing value."""
    for k, v in default.items():
        if k not in target:
            target[k] = copy.deepcopy(v)
        elif isinstance(v, dict) and isinstance(target.get(k), dict):
            _deep_fill(target[k], v)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _read_jsonl(path: str):
    """Yield parsed JSON objects from a JSONL file (empty if absent)."""
    import json
    if not os.path.isfile(path):
        return
    with open(path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_json(path: str):
    """Load a JSON file, or None if absent/unparseable."""
    import json
    try:
        with open(path, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _segment_from_raw_record(rec: dict):
    """Reconstruct a Segment from an old segments.jsonl record (raw fields only)."""
    from classification_tools.data_structures import Segment
    from .segments_io import _int_or_none
    return Segment(
        segment_id=str(rec.get('segment_id', '') or ''),
        trial_id=str(rec.get('trial_id', '') or ''),
        participant_id=str(rec.get('participant_id', '') or ''),
        session_id=str(rec.get('session_id', '') or ''),
        session_number=int(rec.get('session_number') or 0),
        cohort_id=_int_or_none(rec.get('cohort_id')),
        session_variant=str(rec.get('session_variant', '') or ''),
        segment_index=int(rec.get('segment_index') or 0),
        start_time_ms=int(rec.get('start_time_ms') or 0),
        end_time_ms=int(rec.get('end_time_ms') or 0),
        total_segments_in_session=int(rec.get('total_segments_in_session') or 0),
        speaker=str(rec.get('speaker', '') or ''),
        text=str(rec.get('text', '') or ''),
        word_count=int(rec.get('word_count') or 0),
        speakers_in_segment=rec.get('speakers_in_segment'),
        session_file=str(rec.get('session_file', '') or ''),
    )


def _sniff_testset_kind(run_dir: str, n: int) -> str:
    """Detect a flat testset's kind from its (on-disk) human worksheet header."""
    try:
        with open(_paths.testset_human_flat_path(run_dir, n), encoding='utf-8') as fh:
            head = fh.read(4000).upper()
    except OSError:
        return 'vaamr'
    if 'PURER' in head:
        return 'purer'
    if 'CODEBOOK' in head:
        return 'codebook'
    return 'vaamr'


def _relocate_legacy_jsonl(run_dir: str) -> None:
    """Move migrated JSONL/JSON originals to <run_dir>/_legacy_files/ (preserve subpaths)."""
    legacy_root = os.path.join(run_dir, '_legacy_files')

    def _move(abs_path: str) -> None:
        if not os.path.isfile(abs_path):
            return
        dst = os.path.join(legacy_root, os.path.relpath(abs_path, run_dir))
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            try:
                shutil.move(abs_path, dst)
            except OSError:
                pass

    seg_dir = _paths.segmented_sessions_dir(run_dir)
    for sid in _list_dir_safe(seg_dir):
        _move(os.path.join(seg_dir, sid, 'segments.jsonl'))
        _move(os.path.join(seg_dir, sid, 'segmentation_meta.json'))

    cls_dir = _paths.classifications_dir(run_dir)
    for f in _list_dir_safe(cls_dir):
        if f.endswith('.jsonl') or f == 'classification_manifest.json':
            _move(os.path.join(cls_dir, f))

    ts_meta_dir = os.path.join(_paths.meta_dir(run_dir), 'testset_meta')
    for f in _list_dir_safe(ts_meta_dir):
        if f.endswith('.meta.json'):
            _move(os.path.join(ts_meta_dir, f))

    cv_root = _paths.content_validity_dir(run_dir)
    for name in _list_dir_safe(cv_root):
        _move(os.path.join(cv_root, name, 'manifest.json'))
        _move(os.path.join(cv_root, name, 'items.jsonl'))

    ms_dir = _paths.master_segments_dir(run_dir)
    for f in _list_dir_safe(ms_dir):
        if f.startswith('master_segments') and f.endswith('.jsonl'):
            _move(os.path.join(ms_dir, f))


def _load_master_segments_raw(run_dir: str):
    """Load Segment objects from master_segments.jsonl for migration purposes."""
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
