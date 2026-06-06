"""
process/segments_io.py
----------------------
Frozen per-session segmentation I/O and master-segments reader.

Storage is the project SQLite database (``qra.db``, see ``process.db``): frozen
raw-segmentation fields live in the ``segments`` table (one row per segment),
with the former ``segmentation_meta.json`` provenance carried as the
``params_hash`` / ``segmenter_version`` / ``ingest_timestamp`` columns.

The public API is unchanged from the JSONL era — callers don't know whether the
backing store is per-session JSONL files or a SQLite table.  ``write_frozen``-style
overwrite protection is preserved: a session that already has rows raises
:class:`FrozenArtifactError` unless ``force=True``.

``_load_segments_from_jsonl`` is retained: ``legacy_migration`` uses it to read a
genuinely-old monolithic ``master_segments.jsonl`` during v2.0 migration.
"""
import datetime
import hashlib
import json
from typing import List, Optional

from classification_tools.data_structures import Segment
from . import db
from . import output_paths as _paths
from ._freeze import FrozenArtifactError

# Raw-segmentation fields persisted to the ``segments`` table.
# Classification-only fields are intentionally excluded (they live in overlays).
_RAW_FIELDS = (
    'segment_id', 'trial_id', 'participant_id', 'session_id',
    'session_number', 'cohort_id', 'session_variant',
    'segment_index', 'start_time_ms', 'end_time_ms',
    'total_segments_in_session', 'speaker', 'text', 'word_count',
    'speakers_in_segment', 'session_file',
)

# Field stored as a JSON TEXT column rather than a scalar.
_JSON_FIELDS = frozenset({'speakers_in_segment'})

# Segmentation parameters whose values affect output (used for params_hash).
_HASH_FIELDS = (
    'embedding_model', 'silence_threshold_ms', 'semantic_shift_percentile',
    'min_segment_words_conversational', 'max_segment_words_conversational',
    'max_gap_seconds', 'min_words_per_sentence', 'max_segment_duration_seconds',
    'use_adaptive_threshold', 'min_prominence', 'broad_window_size',
    'use_topic_clustering', 'use_llm_refinement', 'llm_refinement_mode',
    'llm_ambiguity_threshold', 'llm_batch_size',
)

# Column order for INSERT into ``segments`` (raw fields + provenance).
_SEGMENT_COLUMNS = _RAW_FIELDS + ('params_hash', 'segmenter_version', 'ingest_timestamp')
_INSERT_SEGMENT_SQL = (
    "INSERT INTO segments (" + ', '.join(_SEGMENT_COLUMNS) + ") "
    "VALUES (" + ', '.join('?' for _ in _SEGMENT_COLUMNS) + ")"
)
SEGMENTER_VERSION = '1'


def resolve_session_id(session_file: str) -> str:
    """Extract the session_id from a transcript file path.

    VTT files are keyed by their basename (stem); JSON diarization files
    are keyed by the parent directory name (which holds result.json).
    """
    import os
    if session_file.lower().endswith('.vtt'):
        return os.path.splitext(os.path.basename(session_file))[0]
    return os.path.basename(os.path.dirname(session_file))


def params_hash(seg_cfg) -> str:
    """
    SHA-256 of canonical-sorted JSON of segmentation parameters that affect output.
    """
    d = {f: getattr(seg_cfg, f, None) for f in _HASH_FIELDS}
    payload = json.dumps(d, sort_keys=True)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Row <-> Segment mapping
# ---------------------------------------------------------------------------

def _seg_insert_row(seg: Segment, params_hash_val: str, ts: str,
                    segmenter_version: str = SEGMENTER_VERSION) -> tuple:
    """Build the INSERT tuple for ``seg`` in _SEGMENT_COLUMNS order."""
    vals = []
    for f in _RAW_FIELDS:
        v = getattr(seg, f, None)
        if f in _JSON_FIELDS:
            v = db.dumps(v)
        vals.append(v)
    vals.extend([params_hash_val, segmenter_version, ts])
    return tuple(vals)


def _row_to_segment(row) -> Segment:
    """Reconstruct a Segment from a ``segments`` row (raw fields only)."""
    return Segment(
        segment_id=str(row['segment_id'] or ''),
        trial_id=str(row['trial_id'] or ''),
        participant_id=str(row['participant_id'] or ''),
        session_id=str(row['session_id'] or ''),
        session_number=int(row['session_number'] or 0),
        cohort_id=_int_or_none(row['cohort_id']),
        session_variant=str(row['session_variant'] or ''),
        segment_index=int(row['segment_index'] or 0),
        start_time_ms=int(row['start_time_ms'] or 0),
        end_time_ms=int(row['end_time_ms'] or 0),
        total_segments_in_session=int(row['total_segments_in_session'] or 0),
        speaker=str(row['speaker'] or ''),
        text=str(row['text'] or ''),
        word_count=int(row['word_count'] or 0),
        speakers_in_segment=db.loads(row['speakers_in_segment']),
        session_file=str(row['session_file'] or ''),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write_session_segments(
    run_dir: str,
    session_id: str,
    segments: List[Segment],
    current_params_hash: str,
    *,
    force: bool = False,
) -> str:
    """
    Write frozen per-session segments into the ``segments`` table.

    Only raw-segmentation fields are persisted; classification fields are dropped.
    A session that already has rows raises :class:`FrozenArtifactError` unless
    ``force=True`` (which deletes the existing rows first).  Returns the DB path.
    """
    with db.open_db(run_dir) as conn:
        existing = conn.execute(
            "SELECT COUNT(*) AS c FROM segments WHERE session_id = ?",
            (session_id,),
        ).fetchone()['c']
        if existing > 0:
            if not force:
                raise FrozenArtifactError(
                    f"Frozen segments already exist for session {session_id!r} "
                    f"in {run_dir}  (pass force=True to overwrite)"
                )
            conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
        ts = datetime.datetime.utcnow().isoformat() + 'Z'
        rows = [_seg_insert_row(seg, current_params_hash, ts) for seg in segments]
        if rows:
            conn.executemany(_INSERT_SEGMENT_SQL, rows)
    return _paths.db_path(run_dir)


def overwrite_segment_texts(run_dir: str, session_id: str, segments: List[Segment]) -> str:
    """
    Rewrite the raw fields of an already-frozen session in place, WITHOUT
    touching segmentation provenance (params_hash / ingest_timestamp).

    Used by `qra apply-anonymization` for retroactive PHI scrubbing of already-frozen
    segments. The caller is responsible for any confirmation prompt.

    Implemented as a session-level replace (DELETE + re-INSERT) so that callers may
    also change ``segment_id`` (the anonymization cascade does) — a per-row UPDATE
    keyed on segment_id could not rename it.  The session's existing segmentation
    provenance (params_hash / segmenter_version / ingest_timestamp) is preserved.

    Returns the DB path.  Raises FileNotFoundError if the session has no rows.
    """
    if not db.db_exists(run_dir):
        raise FileNotFoundError(
            f"No frozen segments found for session {session_id!r} in {run_dir}"
        )
    with db.open_db(run_dir) as conn:
        meta = conn.execute(
            "SELECT params_hash, segmenter_version, ingest_timestamp "
            "FROM segments WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
        if meta is None:
            raise FileNotFoundError(
                f"No frozen segments found for session {session_id!r} in {run_dir}"
            )
        phash = meta['params_hash'] or ''
        sver = meta['segmenter_version'] or SEGMENTER_VERSION
        ts = meta['ingest_timestamp'] or ''
        conn.execute("DELETE FROM segments WHERE session_id = ?", (session_id,))
        rows = [_seg_insert_row(seg, phash, ts, sver) for seg in segments]
        if rows:
            conn.executemany(_INSERT_SEGMENT_SQL, rows)
    return _paths.db_path(run_dir)


def read_session_segments(run_dir: str, session_id: str) -> List[Segment]:
    """
    Reconstruct Segment objects for a session, ordered by segment_index.

    Only raw-segmentation fields are restored; all classification fields
    remain at their dataclass defaults.
    """
    if not db.db_exists(run_dir):
        return []
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT * FROM segments WHERE session_id = ? ORDER BY segment_index",
            (session_id,),
        ).fetchall()
    return [_row_to_segment(r) for r in rows]


def list_segmented_sessions(run_dir: str) -> List[str]:
    """Return session IDs that have frozen segments, sorted."""
    if not db.db_exists(run_dir):
        return []
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM segments ORDER BY session_id"
        ).fetchall()
    return [r['session_id'] for r in rows]


def is_segmentation_fresh(run_dir: str, session_id: str, current_params_hash: str) -> bool:
    """
    True iff frozen segments exist for session_id AND the stored params_hash
    matches current_params_hash, OR the stored hash is the legacy sentinel
    'legacy-pre-modular' (so legacy-migrated sessions don't force re-segmentation).
    """
    if not db.db_exists(run_dir):
        return False
    with db.open_db(run_dir) as conn:
        row = conn.execute(
            "SELECT params_hash FROM segments WHERE session_id = ? LIMIT 1",
            (session_id,),
        ).fetchone()
    if row is None:
        return False
    stored_hash = row['params_hash'] or ''
    return stored_hash == 'legacy-pre-modular' or stored_hash == current_params_hash


def read_master_segments(run_dir: str) -> List[Segment]:
    """
    Reconstruct all Segment objects with every classification overlay applied.

    This is the canonical post-hoc reader used by qra.py testset commands.
    Backed by the SQLite store: frozen segments + theme/purer/codebook/cv/gnn
    overlays (equivalent to ``load_segments_for_stage`` with all overlays).

    Raises FileNotFoundError if no frozen segments exist.
    """
    return load_segments_for_stage(
        run_dir, apply=('theme', 'purer', 'codebook', 'cv', 'gnn')
    )


def _load_segments_from_jsonl(jsonl_path: str) -> List[Segment]:
    """Reconstruct Segment objects from a single (legacy) master_segments JSONL file.

    Retained for ``legacy_migration`` (v2.0 monolithic ``master_segments.jsonl``);
    not used by the SQLite read path.
    """
    import pandas as pd

    df = pd.read_json(jsonl_path, lines=True)
    segments: List[Segment] = []

    for _, row in df.iterrows():
        rater_ids = None
        if isinstance(row.get('rater_ids'), str):
            try:
                rater_ids = json.loads(row['rater_ids'])
            except (json.JSONDecodeError, TypeError):
                pass

        rater_votes = None
        if isinstance(row.get('rater_votes'), str):
            try:
                rater_votes = json.loads(row['rater_votes'])
            except (json.JSONDecodeError, TypeError):
                pass

        seg = Segment(
            segment_id=str(row['segment_id']),
            trial_id=str(row.get('trial_id', '')),
            participant_id=str(row.get('participant_id', '')),
            session_id=str(row['session_id']),
            session_number=_int_or_none(row.get('session_number')) or 1,
            cohort_id=_int_or_none(row.get('cohort_id')),
            session_variant=str(row.get('session_variant', '') or ''),
            segment_index=_int_or_none(row.get('segment_index')) or 0,
            start_time_ms=_int_or_none(row.get('start_time_ms')) or 0,
            end_time_ms=_int_or_none(row.get('end_time_ms')) or 0,
            total_segments_in_session=_int_or_none(row.get('total_segments_in_session')) or 1,
            speaker=str(row.get('speaker', '')),
            text=str(row.get('text', '')),
            word_count=_int_or_none(row.get('word_count')) or 0,
            primary_stage=_int_or_none(row.get('primary_stage')),
            secondary_stage=_int_or_none(row.get('secondary_stage')),
            llm_confidence_primary=_float_or_none(row.get('llm_confidence_primary')),
            llm_confidence_secondary=_float_or_none(row.get('llm_confidence_secondary')),
            llm_justification=row.get('llm_justification'),
            llm_run_consistency=_int_or_none(row.get('llm_run_consistency')),
            rater_ids=rater_ids,
            rater_votes=rater_votes,
            agreement_level=str(row.get('agreement_level') or ''),
            agreement_fraction=_float_or_none(row.get('agreement_fraction')),
            needs_review=bool(row.get('needs_review', False)),
            consensus_vote=row.get('consensus_vote'),
            tie_broken_by_confidence=bool(row.get('tie_broken_by_confidence', False)),
            codebook_labels_embedding=_list_or_none(row.get('codebook_labels_embedding')),
            codebook_labels_llm=_list_or_none(row.get('codebook_labels_llm')),
            codebook_labels_ensemble=_list_or_none(row.get('codebook_labels_ensemble')),
            codebook_disagreements=_list_or_none(row.get('codebook_disagreements')),
            codebook_confidence=None,
            human_label=_int_or_none(row.get('human_label')),
            human_secondary_label=_int_or_none(row.get('human_secondary_label')),
            adjudicated_label=_int_or_none(row.get('adjudicated_label')),
            in_human_coded_subset=bool(row.get('in_human_coded_subset', False)),
            label_status=str(row.get('label_status', 'llm_only')),
            final_label=_int_or_none(row.get('final_label')),
            final_label_source=row.get('final_label_source'),
            label_confidence_tier=row.get('label_confidence_tier'),
            speakers_in_segment=None,
            session_file=str(row.get('session_file', '')),
        )
        segments.append(seg)

    return segments


def load_segments_for_stage(
    run_dir: str,
    *,
    apply: tuple = ('theme', 'purer', 'codebook', 'cv'),
) -> List[Segment]:
    """
    Load all frozen segments and apply the requested overlays.

    ``apply`` controls which overlays are applied; callers exclude the overlay
    they are about to overwrite so stale labels don't bleed in.

    Raises FileNotFoundError if there are no frozen segments yet.
    Missing overlays in the apply tuple are silently skipped.
    """
    sessions = list_segmented_sessions(run_dir)
    if not sessions:
        raise FileNotFoundError(
            f"No frozen segments found in {run_dir}; "
            "run `qra ingest` first."
        )

    segments: List[Segment] = []
    for sid in sessions:
        segments.extend(read_session_segments(run_dir, sid))

    if apply:
        from .classifications_io import apply_overlays
        by_id = {s.segment_id: s for s in segments}
        apply_overlays(run_dir, by_id, keys=apply)

    return segments


# ---------------------------------------------------------------------------
# Coercion helpers (used by _load_segments_from_jsonl)
# ---------------------------------------------------------------------------

def _int_or_none(val) -> Optional[int]:
    try:
        return int(val) if val is not None and str(val) != 'nan' else None
    except (ValueError, TypeError):
        return None


def _float_or_none(val) -> Optional[float]:
    try:
        return float(val) if val is not None and str(val) != 'nan' else None
    except (ValueError, TypeError):
        return None


def _list_or_none(val):
    if val is None:
        return None
    if isinstance(val, list):
        return val if val else None
    return None
