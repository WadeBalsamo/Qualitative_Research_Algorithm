"""
process/segments_io.py
----------------------
Frozen per-session segmentation I/O and master-segments reader.
"""
import datetime
import glob
import hashlib
import json
import os
from typing import Dict, List, Optional

from classification_tools.data_structures import Segment
from . import output_paths as _paths
from ._freeze import write_frozen

# Raw-segmentation fields persisted by write_session_segments.
# Classification-only fields are intentionally excluded.
_RAW_FIELDS = (
    'segment_id', 'trial_id', 'participant_id', 'session_id',
    'session_number', 'cohort_id', 'session_variant',
    'segment_index', 'start_time_ms', 'end_time_ms',
    'total_segments_in_session', 'speaker', 'text', 'word_count',
    'speakers_in_segment', 'session_file',
)

# Segmentation parameters whose values affect output (used for params_hash).
_HASH_FIELDS = (
    'embedding_model', 'silence_threshold_ms', 'semantic_shift_percentile',
    'min_segment_words_conversational', 'max_segment_words_conversational',
    'max_gap_seconds', 'min_words_per_sentence', 'max_segment_duration_seconds',
    'use_adaptive_threshold', 'min_prominence', 'broad_window_size',
    'use_topic_clustering', 'use_llm_refinement', 'llm_refinement_mode',
    'llm_ambiguity_threshold', 'llm_batch_size',
)


def resolve_session_id(session_file: str) -> str:
    """Extract the session_id from a transcript file path.

    VTT files are keyed by their basename (stem); JSON diarization files
    are keyed by the parent directory name (which holds result.json).
    """
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


def write_session_segments(
    run_dir: str,
    session_id: str,
    segments: List[Segment],
    current_params_hash: str,
    *,
    force: bool = False,
) -> str:
    """
    Write frozen per-session segments to
      01_transcripts/segmented/<session_id>/segments.jsonl
    and the corresponding segmentation_meta.json.

    Only raw-segmentation fields are persisted; classification fields are dropped.
    Returns the path to segments.jsonl.
    """
    segs_path = _paths.session_segments_path(run_dir, session_id)
    meta_path = _paths.segmentation_meta_path(run_dir, session_id)

    def _write_segs(fh):
        for seg in segments:
            rec = {f: getattr(seg, f, None) for f in _RAW_FIELDS}
            fh.write(json.dumps(rec, default=str) + '\n')

    def _write_meta(fh):
        meta = {
            'params_hash': current_params_hash,
            'segmenter_version': '1',
            'ingest_timestamp': datetime.datetime.utcnow().isoformat() + 'Z',
            'session_id': session_id,
            'n_segments': len(segments),
        }
        json.dump(meta, fh, indent=2)

    write_frozen(segs_path, _write_segs, force=force)
    write_frozen(meta_path, _write_meta, force=force)
    return segs_path


def overwrite_segment_texts(run_dir: str, session_id: str, segments: List[Segment]) -> str:
    """
    Rewrite segments.jsonl in-place using an atomic rename, WITHOUT touching
    segmentation_meta.json (preserves params_hash and ingest_timestamp).

    Used by `qra apply-anonymization` for retroactive PHI scrubbing of already-frozen
    segments. The caller is responsible for any confirmation prompt.

    Returns the path to the updated segments.jsonl.
    """
    segs_path = _paths.session_segments_path(run_dir, session_id)
    if not os.path.isfile(segs_path):
        raise FileNotFoundError(
            f"No frozen segments found for session {session_id!r} in {run_dir}"
        )
    tmp = segs_path + '.tmp'
    try:
        with open(tmp, 'w', encoding='utf-8') as fh:
            for seg in segments:
                rec = {f: getattr(seg, f, None) for f in _RAW_FIELDS}
                fh.write(json.dumps(rec, default=str) + '\n')
        os.replace(tmp, segs_path)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return segs_path


def read_session_segments(run_dir: str, session_id: str) -> List[Segment]:
    """
    Reconstruct Segment objects from the frozen segments.jsonl.

    Only raw-segmentation fields are restored; all classification fields
    remain at their dataclass defaults.
    """
    segs_path = _paths.session_segments_path(run_dir, session_id)
    segments: List[Segment] = []
    with open(segs_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seg = Segment(
                segment_id=str(d.get('segment_id', '')),
                trial_id=str(d.get('trial_id', '')),
                participant_id=str(d.get('participant_id', '')),
                session_id=str(d.get('session_id', '')),
                session_number=int(d.get('session_number') or 0),
                cohort_id=_int_or_none(d.get('cohort_id')),
                session_variant=str(d.get('session_variant', '') or ''),
                segment_index=int(d.get('segment_index') or 0),
                start_time_ms=int(d.get('start_time_ms') or 0),
                end_time_ms=int(d.get('end_time_ms') or 0),
                total_segments_in_session=int(d.get('total_segments_in_session') or 0),
                speaker=str(d.get('speaker', '')),
                text=str(d.get('text', '')),
                word_count=int(d.get('word_count') or 0),
                speakers_in_segment=d.get('speakers_in_segment'),
                session_file=str(d.get('session_file', '')),
            )
            segments.append(seg)
    return segments


def list_segmented_sessions(run_dir: str) -> List[str]:
    """Return session IDs that have frozen segments on disk."""
    base = _paths.segmented_sessions_dir(run_dir)
    if not os.path.isdir(base):
        return []
    return [
        name for name in sorted(os.listdir(base))
        if os.path.isfile(os.path.join(base, name, 'segments.jsonl'))
    ]


def is_segmentation_fresh(run_dir: str, session_id: str, current_params_hash: str) -> bool:
    """
    True iff frozen segments exist for session_id AND the stored params_hash
    matches current_params_hash, OR the stored hash is the legacy sentinel
    'legacy-pre-modular' (so legacy-migrated sessions don't force re-segmentation).
    """
    segs_path = _paths.session_segments_path(run_dir, session_id)
    meta_path = _paths.segmentation_meta_path(run_dir, session_id)
    if not os.path.isfile(segs_path):
        return False
    if not os.path.isfile(meta_path):
        return False
    try:
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return False
    stored_hash = meta.get('params_hash', '')
    return stored_hash == 'legacy-pre-modular' or stored_hash == current_params_hash


def read_master_segments(run_dir: str) -> List[Segment]:
    """
    Reconstruct all Segment objects from the latest master_segments*.jsonl.

    This is the canonical post-hoc reader used by qra.py analyze/testsets
    commands. Moved here from qra.py:_load_segments_from_jsonl so Phase 2
    has a single canonical reader to extend.
    """
    jsonl_files = sorted(glob.glob(
        os.path.join(_paths.master_segments_dir(run_dir), 'master_segments*.jsonl')
    ))
    if not jsonl_files:
        jsonl_files = sorted(glob.glob(
            os.path.join(run_dir, 'master_segments_*.jsonl')
        ))
    if not jsonl_files:
        raise FileNotFoundError(
            f"No master_segments*.jsonl found in {run_dir}. "
            "Run the pipeline first: python qra.py run --config <config>"
        )
    return _load_segments_from_jsonl(jsonl_files[-1])


def _load_segments_from_jsonl(jsonl_path: str) -> List[Segment]:
    """Reconstruct Segment objects from a single master_segments JSONL file."""
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
    Load all frozen segments from disk and apply the requested overlays.

    ``apply`` controls which overlay files are applied; callers exclude the
    overlay they are about to overwrite so stale labels don't bleed in.

    Raises FileNotFoundError if 01_transcripts/segmented/ is empty.
    Missing overlay files in the apply tuple are silently skipped.
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
