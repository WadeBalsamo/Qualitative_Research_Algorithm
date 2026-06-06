"""
process/classifications_io.py
------------------------------
Overlay read/write for per-classifier classification results (Phase 3).

Each classifier writes its own JSONL overlay file in
  02_meta/classifications/<key>_labels.jsonl

Overlays are intentionally overwritable: re-running a classifier replaces
only its overlay file.  Frozen per-session segments (01_transcripts/segmented/)
are never touched by classification.

Each overlay record contains ``segment_id`` plus the classifier-specific
fields listed in the OVERLAY_FIELDS constants below.

A provenance manifest at 02_meta/classifications/classification_manifest.json
records model, framework version, n_segments, and completion timestamp per key.
"""

import datetime
import json
import os
from typing import Dict, List, Optional, Tuple

from classification_tools.data_structures import Segment
from . import output_paths as _paths
from ._freeze import write_frozen

# ---------------------------------------------------------------------------
# Overlay field constants — must match Segment dataclass field names exactly
# ---------------------------------------------------------------------------

THEME_OVERLAY_FIELDS: Tuple[str, ...] = (
    'primary_stage', 'secondary_stage',
    'llm_confidence_primary', 'llm_confidence_secondary', 'llm_justification',
    'rater_ids', 'rater_votes',
    'agreement_level', 'agreement_fraction',
    'needs_review', 'consensus_vote', 'tie_broken_by_confidence',
    'llm_run_consistency',
    'secondary_agreement_level', 'secondary_agreement_fraction',
)

PURER_OVERLAY_FIELDS: Tuple[str, ...] = (
    'purer_primary', 'purer_secondary',
    'purer_confidence_primary', 'purer_confidence_secondary',
    'purer_justification', 'purer_run_consistency',
    'purer_agreement_level', 'purer_agreement_fraction',
    'purer_needs_review',
    'purer_rater_ids', 'purer_rater_votes',
)

CODEBOOK_OVERLAY_FIELDS: Tuple[str, ...] = (
    'codebook_labels_embedding', 'codebook_labels_llm', 'codebook_labels_ensemble',
    'codebook_disagreements', 'codebook_confidence',
)

CROSS_VALIDATION_OVERLAY_FIELDS: Tuple[str, ...] = (
    'cv_adjudicated_primary', 'cv_adjudicated_secondary',
    'cv_disagreement_score', 'cv_adjudication_method',
)

# GNN consensus-distillation overlay: per-segment graph predictions. Written by the
# GNN layer when produce_consensus_labels=True; become the label of record only when
# gnn_layer.gnn_authoritative=True (see process/assembly/master_dataset.py).
GNN_OVERLAY_FIELDS: Tuple[str, ...] = (
    'gnn_vaamr_pred', 'gnn_vaamr_conf', 'gnn_vaamr_abstain',
    'gnn_purer_pred', 'gnn_purer_conf', 'gnn_purer_abstain',
    'gnn_label_source',
)

OVERLAY_KEYS = ('theme', 'purer', 'codebook', 'cv', 'gnn')

OVERLAY_FILENAMES = {
    'theme': 'theme_labels.jsonl',
    'purer': 'purer_labels.jsonl',
    'codebook': 'codebook_labels.jsonl',
    'cv': 'cross_validation_labels.jsonl',
    'gnn': 'gnn_labels.jsonl',
}

_OVERLAY_FIELDS_MAP = {
    'theme': THEME_OVERLAY_FIELDS,
    'purer': PURER_OVERLAY_FIELDS,
    'codebook': CODEBOOK_OVERLAY_FIELDS,
    'cv': CROSS_VALIDATION_OVERLAY_FIELDS,
    'gnn': GNN_OVERLAY_FIELDS,
}


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def overlay_path(run_dir: str, key: str) -> str:
    """Absolute path to the overlay JSONL for the given classifier key."""
    return _paths.classification_overlay_path(run_dir, key)


def manifest_path(run_dir: str) -> str:
    """Absolute path to the classification provenance manifest."""
    return _paths.classification_manifest_path(run_dir)


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_overlay(run_dir: str, key: str, segments: List[Segment]) -> str:
    """
    Write (or overwrite) the overlay JSONL for ``key``.

    Records contain segment_id plus all fields listed in _OVERLAY_FIELDS_MAP[key].
    Records are sorted by segment_id for stable diffs across runs.
    Returns the path written.
    """
    fields = _OVERLAY_FIELDS_MAP[key]
    path = overlay_path(run_dir, key)
    sorted_segs = sorted(segments, key=lambda s: s.segment_id)

    def _write(fh):
        for seg in sorted_segs:
            rec = {'segment_id': seg.segment_id,
                   **{f: getattr(seg, f, None) for f in fields}}
            fh.write(json.dumps(rec, default=_json_default) + '\n')

    write_frozen(path, _write, force=True)
    return path


def write_theme_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (overwrite) the theme classifier overlay. Returns path."""
    return _write_overlay(run_dir, 'theme', segments)


def write_purer_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (overwrite) the PURER classifier overlay. Returns path."""
    return _write_overlay(run_dir, 'purer', segments)


def write_codebook_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (overwrite) the codebook classifier overlay. Returns path."""
    return _write_overlay(run_dir, 'codebook', segments)


def write_cross_validation_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (overwrite) the cross-validation overlay. Returns path."""
    return _write_overlay(run_dir, 'cv', segments)


def write_gnn_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (overwrite) the GNN consensus overlay. Returns path."""
    return _write_overlay(run_dir, 'gnn', segments)


def merge_overlay(run_dir: str, key: str, segments: List[Segment]) -> str:
    """Merge segments into an existing overlay, replacing rows with matching segment_id.

    Reads the current overlay (if any) into a dict keyed by segment_id, upserts
    rows built from ``segments``, then writes the merged result back atomically.
    Rows are sorted by segment_id for stable diffs.  Returns the path written.
    """
    fields = _OVERLAY_FIELDS_MAP[key]
    path = overlay_path(run_dir, key)

    # Load existing rows keyed by segment_id.
    merged: dict = {}
    try:
        with open(path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get('segment_id')
                if sid is not None:
                    merged[sid] = rec
    except FileNotFoundError:
        pass

    # Build new rows from incoming segments and upsert.
    for seg in segments:
        row = {'segment_id': seg.segment_id,
               **{f: getattr(seg, f, None) for f in fields}}
        merged[row['segment_id']] = row

    # Stable sort by segment_id then atomic write.
    sorted_rows = [merged[k] for k in sorted(merged)]

    def _write(fh):
        for row in sorted_rows:
            fh.write(json.dumps(row, default=_json_default) + '\n')

    write_frozen(path, _write, force=True)
    return path


def merge_theme_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Merge segments into the theme overlay (upsert by segment_id). Returns path."""
    return merge_overlay(run_dir, 'theme', segments)


def merge_purer_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Merge segments into the PURER overlay (upsert by segment_id). Returns path."""
    return merge_overlay(run_dir, 'purer', segments)


def merge_codebook_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Merge segments into the codebook overlay (upsert by segment_id). Returns path."""
    return merge_overlay(run_dir, 'codebook', segments)


def merge_cross_validation_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Merge segments into the cross-validation overlay (upsert by segment_id). Returns path."""
    return merge_overlay(run_dir, 'cv', segments)


def merge_gnn_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Merge segments into the GNN consensus overlay (upsert by segment_id). Returns path."""
    return merge_overlay(run_dir, 'gnn', segments)


# ---------------------------------------------------------------------------
# Readers — apply overlays back onto Segment objects in memory
# ---------------------------------------------------------------------------

def _apply_overlay(run_dir: str, key: str, segments_by_id: Dict[str, Segment]) -> int:
    """
    Read the overlay JSONL for ``key`` and set the listed fields on matching
    Segment objects in ``segments_by_id``.

    Returns the number of segments updated.
    No-op (returns 0) if the overlay file is absent.
    """
    path = overlay_path(run_dir, key)
    fields = _OVERLAY_FIELDS_MAP[key]
    count = 0
    try:
        fh = open(path, encoding='utf-8')
    except FileNotFoundError:
        return 0
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get('segment_id')
            seg = segments_by_id.get(sid)
            if seg is None:
                continue
            for f in fields:
                if f in rec:
                    setattr(seg, f, rec[f])
            count += 1
    return count


def apply_theme_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply theme overlay to in-memory segments. Returns update count."""
    return _apply_overlay(run_dir, 'theme', segments_by_id)


def apply_purer_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply PURER overlay to in-memory segments. Returns update count."""
    return _apply_overlay(run_dir, 'purer', segments_by_id)


def apply_codebook_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply codebook overlay to in-memory segments. Returns update count."""
    return _apply_overlay(run_dir, 'codebook', segments_by_id)


def apply_cross_validation_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply cross-validation overlay to in-memory segments. Returns update count.

    CV results are aggregate statistics; there are no per-segment fields to apply.
    """
    return 0


def apply_gnn_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply GNN consensus overlay to in-memory segments. Returns update count."""
    return _apply_overlay(run_dir, 'gnn', segments_by_id)


_APPLY_FUNCS = {
    'theme': apply_theme_overlay,
    'purer': apply_purer_overlay,
    'codebook': apply_codebook_overlay,
    'cv': apply_cross_validation_overlay,
    'gnn': apply_gnn_overlay,
}


def apply_overlays(
    run_dir: str,
    segments_by_id: Dict[str, Segment],
    keys: Tuple[str, ...] = OVERLAY_KEYS,
) -> Dict[str, int]:
    """
    Apply multiple overlays to in-memory segments.

    Returns a dict mapping key → count of segments updated.
    Missing overlay files are silently skipped.
    """
    return {k: _APPLY_FUNCS[k](run_dir, segments_by_id) for k in keys}


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def update_classification_manifest(
    run_dir: str,
    *,
    key: str,
    entry: dict,
) -> None:
    """
    Read the manifest (or create empty), set manifest[key] = entry (augmented
    with completed_at and segmentation_params_hash), then write back atomically.

    Does not clobber sibling keys.
    """
    mpath = manifest_path(run_dir)
    os.makedirs(os.path.dirname(mpath), exist_ok=True)

    manifest: dict = {}
    try:
        with open(mpath, encoding='utf-8') as fh:
            manifest = json.load(fh)
    except (OSError, json.JSONDecodeError):
        manifest = {}

    full_entry = dict(entry)
    full_entry['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()

    if 'segmentation_params_hash' not in full_entry:
        full_entry['segmentation_params_hash'] = _read_any_params_hash(run_dir)

    manifest[key] = full_entry

    write_frozen(mpath, lambda fh: json.dump(manifest, fh, indent=2), force=True)


def read_classification_manifest(run_dir: str) -> Optional[dict]:
    """Return the manifest dict, or None if it does not exist."""
    mpath = manifest_path(run_dir)
    if not os.path.isfile(mpath):
        return None
    try:
        with open(mpath, encoding='utf-8') as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_any_params_hash(run_dir: str) -> Optional[str]:
    """Read segmentation_params_hash from the first available session meta."""
    seg_dir = _paths.segmented_sessions_dir(run_dir)
    if not os.path.isdir(seg_dir):
        return None
    for name in os.listdir(seg_dir):
        meta_path = _paths.segmentation_meta_path(run_dir, name)
        try:
            with open(meta_path, encoding='utf-8') as fh:
                return json.load(fh).get('params_hash')
        except (OSError, json.JSONDecodeError):
            pass
    return None


def _json_default(obj):
    """Fallback JSON serialiser for non-standard types."""
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    return str(obj)
