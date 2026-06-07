"""
process/classifications_io.py
------------------------------
Overlay read/write for per-classifier classification results (Phase 3).

Each classifier's results live in its own table in the project SQLite store
(``process.db``):

  theme -> theme_labels   purer -> purer_labels   codebook -> codebook_labels
  cv    -> cv_labels       gnn   -> gnn_labels

Overlays are intentionally overwritable: ``write_*_overlay`` replaces a
classifier's whole table (full re-run), ``merge_*_overlay`` upserts by
``segment_id`` (incremental).  Frozen per-session segments are never touched.

A provenance manifest table (``classification_manifest``) records model,
framework version, n_segments, and completion timestamp per key as a JSON blob.

``overlay_path`` / ``manifest_path`` and the ``OVERLAY_*`` / ``*_OVERLAY_FIELDS``
constants are retained for back-compat and as the canonical field-name lists;
the path helpers return their historical .jsonl/.json strings even though those
files no longer exist (the data lives in SQLite).  Use ``read_overlay`` to read
an overlay's records.
"""

import datetime
from typing import Dict, List, Optional, Tuple

from classification_tools.data_structures import Segment
from . import db
from . import output_paths as _paths

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

# Probe scaler overlay: per-segment LLM-free predictions (per-rater ensemble,
# methodology §8.6). Written by classification_tools.probe_classifier.classify_with_probe;
# fills UNLABELED participant segments only and becomes the provenance tier
# 'probe_consensus' ranked BELOW the LLM (see process/assembly/master_dataset.py).
PROBE_OVERLAY_FIELDS: Tuple[str, ...] = (
    'probe_pred', 'probe_conf', 'probe_abstain', 'probe_label_source',
)

OVERLAY_KEYS = ('theme', 'purer', 'codebook', 'cv', 'gnn', 'probe')

# Legacy on-disk filenames (data now lives in SQLite tables — retained so
# overlay_path() and constant-checking tests keep their historical contract).
OVERLAY_FILENAMES = {
    'theme': 'theme_labels.jsonl',
    'purer': 'purer_labels.jsonl',
    'codebook': 'codebook_labels.jsonl',
    'cv': 'cross_validation_labels.jsonl',
    'gnn': 'gnn_labels.jsonl',
    'probe': 'probe_labels.jsonl',
}

# Classifier key -> SQLite table.
_OVERLAY_TABLES = {
    'theme': 'theme_labels',
    'purer': 'purer_labels',
    'codebook': 'codebook_labels',
    'cv': 'cv_labels',
    'gnn': 'gnn_labels',
    'probe': 'probe_labels',
}

_OVERLAY_FIELDS_MAP = {
    'theme': THEME_OVERLAY_FIELDS,
    'purer': PURER_OVERLAY_FIELDS,
    'codebook': CODEBOOK_OVERLAY_FIELDS,
    'cv': CROSS_VALIDATION_OVERLAY_FIELDS,
    'gnn': GNN_OVERLAY_FIELDS,
    'probe': PROBE_OVERLAY_FIELDS,
}

# Fields stored as JSON TEXT (List/Dict, or the heterogeneous consensus_vote
# which is int | 'ABSTAIN' | None): encode on write, decode on read.
_OVERLAY_JSON_FIELDS = {
    'theme': frozenset({'rater_ids', 'rater_votes', 'consensus_vote'}),
    'purer': frozenset({'purer_rater_ids', 'purer_rater_votes'}),
    'codebook': frozenset({
        'codebook_labels_embedding', 'codebook_labels_llm',
        'codebook_labels_ensemble', 'codebook_disagreements', 'codebook_confidence',
    }),
    'cv': frozenset(),
    'gnn': frozenset(),
    'probe': frozenset(),
}

# Fields stored as INTEGER 0/1 standing for a bool: coerce back to bool on read
# (None preserved for the Optional[bool] gnn/probe abstain flags).
_OVERLAY_BOOL_FIELDS = {
    'theme': frozenset({'needs_review', 'tie_broken_by_confidence'}),
    'purer': frozenset({'purer_needs_review'}),
    'codebook': frozenset(),
    'cv': frozenset(),
    'gnn': frozenset({'gnn_vaamr_abstain', 'gnn_purer_abstain'}),
    'probe': frozenset({'probe_abstain'}),
}


# ---------------------------------------------------------------------------
# Path helpers (legacy/back-compat — return historical strings; no live file)
# ---------------------------------------------------------------------------

def overlay_path(run_dir: str, key: str) -> str:
    """Historical overlay JSONL path for ``key``.

    Deprecated: overlay data now lives in the SQLite ``{key}_labels`` table.
    Retained so callers/tests that build the legacy path string keep working.
    """
    return _paths.classification_overlay_path(run_dir, key)


def manifest_path(run_dir: str) -> str:
    """Historical classification-manifest JSON path.

    Deprecated: the manifest now lives in the ``classification_manifest`` table.
    """
    return _paths.classification_manifest_path(run_dir)


# ---------------------------------------------------------------------------
# Row <-> record/segment mapping
# ---------------------------------------------------------------------------

def _seg_to_values(key: str, seg: Segment) -> tuple:
    """Build the INSERT tuple (segment_id + overlay fields) for ``seg``."""
    json_fields = _OVERLAY_JSON_FIELDS[key]
    vals = [seg.segment_id]
    for f in _OVERLAY_FIELDS_MAP[key]:
        v = getattr(seg, f, None)
        if f in json_fields:
            v = db.dumps(v)
        vals.append(v)
    return tuple(vals)


def _row_to_record(key: str, row) -> dict:
    """Reconstruct an overlay record dict (segment_id + fields) from a DB row,
    decoding JSON columns and coercing bool columns."""
    json_fields = _OVERLAY_JSON_FIELDS[key]
    bool_fields = _OVERLAY_BOOL_FIELDS[key]
    rec = {'segment_id': row['segment_id']}
    for f in _OVERLAY_FIELDS_MAP[key]:
        v = row[f]
        if f in json_fields:
            v = db.loads(v)
        elif f in bool_fields:
            v = None if v is None else bool(v)
        rec[f] = v
    return rec


def _insert_sql(key: str) -> str:
    table = _OVERLAY_TABLES[key]
    cols = ('segment_id',) + _OVERLAY_FIELDS_MAP[key]
    return (
        f"INSERT OR REPLACE INTO {table} ({', '.join(cols)}) "
        f"VALUES ({', '.join('?' for _ in cols)})"
    )


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def _write_overlay(run_dir: str, key: str, segments: List[Segment]) -> str:
    """Replace the overlay table for ``key`` with rows built from ``segments``."""
    table = _OVERLAY_TABLES[key]
    with db.open_db(run_dir) as conn:
        conn.execute(f"DELETE FROM {table}")
        rows = [_seg_to_values(key, s) for s in segments]
        if rows:
            conn.executemany(_insert_sql(key), rows)
    return overlay_path(run_dir, key)


def write_theme_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (replace) the theme classifier overlay. Returns legacy path string."""
    return _write_overlay(run_dir, 'theme', segments)


def write_purer_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (replace) the PURER classifier overlay. Returns legacy path string."""
    return _write_overlay(run_dir, 'purer', segments)


def write_codebook_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (replace) the codebook classifier overlay. Returns legacy path string."""
    return _write_overlay(run_dir, 'codebook', segments)


def write_cross_validation_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (replace) the cross-validation overlay. Returns legacy path string."""
    return _write_overlay(run_dir, 'cv', segments)


def write_gnn_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (replace) the GNN consensus overlay. Returns legacy path string."""
    return _write_overlay(run_dir, 'gnn', segments)


def write_probe_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Write (replace) the probe scaler overlay. Returns legacy path string."""
    return _write_overlay(run_dir, 'probe', segments)


def merge_overlay(run_dir: str, key: str, segments: List[Segment]) -> str:
    """Upsert ``segments`` into the overlay table, replacing rows by segment_id.

    Rows for segment_ids not in ``segments`` are left untouched.  Returns the
    legacy path string.
    """
    with db.open_db(run_dir) as conn:
        rows = [_seg_to_values(key, s) for s in segments]
        if rows:
            conn.executemany(_insert_sql(key), rows)
    return overlay_path(run_dir, key)


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


def merge_probe_overlay(run_dir: str, segments: List[Segment]) -> str:
    """Merge segments into the probe scaler overlay (upsert by segment_id). Returns path."""
    return merge_overlay(run_dir, 'probe', segments)


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------

def read_overlay(run_dir: str, key: str) -> List[dict]:
    """Return all overlay records for ``key`` as dicts, sorted by segment_id.

    Each record is ``{'segment_id': ..., <field>: <value>, ...}`` with JSON
    columns decoded and bool columns coerced — the same shape the overlay JSONL
    rows had.  Returns ``[]`` if the store or table is empty.
    """
    if not db.db_exists(run_dir):
        return []
    table = _OVERLAY_TABLES[key]
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            f"SELECT * FROM {table} ORDER BY segment_id"
        ).fetchall()
    return [_row_to_record(key, r) for r in rows]


def clear_overlay(run_dir: str, key: str) -> None:
    """Delete every row from the overlay table for ``key`` (no-op if the DB is absent).

    Used by ``process.reclassify_ops`` to reset a classifier's overlay before a
    from-scratch re-run.
    """
    if not db.db_exists(run_dir):
        return
    table = _OVERLAY_TABLES[key]
    with db.open_db(run_dir) as conn:
        conn.execute(f"DELETE FROM {table}")


def overlay_exists(run_dir: str, key: str) -> bool:
    """True if the overlay table for ``key`` has at least one row.

    Replaces ``os.path.isfile(overlay_path(...))`` checks now that overlays live
    in SQLite rather than per-key JSONL files.
    """
    if not db.db_exists(run_dir):
        return False
    table = _OVERLAY_TABLES[key]
    with db.open_db(run_dir) as conn:
        return conn.execute(f"SELECT 1 FROM {table} LIMIT 1").fetchone() is not None


def remap_overlay_segment_ids(run_dir: str, key: str, segid_map: Dict[str, str]) -> int:
    """Rewrite overlay rows whose segment_id is in ``segid_map`` to the new id.

    Collision-safe across rename cycles: affected rows are captured, deleted,
    then re-inserted under their new segment_id (stored column values copied
    verbatim — no re-encoding).  Returns the number of rows remapped.  Used by
    the anonymization-key cascade.
    """
    if not segid_map or not db.db_exists(run_dir):
        return 0
    table = _OVERLAY_TABLES[key]
    cols = _OVERLAY_FIELDS_MAP[key]
    insert_sql = _insert_sql(key)
    with db.open_db(run_dir) as conn:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
        affected = [r for r in rows if r['segment_id'] in segid_map]
        if not affected:
            return 0
        for r in affected:
            conn.execute(f"DELETE FROM {table} WHERE segment_id = ?", (r['segment_id'],))
        for r in affected:
            vals = [segid_map[r['segment_id']]] + [r[c] for c in cols]
            conn.execute(insert_sql, tuple(vals))
    return len(affected)


def _apply_overlay(run_dir: str, key: str, segments_by_id: Dict[str, Segment]) -> int:
    """
    Set the overlay fields for ``key`` on matching Segment objects in
    ``segments_by_id``.  Returns the number of segments updated; 0 if the store
    or table is empty.
    """
    if not db.db_exists(run_dir):
        return 0
    table = _OVERLAY_TABLES[key]
    fields = _OVERLAY_FIELDS_MAP[key]
    count = 0
    with db.open_db(run_dir) as conn:
        rows = conn.execute(f"SELECT * FROM {table}").fetchall()
    for row in rows:
        seg = segments_by_id.get(row['segment_id'])
        if seg is None:
            continue
        rec = _row_to_record(key, row)
        for f in fields:
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

    CV results are aggregate statistics; there are no per-segment Segment fields
    to apply, so this is a no-op (returns 0), matching the historical behavior.
    """
    return 0


def apply_gnn_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply GNN consensus overlay to in-memory segments. Returns update count."""
    return _apply_overlay(run_dir, 'gnn', segments_by_id)


def apply_probe_overlay(run_dir: str, segments_by_id: Dict[str, Segment]) -> int:
    """Apply probe scaler overlay to in-memory segments. Returns update count."""
    return _apply_overlay(run_dir, 'probe', segments_by_id)


_APPLY_FUNCS = {
    'theme': apply_theme_overlay,
    'purer': apply_purer_overlay,
    'codebook': apply_codebook_overlay,
    'cv': apply_cross_validation_overlay,
    'gnn': apply_gnn_overlay,
    'probe': apply_probe_overlay,
}


def apply_overlays(
    run_dir: str,
    segments_by_id: Dict[str, Segment],
    keys: Tuple[str, ...] = OVERLAY_KEYS,
) -> Dict[str, int]:
    """
    Apply multiple overlays to in-memory segments.

    Returns a dict mapping key → count of segments updated.
    Missing overlays are silently skipped.
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
    Upsert ``manifest[key] = entry`` (augmented with completed_at and
    segmentation_params_hash) into the ``classification_manifest`` table.

    Does not clobber sibling keys (each key is its own row).
    """
    full_entry = dict(entry)
    full_entry['completed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    if 'segmentation_params_hash' not in full_entry:
        full_entry['segmentation_params_hash'] = _read_any_params_hash(run_dir)

    with db.open_db(run_dir) as conn:
        conn.execute(
            "INSERT INTO classification_manifest (key, entry_json) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET entry_json = excluded.entry_json",
            (key, db.dumps(full_entry)),
        )


def read_classification_manifest(run_dir: str) -> Optional[dict]:
    """Return the manifest as a ``{key: entry}`` dict, or None if there are none."""
    if not db.db_exists(run_dir):
        return None
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT key, entry_json FROM classification_manifest"
        ).fetchall()
    if not rows:
        return None
    return {r['key']: db.loads(r['entry_json']) for r in rows}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_any_params_hash(run_dir: str) -> Optional[str]:
    """Read a segmentation params_hash from any frozen segment row."""
    if not db.db_exists(run_dir):
        return None
    with db.open_db(run_dir) as conn:
        row = conn.execute(
            "SELECT params_hash FROM segments LIMIT 1"
        ).fetchone()
    return None if row is None else row['params_hash']
