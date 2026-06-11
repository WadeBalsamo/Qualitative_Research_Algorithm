"""
process/run_registry.py
------------------------
Read/write helpers for the schema-v2 run registry and durable ballots.

A *run* (``classification_runs`` row) is one ``(model, quantization, thinking,
temperature, note)`` sweep over one framework's units (``overlay`` ∈
``'theme' | 'purer' | 'codebook'``).  Its ``rater_label`` is the unique display
id used as the rater id everywhere downstream (κ tables, transcripts, ballots).

*Ballots* (``label_ballots`` rows) are the durable source of truth: one row per
``(overlay, segment_id, run_id)`` carrying the exact parsed ballot in
``raw_json`` so consensus can be re-voted byte-identically from any selected
subset of runs.  A ``vote`` of ``'ERROR'`` records a parse failure (NULL stage /
confidence / raw_json).

This module owns only the SQL against those two tables; the DDL lives in
``process.db`` (which also defines the v1->v2 backfill).  Every public function
opens its own :func:`db.open_db` transaction, mirroring ``classifications_io``.
"""

import datetime
import json
from typing import Any, Dict, List, Optional

from . import db


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Columns a caller may set via ``update_run`` (run_id / overlay are immutable;
# rater_label is immutable to keep κ-history continuity).
_UPDATABLE_FIELDS = frozenset({
    'model', 'backend', 'quantization', 'thinking', 'note', 'temperature',
    'params_json', 'segmentation_params_hash', 'status', 'selected',
    'checkpoint_path', 'started_at', 'completed_at',
    'n_total', 'n_coded', 'n_abstain', 'n_error',
})

# Columns returned by get_run / list_runs (full row, JSON decoded for params).
_RUN_COLUMNS = (
    'run_id', 'overlay', 'rater_label', 'model', 'backend', 'quantization',
    'thinking', 'note', 'temperature', 'params_json', 'segmentation_params_hash',
    'status', 'selected', 'checkpoint_path', 'created_at', 'started_at',
    'completed_at', 'n_total', 'n_coded', 'n_abstain', 'n_error',
)


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (timezone-aware).

    Single-homed in ``db._now_iso``; kept as a module symbol here because
    ``run_executor`` calls ``_rr._now_iso()`` (matching how other modules reuse
    db helpers).
    """
    return db._now_iso()


# ---------------------------------------------------------------------------
# rater_label composition
# ---------------------------------------------------------------------------

def compose_rater_label(
    model: str,
    quantization: Optional[str] = None,
    thinking: Optional[str] = None,
    alias: Optional[str] = None,
    existing: Optional[set] = None,
) -> str:
    """Compose a unique ``rater_label`` for a run.

    ``alias`` wins outright when given.  Otherwise the label is ``model`` with an
    optional ``[quant,think:X]`` suffix (each part omitted when None), e.g.::

        compose_rater_label('qwen-3-70b')                       -> 'qwen-3-70b'
        compose_rater_label('qwen-3-70b', quantization='Q4')    -> 'qwen-3-70b[Q4]'
        compose_rater_label('m', thinking='off')                -> 'm[think:off]'
        compose_rater_label('m', quantization='Q4', thinking='off')
                                                                -> 'm[Q4,think:off]'

    If the composed label collides with an entry in ``existing`` it is
    de-duplicated with a ``#2``, ``#3`` … suffix (the UNIQUE(overlay,
    rater_label) constraint is the hard backstop).
    """
    if alias:
        base = str(alias)
    else:
        parts = []
        if quantization:
            parts.append(str(quantization))
        if thinking:
            parts.append(f'think:{thinking}')
        base = str(model)
        if parts:
            base = f"{base}[{','.join(parts)}]"

    existing = existing or set()
    if base not in existing:
        return base
    n = 2
    while f"{base}#{n}" in existing:
        n += 1
    return f"{base}#{n}"


# ---------------------------------------------------------------------------
# Run CRUD
# ---------------------------------------------------------------------------

def create_run(
    run_dir: str,
    *,
    overlay: str,
    model: str,
    backend: Optional[str] = None,
    quantization: Optional[str] = None,
    thinking: Optional[str] = None,
    note: Optional[str] = None,
    temperature: Optional[float] = None,
    params: Optional[dict] = None,
    rater_label: Optional[str] = None,
) -> int:
    """Create a queued run and return its ``run_id``.

    ``rater_label`` is auto-composed (and de-duplicated against existing labels
    for this overlay) when None.  ``segmentation_params_hash`` is stamped from
    the current frozen segments when available (the staleness guard).
    """
    with db.open_db(run_dir) as conn:
        if rater_label is None:
            existing = {
                r['rater_label'] for r in conn.execute(
                    "SELECT rater_label FROM classification_runs WHERE overlay = ?",
                    (overlay,),
                ).fetchall()
            }
            rater_label = compose_rater_label(
                model, quantization=quantization, thinking=thinking,
                existing=existing,
            )
        seg_hash = db._read_segments_params_hash(conn)
        cur = conn.execute(
            "INSERT INTO classification_runs "
            "(overlay, rater_label, model, backend, quantization, thinking, "
            " note, temperature, params_json, segmentation_params_hash, "
            " status, selected, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', 0, ?)",
            (
                overlay, rater_label, model, backend, quantization, thinking,
                note, temperature, db.dumps(params), seg_hash, _now_iso(),
            ),
        )
        return int(cur.lastrowid)


def _row_to_run(row) -> dict:
    """Reconstruct a run dict from a DB row (decoding params_json)."""
    rec = {c: row[c] for c in _RUN_COLUMNS}
    rec['params'] = db.loads(rec.get('params_json'))
    rec['selected'] = bool(rec['selected'])
    return rec


def get_run(run_dir: str, run_id: int) -> Optional[dict]:
    """Return one run as a dict (with ``params`` decoded), or None if absent."""
    if not db.db_exists(run_dir):
        return None
    with db.open_db(run_dir) as conn:
        row = conn.execute(
            "SELECT * FROM classification_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
    return None if row is None else _row_to_run(row)


def list_runs(
    run_dir: str,
    overlay: Optional[str] = None,
    statuses: Optional[List[str]] = None,
) -> List[dict]:
    """Return runs (optionally filtered by ``overlay`` / ``statuses``).

    Ordered by ``run_id``.  Returns ``[]`` if the store/table is absent.
    """
    if not db.db_exists(run_dir):
        return []
    where = []
    params: List[Any] = []
    if overlay is not None:
        where.append("overlay = ?")
        params.append(overlay)
    if statuses:
        where.append(f"status IN ({', '.join('?' for _ in statuses)})")
        params.extend(statuses)
    sql = "SELECT * FROM classification_runs"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY run_id"
    try:
        with db.open_db(run_dir) as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
    except Exception as exc:  # noqa: BLE001
        # The DB exists (db_exists passed) but the query failed (e.g. a missing
        # classification_runs table on a partially-migrated store) — surface it
        # instead of masquerading as "no runs".
        print(f"  [runs] warning: could not read classification_runs: {exc}")
        return []
    return [_row_to_run(r) for r in rows]


def update_run(run_dir: str, run_id: int, **fields) -> None:
    """Update whitelisted columns on a run (unknown fields raise).

    ``params`` is accepted as a convenience alias for ``params_json`` (it is
    JSON-encoded).  No-op when no recognised fields are given.
    """
    sets: Dict[str, Any] = {}
    for k, v in fields.items():
        if k == 'params':
            sets['params_json'] = db.dumps(v)
            continue
        if k not in _UPDATABLE_FIELDS:
            raise ValueError(f"run_registry.update_run: field {k!r} is not updatable")
        if k == 'selected':
            v = 1 if v else 0
        sets[k] = v
    if not sets:
        return
    assignments = ', '.join(f"{k} = ?" for k in sets)
    with db.open_db(run_dir) as conn:
        conn.execute(
            f"UPDATE classification_runs SET {assignments} WHERE run_id = ?",
            tuple(sets.values()) + (run_id,),
        )


def set_selected(run_dir: str, overlay: str, run_ids: List[int]) -> None:
    """Mark exactly ``run_ids`` selected within ``overlay`` (others cleared)."""
    wanted = [int(r) for r in run_ids]
    with db.open_db(run_dir) as conn:
        conn.execute(
            "UPDATE classification_runs SET selected = 0 WHERE overlay = ?",
            (overlay,),
        )
        if wanted:
            placeholders = ', '.join('?' for _ in wanted)
            conn.execute(
                f"UPDATE classification_runs SET selected = 1 "
                f"WHERE overlay = ? AND run_id IN ({placeholders})",
                (overlay, *wanted),
            )


def selected_runs(run_dir: str, overlay: str) -> List[int]:
    """Return the selected run_ids for ``overlay``, ordered by run_id."""
    if not db.db_exists(run_dir):
        return []
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            "SELECT run_id FROM classification_runs "
            "WHERE overlay = ? AND selected = 1 ORDER BY run_id",
            (overlay,),
        ).fetchall()
    return [r['run_id'] for r in rows]


# ---------------------------------------------------------------------------
# Ballots
# ---------------------------------------------------------------------------

def _decompose_cell(cell: Optional[dict]) -> dict:
    """Map a parsed ballot dict to ``label_ballots`` column values.

    ``cell is None`` -> an ERROR row (NULL stage/confidence, raw_json NULL).
    Otherwise the parsed-ballot keys are decomposed: ``vote`` (inferred from
    ``primary_stage`` when absent, exactly as ``majority_vote`` builds
    rater_votes), ``primary_stage`` -> stage, ``primary_confidence`` ->
    confidence, plus secondary_*/justification.  ``raw_json`` is the verbatim
    cell so re-votes are byte-identical.
    """
    if cell is None:
        return {
            'vote': 'ERROR', 'stage': None, 'confidence': None,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': None, 'raw_json': None,
        }
    vote = cell.get('vote')
    if vote is None:
        # Legacy / parsed-run dict without an explicit vote — infer like
        # majority_vote.vote_single_label does from primary_stage.
        vote = 'ABSTAIN' if cell.get('primary_stage') is None else 'CODED'
    if vote == 'ERROR':
        # An ERROR ballot is a hard parse failure: NULL stage/confidence and
        # NULL raw_json (so ballots_for_runs yields None for the slot), matching
        # the v1->v2 backfill exactly regardless of how the error was encoded.
        return {
            'vote': 'ERROR', 'stage': None, 'confidence': None,
            'secondary_stage': None, 'secondary_confidence': None,
            'justification': None, 'raw_json': None,
        }
    # Accept either the parsed-run shape (primary_stage/primary_confidence) or
    # the rater_votes cache shape (stage/confidence).
    stage = cell.get('primary_stage', cell.get('stage'))
    confidence = cell.get('primary_confidence', cell.get('confidence'))
    return {
        'vote': vote,
        'stage': stage,
        'confidence': confidence,
        'secondary_stage': cell.get('secondary_stage'),
        'secondary_confidence': cell.get('secondary_confidence'),
        'justification': cell.get('justification'),
        'raw_json': json.dumps(cell),
    }


def upsert_ballots(
    run_dir: str,
    overlay: str,
    run_id: int,
    cells: Dict[str, Optional[dict]],
    applies_to: Optional[Dict[str, list]] = None,
) -> int:
    """Insert-or-replace one ballot per ``cells`` entry for ``run_id``.

    ``cells`` maps ``segment_id`` -> parsed ballot dict (or None for a parse
    failure -> an ERROR row).  ``applies_to`` optionally maps ``segment_id`` ->
    the constituent ids this ballot propagates to (PURER cue units).  Replaces
    by the ``(overlay, segment_id, run_id)`` primary key; returns the number of
    ballots written.
    """
    applies_to = applies_to or {}
    now = _now_iso()
    rows = []
    for seg_id, cell in cells.items():
        d = _decompose_cell(cell)
        rows.append((
            overlay, seg_id, run_id, d['vote'], d['stage'], d['confidence'],
            d['secondary_stage'], d['secondary_confidence'], d['justification'],
            db.dumps(applies_to.get(seg_id)), d['raw_json'], now,
        ))
    if not rows:
        return 0
    with db.open_db(run_dir) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO label_ballots "
            "(overlay, segment_id, run_id, vote, stage, confidence, "
            " secondary_stage, secondary_confidence, justification, "
            " applies_to_json, raw_json, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def _normalize_ballot_cell(cell: Optional[dict]) -> Optional[dict]:
    """Coerce a decoded ``raw_json`` cell to the canonical *parsed-run* shape.

    The re-vote path (``llm_classifier.build_merge_result`` ->
    ``majority_vote.vote_single_label``) reads ``primary_stage`` /
    ``primary_confidence``.  Most producers store that shape verbatim, but the
    v1->v2 backfill (and any legacy ``rater_votes``-shaped cell) carry
    ``stage`` / ``confidence`` instead — re-voting those as-is would null a CODED
    ballot's primary.  This maps the legacy keys onto the parsed-run keys so
    every ballot source re-votes identically; cells already carrying
    ``primary_stage`` (and ``None``) pass straight through unchanged, so the
    stored byte-fidelity round-trip is preserved.
    """
    if cell is None or not isinstance(cell, dict):
        return cell
    if 'primary_stage' in cell or 'stage' not in cell:
        return cell  # already parsed-run shape (or nothing to map)
    vote = cell.get('vote')
    if vote == 'ERROR':
        return None
    out = dict(cell)
    out['primary_stage'] = cell.get('stage')
    out['primary_confidence'] = cell.get('confidence')
    out.setdefault('vote', 'ABSTAIN' if cell.get('stage') is None else 'CODED')
    return out


def ballots_for_runs(
    run_dir: str,
    overlay: str,
    run_ids: List[int],
) -> Dict[str, Dict[int, Optional[dict]]]:
    """Return ``{segment_id: {run_id: parsed_ballot | None}}`` for ``run_ids``.

    The parsed ballot is ``json.loads(raw_json)`` normalised to the parsed-run
    shape (see :func:`_normalize_ballot_cell`); an ERROR row whose ``raw_json``
    is NULL maps to ``None`` (a hard parse failure), so the result is directly
    slot-alignable for re-voting.  Returns ``{}`` when ``run_ids`` is empty or
    the store is absent.
    """
    out: Dict[str, Dict[int, Optional[dict]]] = {}
    if not run_ids or not db.db_exists(run_dir):
        return out
    wanted = [int(r) for r in run_ids]
    placeholders = ', '.join('?' for _ in wanted)
    with db.open_db(run_dir) as conn:
        rows = conn.execute(
            f"SELECT segment_id, run_id, raw_json FROM label_ballots "
            f"WHERE overlay = ? AND run_id IN ({placeholders})",
            (overlay, *wanted),
        ).fetchall()
    for r in rows:
        cell = None if r['raw_json'] is None else db.loads(r['raw_json'])
        out.setdefault(r['segment_id'], {})[r['run_id']] = _normalize_ballot_cell(cell)
    return out


def refresh_counters(run_dir: str, run_id: int) -> None:
    """Recompute the CODED/ABSTAIN/ERROR/total counters for ``run_id``."""
    with db.open_db(run_dir) as conn:
        db._refresh_run_counters(conn, run_id)


_BALLOT_COLUMNS = (
    'overlay', 'segment_id', 'run_id', 'vote', 'stage', 'confidence',
    'secondary_stage', 'secondary_confidence', 'justification',
    'applies_to_json', 'raw_json', 'updated_at',
)


def remap_ballot_segment_ids(run_dir: str, segid_map: Dict[str, str]) -> int:
    """Rewrite ``label_ballots`` rows whose ``segment_id`` is in ``segid_map``.

    Mirrors ``classifications_io.remap_overlay_segment_ids``'s collision strategy:
    affected rows are captured, DELETEd, then re-inserted under their new
    ``segment_id`` (so an ``a->b, b->a`` swap is safe — every affected row is
    removed before any re-insert).  ``applies_to_json`` lists (PURER cue-unit ->
    constituent ids) are also rewritten through the map on EVERY ballot, since a
    cue unit's constituent ids may be remapped even when the unit's own
    segment_id is not.  Returns the number of rows written (segment_id-remapped
    rows + applies_to-only rewrites).  No-op (0) when the map is empty or the
    store is absent.  Used by the anonymization-key cascade.
    """
    if not segid_map or not db.db_exists(run_dir):
        return 0

    def _remap_applies(raw):
        ids = db.loads(raw)
        if not isinstance(ids, list) or not ids:
            return raw, False
        new_ids = [segid_map.get(i, i) for i in ids]
        if new_ids == ids:
            return raw, False
        return db.dumps(new_ids), True

    with db.open_db(run_dir) as conn:
        rows = conn.execute("SELECT * FROM label_ballots").fetchall()
        affected = []  # (old_pk_segment_id, new_row_values_tuple)
        for r in rows:
            seg = r['segment_id']
            seg_changed = seg in segid_map
            new_applies, applies_changed = _remap_applies(r['applies_to_json'])
            if not seg_changed and not applies_changed:
                continue
            new_vals = []
            for c in _BALLOT_COLUMNS:
                if c == 'segment_id':
                    new_vals.append(segid_map.get(seg, seg))
                elif c == 'applies_to_json':
                    new_vals.append(new_applies)
                else:
                    new_vals.append(r[c])
            affected.append((r['overlay'], seg, r['run_id'], tuple(new_vals)))

        if not affected:
            return 0
        # Two-phase: delete every affected row by its OLD pk, then re-insert with
        # the new values — collision-safe across id swaps.
        for overlay, old_seg, run_id, _vals in affected:
            conn.execute(
                "DELETE FROM label_ballots "
                "WHERE overlay = ? AND segment_id = ? AND run_id = ?",
                (overlay, old_seg, run_id),
            )
        placeholders = ', '.join('?' for _ in _BALLOT_COLUMNS)
        conn.executemany(
            f"INSERT OR REPLACE INTO label_ballots "
            f"({', '.join(_BALLOT_COLUMNS)}) VALUES ({placeholders})",
            [vals for _o, _s, _r, vals in affected],
        )
    return len(affected)
