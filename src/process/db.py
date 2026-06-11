"""
process/db.py
-------------
Central SQLite schema + connection management for the per-project ``qra.db``.

A single ``qra.db`` file under the run (output) directory is the internal data
store for the pipeline:

  * frozen per-session segments           -> ``segments``
  * classification overlays (theme/purer/  -> ``theme_labels`` / ``purer_labels`` /
    codebook/cv/gnn)                          ``codebook_labels`` / ``cv_labels`` / ``gnn_labels``
  * classification provenance manifest     -> ``classification_manifest``
  * validation testset worksheets + items  -> ``testset_worksheets`` / ``testset_items``
  * content-validity testsets + items      -> ``cv_testsets`` / ``cv_testset_items``
  * inter-rater reliability human codes    -> ``irr_testsets`` / ``irr_human_codes``

All DDL lives here.  The per-table read/write logic lives in the modules that
own each artifact (``segments_io``, ``classifications_io``, ``assembly/human_forms``,
``assembly/content_validity``, ``legacy_migration``) — they open a connection via
``open_db(run_dir)`` and run their own SQL against the schema defined below.

Design notes
------------
* WAL journal mode is enabled so readers don't block the single writer.
* ``foreign_keys`` is ON; the testset/cv item tables reference their parents.
* ``row_factory`` is :class:`sqlite3.Row`, so callers can do ``dict(row)`` or
  ``row['col']``.
* ``open_db`` is a context manager that COMMITS on clean exit, ROLLS BACK and
  re-raises on exception, and ALWAYS closes the connection.  A single ``with``
  block is therefore one atomic transaction — this replaces the old
  ``_freeze.write_frozen`` tmp-file+rename atomicity for SQLite-backed writes.

The classification_manifest stores each entry as a JSON blob (``entry_json``)
rather than flattening to columns: manifest entries carry arbitrary nested
dicts (``framework``/``codebook``) and optional keys, and the only consumers
read the whole entry back as a dict — a blob is a faithful, lossless round-trip.
"""

import contextlib
import json
import os
import sqlite3
from typing import Any, Iterator, Optional

# Bump on any forward-incompatible schema change and add the migration in
# ensure_schema() (forward-only ALTER TABLE / data migration keyed on the
# stored value of _schema_meta['schema_version']).
#
# v2 (2026-06): run-centric classification — ``classification_runs`` registry +
# durable ``label_ballots`` (the source of truth for re-votable consensus).
# ``_migrate_1_to_2`` backfills both from the existing ``rater_votes`` caches.
SCHEMA_VERSION = 2


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def db_path(run_dir: str) -> str:
    """Absolute path to the project SQLite database: ``<run_dir>/qra.db``."""
    return os.path.join(run_dir, 'qra.db')


# ---------------------------------------------------------------------------
# Schema (DDL)
# ---------------------------------------------------------------------------

# segment_id is the natural primary key joining the frozen segments table to
# every per-segment overlay table.
_SCHEMA_STATEMENTS = (
    # -- migration / version tracking -------------------------------------
    """
    CREATE TABLE IF NOT EXISTS _schema_meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,

    # -- frozen segments (raw segmentation only; classification fields live
    #    in the overlay tables).  Last three columns were segmentation_meta.json.
    """
    CREATE TABLE IF NOT EXISTS segments (
        segment_id                TEXT    PRIMARY KEY,
        trial_id                  TEXT    NOT NULL DEFAULT '',
        participant_id            TEXT    NOT NULL DEFAULT '',
        session_id                TEXT    NOT NULL DEFAULT '',
        session_number            INTEGER NOT NULL DEFAULT 0,
        cohort_id                 INTEGER,
        session_variant           TEXT    NOT NULL DEFAULT '',
        segment_index             INTEGER NOT NULL DEFAULT 0,
        start_time_ms             INTEGER NOT NULL DEFAULT 0,
        end_time_ms               INTEGER NOT NULL DEFAULT 0,
        total_segments_in_session INTEGER NOT NULL DEFAULT 0,
        speaker                   TEXT    NOT NULL DEFAULT '',
        text                      TEXT    NOT NULL DEFAULT '',
        word_count                INTEGER NOT NULL DEFAULT 0,
        speakers_in_segment       TEXT,            -- JSON array | NULL
        session_file              TEXT    NOT NULL DEFAULT '',
        params_hash               TEXT    NOT NULL DEFAULT '',
        segmenter_version         TEXT    NOT NULL DEFAULT '1',
        ingest_timestamp          TEXT    NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_seg_session     ON segments (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_seg_speaker     ON segments (speaker)",
    "CREATE INDEX IF NOT EXISTS idx_seg_participant ON segments (participant_id)",

    # -- theme (VAAMR) overlay -------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS theme_labels (
        segment_id                   TEXT PRIMARY KEY,
        primary_stage                INTEGER,
        secondary_stage              INTEGER,
        llm_confidence_primary       REAL,
        llm_confidence_secondary     REAL,
        llm_justification            TEXT,
        rater_ids                    TEXT,    -- JSON array | NULL
        rater_votes                  TEXT,    -- JSON array | NULL
        agreement_level              TEXT,
        agreement_fraction           REAL,
        needs_review                 INTEGER NOT NULL DEFAULT 0,
        consensus_vote               TEXT,    -- JSON-encoded: int | "ABSTAIN" | null
        tie_broken_by_confidence     INTEGER NOT NULL DEFAULT 0,
        llm_run_consistency          INTEGER,
        secondary_agreement_level    TEXT,
        secondary_agreement_fraction REAL
    )
    """,

    # -- purer overlay ----------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS purer_labels (
        segment_id                 TEXT PRIMARY KEY,
        purer_primary              INTEGER,
        purer_secondary            INTEGER,
        purer_confidence_primary   REAL,
        purer_confidence_secondary REAL,
        purer_justification        TEXT,
        purer_run_consistency      INTEGER,
        purer_agreement_level      TEXT,
        purer_agreement_fraction   REAL,
        purer_needs_review         INTEGER NOT NULL DEFAULT 0,
        purer_rater_ids            TEXT,    -- JSON array | NULL
        purer_rater_votes          TEXT     -- JSON array | NULL
    )
    """,

    # -- codebook overlay -------------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS codebook_labels (
        segment_id                 TEXT PRIMARY KEY,
        codebook_labels_embedding  TEXT,    -- JSON array of code_ids | NULL
        codebook_labels_llm        TEXT,    -- JSON array | NULL
        codebook_labels_ensemble   TEXT,    -- JSON array | NULL
        codebook_disagreements     TEXT,    -- JSON array | NULL
        codebook_confidence        TEXT     -- JSON dict {code_id: float} | NULL
    )
    """,

    # -- cross-validation overlay ----------------------------------------
    #    (these fields are not Segment attributes today; the cv overlay
    #     currently round-trips all-NULL rows.  Table preserved for parity.)
    """
    CREATE TABLE IF NOT EXISTS cv_labels (
        segment_id               TEXT PRIMARY KEY,
        cv_adjudicated_primary   INTEGER,
        cv_adjudicated_secondary INTEGER,
        cv_disagreement_score    REAL,
        cv_adjudication_method   TEXT
    )
    """,

    # -- gnn consensus overlay -------------------------------------------
    """
    CREATE TABLE IF NOT EXISTS gnn_labels (
        segment_id        TEXT PRIMARY KEY,
        gnn_vaamr_pred    INTEGER,
        gnn_vaamr_conf    REAL,
        gnn_vaamr_abstain INTEGER,   -- 3-state: NULL | 0 | 1 (preserves None vs False)
        gnn_purer_pred    INTEGER,
        gnn_purer_conf    REAL,
        gnn_purer_abstain INTEGER,
        gnn_label_source  TEXT
    )
    """,

    # -- probe scaler overlay (LLM-free per-rater ensemble; methodology §8.6) --
    #    Fills UNLABELED participant segments only; ranks BELOW the LLM
    #    (provenance tier 'probe_consensus'), never overrides it.
    """
    CREATE TABLE IF NOT EXISTS probe_labels (
        segment_id    TEXT PRIMARY KEY,
        probe_pred    INTEGER,
        probe_conf    REAL,
        probe_abstain INTEGER,   -- 3-state: NULL | 0 | 1 (preserves None vs False)
        probe_label_source TEXT
    )
    """,

    # -- classification provenance manifest ------------------------------
    #    One row per classifier key; the whole entry dict is a JSON blob so
    #    nested framework/codebook dicts and optional keys round-trip exactly.
    """
    CREATE TABLE IF NOT EXISTS classification_manifest (
        key        TEXT PRIMARY KEY,
        entry_json TEXT NOT NULL
    )
    """,

    # -- validation testset worksheets (flat, numbered) ------------------
    #    The human-readable .txt worksheet and AI answer-key stay on disk;
    #    this carries the per-set metadata that was in testset_meta/*.meta.json.
    """
    CREATE TABLE IF NOT EXISTS testset_worksheets (
        worksheet_n   INTEGER PRIMARY KEY,   -- 1-based (matches the .txt filename)
        kind          TEXT    NOT NULL DEFAULT 'vaamr',   -- 'vaamr'|'purer'|'codebook'
        name          TEXT,
        created_at    TEXT    NOT NULL DEFAULT '',
        n_items       INTEGER NOT NULL DEFAULT 0,
        params_hash   TEXT,
        frozen        INTEGER NOT NULL DEFAULT 1,
        legacy_import INTEGER NOT NULL DEFAULT 0
    )
    """,

    # -- validation testset items (replaces testset_meta segments[]) ------
    #    item_num preserves worksheet ordering; (session_id, seg_num) + sha256
    #    are exactly the fields the old .meta.json carried per item.
    """
    CREATE TABLE IF NOT EXISTS testset_items (
        worksheet_n INTEGER NOT NULL,
        item_num    INTEGER NOT NULL,           -- 1-based, preserves order
        session_id  TEXT    NOT NULL DEFAULT '',
        seg_num     INTEGER NOT NULL DEFAULT 0, -- 1-based (= segment_index + 1)
        sha256      TEXT,
        PRIMARY KEY (worksheet_n, item_num),
        FOREIGN KEY (worksheet_n) REFERENCES testset_worksheets (worksheet_n) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_ts_items_ws ON testset_items (worksheet_n)",

    # -- content-validity testsets ---------------------------------------
    """
    CREATE TABLE IF NOT EXISTS cv_testsets (
        name              TEXT PRIMARY KEY,
        kind              TEXT NOT NULL DEFAULT 'vaamr',   -- 'vaamr'|'purer'
        framework_name    TEXT NOT NULL DEFAULT '',
        framework_version TEXT NOT NULL DEFAULT '1',
        created_at        TEXT NOT NULL DEFAULT ''
    )
    """,

    # -- content-validity items (replaces content_validity/<name>/items.jsonl)
    """
    CREATE TABLE IF NOT EXISTS cv_testset_items (
        testset_name   TEXT    NOT NULL,
        item_id        TEXT    NOT NULL,           -- the items.jsonl "id"
        ord            INTEGER NOT NULL DEFAULT 0, -- preserves manifest item_ids order
        text           TEXT    NOT NULL DEFAULT '',
        expected_stage INTEGER,
        difficulty     TEXT,
        source_field   TEXT,
        content_sha256 TEXT,
        PRIMARY KEY (testset_name, item_id),
        FOREIGN KEY (testset_name) REFERENCES cv_testsets (name) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_cv_items_ts ON cv_testset_items (testset_name)",

    # -- inter-rater reliability: imported test-sets --------------------------
    #    One row per imported human-coded worksheet.  ``raters`` is the JSON
    #    roster (ordered) that coded this worksheet; drives Human↔Human IRR.
    """
    CREATE TABLE IF NOT EXISTS irr_testsets (
        worksheet_n INTEGER PRIMARY KEY,
        name        TEXT,
        raters      TEXT,    -- JSON array of rater ids | NULL
        n_items     INTEGER NOT NULL DEFAULT 0,
        created_at  TEXT    NOT NULL DEFAULT ''
    )
    """,

    # -- inter-rater reliability: per-rater human codes ----------------------
    #    Long format: one row per (worksheet, item, rater) for individual rater
    #    ballots, plus one row per (worksheet, item) consensus with
    #    is_consensus=1 and rater='__consensus__'.  ``primary``/``secondary`` are
    #    VAAMR theme_ids (INTEGER), the ABSTAIN sentinel (-1 = "No code"), or NULL
    #    (rater did not code that item / no secondary).  ``segment_id`` is the
    #    resolved frozen-segment id (may be NULL if resolution failed).
    """
    CREATE TABLE IF NOT EXISTS irr_human_codes (
        worksheet_n  INTEGER NOT NULL,
        item_num     INTEGER NOT NULL,
        segment_id   TEXT,
        rater        TEXT    NOT NULL,
        prim         INTEGER,
        secondary    INTEGER,
        is_consensus INTEGER NOT NULL DEFAULT 0,
        source       TEXT,
        notes        TEXT,
        PRIMARY KEY (worksheet_n, item_num, rater),
        FOREIGN KEY (worksheet_n) REFERENCES irr_testsets (worksheet_n) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_irr_codes_ws ON irr_human_codes (worksheet_n)",
    "CREATE INDEX IF NOT EXISTS idx_irr_codes_seg ON irr_human_codes (segment_id)",

    # -- classification runs registry (schema v2) ----------------------------
    #    One row per (model, quantization, thinking, temperature, note) sweep
    #    over one framework's units.  ``rater_label`` is the unique display id
    #    used as the rater id in ballots, kappa tables, and transcripts.
    #    ``selected`` gates which runs feed the derived overlay consensus.
    """
    CREATE TABLE IF NOT EXISTS classification_runs (
        run_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
        overlay                  TEXT    NOT NULL,            -- 'theme'|'purer'|'codebook'
        rater_label              TEXT    NOT NULL,            -- unique display id; rater id downstream
        model                    TEXT    NOT NULL,
        backend                  TEXT,
        quantization             TEXT,
        thinking                 TEXT,                        -- 'on'|'off'|NULL
        note                     TEXT,
        temperature              REAL,
        params_json              TEXT,                        -- JSON dict | NULL
        segmentation_params_hash TEXT,                        -- staleness guard
        status                   TEXT    NOT NULL DEFAULT 'queued',
                                                              -- queued|running|completed|
                                                              --   completed_with_errors|failed|archived
        selected                 INTEGER NOT NULL DEFAULT 0,
        checkpoint_path          TEXT,
        created_at               TEXT    NOT NULL DEFAULT '',
        started_at               TEXT,
        completed_at             TEXT,
        n_total                  INTEGER,
        n_coded                  INTEGER,
        n_abstain                INTEGER,
        n_error                  INTEGER,
        UNIQUE (overlay, rater_label)
    )
    """,

    # -- per-(overlay, segment, run) ballots (schema v2) ---------------------
    #    The durable source of truth: ``raw_json`` is the exact parsed ballot so
    #    consensus can be re-voted byte-identically from any selected subset of
    #    runs.  ``vote`` is 'CODED' | 'ABSTAIN' | 'ERROR' (ERROR rows carry NULL
    #    stage/confidence and NULL raw_json).  ``applies_to_json`` records the
    #    cue-unit -> constituent propagation for PURER.
    """
    CREATE TABLE IF NOT EXISTS label_ballots (
        overlay              TEXT    NOT NULL,
        segment_id           TEXT    NOT NULL,
        run_id               INTEGER NOT NULL,
        vote                 TEXT    NOT NULL,                -- 'CODED'|'ABSTAIN'|'ERROR'
        stage                INTEGER,
        confidence           REAL,
        secondary_stage      INTEGER,
        secondary_confidence REAL,
        justification        TEXT,
        applies_to_json      TEXT,                            -- cue-unit -> constituents (PURER) | NULL
        raw_json             TEXT,                            -- exact parsed ballot | NULL (ERROR)
        updated_at           TEXT    NOT NULL DEFAULT '',
        PRIMARY KEY (overlay, segment_id, run_id),
        FOREIGN KEY (run_id) REFERENCES classification_runs (run_id) ON DELETE CASCADE
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_runs_overlay_status ON classification_runs (overlay, status)",
    "CREATE INDEX IF NOT EXISTS idx_ballots_run         ON label_ballots (run_id)",
    "CREATE INDEX IF NOT EXISTS idx_ballots_overlay_seg ON label_ballots (overlay, segment_id)",
)


class SchemaVersionError(RuntimeError):
    """Raised when the on-disk schema is newer than this code understands."""


# Forward migrations, keyed on the FROM version: ``_MIGRATIONS[n]`` is a
# ``callable(conn)`` that upgrades a v``n`` database to v``n+1``.  Each step must
# be forward-only and idempotent (safe to re-apply).  To evolve the schema:
#   1. add the new column/table to ``_SCHEMA_STATEMENTS`` (use IF NOT EXISTS),
#   2. write ``def _migrate_1_to_2(conn): conn.execute("ALTER TABLE ...")``,
#   3. register it (``_MIGRATIONS = {1: _migrate_1_to_2}``) and bump
#      ``SCHEMA_VERSION`` above.
# Backfill spec: (overlay key, overlay table, rater-ids column, rater-votes column).
# ``_migrate_1_to_2`` reconstructs the run registry + ballots from these caches.
_V2_BACKFILL = (
    ('theme', 'theme_labels', 'rater_ids', 'rater_votes'),
    ('purer', 'purer_labels', 'purer_rater_ids', 'purer_rater_votes'),
)


def _now_iso() -> str:
    """Current UTC time as an ISO-8601 string (timezone-aware)."""
    import datetime
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _migrate_1_to_2(conn: sqlite3.Connection) -> None:
    """Backfill the v2 run registry + ballots from the legacy rater_votes caches.

    The new ``classification_runs`` / ``label_ballots`` tables already exist by
    the time this runs (``ensure_schema`` executes ``_SCHEMA_STATEMENTS`` before
    any migration), so this is pure data backfill — no DDL.

    For each overlay it:
      * derives the rater roster in first-seen order (``rater_ids`` ordering
        across rows, then any stragglers seen only in ``rater_votes``),
      * creates one **born-selected, completed** run per rater
        (``model``/``rater_label`` = the rater string), so the very first
        consensus rebuild reproduces today's overlays,
      * INSERTs (OR IGNORE) one ballot per segment's ``rater_votes`` entry, and
      * refreshes the per-run CODED/ABSTAIN/ERROR/total counters.

    Idempotent: re-running after a partial failure is a no-op for already
    backfilled runs/ballots (get-or-create runs + INSERT OR IGNORE ballots).
    """
    now = _now_iso()
    seg_hash = _read_segments_params_hash(conn)

    for overlay, table, ids_col, votes_col in _V2_BACKFILL:
        rows = conn.execute(
            f"SELECT segment_id, {ids_col} AS ids, {votes_col} AS votes "
            f"FROM {table} WHERE {votes_col} IS NOT NULL"
        ).fetchall()

        # --- rater roster in first-seen order --------------------------------
        roster: list = []
        seen = set()
        # Pass 1: honour the stored rater_ids ordering (the slot order).
        for r in rows:
            for rid in (loads(r['ids']) or []):
                rid = str(rid)
                if rid not in seen:
                    seen.add(rid)
                    roster.append(rid)
        # Pass 2: union any rater seen only in the votes' 'rater' keys.
        for r in rows:
            for entry in (loads(r['votes']) or []):
                rid = entry.get('rater') if isinstance(entry, dict) else None
                if rid is None:
                    continue
                rid = str(rid)
                if rid not in seen:
                    seen.add(rid)
                    roster.append(rid)

        if not roster:
            continue

        # --- one run per rater (get-or-create; born selected + completed) -----
        run_ids: dict = {}
        for rid in roster:
            conn.execute(
                "INSERT OR IGNORE INTO classification_runs "
                "(overlay, rater_label, model, status, selected, note, "
                " segmentation_params_hash, created_at, completed_at) "
                "VALUES (?, ?, ?, 'completed', 1, 'backfilled from rater_votes', ?, ?, ?)",
                (overlay, rid, rid, seg_hash, now, now),
            )
            row = conn.execute(
                "SELECT run_id FROM classification_runs "
                "WHERE overlay = ? AND rater_label = ?",
                (overlay, rid),
            ).fetchone()
            run_ids[rid] = row['run_id']

        # --- ballots from each segment's rater_votes entries -----------------
        for r in rows:
            seg_id = r['segment_id']
            for entry in (loads(r['votes']) or []):
                if not isinstance(entry, dict):
                    continue
                rid = entry.get('rater')
                if rid is None:
                    continue
                run_id = run_ids.get(str(rid))
                if run_id is None:
                    continue
                vote = entry.get('vote') or 'ERROR'
                is_error = (vote == 'ERROR')
                conn.execute(
                    "INSERT OR IGNORE INTO label_ballots "
                    "(overlay, segment_id, run_id, vote, stage, confidence, "
                    " secondary_stage, secondary_confidence, justification, "
                    " raw_json, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        overlay, seg_id, run_id, vote,
                        entry.get('stage'),
                        entry.get('confidence'),
                        entry.get('secondary_stage'),
                        entry.get('secondary_confidence'),
                        entry.get('justification'),
                        None if is_error else dumps(entry),
                        now,
                    ),
                )

        # --- per-run counters -------------------------------------------------
        for run_id in run_ids.values():
            _refresh_run_counters(conn, run_id)


def _read_segments_params_hash(conn: sqlite3.Connection) -> Optional[str]:
    """Single-value segmentation params_hash from any frozen segment, or None."""
    try:
        row = conn.execute("SELECT params_hash FROM segments LIMIT 1").fetchone()
    except sqlite3.Error:
        return None
    if row is None or row['params_hash'] in (None, ''):
        return None
    return row['params_hash']


def _refresh_run_counters(conn: sqlite3.Connection, run_id: int) -> None:
    """Recompute n_coded/n_abstain/n_error/n_total for one run from its ballots."""
    row = conn.execute(
        "SELECT "
        "  SUM(vote = 'CODED')   AS coded, "
        "  SUM(vote = 'ABSTAIN') AS abstain, "
        "  SUM(vote = 'ERROR')   AS error, "
        "  COUNT(*)              AS total "
        "FROM label_ballots WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    conn.execute(
        "UPDATE classification_runs "
        "SET n_coded = ?, n_abstain = ?, n_error = ?, n_total = ? "
        "WHERE run_id = ?",
        (
            int(row['coded'] or 0),
            int(row['abstain'] or 0),
            int(row['error'] or 0),
            int(row['total'] or 0),
            run_id,
        ),
    )


_MIGRATIONS: dict = {1: _migrate_1_to_2}


def ensure_schema(conn: sqlite3.Connection) -> None:
    """
    Create every table/index if absent, then bring the schema up to
    ``SCHEMA_VERSION`` by running any registered forward migrations.

    * Fresh DB  -> create tables, stamp ``schema_version = SCHEMA_VERSION``.
    * Older DB  -> run ``_MIGRATIONS[v]`` for each v in ``[stored, SCHEMA_VERSION)``
      in order, then stamp the new version.
    * Newer DB  -> raise :class:`SchemaVersionError` (never silently downgrade).
    """
    for stmt in _SCHEMA_STATEMENTS:
        conn.execute(stmt)

    stored = get_meta(conn, 'schema_version')
    if stored is None:
        set_meta(conn, 'schema_version', SCHEMA_VERSION)
        conn.commit()
        return

    stored_v = int(stored)
    if stored_v == SCHEMA_VERSION:
        conn.commit()
        return
    if stored_v > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"qra.db schema_version={stored_v} is newer than this build of QRA "
            f"(SCHEMA_VERSION={SCHEMA_VERSION}); upgrade QRA to open this project."
        )

    # stored_v < SCHEMA_VERSION: apply forward migrations in ascending order.
    for v in range(stored_v, SCHEMA_VERSION):
        migrate = _MIGRATIONS.get(v)
        if migrate is None:
            raise SchemaVersionError(
                f"No migration registered for qra.db schema v{v} -> v{v + 1}."
            )
        migrate(conn)
    set_meta(conn, 'schema_version', SCHEMA_VERSION)
    conn.commit()


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def connect(db_file: str) -> sqlite3.Connection:
    """
    Open (creating if absent) a SQLite connection at ``db_file`` with the
    project's standard pragmas and row factory.  Does NOT call ensure_schema().

    Callers that want the schema guaranteed should use :func:`open_db`, or call
    :func:`ensure_schema` themselves (e.g. the migration writing to a temp DB).
    """
    parent = os.path.dirname(db_file)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_file)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextlib.contextmanager
def open_db(run_dir: str) -> Iterator[sqlite3.Connection]:
    """
    Open the project DB for ``run_dir`` (creating + initialising the schema if
    absent) as a single atomic transaction.

    Usage::

        with db.open_db(run_dir) as conn:
            conn.execute("INSERT ...")
            rows = conn.execute("SELECT ...").fetchall()

    Commits on clean exit, rolls back + re-raises on exception, always closes.
    """
    conn = connect(db_path(run_dir))
    try:
        ensure_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def db_exists(run_dir: str) -> bool:
    """True if the project's qra.db file is present on disk."""
    return os.path.isfile(db_path(run_dir))


# ---------------------------------------------------------------------------
# _schema_meta helpers
# ---------------------------------------------------------------------------

def get_meta(conn: sqlite3.Connection, key: str) -> Optional[str]:
    """Return a ``_schema_meta`` value, or None if the key is absent."""
    row = conn.execute(
        "SELECT value FROM _schema_meta WHERE key = ?", (key,)
    ).fetchone()
    return None if row is None else row['value']


def set_meta(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert a ``_schema_meta`` key/value pair."""
    conn.execute(
        "INSERT INTO _schema_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, str(value)),
    )


# ---------------------------------------------------------------------------
# JSON column helpers
# ---------------------------------------------------------------------------

def dumps(value: Any) -> Optional[str]:
    """
    Serialise a value destined for a JSON TEXT column.

    None round-trips to NULL (returns None).  Non-JSON-native iterables are
    coerced to lists; anything else falls back to ``str``.
    """
    if value is None:
        return None
    return json.dumps(value, default=_json_default)


def loads(text: Optional[str]) -> Any:
    """Inverse of :func:`dumps`.  NULL/empty -> None; otherwise json.loads."""
    if text is None or text == '':
        return None
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None


def _json_default(obj: Any) -> Any:
    if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
        return list(obj)
    return str(obj)
