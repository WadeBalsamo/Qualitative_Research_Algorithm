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
SCHEMA_VERSION = 1


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
_MIGRATIONS: dict = {}


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
