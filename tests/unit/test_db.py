"""Tests for process/db.py — SQLite schema + connection management."""
import os
import sqlite3
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from process import db


_EXPECTED_TABLES = {
    '_schema_meta', 'segments',
    'theme_labels', 'purer_labels', 'codebook_labels', 'cv_labels', 'gnn_labels',
    'classification_manifest',
    'testset_worksheets', 'testset_items',
    'cv_testsets', 'cv_testset_items',
    'classification_runs', 'label_ballots',
}


def _table_names(conn):
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r['name'] for r in rows}


class TestDbPath(unittest.TestCase):
    def test_db_path_location(self):
        self.assertTrue(db.db_path('/out').endswith('qra.db'))
        self.assertEqual(db.db_path('/out'), os.path.join('/out', 'qra.db'))

    def test_db_exists_false_then_true(self):
        d = tempfile.mkdtemp()
        self.assertFalse(db.db_exists(d))
        with db.open_db(d):
            pass
        self.assertTrue(db.db_exists(d))


class TestSchema(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_schema_created_fresh(self):
        with db.open_db(self.tmpdir) as conn:
            self.assertTrue(_EXPECTED_TABLES.issubset(_table_names(conn)))

    def test_ensure_schema_idempotent(self):
        with db.open_db(self.tmpdir) as conn:
            db.ensure_schema(conn)
            db.ensure_schema(conn)  # must not raise
            self.assertTrue(_EXPECTED_TABLES.issubset(_table_names(conn)))

    def test_schema_version_recorded(self):
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(db.get_meta(conn, 'schema_version'), str(db.SCHEMA_VERSION))

    def test_wal_mode(self):
        with db.open_db(self.tmpdir) as conn:
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            self.assertEqual(str(mode).lower(), 'wal')

    def test_foreign_keys_on(self):
        with db.open_db(self.tmpdir) as conn:
            fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
            self.assertEqual(int(fk), 1)

    def test_indexes_created(self):
        with db.open_db(self.tmpdir) as conn:
            idx = {r['name'] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        for expected in ('idx_seg_session', 'idx_seg_speaker', 'idx_seg_participant',
                         'idx_ts_items_ws', 'idx_cv_items_ts',
                         'idx_runs_overlay_status', 'idx_ballots_run',
                         'idx_ballots_overlay_seg'):
            self.assertIn(expected, idx)


class TestConnectionSemantics(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_commit_on_clean_exit(self):
        with db.open_db(self.tmpdir) as conn:
            conn.execute("INSERT INTO _schema_meta (key, value) VALUES ('k', 'v')")
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(db.get_meta(conn, 'k'), 'v')

    def test_rollback_on_exception(self):
        try:
            with db.open_db(self.tmpdir) as conn:
                conn.execute("INSERT INTO _schema_meta (key, value) VALUES ('k2', 'v2')")
                raise RuntimeError('boom')
        except RuntimeError:
            pass
        with db.open_db(self.tmpdir) as conn:
            self.assertIsNone(db.get_meta(conn, 'k2'))

    def test_row_factory_is_row(self):
        with db.open_db(self.tmpdir) as conn:
            row = conn.execute("SELECT 1 AS one").fetchone()
            self.assertIsInstance(row, sqlite3.Row)
            self.assertEqual(row['one'], 1)

    def test_set_and_get_meta(self):
        with db.open_db(self.tmpdir) as conn:
            db.set_meta(conn, 'foo', 'bar')
            self.assertEqual(db.get_meta(conn, 'foo'), 'bar')
            db.set_meta(conn, 'foo', 'baz')  # upsert
            self.assertEqual(db.get_meta(conn, 'foo'), 'baz')
            self.assertIsNone(db.get_meta(conn, 'missing'))


class TestJsonHelpers(unittest.TestCase):
    def test_dumps_none_is_none(self):
        self.assertIsNone(db.dumps(None))

    def test_loads_none_and_empty(self):
        self.assertIsNone(db.loads(None))
        self.assertIsNone(db.loads(''))

    def test_roundtrip_list_and_dict(self):
        self.assertEqual(db.loads(db.dumps(['a', 'b'])), ['a', 'b'])
        self.assertEqual(db.loads(db.dumps({'x': 0.5})), {'x': 0.5})

    def test_roundtrip_scalar(self):
        self.assertEqual(db.loads(db.dumps(3)), 3)
        self.assertEqual(db.loads(db.dumps('ABSTAIN')), 'ABSTAIN')

    def test_loads_bad_text_returns_none(self):
        self.assertIsNone(db.loads('not json{'))


def _v1_statements():
    """The schema-v1 subset of _SCHEMA_STATEMENTS (excludes the v2 tables/indexes)."""
    out = []
    for stmt in db._SCHEMA_STATEMENTS:
        s = stmt.lower()
        if 'classification_runs' in s or 'label_ballots' in s:
            continue
        if 'idx_runs_overlay_status' in s or 'idx_ballots_' in s:
            continue
        out.append(stmt)
    return out


def _build_v1_db(run_dir):
    """Create a qra.db at schema_version=1 with theme + purer rater_votes rows."""
    conn = db.connect(db.db_path(run_dir))
    try:
        for stmt in _v1_statements():
            conn.execute(stmt)
        db.set_meta(conn, 'schema_version', 1)

        # theme_labels: two raters in a fixed slot order, incl. an ERROR ballot.
        rater_ids = db.dumps(['mA', 'mB'])
        seg1_votes = db.dumps([
            {'rater': 'mA', 'vote': 'CODED', 'stage': 2, 'confidence': 0.9,
             'secondary_stage': None, 'secondary_confidence': None,
             'justification': 'j1'},
            {'rater': 'mB', 'vote': 'ABSTAIN', 'stage': None, 'confidence': None,
             'secondary_stage': None, 'secondary_confidence': None,
             'justification': ''},
        ])
        seg2_votes = db.dumps([
            {'rater': 'mA', 'vote': 'ERROR', 'stage': None, 'confidence': None,
             'secondary_stage': None, 'secondary_confidence': None,
             'justification': ''},
            {'rater': 'mB', 'vote': 'CODED', 'stage': 1, 'confidence': 0.7,
             'secondary_stage': None, 'secondary_confidence': None,
             'justification': 'j2'},
        ])
        conn.execute(
            "INSERT INTO theme_labels (segment_id, primary_stage, rater_ids, rater_votes) "
            "VALUES (?, ?, ?, ?)", ('seg1', 2, rater_ids, seg1_votes))
        conn.execute(
            "INSERT INTO theme_labels (segment_id, primary_stage, rater_ids, rater_votes) "
            "VALUES (?, ?, ?, ?)", ('seg2', None, rater_ids, seg2_votes))

        # purer_labels: single rater.
        p_ids = db.dumps(['pM'])
        p_votes = db.dumps([
            {'rater': 'pM', 'vote': 'CODED', 'stage': 0, 'confidence': 0.8,
             'secondary_stage': None, 'secondary_confidence': None,
             'justification': 'pj'},
        ])
        conn.execute(
            "INSERT INTO purer_labels (segment_id, purer_primary, purer_rater_ids, "
            "purer_rater_votes) VALUES (?, ?, ?, ?)", ('tseg1', 0, p_ids, p_votes))
        conn.commit()
    finally:
        conn.close()


class TestMigrationV1ToV2(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_migration_runs_on_open(self):
        _build_v1_db(self.tmpdir)
        # Opening triggers ensure_schema -> create v2 tables -> _migrate_1_to_2.
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(db.get_meta(conn, 'schema_version'), '2')

            # theme runs: one per rater, in legacy slot order (mA before mB),
            # born selected + completed.
            theme_runs = conn.execute(
                "SELECT run_id, rater_label, model, status, selected, note, "
                "n_coded, n_abstain, n_error, n_total FROM classification_runs "
                "WHERE overlay='theme' ORDER BY run_id"
            ).fetchall()
            self.assertEqual([r['rater_label'] for r in theme_runs], ['mA', 'mB'])
            for r in theme_runs:
                self.assertEqual(r['status'], 'completed')
                self.assertEqual(r['selected'], 1)
                self.assertEqual(r['model'], r['rater_label'])
                self.assertEqual(r['note'], 'backfilled from rater_votes')

            run_by_label = {r['rater_label']: r for r in theme_runs}
            # mA: CODED on seg1, ERROR on seg2.
            self.assertEqual(run_by_label['mA']['n_coded'], 1)
            self.assertEqual(run_by_label['mA']['n_error'], 1)
            self.assertEqual(run_by_label['mA']['n_total'], 2)
            # mB: ABSTAIN on seg1, CODED on seg2.
            self.assertEqual(run_by_label['mB']['n_coded'], 1)
            self.assertEqual(run_by_label['mB']['n_abstain'], 1)
            self.assertEqual(run_by_label['mB']['n_total'], 2)

            # ballots: ERROR row has NULL raw_json; CODED row has raw_json.
            mA = run_by_label['mA']['run_id']
            err = conn.execute(
                "SELECT vote, stage, raw_json FROM label_ballots "
                "WHERE run_id=? AND segment_id='seg2'", (mA,)
            ).fetchone()
            self.assertEqual(err['vote'], 'ERROR')
            self.assertIsNone(err['stage'])
            self.assertIsNone(err['raw_json'])
            coded = conn.execute(
                "SELECT vote, stage, raw_json FROM label_ballots "
                "WHERE run_id=? AND segment_id='seg1'", (mA,)
            ).fetchone()
            self.assertEqual(coded['vote'], 'CODED')
            self.assertEqual(coded['stage'], 2)
            self.assertIsNotNone(coded['raw_json'])

            # purer run.
            purer_runs = conn.execute(
                "SELECT rater_label, n_coded FROM classification_runs WHERE overlay='purer'"
            ).fetchall()
            self.assertEqual(len(purer_runs), 1)
            self.assertEqual(purer_runs[0]['rater_label'], 'pM')
            self.assertEqual(purer_runs[0]['n_coded'], 1)

    def test_migration_idempotent(self):
        _build_v1_db(self.tmpdir)
        with db.open_db(self.tmpdir):
            pass  # first open migrates
        # Snapshot run + ballot counts.
        with db.open_db(self.tmpdir) as conn:
            runs_before = conn.execute("SELECT COUNT(*) FROM classification_runs").fetchone()[0]
            ballots_before = conn.execute("SELECT COUNT(*) FROM label_ballots").fetchone()[0]
            # Re-run the migration function directly — must be a no-op.
            db._migrate_1_to_2(conn)
            runs_after = conn.execute("SELECT COUNT(*) FROM classification_runs").fetchone()[0]
            ballots_after = conn.execute("SELECT COUNT(*) FROM label_ballots").fetchone()[0]
        self.assertEqual(runs_before, runs_after)
        self.assertEqual(ballots_before, ballots_after)

    def test_fresh_db_has_no_legacy_runs(self):
        # A brand-new DB stamps v2 with no migration -> no backfilled runs.
        with db.open_db(self.tmpdir) as conn:
            self.assertEqual(db.get_meta(conn, 'schema_version'), '2')
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM classification_runs").fetchone()[0], 0)
            self.assertEqual(
                conn.execute("SELECT COUNT(*) FROM label_ballots").fetchone()[0], 0)


if __name__ == '__main__':
    unittest.main()
