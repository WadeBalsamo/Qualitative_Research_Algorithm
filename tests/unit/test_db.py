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
                         'idx_ts_items_ws', 'idx_cv_items_ts'):
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


if __name__ == '__main__':
    unittest.main()
