"""
tests/unit/test_masterpiece_additions.py
-----------------------------------------
Coverage for the control-surface additions:

  - process.reclassify_ops       (delete_checkpoints / clear_overlay / reset_for_fresh)
  - classifications_io.clear_overlay
  - gnn_layer.validation.format_gate_verdict
  - gnn_layer.soft_labels.ballot_coverage
  - process.db.ensure_schema forward-migration + SchemaVersionError
  - CLI smoke: `gnn status` (absent) and `migrate` (preview / no-op)

Hermetic: no network, no models, no Ollama.
"""
import argparse
import os
import shutil
import sqlite3
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import db, segments_io, classifications_io as cio, reclassify_ops, output_paths as paths


def _seg(sid, session='c1s1', idx=0, speaker='participant', text='hello world'):
    return Segment(segment_id=sid, session_id=session, segment_index=idx,
                   speaker=speaker, text=text)


def _seed_project(run_dir):
    """A frozen session + a theme overlay so reset/clear have something to act on."""
    segs = [_seg('s1', idx=0), _seg('s2', idx=1)]
    segments_io.write_session_segments(run_dir, 'c1s1', segs, 'phash')
    t = _seg('s1'); t.primary_stage = 2
    cio.write_theme_overlay(run_dir, [t])


# ---------------------------------------------------------------------------
# reclassify_ops + clear_overlay
# ---------------------------------------------------------------------------

class TestReclassifyOps(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        _seed_project(self.run_dir)

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_clear_overlay_empties_table(self):
        self.assertTrue(cio.overlay_exists(self.run_dir, 'theme'))
        cio.clear_overlay(self.run_dir, 'theme')
        self.assertFalse(cio.overlay_exists(self.run_dir, 'theme'))

    def test_clear_overlay_missing_db_is_noop(self):
        empty = tempfile.mkdtemp()
        try:
            cio.clear_overlay(empty, 'theme')  # must not raise
        finally:
            shutil.rmtree(empty, ignore_errors=True)

    def test_delete_checkpoints_counts_only_matching_prefix(self):
        ckpt_dir = paths.llm_checkpoints_dir(self.run_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        for name in ('llm_results_a_runs.json', 'llm_results_b_runs.json',
                     'codebook_llm_results_x_runs.json', 'purer_cue_results_y_runs.json'):
            open(os.path.join(ckpt_dir, name), 'w').close()
        removed = reclassify_ops.delete_checkpoints(self.run_dir, 'vaamr')
        self.assertEqual(removed, 2)  # only the two llm_results_* files
        # codebook + purer checkpoints survive (distinct prefixes)
        survivors = set(os.listdir(ckpt_dir))
        self.assertIn('codebook_llm_results_x_runs.json', survivors)
        self.assertIn('purer_cue_results_y_runs.json', survivors)

    def test_reset_for_fresh_clears_checkpoints_and_overlay(self):
        ckpt_dir = paths.llm_checkpoints_dir(self.run_dir)
        os.makedirs(ckpt_dir, exist_ok=True)
        open(os.path.join(ckpt_dir, 'llm_results_z_runs.json'), 'w').close()
        res = reclassify_ops.reset_for_fresh(self.run_dir, 'vaamr')
        self.assertEqual(res['checkpoints_removed'], 1)
        self.assertTrue(res['overlay_cleared'])
        self.assertFalse(cio.overlay_exists(self.run_dir, 'theme'))

    def test_reset_for_fresh_cross_validation_has_no_checkpoints(self):
        res = reclassify_ops.reset_for_fresh(self.run_dir, 'cross-validation')
        self.assertEqual(res['checkpoints_removed'], 0)
        self.assertTrue(res['overlay_cleared'])  # cv overlay key still clears


# ---------------------------------------------------------------------------
# db forward-migration scaffold
# ---------------------------------------------------------------------------

class TestSchemaForwardMigration(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_fresh_db_stamps_current_version(self):
        with db.open_db(self.run_dir) as conn:
            self.assertEqual(db.get_meta(conn, 'schema_version'), str(db.SCHEMA_VERSION))

    def test_same_version_is_noop(self):
        with db.open_db(self.run_dir):
            pass
        # Re-open: must not raise and must keep the version.
        with db.open_db(self.run_dir) as conn:
            self.assertEqual(db.get_meta(conn, 'schema_version'), str(db.SCHEMA_VERSION))

    def test_newer_db_raises(self):
        conn = db.connect(db.db_path(self.run_dir))
        db.ensure_schema(conn)
        db.set_meta(conn, 'schema_version', str(db.SCHEMA_VERSION + 5))
        conn.commit()
        conn.close()
        with self.assertRaises(db.SchemaVersionError):
            with db.open_db(self.run_dir):
                pass

    def test_older_db_without_migration_raises(self):
        # Simulate a DB stamped older than the code with no registered migration.
        conn = db.connect(db.db_path(self.run_dir))
        db.ensure_schema(conn)
        db.set_meta(conn, 'schema_version', '0')
        conn.commit()
        conn.close()
        # SCHEMA_VERSION is currently 1; migrating 0->1 has no registered step.
        if db.SCHEMA_VERSION >= 1:
            with self.assertRaises(db.SchemaVersionError):
                with db.open_db(self.run_dir):
                    pass


# ---------------------------------------------------------------------------
# gnn_layer.validation.format_gate_verdict
# ---------------------------------------------------------------------------

class TestFormatGateVerdict(unittest.TestCase):
    def test_none_says_not_run(self):
        from gnn_layer.validation import format_gate_verdict
        out = format_gate_verdict(None, '/x')
        self.assertIn('not run', out.lower())

    def test_ready_verdict_renders_kappa_and_ready(self):
        from gnn_layer.validation import format_gate_verdict
        out = format_gate_verdict({
            'ready_for_scaling': True, 'vaamr_kappa': 0.78, 'vaamr_ready': True,
            'purer_kappa': 0.5, 'purer_ready': False, 'irr_target': 0.61,
            'rare_stage_notes': [], 'calibration_temperature': 1.3,
        }, '/x')
        self.assertIn('READY', out)
        self.assertIn('0.78', out)
        self.assertIn('YES', out)


# ---------------------------------------------------------------------------
# gnn_layer.soft_labels.ballot_coverage
# ---------------------------------------------------------------------------

class TestBallotCoverage(unittest.TestCase):
    def test_counts_multirun_participant_ballots(self):
        import pandas as pd
        from gnn_layer.soft_labels import ballot_coverage
        df = pd.DataFrame([
            {'speaker': 'participant', 'segment_id': 'a', 'rater_votes': [1, 1, 2]},
            {'speaker': 'participant', 'segment_id': 'b', 'rater_votes': [1]},
            {'speaker': 'therapist', 'segment_id': 'c', 'rater_votes': None},
        ])
        cov = ballot_coverage(df)
        self.assertEqual(cov['n_participant'], 2)
        self.assertEqual(cov['n_with_multirun_ballots'], 1)
        self.assertAlmostEqual(cov['multirun_fraction'], 0.5)

    def test_empty_and_missing_column(self):
        import pandas as pd
        from gnn_layer.soft_labels import ballot_coverage
        self.assertEqual(ballot_coverage(pd.DataFrame())['n_participant'], 0)
        df = pd.DataFrame([{'speaker': 'participant', 'segment_id': 'a'}])  # no rater_votes col
        self.assertEqual(ballot_coverage(df)['n_with_multirun_ballots'], 0)


# ---------------------------------------------------------------------------
# CLI smoke: gnn status (absent) and migrate (preview / no-op)
# ---------------------------------------------------------------------------

class TestNewCliCommands(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.run_dir, ignore_errors=True)

    def test_gnn_status_absent_runs(self):
        import qra
        qra.cmd_gnn_status(argparse.Namespace(output_dir=self.run_dir, json=False))  # must not raise

    def test_migrate_no_legacy_files(self):
        import qra
        # An empty dir is neither a JSONL project nor has qra.db → "nothing to migrate".
        qra.cmd_migrate(argparse.Namespace(output_dir=self.run_dir, run=False))

    def test_migrate_skips_when_db_present(self):
        import qra
        with db.open_db(self.run_dir):
            pass  # creates qra.db
        qra.cmd_migrate(argparse.Namespace(output_dir=self.run_dir, run=True))
        # qra.db still present, no _legacy_files created
        self.assertTrue(db.db_exists(self.run_dir))


if __name__ == '__main__':
    unittest.main()
