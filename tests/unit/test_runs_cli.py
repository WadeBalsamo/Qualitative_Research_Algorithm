"""CLI smoke tests for the `qra runs` subcommand group + the classify shim glue.

Argparse-level wiring (subparsers parse, dispatch fields land) plus direct
``cmd_runs_*`` calls on a tmp project seeded with runs + ballots (no LLM, no
network).  Downstream-chaining commands are exercised with ``--no-downstream`` or
via the non-chaining verbs (queue/list/show/select).
"""
import io
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest import mock

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

import qra  # noqa: E402  (top-level CLI module)
from classification_tools.data_structures import Segment
from process import segments_io, run_registry as rr, classifications_io as cio


def _seed_project_with_runs(tmp):
    """Frozen participant segments + two completed, ballot-bearing theme runs."""
    raw = [
        Segment(segment_id=f'p{i}', trial_id='t', participant_id='P1',
                session_id='c1s1', session_number=1, cohort_id=1, segment_index=i,
                speaker='participant', text='I notice the pain and avoid moving.',
                word_count=7, start_time_ms=i * 1000, end_time_ms=i * 1000 + 500)
        for i in range(3)
    ]
    segments_io.write_session_segments(tmp, 'c1s1', raw, 'hash1')
    ids = []
    for model, stage in (('mA', 1), ('mB', 1)):
        rid = rr.create_run(tmp, overlay='theme', model=model, rater_label=model)
        rr.update_run(tmp, rid, status='completed', selected=1)
        cells = {f'p{i}': {'vote': 'CODED', 'primary_stage': stage,
                           'primary_confidence': 0.8, 'secondary_stage': None,
                           'secondary_confidence': None, 'justification': 'j',
                           'evidence_phrase': 'e'} for i in range(3)}
        rr.upsert_ballots(tmp, 'theme', rid, cells)
        rr.refresh_counters(tmp, rid)
        ids.append(rid)
    return ids


class _NS:
    """Minimal argparse-Namespace stand-in for direct cmd_ calls."""
    def __init__(self, **kw):
        self.__dict__.update(kw)
    def __getattr__(self, name):
        return None  # unset flags default to None/falsey


# ---------------------------------------------------------------------------
# Argparse wiring
# ---------------------------------------------------------------------------

class TestRunsArgparse(unittest.TestCase):
    def setUp(self):
        self.parser = qra._build_parser()[0]

    def test_queue_parses_all_flags(self):
        args = self.parser.parse_args([
            'runs', 'queue', '-o', '/tmp/x', '--what', 'purer',
            '--model', 'google/gemma-4-31b', '--quant', 'Q4_K_M',
            '--thinking', 'off', '--note', 'pilot', '--alias', 'gemma-q4',
            '--temperature', '0.2', '--backend', 'lmstudio',
        ])
        self.assertEqual(args.command, 'runs')
        self.assertEqual(args.runs_command, 'queue')
        self.assertEqual(args.what, 'purer')
        self.assertEqual(args.model, 'google/gemma-4-31b')
        self.assertEqual(args.quant, 'Q4_K_M')
        self.assertEqual(args.thinking, 'off')
        self.assertEqual(args.alias, 'gemma-q4')
        self.assertEqual(args.temperature, 0.2)

    def test_start_parses_flags(self):
        args = self.parser.parse_args([
            'runs', 'start', '-o', '/tmp/x', '--retries', '3',
            '--no-downstream', '--force', '--what', 'vaamr',
        ])
        self.assertEqual(args.runs_command, 'start')
        self.assertEqual(args.retries, 3)
        self.assertTrue(args.no_downstream)
        self.assertTrue(args.force)
        self.assertEqual(args.what, 'vaamr')

    def test_select_and_show_parse(self):
        a = self.parser.parse_args(['runs', 'select', '-o', '/t', '--what', 'vaamr',
                                    '--ids', '1,4,7'])
        self.assertEqual(a.runs_command, 'select')
        self.assertEqual(a.ids, '1,4,7')
        b = self.parser.parse_args(['runs', 'show', '-o', '/t', '--run-id', '5'])
        self.assertEqual(b.run_id, 5)


# ---------------------------------------------------------------------------
# Direct command calls on a seeded project
# ---------------------------------------------------------------------------

class TestRunsCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.ids = _seed_project_with_runs(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_queue_creates_a_queued_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_queue(_NS(output_dir=self.tmp, what='vaamr',
                                   model='mC', config=None))
        runs = rr.list_runs(self.tmp, overlay='theme')
        labels = {r['rater_label']: r for r in runs}
        self.assertIn('mC', labels)
        self.assertEqual(labels['mC']['status'], 'queued')
        self.assertFalse(labels['mC']['selected'])

    def test_list_renders_counters(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_list(_NS(output_dir=self.tmp, what='all', json=False))
        out = buf.getvalue()
        self.assertIn('VAAMR runs', out)
        self.assertIn('mA', out)
        self.assertIn('mB', out)

    def test_list_json(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_list(_NS(output_dir=self.tmp, what='theme', json=True))
        import json as _json
        data = _json.loads(buf.getvalue())
        self.assertEqual(len(data['theme']), 2)

    def test_show_one_run(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_show(_NS(output_dir=self.tmp, run_id=self.ids[0], json=False))
        out = buf.getvalue()
        self.assertIn('rater_label', out)
        self.assertIn('mA', out)

    def test_archive_then_excluded(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_archive(_NS(output_dir=self.tmp, run_id=self.ids[1]))
        run = rr.get_run(self.tmp, self.ids[1])
        self.assertEqual(run['status'], 'archived')
        self.assertFalse(run['selected'])

    def test_select_ids_rebuilds_overlay(self):
        # Select only mA → rebuild → overlay reflects exactly that run's ballots.
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_select(_NS(output_dir=self.tmp, what='vaamr',
                                    ids=str(self.ids[0]), config=None, auto=False,
                                    no_downstream=True))
        self.assertEqual(rr.selected_runs(self.tmp, 'theme'), [self.ids[0]])
        overlay = {r['segment_id']: r for r in cio.read_overlay(self.tmp, 'theme')}
        self.assertEqual(overlay['p0']['primary_stage'], 1)
        self.assertEqual(overlay['p0']['rater_ids'], ['mA'])

    def test_select_auto_fallback_selects_all(self):
        # No human IRR codes here → --auto hits the fallback (select all eligible).
        # Downstream suppressed via --no-downstream.
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_select(_NS(output_dir=self.tmp, what='vaamr',
                                    ids=None, auto=True, config=None,
                                    no_downstream=True))
        out = buf.getvalue()
        self.assertIn('FALLBACK', out)
        # Both seeded runs selected; a selection record persisted.
        self.assertEqual(set(rr.selected_runs(self.tmp, 'theme')), set(self.ids))
        from analysis import run_selection as _rsel
        rec = _rsel.load_selection_record(self.tmp, 'theme')
        self.assertIsNotNone(rec)
        self.assertTrue(rec['fallback_used'])

    def test_select_auto_and_ids_conflict_errors(self):
        with self.assertRaises(SystemExit):
            qra.cmd_runs_select(_NS(output_dir=self.tmp, what='vaamr',
                                    ids='1', auto=True, config=None))

    def test_select_neither_auto_nor_ids_errors(self):
        with self.assertRaises(SystemExit):
            qra.cmd_runs_select(_NS(output_dir=self.tmp, what='vaamr',
                                    ids=None, auto=False, config=None))

    def test_select_ids_manual_records_strategy(self):
        # --ids routes through select_runs(strategy='manual'); records + rebuilds.
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_select(_NS(output_dir=self.tmp, what='vaamr',
                                    ids=str(self.ids[0]), config=None, auto=False,
                                    no_downstream=True))
        self.assertEqual(rr.selected_runs(self.tmp, 'theme'), [self.ids[0]])
        from analysis import run_selection as _rsel
        rec = _rsel.load_selection_record(self.tmp, 'theme')
        self.assertEqual(rec['strategy'], 'manual')
        self.assertEqual(rec['selected_run_ids'], [self.ids[0]])

    def test_sync_ballots_no_checkpoints_is_graceful(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_runs_sync_ballots(_NS(output_dir=self.tmp, what='all', config=None))
        self.assertIn('No legacy checkpoints', buf.getvalue())


# ---------------------------------------------------------------------------
# classify shim glue (no LLM — assert run get-or-create + execute_queue wiring)
# ---------------------------------------------------------------------------

class TestClassifyShimGlue(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [Segment(segment_id='p0', trial_id='t', participant_id='P1',
                       session_id='c1s1', session_number=1, cohort_id=1,
                       segment_index=0, speaker='participant',
                       text='I notice the pain.', word_count=4,
                       start_time_ms=0, end_time_ms=500)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'h')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_shim_translates_models_to_runs_and_calls_executor(self):
        from process.config import PipelineConfig
        cfg = PipelineConfig()
        cfg.output_dir = self.tmp
        cfg.theme_classification.per_run_models = ['mA', 'mB', 'mC']
        cfg.theme_classification.n_runs = 3

        captured = {}

        def fake_execute_queue(run_dir, config, *, overlays=(), **kw):
            captured['overlays'] = overlays
            captured['run_ids'] = rr.selected_runs  # sentinel
            return {'per_run': {}, 'overlays_rebuilt': list(overlays),
                    'stopped_early': False, 'skipped_queued': []}

        args = _NS(output_dir=self.tmp, what='vaamr', framework=None,
                   no_downstream=True, fresh=False, zero_shot=False,
                   config=None, resume_from=None)
        with mock.patch.object(qra, '_build_config', return_value=cfg), \
             mock.patch('process.run_executor.execute_queue', side_effect=fake_execute_queue):
            qra.cmd_classify(args)

        # Three theme runs created (one per model), and the executor was invoked
        # for the theme overlay.
        runs = rr.list_runs(self.tmp, overlay='theme')
        self.assertEqual(sorted(r['rater_label'] for r in runs), ['mA', 'mB', 'mC'])
        self.assertEqual(captured['overlays'], ('theme',))


# ---------------------------------------------------------------------------
# fix-errors argparse wiring + dry-run smoke (M4)
# ---------------------------------------------------------------------------

class TestFixErrorsArgparse(unittest.TestCase):
    def setUp(self):
        self.parser = qra._build_parser()[0]

    def test_fix_errors_basic(self):
        args = self.parser.parse_args([
            'fix-errors', '-o', '/tmp/x',
            '--what', 'vaamr',
            '--max-passes', '3',
            '--dry-run',
        ])
        self.assertEqual(args.command, 'fix-errors')
        self.assertEqual(args.what, 'vaamr')
        self.assertEqual(args.max_passes, 3)
        self.assertTrue(args.dry_run)

    def test_fix_errors_all_default(self):
        args = self.parser.parse_args(['fix-errors', '-o', '/tmp/y'])
        self.assertEqual(args.what, 'all')
        self.assertFalse(args.dry_run)
        self.assertFalse(args.force)

    def test_fix_errors_run_id_repeated(self):
        args = self.parser.parse_args([
            'fix-errors', '-o', '/tmp/z', '--run-id', '3', '--run-id', '5',
        ])
        self.assertEqual(args.run_id, ['3', '5'])

    def test_fix_errors_no_downstream(self):
        args = self.parser.parse_args(['fix-errors', '-o', '/tmp/w', '--no-downstream'])
        self.assertTrue(args.no_downstream)

    def test_fix_errors_force(self):
        args = self.parser.parse_args(['fix-errors', '-o', '/tmp/f', '--force'])
        self.assertTrue(args.force)

    def test_fix_errors_what_choices(self):
        for what in ('vaamr', 'purer', 'codebook', 'all'):
            args = self.parser.parse_args(['fix-errors', '-o', '/t', '--what', what])
            self.assertEqual(args.what, what)


class TestFixErrorsDryRunSmoke(unittest.TestCase):
    """Smoke-test cmd_fix_errors with --dry-run on a seeded project."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        # Seed with a completed run + one ERROR ballot.
        raw = [Segment(segment_id='p0', trial_id='t', participant_id='P1',
                       session_id='c1s1', session_number=1, cohort_id=1,
                       segment_index=0, speaker='participant',
                       text='test', word_count=1,
                       start_time_ms=0, end_time_ms=500)]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'h1')
        rid = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')
        rr.upsert_ballots(self.tmp, 'theme', rid, {'p0': None})
        rr.update_run(self.tmp, rid, status='completed_with_errors', selected=1)
        rr.refresh_counters(self.tmp, rid)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_dry_run_does_not_crash(self):
        import sqlite3
        db_path = os.path.join(self.tmp, 'qra.db')
        with sqlite3.connect(db_path) as conn:
            before_count = conn.execute(
                "SELECT COUNT(*) FROM label_ballots").fetchone()[0]

        buf = io.StringIO()
        with redirect_stdout(buf):
            qra.cmd_fix_errors(_NS(
                output_dir=self.tmp, what='vaamr', run_id=None,
                max_passes=2, dry_run=True, force=False,
                no_downstream=True, config=None,
            ))

        with sqlite3.connect(db_path) as conn:
            after_count = conn.execute(
                "SELECT COUNT(*) FROM label_ballots").fetchone()[0]
        self.assertEqual(before_count, after_count,
                         "dry_run must not change label_ballots")
        out = buf.getvalue()
        self.assertIn('dry', out.lower())


if __name__ == '__main__':
    unittest.main()
