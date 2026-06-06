"""
Tests for new Phase 3 CLI subcommands: qra ingest, qra classify, qra assemble.

Uses subprocess to call qra.py so the CLI parsing is also exercised.
Tests verify exit codes, error messages, and on-disk side effects.
"""
import json
import os
import subprocess
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

QRA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'qra.py')
VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    '.venv', 'bin', 'python',
)
PYTHON = VENV_PYTHON if os.path.isfile(VENV_PYTHON) else sys.executable


def _run(args, cwd=None):
    result = subprocess.run(
        [PYTHON, QRA] + args,
        capture_output=True, text=True, cwd=cwd,
    )
    return result


class TestQraHelp(unittest.TestCase):
    """Sanity: new subcommands appear in --help output."""

    def test_ingest_in_help(self):
        r = _run(['--help'])
        # ingest subcommand should be listed
        self.assertIn('ingest', r.stdout + r.stderr)

    def test_classify_in_help(self):
        r = _run(['--help'])
        self.assertIn('classify', r.stdout + r.stderr)

    def test_assemble_in_help(self):
        r = _run(['--help'])
        self.assertIn('assemble', r.stdout + r.stderr)


class TestQraClassifyGuards(unittest.TestCase):
    """qra classify must refuse to run without frozen segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_classify_errors_without_frozen_segments(self):
        r = _run(['classify', '-o', self.tmpdir, '--what', 'vaamr'])
        self.assertNotEqual(r.returncode, 0)
        combined = r.stdout + r.stderr
        # Should mention ingest
        self.assertIn('ingest', combined.lower())

    def test_classify_all_errors_without_frozen_segments(self):
        r = _run(['classify', '-o', self.tmpdir])
        self.assertNotEqual(r.returncode, 0)


class TestQraAssembleGuards(unittest.TestCase):
    """qra assemble must refuse to run without any overlay files."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_assemble_errors_without_overlays_or_segments(self):
        r = _run(['assemble', '-o', self.tmpdir])
        self.assertNotEqual(r.returncode, 0)


class TestQraIngestHelp(unittest.TestCase):
    """qra ingest --help must not crash."""

    def test_ingest_help_exits_zero(self):
        r = _run(['ingest', '--help'])
        self.assertEqual(r.returncode, 0)

    def test_classify_help_exits_zero(self):
        r = _run(['classify', '--help'])
        self.assertEqual(r.returncode, 0)

    def test_assemble_help_exits_zero(self):
        r = _run(['assemble', '--help'])
        self.assertEqual(r.returncode, 0)


class TestQraAssembleWithPreclassifiedData(unittest.TestCase):
    """qra assemble on a directory with frozen segments + theme overlay writes master_segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Write frozen segments manually
        from process import segments_io, classifications_io
        from classification_tools.data_structures import Segment

        seg1 = Segment(
            segment_id='seg_001',
            trial_id='trial_A',
            participant_id='participant_1',
            session_id='c1s1',
            session_number=1,
            cohort_id=1,
            session_variant='',
            segment_index=0,
            start_time_ms=0,
            end_time_ms=5000,
            total_segments_in_session=2,
            speaker='participant',
            text='Mindfulness helps me notice my pain.',
            word_count=7,
        )
        seg2 = Segment(
            segment_id='seg_002',
            trial_id='trial_A',
            participant_id='participant_1',
            session_id='c1s1',
            session_number=1,
            cohort_id=1,
            session_variant='',
            segment_index=1,
            start_time_ms=5000,
            end_time_ms=10000,
            total_segments_in_session=2,
            speaker='participant',
            text='I can observe the sensation without reacting.',
            word_count=8,
        )
        segments_io.write_session_segments(self.tmpdir, 'c1s1', [seg1, seg2], 'testhash')

        # Write theme overlay
        seg1.primary_stage = 2
        seg1.agreement_level = 'unanimous'
        seg1.agreement_fraction = 1.0
        seg2.primary_stage = 3
        seg2.agreement_level = 'unanimous'
        seg2.agreement_fraction = 1.0
        classifications_io.write_theme_overlay(self.tmpdir, [seg1, seg2])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_assemble_writes_master_segments(self):
        r = _run(['assemble', '-o', self.tmpdir])
        if r.returncode != 0:
            self.fail(f"qra assemble failed:\n{r.stdout}\n{r.stderr}")

        from process import output_paths as _paths
        ms_dir = _paths.master_segments_dir(self.tmpdir)
        jsonl_files = [f for f in os.listdir(ms_dir) if 'master_segments' in f]
        self.assertTrue(len(jsonl_files) >= 1, "master_segments JSONL not created")

    def test_assemble_exits_zero(self):
        r = _run(['assemble', '-o', self.tmpdir])
        self.assertEqual(r.returncode, 0, f"stderr: {r.stderr}")


class TestQraClassifyWhatFlag(unittest.TestCase):
    """qra classify --what accepts valid values."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Write frozen segments so --what theme doesn't fail on "no segments" guard
        from process import segments_io
        from classification_tools.data_structures import Segment
        seg = Segment(
            segment_id='seg_001', session_id='c1s1',
            speaker='participant', text='Test.', word_count=1,
            trial_id='t', participant_id='p', session_number=1,
            cohort_id=1, session_variant='', segment_index=0,
            start_time_ms=0, end_time_ms=1000, total_segments_in_session=1,
        )
        segments_io.write_session_segments(self.tmpdir, 'c1s1', [seg], 'h')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_classify_what_invalid_exits_nonzero(self):
        r = _run(['classify', '-o', self.tmpdir, '--what', 'banana'])
        self.assertNotEqual(r.returncode, 0)


if __name__ == '__main__':
    unittest.main()
