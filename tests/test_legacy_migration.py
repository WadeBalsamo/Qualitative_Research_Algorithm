"""Tests for process/legacy_migration.py."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from process import legacy_migration, output_paths as _paths


def _write_master_jsonl(run_dir, rows):
    """Write a minimal master_segments.jsonl for migration tests."""
    ms_dir = _paths.master_segments_dir(run_dir)
    os.makedirs(ms_dir, exist_ok=True)
    path = os.path.join(ms_dir, 'master_segments.jsonl')
    with open(path, 'w') as f:
        for row in rows:
            f.write(json.dumps(row) + '\n')
    return path


def _make_row(segment_id, session_id, segment_index=0, cohort_id=1, text='Test text.'):
    return {
        'segment_id': segment_id,
        'trial_id': 'trial_A',
        'participant_id': 'participant_1',
        'session_id': session_id,
        'session_number': 1,
        'cohort_id': cohort_id,
        'session_variant': '',
        'segment_index': segment_index,
        'start_time_ms': 0,
        'end_time_ms': 3000,
        'total_segments_in_session': 2,
        'speaker': 'participant',
        'text': text,
        'word_count': 2,
        'session_file': '',
        'primary_stage': 1,
        'secondary_stage': None,
        'llm_confidence_primary': 0.9,
        'llm_confidence_secondary': None,
        'agreement_level': 'unanimous',
        'agreement_fraction': 1.0,
        'needs_review': False,
        'label_status': 'llm_only',
    }


class TestIsLegacyProject(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detects_legacy(self):
        _write_master_jsonl(self.tmpdir, [_make_row('seg_1', 'c1s1')])
        self.assertTrue(legacy_migration.is_legacy_project(self.tmpdir))

    def test_not_legacy_when_segmented_exists(self):
        _write_master_jsonl(self.tmpdir, [_make_row('seg_1', 'c1s1')])
        seg_dir = _paths.segmented_session_dir(self.tmpdir, 'c1s1')
        os.makedirs(seg_dir)
        open(os.path.join(seg_dir, 'segments.jsonl'), 'w').close()
        self.assertFalse(legacy_migration.is_legacy_project(self.tmpdir))

    def test_not_legacy_when_no_master(self):
        self.assertFalse(legacy_migration.is_legacy_project(self.tmpdir))


class TestMigrateLegacySegments(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _write_master_jsonl(self.tmpdir, [
            _make_row('seg_1', 'c1s1', segment_index=0),
            _make_row('seg_2', 'c1s1', segment_index=1),
            _make_row('seg_3', 'c1s2', segment_index=0),
        ])

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_per_session_files(self):
        n = legacy_migration.migrate_legacy_segments(self.tmpdir)
        self.assertEqual(n, 2)
        self.assertTrue(os.path.isfile(_paths.session_segments_path(self.tmpdir, 'c1s1')))
        self.assertTrue(os.path.isfile(_paths.session_segments_path(self.tmpdir, 'c1s2')))

    def test_meta_has_legacy_hash(self):
        legacy_migration.migrate_legacy_segments(self.tmpdir)
        meta_path = _paths.segmentation_meta_path(self.tmpdir, 'c1s1')
        with open(meta_path) as f:
            meta = json.load(f)
        self.assertEqual(meta['params_hash'], 'legacy-pre-modular')

    def test_skips_already_migrated(self):
        legacy_migration.migrate_legacy_segments(self.tmpdir)
        n = legacy_migration.migrate_legacy_segments(self.tmpdir)
        self.assertEqual(n, 0)

    def test_segments_have_correct_ids(self):
        from process import segments_io
        legacy_migration.migrate_legacy_segments(self.tmpdir)
        segs = segments_io.read_session_segments(self.tmpdir, 'c1s1')
        self.assertEqual({s.segment_id for s in segs}, {'seg_1', 'seg_2'})


class TestMigrateLegacyTestsets(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        _write_master_jsonl(self.tmpdir, [
            _make_row('seg_1', 'c1s1', segment_index=0, text='First.'),
            _make_row('seg_2', 'c1s1', segment_index=1, text='Second.'),
        ])
        ts_dir = _paths.testsets_dir(self.tmpdir)
        os.makedirs(ts_dir, exist_ok=True)
        human_ws = os.path.join(ts_dir, 'human_classification_testset_worksheet_1.txt')
        with open(human_ws, 'w') as f:
            f.write("=" * 78 + "\n")
            f.write("VALIDATION TEST SET 1 of 1 — HUMAN CODING WORKSHEET\n")
            f.write("=" * 78 + "\n")
            f.write("[ITEM 001]  Session: c1s1   Segment 001\n")
            f.write("            00:00:00–00:00:03   2w   Participant: participant_1\n")
            f.write("  First.\n\n")
        ai_ws = os.path.join(ts_dir, 'AI_classification_testset_worksheet_1.txt')
        with open(ai_ws, 'w') as f:
            f.write("AI worksheet content\n")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_new_layout(self):
        n = legacy_migration.migrate_legacy_testsets(self.tmpdir)
        self.assertEqual(n, 1)
        self.assertTrue(os.path.isfile(
            _paths.testset_human_worksheet_path(self.tmpdir, 'vaamr_testset_1')
        ))
        self.assertTrue(os.path.isfile(
            _paths.testset_answer_key_path(self.tmpdir, 'vaamr_testset_1')
        ))
        self.assertTrue(os.path.isfile(
            _paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_1')
        ))

    def test_legacy_files_moved_not_copied(self):
        ts_dir = _paths.testsets_dir(self.tmpdir)
        legacy_migration.migrate_legacy_testsets(self.tmpdir)
        self.assertFalse(os.path.isfile(
            os.path.join(ts_dir, 'human_classification_testset_worksheet_1.txt')
        ))

    def test_manifest_has_correct_segment(self):
        legacy_migration.migrate_legacy_testsets(self.tmpdir)
        manifest_path = _paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_1')
        with open(manifest_path) as f:
            manifest = json.load(f)
        self.assertIn('seg_1', manifest['segment_ids'])
        self.assertTrue(manifest.get('migrated_from_legacy'))

    def test_skips_already_migrated(self):
        legacy_migration.migrate_legacy_testsets(self.tmpdir)
        n = legacy_migration.migrate_legacy_testsets(self.tmpdir)
        self.assertEqual(n, 0)


if __name__ == '__main__':
    unittest.main()
