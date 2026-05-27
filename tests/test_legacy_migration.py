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


class TestIsV25Layout(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_detects_v25_by_diarized_dir(self):
        os.makedirs(os.path.join(self.tmpdir, '01_transcripts', 'diarized'))
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_detects_v25_by_coded_dir(self):
        os.makedirs(os.path.join(self.tmpdir, '01_transcripts', 'coded'))
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_detects_v25_by_flat_cv_file(self):
        val_dir = os.path.join(self.tmpdir, '04_validation')
        os.makedirs(val_dir)
        open(os.path.join(val_dir, 'content_validity_test_set.jsonl'), 'w').close()
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_detects_v25_by_flat_human_classification(self):
        val_dir = os.path.join(self.tmpdir, '04_validation')
        os.makedirs(val_dir)
        open(os.path.join(val_dir, 'human_classification_c1s1.txt'), 'w').close()
        self.assertTrue(legacy_migration.is_v25_layout(self.tmpdir))

    def test_not_v25_for_fresh_project(self):
        self.assertFalse(legacy_migration.is_v25_layout(self.tmpdir))

    def test_not_v25_for_v3_with_inputs_dir(self):
        os.makedirs(os.path.join(self.tmpdir, '01_transcripts_inputs'))
        os.makedirs(os.path.join(self.tmpdir, '04_validation', 'full_transcripts'))
        self.assertFalse(legacy_migration.is_v25_layout(self.tmpdir))


class TestMigrateV25ToV3(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_dir(self, *parts):
        d = os.path.join(self.tmpdir, *parts)
        os.makedirs(d, exist_ok=True)
        return d

    def _touch(self, *parts):
        path = os.path.join(self.tmpdir, *parts)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, 'w').close()
        return path

    def test_moves_diarized_to_inputs(self):
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertTrue(os.path.isfile(os.path.join(self.tmpdir, '01_transcripts_inputs', 'c1s1.vtt')))
        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, '01_transcripts', 'diarized', 'c1s1.vtt')))

    def test_removes_empty_diarized_dir(self):
        self._make_dir('01_transcripts', 'diarized')
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertFalse(os.path.isdir(os.path.join(self.tmpdir, '01_transcripts', 'diarized')))

    def test_moves_coded_transcripts_to_full_transcripts(self):
        self._touch('01_transcripts', 'coded', 'coded_transcript_c1s1.txt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        full_tx = os.path.join(self.tmpdir, '04_validation', 'full_transcripts')
        self.assertTrue(os.path.isfile(os.path.join(full_tx, 'coded_transcript_c1s1.txt')))
        self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, '01_transcripts', 'coded', 'coded_transcript_c1s1.txt')))

    def test_moves_human_classification_to_full_transcripts(self):
        val_dir = self._make_dir('04_validation')
        self._touch('04_validation', 'human_classification_c1s1.txt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        full_tx = os.path.join(self.tmpdir, '04_validation', 'full_transcripts')
        self.assertTrue(os.path.isfile(os.path.join(full_tx, 'human_classification_c1s1.txt')))
        self.assertFalse(os.path.isfile(os.path.join(val_dir, 'human_classification_c1s1.txt')))

    def test_moves_cv_files_to_content_validity(self):
        for fname in ['content_validity_test_set.jsonl', 'content_validity_human_worksheet.txt',
                      'content_validity_definition_key.txt', 'content_validity_answer_key.txt']:
            self._touch('04_validation', fname)
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        cv_dir = os.path.join(self.tmpdir, '04_validation', 'content_validity')
        for fname in ['content_validity_test_set.jsonl', 'content_validity_human_worksheet.txt',
                      'content_validity_definition_key.txt', 'content_validity_answer_key.txt']:
            self.assertTrue(os.path.isfile(os.path.join(cv_dir, fname)), f"Missing {fname}")
            self.assertFalse(os.path.isfile(os.path.join(self.tmpdir, '04_validation', fname)))

    def test_removes_worksheetN_base_files(self):
        ts_dir = self._make_dir('04_validation', 'testsets')
        self._touch('04_validation', 'testsets', 'worksheet1_base.txt')
        self._touch('04_validation', 'testsets', 'worksheet2_base.txt')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertFalse(os.path.isfile(os.path.join(ts_dir, 'worksheet1_base.txt')))
        self.assertFalse(os.path.isfile(os.path.join(ts_dir, 'worksheet2_base.txt')))

    def test_removes_folder_based_testset_dirs(self):
        ts_dir = self._make_dir('04_validation', 'testsets')
        legacy_dir = self._make_dir('04_validation', 'testsets', 'vaamr_testset_1')
        with open(os.path.join(legacy_dir, 'manifest.json'), 'w') as f:
            json.dump({'kind': 'vaamr'}, f)
        with open(os.path.join(legacy_dir, 'human_worksheet.txt'), 'w') as f:
            f.write('worksheet content')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        self.assertFalse(os.path.isdir(legacy_dir))

    def test_does_not_touch_flat_frozen_worksheets(self):
        ts_dir = self._make_dir('04_validation', 'testsets')
        ws_path = os.path.join(ts_dir, 'human_classification_testset_worksheet_1.txt')
        sentinel = 'FROZEN WORKSHEET - DO NOT MODIFY'
        with open(ws_path, 'w') as f:
            f.write(sentinel)
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        with open(ws_path) as f:
            self.assertEqual(f.read(), sentinel)

    def test_idempotent(self):
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        self._touch('01_transcripts', 'coded', 'coded_transcript_c1s1.txt')
        self._touch('04_validation', 'human_classification_c1s1.txt')
        self._touch('04_validation', 'content_validity_test_set.jsonl')
        self._touch('04_validation', 'testsets', 'worksheet1_base.txt')
        r1 = legacy_migration.migrate_v25_to_v3(self.tmpdir)
        r2 = legacy_migration.migrate_v25_to_v3(self.tmpdir)
        # Second run should move/delete nothing
        self.assertEqual(r2.get('diarized_moved', 0), 0)
        self.assertEqual(r2.get('coded_moved', 0), 0)
        self.assertEqual(r2.get('human_classification_moved', 0), 0)
        self.assertEqual(r2.get('cv_files_moved', 0), 0)
        self.assertEqual(r2.get('legacy_base_removed', 0), 0)

    def test_skips_destination_if_already_exists(self):
        self._touch('01_transcripts', 'diarized', 'c1s1.vtt')
        dst = os.path.join(self.tmpdir, '01_transcripts_inputs', 'c1s1.vtt')
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with open(dst, 'w') as f:
            f.write('existing content')
        legacy_migration.migrate_v25_to_v3(self.tmpdir)
        with open(dst) as f:
            self.assertEqual(f.read(), 'existing content')


if __name__ == '__main__':
    unittest.main()
