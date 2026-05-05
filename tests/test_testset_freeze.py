"""Tests for frozen testset create/refresh logic in process/assembly/human_forms.py."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_tools.data_structures import Segment
from process._freeze import FrozenArtifactError
from process.assembly.human_forms import (
    create_frozen_testset,
    refresh_testset_answer_key,
    generate_or_refresh_validation_testsets,
)
from process import output_paths as _paths


def _make_segment(segment_id, session_id='c1s1', cohort_id=1, speaker='participant',
                  text='A test segment.', segment_index=0, primary_stage=1):
    return Segment(
        segment_id=segment_id,
        trial_id='t1',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=cohort_id,
        session_variant='',
        segment_index=segment_index,
        start_time_ms=0,
        end_time_ms=3000,
        total_segments_in_session=5,
        speaker=speaker,
        text=text,
        word_count=3,
        speakers_in_segment=[speaker],
        session_file='',
        primary_stage=primary_stage,
        llm_confidence_primary=0.9,
        agreement_level='unanimous',
    )


def _make_pool(n=10):
    return [
        _make_segment(f'seg_{i:03d}', segment_index=i, text=f'Segment number {i}.')
        for i in range(n)
    ]


class TestCreateFrozenTestset(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_all_artifacts(self):
        segs = _make_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='vaamr_testset_1', n_sets=2, set_index=1,
            fraction_per_set=0.3, random_seed=42, codebook_enabled=False,
        )
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_snapshot_path(self.tmpdir, 'vaamr_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_human_worksheet_path(self.tmpdir, 'vaamr_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_answer_key_path(self.tmpdir, 'vaamr_testset_1')))

    def test_raises_on_second_create(self):
        segs = _make_pool(10)
        kwargs = dict(name='vaamr_testset_1', n_sets=2, set_index=1,
                      fraction_per_set=0.3, random_seed=42, codebook_enabled=False)
        create_frozen_testset(segs, None, self.tmpdir, **kwargs)
        with self.assertRaises(FrozenArtifactError):
            create_frozen_testset(segs, None, self.tmpdir, **kwargs)

    def test_force_recreates(self):
        segs = _make_pool(10)
        kwargs = dict(name='vaamr_testset_1', n_sets=2, set_index=1,
                      fraction_per_set=0.3, random_seed=42, codebook_enabled=False)
        create_frozen_testset(segs, None, self.tmpdir, **kwargs)
        human_path = _paths.testset_human_worksheet_path(self.tmpdir, 'vaamr_testset_1')
        mtime_before = os.path.getmtime(human_path)
        import time; time.sleep(0.05)
        create_frozen_testset(segs, None, self.tmpdir, force=True, **kwargs)
        self.assertNotEqual(os.path.getmtime(human_path), mtime_before)

    def test_manifest_has_correct_kind(self):
        import json
        segs = _make_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='vaamr_testset_1', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=0, codebook_enabled=False,
        )
        manifest_path = _paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_1')
        with open(manifest_path) as f:
            m = json.load(f)
        self.assertEqual(m['kind'], 'vaamr')
        self.assertEqual(m['set_index'], 1)


class TestRefreshTestsetAnswerKey(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.segs = _make_pool(10)
        create_frozen_testset(
            self.segs, None, self.tmpdir,
            name='vaamr_testset_1', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_refresh_updates_answer_key_only(self):
        import time
        ai_path = _paths.testset_answer_key_path(self.tmpdir, 'vaamr_testset_1')
        human_path = _paths.testset_human_worksheet_path(self.tmpdir, 'vaamr_testset_1')
        mtime_human_before = os.path.getmtime(human_path)
        time.sleep(0.05)

        segs_by_id = {s.segment_id: s for s in self.segs}
        refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 'vaamr_testset_1',
                                   codebook_enabled=False)

        self.assertEqual(os.path.getmtime(human_path), mtime_human_before)

    def test_drift_raises_frozen_artifact_error(self):
        import json as _json
        # Read the manifest to find a segment_id actually in the testset
        manifest_path = _paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_1')
        with open(manifest_path) as f:
            manifest = _json.load(f)
        testset_id = manifest['segment_ids'][0]

        segs_by_id = {s.segment_id: s for s in self.segs}
        segs_by_id[testset_id].text = 'completely different text that drifted'

        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 'vaamr_testset_1',
                                       codebook_enabled=False)

    def test_missing_segment_raises(self):
        with self.assertRaises(FrozenArtifactError):
            refresh_testset_answer_key({}, None, self.tmpdir, 'vaamr_testset_1',
                                       codebook_enabled=False)


class TestGenerateOrRefreshCoordinator(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_testsets_on_first_call(self):
        segs = _make_pool(20)
        dirs = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=2, fraction_per_set=0.2,
            random_seed=42, codebook_enabled=False,
        )
        self.assertEqual(len(dirs), 2)
        for d in dirs:
            self.assertTrue(os.path.isdir(d))

    def test_refreshes_on_second_call(self):
        # With n_sets=1 via legacy kwargs, name has no numeric suffix
        segs = _make_pool(20)
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=1, fraction_per_set=0.3,
            random_seed=42, codebook_enabled=False,
        )
        human_path = _paths.testset_human_worksheet_path(self.tmpdir, 'vaamr_testset')
        mtime_before = os.path.getmtime(human_path)

        import time; time.sleep(0.05)
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, n_sets=1, fraction_per_set=0.3,
            random_seed=42, codebook_enabled=False,
        )
        self.assertEqual(os.path.getmtime(human_path), mtime_before)


if __name__ == '__main__':
    unittest.main()
