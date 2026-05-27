"""Tests for Phase 2 testset kind support (PURER, codebook) in human_forms.py."""
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_tools.data_structures import Segment
from process._freeze import FrozenArtifactError
from process import output_paths as _paths


def _make_segment(segment_id, session_id='c1s1', cohort_id=1, speaker='participant',
                  text='A test segment.', segment_index=0, primary_stage=1,
                  purer_primary=None, codebook_labels_ensemble=None):
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
        purer_primary=purer_primary,
        codebook_labels_ensemble=codebook_labels_ensemble,
    )


def _make_purer_pool(n=10):
    """Make therapist segments with purer_primary labels."""
    segs = []
    for i in range(n):
        segs.append(_make_segment(
            f'tseg_{i:03d}',
            speaker='therapist',
            segment_index=i,
            text=f'Therapist segment {i}.',
            purer_primary=i % 5,
        ))
    return segs


def _make_codebook_pool(n=10):
    """Make participant segments with codebook_labels_ensemble labels."""
    segs = []
    for i in range(n):
        segs.append(_make_segment(
            f'cseg_{i:03d}',
            speaker='participant',
            segment_index=i,
            text=f'Participant segment with codes {i}.',
            codebook_labels_ensemble=['code_A', 'code_B'],
        ))
    return segs


def _make_mixed_pool():
    """Make a pool with participant, therapist+labeled, and therapist+unlabeled segments."""
    return [
        _make_segment('p1', speaker='participant', text='Participant seg.', primary_stage=1),
        _make_segment('t1_labeled', speaker='therapist', text='Therapist with label.', purer_primary=2),
        _make_segment('t2_nolabel', speaker='therapist', text='Therapist no label.', purer_primary=None),
        _make_segment('p2_codes', speaker='participant', text='Participant with codes.',
                      codebook_labels_ensemble=['code_X']),
        _make_segment('p3_nocodes', speaker='participant', text='Participant without codes.'),
    ]


class TestPoolBuilders(unittest.TestCase):
    """Tests for _pool_purer and _pool_codebook."""

    def test_pool_purer_returns_only_labeled_therapist_segments(self):
        from process.assembly.human_forms import _pool_purer
        segs = _make_mixed_pool()
        pool = _pool_purer(segs)
        ids = {s.segment_id for s in pool}
        self.assertIn('t1_labeled', ids)
        self.assertNotIn('t2_nolabel', ids)
        self.assertNotIn('p1', ids)
        self.assertNotIn('p2_codes', ids)

    def test_pool_codebook_returns_only_participant_segs_with_codes(self):
        from process.assembly.human_forms import _pool_codebook
        segs = _make_mixed_pool()
        pool = _pool_codebook(segs)
        ids = {s.segment_id for s in pool}
        self.assertIn('p2_codes', ids)
        self.assertNotIn('p3_nocodes', ids)
        self.assertNotIn('t1_labeled', ids)
        self.assertNotIn('p1', ids)

    def test_pool_purer_empty_when_no_therapist_labels(self):
        from process.assembly.human_forms import _pool_purer
        segs = [_make_segment('p1'), _make_segment('p2')]
        self.assertEqual(_pool_purer(segs), [])

    def test_pool_codebook_empty_when_no_codes(self):
        from process.assembly.human_forms import _pool_codebook
        segs = [_make_segment('p1'), _make_segment('p2')]
        self.assertEqual(_pool_codebook(segs), [])


class TestCreateFrozenTestsetPurer(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_flat_files_for_purer(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 1)))
        self.assertEqual(human_path, _paths.testset_human_flat_path(self.tmpdir, 1))

    def test_purer_worksheet_header(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        with open(human_path) as f:
            content = f.read()
        self.assertIn('PURER', content)

    def test_purer_answer_key_header(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        with open(_paths.testset_ai_flat_path(self.tmpdir, 1)) as f:
            content = f.read()
        self.assertIn('PURER', content)

    def test_purer_worksheet_kind_detected(self):
        from process.assembly.human_forms import create_frozen_testset, _detect_worksheet_kind
        segs = _make_purer_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        self.assertEqual(_detect_worksheet_kind(human_path), 'purer')

    def test_purer_pool_excludes_participant_segments(self):
        """Pool for purer testset must contain only therapist segs — verified via pool builder."""
        from process.assembly.human_forms import _pool_purer, create_frozen_testset
        mixed = _make_mixed_pool()
        for i in range(5):
            mixed.append(_make_segment(f'extra_t{i}', speaker='therapist',
                                        text=f'Extra therapist {i}.', purer_primary=i % 5))
        pool = _pool_purer(mixed)
        for s in pool:
            self.assertEqual(s.speaker, 'therapist')
        # Also verify testset is created successfully from this pool
        human_path = create_frozen_testset(
            mixed, None, self.tmpdir,
            kind='purer', n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=0, codebook_enabled=False,
        )
        self.assertTrue(os.path.isfile(human_path))


class TestCreateFrozenTestsetCodebook(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_flat_files_for_codebook(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_codebook_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=True,
        )
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 1)))
        self.assertEqual(human_path, _paths.testset_human_flat_path(self.tmpdir, 1))

    def test_codebook_worksheet_header(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_codebook_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=True,
        )
        with open(human_path) as f:
            content = f.read()
        self.assertIn('CODEBOOK', content)

    def test_codebook_worksheet_kind_detected(self):
        from process.assembly.human_forms import create_frozen_testset, _detect_worksheet_kind
        segs = _make_codebook_pool(10)
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=True,
        )
        self.assertEqual(_detect_worksheet_kind(human_path), 'codebook')

    def test_codebook_pool_excludes_therapist_segments(self):
        """Pool for codebook testset must contain only participant segs with codes."""
        from process.assembly.human_forms import _pool_codebook, create_frozen_testset
        mixed = _make_mixed_pool()
        for i in range(5):
            mixed.append(_make_segment(f'extra_p{i}', speaker='participant',
                                        text=f'Extra participant {i}.',
                                        codebook_labels_ensemble=['code_Z']))
        pool = _pool_codebook(mixed)
        for s in pool:
            self.assertNotEqual(s.speaker, 'therapist')
        human_path = create_frozen_testset(
            mixed, None, self.tmpdir,
            kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=0, codebook_enabled=True,
        )
        self.assertTrue(os.path.isfile(human_path))


class TestRefreshDispatchesByKind(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_and_refresh(self, kind, segs):
        from process.assembly.human_forms import create_frozen_testset, refresh_testset_answer_key
        human_path = create_frozen_testset(
            segs, None, self.tmpdir,
            kind=kind, n_sets=1, set_index=1,
            fraction_per_set=0.8, random_seed=42, codebook_enabled=(kind == 'codebook'),
        )
        mtime_before = os.path.getmtime(human_path)
        time.sleep(0.05)

        segs_by_id = {s.segment_id: s for s in segs}
        refresh_testset_answer_key(segs_by_id, None, self.tmpdir, 1,
                                   codebook_enabled=(kind == 'codebook'))
        # Human worksheet must be byte-stable (not touched)
        self.assertEqual(os.path.getmtime(human_path), mtime_before)

    def test_refresh_purer_testset(self):
        self._create_and_refresh('purer', _make_purer_pool(10))

    def test_refresh_codebook_testset(self):
        self._create_and_refresh('codebook', _make_codebook_pool(10))


class TestTestSetsConfigCoordinator(unittest.TestCase):
    """Tests for generate_or_refresh_validation_testsets using TestSetsConfig."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_vaamr_and_purer_when_both_enabled(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        vaamr_segs = [_make_segment(f'p{i}', segment_index=i, text=f'Participant {i}.') for i in range(10)]
        purer_segs = _make_purer_pool(10)
        all_segs = vaamr_segs + purer_segs

        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.3, random_seed=42),
            purer=TestSetSpec(enabled=True, name='purer_testset', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
            codebook=TestSetSpec(enabled=False, name='codebook_testset'),
        )
        paths = generate_or_refresh_validation_testsets(
            all_segs, None, self.tmpdir,
            test_sets_config=ts_cfg,
            codebook_enabled=False,
        )
        # Should have created both vaamr (#1) and purer (#2) flat files
        self.assertEqual(len(paths), 2)
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_ai_flat_path(self.tmpdir, 2)))

    def test_skips_disabled_kinds(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segs = [_make_segment(f'p{i}', segment_index=i, text=f'P {i}.') for i in range(10)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.3, random_seed=42),
            purer=TestSetSpec(enabled=False, name='purer_testset'),
            codebook=TestSetSpec(enabled=False, name='codebook_testset'),
        )
        paths = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir,
            test_sets_config=ts_cfg,
            codebook_enabled=False,
        )
        self.assertEqual(len(paths), 1)
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertFalse(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))

    def test_creates_multiple_sets_per_kind(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segs = [_make_segment(f'p{i}', segment_index=i, text=f'P {i}.') for i in range(20)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=2,
                              fraction_per_set=0.2, random_seed=42),
            purer=TestSetSpec(enabled=False, name='purer_testset'),
            codebook=TestSetSpec(enabled=False, name='codebook_testset'),
        )
        paths = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir,
            test_sets_config=ts_cfg,
            codebook_enabled=False,
        )
        self.assertEqual(len(paths), 2)
        # Should produce worksheet #1 and #2
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))

    def test_second_call_creates_new_worksheets(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segs = [_make_segment(f'p{i}', segment_index=i, text=f'P {i}.') for i in range(10)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.3, random_seed=42),
            purer=TestSetSpec(enabled=False, name='purer_testset'),
            codebook=TestSetSpec(enabled=False, name='codebook_testset'),
        )
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, test_sets_config=ts_cfg, codebook_enabled=False)
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, test_sets_config=ts_cfg, codebook_enabled=False)
        # Second call should create #2 (not overwrite #1)
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 1)))
        self.assertTrue(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))

    def test_refresh_only_updates_ai_keys(self):
        from process.assembly.human_forms import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segs = [_make_segment(f'p{i}', segment_index=i, text=f'P {i}.') for i in range(10)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=1,
                              fraction_per_set=0.3, random_seed=42),
            purer=TestSetSpec(enabled=False, name='purer_testset'),
            codebook=TestSetSpec(enabled=False, name='codebook_testset'),
        )
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, test_sets_config=ts_cfg, codebook_enabled=False)
        human_path = _paths.testset_human_flat_path(self.tmpdir, 1)
        mtime_before = os.path.getmtime(human_path)
        time.sleep(0.05)

        # create_missing=False → refresh only
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, test_sets_config=ts_cfg,
            codebook_enabled=False, create_missing=False)
        # Human worksheet untouched, no new worksheet created
        self.assertEqual(os.path.getmtime(human_path), mtime_before)
        self.assertFalse(os.path.isfile(_paths.testset_human_flat_path(self.tmpdir, 2)))


if __name__ == '__main__':
    unittest.main()
