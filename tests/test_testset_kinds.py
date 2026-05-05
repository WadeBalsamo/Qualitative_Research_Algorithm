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
        # Only t1_labeled: therapist with purer_primary
        self.assertIn('t1_labeled', ids)
        self.assertNotIn('t2_nolabel', ids)   # no label
        self.assertNotIn('p1', ids)           # participant
        self.assertNotIn('p2_codes', ids)     # participant

    def test_pool_codebook_returns_only_participant_segs_with_codes(self):
        from process.assembly.human_forms import _pool_codebook
        segs = _make_mixed_pool()
        pool = _pool_codebook(segs)
        ids = {s.segment_id for s in pool}
        # Only p2_codes: participant with non-empty codebook_labels_ensemble
        self.assertIn('p2_codes', ids)
        self.assertNotIn('p3_nocodes', ids)    # no codes
        self.assertNotIn('t1_labeled', ids)    # therapist
        self.assertNotIn('p1', ids)            # no codebook labels

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

    def test_creates_all_artifacts_for_purer(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='purer_testset_1', kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'purer_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_snapshot_path(self.tmpdir, 'purer_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_human_worksheet_path(self.tmpdir, 'purer_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_answer_key_path(self.tmpdir, 'purer_testset_1')))

    def test_manifest_kind_is_purer(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='purer_testset_1', kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        with open(_paths.testset_manifest_path(self.tmpdir, 'purer_testset_1')) as f:
            m = json.load(f)
        self.assertEqual(m['kind'], 'purer')

    def test_purer_worksheet_header(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='purer_testset_1', kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        with open(_paths.testset_human_worksheet_path(self.tmpdir, 'purer_testset_1')) as f:
            content = f.read()
        self.assertIn('PURER', content)

    def test_purer_answer_key_header(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_purer_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='purer_testset_1', kind='purer', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=False,
        )
        with open(_paths.testset_answer_key_path(self.tmpdir, 'purer_testset_1')) as f:
            content = f.read()
        self.assertIn('PURER', content)

    def test_purer_pool_excludes_participant_segments(self):
        """Pool for purer testset must contain only therapist segs."""
        from process.assembly.human_forms import create_frozen_testset
        import json as _json
        # Mix participant + therapist; create purer testset
        mixed = _make_mixed_pool()
        # Add more therapist segs so pool is non-empty
        for i in range(5):
            mixed.append(_make_segment(f'extra_t{i}', speaker='therapist',
                                        text=f'Extra therapist {i}.', purer_primary=i % 5))
        create_frozen_testset(
            mixed, None, self.tmpdir,
            name='purer_testset_1', kind='purer', n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=0, codebook_enabled=False,
        )
        with open(_paths.testset_snapshot_path(self.tmpdir, 'purer_testset_1')) as f:
            snapshot_segs = [_json.loads(ln) for ln in f if ln.strip()]
        # All snapshot segments should be therapist
        for rec in snapshot_segs:
            self.assertEqual(rec['speaker'], 'therapist')


class TestCreateFrozenTestsetCodebook(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_all_artifacts_for_codebook(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_codebook_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='codebook_testset_1', kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=True,
        )
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'codebook_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_snapshot_path(self.tmpdir, 'codebook_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_human_worksheet_path(self.tmpdir, 'codebook_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_answer_key_path(self.tmpdir, 'codebook_testset_1')))

    def test_manifest_kind_is_codebook(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_codebook_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='codebook_testset_1', kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=True,
        )
        with open(_paths.testset_manifest_path(self.tmpdir, 'codebook_testset_1')) as f:
            m = json.load(f)
        self.assertEqual(m['kind'], 'codebook')

    def test_codebook_worksheet_header(self):
        from process.assembly.human_forms import create_frozen_testset
        segs = _make_codebook_pool(10)
        create_frozen_testset(
            segs, None, self.tmpdir,
            name='codebook_testset_1', kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=0.5, random_seed=42, codebook_enabled=True,
        )
        with open(_paths.testset_human_worksheet_path(self.tmpdir, 'codebook_testset_1')) as f:
            content = f.read()
        self.assertIn('CODEBOOK', content)

    def test_codebook_pool_excludes_therapist_segments(self):
        """Pool for codebook testset must contain only participant segs with codes."""
        from process.assembly.human_forms import create_frozen_testset
        import json as _json
        mixed = _make_mixed_pool()
        for i in range(5):
            mixed.append(_make_segment(f'extra_p{i}', speaker='participant',
                                        text=f'Extra participant {i}.',
                                        codebook_labels_ensemble=['code_Z']))
        create_frozen_testset(
            mixed, None, self.tmpdir,
            name='codebook_testset_1', kind='codebook', n_sets=1, set_index=1,
            fraction_per_set=1.0, random_seed=0, codebook_enabled=True,
        )
        with open(_paths.testset_snapshot_path(self.tmpdir, 'codebook_testset_1')) as f:
            snapshot_segs = [_json.loads(ln) for ln in f if ln.strip()]
        for rec in snapshot_segs:
            self.assertNotEqual(rec.get('speaker'), 'therapist')


class TestRefreshDispatchesByKind(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_and_refresh(self, kind, segs):
        from process.assembly.human_forms import create_frozen_testset, refresh_testset_answer_key
        name = f'{kind}_testset_1'
        create_frozen_testset(
            segs, None, self.tmpdir,
            name=name, kind=kind, n_sets=1, set_index=1,
            fraction_per_set=0.8, random_seed=42, codebook_enabled=(kind == 'codebook'),
        )
        human_path = _paths.testset_human_worksheet_path(self.tmpdir, name)
        mtime_before = os.path.getmtime(human_path)
        time.sleep(0.05)

        segs_by_id = {s.segment_id: s for s in segs}
        refresh_testset_answer_key(segs_by_id, None, self.tmpdir, name,
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
        dirs = generate_or_refresh_validation_testsets(
            all_segs, None, self.tmpdir,
            test_sets_config=ts_cfg,
            codebook_enabled=False,
        )
        # Should have created both vaamr and purer directories
        self.assertEqual(len(dirs), 2)
        # Check both manifests exist
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'vaamr_testset')))
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'purer_testset')))
        # Check kind in each manifest
        with open(_paths.testset_manifest_path(self.tmpdir, 'vaamr_testset')) as f:
            m1 = json.load(f)
        with open(_paths.testset_manifest_path(self.tmpdir, 'purer_testset')) as f:
            m2 = json.load(f)
        self.assertEqual(m1['kind'], 'vaamr')
        self.assertEqual(m2['kind'], 'purer')

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
        dirs = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir,
            test_sets_config=ts_cfg,
            codebook_enabled=False,
        )
        self.assertEqual(len(dirs), 1)
        self.assertFalse(os.path.exists(_paths.testset_manifest_path(self.tmpdir, 'purer_testset')))

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
        dirs = generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir,
            test_sets_config=ts_cfg,
            codebook_enabled=False,
        )
        self.assertEqual(len(dirs), 2)
        # Names should be vaamr_testset_1 and vaamr_testset_2 (when n_sets > 1)
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_1')))
        self.assertTrue(os.path.isfile(_paths.testset_manifest_path(self.tmpdir, 'vaamr_testset_2')))

    def test_refreshes_on_second_call(self):
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
        human_path = _paths.testset_human_worksheet_path(self.tmpdir, 'vaamr_testset')
        mtime_before = os.path.getmtime(human_path)
        time.sleep(0.05)
        generate_or_refresh_validation_testsets(
            segs, None, self.tmpdir, test_sets_config=ts_cfg, codebook_enabled=False)
        self.assertEqual(os.path.getmtime(human_path), mtime_before)


if __name__ == '__main__':
    unittest.main()
