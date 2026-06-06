"""Tests for process/segments_io.py."""
import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import segments_io
from process._freeze import FrozenArtifactError


def _make_segment(segment_id='seg_001', session_id='c1s1', segment_index=0, **kwargs):
    defaults = dict(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=segment_index,
        start_time_ms=0,
        end_time_ms=5000,
        total_segments_in_session=3,
        speaker='participant',
        text='This is a test segment.',
        word_count=5,
        speakers_in_segment=['participant'],
        session_file='/data/input/c1s1/session.json',
    )
    defaults.update(kwargs)
    return Segment(**defaults)


class TestWriteReadRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_roundtrip_preserves_raw_fields(self):
        segs = [
            _make_segment('seg_001', segment_index=0, text='First segment.', word_count=2),
            _make_segment('seg_002', segment_index=1, text='Second segment.', word_count=2),
        ]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'abc123')
        loaded = segments_io.read_session_segments(self.tmpdir, 'c1s1')

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0].segment_id, 'seg_001')
        self.assertEqual(loaded[0].text, 'First segment.')
        self.assertEqual(loaded[0].segment_index, 0)
        self.assertEqual(loaded[1].segment_id, 'seg_002')
        self.assertEqual(loaded[1].segment_index, 1)

    def test_classification_fields_not_persisted(self):
        seg = _make_segment()
        seg.primary_stage = 2
        seg.llm_confidence_primary = 0.95
        seg.codebook_labels_ensemble = ['VE.1', 'VE.2']

        segments_io.write_session_segments(self.tmpdir, 'c1s1', [seg], 'hash1')
        loaded = segments_io.read_session_segments(self.tmpdir, 'c1s1')

        self.assertIsNone(loaded[0].primary_stage)
        self.assertIsNone(loaded[0].llm_confidence_primary)
        self.assertIsNone(loaded[0].codebook_labels_ensemble)

    def test_raises_on_overwrite(self):
        segs = [_make_segment()]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'hash1')
        with self.assertRaises(FrozenArtifactError):
            segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'hash1')

    def test_force_overwrites(self):
        segs = [_make_segment()]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'hash1')
        new_segs = [_make_segment(text='Updated.')]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', new_segs, 'hash1', force=True)
        loaded = segments_io.read_session_segments(self.tmpdir, 'c1s1')
        self.assertEqual(loaded[0].text, 'Updated.')


class TestIsSegmentationFresh(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_fresh_when_hash_matches(self):
        segs = [_make_segment()]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'myhash')
        self.assertTrue(segments_io.is_segmentation_fresh(self.tmpdir, 'c1s1', 'myhash'))

    def test_stale_when_hash_differs(self):
        segs = [_make_segment()]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'oldhash')
        self.assertFalse(segments_io.is_segmentation_fresh(self.tmpdir, 'c1s1', 'newhash'))

    def test_fresh_for_legacy_sentinel(self):
        segs = [_make_segment()]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs, 'legacy-pre-modular')
        self.assertTrue(segments_io.is_segmentation_fresh(self.tmpdir, 'c1s1', 'any-new-hash'))

    def test_not_fresh_when_missing(self):
        self.assertFalse(segments_io.is_segmentation_fresh(self.tmpdir, 'nonexistent', 'hash'))


class TestParamsHash(unittest.TestCase):
    def test_same_config_same_hash(self):
        from process.config import SegmentationConfig
        cfg = SegmentationConfig()
        h1 = segments_io.params_hash(cfg)
        h2 = segments_io.params_hash(cfg)
        self.assertEqual(h1, h2)

    def test_different_model_different_hash(self):
        from process.config import SegmentationConfig
        cfg1 = SegmentationConfig(embedding_model='model_A')
        cfg2 = SegmentationConfig(embedding_model='model_B')
        self.assertNotEqual(segments_io.params_hash(cfg1), segments_io.params_hash(cfg2))


if __name__ == '__main__':
    unittest.main()
