"""
Tests for segments_io.load_segments_for_stage (Phase 3).

Covers: frozen-segment loading, overlay application, empty-project error,
partial apply tuple, and missing-overlay graceful no-op.
"""
import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import segments_io, classifications_io


def _make_segment(segment_id='seg_001', session_id='c1s1', speaker='participant', **kwargs):
    defaults = dict(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=0,
        start_time_ms=0,
        end_time_ms=5000,
        total_segments_in_session=2,
        speaker=speaker,
        text='Test segment.',
        word_count=2,
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1/session.json',
    )
    defaults.update(kwargs)
    return Segment(**defaults)


def _write_segs(run_dir, session_id, segs, hash_='testhash'):
    segments_io.write_session_segments(run_dir, session_id, segs, hash_)


class TestLoadSegmentsForStage(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_loads_frozen_segments_from_disk(self):
        segs = [
            _make_segment('seg_001', text='First.'),
            _make_segment('seg_002', text='Second.'),
        ]
        _write_segs(self.tmpdir, 'c1s1', segs)

        loaded = segments_io.load_segments_for_stage(self.tmpdir)
        self.assertEqual(len(loaded), 2)
        ids = {s.segment_id for s in loaded}
        self.assertIn('seg_001', ids)
        self.assertIn('seg_002', ids)

    def test_loads_multiple_sessions(self):
        _write_segs(self.tmpdir, 'c1s1', [_make_segment('s1_001', session_id='c1s1')])
        _write_segs(self.tmpdir, 'c1s2', [_make_segment('s2_001', session_id='c1s2')])

        loaded = segments_io.load_segments_for_stage(self.tmpdir)
        self.assertEqual(len(loaded), 2)

    def test_raises_when_no_frozen_segments(self):
        with self.assertRaises(FileNotFoundError) as ctx:
            segments_io.load_segments_for_stage(self.tmpdir)
        self.assertIn('qra ingest', str(ctx.exception))

    def test_applies_theme_overlay_when_requested(self):
        seg = _make_segment('seg_001')
        _write_segs(self.tmpdir, 'c1s1', [seg])

        # Write a theme overlay
        overlay_seg = _make_segment('seg_001')
        overlay_seg.primary_stage = 2
        overlay_seg.agreement_level = 'unanimous'
        classifications_io.write_theme_overlay(self.tmpdir, [overlay_seg])

        loaded = segments_io.load_segments_for_stage(self.tmpdir, apply=('theme',))
        seg_loaded = next(s for s in loaded if s.segment_id == 'seg_001')
        self.assertEqual(seg_loaded.primary_stage, 2)
        self.assertEqual(seg_loaded.agreement_level, 'unanimous')

    def test_excludes_overlay_when_not_in_apply_tuple(self):
        seg = _make_segment('seg_001')
        _write_segs(self.tmpdir, 'c1s1', [seg])

        overlay_seg = _make_segment('seg_001')
        overlay_seg.primary_stage = 3
        classifications_io.write_theme_overlay(self.tmpdir, [overlay_seg])

        # Request empty apply tuple — no overlays applied
        loaded = segments_io.load_segments_for_stage(self.tmpdir, apply=())
        seg_loaded = next(s for s in loaded if s.segment_id == 'seg_001')
        self.assertIsNone(seg_loaded.primary_stage)

    def test_missing_overlay_in_apply_tuple_is_graceful(self):
        seg = _make_segment('seg_001')
        _write_segs(self.tmpdir, 'c1s1', [seg])

        # purer overlay doesn't exist — should not raise
        loaded = segments_io.load_segments_for_stage(
            self.tmpdir, apply=('theme', 'purer', 'codebook', 'cv'),
        )
        self.assertEqual(len(loaded), 1)
        self.assertIsNone(loaded[0].primary_stage)

    def test_default_apply_includes_all_overlays(self):
        seg = _make_segment('seg_001')
        _write_segs(self.tmpdir, 'c1s1', [seg])

        overlay_seg = _make_segment('seg_001')
        overlay_seg.primary_stage = 1
        classifications_io.write_theme_overlay(self.tmpdir, [overlay_seg])

        # Default apply=('theme','purer','codebook','cv')
        loaded = segments_io.load_segments_for_stage(self.tmpdir)
        self.assertEqual(loaded[0].primary_stage, 1)


if __name__ == '__main__':
    unittest.main()
