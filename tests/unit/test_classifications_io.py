"""
Tests for process/classifications_io.py (Phase 3).

Covers overlay round-trips, apply_*_overlay field restoration,
manifest merge/isolation, sorted-record guarantee, and missing-file no-ops.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment
from process import segments_io
from process._freeze import write_frozen


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
        total_segments_in_session=3,
        speaker=speaker,
        text='Test segment text.',
        word_count=3,
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1/session.json',
    )
    defaults.update(kwargs)
    return Segment(**defaults)


def _write_frozen_segments(run_dir, session_id, segments):
    """Helper: write frozen segments for a session."""
    segments_io.write_session_segments(run_dir, session_id, segments, 'testhash')


# ---------------------------------------------------------------------------
# Import target (will fail until classifications_io exists)
# ---------------------------------------------------------------------------
class TestImport(unittest.TestCase):
    def test_module_imports(self):
        from process import classifications_io  # noqa: F401


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
class TestPathHelpers(unittest.TestCase):
    def test_overlay_path_returns_expected_subpath(self):
        from process import classifications_io as cio
        p = cio.overlay_path('/out', 'theme')
        self.assertTrue(p.endswith('theme_labels.jsonl'))
        self.assertIn('02_meta', p)
        self.assertIn('classifications', p)

    def test_overlay_path_all_keys(self):
        from process import classifications_io as cio
        # Each key produces a distinct absolute path ending in .jsonl
        paths = {key: cio.overlay_path('/out', key) for key in ('theme', 'purer', 'codebook', 'cv')}
        for key, p in paths.items():
            self.assertTrue(os.path.isabs(p), f"{key}: expected absolute path")
            self.assertTrue(p.endswith('.jsonl'), f"{key}: expected .jsonl extension")
        # All paths must be distinct
        self.assertEqual(len(set(paths.values())), 4)

    def test_manifest_path_location(self):
        from process import classifications_io as cio
        p = cio.manifest_path('/out')
        self.assertIn('02_meta', p)
        self.assertIn('classifications', p)
        self.assertTrue(p.endswith('classification_manifest.json'))


# ---------------------------------------------------------------------------
# write_theme_overlay / apply_theme_overlay round-trip
# ---------------------------------------------------------------------------
class TestThemeOverlayRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_classified_seg(self, seg_id, primary, secondary=None):
        seg = _make_segment(segment_id=seg_id)
        seg.primary_stage = primary
        seg.secondary_stage = secondary
        seg.llm_confidence_primary = 0.9
        seg.llm_confidence_secondary = 0.4 if secondary is not None else None
        seg.llm_justification = 'Test justification.'
        seg.rater_ids = ['rater_1', 'rater_2']
        seg.rater_votes = [{'rater': 'rater_1', 'stage': primary, 'confidence': 0.9}]
        seg.agreement_level = 'unanimous'
        seg.agreement_fraction = 1.0
        seg.needs_review = False
        seg.consensus_vote = primary
        seg.tie_broken_by_confidence = False
        seg.llm_run_consistency = 2
        return seg

    def test_roundtrip_restores_theme_fields(self):
        from process import classifications_io as cio
        seg1 = self._make_classified_seg('seg_001', primary=2, secondary=3)
        seg2 = self._make_classified_seg('seg_002', primary=0)
        cio.write_theme_overlay(self.tmpdir, [seg1, seg2])

        # New blank segments
        blank1 = _make_segment('seg_001')
        blank2 = _make_segment('seg_002')
        by_id = {'seg_001': blank1, 'seg_002': blank2}
        n = cio.apply_theme_overlay(self.tmpdir, by_id)

        self.assertEqual(n, 2)
        self.assertEqual(blank1.primary_stage, 2)
        self.assertEqual(blank1.secondary_stage, 3)
        self.assertAlmostEqual(blank1.llm_confidence_primary, 0.9)
        self.assertEqual(blank1.agreement_level, 'unanimous')
        self.assertEqual(blank1.rater_ids, ['rater_1', 'rater_2'])
        self.assertEqual(blank2.primary_stage, 0)
        self.assertIsNone(blank2.secondary_stage)

    def test_overlay_file_exists_after_write(self):
        from process import classifications_io as cio
        seg = self._make_classified_seg('seg_001', 1)
        path = cio.write_theme_overlay(self.tmpdir, [seg])
        self.assertTrue(os.path.isfile(path))

    def test_overlay_is_valid_jsonl(self):
        from process import classifications_io as cio
        seg = self._make_classified_seg('seg_001', 1)
        cio.write_theme_overlay(self.tmpdir, [seg])
        with open(cio.overlay_path(self.tmpdir, 'theme'), encoding='utf-8') as fh:
            lines = [ln for ln in fh if ln.strip()]
        self.assertEqual(len(lines), 1)
        rec = json.loads(lines[0])
        self.assertIn('segment_id', rec)
        self.assertEqual(rec['segment_id'], 'seg_001')
        self.assertEqual(rec['primary_stage'], 1)

    def test_overlay_records_sorted_by_segment_id(self):
        from process import classifications_io as cio
        segs = [
            self._make_classified_seg('seg_003', 0),
            self._make_classified_seg('seg_001', 2),
            self._make_classified_seg('seg_002', 1),
        ]
        cio.write_theme_overlay(self.tmpdir, segs)
        with open(cio.overlay_path(self.tmpdir, 'theme'), encoding='utf-8') as fh:
            ids = [json.loads(ln)['segment_id'] for ln in fh if ln.strip()]
        self.assertEqual(ids, sorted(ids))

    def test_apply_missing_overlay_returns_zero(self):
        from process import classifications_io as cio
        blank = _make_segment('seg_001')
        n = cio.apply_theme_overlay(self.tmpdir, {'seg_001': blank})
        self.assertEqual(n, 0)
        self.assertIsNone(blank.primary_stage)


# ---------------------------------------------------------------------------
# write_purer_overlay / apply_purer_overlay round-trip
# ---------------------------------------------------------------------------
class TestPurerOverlayRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_purer_seg(self, seg_id, primary):
        seg = _make_segment(segment_id=seg_id, speaker='therapist')
        seg.purer_primary = primary
        seg.purer_secondary = None
        seg.purer_confidence_primary = 0.85
        seg.purer_confidence_secondary = None
        seg.purer_justification = 'PURER justification.'
        seg.purer_run_consistency = 2
        seg.purer_agreement_level = 'unanimous'
        seg.purer_agreement_fraction = 1.0
        seg.purer_needs_review = False
        seg.purer_rater_ids = ['rater_1']
        seg.purer_rater_votes = [{'rater': 'rater_1', 'stage': primary}]
        return seg

    def test_roundtrip_restores_purer_fields(self):
        from process import classifications_io as cio
        seg = self._make_purer_seg('th_001', primary=2)
        cio.write_purer_overlay(self.tmpdir, [seg])

        blank = _make_segment('th_001', speaker='therapist')
        n = cio.apply_purer_overlay(self.tmpdir, {'th_001': blank})
        self.assertEqual(n, 1)
        self.assertEqual(blank.purer_primary, 2)
        self.assertAlmostEqual(blank.purer_confidence_primary, 0.85)
        self.assertEqual(blank.purer_agreement_level, 'unanimous')

    def test_apply_missing_purer_overlay_returns_zero(self):
        from process import classifications_io as cio
        blank = _make_segment('th_001', speaker='therapist')
        n = cio.apply_purer_overlay(self.tmpdir, {'th_001': blank})
        self.assertEqual(n, 0)

    def test_purer_overlay_sorted_by_segment_id(self):
        from process import classifications_io as cio
        segs = [
            self._make_purer_seg('th_003', 0),
            self._make_purer_seg('th_001', 1),
        ]
        cio.write_purer_overlay(self.tmpdir, segs)
        with open(cio.overlay_path(self.tmpdir, 'purer'), encoding='utf-8') as fh:
            ids = [json.loads(ln)['segment_id'] for ln in fh if ln.strip()]
        self.assertEqual(ids, sorted(ids))


# ---------------------------------------------------------------------------
# write_codebook_overlay / apply_codebook_overlay round-trip
# ---------------------------------------------------------------------------
class TestCodebookOverlayRoundtrip(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _make_codebook_seg(self, seg_id):
        seg = _make_segment(segment_id=seg_id)
        seg.codebook_labels_embedding = ['VE.1', 'VE.2']
        seg.codebook_labels_llm = ['VE.1']
        seg.codebook_labels_ensemble = ['VE.1']
        seg.codebook_disagreements = ['VE.2']
        seg.codebook_confidence = {'VE.1': 0.9}
        return seg

    def test_roundtrip_restores_codebook_fields(self):
        from process import classifications_io as cio
        seg = self._make_codebook_seg('seg_001')
        cio.write_codebook_overlay(self.tmpdir, [seg])

        blank = _make_segment('seg_001')
        n = cio.apply_codebook_overlay(self.tmpdir, {'seg_001': blank})
        self.assertEqual(n, 1)
        self.assertEqual(blank.codebook_labels_ensemble, ['VE.1'])
        self.assertEqual(blank.codebook_labels_embedding, ['VE.1', 'VE.2'])
        self.assertEqual(blank.codebook_disagreements, ['VE.2'])
        self.assertAlmostEqual(blank.codebook_confidence['VE.1'], 0.9)

    def test_apply_missing_codebook_overlay_returns_zero(self):
        from process import classifications_io as cio
        blank = _make_segment('seg_001')
        n = cio.apply_codebook_overlay(self.tmpdir, {'seg_001': blank})
        self.assertEqual(n, 0)


# ---------------------------------------------------------------------------
# write_cross_validation_overlay / apply_cross_validation_overlay
# ---------------------------------------------------------------------------
class TestCrossValidationOverlay(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_write_cv_overlay_creates_file(self):
        from process import classifications_io as cio
        seg = _make_segment('seg_001')
        # CV overlay fields don't map to Segment; file is written but apply is no-op
        cio.write_cross_validation_overlay(self.tmpdir, [seg])
        self.assertTrue(os.path.isfile(cio.overlay_path(self.tmpdir, 'cv')))

    def test_apply_cv_overlay_missing_returns_zero(self):
        from process import classifications_io as cio
        blank = _make_segment('seg_001')
        n = cio.apply_cross_validation_overlay(self.tmpdir, {'seg_001': blank})
        self.assertEqual(n, 0)


# ---------------------------------------------------------------------------
# update_classification_manifest
# ---------------------------------------------------------------------------
class TestManifest(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_creates_manifest_when_absent(self):
        from process import classifications_io as cio
        cio.update_classification_manifest(
            self.tmpdir,
            key='theme',
            entry={'model': 'test-model', 'n_segments': 5},
        )
        m = cio.read_classification_manifest(self.tmpdir)
        self.assertIsNotNone(m)
        self.assertIn('theme', m)
        self.assertEqual(m['theme']['model'], 'test-model')

    def test_merges_keys_without_clobbering_siblings(self):
        from process import classifications_io as cio
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'model-A'},
        )
        cio.update_classification_manifest(
            self.tmpdir, key='purer', entry={'model': 'model-B'},
        )
        m = cio.read_classification_manifest(self.tmpdir)
        self.assertEqual(m['theme']['model'], 'model-A')
        self.assertEqual(m['purer']['model'], 'model-B')

    def test_update_overwrites_same_key(self):
        from process import classifications_io as cio
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'old-model'},
        )
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'new-model'},
        )
        m = cio.read_classification_manifest(self.tmpdir)
        self.assertEqual(m['theme']['model'], 'new-model')

    def test_manifest_entry_has_completed_at(self):
        from process import classifications_io as cio
        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'x'},
        )
        m = cio.read_classification_manifest(self.tmpdir)
        self.assertIn('completed_at', m['theme'])

    def test_read_manifest_returns_none_when_absent(self):
        from process import classifications_io as cio
        result = cio.read_classification_manifest(self.tmpdir)
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Overlay overwrite semantics (overlays are re-writable unlike frozen segments)
# ---------------------------------------------------------------------------
class TestOverlayOverwrite(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_theme_overlay_can_be_overwritten(self):
        from process import classifications_io as cio
        seg = _make_segment('seg_001')
        seg.primary_stage = 1
        cio.write_theme_overlay(self.tmpdir, [seg])

        seg.primary_stage = 3
        cio.write_theme_overlay(self.tmpdir, [seg])  # Should NOT raise

        blank = _make_segment('seg_001')
        cio.apply_theme_overlay(self.tmpdir, {'seg_001': blank})
        self.assertEqual(blank.primary_stage, 3)

    def test_purer_overlay_can_be_overwritten(self):
        from process import classifications_io as cio
        seg = _make_segment('th_001', speaker='therapist')
        seg.purer_primary = 0
        cio.write_purer_overlay(self.tmpdir, [seg])
        seg.purer_primary = 4
        cio.write_purer_overlay(self.tmpdir, [seg])  # Should NOT raise
        blank = _make_segment('th_001', speaker='therapist')
        cio.apply_purer_overlay(self.tmpdir, {'th_001': blank})
        self.assertEqual(blank.purer_primary, 4)


# ---------------------------------------------------------------------------
# Segment_id not in overlay → segment untouched
# ---------------------------------------------------------------------------
class TestPartialOverlayApplication(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_only_matching_segments_updated(self):
        from process import classifications_io as cio
        seg_in = _make_segment('seg_001')
        seg_in.primary_stage = 2
        cio.write_theme_overlay(self.tmpdir, [seg_in])

        known = _make_segment('seg_001')
        unknown = _make_segment('seg_999')
        n = cio.apply_theme_overlay(self.tmpdir, {'seg_001': known, 'seg_999': unknown})
        self.assertEqual(n, 1)
        self.assertEqual(known.primary_stage, 2)
        self.assertIsNone(unknown.primary_stage)


if __name__ == '__main__':
    unittest.main()
