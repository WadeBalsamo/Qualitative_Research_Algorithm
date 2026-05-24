"""
Tests for process/orchestrator.py stage function decomposition (Phase 3).

Focuses on interface contracts: do stage functions exist, do they write
overlays when given pre-classified in-memory segments, and do they write
the manifest.  Full end-to-end pipeline execution is not exercised here
(that requires LLM backends) — these are structural/contract tests.
"""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        text='Test segment text here.',
        word_count=4,
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1/session.json',
    )
    defaults.update(kwargs)
    return Segment(**defaults)


def _classified_theme_seg(seg_id, primary, session_id='c1s1'):
    seg = _make_segment(seg_id, session_id=session_id)
    seg.primary_stage = primary
    seg.secondary_stage = None
    seg.llm_confidence_primary = 0.9
    seg.agreement_level = 'unanimous'
    seg.agreement_fraction = 1.0
    seg.needs_review = False
    seg.consensus_vote = primary
    seg.tie_broken_by_confidence = False
    seg.llm_run_consistency = 2
    seg.rater_ids = ['rater_1']
    seg.rater_votes = [{'rater': 'rater_1', 'stage': primary, 'confidence': 0.9}]
    return seg


def _classified_purer_seg(seg_id, primary, session_id='c1s1'):
    seg = _make_segment(seg_id, session_id=session_id, speaker='therapist')
    seg.purer_primary = primary
    seg.purer_confidence_primary = 0.8
    seg.purer_agreement_level = 'unanimous'
    seg.purer_agreement_fraction = 1.0
    seg.purer_needs_review = False
    seg.purer_rater_ids = ['rater_1']
    seg.purer_rater_votes = [{'rater': 'rater_1', 'stage': primary}]
    return seg


class TestStageImports(unittest.TestCase):
    """Stage functions must be importable from process.orchestrator."""

    def test_stage_classify_theme_importable(self):
        from process.orchestrator import stage_classify_theme  # noqa

    def test_stage_classify_purer_importable(self):
        from process.orchestrator import stage_classify_purer  # noqa

    def test_stage_classify_codebook_importable(self):
        from process.orchestrator import stage_classify_codebook  # noqa

    def test_stage_cross_validation_importable(self):
        from process.orchestrator import stage_cross_validation  # noqa

    def test_stage_assemble_importable(self):
        from process.orchestrator import stage_assemble  # noqa


class TestStageClassifyThemeOverlay(unittest.TestCase):
    """stage_classify_theme must write an overlay when given pre-classified segs."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Write frozen segments so standalone calls work
        segs_raw = [
            _make_segment('seg_001'),
            _make_segment('seg_002'),
        ]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs_raw, 'testhash')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_theme_overlay_from_classified_segments(self):
        from process.orchestrator import stage_classify_theme

        segs = [
            _classified_theme_seg('seg_001', primary=2),
            _classified_theme_seg('seg_002', primary=0),
        ]
        # Pass pre-classified segments; no LLM call needed
        stage_classify_theme(None, None, segments=segs, output_dir=self.tmpdir)

        overlay = classifications_io.overlay_path(self.tmpdir, 'theme')
        self.assertTrue(os.path.isfile(overlay))

        with open(overlay, encoding='utf-8') as fh:
            records = [json.loads(ln) for ln in fh if ln.strip()]
        self.assertEqual(len(records), 2)
        ids = {r['segment_id'] for r in records}
        self.assertIn('seg_001', ids)

    def test_theme_overlay_updates_manifest(self):
        from process.orchestrator import stage_classify_theme

        segs = [_classified_theme_seg('seg_001', primary=1)]
        stage_classify_theme(None, None, segments=segs, output_dir=self.tmpdir)

        manifest = classifications_io.read_classification_manifest(self.tmpdir)
        self.assertIsNotNone(manifest)
        self.assertIn('theme', manifest)

    def test_theme_stage_returns_segments(self):
        from process.orchestrator import stage_classify_theme

        segs = [_classified_theme_seg('seg_001', primary=3)]
        result = stage_classify_theme(None, None, segments=segs, output_dir=self.tmpdir)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)


class TestStageClassifyPurerOverlay(unittest.TestCase):
    """stage_classify_purer must write purer overlay from pre-classified segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        segs_raw = [_make_segment('th_001', speaker='therapist')]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs_raw, 'testhash')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_purer_overlay(self):
        from process.orchestrator import stage_classify_purer

        segs = [_classified_purer_seg('th_001', primary=2)]
        stage_classify_purer(None, segments=segs, output_dir=self.tmpdir)

        overlay = classifications_io.overlay_path(self.tmpdir, 'purer')
        self.assertTrue(os.path.isfile(overlay))

    def test_purer_overlay_updates_manifest(self):
        from process.orchestrator import stage_classify_purer

        segs = [_classified_purer_seg('th_001', primary=0)]
        stage_classify_purer(None, segments=segs, output_dir=self.tmpdir)

        manifest = classifications_io.read_classification_manifest(self.tmpdir)
        self.assertIn('purer', manifest)

    def test_purer_stage_returns_segments(self):
        from process.orchestrator import stage_classify_purer

        segs = [_classified_purer_seg('th_001', primary=1)]
        result = stage_classify_purer(None, segments=segs, output_dir=self.tmpdir)
        self.assertIsInstance(result, list)


class TestStageClassifyCodebookOverlay(unittest.TestCase):
    """stage_classify_codebook must write codebook overlay from pre-classified segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        segs_raw = [_make_segment('seg_001')]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', segs_raw, 'testhash')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_codebook_overlay(self):
        from process.orchestrator import stage_classify_codebook

        seg = _make_segment('seg_001')
        seg.codebook_labels_ensemble = ['VE.1', 'VE.2']
        seg.codebook_labels_embedding = ['VE.1']
        seg.codebook_labels_llm = ['VE.1', 'VE.2']
        seg.codebook_disagreements = []
        seg.codebook_confidence = {'VE.1': 0.9}
        stage_classify_codebook(None, None, segments=[seg], output_dir=self.tmpdir)

        overlay = classifications_io.overlay_path(self.tmpdir, 'codebook')
        self.assertTrue(os.path.isfile(overlay))

    def test_codebook_stage_returns_segments(self):
        from process.orchestrator import stage_classify_codebook

        seg = _make_segment('seg_001')
        seg.codebook_labels_ensemble = ['VE.1']
        result = stage_classify_codebook(None, None, segments=[seg], output_dir=self.tmpdir)
        self.assertIsInstance(result, list)


class TestStageAssemble(unittest.TestCase):
    """stage_assemble must join segments + overlays and write master_segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        raw = [_classified_theme_seg('seg_001', 2), _classified_theme_seg('seg_002', 0)]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', raw, 'testhash')
        classifications_io.write_theme_overlay(self.tmpdir, raw)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_assemble_writes_master_segments_jsonl(self):
        from process.orchestrator import stage_assemble

        segs = [_classified_theme_seg('seg_001', 2), _classified_theme_seg('seg_002', 0)]
        stage_assemble(None, segments=segs, output_dir=self.tmpdir)

        from process import output_paths as _paths
        ms_dir = _paths.master_segments_dir(self.tmpdir)
        jsonl_files = [f for f in os.listdir(ms_dir) if f.startswith('master_segments')]
        self.assertTrue(len(jsonl_files) >= 1)

    def test_assemble_returns_dataframe(self):
        import pandas as pd
        from process.orchestrator import stage_assemble

        segs = [_classified_theme_seg('seg_001', 2)]
        result = stage_assemble(None, segments=segs, output_dir=self.tmpdir)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertGreater(len(result), 0)

    def test_assemble_standalone_loads_from_disk(self):
        """When segments=None, stage_assemble loads from frozen segments + overlays."""
        from process.orchestrator import stage_assemble

        # No segments passed — should load from disk
        result = stage_assemble(None, output_dir=self.tmpdir)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 2)


class TestStageCrossValidation(unittest.TestCase):
    """stage_cross_validation must write cv overlay and return segments."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        raw = [_classified_theme_seg('seg_001', 2)]
        segments_io.write_session_segments(self.tmpdir, 'c1s1', raw, 'testhash')

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_returns_segments(self):
        from process.orchestrator import stage_cross_validation

        segs = [_classified_theme_seg('seg_001', 2)]
        result = stage_cross_validation(None, None, segments=segs, output_dir=self.tmpdir)
        self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()
