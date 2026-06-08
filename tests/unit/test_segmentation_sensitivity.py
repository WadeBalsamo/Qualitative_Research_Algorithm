"""
tests/unit/test_segmentation_sensitivity.py
--------------------------------------------
Hermetic unit tests for the segmentation-sensitivity check (Feature 2).
No network, no model downloads, no Qwen — a FAKE segmenter (no embeddings) is
injected via `segmenter_factory`, and tiny VTT transcripts + a tiny frozen
master_segments.csv stand in for a real project.

Covers:
  - the result table assembles with ONE row per grid setting (canonical + sweeps)
  - every arm's H1 slope is finite (or cleanly None, never a crash)
  - label projection by token/temporal overlap reuses the frozen labels
  - boundary Jaccard + Spearman columns are present
  - both output artifacts (CSV + report .txt) are written
  - a verdict string is produced
"""

import os
import sys
import tempfile
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from analysis import segmentation_sensitivity as SS
from classification_tools.data_structures import Segment
from process import output_paths as _paths


# ── A fake segmenter: splits each session's participant sentences into N chunks
#    of `min_segment_words_conversational`-ish size, with deterministic time
#    bounds derived from the sentence timings. No embeddings, no network. The
#    chunk size depends on the perturbed config so different arms yield different
#    boundary placements (exercising the Jaccard/Spearman paths). ──────────────
class _FakeSegmenter:
    def __init__(self, seg_cfg: dict):
        self.min_words = int(seg_cfg.get('min_segment_words_conversational', 60))

    def segment_session(self, sentences, metadata):
        sid = metadata.get('session_id', 's')
        pid = metadata.get('participant_id', f'{sid}_p')
        # Only participant sentences feed VAAMR.
        part = [s for s in sentences if s.get('speaker', '').lower().startswith('participant')]
        segs = []
        idx = 0
        buf, buf_words = [], 0
        for s in part:
            buf.append(s)
            buf_words += len(str(s.get('text', '')).split())
            if buf_words >= self.min_words:
                segs.append(self._mk(buf, sid, pid, idx))
                idx += 1
                buf, buf_words = [], 0
        if buf:
            segs.append(self._mk(buf, sid, pid, idx))
        return segs

    @staticmethod
    def _mk(buf, sid, pid, idx):
        text = ' '.join(str(s.get('text', '')) for s in buf)
        start = int(float(buf[0].get('start', 0)) * 1000)
        end = int(float(buf[-1].get('end', 0)) * 1000)
        return Segment(
            segment_id=f'{sid}_part_{idx}', participant_id=pid, session_id=sid,
            session_number=int(sid[-1]) if sid[-1].isdigit() else 0,
            speaker='participant', text=text,
            start_time_ms=start, end_time_ms=max(end, start + 1),
            segment_index=idx, word_count=len(text.split()),
        )


def _fake_factory(seg_cfg):
    return _FakeSegmenter(seg_cfg)


# ── Tiny fixture project: two sessions, ~5 participant sentences each, with a
#    monotone upward stage signal so a positive slope is recoverable. ──────────
_PARTICIPANT_LINES = {
    # session -> list of (start_s, end_s, label, text)
    'p1s1': [
        (0, 10, 0, "The pain just grabs all my attention I cannot think of anything else."),
        (10, 20, 0, "It is constant and sharp and I keep checking whether it is getting worse."),
        (20, 30, 1, "So I avoid moving and I distract myself with the television instead."),
        (30, 40, 1, "I just try not to feel it I push it away whenever it shows up."),
    ],
    'p1s2': [
        (0, 10, 2, "This week I stayed with the breath and watched the sensation without running."),
        (10, 20, 3, "I noticed my mind labeling it as catastrophe and I saw that as just a thought."),
        (20, 30, 3, "I observed the worry arising and passing like clouds across the sky."),
        (30, 40, 4, "The ache became just a texture and it no longer meant my body was failing."),
    ],
}


def _write_vtt(path, lines):
    def _ts(sec):
        h = sec // 3600
        m = (sec % 3600) // 60
        s = sec % 60
        return f"{h:02d}:{m:02d}:{s:02d}.000"
    out = ["WEBVTT", ""]
    for i, (st, en, _lab, text) in enumerate(lines, 1):
        out.append(str(i))
        out.append(f"{_ts(st)} --> {_ts(en)}")
        out.append(f"participant_1: {text}")
        out.append("")
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(out))


def _build_fixture_project(root):
    """Create raw VTTs + a frozen master_segments.csv that load_segments reads."""
    tdir = os.path.join(root, 'input')
    os.makedirs(tdir, exist_ok=True)
    for sid, lines in _PARTICIPANT_LINES.items():
        _write_vtt(os.path.join(tdir, f'{sid}.vtt'), lines)

    # Frozen master_segments.csv: one row per participant line (the labels of record).
    rows = []
    for sid, lines in _PARTICIPANT_LINES.items():
        for i, (st, en, lab, text) in enumerate(lines):
            rows.append({
                'segment_id': f'{sid}_canon_{i}',
                'participant_id': f'{sid}_p',
                'session_id': sid,
                'session_number': int(sid[-1]),
                'segment_index': i,
                'speaker': 'participant',
                'text': text,
                'word_count': len(text.split()),
                'start_time_ms': st * 1000,
                'end_time_ms': en * 1000,
                'primary_stage': lab,
                'final_label': lab,
                'llm_confidence_primary': 0.9,
            })
    md = _paths.master_segments_dir(root)
    os.makedirs(md, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(md, 'master_segments.csv'), index=False)
    return tdir


def _make_config(root, tdir):
    from process.config import PipelineConfig
    cfg = PipelineConfig(transcript_dir=tdir, output_dir=root)
    cfg.segmentation.use_llm_refinement = False
    return cfg


_SMALL_GRID = {
    'semantic_shift_percentile': [25],
    'min_segment_words_conversational': [20, 40],   # 2 boundary regimes
    'max_segment_words_conversational': [500],
    'broad_window_size': [7],
    'use_adaptive_threshold': [True],
}


class TestSegmentationSensitivity(unittest.TestCase):
    def test_table_assembles_one_row_per_setting(self):
        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            table = res['table']
            # OFAT design: canonical + the non-canonical sweep values.
            # _SMALL_GRID adds one new arm (min_segment_words=20) beyond canonical(=40).
            self.assertFalse(table.empty)
            self.assertEqual(len(table), table['setting'].nunique())   # unique settings
            self.assertIn('canonical', set(table['setting']))
            self.assertGreaterEqual(len(table), 2)

    def test_slope_is_finite_for_each_arm(self):
        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            table = res['table']
            for _, r in table.iterrows():
                slope = r['h1_slope']
                # Either a finite float, or cleanly None — never NaN / a crash.
                if slope is not None:
                    self.assertEqual(slope, slope)            # not NaN
                    self.assertTrue(abs(float(slope)) < 1e6)  # finite-ish
            # With the planted monotone-up signal, the canonical arm should be > 0.
            canon = table[table['setting'] == 'canonical'].iloc[0]
            self.assertIsNotNone(canon['h1_slope'])
            self.assertGreater(canon['h1_slope'], 0.0)

    def test_projection_reuses_frozen_labels(self):
        # n_labeled should be > 0 (frozen labels projected onto new segments).
        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            self.assertTrue((res['table']['n_labeled'] > 0).all())

    def test_comparison_columns_present(self):
        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            for col in ('estage_spearman_vs_canon', 'boundary_jaccard_vs_canon',
                        'h1_ci_lo', 'h1_ci_hi', 'barrier_rate'):
                self.assertIn(col, res['table'].columns)
            # Canonical arm boundary-Jaccard vs itself is perfect (or near it).
            canon = res['table'][res['table']['setting'] == 'canonical'].iloc[0]
            self.assertIsNotNone(canon['boundary_jaccard_vs_canon'])

    def test_artifacts_written_and_verdict(self):
        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            csv_path = os.path.join(_paths.analysis_data_dir(root), 'segmentation_sensitivity.csv')
            rpt_path = os.path.join(_paths.reports_outcomes_dir(root), 'segmentation_sensitivity.txt')
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(rpt_path))
            self.assertIn(csv_path, res['files_written'])
            self.assertIn(rpt_path, res['files_written'])
            self.assertIsInstance(res['verdict'], str)
            self.assertTrue(len(res['verdict']) > 0)
            with open(rpt_path, encoding='utf-8') as f:
                txt = f.read()
            self.assertIn('SEGMENTATION-SENSITIVITY CHECK', txt)
            self.assertIn('VERDICT', txt)
            # LLM-refinement-disabled note must be stated.
            self.assertIn('refinement is DISABLED', txt)

    def test_missing_data_skips_cleanly(self):
        with tempfile.TemporaryDirectory() as root:
            # No master_segments.csv at all -> graceful skip, no crash.
            tdir = os.path.join(root, 'input')
            os.makedirs(tdir, exist_ok=True)
            from process.config import PipelineConfig
            cfg = PipelineConfig(transcript_dir=tdir, output_dir=root)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            self.assertTrue(res['table'].empty)
            self.assertIn('skipped', res['verdict'])


# ── F2-A: the instrument must perturb the CONTINUOUS progression coordinate (the
#    superposition mixture — the real H1 statistic), NOT just the hard label. ─────
class TestContinuousCoordinateProjection(unittest.TestCase):
    def _seg(self, sid, idx, start_ms, end_ms, text):
        return Segment(segment_id=f'{sid}_part_{idx}', participant_id=f'{sid}_p',
                       session_id=sid, speaker='participant', text=text,
                       start_time_ms=start_ms, end_time_ms=end_ms, segment_index=idx)

    def test_project_coord_is_overlap_weighted_mean_not_argmax(self):
        # Two canonical units, hard labels 0 and 4 but continuous coords 0.10 and 3.90.
        # A new segment overlapping BOTH (equal token overlap) must get the MEAN coord
        # (~2.0) — a continuous value that is NOT either hard label and NOT their argmax.
        canon = SS._canon_units(pd.DataFrame([
            {'final_label': 0, 'start_time_ms': 0,    'end_time_ms': 1000,
             'text': 'alpha beta', 'progression_coord': 0.10},
            {'final_label': 4, 'start_time_ms': 1000, 'end_time_ms': 2000,
             'text': 'alpha beta', 'progression_coord': 3.90},
        ]), coord_col='progression_coord')
        new = [self._seg('s', 0, 0, 2000, 'alpha beta')]
        labels, coords = SS._project_labels_and_coords(new, canon)
        self.assertEqual(len(coords), 1)
        self.assertIsNotNone(coords[0])
        self.assertAlmostEqual(coords[0], 2.0, places=3)   # weighted MEAN, not a hard label
        self.assertIn(labels[0], (0, 4))                   # hard label is still an argmax

    def test_coord_used_flag_and_coverage_reported(self):
        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_fake_factory, verbose=False,
            )
            # The continuous coordinate must be used (not the hard-label proxy)...
            self.assertTrue(res.get('coord_used'))
            # ...and every arm reports its projected-coordinate coverage.
            self.assertIn('coord_coverage', res['table'].columns)
            self.assertTrue(res['table']['coord_coverage'].notna().all())
            self.assertTrue((res['table']['coord_coverage'] > 0).all())

    def test_slope_tracks_coordinate_not_hard_label(self):
        # When the projected continuous coordinate DIVERGES from the hard labels, the
        # reported H1 slope must follow the COORDINATE. We build a frame whose hard
        # labels are flat (all 2) but whose progression_coord trends DOWN over sessions;
        # a hard-label-only instrument would report ~0 slope, the correct (coord) one
        # reports a clearly negative slope.
        ps_rows = []
        for pnum in range(6):
            pid = f'p{pnum}'
            for sn, coord in ((1, 3.0), (2, 2.0), (3, 1.0)):
                ps_rows.append({'participant_id': pid, 'session_id': f'{pid}s{sn}',
                                'session_number': sn, 'final_label': 2,
                                'progression_coord': coord,
                                'start_time_ms': 0, 'end_time_ms': 1000})
        ps_df = pd.DataFrame(ps_rows)
        arm = SS._evaluate_arm(ps_df, None)
        self.assertIsNotNone(arm['h1_slope'])
        self.assertLess(arm['h1_slope'], -0.5)             # follows the DOWN coordinate
        self.assertEqual(arm['coord_coverage'], 1.0)
        # Sanity: stripping the coordinate (hard-label fallback) flattens the slope.
        flat = SS._evaluate_arm(ps_df.drop(columns=['progression_coord']), None)
        self.assertTrue(flat['h1_slope'] is None or abs(flat['h1_slope']) < 0.2)


# ── F2-D: arms that silently fail to segment must be COUNTED and surfaced. ───────
class TestFailedArmAccounting(unittest.TestCase):
    def test_failed_arm_counted_in_table_and_verdict(self):
        # A factory whose segmenter raises for the 'min_segment_words=20' arm only.
        class _FlakySeg(_FakeSegmenter):
            def segment_session(self, sentences, metadata):
                if self.min_words == 20:
                    raise RuntimeError("planted segmentation failure")
                return super().segment_session(sentences, metadata)

        def _flaky_factory(seg_cfg):
            return _FlakySeg(seg_cfg)

        with tempfile.TemporaryDirectory() as root:
            tdir = _build_fixture_project(root)
            cfg = _make_config(root, tdir)
            res = SS.run_segmentation_sensitivity(
                root, cfg, grid=_SMALL_GRID,
                segmenter_factory=_flaky_factory, verbose=False,
            )
            table = res['table']
            # The failed arm is still a ROW (counted), with no segments / no slope.
            failed = table[table['setting'] == 'min_segment_words_conversational=20']
            self.assertEqual(len(failed), 1)
            self.assertEqual(int(failed.iloc[0]['n_segments']), 0)
            self.assertTrue(pd.isna(failed.iloc[0]['h1_slope'])
                            or failed.iloc[0]['h1_slope'] is None)
            # K/N accounting is reported and K < N (one arm produced no slope).
            self.assertLess(res['n_arms_with_slope'], res['n_arms_total'])
            self.assertIn('arms', res['verdict'])
            # The report text names the K/N split.
            rpt = os.path.join(_paths.reports_outcomes_dir(root),
                               'segmentation_sensitivity.txt')
            with open(rpt, encoding='utf-8') as f:
                txt = f.read()
            self.assertIn('produced a slope', txt)


if __name__ == '__main__':
    unittest.main()
