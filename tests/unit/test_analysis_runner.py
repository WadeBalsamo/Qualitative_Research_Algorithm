"""
tests/unit/test_analysis_runner.py
-----------------------------------
Unit tests for analysis/runner.py — run_analysis() post-hoc Stage 8 entry.

Approach:
- Build a minimal on-disk pipeline output directory containing:
    02_meta/training_data/master_segments.csv  (via make_master_df)
    02_meta/theme_definitions.json             (minimal VAAMR framework)
- Monkeypatch the GNN entry-point and all LLM-summary calls so nothing
  touches the network or a model.
- Assert run_analysis() returns the expected dict shape, populates
  files_generated, creates 06_reports subdirs, and writes
  00_executive_summary.txt.
- Cover force_gnn=True (GNN stub called) / force_gnn=False (skipped) / None.
"""

import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df
from process.output_paths import (
    master_segments_dir,
    meta_dir,
    reports_results_path,
    reports_methods_path,
)


def _make_output_dir(tmp: str) -> str:
    """Create a minimal pipeline output dir with CSV and theme_definitions.json."""
    from process import output_paths as _paths

    # Write master_segments.csv
    seg_dir = master_segments_dir(tmp)
    os.makedirs(seg_dir, exist_ok=True)
    df = make_master_df(n_sessions=2, n_participants=2)
    # Rename confidence_tier → label_confidence_tier (matches loader expectation)
    df = df.rename(columns={'confidence_tier': 'label_confidence_tier'})
    df.to_csv(os.path.join(seg_dir, 'master_segments.csv'), index=False)

    # Write theme_definitions.json
    themes = [
        {'theme_id': i, 'key': f'stage_{i}', 'name': f'Stage {i}',
         'short_name': f'S{i}', 'definition': f'Definition {i}'}
        for i in range(5)
    ]
    meta = meta_dir(tmp)
    os.makedirs(meta, exist_ok=True)
    with open(os.path.join(meta, 'theme_definitions.json'), 'w') as f:
        json.dump({'themes': themes}, f)

    return tmp


# Stub for GNN — records whether it was called
_GNN_CALLS = []


def _fake_gnn(df_all, output_dir, framework, config, llm_client=None):
    _GNN_CALLS.append({'output_dir': output_dir})
    return {'status': 'ok', 'files_written': []}


class TestRunAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_output_dir(self.tmp)
        _GNN_CALLS.clear()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _patches(self):
        """Return list of patches applied for every test: GNN + figures + summaries."""
        return [
            patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn),
            # matplotlib / figure generation — patched to no-ops
            patch('analysis.figures.generate_all_figures', return_value=[]),
            patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]),
        ]

    # ------------------------------------------------------------------ #
    # Basic return-value contract
    # ------------------------------------------------------------------ #
    def test_returns_expected_keys(self):
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            result = run_analysis(self.tmp, verbose=False, force_gnn=False)

        self.assertIn('output_dir', result)
        self.assertIn('n_segments', result)
        self.assertIn('n_participants', result)
        self.assertIn('n_sessions', result)
        self.assertIn('files_generated', result)
        self.assertEqual(result['output_dir'], self.tmp)

    def test_n_segments_participants_sessions(self):
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            result = run_analysis(self.tmp, verbose=False, force_gnn=False)

        # make_master_df(n_sessions=2, n_participants=2) → 4 sessions × 4 participant segs
        # Each participant×session combo has 4 participant rows
        self.assertGreater(result['n_segments'], 0)
        self.assertEqual(result['n_participants'], 2)
        self.assertGreater(result['n_sessions'], 0)

    def test_files_generated_is_list(self):
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            result = run_analysis(self.tmp, verbose=False, force_gnn=False)

        self.assertIsInstance(result['files_generated'], list)

    # ------------------------------------------------------------------ #
    # Top-level artifacts written to disk
    # ------------------------------------------------------------------ #
    def test_results_brief_written(self):
        """Runner step 13 should write 00_RESULTS.txt (top-level synthesis)."""
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)

        rb_path = reports_results_path(self.tmp)
        self.assertTrue(os.path.isfile(rb_path),
                        f"Expected results brief at {rb_path}")

    def test_session_json_files_written(self):
        from process.output_paths import sessions_json_dir
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)

        sess_dir = sessions_json_dir(self.tmp)
        self.assertTrue(os.path.isdir(sess_dir))
        jsons = [f for f in os.listdir(sess_dir) if f.endswith('.json')]
        self.assertGreater(len(jsons), 0, "Expected at least one session JSON")

    def test_participant_json_files_written(self):
        from process.output_paths import participants_json_dir
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)

        par_dir = participants_json_dir(self.tmp)
        self.assertTrue(os.path.isdir(par_dir))
        jsons = [f for f in os.listdir(par_dir) if f.endswith('.json')]
        self.assertGreater(len(jsons), 0, "Expected at least one participant JSON")

    def test_longitudinal_summary_written(self):
        from process.output_paths import analysis_data_dir
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)

        lpath = os.path.join(analysis_data_dir(self.tmp), 'longitudinal_summary.json')
        self.assertTrue(os.path.isfile(lpath), f"Expected longitudinal_summary.json at {lpath}")

    def test_session_progression_csv_written(self):
        from process.output_paths import longitudinal_dir
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)

        csv = os.path.join(longitudinal_dir(self.tmp), 'session_stage_progression.csv')
        self.assertTrue(os.path.isfile(csv), f"Expected progression CSV at {csv}")

    def test_methods_report_written(self):
        """Runner step 13 should write 08_methods.txt ([M#] registry)."""
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)

        self.assertTrue(os.path.isfile(reports_methods_path(self.tmp)))

    # ------------------------------------------------------------------ #
    # force_gnn routing
    # ------------------------------------------------------------------ #
    def test_force_gnn_false_skips_gnn(self):
        _GNN_CALLS.clear()
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn) as mock_gnn, \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=False)
            mock_gnn.assert_not_called()

    def test_force_gnn_none_skips_gnn_when_no_config(self):
        """Without a qra_config.json, gnn_enabled defaults to False."""
        _GNN_CALLS.clear()
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn) as mock_gnn, \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=None)
            mock_gnn.assert_not_called()

    def test_force_gnn_true_calls_gnn(self):
        """force_gnn=True should invoke run_gnn_analysis even with no pipeline config."""
        _GNN_CALLS.clear()
        with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn) as mock_gnn, \
             patch('analysis.figures.generate_all_figures', return_value=[]), \
             patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
            from analysis.runner import run_analysis
            run_analysis(self.tmp, verbose=False, force_gnn=True)
            mock_gnn.assert_called_once()

    # ------------------------------------------------------------------ #
    # Missing CSV → graceful empty dict
    # ------------------------------------------------------------------ #
    def test_missing_csv_returns_empty_result(self):
        """If the CSV is missing, run_analysis should return a zero-count dict."""
        empty_tmp = tempfile.mkdtemp()
        try:
            # Write only theme_definitions.json, no CSV
            meta = meta_dir(empty_tmp)
            os.makedirs(meta, exist_ok=True)
            themes = [{'theme_id': 0, 'key': 's0', 'name': 'Stage 0',
                       'short_name': 'S0', 'definition': ''}]
            with open(os.path.join(meta, 'theme_definitions.json'), 'w') as f:
                json.dump({'themes': themes}, f)

            with patch('gnn_layer.runner.run_gnn_analysis', side_effect=_fake_gnn), \
                 patch('analysis.figures.generate_all_figures', return_value=[]), \
                 patch('analysis.figures.generate_all_session_stage_timelines', return_value=[]):
                from analysis.runner import run_analysis
                result = run_analysis(empty_tmp, verbose=False, force_gnn=False)

            self.assertEqual(result['n_segments'], 0)
        finally:
            shutil.rmtree(empty_tmp, ignore_errors=True)


if __name__ == '__main__':
    unittest.main()
