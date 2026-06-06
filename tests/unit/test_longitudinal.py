"""
tests/unit/test_longitudinal.py
---------------------------------
Unit tests for analysis/longitudinal.py.

Covers:
- compute_group_trajectories: per-session stage-mean columns, equal-weighting
- generate_longitudinal_summary: keys, heatmap, feasibility thresholds,
  validity_indicators, JSON file written to disk
- Edge: single-session input
- Edge: empty participant_reports
- _classify_feasibility thresholds
- _is_non_decreasing helper
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import make_master_df


def _part_df(n_sessions=3, n_participants=2):
    """Return participant-only rows from make_master_df with expected columns."""
    df = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    df = df[df['speaker'] == 'participant'].copy()
    df = df.rename(columns={'confidence_tier': 'label_confidence_tier'})
    df['final_label'] = df['final_label'].astype(int)
    # make_master_df embeds the participant index in session_id (c1p2s3); real
    # session_ids are shared across participants (c1s3). Normalize so per-session
    # aggregation (e.g. n_participants) reflects real session semantics.
    df['session_id'] = df['session_id'].str.replace(r'p\d+', '', regex=True)
    return df


def _mini_framework():
    return {
        0: {'id': 0, 'key': 's0', 'name': 'Vigilance', 'short_name': 'S0', 'definition': ''},
        1: {'id': 1, 'key': 's1', 'name': 'Avoidance', 'short_name': 'S1', 'definition': ''},
        2: {'id': 2, 'key': 's2', 'name': 'AR', 'short_name': 'S2', 'definition': ''},
        3: {'id': 3, 'key': 's3', 'name': 'Meta', 'short_name': 'S3', 'definition': ''},
        4: {'id': 4, 'key': 's4', 'name': 'Reap', 'short_name': 'S4', 'definition': ''},
    }


class TestClassifyFeasibility(unittest.TestCase):
    def test_high(self):
        from analysis.longitudinal import _classify_feasibility
        self.assertEqual(_classify_feasibility(0.60), 'high')
        self.assertEqual(_classify_feasibility(0.95), 'high')

    def test_moderate(self):
        from analysis.longitudinal import _classify_feasibility
        self.assertEqual(_classify_feasibility(0.35), 'moderate')
        self.assertEqual(_classify_feasibility(0.59), 'moderate')

    def test_low(self):
        from analysis.longitudinal import _classify_feasibility
        self.assertEqual(_classify_feasibility(0.00), 'low')
        self.assertEqual(_classify_feasibility(0.34), 'low')


class TestIsNonDecreasing(unittest.TestCase):
    def test_strictly_increasing(self):
        from analysis.longitudinal import _is_non_decreasing
        self.assertTrue(_is_non_decreasing([1, 2, 3]))

    def test_flat(self):
        from analysis.longitudinal import _is_non_decreasing
        self.assertTrue(_is_non_decreasing([2, 2, 2]))

    def test_decreasing(self):
        from analysis.longitudinal import _is_non_decreasing
        self.assertFalse(_is_non_decreasing([3, 2, 1]))

    def test_single(self):
        from analysis.longitudinal import _is_non_decreasing
        self.assertTrue(_is_non_decreasing([5]))


class TestComputeGroupTrajectories(unittest.TestCase):
    def test_returns_dataframe_with_stage_columns(self):
        from analysis.longitudinal import compute_group_trajectories
        df = _part_df(n_sessions=2, n_participants=2)
        fw = _mini_framework()
        result = compute_group_trajectories(df, fw)
        self.assertIsInstance(result, pd.DataFrame)
        for st in range(5):
            self.assertIn(f'stage_{st}_mean', result.columns)

    def test_n_participants_column_correct(self):
        from analysis.longitudinal import compute_group_trajectories
        df = _part_df(n_sessions=2, n_participants=2)
        fw = _mini_framework()
        result = compute_group_trajectories(df, fw)
        # Each session should have 2 participants
        self.assertTrue((result['n_participants'] == 2).all())

    def test_proportions_sum_to_one(self):
        from analysis.longitudinal import compute_group_trajectories
        df = _part_df(n_sessions=2, n_participants=1)
        fw = _mini_framework()
        result = compute_group_trajectories(df, fw)
        stage_cols = [f'stage_{st}_mean' for st in range(5)]
        for _, row in result.iterrows():
            total = sum(row[c] for c in stage_cols)
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg=f"Stage props don't sum to 1: {total}")

    def test_sorted_by_session_order(self):
        from analysis.longitudinal import compute_group_trajectories
        df = _part_df(n_sessions=3, n_participants=1)
        fw = _mini_framework()
        result = compute_group_trajectories(df, fw)
        # session_ids should be in canonical longitudinal order
        from analysis.loader import sort_session_ids
        expected_order = sort_session_ids(result['session_id'].tolist())
        self.assertEqual(list(result['session_id']), expected_order)

    def test_include_secondary_columns(self):
        from analysis.longitudinal import compute_group_trajectories
        df = _part_df(n_sessions=2, n_participants=1)
        fw = _mini_framework()
        result = compute_group_trajectories(df, fw, include_secondary=True)
        self.assertIn('dual_coded_mean', result.columns)

    def test_empty_df_returns_empty(self):
        from analysis.longitudinal import compute_group_trajectories
        fw = _mini_framework()
        empty = pd.DataFrame(columns=['participant_id', 'session_id', 'session_number',
                                      'cohort_id', 'final_label'])
        result = compute_group_trajectories(empty, fw)
        self.assertTrue(result.empty)


class TestGenerateLongitudinalSummary(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()
        # Minimal participant_reports stub
        self.participant_reports = [
            {
                'participant_id': 'P01',
                'progression_score_by_session': {'1': 0.5, '2': 0.8, '3': 1.1},
                'progression_trend': 0.3,
            },
            {
                'participant_id': 'P02',
                'progression_score_by_session': {'1': 0.4, '2': 0.6, '3': 0.9},
                'progression_trend': 0.25,
            },
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_with_expected_keys(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        for key in ('n_participants', 'n_sessions_total', 'session_ids_ordered',
                    'heatmap_data', 'group_progression', 'feasibility_assessment',
                    'validity_indicators', 'narrative_summary'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_n_participants_correct(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        self.assertEqual(result['n_participants'], 2)

    def test_heatmap_data_shape(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        hd = result['heatmap_data']
        self.assertIn('participant_ids', hd)
        self.assertIn('session_ids', hd)
        self.assertIn('dominant_stage_matrix', hd)
        self.assertEqual(len(hd['dominant_stage_matrix']), len(hd['participant_ids']))

    def test_feasibility_assessment_keys(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        fa = result['feasibility_assessment']
        self.assertIn('feasibility_rating', fa)
        self.assertIn('pct_high_confidence', fa)
        self.assertIn('high_plus_medium_pct', fa)

    def test_validity_indicators_keys(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        vi = result['validity_indicators']
        self.assertIn('expected_progression_observed', vi)
        self.assertIn('stage_ordering_consistent', vi)
        self.assertIn('validity_narrative', vi)

    def test_json_file_written(self):
        from analysis.longitudinal import generate_longitudinal_summary
        from process.output_paths import analysis_data_dir
        generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        path = os.path.join(analysis_data_dir(self.tmp), 'longitudinal_summary.json')
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        self.assertIn('n_participants', data)

    def test_narrative_is_string(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        self.assertIsInstance(result['narrative_summary'], str)
        self.assertGreater(len(result['narrative_summary']), 10)

    def test_advancing_participants_counted(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        gp = result['group_progression']
        # Both stubs have positive trend > 0.1, so n_advancing should be 2
        self.assertEqual(gp['n_advancing'], 2)
        self.assertEqual(gp['n_regressing'], 0)

    def test_empty_participant_reports(self):
        from analysis.longitudinal import generate_longitudinal_summary
        result = generate_longitudinal_summary(
            self.df, [], self.fw, self.tmp
        )
        gp = result['group_progression']
        self.assertEqual(gp['mean_progression_trend_overall'], 0.0)

    def test_single_session_edge(self):
        """Single-session input: stage_ordering_consistent defaults to True."""
        from analysis.longitudinal import generate_longitudinal_summary
        df_one = _part_df(n_sessions=1, n_participants=1)
        reports_one = [
            {'participant_id': 'P01',
             'progression_score_by_session': {'1': 0.5},
             'progression_trend': 0.0}
        ]
        result = generate_longitudinal_summary(df_one, reports_one, self.fw, self.tmp)
        vi = result['validity_indicators']
        self.assertTrue(vi['stage_ordering_consistent'])


if __name__ == '__main__':
    unittest.main()
