"""
tests/unit/test_figure_data.py
-------------------------------
Unit tests for analysis/figure_data.py — graph-ready CSV export functions.

Uses make_master_df to supply a small participant-only DataFrame and verifies:
  - Returned structures have the expected columns
  - CSV files land at the expected paths under output_dir/03_analysis_data/graphing/
  - Numeric invariants (proportions sum to 1, etc.)
  - Edge cases: empty df, no codebook labels
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import make_master_df
from analysis.figure_data import (
    export_theme_proportions_by_participant_session,
    export_group_theme_trajectories,
    export_combined_cohort_group_trajectories,
    export_codebook_prevalence_by_participant_session,
    export_confidence_distribution_by_stage,
    export_participant_stage_dominant,
    export_segment_superposition,
    export_all_graphing_datasets,
)
from process import output_paths as _paths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

_FRAMEWORK = {
    0: {'id': 0, 'key': 'vigilance', 'name': 'Vigilance', 'short_name': 'VIG'},
    1: {'id': 1, 'key': 'avoidance', 'name': 'Avoidance', 'short_name': 'AVD'},
    2: {'id': 2, 'key': 'attention_regulation', 'name': 'Attention Regulation', 'short_name': 'ATT'},
    3: {'id': 3, 'key': 'metacognition', 'name': 'Metacognition', 'short_name': 'MET'},
    4: {'id': 4, 'key': 'reappraisal', 'name': 'Reappraisal', 'short_name': 'REA'},
}


def _participant_df(n_sessions=2, n_participants=2):
    """Return participant-only rows with columns figure_data expects."""
    df = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    df = df[df['speaker'] == 'participant'].copy().reset_index(drop=True)
    # figure_data needs label_confidence_tier; make_master_df has confidence_tier
    if 'label_confidence_tier' not in df.columns:
        df['label_confidence_tier'] = df.get('confidence_tier', 'high')
    # ensure final_label is int
    df['final_label'] = df['final_label'].fillna(0).astype(int)
    return df


class TestExportThemeProportionsByParticipantSession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        result = export_theme_proportions_by_participant_session(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = export_theme_proportions_by_participant_session(self.df, _FRAMEWORK, self.tmp)
        expected = ['participant_id', 'session_id', 'session_number', 'n_segments']
        for col in expected:
            self.assertIn(col, result.columns)
        for st in range(5):
            self.assertIn(f'stage_{st}_pct', result.columns)

    def test_stage_proportions_sum_to_one(self):
        result = export_theme_proportions_by_participant_session(self.df, _FRAMEWORK, self.tmp)
        stage_cols = [f'stage_{st}_pct' for st in range(5)]
        for _, row in result.iterrows():
            total = sum(row[c] for c in stage_cols)
            self.assertAlmostEqual(total, 1.0, places=3,
                                   msg=f"Row sum {total} for {row['participant_id']}")

    def test_csv_written(self):
        export_theme_proportions_by_participant_session(self.df, _FRAMEWORK, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp),
                            'theme_proportions_by_participant_session.csv')
        self.assertTrue(os.path.isfile(path))
        self.assertGreater(os.path.getsize(path), 0)

    def test_row_count_matches_participant_sessions(self):
        result = export_theme_proportions_by_participant_session(self.df, _FRAMEWORK, self.tmp)
        expected_rows = self.df.groupby(['participant_id', 'session_id']).ngroups
        self.assertEqual(len(result), expected_rows)


class TestExportGroupThemeTrajectories(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df(n_sessions=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        result = export_group_theme_trajectories(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = export_group_theme_trajectories(self.df, _FRAMEWORK, self.tmp)
        self.assertIn('session_id', result.columns)
        self.assertIn('n_participants', result.columns)
        for st in range(5):
            self.assertIn(f'stage_{st}_mean', result.columns)

    def test_csv_written(self):
        export_group_theme_trajectories(self.df, _FRAMEWORK, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp), 'group_theme_trajectories.csv')
        self.assertTrue(os.path.isfile(path))

    def test_stage_means_sum_to_one(self):
        result = export_group_theme_trajectories(self.df, _FRAMEWORK, self.tmp)
        stage_cols = [f'stage_{st}_mean' for st in range(5)]
        for _, row in result.iterrows():
            total = sum(row[c] for c in stage_cols)
            self.assertAlmostEqual(total, 1.0, places=3)


class TestExportCombinedCohortGroupTrajectories(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df(n_sessions=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        result = export_combined_cohort_group_trajectories(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_session_number_column(self):
        result = export_combined_cohort_group_trajectories(self.df, _FRAMEWORK, self.tmp)
        self.assertIn('session_number', result.columns)

    def test_csv_written(self):
        export_combined_cohort_group_trajectories(self.df, _FRAMEWORK, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp),
                            'combined_cohort_group_trajectories.csv')
        self.assertTrue(os.path.isfile(path))


class TestExportCodebookPrevalence(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe_when_labels_present(self):
        result = export_codebook_prevalence_by_participant_session(self.df, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns_when_data_present(self):
        result = export_codebook_prevalence_by_participant_session(self.df, self.tmp)
        if not result.empty:
            for col in ('participant_id', 'session_id', 'code_id', 'count', 'prevalence'):
                self.assertIn(col, result.columns)

    def test_csv_written(self):
        export_codebook_prevalence_by_participant_session(self.df, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp),
                            'codebook_prevalence_by_participant_session.csv')
        self.assertTrue(os.path.isfile(path))

    def test_empty_without_column(self):
        df_no_codes = self.df.drop(columns=['codebook_labels_ensemble'], errors='ignore')
        result = export_codebook_prevalence_by_participant_session(df_no_codes, self.tmp)
        self.assertTrue(result.empty)


class TestExportConfidenceDistributionByStage(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        result = export_confidence_distribution_by_stage(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = export_confidence_distribution_by_stage(self.df, _FRAMEWORK, self.tmp)
        for col in ('stage_id', 'stage_name', 'label_confidence_tier', 'count', 'proportion'):
            self.assertIn(col, result.columns)

    def test_rows_per_stage(self):
        """3 tiers × 5 stages = 15 rows."""
        result = export_confidence_distribution_by_stage(self.df, _FRAMEWORK, self.tmp)
        self.assertEqual(len(result), 15)

    def test_csv_written(self):
        export_confidence_distribution_by_stage(self.df, _FRAMEWORK, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp),
                            'confidence_distribution_by_stage.csv')
        self.assertTrue(os.path.isfile(path))


class TestExportParticipantStageDominant(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        result = export_participant_stage_dominant(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns(self):
        result = export_participant_stage_dominant(self.df, _FRAMEWORK, self.tmp)
        for col in ('participant_id', 'session_id', 'dominant_stage', 'dominant_stage_name'):
            self.assertIn(col, result.columns)

    def test_dominant_stage_in_range(self):
        result = export_participant_stage_dominant(self.df, _FRAMEWORK, self.tmp)
        for v in result['dominant_stage'].dropna():
            self.assertIn(int(v), range(5))

    def test_csv_written(self):
        export_participant_stage_dominant(self.df, _FRAMEWORK, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp), 'participant_stage_dominant.csv')
        self.assertTrue(os.path.isfile(path))


class TestExportSegmentSuperposition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_without_mixture_column(self):
        result = export_segment_superposition(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(result.empty)

    def test_with_mixture_column(self):
        df = self.df.copy()
        df['mixture'] = [[0.2, 0.2, 0.2, 0.2, 0.2]] * len(df)
        df['progression_coord'] = 2.0
        df['mixture_entropy'] = 1.0
        df['max_stage'] = 0
        df['second_stage'] = 1
        df['n_active_stages'] = 5
        df['is_liminal'] = True
        df['mixture_source'] = 'secondary'
        result = export_segment_superposition(df, _FRAMEWORK, self.tmp)
        self.assertFalse(result.empty)
        self.assertIn('mix_0', result.columns)
        self.assertIn('mix_4', result.columns)

    def test_csv_written_when_mixture_present(self):
        df = self.df.copy()
        df['mixture'] = [[0.2, 0.2, 0.2, 0.2, 0.2]] * len(df)
        df['progression_coord'] = 2.0
        df['mixture_entropy'] = 1.0
        df['max_stage'] = 0
        df['second_stage'] = 1
        df['n_active_stages'] = 5
        df['is_liminal'] = False
        df['mixture_source'] = 'secondary'
        export_segment_superposition(df, _FRAMEWORK, self.tmp)
        path = os.path.join(_paths.graphing_dir(self.tmp), 'segment_superposition.csv')
        self.assertTrue(os.path.isfile(path))


class TestExportAllGraphingDatasets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df(n_sessions=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_list_of_paths(self):
        paths = export_all_graphing_datasets(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)

    def test_all_returned_paths_exist(self):
        paths = export_all_graphing_datasets(self.df, _FRAMEWORK, self.tmp)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f"Missing: {p}")

    def test_graphing_dir_created(self):
        export_all_graphing_datasets(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(os.path.isdir(_paths.graphing_dir(self.tmp)))


if __name__ == '__main__':
    unittest.main()
