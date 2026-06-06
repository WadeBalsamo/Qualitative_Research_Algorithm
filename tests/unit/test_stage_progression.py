"""
tests/unit/test_stage_progression.py
--------------------------------------
Unit tests for analysis/stage_progression.py.

Covers:
- compute_session_stage_progression: forward/backward/lateral counts,
  dominant_stage, max_stage_reached, CSV written, column shape
- compute_cross_session_transitions: matrix shape, participant_sequences
- compute_state_transition_matrix: aggregate matrix shape, counts
- Edge: single segment per session (no transitions → all zeros)
- Edge: empty DataFrame → empty result DataFrame
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df


def _mini_framework():
    return {
        i: {'id': i, 'key': f's{i}', 'name': f'Stage {i}', 'short_name': f'S{i}', 'definition': ''}
        for i in range(5)
    }


def _part_df(n_sessions=3, n_participants=2):
    df = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    df = df[df['speaker'] == 'participant'].copy()
    df = df.rename(columns={'confidence_tier': 'label_confidence_tier'})
    df['final_label'] = df['final_label'].astype(int)
    return df


class TestComputeSessionStageProgression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)

    def test_expected_columns_present(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        for col in ('participant_id', 'session_id', 'session_number', 'n_segments',
                    'forward_transitions', 'backward_transitions', 'lateral_transitions',
                    'max_stage_reached', 'dominant_stage', 'stage_distribution'):
            self.assertIn(col, result.columns, f"Missing column: {col}")

    def test_one_row_per_participant_session(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        n_expected = self.df.groupby(['participant_id', 'session_id']).ngroups
        self.assertEqual(len(result), n_expected)

    def test_forward_plus_backward_plus_lateral_equals_total_transitions(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        for _, row in result.iterrows():
            fwd = row['forward_transitions']
            bwd = row['backward_transitions']
            lat = row['lateral_transitions']
            n_segs = row['n_segments']
            # n transitions = n_segments - 1
            total_trans = n_segs - 1
            self.assertEqual(fwd + bwd + lat, total_trans,
                             f"Transitions don't add up for {row['participant_id']} {row['session_id']}: "
                             f"fwd={fwd}, bwd={bwd}, lat={lat}, expected={total_trans}")

    def test_forward_backward_values_non_negative(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        self.assertTrue((result['forward_transitions'] >= 0).all())
        self.assertTrue((result['backward_transitions'] >= 0).all())
        self.assertTrue((result['lateral_transitions'] >= 0).all())

    def test_max_stage_reached_is_valid_stage(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        valid_stages = set(self.fw.keys())
        for v in result['max_stage_reached']:
            self.assertIn(v, valid_stages)

    def test_dominant_stage_is_valid_stage(self):
        from analysis.stage_progression import compute_session_stage_progression
        result = compute_session_stage_progression(self.df, self.fw, self.tmp)
        valid_stages = set(self.fw.keys())
        for v in result['dominant_stage']:
            self.assertIn(v, valid_stages)

    def test_csv_written(self):
        from analysis.stage_progression import compute_session_stage_progression
        from process.output_paths import longitudinal_dir
        compute_session_stage_progression(self.df, self.fw, self.tmp)
        path = os.path.join(longitudinal_dir(self.tmp), 'session_stage_progression.csv')
        self.assertTrue(os.path.isfile(path))

    def test_csv_parseable(self):
        from analysis.stage_progression import compute_session_stage_progression
        from process.output_paths import longitudinal_dir
        compute_session_stage_progression(self.df, self.fw, self.tmp)
        path = os.path.join(longitudinal_dir(self.tmp), 'session_stage_progression.csv')
        loaded = pd.read_csv(path)
        self.assertGreater(len(loaded), 0)

    def test_single_segment_session_has_zero_transitions(self):
        """Single-segment session → 0 forward, 0 backward, 0 lateral."""
        from analysis.stage_progression import compute_session_stage_progression
        # Build a df with just one row per participant×session
        df_one = self.df.groupby(['participant_id', 'session_id']).head(1).copy()
        result = compute_session_stage_progression(df_one, self.fw, self.tmp)
        self.assertTrue((result['forward_transitions'] == 0).all())
        self.assertTrue((result['backward_transitions'] == 0).all())
        self.assertTrue((result['lateral_transitions'] == 0).all())

    def test_empty_df_returns_empty_dataframe(self):
        from analysis.stage_progression import compute_session_stage_progression
        empty = pd.DataFrame(columns=['participant_id', 'session_id', 'session_number',
                                      'cohort_id', 'segment_index', 'final_label',
                                      'secondary_stage'])
        result = compute_session_stage_progression(empty, self.fw, self.tmp)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 0)

    def test_known_sequence_forward_backward(self):
        """Manually construct a session with a known stage sequence."""
        from analysis.stage_progression import compute_session_stage_progression
        rows = []
        # Stage sequence: 0 → 1 → 2 → 1 = 2 forward, 1 backward, 0 lateral
        for idx, stage in enumerate([0, 1, 2, 1]):
            rows.append({
                'participant_id': 'P99',
                'session_id': 'c1p99s1',
                'session_number': 1,
                'cohort_id': 1,
                'segment_index': idx,
                'final_label': stage,
            })
        df_test = pd.DataFrame(rows)
        result = compute_session_stage_progression(df_test, self.fw, self.tmp)
        self.assertEqual(len(result), 1)
        row = result.iloc[0]
        self.assertEqual(row['forward_transitions'], 2)
        self.assertEqual(row['backward_transitions'], 1)
        self.assertEqual(row['lateral_transitions'], 0)
        self.assertEqual(row['max_stage_reached'], 2)


class TestComputeCrossSessionTransitions(unittest.TestCase):
    def setUp(self):
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()

    def test_returns_matrix_and_sequences(self):
        from analysis.stage_progression import compute_cross_session_transitions
        matrix, seqs = compute_cross_session_transitions(self.df, self.fw)
        self.assertIsInstance(matrix, pd.DataFrame)
        self.assertIsInstance(seqs, dict)

    def test_matrix_shape(self):
        from analysis.stage_progression import compute_cross_session_transitions
        matrix, _ = compute_cross_session_transitions(self.df, self.fw)
        n = len(self.fw)
        self.assertEqual(matrix.shape, (n, n))

    def test_matrix_non_negative(self):
        from analysis.stage_progression import compute_cross_session_transitions
        matrix, _ = compute_cross_session_transitions(self.df, self.fw)
        self.assertTrue((matrix.values >= 0).all())

    def test_participant_sequences_has_all_participants(self):
        from analysis.stage_progression import compute_cross_session_transitions
        _, seqs = compute_cross_session_transitions(self.df, self.fw)
        for pid in self.df['participant_id'].unique():
            self.assertIn(pid, seqs)

    def test_sequence_entries_are_tuples(self):
        from analysis.stage_progression import compute_cross_session_transitions
        _, seqs = compute_cross_session_transitions(self.df, self.fw)
        for pid, seq in seqs.items():
            for entry in seq:
                self.assertIsInstance(entry, tuple)
                self.assertEqual(len(entry), 3)  # (session_id, stage_id, stage_name)


class TestComputeStateTransitionMatrix(unittest.TestCase):
    def setUp(self):
        self.df = _part_df(n_sessions=2, n_participants=2)
        self.fw = _mini_framework()

    def test_returns_dataframe(self):
        from analysis.stage_progression import compute_state_transition_matrix
        matrix = compute_state_transition_matrix(self.df, self.fw)
        self.assertIsInstance(matrix, pd.DataFrame)

    def test_matrix_shape(self):
        from analysis.stage_progression import compute_state_transition_matrix
        matrix = compute_state_transition_matrix(self.df, self.fw)
        n = len(self.fw)
        self.assertEqual(matrix.shape, (n, n))

    def test_matrix_values_non_negative(self):
        from analysis.stage_progression import compute_state_transition_matrix
        matrix = compute_state_transition_matrix(self.df, self.fw)
        self.assertTrue((matrix.values >= 0).all())

    def test_matrix_total_equals_total_transitions(self):
        from analysis.stage_progression import compute_state_transition_matrix
        matrix = compute_state_transition_matrix(self.df, self.fw)
        # Total segments across all participant×session groups - 1 per group
        total = 0
        for (pid, sid), grp in self.df.groupby(['participant_id', 'session_id']):
            total += max(0, len(grp) - 1)
        self.assertEqual(matrix.values.sum(), total)


if __name__ == '__main__':
    unittest.main()
