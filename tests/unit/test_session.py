"""
tests/unit/test_session.py
---------------------------
Unit tests for analysis/session.py.

Covers:
- _build_transition_matrix: shape, self-transitions, forward/back counts
- generate_session_analysis: keys, participant detail, group_props, JSON written
- generate_all_session_analyses: one report per session in df
- Edge: empty session id → returns {}
- Edge: single segment (no transitions)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import make_master_df


def _mini_framework():
    return {
        i: {'id': i, 'key': f's{i}', 'name': f'Stage {i}', 'short_name': f'S{i}', 'definition': ''}
        for i in range(5)
    }


def _part_df(n_sessions=2, n_participants=2):
    df = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    df = df[df['speaker'] == 'participant'].copy()
    df = df.rename(columns={'confidence_tier': 'label_confidence_tier'})
    df['final_label'] = df['final_label'].astype(int)
    return df


class TestBuildTransitionMatrix(unittest.TestCase):
    def test_empty_sequence(self):
        from analysis.session import _build_transition_matrix
        m = _build_transition_matrix([], n_stages=3)
        self.assertIn('0', m)
        self.assertEqual(m['0']['0'], 0)

    def test_single_element_no_transitions(self):
        from analysis.session import _build_transition_matrix
        m = _build_transition_matrix([2], n_stages=3)
        for i in range(3):
            for j in range(3):
                self.assertEqual(m[str(i)][str(j)], 0)

    def test_forward_transition_counted(self):
        from analysis.session import _build_transition_matrix
        m = _build_transition_matrix([0, 1, 2], n_stages=3)
        self.assertEqual(m['0']['1'], 1)
        self.assertEqual(m['1']['2'], 1)

    def test_backward_transition_counted(self):
        from analysis.session import _build_transition_matrix
        m = _build_transition_matrix([2, 1, 0], n_stages=3)
        self.assertEqual(m['2']['1'], 1)
        self.assertEqual(m['1']['0'], 1)

    def test_self_transition(self):
        from analysis.session import _build_transition_matrix
        m = _build_transition_matrix([1, 1, 1], n_stages=3)
        self.assertEqual(m['1']['1'], 2)

    def test_matrix_has_n_stages_keys(self):
        from analysis.session import _build_transition_matrix
        m = _build_transition_matrix([0, 1], n_stages=5)
        self.assertEqual(len(m), 5)
        for v in m.values():
            self.assertEqual(len(v), 5)


class TestGenerateSessionAnalysis(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=2, n_participants=2)
        self.fw = _mini_framework()
        # Pick first session in the dataframe
        self.sid = self.df['session_id'].iloc[0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_with_expected_keys(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        for key in ('session_id', 'n_segments', 'n_participants', 'group_stage_proportions',
                    'stage_transition_matrix', 'confidence_distribution',
                    'stage_exemplars', 'narrative_summary'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_session_id_correct(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        self.assertEqual(result['session_id'], self.sid)

    def test_n_participants_matches_data(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        n_expected = self.df[self.df['session_id'] == self.sid]['participant_id'].nunique()
        self.assertEqual(result['n_participants'], n_expected)

    def test_group_props_sum_to_one(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        total = sum(result['group_stage_proportions'].values())
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_json_file_written(self):
        from analysis.session import generate_session_analysis
        from process.output_paths import sessions_json_dir
        generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        path = os.path.join(sessions_json_dir(self.tmp), f'session_{self.sid}.json')
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data['session_id'], self.sid)

    def test_transition_matrix_shape(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        matrix = result['stage_transition_matrix']
        self.assertEqual(len(matrix), len(self.fw))
        for v in matrix.values():
            self.assertEqual(len(v), len(self.fw))

    def test_confidence_distribution_tiers(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        cd = result['confidence_distribution']
        for tier in ('high', 'medium', 'low'):
            self.assertIn(tier, cd)
            self.assertIn('count', cd[tier])
            self.assertIn('proportion', cd[tier])

    def test_empty_session_id_returns_empty(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, 'NONEXISTENT', self.fw, self.tmp)
        self.assertEqual(result, {})

    def test_participants_detail_present(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        self.assertIn('participants', result)
        for pid, detail in result['participants'].items():
            self.assertIn('n_segments', detail)
            self.assertIn('stage_proportions', detail)
            self.assertIn('dominant_stage', detail)

    def test_stage_exemplars_one_per_stage(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        ex = result['stage_exemplars']
        # Should have a key for each stage
        for st in self.fw:
            self.assertIn(str(st), ex)

    def test_narrative_is_non_empty_string(self):
        from analysis.session import generate_session_analysis
        result = generate_session_analysis(self.df, self.sid, self.fw, self.tmp)
        self.assertIsInstance(result['narrative_summary'], str)
        self.assertGreater(len(result['narrative_summary']), 5)


class TestGenerateAllSessionAnalyses(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_report_per_unique_session(self):
        from analysis.session import generate_all_session_analyses
        reports = generate_all_session_analyses(self.df, self.fw, self.tmp)
        n_sessions = self.df['session_id'].nunique()
        self.assertEqual(len(reports), n_sessions)

    def test_all_reports_have_session_id(self):
        from analysis.session import generate_all_session_analyses
        reports = generate_all_session_analyses(self.df, self.fw, self.tmp)
        for r in reports:
            self.assertIn('session_id', r)

    def test_empty_df_returns_empty_list(self):
        from analysis.session import generate_all_session_analyses
        empty = pd.DataFrame(columns=['session_id', 'participant_id', 'session_number',
                                      'cohort_id', 'segment_index', 'final_label',
                                      'label_confidence_tier'])
        reports = generate_all_session_analyses(empty, self.fw, self.tmp)
        self.assertEqual(reports, [])


if __name__ == '__main__':
    unittest.main()
