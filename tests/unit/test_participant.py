"""
tests/unit/test_participant.py
-------------------------------
Unit tests for analysis/participant.py.

Covers:
- _compute_progression_score: weighted mean, all-zero, all-max
- _compute_progression_trend: positive/negative/flat slope, <2 points → 0.0
- generate_participant_report: keys, session order, trend interpretation, JSON written
- generate_all_participant_reports: one report per participant
- Edge: single session (trend=0.0)
- Edge: empty df for participant → {}
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


class TestComputeProgressionScore(unittest.TestCase):
    def test_all_stage_zero(self):
        from analysis.participant import _compute_progression_score
        score = _compute_progression_score({'0': 1.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 0.0})
        self.assertAlmostEqual(score, 0.0)

    def test_all_stage_four(self):
        from analysis.participant import _compute_progression_score
        score = _compute_progression_score({'0': 0.0, '1': 0.0, '2': 0.0, '3': 0.0, '4': 1.0})
        self.assertAlmostEqual(score, 4.0)

    def test_mixed_stages(self):
        from analysis.participant import _compute_progression_score
        # 50% stage 2, 50% stage 4 → score = 0.5*2 + 0.5*4 = 3.0
        score = _compute_progression_score({'0': 0.0, '1': 0.0, '2': 0.5, '3': 0.0, '4': 0.5})
        self.assertAlmostEqual(score, 3.0)

    def test_uniform_distribution(self):
        from analysis.participant import _compute_progression_score
        # 0.2 each of stages 0-4 → score = 0.2*(0+1+2+3+4) = 2.0
        props = {str(i): 0.2 for i in range(5)}
        score = _compute_progression_score(props)
        self.assertAlmostEqual(score, 2.0)


class TestComputeProgressionTrend(unittest.TestCase):
    def test_positive_trend(self):
        from analysis.participant import _compute_progression_trend
        scores = {'1': 0.5, '2': 1.0, '3': 1.5}
        slope = _compute_progression_trend(scores, ['1', '2', '3'])
        self.assertGreater(slope, 0)

    def test_negative_trend(self):
        from analysis.participant import _compute_progression_trend
        scores = {'1': 1.5, '2': 1.0, '3': 0.5}
        slope = _compute_progression_trend(scores, ['1', '2', '3'])
        self.assertLess(slope, 0)

    def test_flat_returns_zero(self):
        from analysis.participant import _compute_progression_trend
        scores = {'1': 1.0, '2': 1.0, '3': 1.0}
        slope = _compute_progression_trend(scores, ['1', '2', '3'])
        self.assertAlmostEqual(slope, 0.0, places=4)

    def test_single_point_returns_zero(self):
        from analysis.participant import _compute_progression_trend
        slope = _compute_progression_trend({'1': 1.0}, ['1'])
        self.assertEqual(slope, 0.0)

    def test_empty_returns_zero(self):
        from analysis.participant import _compute_progression_trend
        slope = _compute_progression_trend({}, [])
        self.assertEqual(slope, 0.0)


class TestGenerateParticipantReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()
        self.pid = 'P01'

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_with_expected_keys(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        for key in ('participant_id', 'n_sessions', 'session_ids', 'sessions',
                    'longitudinal_trajectory', 'progression_score_by_session',
                    'progression_trend', 'progression_trend_interpretation',
                    'stage_exemplars_overall', 'narrative_summary'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_participant_id_correct(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        self.assertEqual(result['participant_id'], self.pid)

    def test_n_sessions_matches_data(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        n_exp = self.df[self.df['participant_id'] == self.pid]['session_id'].nunique()
        self.assertEqual(result['n_sessions'], n_exp)

    def test_json_file_written(self):
        from analysis.participant import generate_participant_report
        from process.output_paths import participants_json_dir
        generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        path = os.path.join(participants_json_dir(self.tmp), f'participant_{self.pid}.json')
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(data['participant_id'], self.pid)

    def test_progression_score_by_session_is_dict(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        self.assertIsInstance(result['progression_score_by_session'], dict)
        for v in result['progression_score_by_session'].values():
            self.assertIsInstance(v, float)

    def test_progression_scores_in_valid_range(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        for v in result['progression_score_by_session'].values():
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 4.0)

    def test_trend_interpretation_is_one_of_three(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        interp = result['progression_trend_interpretation']
        valid = {'advancing — progression scores increase across sessions',
                 'regressing — progression scores decrease across sessions',
                 'stable — no consistent directional trend across sessions'}
        self.assertIn(interp, valid)

    def test_narrative_is_non_empty_string(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        self.assertIsInstance(result['narrative_summary'], str)
        self.assertGreater(len(result['narrative_summary']), 10)

    def test_stage_exemplars_overall_has_stage_keys(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        for st in self.fw:
            self.assertIn(str(st), result['stage_exemplars_overall'])

    def test_nonexistent_participant_returns_empty(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, 'NOSUCHPARTICIPANT', self.fw, self.tmp)
        self.assertEqual(result, {})

    def test_single_session_trend_is_zero(self):
        from analysis.participant import generate_participant_report
        df_one = _part_df(n_sessions=1, n_participants=1)
        result = generate_participant_report(df_one, 'P01', self.fw, self.tmp)
        self.assertEqual(result['progression_trend'], 0.0)

    def test_progression_score_overall_is_mean_of_sessions(self):
        from analysis.participant import generate_participant_report
        result = generate_participant_report(self.df, self.pid, self.fw, self.tmp)
        scores = list(result['progression_score_by_session'].values())
        expected_overall = round(sum(scores) / len(scores), 4)
        self.assertAlmostEqual(result['progression_score_overall'], expected_overall, places=3)


class TestGenerateAllParticipantReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=2, n_participants=3)
        self.fw = _mini_framework()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_one_report_per_participant(self):
        from analysis.participant import generate_all_participant_reports
        reports = generate_all_participant_reports(self.df, self.fw, self.tmp)
        self.assertEqual(len(reports), 3)

    def test_all_have_participant_id(self):
        from analysis.participant import generate_all_participant_reports
        reports = generate_all_participant_reports(self.df, self.fw, self.tmp)
        for r in reports:
            self.assertIn('participant_id', r)

    def test_empty_df_returns_empty_list(self):
        from analysis.participant import generate_all_participant_reports
        empty = pd.DataFrame(columns=['participant_id', 'session_id', 'session_number',
                                      'cohort_id', 'segment_index', 'final_label',
                                      'label_confidence_tier', 'llm_confidence_primary'])
        reports = generate_all_participant_reports(empty, self.fw, self.tmp)
        self.assertEqual(reports, [])


if __name__ == '__main__':
    unittest.main()
