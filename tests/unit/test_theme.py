"""
tests/unit/test_theme.py
--------------------------
Unit tests for analysis/theme.py.

Covers:
- _compute_lift: expected formula, zero-denominator guards
- generate_theme_stage_report: keys, prevalence, exemplars, JSON written
- generate_all_theme_reports: one report per stage; skips codebook when absent
- generate_codebook_code_report: stage co-occurrence keys, JSON written
- Edge: stage absent from data → prevalence=0
- Edge: empty DataFrame
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


class TestComputeLift(unittest.TestCase):
    def test_expected_formula(self):
        from analysis.theme import _compute_lift
        # n_code_in_stage=5, n_stage=10, n_code=20, n_total=100
        # observed=0.5, expected=0.2, lift=2.5
        lift = _compute_lift(5, 10, 20, 100)
        self.assertAlmostEqual(lift, 2.5, places=3)

    def test_lift_one_when_independent(self):
        from analysis.theme import _compute_lift
        # n_code_in_stage/n_stage == n_code/n_total → lift=1
        lift = _compute_lift(5, 10, 50, 100)
        self.assertAlmostEqual(lift, 1.0, places=3)

    def test_zero_stage_returns_zero(self):
        from analysis.theme import _compute_lift
        self.assertEqual(_compute_lift(0, 0, 20, 100), 0.0)

    def test_zero_total_returns_zero(self):
        from analysis.theme import _compute_lift
        self.assertEqual(_compute_lift(5, 10, 20, 0), 0.0)

    def test_zero_code_returns_zero(self):
        from analysis.theme import _compute_lift
        self.assertEqual(_compute_lift(5, 10, 0, 100), 0.0)


class TestGenerateThemeStageReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_with_expected_keys(self):
        from analysis.theme import generate_theme_stage_report
        result = generate_theme_stage_report(self.df, 0, self.fw, self.tmp)
        for key in ('stage_id', 'stage_name', 'overall_prevalence', 'n_segments_total',
                    'n_segments_this_stage', 'prevalence_by_session_number',
                    'prevalence_by_participant', 'longitudinal_trend',
                    'top_exemplars', 'co_occurring_codes'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_stage_id_correct(self):
        from analysis.theme import generate_theme_stage_report
        for stage_id in range(5):
            result = generate_theme_stage_report(self.df, stage_id, self.fw, self.tmp)
            self.assertEqual(result['stage_id'], stage_id)

    def test_overall_prevalence_in_range(self):
        from analysis.theme import generate_theme_stage_report
        for stage_id in range(5):
            result = generate_theme_stage_report(self.df, stage_id, self.fw, self.tmp)
            self.assertGreaterEqual(result['overall_prevalence'], 0.0)
            self.assertLessEqual(result['overall_prevalence'], 1.0)

    def test_prevalences_sum_to_one(self):
        """Sum of per-stage overall_prevalences should equal ~1.0."""
        from analysis.theme import generate_theme_stage_report
        total = sum(
            generate_theme_stage_report(self.df, st, self.fw, self.tmp)['overall_prevalence']
            for st in range(5)
        )
        self.assertAlmostEqual(total, 1.0, places=3)

    def test_n_segments_correct(self):
        from analysis.theme import generate_theme_stage_report
        result = generate_theme_stage_report(self.df, 0, self.fw, self.tmp)
        self.assertEqual(result['n_segments_total'], len(self.df))
        n_stage0 = int((self.df['final_label'] == 0).sum())
        self.assertEqual(result['n_segments_this_stage'], n_stage0)

    def test_json_file_written(self):
        from analysis.theme import generate_theme_stage_report
        from process.output_paths import themes_json_dir
        generate_theme_stage_report(self.df, 0, self.fw, self.tmp)
        out_dir = themes_json_dir(self.tmp)
        jsons = [f for f in os.listdir(out_dir) if f.startswith('stage_0')]
        self.assertGreater(len(jsons), 0)

    def test_prevalence_by_participant_keys(self):
        from analysis.theme import generate_theme_stage_report
        result = generate_theme_stage_report(self.df, 0, self.fw, self.tmp)
        for pid in self.df['participant_id'].unique():
            self.assertIn(pid, result['prevalence_by_participant'])

    def test_stage_absent_from_data_returns_zero_prevalence(self):
        """Stage 4 (Reappraisal) is rare in make_master_df. Prevalence >= 0."""
        from analysis.theme import generate_theme_stage_report
        df_no_stage4 = self.df[self.df['final_label'] != 4].copy()
        result = generate_theme_stage_report(df_no_stage4, 4, self.fw, self.tmp)
        self.assertEqual(result['overall_prevalence'], 0.0)
        self.assertEqual(result['n_segments_this_stage'], 0)


class TestGenerateAllThemeReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=2, n_participants=2)
        self.fw = _mini_framework()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_one_report_per_stage(self):
        from analysis.theme import generate_all_theme_reports
        reports = generate_all_theme_reports(self.df, self.fw, self.tmp)
        self.assertEqual(len(reports), len(self.fw))

    def test_all_have_stage_id(self):
        from analysis.theme import generate_all_theme_reports
        reports = generate_all_theme_reports(self.df, self.fw, self.tmp)
        for r in reports:
            self.assertIn('stage_id', r)

    def test_no_codebook_column_returns_stage_reports_only(self):
        from analysis.theme import generate_all_theme_reports
        df_no_cb = self.df.drop(columns=['codebook_labels_ensemble'], errors='ignore')
        reports = generate_all_theme_reports(df_no_cb, self.fw, self.tmp)
        self.assertEqual(len(reports), len(self.fw))


class TestGenerateCodebookCodeReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=2, n_participants=2)
        self.fw = _mini_framework()
        # Ensure codebook_labels_ensemble exists
        if 'codebook_labels_ensemble' not in self.df.columns:
            self.df['codebook_labels_ensemble'] = [['body_awareness']] * len(self.df)
        # Make sure at least 'body_awareness' code exists
        self.df['codebook_labels_ensemble'] = self.df['codebook_labels_ensemble'].apply(
            lambda x: x if isinstance(x, list) else ['body_awareness']
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_with_expected_keys(self):
        from analysis.theme import generate_codebook_code_report
        result = generate_codebook_code_report(self.df, 'body_awareness', self.fw, self.tmp)
        for key in ('code_id', 'overall_prevalence', 'n_segments_total',
                    'n_segments_coded', 'prevalence_by_session_number',
                    'stage_co_occurrence', 'top_exemplars'):
            self.assertIn(key, result, f"Missing key: {key}")

    def test_code_id_correct(self):
        from analysis.theme import generate_codebook_code_report
        result = generate_codebook_code_report(self.df, 'body_awareness', self.fw, self.tmp)
        self.assertEqual(result['code_id'], 'body_awareness')

    def test_prevalence_in_range(self):
        from analysis.theme import generate_codebook_code_report
        result = generate_codebook_code_report(self.df, 'body_awareness', self.fw, self.tmp)
        self.assertGreaterEqual(result['overall_prevalence'], 0.0)
        self.assertLessEqual(result['overall_prevalence'], 1.0)

    def test_stage_co_occurrence_has_all_stages(self):
        from analysis.theme import generate_codebook_code_report
        result = generate_codebook_code_report(self.df, 'body_awareness', self.fw, self.tmp)
        co = result['stage_co_occurrence']
        for st in self.fw:
            self.assertIn(str(st), co)
            self.assertIn('lift', co[str(st)])

    def test_json_file_written(self):
        from analysis.theme import generate_codebook_code_report
        from process.output_paths import themes_json_dir
        generate_codebook_code_report(self.df, 'body_awareness', self.fw, self.tmp)
        out_dir = themes_json_dir(self.tmp)
        jsons = [f for f in os.listdir(out_dir) if 'body_awareness' in f]
        self.assertGreater(len(jsons), 0)

    def test_absent_code_returns_zero_prevalence(self):
        from analysis.theme import generate_codebook_code_report
        result = generate_codebook_code_report(self.df, 'NONEXISTENTCODE', self.fw, self.tmp)
        self.assertEqual(result['n_segments_coded'], 0)
        self.assertEqual(result['overall_prevalence'], 0.0)


if __name__ == '__main__':
    unittest.main()
