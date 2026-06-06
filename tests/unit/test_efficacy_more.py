"""
tests/unit/test_efficacy_more.py
---------------------------------
Gap coverage for analysis/efficacy.py — complementing test_efficacy.py.

New coverage:
  - run_efficacy_analysis writes progression_summary.txt + efficacy_summary.json
    on a make_master_df input (the canonical shared fixture).
  - Descriptive / single-arm framing: the report must NOT claim efficacy; key
    descriptive-framing strings must be present, methodology §8.3.
  - link_to_external with a tiny wide pre/post outcomes frame.
  - No-variance guard: constant external outcome yields a clean correlation row
    (r rendered as n/a), not a crash.
  - Empty-DataFrame edge case: run_efficacy_analysis returns cleanly (no crash,
    files_written may be empty).
"""

import json
import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from analysis import efficacy as E
from analysis.superposition import attach_superposition
from process.config import SuperpositionConfig, EfficacyConfig
from process import output_paths as _paths
from tests.testhelpers.fixtures import make_master_df


_FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


def _build_df(tmp, n_sessions=3, n_participants=2):
    """Attach superposition to a make_master_df participant slice."""
    df_all = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    attach_superposition(df_all, tmp, config=SuperpositionConfig())
    # efficacy operates on participant segments only
    return df_all[df_all['speaker'] == 'participant'].copy()


class TestRunEfficacyWritesArtifacts(unittest.TestCase):
    """run_efficacy_analysis produces the expected files from make_master_df."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _build_df(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_progression_summary_txt(self):
        res = E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        files = res['files_written']
        report_files = [f for f in files if 'progression_summary.txt' in f]
        self.assertGreater(len(report_files), 0, 'progression_summary.txt not in files_written')
        self.assertTrue(os.path.isfile(report_files[0]))

    def test_writes_efficacy_summary_json(self):
        E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        sp = os.path.join(_paths.efficacy_dir(self.tmp), 'efficacy_summary.json')
        self.assertTrue(os.path.isfile(sp), 'efficacy_summary.json not written')

    def test_efficacy_summary_json_has_expected_keys(self):
        E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        sp = os.path.join(_paths.efficacy_dir(self.tmp), 'efficacy_summary.json')
        with open(sp, encoding='utf-8') as f:
            data = json.load(f)
        for key in ('n_participants', 'n_sessions', 'barrier_crossed', 'barrier_total',
                    'underpowered', 'mk_adaptive_occupancy', 'sign_test'):
            self.assertIn(key, data, f'Missing key: {key}')

    def test_writes_participant_session_outcomes_csv(self):
        E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        p = os.path.join(_paths.efficacy_dir(self.tmp), 'participant_session_outcomes.csv')
        self.assertTrue(os.path.isfile(p))

    def test_writes_barrier_crossing_csv(self):
        E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        p = os.path.join(_paths.efficacy_dir(self.tmp), 'barrier_crossing.csv')
        self.assertTrue(os.path.isfile(p))

    def test_n_participants_matches_input(self):
        E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        sp = os.path.join(_paths.efficacy_dir(self.tmp), 'efficacy_summary.json')
        with open(sp, encoding='utf-8') as f:
            data = json.load(f)
        expected_participants = self.df['participant_id'].nunique()
        self.assertEqual(data['n_participants'], expected_participants)


class TestDescriptiveSingleArmFraming(unittest.TestCase):
    """methodology §8.3 — the report is DESCRIPTIVE, not an efficacy claim.

    The written progression_summary.txt must carry explicit single-arm /
    descriptive framing and must NOT contain unhedged efficacy language.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _build_df(self.tmp)
        E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        rep_path = os.path.join(_paths.reports_outcomes_dir(self.tmp), 'progression_summary.txt')
        with open(rep_path, encoding='utf-8') as f:
            self.report_text = f.read()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_report_contains_descriptive_framing_header(self):
        # The title line must name it descriptive / single-arm
        self.assertIn('descriptive', self.report_text.lower())

    def test_report_contains_no_control_arm(self):
        # Must acknowledge no control arm
        self.assertIn('no control arm', self.report_text.lower())

    def test_report_does_not_claim_efficacy(self):
        # The report must explicitly say NOT an efficacy claim or similar
        lower = self.report_text.lower()
        self.assertIn('not an efficacy', lower)

    def test_report_contains_hypothesis_generating_hedge(self):
        # Must frame results as hypothesis-generating, not confirmatory
        lower = self.report_text.lower()
        self.assertTrue(
            'hypothesis' in lower or 'hypothes' in lower,
            'Report should contain "hypothesis" or similar hedge'
        )

    def test_report_contains_observational_caveat(self):
        lower = self.report_text.lower()
        self.assertTrue(
            'observational' in lower or 'associational' in lower or 'associations' in lower,
            'Report should flag observational/associational nature'
        )

    def test_report_primary_outcome_is_ordinal_safe(self):
        # Mann-Kendall is primary; the sensitivity line must be labeled sensitivity
        self.assertIn('SENSITIVITY', self.report_text.upper())

    def test_efficacy_summary_json_underpowered_flag_is_bool(self):
        sp = os.path.join(_paths.efficacy_dir(self.tmp), 'efficacy_summary.json')
        with open(sp, encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data['underpowered'], bool)


class TestLinkToExternal(unittest.TestCase):
    """link_to_external: correlation with a tiny wide pre/post outcomes frame."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df_all = make_master_df(n_sessions=4, n_participants=3)
        attach_superposition(df_all, self.tmp, config=SuperpositionConfig())
        self.df = df_all[df_all['speaker'] == 'participant'].copy()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _participant_ids(self):
        return sorted(self.df['participant_id'].unique().tolist())

    def _make_wide_outcomes(self, values_pre, values_post):
        pids = self._participant_ids()
        return {
            'mode': 'wide',
            'measures': ['pain'],
            'change_cols': ['pain_change'],
            'data': pd.DataFrame({
                'participant_id': pids[:len(values_pre)],
                'pain_pre': values_pre,
                'pain_post': values_post,
                'pain_change': [b - a for a, b in zip(values_pre, values_post)],
            }),
        }

    def test_link_to_external_returns_dataframe(self):
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        slopes = E.compute_participant_slopes(ps, 'progression_coord')
        pids = self._participant_ids()
        outcomes = self._make_wide_outcomes([8, 7, 9][:len(pids)], [5, 6, 4][:len(pids)])
        result = E.link_to_external(slopes, outcomes)
        # With 3 participants and matched slopes we should get correlation rows
        # (may be empty if fewer than 3 matched, but must not crash)
        self.assertIsInstance(result, pd.DataFrame)

    def test_link_to_external_none_outcomes_returns_empty(self):
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        slopes = E.compute_participant_slopes(ps, 'progression_coord')
        result = E.link_to_external(slopes, None)
        self.assertTrue(result.empty)

    def test_link_to_external_non_wide_returns_empty(self):
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        slopes = E.compute_participant_slopes(ps, 'progression_coord')
        long_outcomes = {'mode': 'long', 'measures': ['craving'], 'change_cols': [],
                         'data': pd.DataFrame({'participant_id': ['P01'], 'craving': [3.0]})}
        result = E.link_to_external(slopes, long_outcomes)
        self.assertTrue(result.empty)

    def test_correlation_columns_present_when_data_sufficient(self):
        """When ≥3 participants match we get pearson_r and p_value columns."""
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        slopes = E.compute_participant_slopes(ps, 'progression_coord')
        pids = sorted(slopes['participant_id'].unique())[:3]
        # Force at least 3 participants in slopes
        slopes3 = slopes[slopes['participant_id'].isin(pids)].copy()
        outcomes = {
            'mode': 'wide',
            'measures': ['pain'],
            'change_cols': ['pain_change'],
            'data': pd.DataFrame({
                'participant_id': pids,
                'pain_change': [-3.0, -1.0, -2.0],
            }),
        }
        result = E.link_to_external(slopes3, outcomes)
        if not result.empty:
            self.assertIn('pearson_r', result.columns)
            self.assertIn('p_value', result.columns)
            self.assertIn('n', result.columns)


class TestNoVarianceGuard(unittest.TestCase):
    """Constant external outcome must render cleanly, not crash.

    When all participants have the same outcome change (zero variance),
    pearsonr raises ValueError or returns nan.  link_to_external must handle
    this gracefully — either by returning an empty DataFrame or by rendering
    the correlation as NaN/None rather than crashing.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df_all = make_master_df(n_sessions=4, n_participants=3)
        attach_superposition(df_all, self.tmp, config=SuperpositionConfig())
        self.df = df_all[df_all['speaker'] == 'participant'].copy()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_constant_outcome_does_not_crash(self):
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        slopes = E.compute_participant_slopes(ps, 'progression_coord')
        pids = sorted(slopes['participant_id'].unique())[:3]
        slopes3 = slopes[slopes['participant_id'].isin(pids)].copy()
        # All participants have the same outcome change: variance = 0
        outcomes = {
            'mode': 'wide',
            'measures': ['pain'],
            'change_cols': ['pain_change'],
            'data': pd.DataFrame({
                'participant_id': pids,
                'pain_change': [5.0, 5.0, 5.0],   # constant — zero variance
            }),
        }
        # Must not raise; result is either empty or has NaN-safe correlation
        try:
            result = E.link_to_external(slopes3, outcomes)
        except Exception as exc:
            self.fail(f'link_to_external raised on zero-variance outcome: {exc}')
        # If a row was returned, pearson_r must be nan/None (not a valid float)
        if isinstance(result, pd.DataFrame) and not result.empty:
            for _, row in result.iterrows():
                rv = row.get('pearson_r')
                self.assertTrue(
                    rv is None or (isinstance(rv, float) and (rv != rv)),
                    f'Expected nan/None pearson_r for constant outcome, got {rv!r}'
                )

    def test_full_pipeline_constant_outcome_no_crash(self):
        """run_efficacy_analysis must survive when the loaded outcomes.csv has constant change."""
        os.makedirs(_paths.meta_dir(self.tmp), exist_ok=True)
        pids = sorted(self.df['participant_id'].unique())
        pd.DataFrame({
            'participant_id': pids,
            'pain_pre': [5] * len(pids),
            'pain_post': [5] * len(pids),   # change = 0 for all
        }).to_csv(os.path.join(_paths.meta_dir(self.tmp), 'outcomes.csv'), index=False)
        try:
            E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        except Exception as exc:
            self.fail(f'run_efficacy_analysis raised with constant-outcome CSV: {exc}')


class TestEmptyDataFrameEdgeCase(unittest.TestCase):
    """Empty / degenerate inputs must return cleanly without crashing."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_df_returns_no_crash(self):
        empty = pd.DataFrame(columns=[
            'participant_id', 'session_id', 'session_number',
            'final_label', 'progression_coord', 'speaker',
        ])
        try:
            result = E.run_efficacy_analysis(empty, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        except Exception as exc:
            self.fail(f'run_efficacy_analysis raised on empty DataFrame: {exc}')
        self.assertIn('files_written', result)

    def test_empty_df_files_written_is_list(self):
        empty = pd.DataFrame(columns=[
            'participant_id', 'session_id', 'session_number',
            'final_label', 'progression_coord', 'speaker',
        ])
        result = E.run_efficacy_analysis(empty, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        self.assertIsInstance(result['files_written'], list)

    def test_compute_participant_session_outcomes_empty(self):
        empty = pd.DataFrame(columns=['participant_id', 'session_id', 'session_number', 'final_label'])
        result = E.compute_participant_session_outcomes(empty, EfficacyConfig())
        self.assertTrue(result.empty)

    def test_compute_group_trajectory_empty(self):
        empty = pd.DataFrame(columns=['participant_id', 'session_number', 'progression_coord'])
        result = E.compute_group_trajectory(empty, 'progression_coord')
        self.assertTrue(result.empty)

    def test_compute_barrier_crossing_empty(self):
        empty = pd.DataFrame(columns=['participant_id', 'session_id', 'final_label'])
        result = E.compute_barrier_crossing(empty, EfficacyConfig())
        self.assertTrue(result.empty)

    def test_single_participant_one_session_no_slope(self):
        """Single session → slope = None (< 2 data points); must not crash."""
        df_all = make_master_df(n_sessions=1, n_participants=1)
        attach_superposition(df_all, self.tmp, config=SuperpositionConfig())
        df = df_all[df_all['speaker'] == 'participant'].copy()
        ps = E.compute_participant_session_outcomes(df, EfficacyConfig())
        slopes = E.compute_participant_slopes(ps, 'progression_coord')
        self.assertFalse(slopes.empty)
        # With 1 session slope should be None or NaN
        slope_val = slopes.iloc[0]['slope']
        self.assertTrue(slope_val is None or (isinstance(slope_val, float) and slope_val != slope_val),
                        f'Expected None/NaN slope for single session, got {slope_val!r}')


if __name__ == '__main__':
    unittest.main()
