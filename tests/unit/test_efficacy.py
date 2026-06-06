"""
tests/test_efficacy.py
----------------------
Unit tests for analysis/efficacy.py — internal VAAMR outcomes, external-outcome
ingestion (wide pre/post + long auto-detect), group trajectory CIs, barrier
crossing, and external linkage. No model backends.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis import efficacy as E
from analysis.superposition import attach_superposition
from process.config import SuperpositionConfig, EfficacyConfig
from process import output_paths as _paths

_FRAMEWORK = {0: {'short_name': 'Vigilance'}, 1: {'short_name': 'Avoidance'},
              2: {'short_name': 'AttnReg'}, 3: {'short_name': 'Metacog'},
              4: {'short_name': 'Reappraisal'}}


def _df_advancing(tmp):
    """3 participants, 4 sessions, rising stages."""
    rows = []
    for p in range(3):
        for s in range(1, 5):
            stage = min(1 + s // 2 + (p % 2), 4)
            for j in range(4):
                rows.append(dict(
                    segment_id=f'P{p}_c1s{s}_{j}', participant_id=f'P{p}',
                    session_id=f'c1s{s}', session_number=s, segment_index=j,
                    speaker='participant', word_count=30, text='x',
                    final_label=stage, primary_stage=stage, secondary_stage=np.nan,
                    llm_confidence_primary=0.8, llm_confidence_secondary=np.nan,
                    rater_votes=None,
                ))
    df = pd.DataFrame(rows)
    attach_superposition(df, tmp, config=SuperpositionConfig())
    return df


class TestInternalOutcomes(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _df_advancing(self.tmp)

    def test_participant_session_outcomes(self):
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        self.assertFalse(ps.empty)
        for col in ('progression_coord', 'adaptive_occupancy', 'maladaptive_occupancy'):
            self.assertIn(col, ps.columns)

    def test_group_trajectory_has_ci(self):
        ps = E.compute_participant_session_outcomes(self.df, EfficacyConfig())
        traj = E.compute_group_trajectory(ps, 'progression_coord')
        self.assertIn('ci_lo', traj.columns)
        self.assertIn('ci_hi', traj.columns)
        # advancing cohort → later session mean > earlier
        self.assertGreaterEqual(traj['mean'].iloc[-1], traj['mean'].iloc[0])

    def test_barrier_crossing(self):
        bc = E.compute_barrier_crossing(self.df, EfficacyConfig())
        self.assertIn('crossed_to_attention_regulation', bc.columns)
        self.assertTrue(bc['crossed_to_attention_regulation'].any())


class TestExternalIngestion(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.makedirs(_paths.meta_dir(self.tmp), exist_ok=True)

    def test_wide_pre_post_change(self):
        pd.DataFrame({
            'participant_id': ['P0', 'P1', 'P2'],
            'pain_pre': [8, 7, 9], 'pain_post': [5, 6, 4],
        }).to_csv(os.path.join(_paths.meta_dir(self.tmp), 'outcomes.csv'), index=False)
        out = E.load_external_outcomes(self.tmp, EfficacyConfig())
        self.assertEqual(out['mode'], 'wide')
        self.assertIn('pain', out['measures'])
        self.assertIn('pain_change', out['change_cols'])
        self.assertEqual(out['data']['pain_change'].tolist(), [-3, -1, -5])

    def test_long_autodetect(self):
        pd.DataFrame({
            'participant_id': ['P0', 'P0', 'P1'],
            'session_number': [1, 2, 1],
            'craving': [5.0, 3.0, 4.0],
        }).to_csv(os.path.join(_paths.meta_dir(self.tmp), 'outcomes.csv'), index=False)
        out = E.load_external_outcomes(self.tmp, EfficacyConfig())
        self.assertEqual(out['mode'], 'long')
        self.assertIn('craving', out['measures'])

    def test_absent_returns_none(self):
        self.assertIsNone(E.load_external_outcomes(self.tmp, EfficacyConfig()))


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _df_advancing(self.tmp)

    def test_run_writes_artifacts(self):
        res = E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        files = res['files_written']
        self.assertTrue(any(os.path.join('01_outcomes', 'progression_summary.txt') in f for f in files))
        self.assertTrue(os.path.isfile(os.path.join(_paths.efficacy_dir(self.tmp),
                                                     'group_progression_trajectory.csv')))

    def test_external_linkage(self):
        os.makedirs(_paths.meta_dir(self.tmp), exist_ok=True)
        pd.DataFrame({
            'participant_id': ['P0', 'P1', 'P2'],
            'function_pre': [3, 4, 2], 'function_post': [6, 6, 7],
        }).to_csv(os.path.join(_paths.meta_dir(self.tmp), 'outcomes.csv'), index=False)
        res = E.run_efficacy_analysis(self.df, _FRAMEWORK, self.tmp, config=EfficacyConfig())
        link = os.path.join(_paths.efficacy_dir(self.tmp), 'external_outcome_linkage.csv')
        self.assertTrue(os.path.isfile(link))


if __name__ == '__main__':
    unittest.main()
