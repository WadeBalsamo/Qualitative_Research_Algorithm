"""
tests/unit/test_gnn_lift.py
----------------------------
Unit tests for gnn_layer/gnn_lift.py:
  - gnn_vaamr_vce_lift
  - llm_vaamr_vce_lift
  - compare_gnn_vs_llm (both-elevated flag)
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import synthetic_df, embedding_patch, make_master_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer import gnn_lift as _lift


def _participant_df(n=12):
    """Minimal participant DataFrame with codebook_labels_ensemble and final_label."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n):
        stage = i % 5
        rows.append({
            'segment_id': f's{i}',
            'speaker': 'participant',
            'final_label': stage,
            'codebook_labels_ensemble': ['code_a', 'code_b'] if i % 2 == 0 else ['code_c'],
        })
    return pd.DataFrame(rows)


def _mixed_df(n_part=10, n_ther=5):
    rows = []
    for i in range(n_part):
        rows.append({
            'segment_id': f'p{i}',
            'speaker': 'participant',
            'final_label': i % 5,
            'codebook_labels_ensemble': ['code_x'] if i % 3 == 0 else ['code_y'],
        })
    for j in range(n_ther):
        rows.append({
            'segment_id': f't{j}',
            'speaker': 'therapist',
            'final_label': np.nan,
            'codebook_labels_ensemble': [],
        })
    return pd.DataFrame(rows)


class TestGnnVaamrVceLift(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _stage_map(self, df):
        """Build a gnn_stage_by_id dict from the df's final_label for participants."""
        return {
            str(r['segment_id']): int(r['final_label'])
            for _, r in df.iterrows()
            if r['speaker'] == 'participant' and not pd.isna(r.get('final_label'))
        }

    def test_returns_dataframe(self):
        df = _participant_df()
        sm = self._stage_map(df)
        t = _lift.gnn_vaamr_vce_lift(df, sm)
        self.assertIsInstance(t, pd.DataFrame)

    def test_columns(self):
        df = _participant_df()
        sm = self._stage_map(df)
        t = _lift.gnn_vaamr_vce_lift(df, sm)
        for col in ('vaamr_stage', 'vce_code', 'lift', 'count', 'p_b'):
            self.assertIn(col, t.columns)

    def test_lift_positive(self):
        df = _participant_df()
        sm = self._stage_map(df)
        t = _lift.gnn_vaamr_vce_lift(df, sm)
        self.assertTrue((t['lift'] >= 0).all())

    def test_only_participant_segments_used(self):
        """Therapist segments must not contribute to lift."""
        df = _mixed_df(10, 5)
        sm = self._stage_map(df)
        t_mixed = _lift.gnn_vaamr_vce_lift(df, sm)
        df_part_only = df[df['speaker'] == 'participant'].copy()
        t_part = _lift.gnn_vaamr_vce_lift(df_part_only, sm)
        # Both tables should have same rows (therapist rows contributed nothing)
        self.assertEqual(len(t_mixed), len(t_part))

    def test_empty_stage_map_returns_empty_df(self):
        df = _participant_df()
        t = _lift.gnn_vaamr_vce_lift(df, {})
        self.assertEqual(len(t), 0)


class TestLlmVaamrVceLift(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dataframe(self):
        df = _participant_df()
        t = _lift.llm_vaamr_vce_lift(df)
        self.assertIsInstance(t, pd.DataFrame)

    def test_columns(self):
        df = _participant_df()
        t = _lift.llm_vaamr_vce_lift(df)
        for col in ('vaamr_stage', 'vce_code', 'lift', 'count', 'p_b'):
            self.assertIn(col, t.columns)

    def test_lift_positive(self):
        df = _participant_df()
        t = _lift.llm_vaamr_vce_lift(df)
        self.assertTrue((t['lift'] >= 0).all())

    def test_nan_final_label_excluded(self):
        df = _participant_df()
        df.at[0, 'final_label'] = np.nan
        t = _lift.llm_vaamr_vce_lift(df)
        # Should still produce a table for the remaining rows
        self.assertIsInstance(t, pd.DataFrame)

    def test_empty_codebook_labels_excluded(self):
        df = pd.DataFrame([
            {'segment_id': 's0', 'speaker': 'participant', 'final_label': 0,
             'codebook_labels_ensemble': []},
            {'segment_id': 's1', 'speaker': 'participant', 'final_label': 1,
             'codebook_labels_ensemble': ['code_a']},
        ])
        t = _lift.llm_vaamr_vce_lift(df)
        # Only code_a has counts; no rows for empty list
        self.assertGreater(len(t), 0)


class TestCompareGnnVsLlm(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _tables(self):
        df = _participant_df(15)
        sm = {str(r['segment_id']): int(r['final_label'])
              for _, r in df.iterrows() if not pd.isna(r.get('final_label'))}
        gnn_t = _lift.gnn_vaamr_vce_lift(df, sm)
        llm_t = _lift.llm_vaamr_vce_lift(df)
        return gnn_t, llm_t

    def test_returns_dataframe(self):
        gnn_t, llm_t = self._tables()
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        self.assertIsInstance(out, pd.DataFrame)

    def test_columns_present(self):
        gnn_t, llm_t = self._tables()
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        for col in ('vaamr_stage', 'vce_code', 'lift_gnn', 'lift_llm', 'both_elevated'):
            self.assertIn(col, out.columns)

    def test_both_elevated_is_bool(self):
        gnn_t, llm_t = self._tables()
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        self.assertEqual(out['both_elevated'].dtype, bool)

    def test_both_elevated_flag_correct(self):
        """Entries with lift_gnn >= 1.5 AND lift_llm >= 1.5 must be True."""
        gnn_t = pd.DataFrame([
            {'vaamr_stage': 0, 'vce_code': 'x', 'lift': 2.0, 'count': 3, 'p_b': 0.1},
        ])
        llm_t = pd.DataFrame([
            {'vaamr_stage': 0, 'vce_code': 'x', 'lift': 1.8, 'count': 3, 'p_b': 0.1},
        ])
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        self.assertTrue(out.loc[out['vce_code'] == 'x', 'both_elevated'].iloc[0])

    def test_not_elevated_when_one_below(self):
        gnn_t = pd.DataFrame([
            {'vaamr_stage': 0, 'vce_code': 'y', 'lift': 2.0, 'count': 3, 'p_b': 0.1},
        ])
        llm_t = pd.DataFrame([
            {'vaamr_stage': 0, 'vce_code': 'y', 'lift': 0.8, 'count': 3, 'p_b': 0.1},
        ])
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        self.assertFalse(out.loc[out['vce_code'] == 'y', 'both_elevated'].iloc[0])

    def test_empty_both_returns_empty_with_columns(self):
        gnn_t = pd.DataFrame(columns=['vaamr_stage', 'vce_code', 'lift', 'count', 'p_b'])
        llm_t = pd.DataFrame(columns=['vaamr_stage', 'vce_code', 'lift', 'count', 'p_b'])
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        self.assertIn('both_elevated', out.columns)
        self.assertEqual(len(out), 0)

    def test_fillna_zero_for_missing_pairs(self):
        """If a (stage, code) pair exists in GNN but not LLM, lift_llm should be 0."""
        gnn_t = pd.DataFrame([
            {'vaamr_stage': 0, 'vce_code': 'unique_code', 'lift': 1.9, 'count': 2, 'p_b': 0.1},
        ])
        llm_t = pd.DataFrame(columns=['vaamr_stage', 'vce_code', 'lift', 'count', 'p_b'])
        out = _lift.compare_gnn_vs_llm(gnn_t, llm_t)
        row = out[out['vce_code'] == 'unique_code']
        self.assertEqual(len(row), 1)
        self.assertAlmostEqual(float(row['lift_llm'].iloc[0]), 0.0)


if __name__ == '__main__':
    unittest.main()
