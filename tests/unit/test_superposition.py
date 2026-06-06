"""
tests/test_superposition.py
---------------------------
Unit tests for analysis/superposition.py — the unified VAAMR stage-mixture
provider that surfaces superposition from GNN positions → LLM ballots →
secondary_stage. No embedding model or LLM backend is touched.
"""

import json
import math
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.superposition import (
    attach_superposition,
    mixture_entropy,
    stage_cooccurrence_matrix,
    dominant_source,
)
from process import output_paths as _paths
from process.config import SuperpositionConfig


def _base_row(**kw):
    row = dict(
        segment_id='c1s1_0', participant_id='P1', session_id='c1s1',
        session_number=1, segment_index=0, word_count=30, text='hello',
        final_label=1, primary_stage=1, secondary_stage=np.nan,
        llm_confidence_primary=0.8, llm_confidence_secondary=np.nan,
        rater_votes=None,
    )
    row.update(kw)
    return row


class TestEntropy(unittest.TestCase):
    def test_pure_stage_zero_entropy(self):
        self.assertAlmostEqual(mixture_entropy([1, 0, 0, 0, 0]), 0.0, places=6)

    def test_uniform_max_entropy(self):
        self.assertAlmostEqual(mixture_entropy([0.2] * 5), 1.0, places=6)

    def test_two_point_between(self):
        e = mixture_entropy([0.5, 0.5, 0, 0, 0])
        self.assertTrue(0 < e < 1)
        # ln2 / ln5
        self.assertAlmostEqual(e, math.log(2) / math.log(5), places=6)


class TestSourcePriority(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_gnn_positions(self, mapping):
        d = _paths.gnn_data_dir(self.tmp)
        os.makedirs(d, exist_ok=True)
        rows = []
        for sid, mix in mapping.items():
            r = {'segment_id': sid, 'node_type': 'participant_segment',
                 'progression_coord': float(np.dot(np.arange(len(mix)), mix))}
            for k, v in enumerate(mix):
                r[f'vaamr_mix_{k}'] = v
            rows.append(r)
        pd.DataFrame(rows).to_csv(os.path.join(d, 'segment_positions.csv'), index=False)

    def test_gnn_takes_priority(self):
        self._write_gnn_positions({'c1s1_0': [0.1, 0.1, 0.6, 0.1, 0.1]})
        df = pd.DataFrame([_base_row(rater_votes=json.dumps([{'stage': 1, 'confidence': 1.0}]))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertEqual(df.iloc[0]['mixture_source'], 'gnn')
        self.assertEqual(df.iloc[0]['max_stage'], 2)  # GNN says stage 2 dominates

    def test_ballots_when_no_gnn(self):
        votes = [{'stage': 1, 'confidence': 0.8}, {'stage': 2, 'confidence': 0.6}]
        df = pd.DataFrame([_base_row(rater_votes=json.dumps(votes))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertEqual(df.iloc[0]['mixture_source'], 'ballots')
        mix = df.iloc[0]['mixture']
        self.assertAlmostEqual(sum(mix), 1.0, places=4)
        self.assertGreater(mix[1], 0)
        self.assertGreater(mix[2], 0)

    def test_secondary_fallback(self):
        df = pd.DataFrame([_base_row(secondary_stage=2, llm_confidence_secondary=0.5)])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertEqual(df.iloc[0]['mixture_source'], 'secondary')
        mix = df.iloc[0]['mixture']
        self.assertAlmostEqual(sum(mix), 1.0, places=4)
        # primary (1) heavier than secondary (2)
        self.assertGreater(mix[1], mix[2])

    def test_forced_ballots_mode(self):
        self._write_gnn_positions({'c1s1_0': [0.1, 0.1, 0.6, 0.1, 0.1]})
        votes = [{'stage': 0, 'confidence': 1.0}]
        df = pd.DataFrame([_base_row(rater_votes=json.dumps(votes))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig(mixture_source='ballots'))
        # GNN ignored in forced ballots mode
        self.assertEqual(df.iloc[0]['mixture_source'], 'ballots')
        self.assertEqual(df.iloc[0]['max_stage'], 0)


class TestColumnsAndLiminality(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_all_columns_present_and_normalized(self):
        votes = [{'stage': 1, 'confidence': 0.5}, {'stage': 2, 'confidence': 0.5}]
        df = pd.DataFrame([_base_row(rater_votes=json.dumps(votes))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        for col in ('mixture', 'progression_coord', 'mixture_entropy', 'max_stage',
                    'second_stage', 'n_active_stages', 'is_liminal', 'mixture_source'):
            self.assertIn(col, df.columns)
        self.assertAlmostEqual(sum(df.iloc[0]['mixture']), 1.0, places=4)

    def test_liminal_flag_on_split(self):
        # 50/50 split → small top1-top2 gap → liminal
        votes = [{'stage': 1, 'confidence': 0.5}, {'stage': 2, 'confidence': 0.5}]
        df = pd.DataFrame([_base_row(rater_votes=json.dumps(votes))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertTrue(bool(df.iloc[0]['is_liminal']))

    def test_not_liminal_on_pure(self):
        votes = [{'stage': 1, 'confidence': 1.0}]
        df = pd.DataFrame([_base_row(rater_votes=json.dumps(votes))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertFalse(bool(df.iloc[0]['is_liminal']))
        self.assertEqual(df.iloc[0]['n_active_stages'], 1)

    def test_cooccurrence_matrix_shape(self):
        votes = [{'stage': 1, 'confidence': 0.5}, {'stage': 2, 'confidence': 0.5}]
        df = pd.DataFrame([_base_row(rater_votes=json.dumps(votes))])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        mat = stage_cooccurrence_matrix(df, 5)
        self.assertEqual(len(mat), 5)
        self.assertEqual(len(mat[0]), 5)
        # off-diagonal mass on the 1↔2 cusp
        self.assertGreater(mat[1][2], 0)

    def test_dominant_source(self):
        df = pd.DataFrame([_base_row(secondary_stage=2, llm_confidence_secondary=0.4)])
        attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertEqual(dominant_source(df), 'secondary')

    def test_empty_df_noop(self):
        df = pd.DataFrame()
        out = attach_superposition(df, self.tmp, config=SuperpositionConfig())
        self.assertEqual(len(out), 0)


if __name__ == '__main__':
    unittest.main()
