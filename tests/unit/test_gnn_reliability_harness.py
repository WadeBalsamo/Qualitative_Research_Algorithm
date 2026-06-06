"""
tests/unit/test_gnn_reliability_harness.py
------------------------------------------
Hermetic tests for the GNN reliability battery harness
(experiments/gnn_reliability/harness.py).

Covers the parts every arm depends on:
  * build_folds — participant-grouped (no participant in two folds) + full coverage
  * score_arm   — LLM/human κ wiring (vs an independent cohen_kappa), the 6-class
                  No-code->-1 mapping, and the ledger row append (to a temp ledger)
  * run_gnn_arm — a tiny end-to-end grouped-CV smoke (random embeddings + torch)

No network, no model downloads: human codes are monkeypatched and embeddings are
random vectors.
"""

import os
import sys
import tempfile
import unittest
from unittest import mock

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

import numpy as np
import pandas as pd

from experiments.gnn_reliability import harness
from analysis import irr_stats


def _consensus(segment_id, primary, source='unanimous'):
    return {'worksheet_n': 1, 'item_num': 1, 'segment_id': segment_id,
            'rater': '__consensus__', 'primary': primary, 'secondary': None,
            'is_consensus': True, 'source': source, 'notes': None}


class TestBuildFolds(unittest.TestCase):
    def _df(self, n_participants=8, per=3):
        rows = []
        for p in range(n_participants):
            for j in range(per):
                rows.append({
                    'segment_id': f'p{p}_s{j}',
                    'participant_id': f'P{p}',
                    'speaker': 'participant',
                    'text': f'text {p} {j}',
                    'final_label': (p + j) % 2,  # 2 classes spread across participants
                })
        return pd.DataFrame(rows)

    def test_grouped_and_covers_all_labeled(self):
        df = self._df()
        folds = harness.build_folds(df, n_folds=4, seed=42, verbose=False)
        # every labeled participant segment got a fold
        self.assertEqual(len(folds), len(df))
        # participant-grouping: no participant spans two folds
        part_folds = {}
        for sid, f in folds.items():
            p = sid.split('_')[0]
            part_folds.setdefault(p, set()).add(f)
        for p, fs in part_folds.items():
            self.assertEqual(len(fs), 1, f"participant {p} leaked across folds {fs}")
        self.assertTrue(all(0 <= f < 4 for f in folds.values()))

    def test_deterministic(self):
        df = self._df()
        a = harness.build_folds(df, n_folds=4, seed=42, verbose=False)
        b = harness.build_folds(df, n_folds=4, seed=42, verbose=False)
        self.assertEqual(a, b)


class TestScoreArm(unittest.TestCase):
    def _df(self):
        return pd.DataFrame({
            'segment_id': ['s1', 's2', 's3', 's4'],
            'participant_id': ['p1', 'p1', 'p2', 'p2'],
            'speaker': ['participant'] * 4,
            'text': ['a', 'b', 'c', 'd'],
            'final_label': [2, 0, 4, 1],
        })

    def test_llm_and_human_axes_match_independent_kappa(self):
        df = self._df()
        codes = [_consensus('s1', 2), _consensus('s3', -1),
                 _consensus('s2', 0, source='unresolved')]  # unresolved excluded
        oof = {'s1': 2, 's2': 0, 's3': 2, 's4': 1}
        tmp_ledger = os.path.join(tempfile.mkdtemp(), 'ledger.csv')
        with mock.patch('process.irr_import.read_human_codes', return_value=codes), \
                mock.patch.object(harness, 'LEDGER_PATH', tmp_ledger):
            res = harness.score_arm('UNIT', oof, df, 'ignored', n_classes=5,
                                    meta={'embedding': 'minilm', 'embed_dim': 8,
                                          'method': 'GraphSAGE', 'imbalance': 'none',
                                          'seed': 42, 'branch': 'test'})
        # LLM axis: oof vs final_label over all 4 labeled
        exp_llm = irr_stats.cohen_kappa([2, 0, 2, 1], [2, 0, 4, 1])
        self.assertAlmostEqual(res['llm_axis']['cohen_kappa_205'], exp_llm, places=10)
        self.assertEqual(res['llm_axis']['n'], 4)
        # Human axis: usable consensus s1 (h=2,m=2) + s3 (h=-1,m=2); unresolved excluded
        exp_hum = irr_stats.cohen_kappa([2, 2], [2, -1])
        self.assertAlmostEqual(res['human_axis']['cohen_kappa'], exp_hum, places=10)
        self.assertEqual(res['human_axis']['n'], 2)
        # ledger row written with the headline κ
        self.assertTrue(os.path.isfile(tmp_ledger))
        led = pd.read_csv(tmp_ledger)
        self.assertEqual(list(led.columns), harness.LEDGER_COLUMNS)
        self.assertEqual(led.iloc[0]['arm'], 'UNIT')
        self.assertAlmostEqual(float(led.iloc[0]['gnn_human_kappa']), exp_hum, places=3)

    def test_six_class_maps_nocode_pred(self):
        df = self._df()
        codes = [_consensus('s1', 2), _consensus('s3', -1)]  # coded + No-code
        oof = {'s1': 2, 's3': 5}                 # 6-class GNN predicts class 5 (No-code) for s3
        with mock.patch('process.irr_import.read_human_codes', return_value=codes):
            res = harness.score_arm('UNIT6', oof, df, 'ignored', n_classes=6,
                                    meta={}, write_ledger=False)
        # s3 predicted 5 -> -1 matches human -1; s1 2 matches 2 -> perfect agreement
        self.assertEqual(res['human_axis']['n'], 2)
        self.assertEqual(res['human_axis']['cohen_kappa'], 1.0)

    def test_deferred_when_no_oof(self):
        df = self._df()
        codes = [_consensus('s1', 2), _consensus('s3', -1)]
        oof = {'s1': 2}                          # s3 has no out-of-fold prediction
        with mock.patch('process.irr_import.read_human_codes', return_value=codes):
            res = harness.score_arm('UNITD', oof, df, 'ignored', n_classes=5,
                                    meta={}, write_ledger=False)
        self.assertEqual(res['human_axis']['n'], 1)
        self.assertEqual(res['human_axis']['n_deferred'], 1)


class TestRunGnnArmSmoke(unittest.TestCase):
    def test_grouped_cv_returns_inrange_predictions(self):
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest('torch unavailable')
        from gnn_layer.config import GnnLayerConfig

        rng = np.random.default_rng(0)
        rows, emb = [], {}
        for p in range(4):
            for j in range(4):
                sid = f'p{p}_s{j}'
                rows.append({'segment_id': sid, 'participant_id': f'P{p}',
                             'session_id': f'sess{p}', 'speaker': 'participant',
                             'text': f't{p}{j}', 'final_label': (p + j) % 5,
                             'rater_votes': None})
                emb[sid] = rng.standard_normal(16).astype('float32')
        df = pd.DataFrame(rows)
        folds = harness.build_folds(df, n_folds=4, seed=42, verbose=False)
        cfg = GnnLayerConfig(knn_k=3, hidden_dim=8, n_layers=1, epochs=5,
                             patience=5, seed=42)
        oof = harness.run_gnn_arm(df, emb, folds, cfg)
        # every labeled participant segment got an out-of-fold prediction in range
        self.assertEqual(set(oof.keys()), set(df['segment_id']))
        self.assertTrue(all(0 <= v < 5 for v in oof.values()))


if __name__ == '__main__':
    unittest.main()
