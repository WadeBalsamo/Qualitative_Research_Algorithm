"""
tests/unit/test_gnn_discriminant.py
-----------------------------------
Unit tests for WS1 — H6 discriminant validity (gnn_layer/discriminant.py).

Hermetic: crafts embeddings that are either stage-homophilous (same-stage segments cluster) or
content-random (stage independent of similarity), then exercises the pure geometry/statistic
functions and the report/CSV/figure writers. No model download, no network, no qra.db — the
harness-dependent arms/scoring are covered by the integration smoke run, not here. The one
harness-touching helper (_chance_oof) is guarded and skipped if experiments/ is unimportable.
numpy + sklearn only.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df
from gnn_layer import discriminant as D
from gnn_layer import figures as FIG


def _stage_emb(df, dim=32, homophily=True, seed=0):
    """Embeddings keyed on VAAMR stage (homophilous) or random (stage ⟂ similarity)."""
    rng = np.random.default_rng(seed)
    cent = (rng.standard_normal((6, dim)) * 5.0).astype('float32')
    emb = {}
    for _, r in df.iterrows():
        sid = str(r['segment_id'])
        fl = r.get('final_label')
        labeled = str(r.get('speaker')) == 'participant' and not (fl is None or (isinstance(fl, float) and np.isnan(fl)))
        if homophily and labeled:
            emb[sid] = cent[int(fl)] + (rng.standard_normal(dim) * 0.3).astype('float32')
        else:
            emb[sid] = rng.standard_normal(dim).astype('float32')
    return emb


class TestKnnHomophily(unittest.TestCase):

    def test_homophilous_embedding_has_high_lift(self):
        df = make_master_df(n_sessions=4, n_participants=8)
        emb = _stage_emb(df, homophily=True, seed=1)
        _, X_lab, y_lab, _ = D._participant_matrices(df, emb)
        hom = D._knn_stage_homophily(X_lab, y_lab, ks=(1, 5))
        self.assertTrue(hom['available'])
        # same-stage clustering -> nearest neighbour shares stage far above base rate
        self.assertGreater(hom['rows'][0]['lift_over_base'], 2.0)

    def test_content_random_embedding_near_chance(self):
        df = make_master_df(n_sessions=4, n_participants=8)
        emb = _stage_emb(df, homophily=False, seed=2)
        _, X_lab, y_lab, _ = D._participant_matrices(df, emb)
        hom = D._knn_stage_homophily(X_lab, y_lab, ks=(1, 5))
        self.assertTrue(hom['available'])
        # stage independent of similarity -> neighbours stage-mixed, lift ~ 1
        self.assertLess(hom['rows'][0]['lift_over_base'], 1.6)


class TestStageVarianceByPcs(unittest.TestCase):

    def test_returns_structure_and_recovers_when_separable(self):
        df = make_master_df(n_sessions=4, n_participants=8)
        emb = _stage_emb(df, homophily=True, seed=3)
        X_all, X_lab, y_lab, groups = D._participant_matrices(df, emb)
        res = D._stage_variance_by_pcs(X_all, X_lab, y_lab, groups, ks=(5, 10), seed=42)
        self.assertIn('pc_rows', res)
        self.assertTrue(res['pc_rows'])
        self.assertIn('full_embedding_kappa', res)
        self.assertIn('chance_modal_acc', res)
        # separable stage -> the full-embedding probe beats the most-frequent floor
        if res['full_embedding_kappa'] is not None:
            self.assertGreater(res['full_embedding_kappa'], 0.0)


class TestParticipantMatrices(unittest.TestCase):

    def test_shapes_and_l2_norm(self):
        df = make_master_df(n_sessions=3, n_participants=4)
        emb = _stage_emb(df, homophily=True, seed=4)
        out = D._participant_matrices(df, emb)
        self.assertIsNotNone(out)
        X_all, X_lab, y_lab, groups = out
        self.assertEqual(X_lab.shape[0], len(y_lab))
        self.assertEqual(len(groups), len(y_lab))
        self.assertTrue(np.allclose(np.linalg.norm(X_lab, axis=1), 1.0, atol=1e-5))
        self.assertTrue(set(y_lab.tolist()).issubset(set(range(5))))


class TestChanceOof(unittest.TestCase):

    def test_most_frequent_in_range(self):
        harness, baselines = D._load_harness()
        if baselines is None:
            self.skipTest("experiments.gnn_reliability.baselines unavailable")
        from gnn_layer.config import GnnLayerConfig
        df = make_master_df(n_sessions=4, n_participants=8)
        emb = _stage_emb(df, homophily=True, seed=5)
        lab = df[(df.speaker == 'participant') & (df.final_label.notna())]
        folds = {str(r['segment_id']): i % 5 for i, (_, r) in enumerate(lab.iterrows())}
        preds = D._chance_oof(baselines, df, emb, folds, GnnLayerConfig(vaamr_n_classes=5),
                              'most_frequent', seed=7)
        self.assertTrue(preds)
        self.assertTrue(all(0 <= v < 5 for v in preds.values()))


class TestCommunityStageDegenerateGuard(unittest.TestCase):

    def test_singleton_communities_report_degenerate(self):
        # random high-dim embeddings -> no cosine edges above threshold -> all singletons
        df = make_master_df(n_sessions=4, n_participants=6)
        emb = _stage_emb(df, homophily=False, dim=64, seed=6)
        from gnn_layer.config import GnnLayerConfig
        res = D._community_stage_independence(df, emb, GnnLayerConfig(), seed=42)
        self.assertFalse(res.get('available'))
        self.assertTrue(res.get('degenerate'))


class TestWriters(unittest.TestCase):

    def _result(self):
        return {
            'status': 'ok',
            'arms': {
                'H6-probe': {'human_axis': {'cohen_kappa': 0.365, 'ci95': [0.23, 0.51], 'n': 66},
                             'llm_axis': {'cohen_kappa_205': 0.283, 'ci95': [0.20, 0.35], 'n': 205}},
                'H6-content': {'human_axis': {'cohen_kappa': 0.196, 'ci95': [0.12, 0.32], 'n': 66},
                               'llm_axis': {'cohen_kappa_205': 0.069, 'ci95': [0.01, 0.14], 'n': 205}},
                'H6-chance-mode': {'human_axis': {'cohen_kappa': 0.0, 'ci95': [0.0, 0.0], 'n': 66},
                                   'llm_axis': {'cohen_kappa_205': 0.0, 'ci95': [0.0, 0.0], 'n': 205}},
            },
            'contrast': {'human': {'delta': 0.17, 'lo': 0.0, 'hi': 0.32, 'n': 66, 'n_clusters': 15},
                         'llm': {'delta': 0.21, 'lo': 0.15, 'hi': 0.27, 'n': 205, 'n_clusters': 20}},
            'geometry': {
                'variance': {'pc_rows': [{'k': 10, 'cum_var': 0.30, 'stage_kappa_from_pcs': 0.25}],
                             'full_embedding_kappa': 0.31, 'chance_modal_acc': 0.36},
                'homophily': {'available': True, 'base_rate': 0.2, 'per_stage_k': 5,
                              'per_stage': {0: 0.3, 1: 0.2},
                              'rows': [{'k': 1, 'mean_same_stage_frac': 0.4, 'base_rate': 0.2,
                                        'lift_over_base': 2.0}]},
                'community_stage': {'available': False, 'degenerate': True, 'threshold': 0.55},
            },
            'operationalization': {'human_nocode': 24, 'human_total': 66, 'human_nocode_frac': 0.36,
                                   'n_nolabel': 134, 'n_participant': 339, 'corpus_nolabel_frac': 0.40},
        }

    def test_csv_and_report_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = D.write_discriminant_csv(self._result(), tmp)
            self.assertTrue(csv and os.path.isfile(csv))
            df = pd.read_csv(csv)
            self.assertIn('arm', set(df['section']))
            self.assertIn('knn_homophily', set(df['section']))

            rep = D.write_discriminant_report(self._result(), tmp)
            self.assertTrue(os.path.isfile(rep))
            text = open(rep, encoding='utf-8').read()
            self.assertIn('H6', text)
            self.assertIn('homophily', text.lower())
            self.assertIn('probe', text.lower())

    def test_pca_coords_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            coords = {'pc1': [0.1, 0.2], 'pc2': [0.3, 0.4], 'stage': [0, 4],
                      'stage_name': ['Vigilance', 'Reappraisal']}
            p = D.write_pca_coords_csv(coords, tmp)
            self.assertTrue(p and os.path.isfile(p))
            self.assertEqual(len(pd.read_csv(p)), 2)


class TestFigureGuard(unittest.TestCase):

    def test_returns_none_without_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(FIG.plot_discriminant_validity(tmp))


if __name__ == '__main__':
    unittest.main()
