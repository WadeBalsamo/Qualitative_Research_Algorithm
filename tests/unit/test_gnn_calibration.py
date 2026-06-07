"""
tests/unit/test_gnn_calibration.py
----------------------------------
Unit tests for Track A3 — confidence calibration for domain shift.

Covers (hermetic): temperature fitting recovers an over-confidence scale and reduces
ECE; apply_temperature is a no-op when disabled; ECE behaves sanely; OOD scores rank a
far point above a near one; temperature_from_cv pairs held-out logits with the reference;
inference applies the configured temperature; crossval_predictions returns held-out logits.
"""

import os
import sys
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df
from gnn_layer.config import GnnLayerConfig
from gnn_layer.classifier import calibration as CAL


def _softmax(a):
    e = np.exp(a - a.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


class TestTemperature(unittest.TestCase):

    def test_recovers_overconfidence_scale_and_lowers_ece(self):
        rng = np.random.default_rng(1)
        N, C = 400, 5
        base = rng.standard_normal((N, C)).astype('float32')
        p = _softmax(base)
        y = np.array([rng.choice(C, p=p[i]) for i in range(N)])
        logits = base * 3.0  # over-sharpened → over-confident
        T = CAL.fit_temperature(logits, y)
        self.assertGreater(T, 1.5)
        ece_b = CAL.expected_calibration_error(_softmax(logits), y)
        ece_a = CAL.expected_calibration_error(_softmax(logits / T), y)
        self.assertLess(ece_a, ece_b)

    def test_apply_temperature_noop_when_none_or_nonpositive(self):
        import torch
        lt = torch.randn(3, 5)
        self.assertTrue(torch.equal(CAL.apply_temperature(lt, None), lt))
        self.assertTrue(torch.equal(CAL.apply_temperature(lt, 0.0), lt))

    def test_apply_temperature_divides(self):
        import torch
        lt = torch.ones(2, 3)
        out = CAL.apply_temperature(lt, 2.0)
        self.assertTrue(torch.allclose(out, torch.full((2, 3), 0.5)))

    def test_fit_temperature_degenerate_returns_one(self):
        self.assertEqual(CAL.fit_temperature(np.zeros((1, 5)), [0]), 1.0)
        # single-class reference is degenerate
        self.assertEqual(CAL.fit_temperature(np.zeros((4, 5)), [2, 2, 2, 2]), 1.0)

    def test_ece_none_on_empty(self):
        self.assertIsNone(CAL.expected_calibration_error(np.zeros((0, 5)), np.array([])))


class TestOOD(unittest.TestCase):

    def test_far_point_scores_higher(self):
        rng = np.random.default_rng(0)
        ref = rng.standard_normal((50, 8)).astype('float32')
        near = ref[0] + 0.01 * rng.standard_normal(8)
        far = rng.standard_normal(8) * 5 + 100
        s = CAL.ood_scores(np.vstack([near, far]).astype('float32'), ref, k=5)
        self.assertGreater(float(s[1]), float(s[0]))

    def test_empty_query_returns_empty(self):
        ref = np.zeros((5, 8), dtype='float32')
        self.assertEqual(CAL.ood_scores(np.zeros((0, 8), dtype='float32'), ref).shape, (0,))

    def test_empty_ref_returns_zeros(self):
        q = np.zeros((3, 8), dtype='float32')
        out = CAL.ood_scores(q, np.zeros((0, 8), dtype='float32'))
        self.assertEqual(out.shape, (3,))
        self.assertTrue(np.all(out == 0))


class TestCrossvalLogitsAndTempFromCv(unittest.TestCase):

    def _setup(self):
        from gnn_layer.classifier import graph_builder as gb, train as tr
        from gnn_layer.soft_labels import build_soft_targets
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6, validation_folds=2,
                             cache_embeddings=False, seed=1,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        df = synthetic_df(n_sessions=4)
        g = gb.build_graph(df, _seg_emb(df, seed=2), cfg)
        tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
        return g, tgts, cfg, df

    def test_crossval_returns_logits(self):
        from gnn_layer.classifier import train as tr
        g, tgts, cfg, df = self._setup()
        cv = tr.crossval_predictions(g, tgts, cfg, return_logits=True)
        self.assertIn('vaamr_logits', cv)
        self.assertTrue(len(cv['vaamr_logits']) > 0)
        sid, logit = cv['vaamr_logits'][0]
        self.assertEqual(len(np.asarray(logit)), 5)

    def test_temperature_from_cv_structure(self):
        from gnn_layer.classifier import train as tr
        g, tgts, cfg, df = self._setup()
        cv = tr.crossval_predictions(g, tgts, cfg, return_logits=True)
        cal = CAL.temperature_from_cv(cv, df)
        self.assertIn('temperature', cal)
        self.assertGreater(cal['temperature'], 0.0)
        self.assertEqual(cal['n'], len(cv['vaamr_logits']))

    def test_temperature_from_empty_cv_defaults_one(self):
        cal = CAL.temperature_from_cv({'vaamr_logits': []}, synthetic_df(n_sessions=1))
        self.assertEqual(cal['temperature'], 1.0)


class TestInferenceAppliesTemperature(unittest.TestCase):

    def test_temperature_changes_confidence(self):
        from gnn_layer.classifier import graph_builder as gb, train as tr, inference as inf
        from gnn_layer.soft_labels import build_soft_targets
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6,
                             cache_embeddings=False, seed=1,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        df = synthetic_df(n_sessions=3)
        g = gb.build_graph(df, _seg_emb(df), cfg)
        tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
        model, _ = tr.train_model(g, tgts, cfg)

        cfg.calibration_temperature = None
        hp0 = inf.infer_head_predictions(model, g, cfg)
        cfg.calibration_temperature = 5.0  # strong softening
        hp1 = inf.infer_head_predictions(model, g, cfg)
        # A higher temperature must not raise mean confidence (it softens it).
        self.assertLessEqual(np.mean(hp1['gnn_vaamr_conf']), np.mean(hp0['gnn_vaamr_conf']) + 1e-6)


if __name__ == '__main__':
    unittest.main()
