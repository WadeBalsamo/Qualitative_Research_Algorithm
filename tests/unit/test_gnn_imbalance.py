"""
tests/unit/test_gnn_imbalance.py
--------------------------------
Hermetic tests for the VAAMR class-imbalance handling and the optional 6th
"No code" class added to the GNN soft-VAAMR head (design_decisions.md §1B/§4/§6,
arms A4 / A4n).

All four flags (``vaamr_n_classes`` / ``vaamr_class_balance`` /
``vaamr_focal_gamma`` / ``vaamr_hard_ce_weight``) plus the stretch ``vaamr_tam``
default to a no-op, so the default soft-VAAMR loss and head must be byte-identical
to the original unweighted 5-class path — the first test is the regression guard.

No network, no model downloads: synthetic embeddings / fabricated logits only.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from gnn_layer.config import GnnLayerConfig
from gnn_layer.model import build_model, compute_losses
from gnn_layer import soft_labels, graph_builder
from gnn_layer.train import assemble_targets, train_model

from tests.testhelpers.fixtures import synthetic_df


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _one_hot_rows(classes, n_classes):
    """[N, C] one-hot float matrix from a list of integer class indices."""
    m = torch.zeros((len(classes), n_classes), dtype=torch.float32)
    for i, c in enumerate(classes):
        m[i, int(c)] = 1.0
    return m


def _call_soft_vaamr(logits, mix, **flags):
    """Run compute_losses with only the soft_vaamr term active (+ optional hard CE).

    ``vaamr_idx`` is the identity over the rows, so outputs['soft_vaamr'][idx] == logits.
    """
    cfg = GnnLayerConfig(objectives=['soft_vaamr'], **flags)
    outputs = {'emb': torch.zeros((logits.shape[0], 4)), 'soft_vaamr': logits}
    targets = {'vaamr_idx': torch.arange(logits.shape[0]), 'vaamr_mix': mix}
    return compute_losses(outputs, targets, cfg)


# --------------------------------------------------------------------------- #
# 0. config defaults are the no-op
# --------------------------------------------------------------------------- #
class TestImbalanceConfigDefaults(unittest.TestCase):
    def test_defaults_are_noop(self):
        c = GnnLayerConfig()
        self.assertEqual(c.vaamr_n_classes, 5)
        self.assertFalse(c.vaamr_class_balance)
        self.assertEqual(c.vaamr_focal_gamma, 0.0)
        self.assertEqual(c.vaamr_hard_ce_weight, 0.0)
        self.assertFalse(c.vaamr_tam)


# --------------------------------------------------------------------------- #
# 1. REGRESSION GUARD — default path is byte-identical to the original KL
# --------------------------------------------------------------------------- #
class TestDefaultPathByteIdentical(unittest.TestCase):
    def test_compute_losses_matches_handcomputed_batchmean_kl(self):
        df = synthetic_df(n_sessions=1)
        rng = np.random.default_rng(0)
        seg_emb = {sid: rng.standard_normal(16).astype('float32')
                   for sid in df['segment_id']}
        cfg = GnnLayerConfig(enabled=True, hidden_dim=8, n_layers=2, knn_k=2,
                             cache_embeddings=False, objectives=['soft_vaamr'])
        g = graph_builder.build_graph(df, seg_emb, cfg)
        soft = soft_labels.build_soft_targets(df, label_mode='weak')  # default n_stages=5
        targets = assemble_targets(g, soft, cfg, df_all=df)
        self.assertGreater(int(targets['vaamr_idx'].numel()), 0)

        model = build_model(g, cfg)
        model.eval()
        with torch.no_grad():
            out = model(g.x, g.edge_index, g.edge_weight)

        # Hand-computed reference = the exact pre-change expression.
        idx = targets['vaamr_idx']
        logp = F.log_softmax(out['soft_vaamr'][idx], dim=1)
        ref = F.kl_div(logp, targets['vaamr_mix'], reduction='batchmean')

        got = compute_losses(out, targets, cfg)
        self.assertTrue(torch.equal(got['soft_vaamr'], ref))     # bit-identical
        self.assertNotIn('vaamr_hard_ce', got)                   # no aux term by default
        # default head is 5-wide
        self.assertEqual(model.heads['soft_vaamr'].out_features, 5)


# --------------------------------------------------------------------------- #
# 2. class balancing up-weights the rare class
# --------------------------------------------------------------------------- #
class TestClassBalance(unittest.TestCase):
    def setUp(self):
        # 4 rows of class 0 (common), 1 row of class 1 (rare).
        self.classes = [0, 0, 0, 0, 1]
        self.mix = _one_hot_rows(self.classes, 5)
        # Common rows: confidently correct (low KL). Rare row: confidently WRONG (high KL).
        self.logits = torch.tensor([
            [5., 0., 0., 0., 0.],
            [5., 0., 0., 0., 0.],
            [5., 0., 0., 0., 0.],
            [5., 0., 0., 0., 0.],
            [5., 0., 0., 0., 0.],   # true class is 1, but mass is on 0 → high KL
        ])

    def _expected_weighted(self):
        logp = F.log_softmax(self.logits, dim=1)
        kl_row = F.kl_div(logp, self.mix, reduction='none').sum(dim=1)
        labels = torch.tensor(self.classes)
        counts = torch.bincount(labels, minlength=5).float()
        inv = torch.zeros(5)
        inv[counts > 0] = len(self.classes) / counts[counts > 0]
        w = inv[labels]
        w = w / w.mean()
        return w, kl_row, float((w * kl_row).mean())

    def test_matches_documented_weighting_formula(self):
        w, _, expected = self._expected_weighted()
        got = _call_soft_vaamr(self.logits, self.mix, vaamr_class_balance=True)
        self.assertAlmostEqual(float(got['soft_vaamr']), expected, places=6)
        # the rare class (1, count=1) is up-weighted above the common class (0, count=4)
        self.assertGreater(float(w[4]), float(w[0]))
        self.assertGreater(float(w[4]), 1.0)

    def test_rare_high_kl_row_raises_total_vs_unweighted(self):
        unweighted = float(_call_soft_vaamr(self.logits, self.mix)['soft_vaamr'])
        weighted = float(_call_soft_vaamr(
            self.logits, self.mix, vaamr_class_balance=True)['soft_vaamr'])
        # the rare row carries the largest KL and gets the largest weight → loss rises
        self.assertGreater(weighted, unweighted)

    def test_balanced_mix_is_unchanged(self):
        # When every class is equally frequent, the inverse-frequency weights are all 1
        # → the weighted KL collapses back to the plain batchmean KL.
        mix = _one_hot_rows([0, 1, 2, 3, 4], 5)
        logits = torch.randn(5, 5, generator=torch.Generator().manual_seed(1))
        plain = float(_call_soft_vaamr(logits, mix)['soft_vaamr'])
        bal = float(_call_soft_vaamr(logits, mix, vaamr_class_balance=True)['soft_vaamr'])
        self.assertAlmostEqual(plain, bal, places=6)


# --------------------------------------------------------------------------- #
# 3. focal modulation shrinks loss on a confidently-correct row
# --------------------------------------------------------------------------- #
class TestFocal(unittest.TestCase):
    def test_focal_reduces_confident_correct_loss(self):
        mix = _one_hot_rows([0], 5)
        logits = torch.tensor([[4., 0., 0., 0., 0.]])   # softmax ~0.93 on the true class
        l0 = float(_call_soft_vaamr(logits, mix, vaamr_focal_gamma=0.0)['soft_vaamr'])
        l2 = float(_call_soft_vaamr(logits, mix, vaamr_focal_gamma=2.0)['soft_vaamr'])
        self.assertLess(l2, l0)
        self.assertGreater(l2, 0.0)

    def test_gamma_zero_is_fast_path(self):
        mix = _one_hot_rows([0, 1], 5)
        logits = torch.randn(2, 5, generator=torch.Generator().manual_seed(2))
        # gamma 0.0 with no other flags → identical to the plain KL.
        plain = float(_call_soft_vaamr(logits, mix)['soft_vaamr'])
        g0 = float(_call_soft_vaamr(logits, mix, vaamr_focal_gamma=0.0)['soft_vaamr'])
        self.assertAlmostEqual(plain, g0, places=7)


# --------------------------------------------------------------------------- #
# 4. auxiliary hard-label CE adds a positive term on top of the soft KL
# --------------------------------------------------------------------------- #
class TestHardCE(unittest.TestCase):
    def test_hard_ce_adds_positive_term(self):
        mix = _one_hot_rows([0, 0, 1], 5)
        logits = torch.tensor([
            [2., 0., 0., 0., 0.],
            [1., 0.5, 0., 0., 0.],
            [0., 1., 0., 0., 0.],
        ])
        soft_only = _call_soft_vaamr(logits, mix)
        with_ce = _call_soft_vaamr(logits, mix, vaamr_hard_ce_weight=2.0)

        self.assertNotIn('vaamr_hard_ce', soft_only)
        self.assertIn('vaamr_hard_ce', with_ce)
        self.assertGreater(float(with_ce['vaamr_hard_ce']), 0.0)
        # the soft KL term itself is unchanged; only an extra term is ADDED
        self.assertAlmostEqual(float(with_ce['soft_vaamr']),
                               float(soft_only['soft_vaamr']), places=6)
        self.assertGreater(float(with_ce['total']), float(soft_only['total']))

        # the term equals weight * class-weighted CE(argmax target)
        labels = torch.tensor([0, 0, 1])
        counts = torch.bincount(labels, minlength=5).float()
        w = torch.zeros(5)
        w[counts > 0] = 3.0 / counts[counts > 0]
        expected = 2.0 * float(F.cross_entropy(logits, labels, weight=w))
        self.assertAlmostEqual(float(with_ce['vaamr_hard_ce']), expected, places=5)


# --------------------------------------------------------------------------- #
# 5. stretch — logit adjustment (TAM)
# --------------------------------------------------------------------------- #
class TestTAM(unittest.TestCase):
    def test_tam_changes_loss_and_is_finite(self):
        mix = _one_hot_rows([0, 0, 0, 0, 1], 5)   # imbalanced → priors differ
        logits = torch.randn(5, 5, generator=torch.Generator().manual_seed(5))
        plain = float(_call_soft_vaamr(logits, mix)['soft_vaamr'])
        tam = float(_call_soft_vaamr(logits, mix, vaamr_tam=True)['soft_vaamr'])
        self.assertTrue(np.isfinite(tam))
        self.assertNotAlmostEqual(plain, tam, places=4)

    def test_tam_composes_with_balance_and_focal(self):
        mix = _one_hot_rows([0, 0, 0, 1, 2], 5)
        logits = torch.randn(5, 5, generator=torch.Generator().manual_seed(6))
        out = _call_soft_vaamr(logits, mix, vaamr_tam=True,
                               vaamr_class_balance=True, vaamr_focal_gamma=1.5)
        self.assertTrue(np.isfinite(float(out['soft_vaamr'])))
        self.assertGreater(float(out['soft_vaamr']), 0.0)


# --------------------------------------------------------------------------- #
# 6. the optional 6th "No code" class
# --------------------------------------------------------------------------- #
class TestSixClassNoCode(unittest.TestCase):
    def test_build_soft_targets_six_class(self):
        df = pd.DataFrame([
            # labeled participant (stage 2) with concentrated ballots
            dict(segment_id='p_lab', speaker='participant', final_label=2,
                 primary_stage=2, rater_votes=[{'stage': 2, 'confidence': 0.9}]),
            # No-code participant: final_label null, no usable ballots
            dict(segment_id='p_nocode', speaker='participant', final_label=np.nan,
                 primary_stage=np.nan, rater_votes=None),
            # therapist row is ignored by build_soft_targets
            dict(segment_id='t0', speaker='therapist', final_label=np.nan,
                 primary_stage=np.nan, rater_votes=None),
        ])

        soft6 = soft_labels.build_soft_targets(df, label_mode='weak', n_stages=6)
        self.assertIn('p_lab', soft6)
        self.assertIn('p_nocode', soft6)
        self.assertNotIn('t0', soft6)

        lab = np.asarray(soft6['p_lab'])
        noc = np.asarray(soft6['p_nocode'])
        self.assertEqual(lab.shape, (6,))
        self.assertEqual(noc.shape, (6,))
        # labeled row keeps its 5-stage mixture in dims 0..4, dim 5 == 0
        self.assertEqual(float(lab[5]), 0.0)
        self.assertAlmostEqual(float(lab[:5].sum()), 1.0, places=6)
        self.assertAlmostEqual(float(lab[2]), 1.0, places=6)
        # No-code row → clean one-hot on class 5 (was uniform noise before)
        self.assertTrue(np.allclose(noc, [0, 0, 0, 0, 0, 1]))

        # 5-class path is unchanged: the No-code row still falls through to uniform.
        soft5 = soft_labels.build_soft_targets(df, label_mode='weak', n_stages=5)
        self.assertTrue(np.allclose(np.asarray(soft5['p_nocode']), 0.2))
        self.assertAlmostEqual(float(np.asarray(soft5['p_lab'])[2]), 1.0, places=6)

    def test_build_model_six_class_head(self):
        df = synthetic_df(n_sessions=1)
        rng = np.random.default_rng(0)
        seg_emb = {sid: rng.standard_normal(16).astype('float32')
                   for sid in df['segment_id']}
        cfg5 = GnnLayerConfig(enabled=True, hidden_dim=8, knn_k=2, cache_embeddings=False,
                              objectives=['soft_vaamr'])
        cfg6 = GnnLayerConfig(enabled=True, hidden_dim=8, knn_k=2, cache_embeddings=False,
                              objectives=['soft_vaamr'], vaamr_n_classes=6)
        g = graph_builder.build_graph(df, seg_emb, cfg5)
        self.assertEqual(build_model(g, cfg5).heads['soft_vaamr'].out_features, 5)
        self.assertEqual(build_model(g, cfg6).heads['soft_vaamr'].out_features, 6)

    def test_end_to_end_train_predicts_zero_to_five(self):
        df = synthetic_df(n_sessions=2)
        # turn one participant row into a No-code row (null final_label).
        mask = df['segment_id'] == 'c1s1_0'
        df.loc[mask, 'final_label'] = np.nan
        df.loc[mask, 'primary_stage'] = np.nan
        rng = np.random.default_rng(3)
        seg_emb = {sid: rng.standard_normal(16).astype('float32')
                   for sid in df['segment_id']}
        cfg = GnnLayerConfig(enabled=True, hidden_dim=8, n_layers=2, knn_k=3, epochs=10,
                             cache_embeddings=False, seed=1, vaamr_n_classes=6,
                             vaamr_class_balance=True, vaamr_focal_gamma=1.0,
                             device='cpu',  # hermetic: keep train + forward on one device
                             objectives=['soft_vaamr', 'progression'])
        g = graph_builder.build_graph(df, seg_emb, cfg)
        soft = soft_labels.build_soft_targets(df, label_mode='weak', n_stages=6)
        targets = assemble_targets(g, soft, cfg, df_all=df)

        # No-code row carries a real (one-hot class 5) target now, so it is supervised.
        self.assertEqual(int(targets['vaamr_mix'].shape[1]), 6)
        nocode_pos = [i for i, gi in enumerate(targets['vaamr_idx'].tolist())
                      if g.node_ids[gi] == 'c1s1_0']
        self.assertEqual(len(nocode_pos), 1)
        self.assertEqual(int(targets['vaamr_mix'][nocode_pos[0]].argmax()), 5)
        # progression target for the No-code row is 0.0 (E[stage] over the 5 real dims).
        self.assertAlmostEqual(float(targets['prog_val'][nocode_pos[0]]), 0.0, places=6)

        model, metrics = train_model(g, targets, cfg)
        with torch.no_grad():
            out = model(g.x, g.edge_index, g.edge_weight)
        preds = out['soft_vaamr'].argmax(dim=1).tolist()
        self.assertEqual(out['soft_vaamr'].shape[1], 6)
        self.assertTrue(all(0 <= p <= 5 for p in preds))


if __name__ == '__main__':
    unittest.main()
