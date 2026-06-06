"""
tests/test_gnn_consensus.py
---------------------------
Tests for the graph-distilled consensus layer: the gnn_labels overlay, the
gnn_consensus provenance tier (gated on gnn_authoritative), the out-of-sample
reliability gate (per-stage κ + rare-stage check), and LLM-free scale-mode
classification.
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment
from process import classifications_io as cio
from process.assembly.master_dataset import assemble_master_dataset
from tests.testhelpers import slow_test


def _mkdir_overlay_root():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, '02_meta', 'classifications'), exist_ok=True)
    return d


class TestGnnOverlay(unittest.TestCase):
    def test_roundtrip_and_merge(self):
        d = _mkdir_overlay_root()
        p = Segment(segment_id='p1', text='x', speaker='participant')
        p.gnn_vaamr_pred, p.gnn_vaamr_conf, p.gnn_label_source = 3, 0.8, 'gnn_trained'
        t = Segment(segment_id='t1', text='y', speaker='therapist')
        t.gnn_purer_pred, t.gnn_purer_conf, t.gnn_label_source = 2, 0.7, 'gnn_trained'
        cio.write_gnn_overlay(d, [p, t])

        by_id = {'p1': Segment(segment_id='p1'), 't1': Segment(segment_id='t1')}
        n = cio.apply_gnn_overlay(d, by_id)
        self.assertEqual(n, 2)
        self.assertEqual(by_id['p1'].gnn_vaamr_pred, 3)
        self.assertEqual(by_id['t1'].gnn_purer_pred, 2)

        # merge upsert
        p.gnn_vaamr_pred = 4
        cio.merge_gnn_overlay(d, [p])
        by_id2 = {'p1': Segment(segment_id='p1'), 't1': Segment(segment_id='t1')}
        cio.apply_gnn_overlay(d, by_id2)
        self.assertEqual(by_id2['p1'].gnn_vaamr_pred, 4)
        self.assertEqual(by_id2['t1'].gnn_purer_pred, 2)  # untouched by merge

    def test_gnn_in_overlay_registry(self):
        self.assertIn('gnn', cio.OVERLAY_KEYS)
        self.assertIn('gnn', cio.OVERLAY_FILENAMES)


class TestProvenanceTier(unittest.TestCase):
    def setUp(self):
        self.p = Segment(segment_id='p1', text='x', speaker='participant')
        self.p.primary_stage, self.p.gnn_vaamr_pred, self.p.gnn_vaamr_conf = 1, 3, 0.9
        self.t = Segment(segment_id='t1', text='y', speaker='therapist')
        self.t.purer_primary, self.t.gnn_purer_pred, self.t.gnn_purer_conf = 0, 2, 0.8
        self.out = os.path.join(tempfile.mkdtemp(), 'm.jsonl')

    def test_authoritative_off_keeps_llm(self):
        df = assemble_master_dataset([self.p, self.t], self.out, gnn_authoritative=False)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label'], 1)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'llm_zero_shot')
        self.assertEqual(r.loc['t1', 'purer_final'], 0)
        self.assertEqual(r.loc['t1', 'purer_final_source'], 'llm_zero_shot')

    def test_authoritative_on_uses_gnn_and_preserves_raw(self):
        # Promotion requires BOTH the operator opt-in AND a passing gate (Track 0.2).
        df = assemble_master_dataset([self.p, self.t], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label'], 3)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'gnn_consensus')
        self.assertEqual(r.loc['t1', 'purer_final'], 2)
        self.assertEqual(r.loc['t1', 'purer_final_source'], 'gnn_consensus')
        # raw LLM labels preserved
        self.assertEqual(r.loc['p1', 'primary_stage'], 1)
        self.assertEqual(r.loc['t1', 'purer_primary'], 0)

    def test_authoritative_on_but_gate_not_passed_keeps_llm(self):
        # The gate is the hard precondition: a config flag alone can NOT promote an
        # un-gated graph to the label of record (Track 0.2 — gate-gated promotion).
        df = assemble_master_dataset([self.p, self.t], self.out,
                                     gnn_authoritative=True, gate_passed=False)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label'], 1)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'llm_zero_shot')
        self.assertEqual(r.loc['t1', 'purer_final'], 0)
        self.assertEqual(r.loc['t1', 'purer_final_source'], 'llm_zero_shot')

    def test_authoritative_defaults_gate_passed_false(self):
        # gate_passed defaults to False, so the safeguard holds even if a caller
        # forgets to thread the gate verdict through.
        df = assemble_master_dataset([self.p, self.t], self.out, gnn_authoritative=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label_source'], 'llm_zero_shot')
        self.assertEqual(r.loc['t1', 'purer_final_source'], 'llm_zero_shot')

    def test_human_outranks_gnn(self):
        self.p.human_label = 1  # equals primary_stage -> human_consensus
        df = assemble_master_dataset([self.p], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label_source'], 'human_consensus')
        self.assertEqual(r.loc['p1', 'final_label'], 1)


class TestReliabilityGate(unittest.TestCase):
    def test_per_class_and_verdict(self):
        from gnn_layer import validation as val
        from gnn_layer.config import GnnLayerConfig

        # Build a tiny df with known LLM labels; cv_preds reproduce most of them.
        rows = []
        for i in range(10):
            rows.append(dict(segment_id=f's{i}', final_label=i % 5,
                             purer_primary=np.nan, in_human_coded_subset=False,
                             human_label=np.nan))
        df = pd.DataFrame(rows)
        # graph predicts perfectly except it collapses stage 4 (Reappraisal) to 0
        preds = []
        for i in range(10):
            ref = i % 5
            preds.append((f's{i}', 0 if ref == 4 else ref))
        cv = {'vaamr': preds, 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = val.evaluate_crossval(df, cv, cfg)
        # per-class present for stage 4 with recall 0
        s4 = [r for r in m['vaamr_per_class'] if r['class_id'] == 4]
        self.assertTrue(s4)
        self.assertEqual(s4[0]['recall'], 0.0)
        # overall agreement is 8/10
        self.assertAlmostEqual(m['vaamr_overall']['percent_agreement'], 0.8, places=3)

    def test_reports_written(self):
        from gnn_layer import validation as val
        from gnn_layer.config import GnnLayerConfig
        df = pd.DataFrame([dict(segment_id=f's{i}', final_label=i % 2,
                                purer_primary=np.nan, in_human_coded_subset=False,
                                human_label=np.nan) for i in range(6)])
        cv = {'vaamr': [(f's{i}', i % 2) for i in range(6)], 'purer': []}
        m = val.evaluate_crossval(df, cv, GnnLayerConfig(irr_target=0.5))
        out = tempfile.mkdtemp()
        rp = val.write_validation_report(m, out)
        cp = val.write_validation_csv(m, out)
        self.assertTrue(os.path.isfile(rp))
        self.assertTrue(os.path.isfile(cp))
        self.assertIn('LLM-FREE SCALING?', open(rp).read())

    def test_gate_verdict_persisted_and_read_back(self):
        """write_gate_verdict persists ready_for_scaling; gate_ready_for_scaling reads it."""
        from gnn_layer import validation as val
        from gnn_layer.config import GnnLayerConfig
        # Perfect reproduction → gate passes.
        df = pd.DataFrame([dict(segment_id=f's{i}', final_label=i % 2,
                                purer_primary=np.nan, in_human_coded_subset=False,
                                human_label=np.nan) for i in range(6)])
        cv = {'vaamr': [(f's{i}', i % 2) for i in range(6)], 'purer': []}
        m = val.evaluate_crossval(df, cv, GnnLayerConfig(irr_target=0.5))
        out = tempfile.mkdtemp()
        gp = val.write_gate_verdict(m, out)
        self.assertTrue(os.path.isfile(gp))
        verdict = val.read_gate_verdict(out)
        self.assertTrue(verdict['ready_for_scaling'])
        self.assertTrue(val.gate_ready_for_scaling(out))

    def test_gate_ready_false_when_no_verdict_file(self):
        """A missing verdict file means not-ready (safe default — no promotion)."""
        from gnn_layer import validation as val
        out = tempfile.mkdtemp()
        self.assertIsNone(val.read_gate_verdict(out))
        self.assertFalse(val.gate_ready_for_scaling(out))

    def test_gate_verdict_records_failing_gate(self):
        """A failing gate (κ below target) persists ready_for_scaling=False."""
        from gnn_layer import validation as val
        from gnn_layer.config import GnnLayerConfig
        # Graph disagrees with LLM on half the rows → κ below a high target.
        df = pd.DataFrame([dict(segment_id=f's{i}', final_label=i % 2,
                                purer_primary=np.nan, in_human_coded_subset=False,
                                human_label=np.nan) for i in range(6)])
        cv = {'vaamr': [(f's{i}', (i + 1) % 2) for i in range(6)], 'purer': []}
        m = val.evaluate_crossval(df, cv, GnnLayerConfig(irr_target=0.9))
        out = tempfile.mkdtemp()
        val.write_gate_verdict(m, out)
        self.assertFalse(val.gate_ready_for_scaling(out))


class TestCrossvalHoldout(unittest.TestCase):
    def test_subset_targets_masks_rows(self):
        import torch
        from gnn_layer.train import _subset_targets
        targets = {
            'vaamr_idx': torch.tensor([10, 11, 12, 13]),
            'vaamr_mix': torch.zeros((4, 5)),
            'prog_val': torch.zeros(4),
            'contrast_idx': torch.tensor([10, 11, 12, 13]),
            'contrast_label': torch.tensor([0, 1, 2, 3]),
            'purer_idx': torch.tensor([20, 21]),
            'purer_label': torch.tensor([0, 1]),
        }
        sub = _subset_targets(targets, keep_v_pos=[0, 2], keep_p_pos=[1])
        self.assertEqual(sub['vaamr_idx'].tolist(), [10, 12])
        self.assertEqual(sub['contrast_label'].tolist(), [0, 2])
        self.assertEqual(sub['purer_idx'].tolist(), [21])


class TestScaleModeNoLLM(unittest.TestCase):
    """Graph classifies a new unlabeled segment with no LLM client present."""

    def setUp(self):
        from gnn_layer import embeddings as emb
        self._orig = emb.embed_segment_texts
        rng = np.random.default_rng(3)
        emb.embed_segment_texts = lambda texts, config: rng.standard_normal(
            (len(texts), 16)).astype('float32')

    def tearDown(self):
        from gnn_layer import embeddings as emb
        emb.embed_segment_texts = self._orig

    @slow_test
    def test_train_then_classify_new(self):
        from tests.testhelpers import fixtures as T
        from gnn_layer import runner
        from gnn_layer.config import GnnLayerConfig

        df = T.synthetic_df(n_sessions=3)
        cfg = GnnLayerConfig(enabled=True, hidden_dim=16, n_layers=2, knn_k=3, epochs=10,
                             n_motif_clusters=3, cache_embeddings=False, seed=1,
                             interpret_against_cf_ic=False, validation_folds=2,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        out = tempfile.mkdtemp()
        runner.run_gnn_analysis(df, out, config=cfg, verbose=False)
        self.assertTrue(os.path.isfile(os.path.join(out, '02_meta', 'gnn', 'weights.pt')))

        new = pd.DataFrame([
            dict(segment_id='cNEW_0', session_id='cNEW', speaker='participant',
                 text='a new participant turn', start_time_ms=1000, end_time_ms=1800,
                 final_label=np.nan, primary_stage=np.nan, purer_primary=np.nan),
        ])
        df2 = pd.concat([df, new], ignore_index=True)
        res = runner.run_gnn_classify(df2, out, config=cfg, verbose=False, only_unlabeled=True)
        self.assertEqual(res['status'], 'ok')
        self.assertGreaterEqual(res['n_classified'], 1)
        # the new segment got a graph label in the overlay
        by_id = {'cNEW_0': Segment(segment_id='cNEW_0')}
        cio.apply_gnn_overlay(out, by_id)
        self.assertIsNotNone(by_id['cNEW_0'].gnn_vaamr_pred)
        self.assertEqual(by_id['cNEW_0'].gnn_label_source, 'gnn_scale_mode')


if __name__ == '__main__':
    unittest.main()
