"""
tests/unit/test_gnn_abstention.py
---------------------------------
Unit tests for Track A2 — abstention / deferral.

A confident-wrong graph label on a novel segment would poison the MindfulBERT
training set, so the graph must be able to defer ("ABSTAIN") below a per-stage
confidence floor and let master_dataset keep the LLM label. Covers (all hermetic):

  * floor resolution precedence (off / global / rare-stage / explicit per-stage)
  * infer_head_predictions emits abstain flags only when configured; a high floor
    abstains everything, a zero floor abstains nothing
  * the gnn overlay round-trips the abstain flags
  * master_dataset keeps the LLM label for abstained segments even when the graph is
    authoritative + gate-passed; non-abstained predictions still promote
  * calibrate_abstain_floors returns clamped per-stage floors for all 5 stages
"""

import os
import sys
import tempfile
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df
from gnn_layer.config import GnnLayerConfig


def _seg_emb(df, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    return {sid: rng.standard_normal(dim).astype('float32') for sid in df['segment_id']}


class TestFloorResolution(unittest.TestCase):

    def test_disabled_by_default(self):
        from gnn_layer.inference import resolve_abstain_floors, resolve_purer_abstain_floor
        cfg = GnnLayerConfig()
        self.assertIsNone(resolve_abstain_floors(cfg))
        self.assertIsNone(resolve_purer_abstain_floor(cfg))

    def test_global_floor_applies_to_all_stages(self):
        from gnn_layer.inference import resolve_abstain_floors, resolve_purer_abstain_floor
        cfg = GnnLayerConfig(abstain_threshold=0.5)
        floors = resolve_abstain_floors(cfg)
        self.assertEqual(set(floors), set(range(5)))
        self.assertTrue(all(v == 0.5 for v in floors.values()))
        self.assertEqual(resolve_purer_abstain_floor(cfg), 0.5)

    def test_rare_stage_floor_overrides_common(self):
        from gnn_layer.inference import resolve_abstain_floors
        cfg = GnnLayerConfig(abstain_threshold=0.5, abstain_rare_stage_threshold=0.8)
        floors = resolve_abstain_floors(cfg)
        self.assertEqual(floors[0], 0.5)
        self.assertEqual(floors[2], 0.5)
        self.assertEqual(floors[3], 0.8)   # Metacognition
        self.assertEqual(floors[4], 0.8)   # Reappraisal

    def test_explicit_per_stage_takes_precedence(self):
        from gnn_layer.inference import resolve_abstain_floors
        cfg = GnnLayerConfig(abstain_threshold=0.5, abstain_rare_stage_threshold=0.8,
                             abstain_per_stage={0: 0.1, 4: 0.99})
        self.assertEqual(resolve_abstain_floors(cfg), {0: 0.1, 4: 0.99})

    def test_none_config_returns_none(self):
        from gnn_layer.inference import resolve_abstain_floors, resolve_purer_abstain_floor
        self.assertIsNone(resolve_abstain_floors(None))
        self.assertIsNone(resolve_purer_abstain_floor(None))


class TestInferenceAbstain(unittest.TestCase):

    def _train(self, cfg):
        from gnn_layer import graph_builder as gb, train as tr
        from gnn_layer.soft_labels import build_soft_targets
        df = synthetic_df(n_sessions=3)
        g = gb.build_graph(df, _seg_emb(df), cfg)
        tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
        model, _ = tr.train_model(g, tgts, cfg)
        return model, g

    def _cfg(self, **kw):
        return GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=5,
                              cache_embeddings=False, seed=1,
                              objectives=['soft_vaamr', 'progression', 'purer'], **kw)

    def test_no_abstain_keys_when_disabled(self):
        from gnn_layer.inference import infer_head_predictions
        cfg = self._cfg()
        model, g = self._train(cfg)
        hp = infer_head_predictions(model, g, cfg)
        self.assertNotIn('gnn_vaamr_abstain', hp)
        self.assertNotIn('gnn_purer_abstain', hp)

    def test_high_floor_abstains_everything(self):
        from gnn_layer.inference import infer_head_predictions
        cfg = self._cfg(abstain_threshold=1.01)  # impossible to clear
        model, g = self._train(cfg)
        hp = infer_head_predictions(model, g, cfg)
        self.assertIn('gnn_vaamr_abstain', hp)
        self.assertTrue(all(hp['gnn_vaamr_abstain']))
        self.assertTrue(all(hp['gnn_purer_abstain']))

    def test_zero_floor_abstains_nothing(self):
        from gnn_layer.inference import infer_head_predictions
        cfg = self._cfg(abstain_threshold=0.0)
        model, g = self._train(cfg)
        hp = infer_head_predictions(model, g, cfg)
        self.assertIn('gnn_vaamr_abstain', hp)
        self.assertFalse(any(hp['gnn_vaamr_abstain']))
        self.assertFalse(any(hp['gnn_purer_abstain']))


class TestOverlayRoundtrip(unittest.TestCase):

    def test_abstain_flags_persist_through_overlay(self):
        from classification_tools.data_structures import Segment
        from process import classifications_io as cio
        d = tempfile.mkdtemp()
        os.makedirs(os.path.join(d, '02_meta', 'classifications'), exist_ok=True)
        p = Segment(segment_id='p1', text='x', speaker='participant')
        p.gnn_vaamr_pred, p.gnn_vaamr_conf, p.gnn_vaamr_abstain = 3, 0.4, True
        t = Segment(segment_id='t1', text='y', speaker='therapist')
        t.gnn_purer_pred, t.gnn_purer_conf, t.gnn_purer_abstain = 2, 0.45, True
        cio.write_gnn_overlay(d, [p, t])
        by_id = {'p1': Segment(segment_id='p1'), 't1': Segment(segment_id='t1')}
        cio.apply_gnn_overlay(d, by_id)
        self.assertTrue(by_id['p1'].gnn_vaamr_abstain)
        self.assertTrue(by_id['t1'].gnn_purer_abstain)


class TestMasterDatasetDeferral(unittest.TestCase):

    def setUp(self):
        self.out = os.path.join(tempfile.mkdtemp(), 'm.jsonl')

    def _seg(self, sid, speaker, **kw):
        from classification_tools.data_structures import Segment
        s = Segment(segment_id=sid, text='x', speaker=speaker)
        for k, v in kw.items():
            setattr(s, k, v)
        return s

    def test_abstained_vaamr_keeps_llm_label(self):
        from process.assembly.master_dataset import assemble_master_dataset
        p = self._seg('p1', 'participant', primary_stage=1,
                      gnn_vaamr_pred=3, gnn_vaamr_conf=0.4, gnn_vaamr_abstain=True)
        df = assemble_master_dataset([p], self.out,
                                     gnn_authoritative=True, gate_passed=True).set_index('segment_id')
        self.assertEqual(df.loc['p1', 'final_label'], 1)
        self.assertEqual(df.loc['p1', 'final_label_source'], 'llm_zero_shot')

    def test_non_abstained_vaamr_promotes(self):
        from process.assembly.master_dataset import assemble_master_dataset
        p = self._seg('p1', 'participant', primary_stage=1,
                      gnn_vaamr_pred=3, gnn_vaamr_conf=0.95, gnn_vaamr_abstain=False)
        df = assemble_master_dataset([p], self.out,
                                     gnn_authoritative=True, gate_passed=True).set_index('segment_id')
        self.assertEqual(df.loc['p1', 'final_label'], 3)
        self.assertEqual(df.loc['p1', 'final_label_source'], 'gnn_consensus')

    def test_abstained_purer_keeps_llm_label(self):
        from process.assembly.master_dataset import assemble_master_dataset
        t = self._seg('t1', 'therapist', purer_primary=0,
                      gnn_purer_pred=4, gnn_purer_conf=0.4, gnn_purer_abstain=True)
        df = assemble_master_dataset([t], self.out,
                                     gnn_authoritative=True, gate_passed=True).set_index('segment_id')
        self.assertEqual(df.loc['t1', 'purer_final'], 0)
        self.assertEqual(df.loc['t1', 'purer_final_source'], 'llm_zero_shot')

    def test_abstain_columns_present_in_output(self):
        from process.assembly.master_dataset import assemble_master_dataset
        p = self._seg('p1', 'participant', primary_stage=1, gnn_vaamr_pred=3,
                      gnn_vaamr_conf=0.4, gnn_vaamr_abstain=True)
        df = assemble_master_dataset([p], self.out)
        self.assertIn('gnn_vaamr_abstain', df.columns)
        self.assertIn('gnn_purer_abstain', df.columns)


class TestCalibration(unittest.TestCase):

    def test_calibrate_returns_clamped_floors_for_all_stages(self):
        from gnn_layer import graph_builder as gb, train as tr
        from gnn_layer.soft_labels import build_soft_targets
        cfg = GnnLayerConfig(hidden_dim=16, n_layers=2, knn_k=3, epochs=6, validation_folds=2,
                             cache_embeddings=False, seed=1, abstain_target_precision=0.7,
                             objectives=['soft_vaamr', 'progression', 'purer'])
        df = synthetic_df(n_sessions=4)
        g = gb.build_graph(df, _seg_emb(df, seed=2), cfg)
        tgts = tr.assemble_targets(g, build_soft_targets(df, 'weak'), cfg, df_all=df)
        cal = tr.calibrate_abstain_floors(g, tgts, cfg, df, n_vce=0)
        self.assertEqual(set(cal['floors']), set(range(5)))
        self.assertTrue(all(0.0 <= f <= 1.0 for f in cal['floors'].values()))
        self.assertEqual(len(cal['per_stage']), 5)
        self.assertEqual(cal['target_precision'], 0.7)


if __name__ == '__main__':
    unittest.main()
