"""
tests/unit/test_mindfulbert_dataset.py
--------------------------------------
Unit tests for Track C — the MindfulBERT training-set builder.

Hermetic: builds the (cue language → observed Δprogression) dataset from a synthetic master
frame, checks per-example provenance + labels, the datasheet, augmentation gating (suppressed
without a passing gate), the C4 ablation proxy, and the OFF-by-default flags.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df
from gnn_layer.config import GnnLayerConfig
from process.assembly import mindfulbert_dataset as MB
from process.assembly import build_mindfulbert_dataset
from process import output_paths as _paths


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


class TestBuildExamples(unittest.TestCase):

    def test_examples_and_provenance(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        res = build_mindfulbert_dataset(df, out, config=GnnLayerConfig(), verbose=False)
        self.assertEqual(res['status'], 'ok')
        self.assertGreater(res['n_examples'], 0)
        rows = _read_jsonl(os.path.join(_paths.training_data_dir(out), 'mindfulbert_dataset.jsonl'))
        self.assertEqual(len(rows), res['n_examples'])
        ex = rows[0]
        for key in ('cue_block_id', 'context_text', 'cue_text', 'delta_progression',
                    'direction', 'provenance'):
            self.assertIn(key, ex)
        self.assertIn(ex['direction'], ('advanced', 'stayed', 'regressed'))
        self.assertIn(ex['provenance']['tier'],
                      ('adjudicated', 'human_consensus', 'gnn_consensus', 'llm_zero_shot'))
        self.assertFalse(ex['provenance']['gate_passed'])  # no gate verdict on disk

    def test_progression_coord_basis(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        # Give participants a continuous progression coordinate → label_basis switches.
        df['progression_coord'] = [
            (r['final_label'] + 0.3) if r['speaker'] == 'participant' and r['final_label'] == r['final_label']
            else np.nan for _, r in df.iterrows()
        ]
        out = tempfile.mkdtemp()
        build_mindfulbert_dataset(df, out, config=GnnLayerConfig(), verbose=False)
        rows = _read_jsonl(os.path.join(_paths.training_data_dir(out), 'mindfulbert_dataset.jsonl'))
        self.assertTrue(any(r['label_basis'] == 'progression_coord' for r in rows))

    def test_datasheet_written(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        build_mindfulbert_dataset(df, out, config=GnnLayerConfig(), verbose=False)
        tdir = _paths.training_data_dir(out)
        self.assertTrue(os.path.isfile(os.path.join(tdir, 'mindfulbert_datasheet.json')))
        txt = os.path.join(tdir, 'mindfulbert_datasheet.txt')
        self.assertTrue(os.path.isfile(txt))
        with open(txt) as f:
            content = f.read()
        self.assertIn('MINDFULBERT TRAINING-SET DATASHEET', content)
        self.assertIn('CAVEATS', content)


class TestAugmentationGating(unittest.TestCase):

    def test_augmentation_suppressed_without_gate(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        cfg = GnnLayerConfig(augmentation_enabled=True)  # but no gate verdict on disk
        res = build_mindfulbert_dataset(df, out, config=cfg, verbose=False)
        self.assertFalse(res['augmentation_retained'])
        with open(os.path.join(_paths.training_data_dir(out), 'mindfulbert_datasheet.json')) as f:
            sheet = json.load(f)
        self.assertTrue(sheet['augmentation']['enabled'])
        self.assertEqual(sheet['augmentation']['n_augmented'], 0)
        # No example should carry an augmentation channel.
        rows = _read_jsonl(os.path.join(_paths.training_data_dir(out), 'mindfulbert_dataset.jsonl'))
        self.assertFalse(any('augmentation' in r for r in rows))

    def test_augmentation_attached_from_transition_model(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        # The dyadic transition model's per-move learned counterfactual (replaces the retired
        # influence.py CSV). Augmentation is now sourced here and DECOUPLED from the classifier
        # gate — note gate_passed=False below still yields the channel.
        gdir = _paths.gnn_data_dir(out)
        os.makedirs(gdir, exist_ok=True)
        import pandas as pd
        pd.DataFrame([{'move': m, 'move_name': str(m), 'mean_influence': 0.1 * m}
                      for m in range(5)]).to_csv(
            os.path.join(gdir, 'transition_per_move.csv'), index=False)
        cfg = GnnLayerConfig(augmentation_enabled=True)
        res = build_mindfulbert_dataset(df, out, config=cfg, gate_passed=False, verbose=False)
        with open(os.path.join(_paths.training_data_dir(out), 'mindfulbert_datasheet.json')) as f:
            sheet = json.load(f)
        self.assertGreater(sheet['augmentation']['n_augmented'], 0)
        self.assertIn('ablation', sheet['augmentation'])


class TestAblationProxy(unittest.TestCase):

    def _aug_examples(self, n=40, informative=True, seed=0):
        rng = np.random.default_rng(seed)
        exs = []
        for i in range(n):
            stage = int(rng.integers(0, 5))
            move = int(rng.integers(0, 5))
            # When informative, the augmentation feature aligns with the outcome.
            adv = 1 if (rng.random() < (0.2 + 0.6 * (move >= 3))) else 0
            wp = (0.5 if adv else -0.5) + rng.normal(0, 0.1) if informative else rng.normal(0, 1)
            exs.append({
                'from_stage': stage, 'dominant_purer': move, 'n_cue_words': int(rng.integers(3, 40)),
                'n_therapist_segments': int(rng.integers(1, 4)),
                'direction': 'advanced' if adv else 'stayed',
                'participant_id': f'P{i % 4}',
                'augmentation': {'provenance': 'transition_counterfactual', 'would_progress': float(wp)},
            })
        return exs

    def test_ablation_returns_metrics(self):
        cfg = GnnLayerConfig(augmentation_enabled=True, augmentation_min_gain=0.0)
        res = MB._augmentation_ablation(self._aug_examples(informative=True), cfg)
        self.assertIn(res['status'], ('ok', 'inconclusive: single-class outcome'))
        if res['status'] == 'ok':
            self.assertIsNotNone(res['base_accuracy'])
            self.assertIsNotNone(res['augmented_accuracy'])
            self.assertIn('retain', res)

    def test_ablation_too_few(self):
        cfg = GnnLayerConfig()
        res = MB._augmentation_ablation(self._aug_examples(n=4), cfg)
        self.assertIn('inconclusive', res['status'])


class TestDefaults(unittest.TestCase):

    def test_build_on_by_default_augmentation_off(self):
        c = GnnLayerConfig()
        # Track C ships on by default (feeds the fine-tuning workshop) ...
        self.assertTrue(c.build_mindfulbert_dataset)
        # ... but the gate-gated augmentation channel stays off (Decision D10).
        self.assertFalse(c.augmentation_enabled)
        self.assertEqual(c.augmentation_min_gain, 0.0)


class TestContractAdditions(unittest.TestCase):
    """Contract §3 additions exposed on cue blocks + sibling exports."""

    def test_cue_block_enriched_fields(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        build_mindfulbert_dataset(df, out, config=GnnLayerConfig(), verbose=False)
        rows = _read_jsonl(os.path.join(_paths.training_data_dir(out), 'mindfulbert_dataset.jsonl'))
        ex = rows[0]
        for key in ('from_coord', 'to_coord', 'from_stage_mixture', 'to_stage_mixture',
                    'from_confidence', 'to_confidence', 'text_sha'):
            self.assertIn(key, ex)
        # Mixtures recomputed from rater_votes → normalized 5-vectors.
        self.assertEqual(len(ex['from_stage_mixture']), 5)
        self.assertAlmostEqual(sum(ex['from_stage_mixture']), 1.0, places=5)
        # text_sha is a stable sha256 hex digest.
        self.assertEqual(len(ex['text_sha']), 64)

    def test_cue_pool_and_sft_written(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        build_mindfulbert_dataset(df, out, config=GnnLayerConfig(), verbose=False)
        tdir = _paths.training_data_dir(out)
        pool = _read_jsonl(os.path.join(tdir, 'therapist_cue_pool.jsonl'))
        self.assertTrue(pool)
        for row in pool:
            for key in ('cue_text', 'dominant_purer', 'n_uses', 'p_advance', 'text_sha'):
                self.assertIn(key, row)
            self.assertGreaterEqual(row['n_uses'], 1)
            self.assertTrue(0.0 <= row['p_advance'] <= 1.0)
        # Cue pool is deduplicated by text_sha.
        shas = [r['text_sha'] for r in pool]
        self.assertEqual(len(shas), len(set(shas)))
        # SFT view: advancers only.
        examples = _read_jsonl(os.path.join(tdir, 'mindfulbert_dataset.jsonl'))
        n_adv = sum(1 for e in examples if e['direction'] == 'advanced')
        sft = _read_jsonl(os.path.join(tdir, 'mindfulbert_sft.jsonl'))
        self.assertEqual(len(sft), n_adv)
        if sft:
            for key in ('instruction', 'input', 'output', 'provenance_tier',
                        'from_stage', 'dominant_purer', 'text_sha'):
                self.assertIn(key, sft[0])

    def test_datasheet_has_stage_counts_and_weights(self):
        df = make_master_df(n_sessions=3, n_participants=2)
        out = tempfile.mkdtemp()
        build_mindfulbert_dataset(df, out, config=GnnLayerConfig(), verbose=False)
        with open(os.path.join(_paths.training_data_dir(out), 'mindfulbert_datasheet.json')) as f:
            sheet = json.load(f)
        self.assertIn('theme_label_counts', sheet)
        self.assertIn('class_weights', sheet)
        self.assertTrue(sheet['theme_label_counts'])


if __name__ == '__main__':
    unittest.main()
