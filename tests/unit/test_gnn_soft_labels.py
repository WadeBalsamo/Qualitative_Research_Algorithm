"""
tests/unit/test_gnn_soft_labels.py
------------------------------------
Unit tests for gnn_layer/soft_labels.py.

Covers: ballots_to_mixture (normalizes to sum 1; None/empty → uniform 0.2;
secondary_stage weighting), one_hot (valid/invalid stage), mixture_to_progression
(E[stage] = sum_k k*p_k), build_soft_targets for 'weak', 'human',
'self_supervised' label_mode variants.
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df, embedding_patch, make_master_df
from gnn_layer.config import GnnLayerConfig


# ---------------------------------------------------------------------------
# ballots_to_mixture
# ---------------------------------------------------------------------------

class TestBallotsToMixture(unittest.TestCase):

    def setUp(self):
        from gnn_layer.soft_labels import ballots_to_mixture
        self._fn = ballots_to_mixture

    def test_returns_numpy_array_length_5(self):
        m = self._fn([{'stage': 0, 'confidence': 1.0}])
        self.assertIsInstance(m, np.ndarray)
        self.assertEqual(m.shape, (5,))

    def test_sums_to_one(self):
        m = self._fn([{'stage': 1, 'confidence': 0.9},
                      {'stage': 2, 'confidence': 0.6}])
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)

    def test_none_gives_uniform(self):
        m = self._fn(None)
        self.assertTrue(np.allclose(m, 0.2),
                        f"Expected uniform 0.2, got {m}")

    def test_empty_list_gives_uniform(self):
        m = self._fn([])
        self.assertTrue(np.allclose(m, 0.2))

    def test_single_ballot_concentrates_on_stage(self):
        m = self._fn([{'stage': 3, 'confidence': 1.0}])
        self.assertAlmostEqual(float(m[3]), 1.0, places=6)
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)

    def test_secondary_stage_receives_weight(self):
        # primary stage 1, secondary stage 3
        m_no_sec = self._fn([{'stage': 1, 'confidence': 1.0}])
        m_sec = self._fn([{'stage': 1, 'confidence': 1.0,
                            'secondary_stage': 3, 'secondary_confidence': 1.0}])
        # secondary stage 3 should have > 0 mass
        self.assertGreater(float(m_sec[3]), 0.0)
        # stage 1 should have more than stage 3 (primary weight > secondary weight)
        self.assertGreater(float(m_sec[1]), float(m_sec[3]))

    def test_confidence_determines_weight(self):
        m = self._fn([{'stage': 0, 'confidence': 0.9},
                      {'stage': 4, 'confidence': 0.1}])
        self.assertGreater(float(m[0]), float(m[4]))

    def test_missing_confidence_defaults_to_1(self):
        m = self._fn([{'stage': 2}])  # no 'confidence' key
        self.assertAlmostEqual(float(m[2]), 1.0, places=6)

    def test_multi_ballot_mixture(self):
        m = self._fn([{'stage': 1, 'confidence': 0.9},
                      {'stage': 2, 'confidence': 0.1, 'secondary_stage': 1}])
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)
        self.assertGreater(float(m[1]), float(m[3]))  # stage 1 carried weight

    def test_invalid_ballot_type_ignored(self):
        m = self._fn(["not_a_dict", None, {'stage': 2, 'confidence': 1.0}])
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(m[2]), 1.0, places=6)

    def test_out_of_range_stage_ignored(self):
        # stage 99 should be ignored; only stage 2 counts
        m = self._fn([{'stage': 99, 'confidence': 1.0},
                      {'stage': 2, 'confidence': 1.0}])
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)
        self.assertAlmostEqual(float(m[2]), 1.0, places=6)

    def test_vote_key_alias_for_stage(self):
        m = self._fn([{'vote': 4, 'confidence': 1.0}])
        self.assertAlmostEqual(float(m[4]), 1.0, places=6)

    def test_custom_n_stages(self):
        m = self._fn([{'stage': 2, 'confidence': 1.0}], n_stages=3)
        self.assertEqual(m.shape, (3,))
        self.assertAlmostEqual(float(m.sum()), 1.0, places=6)


# ---------------------------------------------------------------------------
# one_hot
# ---------------------------------------------------------------------------

class TestOneHot(unittest.TestCase):

    def setUp(self):
        from gnn_layer.soft_labels import one_hot
        self._fn = one_hot

    def test_shape_is_5(self):
        self.assertEqual(self._fn(0).shape, (5,))

    def test_only_specified_index_is_one(self):
        for k in range(5):
            v = self._fn(k)
            self.assertAlmostEqual(float(v[k]), 1.0, places=6)
            self.assertAlmostEqual(float(v.sum()), 1.0, places=6)

    def test_invalid_stage_returns_uniform(self):
        v = self._fn(99)
        self.assertTrue(np.allclose(v, 0.2))

    def test_none_stage_returns_uniform(self):
        v = self._fn(None)
        self.assertTrue(np.allclose(v, 0.2))

    def test_custom_n_stages(self):
        v = self._fn(1, n_stages=3)
        self.assertEqual(v.shape, (3,))
        self.assertAlmostEqual(float(v[1]), 1.0, places=6)


# ---------------------------------------------------------------------------
# mixture_to_progression
# ---------------------------------------------------------------------------

class TestMixtureToProgression(unittest.TestCase):

    def setUp(self):
        from gnn_layer.soft_labels import mixture_to_progression, one_hot
        self._fn = mixture_to_progression
        self._oh = one_hot

    def test_one_hot_returns_exact_stage(self):
        for k in range(5):
            self.assertAlmostEqual(self._fn(self._oh(k)), float(k), places=6)

    def test_uniform_returns_two(self):
        # E[stage] = sum(k * 0.2 for k in range(5)) = 0.2*(0+1+2+3+4) = 2.0
        self.assertAlmostEqual(self._fn(np.full(5, 0.2)), 2.0, places=6)

    def test_zero_vector_returns_zero(self):
        # Edge case: zero vector → 0.0 (no numeric crash)
        self.assertEqual(self._fn(np.zeros(5)), 0.0)

    def test_returns_float(self):
        result = self._fn(np.full(5, 0.2))
        self.assertIsInstance(result, float)

    def test_handles_torch_tensor_input(self):
        import torch
        m = torch.tensor([0., 0., 1., 0., 0.])
        result = self._fn(m)
        self.assertAlmostEqual(result, 2.0, places=5)


# ---------------------------------------------------------------------------
# build_soft_targets
# ---------------------------------------------------------------------------

class TestBuildSoftTargets(unittest.TestCase):

    def setUp(self):
        from gnn_layer.soft_labels import build_soft_targets
        self._fn = build_soft_targets

    def _df(self):
        return synthetic_df(n_sessions=2)

    def test_weak_returns_participant_segments_only(self):
        df = self._df()
        t = self._fn(df, 'weak')
        # participant segment IDs
        part_ids = set(df[df['speaker'] == 'participant']['segment_id'].astype(str))
        ther_ids = set(df[df['speaker'] == 'therapist']['segment_id'].astype(str))
        self.assertTrue(part_ids.issuperset(set(t.keys())))
        self.assertEqual(set(t.keys()) & ther_ids, set())

    def test_weak_all_mixtures_sum_to_one(self):
        df = self._df()
        t = self._fn(df, 'weak')
        for sid, mix in t.items():
            self.assertAlmostEqual(float(np.asarray(mix).sum()), 1.0, places=5,
                                   msg=f"Mixture for {sid} does not sum to 1")

    def test_weak_non_empty(self):
        df = self._df()
        t = self._fn(df, 'weak')
        self.assertGreater(len(t), 0)

    def test_self_supervised_returns_empty_dict(self):
        df = self._df()
        t = self._fn(df, 'self_supervised')
        self.assertEqual(t, {})

    def test_human_returns_only_human_coded_rows(self):
        # synthetic_df has in_human_coded_subset=False for all rows → empty dict
        df = self._df()
        t = self._fn(df, 'human')
        self.assertEqual(len(t), 0)

    def test_human_with_human_subset(self):
        # make_master_df includes human-coded rows
        df = make_master_df(n_sessions=2, n_participants=1, with_human_subset=True)
        t = self._fn(df, 'human')
        self.assertGreater(len(t), 0)
        # all returned keys should be participant segments with human labels
        for sid, mix in t.items():
            row = df[df['segment_id'].astype(str) == sid].iloc[0]
            self.assertTrue(bool(row.get('in_human_coded_subset', False)))
            self.assertAlmostEqual(float(np.asarray(mix).sum()), 1.0, places=5)

    def test_weak_fallback_to_final_label_when_no_valid_votes(self):
        """When rater_votes is None, build_soft_targets falls back to final_label one-hot."""
        df = pd.DataFrame([dict(
            segment_id='s0', speaker='participant', session_id='test',
            final_label=3, primary_stage=3, rater_votes=None,
            in_human_coded_subset=False, human_label=float('nan'),
            codebook_labels_ensemble=[],
        )])
        t = self._fn(df, 'weak')
        self.assertIn('s0', t)
        mix = np.asarray(t['s0'])
        # one-hot on stage 3
        self.assertAlmostEqual(float(mix[3]), 1.0, places=6)

    def test_mixture_shape_is_5(self):
        df = self._df()
        t = self._fn(df, 'weak')
        for sid, mix in t.items():
            self.assertEqual(np.asarray(mix).shape, (5,))

    def test_df_without_speaker_column_treated_as_all_participant(self):
        """When 'speaker' column absent, all rows are treated as participants (source: soft_labels.py)."""
        df = self._df().drop(columns=['speaker'])
        t = self._fn(df, 'weak')
        self.assertGreater(len(t), 0)


if __name__ == '__main__':
    unittest.main()
