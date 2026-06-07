"""
Golden tests for the shared lift kernel (analysis/stats.py:lift_ratio).

`lift_ratio` was extracted from three call sites that each compute association
lift = P(b|a)/P(b) but keep their own interface, rounding, and filtering:

  * analysis/theme.py:_compute_lift            (counts -> scalar, round 4)
  * gnn_layer/gnn_lift.py:_lift_table          (lists  -> DataFrame, round 3)
  * gnn_layer/graph_builder.py:compute_cross_framework_lift  (df -> filtered dict, round 3)

These tests pin the kernel directly AND the exact numeric output of each call
site on a fixed fixture, so the extraction is provably behavior-preserving.
"""

import unittest

import pandas as pd

from analysis.stats import lift_ratio
from analysis.theme import _compute_lift
from gnn_layer.classifier.gnn_lift import _lift_table
from gnn_layer.classifier.graph_builder import compute_cross_framework_lift


class TestLiftRatioKernel(unittest.TestCase):
    def test_independence_is_one(self):
        self.assertEqual(lift_ratio(0.5, 0.5), 1.0)

    def test_positive_association(self):
        self.assertEqual(lift_ratio(0.8, 0.2), 4.0)

    def test_negative_association(self):
        self.assertEqual(lift_ratio(0.3, 0.6), 0.5)

    def test_nonpositive_marginal_returns_zero(self):
        self.assertEqual(lift_ratio(0.5, 0.0), 0.0)
        self.assertEqual(lift_ratio(0.5, -1.0), 0.0)

    def test_zero_conditional(self):
        self.assertEqual(lift_ratio(0.0, 0.5), 0.0)


class TestThemeComputeLiftGolden(unittest.TestCase):
    def test_known_lift(self):
        # observed = 4/10 = 0.4 ; expected = 8/40 = 0.2 ; lift = 2.0
        self.assertEqual(_compute_lift(4, 10, 8, 40), 2.0)

    def test_independence(self):
        # observed = 3/6 = 0.5 ; expected = 10/20 = 0.5 ; lift = 1.0
        self.assertEqual(_compute_lift(3, 6, 10, 20), 1.0)

    def test_guards_return_zero(self):
        self.assertEqual(_compute_lift(5, 0, 8, 40), 0.0)   # n_stage == 0
        self.assertEqual(_compute_lift(5, 10, 0, 40), 0.0)  # n_code == 0
        self.assertEqual(_compute_lift(5, 10, 8, 0), 0.0)   # n_total == 0
        self.assertEqual(_compute_lift(0, 10, 8, 40), 0.0)  # no co-occurrence


class TestGnnLiftTableGolden(unittest.TestCase):
    def test_known_table(self):
        # 4 segments; code 'x' present in 3 of them -> p_b = 0.75
        df = _lift_table([0, 0, 1, 1], [['x'], ['x'], ['x'], []], 'stage', 'code')
        by_stage = {int(r['stage']): r for _, r in df.iterrows()}
        # stage 0: p(x|0)=2/2=1.0 ; lift = 1.0/0.75 = 1.333
        self.assertAlmostEqual(by_stage[0]['lift'], 1.333)
        self.assertEqual(int(by_stage[0]['count']), 2)
        self.assertAlmostEqual(by_stage[0]['p_b'], 0.75)
        # stage 1: p(x|1)=1/2=0.5 ; lift = 0.5/0.75 = 0.667
        self.assertAlmostEqual(by_stage[1]['lift'], 0.667)
        self.assertEqual(int(by_stage[1]['count']), 1)


class TestCrossFrameworkLiftGolden(unittest.TestCase):
    def test_known_dict_with_threshold(self):
        df = pd.DataFrame({
            'final_label': [0, 0, 1, 1],
            'codebook_labels_ensemble': [['x'], ['x'], [], []],
        })
        # code 'x': p(x)=2/4=0.5 ; p(x|0)=1.0 -> lift 2.0 (>=1.5, kept)
        #                          p(x|1)=0.0 -> lift 0.0 (dropped)
        self.assertEqual(compute_cross_framework_lift(df), {('vaamr_0', 'x'): 2.0})


if __name__ == '__main__':
    unittest.main()
