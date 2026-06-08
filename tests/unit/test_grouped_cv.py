"""
tests/unit/test_grouped_cv.py
-----------------------------
Hermetic tests for analysis/grouped_cv.py — the leakage-free participant-grouped
folds + clustered-bootstrap κ-CI that gate every QRA classifier (methodology §5.3).
"""

import os
import sys
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from analysis import grouped_cv as gcv  # noqa: E402
from analysis import irr_stats  # noqa: E402


def _df(n_participants=10, per_class=2):
    """Each participant carries `per_class` segments of each VAAMR stage 0..4."""
    rows = []
    for p in range(n_participants):
        pid = f'P{p:02d}'
        for stage in range(5):
            for k in range(per_class):
                rows.append(dict(segment_id=f'{pid}_s{stage}_{k}', participant_id=pid,
                                 speaker='participant', final_label=stage))
    return pd.DataFrame(rows)


class TestBuildFolds(unittest.TestCase):
    def test_participant_pure(self):
        df = _df()
        folds = gcv.build_folds(df, n_folds=5, seed=42)
        # every labeled participant segment is assigned a fold
        self.assertEqual(len(folds), len(df))
        # no participant spans more than one fold (leakage-free)
        part_of = dict(zip(df['segment_id'].astype(str), df['participant_id'].astype(str)))
        p2folds = {}
        for sid, f in folds.items():
            p2folds.setdefault(part_of[sid], set()).add(f)
        self.assertTrue(all(len(fs) == 1 for fs in p2folds.values()))

    def test_deterministic(self):
        df = _df()
        self.assertEqual(gcv.build_folds(df, seed=42), gcv.build_folds(df, seed=42))

    def test_degrades_to_single_fold_when_tiny(self):
        df = pd.DataFrame([dict(segment_id='a', participant_id='P1', speaker='participant',
                                final_label=0)])
        folds = gcv.build_folds(df, n_folds=5, seed=42)
        self.assertEqual(set(folds.values()), {0})

    def test_empty_when_no_labels(self):
        df = pd.DataFrame([dict(segment_id='a', participant_id='P1', speaker='therapist',
                                final_label=None)])
        self.assertEqual(gcv.build_folds(df), {})


class TestKappaClusterCI(unittest.TestCase):
    def test_point_matches_cohen(self):
        a = [0, 1, 2, 3, 4, 0, 1, 2]
        b = [0, 1, 2, 3, 4, 1, 1, 2]
        clusters = ['P1', 'P1', 'P2', 'P2', 'P3', 'P3', 'P4', 'P4']
        res = gcv.kappa_cluster_ci(a, b, clusters, seed=42, n_boot=200)
        self.assertAlmostEqual(res['point'], irr_stats.cohen_kappa(a, b), places=6)
        self.assertEqual(res['n'], len(a))
        self.assertEqual(res['n_clusters'], 4)

    def test_handles_abstain_minus_one(self):
        # labels in [-1, 5] must pack/unpack correctly (-1 = No code)
        a = [-1, 0, 1, -1, 4]
        b = [-1, 0, 2, -1, 4]
        res = gcv.kappa_cluster_ci(a, b, ['P1', 'P1', 'P2', 'P2', 'P3'], n_boot=100)
        self.assertAlmostEqual(res['point'], irr_stats.cohen_kappa(a, b), places=6)

    def test_too_few_items(self):
        res = gcv.kappa_cluster_ci([0], [0], ['P1'])
        self.assertIsNone(res['lo'])


class TestPerClass(unittest.TestCase):
    def test_recall_precision(self):
        # class 1: support=2 (refs (1,1),(0,1)); pred==1 twice ((1,1),(1,0)), 1 correct.
        # → recall = 1/2 = 0.5; precision = 1/2 = 0.5
        pairs = [(1, 1), (0, 1), (1, 0)]
        rows = {r['class_id']: r for r in gcv.per_class_recall_precision(pairs)}
        self.assertEqual(rows[1]['support'], 2)
        self.assertAlmostEqual(rows[1]['recall'], 0.5)
        self.assertAlmostEqual(rows[1]['precision'], 0.5)


if __name__ == '__main__':
    unittest.main()
