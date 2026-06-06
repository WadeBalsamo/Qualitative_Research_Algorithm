"""
tests/unit/test_splits.py
-------------------------
Unit tests for the frozen, leakage-safe split manifest (contract P0).

Hermetic: builds the participant-grouped split manifest from synthetic segments and checks
determinism, the schema, participant grouping (no leakage), and the holdout carve-out.
"""

import json
import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process.assembly import export_split_manifest
from process import output_paths as _paths


def _segments(n_participants=7, per=3):
    segs = []
    for p in range(n_participants):
        pid = f'P{p + 1:02d}'
        for j in range(per):
            segs.append(Segment(segment_id=f'{pid}_{j}', speaker='participant',
                                text='x', participant_id=pid))
    return segs


def _load(path):
    with open(path) as f:
        return json.load(f)


class TestSplitManifest(unittest.TestCase):

    def test_schema_and_grouping(self):
        out = tempfile.mkdtemp()
        path = export_split_manifest(_segments(7), out, k=5, seed=42)
        self.assertEqual(path, os.path.join(_paths.training_data_dir(out), 'splits.json'))
        m = _load(path)
        self.assertEqual(m['strategy'], 'grouped_by_participant')
        self.assertEqual(m['seed'], 42)
        self.assertEqual(set(m['assignment'].keys()),
                         {f'P{p + 1:02d}' for p in range(7)})
        # Every fold index is within range, and a participant maps to exactly one fold.
        for fold in m['assignment'].values():
            self.assertTrue(0 <= fold < m['k'])

    def test_deterministic(self):
        out1, out2 = tempfile.mkdtemp(), tempfile.mkdtemp()
        m1 = _load(export_split_manifest(_segments(7), out1, k=5, seed=42))
        m2 = _load(export_split_manifest(_segments(7), out2, k=5, seed=42))
        self.assertEqual(m1['assignment'], m2['assignment'])

    def test_holdout_excluded(self):
        out = tempfile.mkdtemp()
        m = _load(export_split_manifest(_segments(7), out, k=5, seed=42,
                                        holdout_participants=['P06', 'P07']))
        self.assertEqual(m['holdout_participants'], ['P06', 'P07'])
        self.assertNotIn('P06', m['assignment'])
        self.assertNotIn('P07', m['assignment'])

    def test_fewer_participants_than_k(self):
        out = tempfile.mkdtemp()
        m = _load(export_split_manifest(_segments(3), out, k=5, seed=42))
        # Effective k capped at participant count → every fold non-empty.
        self.assertEqual(m['k'], 3)
        self.assertEqual(sorted(set(m['assignment'].values())), [0, 1, 2])


if __name__ == '__main__':
    unittest.main()
