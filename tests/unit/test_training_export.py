"""
tests/unit/test_training_export.py
----------------------------------
Unit tests for assembly/training_export.export_training_data.

Hermetic: builds a tiny set of classified participant segments and checks the contract
additions — the per-record soft ``stage_mixture`` on theme_classification.jsonl and the
sibling theme_classification_datasheet.json (per-stage counts + class weights).
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
from process.assembly import export_training_data
from process import output_paths as _paths
from theme_framework.registry import load as load_framework


def _read_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _participant(sid, stage, votes=None):
    seg = Segment(segment_id=sid, speaker='participant', text=f'text {sid}',
                  participant_id='P01', session_id='c1p1s1', session_number=1)
    seg.primary_stage = stage
    seg.final_label = stage
    seg.final_label_source = 'llm_zero_shot'
    seg.llm_confidence_primary = 0.9
    seg.llm_run_consistency = 3
    seg.label_confidence_tier = 'high'
    if votes is not None:
        seg.rater_votes = votes
    return seg


class TestTrainingExport(unittest.TestCase):

    def setUp(self):
        self.out = tempfile.mkdtemp()
        self.fw = load_framework('vaamr')

    def test_stage_mixture_on_theme_records(self):
        segs = [
            _participant('p1', 2, votes=[{'stage': 2, 'confidence': 1.0},
                                         {'stage': 3, 'confidence': 1.0}]),
            _participant('p2', 0),  # no ballots → one-hot fallback
        ]
        export_training_data(segs, self.fw, None, self.out)
        rows = _read_jsonl(os.path.join(_paths.training_data_dir(self.out),
                                        'theme_classification.jsonl'))
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn('stage_mixture', r)
            self.assertEqual(len(r['stage_mixture']), 5)
            self.assertAlmostEqual(sum(r['stage_mixture']), 1.0, places=5)
        # The split-ballot record is a genuine mixture (not one-hot).
        r1 = next(r for r in rows if r['segment_id'] == 'p1')
        self.assertAlmostEqual(r1['stage_mixture'][2], 0.5, places=5)
        self.assertAlmostEqual(r1['stage_mixture'][3], 0.5, places=5)

    def test_theme_datasheet(self):
        segs = [_participant('p1', 2), _participant('p2', 2), _participant('p3', 0)]
        export_training_data(segs, self.fw, None, self.out)
        path = os.path.join(_paths.training_data_dir(self.out),
                            'theme_classification_datasheet.json')
        self.assertTrue(os.path.isfile(path))
        with open(path) as f:
            sheet = json.load(f)
        self.assertEqual(sheet['n_examples'], 3)
        self.assertEqual(sheet['theme_label_counts'], {'0': 1, '2': 2})
        self.assertIn('class_weights', sheet)
        # The rarer class (0) gets a higher inverse-frequency weight than the common (2).
        self.assertGreater(sheet['class_weights']['0'], sheet['class_weights']['2'])


if __name__ == '__main__':
    unittest.main()
