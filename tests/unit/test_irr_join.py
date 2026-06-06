"""
tests/unit/test_irr_join.py
---------------------------
Hermetic tests for the human-subset integration join (analysis/irr_join.py).

The human consensus codes live in qra.db; these tests monkeypatch
``process.irr_import.read_human_codes`` so no SQLite/file I/O is needed and assert
that ``populate_human_columns`` / ``human_consensus_map`` write the two
master_segments columns with the irr_import encoding ("No code" -> -1), mark only
usable consensus rows, and leave non-human rows untouched.
"""

import os
import sys
import unittest
from unittest import mock

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

import pandas as pd

from analysis import irr_join


def _code(segment_id, primary, *, is_consensus=True, source='unanimous', rater='__consensus__'):
    """Build one read_human_codes()-shaped dict."""
    return {
        'worksheet_n': 1, 'item_num': 1, 'segment_id': segment_id, 'rater': rater,
        'primary': primary, 'secondary': None, 'is_consensus': is_consensus,
        'source': source, 'notes': None,
    }


# A coded consensus, a No-code consensus, an UNRESOLVED consensus (excluded), a
# consensus with no resolved segment_id (excluded), and a per-rater (non-consensus) row.
SYNTH_CODES = [
    _code('s1', 2, source='unanimous'),
    _code('s2', -1, source='majority'),                 # "No code" -> -1
    _code('s3', 0, source='unresolved'),                # excluded (unusable)
    _code(None, 3, source='explicit'),                  # excluded (no segment)
    _code('s1', 2, is_consensus=False, source=None, rater='wade'),  # rater ballot, ignored
]


def _df():
    return pd.DataFrame({
        'segment_id': ['s1', 's2', 's3', 's4', 't1'],
        'participant_id': ['p1', 'p1', 'p2', 'p2', 'p1'],
        'speaker': ['participant', 'participant', 'participant', 'participant', 'therapist'],
        'text': ['a', 'b', 'c', 'd', 'e'],
        'final_label': [2, None, 0, None, None],
    })


class TestHumanConsensusMap(unittest.TestCase):
    def test_map_encoding(self):
        with mock.patch('process.irr_import.read_human_codes', return_value=SYNTH_CODES):
            m = irr_join.human_consensus_map('ignored')
        self.assertEqual(m, {'s1': 2, 's2': -1})          # No-code -> -1; unresolved/None excluded
        self.assertNotIn('s3', m)
        self.assertNotIn(None, m)

    def test_no_codes_returns_empty(self):
        with mock.patch('process.irr_import.read_human_codes', return_value=[]):
            self.assertEqual(irr_join.human_consensus_map('ignored'), {})


class TestPopulateHumanColumns(unittest.TestCase):
    def test_columns_populated(self):
        df = _df()
        with mock.patch('process.irr_import.read_human_codes', return_value=SYNTH_CODES):
            out = irr_join.populate_human_columns(df, 'ignored')
        by_id = {r['segment_id']: r for _, r in out.iterrows()}

        # matching usable-consensus rows are marked + carry the consensus primary
        self.assertTrue(bool(by_id['s1']['in_human_coded_subset']))
        self.assertEqual(int(by_id['s1']['human_label']), 2)
        self.assertTrue(bool(by_id['s2']['in_human_coded_subset']))
        self.assertEqual(int(by_id['s2']['human_label']), -1)    # No-code maps to -1

        # unresolved consensus / non-human rows stay non-human
        for sid in ('s3', 's4', 't1'):
            self.assertFalse(bool(by_id[sid]['in_human_coded_subset']))
            self.assertTrue(pd.isna(by_id[sid]['human_label']))

        # exactly the two usable consensus segments are in the subset
        self.assertEqual(int((out['in_human_coded_subset'] == True).sum()), 2)  # noqa: E712

    def test_no_codes_is_noop_but_ensures_columns(self):
        df = _df().drop(columns=[])  # no pre-existing human columns
        with mock.patch('process.irr_import.read_human_codes', return_value=[]):
            out = irr_join.populate_human_columns(df, 'ignored')
        self.assertIn('in_human_coded_subset', out.columns)
        self.assertIn('human_label', out.columns)
        self.assertEqual(int((out['in_human_coded_subset'] == True).sum()), 0)  # noqa: E712
        self.assertTrue(out['human_label'].isna().all())

    def test_human_label_int_parseable_for_gate(self):
        # gnn_layer.validation._human_axis does int(row['human_label']); ensure that works
        # for both a coded stage and the No-code sentinel, and raises for non-human rows.
        df = _df()
        with mock.patch('process.irr_import.read_human_codes', return_value=SYNTH_CODES):
            out = irr_join.populate_human_columns(df, 'ignored')
        by_id = {r['segment_id']: r for _, r in out.iterrows()}
        self.assertEqual(int(by_id['s1']['human_label']), 2)
        self.assertEqual(int(by_id['s2']['human_label']), -1)
        with self.assertRaises((TypeError, ValueError)):
            int(by_id['s4']['human_label'])


if __name__ == '__main__':
    unittest.main()
