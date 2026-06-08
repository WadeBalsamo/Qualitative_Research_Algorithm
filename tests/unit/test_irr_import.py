"""
tests/unit/test_irr_import.py
-----------------------------
Hermetic tests for the IRR importer (no network, no models).

Covers:
  - free-text label normalization ("No code" -> abstain, empty -> missing, alias map)
  - consensus recompute (unanimous / majority / unresolved / abstain)
  - worksheet item -> segment_id resolution against a fixture qra.db
  - consensus-of-record persistence + validation warnings (explicit override, drift)
  - idempotent re-import
"""

import hashlib
import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from classification_tools.data_structures import Segment
from process import db, segments_io
from process import irr_import
from process.irr_import import (
    _normalize_label, _recompute_consensus, ABSTAIN_CODE,
)

# VAAMR ids
VIG, AVO, ATT, MET, REA = 0, 1, 2, 3, 4

CSV_HEADER = ("testset,worksheet_n,item,becca_p,becca_s,ryan_p,ryan_s,wade_p,wade_s,"
              "adam_p,adam_s,human_consensus_p,human_consensus_s,consensus_source,notes")


def _write_csv(rows):
    fd, path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(CSV_HEADER + "\n")
        for r in rows:
            f.write(r + "\n")
    return path


def _build_fixture_db(run_dir, session_id, texts, *, sha_override=None):
    """Write `len(texts)` frozen participant segments + a matching worksheet."""
    segs = []
    for i, t in enumerate(texts):
        s = Segment(segment_id=f'{session_id}_{i:03d}', session_id=session_id,
                    speaker='participant', text=t, segment_index=i,
                    start_time_ms=1000 * i, end_time_ms=1000 * i + 500)
        segs.append(s)
    segments_io.write_session_segments(run_dir, session_id, segs, 'h')
    with db.open_db(run_dir) as conn:
        conn.execute("INSERT INTO testset_worksheets (worksheet_n,kind,name,n_items) "
                     "VALUES (1,'vaamr','ws1',?)", (len(texts),))
        for i, t in enumerate(texts):
            sha = sha_override or hashlib.sha256(t.encode()).hexdigest()
            conn.execute("INSERT INTO testset_items (worksheet_n,item_num,session_id,seg_num,sha256) "
                         "VALUES (1,?,?,?,?)", (i + 1, session_id, i + 1, sha))
    return segs


class TestNormalization(unittest.TestCase):
    def setUp(self):
        from constructs.registry import load
        self.m = load('vaamr').build_name_to_id_map()

    def test_known_labels(self):
        w = []
        self.assertEqual(_normalize_label('Vigilance', self.m, w, 'x'), ('coded', VIG))
        self.assertEqual(_normalize_label('Attention', self.m, w, 'x'), ('coded', ATT))
        self.assertEqual(_normalize_label('Reappraisal', self.m, w, 'x'), ('coded', REA))
        self.assertEqual(w, [])

    def test_no_code_is_abstain(self):
        w = []
        self.assertEqual(_normalize_label('No code', self.m, w, 'x'), ('abstain', ABSTAIN_CODE))

    def test_empty_is_missing(self):
        w = []
        self.assertEqual(_normalize_label('', self.m, w, 'x'), ('missing', None))
        self.assertEqual(_normalize_label('   ', self.m, w, 'x'), ('missing', None))
        self.assertEqual(w, [])

    def test_unknown_warns_and_is_missing(self):
        w = []
        self.assertEqual(_normalize_label('Frobnicate', self.m, w, 'ws1 item5'), ('missing', None))
        self.assertEqual(len(w), 1)
        self.assertIn('Frobnicate', w[0])


class TestConsensusRecompute(unittest.TestCase):
    def test_unanimous(self):
        ballots = [('a', 'coded', VIG), ('b', 'coded', VIG), ('c', 'coded', VIG)]
        self.assertEqual(_recompute_consensus(ballots), ('unanimous', VIG))

    def test_majority(self):
        ballots = [('a', 'coded', ATT), ('b', 'coded', ATT), ('c', 'coded', MET)]
        self.assertEqual(_recompute_consensus(ballots), ('majority', ATT))

    def test_unresolved_all_different(self):
        ballots = [('a', 'coded', VIG), ('b', 'coded', ATT), ('c', 'coded', MET)]
        self.assertEqual(_recompute_consensus(ballots), ('unresolved', None))

    def test_abstain_consensus(self):
        ballots = [('a', 'abstain', ABSTAIN_CODE), ('b', 'abstain', ABSTAIN_CODE),
                   ('c', 'abstain', ABSTAIN_CODE)]
        self.assertEqual(_recompute_consensus(ballots), ('unanimous', ABSTAIN_CODE))

    def test_two_of_three_with_one_missing_is_majority(self):
        # Only two raters cast a ballot; both agree -> majority over the roster.
        ballots = [('a', 'coded', REA), ('b', 'coded', REA)]
        self.assertEqual(_recompute_consensus(ballots), ('unanimous', REA))


class TestImport(unittest.TestCase):
    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        self.texts = ['vig text', 'att text', 'split text', 'abstain text', 'rea text']
        _build_fixture_db(self.run_dir, 'c1s1', self.texts)
        self.csv = _write_csv([
            f"1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",
            f"1,1,2,Attention,,Attention,,Metacognition,,,,Attention,,majority,",
            f"1,1,3,Vigilance,,Attention,,Metacognition,,,,,,unresolved,",
            f"1,1,4,No code,,No code,,No code,,,,No code,,unanimous,",
            f"1,1,5,Reappraisal,,Reappraisal,,,,,,Reappraisal,,majority,",
        ])

    def tearDown(self):
        if os.path.isfile(self.csv):
            os.remove(self.csv)

    def test_import_resolves_segments(self):
        summary = irr_import.import_irr_csv(self.run_dir, self.csv, verbose=False)
        self.assertEqual(summary['worksheets'], [1])
        codes = irr_import.read_human_codes(self.run_dir)
        # every code row should have a resolved segment_id
        for c in codes:
            self.assertEqual(c['segment_id'], f"c1s1_{c['item_num'] - 1:03d}")

    def test_consensus_rows_persisted(self):
        irr_import.import_irr_csv(self.run_dir, self.csv, verbose=False)
        codes = irr_import.read_human_codes(self.run_dir)
        cons = {c['item_num']: c for c in codes if c['is_consensus']}
        self.assertEqual(cons[1]['primary'], VIG)
        self.assertEqual(cons[2]['primary'], ATT)
        self.assertIsNone(cons[3]['primary'])               # unresolved
        self.assertEqual(cons[3]['source'], 'unresolved')
        self.assertEqual(cons[4]['primary'], ABSTAIN_CODE)  # No code
        self.assertEqual(cons[5]['primary'], REA)

    def test_missing_rater_cell_yields_no_ballot(self):
        irr_import.import_irr_csv(self.run_dir, self.csv, verbose=False)
        codes = irr_import.read_human_codes(self.run_dir)
        # item5: wade has a blank primary -> no wade rater row.
        item5_raters = {c['rater'] for c in codes
                        if c['item_num'] == 5 and not c['is_consensus']}
        self.assertEqual(item5_raters, {'becca', 'ryan'})

    def test_roster_recorded(self):
        irr_import.import_irr_csv(self.run_dir, self.csv, verbose=False)
        ts = irr_import.list_imported_testsets(self.run_dir)
        self.assertEqual(len(ts), 1)
        self.assertEqual(set(ts[0]['raters']), {'becca', 'ryan', 'wade'})

    def test_idempotent_reimport(self):
        s1 = irr_import.import_irr_csv(self.run_dir, self.csv, verbose=False)
        s2 = irr_import.import_irr_csv(self.run_dir, self.csv, verbose=False)
        self.assertEqual(s1['n_codes'], s2['n_codes'])
        codes = irr_import.read_human_codes(self.run_dir)
        # exactly one consensus row per item (no duplication on re-import)
        cons = [c for c in codes if c['is_consensus']]
        self.assertEqual(len(cons), 5)

    def test_sha_drift_warning(self):
        run_dir = tempfile.mkdtemp()
        _build_fixture_db(run_dir, 'c1s1', self.texts, sha_override='deadbeef')
        summary = irr_import.import_irr_csv(run_dir, self.csv, verbose=False)
        self.assertTrue(any('SHA drift' in w for w in summary['warnings']))


class TestDriftGuard(unittest.TestCase):
    """Run-time guard: the segments scored by IRR must still be byte-identical to the
    frozen worksheet text the humans coded (see check_testset_drift)."""

    def setUp(self):
        self.run_dir = tempfile.mkdtemp()
        self.texts = ['vig text', 'att text', 'split text', 'abstain text', 'rea text']
        _build_fixture_db(self.run_dir, 'c1s1', self.texts)

    def test_clean_project_has_no_drift(self):
        self.assertEqual(irr_import.check_testset_drift(self.run_dir), [])
        self.assertTrue(irr_import.testset_content_signature(self.run_dir))  # non-empty hash

    def test_mutated_segment_is_drift_and_flips_signature(self):
        sig_before = irr_import.testset_content_signature(self.run_dir)
        with db.open_db(self.run_dir) as conn:
            # seg_num 2 -> segment_index 1 -> segment c1s1_001 (worksheet item 2).
            conn.execute("UPDATE segments SET text=? WHERE segment_id=?",
                         ('MUTATED CONTENT', 'c1s1_001'))
        drift = irr_import.check_testset_drift(self.run_dir)
        self.assertEqual(len(drift), 1)
        self.assertEqual(drift[0]['kind'], 'drift')
        self.assertEqual(drift[0]['item_num'], 2)
        self.assertEqual(drift[0]['segment_id'], 'c1s1_001')
        # signature must change so qra analyze's change-gate regenerates + flags it
        self.assertNotEqual(irr_import.testset_content_signature(self.run_dir), sig_before)
        self.assertIn('item2', irr_import.format_drift_banner(drift))

    def test_missing_segment_is_unresolved(self):
        with db.open_db(self.run_dir) as conn:
            # seg_num 3 -> segment_index 2 -> segment c1s1_002 (worksheet item 3).
            conn.execute("DELETE FROM segments WHERE segment_id=?", ('c1s1_002',))
        drift = irr_import.check_testset_drift(self.run_dir)
        self.assertIn((3, 'unresolved'), {(d['item_num'], d['kind']) for d in drift})


if __name__ == '__main__':
    unittest.main()
