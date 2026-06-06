"""
tests/unit/test_irr_analysis.py
-------------------------------
Hermetic tests for the IRR analysis orchestrator + library-backed statistics.

Covers:
  - irr_stats wrappers (Cohen via sklearn, Fleiss via statsmodels, Krippendorff
    via the krippendorff package) on hand-verifiable inputs
  - Human↔Human (perfect agreement -> α = κ = 1.0)
  - Human-consensus ↔ LLM κ + percent agreement, unresolved items excluded
  - per-LLM-rater agreement from rater_votes
  - GNN abstain -> deferred + excluded
  - discrepancy extraction
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
from process import db, segments_io, classifications_io, irr_import
from analysis import irr_analysis, irr_stats

VIG, AVO, ATT, MET, REA = 0, 1, 2, 3, 4
ABSTAIN = irr_stats.ABSTAIN_CODE

CSV_HEADER = ("testset,worksheet_n,item,becca_p,becca_s,ryan_p,ryan_s,wade_p,wade_s,"
              "adam_p,adam_s,human_consensus_p,human_consensus_s,consensus_source,notes")


def _write_csv(rows):
    fd, path = tempfile.mkstemp(suffix='.csv')
    with os.fdopen(fd, 'w', encoding='utf-8') as f:
        f.write(CSV_HEADER + "\n")
        for r in rows:
            f.write(r + "\n")
    return path


class TestStatsWrappers(unittest.TestCase):
    def test_cohen_perfect(self):
        self.assertEqual(irr_stats.cohen_kappa([0, 1, 2, 3], [0, 1, 2, 3]), 1.0)

    def test_cohen_constant_identical(self):
        self.assertEqual(irr_stats.cohen_kappa([2, 2, 2], [2, 2, 2]), 1.0)

    def test_observed_agreement(self):
        self.assertAlmostEqual(irr_stats.observed_agreement([0, 1, 2], [0, 1, 3]), 2 / 3)

    def test_krippendorff_perfect(self):
        matrix = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]
        self.assertAlmostEqual(irr_stats.krippendorff_alpha(matrix), 1.0, places=6)

    def test_fleiss_complete_case(self):
        # 3 raters, 4 items, perfect agreement -> κ = 1.0
        matrix = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 4, 4]]
        res = irr_stats.fleiss_kappa(matrix)
        self.assertEqual(res['n_complete'], 4)
        self.assertAlmostEqual(res['kappa'], 1.0, places=6)

    def test_fleiss_ignores_incomplete_rows(self):
        matrix = [[0, 0, 0], [1, 1, None], [2, 2, 2]]
        res = irr_stats.fleiss_kappa(matrix)
        self.assertEqual(res['n_complete'], 2)  # the None row is dropped

    def test_krippendorff_tolerates_missing(self):
        matrix = [[0, 0, None], [1, None, 1], [2, 2, 2]]
        self.assertIsNotNone(irr_stats.krippendorff_alpha(matrix))


def _seg(sess, i, text, primary, gnn_pred, *, gnn_abstain=0, rater_votes=None):
    s = Segment(segment_id=f'{sess}_{i:03d}', session_id=sess, speaker='participant',
                text=text, segment_index=i, start_time_ms=1000 * i, end_time_ms=1000 * i + 5)
    s.primary_stage = primary
    s.llm_confidence_primary = 0.8
    s.gnn_vaamr_pred = gnn_pred
    s.gnn_vaamr_conf = 0.7
    s.gnn_vaamr_abstain = gnn_abstain
    if rater_votes is not None:
        s.rater_votes = rater_votes
        s.rater_ids = [rv['rater'] for rv in rater_votes]
    return s


def _build(run_dir, segs, n_items, session='c1s1'):
    segments_io.write_session_segments(run_dir, session, segs, 'h')
    classifications_io.write_theme_overlay(run_dir, segs)
    classifications_io.write_gnn_overlay(run_dir, segs)
    with db.open_db(run_dir) as conn:
        conn.execute("INSERT INTO testset_worksheets (worksheet_n,kind,name,n_items) "
                     "VALUES (1,'vaamr','ws1',?)", (n_items,))
        for s in segs:
            sha = hashlib.sha256(s.text.encode()).hexdigest()
            conn.execute("INSERT INTO testset_items (worksheet_n,item_num,session_id,seg_num,sha256) "
                         "VALUES (1,?,?,?,?)", (s.segment_index + 1, session, s.segment_index + 1, sha))


class TestAnalysis(unittest.TestCase):
    def _run(self, csv_rows, segs, n_items):
        run_dir = tempfile.mkdtemp()
        _build(run_dir, segs, n_items)
        csv = _write_csv(csv_rows)
        irr_import.import_irr_csv(run_dir, csv, verbose=False)
        res = irr_analysis.run_irr_analysis(run_dir, verbose=False)
        os.remove(csv)
        return res

    def test_human_human_perfect(self):
        rows = [
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",
        ]
        segs = [_seg('c1s1', 0, 'a', VIG, VIG), _seg('c1s1', 1, 'b', ATT, ATT)]
        res = self._run(rows, segs, 2)
        pri = res['human_human']['1']['primary']
        self.assertAlmostEqual(pri['krippendorff_alpha'], 1.0, places=6)
        for pr in pri['pairwise']:
            self.assertEqual(pr['cohen_kappa'], 1.0)

    def test_human_vs_llm_and_unresolved_excluded(self):
        rows = [
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",   # llm agrees
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",    # llm disagrees
            "1,1,3,Vigilance,,Attention,,Metacognition,,,,,,unresolved,",        # excluded
            "1,1,4,No code,,No code,,No code,,,,No code,,unanimous,",             # llm abstains (match)
        ]
        segs = [
            _seg('c1s1', 0, 'a', VIG, VIG),
            _seg('c1s1', 1, 'b', MET, MET),    # LLM = Metacognition, human = Attention
            _seg('c1s1', 2, 'c', ATT, ATT),
            _seg('c1s1', 3, 'd', None, None, gnn_abstain=1),  # LLM/GNN abstain
        ]
        res = self._run(rows, segs, 4)
        llm = res['human_vs_llm']
        self.assertEqual(llm['n'], 3)  # items 1,2,4 (3 excluded as unresolved)
        self.assertAlmostEqual(llm['percent_agreement'], 2 / 3)  # 1 & 4 agree, 2 disagrees
        # item 2 is the discrepancy
        disc_items = {d['item_num'] for d in res['_discrepancies']}
        self.assertIn(2, disc_items)
        self.assertNotIn(1, disc_items)

    def test_gnn_abstain_deferred(self):
        rows = [
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",
        ]
        segs = [
            _seg('c1s1', 0, 'a', VIG, VIG, gnn_abstain=1),  # deferred
            _seg('c1s1', 1, 'b', ATT, ATT, gnn_abstain=0),
        ]
        res = self._run(rows, segs, 2)
        gnn = res['human_vs_gnn']
        self.assertEqual(gnn['n_deferred'], 1)
        self.assertEqual(gnn['n'], 1)

    def test_per_llm_rater(self):
        rows = [
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",
        ]
        # modelA always matches the human consensus; modelB never does.
        rv1 = [{'rater': 'modelA', 'vote': 'CODED', 'stage': VIG},
               {'rater': 'modelB', 'vote': 'CODED', 'stage': REA}]
        rv2 = [{'rater': 'modelA', 'vote': 'CODED', 'stage': ATT},
               {'rater': 'modelB', 'vote': 'CODED', 'stage': REA}]
        segs = [_seg('c1s1', 0, 'a', VIG, VIG, rater_votes=rv1),
                _seg('c1s1', 1, 'b', ATT, ATT, rater_votes=rv2)]
        res = self._run(rows, segs, 2)
        per = res['human_vs_llm']['per_llm_rater']
        self.assertIn('modelA', per)
        self.assertIn('modelB', per)
        self.assertEqual(per['modelA']['percent_agreement'], 1.0)
        self.assertEqual(per['modelB']['percent_agreement'], 0.0)

    def test_no_human_codes_raises(self):
        run_dir = tempfile.mkdtemp()
        with self.assertRaises(RuntimeError):
            irr_analysis.run_irr_analysis(run_dir, verbose=False)

    def test_heldout_gnn_axis_preferred(self):
        from gnn_layer import validation as gval
        import numpy as np
        rows = [
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,reason one",
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",
        ]
        # distillation overlay says REA for both; held-out says the human label.
        segs = [_seg('c1s1', 0, 'a', VIG, REA), _seg('c1s1', 1, 'b', ATT, REA)]
        run_dir = tempfile.mkdtemp()
        _build(run_dir, segs, 2)
        # held-out predictions (out-of-fold) agree with the human consensus
        cv = {'vaamr': [('c1s1_000', VIG), ('c1s1_001', ATT)],
              'vaamr_logits': [('c1s1_000', np.array([5., 0, 0, 0, 0])),
                               ('c1s1_001', np.array([0, 0, 5., 0, 0]))]}
        gval.write_heldout_predictions_csv(cv, run_dir)
        csv = _write_csv(rows)
        irr_import.import_irr_csv(run_dir, csv, verbose=False)
        res = irr_analysis.run_irr_analysis(run_dir, verbose=False)
        os.remove(csv)
        self.assertEqual(res['gnn_axis'], 'heldout')
        # held-out matches human on both -> perfect agreement (distillation would be 0)
        self.assertEqual(res['human_vs_gnn']['percent_agreement'], 1.0)

    def test_item_details_content(self):
        rows = [
            "1,1,1,Vigilance,Attention,Vigilance,,Vigilance,,,,Vigilance,,unanimous,a debate note",
            "1,1,2,Vigilance,,Attention,,Metacognition,,,,,,unresolved,",
        ]
        rv = [{'rater': 'qwen', 'vote': 'CODED', 'stage': VIG, 'justification': 'qwen reason'}]
        segs = [_seg('c1s1', 0, 'first quote', VIG, VIG, rater_votes=rv),
                _seg('c1s1', 1, 'second quote', MET, MET, rater_votes=rv)]
        res = self._run(rows, segs, 2)
        items = res['_item_details']
        self.assertEqual(len(items), 2)  # includes the unresolved item
        it1 = items[0]
        self.assertEqual(it1['human']['consensus'], 'Vigilance')
        self.assertEqual(it1['human']['notes'], 'a debate note')
        self.assertEqual(it1['human']['raters']['becca']['secondary'], 'Attention Regulation')
        self.assertEqual(it1['text'], 'first quote')
        self.assertEqual(it1['llm']['raters'][0]['rater'], 'qwen')
        self.assertEqual(it1['llm']['raters'][0]['justification'], 'qwen reason')
        self.assertIn('agree', it1['llm_gnn_consensus'])
        # unresolved item is still detailed
        self.assertEqual(items[1]['human']['consensus'], 'unresolved')


class TestHeldoutRoundTrip(unittest.TestCase):
    def test_writer_reader(self):
        from gnn_layer import validation as gval
        import numpy as np
        run_dir = tempfile.mkdtemp()
        cv = {'vaamr': [('s0', 2), ('s1', 4)],
              'vaamr_logits': [('s0', np.array([0, 0, 9., 0, 0])),
                               ('s1', np.array([0, 0, 0, 0, 9.]))]}
        gval.write_heldout_predictions_csv(cv, run_dir)
        back = gval.read_heldout_predictions(run_dir)
        self.assertEqual(back['s0']['vaamr_pred'], 2)
        self.assertEqual(back['s1']['vaamr_pred'], 4)
        self.assertGreater(back['s0']['vaamr_conf'], 0.9)

    def test_reader_absent_returns_empty(self):
        from gnn_layer import validation as gval
        self.assertEqual(gval.read_heldout_predictions(tempfile.mkdtemp()), {})


class TestAutoRegen(unittest.TestCase):
    def _project(self):
        run_dir = tempfile.mkdtemp()
        segs = [_seg('c1s1', 0, 'a', VIG, VIG), _seg('c1s1', 1, 'b', ATT, ATT)]
        _build(run_dir, segs, 2)
        csv = _write_csv([
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",
        ])
        return run_dir, csv

    def test_skipped_without_human_codes(self):
        run_dir = tempfile.mkdtemp()
        self.assertIsNone(irr_analysis.maybe_run_irr(run_dir, verbose=False))

    def test_regen_then_unchanged(self):
        run_dir, csv = self._project()
        irr_import.import_irr_csv(run_dir, csv, verbose=False)
        os.remove(csv)
        first = irr_analysis.maybe_run_irr(run_dir, verbose=False)
        self.assertIsInstance(first, dict)
        # Second call with no machine change -> skipped.
        self.assertEqual(irr_analysis.maybe_run_irr(run_dir, verbose=False), 'unchanged')
        # force re-runs regardless.
        self.assertIsInstance(
            irr_analysis.maybe_run_irr(run_dir, verbose=False, force=True), dict)


class TestDriftEnforcement(unittest.TestCase):
    """run_irr_analysis must not silently score segments that drifted from the text the
    humans coded: strict_drift=True refuses; otherwise it records + reports the drift."""

    def _project(self):
        run_dir = tempfile.mkdtemp()
        segs = [_seg('c1s1', 0, 'alpha quote', VIG, VIG),
                _seg('c1s1', 1, 'beta quote', ATT, ATT)]
        _build(run_dir, segs, 2)
        csv = _write_csv([
            "1,1,1,Vigilance,,Vigilance,,Vigilance,,,,Vigilance,,unanimous,",
            "1,1,2,Attention,,Attention,,Attention,,,,Attention,,unanimous,",
        ])
        irr_import.import_irr_csv(run_dir, csv, verbose=False)
        os.remove(csv)
        return run_dir

    def _mutate(self, run_dir):
        with db.open_db(run_dir) as conn:
            conn.execute("UPDATE segments SET text=? WHERE segment_id=?",
                         ('TAMPERED CONTENT', 'c1s1_001'))

    def test_clean_run_has_empty_drift(self):
        res = irr_analysis.run_irr_analysis(self._project(), verbose=False)
        self.assertEqual(res['testset_drift'], [])

    def test_strict_drift_raises(self):
        run_dir = self._project()
        self._mutate(run_dir)
        with self.assertRaises(irr_import.TestsetDriftError):
            irr_analysis.run_irr_analysis(run_dir, verbose=False, strict_drift=True)

    def test_nonstrict_records_drift(self):
        run_dir = self._project()
        self._mutate(run_dir)
        res = irr_analysis.run_irr_analysis(run_dir, verbose=False, strict_drift=False)
        self.assertTrue(res['testset_drift'])
        self.assertEqual(res['testset_drift'][0]['kind'], 'drift')

    def test_machine_signature_tracks_coded_content(self):
        run_dir = self._project()
        sig_before = irr_analysis.machine_signature(run_dir)
        self._mutate(run_dir)
        self.assertNotEqual(irr_analysis.machine_signature(run_dir), sig_before)


if __name__ == '__main__':
    unittest.main()
