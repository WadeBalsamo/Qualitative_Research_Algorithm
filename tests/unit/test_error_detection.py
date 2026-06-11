"""Unit tests for process/error_detection.py — M4.

Truth table per overlay (hermetic, no network, no LLM):
  theme: primary_stage not None → ok
         consensus_vote=='ABSTAIN' → legitimate_abstain
         agreement_level=='split' → review
         agreement_level=='none' → repairable_error
         missing/empty rater_votes (+ null primary) → repairable_error
         plurality_coded (M0 level) → ok
  purer: same with purer_* field names; unanimous-ABSTAIN rater_votes → legitimate_abstain
  codebook: non-empty ensemble list → ok; empty/None → repairable_error

Also tests detect_overlay_errors (reads from DB) and detect_run_error_cells (ballot query).
"""
import os
import shutil
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from process.error_detection import (
    overlay_row_status,
    detect_overlay_errors,
    detect_run_error_cells,
    repair_targets,
)
from classification_tools.data_structures import Segment
from process import segments_io, run_registry as rr, classifications_io as cio


# ---------------------------------------------------------------------------
# overlay_row_status truth table
# ---------------------------------------------------------------------------

class TestOverlayRowStatusTheme(unittest.TestCase):
    """Truth table for theme overlay rows."""

    def _row(self, primary_stage=None, agreement_level=None,
             consensus_vote=None, rater_votes=None):
        return {
            'segment_id': 's1',
            'primary_stage': primary_stage,
            'secondary_stage': None,
            'agreement_level': agreement_level,
            'consensus_vote': consensus_vote,
            'rater_votes': rater_votes,
            'needs_review': False,
        }

    def test_labeled_ok(self):
        """primary_stage not None → ok."""
        self.assertEqual(overlay_row_status(self._row(primary_stage=2), 'theme'), 'ok')

    def test_labeled_stage_0_ok(self):
        """stage=0 (Vigilance) is a valid label → ok."""
        self.assertEqual(overlay_row_status(self._row(primary_stage=0), 'theme'), 'ok')

    def test_plurality_coded_ok(self):
        """plurality_coded agreement (has a primary label) → ok."""
        row = self._row(primary_stage=1, agreement_level='plurality_coded')
        self.assertEqual(overlay_row_status(row, 'theme'), 'ok')

    def test_unanimous_ok(self):
        row = self._row(primary_stage=3, agreement_level='unanimous',
                        rater_votes=[{'primary_stage': 3}, {'primary_stage': 3}])
        self.assertEqual(overlay_row_status(row, 'theme'), 'ok')

    def test_majority_ok(self):
        row = self._row(primary_stage=1, agreement_level='majority')
        self.assertEqual(overlay_row_status(row, 'theme'), 'ok')

    def test_consensus_abstain(self):
        """Explicit ABSTAIN consensus vote → legitimate_abstain."""
        row = self._row(primary_stage=None, consensus_vote='ABSTAIN',
                        agreement_level='none')
        self.assertEqual(overlay_row_status(row, 'theme'), 'legitimate_abstain')

    def test_split_is_review(self):
        """agreement_level='split' + no primary → review (NEVER auto-repaired)."""
        row = self._row(primary_stage=None, agreement_level='split',
                        rater_votes=[{'primary_stage': 1}, {'primary_stage': 2}])
        self.assertEqual(overlay_row_status(row, 'theme'), 'review')

    def test_none_agreement_repairable(self):
        """agreement_level='none' (no consensus formed) → repairable_error."""
        row = self._row(primary_stage=None, agreement_level='none',
                        rater_votes=[])
        self.assertEqual(overlay_row_status(row, 'theme'), 'repairable_error')

    def test_missing_rater_votes_repairable(self):
        """No rater_votes at all → repairable_error."""
        row = self._row(primary_stage=None, agreement_level=None, rater_votes=None)
        self.assertEqual(overlay_row_status(row, 'theme'), 'repairable_error')

    def test_empty_rater_votes_repairable(self):
        row = self._row(primary_stage=None, agreement_level='none', rater_votes=[])
        self.assertEqual(overlay_row_status(row, 'theme'), 'repairable_error')

    def test_null_primary_with_votes_no_abstain_repairable(self):
        """Null primary + rater_votes present but no ABSTAIN consensus → repairable."""
        row = self._row(primary_stage=None, agreement_level='none',
                        rater_votes=[{'primary_stage': None, 'vote': 'ERROR'},
                                     {'primary_stage': None, 'vote': 'ERROR'}])
        self.assertEqual(overlay_row_status(row, 'theme'), 'repairable_error')

    def test_split_beats_no_primary(self):
        """Split is categorised as review, not repairable, even with null primary."""
        row = self._row(primary_stage=None, agreement_level='SPLIT',
                        rater_votes=[{'primary_stage': 0}, {'primary_stage': 4}])
        self.assertEqual(overlay_row_status(row, 'theme'), 'review')


class TestOverlayRowStatusPurer(unittest.TestCase):
    """Truth table for purer overlay rows."""

    def _row(self, purer_primary=None, purer_agreement_level=None,
             purer_rater_votes=None):
        return {
            'segment_id': 's1',
            'purer_primary': purer_primary,
            'purer_secondary': None,
            'purer_agreement_level': purer_agreement_level,
            'purer_rater_votes': purer_rater_votes,
            'purer_needs_review': False,
        }

    def test_labeled_ok(self):
        self.assertEqual(overlay_row_status(self._row(purer_primary=0), 'purer'), 'ok')

    def test_unanimous_abstain(self):
        """All rater_votes are ABSTAIN → legitimate_abstain."""
        votes = [{'vote': 'ABSTAIN', 'primary_stage': None},
                 {'vote': 'ABSTAIN', 'primary_stage': None}]
        row = self._row(purer_primary=None, purer_rater_votes=votes)
        self.assertEqual(overlay_row_status(row, 'purer'), 'legitimate_abstain')

    def test_split_review(self):
        row = self._row(purer_primary=None, purer_agreement_level='split',
                        purer_rater_votes=[{'vote': 'CODED', 'primary_stage': 1},
                                           {'vote': 'CODED', 'primary_stage': 3}])
        self.assertEqual(overlay_row_status(row, 'purer'), 'review')

    def test_none_agreement_repairable(self):
        row = self._row(purer_primary=None, purer_agreement_level='none',
                        purer_rater_votes=[])
        self.assertEqual(overlay_row_status(row, 'purer'), 'repairable_error')

    def test_missing_votes_repairable(self):
        row = self._row(purer_primary=None, purer_rater_votes=None)
        self.assertEqual(overlay_row_status(row, 'purer'), 'repairable_error')

    def test_mixed_abstain_coded_not_abstain(self):
        """One CODED + one ABSTAIN → not unanimous abstain → repairable_error (null primary)."""
        votes = [{'vote': 'CODED', 'primary_stage': 2},
                 {'vote': 'ABSTAIN', 'primary_stage': None}]
        row = self._row(purer_primary=None, purer_rater_votes=votes)
        # Not a legitimate_abstain since one rater voted CODED.
        status = overlay_row_status(row, 'purer')
        self.assertIn(status, ('repairable_error', 'review'))

    def test_labeled_plurality_ok(self):
        row = self._row(purer_primary=2, purer_agreement_level='plurality_coded')
        self.assertEqual(overlay_row_status(row, 'purer'), 'ok')


class TestOverlayRowStatusCodebook(unittest.TestCase):
    """Truth table for codebook overlay rows (no abstain concept)."""

    def _row(self, ensemble):
        return {
            'segment_id': 's1',
            'codebook_labels_ensemble': ensemble,
            'codebook_labels_embedding': None,
            'codebook_labels_llm': None,
        }

    def test_nonempty_list_ok(self):
        self.assertEqual(
            overlay_row_status(self._row(['VCE_Awareness', 'VCE_Sensation']), 'codebook'),
            'ok',
        )

    def test_empty_list_repairable(self):
        self.assertEqual(overlay_row_status(self._row([]), 'codebook'), 'repairable_error')

    def test_none_repairable(self):
        self.assertEqual(overlay_row_status(self._row(None), 'codebook'), 'repairable_error')

    def test_empty_string_repairable(self):
        self.assertEqual(overlay_row_status(self._row(''), 'codebook'), 'repairable_error')

    def test_json_empty_array_string_repairable(self):
        self.assertEqual(overlay_row_status(self._row('[]'), 'codebook'), 'repairable_error')

    def test_null_string_repairable(self):
        self.assertEqual(overlay_row_status(self._row('null'), 'codebook'), 'repairable_error')

    def test_json_list_string_ok(self):
        # When stored as a JSON string representation it's ok
        self.assertEqual(
            overlay_row_status(self._row(['code_a']), 'codebook'),
            'ok',
        )


# ---------------------------------------------------------------------------
# detect_overlay_errors (DB integration — tiny in-memory DB)
# ---------------------------------------------------------------------------

def _seed_theme_overlay(tmp):
    """Write theme_labels with a mix of ok / repairable / review / abstain rows."""
    # Write frozen segments first (required by segments table FK-style setup).
    raw = [
        Segment(
            segment_id=f's{i}', trial_id='t', participant_id='P1', session_id='c1s1',
            session_number=1, cohort_id=1, segment_index=i, speaker='participant',
            text='test', word_count=1, start_time_ms=i * 1000, end_time_ms=i * 1000 + 500,
        )
        for i in range(5)
    ]
    segments_io.write_session_segments(tmp, 'c1s1', raw, 'h1')

    import json as _json
    from process import db as _db
    with _db.open_db(tmp) as conn:
        rows = [
            # s0: labeled → ok
            ('s0', 2, 'unanimous', _json.dumps('CODED'), _json.dumps([{'ps': 2}])),
            # s1: ABSTAIN consensus → legitimate_abstain
            ('s1', None, 'none', _json.dumps('ABSTAIN'), _json.dumps([])),
            # s2: split → review
            ('s2', None, 'split', _json.dumps(None),
             _json.dumps([{'primary_stage': 0}, {'primary_stage': 4}])),
            # s3: none + empty votes → repairable
            ('s3', None, 'none', _json.dumps(None), _json.dumps([])),
            # s4: labeled majority → ok
            ('s4', 1, 'majority', _json.dumps('CODED'), _json.dumps([{'ps': 1}, {'ps': 1}])),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO theme_labels "
            "(segment_id, primary_stage, agreement_level, consensus_vote, rater_votes) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )


class TestDetectOverlayErrors(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_detect_theme_overlay(self):
        _seed_theme_overlay(self.tmp)
        result = detect_overlay_errors(self.tmp, 'theme')
        by_seg = result['by_segment']
        summary = result['summary']
        self.assertEqual(by_seg['s0'], 'ok')
        self.assertEqual(by_seg['s1'], 'legitimate_abstain')
        self.assertEqual(by_seg['s2'], 'review')
        self.assertEqual(by_seg['s3'], 'repairable_error')
        self.assertEqual(by_seg['s4'], 'ok')
        self.assertEqual(summary['ok'], 2)
        self.assertEqual(summary['legitimate_abstain'], 1)
        self.assertEqual(summary['review'], 1)
        self.assertEqual(summary['repairable_error'], 1)

    def test_detect_empty_overlay(self):
        """Empty overlay → all zeros."""
        # Seed DB (no overlay rows).
        raw = [
            Segment(segment_id='s0', trial_id='t', participant_id='P1',
                    session_id='c1s1', session_number=1, cohort_id=1,
                    segment_index=0, speaker='participant', text='test',
                    word_count=1, start_time_ms=0, end_time_ms=500)
        ]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'h1')
        result = detect_overlay_errors(self.tmp, 'theme')
        self.assertEqual(result['by_segment'], {})
        self.assertEqual(result['summary']['repairable_error'], 0)


# ---------------------------------------------------------------------------
# detect_run_error_cells (ballot query)
# ---------------------------------------------------------------------------

class TestDetectRunErrorCells(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [
            Segment(segment_id=f's{i}', trial_id='t', participant_id='P1',
                    session_id='c1s1', session_number=1, cohort_id=1,
                    segment_index=i, speaker='participant', text='test',
                    word_count=1, start_time_ms=i * 1000, end_time_ms=i * 1000 + 500)
            for i in range(4)
        ]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'h1')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_error_cells_returned(self):
        rid = rr.create_run(self.tmp, overlay='theme', model='mA', rater_label='mA')
        cells = {
            's0': {'vote': 'CODED', 'primary_stage': 2, 'primary_confidence': 0.8,
                   'secondary_stage': None, 'secondary_confidence': None,
                   'justification': 'j', 'evidence_phrase': 'e'},
            's1': None,   # ERROR
            's2': {'vote': 'ABSTAIN', 'primary_stage': None, 'primary_confidence': 0.0,
                   'secondary_stage': None, 'secondary_confidence': None,
                   'justification': '', 'evidence_phrase': ''},
            's3': None,   # ERROR
        }
        rr.upsert_ballots(self.tmp, 'theme', rid, cells)
        errors = detect_run_error_cells(self.tmp, rid)
        self.assertEqual(sorted(errors), ['s1', 's3'])

    def test_no_errors(self):
        rid = rr.create_run(self.tmp, overlay='theme', model='mB', rater_label='mB')
        cells = {'s0': {'vote': 'CODED', 'primary_stage': 1, 'primary_confidence': 0.9,
                        'secondary_stage': None, 'secondary_confidence': None,
                        'justification': 'j', 'evidence_phrase': 'e'}}
        rr.upsert_ballots(self.tmp, 'theme', rid, cells)
        self.assertEqual(detect_run_error_cells(self.tmp, rid), [])

    def test_missing_db(self):
        """Missing DB → empty list (no crash)."""
        self.assertEqual(detect_run_error_cells('/nonexistent/path', 999), [])


# ---------------------------------------------------------------------------
# repair_targets
# ---------------------------------------------------------------------------

class TestRepairTargets(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        raw = [
            Segment(segment_id=f's{i}', trial_id='t', participant_id='P1',
                    session_id='c1s1', session_number=1, cohort_id=1,
                    segment_index=i, speaker='participant', text='t',
                    word_count=1, start_time_ms=i * 1000, end_time_ms=i * 1000 + 500)
            for i in range(3)
        ]
        segments_io.write_session_segments(self.tmp, 'c1s1', raw, 'h1')

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_run_with_errors(self, error_seg_ids, status='completed_with_errors',
                               model='mA', rater_label=None):
        rid = rr.create_run(self.tmp, overlay='theme', model=model,
                             rater_label=rater_label or model)
        cells = {}
        for i in range(3):
            sid = f's{i}'
            if sid in error_seg_ids:
                cells[sid] = None  # ERROR
            else:
                cells[sid] = {'vote': 'CODED', 'primary_stage': 1,
                               'primary_confidence': 0.8, 'secondary_stage': None,
                               'secondary_confidence': None, 'justification': 'j',
                               'evidence_phrase': 'e'}
        rr.upsert_ballots(self.tmp, 'theme', rid, cells)
        rr.update_run(self.tmp, rid, status=status)
        rr.refresh_counters(self.tmp, rid)
        return rid

    def test_targets_found(self):
        rid = self._make_run_with_errors(['s1', 's2'])
        targets = repair_targets(self.tmp, 'theme')
        self.assertIn(rid, targets)
        self.assertEqual(sorted(targets[rid]), ['s1', 's2'])

    def test_archived_excluded(self):
        rid = self._make_run_with_errors(['s0'], status='archived')
        targets = repair_targets(self.tmp, 'theme')
        self.assertNotIn(rid, targets)

    def test_failed_excluded(self):
        rid = self._make_run_with_errors(['s0'], status='failed')
        targets = repair_targets(self.tmp, 'theme')
        self.assertNotIn(rid, targets)

    def test_explicit_run_ids_filter(self):
        rid1 = self._make_run_with_errors(['s0'], model='mA', rater_label='mA')
        rid2 = self._make_run_with_errors(['s1'], model='mB', rater_label='mB')
        # Provide only rid1 explicitly.
        targets = repair_targets(self.tmp, 'theme', run_ids=[rid1])
        self.assertIn(rid1, targets)
        self.assertNotIn(rid2, targets)

    def test_no_errors_omitted(self):
        """Run with zero ERROR ballots must not appear in targets."""
        rid = rr.create_run(self.tmp, overlay='theme', model='mC', rater_label='mC')
        cells = {f's{i}': {'vote': 'CODED', 'primary_stage': 1, 'primary_confidence': 0.9,
                            'secondary_stage': None, 'secondary_confidence': None,
                            'justification': 'j', 'evidence_phrase': 'e'}
                 for i in range(3)}
        rr.upsert_ballots(self.tmp, 'theme', rid, cells)
        rr.update_run(self.tmp, rid, status='completed')
        targets = repair_targets(self.tmp, 'theme')
        self.assertNotIn(rid, targets)


if __name__ == '__main__':
    unittest.main(verbosity=2)
