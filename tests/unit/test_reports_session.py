"""
tests/unit/test_reports_session.py
-----------------------------------
Unit tests for:
  - analysis/reports/session_report.py  (generate_comprehensive_session_report)
  - analysis/reports/session_txt_report.py (generate_session_txt_report,
                                            generate_all_session_txt_reports)

Covers:
  * File is written at the correct 06_reports/03_per_session/ path.
  * Key section headings and labels appear in the output.
  * Degenerate / empty input cases produce a valid file rather than crashing.
"""

import os
import sys
import shutil
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

import numpy as np
import pandas as pd

from process import output_paths as _paths
from analysis.reports.session_report import generate_comprehensive_session_report
from analysis.reports.session_txt_report import (
    generate_session_txt_report,
    generate_all_session_txt_reports,
)
from tests.testhelpers.fixtures import make_master_df


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FRAMEWORK = {
    0: {'short_name': 'Vigilance',            'name': 'Vigilance'},
    1: {'short_name': 'Avoidance',            'name': 'Avoidance'},
    2: {'short_name': 'Attention Regulation', 'name': 'Attention Regulation'},
    3: {'short_name': 'Metacognition',        'name': 'Metacognition'},
    4: {'short_name': 'Reappraisal',          'name': 'Reappraisal'},
}


def _make_session_json(session_id: str, session_number: int,
                       cohort_id: int = 1, n_participants: int = 2,
                       n_segments: int = 8) -> dict:
    """Minimal session-stats dict as produced by the analysis runner."""
    group_props = {str(s): round(0.2, 2) for s in range(5)}  # equal split
    exemplars = {
        '0': [{'participant_id': 'P01', 'text': 'I feel tense all day.', 'confidence': 0.82}],
        '2': [{'participant_id': 'P02', 'text': 'I can stay with the sensation.', 'confidence': 0.88}],
    }
    trans_mat = {
        '0': {'1': 2, '0': 1},
        '1': {'2': 3},
        '2': {'2': 2, '3': 1},
        '3': {'3': 1},
        '4': {},
    }
    participants = {
        'P01': {
            'n_segments': 4,
            'dominant_stage_name': 'Vigilance',
            'stage_proportions': {'0': 0.5, '1': 0.25, '2': 0.25, '3': 0.0, '4': 0.0},
        },
        'P02': {
            'n_segments': 4,
            'dominant_stage_name': 'Attention Regulation',
            'stage_proportions': {'0': 0.0, '1': 0.25, '2': 0.5, '3': 0.25, '4': 0.0},
        },
    }
    return {
        'session_id': session_id,
        'session_number': session_number,
        'cohort_id': cohort_id,
        'n_participants': n_participants,
        'n_segments': n_segments,
        'group_stage_proportions': group_props,
        'stage_exemplars': exemplars,
        'stage_transition_matrix': trans_mat,
        'participants': participants,
    }


# ---------------------------------------------------------------------------
# Tests for session_report.py
# ---------------------------------------------------------------------------

class TestComprehensiveSessionReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_df_two_sessions(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        # Rename session_ids to match our two test sessions
        mapping = {}
        for i, sid in enumerate(sorted(df['session_id'].unique())):
            mapping[sid] = f'c1s{i + 1}' if i < 2 else sid
        df['session_id'] = df['session_id'].map(mapping).fillna(df['session_id'])
        return df

    def test_writes_file_to_correct_path(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        session_ids = sorted(df['session_id'].unique())[:2]
        reports = [_make_session_json(sid, i + 1) for i, sid in enumerate(session_ids)]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)

        expected_dir = _paths.reports_per_session_dir(self.tmp)
        expected_path = os.path.join(expected_dir, '_overview.txt')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.isfile(path), f'File not found: {path}')

    def test_key_headings_present(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        session_ids = sorted(df['session_id'].unique())[:2]
        reports = [_make_session_json(sid, i + 1) for i, sid in enumerate(session_ids)]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)
        content = open(path, encoding='utf-8').read()

        self.assertIn('QRA COMPREHENSIVE THEME ANALYSIS', content)
        self.assertIn('THEME CONSISTENCY ANALYSIS', content)
        self.assertIn('Theme Distribution', content)
        self.assertIn('Transitions:', content)

    def test_stage_names_in_distribution(self):
        df = make_master_df(n_sessions=1, n_participants=2)
        session_ids = sorted(df['session_id'].unique())[:1]
        reports = [_make_session_json(session_ids[0], 1)]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)
        content = open(path, encoding='utf-8').read()

        for name in ('Vigilance', 'Avoidance', 'Attention Regulation'):
            self.assertIn(name, content)

    def test_exemplar_quotes_included(self):
        df = make_master_df(n_sessions=1, n_participants=2)
        session_ids = sorted(df['session_id'].unique())[:1]
        reports = [_make_session_json(session_ids[0], 1)]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)
        content = open(path, encoding='utf-8').read()

        self.assertIn('I feel tense all day', content)

    def test_empty_session_reports_list(self):
        """Empty session_reports list should still produce a valid (minimal) file."""
        df = pd.DataFrame(columns=['participant_id', 'session_id', 'final_label'])
        path = generate_comprehensive_session_report(df, _FRAMEWORK, [], self.tmp)

        self.assertTrue(os.path.isfile(path))
        content = open(path, encoding='utf-8').read()
        self.assertIn('QRA COMPREHENSIVE THEME ANALYSIS', content)

    def test_none_reports_skipped(self):
        """None entries in session_reports should be skipped gracefully."""
        df = make_master_df(n_sessions=1, n_participants=1)
        session_ids = sorted(df['session_id'].unique())[:1]
        reports = [_make_session_json(session_ids[0], 1), None]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)
        self.assertTrue(os.path.isfile(path))

    def test_theme_consistency_none_present_all(self):
        """With only one session each stage appears at most in one session.

        The report's consistency block should appear and not crash.
        """
        df = make_master_df(n_sessions=1, n_participants=1)
        session_ids = sorted(df['session_id'].unique())[:1]
        reports = [_make_session_json(session_ids[0], 1)]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)
        content = open(path, encoding='utf-8').read()
        # Single session → themes_all is empty OR all present — either is fine
        self.assertIn('THEME CONSISTENCY ANALYSIS', content)

    def test_multiple_sessions_session_ids_appear(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        session_ids = sorted(df['session_id'].unique())[:2]
        reports = [_make_session_json(sid, i + 1) for i, sid in enumerate(session_ids)]

        path = generate_comprehensive_session_report(df, _FRAMEWORK, reports, self.tmp)
        content = open(path, encoding='utf-8').read()

        for sid in session_ids:
            self.assertIn(sid, content, f'Session ID {sid} not found in overview')


# ---------------------------------------------------------------------------
# Tests for session_txt_report.py
# ---------------------------------------------------------------------------

class TestSessionTxtReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_df_for_session(self, session_id: str) -> pd.DataFrame:
        """Participant-only DataFrame for a single session."""
        df = make_master_df(n_sessions=1, n_participants=2)
        # Remap the session_id to the one we want
        df['session_id'] = session_id
        return df[df['speaker'] == 'participant'].copy()

    def test_writes_file_to_correct_path(self):
        sid = 'c1s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)

        expected_dir = _paths.reports_per_session_dir(self.tmp)
        expected_path = os.path.join(expected_dir, f'session_{sid}.txt')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.isfile(path))

    def test_key_sections_present(self):
        sid = 'c1s2'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 2)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        self.assertIn('SESSION INSTRUCTION SUMMARY', content)
        self.assertIn('OVERVIEW', content)
        self.assertIn('STAGE DISTRIBUTION', content)
        self.assertIn('TRANSITIONS THIS SESSION', content)
        self.assertIn('PER-PARTICIPANT BREAKDOWN', content)

    def test_session_id_and_number_in_header(self):
        sid = 'c1s3'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 3)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        self.assertIn(f'SESSION {sid}', content)
        self.assertIn('Session 3', content)

    def test_stage_names_in_distribution(self):
        sid = 'c1s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        for name in ('Vigilance', 'Avoidance', 'Attention Regulation', 'Metacognition'):
            self.assertIn(name, content)

    def test_participant_breakdown_present(self):
        sid = 'c1s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        self.assertIn('P01', content)
        self.assertIn('P02', content)

    def test_purer_distribution_when_present(self):
        """When df_all contains therapist segments with purer_primary, PURER section appears."""
        sid = 'c1s1'
        df_all = make_master_df(n_sessions=1, n_participants=2)
        df_all['session_id'] = sid
        df_part = df_all[df_all['speaker'] == 'participant'].copy()
        session_json = _make_session_json(sid, 1)

        path = generate_session_txt_report(
            df_part, sid, session_json, _FRAMEWORK, self.tmp,
            df_all=df_all,
        )
        content = open(path, encoding='utf-8').read()

        # PURER distribution section should appear (therapists have purer_primary)
        self.assertIn('PURER', content)

    def test_session_summaries_used_when_provided(self):
        sid = 'c1s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1)
        summaries = {sid: 'Today the group focused on body scanning and pain acceptance.'}

        path = generate_session_txt_report(
            df, sid, session_json, _FRAMEWORK, self.tmp,
            session_summaries=summaries,
        )
        content = open(path, encoding='utf-8').read()
        self.assertIn('body scanning', content)

    def test_placeholder_when_no_summaries(self):
        sid = 'c1s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('[session summaries not generated]', content)

    def test_cohort_label_in_header(self):
        sid = 'c2s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1, cohort_id=2)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('Cohort 2', content)

    def test_empty_dataframe_does_not_crash(self):
        """An empty DataFrame for the session should produce a file without crashing."""
        sid = 'c1s1'
        df_empty = pd.DataFrame(columns=[
            'participant_id', 'session_id', 'speaker', 'final_label',
            'text', 'start_time_ms', 'end_time_ms', 'segment_index',
            'llm_confidence_primary',
        ])
        session_json = _make_session_json(sid, 1, n_segments=0)

        path = generate_session_txt_report(df_empty, sid, session_json, _FRAMEWORK, self.tmp)
        self.assertTrue(os.path.isfile(path))
        content = open(path, encoding='utf-8').read()
        self.assertIn('SESSION INSTRUCTION SUMMARY', content)

    def test_transition_counts_rendered(self):
        """Forward/backward/lateral transition counts should appear."""
        sid = 'c1s1'
        df = self._make_df_for_session(sid)
        session_json = _make_session_json(sid, 1)

        path = generate_session_txt_report(df, sid, session_json, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('Forward:', content)
        self.assertIn('Backward:', content)
        self.assertIn('Lateral:', content)


class TestGenerateAllSessionTxtReports(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generates_one_file_per_session(self):
        df = make_master_df(n_sessions=2, n_participants=2)
        session_ids = sorted(df['session_id'].unique())[:2]
        reports = [_make_session_json(sid, i + 1) for i, sid in enumerate(session_ids)]

        # Filter to participant rows only (generator adds df filtering internally)
        df_part = df[df['speaker'] == 'participant'].copy()

        paths = generate_all_session_txt_reports(df_part, reports, _FRAMEWORK, self.tmp)

        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f'Missing: {p}')

    def test_skips_reports_with_no_session_id(self):
        df = make_master_df(n_sessions=1, n_participants=1)
        df_part = df[df['speaker'] == 'participant'].copy()

        bad_report = {'session_number': 1}   # missing session_id key
        sid = sorted(df['session_id'].unique())[0]
        good_report = _make_session_json(sid, 1)

        paths = generate_all_session_txt_reports(
            df_part, [bad_report, good_report], _FRAMEWORK, self.tmp
        )
        self.assertEqual(len(paths), 1)

    def test_empty_reports_list_returns_empty(self):
        df = make_master_df(n_sessions=1, n_participants=1)
        df_part = df[df['speaker'] == 'participant'].copy()

        paths = generate_all_session_txt_reports(df_part, [], _FRAMEWORK, self.tmp)
        self.assertEqual(paths, [])


if __name__ == '__main__':
    unittest.main()
