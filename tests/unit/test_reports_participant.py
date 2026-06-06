"""
tests/unit/test_reports_participant.py
---------------------------------------
Unit tests for analysis/reports/participant_txt_report.py.

Covers:
  * generate_participant_txt_report  — file written at correct 06_reports path,
    key sections and labels appear, participant ID slug stripping.
  * generate_all_participant_txt_reports — one file per participant, graceful
    handling of missing/empty participant ids.
  * Degenerate / empty DataFrame does not crash.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from process import output_paths as _paths
from analysis.reports.participant_txt_report import (
    generate_participant_txt_report,
    generate_all_participant_txt_reports,
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


def _make_participant_json(participant_id: str, session_ids: list,
                           cohort_id: int = 1,
                           trend: float = 0.25) -> dict:
    """Minimal per-participant analysis dict."""
    sessions_detail = {}
    for i, sid in enumerate(session_ids):
        dom = i % 5  # cycle through stages across sessions
        sessions_detail[sid] = {
            'n_segments': 4,
            'dominant_stage': dom,
            'dominant_stage_name': _FRAMEWORK[dom]['short_name'],
            'mean_confidence': 0.82,
            'progression_score': float(dom),
        }

    stage_exemplars_overall = {
        '0': {'text': 'I keep noticing pain signals.', 'confidence': 0.85, 'session_id': session_ids[0]},
        '2': {'text': 'I can stay present with discomfort.', 'confidence': 0.90, 'session_id': session_ids[-1]},
    }

    return {
        'participant_id': participant_id,
        'cohort_id': cohort_id,
        'session_ids': list(session_ids),
        'sessions': sessions_detail,
        'n_sessions': len(session_ids),
        'progression_trend': trend,
        'stage_exemplars_overall': stage_exemplars_overall,
    }


# ---------------------------------------------------------------------------
# Tests for generate_participant_txt_report
# ---------------------------------------------------------------------------

class TestParticipantTxtReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _df_for_participant(self, pid: str, session_ids: list) -> pd.DataFrame:
        df = make_master_df(n_sessions=len(session_ids), n_participants=1)
        # remap session_ids to those we supply
        old_sids = sorted(df['session_id'].unique())
        remap = {old: new for old, new in zip(old_sids, session_ids)}
        df['session_id'] = df['session_id'].map(remap).fillna(df['session_id'])
        df['participant_id'] = pid
        return df[df['speaker'] == 'participant'].copy()

    def test_writes_file_to_correct_path(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)

        expected_dir = _paths.reports_per_participant_dir(self.tmp)
        expected_path = os.path.join(expected_dir, 'participant_P01.txt')
        self.assertEqual(path, expected_path)
        self.assertTrue(os.path.isfile(path))

    def test_key_sections_present(self):
        pid = 'P02'
        session_ids = ['c1s1', 'c1s2', 'c1s3']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        for heading in (
            'OVERVIEW',
            'PARTICIPATION STATISTICS',
            'STAGE DISTRIBUTION',
            'LONGITUDINAL TRAJECTORY',
            'BEST EXPRESSIONS BY STAGE',
        ):
            self.assertIn(heading, content, f'Missing section: {heading}')

    def test_participant_id_in_header(self):
        pid = 'P03'
        session_ids = ['c1s1']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn(f'PARTICIPANT {pid}', content)

    def test_cohort_label_in_header(self):
        pid = 'P01'
        session_ids = ['c2s1', 'c2s2']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids, cohort_id=2)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('Cohort 2', content)

    def test_progression_trend_advancing(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids, trend=0.5)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('advancing', content)

    def test_progression_trend_regressing(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids, trend=-0.5)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('regressing', content)

    def test_progression_trend_stable(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids, trend=0.0)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('stable', content)

    def test_session_ids_appear_in_trajectory(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2', 'c1s3']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        for sid in session_ids:
            self.assertIn(sid, content, f'Session {sid} missing from trajectory')

    def test_between_session_transitions_section(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2', 'c1s3']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('BETWEEN-SESSION TRANSITIONS', content)

    def test_exemplar_quotes_present(self):
        pid = 'P01'
        session_ids = ['c1s1']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        # Our helper injects these texts into stage_exemplars_overall
        self.assertIn('I keep noticing pain signals', content)
        self.assertIn('I can stay present with discomfort', content)

    def test_stage_names_in_distribution(self):
        pid = 'P01'
        session_ids = ['c1s1', 'c1s2']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()

        for name in ('Vigilance', 'Avoidance', 'Attention Regulation', 'Metacognition', 'Reappraisal'):
            self.assertIn(name, content)

    def test_participant_prefix_stripped_from_filename(self):
        """participant_P01 should yield participant_P01.txt, not participant_Participant_P01.txt."""
        pid = 'Participant_P01'
        session_ids = ['c1s1']
        df = make_master_df(n_sessions=1, n_participants=1)
        df['participant_id'] = pid
        df['session_id'] = 'c1s1'
        df_part = df[df['speaker'] == 'participant'].copy()
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df_part, pid, pjson, _FRAMEWORK, self.tmp)

        basename = os.path.basename(path)
        # After stripping 'Participant_', the file should be participant_P01.txt
        self.assertEqual(basename, 'participant_P01.txt')
        self.assertTrue(os.path.isfile(path))

    def test_empty_dataframe_does_not_crash(self):
        """If the participant has no rows in the DataFrame, report is minimal but valid."""
        pid = 'P99'
        session_ids = ['c1s1']
        df_empty = pd.DataFrame(columns=[
            'participant_id', 'session_id', 'speaker', 'final_label', 'text',
            'start_time_ms', 'end_time_ms', 'segment_index', 'llm_confidence_primary',
        ])
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df_empty, pid, pjson, _FRAMEWORK, self.tmp)
        self.assertTrue(os.path.isfile(path))
        content = open(path, encoding='utf-8').read()
        self.assertIn(f'PARTICIPANT {pid}', content)

    def test_single_session_no_between_session_section(self):
        """With only one session the between-session transitions section is omitted."""
        pid = 'P01'
        session_ids = ['c1s1']
        df = self._df_for_participant(pid, session_ids)
        pjson = _make_participant_json(pid, session_ids)

        path = generate_participant_txt_report(df, pid, pjson, _FRAMEWORK, self.tmp)
        content = open(path, encoding='utf-8').read()
        # One session — no between-session block expected
        self.assertNotIn('BETWEEN-SESSION TRANSITIONS', content)


# ---------------------------------------------------------------------------
# Tests for generate_all_participant_txt_reports
# ---------------------------------------------------------------------------

class TestGenerateAllParticipantTxtReports(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_generates_one_file_per_participant(self):
        df = make_master_df(n_sessions=2, n_participants=3)
        df_part = df[df['speaker'] == 'participant'].copy()

        participant_ids = sorted(df_part['participant_id'].unique())
        reports = []
        for pid in participant_ids:
            sids = sorted(df_part[df_part['participant_id'] == pid]['session_id'].unique())
            reports.append(_make_participant_json(pid, sids))

        paths = generate_all_participant_txt_reports(df_part, reports, _FRAMEWORK, self.tmp)

        self.assertEqual(len(paths), 3)
        for p in paths:
            self.assertTrue(os.path.isfile(p), f'Missing: {p}')

    def test_skips_reports_without_participant_id(self):
        df = make_master_df(n_sessions=1, n_participants=1)
        df_part = df[df['speaker'] == 'participant'].copy()
        pid = df_part['participant_id'].iloc[0]
        sids = sorted(df_part['session_id'].unique())

        good = _make_participant_json(pid, sids)
        bad = {'session_ids': sids}   # missing participant_id

        paths = generate_all_participant_txt_reports(df_part, [bad, good], _FRAMEWORK, self.tmp)
        self.assertEqual(len(paths), 1)

    def test_empty_list_returns_empty(self):
        df = make_master_df(n_sessions=1, n_participants=1)
        df_part = df[df['speaker'] == 'participant'].copy()

        paths = generate_all_participant_txt_reports(df_part, [], _FRAMEWORK, self.tmp)
        self.assertEqual(paths, [])


if __name__ == '__main__':
    unittest.main()
