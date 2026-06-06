"""
tests/unit/test_text_reports.py
---------------------------------
Unit tests for text-building helpers in the analysis/reports layer.

Since there is no single analysis/text_reports.py module, this file covers
the shared formatting helpers in analysis/reports/_formatting.py and the
public output-generating functions exported from analysis/reports/__init__.py
(generate_comprehensive_session_report, generate_longitudinal_text_report,
generate_all_stage_text_reports).

All tests use tiny in-memory DataFrames; no network, no LLM, no model weights.
"""

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import make_master_df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mini_framework():
    return {
        i: {'id': i, 'key': f's{i}', 'name': f'Stage {i}', 'short_name': f'S{i}', 'definition': ''}
        for i in range(5)
    }


def _part_df(n_sessions=2, n_participants=2):
    df = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    df = df[df['speaker'] == 'participant'].copy()
    df = df.rename(columns={'confidence_tier': 'label_confidence_tier'})
    df['final_label'] = df['final_label'].astype(int)
    return df


def _fake_session_reports(df):
    """Build minimal session-report dicts like generate_all_session_analyses produces."""
    fw = _mini_framework()
    stage_ids = sorted(fw.keys())
    reports = []
    for sid in df['session_id'].unique():
        sdf = df[df['session_id'] == sid]
        n = len(sdf)
        if n == 0:
            continue
        stage_counts = sdf['final_label'].value_counts().to_dict()
        group_props = {str(st): round(stage_counts.get(st, 0) / n, 4) for st in stage_ids}
        dominant = str(int(sdf['final_label'].mode().iloc[0]))
        reports.append({
            'session_id': sid,
            'session_number': int(sdf['session_number'].iloc[0]),
            'n_segments': n,
            'n_participants': sdf['participant_id'].nunique(),
            'participant_ids': sorted(sdf['participant_id'].unique().tolist()),
            'participants': {},
            'group_stage_proportions': group_props,
            'stage_transition_matrix': {str(i): {str(j): 0 for j in range(5)} for i in range(5)},
            'confidence_distribution': {
                'high': {'count': n, 'proportion': 1.0},
                'medium': {'count': 0, 'proportion': 0.0},
                'low': {'count': 0, 'proportion': 0.0},
            },
            'top_codebook_codes': [],
            'stage_exemplars': {str(st): [] for st in stage_ids},
            'secondary_stage_exemplars': {},
            'superposition': None,
            'narrative_summary': f'Session {sid} narrative.',
            'dual_coded_count': 0,
            'dual_coded_pct': 0.0,
            'cohort_id': int(sdf['cohort_id'].iloc[0]) if 'cohort_id' in sdf.columns else None,
            'stage_sequence': sdf['final_label'].tolist(),
        })
    return reports


def _fake_participant_reports(df):
    """Build minimal participant-report dicts like generate_all_participant_reports produces."""
    reports = []
    for pid in df['participant_id'].unique():
        pdf = df[df['participant_id'] == pid]
        sessions = pdf['session_id'].unique().tolist()
        prog = {str(i + 1): float(i) * 0.5 for i, _ in enumerate(sessions)}
        reports.append({
            'participant_id': pid,
            'n_sessions': len(sessions),
            'session_ids': sessions,
            'sessions': {
                sid: {
                    'session_id': sid,
                    'session_number': int(pdf[pdf['session_id'] == sid]['session_number'].iloc[0]),
                    'n_segments': len(pdf[pdf['session_id'] == sid]),
                    'stage_proportions': {'0': 0.5, '1': 0.25, '2': 0.25, '3': 0.0, '4': 0.0},
                    'dominant_stage': 0,
                    'dominant_stage_name': 'S0',
                    'mean_confidence': 0.85,
                    'progression_score': 0.75,
                }
                for sid in sessions
            },
            'longitudinal_trajectory': {
                str(i + 1): {'0': 0.5, '1': 0.25, '2': 0.25, '3': 0.0, '4': 0.0}
                for i, _ in enumerate(sessions)
            },
            'progression_score_by_session': prog,
            'progression_score_overall': 0.5,
            'progression_trend': 0.25,
            'progression_trend_interpretation': 'advancing — progression scores increase across sessions',
            'stage_exemplars_overall': {str(i): None for i in range(5)},
            'human_coding_agreement': None,
            'narrative_summary': f'{pid} narrative.',
        })
    return reports


# ---------------------------------------------------------------------------
# _bar helper
# ---------------------------------------------------------------------------

class TestBarHelper(unittest.TestCase):
    def test_zero_value(self):
        from analysis.reports._formatting import _bar
        result = _bar(0.0)
        self.assertEqual(result, '░' * 30)

    def test_full_value(self):
        from analysis.reports._formatting import _bar
        result = _bar(1.0)
        self.assertEqual(result, '█' * 30)

    def test_half_value(self):
        from analysis.reports._formatting import _bar
        result = _bar(0.5)
        self.assertEqual(len(result), 30)
        # half filled, half empty
        n_filled = result.count('█')
        self.assertEqual(n_filled, 15)

    def test_custom_width(self):
        from analysis.reports._formatting import _bar
        result = _bar(0.5, width=10)
        self.assertEqual(len(result), 10)


# ---------------------------------------------------------------------------
# _pct helper
# ---------------------------------------------------------------------------

class TestPctHelper(unittest.TestCase):
    def test_zero(self):
        from analysis.reports._formatting import _pct
        self.assertEqual(_pct(0.0), '0.0%')

    def test_one(self):
        from analysis.reports._formatting import _pct
        self.assertEqual(_pct(1.0), '100.0%')

    def test_half(self):
        from analysis.reports._formatting import _pct
        self.assertEqual(_pct(0.5), '50.0%')

    def test_fraction(self):
        from analysis.reports._formatting import _pct
        result = _pct(0.123)
        self.assertIn('12.3%', result)


# ---------------------------------------------------------------------------
# _wrap_quote helper
# ---------------------------------------------------------------------------

class TestWrapQuote(unittest.TestCase):
    def test_empty_string(self):
        from analysis.reports._formatting import _wrap_quote
        result = _wrap_quote('')
        self.assertIn('""', result)

    def test_short_string_no_wrapping(self):
        from analysis.reports._formatting import _wrap_quote
        result = _wrap_quote('hello world')
        self.assertIn('hello world', result)
        self.assertIn('"', result)

    def test_long_string_wraps(self):
        from analysis.reports._formatting import _wrap_quote
        long_text = 'word ' * 40
        result = _wrap_quote(long_text, indent=2, max_width=40)
        lines = result.split('\n')
        self.assertGreater(len(lines), 1)

    def test_all_lines_start_with_quote_or_indent(self):
        from analysis.reports._formatting import _wrap_quote
        long_text = 'word ' * 30
        result = _wrap_quote(long_text, indent=4, max_width=40)
        for line in result.split('\n'):
            self.assertTrue(
                line.startswith('    "') or line.startswith('     '),
                f"Line doesn't start with expected indent/quote: {line!r}"
            )

    def test_preserves_all_words(self):
        from analysis.reports._formatting import _wrap_quote
        text = 'the quick brown fox'
        result = _wrap_quote(text)
        for word in text.split():
            self.assertIn(word, result)


# ---------------------------------------------------------------------------
# _collect_therapist_cue
# ---------------------------------------------------------------------------

class TestCollectTherapistCue(unittest.TestCase):
    def _build_df(self):
        rows = [
            {'session_id': 'c1s1', 'speaker': 'therapist',
             'start_time_ms': 1000, 'end_time_ms': 2000, 'text': 'therapist says A'},
            {'session_id': 'c1s1', 'speaker': 'therapist',
             'start_time_ms': 2100, 'end_time_ms': 3000, 'text': 'therapist says B'},
            {'session_id': 'c1s1', 'speaker': 'participant',
             'start_time_ms': 500, 'end_time_ms': 1000, 'text': 'participant before'},
            {'session_id': 'c1s1', 'speaker': 'participant',
             'start_time_ms': 3200, 'end_time_ms': 4000, 'text': 'participant after'},
        ]
        return pd.DataFrame(rows)

    def test_captures_therapist_in_window(self):
        from analysis.reports._formatting import _collect_therapist_cue
        df = self._build_df()
        result = _collect_therapist_cue(df, 'c1s1', 1000, 3200)
        self.assertIn('therapist says A', result)
        self.assertIn('therapist says B', result)

    def test_empty_window_returns_empty(self):
        from analysis.reports._formatting import _collect_therapist_cue
        df = self._build_df()
        # Window before any therapist segment
        result = _collect_therapist_cue(df, 'c1s1', 0, 500)
        self.assertEqual(result, '')

    def test_zero_bounds_return_empty(self):
        from analysis.reports._formatting import _collect_therapist_cue
        df = self._build_df()
        result = _collect_therapist_cue(df, 'c1s1', 0, 0)
        self.assertEqual(result, '')

    def test_reversed_bounds_return_empty(self):
        from analysis.reports._formatting import _collect_therapist_cue
        df = self._build_df()
        result = _collect_therapist_cue(df, 'c1s1', 3000, 1000)
        self.assertEqual(result, '')

    def test_excludes_participant_rows(self):
        from analysis.reports._formatting import _collect_therapist_cue
        df = self._build_df()
        result = _collect_therapist_cue(df, 'c1s1', 1000, 3200)
        self.assertNotIn('participant', result)

    def test_missing_speaker_column_returns_empty(self):
        from analysis.reports._formatting import _collect_therapist_cue
        df = pd.DataFrame([{'session_id': 'c1s1', 'text': 'x'}])
        result = _collect_therapist_cue(df, 'c1s1', 1000, 2000)
        self.assertEqual(result, '')


# ---------------------------------------------------------------------------
# _format_purer_profile
# ---------------------------------------------------------------------------

class TestFormatPurerProfile(unittest.TestCase):
    def test_empty_profile(self):
        from analysis.reports._formatting import _format_purer_profile
        result = _format_purer_profile({})
        self.assertEqual(result, '')

    def test_single_entry(self):
        from analysis.reports._formatting import _format_purer_profile
        result = _format_purer_profile({3: 2})
        # PURER id 3 → 'E'
        self.assertIn('E', result)
        self.assertIn('2', result)

    def test_multiple_entries_sorted_by_count(self):
        from analysis.reports._formatting import _format_purer_profile
        result = _format_purer_profile({0: 3, 1: 1})
        # P (id=0) appears first (higher count)
        idx_p = result.index('P')
        idx_u = result.index('U')
        self.assertLess(idx_p, idx_u)

    def test_format_has_brackets(self):
        from analysis.reports._formatting import _format_purer_profile
        result = _format_purer_profile({2: 5})
        self.assertTrue(result.startswith('['))
        self.assertTrue(result.endswith(']'))


# ---------------------------------------------------------------------------
# generate_comprehensive_session_report (integration-lite)
# ---------------------------------------------------------------------------

class TestGenerateComprehensiveSessionReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=2, n_participants=2)
        self.fw = _mini_framework()
        self.session_reports = _fake_session_reports(self.df)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_path_string(self):
        from analysis.reports import generate_comprehensive_session_report
        path = generate_comprehensive_session_report(
            self.df, self.fw, self.session_reports, self.tmp
        )
        self.assertIsInstance(path, str)

    def test_file_written(self):
        from analysis.reports import generate_comprehensive_session_report
        path = generate_comprehensive_session_report(
            self.df, self.fw, self.session_reports, self.tmp
        )
        self.assertTrue(os.path.isfile(path), f"Expected file at {path}")

    def test_file_contains_session_ids(self):
        from analysis.reports import generate_comprehensive_session_report
        path = generate_comprehensive_session_report(
            self.df, self.fw, self.session_reports, self.tmp
        )
        with open(path, encoding='utf-8') as f:
            content = f.read()
        for r in self.session_reports:
            self.assertIn(r['session_id'], content)

    def test_empty_session_reports_still_writes(self):
        from analysis.reports import generate_comprehensive_session_report
        path = generate_comprehensive_session_report(
            self.df, self.fw, [], self.tmp
        )
        self.assertTrue(os.path.isfile(path))


# ---------------------------------------------------------------------------
# generate_longitudinal_text_report (integration-lite)
# ---------------------------------------------------------------------------

class TestGenerateLongitudinalTextReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=3, n_participants=2)
        self.fw = _mini_framework()
        self.participant_reports = _fake_participant_reports(self.df)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_path_or_none(self):
        from analysis.reports import generate_longitudinal_text_report
        result = generate_longitudinal_text_report(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        self.assertTrue(result is None or isinstance(result, str))

    def test_file_written_when_path_returned(self):
        from analysis.reports import generate_longitudinal_text_report
        path = generate_longitudinal_text_report(
            self.df, self.participant_reports, self.fw, self.tmp
        )
        if path:
            self.assertTrue(os.path.isfile(path))

    def test_empty_participant_reports_does_not_crash(self):
        from analysis.reports import generate_longitudinal_text_report
        try:
            generate_longitudinal_text_report(self.df, [], self.fw, self.tmp)
        except Exception as e:
            self.fail(f"Raised exception with empty participant_reports: {e}")


# ---------------------------------------------------------------------------
# generate_all_stage_text_reports (integration-lite)
# ---------------------------------------------------------------------------

class TestGenerateAllStageTextReports(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _part_df(n_sessions=2, n_participants=2)
        self.fw = _mini_framework()
        # Minimal stage_reports (from generate_all_theme_reports output shape)
        self.stage_reports = [
            {
                'stage_id': i,
                'stage_name': f'Stage {i}',
                'stage_short_name': f'S{i}',
                'overall_prevalence': 0.2,
                'n_segments_total': len(self.df),
                'n_segments_this_stage': int(len(self.df) / 5),
                'prevalence_by_session_number': {'1': 0.2, '2': 0.2},
                'prevalence_by_participant': {},
                'prevalence_by_cohort': {},
                'longitudinal_trend': 0.0,
                'top_exemplars': [],
                'co_occurring_codes': [],
            }
            for i in range(5)
        ]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_list(self):
        from analysis.reports import generate_all_stage_text_reports
        result = generate_all_stage_text_reports(
            self.df, self.fw, self.stage_reports, self.tmp
        )
        self.assertIsInstance(result, list)

    def test_files_written(self):
        from analysis.reports import generate_all_stage_text_reports
        paths = generate_all_stage_text_reports(
            self.df, self.fw, self.stage_reports, self.tmp
        )
        for path in paths:
            self.assertTrue(os.path.isfile(path), f"Expected file: {path}")

    def test_empty_stage_reports_does_not_crash(self):
        from analysis.reports import generate_all_stage_text_reports
        try:
            generate_all_stage_text_reports(self.df, self.fw, [], self.tmp)
        except Exception as e:
            self.fail(f"Raised exception with empty stage_reports: {e}")


if __name__ == '__main__':
    unittest.main()
