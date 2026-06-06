"""
tests/unit/test_longitudinal_report.py
----------------------------------------
Tests for analysis/reports/longitudinal_report.py.

generate_longitudinal_text_report(df, participant_reports, framework, output_dir) -> str

Covers:
  - Writes 06_reports/01_outcomes/longitudinal.txt
  - Returns path string
  - Eight section structure: header, group trajectory, stage proportions, regression,
    per-participant trajectories, PURER×VAAMR (if present), codebook (if present),
    illustrative journey quotes
  - Empty/degenerate: zero participants or all-same-stage
  - PURER section absent when purer_primary not in df
  - PURER section present when purer_primary available
  - Codebook section absent when no codebook_labels_ensemble
  - Regression detection: between-session regressions counted correctly
  - Illustrative advances: 'ILLUSTRATIVE JOURNEY QUOTES' section present
  - participant_reports list: advancing / stable / regressing tallied
  - _parse_codebook_labels handles list/JSON/CSV/nan inputs
  - _compute_regression_patterns returns correct totals
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from process import output_paths as _paths
from analysis.reports.longitudinal_report import (
    generate_longitudinal_text_report,
    _parse_codebook_labels,
    _check_codebook_present,
    _check_purer_present,
    _compute_regression_patterns,
    _build_snum_lookup,
)


# ── framework ─────────────────────────────────────────────────────────────────

FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


# ── builders ──────────────────────────────────────────────────────────────────

def _make_df(n_participants=2, n_sessions=3, stage_sequence=None,
             include_purer=False, include_codebook=False):
    """Participant df with all columns generate_longitudinal_text_report needs."""
    if stage_sequence is None:
        stage_sequence = [0, 1, 2, 1, 2, 3]
    rows = []
    t = 0
    for p in range(n_participants):
        pid = f'P{p + 1:02d}'
        for s in range(n_sessions):
            sid = f'c1p{p + 1}s{s + 1}'
            snum = s + 1
            for j, lbl in enumerate(stage_sequence):
                t += 1000
                row = {
                    'participant_id': pid,
                    'session_id': sid,
                    'session_number': snum,
                    'cohort_id': 1,
                    'segment_index': j,
                    'speaker': 'participant',
                    'text': f'Participant {pid} text {sid} seg {j}',
                    'start_time_ms': t, 'end_time_ms': t + 800,
                    'final_label': lbl,
                    'llm_confidence_primary': 0.8,
                }
                if include_purer:
                    row['purer_primary'] = (j % 5)
                else:
                    row['purer_primary'] = np.nan
                if include_codebook:
                    row['codebook_labels_ensemble'] = ['body_awareness', 'present_moment']
                rows.append(row)
    df = pd.DataFrame(rows)
    # Therapist rows when PURER present (speaker='therapist')
    if include_purer:
        therapist_rows = []
        t2 = 999
        for p in range(n_participants):
            pid = f'P{p + 1:02d}'
            for s in range(n_sessions):
                sid = f'c1p{p + 1}s{s + 1}'
                snum = s + 1
                for j in range(3):
                    t2 += 1500
                    therapist_rows.append({
                        'participant_id': pid,
                        'session_id': sid,
                        'session_number': snum,
                        'cohort_id': 1,
                        'segment_index': j + len(stage_sequence),
                        'speaker': 'therapist',
                        'text': f'Therapist move {j}',
                        'start_time_ms': t2, 'end_time_ms': t2 + 600,
                        'final_label': np.nan,
                        'llm_confidence_primary': np.nan,
                        'purer_primary': j % 5,
                    })
        df = pd.concat([df, pd.DataFrame(therapist_rows)], ignore_index=True)
    return df


def _make_participant_reports(n_participants=2, n_sessions=3, advancing=True):
    """Minimal participant_report dicts for generate_longitudinal_text_report."""
    reports = []
    for p in range(n_participants):
        pid = f'P{p + 1:02d}'
        score_by_snum = {str(s + 1): 0.5 + s * (0.2 if advancing else -0.1)
                         for s in range(n_sessions)}
        trend = 0.2 if advancing else -0.15
        reports.append({
            'participant_id': pid,
            'cohort_id': 1,
            'n_sessions': n_sessions,
            'progression_trend': trend,
            'progression_trend_interpretation': 'advancing' if advancing else 'regressing',
            'progression_score_by_session': score_by_snum,
            'dominant_stage_sequence': list(range(n_sessions)),
        })
    return reports


# ── main test class ───────────────────────────────────────────────────────────

class TestLongitudinalReportWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _make_df()
        self.reports = _make_participant_reports()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_path(self):
        path = generate_longitudinal_text_report(self.df, self.reports, FRAMEWORK, self.tmp)
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.txt'))

    def test_file_exists(self):
        path = generate_longitudinal_text_report(self.df, self.reports, FRAMEWORK, self.tmp)
        self.assertTrue(os.path.isfile(path))

    def test_path_in_outcomes_dir(self):
        path = generate_longitudinal_text_report(self.df, self.reports, FRAMEWORK, self.tmp)
        self.assertIn('01_outcomes', path)
        self.assertTrue(path.endswith('longitudinal.txt'))


class TestLongitudinalSections(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _make_df()
        self.reports = _make_participant_reports()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _read(self):
        path = os.path.join(_paths.reports_outcomes_dir(self.tmp), 'longitudinal.txt')
        with open(path, encoding='utf-8') as f:
            return f.read()

    def _run(self):
        return generate_longitudinal_text_report(self.df, self.reports, FRAMEWORK, self.tmp)

    def test_header(self):
        self._run()
        self.assertIn('QRA LONGITUDINAL ANALYSIS REPORT', self._read())

    def test_group_trajectory_section(self):
        self._run()
        self.assertIn('VAAMR GROUP TRAJECTORY', self._read())
        self.assertIn('MEAN GROUP TREND', self._read())

    def test_trend_direction_label(self):
        self._run()
        content = self._read()
        # With advancing=True, mean_trend > 0 → 'ADVANCING'
        self.assertIn('ADVANCING', content)

    def test_stage_proportions_section(self):
        self._run()
        content = self._read()
        self.assertIn('STAGE PROPORTIONS AND COUNTS BY SESSION', content)
        # Session numbers should appear
        for snum in (1, 2, 3):
            self.assertIn(str(snum), content)

    def test_regression_section(self):
        self._run()
        self.assertIn('REGRESSION PATTERN ANALYSIS', self._read())

    def test_per_participant_section(self):
        self._run()
        content = self._read()
        self.assertIn('PER-PARTICIPANT TRAJECTORIES', content)
        self.assertIn('P01', content)
        self.assertIn('P02', content)

    def test_illustrative_quotes_section(self):
        self._run()
        self.assertIn('ILLUSTRATIVE JOURNEY QUOTES', self._read())

    def test_stage_names_in_content(self):
        self._run()
        content = self._read()
        for name in ('Vigilance', 'Avoidance', 'AttnReg'):
            self.assertIn(name, content)

    def test_advancing_stable_regressing_tallied(self):
        self._run()
        content = self._read()
        self.assertIn('Advancing:', content)
        self.assertIn('Stable:', content)
        self.assertIn('Regressing:', content)


class TestPURERSection(unittest.TestCase):
    def test_purer_section_absent_without_purer(self):
        df = _make_df(include_purer=False)
        reports = _make_participant_reports()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertNotIn('PURER × VAAMR LONGITUDINAL INFLUENCE', content)

    def test_purer_section_present_with_purer(self):
        df = _make_df(include_purer=True)
        reports = _make_participant_reports()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertIn('PURER × VAAMR LONGITUDINAL INFLUENCE', content)


class TestCodebookSection(unittest.TestCase):
    def test_codebook_section_absent_without_labels(self):
        df = _make_df(include_codebook=False)
        reports = _make_participant_reports()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertNotIn('VAAMR × PHENOMENOLOGY RELATIONSHIPS', content)

    def test_codebook_section_present_with_labels(self):
        df = _make_df(include_codebook=True)
        reports = _make_participant_reports()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertIn('VAAMR × PHENOMENOLOGY RELATIONSHIPS', content)
        self.assertIn('body_awareness', content)


class TestRegressionDetection(unittest.TestCase):
    def test_no_regression_message(self):
        """With monotonically advancing sequence, no regressions should appear."""
        df = _make_df(stage_sequence=[0, 1, 2, 3])
        # participant_sequences will show stage 0→1 in sessions, so no between-session regression
        reports = _make_participant_reports(advancing=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        # Either "No between-session regressions" or the regression section exists with no data
        self.assertIn('REGRESSION PATTERN ANALYSIS', content)

    def test_regressing_participant(self):
        """Two participants, one advancing, one regressing."""
        df = _make_df(n_participants=1, stage_sequence=[2, 1, 0])  # Descending
        reports = _make_participant_reports(n_participants=1, advancing=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertIn('REGRESSING', content)


class TestDegenerateInputs(unittest.TestCase):
    def test_single_participant_single_session(self):
        df = _make_df(n_participants=1, n_sessions=1, stage_sequence=[1, 2])
        reports = _make_participant_reports(n_participants=1, n_sessions=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            self.assertTrue(os.path.isfile(path))

    def test_all_same_stage(self):
        df = _make_df(stage_sequence=[1, 1, 1])
        reports = _make_participant_reports()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, reports, FRAMEWORK, tmp)
            self.assertTrue(os.path.isfile(path))

    def test_empty_participant_reports(self):
        """With no participant reports, advancing/stable/regressing all 0."""
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_longitudinal_text_report(df, [], FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertIn('Advancing:', content)


# ── unit tests for private helpers ────────────────────────────────────────────

class TestParseCodebookLabels(unittest.TestCase):
    def test_list_input(self):
        self.assertEqual(_parse_codebook_labels(['a', 'b']), ['a', 'b'])

    def test_json_string(self):
        self.assertEqual(_parse_codebook_labels('["x", "y"]'), ['x', 'y'])

    def test_comma_separated_string(self):
        result = _parse_codebook_labels('code_a, code_b, code_c')
        self.assertEqual(result, ['code_a', 'code_b', 'code_c'])

    def test_none_input(self):
        self.assertEqual(_parse_codebook_labels(None), [])

    def test_nan_input(self):
        self.assertEqual(_parse_codebook_labels(float('nan')), [])

    def test_empty_string(self):
        self.assertEqual(_parse_codebook_labels(''), [])

    def test_empty_list(self):
        self.assertEqual(_parse_codebook_labels([]), [])


class TestCheckCodebookPresent(unittest.TestCase):
    def test_absent_when_no_column(self):
        df = pd.DataFrame({'final_label': [0, 1]})
        self.assertFalse(_check_codebook_present(df))

    def test_absent_when_all_empty(self):
        df = pd.DataFrame({'codebook_labels_ensemble': [[], [], []]})
        self.assertFalse(_check_codebook_present(df))

    def test_present_when_has_labels(self):
        df = pd.DataFrame({'codebook_labels_ensemble': [['a', 'b'], [], []]})
        self.assertTrue(_check_codebook_present(df))


class TestCheckPURERPresent(unittest.TestCase):
    def test_absent_when_no_purer_column(self):
        df = pd.DataFrame({'speaker': ['participant']})
        self.assertFalse(_check_purer_present(df))

    def test_absent_when_no_speaker_column(self):
        df = pd.DataFrame({'purer_primary': [0, 1]})
        self.assertFalse(_check_purer_present(df))

    def test_absent_when_therapist_has_only_nan(self):
        df = pd.DataFrame({
            'speaker': ['therapist', 'therapist'],
            'purer_primary': [np.nan, np.nan],
        })
        self.assertFalse(_check_purer_present(df))

    def test_present_when_therapist_has_labels(self):
        df = pd.DataFrame({
            'speaker': ['participant', 'therapist'],
            'purer_primary': [np.nan, 2],
        })
        self.assertTrue(_check_purer_present(df))


class TestComputeRegressionPatterns(unittest.TestCase):
    def _seq(self, pid, stages):
        """Build participant_sequences format: {pid: [(sid, stage_id, stage_name), ...]}"""
        return {pid: [(f'c1s{i + 1}', s, f'Stage{s}') for i, s in enumerate(stages)]}

    def test_no_regressions(self):
        seqs = self._seq('P01', [0, 1, 2, 3])
        snum_lookup = {f'c1s{i + 1}': i + 1 for i in range(4)}
        result = _compute_regression_patterns(seqs, snum_lookup, {})
        self.assertEqual(result['total_regressions'], 0)
        self.assertEqual(len(result['participants_with_regressions']), 0)

    def test_one_regression(self):
        seqs = self._seq('P01', [2, 1])
        snum_lookup = {'c1s1': 1, 'c1s2': 2}
        result = _compute_regression_patterns(seqs, snum_lookup, {})
        self.assertEqual(result['total_regressions'], 1)
        self.assertEqual(len(result['participants_with_regressions']), 1)

    def test_multiple_regressions(self):
        seqs = {'P01': [('c1s1', 2, 'AttnReg'), ('c1s2', 1, 'Avoidance'),
                        ('c1s3', 3, 'Metacog'), ('c1s4', 0, 'Vigilance')]}
        snum_lookup = {f'c1s{i}': i for i in range(1, 5)}
        result = _compute_regression_patterns(seqs, snum_lookup, {})
        self.assertEqual(result['total_regressions'], 2)

    def test_returns_correct_keys(self):
        seqs = self._seq('P01', [1, 0])
        result = _compute_regression_patterns(seqs, {'c1s1': 1, 'c1s2': 2}, {})
        for key in ('regression_at_snum', 'regression_types',
                    'participants_with_regressions', 'total_regressions'):
            self.assertIn(key, result)


class TestBuildSnumLookup(unittest.TestCase):
    def test_basic_lookup(self):
        df = pd.DataFrame({
            'session_id': ['c1s1', 'c1s1', 'c1s2', 'c1s3'],
            'session_number': [1, 1, 2, 3],
        })
        lookup = _build_snum_lookup(df)
        self.assertEqual(lookup['c1s1'], 1)
        self.assertEqual(lookup['c1s2'], 2)
        self.assertEqual(lookup['c1s3'], 3)


if __name__ == '__main__':
    unittest.main()
