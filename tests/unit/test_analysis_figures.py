"""
tests/unit/test_analysis_figures.py
-------------------------------------
Unit tests for analysis/figures.py and analysis/purer_figures.py.

Uses the Agg backend (headless) and make_master_df for small synthetic data.
All pure-matplotlib functions are quick and not marked @slow_test.
Tests verify that PNG files are created (non-empty) at the expected paths.
"""

import matplotlib
matplotlib.use('Agg')  # MUST be before any other matplotlib/figures import

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import make_master_df

# analysis/figures.py already sets Agg internally; import after our set above
from analysis.figures import (
    plot_dominant_stage_heatmap,
    plot_group_longitudinal_trajectory,
    plot_participant_longitudinal_trajectory,
    plot_all_participant_trajectories,
    plot_transition_heatmap,
    plot_cross_session_transition_heatmap,
    plot_stage_prevalence_by_session,
    plot_all_stage_prevalence_charts,
    plot_session_stage_timeline,
    generate_all_session_stage_timelines,
    plot_program_longitudinal_progression,
    plot_session_mixture_timeline,
    plot_participant_progression_trajectory,
    plot_superposition_entropy_by_session,
    plot_stage_cooccurrence_matrix,
    generate_superposition_figures,
)
from analysis.purer_figures import (
    plot_purer_lift_heatmap,
    plot_purer_transition_profiles,
    generate_purer_figures,
)
from process import output_paths as _paths


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_FRAMEWORK = {
    0: {'id': 0, 'key': 'vigilance', 'name': 'Vigilance', 'short_name': 'VIG'},
    1: {'id': 1, 'key': 'avoidance', 'name': 'Avoidance', 'short_name': 'AVD'},
    2: {'id': 2, 'key': 'attention_regulation', 'name': 'Attention Regulation', 'short_name': 'ATT'},
    3: {'id': 3, 'key': 'metacognition', 'name': 'Metacognition', 'short_name': 'MET'},
    4: {'id': 4, 'key': 'reappraisal', 'name': 'Reappraisal', 'short_name': 'REA'},
}


def _participant_df(n_sessions=3, n_participants=2):
    """Participant-only df suitable for analysis/figures.py functions."""
    df = make_master_df(n_sessions=n_sessions, n_participants=n_participants)
    df = df[df['speaker'] == 'participant'].copy().reset_index(drop=True)
    df['final_label'] = df['final_label'].fillna(0).astype(int)
    df['primary_stage'] = df['primary_stage'].fillna(0).astype(int)
    if 'label_confidence_tier' not in df.columns:
        df['label_confidence_tier'] = df.get('confidence_tier', 'high')
    if 'llm_run_consistency' not in df.columns:
        df['llm_run_consistency'] = 3
    return df


def _png_ok(path):
    """Return True iff path is a non-empty file."""
    return os.path.isfile(path) and os.path.getsize(path) > 0


class TestDominantStageHeatmap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png(self):
        path = plot_dominant_stage_heatmap(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(_png_ok(path), f"PNG not created: {path}")

    def test_path_ends_with_png(self):
        path = plot_dominant_stage_heatmap(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(path.endswith('.png'))

    def test_saved_in_figures_dir(self):
        path = plot_dominant_stage_heatmap(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(path.startswith(_paths.figures_dir(self.tmp)))


class TestGroupLongitudinalTrajectory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png(self):
        path = plot_group_longitudinal_trajectory(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(_png_ok(path))

    def test_empty_df_returns_empty_string(self):
        result = plot_group_longitudinal_trajectory(pd.DataFrame(), _FRAMEWORK, self.tmp)
        self.assertEqual(result, '')


class TestParticipantLongitudinalTrajectory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()
        self.pid = self.df['participant_id'].iloc[0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png_for_known_participant(self):
        path = plot_participant_longitudinal_trajectory(
            self.df, self.pid, _FRAMEWORK, self.tmp
        )
        self.assertTrue(_png_ok(path))

    def test_unknown_participant_returns_empty(self):
        result = plot_participant_longitudinal_trajectory(
            self.df, 'UNKNOWN_XYZ', _FRAMEWORK, self.tmp
        )
        self.assertEqual(result, '')

    def test_all_participants_generates_list(self):
        paths = plot_all_participant_trajectories(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(paths, list)
        n_pids = len(self.df['participant_id'].unique())
        self.assertEqual(len(paths), n_pids)
        for p in paths:
            self.assertTrue(_png_ok(p))


class TestTransitionHeatmap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png(self):
        path = plot_transition_heatmap(self.df, _FRAMEWORK, self.tmp)
        # May return '' if no transitions (single segment per session)
        if path:
            self.assertTrue(_png_ok(path))


class TestCrossSessionTransitionHeatmap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df(n_sessions=3)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png_or_empty(self):
        path = plot_cross_session_transition_heatmap(self.df, _FRAMEWORK, self.tmp)
        if path:
            self.assertTrue(_png_ok(path))


class TestStagePrevalenceBySession(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png_for_each_stage(self):
        for stage_id in range(5):
            path = plot_stage_prevalence_by_session(self.df, stage_id, _FRAMEWORK, self.tmp)
            self.assertTrue(_png_ok(path), f"PNG missing for stage {stage_id}: {path}")

    def test_all_stage_prevalence_charts_list(self):
        paths = plot_all_stage_prevalence_charts(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertTrue(_png_ok(p))


class TestSessionStageTimeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()
        self.sid = self.df['session_id'].iloc[0]

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png_for_known_session(self):
        path = plot_session_stage_timeline(self.df, self.sid, _FRAMEWORK, self.tmp)
        self.assertTrue(_png_ok(path), f"PNG not created for session {self.sid}: {path}")

    def test_unknown_session_returns_empty(self):
        result = plot_session_stage_timeline(self.df, 'NOSUCHSESSION', _FRAMEWORK, self.tmp)
        self.assertEqual(result, '')

    def test_generate_all_session_timelines(self):
        paths = generate_all_session_stage_timelines(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(paths, list)
        n_sessions = len(self.df['session_id'].unique())
        self.assertEqual(len(paths), n_sessions)
        for p in paths:
            self.assertTrue(_png_ok(p))


class TestProgramLongitudinalProgression(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _participant_df()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png(self):
        path = plot_program_longitudinal_progression(self.df, _FRAMEWORK, self.tmp)
        self.assertTrue(_png_ok(path))

    def test_empty_df_returns_empty_string(self):
        result = plot_program_longitudinal_progression(
            pd.DataFrame(), _FRAMEWORK, self.tmp
        )
        self.assertEqual(result, '')


class TestSuperpositionFigures(unittest.TestCase):
    """Figures that require 'mixture' and 'progression_coord' columns."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        df = _participant_df()
        n = len(df)
        df['mixture'] = [[0.2, 0.2, 0.2, 0.2, 0.2]] * n
        df['progression_coord'] = 2.0
        df['mixture_entropy'] = 0.8
        df['max_stage'] = 0
        df['second_stage'] = 1
        df['n_active_stages'] = 5
        df['is_liminal'] = True
        df['mixture_source'] = 'secondary'
        self.df = df

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_session_mixture_timeline(self):
        sid = self.df['session_id'].iloc[0]
        path = plot_session_mixture_timeline(self.df, sid, _FRAMEWORK, self.tmp)
        self.assertIsNotNone(path)
        self.assertTrue(_png_ok(path))

    def test_session_mixture_timeline_no_mixture_col(self):
        df_no = self.df.drop(columns=['mixture'])
        result = plot_session_mixture_timeline(df_no, 'any', _FRAMEWORK, self.tmp)
        self.assertIsNone(result)

    def test_participant_progression_trajectory(self):
        pid = self.df['participant_id'].iloc[0]
        path = plot_participant_progression_trajectory(self.df, pid, _FRAMEWORK, self.tmp)
        self.assertIsNotNone(path)
        self.assertTrue(_png_ok(path))

    def test_participant_progression_unknown_pid(self):
        result = plot_participant_progression_trajectory(
            self.df, 'UNKNOWN', _FRAMEWORK, self.tmp
        )
        self.assertIsNone(result)

    def test_superposition_entropy_by_session(self):
        path = plot_superposition_entropy_by_session(self.df, _FRAMEWORK, self.tmp)
        self.assertIsNotNone(path)
        self.assertTrue(_png_ok(path))

    def test_superposition_entropy_no_col(self):
        df_no = self.df.drop(columns=['mixture_entropy'])
        result = plot_superposition_entropy_by_session(df_no, _FRAMEWORK, self.tmp)
        self.assertIsNone(result)

    def test_stage_cooccurrence_matrix_figure(self):
        path = plot_stage_cooccurrence_matrix(self.df, _FRAMEWORK, self.tmp)
        self.assertIsNotNone(path)
        self.assertTrue(_png_ok(path))

    def test_generate_superposition_figures_returns_list(self):
        paths = generate_superposition_figures(self.df, _FRAMEWORK, self.tmp)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertTrue(_png_ok(p))


# ---------------------------------------------------------------------------
# analysis/purer_figures.py tests
# ---------------------------------------------------------------------------

def _minimal_lift_df():
    """Minimal lift_matrix DataFrame as returned by purer_analysis."""
    rows = []
    for pid, pshort, pname in [
        (0, 'P', 'Phenomenology'),
        (1, 'U', 'Utilization'),
        (2, 'R', 'Reframing'),
        (3, 'E', 'Education'),
        (4, 'R2', 'Reinforcement'),
    ]:
        row = {'purer_short': pshort, 'purer_construct': pname, 'n_blocks': 5}
        for st in range(5):
            row[f'lift_to_{st}'] = 0.8 + 0.1 * st  # synthetic lift values
        rows.append(row)
    return pd.DataFrame(rows)


def _minimal_transition_profiles():
    """Minimal transition_profiles DataFrame."""
    rows = []
    for fr, to in [(0, 1), (1, 2), (2, 1), (2, 3), (0, 2)]:
        for pid in range(5):
            rows.append({
                'from_stage': fr,
                'to_stage': to,
                'dominant_purer': pid,
                'fraction_of_mediated': 0.2,
                'mediated_total': 10,
            })
    return pd.DataFrame(rows)


class TestPurerLiftHeatmap(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.influence = {
            'lift_matrix': _minimal_lift_df(),
            'n_mediated': 42,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png(self):
        path = plot_purer_lift_heatmap(self.influence, _FRAMEWORK, self.tmp)
        self.assertTrue(_png_ok(path), f"PNG not created: {path}")

    def test_path_in_figures_dir(self):
        path = plot_purer_lift_heatmap(self.influence, _FRAMEWORK, self.tmp)
        self.assertTrue(path.startswith(_paths.figures_dir(self.tmp)))

    def test_empty_lift_returns_empty_string(self):
        result = plot_purer_lift_heatmap({'lift_matrix': pd.DataFrame()}, _FRAMEWORK, self.tmp)
        self.assertEqual(result, '')

    def test_none_lift_returns_empty_string(self):
        result = plot_purer_lift_heatmap({'lift_matrix': None}, _FRAMEWORK, self.tmp)
        self.assertEqual(result, '')


class TestPurerTransitionProfiles(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.influence = {
            'transition_profiles': _minimal_transition_profiles(),
            'n_mediated': 10,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_creates_png(self):
        path = plot_purer_transition_profiles(self.influence, _FRAMEWORK, self.tmp)
        self.assertTrue(_png_ok(path), f"PNG not created: {path}")

    def test_empty_profiles_returns_empty_string(self):
        result = plot_purer_transition_profiles(
            {'transition_profiles': pd.DataFrame()}, _FRAMEWORK, self.tmp
        )
        self.assertEqual(result, '')

    def test_lateral_only_returns_empty_string(self):
        """Only self-transitions (from == to) -> non_lateral is empty -> ''."""
        df = pd.DataFrame([{
            'from_stage': 1, 'to_stage': 1, 'dominant_purer': 0,
            'fraction_of_mediated': 1.0, 'mediated_total': 5,
        }])
        result = plot_purer_transition_profiles({'transition_profiles': df}, _FRAMEWORK, self.tmp)
        self.assertEqual(result, '')


class TestGeneratePurerFigures(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.influence = {
            'lift_matrix': _minimal_lift_df(),
            'transition_profiles': _minimal_transition_profiles(),
            'n_mediated': 42,
        }

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_list_of_png_paths(self):
        paths = generate_purer_figures(self.influence, _FRAMEWORK, self.tmp)
        self.assertIsInstance(paths, list)
        self.assertGreater(len(paths), 0)
        for p in paths:
            self.assertTrue(_png_ok(p))

    def test_empty_influence_no_crash(self):
        result = generate_purer_figures({}, _FRAMEWORK, self.tmp)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
