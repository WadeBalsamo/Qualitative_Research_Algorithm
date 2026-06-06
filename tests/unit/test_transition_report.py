"""
tests/unit/test_transition_report.py
--------------------------------------
Tests for analysis/reports/transition_report.py.

generate_transition_explanation(df, framework, output_dir,
    therapist_cue_config=None, llm_client=None, df_all=None) -> str

generate_therapist_cues_report(df, framework, output_dir,
    therapist_cue_config, llm_client, df_all=None) -> str

Covers:
  - File is written to 02_mechanism/transitions.txt
  - File is written to 02_mechanism/cue_response.txt
  - Within-session and between-session section headers present
  - All transitions listed (including forward/backward)
  - Per-participant trajectories section present
  - Empty df (all participants have <2 segments) → still writes file
  - PURER × transition section appears when purer_primary present
  - Exemplar blocks appear for non-self transitions
  - _find_transition_examples finds correct pairs
  - _find_transition_examples_by_cohort_session returns one-per-(cohort,session)
"""

import os
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from process import output_paths as _paths
from analysis.reports.transition_report import (
    generate_transition_explanation,
    generate_therapist_cues_report,
    _find_transition_examples,
    _find_transition_examples_by_cohort_session,
    _build_purer_transition_profiles,
)


# ── framework ─────────────────────────────────────────────────────────────────

FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


# ── df builder ────────────────────────────────────────────────────────────────

def _make_df(n_participants=2, n_sessions=2, labels_sequence=None):
    """Participant-only df with final_label, segment_index, session_number."""
    if labels_sequence is None:
        labels_sequence = [0, 1, 2, 1, 0, 2]
    rows = []
    t = 0
    for p in range(n_participants):
        pid = f'P{p + 1:02d}'
        for s in range(n_sessions):
            sid = f'c1p{p + 1}s{s + 1}'
            snum = s + 1
            for j, lbl in enumerate(labels_sequence):
                t += 1000
                rows.append({
                    'participant_id': pid,
                    'session_id': sid,
                    'session_number': snum,
                    'cohort_id': 1,
                    'segment_index': j,
                    'speaker': 'participant',
                    'text': f'Participant {pid} session {sid} segment {j} text here',
                    'start_time_ms': t,
                    'end_time_ms': t + 800,
                    'final_label': lbl,
                    'llm_confidence_primary': 0.8,
                })
    return pd.DataFrame(rows)


def _make_df_with_therapist(n_participants=2, n_sessions=2):
    """Interleaved participant+therapist df for cue report tests."""
    rows = []
    t = 0
    labels = [0, 1, 2, 1]
    purer = [0, 1, 2, 3]
    for p in range(n_participants):
        pid = f'P{p + 1:02d}'
        for s in range(n_sessions):
            sid = f'c1p{p + 1}s{s + 1}'
            snum = s + 1
            seg_idx = 0
            for j in range(len(labels)):
                # participant segment
                t += 1000
                rows.append({
                    'participant_id': pid,
                    'session_id': sid,
                    'session_number': snum,
                    'cohort_id': 1,
                    'segment_index': seg_idx,
                    'speaker': 'participant',
                    'text': f'Participant text session {sid} idx {j}',
                    'start_time_ms': t, 'end_time_ms': t + 800,
                    'final_label': labels[j],
                    'llm_confidence_primary': 0.8,
                    'purer_primary': np.nan,
                    'purer_secondary': np.nan,
                })
                seg_idx += 1
                t += 900
                # therapist segment between participant turns
                rows.append({
                    'participant_id': pid,
                    'session_id': sid,
                    'session_number': snum,
                    'cohort_id': 1,
                    'segment_index': seg_idx,
                    'speaker': 'therapist',
                    'text': f'Therapist cue text for move {purer[j]}',
                    'start_time_ms': t, 'end_time_ms': t + 500,
                    'final_label': np.nan,
                    'llm_confidence_primary': np.nan,
                    'purer_primary': purer[j],
                    'purer_secondary': (purer[j] + 1) % 5,
                })
                seg_idx += 1
    return pd.DataFrame(rows)


# ── TherapistCueConfig stub ───────────────────────────────────────────────────

def _cue_cfg(enabled=False, max_length=200):
    return SimpleNamespace(
        enabled=enabled,
        max_length_per_cue=max_length,
        max_length_of_average_cue_responses=max_length,
    )


# ── tests: generate_transition_explanation ────────────────────────────────────

class TestTransitionExplanationWrites(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _make_df()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_path_string(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        self.assertIsInstance(path, str)
        self.assertTrue(path.endswith('.txt'))

    def test_file_exists(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        self.assertTrue(os.path.isfile(path))

    def test_path_in_mechanism_dir(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        self.assertIn('02_mechanism', path)
        self.assertTrue(path.endswith('transitions.txt'))

    def test_within_session_header(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('WITHIN-SESSION TRANSITIONS', content)

    def test_between_session_header(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('BETWEEN-SESSION TRANSITIONS', content)

    def test_generated_date_in_header(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('Generated:', content)

    def test_per_participant_trajectories_section(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('Individual participant trajectories', content)

    def test_stage_names_rendered(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        for name in ('Vigilance', 'Avoidance', 'AttnReg'):
            self.assertIn(name, content)

    def test_transition_directions_labeled(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        # labels_sequence [0,1,2,1,0,2] has forward and backward transitions
        self.assertIn('forward', content)
        self.assertIn('backward', content)

    def test_participant_ids_in_trajectory_section(self):
        path = generate_transition_explanation(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('P01', content)
        self.assertIn('P02', content)


class TestTransitionExplanationWithPURER(unittest.TestCase):
    """When df_all has purer_primary, the PURER × transition section appears."""

    def test_purer_section_present(self):
        df_all = _make_df_with_therapist()
        df_participant = df_all[df_all['speaker'] == 'participant'].copy()
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_transition_explanation(
                df_participant, FRAMEWORK, tmp, df_all=df_all
            )
            with open(path, encoding='utf-8') as f:
                content = f.read()
        self.assertIn('PURER', content)


class TestTransitionExplanationDegenerate(unittest.TestCase):
    """With no transitions (all segments same label or only 1 segment per session), still writes."""

    def test_all_same_label(self):
        df = _make_df(labels_sequence=[2, 2, 2])
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_transition_explanation(df, FRAMEWORK, tmp)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('WITHIN-SESSION TRANSITIONS', content)

    def test_single_segment_per_session(self):
        df = _make_df(labels_sequence=[1])
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_transition_explanation(df, FRAMEWORK, tmp)
            self.assertTrue(os.path.isfile(path))


# ── tests: generate_therapist_cues_report ─────────────────────────────────────

class TestTherapistCuesReport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df_all = _make_df_with_therapist()
        self.df = self.df_all[self.df_all['speaker'] == 'participant'].copy()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_writes_cue_response_txt(self):
        path = generate_therapist_cues_report(
            self.df, FRAMEWORK, self.tmp, _cue_cfg(), None, self.df_all
        )
        self.assertTrue(os.path.isfile(path))
        self.assertTrue(path.endswith('cue_response.txt'))

    def test_header_present(self):
        path = generate_therapist_cues_report(
            self.df, FRAMEWORK, self.tmp, _cue_cfg(), None, self.df_all
        )
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('THERAPIST CUE ANALYSIS', content)
        self.assertIn('Generated:', content)

    def test_total_transitions_reported(self):
        path = generate_therapist_cues_report(
            self.df, FRAMEWORK, self.tmp, _cue_cfg(), None, self.df_all
        )
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('Total within-session transitions', content)

    def test_forward_backward_counts(self):
        path = generate_therapist_cues_report(
            self.df, FRAMEWORK, self.tmp, _cue_cfg(), None, self.df_all
        )
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('Forward:', content)
        self.assertIn('Backward:', content)

    def test_transition_blocks_rendered(self):
        path = generate_therapist_cues_report(
            self.df, FRAMEWORK, self.tmp, _cue_cfg(), None, self.df_all
        )
        with open(path, encoding='utf-8') as f:
            content = f.read()
        # At least one transition block header: "── StageA → StageB"
        self.assertIn('──', content)


# ── tests: private helpers ────────────────────────────────────────────────────

class TestFindTransitionExamples(unittest.TestCase):
    def setUp(self):
        self.df = _make_df(n_participants=1, n_sessions=1, labels_sequence=[0, 1, 2, 0])

    def test_finds_forward_pair(self):
        examples = _find_transition_examples(self.df, from_stage=0, to_stage=1,
                                             framework=FRAMEWORK)
        self.assertGreater(len(examples), 0)
        ex = examples[0]
        self.assertIn('from_text', ex)
        self.assertIn('to_text', ex)
        self.assertIn('participant_id', ex)

    def test_no_pair_for_impossible_transition(self):
        # [0,1,2,0] has no 3→4
        examples = _find_transition_examples(self.df, from_stage=3, to_stage=4,
                                             framework=FRAMEWORK)
        self.assertEqual(len(examples), 0)

    def test_respects_n_limit(self):
        # Large df: fill with 0→1 transitions
        rows = []
        for i in range(20):
            rows.append({'participant_id': 'P01', 'session_id': 'c1s1',
                         'segment_index': i * 2, 'final_label': 0,
                         'text': f'from {i}', 'session_number': 1})
            rows.append({'participant_id': 'P01', 'session_id': 'c1s1',
                         'segment_index': i * 2 + 1, 'final_label': 1,
                         'text': f'to {i}', 'session_number': 1})
        df = pd.DataFrame(rows)
        examples = _find_transition_examples(df, from_stage=0, to_stage=1,
                                             framework=FRAMEWORK, n=3)
        self.assertLessEqual(len(examples), 3)


class TestFindTransitionExamplesByCohortSession(unittest.TestCase):
    def setUp(self):
        self.df = _make_df(n_participants=2, n_sessions=1,
                           labels_sequence=[0, 1, 0, 1])

    def test_returns_one_per_cohort_session(self):
        examples = _find_transition_examples_by_cohort_session(
            self.df, from_stage=0, to_stage=1
        )
        # 2 participants, 1 session each → at most 2 examples, one per (cohort, session)
        self.assertLessEqual(len(examples), 2)
        keys = [(e['cohort_id'], e['session_id']) for e in examples]
        self.assertEqual(len(keys), len(set(keys)))

    def test_examples_have_required_fields(self):
        examples = _find_transition_examples_by_cohort_session(
            self.df, from_stage=0, to_stage=1
        )
        for ex in examples:
            for key in ('participant_id', 'session_id', 'from_text', 'to_text',
                        'from_conf', 'to_conf', 'from_seg_idx', 'to_seg_idx'):
                self.assertIn(key, ex)


class TestBuildPURERTransitionProfiles(unittest.TestCase):
    def test_empty_when_no_purer_column(self):
        df = _make_df()
        df_all = _make_df()
        result = _build_purer_transition_profiles(df, df_all)
        self.assertEqual(result, {})

    def test_returns_dict_with_purer_present(self):
        df_all = _make_df_with_therapist()
        df = df_all[df_all['speaker'] == 'participant'].copy()
        result = _build_purer_transition_profiles(df, df_all)
        # Any result is valid; just must be a dict
        self.assertIsInstance(result, dict)


if __name__ == '__main__':
    unittest.main()
