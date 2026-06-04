"""
tests/test_mechanism.py
-----------------------
Unit tests for analysis/mechanism.py and the soft cross-validation lift.

Synthetic participant/therapist segments + injected GNN positions exercise the
continuous Δprogression aggregation, liminality binning, avoidance-barrier
ranking, trajectory typology, and the expected-count soft-lift math. No model
backends are touched.
"""

import json
import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analysis.superposition import attach_superposition
from analysis.mechanism import run_mechanism_analysis, _trajectory_typology, _slope
from process.config import SuperpositionConfig
from process.cross_validation import compute_soft_theme_codebook_cooccurrence
from process import output_paths as _paths


_FRAMEWORK = {
    0: {'key': 'vigilance', 'short_name': 'Vigilance'},
    1: {'key': 'avoidance', 'short_name': 'Avoidance'},
    2: {'key': 'attn_reg', 'short_name': 'AttnReg'},
    3: {'key': 'metacog', 'short_name': 'Metacog'},
    4: {'key': 'reappraisal', 'short_name': 'Reappraisal'},
}


def _synthetic_all_df():
    """Participant turns alternating with therapist cues; Avoidance→AttnReg moves."""
    rows = []
    for s in range(1, 4):
        sess = f'c1s{s}'
        t = 0
        # participant(avoidance) -> therapist(reframing) -> participant(attnreg) -> ...
        seq = [('participant', 1), ('therapist', 2), ('participant', 2),
               ('therapist', 2), ('participant', 3)]
        for j, (spk, stage) in enumerate(seq):
            t += 1000
            is_p = spk == 'participant'
            rows.append(dict(
                segment_id=f'{sess}_{j}', participant_id='P1', session_id=sess,
                session_number=s, segment_index=j, speaker=spk, word_count=40,
                text=f'{spk} text {sess} {j}',
                start_time_ms=t, end_time_ms=t + 800,
                final_label=(stage if is_p else np.nan),
                primary_stage=(stage if is_p else np.nan),
                secondary_stage=np.nan,
                llm_confidence_primary=0.8, llm_confidence_secondary=np.nan,
                rater_votes=(json.dumps([{'stage': stage, 'confidence': 0.9}]) if is_p else None),
                codebook_labels_ensemble=(['somatic_x'] if is_p else []),
                purer_primary=(2 if not is_p else np.nan),   # Reframing cue
                microskill_labels_ensemble=(['reframe'] if not is_p else []),
            ))
    return pd.DataFrame(rows)


class TestHelpers(unittest.TestCase):
    def test_slope_positive(self):
        self.assertGreater(_slope([1.0, 1.5, 2.0, 2.5]), 0)

    def test_slope_single_point(self):
        self.assertEqual(_slope([1.0]), 0.0)


class TestSoftLift(unittest.TestCase):
    def test_expected_count_math(self):
        # Two segments: one pure stage1, one 50/50 stage1/stage2, both carry code 'c'.
        df = pd.DataFrame([
            {'mixture': [0, 1, 0, 0, 0], 'codebook_labels_ensemble': ['c']},
            {'mixture': [0, 0.5, 0.5, 0, 0], 'codebook_labels_ensemble': ['c']},
        ])
        soft = compute_soft_theme_codebook_cooccurrence(df, _FRAMEWORK)
        # Stage 1 expected count for 'c' = 1.0 + 0.5 = 1.5; mass = 1.5
        s1 = soft['avoidance']['codes']['c']
        self.assertAlmostEqual(s1['expected_count'], 1.5, places=3)
        # Stage 2 expected count = 0.5; mass = 0.5 → soft_rate 1.0
        s2 = soft['attn_reg']['codes']['c']
        self.assertAlmostEqual(s2['expected_count'], 0.5, places=3)
        self.assertAlmostEqual(s2['soft_rate'], 1.0, places=3)


class TestTrajectoryTypology(unittest.TestCase):
    def test_climber_detected(self):
        rows = []
        for s, coord in enumerate([1.0, 2.0, 3.0], start=1):
            rows.append(dict(participant_id='P1', session_id=f'c1s{s}',
                             progression_coord=coord))
        df = pd.DataFrame(rows)
        types = _trajectory_typology(df, _FRAMEWORK)
        self.assertEqual(types[0]['trajectory_type'], 'climber')
        self.assertGreater(types[0]['progression_slope'], 0)


class TestRunMechanism(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def test_end_to_end_writes_files(self):
        df_all = _synthetic_all_df()
        attach_superposition(df_all, self.tmp, config=SuperpositionConfig())
        df = df_all[df_all['speaker'] == 'participant'].copy()

        result = run_mechanism_analysis(df, df_all, self.tmp, _FRAMEWORK)
        self.assertGreater(result['n_blocks'], 0)

        mech_dir = _paths.mechanism_dir(self.tmp)
        self.assertTrue(os.path.isfile(os.path.join(mech_dir, 'mechanism_delta_progression.csv')))
        self.assertTrue(os.path.isfile(os.path.join(mech_dir, 'participant_trajectory_types.csv')))

        rep = os.path.join(_paths.human_reports_dir(self.tmp), 'report_mechanism.txt')
        self.assertTrue(os.path.isfile(rep))
        av = os.path.join(_paths.human_reports_dir(self.tmp), 'report_avoidance_barrier.txt')
        self.assertTrue(os.path.isfile(av))

    def test_delta_csv_has_inference_columns(self):
        df_all = _synthetic_all_df()
        attach_superposition(df_all, self.tmp, config=SuperpositionConfig())
        df = df_all[df_all['speaker'] == 'participant'].copy()
        run_mechanism_analysis(df, df_all, self.tmp, _FRAMEWORK)
        ddf = pd.read_csv(os.path.join(_paths.mechanism_dir(self.tmp),
                                       'mechanism_delta_progression.csv'))
        for col in ('ci_lo', 'ci_hi', 'perm_p', 'fdr_significant',
                    'n_progress', 'n_stabilize', 'n_regress', 'n_participants'):
            self.assertIn(col, ddf.columns)

    def test_avoidance_delta_positive(self):
        df_all = _synthetic_all_df()
        attach_superposition(df_all, self.tmp, config=SuperpositionConfig())
        df = df_all[df_all['speaker'] == 'participant'].copy()
        run_mechanism_analysis(df, df_all, self.tmp, _FRAMEWORK)
        # Avoidance(1) → AttnReg(2) is a forward move, so Δprogression should be > 0.
        delta_csv = os.path.join(_paths.mechanism_dir(self.tmp), 'mechanism_delta_progression.csv')
        ddf = pd.read_csv(delta_csv)
        av = ddf[ddf['from_stage'] == 1]
        self.assertTrue((av['mean_delta_prog'] > 0).any())


if __name__ == '__main__':
    unittest.main()
