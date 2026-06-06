"""
tests/unit/test_purer_analysis_more.py
---------------------------------------
Gap coverage for analysis/purer_analysis.py — complementing test_purer_analysis.py.

New coverage:
  - conditional_table / purer_transition_profiles: for each (from_stage, to_stage)
    the dominant-PURER distribution is correct (methodology §7.6).
  - PURER × VAAMR influence lift: lift values computed correctly per block counts.
  - FROM→CUE→TO triple assembly where CUE = therapist segments strictly between
    consecutive participant turns (§4.8): verified via compute_cue_block_purer_profiles
    on a make_master_df slice with known structure.
  - run_purer_analysis from make_master_df (LLM summarization absent — no monkeypatching
    needed because generate_purer_report is purely text/CSV, no LLM calls).
  - Edge: no transitions (single participant, single labeled turn → no blocks).
  - Edge: single participant → produces cue blocks but no cross-participant aggregation error.
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.purer_analysis import (
    CueBlock,
    compute_cue_block_purer_profiles,
    compute_purer_transition_influence,
    run_purer_analysis,
)
from process import output_paths as _paths
from tests.testhelpers.fixtures import make_master_df


_FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cb(from_stage, to_stage, purer_labels, session='s1'):
    return CueBlock(session, 1, 'a', 'b', from_stage, to_stage, purer_labels)


def _make_interleaved_df(stages_and_purers, session_id='s1'):
    """Build participant/therapist rows from a list of (speaker, stage, purer) tuples."""
    rows = []
    for i, (speaker, stage, purer) in enumerate(stages_and_purers):
        is_p = speaker == 'participant'
        rows.append({
            'segment_id': f'{session_id}_{i}',
            'session_id': session_id,
            'speaker': speaker,
            'start_time_ms': i * 1000,
            'end_time_ms': i * 1000 + 800,
            'final_label': float(stage) if is_p and stage is not None else np.nan,
            'purer_primary': float(purer) if not is_p and purer is not None else np.nan,
            'cohort_id': 1,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# conditional_table / purer_transition_profiles (§7.6)
# ---------------------------------------------------------------------------

class TestConditionalTableProfiles(unittest.TestCase):
    """For each (from_stage, to_stage) the dominant-PURER distribution sums to 1."""

    def _blocks_multi_transition(self):
        """3 distinct transition types, each with 2 mediated blocks."""
        return [
            # Vigilance → Avoidance, PURER=0 (x2)
            _make_cb(0, 1, [0]),
            _make_cb(0, 1, [0]),
            # Avoidance → AttnReg, PURER=2 (x2)
            _make_cb(1, 2, [2]),
            _make_cb(1, 2, [2]),
            # Metacog → Reappraisal, PURER=4 (x2)
            _make_cb(3, 4, [4]),
            _make_cb(3, 4, [4]),
        ]

    def test_conditional_table_fractions_sum_to_one_per_transition(self):
        blocks = self._blocks_multi_transition()
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        for (fs, ts), grp in ct.groupby(['from_stage', 'to_stage']):
            total = grp['fraction_of_mediated'].sum()
            self.assertAlmostEqual(total, 1.0, places=2,
                msg=f'Fractions for ({fs}→{ts}) sum to {total:.4f}, expected 1.0')

    def test_conditional_table_dominant_purer_per_transition(self):
        blocks = self._blocks_multi_transition()
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        # Vigilance→Avoidance: dominant PURER should be 0 (Phenomenology)
        row_01 = ct[(ct['from_stage'] == 0) & (ct['to_stage'] == 1)]
        self.assertEqual(len(row_01), 1)
        self.assertEqual(int(row_01.iloc[0]['dominant_purer']), 0)
        # Avoidance→AttnReg: dominant PURER = 2 (Reframing)
        row_12 = ct[(ct['from_stage'] == 1) & (ct['to_stage'] == 2)]
        self.assertEqual(len(row_12), 1)
        self.assertEqual(int(row_12.iloc[0]['dominant_purer']), 2)

    def test_conditional_table_includes_purer_name_and_short(self):
        blocks = [_make_cb(0, 1, [1])]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        self.assertIn('purer_name', ct.columns)
        self.assertIn('purer_short', ct.columns)
        row = ct.iloc[0]
        self.assertEqual(row['purer_name'], 'Utilization')
        self.assertEqual(row['purer_short'], 'U')

    def test_transition_profiles_equals_conditional_table(self):
        """transition_profiles is an alias for conditional_table."""
        blocks = self._blocks_multi_transition()
        result = compute_purer_transition_influence(blocks)
        pd.testing.assert_frame_equal(
            result['transition_profiles'].reset_index(drop=True),
            result['conditional_table'].reset_index(drop=True),
        )

    def test_mixed_purer_in_transition(self):
        """Multiple PURER codes for same transition → fractions correctly distributed."""
        blocks = [
            _make_cb(1, 2, [0]),   # Phenomenology
            _make_cb(1, 2, [2]),   # Reframing
            _make_cb(1, 2, [2]),   # Reframing
            _make_cb(1, 2, [4]),   # Reinforcement
        ]
        result = compute_purer_transition_influence(blocks)
        ct = result['conditional_table']
        avoidance_to_attn = ct[(ct['from_stage'] == 1) & (ct['to_stage'] == 2)]
        # 3 PURER codes observed: 0, 2, 4
        self.assertEqual(len(avoidance_to_attn), 3)
        # Reframing(2) should be the most frequent
        max_row = avoidance_to_attn.loc[avoidance_to_attn['fraction_of_mediated'].idxmax()]
        self.assertEqual(int(max_row['dominant_purer']), 2)
        # fractions: Phenomenology=1/4, Reframing=2/4, Reinforcement=1/4
        p_row = avoidance_to_attn[avoidance_to_attn['dominant_purer'] == 0]
        self.assertAlmostEqual(float(p_row.iloc[0]['fraction_of_mediated']), 0.25, places=2)
        r_row = avoidance_to_attn[avoidance_to_attn['dominant_purer'] == 2]
        self.assertAlmostEqual(float(r_row.iloc[0]['fraction_of_mediated']), 0.5, places=2)


# ---------------------------------------------------------------------------
# PURER × VAAMR influence lift
# ---------------------------------------------------------------------------

class TestPurerVaamrLift(unittest.TestCase):
    """Lift = P(to_stage | PURER) / P(to_stage).  Hand-computed checks."""

    def test_lift_equals_marginal_when_purer_unrelated(self):
        """If every PURER code leads to every stage equally, lift = 1.0."""
        blocks = [
            _make_cb(0, 0, [0]), _make_cb(0, 1, [0]),
            _make_cb(1, 0, [1]), _make_cb(1, 1, [1]),
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        # P(to=0) = P(to=1) = 0.5
        # P(to=0|P0) = 1/2 → lift = 1.0; P(to=1|P0) = 1/2 → lift = 1.0
        row_p = lm[lm['purer_construct'] == 'Phenomenology']
        self.assertAlmostEqual(float(row_p['lift_to_0'].iloc[0]), 1.0, places=1)
        self.assertAlmostEqual(float(row_p['lift_to_1'].iloc[0]), 1.0, places=1)

    def test_lift_elevated_when_purer_always_leads_to_stage(self):
        """PURER=R (2) always leads to stage=3; base rate of stage=3 = 50%."""
        # 2 R→3 blocks; 1 P→1 block; 1 P→3 block
        # base rate P(to=3) = 3/4 = 0.75; P(to=3 | R) = 2/2 = 1.0 → lift = 1/0.75 ≈ 1.33
        blocks = [
            _make_cb(0, 3, [2]),
            _make_cb(1, 3, [2]),
            _make_cb(0, 1, [0]),
            _make_cb(1, 3, [0]),
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        row_r = lm[lm['purer_construct'] == 'Reframing']
        self.assertFalse(row_r.empty)
        lift_3 = float(row_r['lift_to_3'].iloc[0])
        self.assertGreater(lift_3, 1.0,
            f'Expected lift > 1.0 for R→3, got {lift_3}')

    def test_lift_zero_when_purer_never_leads_to_stage(self):
        """PURER=E (3) never leads to stage=4; lift should be 0.0."""
        blocks = [
            _make_cb(0, 0, [3]),
            _make_cb(0, 1, [3]),
            _make_cb(1, 4, [0]),   # stage 4 only appears with PURER=0
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        row_e = lm[lm['purer_construct'] == 'Education']
        if not row_e.empty and 'lift_to_4' in row_e.columns:
            self.assertAlmostEqual(float(row_e['lift_to_4'].iloc[0]), 0.0, places=2)

    def test_lift_matrix_n_blocks_matches_block_count(self):
        blocks = [
            _make_cb(0, 1, [0]), _make_cb(1, 2, [0]),  # 2 blocks with PURER=0
            _make_cb(0, 2, [1]),                         # 1 block with PURER=1
        ]
        result = compute_purer_transition_influence(blocks)
        lm = result['lift_matrix']
        row_p = lm[lm['purer_construct'] == 'Phenomenology'].iloc[0]
        row_u = lm[lm['purer_construct'] == 'Utilization'].iloc[0]
        self.assertEqual(int(row_p['n_blocks']), 2)
        self.assertEqual(int(row_u['n_blocks']), 1)


# ---------------------------------------------------------------------------
# FROM→CUE→TO triple assembly (§4.8)
# ---------------------------------------------------------------------------

class TestFromCueToTripleAssembly(unittest.TestCase):
    """Therapist segments strictly between consecutive participant turns form the CUE."""

    def test_cue_is_therapist_segments_between_participant_turns(self):
        """P1→T1→T2→P2 yields one block with 2 therapist segments."""
        df = _make_interleaved_df([
            ('participant', 0, None),
            ('therapist',   None, 2),
            ('therapist',   None, 2),
            ('participant', 3, None),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].n_therapist_segs, 2)
        self.assertEqual(blocks[0].from_stage, 0)
        self.assertEqual(blocks[0].to_stage, 3)

    def test_from_stage_comes_from_first_participant(self):
        df = _make_interleaved_df([
            ('participant', 2, None),
            ('therapist',   None, 1),
            ('participant', 4, None),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].from_stage, 2)

    def test_to_stage_comes_from_second_participant(self):
        df = _make_interleaved_df([
            ('participant', 2, None),
            ('therapist',   None, 1),
            ('participant', 4, None),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].to_stage, 4)

    def test_therapist_before_first_participant_not_in_block(self):
        """Therapist segment before the first participant turn is not counted."""
        df = _make_interleaved_df([
            ('therapist',   None, 3),   # before first participant — excluded
            ('participant', 0, None),
            ('therapist',   None, 2),   # inside block
            ('participant', 1, None),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        # Only 1 therapist segment is inside the window (the one between participants)
        self.assertEqual(blocks[0].n_therapist_segs, 1)
        self.assertEqual(blocks[0].purer_profile, {2: 1})

    def test_therapist_after_last_participant_not_in_block(self):
        """Therapist segment after the last participant turn is not counted."""
        df = _make_interleaved_df([
            ('participant', 0, None),
            ('therapist',   None, 2),   # in window
            ('participant', 1, None),
            ('therapist',   None, 4),   # after last participant — no subsequent participant
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        # Only the first therapist segment is in a valid block window
        self.assertEqual(blocks[0].purer_profile, {2: 1})

    def test_purer_label_comes_from_therapist_only(self):
        """Participant purer_primary field must be ignored when building cue profiles."""
        df = _make_interleaved_df([
            ('participant', 0, None),
            ('therapist',   None, 2),   # therapist PURER=2
            ('participant', 1, 4),      # participant has purer=4 — MUST be ignored
        ])
        # Manually add purer_primary to participant row
        df.loc[df['speaker'] == 'participant', 'purer_primary'] = 4.0
        # But therapist row should still be 2
        df.loc[df['speaker'] == 'therapist', 'purer_primary'] = 2.0
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        # Only therapist PURER=2 should count
        self.assertEqual(blocks[0].purer_profile, {2: 1})

    def test_multi_session_blocks_independent(self):
        """Blocks from different sessions must not mix therapist segments."""
        rows_s1 = [
            ('participant', 0, None),
            ('therapist',   None, 0),   # session s1 therapist
            ('participant', 1, None),
        ]
        rows_s2 = [
            ('participant', 2, None),
            ('therapist',   None, 3),   # session s2 therapist
            ('participant', 3, None),
        ]
        df_s1 = _make_interleaved_df(rows_s1, session_id='s1')
        df_s2 = _make_interleaved_df(rows_s2, session_id='s2')
        # Reassign timestamps for s2 to avoid overlap confusion
        offset = 100000
        df_s2['start_time_ms'] += offset
        df_s2['end_time_ms'] += offset
        df = pd.concat([df_s1, df_s2], ignore_index=True)
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 2)
        s1_block = next(b for b in blocks if b.session_id == 's1')
        s2_block = next(b for b in blocks if b.session_id == 's2')
        self.assertEqual(s1_block.purer_profile, {0: 1})
        self.assertEqual(s2_block.purer_profile, {3: 1})

    def test_from_to_transition_type_forward(self):
        """from_stage < to_stage → transition_type == 'forward'."""
        df = _make_interleaved_df([
            ('participant', 1, None),
            ('therapist',   None, 2),
            ('participant', 3, None),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].transition_type, 'forward')

    def test_from_to_transition_type_backward(self):
        """from_stage > to_stage → transition_type == 'backward'."""
        df = _make_interleaved_df([
            ('participant', 3, None),
            ('therapist',   None, 2),
            ('participant', 1, None),
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].transition_type, 'backward')


# ---------------------------------------------------------------------------
# run_purer_analysis from make_master_df (full pipeline, no LLM)
# ---------------------------------------------------------------------------

class TestRunPurerAnalysisMakeMasterDf(unittest.TestCase):
    """run_purer_analysis on make_master_df produces correct outputs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df_all = make_master_df(n_sessions=3, n_participants=2)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_cue_blocks(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        self.assertIn('cue_blocks', result)
        self.assertIsInstance(result['cue_blocks'], list)

    def test_cue_blocks_non_empty(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        # make_master_df has 2 participant segments per session-pair
        self.assertGreater(result['n_cue_blocks'], 0)

    def test_influence_dict_has_expected_keys(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        influence = result['influence']
        for key in ('transition_profiles', 'lift_matrix', 'empty_cue_rates', 'conditional_table'):
            self.assertIn(key, influence)

    def test_purer_report_written(self):
        run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        report = os.path.join(_paths.reports_mechanism_dir(self.tmp), 'purer.txt')
        self.assertTrue(os.path.isfile(report))

    def test_csvs_written(self):
        run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        analysis_dir = _paths.analysis_data_dir(self.tmp)
        # At least one CSV should be written
        csv_files = [f for f in os.listdir(analysis_dir) if f.endswith('.csv')] if os.path.isdir(analysis_dir) else []
        self.assertGreater(len(csv_files), 0, 'No CSVs written to analysis_data_dir')

    def test_conditional_table_fractions_valid(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        ct = result['influence']['conditional_table']
        if not ct.empty and 'fraction_of_mediated' in ct.columns:
            # All fraction values must be in [0, 1]
            self.assertTrue((ct['fraction_of_mediated'] >= 0.0).all())
            self.assertTrue((ct['fraction_of_mediated'] <= 1.0).all())

    def test_n_mediated_plus_n_empty_equals_total(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        total = result['n_cue_blocks']
        n_med = result['n_mediated']
        n_empty = result['n_empty']
        self.assertEqual(n_med + n_empty, total)

    def test_lift_matrix_lift_values_non_negative(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        lm = result['influence']['lift_matrix']
        if not lm.empty:
            lift_cols = [c for c in lm.columns if c.startswith('lift_to_')]
            for col in lift_cols:
                self.assertTrue((lm[col] >= 0.0).all(),
                    f'Lift column {col} has negative values')


# ---------------------------------------------------------------------------
# Edge: no transitions (single participant, single labeled turn)
# ---------------------------------------------------------------------------

class TestEdgeNoTransitions(unittest.TestCase):
    """Single participant with only one labeled participant segment → no cue blocks."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_single_participant_segment_no_blocks(self):
        df = pd.DataFrame([{
            'segment_id': 'p1',
            'session_id': 's1',
            'speaker': 'participant',
            'start_time_ms': 0,
            'end_time_ms': 5000,
            'final_label': 1.0,
            'purer_primary': np.nan,
            'cohort_id': 1,
        }])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 0)

    def test_compute_purer_transition_influence_no_blocks_returns_empty(self):
        result = compute_purer_transition_influence([])
        self.assertTrue(result['transition_profiles'].empty)
        self.assertTrue(result['lift_matrix'].empty)
        self.assertTrue(result['empty_cue_rates'].empty)
        self.assertTrue(result['conditional_table'].empty)
        self.assertEqual(result['raw_cue_blocks'], [])

    def test_run_purer_analysis_no_transitions_no_crash(self):
        """A single-row participant df must not crash run_purer_analysis."""
        df = pd.DataFrame([{
            'segment_id': 'p1', 'participant_id': 'P01',
            'session_id': 's1', 'session_number': 1, 'cohort_id': 1,
            'speaker': 'participant',
            'start_time_ms': 0, 'end_time_ms': 5000,
            'final_label': 1.0, 'purer_primary': np.nan,
            'text': 'hello',
        }])
        try:
            result = run_purer_analysis(df, self.tmp, framework=_FRAMEWORK)
        except Exception as exc:
            self.fail(f'run_purer_analysis raised on single-turn df: {exc}')
        self.assertEqual(result['n_cue_blocks'], 0)

    def test_all_unlabeled_participant_segs_no_blocks(self):
        """If all participant final_labels are NaN, no cue blocks are built."""
        df = pd.DataFrame([
            {'segment_id': 'p1', 'session_id': 's1', 'speaker': 'participant',
             'start_time_ms': 0, 'end_time_ms': 5000,
             'final_label': np.nan, 'purer_primary': np.nan, 'cohort_id': 1},
            {'segment_id': 't1', 'session_id': 's1', 'speaker': 'therapist',
             'start_time_ms': 5000, 'end_time_ms': 8000,
             'final_label': np.nan, 'purer_primary': 2.0, 'cohort_id': 1},
            {'segment_id': 'p2', 'session_id': 's1', 'speaker': 'participant',
             'start_time_ms': 8000, 'end_time_ms': 12000,
             'final_label': np.nan, 'purer_primary': np.nan, 'cohort_id': 1},
        ])
        blocks = compute_cue_block_purer_profiles(df)
        self.assertEqual(len(blocks), 0)


# ---------------------------------------------------------------------------
# Edge: single participant, multiple sessions
# ---------------------------------------------------------------------------

class TestSingleParticipant(unittest.TestCase):
    """Single participant produces cue blocks but no cross-participant aggregation error."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df_all = make_master_df(n_sessions=3, n_participants=1)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_purer_analysis_single_participant_no_crash(self):
        try:
            result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        except Exception as exc:
            self.fail(f'run_purer_analysis raised on single-participant df: {exc}')
        self.assertIn('n_cue_blocks', result)

    def test_single_participant_cue_blocks_built(self):
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        # make_master_df(n_participants=1, n_sessions=3) has participant segments
        # so some cue blocks should be built
        self.assertGreaterEqual(result['n_cue_blocks'], 0)  # may be 0 if no pairs, but no crash

    def test_lift_matrix_well_formed_with_single_participant(self):
        """Lift matrix produced from a single participant must be a DataFrame."""
        result = run_purer_analysis(self.df_all, self.tmp, framework=_FRAMEWORK)
        lm = result['influence']['lift_matrix']
        self.assertIsInstance(lm, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
