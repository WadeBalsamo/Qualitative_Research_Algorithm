"""Unit tests for analysis.reports.case_studies — deterministic per-cohort
case-study selection + grant-doc report generation."""

import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

import pandas as pd

from analysis.reports.case_studies import (
    compute_candidate_table, generate_case_studies, select_case_studies)
from process import output_paths as _paths


def _row(pid, cohort, sess, idx, stage, conf=0.9, tier='high'):
    return {
        'segment_id': f'T_c{cohort}s{sess}_{pid}_seg{idx:04d}',
        'participant_id': f'Participant_{pid}',
        'session_id': f'c{cohort}s{sess}',
        'session_number': sess,
        'cohort_id': cohort,
        'segment_index': idx,
        'speaker': 'participant',
        'text': f'[Participant_{pid}]: utterance {idx} at stage {stage} with enough '
                f'words to quote in a report body for testing purposes.',
        'word_count': 20,
        'final_label': stage,
        'llm_confidence_primary': conf,
        'llm_run_consistency': 3,
        'label_confidence_tier': tier,
    }


def _make_df():
    """Cohort 1: P1 climbs 0->4 over 5 sessions (Tier A); P2 flat & sparse.
    Cohort 2: P3 has no positive trend but rising MR share (Tier B).
    Cohort 3: only a sparse participant (Tier C)."""
    rows = []
    # P1 climber: sessions 1..5, three coded utterances each, stages rise.
    stages_by_sess = {1: [0, 0, 1], 2: [0, 1, 2], 3: [2, 2, 2], 4: [2, 3, 3], 5: [4, 4, 3]}
    for s, stages in stages_by_sess.items():
        for i, st in enumerate(stages):
            rows.append(_row('P1', 1, s, i, st))
    # P2 flat, sparse (ineligible: n_coded < 10).
    for s in (1, 2, 3, 4):
        rows.append(_row('P2', 1, s, 0, 2))
    # P3 consolidator: high stages throughout, MR share rises, tau ~ 0/negative.
    stages_by_sess = {1: [3, 2, 2], 2: [4, 2, 2], 3: [2, 2, 3], 4: [3, 4, 2], 5: [4, 4, 4]}
    for s, stages in stages_by_sess.items():
        for i, st in enumerate(stages):
            rows.append(_row('P3', 2, s, i, st))
    # P4 sparse in cohort 3.
    for s in (1, 2):
        rows.append(_row('P4', 3, s, 0, 1))
    return pd.DataFrame(rows)


class TestSelection(unittest.TestCase):

    def test_candidate_table_stats(self):
        table = compute_candidate_table(_make_df())
        p1 = table[table['participant_id'] == 'Participant_P1'].iloc[0]
        self.assertEqual(p1['cohort'], 1)
        self.assertEqual(p1['n_coded'], 15)
        self.assertGreater(p1['tau'], 0)
        self.assertFalse(p1['dual_cohort'])

    def test_one_pick_per_cohort_with_tiers(self):
        picks = select_case_studies(_make_df())
        by_cohort = {int(p['cohort']): p for p in picks}
        self.assertEqual(sorted(by_cohort), [1, 2, 3])
        self.assertEqual(by_cohort[1]['participant_id'], 'Participant_P1')
        self.assertEqual(by_cohort[1]['archetype'], 'climber')
        self.assertEqual(by_cohort[2]['participant_id'], 'Participant_P3')
        self.assertIn(by_cohort[2]['archetype'], ('climber', 'consolidator'))
        self.assertEqual(by_cohort[3]['archetype'], 'insufficient')

    def test_selection_is_deterministic(self):
        df = _make_df()
        a = select_case_studies(df)
        b = select_case_studies(df.sample(frac=1.0, random_state=7))  # row order shuffled
        self.assertEqual([p['participant_id'] for p in a],
                         [p['participant_id'] for p in b])
        self.assertEqual([p['tier'] for p in a], [p['tier'] for p in b])

    def test_cohort_derived_from_session_prefix_when_cohort_id_missing(self):
        df = _make_df()
        df.loc[df['participant_id'] == 'Participant_P3', 'cohort_id'] = float('nan')
        picks = select_case_studies(df)
        self.assertIn(2, [int(p['cohort']) for p in picks])


class TestGeneration(unittest.TestCase):

    def test_generates_doc_and_figure(self):
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            paths = generate_case_studies(df, df, framework=None, output_dir=tmp)
            self.assertEqual(len(paths), 2)
            txt, fig = paths
            self.assertEqual(txt, _paths.reports_case_studies_path(tmp))
            self.assertEqual(fig, _paths.reports_case_studies_figure_path(tmp))
            self.assertTrue(os.path.isfile(txt))
            self.assertTrue(os.path.isfile(fig))
            content = open(txt, encoding='utf-8').read()
            self.assertIn('CASE 1 - PARTICIPANT P1', content)
            self.assertIn('SELECTION AUDIT', content)
            self.assertIn('REDCAP LOOKUP CHECKLIST', content)
            self.assertIn('___', content)          # PRO blanks present
            self.assertIn('Kendall tau', content)

    def test_annotations_sidecar_is_included(self):
        df = _make_df()
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, '02_meta'), exist_ok=True)
            with open(os.path.join(tmp, '02_meta', 'case_study_annotations.json'),
                      'w', encoding='utf-8') as f:
                f.write('{"Participant_P1": {"profile": "TEST-PROFILE-XYZ", '
                        '"confounds": ["TEST-CONFOUND-ABC."]}}')
            txt = generate_case_studies(df, df, None, tmp)[0]
            content = open(txt, encoding='utf-8').read()
            self.assertIn('TEST-PROFILE-XYZ', content)
            self.assertIn('TEST-CONFOUND-ABC', content)

    def test_empty_df_returns_no_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(generate_case_studies(pd.DataFrame(), None, None, tmp), [])

    def test_output_path_accessors(self):
        self.assertEqual(_paths.reports_case_studies_path('/x'),
                         os.path.join('/x', '06_reports', '02_outcomes', 'case_studies.txt'))
        self.assertEqual(_paths.reports_case_studies_figure_path('/x'),
                         os.path.join('/x', '06_reports', '02_outcomes', 'case_studies_fig.png'))


if __name__ == '__main__':
    unittest.main()
