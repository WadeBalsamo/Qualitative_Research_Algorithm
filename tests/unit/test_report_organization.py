"""
tests/test_report_organization.py
---------------------------------
Covers the tiered 06_reports/ reorganization:
  • path helpers resolve under 06_reports/NN_* ;
  • the deterministic executive summary + reports guide write non-empty files
    from minimal analysis artifacts (no LLM, no real pipeline).
"""

import json
import os
import tempfile
import unittest

import pandas as pd

from process import output_paths as _paths
from analysis.reports.executive_summary import generate_executive_summary
from analysis.reports.reports_guide import (
    generate_methods_appendix,
    generate_reports_readme,
)


class TestReportPathHelpers(unittest.TestCase):
    def test_tiered_dirs_under_06_reports(self):
        run = '/tmp/run'
        self.assertEqual(_paths.reports_outcomes_dir(run), '/tmp/run/06_reports/01_outcomes')
        self.assertEqual(_paths.reports_mechanism_dir(run), '/tmp/run/06_reports/02_mechanism')
        self.assertEqual(_paths.reports_per_session_dir(run), '/tmp/run/06_reports/03_per_session')
        self.assertEqual(_paths.reports_per_participant_dir(run), '/tmp/run/06_reports/04_per_participant')
        self.assertEqual(_paths.themes_dir(run), '/tmp/run/06_reports/05_per_stage')
        self.assertEqual(_paths.reports_gnn_dir(run), '/tmp/run/06_reports/06_gnn')
        self.assertTrue(_paths.executive_summary_path(run).endswith('06_reports/00_executive_summary.txt'))
        self.assertTrue(_paths.methods_appendix_path(run).endswith('06_reports/07_methods_appendix.txt'))


def _seed_artifacts(tmp):
    """Write minimal efficacy + mechanism + longitudinal artifacts."""
    eff_dir = _paths.efficacy_dir(tmp)
    mech_dir = _paths.mechanism_dir(tmp)
    data_dir = _paths.analysis_data_dir(tmp)
    for d in (eff_dir, mech_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    pd.DataFrame([
        {'session_number': 1, 'n_participants': 5, 'mean': 1.2, 'ci_lo': 0.9, 'ci_hi': 1.5},
        {'session_number': 8, 'n_participants': 5, 'mean': 2.1, 'ci_lo': 1.7, 'ci_hi': 2.5},
    ]).to_csv(os.path.join(eff_dir, 'group_progression_trajectory.csv'), index=False)

    pd.DataFrame([
        {'participant_id': 'P1', 'n_sessions': 8, 'expressed_barrier_from': True,
         'crossed_to_attention_regulation': True, 'first_passage_session_index': 3},
        {'participant_id': 'P2', 'n_sessions': 8, 'expressed_barrier_from': True,
         'crossed_to_attention_regulation': False, 'first_passage_session_index': None},
    ]).to_csv(os.path.join(eff_dir, 'barrier_crossing.csv'), index=False)

    with open(os.path.join(eff_dir, 'efficacy_summary.json'), 'w') as f:
        json.dump({
            'mk_adaptive_occupancy': {'n': 4, 'direction': 'increasing', 'p_value': 0.08, 'sen_slope': 0.05, 'tau': 0.6},
            'mk_progression_coord': {'n': 4, 'direction': 'increasing', 'p_value': 0.1},
            'trend_interval_sensitivity': {'slope': -0.02, 'p_value': 0.3, 'method': 'mixedlm',
                                           'ci_lo': -0.1, 'ci_hi': 0.06, 'n': 10, 'n_groups': 5},
            'sign_test': {'n_positive': 2, 'n_total': 5, 'p_value': 0.5},
            'n_advancing': 2, 'n_participants': 5, 'n_sessions': 2,
            'underpowered': True, 'power_note': 'UNDERPOWERED (n=5 participants, 2 sessions).',
            'mixture_source': 'llm_ballots',
            'barrier_crossed': 1, 'barrier_total': 2,
            'adaptive_first_mean': 0.3, 'adaptive_last_mean': 0.5,
            'group_first_mean': 1.2, 'group_last_mean': 2.1,
        }, f)

    pd.DataFrame([
        {'grouping': 'purer', 'from_stage': 1, 'from_stage_name': 'Avoidance',
         'behavior': 'Reframing', 'n': 12, 'mean_delta_prog': 0.45, 'fdr_significant': True},
        {'grouping': 'purer', 'from_stage': 0, 'from_stage_name': 'Vigilance',
         'behavior': 'Utilization', 'n': 7, 'mean_delta_prog': -0.30, 'fdr_significant': True},
    ]).to_csv(os.path.join(mech_dir, 'mechanism_delta_progression.csv'), index=False)

    with open(os.path.join(data_dir, 'longitudinal_summary.json'), 'w') as f:
        json.dump({
            'group_progression': {'n_advancing': 2, 'n_stable': 2, 'n_regressing': 1},
            'feasibility_assessment': {'feasibility_rating': 'medium', 'high_plus_medium_pct': 0.6},
            'validity_indicators': {'validity_narrative': 'Expected progression partially observed.'},
        }, f)


class TestExecutiveSummary(unittest.TestCase):
    def test_writes_brief_with_both_directions(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_artifacts(tmp)
            path = generate_executive_summary(tmp, df=None, framework={})
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('PROGRAM-IMPROVEMENT EXECUTIVE SUMMARY', content)
            self.assertIn('FORWARD-associated', content)
            self.assertIn('BACKWARD', content)
            self.assertIn('Reframing', content)   # top forward mover
            self.assertIn('Utilization', content)  # backward mover
            self.assertIn('CANDIDATE RECOMMENDATIONS', content)
            self.assertIn('VALIDATION CAVEATS', content)

    def test_guides_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_artifacts(tmp)
            ma = generate_methods_appendix(tmp)
            rm = generate_reports_readme(tmp)
            self.assertTrue(os.path.getsize(ma) > 0)
            self.assertTrue(os.path.getsize(rm) > 0)


if __name__ == '__main__':
    unittest.main()
