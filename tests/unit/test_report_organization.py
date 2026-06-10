"""
tests/unit/test_report_organization.py
---------------------------------------
Covers the tiered 06_reports/ reorganization:
  - Path helpers resolve under the correct 06_reports/NN_* subdirectories.
  - generate_results_brief and generate_methods_report write non-empty files
    from minimal analysis artifacts (no LLM, no real pipeline).
  - The new supplementary / reliability / thesis-figure paths are reachable.
"""

import json
import os
import tempfile
import unittest

import pandas as pd

from process import output_paths as _paths
from analysis.reports.results_brief import generate_results_brief
from analysis.reports.methods_report import generate_methods_report


class TestReportPathHelpers(unittest.TestCase):
    """Assert the exact relative sub-paths documented in CLAUDE.md."""

    def test_tiered_dirs_under_06_reports(self):
        run = '/tmp/run'
        self.assertEqual(_paths.reports_outcomes_dir(run),
                         '/tmp/run/06_reports/02_outcomes')
        self.assertEqual(_paths.reports_mechanism_dir(run),
                         '/tmp/run/06_reports/03_mechanism')
        self.assertEqual(_paths.reports_per_session_dir(run),
                         '/tmp/run/06_reports/04_per_session')
        self.assertEqual(_paths.reports_per_participant_dir(run),
                         '/tmp/run/06_reports/05_per_participant')
        self.assertEqual(_paths.themes_dir(run),
                         '/tmp/run/06_reports/06_per_stage')
        self.assertEqual(_paths.reports_gnn_dir(run),
                         '/tmp/run/06_reports/07_gnn')

    def test_top_level_files(self):
        run = '/tmp/run'
        self.assertTrue(
            _paths.reports_results_path(run).endswith('06_reports/00_RESULTS.txt'))
        self.assertTrue(
            _paths.reports_methods_path(run).endswith('06_reports/08_methods.txt'))

    def test_reliability_tier(self):
        run = '/tmp/run'
        self.assertEqual(_paths.reports_reliability_dir(run),
                         '/tmp/run/06_reports/01_reliability')
        self.assertEqual(_paths.reports_irr_path(run),
                         '/tmp/run/06_reports/01_reliability/irr_report.txt')
        # classifier reports alias reliability
        self.assertEqual(_paths.reports_classifier_dir(run),
                         _paths.reports_reliability_dir(run))

    def test_supplementary_tier(self):
        run = '/tmp/run'
        self.assertEqual(_paths.reports_supplementary_dir(run),
                         '/tmp/run/06_reports/09_supplementary')

    def test_thesis_figure_paths(self):
        run = '/tmp/run'
        self.assertTrue(
            _paths.thesis_figure_path(run, 1).endswith('00_fig1_rehabituation_arc.png'))
        self.assertTrue(
            _paths.thesis_figure_path(run, 2).endswith('00_fig2_dyadic_mechanism.png'))
        self.assertTrue(
            _paths.thesis_figure_path(run, 3).endswith('00_fig3_dashboard.png'))
        # All three sit in the 06_reports root alongside 00_RESULTS.txt
        for n in (1, 2, 3):
            fig = _paths.thesis_figure_path(run, n)
            self.assertEqual(os.path.dirname(fig), _paths.human_reports_dir(run))


def _seed_artifacts(tmp):
    """Write minimal efficacy + mechanism + longitudinal artifacts."""
    eff_dir = _paths.efficacy_dir(tmp)
    mech_dir = _paths.mechanism_dir(tmp)
    data_dir = _paths.analysis_data_dir(tmp)
    irr_dir = _paths.irr_validation_dir(tmp)
    for d in (eff_dir, mech_dir, data_dir, irr_dir):
        os.makedirs(d, exist_ok=True)

    pd.DataFrame([
        {'session_number': 1, 'n_participants': 5, 'mean': 1.2, 'ci_lo': 0.9, 'ci_hi': 1.5},
        {'session_number': 8, 'n_participants': 5, 'mean': 2.1, 'ci_lo': 1.7, 'ci_hi': 2.5},
    ]).to_csv(os.path.join(eff_dir, 'group_progression_trajectory.csv'), index=False)

    pd.DataFrame([
        {'participant_id': 'P1', 'crossed_to_attention_regulation': True},
        {'participant_id': 'P2', 'crossed_to_attention_regulation': False},
    ]).to_csv(os.path.join(eff_dir, 'barrier_crossing.csv'), index=False)

    with open(os.path.join(eff_dir, 'efficacy_summary.json'), 'w') as f:
        json.dump({
            'mk_adaptive_occupancy': {
                'n': 4, 'direction': 'increasing', 'p_value': 0.08,
                'sen_slope': 0.05, 'tau': 0.6,
            },
            'trend_interval_sensitivity': {
                'slope': -0.02, 'p_value': 0.3, 'method': 'mixedlm',
                'ci_lo': -0.1, 'ci_hi': 0.06, 'n': 10, 'n_groups': 5,
            },
            'underpowered': True,
            'power_note': 'UNDERPOWERED (n=5 participants, 2 sessions).',
            'mixture_source': 'llm_ballots',
            'barrier_crossed': 1, 'barrier_total': 2,
            'adaptive_first_mean': 0.3, 'adaptive_last_mean': 0.5,
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
            'feasibility_assessment': {
                'feasibility_rating': 'medium', 'high_plus_medium_pct': 0.6,
            },
            'validity_indicators': {
                'validity_narrative': 'Expected progression partially observed.',
            },
        }, f)


class TestResultsBrief(unittest.TestCase):
    def test_writes_brief_with_core_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_artifacts(tmp)
            path = generate_results_brief(tmp, df=None, framework={})
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('RESULTS', content)
            # Outcomes data from the seeded artifacts
            self.assertIn('Reframing', content)
            self.assertIn('Utilization', content)

    def test_results_brief_written_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_artifacts(tmp)
            path = generate_results_brief(tmp)
            self.assertEqual(path, _paths.reports_results_path(tmp))
            self.assertTrue(path.endswith('00_RESULTS.txt'))


class TestMethodsReport(unittest.TestCase):
    def test_methods_report_writes_nonempty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            _seed_artifacts(tmp)
            path = generate_methods_report(tmp)
            self.assertGreater(os.path.getsize(path), 0)

    def test_methods_report_written_to_correct_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_methods_report(tmp)
            self.assertEqual(path, _paths.reports_methods_path(tmp))
            self.assertTrue(path.endswith('08_methods.txt'))


if __name__ == '__main__':
    unittest.main()
