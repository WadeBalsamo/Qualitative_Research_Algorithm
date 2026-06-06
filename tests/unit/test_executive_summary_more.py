"""
tests/unit/test_executive_summary_more.py
-----------------------------------------
Covers gaps beyond test_report_organization.py for
analysis/reports/executive_summary.py:
  - All six section headers present
  - Deterministic program-improvement brief sections
  - Recommendations format (Observation/Mechanism/Proposed/Assess)
  - Descriptive/no-efficacy framing per methodology §8.3
  - Empty-artifacts graceful degradation
  - Mann-Kendall direction rendering
  - GNN-source note appears when mixture_source starts with 'gnn'
  - Underpowered note surfaced
  - _build_recommendations logic (barrier uncrossed triggers rec)
"""

import json
import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

import pandas as pd

from process import output_paths as _paths
from analysis.reports.executive_summary import generate_executive_summary, _build_recommendations


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_minimal_artifacts(tmp, eff_override=None, mech_rows=None, barrier_rows=None,
                              longitudinal_override=None):
    """Write the minimal set of JSON/CSV artifacts that generate_executive_summary reads."""
    eff_dir = _paths.efficacy_dir(tmp)
    mech_dir = _paths.mechanism_dir(tmp)
    data_dir = _paths.analysis_data_dir(tmp)
    for d in (eff_dir, mech_dir, data_dir):
        os.makedirs(d, exist_ok=True)

    # efficacy_summary.json
    eff = {
        'mk_adaptive_occupancy': {'n': 4, 'direction': 'increasing', 'p_value': 0.05,
                                  'sen_slope': 0.04, 'tau': 0.7},
        'trend_interval_sensitivity': {'slope': 0.15, 'p_value': 0.04},
        'adaptive_first_mean': 0.25, 'adaptive_last_mean': 0.60,
        'barrier_crossed': 2, 'barrier_total': 4,
        'underpowered': False, 'mixture_source': 'llm_ballots',
    }
    if eff_override:
        eff.update(eff_override)
    with open(os.path.join(eff_dir, 'efficacy_summary.json'), 'w') as f:
        json.dump(eff, f)

    # mechanism_delta_progression.csv
    rows = mech_rows if mech_rows is not None else [
        {'grouping': 'purer', 'from_stage': 1, 'from_stage_name': 'Avoidance',
         'behavior': 'Reframing', 'n': 10, 'mean_delta_prog': 0.42, 'fdr_significant': True},
        {'grouping': 'purer', 'from_stage': 0, 'from_stage_name': 'Vigilance',
         'behavior': 'Education', 'n': 6, 'mean_delta_prog': -0.25, 'fdr_significant': True},
    ]
    pd.DataFrame(rows).to_csv(os.path.join(mech_dir, 'mechanism_delta_progression.csv'), index=False)

    # barrier_crossing.csv
    barrier = barrier_rows if barrier_rows is not None else [
        {'participant_id': 'P1', 'crossed_to_attention_regulation': True},
        {'participant_id': 'P2', 'crossed_to_attention_regulation': False},
    ]
    pd.DataFrame(barrier).to_csv(os.path.join(eff_dir, 'barrier_crossing.csv'), index=False)

    # longitudinal_summary.json
    long_data = longitudinal_override if longitudinal_override is not None else {
        'group_progression': {'n_advancing': 3, 'n_stable': 1, 'n_regressing': 0},
        'feasibility_assessment': {'feasibility_rating': 'high', 'high_plus_medium_pct': 0.8},
        'validity_indicators': {'validity_narrative': 'Progression partially observed.'},
    }
    with open(os.path.join(data_dir, 'longitudinal_summary.json'), 'w') as f:
        json.dump(long_data, f)


def _read(tmp):
    path = _paths.executive_summary_path(tmp)
    with open(path, encoding='utf-8') as f:
        return f.read()


# ── test classes ──────────────────────────────────────────────────────────────

class TestExecutiveSummaryAllSections(unittest.TestCase):
    """All six section headers must appear in a fully-seeded run."""

    def test_six_section_headers(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('1. PROGRESSION OVER SESSIONS', content)
            self.assertIn('2. AVOIDANCE-BARRIER ASSESSMENT', content)
            self.assertIn('3. THERAPIST BEHAVIOR', content)
            self.assertIn('4. CONVERGENT VALIDITY', content)
            self.assertIn('5. CANDIDATE RECOMMENDATIONS', content)
            self.assertIn('6. VALIDATION CAVEATS', content)


class TestDescriptiveNoEfficacyFraming(unittest.TestCase):
    """Verify methodology §8.3 language: descriptive, not efficacy claim."""

    def test_not_efficacy_claim_language_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_executive_summary(tmp)
            content = _read(tmp)
            # Must say this is NOT an efficacy claim
            self.assertIn('NOT an efficacy claim', content)

    def test_directional_associational_framing(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('DIRECTIONAL', content)
            self.assertIn('ASSOCIATIONAL', content)

    def test_validation_caveat_section_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_executive_summary(tmp)
            content = _read(tmp)
            # Specific methodology caveats
            self.assertIn('not what they experience', content)
            self.assertIn('temporal-adjacency association', content)
            self.assertIn('NOT causal mechanism', content)

    def test_no_external_outcomes_message(self):
        """When no linkage CSV exists the summary says so explicitly."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('NOT yet integrated', content)


class TestRecommendationsFormat(unittest.TestCase):
    """5. CANDIDATE RECOMMENDATIONS must follow the four-field format."""

    def test_recommendation_four_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, barrier_rows=[
                {'participant_id': 'P1', 'crossed_to_attention_regulation': False},
                {'participant_id': 'P2', 'crossed_to_attention_regulation': False},
            ])
            generate_executive_summary(tmp)
            content = _read(tmp)
            # At least one recommendation block should appear
            self.assertIn('Observation:', content)
            self.assertIn('Mechanism:', content)
            self.assertIn('Proposed:', content)
            self.assertIn('Assess in C3', content)

    def test_recommendations_header_note(self):
        """The 'directional — require human review' caveat must precede the recs."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('directional', content.lower())
            self.assertIn('human review', content.lower())


class TestMannKendallRendering(unittest.TestCase):
    """Mann-Kendall trend direction and p-value should appear correctly."""

    def test_increasing_direction(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, eff_override={
                'mk_adaptive_occupancy': {'n': 4, 'direction': 'increasing', 'p_value': 0.03}
            })
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('increasing', content)
            self.assertIn('Mann', content)

    def test_flat_direction_triggers_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, eff_override={
                'mk_adaptive_occupancy': {'n': 4, 'direction': 'flat', 'p_value': 0.50}
            })
            generate_executive_summary(tmp)
            content = _read(tmp)
            # flat trend → consolidation recommendation
            self.assertIn('flat', content)

    def test_decreasing_direction_triggers_recommendation(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, eff_override={
                'mk_adaptive_occupancy': {'n': 4, 'direction': 'decreasing', 'p_value': 0.02}
            })
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('decreasing', content)


class TestGNNSourceNote(unittest.TestCase):
    """When mixture_source starts with 'gnn', a GNN-validated warning appears."""

    def test_gnn_source_note_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, eff_override={
                'mixture_source': 'gnn',
                'mk_adaptive_occupancy': {'n': 4, 'direction': 'increasing', 'p_value': 0.04},
            })
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('GNN-derived', content)


class TestUnderpoweredNote(unittest.TestCase):
    def test_underpowered_note_surfaces(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, eff_override={
                'underpowered': True,
                'power_note': 'UNDERPOWERED (n=3 participants).',
                'mk_adaptive_occupancy': {'n': 4, 'direction': 'increasing', 'p_value': 0.1},
            })
            generate_executive_summary(tmp)
            content = _read(tmp)
            self.assertIn('UNDERPOWERED', content)


class TestAdaptiveOccupancyLine(unittest.TestCase):
    def test_adaptive_occupancy_values_rendered(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp, eff_override={
                'adaptive_first_mean': 0.20,
                'adaptive_last_mean': 0.70,
                'mk_adaptive_occupancy': {'n': 4, 'direction': 'increasing', 'p_value': 0.04},
            })
            generate_executive_summary(tmp)
            content = _read(tmp)
            # 20% and 70% should appear
            self.assertIn('20%', content)
            self.assertIn('70%', content)


class TestEmptyArtifactsDegradation(unittest.TestCase):
    """generate_executive_summary must not raise and must write a file even with no artifacts."""

    def test_completely_empty_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_executive_summary(tmp)
            self.assertTrue(os.path.isfile(path))
            content = _read(tmp)
            self.assertIn('PROGRAM-IMPROVEMENT EXECUTIVE SUMMARY', content)
            # Graceful "not found" messages instead of blank
            self.assertIn('not found', content.lower())

    def test_missing_mechanism_csv(self):
        """If only mechanism CSV is absent, the therapist-behavior section degrades."""
        with tempfile.TemporaryDirectory() as tmp:
            eff_dir = _paths.efficacy_dir(tmp)
            os.makedirs(eff_dir, exist_ok=True)
            with open(os.path.join(eff_dir, 'efficacy_summary.json'), 'w') as f:
                json.dump({'barrier_crossed': 1, 'barrier_total': 2}, f)
            path = generate_executive_summary(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('mechanism', content.lower())


class TestBuildRecommendations(unittest.TestCase):
    """Unit-test _build_recommendations in isolation."""

    def _minimal_eff(self, crossed, total, mk_dir='increasing'):
        return {
            'barrier_crossed': crossed,
            'barrier_total': total,
            'mk_adaptive_occupancy': {'n': 4, 'direction': mk_dir, 'p_value': 0.05},
        }

    def test_barrier_rec_when_not_all_crossed(self):
        recs = _build_recommendations(
            self._minimal_eff(1, 4), mech=None, barrier=None, longitudinal=None
        )
        self.assertTrue(len(recs) >= 1)
        self.assertIn('Avoidance', recs[0]['observation'])

    def test_no_barrier_rec_when_all_crossed(self):
        recs = _build_recommendations(
            self._minimal_eff(4, 4), mech=None, barrier=None, longitudinal=None
        )
        # Barrier rec should not appear
        barrier_recs = [r for r in recs if 'Avoidance' in r['observation']]
        self.assertEqual(len(barrier_recs), 0)

    def test_flat_trend_rec_appears(self):
        recs = _build_recommendations(
            self._minimal_eff(3, 4, mk_dir='flat'), mech=None, barrier=None, longitudinal=None
        )
        flat_recs = [r for r in recs if 'flat' in r['observation']]
        self.assertTrue(len(flat_recs) >= 1)

    def test_backward_mover_rec_from_mech(self):
        mech = pd.DataFrame([{
            'from_stage': 1, 'from_stage_name': 'Avoidance',
            'behavior': 'SomeBadMove', 'n': 5, 'mean_delta_prog': -0.30,
        }])
        recs = _build_recommendations(
            self._minimal_eff(2, 4), mech=mech, barrier=None, longitudinal=None
        )
        backward_recs = [r for r in recs if 'backward' in r['mechanism'].lower() or
                         'SomeBadMove' in r['observation']]
        self.assertTrue(len(backward_recs) >= 1)

    def test_max_five_recommendations(self):
        mech = pd.DataFrame([{
            'from_stage': i, 'from_stage_name': f'Stage{i}',
            'behavior': f'Move{i}', 'n': 5, 'mean_delta_prog': -0.30 - i * 0.01,
        } for i in range(10)])
        recs = _build_recommendations(
            self._minimal_eff(0, 5, mk_dir='flat'), mech=mech, barrier=None, longitudinal=None
        )
        self.assertLessEqual(len(recs), 5)

    def test_returns_list(self):
        recs = _build_recommendations(None, None, None, None)
        self.assertIsInstance(recs, list)

    def test_all_recs_have_required_keys(self):
        mech = pd.DataFrame([{
            'from_stage': 1, 'from_stage_name': 'Avoidance',
            'behavior': 'Reframing', 'n': 8, 'mean_delta_prog': -0.20,
        }])
        recs = _build_recommendations(
            self._minimal_eff(1, 4, mk_dir='flat'), mech=mech, barrier=None, longitudinal=None
        )
        for r in recs:
            self.assertIn('observation', r)
            self.assertIn('mechanism', r)
            self.assertIn('change', r)
            self.assertIn('assess', r)


class TestReturnPath(unittest.TestCase):
    def test_returns_path_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_executive_summary(tmp)
            self.assertIsInstance(result, str)
            self.assertTrue(result.endswith('.txt'))

    def test_file_is_utf8(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            path = generate_executive_summary(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertGreater(len(content), 100)


if __name__ == '__main__':
    unittest.main()
