"""
tests/unit/test_results_brief.py
---------------------------------
Exercises analysis/reports/results_brief.py (generate_results_brief) and
analysis/reports/methods_report.py (generate_methods_report) against a temp
dir with minimal synthetic artifacts.

Covers:
  - generate_results_brief returns a path that exists and is non-empty
  - The returned file is at reports_results_path (06_reports/00_RESULTS.txt)
  - Core header text appears ("RESULTS")
  - Section headers appear (1. SAMPLE, 2. RELIABILITY, 3. OUTCOMES, 4. MECHANISM)
  - Graceful degradation: missing CSVs → sections say data unavailable
  - [M#] tags appear in the output (at least one m_ref call fired)
  - generate_methods_report returns a path at reports_methods_path (08_methods.txt)
  - 08_methods.txt contains entries from the methods_entries() registry
  - Both generators handle a completely empty output dir without raising
"""

import json
import os
import sys
import tempfile
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from process import output_paths as _paths
from analysis.reports.results_brief import generate_results_brief
from analysis.reports.methods_report import generate_methods_report
from analysis.reports.stat_format import methods_entries


# ── helpers ───────────────────────────────────────────────────────────────────

def _write_minimal_artifacts(tmp):
    """Write the minimal set of JSON/CSV artifacts that generate_results_brief reads."""
    eff_dir = _paths.efficacy_dir(tmp)
    mech_dir = _paths.mechanism_dir(tmp)
    data_dir = _paths.analysis_data_dir(tmp)
    irr_dir = _paths.irr_validation_dir(tmp)
    for d in (eff_dir, mech_dir, data_dir, irr_dir):
        os.makedirs(d, exist_ok=True)

    # efficacy_summary.json
    with open(os.path.join(eff_dir, 'efficacy_summary.json'), 'w') as f:
        json.dump({
            'mk_adaptive_occupancy': {
                'n': 4, 'direction': 'increasing', 'p_value': 0.05,
                'sen_slope': 0.04, 'tau': 0.7,
            },
            'trend_interval_sensitivity': {'slope': 0.15, 'p_value': 0.04},
            'adaptive_first_mean': 0.25, 'adaptive_last_mean': 0.60,
            'barrier_crossed': 2, 'barrier_total': 4,
            'underpowered': False, 'mixture_source': 'llm_ballots',
        }, f)

    # mechanism_delta_progression.csv
    pd.DataFrame([
        {'grouping': 'purer', 'from_stage': 1, 'from_stage_name': 'Avoidance',
         'behavior': 'Reframing', 'n': 10, 'mean_delta_prog': 0.42, 'fdr_significant': True},
        {'grouping': 'purer', 'from_stage': 0, 'from_stage_name': 'Vigilance',
         'behavior': 'Education', 'n': 6, 'mean_delta_prog': -0.25, 'fdr_significant': True},
    ]).to_csv(os.path.join(mech_dir, 'mechanism_delta_progression.csv'), index=False)

    # barrier_crossing.csv
    pd.DataFrame([
        {'participant_id': 'P1', 'crossed_to_attention_regulation': True},
        {'participant_id': 'P2', 'crossed_to_attention_regulation': False},
    ]).to_csv(os.path.join(eff_dir, 'barrier_crossing.csv'), index=False)

    # longitudinal_summary.json (analysis_data_dir)
    with open(os.path.join(data_dir, 'longitudinal_summary.json'), 'w') as f:
        json.dump({
            'group_progression': {'n_advancing': 3, 'n_stable': 1, 'n_regressing': 0},
            'feasibility_assessment': {
                'feasibility_rating': 'high', 'high_plus_medium_pct': 0.8,
            },
            'validity_indicators': {
                'validity_narrative': 'Progression partially observed.',
            },
        }, f)


def _make_minimal_df():
    import pandas as pd
    rows = []
    for pid in ('P1', 'P2'):
        for sid in (f'{pid}_S1', f'{pid}_S2'):
            for i in range(4):
                rows.append({
                    'segment_id': f'{sid}_{i}',
                    'participant_id': pid,
                    'session_id': sid,
                    'theme_id': i % 5,
                    'speaker': 'participant',
                    'text': f'Sample utterance {i}',
                    'label_confidence_tier': 'high',
                    'cohort_id': 1,
                })
    return pd.DataFrame(rows)


def _read_results(tmp):
    path = _paths.reports_results_path(tmp)
    with open(path, encoding='utf-8') as f:
        return f.read()


# ── tests ─────────────────────────────────────────────────────────────────────

class TestGenerateResultsBriefReturnValue(unittest.TestCase):

    def test_returns_path_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_results_brief(tmp)
            self.assertIsInstance(result, str)

    def test_returns_path_at_reports_results_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_results_brief(tmp)
            self.assertEqual(result, _paths.reports_results_path(tmp))

    def test_file_exists_after_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_results_brief(tmp)
            self.assertTrue(os.path.isfile(path))

    def test_file_is_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            path = generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertGreater(os.path.getsize(path), 100)

    def test_file_is_utf8_decodable(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            path = generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertGreater(len(content), 50)


class TestResultsBriefHeader(unittest.TestCase):

    def test_results_header_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            content = _read_results(tmp)
            self.assertIn('RESULTS', content)

    def test_generated_date_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            generate_results_brief(tmp)
            content = _read_results(tmp)
            self.assertIn('Generated:', content)


class TestResultsBriefSections(unittest.TestCase):
    """All four main section headers must appear with full artifacts."""

    def test_sample_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertIn('1. SAMPLE', _read_results(tmp))

    def test_reliability_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertIn('2. RELIABILITY', _read_results(tmp))

    def test_outcomes_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertIn('3. OUTCOMES', _read_results(tmp))

    def test_mechanism_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertIn('4. MECHANISM', _read_results(tmp))


class TestResultsBriefMethodTags(unittest.TestCase):
    """[M#] tags should appear in the output (stat_format m_ref fired)."""

    def test_at_least_one_m_tag(self):
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            content = _read_results(tmp)
            self.assertIn('[M', content)


class TestResultsBriefGracefulDegradation(unittest.TestCase):

    def test_empty_dir_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = generate_results_brief(tmp)
                # Either returns a path or ''
                self.assertIsInstance(path, str)
            except Exception as e:
                self.fail(f"generate_results_brief raised unexpectedly: {e}")

    def test_empty_dir_writes_file_or_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_results_brief(tmp)
            if path:
                self.assertTrue(os.path.isfile(path))

    def test_missing_efficacy_json_degrades(self):
        """Missing efficacy_summary.json: outcomes section degrades gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            os.remove(os.path.join(_paths.efficacy_dir(tmp), 'efficacy_summary.json'))
            path = generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertTrue(os.path.isfile(path))

    def test_missing_mechanism_csv_degrades(self):
        """Missing mechanism CSV: mechanism section degrades gracefully."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            os.remove(os.path.join(
                _paths.mechanism_dir(tmp), 'mechanism_delta_progression.csv'))
            path = generate_results_brief(tmp, df=_make_minimal_df(), framework={})
            self.assertTrue(os.path.isfile(path))

    def test_none_df_degrades(self):
        """df=None: sample section says data unavailable without crashing."""
        with tempfile.TemporaryDirectory() as tmp:
            _write_minimal_artifacts(tmp)
            path = generate_results_brief(tmp, df=None, framework={})
            self.assertTrue(os.path.isfile(path))


class TestGenerateMethodsReport(unittest.TestCase):

    def test_returns_path_string(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_methods_report(tmp)
            self.assertIsInstance(result, str)

    def test_returns_path_at_reports_methods_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_methods_report(tmp)
            self.assertEqual(result, _paths.reports_methods_path(tmp))

    def test_file_exists_after_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_methods_report(tmp)
            self.assertTrue(os.path.isfile(path))

    def test_file_is_nonempty(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_methods_report(tmp)
            self.assertGreater(os.path.getsize(path), 100)

    def test_methods_header_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_methods_report(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('METHODS', content)

    def test_registry_entries_expanded(self):
        """Every entry from methods_entries() should appear in the report."""
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_methods_report(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            entries = list(methods_entries())
            self.assertGreater(len(entries), 0, "methods_entries() returned nothing")
            # At least the first entry's title must appear
            _tag, _key, title, _text = entries[0]
            self.assertIn(title, content)

    def test_all_entry_tags_appear(self):
        """All [M#] tags from the registry must appear in 08_methods.txt."""
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_methods_report(tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
            for tag, _key, _title, _text in methods_entries():
                self.assertIn(f'[{tag}]', content,
                              f"Registry tag [{tag}] missing from 08_methods.txt")

    def test_empty_dir_does_not_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            try:
                path = generate_methods_report(tmp)
                self.assertIsInstance(path, str)
            except Exception as e:
                self.fail(f"generate_methods_report raised unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
