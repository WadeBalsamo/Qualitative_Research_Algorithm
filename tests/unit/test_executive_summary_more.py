"""
tests/unit/test_executive_summary_more.py
-----------------------------------------
The executive_summary module was replaced by analysis/reports/results_brief.py
(generate_results_brief → 06_reports/00_RESULTS.txt) and
analysis/reports/methods_report.py (generate_methods_report → 06_reports/08_methods.txt).

All coverage for those generators now lives in:
    tests/unit/test_results_brief.py

This file is kept as an empty placeholder so git history and any
external references to the filename remain valid.
"""

import unittest


class TestExecutiveSummaryDeprecated(unittest.TestCase):
    """Coverage moved to test_results_brief.py (see module docstring)."""

    def test_placeholder(self):
        """Placeholder: real tests are in test_results_brief.py."""
        pass


if __name__ == '__main__':
    unittest.main()
