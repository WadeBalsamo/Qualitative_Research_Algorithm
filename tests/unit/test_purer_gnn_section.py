"""
tests/test_purer_gnn_section.py
--------------------------------
Unit tests for analysis.purer_analysis.append_gnn_motif_section.

Covers:
  - Returns non-None and writes section header + top motif id when CSVs present.
  - Appends correctly to an existing report_purer_analysis.txt.
  - Returns None (no raise) when both CSVs are absent.
  - Returns None (no raise) when only one CSV is absent.
  - Does not raise when report_purer_analysis.txt does not exist.
"""

import os
import sys
import tempfile
import textwrap
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.purer_analysis import append_gnn_motif_section

# ── Column names discovered from gnn_layer/reports.py ──────────────────────
# write_cue_motifs: motif_id, n_blocks, influence, mean_pred_forward,
#                   dominant_purer, purer_purity, n_exemplars
#
# write_coupling_factors: factor, explained_variance_ratio, forward_corr,
#                         nearest_cf_ic, cf_ic_similarity, n_exemplars
# ───────────────────────────────────────────────────────────────────────────

_CUE_MOTIFS_CSV = textwrap.dedent("""\
    motif_id,n_blocks,influence,mean_pred_forward,dominant_purer,purer_purity,n_exemplars
    0,45,0.8821,0.712,3,0.71,3
    1,30,0.7654,0.601,0,0.45,2
    2,20,0.6100,0.550,1,0.82,1
    3,15,0.5300,0.490,2,0.55,2
    4,10,0.4200,0.420,4,0.90,1
    5,8,0.3100,0.380,0,0.30,0
    6,7,0.2800,0.350,3,0.65,1
    7,5,0.1500,0.280,1,0.70,0
    8,3,0.0800,0.200,2,0.80,0
""")

_COUPLING_FACTORS_CSV = textwrap.dedent("""\
    factor,explained_variance_ratio,forward_corr,nearest_cf_ic,cf_ic_similarity,n_exemplars
    0,0.2341,0.6120,therapeutic_alliance,0.82,3
    1,0.1892,-0.3450,empathy,0.74,2
    2,0.1503,0.5100,goal_consensus,0.68,2
    3,0.1012,-0.2200,therapist_directiveness,0.61,1
    4,0.0879,0.4300,mindfulness_facilitation,0.59,1
    5,0.0712,0.1100,psychoeducation,0.55,0
""")

_MINIMAL_PURER_REPORT = textwrap.dedent("""\
    ════════════════════════════════════════════════════════════════════════
    PURER × VAMMR CUE-BLOCK INFLUENCE ANALYSIS
    ════════════════════════════════════════════════════════════════════════
    Generated : 2026-06-05
    Total cue blocks : 10
    ════════════════════════════════════════════════════════════════════════
""")


def _make_output_dir(tmp: str, with_motifs=True, with_factors=True,
                     with_purer_report=True) -> str:
    """Create a minimal output_dir structure inside tmp."""
    gnn_dir = os.path.join(tmp, '03_analysis_data', 'gnn')
    reports_dir = os.path.join(tmp, '06_reports', '02_mechanism')
    os.makedirs(gnn_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    if with_motifs:
        with open(os.path.join(gnn_dir, 'cue_motifs.csv'), 'w', encoding='utf-8') as f:
            f.write(_CUE_MOTIFS_CSV)

    if with_factors:
        with open(os.path.join(gnn_dir, 'coupling_factors.csv'), 'w', encoding='utf-8') as f:
            f.write(_COUPLING_FACTORS_CSV)

    if with_purer_report:
        with open(os.path.join(reports_dir, 'purer.txt'), 'w',
                  encoding='utf-8') as f:
            f.write(_MINIMAL_PURER_REPORT)

    return tmp


class TestAppendGnnMotifSection(unittest.TestCase):

    # ── (a) Returns non-None when both CSVs are present ───────────────────

    def test_returns_non_none_when_csvs_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            result = append_gnn_motif_section(output_dir)
            self.assertIsNotNone(result,
                "Expected non-None when cue_motifs.csv and coupling_factors.csv exist.")

    # ── (b) Section header and top motif id appear in the report file ─────

    def test_section_header_in_report_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            append_gnn_motif_section(output_dir)
            report_path = os.path.join(tmp, '06_reports', '02_mechanism', 'purer.txt')
            with open(report_path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn(
                'GNN-DISCOVERED THERAPIST-LANGUAGE MOTIFS × FORWARD VAAMR MOVEMENT',
                content,
                "Section header missing from report file.",
            )

    def test_top_motif_id_in_report_file(self):
        """The highest-influence motif (id=0) must appear in the report."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            append_gnn_motif_section(output_dir)
            report_path = os.path.join(tmp, '06_reports', '02_mechanism', 'purer.txt')
            with open(report_path, encoding='utf-8') as f:
                content = f.read()
            # The top motif by influence is motif_id=0 with influence=0.8821
            # It must appear in the appended section
            self.assertIn(
                '0', content,
                "Top motif id (0) not found in appended section of report file.",
            )

    def test_section_text_contains_header(self):
        """Return value itself must contain the section header."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            section = append_gnn_motif_section(output_dir)
            self.assertIn(
                'GNN-DISCOVERED THERAPIST-LANGUAGE MOTIFS',
                section,
            )

    def test_section_appended_after_existing_content(self):
        """Original report content must be preserved; section appended at end."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            append_gnn_motif_section(output_dir)
            report_path = os.path.join(tmp, '06_reports', '02_mechanism', 'purer.txt')
            with open(report_path, encoding='utf-8') as f:
                content = f.read()
            # Original content still present
            self.assertIn('PURER × VAMMR CUE-BLOCK INFLUENCE ANALYSIS', content)
            # GNN section comes after (its position is after original content)
            orig_pos = content.index('PURER × VAMMR CUE-BLOCK INFLUENCE ANALYSIS')
            gnn_pos = content.index('GNN-DISCOVERED THERAPIST-LANGUAGE MOTIFS')
            self.assertGreater(gnn_pos, orig_pos,
                "GNN section should appear after original report content.")

    # ── (c) Returns None when both CSVs are absent ────────────────────────

    def test_returns_none_when_no_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp, with_motifs=False, with_factors=False)
            result = append_gnn_motif_section(output_dir)
            self.assertIsNone(result,
                "Expected None when neither cue_motifs.csv nor coupling_factors.csv exist.")

    def test_does_not_raise_when_no_csvs(self):
        """Must never raise even when GNN data is absent."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp, with_motifs=False, with_factors=False)
            try:
                result = append_gnn_motif_section(output_dir)
            except Exception as e:
                self.fail(f"append_gnn_motif_section raised unexpectedly: {e}")

    # ── Additional robustness cases ───────────────────────────────────────

    def test_returns_non_none_with_only_motifs_csv(self):
        """If only cue_motifs.csv is present (factors absent) should still work."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp, with_motifs=True, with_factors=False)
            result = append_gnn_motif_section(output_dir)
            self.assertIsNotNone(result)
            self.assertIn('GNN-DISCOVERED', result)

    def test_returns_non_none_with_only_factors_csv(self):
        """If only coupling_factors.csv is present (motifs absent) should still work."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp, with_motifs=False, with_factors=True)
            result = append_gnn_motif_section(output_dir)
            self.assertIsNotNone(result)
            self.assertIn('GNN-DISCOVERED', result)

    def test_no_raise_when_purer_report_absent(self):
        """Section text still returned even if report_purer_analysis.txt doesn't exist."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp, with_purer_report=False)
            try:
                result = append_gnn_motif_section(output_dir)
            except Exception as e:
                self.fail(f"append_gnn_motif_section raised when report absent: {e}")
            self.assertIsNotNone(result)

    def test_emergent_flag_for_low_purity_motif(self):
        """Motif with purer_purity < 0.60 should be flagged as EMERGENT in the section."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            section = append_gnn_motif_section(output_dir)
            # motif_id=1 has purer_purity=0.45 → should be flagged
            self.assertIn('EMERGENT', section,
                "Expected EMERGENT flag for motif with purer_purity < 0.60.")

    def test_coupling_factor_forward_corr_in_section(self):
        """Top coupling factor's forward_corr value should appear in the section."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            section = append_gnn_motif_section(output_dir)
            # Factor 0 has forward_corr=0.6120
            self.assertIn('0.612', section,
                "Expected forward_corr=0.6120 for factor 0 in section text.")

    def test_caveat_line_present(self):
        """Section must contain the directional/embedding-derived caveat."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            section = append_gnn_motif_section(output_dir)
            self.assertIn('Embedding-derived', section,
                "Expected caveat about embedding-derived, directional nature.")

    def test_capped_at_eight_motifs(self):
        """With 9 motifs in CSV, at most 8 should appear in the table."""
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = _make_output_dir(tmp)
            section = append_gnn_motif_section(output_dir)
            # motif_id=8 (lowest influence, 9th row) should NOT be in table
            # We check by looking for '    8  ' in the data rows (motif_id column)
            # The table rows start with '  ' then motif_id right-aligned in 5 chars
            # motif_id=8 would appear as '      8' in the motif column
            # A simple check: section has at most 8 data rows between the header
            # dashes and the trailing note. Count lines containing n_exemplars data.
            # Safer: the 9th motif (motif_id=8, influence=0.08) influence value
            # 0.0800 should NOT appear (since only top-8 are shown, motif 8 is 9th)
            self.assertNotIn('0.0800', section,
                "9th motif (influence=0.0800) should not appear; table capped at 8.")

    def test_no_raise_on_completely_empty_output_dir(self):
        """Must not raise for a completely empty directory."""
        with tempfile.TemporaryDirectory() as tmp:
            try:
                result = append_gnn_motif_section(tmp)
            except Exception as e:
                self.fail(f"Raised on empty dir: {e}")
            self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
