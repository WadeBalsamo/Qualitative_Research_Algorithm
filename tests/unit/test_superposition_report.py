"""
tests/unit/test_superposition_report.py
-----------------------------------------
Tests for analysis/reports/superposition_report.py.

generate_superposition_report(df, framework, output_dir) -> str

Covers:
  - Returns '' when 'mixture' column absent or df empty
  - Returns '' when df is empty (len == 0)
  - Writes 06_reports/02_mechanism/superposition.txt with valid df
  - All five section headers present
  - Corpus stats (n segments, liminal pct, entropy, active stages)
  - Co-occurrence matrix rendered (stage names in header)
  - Liminality per session series rendered
  - Liminality trend note when ≥2 sessions
  - Avoidance↔Attention-Regulation cusp section: fallback when no CSV
  - Most-liminal exemplars section: at least one quote block
  - Dominant mixture source rendered from df
"""

import os
import sys
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

import numpy as np
import pandas as pd

from process import output_paths as _paths
from analysis.reports.superposition_report import generate_superposition_report


# ── shared fixtures ───────────────────────────────────────────────────────────

FRAMEWORK = {
    0: {'short_name': 'Vigilance'},
    1: {'short_name': 'Avoidance'},
    2: {'short_name': 'AttnReg'},
    3: {'short_name': 'Metacog'},
    4: {'short_name': 'Reappraisal'},
}


def _make_df(n_sessions=2, n_per_session=4, seed=42):
    """Participant-only df with 'mixture' + all superposition columns."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(n_sessions):
        sid = f'c1s{s + 1}'
        for j in range(n_per_session):
            raw = rng.dirichlet([2.0, 1.5, 1.0, 0.7, 0.5])
            ent = -float(np.sum(raw * np.log(raw + 1e-12))) / np.log(5)
            order = np.argsort(raw)[::-1]
            rows.append({
                'segment_id': f'{sid}_{j}',
                'session_id': sid,
                'speaker': 'participant',
                'text': f'Some longer participant text for session {s + 1} segment {j} about pain and mindfulness',
                'word_count': 20,
                'mixture': raw.round(4).tolist(),
                'mixture_entropy': round(ent, 4),
                'is_liminal': bool(ent >= 0.5),
                'n_active_stages': int((raw >= 0.15).sum()),
                'max_stage': int(order[0]),
                'second_stage': int(order[1]),
                'mixture_source': 'ballots',
                'final_label': int(order[0]),
                'progression_coord': round(float(np.dot(raw, [0, 1, 2, 3, 4])), 4),
            })
    return pd.DataFrame(rows)


# ── tests ─────────────────────────────────────────────────────────────────────

class TestSuperpositionEarlyReturn(unittest.TestCase):
    """Returns '' when inputs are unusable."""

    def test_no_mixture_column(self):
        df = _make_df()
        df = df.drop(columns=['mixture'])
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_superposition_report(df, FRAMEWORK, tmp)
        self.assertEqual(result, '')

    def test_empty_df(self):
        df = _make_df().iloc[0:0]  # 0 rows but keeps columns including 'mixture'
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_superposition_report(df, FRAMEWORK, tmp)
        self.assertEqual(result, '')


class TestSuperpositionWritesFile(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.df = _make_df(n_sessions=3, n_per_session=4)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_nonempty_path(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        self.assertTrue(path)
        self.assertTrue(os.path.isfile(path))

    def test_path_in_mechanism_dir(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        self.assertIn('02_mechanism', path)
        self.assertTrue(path.endswith('superposition.txt'))

    def test_five_section_headers(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('VAAMR SUPERPOSITION REPORT', content)
        self.assertIn('1. CORPUS SUPERPOSITION', content)
        self.assertIn('2. STAGE CO-OCCURRENCE (CUSP) MATRIX', content)
        self.assertIn('3. LIMINALITY ACROSS THE PROGRAM', content)
        self.assertIn('4. AVOIDANCE', content)
        self.assertIn('5. MOST-LIMINAL EXEMPLARS', content)

    def test_corpus_stats_rendered(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        # Should have counts and percentage signs
        self.assertIn('Segments:', content)
        self.assertIn('Liminal', content)
        self.assertIn('Mean mixture entropy', content)
        self.assertIn('Mean active stages', content)

    def test_stage_names_in_cooccurrence_header(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        for name in ('Vigilance', 'Avoidance', 'AttnReg'):
            self.assertIn(name, content)

    def test_liminality_series_has_session_entries(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        # Each session id should appear in the liminality section
        for sid in ('c1s1', 'c1s2', 'c1s3'):
            self.assertIn(sid, content)

    def test_liminality_trend_note_present(self):
        """With ≥2 sessions the trend sentence ('fell', 'rose', or 'held steady') appears."""
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        has_trend_word = ('fell' in content or 'rose' in content or 'held steady' in content)
        self.assertTrue(has_trend_word)

    def test_avoidance_cusp_fallback_without_csv(self):
        """Without avoidance_cusp_density_by_session.csv the section shows fallback text."""
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('run with mechanism analysis enabled', content)

    def test_avoidance_cusp_with_csv(self):
        """When the CSV exists the bars render."""
        mech_dir = _paths.mechanism_dir(self.tmp)
        os.makedirs(mech_dir, exist_ok=True)
        cusp_path = os.path.join(mech_dir, 'avoidance_cusp_density_by_session.csv')
        pd.DataFrame([
            {'session_id': 'c1s1', 'cusp_density': 0.25},
            {'session_id': 'c1s2', 'cusp_density': 0.40},
        ]).to_csv(cusp_path, index=False)
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('25.0%', content)

    def test_exemplars_section_has_quotes(self):
        """Section 5 should have at least one quoted block."""
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        # _wrap_quote wraps text in '"'
        self.assertIn('"', content)

    def test_mixture_source_in_report(self):
        path = generate_superposition_report(self.df, FRAMEWORK, self.tmp)
        with open(path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('ballots', content)


class TestSuperpositionSingleSession(unittest.TestCase):
    """With one session, no trend note and no Strongest cusp (only diagonal possible)."""

    def test_single_session_no_trend(self):
        df = _make_df(n_sessions=1, n_per_session=3)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_superposition_report(df, FRAMEWORK, tmp)
            with open(path, encoding='utf-8') as f:
                content = f.read()
        # No trend note with a single session (needs ≥2 series entries)
        for word in ('fell', 'rose', 'held steady'):
            self.assertNotIn(word, content)


class TestSuperpositionMinimalMixture(unittest.TestCase):
    """A pure-stage (non-liminal) df should still render correctly."""

    def test_all_pure_segments(self):
        rows = []
        for i in range(4):
            mix = [0.0, 0.0, 0.0, 0.0, 0.0]
            mix[i % 5] = 1.0
            rows.append({
                'segment_id': f's{i}',
                'session_id': 'c1s1',
                'speaker': 'participant',
                'text': 'Pure stage text utterance here',
                'word_count': 5,
                'mixture': mix,
                'mixture_entropy': 0.0,
                'is_liminal': False,
                'n_active_stages': 1,
                'max_stage': i % 5,
                'second_stage': (i + 1) % 5,
                'mixture_source': 'secondary',
                'final_label': i % 5,
                'progression_coord': float(i % 5),
            })
        df = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as tmp:
            path = generate_superposition_report(df, FRAMEWORK, tmp)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding='utf-8') as f:
                content = f.read()
            self.assertIn('Liminal', content)
            self.assertIn('0.0%', content)


if __name__ == '__main__':
    unittest.main()
