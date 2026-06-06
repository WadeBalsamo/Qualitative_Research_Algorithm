"""
tests/unit/test_exemplars.py
-----------------------------
Unit tests for analysis/exemplars.py.

Covers:
  - select_prototypical_exemplars: top-confidence selection, per-stage grouping,
    empty df, word-count bounds, fallback relaxation
  - format_exemplar: dict keys, optional superposition fields
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.exemplars import select_prototypical_exemplars, format_exemplar


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _row(**kw):
    defaults = dict(
        segment_id='c1s1_0',
        participant_id='P01',
        session_id='c1s1',
        session_number=1,
        text='A sufficiently long segment text for testing purposes here.',
        word_count=9,
        label_confidence_tier='high',
        llm_run_consistency=3,
        llm_confidence_primary=0.85,
        llm_justification='Good exemplar.',
    )
    defaults.update(kw)
    return defaults


def _df(*rows):
    return pd.DataFrame(list(rows))


class TestSelectPrototypicalExemplarsEmpty(unittest.TestCase):
    def test_empty_df_returns_empty(self):
        df = pd.DataFrame(columns=[
            'segment_id', 'participant_id', 'session_id', 'session_number',
            'text', 'word_count', 'label_confidence_tier',
            'llm_run_consistency', 'llm_confidence_primary', 'llm_justification',
        ])
        result = select_prototypical_exemplars(df, n=3)
        self.assertEqual(len(result), 0)
        # Should return a DataFrame, not raise
        self.assertIsInstance(result, pd.DataFrame)


class TestSelectPrototypicalExemplarsTopConfidence(unittest.TestCase):
    """High-confidence rows should appear before lower-confidence ones."""

    def _make_df(self):
        rows = [
            _row(segment_id='s1', llm_confidence_primary=0.95, llm_run_consistency=3,
                 label_confidence_tier='high', word_count=20,
                 text=' '.join(['word'] * 20)),
            _row(segment_id='s2', llm_confidence_primary=0.50, llm_run_consistency=1,
                 label_confidence_tier='low', word_count=20,
                 text=' '.join(['word'] * 20)),
            _row(segment_id='s3', llm_confidence_primary=0.90, llm_run_consistency=3,
                 label_confidence_tier='high', word_count=20,
                 text=' '.join(['word'] * 20)),
        ]
        return pd.DataFrame(rows)

    def test_n_limits_result(self):
        df = self._make_df()
        result = select_prototypical_exemplars(df, n=2)
        self.assertLessEqual(len(result), 2)

    def test_high_confidence_preferred(self):
        df = self._make_df()
        result = select_prototypical_exemplars(df, n=1)
        # The top result should be the highest-consistency + highest-confidence
        self.assertEqual(result.iloc[0]['segment_id'], 's1')

    def test_two_high_confidence_returned(self):
        df = self._make_df()
        result = select_prototypical_exemplars(df, n=2)
        ids = set(result['segment_id'].tolist())
        self.assertIn('s1', ids)
        self.assertIn('s3', ids)


class TestSelectPrototypicalExemplarsWordCount(unittest.TestCase):
    """Word-count bounds filter out trivially short / overly long segments."""

    def test_short_segment_excluded(self):
        rows = [
            _row(segment_id='short', word_count=3, text='too short'),
            _row(segment_id='ok', word_count=30,
                 text=' '.join(['w'] * 30)),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=5, min_words=15, max_words=200)
        ids = result['segment_id'].tolist()
        self.assertNotIn('short', ids)
        self.assertIn('ok', ids)

    def test_long_segment_excluded(self):
        rows = [
            _row(segment_id='long', word_count=300,
                 text=' '.join(['w'] * 300)),
            _row(segment_id='ok', word_count=30,
                 text=' '.join(['w'] * 30)),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=5, min_words=15, max_words=200)
        ids = result['segment_id'].tolist()
        self.assertNotIn('long', ids)
        self.assertIn('ok', ids)


class TestSelectPrototypicalExemplarsFallback(unittest.TestCase):
    """When no high-confidence candidates exist, relax to any labeled segment."""

    def test_fallback_relaxes_confidence_filter(self):
        rows = [
            _row(segment_id='low', word_count=25,
                 text=' '.join(['w'] * 25),
                 label_confidence_tier='low',
                 llm_run_consistency=1,
                 llm_confidence_primary=0.4),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=3)
        self.assertGreater(len(result), 0)
        self.assertEqual(result.iloc[0]['segment_id'], 'low')

    def test_fallback_to_all_when_word_count_empty(self):
        """If no segment passes word-count bounds, return any available."""
        rows = [
            _row(segment_id='s1', word_count=3, text='tiny',
                 label_confidence_tier='low',
                 llm_run_consistency=1, llm_confidence_primary=0.3),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=3, min_words=15, max_words=200)
        # Should still return a non-empty df (ultimate fallback)
        self.assertGreater(len(result), 0)


class TestSelectPrototypicalExemplarsPerStageGrouping(unittest.TestCase):
    """Caller can filter df by stage before calling; verify per-stage semantics."""

    def _make_multi_stage_df(self):
        rows = []
        for stage in range(5):
            for i in range(4):
                conf = 0.9 - 0.1 * i
                rows.append(_row(
                    segment_id=f's{stage}_{i}',
                    word_count=25,
                    text=' '.join(['w'] * 25),
                    llm_confidence_primary=conf,
                    llm_run_consistency=3 if i < 2 else 1,
                    label_confidence_tier='high' if i < 2 else 'low',
                    final_label=stage,
                ))
        return pd.DataFrame(rows)

    def test_per_stage_selection(self):
        df = self._make_multi_stage_df()
        for stage in range(5):
            stage_df = df[df['final_label'] == stage]
            result = select_prototypical_exemplars(stage_df, n=2)
            self.assertLessEqual(len(result), 2)
            # All returned segments should belong to the correct stage
            self.assertTrue((result['final_label'] == stage).all())


class TestSelectPrototypicalExemplarsOrConsistency(unittest.TestCase):
    """OR logic: high tier OR (consistency==3 AND confidence>0.7)."""

    def test_high_tier_selected_regardless_of_consistency(self):
        rows = [
            _row(segment_id='hi_tier', word_count=20, text=' '.join(['w']*20),
                 label_confidence_tier='high', llm_run_consistency=1,
                 llm_confidence_primary=0.4),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=3)
        self.assertIn('hi_tier', result['segment_id'].tolist())

    def test_consistent_above_threshold_selected(self):
        rows = [
            _row(segment_id='cons', word_count=20, text=' '.join(['w']*20),
                 label_confidence_tier='low', llm_run_consistency=3,
                 llm_confidence_primary=0.8),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=3)
        self.assertIn('cons', result['segment_id'].tolist())

    def test_low_confidence_low_tier_not_preferred(self):
        """Low tier + low consistency should only appear via fallback."""
        rows = [
            _row(segment_id='good', word_count=20, text=' '.join(['w']*20),
                 label_confidence_tier='high', llm_run_consistency=3,
                 llm_confidence_primary=0.9),
            _row(segment_id='bad', word_count=20, text=' '.join(['w']*20),
                 label_confidence_tier='low', llm_run_consistency=1,
                 llm_confidence_primary=0.3),
        ]
        df = pd.DataFrame(rows)
        result = select_prototypical_exemplars(df, n=1)
        self.assertEqual(result.iloc[0]['segment_id'], 'good')


class TestFormatExemplar(unittest.TestCase):
    """format_exemplar converts a row to the standard dict."""

    def _series(self, **kw):
        data = dict(
            segment_id='c1s1_0',
            participant_id='P01',
            session_id='c1s1',
            session_number=1,
            text='The patient described pain as a constant companion.',
            llm_confidence_primary=0.9,
            llm_run_consistency=3,
            llm_justification='Clearly reappraisal.',
        )
        data.update(kw)
        return pd.Series(data)

    def test_basic_keys_present(self):
        row = self._series()
        result = format_exemplar(row)
        for key in ('segment_id', 'participant_id', 'session_id', 'session_number',
                    'text', 'confidence', 'consistency', 'justification'):
            self.assertIn(key, result)

    def test_confidence_value(self):
        row = self._series(llm_confidence_primary=0.87)
        result = format_exemplar(row)
        self.assertAlmostEqual(result['confidence'], 0.87, places=3)

    def test_consistency_value(self):
        row = self._series(llm_run_consistency=2)
        result = format_exemplar(row)
        self.assertEqual(result['consistency'], 2)

    def test_nan_confidence_returns_none(self):
        row = self._series(llm_confidence_primary=np.nan)
        result = format_exemplar(row)
        self.assertIsNone(result['confidence'])

    def test_nan_consistency_returns_none(self):
        row = self._series(llm_run_consistency=np.nan)
        result = format_exemplar(row)
        self.assertIsNone(result['consistency'])

    def test_superposition_fields_added_when_present(self):
        row = self._series(
            mixture=[0.1, 0.2, 0.4, 0.2, 0.1],
            mixture_entropy=0.75,
            is_liminal=True,
        )
        result = format_exemplar(row)
        self.assertIn('mixture', result)
        self.assertIn('mixture_entropy', result)
        self.assertIn('is_liminal', result)
        self.assertAlmostEqual(result['mixture_entropy'], 0.75, places=3)

    def test_superposition_fields_absent_when_no_mixture(self):
        row = self._series()
        result = format_exemplar(row)
        self.assertNotIn('mixture', result)

    def test_empty_justification(self):
        row = self._series(llm_justification=None)
        result = format_exemplar(row)
        self.assertEqual(result['justification'], '')


if __name__ == '__main__':
    unittest.main()
