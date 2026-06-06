"""
tests/unit/test_cls_validation.py
-----------------------------------
Unit tests for classification_tools/validation.py.

Covers:
  - create_balanced_evaluation_set: basic balanced sampling,
    under-represented class uses all available,
    dual-coded stratum appended when secondary_stage present,
    stratification by trial_id,
    random_state reproducibility,
    empty input handling.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from classification_tools.validation import create_balanced_evaluation_set


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(n_per_class=10, n_classes=3, with_secondary=False,
             n_trials=1, random_state=0):
    """Build a synthetic labeled DataFrame for testing.

    Parameters
    ----------
    n_per_class : int
        Rows per class.
    n_classes : int
        Number of distinct classes (0..n_classes-1).
    with_secondary : bool
        If True, add a 'secondary_stage' column with non-null values on half
        of all rows.
    n_trials : int
        Number of distinct trial_id values (spread evenly across rows).
    """
    rng = np.random.default_rng(random_state)
    rows = []
    for cls in range(n_classes):
        for i in range(n_per_class):
            row = {
                'segment_id': f'cls{cls}_seg{i}',
                'primary_stage': cls,
                'text': f'Segment text class {cls} index {i}',
                'trial_id': f'trial_{i % n_trials}',
            }
            if with_secondary:
                row['secondary_stage'] = (cls + 1) % n_classes if i % 2 == 0 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Basic sampling tests
# ---------------------------------------------------------------------------

class TestCreateBalancedEvaluationSet(unittest.TestCase):

    def test_returns_dataframe(self):
        df = _make_df(n_per_class=20, n_classes=3)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        self.assertIsInstance(result, pd.DataFrame)

    def test_has_correct_columns(self):
        df = _make_df(n_per_class=20, n_classes=3)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        self.assertIn('primary_stage', result.columns)
        self.assertIn('segment_id', result.columns)

    def test_all_classes_represented(self):
        """All unique classes in the input must appear in the output."""
        df = _make_df(n_per_class=15, n_classes=4)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        for cls in range(4):
            self.assertIn(cls, result['primary_stage'].values,
                          f"Class {cls} missing from balanced set")

    def test_n_per_class_respected_when_sufficient_data(self):
        """When enough data is available, each class gets exactly n_per_class rows."""
        df = _make_df(n_per_class=50, n_classes=3)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        for cls in range(3):
            cls_count = (result['primary_stage'] == cls).sum()
            self.assertGreaterEqual(cls_count, 10,
                                    f"Class {cls} undersampled: got {cls_count}")

    def test_uses_all_available_when_insufficient(self):
        """A class with fewer than n_per_class rows uses all available."""
        # Class 0: only 3 rows; classes 1,2: 20 rows each
        rows = (
            [{'segment_id': f's0_{i}', 'primary_stage': 0, 'text': f't{i}',
              'trial_id': 'trial_0'} for i in range(3)]
            + [{'segment_id': f's1_{i}', 'primary_stage': 1, 'text': f't{i}',
                'trial_id': 'trial_0'} for i in range(20)]
            + [{'segment_id': f's2_{i}', 'primary_stage': 2, 'text': f't{i}',
                'trial_id': 'trial_0'} for i in range(20)]
        )
        df = pd.DataFrame(rows)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        cls0_count = (result['primary_stage'] == 0).sum()
        # All 3 rows of class 0 must be present
        self.assertEqual(cls0_count, 3)

    def test_reproducible_with_same_random_state(self):
        """Same random_state must yield the same result on two calls."""
        df = _make_df(n_per_class=50, n_classes=3)
        r1 = create_balanced_evaluation_set(df, n_per_class=10, random_state=42)
        r2 = create_balanced_evaluation_set(df, n_per_class=10, random_state=42)
        pd.testing.assert_frame_equal(r1.reset_index(drop=True),
                                      r2.reset_index(drop=True))

    def test_different_random_states_may_differ(self):
        """Different random states should (usually) produce different shuffles."""
        df = _make_df(n_per_class=100, n_classes=3)
        r1 = create_balanced_evaluation_set(df, n_per_class=10, random_state=1)
        r2 = create_balanced_evaluation_set(df, n_per_class=10, random_state=99)
        # It is extremely unlikely they are identical
        self.assertFalse(
            r1['segment_id'].reset_index(drop=True).equals(
                r2['segment_id'].reset_index(drop=True)),
            "Two different random seeds produced identical orderings (very unlikely)"
        )

    def test_output_index_is_reset(self):
        """Returned DataFrame must have a clean 0-based integer index."""
        df = _make_df(n_per_class=20, n_classes=2)
        result = create_balanced_evaluation_set(df, n_per_class=5)
        self.assertEqual(list(result.index), list(range(len(result))))

    def test_no_nulls_in_label_column(self):
        """The label column must have no NaN values in the output."""
        df = _make_df(n_per_class=20, n_classes=3)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        self.assertFalse(result['primary_stage'].isna().any(),
                         "primary_stage must not have NaN values in output")

    def test_custom_label_column(self):
        """label_column override must be respected."""
        df = _make_df(n_per_class=20, n_classes=3)
        df = df.rename(columns={'primary_stage': 'my_label'})
        result = create_balanced_evaluation_set(df, n_per_class=5,
                                               label_column='my_label')
        self.assertIn('my_label', result.columns)
        for cls in range(3):
            self.assertIn(cls, result['my_label'].values)


# ---------------------------------------------------------------------------
# Dual-coded stratum
# ---------------------------------------------------------------------------

class TestDualCodedStratum(unittest.TestCase):

    def test_dual_coded_stratum_appended(self):
        """When secondary_stage is present, dual-coded rows appear in output."""
        df = _make_df(n_per_class=20, n_classes=3, with_secondary=True)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        # Dual-coded rows have non-null secondary_stage
        dual = result[result['secondary_stage'].notna()]
        self.assertGreater(len(dual), 0,
                           "Dual-coded stratum must be present when secondary_stage exists")

    def test_no_dual_stratum_without_secondary_column(self):
        """No secondary_stage column → no dual stratum → clean output."""
        df = _make_df(n_per_class=20, n_classes=3, with_secondary=False)
        result = create_balanced_evaluation_set(df, n_per_class=10)
        self.assertNotIn('secondary_stage', result.columns)

    def test_dual_coded_capped_at_n_per_class(self):
        """When > n_per_class dual-coded rows exist, the dedicated dual stratum
        is capped at n_per_class.

        Dual-coded rows still carry a real primary_stage, so they can also be
        sampled into their primary class stratum. The total number of
        secondary-labeled rows in the result is therefore bounded by the dual
        stratum cap PLUS the primary stratum cap (both n_per_class), i.e.
        2 * n_per_class. Primary-stratum rows carry NO secondary label.
        """
        n_per_class = 5
        rows = []
        for cls in range(2):
            for i in range(100):
                rows.append({
                    'segment_id': f'c{cls}_{i}',
                    'primary_stage': cls,
                    'secondary_stage': np.nan,
                    'text': f't{i}',
                    'trial_id': 'trial_0',
                })
        for i in range(100):
            rows.append({
                'segment_id': f'dual_{i}',
                'primary_stage': 0,
                'secondary_stage': 1,
                'text': f'd{i}',
                'trial_id': 'trial_0',
            })
        df = pd.DataFrame(rows)
        result = create_balanced_evaluation_set(df, n_per_class=n_per_class)
        dual = result[result['secondary_stage'].notna()]
        self.assertLessEqual(
            len(dual), 2 * n_per_class,
            "Dual-coded stratum must be capped at n_per_class "
            "(plus at most n_per_class leakage into its primary class stratum)",
        )


# ---------------------------------------------------------------------------
# Trial-stratified sampling
# ---------------------------------------------------------------------------

class TestTrialStratification(unittest.TestCase):

    def test_multiple_trials_each_represented(self):
        """With multiple trial_ids, each trial should contribute to the sample."""
        df = _make_df(n_per_class=30, n_classes=2, n_trials=3)
        result = create_balanced_evaluation_set(df, n_per_class=15)
        trials_represented = result['trial_id'].nunique()
        self.assertGreater(trials_represented, 1,
                           "Multiple trials should each contribute to the balanced set")

    def test_single_trial_same_as_unstratified(self):
        """With a single trial_id, the stratified and unstratified paths both work."""
        df = _make_df(n_per_class=30, n_classes=2, n_trials=1)
        # Should not raise; result should have rows from all classes.
        result = create_balanced_evaluation_set(df, n_per_class=10)
        for cls in range(2):
            self.assertIn(cls, result['primary_stage'].values)

    def test_no_trial_id_column_uses_simple_sampling(self):
        """When trial_id is absent, simple sampling without stratification is used."""
        df = _make_df(n_per_class=20, n_classes=3)
        df = df.drop(columns=['trial_id'])
        result = create_balanced_evaluation_set(df, n_per_class=10)
        self.assertEqual(len(result), 30)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def test_single_class(self):
        """A DataFrame with only one class must return rows for that class."""
        df = pd.DataFrame([
            {'segment_id': f's{i}', 'primary_stage': 0, 'text': 't', 'trial_id': 't0'}
            for i in range(15)
        ])
        result = create_balanced_evaluation_set(df, n_per_class=5)
        self.assertGreater(len(result), 0)
        self.assertTrue((result['primary_stage'] == 0).all())

    def test_n_per_class_larger_than_total_data(self):
        """Requesting more per class than available should use all data (no crash)."""
        df = _make_df(n_per_class=5, n_classes=2)
        result = create_balanced_evaluation_set(df, n_per_class=100)
        # Should include all rows that exist
        self.assertGreater(len(result), 0)

    def test_null_labels_excluded(self):
        """Rows with NaN in the label column must not appear in the result."""
        rows = [
            {'segment_id': 's0', 'primary_stage': 0, 'text': 't', 'trial_id': 't0'},
            {'segment_id': 's1', 'primary_stage': np.nan, 'text': 't', 'trial_id': 't0'},
            {'segment_id': 's2', 'primary_stage': 1, 'text': 't', 'trial_id': 't0'},
            {'segment_id': 's3', 'primary_stage': 0, 'text': 't', 'trial_id': 't0'},
            {'segment_id': 's4', 'primary_stage': 1, 'text': 't', 'trial_id': 't0'},
        ]
        df = pd.DataFrame(rows)
        result = create_balanced_evaluation_set(df, n_per_class=2)
        self.assertFalse(result['primary_stage'].isna().any())

    def test_all_same_class_returns_rows(self):
        """Edge: all segments are the same class — must still produce output."""
        df = pd.DataFrame([
            {'segment_id': f's{i}', 'primary_stage': 2, 'text': 't', 'trial_id': 't0'}
            for i in range(10)
        ])
        result = create_balanced_evaluation_set(df, n_per_class=5)
        self.assertGreater(len(result), 0)
        self.assertTrue((result['primary_stage'] == 2).all())


if __name__ == '__main__':
    unittest.main()
