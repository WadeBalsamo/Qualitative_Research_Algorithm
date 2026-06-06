"""
tests/unit/test_loader.py
-------------------------
Unit tests for analysis/loader.py.

Builds minimal master_segments.csv fixtures in a tmpdir and verifies:
  - load_segments() returns a DataFrame with the expected columns/dtypes
  - find_master_csv() finds the CSV in the new canonical location
  - speaker_filter and require_labeled work correctly
  - list-column parsing (codebook_labels_ensemble)
  - cohort_id derivation from session_id
  - sort_session_ids canonical ordering
  - missing-file and empty-file handling
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from analysis.loader import (
    load_segments,
    find_master_csv,
    sort_session_ids,
    _parse_list_column,
    _derive_cohort_id,
)
from process import output_paths as _paths


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _master_csv_path(output_dir: str) -> str:
    d = _paths.master_segments_dir(output_dir)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, 'master_segments.csv')


def _make_minimal_csv(output_dir: str, rows=None) -> str:
    """Write a minimal master_segments.csv with sane defaults."""
    if rows is None:
        rows = [
            dict(
                segment_id='c1s1_0', participant_id='P01',
                session_id='c1s1', session_number=1, cohort_id=1,
                session_variant='', segment_index=0,
                speaker='participant',
                text='This is a test segment with some words.',
                word_count=8,
                start_time_ms=1000, end_time_ms=3000,
                primary_stage=2, final_label=2,
                secondary_stage=3,
                llm_confidence_primary=0.85,
                llm_confidence_secondary=0.4,
                llm_run_consistency=3,
                label_confidence_tier='high',
                llm_justification='Clearly AR.',
                codebook_labels_ensemble="['body_awareness', 'somatic']",
                codebook_labels_embedding="['body_awareness']",
                codebook_labels_llm="['body_awareness']",
                codebook_disagreements="[]",
                in_human_coded_subset=False,
            ),
            dict(
                segment_id='c1s1_1', participant_id='P01',
                session_id='c1s1', session_number=1, cohort_id=1,
                session_variant='', segment_index=1,
                speaker='therapist',
                text='That is interesting.',
                word_count=3,
                start_time_ms=3100, end_time_ms=4000,
                primary_stage=float('nan'), final_label=float('nan'),
                secondary_stage=float('nan'),
                llm_confidence_primary=float('nan'),
                llm_confidence_secondary=float('nan'),
                llm_run_consistency=float('nan'),
                label_confidence_tier='',
                llm_justification='',
                codebook_labels_ensemble="[]",
                codebook_labels_embedding="[]",
                codebook_labels_llm="[]",
                codebook_disagreements="[]",
                in_human_coded_subset=False,
            ),
        ]
    path = _master_csv_path(output_dir)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestFindMasterCsv(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_finds_in_canonical_location(self):
        path = _make_minimal_csv(self.tmp)
        found = find_master_csv(self.tmp)
        self.assertEqual(os.path.realpath(found), os.path.realpath(path))

    def test_finds_legacy_root_csv(self):
        """Legacy: master_segments.csv at output_dir root."""
        legacy = os.path.join(self.tmp, 'master_segments.csv')
        pd.DataFrame([]).to_csv(legacy, index=False)
        found = find_master_csv(self.tmp)
        self.assertEqual(os.path.realpath(found), os.path.realpath(legacy))

    def test_raises_when_missing(self):
        empty_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                find_master_csv(empty_dir)
        finally:
            shutil.rmtree(empty_dir, ignore_errors=True)


class TestLoadSegmentsColumns(unittest.TestCase):
    """load_segments() returns expected columns and dtypes."""

    REQUIRED_COLUMNS = [
        'segment_id', 'participant_id', 'session_id', 'session_number',
        'cohort_id', 'session_variant', 'segment_index', 'text',
        'word_count', 'primary_stage', 'final_label',
        'llm_confidence_primary', 'llm_confidence_secondary',
        'llm_run_consistency', 'secondary_stage',
        'label_confidence_tier', 'codebook_labels_ensemble',
        'llm_justification',
    ]

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_minimal_csv(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_required_columns_present(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        for col in self.REQUIRED_COLUMNS:
            self.assertIn(col, df.columns, f"Missing column: {col}")

    def test_final_label_is_int(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertTrue(df['final_label'].dtype in (int, np.int64, np.int32),
                        f"final_label dtype: {df['final_label'].dtype}")

    def test_session_number_is_int(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertTrue(df['session_number'].dtype in (int, np.int64, np.int32))

    def test_word_count_is_int(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertTrue(df['word_count'].dtype in (int, np.int64, np.int32))

    def test_codebook_labels_is_list(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        for v in df['codebook_labels_ensemble']:
            self.assertIsInstance(v, list)

    def test_codebook_labels_parsed(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertEqual(df.iloc[0]['codebook_labels_ensemble'], ['body_awareness', 'somatic'])

    def test_confidence_primary_numeric(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertAlmostEqual(float(df.iloc[0]['llm_confidence_primary']), 0.85, places=5)


class TestLoadSegmentsSpeakerFilter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        _make_minimal_csv(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_participant_filter(self):
        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertTrue((df['speaker'] == 'participant').all())
        self.assertEqual(len(df), 1)

    def test_therapist_filter(self):
        # Therapist rows have NaN final_label, so require_labeled=False
        df = load_segments(self.tmp, speaker_filter='therapist', require_labeled=False)
        self.assertTrue((df['speaker'] == 'therapist').all())
        self.assertEqual(len(df), 1)

    def test_no_filter_returns_all(self):
        df = load_segments(self.tmp, speaker_filter=None, require_labeled=False)
        self.assertEqual(len(df), 2)

    def test_require_labeled_drops_nan(self):
        df = load_segments(self.tmp, speaker_filter=None, require_labeled=True)
        self.assertFalse(df['final_label'].isna().any())


class TestLoadSegmentsCohortDerivation(unittest.TestCase):
    """cohort_id is derived from session_id when absent."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cohort_derived_from_session_id(self):
        """When cohort_id column is absent, derive from session_id 'c2s3' -> 2."""
        rows = [dict(
            segment_id='c2s3_0', participant_id='P02',
            session_id='c2s3', session_number=3,
            session_variant='', segment_index=0,
            speaker='participant', text='hello', word_count=1,
            start_time_ms=0, end_time_ms=500,
            primary_stage=1, final_label=1,
            secondary_stage=float('nan'),
            llm_confidence_primary=0.7,
            llm_confidence_secondary=float('nan'),
            llm_run_consistency=2,
            label_confidence_tier='medium', llm_justification='',
            codebook_labels_ensemble='[]',
            codebook_labels_embedding='[]',
            codebook_labels_llm='[]',
            codebook_disagreements='[]',
            in_human_coded_subset=False,
        )]
        path = _master_csv_path(self.tmp)
        df_in = pd.DataFrame(rows)
        df_in = df_in.drop(columns=['cohort_id'], errors='ignore')
        df_in.to_csv(path, index=False)

        df = load_segments(self.tmp, speaker_filter='participant')
        self.assertEqual(int(df.iloc[0]['cohort_id']), 2)


class TestParseListColumn(unittest.TestCase):
    def test_string_list(self):
        self.assertEqual(_parse_list_column("['a', 'b']"), ['a', 'b'])

    def test_empty_list_string(self):
        self.assertEqual(_parse_list_column("[]"), [])

    def test_none_returns_empty(self):
        self.assertEqual(_parse_list_column(None), [])

    def test_nan_returns_empty(self):
        self.assertEqual(_parse_list_column(float('nan')), [])

    def test_already_list(self):
        self.assertEqual(_parse_list_column(['x', 'y']), ['x', 'y'])

    def test_invalid_string_returns_empty(self):
        self.assertEqual(_parse_list_column("not_a_list"), [])


class TestDeriveCohortId(unittest.TestCase):
    def test_standard_format(self):
        self.assertEqual(_derive_cohort_id('c1s3'), 1)
        self.assertEqual(_derive_cohort_id('c2s7'), 2)
        self.assertEqual(_derive_cohort_id('C3S1'), 3)

    def test_no_match_returns_none(self):
        self.assertIsNone(_derive_cohort_id('session_1'))
        self.assertIsNone(_derive_cohort_id(''))


class TestSortSessionIds(unittest.TestCase):
    def test_longitudinal_order(self):
        ids = ['c1s3', 'c1s1', 'c2s2', 'c1s2', 'c1s4a', 'c1s4']
        result = sort_session_ids(ids)
        # c1s1 < c1s2 < c1s3 < c1s4 < c1s4a < c2s2
        self.assertEqual(result.index('c1s1'), 0)
        self.assertLess(result.index('c1s4'), result.index('c1s4a'))
        self.assertLess(result.index('c1s4a'), result.index('c2s2'))

    def test_single_element(self):
        self.assertEqual(sort_session_ids(['c1s1']), ['c1s1'])

    def test_empty(self):
        self.assertEqual(sort_session_ids([]), [])


class TestLoadSegmentsMissingFile(unittest.TestCase):
    def test_missing_file_raises(self):
        tmp = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                load_segments(tmp)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestLoadSegmentsEmptyFile(unittest.TestCase):
    """A CSV with header only (0 data rows) should load to an empty DataFrame."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_empty_csv_no_crash(self):
        # Write a header-only CSV (zero rows)
        path = _master_csv_path(self.tmp)
        pd.DataFrame(columns=[
            'segment_id', 'participant_id', 'session_id', 'session_number',
            'cohort_id', 'session_variant', 'segment_index', 'speaker',
            'text', 'word_count', 'start_time_ms', 'end_time_ms',
            'primary_stage', 'final_label', 'secondary_stage',
            'llm_confidence_primary', 'llm_confidence_secondary',
            'llm_run_consistency', 'label_confidence_tier',
            'llm_justification', 'codebook_labels_ensemble',
            'codebook_labels_embedding', 'codebook_labels_llm',
            'codebook_disagreements', 'in_human_coded_subset',
        ]).to_csv(path, index=False)

        df = load_segments(self.tmp, speaker_filter=None, require_labeled=False)
        self.assertEqual(len(df), 0)


if __name__ == '__main__':
    unittest.main()
