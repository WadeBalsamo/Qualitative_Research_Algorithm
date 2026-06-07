"""
tests/unit/test_gnn_validation.py
----------------------------------
Unit tests for gnn_layer/validation.py.

Covers: _per_class (recall/precision/binary-kappa; zero-support rows),
_overall, n_or_true, evaluate_crossval (rare-stage notes, ready_for_scaling,
human axis), write_validation_report (exact path, YES/NO verdict line),
write_validation_csv (exact path, expected columns/rows).
"""

import os
import sys
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from tests.testhelpers import synthetic_df, embedding_patch, make_master_df
from gnn_layer.config import GnnLayerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df(n=10):
    """10-row dataframe cycling stages 0..4, no human subset, no PURER labels."""
    rows = []
    for i in range(n):
        rows.append(dict(
            segment_id=f's{i}',
            final_label=i % 5,
            purer_primary=np.nan,
            in_human_coded_subset=False,
            human_label=np.nan,
        ))
    return pd.DataFrame(rows)


def _df_with_purer(n=10):
    rows = []
    for i in range(n):
        rows.append(dict(
            segment_id=f's{i}',
            final_label=i % 5,
            purer_primary=i % 5,
            in_human_coded_subset=False,
            human_label=np.nan,
        ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# _per_class
# ---------------------------------------------------------------------------

class TestPerClass(unittest.TestCase):

    def setUp(self):
        from gnn_layer.classifier.validation import _per_class, VAAMR_NAMES
        self._per_class = _per_class
        self._names = VAAMR_NAMES

    def test_basic_shape_and_keys(self):
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        rows = self._per_class(pairs, self._names)
        self.assertEqual(len(rows), 5)
        expected_keys = {'class_id', 'class_name', 'support', 'recall', 'precision', 'kappa'}
        for row in rows:
            self.assertEqual(set(row.keys()), expected_keys)

    def test_perfect_recall_and_precision(self):
        pairs = [(0, 0), (1, 1), (2, 2)]
        rows = self._per_class(pairs, self._names)
        for row in rows:
            self.assertAlmostEqual(row['recall'], 1.0, places=5)
            self.assertAlmostEqual(row['precision'], 1.0, places=5)

    def test_zero_support_recall_is_none(self):
        # class 4 appears only in preds, not in reference → support=0, recall=None
        pairs = [(4, 0), (4, 1), (0, 0)]
        rows = self._per_class(pairs, self._names)
        s4 = next(r for r in rows if r['class_id'] == 4)
        self.assertEqual(s4['support'], 0)
        self.assertIsNone(s4['recall'])

    def test_zero_pred_precision_is_none(self):
        # class 4 only appears in reference, never predicted → precision=None
        pairs = [(0, 4), (1, 4)]
        rows = self._per_class(pairs, self._names)
        s4 = next(r for r in rows if r['class_id'] == 4)
        self.assertIsNone(s4['precision'])
        self.assertEqual(s4['support'], 2)

    def test_partial_recall_computed(self):
        # stage 1: 2 ref occurrences, 1 predicted correctly
        pairs = [(1, 1), (0, 1)]
        rows = self._per_class(pairs, self._names)
        s1 = next(r for r in rows if r['class_id'] == 1)
        self.assertAlmostEqual(s1['recall'], 0.5, places=5)

    def test_class_names_resolved(self):
        pairs = [(0, 0)]
        rows = self._per_class(pairs, self._names)
        self.assertEqual(rows[0]['class_name'], 'Vigilance')

    def test_unknown_class_name_falls_back_to_str(self):
        pairs = [(99, 99)]
        rows = self._per_class(pairs, self._names)
        r99 = next(r for r in rows if r['class_id'] == 99)
        self.assertEqual(r99['class_name'], '99')

    def test_kappa_present_and_float(self):
        pairs = [(0, 0), (1, 1), (2, 2)]
        rows = self._per_class(pairs, self._names)
        for row in rows:
            self.assertIsNotNone(row['kappa'])
            self.assertIsInstance(float(row['kappa']), float)

    def test_empty_pairs_returns_empty(self):
        rows = self._per_class([], self._names)
        self.assertEqual(rows, [])


# ---------------------------------------------------------------------------
# _overall
# ---------------------------------------------------------------------------

class TestOverall(unittest.TestCase):

    def setUp(self):
        from gnn_layer.classifier.validation import _overall
        self._overall = _overall

    def test_empty_pairs(self):
        res = self._overall([])
        self.assertEqual(res['n'], 0)
        self.assertIsNone(res['percent_agreement'])
        self.assertIsNone(res['cohen_kappa'])

    def test_perfect_agreement(self):
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        res = self._overall(pairs)
        self.assertEqual(res['n'], 5)
        self.assertAlmostEqual(res['percent_agreement'], 1.0, places=4)

    def test_partial_agreement(self):
        pairs = [(0, 0), (1, 1), (2, 2), (3, 3), (0, 4)]  # 4/5 agree
        res = self._overall(pairs)
        self.assertAlmostEqual(res['percent_agreement'], 0.8, places=4)

    def test_keys_present(self):
        res = self._overall([(0, 0)])
        self.assertIn('n', res)
        self.assertIn('percent_agreement', res)
        self.assertIn('cohen_kappa', res)

    def test_kappa_is_float(self):
        pairs = [(i % 3, i % 3) for i in range(9)]
        res = self._overall(pairs)
        self.assertIsInstance(res['cohen_kappa'], float)


# ---------------------------------------------------------------------------
# n_or_true
# ---------------------------------------------------------------------------

class TestNOrTrue(unittest.TestCase):

    def setUp(self):
        from gnn_layer.classifier.validation import n_or_true
        self._n_or_true = n_or_true

    def test_empty_purer_returns_true_regardless_of_flag(self):
        self.assertTrue(self._n_or_true([], True))
        self.assertTrue(self._n_or_true([], False))

    def test_non_empty_purer_uses_purer_ready_flag(self):
        self.assertFalse(self._n_or_true([(0, 0)], False))
        self.assertTrue(self._n_or_true([(0, 0)], True))


# ---------------------------------------------------------------------------
# evaluate_crossval
# ---------------------------------------------------------------------------

class TestEvaluateCrossval(unittest.TestCase):

    def setUp(self):
        from gnn_layer.classifier.validation import evaluate_crossval, RARE_STAGES, MIN_SUPPORT_FOR_FLOOR
        self._eval = evaluate_crossval
        self._RARE_STAGES = RARE_STAGES
        self._MIN_SUPPORT = MIN_SUPPORT_FOR_FLOOR

    def test_returns_expected_top_level_keys(self):
        df = _df()
        cv = {'vaamr': [(f's{i}', i % 5) for i in range(10)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = self._eval(df, cv, cfg)
        for k in ('irr_target', 'vaamr_overall', 'vaamr_per_class',
                  'purer_overall', 'purer_per_class', 'vaamr_human',
                  'rare_stage_notes', 'vaamr_ready', 'purer_ready', 'ready_for_scaling'):
            self.assertIn(k, m)

    def test_perfect_vaamr_is_ready(self):
        df = _df()
        cv = {'vaamr': [(f's{i}', i % 5) for i in range(10)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = self._eval(df, cv, cfg)
        self.assertTrue(m['vaamr_ready'])
        self.assertTrue(m['ready_for_scaling'])
        self.assertAlmostEqual(m['vaamr_overall']['percent_agreement'], 1.0, places=4)

    def test_bad_vaamr_not_ready(self):
        df = _df()
        cv = {'vaamr': [(f's{i}', 0) for i in range(10)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.7)
        m = self._eval(df, cv, cfg)
        self.assertFalse(m['vaamr_ready'])
        self.assertFalse(m['ready_for_scaling'])

    def test_partial_agreement_correct_fraction(self):
        df = _df()
        # 8/10 correct (s8→stage3 but df has label 3; s9→stage4 but predict 0)
        preds = [(f's{i}', i % 5) for i in range(8)]
        preds += [('s8', 3), ('s9', 0)]  # s9's true label is 4, predict 0
        cv = {'vaamr': preds, 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = self._eval(df, cv, cfg)
        self.assertAlmostEqual(m['vaamr_overall']['percent_agreement'], 0.9, places=3)

    def test_rare_stage_note_triggered_when_zero_recall_and_enough_support(self):
        # Stage 4 (Reappraisal) has support >= MIN_SUPPORT_FOR_FLOOR and recall 0
        rows = [dict(segment_id=f's{i}', final_label=4,
                     purer_primary=np.nan, in_human_coded_subset=False, human_label=np.nan)
                for i in range(self._MIN_SUPPORT)]  # exactly at floor
        df = pd.DataFrame(rows)
        preds = [(f's{i}', 0) for i in range(self._MIN_SUPPORT)]  # all wrong
        cv = {'vaamr': preds, 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = self._eval(df, cv, cfg)
        self.assertTrue(len(m['rare_stage_notes']) > 0)
        self.assertIn('Reappraisal', m['rare_stage_notes'][0])
        self.assertFalse(m['vaamr_ready'])

    def test_rare_stage_note_not_triggered_below_min_support(self):
        # Only 1 stage-4 segment → below MIN_SUPPORT_FOR_FLOOR → no note
        rows = [dict(segment_id='s0', final_label=4, purer_primary=np.nan,
                     in_human_coded_subset=False, human_label=np.nan)]
        df = pd.DataFrame(rows)
        cv = {'vaamr': [('s0', 0)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.0)
        m = self._eval(df, cv, cfg)
        self.assertEqual(m['rare_stage_notes'], [])

    def test_purer_gates_scaling_when_data_present(self):
        df = _df_with_purer()
        # perfect vaamr, bad purer
        cv = {
            'vaamr': [(f's{i}', i % 5) for i in range(10)],
            'purer': [(f's{i}', 0) for i in range(10)],  # all wrong
        }
        cfg = GnnLayerConfig(irr_target=0.7)
        m = self._eval(df, cv, cfg)
        # vaamr_ready may be True but purer_ready False → ready_for_scaling False
        self.assertFalse(m['purer_ready'])
        self.assertFalse(m['ready_for_scaling'])

    def test_purer_doesnt_gate_when_no_purer_data(self):
        df = _df()
        cv = {'vaamr': [(f's{i}', i % 5) for i in range(10)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = self._eval(df, cv, cfg)
        # purer_ready False but no purer data → n_or_true([],purer_ready)=True → scaling OK
        self.assertTrue(m['ready_for_scaling'])

    def test_human_axis_computed_from_subset(self):
        rows = [
            dict(segment_id='s0', final_label=2, purer_primary=np.nan,
                 in_human_coded_subset=True, human_label=2),   # graph correct, llm correct
            dict(segment_id='s1', final_label=3, purer_primary=np.nan,
                 in_human_coded_subset=True, human_label=2),   # graph wrong, llm correct vs human
            dict(segment_id='s2', final_label=1, purer_primary=np.nan,
                 in_human_coded_subset=False, human_label=np.nan),
        ]
        df = pd.DataFrame(rows)
        cv = {'vaamr': [('s0', 2), ('s1', 3), ('s2', 1)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.0)
        m = self._eval(df, cv, cfg)
        gvh = m['vaamr_human']['graph_vs_human']
        self.assertEqual(gvh['n'], 2)          # only 2 human-coded rows
        self.assertAlmostEqual(gvh['percent_agreement'], 0.5, places=3)

    def test_human_axis_empty_when_no_human_subset(self):
        df = _df()
        cv = {'vaamr': [(f's{i}', i % 5) for i in range(10)], 'purer': []}
        cfg = GnnLayerConfig(irr_target=0.5)
        m = self._eval(df, cv, cfg)
        self.assertEqual(m['vaamr_human']['graph_vs_human']['n'], 0)


# ---------------------------------------------------------------------------
# write_validation_report
# ---------------------------------------------------------------------------

class TestWriteValidationReport(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_metrics(self, perfect=True):
        from gnn_layer.classifier.validation import evaluate_crossval
        df = _df()
        if perfect:
            preds = [(f's{i}', i % 5) for i in range(10)]
        else:
            preds = [(f's{i}', 0) for i in range(10)]
        cv = {'vaamr': preds, 'purer': []}
        return evaluate_crossval(df, cv, GnnLayerConfig(irr_target=0.5))

    def test_file_written_to_exact_path(self):
        from process import output_paths as op
        from gnn_layer.classifier.validation import write_validation_report
        m = self._make_metrics(perfect=True)
        path = write_validation_report(m, self.tmp)
        expected = os.path.join(op.reports_gnn_dir(self.tmp), 'validation.txt')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_report_contains_verdict_line(self):
        from gnn_layer.classifier.validation import write_validation_report
        m = self._make_metrics(perfect=True)
        path = write_validation_report(m, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('LLM-FREE SCALING?', content)

    def test_report_says_yes_when_ready(self):
        from gnn_layer.classifier.validation import write_validation_report
        m = self._make_metrics(perfect=True)
        m['ready_for_scaling'] = True
        path = write_validation_report(m, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('YES', content)

    def test_report_says_no_when_not_ready(self):
        from gnn_layer.classifier.validation import write_validation_report
        m = self._make_metrics(perfect=False)
        path = write_validation_report(m, self.tmp)
        content = open(path, encoding='utf-8').read()
        self.assertIn('LLM-FREE SCALING?', content)
        self.assertIn('NO', content)

    def test_parent_dir_created_automatically(self):
        from gnn_layer.classifier.validation import write_validation_report
        # delete any previously created dir
        import shutil as _sh
        from process import output_paths as op
        _sh.rmtree(op.reports_gnn_dir(self.tmp), ignore_errors=True)
        m = self._make_metrics()
        write_validation_report(m, self.tmp)
        self.assertTrue(os.path.isdir(op.reports_gnn_dir(self.tmp)))


# ---------------------------------------------------------------------------
# write_validation_csv
# ---------------------------------------------------------------------------

class TestWriteValidationCsv(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_full_metrics(self):
        from gnn_layer.classifier.validation import evaluate_crossval
        df = _df_with_purer()
        cv = {
            'vaamr': [(f's{i}', i % 5) for i in range(10)],
            'purer': [(f's{i}', i % 5) for i in range(10)],
        }
        return evaluate_crossval(df, cv, GnnLayerConfig(irr_target=0.5))

    def test_file_written_to_exact_path(self):
        from process import output_paths as op
        from gnn_layer.classifier.validation import write_validation_csv
        m = self._make_full_metrics()
        path = write_validation_csv(m, self.tmp)
        expected = os.path.join(op.gnn_data_dir(self.tmp), 'gnn_validation.csv')
        self.assertEqual(path, expected)
        self.assertTrue(os.path.isfile(path))

    def test_csv_has_expected_columns(self):
        from gnn_layer.classifier.validation import write_validation_csv
        m = self._make_full_metrics()
        path = write_validation_csv(m, self.tmp)
        df_csv = pd.read_csv(path)
        expected_cols = ['framework', 'scope', 'class_id', 'class_name',
                         'support', 'recall', 'precision', 'kappa']
        self.assertEqual(list(df_csv.columns), expected_cols)

    def test_csv_has_both_frameworks(self):
        from gnn_layer.classifier.validation import write_validation_csv
        m = self._make_full_metrics()
        path = write_validation_csv(m, self.tmp)
        df_csv = pd.read_csv(path)
        self.assertIn('vaamr', df_csv['framework'].values)
        self.assertIn('purer', df_csv['framework'].values)

    def test_csv_has_overall_and_per_class_rows(self):
        from gnn_layer.classifier.validation import write_validation_csv
        m = self._make_full_metrics()
        path = write_validation_csv(m, self.tmp)
        df_csv = pd.read_csv(path)
        self.assertIn('overall', df_csv['scope'].values)
        self.assertIn('per_class', df_csv['scope'].values)

    def test_csv_row_count(self):
        # 2 frameworks × (1 overall + 5 per_class) = 12 rows
        from gnn_layer.classifier.validation import write_validation_csv
        m = self._make_full_metrics()
        path = write_validation_csv(m, self.tmp)
        df_csv = pd.read_csv(path)
        self.assertEqual(len(df_csv), 12)

    def test_parent_dir_created_automatically(self):
        from gnn_layer.classifier.validation import write_validation_csv
        from process import output_paths as op
        import shutil as _sh
        _sh.rmtree(op.gnn_data_dir(self.tmp), ignore_errors=True)
        m = self._make_full_metrics()
        write_validation_csv(m, self.tmp)
        self.assertTrue(os.path.isdir(op.gnn_data_dir(self.tmp)))


if __name__ == '__main__':
    unittest.main()
