"""
tests/unit/test_master_dataset.py
----------------------------------
Unit tests for process/assembly/master_dataset.assemble_master_dataset.

Covers:
- All provenance tiers in priority order:
    adjudicated > human_consensus > gnn_consensus > llm_zero_shot
- GNN tier engaged only when gnn_authoritative=True
- LLM tier wins over GNN when gnn_authoritative=False
- Raw labels (primary_stage, purer_primary, rater_votes, gnn_* fields) preserved
- Confidence tiering: high / medium / low per ConfidenceTierConfig logic
- purer_final / purer_final_source populated for therapist segments
- gnn_* columns present in output
- Both master_segments.jsonl AND master_segments.csv written, rows roundtrip
- Empty segment list produces empty DataFrame (no crash)
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment
from process.assembly.master_dataset import assemble_master_dataset


def _make_participant(segment_id='p1', primary_stage=2, **kwargs):
    """Minimal participant segment with classification fields."""
    seg = Segment(segment_id=segment_id, speaker='participant', text='some participant text')
    seg.primary_stage = primary_stage
    seg.llm_confidence_primary = kwargs.pop('llm_confidence_primary', 0.9)
    seg.llm_run_consistency = kwargs.pop('llm_run_consistency', 3)
    for k, v in kwargs.items():
        setattr(seg, k, v)
    return seg


def _make_therapist(segment_id='t1', purer_primary=1, **kwargs):
    """Minimal therapist segment with PURER classification fields."""
    seg = Segment(segment_id=segment_id, speaker='therapist', text='some therapist text')
    seg.purer_primary = purer_primary
    for k, v in kwargs.items():
        setattr(seg, k, v)
    return seg


class TestProvenanceTierOrder(unittest.TestCase):
    """Priority: adjudicated > human_consensus > gnn_consensus > llm_zero_shot."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.out = os.path.join(self.tmpdir, 'master_segments.jsonl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_adjudicated_wins_over_everything(self):
        """adjudicated_label beats human_label, GNN, and LLM."""
        seg = _make_participant('p1', primary_stage=1)
        seg.human_label = 1   # matches primary_stage so human_consensus would fire
        seg.adjudicated_label = 4
        seg.gnn_vaamr_pred = 3
        df = assemble_master_dataset([seg], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label'], 4)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'adjudicated')

    def test_human_consensus_when_human_matches_llm(self):
        """human_label == primary_stage triggers human_consensus tier."""
        seg = _make_participant('p1', primary_stage=2)
        seg.human_label = 2   # equals primary_stage
        seg.adjudicated_label = None
        seg.gnn_vaamr_pred = 4
        df = assemble_master_dataset([seg], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label'], 2)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'human_consensus')

    def test_human_label_mismatch_does_not_trigger_human_consensus(self):
        """human_label != primary_stage means no human_consensus — falls through."""
        seg = _make_participant('p1', primary_stage=2)
        seg.human_label = 3   # disagrees with LLM
        seg.adjudicated_label = None
        seg.gnn_vaamr_pred = None
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=False)
        r = df.set_index('segment_id')
        # No adjudicated, no matching human, no GNN — falls to llm_zero_shot
        self.assertEqual(r.loc['p1', 'final_label'], 2)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'llm_zero_shot')

    def test_gnn_consensus_only_when_authoritative_true(self):
        """gnn_consensus fires only when gnn_authoritative=True."""
        seg = _make_participant('p1', primary_stage=1)
        seg.gnn_vaamr_pred = 3
        seg.gnn_vaamr_conf = 0.9

        df_off = assemble_master_dataset([seg], self.out, gnn_authoritative=False)
        r_off = df_off.set_index('segment_id')
        self.assertEqual(r_off.loc['p1', 'final_label'], 1)
        self.assertEqual(r_off.loc['p1', 'final_label_source'], 'llm_zero_shot')

        df_on = assemble_master_dataset([seg], self.out,
                                        gnn_authoritative=True, gate_passed=True)
        r_on = df_on.set_index('segment_id')
        self.assertEqual(r_on.loc['p1', 'final_label'], 3)
        self.assertEqual(r_on.loc['p1', 'final_label_source'], 'gnn_consensus')

    def test_gnn_consensus_not_applied_to_therapist_vaamr(self):
        """GNN VAAMR prediction must not become VAAMR label for therapist segments."""
        seg = _make_therapist('t1', purer_primary=0)
        seg.gnn_vaamr_pred = 2  # should be ignored — therapist gets no VAAMR label
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=True)
        r = df.set_index('segment_id')
        # Therapist has no primary_stage, so final_label stays None
        self.assertIsNone(r.loc['t1', 'final_label'])

    def test_llm_zero_shot_fallback(self):
        """Pure LLM path: no adjudicated, no matching human, no GNN."""
        seg = _make_participant('p1', primary_stage=0)
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=False)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'final_label'], 0)
        self.assertEqual(r.loc['p1', 'final_label_source'], 'llm_zero_shot')

    def test_no_label_produces_none(self):
        """Segment with no classification → final_label and source are None."""
        seg = Segment(segment_id='p_bare', speaker='participant', text='no label')
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=False)
        r = df.set_index('segment_id')
        self.assertIsNone(r.loc['p_bare', 'final_label'])
        self.assertIsNone(r.loc['p_bare', 'final_label_source'])


class TestRawLabelAuditability(unittest.TestCase):
    """Raw labels must be preserved alongside the derived final_label."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.out = os.path.join(self.tmpdir, 'master_segments.jsonl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_primary_stage_preserved_when_gnn_wins(self):
        seg = _make_participant('p1', primary_stage=1)
        seg.gnn_vaamr_pred = 4
        seg.gnn_vaamr_conf = 0.88
        df = assemble_master_dataset([seg], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        # final_label is GNN
        self.assertEqual(r.loc['p1', 'final_label'], 4)
        # raw LLM label still present
        self.assertEqual(r.loc['p1', 'primary_stage'], 1)

    def test_purer_primary_preserved_when_gnn_wins_purer(self):
        seg = _make_therapist('t1', purer_primary=0)
        seg.gnn_purer_pred = 3
        seg.gnn_purer_conf = 0.75
        df = assemble_master_dataset([seg], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['t1', 'purer_final'], 3)
        self.assertEqual(r.loc['t1', 'purer_primary'], 0)

    def test_rater_votes_preserved(self):
        seg = _make_participant('p1', primary_stage=2)
        seg.rater_votes = [{'rater': 'r1', 'stage': 2, 'confidence': 0.9}]
        df = assemble_master_dataset([seg], self.out)
        r = df.set_index('segment_id')
        # rater_votes serialized as JSON string in the DataFrame
        rv = r.loc['p1', 'rater_votes']
        # may be a JSON string
        if isinstance(rv, str):
            rv = json.loads(rv)
        self.assertEqual(len(rv), 1)
        self.assertEqual(rv[0]['stage'], 2)

    def test_gnn_fields_present_in_output(self):
        """gnn_vaamr_pred/conf/purer_pred/purer_conf/gnn_label_source columns exist."""
        seg = _make_participant('p1', primary_stage=2)
        seg.gnn_vaamr_pred = 3
        seg.gnn_vaamr_conf = 0.82
        seg.gnn_label_source = 'gnn_trained'
        df = assemble_master_dataset([seg], self.out)
        for col in ('gnn_vaamr_pred', 'gnn_vaamr_conf', 'gnn_purer_pred',
                    'gnn_purer_conf', 'gnn_label_source'):
            self.assertIn(col, df.columns, f'column {col!r} missing from DataFrame')
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['p1', 'gnn_vaamr_pred'], 3)
        self.assertAlmostEqual(r.loc['p1', 'gnn_vaamr_conf'], 0.82)
        self.assertEqual(r.loc['p1', 'gnn_label_source'], 'gnn_trained')


class TestPurerFinalColumn(unittest.TestCase):
    """purer_final / purer_final_source populated correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.out = os.path.join(self.tmpdir, 'master_segments.jsonl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_purer_final_llm_zero_shot_default(self):
        seg = _make_therapist('t1', purer_primary=2)
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=False)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['t1', 'purer_final'], 2)
        self.assertEqual(r.loc['t1', 'purer_final_source'], 'llm_zero_shot')

    def test_purer_final_gnn_consensus_when_authoritative(self):
        seg = _make_therapist('t1', purer_primary=0)
        seg.gnn_purer_pred = 4
        seg.gnn_purer_conf = 0.78
        df = assemble_master_dataset([seg], self.out,
                                     gnn_authoritative=True, gate_passed=True)
        r = df.set_index('segment_id')
        self.assertEqual(r.loc['t1', 'purer_final'], 4)
        self.assertEqual(r.loc['t1', 'purer_final_source'], 'gnn_consensus')

    def test_purer_final_none_for_participant(self):
        """Participant segments have no PURER label."""
        seg = _make_participant('p1', primary_stage=2)
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=False)
        r = df.set_index('segment_id')
        self.assertIsNone(r.loc['p1', 'purer_final'])
        self.assertIsNone(r.loc['p1', 'purer_final_source'])

    def test_purer_gnn_not_applied_to_participant(self):
        """gnn_purer_pred on a participant segment must not set purer_final."""
        seg = _make_participant('p1', primary_stage=2)
        seg.gnn_purer_pred = 3   # spurious GNN field on a participant
        df = assemble_master_dataset([seg], self.out, gnn_authoritative=True)
        r = df.set_index('segment_id')
        self.assertIsNone(r.loc['p1', 'purer_final'])


class TestConfidenceTiering(unittest.TestCase):
    """Confidence tier logic: high / medium / low."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.out = os.path.join(self.tmpdir, 'master_segments.jsonl')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _assemble(self, seg, tiers=None):
        return assemble_master_dataset([seg], self.out,
                                       confidence_tiers=tiers).set_index('segment_id')

    def test_high_tier_unanimous_and_above_threshold(self):
        """llm_run_consistency==3 and conf>0.8 → high."""
        seg = _make_participant('p1', primary_stage=2,
                                llm_run_consistency=3, llm_confidence_primary=0.85)
        r = self._assemble(seg)
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'high')

    def test_high_tier_requires_conf_strictly_above_0_8(self):
        """conf exactly 0.8 is NOT strictly > 0.8 → falls to medium if consistency sufficient."""
        seg = _make_participant('p1', primary_stage=2,
                                llm_run_consistency=3, llm_confidence_primary=0.80)
        r = self._assemble(seg)
        # 0.80 is not > 0.80, so not high; but consistency=3 >= 2 and 0.80 > 0.60 → medium
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'medium')

    def test_medium_tier_majority_and_above_medium_threshold(self):
        """llm_run_consistency==2 and conf>0.6 → medium."""
        seg = _make_participant('p1', primary_stage=1,
                                llm_run_consistency=2, llm_confidence_primary=0.75)
        r = self._assemble(seg)
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'medium')

    def test_low_tier_low_consistency(self):
        """llm_run_consistency==1 → low regardless of confidence."""
        seg = _make_participant('p1', primary_stage=1,
                                llm_run_consistency=1, llm_confidence_primary=0.95)
        r = self._assemble(seg)
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'low')

    def test_low_tier_none_consistency(self):
        """None consistency → low."""
        seg = _make_participant('p1', primary_stage=1)
        seg.llm_run_consistency = None
        seg.llm_confidence_primary = 0.95
        r = self._assemble(seg)
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'low')

    def test_low_tier_low_confidence_despite_high_consistency(self):
        """High consistency but conf<=0.6 → low."""
        seg = _make_participant('p1', primary_stage=2,
                                llm_run_consistency=3, llm_confidence_primary=0.55)
        r = self._assemble(seg)
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'low')

    def test_custom_tier_config_respected(self):
        """Caller can override threshold values via confidence_tiers dict."""
        # lower the bar: high requires consistency >= 2, conf > 0.5
        seg = _make_participant('p1', primary_stage=2,
                                llm_run_consistency=2, llm_confidence_primary=0.6)
        custom_tiers = {
            'high_consistency': 2,
            'high_confidence': 0.5,
            'medium_min_consistency': 1,
            'medium_min_confidence': 0.3,
        }
        r = self._assemble(seg, tiers=custom_tiers)
        self.assertEqual(r.loc['p1', 'label_confidence_tier'], 'high')

    def test_all_three_tiers_in_mixed_batch(self):
        """A batch with high/medium/low segments produces all three tiers."""
        segs = [
            _make_participant('ph', primary_stage=2,
                              llm_run_consistency=3, llm_confidence_primary=0.92),
            _make_participant('pm', primary_stage=1,
                              llm_run_consistency=2, llm_confidence_primary=0.72),
            _make_participant('pl', primary_stage=0,
                              llm_run_consistency=1, llm_confidence_primary=0.4),
        ]
        df = assemble_master_dataset(segs, self.out)
        tiers = dict(zip(df['segment_id'], df['label_confidence_tier']))
        self.assertEqual(tiers['ph'], 'high')
        self.assertEqual(tiers['pm'], 'medium')
        self.assertEqual(tiers['pl'], 'low')


class TestDualOutputFiles(unittest.TestCase):
    """assemble_master_dataset writes both .jsonl and .csv, and rows roundtrip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.jsonl_path = os.path.join(self.tmpdir, 'master_segments.jsonl')
        self.csv_path = os.path.join(self.tmpdir, 'master_segments.csv')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _build_segments(self):
        p = _make_participant('p1', primary_stage=2,
                              llm_run_consistency=3, llm_confidence_primary=0.85)
        t = _make_therapist('t1', purer_primary=1)
        return [p, t]

    def test_jsonl_file_written(self):
        assemble_master_dataset(self._build_segments(), self.jsonl_path)
        self.assertTrue(os.path.isfile(self.jsonl_path),
                        'master_segments.jsonl was not created')

    def test_csv_file_written(self):
        assemble_master_dataset(self._build_segments(), self.jsonl_path)
        self.assertTrue(os.path.isfile(self.csv_path),
                        'master_segments.csv was not created')

    def test_jsonl_roundtrip_segment_ids(self):
        assemble_master_dataset(self._build_segments(), self.jsonl_path)
        ids = []
        with open(self.jsonl_path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    ids.append(json.loads(line)['segment_id'])
        self.assertIn('p1', ids)
        self.assertIn('t1', ids)

    def test_csv_roundtrip_segment_ids(self):
        import pandas as pd
        assemble_master_dataset(self._build_segments(), self.jsonl_path)
        df = pd.read_csv(self.csv_path)
        self.assertIn('p1', df['segment_id'].tolist())
        self.assertIn('t1', df['segment_id'].tolist())

    def test_jsonl_contains_final_label(self):
        assemble_master_dataset(self._build_segments(), self.jsonl_path)
        records = {}
        with open(self.jsonl_path, encoding='utf-8') as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    records[rec['segment_id']] = rec
        self.assertEqual(records['p1']['final_label'], 2)
        self.assertEqual(records['p1']['final_label_source'], 'llm_zero_shot')
        self.assertEqual(records['t1']['purer_final'], 1)

    def test_returned_dataframe_matches_jsonl_row_count(self):
        segs = self._build_segments()
        df = assemble_master_dataset(segs, self.jsonl_path)
        lines = [l for l in open(self.jsonl_path) if l.strip()]
        self.assertEqual(len(df), len(lines))
        self.assertEqual(len(df), len(segs))

    def test_empty_segment_list_produces_empty_files(self):
        df = assemble_master_dataset([], self.jsonl_path)
        self.assertTrue(os.path.isfile(self.jsonl_path))
        self.assertTrue(os.path.isfile(self.csv_path))
        self.assertEqual(len(df), 0)
        lines = [l for l in open(self.jsonl_path) if l.strip()]
        self.assertEqual(len(lines), 0)

    def test_required_columns_present(self):
        """All expected output columns exist in the returned DataFrame."""
        segs = self._build_segments()
        df = assemble_master_dataset(segs, self.jsonl_path)
        required = [
            'segment_id', 'speaker', 'text',
            'primary_stage', 'purer_primary',
            'final_label', 'final_label_source', 'label_confidence_tier',
            'purer_final', 'purer_final_source',
            'gnn_vaamr_pred', 'gnn_vaamr_conf',
            'gnn_purer_pred', 'gnn_purer_conf', 'gnn_label_source',
            'human_label', 'adjudicated_label', 'rater_votes',
        ]
        for col in required:
            self.assertIn(col, df.columns, f'expected column {col!r} missing')


if __name__ == '__main__':
    unittest.main()
