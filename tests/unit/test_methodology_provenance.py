"""
tests/unit/test_methodology_provenance.py
------------------------------------------
Conceptual conformance: the provenance hierarchy is correctly adjudicated.

Methodology references:
  §4.7 Stage 6 — Dataset Assembly:
    Priority: adjudicated > human_consensus > gnn_consensus > llm_zero_shot
  §8.5 / CLAUDE.md:
    gnn_consensus is engaged ONLY when gnn_authoritative=True.
    Raw LLM ballots (primary_stage, purer_primary, rater_votes) are always
    preserved (auditability invariant).
    The GNN is NOT injected as an extra rater vote — distillation replaces the
    aggregator, it does NOT add a ballot. rater_votes length is unchanged by
    gnn fields.

Tests:
  1. adjudicated label takes top priority.
  2. human_consensus (human_label == primary_stage) takes priority over GNN and LLM.
  3. gnn_consensus engaged ONLY when gnn_authoritative=True.
  4. llm_zero_shot is the default source when no higher tier is present.
  5. gnn_authoritative=False leaves final_label_source as llm_zero_shot even when
     gnn_vaamr_pred is populated.
  6. rater_votes length is unchanged by the presence of GNN fields
     (GNN does not inject an extra ballot).
  7. primary_stage (raw LLM label) is always preserved in the output row.
"""

import os
import sys
import tempfile
import shutil
import json
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment
from process.assembly.master_dataset import assemble_master_dataset


# ---------------------------------------------------------------------------
# Builder helper
# ---------------------------------------------------------------------------

def _seg(sid, primary_stage=2, human_label=None, adjudicated_label=None,
         gnn_vaamr_pred=None, speaker='participant',
         rater_votes=None):
    """Build a participant Segment exercising provenance fields."""
    seg = Segment(
        segment_id=sid,
        session_id='c1s1',
        speaker=speaker,
        text='Segment text for provenance testing.',
    )
    seg.primary_stage = primary_stage
    seg.llm_confidence_primary = 0.82
    seg.llm_run_consistency = 3
    seg.human_label = human_label
    seg.adjudicated_label = adjudicated_label
    seg.gnn_vaamr_pred = gnn_vaamr_pred
    seg.rater_votes = rater_votes  # raw LLM ballots
    seg.purer_primary = None
    return seg


def _run(segments, gnn_authoritative=False, gate_passed=True):
    # gate_passed defaults to True here so the provenance-hierarchy tests treat the
    # GNN as gate-eligible; the gate safeguard itself is exercised separately.
    tmpdir = tempfile.mkdtemp()
    try:
        outpath = os.path.join(tmpdir, 'master.jsonl')
        df = assemble_master_dataset(
            segments=segments,
            output_path=outpath,
            gnn_authoritative=gnn_authoritative,
            gate_passed=gate_passed,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestProvenanceHierarchy(unittest.TestCase):
    """
    §4.7: adjudicated > human_consensus > gnn_consensus > llm_zero_shot
    """

    def test_adjudicated_label_is_top_priority(self):
        """adjudicated_label wins over human_label, gnn_vaamr_pred, and primary_stage."""
        seg = _seg('s1', primary_stage=0, human_label=1,
                   adjudicated_label=4, gnn_vaamr_pred=3)
        df = _run([seg], gnn_authoritative=True)
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertEqual(row['final_label'], 4)
        self.assertEqual(row['final_label_source'], 'adjudicated')

    def test_human_consensus_beats_gnn_and_llm(self):
        """
        human_label == primary_stage → human_consensus, even when gnn_vaamr_pred differs.
        (adjudicated_label=None, so human_consensus tier applies)
        """
        seg = _seg('s1', primary_stage=2, human_label=2, gnn_vaamr_pred=4)
        df = _run([seg], gnn_authoritative=True)
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertEqual(row['final_label'], 2)
        self.assertEqual(row['final_label_source'], 'human_consensus')

    def test_human_label_disagreeing_with_llm_is_not_human_consensus(self):
        """
        When human_label != primary_stage, the human_consensus condition is false.
        Tier falls to gnn_consensus (if authoritative) or llm_zero_shot.
        """
        seg = _seg('s1', primary_stage=2, human_label=3, gnn_vaamr_pred=4)
        df = _run([seg], gnn_authoritative=True)
        row = df[df['segment_id'] == 's1'].iloc[0]
        # human_label != primary_stage → not human_consensus
        self.assertNotEqual(row['final_label_source'], 'human_consensus')

    def test_gnn_consensus_engaged_when_gnn_authoritative_true(self):
        """
        §4.7 / §8.5: gnn_consensus tier requires gnn_authoritative=True.
        When set, gnn_vaamr_pred becomes the final_label for participants.
        """
        seg = _seg('s1', primary_stage=1, gnn_vaamr_pred=3)
        df = _run([seg], gnn_authoritative=True)
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertEqual(row['final_label'], 3)
        self.assertEqual(row['final_label_source'], 'gnn_consensus')

    def test_gnn_consensus_NOT_engaged_when_gate_not_passed(self):
        """
        Track 0.2: even with gnn_authoritative=True, the gnn_consensus tier requires
        a passing reliability gate. With gate_passed=False the GNN must NOT become the
        label of record — a config flag alone can never promote an un-gated graph.
        """
        seg = _seg('s1', primary_stage=1, gnn_vaamr_pred=3)
        df = _run([seg], gnn_authoritative=True, gate_passed=False)
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertNotEqual(row['final_label_source'], 'gnn_consensus')
        self.assertEqual(row['final_label_source'], 'llm_zero_shot')
        self.assertEqual(row['final_label'], 1)

    def test_gnn_consensus_NOT_engaged_when_gnn_authoritative_false(self):
        """
        §4.7 / §8.5: when gnn_authoritative=False (default), the GNN label is
        stored but does NOT become final_label_source='gnn_consensus'.
        final_label stays as the LLM label.
        """
        seg = _seg('s1', primary_stage=1, gnn_vaamr_pred=3)
        df = _run([seg], gnn_authoritative=False)
        row = df[df['segment_id'] == 's1'].iloc[0]
        # GNN should not override when not authoritative
        self.assertNotEqual(row['final_label_source'], 'gnn_consensus')
        self.assertEqual(row['final_label_source'], 'llm_zero_shot')
        self.assertEqual(row['final_label'], 1)  # stays as primary_stage

    def test_llm_zero_shot_is_default_source(self):
        """Default tier when no human or GNN override is present."""
        seg = _seg('s1', primary_stage=2)
        df = _run([seg])
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertEqual(row['final_label'], 2)
        self.assertEqual(row['final_label_source'], 'llm_zero_shot')

    def test_null_primary_stage_yields_null_final_label(self):
        """A segment with no primary_stage and no other override has final_label=None."""
        seg = _seg('s1', primary_stage=None)
        df = _run([seg])
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertIsNone(row['final_label'])
        self.assertIsNone(row['final_label_source'])


# ---------------------------------------------------------------------------
# Auditability: raw LLM ballots preserved
# ---------------------------------------------------------------------------

class TestAuditabilityInvariants(unittest.TestCase):
    """
    §4.7 / §8.5: raw LLM ballots are always preserved in the output.
    The GNN does NOT inject an extra rater vote — it replaces the aggregator,
    not the ballots. rater_votes length must not change with/without GNN fields.
    """

    def test_primary_stage_preserved_in_output(self):
        """primary_stage (raw LLM classification) always present in the row."""
        seg = _seg('s1', primary_stage=3, gnn_vaamr_pred=4)
        df = _run([seg], gnn_authoritative=True)
        row = df[df['segment_id'] == 's1'].iloc[0]
        # primary_stage must survive even when gnn_consensus overrides final_label
        self.assertEqual(int(row['primary_stage']), 3,
                         "primary_stage (raw LLM ballot) must be preserved in output")
        self.assertEqual(row['final_label_source'], 'gnn_consensus')

    def test_rater_votes_length_unchanged_by_gnn_fields(self):
        """
        §8.5: GNN is a distillation layer — it replaces the aggregator, does NOT
        add a ballot. rater_votes should have the same length regardless of whether
        gnn fields are populated.

        We run one segment without GNN fields and one with, and confirm rater_votes
        is identical (it is set once, by the LLM classification loop, and the
        assembly function never appends to it).
        """
        votes = [
            {'rater': 'run_0', 'stage': 2, 'confidence': 0.85},
            {'rater': 'run_1', 'stage': 2, 'confidence': 0.80},
            {'rater': 'run_2', 'stage': 3, 'confidence': 0.60},
        ]
        seg_no_gnn = _seg('no_gnn', primary_stage=2, rater_votes=votes)
        seg_with_gnn = _seg('with_gnn', primary_stage=2, rater_votes=votes,
                            gnn_vaamr_pred=3)

        df = _run([seg_no_gnn, seg_with_gnn], gnn_authoritative=True)

        def _votes_list(row):
            rv = row['rater_votes']
            if rv is None:
                return []
            if isinstance(rv, str):
                return json.loads(rv)
            return rv

        r_no_gnn = df[df['segment_id'] == 'no_gnn'].iloc[0]
        r_with_gnn = df[df['segment_id'] == 'with_gnn'].iloc[0]

        votes_no_gnn = _votes_list(r_no_gnn)
        votes_with_gnn = _votes_list(r_with_gnn)

        self.assertEqual(
            len(votes_no_gnn), len(votes_with_gnn),
            "rater_votes length must be unchanged by GNN fields — GNN is a "
            "distillation layer, not an extra rater (§8.5)"
        )
        # Both should be 3 (the three LLM runs)
        self.assertEqual(len(votes_with_gnn), 3)

    def test_gnn_fields_stored_alongside_llm_when_authoritative(self):
        """
        Even when gnn_authoritative=True, the raw LLM fields are NOT wiped.
        Both gnn_vaamr_pred and primary_stage appear in the row.
        """
        seg = _seg('s1', primary_stage=1, gnn_vaamr_pred=3)
        df = _run([seg], gnn_authoritative=True)
        row = df[df['segment_id'] == 's1'].iloc[0]
        self.assertEqual(int(row['primary_stage']), 1,
                         "primary_stage (LLM ballot) must survive alongside gnn override")
        self.assertEqual(int(row['gnn_vaamr_pred']), 3,
                         "gnn_vaamr_pred should be stored for auditability")


if __name__ == '__main__':
    unittest.main()
