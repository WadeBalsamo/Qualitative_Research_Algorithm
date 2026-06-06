"""
tests/unit/test_data_structures.py
------------------------------------
Unit tests for classification_tools/data_structures.py

Covers:
- Segment() instantiation with all defaults (no arguments)
- Default values for every field group: identity, temporal, speaker/text,
  theme labels, IRR fields, codebook fields, PURER fields, GNN fields,
  validation fields, final-label fields, conversational segmenter fields
- Mutable default isolation: Optional[List] and Optional[Dict] fields that
  are default None stay None independently across instances (not shared)
- Basic attribute override works at construction time
- GNN fields specifically: gnn_vaamr_pred/conf, gnn_purer_pred/conf,
  gnn_label_source all default to None
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.data_structures import Segment


class TestSegmentDefaults(unittest.TestCase):
    """Default values match the schema documented in data_structures.py."""

    def setUp(self):
        self.seg = Segment()

    # -- Identity fields -----------------------------------------------------

    def test_segment_id_default(self):
        self.assertEqual(self.seg.segment_id, "")

    def test_trial_id_default(self):
        self.assertEqual(self.seg.trial_id, "")

    def test_participant_id_default(self):
        self.assertEqual(self.seg.participant_id, "")

    def test_session_id_default(self):
        self.assertEqual(self.seg.session_id, "")

    def test_session_number_default(self):
        self.assertEqual(self.seg.session_number, 0)

    def test_cohort_id_default(self):
        self.assertIsNone(self.seg.cohort_id)

    def test_session_variant_default(self):
        self.assertEqual(self.seg.session_variant, '')

    # -- Temporal fields -----------------------------------------------------

    def test_segment_index_default(self):
        self.assertEqual(self.seg.segment_index, 0)

    def test_start_time_ms_default(self):
        self.assertEqual(self.seg.start_time_ms, 0)

    def test_end_time_ms_default(self):
        self.assertEqual(self.seg.end_time_ms, 0)

    def test_total_segments_in_session_default(self):
        self.assertEqual(self.seg.total_segments_in_session, 0)

    # -- Speaker / text fields -----------------------------------------------

    def test_speaker_default(self):
        self.assertEqual(self.seg.speaker, "")

    def test_text_default(self):
        self.assertEqual(self.seg.text, "")

    def test_word_count_default(self):
        self.assertEqual(self.seg.word_count, 0)

    # -- Theme label fields --------------------------------------------------

    def test_primary_stage_default(self):
        self.assertIsNone(self.seg.primary_stage)

    def test_secondary_stage_default(self):
        self.assertIsNone(self.seg.secondary_stage)

    def test_llm_confidence_primary_default(self):
        self.assertIsNone(self.seg.llm_confidence_primary)

    def test_llm_confidence_secondary_default(self):
        self.assertIsNone(self.seg.llm_confidence_secondary)

    def test_llm_justification_default(self):
        self.assertIsNone(self.seg.llm_justification)

    def test_llm_run_consistency_default(self):
        self.assertIsNone(self.seg.llm_run_consistency)

    # -- IRR fields ----------------------------------------------------------

    def test_rater_ids_default(self):
        self.assertIsNone(self.seg.rater_ids)

    def test_rater_votes_default(self):
        self.assertIsNone(self.seg.rater_votes)

    def test_agreement_level_default(self):
        self.assertIsNone(self.seg.agreement_level)

    def test_agreement_fraction_default(self):
        self.assertIsNone(self.seg.agreement_fraction)

    def test_needs_review_default(self):
        self.assertFalse(self.seg.needs_review)

    def test_consensus_vote_default(self):
        self.assertIsNone(self.seg.consensus_vote)

    def test_tie_broken_by_confidence_default(self):
        self.assertFalse(self.seg.tie_broken_by_confidence)

    def test_secondary_agreement_level_default(self):
        self.assertIsNone(self.seg.secondary_agreement_level)

    def test_secondary_agreement_fraction_default(self):
        self.assertIsNone(self.seg.secondary_agreement_fraction)

    # -- Codebook label fields -----------------------------------------------

    def test_codebook_labels_embedding_default(self):
        self.assertIsNone(self.seg.codebook_labels_embedding)

    def test_codebook_labels_llm_default(self):
        self.assertIsNone(self.seg.codebook_labels_llm)

    def test_codebook_labels_ensemble_default(self):
        self.assertIsNone(self.seg.codebook_labels_ensemble)

    def test_codebook_disagreements_default(self):
        self.assertIsNone(self.seg.codebook_disagreements)

    def test_codebook_confidence_default(self):
        self.assertIsNone(self.seg.codebook_confidence)

    # -- PURER label fields --------------------------------------------------

    def test_purer_primary_default(self):
        self.assertIsNone(self.seg.purer_primary)

    def test_purer_secondary_default(self):
        self.assertIsNone(self.seg.purer_secondary)

    def test_purer_confidence_primary_default(self):
        self.assertIsNone(self.seg.purer_confidence_primary)

    def test_purer_confidence_secondary_default(self):
        self.assertIsNone(self.seg.purer_confidence_secondary)

    def test_purer_justification_default(self):
        self.assertIsNone(self.seg.purer_justification)

    def test_purer_run_consistency_default(self):
        self.assertIsNone(self.seg.purer_run_consistency)

    def test_purer_agreement_level_default(self):
        self.assertIsNone(self.seg.purer_agreement_level)

    def test_purer_agreement_fraction_default(self):
        self.assertIsNone(self.seg.purer_agreement_fraction)

    def test_purer_needs_review_default(self):
        self.assertFalse(self.seg.purer_needs_review)

    def test_purer_rater_ids_default(self):
        self.assertIsNone(self.seg.purer_rater_ids)

    def test_purer_rater_votes_default(self):
        self.assertIsNone(self.seg.purer_rater_votes)

    # -- GNN fields ----------------------------------------------------------

    def test_gnn_vaamr_pred_default(self):
        self.assertIsNone(self.seg.gnn_vaamr_pred)

    def test_gnn_vaamr_conf_default(self):
        self.assertIsNone(self.seg.gnn_vaamr_conf)

    def test_gnn_purer_pred_default(self):
        self.assertIsNone(self.seg.gnn_purer_pred)

    def test_gnn_purer_conf_default(self):
        self.assertIsNone(self.seg.gnn_purer_conf)

    def test_gnn_label_source_default(self):
        self.assertIsNone(self.seg.gnn_label_source)

    def test_gnn_fields_are_all_none(self):
        """Convenience: all five GNN fields are None on a fresh Segment."""
        for attr in ('gnn_vaamr_pred', 'gnn_vaamr_conf',
                     'gnn_purer_pred', 'gnn_purer_conf', 'gnn_label_source'):
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(self.seg, attr))

    # -- Validation fields ---------------------------------------------------

    def test_human_label_default(self):
        self.assertIsNone(self.seg.human_label)

    def test_human_secondary_label_default(self):
        self.assertIsNone(self.seg.human_secondary_label)

    def test_adjudicated_label_default(self):
        self.assertIsNone(self.seg.adjudicated_label)

    def test_in_human_coded_subset_default(self):
        self.assertFalse(self.seg.in_human_coded_subset)

    def test_label_status_default(self):
        self.assertEqual(self.seg.label_status, "llm_only")

    # -- Final training label fields -----------------------------------------

    def test_final_label_default(self):
        self.assertIsNone(self.seg.final_label)

    def test_final_label_source_default(self):
        self.assertIsNone(self.seg.final_label_source)

    def test_label_confidence_tier_default(self):
        self.assertIsNone(self.seg.label_confidence_tier)

    # -- Conversational segmenter fields -------------------------------------

    def test_speakers_in_segment_default(self):
        self.assertIsNone(self.seg.speakers_in_segment)

    def test_session_file_default(self):
        self.assertEqual(self.seg.session_file, "")


class TestSegmentMutableDefaultIsolation(unittest.TestCase):
    """
    Optional[List] and Optional[Dict] fields default to None (not a shared
    mutable container). Assigning a list/dict to one instance must not affect
    another.
    """

    def _check_isolation(self, attr, value):
        """Assign *value* to seg1.attr and verify seg2.attr is still None."""
        seg1 = Segment()
        seg2 = Segment()
        setattr(seg1, attr, value)
        self.assertIsNone(getattr(seg2, attr),
                          f"{attr} on seg2 was unexpectedly mutated")

    def test_rater_ids_isolation(self):
        self._check_isolation('rater_ids', ['run_1', 'run_2'])

    def test_rater_votes_isolation(self):
        self._check_isolation('rater_votes', [{'rater': 'r', 'vote': 'CODED'}])

    def test_codebook_labels_embedding_isolation(self):
        self._check_isolation('codebook_labels_embedding', ['affect_x'])

    def test_codebook_labels_llm_isolation(self):
        self._check_isolation('codebook_labels_llm', ['present_moment'])

    def test_codebook_labels_ensemble_isolation(self):
        self._check_isolation('codebook_labels_ensemble', ['body_awareness'])

    def test_codebook_disagreements_isolation(self):
        self._check_isolation('codebook_disagreements', ['code_x'])

    def test_codebook_confidence_isolation(self):
        self._check_isolation('codebook_confidence', {'affect_x': 0.9})

    def test_purer_rater_ids_isolation(self):
        self._check_isolation('purer_rater_ids', ['model_a'])

    def test_purer_rater_votes_isolation(self):
        self._check_isolation('purer_rater_votes', [{'rater': 'model_a', 'vote': 'CODED'}])

    def test_speakers_in_segment_isolation(self):
        self._check_isolation('speakers_in_segment', ['participant', 'therapist'])


class TestSegmentConstruction(unittest.TestCase):
    """Segment correctly stores values provided at construction time."""

    def test_keyword_arguments(self):
        seg = Segment(
            segment_id='seg_001',
            session_id='c1s3',
            speaker='participant',
            text='I notice pain in my lower back.',
            primary_stage=2,
            llm_confidence_primary=0.87,
        )
        self.assertEqual(seg.segment_id, 'seg_001')
        self.assertEqual(seg.session_id, 'c1s3')
        self.assertEqual(seg.speaker, 'participant')
        self.assertEqual(seg.text, 'I notice pain in my lower back.')
        self.assertEqual(seg.primary_stage, 2)
        self.assertAlmostEqual(seg.llm_confidence_primary, 0.87)

    def test_gnn_fields_can_be_set(self):
        seg = Segment(
            gnn_vaamr_pred=3,
            gnn_vaamr_conf=0.92,
            gnn_purer_pred=1,
            gnn_purer_conf=0.78,
            gnn_label_source='gnn_trained',
        )
        self.assertEqual(seg.gnn_vaamr_pred, 3)
        self.assertAlmostEqual(seg.gnn_vaamr_conf, 0.92)
        self.assertEqual(seg.gnn_purer_pred, 1)
        self.assertAlmostEqual(seg.gnn_purer_conf, 0.78)
        self.assertEqual(seg.gnn_label_source, 'gnn_trained')

    def test_purer_fields_can_be_set(self):
        seg = Segment(
            speaker='therapist',
            purer_primary=0,
            purer_secondary=4,
            purer_confidence_primary=0.9,
            purer_agreement_level='unanimous',
            purer_needs_review=False,
        )
        self.assertEqual(seg.purer_primary, 0)
        self.assertEqual(seg.purer_secondary, 4)
        self.assertAlmostEqual(seg.purer_confidence_primary, 0.9)
        self.assertEqual(seg.purer_agreement_level, 'unanimous')
        self.assertFalse(seg.purer_needs_review)

    def test_needs_review_can_be_set_true(self):
        seg = Segment(needs_review=True)
        self.assertTrue(seg.needs_review)

    def test_label_status_can_be_overridden(self):
        seg = Segment(label_status='human_adjudicated')
        self.assertEqual(seg.label_status, 'human_adjudicated')

    def test_boolean_defaults_are_actual_bools(self):
        seg = Segment()
        self.assertIs(type(seg.needs_review), bool)
        self.assertIs(type(seg.tie_broken_by_confidence), bool)
        self.assertIs(type(seg.purer_needs_review), bool)
        self.assertIs(type(seg.in_human_coded_subset), bool)


class TestSegmentFieldGroups(unittest.TestCase):
    """Holistic tests confirming field groups are complete and coherent."""

    def test_all_optional_int_fields_default_none(self):
        """Every Optional[int] field is None by default."""
        seg = Segment()
        optional_int_fields = [
            'cohort_id', 'primary_stage', 'secondary_stage',
            'llm_run_consistency', 'purer_primary', 'purer_secondary',
            'purer_run_consistency', 'gnn_vaamr_pred', 'gnn_purer_pred',
            'human_label', 'human_secondary_label', 'adjudicated_label',
            'final_label',
        ]
        for attr in optional_int_fields:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(seg, attr))

    def test_all_optional_float_fields_default_none(self):
        """Every Optional[float] field is None by default."""
        seg = Segment()
        optional_float_fields = [
            'llm_confidence_primary', 'llm_confidence_secondary',
            'agreement_fraction', 'secondary_agreement_fraction',
            'purer_confidence_primary', 'purer_confidence_secondary',
            'purer_agreement_fraction',
            'gnn_vaamr_conf', 'gnn_purer_conf',
            'label_confidence_tier',  # actually Optional[str] — skip
        ]
        # Filter out non-float ones
        float_fields = [
            'llm_confidence_primary', 'llm_confidence_secondary',
            'agreement_fraction', 'secondary_agreement_fraction',
            'purer_confidence_primary', 'purer_confidence_secondary',
            'purer_agreement_fraction', 'gnn_vaamr_conf', 'gnn_purer_conf',
        ]
        for attr in float_fields:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(seg, attr))

    def test_all_optional_str_fields_default_none(self):
        seg = Segment()
        optional_str_fields = [
            'llm_justification', 'agreement_level', 'consensus_vote',
            'secondary_agreement_level', 'purer_justification',
            'purer_agreement_level', 'gnn_label_source',
            'final_label_source', 'label_confidence_tier',
        ]
        for attr in optional_str_fields:
            with self.subTest(attr=attr):
                self.assertIsNone(getattr(seg, attr))

    def test_string_defaults_are_empty_strings(self):
        """Non-optional string fields default to empty string."""
        seg = Segment()
        empty_str_fields = [
            'segment_id', 'trial_id', 'participant_id',
            'session_id', 'speaker', 'text', 'session_variant',
            'session_file',
        ]
        for attr in empty_str_fields:
            with self.subTest(attr=attr):
                self.assertEqual(getattr(seg, attr), "")

    def test_int_defaults_are_zero(self):
        """Non-optional int fields default to 0."""
        seg = Segment()
        zero_int_fields = [
            'session_number', 'segment_index',
            'start_time_ms', 'end_time_ms',
            'total_segments_in_session', 'word_count',
        ]
        for attr in zero_int_fields:
            with self.subTest(attr=attr):
                self.assertEqual(getattr(seg, attr), 0)


if __name__ == '__main__':
    unittest.main()
