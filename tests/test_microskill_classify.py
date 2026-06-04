"""
tests/test_microskill_classify.py
---------------------------------
Tests for the therapist microcounseling-skill classifier stage (VCE mirror).

Uses the config=None path (pre-populated microskill fields) so no embedding model or
LLM backend is needed — mirroring test_orchestrator_stages / test_zero_shot_classify.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from classification_tools.data_structures import Segment
from process.orchestrator import stage_classify_microskill
from process import classifications_io as cio
from process import output_paths as paths
from process.assembly.master_dataset import assemble_master_dataset


def _segs():
    return [
        Segment(segment_id='s1', speaker='therapist', text='What did you notice?',
                microskill_labels_embedding=['open_ended_question'],
                microskill_labels_llm=['open_ended_question'],
                microskill_labels_ensemble=['open_ended_question'],
                microskill_disagreements=[],
                microskill_confidence={'open_ended_question': 0.9}),
        Segment(segment_id='s2', speaker='participant', text='I felt the pain ease.',
                primary_stage=2, final_label=2),
    ]


class TestMicroskillCodebook(unittest.TestCase):
    def test_codebook_parses(self):
        from codebook.microcounseling_codebook import get_microcounseling_codebook
        cb = get_microcounseling_codebook()
        self.assertEqual(len(cb.codes), 8)
        ids = {c.code_id for c in cb.codes}
        self.assertIn('open_ended_question', ids)
        self.assertIn('reflective_listening', ids)
        for c in cb.codes:
            self.assertEqual(len(c.exemplar_utterances), 4)
            self.assertTrue(c.inclusive_criteria and c.exclusive_criteria)


class TestMicroskillStage(unittest.TestCase):
    def test_overlay_written_and_roundtrips(self):
        d = tempfile.mkdtemp()
        stage_classify_microskill(None, None, segments=_segs(), output_dir=d)
        ov = paths.classification_overlay_path(d, 'microskill')
        self.assertTrue(os.path.isfile(ov))
        self.assertTrue(ov.endswith('microskill_labels.jsonl'))
        # round-trip onto fresh segments
        fresh = {'s1': Segment(segment_id='s1', speaker='therapist'),
                 's2': Segment(segment_id='s2', speaker='participant')}
        n = cio.apply_microskill_overlay(d, fresh)
        self.assertEqual(n, 2)
        self.assertEqual(fresh['s1'].microskill_labels_ensemble, ['open_ended_question'])

    def test_manifest_records_microskill(self):
        d = tempfile.mkdtemp()
        stage_classify_microskill(None, None, segments=_segs(), output_dir=d)
        man = cio.read_classification_manifest(d)
        self.assertIn('microskill', man)

    def test_overlay_key_registered(self):
        self.assertIn('microskill', cio.OVERLAY_KEYS)
        self.assertIn('microskill', cio._OVERLAY_FIELDS_MAP)
        self.assertIn('microskill_labels_ensemble', cio.MICROSKILL_OVERLAY_FIELDS)


class TestMasterDatasetColumns(unittest.TestCase):
    def test_microskill_columns_present(self):
        d = tempfile.mkdtemp()
        out = os.path.join(d, 'master_segments.jsonl')
        df = assemble_master_dataset(_segs(), out)
        for col in ('microskill_labels_embedding', 'microskill_labels_llm',
                    'microskill_labels_ensemble', 'microskill_disagreements'):
            self.assertIn(col, df.columns)
        s1 = df[df['segment_id'] == 's1'].iloc[0]
        self.assertEqual(s1['microskill_labels_ensemble'], ['open_ended_question'])


if __name__ == '__main__':
    unittest.main()
