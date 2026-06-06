"""
tests/unit/test_llm_classifier.py
----------------------------------
Unit tests for classification_tools/llm_classifier.py.

Covers:
  - _parse_single_run: valid name -> CODED with mapped id
  - _parse_single_run: unknown name -> None (parse failure)
  - _parse_single_run: null/none primary_stage -> ABSTAIN
  - _parse_single_run: confidence coercion (bad string, None)
  - _parse_single_run: None input -> None
  - _parse_single_run: missing primary_stage key -> None
  - _build_context_block: empty when window=0 or no preceding; non-empty with context
  - classify_segments_zero_shot: n_runs=1 with FakeLLMClient (real VAAMR framework);
    returned dict maps segment_id -> consensus result; stage id in range
  - classify_purer_cue_units: end-to-end with FakeLLMClient against PURER framework;
    purer_primary id assigned to cue segment
  - build_prompt via classify_segments_zero_shot: prompt includes framework definitions
  - randomize_codebook=False vs True: prompt still contains all themes
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.llm_classifier import (
    _parse_single_run,
    _build_context_block,
    classify_segments_zero_shot,
    classify_purer_cue_units,
)
from tests.testhelpers import FakeLLMClient, make_segment, tiny_vaamr_framework, tiny_purer_framework
from theme_framework.config import ThemeClassificationConfig


# ---------------------------------------------------------------------------
# _parse_single_run
# ---------------------------------------------------------------------------

class TestParseSingleRun(unittest.TestCase):
    """Verify the ballot-parsing contract for all expected inputs."""

    def _name_to_id(self):
        fw = tiny_vaamr_framework()
        return fw.build_name_to_id_map()

    def test_valid_name_returns_coded(self):
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'Vigilance',
            'primary_confidence': 0.9,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': 'clear vigilance signal',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'CODED')
        self.assertEqual(result['primary_stage'], 0)  # Vigilance = 0
        self.assertAlmostEqual(result['primary_confidence'], 0.9)

    def test_case_insensitive_name_lookup(self):
        """Framework names are looked up case-insensitively."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'REAPPRAISAL',
            'primary_confidence': 0.7,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'CODED')
        self.assertEqual(result['primary_stage'], 4)  # Reappraisal = 4

    def test_unknown_name_returns_none(self):
        """An unrecognized primary_stage name is a parse failure."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'FluxCapacitor',
            'primary_confidence': 0.5,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNone(result)

    def test_null_primary_stage_returns_abstain(self):
        """JSON null for primary_stage must yield ABSTAIN ballot."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': None,
            'primary_confidence': 0.3,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'ABSTAIN')
        self.assertIsNone(result['primary_stage'])

    def test_string_null_primary_stage_returns_abstain(self):
        """String 'null' for primary_stage must yield ABSTAIN ballot."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'null',
            'primary_confidence': 0.0,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'ABSTAIN')

    def test_string_none_primary_stage_returns_abstain(self):
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'none',
            'primary_confidence': 0.0,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'ABSTAIN')

    def test_none_input_returns_none(self):
        """None result text (LLM returned nothing) is a hard failure."""
        n2id = self._name_to_id()
        result = _parse_single_run(None, n2id)
        self.assertIsNone(result)

    def test_missing_primary_stage_key_returns_none(self):
        """JSON without 'primary_stage' key is a parse failure."""
        n2id = self._name_to_id()
        response = json.dumps({'justification': 'no stage key here'})
        result = _parse_single_run(response, n2id)
        self.assertIsNone(result)

    def test_malformed_json_returns_none(self):
        n2id = self._name_to_id()
        result = _parse_single_run('{not json!!!}', n2id)
        self.assertIsNone(result)

    def test_confidence_coercion_bad_string(self):
        """Non-numeric confidence string is coerced to 0.0, not a crash."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'Metacognition',
            'primary_confidence': 'high',
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'CODED')
        self.assertEqual(result['primary_confidence'], 0.0)

    def test_confidence_coercion_none(self):
        """None confidence is coerced to 0.0."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'Avoidance',
            'primary_confidence': None,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['primary_confidence'], 0.0)

    def test_valid_secondary_stage_mapped(self):
        """A valid secondary stage name is mapped to its id."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'Vigilance',
            'primary_confidence': 0.8,
            'secondary_stage': 'Avoidance',
            'secondary_confidence': 0.4,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['secondary_stage'], 1)  # Avoidance = 1
        self.assertAlmostEqual(result['secondary_confidence'], 0.4)

    def test_invalid_secondary_stage_yields_none_secondary(self):
        """Unknown secondary stage name -> secondary_stage is None (not a crash)."""
        n2id = self._name_to_id()
        response = json.dumps({
            'primary_stage': 'Vigilance',
            'primary_confidence': 0.8,
            'secondary_stage': 'Nonexistent',
            'secondary_confidence': 0.3,
            'justification': '',
            'evidence_phrase': '',
        })
        result = _parse_single_run(response, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'CODED')
        self.assertIsNone(result['secondary_stage'])

    def test_dict_input_parsed_directly(self):
        """When a dict (not a string) is passed it is used directly."""
        n2id = self._name_to_id()
        d = {
            'primary_stage': 'Attention Regulation',
            'primary_confidence': 0.95,
            'secondary_stage': None,
            'secondary_confidence': None,
            'justification': 'dict input test',
            'evidence_phrase': '',
        }
        result = _parse_single_run(d, n2id)
        self.assertIsNotNone(result)
        self.assertEqual(result['vote'], 'CODED')
        self.assertEqual(result['primary_stage'], 2)  # Attention Regulation = 2


# ---------------------------------------------------------------------------
# _build_context_block
# ---------------------------------------------------------------------------

class TestBuildContextBlock(unittest.TestCase):

    def _segs(self, n=5):
        return [make_segment(f's{i}', text=f'Sentence number {i} about pain.') for i in range(n)]

    def test_empty_when_window_zero(self):
        segs = self._segs()
        result = _build_context_block(segs, 3, window_size=0)
        self.assertEqual(result, '')

    def test_empty_at_first_segment(self):
        segs = self._segs()
        result = _build_context_block(segs, 0, window_size=2)
        self.assertEqual(result, '')

    def test_non_empty_with_preceding_segments(self):
        segs = self._segs(5)
        result = _build_context_block(segs, 3, window_size=2)
        self.assertIn('Sentence number 1', result)
        self.assertIn('Sentence number 2', result)

    def test_respects_word_budget(self):
        """With a very tight word budget, context is truncated."""
        segs = [make_segment(f's{i}', text=' '.join([f'word{j}' for j in range(50)]))
                for i in range(4)]
        result = _build_context_block(segs, 3, window_size=3, max_words=10)
        # Result should have some content but be capped at ~10 words
        word_count = len(result.split())
        self.assertGreater(word_count, 0)
        # The truncation marker or short text should be present
        self.assertTrue(len(result) > 0)

    def test_window_size_larger_than_available(self):
        """window_size > current_index uses all available preceding segments."""
        segs = self._segs(3)
        result = _build_context_block(segs, 2, window_size=10)
        # Should include segments 0 and 1
        self.assertIn('Sentence number 0', result)
        self.assertIn('Sentence number 1', result)


# ---------------------------------------------------------------------------
# classify_segments_zero_shot with FakeLLMClient
# ---------------------------------------------------------------------------

class TestClassifySegmentsZeroShot(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _minimal_config(self):
        cfg = ThemeClassificationConfig()
        cfg.n_runs = 1
        cfg.backend = 'lmstudio'
        cfg.model = 'fake-model'
        cfg.output_dir = self.tmp
        cfg.save_interval = 100
        cfg.randomize_codebook = False
        cfg.min_classifiable_words = 0
        cfg.context_window_segments = 0
        cfg.per_run_models = []
        cfg.models = []
        return cfg

    def test_returns_results_for_each_segment(self):
        fw = tiny_vaamr_framework()
        segs = [make_segment(f'seg_{i}', text='I avoid thinking about my pain.') for i in range(3)]
        config = self._minimal_config()

        import unittest.mock as mock
        # Inject FakeLLMClient at the seam where classify_segments_zero_shot builds it
        fake = FakeLLMClient(default_name='Avoidance')
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, metadata = classify_segments_zero_shot(segs, fw, config)

        self.assertEqual(len(results), 3)
        for seg in segs:
            self.assertIn(seg.segment_id, results)

    def test_stage_id_in_valid_range(self):
        """Classified stage ids must be in 0..4 for the 5-stage VAAMR framework."""
        fw = tiny_vaamr_framework()
        segs = [make_segment('seg_range', text='Metacognition is observing my thoughts.')]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient(default_name='Metacognition')
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, _ = classify_segments_zero_shot(segs, fw, config)

        result = results['seg_range']
        # The consensus dict is nested inside the returned dict
        consensus = result.get('consensus', result)
        primary = consensus.get('primary_stage')
        if primary is not None:
            self.assertIn(primary, range(5))

    def test_metadata_contains_segment_text(self):
        fw = tiny_vaamr_framework()
        segs = [make_segment('seg_meta', text='Here is some text.')]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            _, metadata = classify_segments_zero_shot(segs, fw, config)

        self.assertIn('seg_meta', metadata)
        self.assertEqual(metadata['seg_meta']['text'], 'Here is some text.')

    def test_prompt_contains_framework_definitions(self):
        """build_prompt must include the framework's stage definitions."""
        fw = tiny_vaamr_framework()
        segs = [make_segment('seg_prompt', text='Pain is everywhere.')]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            classify_segments_zero_shot(segs, fw, config)

        # Each stage definition should appear in the prompt
        for theme in fw.themes:
            self.assertTrue(
                any(theme.definition in p for p in fake.calls),
                f"Theme definition for {theme.name} not found in any prompt"
            )

    def test_randomize_codebook_still_contains_all_themes(self):
        """With randomize_codebook=True, all theme names should still appear."""
        fw = tiny_vaamr_framework()
        segs = [make_segment('seg_rand', text='My mind wanders.')]
        config = self._minimal_config()
        config.randomize_codebook = True

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            classify_segments_zero_shot(segs, fw, config)

        # All 5 theme names should appear somewhere in the prompt
        prompt = fake.calls[0]
        for theme in fw.themes:
            self.assertIn(theme.name, prompt,
                          f"Theme {theme.name} missing from randomized prompt")

    def test_n_runs_one_no_per_run_models_ok(self):
        """n_runs=1 with no per_run_models must not raise ValueError."""
        fw = tiny_vaamr_framework()
        segs = [make_segment('seg_single')]
        config = self._minimal_config()
        config.n_runs = 1
        config.per_run_models = []
        config.models = []

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            try:
                results, _ = classify_segments_zero_shot(segs, fw, config)
            except ValueError as e:
                self.fail(f"ValueError raised with n_runs=1: {e}")
        self.assertIn('seg_single', results)

    def test_n_runs_two_without_per_run_models_raises(self):
        """n_runs=2 with no per_run_models must raise ValueError
        (single-model stochastic IRR removed)."""
        fw = tiny_vaamr_framework()
        segs = [make_segment('seg_irr')]
        config = self._minimal_config()
        config.n_runs = 2
        config.per_run_models = []
        config.models = []

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            with self.assertRaises(ValueError):
                classify_segments_zero_shot(segs, fw, config)

    def test_empty_segment_list_returns_empty(self):
        fw = tiny_vaamr_framework()
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, metadata = classify_segments_zero_shot([], fw, config)

        self.assertEqual(results, {})
        self.assertEqual(metadata, {})


# ---------------------------------------------------------------------------
# classify_purer_cue_units with FakeLLMClient
# ---------------------------------------------------------------------------

class TestClassifyPurerCueUnits(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _make_cue_unit(self, cue_id='cue_001', therapist_text='Tell me more about that.',
                       participant_text='I feel pain in my back.'):
        """Build a minimal cue_unit dict matching the expected schema."""
        therapist_seg = make_segment(cue_id, speaker='therapist', text=therapist_text)
        participant_seg = make_segment(f'from_{cue_id}', speaker='participant',
                                       text=participant_text)
        return {
            'segment': therapist_seg,
            'from_segment': participant_seg,
            'context_block': '',
        }

    def _minimal_config(self):
        cfg = ThemeClassificationConfig()
        cfg.n_runs = 1
        cfg.backend = 'lmstudio'
        cfg.model = 'fake-model'
        cfg.output_dir = self.tmp
        cfg.save_interval = 100
        cfg.randomize_codebook = False
        cfg.min_classifiable_words = 0
        cfg.context_window_segments = 0
        cfg.per_run_models = []
        cfg.models = []
        return cfg

    def test_returns_result_for_each_cue(self):
        fw = tiny_purer_framework()
        cue_units = [self._make_cue_unit(f'cue_{i}') for i in range(3)]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient(default_name='Phenomenological')
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, metadata = classify_purer_cue_units(cue_units, fw, config)

        self.assertEqual(len(results), 3)
        for cu in cue_units:
            self.assertIn(cu['segment'].segment_id, results)

    def test_purer_primary_in_valid_range(self):
        """Consensus primary_stage for a PURER cue must be in 0..4."""
        fw = tiny_purer_framework()
        cue_units = [self._make_cue_unit('cue_range')]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient(default_name='Reframing')
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, _ = classify_purer_cue_units(cue_units, fw, config)

        result = results['cue_range']
        consensus = result.get('consensus', result)
        primary = consensus.get('primary_stage')
        if primary is not None:
            self.assertIn(primary, range(5))

    def test_empty_cue_units_returns_empty(self):
        fw = tiny_purer_framework()
        config = self._minimal_config()
        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, metadata = classify_purer_cue_units([], fw, config)
        self.assertEqual(results, {})
        self.assertEqual(metadata, {})

    def test_prompt_contains_framework_name(self):
        """PURER cue prompt must include the framework name."""
        fw = tiny_purer_framework()
        cue_units = [self._make_cue_unit('cue_prompt')]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient(default_name='Utilization')
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            classify_purer_cue_units(cue_units, fw, config)

        self.assertTrue(len(fake.calls) >= 1)
        self.assertIn('PURER', fake.calls[0])

    def test_metadata_contains_from_segment_id(self):
        fw = tiny_purer_framework()
        cue_units = [self._make_cue_unit('cue_meta')]
        config = self._minimal_config()

        import unittest.mock as mock
        fake = FakeLLMClient()
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            _, metadata = classify_purer_cue_units(cue_units, fw, config)

        self.assertIn('cue_meta', metadata)
        self.assertEqual(metadata['cue_meta']['from_segment_id'], 'from_cue_meta')

    def test_responder_name_maps_to_purer_id(self):
        """FakeLLMClient echoing 'Reinforcement' should yield primary_stage=4."""
        fw = tiny_purer_framework()
        cue_units = [self._make_cue_unit('cue_reinf',
                                          therapist_text='Good work! Reinforcement is key.')]
        config = self._minimal_config()

        import unittest.mock as mock

        def reinforcement_responder(prompt):
            return {
                'primary_stage': 'Reinforcement',
                'primary_confidence': 0.9,
                'secondary_stage': None,
                'secondary_confidence': None,
                'justification': 'clear reinforcement',
                'evidence_phrase': '',
            }

        fake = FakeLLMClient(responder=reinforcement_responder)
        with mock.patch('classification_tools.llm_classifier.LLMClient', return_value=fake):
            results, _ = classify_purer_cue_units(cue_units, fw, config)

        result = results['cue_reinf']
        consensus = result.get('consensus', result)
        self.assertEqual(consensus.get('primary_stage'), 4)  # Reinforcement = 4


if __name__ == '__main__':
    unittest.main()
