"""
tests/unit/test_config_more.py
-------------------------------
Additional unit tests for process/config.py.

Covers gaps not already tested: PipelineConfig defaults, to_json/from_json
roundtrip with nested sub-configs (incl. gnn_layer), tolerance of unknown keys
and legacy 'pipeline'-nested format, and methodology-encoding sub-config defaults.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path: sys.path.insert(1, _QRA_ROOT)

from process.config import (
    PipelineConfig,
    ConfidenceTierConfig,
    EfficacyConfig,
    SuperpositionConfig,
    PurerCueConfig,
    PurerCueConfig,
    TherapistCueConfig,
    SegmentationConfig,
    ValidationConfig,
    TestSetSpec,
    TestSetsConfig,
    ContentValidityConfig,
    ContentValiditySpec,
    SessionSummariesConfig,
    ParticipantSummariesConfig,
)
from gnn_layer.config import GnnLayerConfig


class TestPipelineConfigDefaults(unittest.TestCase):
    """PipelineConfig() must reflect the correct methodology decisions."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_run_codebook_classifier_is_false(self):
        cfg = PipelineConfig()
        self.assertFalse(cfg.run_codebook_classifier)

    def test_therapist_framework_is_purer(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.therapist_framework, 'purer')

    def test_participant_framework_is_vaamr(self):
        cfg = PipelineConfig()
        self.assertEqual(cfg.participant_framework, 'vaamr')

    def test_gnn_layer_present_and_correct_type(self):
        cfg = PipelineConfig()
        self.assertIsInstance(cfg.gnn_layer, GnnLayerConfig)

    def test_run_theme_labeler_true(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.run_theme_labeler)

    def test_run_purer_labeler_true(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.run_purer_labeler)

    def test_auto_analyze_true(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.auto_analyze)

    def test_anonymize_transcript_text_true(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.anonymize_transcript_text)


class TestJsonRoundtrip(unittest.TestCase):
    """to_json() / from_json() must preserve all nested sub-config values."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_basic_roundtrip(self):
        cfg = PipelineConfig(transcript_dir='/a/b', output_dir='/c/d', trial_id='test')
        restored = PipelineConfig.from_json(cfg.to_json())
        self.assertEqual(restored.transcript_dir, '/a/b')
        self.assertEqual(restored.output_dir, '/c/d')
        self.assertEqual(restored.trial_id, 'test')

    def test_gnn_layer_non_defaults_roundtrip(self):
        cfg = PipelineConfig()
        cfg.gnn_layer.hidden_dim = 64
        cfg.gnn_layer.n_layers = 3
        cfg.gnn_layer.epochs = 50
        cfg.gnn_layer.gnn_authoritative = True

        data = cfg.to_json()
        restored = PipelineConfig.from_json(data)

        self.assertEqual(restored.gnn_layer.hidden_dim, 64)
        self.assertEqual(restored.gnn_layer.n_layers, 3)
        self.assertEqual(restored.gnn_layer.epochs, 50)
        self.assertTrue(restored.gnn_layer.gnn_authoritative)

    def test_confidence_tiers_roundtrip(self):
        cfg = PipelineConfig()
        cfg.confidence_tiers.high_consistency = 5
        cfg.confidence_tiers.high_confidence = 0.9

        data = cfg.to_json()
        restored = PipelineConfig.from_json(data)

        self.assertEqual(restored.confidence_tiers.high_consistency, 5)
        self.assertEqual(restored.confidence_tiers.high_confidence, 0.9)

    def test_purer_cue_roundtrip(self):
        cfg = PipelineConfig()
        cfg.purer_cue.skip_lesson_content = False
        cfg.purer_cue.max_lesson_words = 200
        cfg.purer_cue.max_cue_words = 150

        data = cfg.to_json()
        restored = PipelineConfig.from_json(data)

        self.assertFalse(restored.purer_cue.skip_lesson_content)
        self.assertEqual(restored.purer_cue.max_lesson_words, 200)
        self.assertEqual(restored.purer_cue.max_cue_words, 150)

    def test_efficacy_roundtrip(self):
        cfg = PipelineConfig()
        cfg.efficacy.barrier_from = 0
        cfg.efficacy.barrier_to = 3
        cfg.efficacy.adaptive_stages = [3, 4]
        cfg.efficacy.maladaptive_stages = [0]

        data = cfg.to_json()
        restored = PipelineConfig.from_json(data)

        self.assertEqual(restored.efficacy.barrier_from, 0)
        self.assertEqual(restored.efficacy.barrier_to, 3)
        self.assertEqual(restored.efficacy.adaptive_stages, [3, 4])
        self.assertEqual(restored.efficacy.maladaptive_stages, [0])

    def test_superposition_roundtrip(self):
        cfg = PipelineConfig()
        cfg.superposition.liminal_entropy_threshold = 0.75
        cfg.superposition.active_stage_threshold = 0.20

        data = cfg.to_json()
        restored = PipelineConfig.from_json(data)

        self.assertAlmostEqual(restored.superposition.liminal_entropy_threshold, 0.75)
        self.assertAlmostEqual(restored.superposition.active_stage_threshold, 0.20)

    def test_json_is_serializable(self):
        """to_json() output must be round-trippable through json.dumps/loads."""
        cfg = PipelineConfig()
        data = cfg.to_json()
        serialized = json.dumps(data)
        loaded = json.loads(serialized)
        restored = PipelineConfig.from_json(loaded)
        self.assertEqual(restored.therapist_framework, 'purer')

    def test_api_key_blanked_in_json(self):
        """Secret api_key fields must be blanked, never written."""
        cfg = PipelineConfig()
        cfg.theme_classification.api_key = 'super-secret-key'
        data = cfg.to_json()
        serialized = json.dumps(data)
        self.assertNotIn('super-secret-key', serialized)


class TestFromJsonToleratesUnknownKeys(unittest.TestCase):
    """from_json() must silently ignore unknown keys (forward-compatibility)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_unknown_top_level_key_ignored(self):
        cfg = PipelineConfig()
        data = cfg.to_json()
        data['totally_unknown_future_field'] = 'some_value'
        # Must not raise
        restored = PipelineConfig.from_json(data)
        self.assertEqual(restored.therapist_framework, 'purer')

    def test_unknown_key_in_sub_config_ignored(self):
        cfg = PipelineConfig()
        data = cfg.to_json()
        data['gnn_layer']['future_gnn_option'] = True
        restored = PipelineConfig.from_json(data)
        self.assertIsInstance(restored.gnn_layer, GnnLayerConfig)

    def test_unknown_key_in_confidence_tiers_ignored(self):
        cfg = PipelineConfig()
        data = cfg.to_json()
        data['confidence_tiers']['future_tier_level'] = 99
        restored = PipelineConfig.from_json(data)
        self.assertEqual(restored.confidence_tiers.high_consistency, 3)


class TestFromJsonLegacyPipelineFormat(unittest.TestCase):
    """from_json() must handle the legacy 'pipeline'-nested dict format."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_pipeline_nested_format_is_flattened(self):
        """Keys nested under 'pipeline' must be promoted to top level."""
        legacy_data = {
            'pipeline': {
                'transcript_dir': '/legacy/input',
                'output_dir': '/legacy/output',
                'trial_id': 'legacy_trial',
            }
        }
        cfg = PipelineConfig.from_json(legacy_data)
        self.assertEqual(cfg.transcript_dir, '/legacy/input')
        self.assertEqual(cfg.output_dir, '/legacy/output')
        self.assertEqual(cfg.trial_id, 'legacy_trial')

    def test_pipeline_nested_with_extra_top_level(self):
        """Top-level keys outside 'pipeline' should also be respected."""
        legacy_data = {
            'trial_id': 'outer',
            'pipeline': {
                'transcript_dir': '/leg/in',
                'trial_id': 'inner',
            }
        }
        cfg = PipelineConfig.from_json(legacy_data)
        # 'inner' from pipeline dict overwrites 'outer' after merge
        self.assertEqual(cfg.trial_id, 'inner')


class TestConfidenceTierConfigDefaults(unittest.TestCase):
    """ConfidenceTierConfig defaults encode the methodology's tier thresholds."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_high_consistency(self):
        self.assertEqual(ConfidenceTierConfig().high_consistency, 3)

    def test_high_confidence(self):
        self.assertAlmostEqual(ConfidenceTierConfig().high_confidence, 0.8)

    def test_medium_min_consistency(self):
        self.assertEqual(ConfidenceTierConfig().medium_min_consistency, 2)

    def test_medium_min_confidence(self):
        self.assertAlmostEqual(ConfidenceTierConfig().medium_min_confidence, 0.6)


class TestEfficacyConfigDefaults(unittest.TestCase):
    """EfficacyConfig defaults encode the VAAMR barrier / adaptive-stage definitions."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_barrier_from_is_avoidance(self):
        self.assertEqual(EfficacyConfig().barrier_from, 1)

    def test_barrier_to_is_attention_regulation(self):
        self.assertEqual(EfficacyConfig().barrier_to, 2)

    def test_adaptive_stages(self):
        self.assertEqual(EfficacyConfig().adaptive_stages, [2, 3, 4])

    def test_maladaptive_stages(self):
        self.assertEqual(EfficacyConfig().maladaptive_stages, [0, 1])

    def test_enabled_true(self):
        self.assertTrue(EfficacyConfig().enabled)


class TestSuperpositionConfigDefaults(unittest.TestCase):
    """SuperpositionConfig defaults encode the liminality / mixture thresholds."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_liminal_entropy_threshold(self):
        self.assertAlmostEqual(SuperpositionConfig().liminal_entropy_threshold, 0.6)

    def test_liminal_gap_threshold(self):
        self.assertAlmostEqual(SuperpositionConfig().liminal_gap_threshold, 0.25)

    def test_active_stage_threshold(self):
        self.assertAlmostEqual(SuperpositionConfig().active_stage_threshold, 0.15)

    def test_enabled_true(self):
        self.assertTrue(SuperpositionConfig().enabled)

    def test_mixture_source_auto(self):
        self.assertEqual(SuperpositionConfig().mixture_source, 'auto')

    def test_run_mechanism_analysis_true(self):
        self.assertTrue(SuperpositionConfig().run_mechanism_analysis)


class TestPurerCueConfigDefaults(unittest.TestCase):
    """PurerCueConfig defaults encode the cue-block PURER classification behaviour."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skip_lesson_content_true(self):
        self.assertTrue(PurerCueConfig().skip_lesson_content)

    def test_max_lesson_words(self):
        self.assertEqual(PurerCueConfig().max_lesson_words, 400)

    def test_max_cue_words(self):
        self.assertEqual(PurerCueConfig().max_cue_words, 300)

    def test_therapist_max_gap_seconds(self):
        self.assertAlmostEqual(PurerCueConfig().therapist_max_gap_seconds, 120.0)

    def test_max_context_words(self):
        self.assertEqual(PurerCueConfig().max_context_words, 1000)


class TestFromDictIsAlias(unittest.TestCase):
    """from_dict() is a documented alias for from_json()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_from_dict_alias(self):
        cfg = PipelineConfig(trial_id='alias_test')
        restored = PipelineConfig.from_dict(cfg.to_dict())
        self.assertEqual(restored.trial_id, 'alias_test')


class TestTestSetsAndContentValidityDefaults(unittest.TestCase):
    """Check default-constructed test set configs are coherent."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_test_sets_vaamr_enabled_by_default(self):
        cfg = PipelineConfig()
        self.assertTrue(cfg.test_sets.vaamr.enabled)

    def test_test_sets_purer_disabled_by_default(self):
        cfg = PipelineConfig()
        self.assertFalse(cfg.test_sets.purer.enabled)

    def test_test_sets_codebook_disabled_by_default(self):
        cfg = PipelineConfig()
        self.assertFalse(cfg.test_sets.codebook.enabled)

    def test_any_enabled_true_when_vaamr_on(self):
        ts = TestSetsConfig()
        self.assertTrue(ts.any_enabled())

    def test_content_validity_any_enabled_true_by_default(self):
        # CV vaamr defaults to enabled (cv_vaamr_v1) so any_enabled() is True;
        # see ContentValidityConfig in process/config.py.
        cv = ContentValidityConfig()
        self.assertTrue(cv.any_enabled())

    def test_test_sets_roundtrip(self):
        cfg = PipelineConfig()
        cfg.test_sets.purer.enabled = True
        cfg.test_sets.purer.name = 'purer_testset_v2'
        restored = PipelineConfig.from_json(cfg.to_json())
        self.assertTrue(restored.test_sets.purer.enabled)
        self.assertEqual(restored.test_sets.purer.name, 'purer_testset_v2')


if __name__ == '__main__':
    unittest.main()
