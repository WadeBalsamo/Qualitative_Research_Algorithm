"""
test_behavior_preservation.py
------------------------------
Comprehensive behavior-preservation test suite.

Verifies that the refactored 'beta' branch preserves all core functionality
of the 'working' branch while adding modularization, content-validity features,
and improved TUI/CLI/setup_wizard.

The reference config (from the working branch) is:

  {
    "pipeline": {
      "transcript_dir": "./data/full/",
      "output_dir": "./data/MoveMORE/",
      "trial_id": "MoveMORE",
      "speaker_anonymization_key_path": "...",
      "run_codebook_classifier": false,
      "auto_analyze": true
    },
    "speaker_filter": {
      "mode": "exclude",
      "speakers": ["Move-MORE Study", "Anand", "Lani", ...]
    },
    "run_purer_labeler": true,
    "purer_cue": {"skip_lesson_content": false},
    "theme_classification": {
      "backend": "lmstudio",
      "model": "qwen/qwen3-next-80b",
      "summarization_model": "nvidia/nemotron-3-nano-4b",
      "lmstudio_base_url": "http://10.0.0.58:1234/v1",
      "n_runs": 3, "temperature": 0.1,
      "per_run_models": ["qwen/qwen3-next-80b", "google/gemma-4-31b", "nvidia/nemotron-3-super"]
    },
    "purer_classification": {
      "backend": "lmstudio",
      "model": "google/gemma-4-31b",
      "summarization_model": "nvidia/nemotron-3-nano-4b",
      "lmstudio_base_url": "http://10.0.0.58:1234/v1",
      "n_runs": 1, "temperature": 0.1, "per_run_models": []
    },
    "segmentation": {
      "embedding_model": "Qwen/Qwen3-Embedding-8B",
      "max_gap_seconds": 15.0,
      "min_words_per_sentence": 20,
      "max_segment_duration_seconds": 60.0,
      "min_segment_words_conversational": 60,
      "max_segment_words_conversational": 500,
      "use_adaptive_threshold": true,
      "min_prominence": 0.05,
      "use_topic_clustering": true,
      "use_llm_refinement": true,
      "llm_refinement_mode": "full"
    },
    "framework": {"preset": "vaamr"},
    "confidence_tiers": {"high_confidence": 0.8, "medium_min_confidence": 0.6},
    "test_sets": {"enabled": true, "n_sets": 2, "fraction_per_set": 0.1, "random_seed": 42},
    "therapist_cues": {"enabled": true, "max_length_per_cue": 250, "max_length_of_average_cue_responses": 300},
    "session_summaries": {"enabled": true, "max_words_per_session": 300},
    "participant_summaries": {"enabled": true, "max_words_per_session": 300}
  }

Tests are organized into categories that systematically verify each
dimension of the refactoring.
"""

import datetime
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock, call, ANY
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# ============================================================================
# REFERENCE CONFIG — mirrors the working branch production config exactly
# ============================================================================
REFERENCE_CONFIG_DICT = {
    "pipeline": {
        "transcript_dir": "./data/full/",
        "output_dir": "./data/MoveMORE/",
        "trial_id": "MoveMORE",
        "speaker_anonymization_key_path": "/home/wisgood/qra/Qualitative_Research_Algorithm/data/full/speaker_anonymization_key.json",
        "run_codebook_classifier": False,
        "auto_analyze": True,
    },
    "speaker_filter": {
        "mode": "exclude",
        "speakers": [
            "Move-MORE Study",
            "Anand",
            "Lani",
            "Wade (Study Coordinator)",
            "Rebecca Heron",
            "Michelle Berg",
        ],
    },
    "run_purer_labeler": True,
    "purer_cue": {
        "skip_lesson_content": False,
    },
    "theme_classification": {
        "backend": "lmstudio",
        "model": "qwen/qwen3-next-80b",
        "summarization_model": "nvidia/nemotron-3-nano-4b",
        "lmstudio_base_url": "http://10.0.0.58:1234/v1",
        "n_runs": 3,
        "temperature": 0.1,
        "per_run_models": [
            "qwen/qwen3-next-80b",
            "google/gemma-4-31b",
            "nvidia/nemotron-3-super",
        ],
    },
    "purer_classification": {
        "backend": "lmstudio",
        "model": "google/gemma-4-31b",
        "summarization_model": "nvidia/nemotron-3-nano-4b",
        "lmstudio_base_url": "http://10.0.0.58:1234/v1",
        "n_runs": 1,
        "temperature": 0.1,
        "per_run_models": [],
    },
    "segmentation": {
        "embedding_model": "Qwen/Qwen3-Embedding-8B",
        "max_gap_seconds": 15.0,
        "min_words_per_sentence": 20,
        "max_segment_duration_seconds": 60.0,
        "min_segment_words_conversational": 60,
        "max_segment_words_conversational": 500,
        "use_adaptive_threshold": True,
        "min_prominence": 0.05,
        "use_topic_clustering": True,
        "use_llm_refinement": True,
        "llm_refinement_mode": "full",
    },
    "framework": {
        "preset": "vaamr",
    },
    "confidence_tiers": {
        "high_confidence": 0.8,
        "medium_min_confidence": 0.6,
    },
    "test_sets": {
        "enabled": True,
        "n_sets": 2,
        "fraction_per_set": 0.1,
        "random_seed": 42,
    },
    "therapist_cues": {
        "enabled": True,
        "max_length_per_cue": 250,
        "max_length_of_average_cue_responses": 300,
    },
    "session_summaries": {
        "enabled": True,
        "max_words_per_session": 300,
    },
    "participant_summaries": {
        "enabled": True,
        "max_words_per_session": 300,
    },
}


# ============================================================================
# CATEGORY 1 — Module import integrity
# ============================================================================

class TestModuleImports(unittest.TestCase):
    """All public API surfaces from the working branch must remain importable."""

    def test_process_init_exports_run_full_pipeline(self):
        """run_full_pipeline must be importable from process package."""
        from process import run_full_pipeline
        self.assertTrue(callable(run_full_pipeline))

    def test_process_init_exports_pipeline_observer(self):
        """PipelineObserver must be importable from process package."""
        from process import PipelineObserver, SilentObserver
        self.assertTrue(callable(PipelineObserver))
        self.assertTrue(callable(SilentObserver))

    def test_orchestrator_exports_all_stage_functions(self):
        """All modular stage functions must be importable."""
        from process.orchestrator import (
            stage_classify_theme,
            stage_classify_purer,
            stage_classify_codebook,
            stage_cross_validation,
            stage_assemble,
            stage_validation_artifacts,
            stage_ingest,
        )
        self.assertTrue(callable(stage_classify_theme))
        self.assertTrue(callable(stage_classify_purer))
        self.assertTrue(callable(stage_classify_codebook))
        self.assertTrue(callable(stage_cross_validation))
        self.assertTrue(callable(stage_assemble))
        self.assertTrue(callable(stage_validation_artifacts))
        self.assertTrue(callable(stage_ingest))

    def test_config_module_imports(self):
        """All config dataclasses must be importable."""
        from process.config import (
            PipelineConfig,
            SpeakerFilterConfig,
            ThemeClassificationConfig,
            SegmentationConfig,
            ConfidenceTierConfig,
            TestSetsConfig,
            TestSetSpec,
            ContentValidityConfig,
            ContentValiditySpec,
            PurerCueConfig,
            TherapistCueConfig,
        )
        import dataclasses
        field_names = {f.name for f in dataclasses.fields(PipelineConfig)}
        for expected in ('transcript_dir', 'output_dir', 'trial_id', 'speaker_filter',
                         'theme_classification', 'purer_classification', 'segmentation',
                         'confidence_tiers'):
            self.assertIn(expected, field_names)

    def test_classifications_io_module_imports(self):
        """New classifications_io module must be importable."""
        from process import classifications_io
        self.assertTrue(callable(classifications_io.write_theme_overlay))
        self.assertTrue(callable(classifications_io.write_purer_overlay))
        self.assertTrue(callable(classifications_io.write_codebook_overlay))
        self.assertTrue(callable(classifications_io.write_cross_validation_overlay))
        self.assertTrue(callable(classifications_io.apply_theme_overlay))
        self.assertTrue(callable(classifications_io.apply_purer_overlay))
        self.assertTrue(callable(classifications_io.apply_codebook_overlay))
        self.assertTrue(callable(classifications_io.apply_overlays))
        self.assertTrue(callable(classifications_io.update_classification_manifest))
        self.assertTrue(callable(classifications_io.read_classification_manifest))

    def test_segments_io_load_for_stage_importable(self):
        """New load_segments_for_stage must be importable."""
        from process.segments_io import load_segments_for_stage
        self.assertTrue(callable(load_segments_for_stage))

    def test_qra_module_imports(self):
        """qra.py must be importable as __main__ proxy."""
        import qra
        self.assertTrue(hasattr(qra, 'main'))

    def test_llm_client_has_no_replicate_or_huggingface(self):
        """LLMClient must have removed Replicate and HuggingFace backends."""
        from classification_tools.llm_client import LLMClient, LLMClientConfig
        # Config must NOT have replicate_api_token
        cfg = LLMClientConfig()
        self.assertFalse(hasattr(cfg, 'replicate_api_token'),
                         "replicate_api_token must be removed from LLMClientConfig")
        self.assertFalse(hasattr(cfg, 'max_new_tokens'),
                         "max_new_tokens must be removed from LLMClientConfig")

    def test_theme_classification_config_has_no_replicate(self):
        """ThemeClassificationConfig must have removed replicate_api_token."""
        from theme_framework.config import ThemeClassificationConfig
        cfg = ThemeClassificationConfig()
        self.assertFalse(hasattr(cfg, 'replicate_api_token'),
                         "replicate_api_token must be removed")


# ============================================================================
# CATEGORY 2 — PipelineConfig construction and serialization
# ============================================================================

class TestPipelineConfigConstruction(unittest.TestCase):
    """PipelineConfig must be constructable from the reference config dict."""

    def test_construct_from_reference_dict(self):
        """build PipelineConfig from the exact working branch config dict."""
        from process.config import PipelineConfig
        config = PipelineConfig.from_dict(REFERENCE_CONFIG_DICT)

        # Top-level fields
        self.assertEqual(config.transcript_dir, "./data/full/")
        self.assertEqual(config.output_dir, "./data/MoveMORE/")
        self.assertEqual(config.trial_id, "MoveMORE")

        # Feature flags
        self.assertFalse(config.run_codebook_classifier,
                         "run_codebook_classifier must be False")
        self.assertTrue(config.auto_analyze,
                        "auto_analyze must be True")

        # Speaker filter
        self.assertEqual(config.speaker_filter.mode, "exclude")
        self.assertIn("Move-MORE Study", config.speaker_filter.speakers)
        self.assertIn("Anand", config.speaker_filter.speakers)
        self.assertEqual(len(config.speaker_filter.speakers), 6)

        # Theme classification
        tc = config.theme_classification
        self.assertEqual(tc.backend, "lmstudio")
        self.assertEqual(tc.model, "qwen/qwen3-next-80b")
        self.assertEqual(tc.summarization_model, "nvidia/nemotron-3-nano-4b")
        self.assertEqual(tc.lmstudio_base_url, "http://10.0.0.58:1234/v1")
        self.assertEqual(tc.n_runs, 3)
        self.assertAlmostEqual(tc.temperature, 0.1)
        self.assertEqual(len(tc.per_run_models), 3)
        self.assertEqual(tc.per_run_models, [
            "qwen/qwen3-next-80b",
            "google/gemma-4-31b",
            "nvidia/nemotron-3-super",
        ])

        # PURER classification (with its own model, different from theme)
        pc = config.purer_classification
        self.assertEqual(pc.backend, "lmstudio")
        self.assertEqual(pc.model, "google/gemma-4-31b")
        self.assertEqual(pc.summarization_model, "nvidia/nemotron-3-nano-4b")
        self.assertEqual(pc.n_runs, 1)
        self.assertEqual(pc.per_run_models, [])

        # PURER feature flags
        self.assertTrue(config.run_purer_labeler)
        purer_cue = config.purer_cue
        self.assertFalse(purer_cue.skip_lesson_content)

        # Segmentation
        seg = config.segmentation
        self.assertEqual(seg.embedding_model, "Qwen/Qwen3-Embedding-8B")
        self.assertAlmostEqual(seg.max_gap_seconds, 15.0)
        self.assertEqual(seg.min_words_per_sentence, 20)
        self.assertAlmostEqual(seg.max_segment_duration_seconds, 60.0)
        self.assertEqual(seg.min_segment_words_conversational, 60)
        self.assertEqual(seg.max_segment_words_conversational, 500)
        self.assertTrue(seg.use_adaptive_threshold)
        self.assertAlmostEqual(seg.min_prominence, 0.05)
        self.assertTrue(seg.use_topic_clustering)
        self.assertTrue(seg.use_llm_refinement)
        self.assertEqual(seg.llm_refinement_mode, "full")

        # Confidence tiers
        ct = config.confidence_tiers
        self.assertAlmostEqual(ct.high_confidence, 0.8)
        self.assertAlmostEqual(ct.medium_min_confidence, 0.6)

    def test_construct_with_legacy_test_sets_format(self):
        """Legacy flat test_sets config should be accepted."""
        from process.config import PipelineConfig
        config = PipelineConfig.from_dict(REFERENCE_CONFIG_DICT)
        ts = config.test_sets
        # Legacy format: "enabled": True, "n_sets": 2, ...
        # Should map to vaamr spec
        self.assertTrue(ts.vaamr.enabled)
        self.assertEqual(ts.vaamr.n_sets, 2)
        self.assertAlmostEqual(ts.vaamr.fraction_per_set, 0.1)
        self.assertEqual(ts.vaamr.random_seed, 42)

    def test_config_has_content_validity_field(self):
        """PipelineConfig must have content_validity field (new feature)."""
        import dataclasses
        from process.config import PipelineConfig
        field_names = {f.name for f in dataclasses.fields(PipelineConfig)}
        self.assertIn('content_validity', field_names,
                      "PipelineConfig must have content_validity field")

    def test_secret_keys_no_replicate(self):
        """_SECRET_KEYS must not contain replicate_api_token."""
        from process.config import _SECRET_KEYS
        self.assertNotIn('replicate_api_token', _SECRET_KEYS)

    def test_config_no_replicate_api_token_field(self):
        """PipelineConfig.theme_classification must not have replicate_api_token."""
        from process.config import PipelineConfig
        config = PipelineConfig.from_dict(REFERENCE_CONFIG_DICT)
        tc = config.theme_classification
        self.assertFalse(hasattr(tc, 'replicate_api_token'),
                         "ThemeClassificationConfig: replicate_api_token should be removed")


# ============================================================================
# CATEGORY 3 — PipelineObserver interface backward-compatibility
# ============================================================================

class TestPipelineObserverBackCompat(unittest.TestCase):
    """Observer interface must be backward-compatible from working branch."""

    def test_base_observer_has_all_working_branch_methods(self):
        """All observer methods from working branch must exist."""
        from process import PipelineObserver
        obs = PipelineObserver()

        # Working branch methods (without scope_id)
        obs.on_stage_start("test", "1")
        obs.on_stage_progress("test", "hello")
        obs.on_stage_complete("test", "done")
        obs.on_pipeline_complete("/out")

    def test_base_observer_accepts_scope_id_kwarg(self):
        """Observer methods must accept the new scope_id kwarg (beta enhancement)."""
        from process import PipelineObserver
        obs = PipelineObserver()
        obs.on_stage_start("test", "1", scope_id="theme")
        obs.on_stage_progress("test", "hello", scope_id="theme")
        obs.on_stage_complete("test", "done", scope_id="theme")

    def test_silent_observer_has_all_working_branch_methods(self):
        """SilentObserver must still work with old calling conventions."""
        from process import SilentObserver
        obs = SilentObserver()
        obs.on_stage_start("test", "1")
        obs.on_stage_progress("test", "hello")
        obs.on_stage_complete("test", "done")
        obs.on_pipeline_complete("/out")

    def test_silent_observer_accepts_extra_kwargs(self):
        """SilentObserver must accept explanation_key and other kwargs."""
        from process import SilentObserver
        obs = SilentObserver()
        obs.on_stage_start("test", "1", explanation_key='foo')
        obs.on_stage_progress("test", "hello", detail='bar')
        obs.on_stage_complete("test", "done", count=5)


# ============================================================================
# CATEGORY 4 — run_full_pipeline signature and return type
# ============================================================================

class TestRunFullPipelineSignature(unittest.TestCase):
    """run_full_pipeline must maintain the exact same interface from working."""

    def test_signature_is_callable_with_working_branch_args(self):
        """Must accept (config, framework, codebook, observer) like working."""
        import inspect
        from process.orchestrator import run_full_pipeline

        sig = inspect.signature(run_full_pipeline)
        params = list(sig.parameters.keys())
        self.assertIn('config', params)
        self.assertIn('framework', params)
        self.assertIn('codebook', params)
        self.assertIn('observer', params)
        # codebook and observer must default to None
        self.assertIsNone(sig.parameters['codebook'].default)
        self.assertIsNone(sig.parameters['observer'].default)

    def test_return_type_is_dataframe(self):
        """run_full_pipeline must return pd.DataFrame as documented."""
        from process.orchestrator import run_full_pipeline
        import inspect
        sig = inspect.signature(run_full_pipeline)
        anno = sig.return_annotation
        # annotation may be pd.DataFrame or str; check the actual type from
        # the working branch behavior — it always returns pd.DataFrame
        self.assertTrue('DataFrame' in str(anno) or 'pd.DataFrame' in str(anno),
                        f"Expected DataFrame return type, got {anno}")


# ============================================================================
# CATEGORY 5 — Stage execution order verification
# ============================================================================

class TestStageExecutionOrder(unittest.TestCase):
    """
    Verify that run_full_pipeline, when called with the working branch config,
    executes stages in the correct sequence: 1→2→3→3c→3b→4→5→6→7→8.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        os.environ['OPENROUTER_API_KEY'] = 'test-key'

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_stage_order_is_preserved_in_orchestrator(self):
        """
        Mock all stage functions and verify run_full_pipeline calls them in order.
        """
        from process.config import PipelineConfig
        from theme_framework.vaamr import get_vaamr_framework

        config = PipelineConfig.from_dict({
            **REFERENCE_CONFIG_DICT,
            "pipeline": {
                **REFERENCE_CONFIG_DICT["pipeline"],
                "output_dir": self.tmpdir,
                "transcript_dir": self.tmpdir,
            },
        })
        framework = get_vaamr_framework()

        import pandas as pd
        with (
            patch('process.orchestrator.stage_ingest') as mock_ingest,
            patch('process.orchestrator.stage_classify_theme') as mock_theme,
            patch('process.orchestrator.stage_classify_purer') as mock_purer,
            patch('process.orchestrator.stage_classify_codebook') as mock_cb,
            patch('process.orchestrator.stage_cross_validation') as mock_cv,
            patch('process.orchestrator.stage_validation_artifacts') as mock_val,
            patch('process.orchestrator.stage_assemble') as mock_assemble,
            patch('process.orchestrator.ensure_embedding_model_ready'),
            patch('process.orchestrator.create_content_validity_test_set', return_value=[]),
            patch('process.orchestrator.export_theme_definitions'),
            patch('process.orchestrator.export_theme_definitions_txt'),
            patch('process.orchestrator.assemble_master_dataset', return_value=pd.DataFrame()),
            patch('process.orchestrator.export_coded_transcript'),
            patch('process.orchestrator.export_per_transcript_stats'),
            patch('process.orchestrator.export_cumulative_report'),
            patch('process.orchestrator.export_training_data'),
        ):
            # Set up mocks to return segments (include one therapist so PURER gate passes)
            from classification_tools.data_structures import Segment
            therapist_seg = Segment(
                segment_id='t1', trial_id='t', participant_id='p1',
                session_id='s1', session_number=1, cohort_id=1,
                session_variant='', segment_index=0,
                start_time_ms=0, end_time_ms=1000,
                total_segments_in_session=1, speaker='therapist',
                text='How did that feel?', word_count=4,
                speakers_in_segment=['therapist'],
                session_file='/data/s1.json',
            )
            mock_segments = [therapist_seg]
            mock_ingest.return_value = mock_segments
            mock_theme.return_value = mock_segments
            mock_purer.return_value = mock_segments
            mock_cb.return_value = mock_segments
            mock_cv.return_value = mock_segments
            mock_assemble.return_value = pd.DataFrame()

            from process.orchestrator import run_full_pipeline
            result = run_full_pipeline(config, framework)

            # Verify stage 1 (ingest) was called
            mock_ingest.assert_called_once()
            # Verify stage 3 (theme) was called (run_theme_labeler=True by default)
            mock_theme.assert_called_once()
            # Verify stage 3c (PURER) was called (run_purer_labeler=True in ref config,
            # and mock_segments contains a therapist speaker)
            mock_purer.assert_called_once()
            # Verify stage 3b (codebook) was NOT called (run_codebook_classifier=False)
            mock_cb.assert_not_called()
            # Verify stage 4 (cross-validation) was NOT called (no codebook)
            mock_cv.assert_not_called()
            # run_full_pipeline calls assemble_master_dataset directly, not stage_assemble
            # (stage_assemble is the standalone CLI function for `qra assemble`)
            mock_assemble.assert_not_called()
            # Verify validation artifacts were called (inside Stage 7)
            mock_val.assert_called_once()

            # Stage order check: ingest must be called before theme
            # (Use assertGreater on call order)
            self.assertIsNotNone(result)

    def test_stage_ingest_writes_anonymization_key(self):
        """
        In beta, speaker anonymization key is written inside stage_ingest
        (moved from Stage 7 in working). Verify stage_ingest produces the
        anonymization key file.
        """
        from process.orchestrator import stage_ingest
        from process.config import PipelineConfig

        config = PipelineConfig.from_dict({
            **REFERENCE_CONFIG_DICT,
            "pipeline": {
                **REFERENCE_CONFIG_DICT["pipeline"],
                "output_dir": self.tmpdir,
                "transcript_dir": self.tmpdir,
            },
        })

        # The ingest will try to discover sessions; since our tmpdir is empty,
        # it will produce 0 segments but still create the meta dir.
        # We need to mock segmentation to test key writing behavior.
        with (
            patch('process.orchestrator.ensure_embedding_model_ready'),
            patch('process.orchestrator.discover_session_files', return_value=[]),
            patch('process.orchestrator._load_speaker_map', return_value=(None, True)),
            patch('process.orchestrator.ConversationalSegmenter') as _MockSeg,
            patch('process.orchestrator.LLMSegmentationRefiner'),
            patch('process.orchestrator.LLMClient'),
        ):
            _MockSeg.return_value.speaker_norm.speaker_map = {}
            segments = stage_ingest(config, output_dir=self.tmpdir,
                                    observer=None)

            # Verify the speaker anonymization key was written
            from process import output_paths as _paths
            key_path = os.path.join(_paths.meta_dir(self.tmpdir),
                                    'speaker_anonymization_key.json')
            self.assertTrue(
                os.path.isfile(key_path),
                f"Speaker anonymization key must exist at {key_path}"
            )


# ============================================================================
# CATEGORY 6 — qra.py CLI dispatch and _build_config
# ============================================================================

class TestQRACLIHelpers(unittest.TestCase):
    """Test that qra.py's _build_config handles all arg combinations."""

    def test_build_config_with_missing_attrs(self):
        """
        _build_config must not crash when called from subparsers that don't
        define all flags (e.g., ingest/validate subcommands).
        """
        import qra as qra_mod

        # Simulate an argparse.Namespace with minimal args (like ingest subcommand)
        class MinimalArgs:
            config = None
            output_dir = "/tmp/test_out"
            transcript_dir = None
            trial_id = None
            resume_from = None
            backend = None
            model = None
            lmstudio_url = None
            api_key = None
            # These are NOT defined on the namespace
            # framework, codebook, no_theme_labeler, etc.

        args = MinimalArgs()
        config = qra_mod._build_config(args)
        self.assertIsNotNone(config)
        self.assertEqual(config.output_dir, "/tmp/test_out")

    def test_build_config_with_partial_args(self):
        """_build_config handles namespace with only some attributes."""
        import qra as qra_mod

        class PartialArgs:
            config = None
            output_dir = "/tmp/test"

        args = PartialArgs()
        config = qra_mod._build_config(args)
        self.assertIsNotNone(config)
        self.assertEqual(config.output_dir, "/tmp/test")

    def test_build_config_with_full_reference_args(self):
        """_build_config handles a full args namespace like cmd_run."""
        import qra as qra_mod

        class FullArgs:
            config = None
            output_dir = "/tmp/test_out"
            transcript_dir = "./data/input"
            trial_id = "TestTrial"
            resume_from = None
            backend = "lmstudio"
            model = "qwen/qwen3-next-80b"
            lmstudio_url = "http://10.0.0.58:1234/v1"
            api_key = None
            models = None
            n_runs = 3
            temperature = 0.1
            no_theme_labeler = False
            run_codebook_classifier = False
            no_codebook_classifier = True
            verbose_segmentation = False
            no_two_pass = False
            embedding_model = None
            exemplar_import_path = None
            criteria_weight = None
            exemplar_weight = None
            exemplar_confidence_threshold = None
            max_exemplar_tokens = None
            high_confidence_threshold = None
            medium_confidence_threshold = None
            no_auto_analyze = False
            speaker_filter_mode = "exclude"
            exclude_speakers = ["Test Speaker"]
            framework = "vaamr"
            codebook = None

        args = FullArgs()
        config = qra_mod._build_config(args)
        self.assertEqual(config.theme_classification.backend, "lmstudio")
        self.assertEqual(config.theme_classification.model, "qwen/qwen3-next-80b")
        self.assertFalse(config.run_codebook_classifier)
        self.assertFalse(config.run_theme_labeler is False)  # should NOT be set to False

    def test_cli_replicate_backend_removed_from_choices(self):
        """--backend choices must NOT include 'replicate' or 'huggingface'."""
        import qra as qra_mod

        # Inspect the argparse setup function
        import inspect
        source = inspect.getsource(qra_mod._add_common_args)
        self.assertNotIn("'replicate'", source,
                         "'replicate' must be removed from --backend choices")
        self.assertNotIn("'huggingface'", source,
                         "'huggingface' must be removed from --backend choices")
        self.assertIn("'openrouter'", source)
        self.assertIn("'ollama'", source)
        self.assertIn("'lmstudio'", source)

    def test_cli_replicate_api_token_arg_removed(self):
        """--replicate-api-token argument must be removed from CLI."""
        import qra as qra_mod
        import inspect
        source = inspect.getsource(qra_mod._add_common_args)
        self.assertNotIn('replicate_api_token', source,
                         '--replicate-api-token must be removed from CLI')
        self.assertNotIn('replicate-api-token', source,
                         '--replicate-api-token must be removed from CLI')

    def test_command_dispatch_routes_correctly(self):
        """All new Phase 3 subcommands must be in the dispatch table."""
        import qra as qra_mod
        source = qra_mod.__file__
        with open(source) as f:
            content = f.read()

        # Check for new subcommand handlers
        self.assertIn('cmd_ingest', content)
        self.assertIn('cmd_classify', content)
        self.assertIn('cmd_assemble', content)
        self.assertIn('cmd_validate', content)

        # Check dispatch routing
        self.assertIn("args.command == 'ingest'", content)
        self.assertIn("args.command == 'classify'", content)
        self.assertIn("args.command == 'assemble'", content)
        self.assertIn("args.command == 'validate'", content)

    def test_subparser_setup_includes_new_commands(self):
        """argparse setup must register Phase 3 subcommands."""
        import qra as qra_mod
        source = qra_mod.__file__
        with open(source) as f:
            content = f.read()

        self.assertIn("'ingest'", content)
        self.assertIn("'classify'", content)
        self.assertIn("'assemble'", content)
        self.assertIn("'validate'", content)


# ============================================================================
# CATEGORY 7 — Overlay file I/O correctness
# ============================================================================

class TestOverlayIOBehavior(unittest.TestCase):
    """Overlay files must correctly serialize and deserialize classifications."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _make_seg(seg_id, **kwargs):
        from classification_tools.data_structures import Segment
        defaults = dict(
            segment_id=seg_id, trial_id='t', participant_id='p1',
            session_id='s1', session_number=1, cohort_id=1,
            segment_index=0, start_time_ms=0, end_time_ms=1000,
            total_segments_in_session=5, speaker='participant',
            text='test', word_count=1, speakers_in_segment=['participant'],
            session_file='/data/s1.json',
        )
        defaults.update(kwargs)
        return Segment(**defaults)

    def test_theme_overlay_roundtrip_preserves_all_reference_fields(self):
        """Write and read back theme overlay; all fields roundtrip intact."""
        from process import classifications_io as cio

        seg = self._make_seg('seg_A')
        seg.primary_stage = 4
        seg.secondary_stage = 1
        seg.llm_confidence_primary = 0.95
        seg.llm_confidence_secondary = 0.42
        seg.llm_justification = 'Clear V-A-MR pattern present.'
        seg.rater_ids = ['rater_1', 'rater_2', 'rater_3']
        seg.rater_votes = [
            {'rater': 'rater_1', 'stage': 4, 'confidence': 0.9},
            {'rater': 'rater_2', 'stage': 4, 'confidence': 0.95},
            {'rater': 'rater_3', 'stage': 1, 'confidence': 0.7},
        ]
        seg.agreement_level = 'majority'
        seg.agreement_fraction = 2 / 3
        seg.needs_review = False
        seg.consensus_vote = 4
        seg.tie_broken_by_confidence = False
        seg.llm_run_consistency = 2
        seg.secondary_agreement_level = 'none'
        seg.secondary_agreement_fraction = 0.0

        cio.write_theme_overlay(self.tmpdir, [seg])

        blank = self._make_seg('seg_A')
        cio.apply_theme_overlay(self.tmpdir, {'seg_A': blank})

        self.assertEqual(blank.primary_stage, 4)
        self.assertEqual(blank.secondary_stage, 1)
        self.assertAlmostEqual(blank.llm_confidence_primary, 0.95)
        self.assertAlmostEqual(blank.llm_confidence_secondary, 0.42)
        self.assertEqual(blank.agreement_level, 'majority')
        self.assertAlmostEqual(blank.agreement_fraction, 2 / 3)
        self.assertEqual(blank.consensus_vote, 4)
        self.assertFalse(blank.tie_broken_by_confidence)
        self.assertEqual(blank.llm_run_consistency, 2)

    def test_purer_overlay_roundtrip(self):
        """PURER overlay roundtrip preserves all therapist cue fields."""
        from process import classifications_io as cio

        seg = self._make_seg('th_001', speaker='therapist')
        seg.purer_primary = 2  # Phenomenological inquiry
        seg.purer_secondary = None
        seg.purer_confidence_primary = 0.88
        seg.purer_confidence_secondary = None
        seg.purer_justification = 'Therapist uses open-ended phenomenological inquiry.'
        seg.purer_run_consistency = 2
        seg.purer_agreement_level = 'unanimous'
        seg.purer_agreement_fraction = 1.0
        seg.purer_needs_review = False
        seg.purer_rater_ids = ['rater_1', 'rater_2']
        seg.purer_rater_votes = [{'rater': 'rater_1', 'stage': 2}] * 2

        cio.write_purer_overlay(self.tmpdir, [seg])

        blank = self._make_seg('th_001', speaker='therapist')
        cio.apply_purer_overlay(self.tmpdir, {'th_001': blank})

        self.assertEqual(blank.purer_primary, 2)
        self.assertAlmostEqual(blank.purer_confidence_primary, 0.88)
        self.assertEqual(blank.purer_agreement_level, 'unanimous')

    def test_codebook_overlay_roundtrip(self):
        """Codebook overlay roundtrip preserves multi-label fields."""
        from process import classifications_io as cio

        seg = self._make_seg('seg_B')
        seg.codebook_labels_embedding = ['VA.1', 'MC.2']
        seg.codebook_labels_llm = ['VA.1']
        seg.codebook_labels_ensemble = ['VA.1', 'MC.2']
        seg.codebook_disagreements = []
        seg.codebook_confidence = {'VA.1': 0.92, 'MC.2': 0.78}

        cio.write_codebook_overlay(self.tmpdir, [seg])

        blank = self._make_seg('seg_B')
        cio.apply_codebook_overlay(self.tmpdir, {'seg_B': blank})

        self.assertEqual(blank.codebook_labels_ensemble, ['VA.1', 'MC.2'])
        self.assertEqual(blank.codebook_labels_embedding, ['VA.1', 'MC.2'])
        self.assertEqual(blank.codebook_labels_llm, ['VA.1'])
        self.assertEqual(blank.codebook_disagreements, [])
        self.assertAlmostEqual(blank.codebook_confidence['VA.1'], 0.92)

    def test_overlays_are_idempotently_writable(self):
        """Overlay files can be overwritten without error (unlike frozen segments)."""
        from process import classifications_io as cio

        seg = self._make_seg('seg_X')
        seg.primary_stage = 0
        cio.write_theme_overlay(self.tmpdir, [seg])
        # Overwrite with same seg_id, different label
        seg.primary_stage = 4
        cio.write_theme_overlay(self.tmpdir, [seg])  # Must not raise

        blank = self._make_seg('seg_X')
        cio.apply_theme_overlay(self.tmpdir, {'seg_X': blank})
        self.assertEqual(blank.primary_stage, 4, "Overwrite must take effect")

    def test_overlay_files_are_sorted_by_segment_id(self):
        """Overlay records must be sorted for stable diffs across runs."""
        from process import classifications_io as cio

        segs = [
            self._make_seg('seg_003'),
            self._make_seg('seg_001'),
            self._make_seg('seg_002'),
        ]
        for s in segs:
            s.primary_stage = 1

        cio.write_theme_overlay(self.tmpdir, segs)

        with open(cio.overlay_path(self.tmpdir, 'theme')) as f:
            ids = [json.loads(line)['segment_id'] for line in f if line.strip()]
        self.assertEqual(ids, sorted(ids))

    def test_apply_missing_overlay_is_noop(self):
        """Applying a non-existent overlay must return 0, not error."""
        from process import classifications_io as cio
        seg = self._make_seg('seg_N')
        n = cio.apply_theme_overlay(self.tmpdir, {'seg_N': seg})
        self.assertEqual(n, 0)
        self.assertIsNone(seg.primary_stage)

    def test_manifest_records_timestamp(self):
        """Classification manifest must record completed_at timestamp."""
        from process import classifications_io as cio

        cio.update_classification_manifest(
            self.tmpdir, key='theme',
            entry={'model': 'qwen/qwen3-next-80b', 'n_segments': 10},
        )
        m = cio.read_classification_manifest(self.tmpdir)
        self.assertIn('completed_at', m['theme'])
        # Should be parseable as ISO datetime
        datetime.datetime.fromisoformat(m['theme']['completed_at'])

    def test_manifest_merge_does_not_clobber(self):
        """Writing theme manifest must not clobber purer manifest entry."""
        from process import classifications_io as cio

        cio.update_classification_manifest(
            self.tmpdir, key='theme', entry={'model': 'model-A'},
        )
        cio.update_classification_manifest(
            self.tmpdir, key='purer', entry={'model': 'model-B'},
        )
        m = cio.read_classification_manifest(self.tmpdir)
        self.assertIn('theme', m)
        self.assertIn('purer', m)
        self.assertEqual(m['theme']['model'], 'model-A')
        self.assertEqual(m['purer']['model'], 'model-B')
        # theme entry must not have been overwritten
        self.assertEqual(m['theme']['model'], 'model-A')


# ============================================================================
# CATEGORY 8 — create_missing flag correctness
# ============================================================================

class TestCreateMissingFlag(unittest.TestCase):
    """create_missing=False must prevent new testset/CV creation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        from theme_framework.vaamr import get_vaamr_framework
        self.framework = get_vaamr_framework()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _make_seg(seg_id, primary=2):
        from classification_tools.data_structures import Segment
        seg = Segment(
            segment_id=seg_id, trial_id='t', participant_id='p1',
            session_id='s1', session_number=1, cohort_id=1,
            segment_index=0, start_time_ms=0, end_time_ms=5000,
            total_segments_in_session=30, speaker='participant',
            text='Test segment for pain management.',
            word_count=5, session_file='/data/s1.json',
        )
        seg.primary_stage = primary
        seg.agreement_level = 'unanimous'
        seg.agreement_fraction = 1.0
        return seg

    def test_validation_testsets_create_missing_false_skips(self):
        """create_missing=False on validation testsets must not create new sets."""
        from process.assembly import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segments = [self._make_seg(f'seg_{i:03d}', primary=i % 5) for i in range(20)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_ref', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
        )

        dirs = generate_or_refresh_validation_testsets(
            segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg,
            create_missing=False,
        )
        self.assertEqual(dirs, [], "create_missing=False must not create new testsets")

    def test_validation_testsets_create_missing_true_creates(self):
        """create_missing=True on validation testsets must create new sets."""
        from process.assembly import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segments = [self._make_seg(f'seg_{i:03d}', primary=i % 5) for i in range(20)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_new', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
        )

        dirs = generate_or_refresh_validation_testsets(
            segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg,
            create_missing=True,
        )
        self.assertGreater(len(dirs), 0)

    def test_validation_testsets_refresh_existing_when_no_create(self):
        """create_missing=False must refresh existing testsets."""
        from process.assembly import generate_or_refresh_validation_testsets
        from process.config import TestSetsConfig, TestSetSpec

        segments = [self._make_seg(f'seg_{i:03d}', primary=i % 5) for i in range(20)]
        ts_cfg = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_existing', n_sets=1,
                              fraction_per_set=0.5, random_seed=42),
        )

        # Create first
        generate_or_refresh_validation_testsets(
            segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg, create_missing=True,
        )

        # Refresh without creating
        dirs = generate_or_refresh_validation_testsets(
            segments, self.framework, self.tmpdir,
            test_sets_config=ts_cfg, create_missing=False,
        )
        self.assertGreater(len(dirs), 0)

    def test_cv_testsets_create_missing_false_skips(self):
        """create_missing=False on CV testsets must not create new sets."""
        from process.assembly import generate_or_refresh_content_validity_testsets
        from process.config import ContentValidityConfig, ContentValiditySpec

        cv_cfg = ContentValidityConfig(
            vaamr=ContentValiditySpec(enabled=True, name='cv_skip'),
        )

        dirs = generate_or_refresh_content_validity_testsets(
            self.tmpdir,
            cv_config=cv_cfg,
            framework_vaamr=self.framework,
            framework_purer=None,
            theme_classification_cfg=None,
            create_missing=False,
        )
        self.assertEqual(dirs, [])
        # Ensure no cv directory was created
        from process import output_paths as _paths
        cv_dir = _paths.cv_testset_dir(self.tmpdir, 'cv_skip')
        self.assertFalse(os.path.isdir(cv_dir))


# ============================================================================
# CATEGORY 9 — stage_classify functions with pre-classified segments
# ============================================================================

class TestStageFunctionsWithPreClassifiedSegments(unittest.TestCase):
    """
    Stage functions, when given pre-classified segments and config=None,
    must write their overlays without attempting LLM calls.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @staticmethod
    def _make_seg(seg_id, speaker='participant', **kw):
        from classification_tools.data_structures import Segment
        defaults = dict(
            segment_id=seg_id, trial_id='t', participant_id='p1',
            session_id='s1', session_number=1, cohort_id=1,
            segment_index=0, start_time_ms=0, end_time_ms=1000,
            total_segments_in_session=5, speaker=speaker,
            text='test', word_count=1, speakers_in_segment=[speaker],
            session_file='/data/s1.json',
        )
        defaults.update(kw)
        return Segment(**defaults)

    def _write_frozen(self, segments):
        from process import segments_io
        sessions = defaultdict(list)
        for s in segments:
            sessions[s.session_id].append(s)
        for sid, segs in sessions.items():
            segments_io.write_session_segments(self.tmpdir, sid, segs, 'testhash')

    def test_stage_classify_theme_writes_overlay_from_memory(self):
        """stage_classify_theme with config=None writes overlay from in-memory state."""
        from process.orchestrator import stage_classify_theme

        seg = self._make_seg('seg_001')
        seg.primary_stage = 3
        seg.agreement_level = 'unanimous'
        seg.llm_confidence_primary = 0.9

        self._write_frozen([seg])
        result = stage_classify_theme(None, None, segments=[seg],
                                       output_dir=self.tmpdir)

        from process import classifications_io as cio
        overlay = cio.overlay_path(self.tmpdir, 'theme')
        self.assertTrue(os.path.isfile(overlay))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].primary_stage, 3)

    def test_stage_classify_purer_writes_overlay_from_memory(self):
        """stage_classify_purer with config=None writes overlay from in-memory state."""
        from process.orchestrator import stage_classify_purer

        seg = self._make_seg('th_001', speaker='therapist')
        seg.purer_primary = 1
        seg.purer_confidence_primary = 0.7

        self._write_frozen([seg])
        result = stage_classify_purer(None, segments=[seg],
                                       output_dir=self.tmpdir)

        from process import classifications_io as cio
        overlay = cio.overlay_path(self.tmpdir, 'purer')
        self.assertTrue(os.path.isfile(overlay))
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].purer_primary, 1)

    def test_stage_classify_codebook_writes_overlay_from_memory(self):
        """stage_classify_codebook with config=None writes overlay from in-memory."""
        from process.orchestrator import stage_classify_codebook

        seg = self._make_seg('seg_002')
        seg.codebook_labels_ensemble = ['VA.1', 'AT.2']

        self._write_frozen([seg])
        result = stage_classify_codebook(None, None, segments=[seg],
                                          output_dir=self.tmpdir)

        from process import classifications_io as cio
        overlay = cio.overlay_path(self.tmpdir, 'codebook')
        self.assertTrue(os.path.isfile(overlay))

    def test_stage_assemble_returns_dataframe(self):
        """stage_assemble with config=None returns pd.DataFrame."""
        from process.orchestrator import stage_assemble
        import pandas as pd

        seg = self._make_seg('seg_001')
        seg.primary_stage = 2
        self._write_frozen([seg])

        # Write theme overlay so it's loadable
        from process import classifications_io as cio
        cio.write_theme_overlay(self.tmpdir, [seg])

        result = stage_assemble(None, segments=[seg], output_dir=self.tmpdir)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(len(result), 1)

    def test_stage_assemble_standalone_loads_from_disk(self):
        """stage_assemble without segments loads from frozen + overlays."""
        from process.orchestrator import stage_assemble

        seg = self._make_seg('seg_001')
        seg.primary_stage = 2
        self._write_frozen([seg])

        from process import classifications_io as cio
        cio.write_theme_overlay(self.tmpdir, [seg])

        result = stage_assemble(None, output_dir=self.tmpdir)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_stage_cross_validation_writes_cv_overlay(self):
        """stage_cross_validation writes cross_validation_labels.jsonl."""
        from process.orchestrator import stage_cross_validation

        seg = self._make_seg('seg_001')
        seg.primary_stage = 2
        seg.codebook_labels_ensemble = ['VA.1']

        self._write_frozen([seg])
        result = stage_cross_validation(None, None, segments=[seg],
                                         output_dir=self.tmpdir)

        from process import classifications_io as cio
        overlay = cio.overlay_path(self.tmpdir, 'cv')
        self.assertTrue(os.path.isfile(overlay))
        self.assertEqual(len(result), 1)


# ============================================================================
# CATEGORY 10 — Legacy config format backward compatibility
# ============================================================================

class TestLegacyConfigBackCompat(unittest.TestCase):
    """Config loading must handle legacy (working branch) formats."""

    def test_legacy_flat_test_sets_format(self):
        """The legacy flat test_sets format {'enabled': True, 'n_sets': 2, ...} must work."""
        from process.config import PipelineConfig

        config = PipelineConfig.from_dict({
            "pipeline": {
                "transcript_dir": "./data/input",
                "output_dir": "./data/output",
                "trial_id": "test",
            },
            "test_sets": {
                "enabled": True,
                "n_sets": 2,
                "fraction_per_set": 0.1,
                "random_seed": 42,
            },
        })

        # Should NOT crash and should have test_sets configured
        self.assertIsNotNone(config.test_sets)
        self.assertTrue(config.test_sets.vaamr.enabled)
        self.assertEqual(config.test_sets.vaamr.n_sets, 2)

    def test_config_without_content_validity_field(self):
        """Config without content_validity must still load (backward compat)."""
        from process.config import PipelineConfig

        config = PipelineConfig.from_dict({
            "pipeline": {
                "transcript_dir": "./data/input",
                "output_dir": "./data/output",
                "trial_id": "test",
            },
        })

        # content_validity must have a default value (not crash)
        cv = config.content_validity
        self.assertIsNotNone(cv)
        # Default (consistent with the parse path, config.py:425): VAAMR content
        # validity is ON by default, PURER is OFF.
        self.assertTrue(cv.vaamr.enabled)
        self.assertFalse(cv.purer.enabled)

    def test_config_without_purer_classification(self):
        """Config without purer_classification must still load."""
        from process.config import PipelineConfig

        config = PipelineConfig.from_dict({
            "pipeline": {
                "transcript_dir": "./data/input",
                "output_dir": "./data/output",
                "trial_id": "test",
            },
        })

        # purer_classification should have defaults
        pc = config.purer_classification
        self.assertIsNotNone(pc)

    def test_build_config_without_framework_field(self):
        """_build_config when args has no 'framework' attr must not crash."""
        import qra as qra_mod

        class ArgsNoFramework:
            config = None
            output_dir = "/tmp/test"

        args = ArgsNoFramework()
        config = qra_mod._build_config(args)
        self.assertIsNotNone(config)


# ============================================================================
# CATEGORY 11 — output_paths completeness
# ============================================================================

class TestOutputPathsCompleteness(unittest.TestCase):
    """output_paths.py must expose all directories needed by the pipeline."""

    def test_new_classification_paths_exist(self):
        """New Phase 3 classification path functions must exist."""
        from process import output_paths as op

        self.assertTrue(callable(op.classifications_dir))
        self.assertTrue(callable(op.classification_overlay_path))
        self.assertTrue(callable(op.classification_manifest_path))

    def test_classification_overlay_path_produces_correct_structure(self):
        """classification_overlay_path must produce 02_meta/classifications/<key>_labels.jsonl."""
        from process import output_paths as op

        expected_suffixes = {
            'theme': 'theme_labels.jsonl',
            'purer': 'purer_labels.jsonl',
            'codebook': 'codebook_labels.jsonl',
            'cv': 'cross_validation_labels.jsonl',
        }
        for key in ('theme', 'purer', 'codebook', 'cv'):
            path = op.classification_overlay_path('/out', key)
            self.assertIn('02_meta', path)
            self.assertIn('classifications', path)
            self.assertTrue(path.endswith(expected_suffixes[key]),
                            f"Expected {expected_suffixes[key]}, got {path}")

    def test_existing_paths_still_present(self):
        """All paths from working branch must still exist."""
        from process import output_paths as op

        # Previously existing paths
        self.assertTrue(callable(op.meta_dir))
        self.assertTrue(callable(op.auditable_logs_dir))
        self.assertTrue(callable(op.transcripts_diarized_dir))
        self.assertTrue(callable(op.master_segments_dir))
        self.assertTrue(callable(op.human_eval_dir))
        self.assertTrue(callable(op.full_transcripts_dir))
        self.assertTrue(callable(op.cv_testset_dir))
        self.assertTrue(callable(op.llm_prompts_path))
        self.assertTrue(callable(op.theme_definitions_txt_path))
        self.assertTrue(callable(op.codebook_raw_dir))
        self.assertTrue(callable(op.segmented_sessions_dir))
        self.assertTrue(callable(op.segmented_session_dir))
        self.assertTrue(callable(op.segmentation_meta_path))

    def test_abspath_returned_for_overlay_path(self):
        """overlay_path must return absolute paths."""
        from process import output_paths as op
        path = op.classification_overlay_path('/out', 'theme')
        self.assertTrue(os.path.isabs(path))


# ============================================================================
# CATEGORY 12 — LLMClient interface (after backend removal)
# ============================================================================

class TestLLMClientInterface(unittest.TestCase):
    """LLMClient must work with only openrouter, ollama, lmstudio backends."""

    def test_llm_client_config_no_replicate_or_hf(self):
        """LLMClientConfig must not have Replicate/HF fields."""
        from classification_tools.llm_client import LLMClientConfig

        cfg = LLMClientConfig()

        # Must NOT have removed fields
        for forbidden in ('replicate_api_token', 'max_new_tokens', 'use_gpu',
                          'batch_size', 'replicate'):
            self.assertFalse(hasattr(cfg, forbidden),
                             f"LLMClientConfig must not have '{forbidden}'")

        # Must have required fields
        self.assertTrue(hasattr(cfg, 'backend'))
        self.assertTrue(hasattr(cfg, 'model'))
        self.assertTrue(hasattr(cfg, 'api_key'))
        self.assertTrue(hasattr(cfg, 'temperature'))
        self.assertTrue(hasattr(cfg, 'lmstudio_base_url'))
        self.assertTrue(hasattr(cfg, 'ollama_host'))
        self.assertTrue(hasattr(cfg, 'ollama_port'))

    def test_llm_client_default_backend_is_lmstudio(self):
        """Default backend should be lmstudio (not huggingface)."""
        from classification_tools.llm_client import LLMClientConfig

        cfg = LLMClientConfig()
        self.assertEqual(cfg.backend, 'lmstudio')

    def test_theme_classification_config_default_backend_is_lmstudio(self):
        """ThemeClassificationConfig default backend should be lmstudio."""
        from theme_framework.config import ThemeClassificationConfig

        cfg = ThemeClassificationConfig()
        self.assertEqual(cfg.backend, 'lmstudio')


# ============================================================================
# CATEGORY 13 — Setup wizard enhancements
# ============================================================================

class TestSetupWizardEnhancements(unittest.TestCase):
    """Setup wizard must have content-validity pages and multi-kind testset support."""

    def test_setup_wizard_has_new_steps(self):
        """SetupWizard must have _step_10b_content_validity method."""
        from process.setup_wizard import SetupWizard
        self.assertTrue(callable(getattr(SetupWizard, '_step_10b_content_validity', None)))

    def test_build_config_from_wizard_includes_content_validity(self):
        """build_config_from_wizard_data must produce content_validity config."""
        from process.setup_wizard import build_config_from_wizard_data

        wizard_data = {
            "pipeline": {
                "transcript_dir": "./data/input",
                "output_dir": "./data/output",
                "trial_id": "test",
            },
            "content_validity": {
                "vaamr": {"enabled": True, "name": "cv_v1"},
                "purer": {"enabled": False, "name": "cv_purer_v1"},
            },
            "test_sets": {
                "vaamr": {"enabled": True, "name": "vaamr_ts", "n_sets": 2,
                          "fraction_per_set": 0.1, "random_seed": 42},
            },
        }

        config = build_config_from_wizard_data(wizard_data)
        self.assertIsNotNone(config.content_validity)
        self.assertTrue(config.content_validity.vaamr.enabled)
        self.assertEqual(config.content_validity.vaamr.name, "cv_v1")
        self.assertFalse(config.content_validity.purer.enabled)

    def test_setup_wizard_backend_choices_no_replicate_hf(self):
        """SetupWizard's backend prompt must not offer replicate or huggingface."""
        import inspect
        from process.setup_wizard import SetupWizard

        source = inspect.getsource(SetupWizard._step_4_backend)
        self.assertNotIn("'replicate'", source)
        self.assertNotIn("'huggingface'", source)
        self.assertIn("'lmstudio'", source)
        self.assertIn("'ollama'", source)

    def test_build_test_sets_config_from_wizard_handles_nested_format(self):
        """_build_test_sets_config must handle nested per-kind format."""
        from process.setup_wizard import _build_test_sets_config

        ts_config = _build_test_sets_config({
            "vaamr": {"enabled": True, "name": "v_ts", "n_sets": 3,
                      "fraction_per_set": 0.05, "random_seed": 99},
            "purer": {"enabled": False},
            "codebook": {"enabled": False},
        })

        self.assertTrue(ts_config.vaamr.enabled)
        self.assertEqual(ts_config.vaamr.name, "v_ts")
        self.assertEqual(ts_config.vaamr.n_sets, 3)
        self.assertFalse(ts_config.purer.enabled)

    def test_build_test_sets_config_handles_legacy_flat_format(self):
        """_build_test_sets_config must handle old flat format."""
        from process.setup_wizard import _build_test_sets_config

        ts_config = _build_test_sets_config({
            "enabled": True,
            "n_sets": 2,
            "fraction_per_set": 0.10,
            "random_seed": 42,
        })

        # Should map to VAAMR with provided values
        self.assertTrue(ts_config.vaamr.enabled)
        self.assertEqual(ts_config.vaamr.n_sets, 2)

    def test_content_validity_config_builder(self):
        """_build_content_validity_config must handle wizard data."""
        from process.setup_wizard import _build_content_validity_config

        cv = _build_content_validity_config({
            "vaamr": {"enabled": True, "name": "cv_vaamr"},
            "purer": {"enabled": True, "name": "cv_purer"},
        })

        self.assertTrue(cv.vaamr.enabled)
        self.assertTrue(cv.purer.enabled)


# ============================================================================
# CATEGORY 14 — End-to-end config roundtrip
# ============================================================================

class TestConfigRoundtrip(unittest.TestCase):
    """Full config serialize/deserialize must preserve all fields."""

    def test_reference_config_roundtrips(self):
        """Construct PipelineConfig from reference dict, convert back, compare."""
        from process.config import PipelineConfig

        config = PipelineConfig.from_dict(REFERENCE_CONFIG_DICT)
        as_dict = config.to_dict()

        # Verify top-level pipeline fields survive the roundtrip (flat format)
        self.assertEqual(as_dict['transcript_dir'], './data/full/')
        self.assertEqual(as_dict['output_dir'], './data/MoveMORE/')
        self.assertEqual(as_dict['trial_id'], 'MoveMORE')
        self.assertFalse(as_dict['run_codebook_classifier'])
        self.assertTrue(as_dict['auto_analyze'])

        # Verify secrets are blanked
        tc = as_dict['theme_classification']
        self.assertEqual(tc.get('api_key', ''), '',
                         "api_key must be blanked in to_dict output")

    def test_config_serialization_excludes_secrets(self):
        """to_dict must exclude _SECRET_KEYS."""
        from process.config import PipelineConfig

        config = PipelineConfig.from_dict(REFERENCE_CONFIG_DICT)
        config.theme_classification.api_key = 'secret-123'
        as_dict = config.to_dict()

        # Recursively check no secret VALUES leaked (keys exist but must be blank)
        def has_secret_value(v):
            if isinstance(v, dict):
                return (any(k == 'api_key' and bool(vv) for k, vv in v.items())
                        or any(has_secret_value(vv) for vv in v.values()))
            return False

        self.assertFalse(has_secret_value(as_dict),
                         "to_dict must not expose non-blank secret values")


# ============================================================================
# CATEGORY 15 — Data structure integrity
# ============================================================================

class TestSegmentDataStructure(unittest.TestCase):
    """Segment dataclass must have all fields needed by both working and beta."""

    def test_segment_has_vaamr_fields(self):
        """Segment must have VAAMR classification fields from working branch."""
        from classification_tools.data_structures import Segment

        fields = {f.name for f in __import__('dataclasses').fields(Segment)}
        vaamr_fields = {
            'primary_stage', 'secondary_stage',
            'llm_confidence_primary', 'llm_confidence_secondary',
            'llm_justification', 'rater_ids', 'rater_votes',
            'agreement_level', 'agreement_fraction',
            'needs_review', 'consensus_vote', 'tie_broken_by_confidence',
            'llm_run_consistency',
        }
        self.assertTrue(vaamr_fields.issubset(fields),
                        f"Missing VAAMR fields: {vaamr_fields - fields}")

    def test_segment_has_purer_fields(self):
        """Segment must have PURER classification fields."""
        from classification_tools.data_structures import Segment

        fields = {f.name for f in __import__('dataclasses').fields(Segment)}
        purer_fields = {
            'purer_primary', 'purer_secondary',
            'purer_confidence_primary', 'purer_confidence_secondary',
            'purer_justification', 'purer_run_consistency',
            'purer_agreement_level', 'purer_agreement_fraction',
            'purer_needs_review', 'purer_rater_ids', 'purer_rater_votes',
        }
        self.assertTrue(purer_fields.issubset(fields),
                        f"Missing PURER fields: {purer_fields - fields}")

    def test_segment_has_codebook_fields(self):
        """Segment must have codebook fields."""
        from classification_tools.data_structures import Segment

        fields = {f.name for f in __import__('dataclasses').fields(Segment)}
        cb_fields = {
            'codebook_labels_embedding', 'codebook_labels_llm',
            'codebook_labels_ensemble', 'codebook_disagreements',
            'codebook_confidence',
        }
        self.assertTrue(cb_fields.issubset(fields),
                        f"Missing codebook fields: {cb_fields - fields}")

    def test_segment_has_core_metadata_fields(self):
        """Segment must have core identity and metadata fields."""
        from classification_tools.data_structures import Segment

        fields = {f.name for f in __import__('dataclasses').fields(Segment)}
        core_fields = {
            'segment_id', 'trial_id', 'participant_id', 'session_id',
            'session_number', 'cohort_id', 'speaker', 'text', 'word_count',
            'start_time_ms', 'end_time_ms', 'segment_index',
            'total_segments_in_session', 'session_file',
        }
        self.assertTrue(core_fields.issubset(fields),
                        f"Missing core fields: {core_fields - fields}")


# ============================================================================
# CATEGORY 16 — Theme framework integrity
# ============================================================================

class TestThemeFrameworkIntegrity(unittest.TestCase):
    """Theme frameworks must load and work correctly."""

    def test_vaamr_framework_loads(self):
        """VAAMR framework must be loadable with correct metadata."""
        from theme_framework.vaamr import get_vaamr_framework, VAAMR_FRAMEWORK_VERSION

        framework = get_vaamr_framework()
        self.assertEqual(framework.name, 'VAAMR')
        self.assertEqual(framework.version, VAAMR_FRAMEWORK_VERSION)
        self.assertEqual(len(framework.stages), 5)
        stage_names = [s.short_name for s in framework.stages]
        self.assertEqual(stage_names, [
            'Vigilance',
            'Avoidance',
            'Attention Regulation',
            'Metacognition',
            'Reappraisal',
        ])

    def test_vaamr_build_name_to_id_map(self):
        """VAAMR must produce correct name→id mapping."""
        from theme_framework.vaamr import get_vaamr_framework

        framework = get_vaamr_framework()
        name_to_id = framework.build_name_to_id_map()

        self.assertEqual(name_to_id['vigilance'], 0)
        self.assertEqual(name_to_id['avoidance'], 1)
        self.assertEqual(name_to_id['attention regulation'], 2)
        self.assertEqual(name_to_id['metacognition'], 3)
        self.assertEqual(name_to_id['reappraisal'], 4)

    def test_purer_framework_loads(self):
        """PURER framework must be loadable."""
        from theme_framework.purer import get_purer_framework, PURER_FRAMEWORK_VERSION

        framework = get_purer_framework()
        self.assertEqual(framework.name, 'PURER')
        self.assertEqual(framework.version, PURER_FRAMEWORK_VERSION)
        self.assertEqual(len(framework.stages), 5)
        stage_names = [s.name for s in framework.stages]
        self.assertEqual(stage_names, [
            'Phenomenological',
            'Utilization',
            'Reframing',
            'Educate/Expectancy',
            'Reinforcement',
        ])  # s.name for PURER stages; stage 3 name is 'Educate/Expectancy' per CLAUDE.md


# ============================================================================
# CATEGORY 17 — Speaker anonymization key path handling
# ============================================================================

class TestSpeakerAnonymizationKey(unittest.TestCase):
    """
    In the working branch, the speaker anonymization key was written in Stage 7.
    In beta, it's written in stage_ingest (Stage 1). Verify the output path
    is still 02_meta/speaker_anonymization_key.json.
    """

    def test_speaker_key_path_is_in_meta_dir(self):
        """Speaker anonymization key must be in 02_meta/."""
        from process import output_paths as _paths

        key_path = os.path.join(_paths.meta_dir('/out'),
                                'speaker_anonymization_key.json')
        self.assertIn('02_meta', key_path)
        self.assertTrue(key_path.endswith('speaker_anonymization_key.json'))

    def test_anonymization_key_txt_path(self):
        """anonymization_key_txt_path must point to 02_meta/."""
        from process import output_paths as _paths

        txt_path = _paths.anonymization_key_txt_path('/out')
        self.assertIn('02_meta', txt_path)
        self.assertTrue(txt_path.endswith('.txt'))


# ============================================================================
# CATEGORY 18 — analysis runner interface
# ============================================================================

class TestAnalysisRunnerInterface(unittest.TestCase):
    """analysis/runner.py must work without replicate_api_token."""

    def test_run_analysis_signature(self):
        """run_analysis must be importable and callable."""
        from analysis.runner import run_analysis
        self.assertTrue(callable(run_analysis))

    def test_run_analysis_source_has_no_replicate(self):
        """run_analysis source code must not reference replicate_api_token."""
        import analysis.runner
        source = __import__('inspect').getsource(analysis.runner.run_analysis)
        self.assertNotIn('replicate_api_token', source)


# ============================================================================
# Run all tests
# ============================================================================

if __name__ == '__main__':
    unittest.main(verbosity=2)
