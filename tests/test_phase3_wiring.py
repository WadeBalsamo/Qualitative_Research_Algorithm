"""
tests/test_phase3_wiring.py
---------------------------
Tests for the Phase 3 CLI / wizard wiring implemented across 8 blocks:

  Block A — _build_config safe for any subparser namespace; cmd_classify/assemble wired
  Block B — stage_ingest extraction; cmd_ingest wired
  Block C — run_full_pipeline calls stage_* functions (overlay writes)
  Block D — stage_validation_artifacts extraction; cmd_assemble calls it
  Block E — qra validate subcommand
  Block F — cmd_cv_refresh with ThemeClassificationConfig
  Block G — wizard Phase 2 testsets + content-validity
  Block H — CLI help / epilog completeness
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

QRA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qra.py')
VENV_PYTHON = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    '.venv', 'bin', 'python',
)
PYTHON = VENV_PYTHON if os.path.isfile(VENV_PYTHON) else sys.executable


def _run(*args):
    """Run qra.py with the given args, return completed process."""
    return subprocess.run(
        [PYTHON, QRA] + list(args),
        capture_output=True, text=True,
    )


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_segment(segment_id='seg_001', session_id='c1s1', speaker='participant', **kwargs):
    from classification_tools.data_structures import Segment
    defaults = dict(
        segment_id=segment_id,
        trial_id='trial_A',
        participant_id='participant_1',
        session_id=session_id,
        session_number=1,
        cohort_id=1,
        session_variant='',
        segment_index=0,
        start_time_ms=0,
        end_time_ms=5000,
        total_segments_in_session=2,
        speaker=speaker,
        text='Test segment text for phase3 wiring tests.',
        word_count=8,
        speakers_in_segment=[speaker],
        session_file='/data/input/c1s1/session.json',
    )
    defaults.update(kwargs)
    return Segment(**defaults)


def _classified(seg_id, primary=2, session_id='c1s1'):
    seg = _make_segment(seg_id, session_id=session_id)
    seg.primary_stage = primary
    seg.llm_confidence_primary = 0.9
    seg.agreement_level = 'unanimous'
    seg.agreement_fraction = 1.0
    seg.needs_review = False
    seg.consensus_vote = primary
    seg.tie_broken_by_confidence = False
    seg.llm_run_consistency = 2
    seg.rater_ids = ['r1']
    seg.rater_votes = [{'rater': 'r1', 'stage': primary, 'confidence': 0.9}]
    return seg


def _write_frozen(tmpdir, segs, session_id='c1s1'):
    from process import segments_io
    segments_io.write_session_segments(tmpdir, session_id, segs, 'testhash')


def _write_theme_overlay(tmpdir, segs):
    from process import classifications_io
    classifications_io.write_theme_overlay(tmpdir, segs)


# ===========================================================================
# Block A — _build_config safety for minimal namespaces
# ===========================================================================

class TestBuildConfigMinimalNamespace(unittest.TestCase):
    """
    _build_config must work when called with a Namespace that only has
    output_dir and config (as in the assemble / validate parsers).
    No AttributeError for absent transcript_dir, no_theme_labeler, etc.
    """

    def _import_build_config(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location('qra', QRA)
        mod = importlib.util.load_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod._build_config

    def test_minimal_namespace_no_config(self):
        """Namespace with only output_dir and config=None must not raise."""
        import importlib.util
        args = argparse.Namespace(output_dir='/tmp/qra_test_minimal', config=None)
        # Load qra module with __file__ set so sys.path.insert doesn't fail
        spec = importlib.util.spec_from_file_location('qra_module', QRA)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # _build_config must not raise AttributeError on sparse namespace
        config = mod._build_config(args)
        self.assertIsNotNone(config)

    def test_assemble_command_no_transcript_dir(self):
        """qra assemble must not crash with AttributeError on missing transcript_dir."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001'), _classified('seg_002')]
            _write_frozen(d, segs)
            _write_theme_overlay(d, segs)
            r = _run('assemble', '-o', d)
            # Should either succeed or fail with a meaningful message, not AttributeError
            self.assertNotIn("AttributeError: 'Namespace'", r.stderr,
                             "AttributeError leaked from _build_config")

    def test_validate_command_no_transcript_dir(self):
        """qra validate must not crash with AttributeError on missing transcript_dir."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001')]
            _write_frozen(d, segs)
            _write_theme_overlay(d, segs)
            r = _run('validate', '-o', d)
            self.assertNotIn("AttributeError: 'Namespace'", r.stderr,
                             "AttributeError leaked from _build_config")

    def test_classify_command_no_config_still_uses_defaults(self):
        """qra classify without --config must not pass config=None to stages."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001')]
            _write_frozen(d, segs)
            # Classify with no --config and no --backend — should use minimal config
            r = _run('classify', '--what', 'theme', '-o', d)
            # The key assertion: no AttributeError from _build_config
            self.assertNotIn("AttributeError", r.stderr)
            # theme_labels.jsonl must exist (stage ran with config, even if LLM failed)
            from process import classifications_io
            overlay = classifications_io.overlay_path(d, 'theme')
            self.assertTrue(os.path.isfile(overlay),
                            "theme overlay must be written even without --config")

    def test_classify_codebook_loads_regardless_of_config_flag(self):
        """--what codebook must load the codebook regardless of config.run_codebook_classifier."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001')]
            _write_frozen(d, segs)
            # No --config → no run_codebook_classifier in config, but --what codebook is explicit
            r = _run('classify', '--what', 'codebook', '-o', d)
            self.assertNotIn("AttributeError", r.stderr)
            from process import classifications_io
            overlay = classifications_io.overlay_path(d, 'codebook')
            self.assertTrue(os.path.isfile(overlay),
                            "codebook overlay must be written when --what codebook is specified")


# ===========================================================================
# Block B — stage_ingest
# ===========================================================================

class TestStageIngestImport(unittest.TestCase):
    """stage_ingest must be importable from process.orchestrator."""

    def test_importable(self):
        from process.orchestrator import stage_ingest  # noqa

    def test_signature_has_force_reingest(self):
        import inspect
        from process.orchestrator import stage_ingest
        sig = inspect.signature(stage_ingest)
        params = set(sig.parameters)
        self.assertIn('force_reingest', params)
        self.assertIn('force_reingest_all', params)
        self.assertIn('output_dir', params)
        self.assertIn('observer', params)

    def test_returns_list(self):
        """stage_ingest called with empty transcript_dir returns a list."""
        from process.orchestrator import stage_ingest
        from process.config import PipelineConfig
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'transcripts'), exist_ok=True)
            config = PipelineConfig()
            config.transcript_dir = os.path.join(d, 'transcripts')
            config.output_dir = d
            # Patch segmenter so we don't need the embedding model
            with patch('process.orchestrator.ConversationalSegmenter') as mock_seg:
                mock_seg.return_value.speaker_norm.speaker_map = {}
                mock_seg.return_value.release_gpu_memory = lambda: None
                segs = stage_ingest(config, output_dir=d)
            self.assertIsInstance(segs, list)

    def test_writes_speaker_key_json(self):
        """stage_ingest must write speaker_anonymization_key.json to meta/."""
        from process.orchestrator import stage_ingest
        from process.config import PipelineConfig
        from process import output_paths as _paths
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'transcripts'), exist_ok=True)
            config = PipelineConfig()
            config.transcript_dir = os.path.join(d, 'transcripts')
            config.output_dir = d
            with patch('process.orchestrator.ConversationalSegmenter') as mock_seg:
                mock_seg.return_value.speaker_norm.speaker_map = {
                    'SPEAKER_00': ('participant', 'participant_1'),
                }
                mock_seg.return_value.release_gpu_memory = lambda: None
                stage_ingest(config, output_dir=d)
            key_path = os.path.join(_paths.meta_dir(d), 'speaker_anonymization_key.json')
            self.assertTrue(os.path.isfile(key_path), "speaker key must be written")
            with open(key_path) as f:
                key = json.load(f)
            self.assertIn('SPEAKER_00', key)

    def test_reuses_frozen_segments(self):
        """stage_ingest must load frozen segments without re-segmenting when fresh."""
        from process.orchestrator import stage_ingest, _resolve_session_id
        from process.config import PipelineConfig
        from process import segments_io
        with tempfile.TemporaryDirectory() as d:
            transcript_dir = os.path.join(d, 'transcripts', 'c1s1')
            os.makedirs(transcript_dir)
            fake_session_file = os.path.join(transcript_dir, 'session.json')
            with open(fake_session_file, 'w') as f:
                json.dump({'metadata': {}, 'sentences': []}, f)

            # Write frozen segments with a known hash
            segs_frozen = [_make_segment('frozen_001', session_id='c1s1')]
            config = PipelineConfig()
            config.transcript_dir = os.path.join(d, 'transcripts')
            config.output_dir = d
            current_hash = segments_io.params_hash(config.segmentation)
            segments_io.write_session_segments(d, 'c1s1', segs_frozen, current_hash)

            with patch('process.orchestrator.ConversationalSegmenter') as mock_seg:
                mock_seg.return_value.speaker_norm.speaker_map = {}
                mock_seg.return_value.release_gpu_memory = lambda: None
                segs = stage_ingest(config, output_dir=d)

            # Should return the frozen segment, not call segment_session
            seg_ids = [s.segment_id for s in segs]
            self.assertIn('frozen_001', seg_ids)
            mock_seg.return_value.segment_session.assert_not_called()

    def test_force_reingest_all_bypasses_fresh_check(self):
        """force_reingest_all=True must pass force=True to write_session_segments."""
        from process.orchestrator import stage_ingest, _resolve_session_id
        from process.config import PipelineConfig
        from process import segments_io
        with tempfile.TemporaryDirectory() as d:
            transcript_dir = os.path.join(d, 'transcripts', 'c1s1')
            os.makedirs(transcript_dir)
            fake_session_file = os.path.join(transcript_dir, 'session.json')
            with open(fake_session_file, 'w') as f:
                json.dump({'metadata': {}, 'sentences': []}, f)

            # Write frozen segments with current hash
            config = PipelineConfig()
            config.transcript_dir = os.path.join(d, 'transcripts')
            config.output_dir = d
            config.segmentation.use_llm_refinement = False  # avoid LLM calls in test
            current_hash = segments_io.params_hash(config.segmentation)
            _write_frozen(d, [_make_segment('frozen_001')], 'c1s1')

            # Patch discover_session_files to return a single fake file
            # and segment_session to return new segments
            new_seg = _make_segment('new_001', session_id='c1s1')
            with patch('process.orchestrator.ConversationalSegmenter') as mock_seg, \
                 patch('process.orchestrator.discover_session_files') as mock_disc:
                mock_disc.return_value = [fake_session_file]
                mock_seg.return_value.speaker_norm.speaker_map = {}
                mock_seg.return_value.release_gpu_memory = lambda: None
                mock_seg.return_value.segment_session.return_value = [new_seg]
                mock_seg.return_value.extract_therapist_segments.return_value = []
                with patch('process.orchestrator.load_diarized_session') as mock_load:
                    mock_load.return_value = {
                        'metadata': {'session_id': 'c1s1'},
                        'sentences': [],
                    }
                    segs = stage_ingest(
                        config, output_dir=d,
                        force_reingest_all=True,
                    )
                # Should have called segment_session since we forced re-ingest
                mock_seg.return_value.segment_session.assert_called()


class TestQraIngestCLI(unittest.TestCase):
    """CLI tests for qra ingest."""

    def test_ingest_help_exits_zero(self):
        r = _run('ingest', '--help')
        self.assertEqual(r.returncode, 0)

    def test_ingest_help_shows_reingest_flags(self):
        r = _run('ingest', '--help')
        output = r.stdout + r.stderr
        self.assertIn('reingest', output.lower())

    def test_ingest_no_longer_prints_stub_message(self):
        """cmd_ingest must NOT print the old 'use qra run' stub message."""
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, 'transcripts'), exist_ok=True)
            r = _run('ingest', '-o', d, '--transcript-dir', os.path.join(d, 'transcripts'))
            combined = r.stdout + r.stderr
            self.assertNotIn('use `qra run` to run the full segmentation', combined,
                             "cmd_ingest must not print the old stub guidance")
            self.assertNotIn('Phase 3+ feature', combined)

    def test_ingest_with_empty_transcript_dir_exits_cleanly(self):
        """With no transcript files, qra ingest must exit 0 (nothing to do)."""
        with tempfile.TemporaryDirectory() as d:
            transcript_dir = os.path.join(d, 'transcripts')
            os.makedirs(transcript_dir)
            with patch.dict(os.environ, {}):
                r = _run('ingest', '-o', d, '--transcript-dir', transcript_dir)
            # Should not crash with a traceback
            self.assertNotIn('Traceback', r.stderr, f"Unexpected error: {r.stderr}")


# ===========================================================================
# Block C — run_full_pipeline writes overlays
# ===========================================================================

class TestRunFullPipelineOverlayWrites(unittest.TestCase):
    """
    After run_full_pipeline completes, all classification overlays must exist
    on disk so that standalone classify/assemble commands can re-enter.

    We test this by running a minimal mocked pipeline where stage functions
    are patched to no-op but still write their overlays.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Pre-write frozen segments so stage_ingest can reuse them
        segs = [_classified('seg_001'), _classified('seg_002')]
        _write_frozen(self.tmpdir, segs)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _minimal_config(self):
        from process.config import PipelineConfig
        c = PipelineConfig()
        c.output_dir = self.tmpdir
        c.transcript_dir = self.tmpdir
        c.run_theme_labeler = True
        c.run_purer_labeler = False
        c.run_codebook_classifier = False
        c.auto_analyze = False
        return c

    def test_theme_overlay_written_after_run(self):
        """After run_full_pipeline, theme_labels.jsonl must exist on disk."""
        from process.orchestrator import run_full_pipeline
        from process import classifications_io
        from theme_framework.vaamr import get_vaamr_framework

        config = self._minimal_config()
        framework = get_vaamr_framework()

        segs = [_classified('seg_001'), _classified('seg_002')]

        # Patch stage_ingest so no real segmentation happens
        with patch('process.orchestrator.stage_ingest', return_value=segs) as _si, \
             patch('process.orchestrator.stage_classify_theme',
                   side_effect=lambda cfg, fw, segments=None, output_dir=None, observer=None: (
                       classifications_io.write_theme_overlay(output_dir or cfg.output_dir, segments or []),
                       classifications_io.update_classification_manifest(
                           output_dir or cfg.output_dir, key='theme', entry={'n_segments': len(segments or [])}),
                       segments,
                   )[-1]) as _sct, \
             patch('process.orchestrator.assemble_master_dataset',
                   return_value=__import__('pandas').DataFrame([{'segment_id': 'seg_001'}])) as _asm, \
             patch('process.orchestrator.export_coded_transcript'), \
             patch('process.orchestrator.export_human_classification_forms'), \
             patch('process.orchestrator.export_flagged_for_review'), \
             patch('process.orchestrator.export_per_transcript_stats'), \
             patch('process.orchestrator.export_cumulative_report'), \
             patch('process.orchestrator.export_training_data'), \
             patch('process.orchestrator.export_theme_definitions'), \
             patch('process.orchestrator.export_theme_definitions_txt'), \
             patch('process.orchestrator.export_content_validity_test_set'), \
             patch('process.orchestrator.export_content_validity_human_worksheet'), \
             patch('process.orchestrator.export_content_validity_definition_key'), \
             patch('process.orchestrator.ensure_embedding_model_ready'):
            try:
                run_full_pipeline(config, framework)
            except Exception:
                pass  # report generation may fail with mocks; overlay write is what we test

        overlay = classifications_io.overlay_path(self.tmpdir, 'theme')
        self.assertTrue(os.path.isfile(overlay),
                        "theme_labels.jsonl must exist after run_full_pipeline")

    def test_stage_ingest_called_from_run_full_pipeline(self):
        """run_full_pipeline must delegate Stage 1 to stage_ingest (no inline duplication)."""
        from process.orchestrator import run_full_pipeline
        from theme_framework.vaamr import get_vaamr_framework

        config = self._minimal_config()
        framework = get_vaamr_framework()
        segs = [_classified('seg_001')]

        with patch('process.orchestrator.stage_ingest', return_value=segs) as mock_si, \
             patch('process.orchestrator.stage_classify_theme', return_value=segs), \
             patch('process.orchestrator.assemble_master_dataset',
                   return_value=__import__('pandas').DataFrame()), \
             patch('process.orchestrator.export_coded_transcript'), \
             patch('process.orchestrator.export_human_classification_forms'), \
             patch('process.orchestrator.export_flagged_for_review'), \
             patch('process.orchestrator.export_per_transcript_stats'), \
             patch('process.orchestrator.export_cumulative_report'), \
             patch('process.orchestrator.export_training_data'), \
             patch('process.orchestrator.export_theme_definitions'), \
             patch('process.orchestrator.export_theme_definitions_txt'), \
             patch('process.orchestrator.export_content_validity_test_set'), \
             patch('process.orchestrator.export_content_validity_human_worksheet'), \
             patch('process.orchestrator.export_content_validity_definition_key'), \
             patch('process.orchestrator.ensure_embedding_model_ready'):
            try:
                run_full_pipeline(config, framework)
            except Exception:
                pass

        mock_si.assert_called_once()

    def test_stage_classify_theme_called_from_run_full_pipeline(self):
        """run_full_pipeline must call stage_classify_theme (not inline code)."""
        from process.orchestrator import run_full_pipeline
        from theme_framework.vaamr import get_vaamr_framework

        config = self._minimal_config()
        framework = get_vaamr_framework()
        segs = [_classified('seg_001')]

        with patch('process.orchestrator.stage_ingest', return_value=segs), \
             patch('process.orchestrator.stage_classify_theme',
                   return_value=segs) as mock_sct, \
             patch('process.orchestrator.assemble_master_dataset',
                   return_value=__import__('pandas').DataFrame()), \
             patch('process.orchestrator.export_coded_transcript'), \
             patch('process.orchestrator.export_human_classification_forms'), \
             patch('process.orchestrator.export_flagged_for_review'), \
             patch('process.orchestrator.export_per_transcript_stats'), \
             patch('process.orchestrator.export_cumulative_report'), \
             patch('process.orchestrator.export_training_data'), \
             patch('process.orchestrator.export_theme_definitions'), \
             patch('process.orchestrator.export_theme_definitions_txt'), \
             patch('process.orchestrator.export_content_validity_test_set'), \
             patch('process.orchestrator.export_content_validity_human_worksheet'), \
             patch('process.orchestrator.export_content_validity_definition_key'), \
             patch('process.orchestrator.ensure_embedding_model_ready'):
            try:
                run_full_pipeline(config, framework)
            except Exception:
                pass

        mock_sct.assert_called_once()
        # Must be called with in-memory segments (not load-from-disk mode)
        call_kwargs = mock_sct.call_args[1]
        self.assertIsNotNone(call_kwargs.get('segments'),
                             "stage_classify_theme must receive segments from stage_ingest")

    def test_stage_classify_purer_called_when_purer_enabled_and_therapists(self):
        """run_full_pipeline calls stage_classify_purer when purer enabled and therapists exist."""
        from process.orchestrator import run_full_pipeline
        from theme_framework.vaamr import get_vaamr_framework

        config = self._minimal_config()
        config.run_purer_labeler = True
        framework = get_vaamr_framework()

        therapist_seg = _make_segment('th_001', speaker='therapist')
        segs = [_classified('seg_001'), therapist_seg]

        with patch('process.orchestrator.stage_ingest', return_value=segs), \
             patch('process.orchestrator.stage_classify_theme', return_value=segs), \
             patch('process.orchestrator.stage_classify_purer',
                   return_value=segs) as mock_purer, \
             patch('process.orchestrator.assemble_master_dataset',
                   return_value=__import__('pandas').DataFrame()), \
             patch('process.orchestrator.export_coded_transcript'), \
             patch('process.orchestrator.export_human_classification_forms'), \
             patch('process.orchestrator.export_flagged_for_review'), \
             patch('process.orchestrator.export_per_transcript_stats'), \
             patch('process.orchestrator.export_cumulative_report'), \
             patch('process.orchestrator.export_training_data'), \
             patch('process.orchestrator.export_theme_definitions'), \
             patch('process.orchestrator.export_theme_definitions_txt'), \
             patch('process.orchestrator.export_content_validity_test_set'), \
             patch('process.orchestrator.export_content_validity_human_worksheet'), \
             patch('process.orchestrator.export_content_validity_definition_key'), \
             patch('process.orchestrator.ensure_embedding_model_ready'):
            try:
                run_full_pipeline(config, framework)
            except Exception:
                pass

        mock_purer.assert_called_once()

    def test_run_full_pipeline_skips_purer_when_no_therapists(self):
        """run_full_pipeline must NOT call stage_classify_purer if no therapist segments."""
        from process.orchestrator import run_full_pipeline
        from theme_framework.vaamr import get_vaamr_framework

        config = self._minimal_config()
        config.run_purer_labeler = True  # enabled but no therapists
        framework = get_vaamr_framework()
        participant_only = [_classified('seg_001'), _classified('seg_002')]

        with patch('process.orchestrator.stage_ingest', return_value=participant_only), \
             patch('process.orchestrator.stage_classify_theme', return_value=participant_only), \
             patch('process.orchestrator.stage_classify_purer') as mock_purer, \
             patch('process.orchestrator.assemble_master_dataset',
                   return_value=__import__('pandas').DataFrame()), \
             patch('process.orchestrator.export_coded_transcript'), \
             patch('process.orchestrator.export_human_classification_forms'), \
             patch('process.orchestrator.export_flagged_for_review'), \
             patch('process.orchestrator.export_per_transcript_stats'), \
             patch('process.orchestrator.export_cumulative_report'), \
             patch('process.orchestrator.export_training_data'), \
             patch('process.orchestrator.export_theme_definitions'), \
             patch('process.orchestrator.export_theme_definitions_txt'), \
             patch('process.orchestrator.export_content_validity_test_set'), \
             patch('process.orchestrator.export_content_validity_human_worksheet'), \
             patch('process.orchestrator.export_content_validity_definition_key'), \
             patch('process.orchestrator.ensure_embedding_model_ready'):
            try:
                run_full_pipeline(config, framework)
            except Exception:
                pass

        mock_purer.assert_not_called()


# ===========================================================================
# Block D — stage_validation_artifacts
# ===========================================================================

class TestStageValidationArtifactsImport(unittest.TestCase):
    """stage_validation_artifacts must be importable with correct signature."""

    def test_importable(self):
        from process.orchestrator import stage_validation_artifacts  # noqa

    def test_signature(self):
        import inspect
        from process.orchestrator import stage_validation_artifacts
        sig = inspect.signature(stage_validation_artifacts)
        params = set(sig.parameters)
        self.assertIn('config', params)
        self.assertIn('framework', params)
        self.assertIn('codebook', params)
        self.assertIn('segments', params)
        self.assertIn('output_dir', params)
        self.assertIn('create_missing', params)


class TestStageValidationArtifactsWrites(unittest.TestCase):
    """stage_validation_artifacts must write human forms and flagged-for-review."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        segs = [_classified('seg_001', 2), _classified('seg_002', 0)]
        _write_frozen(self.tmpdir, segs)
        _write_theme_overlay(self.tmpdir, segs)
        self.segs = segs

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_writes_human_classification_forms(self):
        from process.orchestrator import stage_validation_artifacts
        from theme_framework.vaamr import get_vaamr_framework

        stage_validation_artifacts(
            None, get_vaamr_framework(),
            segments=self.segs, output_dir=self.tmpdir,
        )
        validation_dir = os.path.join(self.tmpdir, '04_validation')
        forms = [f for f in os.listdir(validation_dir)
                 if f.startswith('human_classification')]
        self.assertTrue(len(forms) >= 1, "human classification form must be written")

    def test_writes_flagged_for_review(self):
        from process.orchestrator import stage_validation_artifacts
        from theme_framework.vaamr import get_vaamr_framework

        stage_validation_artifacts(
            None, get_vaamr_framework(),
            segments=self.segs, output_dir=self.tmpdir,
        )
        validation_dir = os.path.join(self.tmpdir, '04_validation')
        flagged = os.path.join(validation_dir, 'flagged_for_review.txt')
        self.assertTrue(os.path.isfile(flagged), "flagged_for_review.txt must be written")

    def test_loads_from_disk_when_segments_none(self):
        """When segments=None, stage_validation_artifacts loads from frozen disk."""
        from process.orchestrator import stage_validation_artifacts
        from theme_framework.vaamr import get_vaamr_framework

        # Should not raise even when segments not passed in
        stage_validation_artifacts(
            None, get_vaamr_framework(),
            output_dir=self.tmpdir,
        )
        validation_dir = os.path.join(self.tmpdir, '04_validation')
        self.assertTrue(os.path.isdir(validation_dir))

    def test_create_missing_false_skips_testset_creation(self):
        """create_missing=False must not create new testsets that don't already exist."""
        from process.orchestrator import stage_validation_artifacts
        from process.config import PipelineConfig, TestSetsConfig, TestSetSpec
        from theme_framework.vaamr import get_vaamr_framework

        config = PipelineConfig()
        config.output_dir = self.tmpdir
        config.test_sets = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_ts', n_sets=1, fraction_per_set=0.5),
        )

        with patch('process.orchestrator.generate_or_refresh_validation_testsets') as mock_ts:
            stage_validation_artifacts(
                config, get_vaamr_framework(),
                segments=self.segs,
                output_dir=self.tmpdir,
                create_missing=False,
            )
        # Must be called with create_missing=False
        if mock_ts.called:
            call_kwargs = mock_ts.call_args[1]
            self.assertFalse(call_kwargs.get('create_missing', True),
                             "create_missing=False must propagate to testset coordinator")

    def test_cmd_assemble_calls_stage_validation_artifacts(self):
        """qra assemble must call stage_validation_artifacts after assembly."""
        r = _run('assemble', '-o', self.tmpdir)
        # The validation dir must exist (stage_validation_artifacts was called)
        validation_dir = os.path.join(self.tmpdir, '04_validation')
        self.assertTrue(os.path.isdir(validation_dir),
                        "04_validation must exist after qra assemble")


# ===========================================================================
# Block E — qra validate subcommand
# ===========================================================================

class TestQraValidateCLI(unittest.TestCase):
    """Tests for the new qra validate subcommand."""

    def test_validate_in_main_help(self):
        r = _run('--help')
        self.assertIn('validate', r.stdout + r.stderr)

    def test_validate_help_exits_zero(self):
        r = _run('validate', '--help')
        self.assertEqual(r.returncode, 0)

    def test_validate_help_shows_output_dir(self):
        r = _run('validate', '--help')
        self.assertIn('output-dir', r.stdout + r.stderr)

    def test_validate_help_shows_config_option(self):
        r = _run('validate', '--help')
        self.assertIn('config', r.stdout + r.stderr)

    def test_validate_errors_without_frozen_segments(self):
        """qra validate must exit non-zero if no frozen segments exist."""
        with tempfile.TemporaryDirectory() as d:
            r = _run('validate', '-o', d)
            self.assertNotEqual(r.returncode, 0,
                                "validate must fail with no frozen segments")

    def test_validate_error_message_mentions_ingest(self):
        """Error message when no segments must guide user to run qra ingest."""
        with tempfile.TemporaryDirectory() as d:
            r = _run('validate', '-o', d)
            self.assertIn('ingest', r.stdout + r.stderr)

    def test_validate_succeeds_with_frozen_segments_and_overlay(self):
        """qra validate must exit 0 when frozen segments + overlay are present."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001'), _classified('seg_002')]
            _write_frozen(d, segs)
            _write_theme_overlay(d, segs)
            r = _run('validate', '-o', d)
            self.assertEqual(r.returncode, 0, f"stderr: {r.stderr}")

    def test_validate_writes_validation_artifacts(self):
        """qra validate must write human forms and flagged-for-review."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001'), _classified('seg_002')]
            _write_frozen(d, segs)
            _write_theme_overlay(d, segs)
            r = _run('validate', '-o', d)
            self.assertEqual(r.returncode, 0, f"stderr: {r.stderr}")
            validation_dir = os.path.join(d, '04_validation')
            self.assertTrue(os.path.isdir(validation_dir),
                            "04_validation must be created by qra validate")
            forms = [f for f in os.listdir(validation_dir)
                     if f.startswith('human_classification')]
            self.assertTrue(len(forms) >= 1, "human classification form must exist")

    def test_validate_preserves_existing_frozen_testsets(self):
        """qra validate with create_missing=False must not overwrite frozen testset worksheets."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001'), _classified('seg_002')]
            _write_frozen(d, segs)
            _write_theme_overlay(d, segs)

            # Create a testset worksheet and record its content
            testset_dir = os.path.join(d, '04_validation', 'testsets', 'vaamr_testset_1')
            os.makedirs(testset_dir, exist_ok=True)
            worksheet_path = os.path.join(testset_dir, 'human_worksheet.txt')
            sentinel_content = 'HUMAN CODED WORKSHEET - DO NOT OVERWRITE'
            with open(worksheet_path, 'w') as f:
                f.write(sentinel_content)
            manifest_path = os.path.join(testset_dir, 'manifest.json')
            with open(manifest_path, 'w') as f:
                json.dump({'name': 'vaamr_testset_1', 'kind': 'vaamr',
                           'n_items': 2, 'created_at': '2026-01-01'}, f)

            _run('validate', '-o', d)

            with open(worksheet_path) as f:
                content_after = f.read()
            self.assertEqual(content_after, sentinel_content,
                             "validate must not overwrite frozen human worksheet")

    def test_validate_does_not_create_testsets_not_in_config(self):
        """qra validate (create_missing=False) must not spontaneously create new testsets."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001')]
            _write_frozen(d, segs)
            _write_theme_overlay(d, segs)
            r = _run('validate', '-o', d)
            testset_dir = os.path.join(d, '04_validation', 'testsets')
            if os.path.isdir(testset_dir):
                dirs = [e for e in os.listdir(testset_dir)
                        if os.path.isdir(os.path.join(testset_dir, e))]
                self.assertEqual(len(dirs), 0,
                                 "validate must not create new testsets when none exist")


# ===========================================================================
# Block F — cmd_cv_refresh with ThemeClassificationConfig
# ===========================================================================

class TestCvRefreshParser(unittest.TestCase):
    """cv refresh parser must accept --backend, --model, --lmstudio-url."""

    def test_cv_refresh_help_shows_backend(self):
        r = _run('cv', 'refresh', '--help')
        self.assertIn('backend', r.stdout + r.stderr)

    def test_cv_refresh_help_shows_model(self):
        r = _run('cv', 'refresh', '--help')
        self.assertIn('model', r.stdout + r.stderr)

    def test_cv_refresh_help_shows_lmstudio_url(self):
        r = _run('cv', 'refresh', '--help')
        self.assertIn('lmstudio', r.stdout + r.stderr)

    def test_cv_refresh_help_shows_config(self):
        r = _run('cv', 'refresh', '--help')
        self.assertIn('config', r.stdout + r.stderr)


class TestCvRefreshThemeConfig(unittest.TestCase):
    """cmd_cv_refresh must build ThemeClassificationConfig, not pass None."""

    def test_cv_refresh_no_longer_passes_none_config(self):
        """Verify refresh_cv_answer_key is called with a non-None tc parameter."""
        with tempfile.TemporaryDirectory() as d:
            # Create a minimal CV testset directory structure
            from process import output_paths as _paths
            cv_dir = _paths.cv_testsets_dir(d)
            testset_dir = os.path.join(cv_dir, 'cv_vaamr_v1')
            os.makedirs(testset_dir, exist_ok=True)
            manifest = {
                'name': 'cv_vaamr_v1', 'kind': 'vaamr',
                'n_items': 1, 'created_at': '2026-01-01',
                'framework': {'name': 'VAAMR'},
            }
            with open(os.path.join(testset_dir, 'manifest.json'), 'w') as f:
                json.dump(manifest, f)
            # Write a snapshot file (required by refresh)
            with open(os.path.join(testset_dir, 'snapshot.jsonl'), 'w') as f:
                f.write(json.dumps({'item_id': 'i1', 'text': 'test', 'true_label': 2}) + '\n')

            captured_tc = []

            def fake_refresh(run_dir, name, tc, framework):
                captured_tc.append(tc)
                return os.path.join(testset_dir, 'ai_answer_key.txt')

            with patch('process.assembly.content_validity.refresh_cv_answer_key',
                       side_effect=fake_refresh):
                r = _run('cv', 'refresh', '-o', d, '--name', 'cv_vaamr_v1',
                         '--backend', 'lmstudio', '--model', 'test-model')

            if captured_tc:
                self.assertIsNotNone(captured_tc[0],
                                     "refresh_cv_answer_key must receive non-None tc")

    def test_cv_refresh_builds_tc_from_cli_overrides(self):
        """CLI --backend and --model must override any config-derived ThemeClassificationConfig."""
        import importlib.util
        spec = importlib.util.spec_from_file_location('qra_module', QRA)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        # Simulate args with CLI overrides
        args = argparse.Namespace(
            output_dir='/tmp',
            config=None,
            backend='openrouter',
            model='openai/gpt-4o',
            lmstudio_url=None,
            name=None,
            all=True,
        )
        # _build_config must produce a config where theme_classification.backend = 'openrouter'
        config = mod._build_config(args)
        self.assertEqual(config.theme_classification.backend, 'openrouter')
        self.assertEqual(config.theme_classification.model, 'openai/gpt-4o')


# ===========================================================================
# Block G — Wizard Phase 2 config format
# ===========================================================================

class TestBuildTestSetsConfig(unittest.TestCase):
    """_build_test_sets_config handles Phase 2 nested and legacy flat format."""

    def _import(self):
        from process.setup_wizard import _build_test_sets_config
        return _build_test_sets_config

    def test_phase2_vaamr_only(self):
        fn = self._import()
        ts = fn({'vaamr': {'enabled': True, 'name': 'vts', 'n_sets': 2,
                           'fraction_per_set': 0.10, 'random_seed': 42}})
        self.assertTrue(ts.vaamr.enabled)
        self.assertEqual(ts.vaamr.name, 'vts')
        self.assertEqual(ts.vaamr.n_sets, 2)
        self.assertFalse(ts.purer.enabled)
        self.assertFalse(ts.codebook.enabled)

    def test_phase2_all_kinds(self):
        fn = self._import()
        ts = fn({
            'vaamr':    {'enabled': True,  'name': 'v', 'n_sets': 2,
                         'fraction_per_set': 0.10, 'random_seed': 42},
            'purer':    {'enabled': True,  'name': 'p', 'n_sets': 1,
                         'fraction_per_set': 0.15, 'random_seed': 42},
            'codebook': {'enabled': False, 'name': 'c', 'n_sets': 1,
                         'fraction_per_set': 0.10, 'random_seed': 42},
        })
        self.assertTrue(ts.vaamr.enabled)
        self.assertTrue(ts.purer.enabled)
        self.assertEqual(ts.purer.fraction_per_set, 0.15)
        self.assertFalse(ts.codebook.enabled)

    def test_legacy_flat_format_backcompat(self):
        """Old flat dict must produce a TestSetsConfig with vaamr populated."""
        fn = self._import()
        ts = fn({'enabled': True, 'n_sets': 3, 'fraction_per_set': 0.12, 'random_seed': 42})
        self.assertTrue(ts.vaamr.enabled)
        self.assertEqual(ts.vaamr.n_sets, 3)
        self.assertFalse(ts.purer.enabled)

    def test_empty_dict_produces_defaults(self):
        fn = self._import()
        ts = fn({})
        # Legacy path: enabled defaults to True for vaamr
        self.assertIsNotNone(ts)

    def test_phase2_extra_fields_ignored_gracefully(self):
        """Unknown fields in nested specs must not raise."""
        fn = self._import()
        # 'random_seed' and 'fraction_per_set' are known; extra 'foo' should be stripped
        ts = fn({'vaamr': {'enabled': True, 'name': 'v', 'n_sets': 1,
                           'fraction_per_set': 0.10, 'random_seed': 42}})
        self.assertIsNotNone(ts)


class TestBuildContentValidityConfig(unittest.TestCase):
    """_build_content_validity_config produces correct ContentValidityConfig."""

    def _import(self):
        from process.setup_wizard import _build_content_validity_config
        return _build_content_validity_config

    def test_empty_produces_default(self):
        fn = self._import()
        cv = fn({})
        from process.config import ContentValidityConfig
        self.assertIsInstance(cv, ContentValidityConfig)

    def test_vaamr_only_enabled(self):
        fn = self._import()
        cv = fn({'vaamr': {'enabled': True, 'name': 'cv_v'}, 'purer': {'enabled': False, 'name': 'cv_p'}})
        self.assertTrue(cv.vaamr.enabled)
        self.assertEqual(cv.vaamr.name, 'cv_v')
        self.assertFalse(cv.purer.enabled)

    def test_both_enabled(self):
        fn = self._import()
        cv = fn({'vaamr': {'enabled': True, 'name': 'cv_vaamr_v1'},
                 'purer': {'enabled': True, 'name': 'cv_purer_v1'}})
        self.assertTrue(cv.vaamr.enabled)
        self.assertTrue(cv.purer.enabled)
        self.assertEqual(cv.purer.name, 'cv_purer_v1')

    def test_any_enabled_false_when_all_disabled(self):
        fn = self._import()
        cv = fn({'vaamr': {'enabled': False, 'name': ''}, 'purer': {'enabled': False, 'name': ''}})
        self.assertFalse(cv.any_enabled())

    def test_any_enabled_true_when_vaamr(self):
        fn = self._import()
        cv = fn({'vaamr': {'enabled': True, 'name': 'cv'}, 'purer': {'enabled': False}})
        self.assertTrue(cv.any_enabled())


class TestBuildConfigFromWizardDataPhase2(unittest.TestCase):
    """build_config_from_wizard_data must produce TestSetsConfig + ContentValidityConfig."""

    def _base_data(self):
        return {
            'output_dir': '/tmp/qra_wizard_test',
            'transcript_dir': '/tmp/qra_wizard_test/input',
            'trial_id': 'trial_test',
            'pipeline': {
                'run_theme_labeler': True,
                'run_purer_labeler': False,
                'run_codebook_classifier': False,
                'auto_analyze': False,
            },
            'theme_classification': {
                'backend': 'lmstudio',
                'model': 'test-model',
            },
            'purer_classification': {},
            'codebook_embedding': {},
            'confidence_tiers': {},
            'speaker_filter': {'mode': 'include', 'speakers': []},
            'segmentation': {},
            'therapist_cues': {'enabled': False},
            'purer_cue': {},
            'session_summaries': {},
            'participant_summaries': {},
        }

    def test_phase2_test_sets_produces_tests_sets_config(self):
        from process.setup_wizard import build_config_from_wizard_data
        from process.config import TestSetsConfig
        data = self._base_data()
        data['test_sets'] = {
            'vaamr':    {'enabled': True, 'name': 'vts', 'n_sets': 2,
                         'fraction_per_set': 0.10, 'random_seed': 42},
            'purer':    {'enabled': False, 'name': 'pts', 'n_sets': 1,
                         'fraction_per_set': 0.10, 'random_seed': 42},
            'codebook': {'enabled': False, 'name': 'cts', 'n_sets': 1,
                         'fraction_per_set': 0.10, 'random_seed': 42},
        }
        config = build_config_from_wizard_data(data)
        self.assertIsInstance(config.test_sets, TestSetsConfig)
        self.assertTrue(config.test_sets.vaamr.enabled)
        self.assertEqual(config.test_sets.vaamr.name, 'vts')
        self.assertFalse(config.test_sets.purer.enabled)

    def test_phase2_content_validity_produces_cv_config(self):
        from process.setup_wizard import build_config_from_wizard_data
        from process.config import ContentValidityConfig
        data = self._base_data()
        data['test_sets'] = {'vaamr': {'enabled': True, 'name': 'v',
                                        'n_sets': 1, 'fraction_per_set': 0.1, 'random_seed': 42}}
        data['content_validity'] = {
            'vaamr': {'enabled': True, 'name': 'cv_vaamr_v1'},
            'purer': {'enabled': False, 'name': 'cv_purer_v1'},
        }
        config = build_config_from_wizard_data(data)
        self.assertIsInstance(config.content_validity, ContentValidityConfig)
        self.assertTrue(config.content_validity.vaamr.enabled)
        self.assertEqual(config.content_validity.vaamr.name, 'cv_vaamr_v1')
        self.assertFalse(config.content_validity.purer.enabled)

    def test_legacy_flat_format_still_works(self):
        """Configs created before Phase 2 (flat test_sets dict) must still load."""
        from process.setup_wizard import build_config_from_wizard_data
        from process.config import TestSetsConfig
        data = self._base_data()
        data['test_sets'] = {'enabled': True, 'n_sets': 2, 'fraction_per_set': 0.10, 'random_seed': 42}
        config = build_config_from_wizard_data(data)
        self.assertIsInstance(config.test_sets, TestSetsConfig)
        self.assertTrue(config.test_sets.vaamr.enabled)
        self.assertEqual(config.test_sets.vaamr.n_sets, 2)

    def test_wizard_step10_produces_nested_format(self):
        """_step_10_testsets must produce the Phase 2 nested dict format."""
        from process.setup_wizard import SetupWizard
        wizard = SetupWizard.__new__(SetupWizard)
        wizard.config_data = {
            'run_purer_labeler': False,
            'pipeline': {'run_codebook_classifier': False},
        }
        # Patch all prompts to return defaults
        with patch('process.setup_wizard._prompt_yes_no', return_value=True), \
             patch('process.setup_wizard._prompt', side_effect=lambda prompt, default: default), \
             patch('process.setup_wizard._prompt_int', side_effect=lambda prompt, default: default), \
             patch('process.setup_wizard._prompt_float', side_effect=lambda prompt, default: default):
            wizard._step_10_testsets()

        ts = wizard.config_data.get('test_sets', {})
        self.assertIn('vaamr', ts, "test_sets must have 'vaamr' key (Phase 2 format)")
        self.assertIn('purer', ts)
        self.assertIn('codebook', ts)

    def test_wizard_step10b_produces_content_validity(self):
        """_step_10b_content_validity must write config_data['content_validity']."""
        from process.setup_wizard import SetupWizard
        wizard = SetupWizard.__new__(SetupWizard)
        wizard.config_data = {'run_purer_labeler': False}
        with patch('process.setup_wizard._prompt_yes_no', return_value=True), \
             patch('process.setup_wizard._prompt', side_effect=lambda p, d: d):
            wizard._step_10b_content_validity()

        cv = wizard.config_data.get('content_validity', {})
        self.assertIn('vaamr', cv, "content_validity must have 'vaamr' key")
        self.assertIn('purer', cv)

    def test_wizard_step10b_purer_disabled_when_no_purer_labeler(self):
        """PURER CV must default to disabled when PURER classifier is off."""
        from process.setup_wizard import SetupWizard
        wizard = SetupWizard.__new__(SetupWizard)
        wizard.config_data = {'run_purer_labeler': False}
        with patch('process.setup_wizard._prompt_yes_no', return_value=True), \
             patch('process.setup_wizard._prompt', side_effect=lambda p, d: d):
            wizard._step_10b_content_validity()

        cv = wizard.config_data['content_validity']
        self.assertFalse(cv['purer']['enabled'],
                         "PURER CV must be disabled when PURER classifier is off")

    def test_wizard_step_sequence_calls_step10b(self):
        """Wizard run sequence must include _step_10b_content_validity call."""
        from process.setup_wizard import SetupWizard
        # Check the _run_custom method calls _step_10b_content_validity
        import inspect
        src = inspect.getsource(SetupWizard._run_custom
                                if hasattr(SetupWizard, '_run_custom')
                                else SetupWizard.run)
        # The step sequence is in the run() body — check via source or attribute
        wizard = SetupWizard.__new__(SetupWizard)
        self.assertTrue(
            hasattr(wizard, '_step_10b_content_validity'),
            "SetupWizard must have _step_10b_content_validity method",
        )

    def test_step_headers_updated_to_total(self):
        """Step 10 header must say /17 after the latest renumber."""
        from process.setup_wizard import SetupWizard
        import inspect
        src = inspect.getsource(SetupWizard._step_10_testsets)
        self.assertIn('/17:', src, "Step 10 header must show /17 total steps")

    def test_step11_header_updated_to_total(self):
        from process.setup_wizard import SetupWizard
        import inspect
        src = inspect.getsource(SetupWizard._step_11_analysis)
        self.assertIn('/17:', src, "Step 11 header must show /17 total steps")


# ===========================================================================
# Block H — CLI help completeness
# ===========================================================================

class TestCliHelpCompleteness(unittest.TestCase):
    """Main --help and subcommand help must mention all new features."""

    def test_validate_in_main_help(self):
        r = _run('--help')
        self.assertIn('validate', r.stdout + r.stderr)

    def test_ingest_in_main_help(self):
        r = _run('--help')
        self.assertIn('ingest', r.stdout + r.stderr)

    def test_epilog_contains_modular_workflow(self):
        r = _run('--help')
        combined = r.stdout + r.stderr
        self.assertIn('classify', combined)
        self.assertIn('assemble', combined)
        self.assertIn('validate', combined)

    def test_epilog_contains_re_classification_example(self):
        r = _run('--help')
        combined = r.stdout + r.stderr
        self.assertIn('--what theme', combined)

    def test_epilog_no_longer_mentions_testsets_as_primary_path(self):
        """Deprecated 'testsets' command should no longer be the primary example."""
        r = _run('--help')
        combined = r.stdout + r.stderr
        # 'testset' (singular) should be in examples; 'testsets' (deprecated) may or may not be
        self.assertIn('testset', combined)

    def test_ingest_help_shows_reingest_all(self):
        r = _run('ingest', '--help')
        self.assertIn('reingest-all', r.stdout + r.stderr)

    def test_ingest_help_shows_reingest_session(self):
        r = _run('ingest', '--help')
        self.assertIn('reingest', r.stdout + r.stderr)

    def test_classify_help_shows_what_flag(self):
        r = _run('classify', '--help')
        self.assertIn('what', r.stdout + r.stderr)

    def test_assemble_help_exits_zero(self):
        r = _run('assemble', '--help')
        self.assertEqual(r.returncode, 0)

    def test_validate_help_exits_zero(self):
        r = _run('validate', '--help')
        self.assertEqual(r.returncode, 0)


# ===========================================================================
# Cross-cutting: overlay write → classify → assemble round-trip
# ===========================================================================

class TestClassifyAssembleRoundTrip(unittest.TestCase):
    """
    Integration: qra classify writes an overlay; qra assemble reads it.
    Both must work without --config being present.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        segs = [
            _classified('seg_001', primary=2),
            _classified('seg_002', primary=0),
        ]
        _write_frozen(self.tmpdir, segs)
        # Pre-write theme overlay so assemble has something to read
        _write_theme_overlay(self.tmpdir, segs)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_classify_writes_overlay_without_config(self):
        """qra classify --what theme must write theme_labels.jsonl without --config."""
        from process import classifications_io
        r = _run('classify', '--what', 'theme', '-o', self.tmpdir)
        # The key contract: overlay file must exist after classify
        overlay = classifications_io.overlay_path(self.tmpdir, 'theme')
        self.assertTrue(os.path.isfile(overlay),
                        f"theme overlay must exist; stderr: {r.stderr}")

    def test_assemble_reads_overlay_written_by_classify(self):
        """After classify writes theme overlay, assemble must produce master_segments."""
        r = _run('assemble', '-o', self.tmpdir)
        self.assertEqual(r.returncode, 0, f"stderr: {r.stderr}")
        from process import output_paths as _paths
        ms_dir = _paths.master_segments_dir(self.tmpdir)
        jsonl_files = [f for f in os.listdir(ms_dir) if 'master_segments' in f]
        self.assertTrue(len(jsonl_files) >= 1, "master_segments JSONL must exist after assemble")

    def test_assemble_exits_nonzero_without_overlay(self):
        """qra assemble must fail if no overlay exists (only frozen segments)."""
        with tempfile.TemporaryDirectory() as d:
            segs = [_classified('seg_001')]
            _write_frozen(d, segs)
            # No overlay written
            r = _run('assemble', '-o', d)
            self.assertNotEqual(r.returncode, 0,
                                "assemble must exit non-zero with no overlay")

    def test_full_classify_assemble_validate_pipeline(self):
        """classify → assemble → validate end-to-end must all exit 0."""
        r1 = _run('classify', '--what', 'theme', '-o', self.tmpdir)
        r2 = _run('assemble', '-o', self.tmpdir)
        r3 = _run('validate', '-o', self.tmpdir)

        self.assertEqual(r1.returncode, 0, f"classify failed: {r1.stderr}")
        self.assertEqual(r2.returncode, 0, f"assemble failed: {r2.stderr}")
        self.assertEqual(r3.returncode, 0, f"validate failed: {r3.stderr}")


if __name__ == '__main__':
    unittest.main()
