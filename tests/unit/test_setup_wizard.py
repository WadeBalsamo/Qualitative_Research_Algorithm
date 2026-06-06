"""
tests/unit/test_setup_wizard.py
--------------------------------
Unit tests for process/setup_wizard.py.

Covers the pure/deterministic helpers without real interactivity:
  1. build_config_from_wizard_data — maps wizard dict onto PipelineConfig,
     including gnn_layer sub-dict applied to GnnLayerConfig fields.
  2. _GNN_KNOB_REFERENCE — constant must be a list of strings documenting the
     GNN knobs; checked for presence and expected members.
  3. _step_11d_gnn — GNN wizard sub-step; monkeypatching builtins.input feeds
     deterministic answers and the method must write config_data['gnn_layer']
     with the correct keys/values.
  4. _validate_speaker_anonymization_key — pure path (file-based but uses
     tempfile, no network).
  5. _build_test_sets_config / _build_content_validity_config — standalone
     builder helpers.
"""
import builtins
import json
import os
import sys
import tempfile
import shutil
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from process.setup_wizard import (
    build_config_from_wizard_data,
    _GNN_KNOB_REFERENCE,
    _validate_speaker_anonymization_key,
    _build_test_sets_config,
    _build_content_validity_config,
    SetupWizard,
)
from process.config import PipelineConfig
from gnn_layer.config import GnnLayerConfig


# ---------------------------------------------------------------------------
# Helper: minimal wizard data that build_config_from_wizard_data accepts
# ---------------------------------------------------------------------------

def _minimal_wizard_data(**overrides):
    base = {
        'pipeline': {
            'transcript_dir': '/tmp/transcripts',
            'output_dir': '/tmp/output',
            'trial_id': 'test_trial',
            'run_theme_labeler': True,
            'run_codebook_classifier': False,
            'anonymize_transcript_text': True,
            'anonymize_text_model': 'obi/deid_roberta_i2b2',
            'auto_analyze': False,
        },
        'theme_classification': {
            'backend': 'lmstudio',
            'model': 'test/model',
            'summarization_model': 'test/summ',
            'lmstudio_base_url': 'http://localhost:1234/v1',
            'n_runs': 3,
            'temperature': 0.1,
            'per_run_models': ['test/model', 'test/checker'],
        },
        'segmentation': {},
        'speaker_filter': {'mode': 'none', 'speakers': []},
        'confidence_tiers': {'high_confidence': 0.8, 'medium_min_confidence': 0.6},
        'test_sets': {},
        'content_validity': {},
        'therapist_cues': {},
        'purer_cue': {},
        'session_summaries': {},
        'participant_summaries': {},
        'gnn_layer': {},
        'superposition': {},
        'efficacy': {},
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# build_config_from_wizard_data
# ---------------------------------------------------------------------------

class TestBuildConfigFromWizardData(unittest.TestCase):

    def test_returns_pipeline_config_instance(self):
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertIsInstance(cfg, PipelineConfig)

    def test_pipeline_fields_mapped(self):
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertEqual(cfg.transcript_dir, '/tmp/transcripts')
        self.assertEqual(cfg.output_dir, '/tmp/output')
        self.assertEqual(cfg.trial_id, 'test_trial')

    def test_auto_analyze_false_propagated(self):
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertFalse(cfg.auto_analyze)

    def test_anonymize_fields(self):
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertTrue(cfg.anonymize_transcript_text)
        self.assertEqual(cfg.anonymize_text_model, 'obi/deid_roberta_i2b2')

    def test_theme_classification_backend(self):
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertEqual(cfg.theme_classification.backend, 'lmstudio')
        self.assertEqual(cfg.theme_classification.model, 'test/model')

    def test_gnn_layer_defaults_when_empty(self):
        """Empty gnn_layer dict produces a GnnLayerConfig with default values."""
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertIsInstance(cfg.gnn_layer, GnnLayerConfig)
        # Default enabled=True from GnnLayerConfig dataclass
        self.assertTrue(cfg.gnn_layer.enabled)

    def test_gnn_layer_enabled_false_propagated(self):
        data = _minimal_wizard_data(gnn_layer={'enabled': False})
        cfg = build_config_from_wizard_data(data)
        self.assertFalse(cfg.gnn_layer.enabled)

    def test_gnn_layer_fields_applied(self):
        """Non-default GNN fields from wizard dict are reflected in config."""
        gnn_data = {
            'enabled': True,
            'label_mode': 'human',
            'validation_folds': 10,
            'irr_target': 0.85,
            'run_on_participants': False,
            'run_on_therapists': True,
            'run_gnn_ablation': True,
            'n_motif_clusters': 8,
            'n_latent_factors': 3,
            'gnn_authoritative': True,
        }
        data = _minimal_wizard_data(gnn_layer=gnn_data)
        cfg = build_config_from_wizard_data(data)
        gnn = cfg.gnn_layer
        self.assertEqual(gnn.label_mode, 'human')
        self.assertEqual(gnn.validation_folds, 10)
        self.assertAlmostEqual(gnn.irr_target, 0.85)
        self.assertFalse(gnn.run_on_participants)
        self.assertTrue(gnn.run_on_therapists)
        self.assertTrue(gnn.run_gnn_ablation)
        self.assertEqual(gnn.n_motif_clusters, 8)
        self.assertEqual(gnn.n_latent_factors, 3)
        self.assertTrue(gnn.gnn_authoritative)

    def test_unknown_gnn_keys_ignored(self):
        """Extra keys in gnn_layer that don't exist as GnnLayerConfig fields are ignored."""
        data = _minimal_wizard_data(gnn_layer={'enabled': False, '__nonexistent__': 'x'})
        # Must not raise
        cfg = build_config_from_wizard_data(data)
        self.assertFalse(cfg.gnn_layer.enabled)

    def test_confidence_tiers(self):
        data = _minimal_wizard_data(
            confidence_tiers={'high_confidence': 0.9, 'medium_min_confidence': 0.65}
        )
        cfg = build_config_from_wizard_data(data)
        self.assertAlmostEqual(cfg.confidence_tiers.high_confidence, 0.9)
        self.assertAlmostEqual(cfg.confidence_tiers.medium_min_confidence, 0.65)

    def test_purer_classification_falls_back_to_theme_classification(self):
        """When no 'purer_classification' key exists, it inherits from theme_classification."""
        data = _minimal_wizard_data()
        data.pop('purer_classification', None)
        cfg = build_config_from_wizard_data(data)
        # Must not raise; purer backend inherits from tc
        self.assertEqual(cfg.purer_classification.backend, 'lmstudio')

    def test_segmentation_defaults_applied(self):
        """Empty segmentation dict uses sensible defaults."""
        cfg = build_config_from_wizard_data(_minimal_wizard_data())
        self.assertIsNotNone(cfg.segmentation.embedding_model)
        self.assertGreater(cfg.segmentation.max_gap_seconds, 0)


# ---------------------------------------------------------------------------
# _GNN_KNOB_REFERENCE
# ---------------------------------------------------------------------------

class TestGnnKnobReference(unittest.TestCase):

    def test_is_list(self):
        self.assertIsInstance(_GNN_KNOB_REFERENCE, list)

    def test_non_empty(self):
        self.assertGreater(len(_GNN_KNOB_REFERENCE), 0)

    def test_all_strings(self):
        for item in _GNN_KNOB_REFERENCE:
            self.assertIsInstance(item, str, msg=f"Non-string entry: {item!r}")

    def test_enabled_knob_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('enabled', joined)

    def test_label_mode_knob_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('label_mode', joined)

    def test_gnn_authoritative_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('gnn_authoritative', joined)

    def test_validation_folds_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('validation_folds', joined)

    def test_irr_target_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('irr_target', joined)

    def test_n_motif_clusters_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('n_motif_clusters', joined)

    def test_n_latent_factors_documented(self):
        joined = '\n'.join(_GNN_KNOB_REFERENCE)
        self.assertIn('n_latent_factors', joined)


# ---------------------------------------------------------------------------
# _step_11d_gnn — monkeypatch builtins.input
# ---------------------------------------------------------------------------

class TestStep11dGnn(unittest.TestCase):
    """
    Drive _step_11d_gnn with deterministic input() answers.

    Input sequence for GNN ENABLED path:
      1. "Enable the GNN layer?" -> 'y'
      2. "Label mode" -> 'weak'
      3. "Position participant segments?" -> 'y'
      4. "Position therapist segments?" -> 'y'
      5. "Cross-validation folds" -> '5'
      6. "Target kappa" -> '0.70'
      7. "Make GNN labels authoritative now?" -> 'n'
      8. "Run construct-signal ablation?" -> 'n'
      9. "Number of cue-language motif clusters" -> '12'
      10. "Number of coupling latent factors" -> '5'
    """

    def _run_step(self, answers):
        wizard = SetupWizard()
        wizard.config_data = {}
        answers_iter = iter(answers)

        def fake_input(prompt=''):
            return next(answers_iter)

        original_input = builtins.input
        builtins.input = fake_input
        try:
            wizard._step_11d_gnn()
        finally:
            builtins.input = original_input
        return wizard.config_data

    def test_gnn_disabled_path(self):
        """Answer 'n' to 'Enable GNN' -> config_data['gnn_layer'] = {'enabled': False}."""
        config = self._run_step(['n'])
        self.assertIn('gnn_layer', config)
        self.assertFalse(config['gnn_layer']['enabled'])
        # Only the 'enabled' key set in disabled branch
        self.assertEqual(config['gnn_layer'], {'enabled': False})

    def test_gnn_enabled_sets_enabled_true(self):
        """Answer 'y' -> gnn_layer['enabled'] = True."""
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertTrue(config['gnn_layer']['enabled'])

    def test_gnn_enabled_label_mode_weak(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertEqual(config['gnn_layer']['label_mode'], 'weak')

    def test_gnn_enabled_label_mode_human(self):
        answers = ['y', 'human', 'y', 'y', '5', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertEqual(config['gnn_layer']['label_mode'], 'human')

    def test_gnn_enabled_run_on_participants_true(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertTrue(config['gnn_layer']['run_on_participants'])

    def test_gnn_enabled_run_on_therapists_false(self):
        """Answer 'n' to therapist scope -> run_on_therapists=False."""
        answers = ['y', 'weak', 'y', 'n', '5', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertFalse(config['gnn_layer']['run_on_therapists'])

    def test_gnn_enabled_validation_folds(self):
        answers = ['y', 'weak', 'y', 'y', '7', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertEqual(config['gnn_layer']['validation_folds'], 7)

    def test_gnn_enabled_irr_target(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.80', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertAlmostEqual(config['gnn_layer']['irr_target'], 0.80)

    def test_gnn_enabled_authoritative_false(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertFalse(config['gnn_layer']['gnn_authoritative'])

    def test_gnn_enabled_authoritative_true(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'y', 'n', '12', '5']
        config = self._run_step(answers)
        self.assertTrue(config['gnn_layer']['gnn_authoritative'])

    def test_gnn_enabled_ablation_true(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'y', '12', '5']
        config = self._run_step(answers)
        self.assertTrue(config['gnn_layer']['run_gnn_ablation'])

    def test_gnn_enabled_n_motif_clusters(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'n', '9', '5']
        config = self._run_step(answers)
        self.assertEqual(config['gnn_layer']['n_motif_clusters'], 9)

    def test_gnn_enabled_n_latent_factors(self):
        answers = ['y', 'weak', 'y', 'y', '5', '0.70', 'n', 'n', '12', '3']
        config = self._run_step(answers)
        self.assertEqual(config['gnn_layer']['n_latent_factors'], 3)


# ---------------------------------------------------------------------------
# _validate_speaker_anonymization_key
# ---------------------------------------------------------------------------

class TestValidateSpeakerAnonymizationKey(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _write_key(self, data):
        path = os.path.join(self.tmpdir, 'key.json')
        with open(path, 'w') as f:
            json.dump(data, f)
        return path

    def test_valid_key_accepted(self):
        data = {
            'Alice Jones': {'role': 'participant', 'anonymized_id': 'P001'},
            'Dr. Smith': {'role': 'therapist', 'anonymized_id': 'T001'},
        }
        path = self._write_key(data)
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertTrue(ok)
        self.assertIsNone(err)

    def test_missing_file_returns_error(self):
        ok, err = _validate_speaker_anonymization_key('/nonexistent/path.json')
        self.assertFalse(ok)
        self.assertIn('not found', err.lower())

    def test_invalid_json_returns_error(self):
        path = os.path.join(self.tmpdir, 'bad.json')
        with open(path, 'w') as f:
            f.write('{not valid json')
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertFalse(ok)
        self.assertIn('json', err.lower())

    def test_root_not_dict_fails(self):
        path = self._write_key([1, 2, 3])
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertFalse(ok)
        self.assertIn('object', err.lower())

    def test_entry_not_dict_fails(self):
        path = self._write_key({'Alice': 'not_a_dict'})
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertFalse(ok)

    def test_extra_keys_fail(self):
        data = {'Alice': {'role': 'participant', 'anonymized_id': 'P001', 'extra': 'bad'}}
        path = self._write_key(data)
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertFalse(ok)

    def test_bad_role_fails(self):
        data = {'Alice': {'role': 'admin', 'anonymized_id': 'P001'}}
        path = self._write_key(data)
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertFalse(ok)
        self.assertIn('role', err.lower())

    def test_empty_anon_id_fails(self):
        data = {'Alice': {'role': 'participant', 'anonymized_id': '  '}}
        path = self._write_key(data)
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertFalse(ok)

    def test_therapist_role_accepted(self):
        data = {'Bob': {'role': 'therapist', 'anonymized_id': 'T001'}}
        path = self._write_key(data)
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertTrue(ok)

    def test_empty_object_accepted(self):
        path = self._write_key({})
        ok, err = _validate_speaker_anonymization_key(path)
        self.assertTrue(ok)


# ---------------------------------------------------------------------------
# _build_test_sets_config
# ---------------------------------------------------------------------------

class TestBuildTestSetsConfig(unittest.TestCase):

    def test_nested_phase2_format(self):
        raw = {
            'vaamr':    {'enabled': True, 'name': 'vaamr_ts', 'n_sets': 2,
                         'fraction_per_set': 0.1, 'random_seed': 42},
            'purer':    {'enabled': False, 'name': 'purer_ts', 'n_sets': 1,
                         'fraction_per_set': 0.1, 'random_seed': 42},
            'codebook': {'enabled': False, 'name': 'cb_ts', 'n_sets': 1,
                         'fraction_per_set': 0.1, 'random_seed': 42},
        }
        ts = _build_test_sets_config(raw)
        self.assertTrue(ts.vaamr.enabled)
        self.assertEqual(ts.vaamr.name, 'vaamr_ts')
        self.assertEqual(ts.vaamr.n_sets, 2)
        self.assertFalse(ts.purer.enabled)

    def test_empty_dict_returns_default(self):
        ts = _build_test_sets_config({})
        # Legacy path: empty dict produces a default TestSetsConfig
        from process.config import TestSetsConfig
        self.assertIsInstance(ts, TestSetsConfig)

    def test_legacy_flat_format(self):
        raw = {'enabled': True, 'n_sets': 1, 'fraction_per_set': 0.15, 'random_seed': 7}
        ts = _build_test_sets_config(raw)
        self.assertTrue(ts.vaamr.enabled)
        self.assertEqual(ts.vaamr.n_sets, 1)
        self.assertAlmostEqual(ts.vaamr.fraction_per_set, 0.15)


# ---------------------------------------------------------------------------
# _build_content_validity_config
# ---------------------------------------------------------------------------

class TestBuildContentValidityConfig(unittest.TestCase):

    def test_empty_dict_returns_default(self):
        from process.config import ContentValidityConfig
        cv = _build_content_validity_config({})
        self.assertIsInstance(cv, ContentValidityConfig)

    def test_vaamr_enabled(self):
        raw = {
            'vaamr': {'enabled': True, 'name': 'cv_vaamr_v1'},
            'purer': {'enabled': False, 'name': 'cv_purer_v1'},
        }
        cv = _build_content_validity_config(raw)
        self.assertTrue(cv.vaamr.enabled)
        self.assertEqual(cv.vaamr.name, 'cv_vaamr_v1')
        self.assertFalse(cv.purer.enabled)

    def test_none_dict_returns_default(self):
        from process.config import ContentValidityConfig
        cv = _build_content_validity_config(None or {})
        self.assertIsInstance(cv, ContentValidityConfig)


if __name__ == '__main__':
    unittest.main()
