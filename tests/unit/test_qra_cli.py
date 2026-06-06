"""
tests/unit/test_qra_cli.py
--------------------------
Pure arg/config helper tests for qra.py CLI entry point.

Covers (without running the pipeline or any LLM):
  1. _flatten_wizard_config passes gnn_layer sub-dict through unchanged
  2. argparse: `classify --backend gnn` routes to run_gnn_classify
     (monkeypatched — no real classify)
  3. argparse: `analyze --gnn` / `analyze --no-gnn` parse to the correct
     force_gnn flag values and run_analysis is called with the right kwarg
"""
import argparse
import os
import sys
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch, call

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import qra


# ---------------------------------------------------------------------------
# 1. _flatten_wizard_config — gnn_layer pass-through
# ---------------------------------------------------------------------------

class TestFlattenWizardConfigGnn(unittest.TestCase):
    """_flatten_wizard_config must pass gnn_layer through to the result dict."""

    def test_gnn_layer_present_when_in_data(self):
        data = {
            'gnn_layer': {'enabled': True, 'gnn_authoritative': False},
        }
        result = qra._flatten_wizard_config(data)
        self.assertIn('gnn_layer', result)
        self.assertEqual(result['gnn_layer']['enabled'], True)
        self.assertEqual(result['gnn_layer']['gnn_authoritative'], False)

    def test_gnn_layer_absent_when_not_in_data(self):
        data = {'theme_classification': {'model': 'some-model'}}
        result = qra._flatten_wizard_config(data)
        self.assertNotIn('gnn_layer', result)

    def test_gnn_layer_full_wizard_structure(self):
        """Wizard-style nested config with pipeline block + gnn_layer sibling."""
        data = {
            'pipeline': {
                'transcript_dir': '/data/input',
                'output_dir': '/data/output',
                'run_theme_labeler': True,
            },
            'theme_classification': {'model': 'qwen/qwen-3-70b', 'n_runs': 3},
            'gnn_layer': {
                'enabled': True,
                'gnn_authoritative': True,
                'n_layers': 3,
            },
        }
        result = qra._flatten_wizard_config(data)

        # Pipeline-level keys lifted to top
        self.assertEqual(result.get('transcript_dir'), '/data/input')
        self.assertEqual(result.get('output_dir'), '/data/output')

        # theme_classification preserved
        self.assertIn('theme_classification', result)
        self.assertEqual(result['theme_classification']['n_runs'], 3)

        # gnn_layer preserved verbatim
        self.assertIn('gnn_layer', result)
        self.assertEqual(result['gnn_layer']['n_layers'], 3)

    def test_all_sub_config_keys_are_forwarded(self):
        """All documented sub-config keys must pass through."""
        expected_keys = [
            'segmentation', 'speaker_filter', 'theme_classification',
            'codebook_embedding', 'codebook_llm', 'codebook_ensemble',
            'validation', 'confidence_tiers', 'test_sets', 'content_validity',
            'purer_classification', 'purer_cue', 'therapist_cues',
            'session_summaries', 'participant_summaries', 'gnn_layer',
        ]
        data = {k: {'dummy': 1} for k in expected_keys}
        result = qra._flatten_wizard_config(data)
        for k in expected_keys:
            self.assertIn(k, result, f"Sub-config key '{k}' missing from flattened result")


# ---------------------------------------------------------------------------
# 2. classify --backend gnn routes to run_gnn_classify
# ---------------------------------------------------------------------------

class TestClassifyBackendGnn(unittest.TestCase):
    """
    `qra classify --backend gnn` must call gnn_layer.runner.run_gnn_classify
    instead of the LLM-based classifiers, and must not call stage_classify_theme.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create a minimal frozen-segment directory so classify doesn't
        # error out on "no frozen segments" guard.
        seg_dir = os.path.join(self.tmpdir, '01_transcripts', 'segmented', 'c1s1')
        os.makedirs(seg_dir, exist_ok=True)
        # Write a minimal segments.jsonl so list_segmented_sessions finds it
        with open(os.path.join(seg_dir, 'segments.jsonl'), 'w') as f:
            f.write('')

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _parse(self, extra_args):
        parser, *_ = qra._build_parser()
        return parser.parse_args(['classify', '-o', self.tmpdir] + extra_args)

    def test_backend_gnn_in_choices(self):
        """
        The classify parser accepts arbitrary --backend values (it's not
        restricted to a choices list), so --backend gnn must parse cleanly.
        """
        args = self._parse(['--backend', 'gnn', '--no-downstream'])
        self.assertEqual(args.backend, 'gnn')

    def test_classify_backend_gnn_calls_run_gnn_classify(self):
        """
        When --backend gnn, cmd_classify must call run_gnn_classify and
        must NOT call stage_classify_theme.
        """
        fake_classify_result = {'status': 'ok', 'n_classified': 5}

        with patch('process.segments_io.list_segmented_sessions',
                   return_value=['c1s1']), \
             patch('process.segments_io.load_segments_for_stage',
                   return_value=[]), \
             patch('process.classifications_io.apply_overlays'), \
             patch('process.orchestrator.stage_classify_theme') as mock_theme, \
             patch('gnn_layer.runner.run_gnn_classify',
                   return_value=fake_classify_result) as mock_gnn, \
             patch('pandas.DataFrame') as mock_df:

            # Build a minimal args namespace directly
            args = argparse.Namespace(
                output_dir=self.tmpdir,
                what='all',
                backend='gnn',
                model=None,
                api_key=None,
                config=None,
                framework=None,
                codebook=None,
                no_downstream=True,      # skip assemble/analyze cascade
                zero_shot=False,
                transcript_dir=None,
                trial_id=None,
                n_runs=None,
                temperature=None,
                resume_from=None,
                speaker_filter_mode=None,
                exclude_speakers=None,
                no_theme_labeler=False,
                run_codebook_classifier=False,
                no_codebook_classifier=False,
                verbose_segmentation=False,
                no_auto_analyze=False,
                no_text_anonymization=False,
                lmstudio_url=None,
                models=None,
                no_two_pass=False,
                embedding_model=None,
                exemplar_import_path=None,
                criteria_weight=None,
                exemplar_weight=None,
                exemplar_confidence_threshold=None,
                max_exemplar_tokens=None,
                high_confidence_threshold=None,
                medium_confidence_threshold=None,
            )

            qra.cmd_classify(args)

        mock_gnn.assert_called_once()
        mock_theme.assert_not_called()


# ---------------------------------------------------------------------------
# 3. analyze --gnn / --no-gnn arg parsing
# ---------------------------------------------------------------------------

class TestAnalyzeGnnFlags(unittest.TestCase):
    """
    `qra analyze --gnn` and `qra analyze --no-gnn` must parse correctly
    and call run_analysis with the appropriate force_gnn value.
    """

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        # Create the output dir so cmd_analyze doesn't fail the os.path.isdir check
        os.makedirs(self.tmpdir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _parse_analyze(self, extra_args):
        parser, *_ = qra._build_parser()
        return parser.parse_args(['analyze', '-o', self.tmpdir] + extra_args)

    # --- parser-level assertions ---

    def test_no_flags_gnn_and_no_gnn_both_false(self):
        args = self._parse_analyze([])
        self.assertFalse(args.gnn)
        self.assertFalse(args.no_gnn)

    def test_gnn_flag_sets_gnn_true(self):
        args = self._parse_analyze(['--gnn'])
        self.assertTrue(args.gnn)
        self.assertFalse(args.no_gnn)

    def test_no_gnn_flag_sets_no_gnn_true(self):
        args = self._parse_analyze(['--no-gnn'])
        self.assertFalse(args.gnn)
        self.assertTrue(args.no_gnn)

    def test_gnn_and_no_gnn_are_mutually_exclusive(self):
        """Passing both --gnn and --no-gnn must raise a parse error."""
        parser, *_ = qra._build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(['analyze', '-o', self.tmpdir, '--gnn', '--no-gnn'])

    # --- cmd_analyze calls run_analysis with the right force_gnn ---

    def _run_analyze_with_mock(self, extra_args):
        """Call cmd_analyze with mocked run_analysis; return the mock."""
        args = self._parse_analyze(extra_args)
        fake_result = {
            'n_segments': 10,
            'n_participants': 2,
            'n_sessions': 3,
            'files_generated': ['a.txt'],
        }
        with patch('analysis.runner.run_analysis', return_value=fake_result) as mock_run:
            qra.cmd_analyze(args)
        return mock_run

    def test_no_flags_calls_run_analysis_force_gnn_none(self):
        mock_run = self._run_analyze_with_mock([])
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertIsNone(kwargs.get('force_gnn'))

    def test_gnn_flag_calls_run_analysis_force_gnn_true(self):
        mock_run = self._run_analyze_with_mock(['--gnn'])
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertTrue(kwargs.get('force_gnn'))

    def test_no_gnn_flag_calls_run_analysis_force_gnn_false(self):
        mock_run = self._run_analyze_with_mock(['--no-gnn'])
        mock_run.assert_called_once()
        _, kwargs = mock_run.call_args
        self.assertIs(kwargs.get('force_gnn'), False)


# ---------------------------------------------------------------------------
# 4. _build_parser returns a 2-tuple
# ---------------------------------------------------------------------------

class TestBuildParser(unittest.TestCase):
    """Ensure _build_parser is importable and returns the expected shape."""

    def test_build_parser_returns_tuple(self):
        result = qra._build_parser()
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)

    def test_first_element_is_argument_parser(self):
        parser, *_ = qra._build_parser()
        self.assertIsInstance(parser, argparse.ArgumentParser)


if __name__ == '__main__':
    unittest.main()
