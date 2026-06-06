"""
tests/unit/test_output_paths.py
--------------------------------
Unit tests for process/output_paths.py.

Verifies every public path helper returns the correct relative layout under
a given run_dir.  No filesystem I/O is required — helpers are pure string
operations.
"""

import os
import sys
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from process import output_paths as P


class TestCriticalLayouts(unittest.TestCase):
    """Assert the exact relative sub-paths documented in CLAUDE.md."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _rel(self, path: str) -> str:
        """Return path relative to self.tmp for readable assertions."""
        return os.path.relpath(path, self.tmp)

    # ── GNN-specific paths ────────────────────────────────────────────────

    def test_reports_gnn_dir(self):
        self.assertEqual(
            P.reports_gnn_dir(self.tmp),
            os.path.join(self.tmp, '06_reports', '06_gnn'),
        )

    def test_gnn_model_dir(self):
        self.assertEqual(
            P.gnn_model_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'gnn'),
        )

    def test_gnn_data_dir(self):
        self.assertEqual(
            P.gnn_data_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'gnn'),
        )

    def test_classification_overlay_path_gnn(self):
        self.assertEqual(
            P.classification_overlay_path(self.tmp, 'gnn'),
            os.path.join(self.tmp, '02_meta', 'classifications', 'gnn_labels.jsonl'),
        )

    # ── Tiered reports layout ─────────────────────────────────────────────

    def test_reports_outcomes_dir(self):
        self.assertEqual(
            P.reports_outcomes_dir(self.tmp),
            os.path.join(self.tmp, '06_reports', '01_outcomes'),
        )

    def test_reports_mechanism_dir(self):
        self.assertEqual(
            P.reports_mechanism_dir(self.tmp),
            os.path.join(self.tmp, '06_reports', '02_mechanism'),
        )

    def test_themes_dir(self):
        self.assertEqual(
            P.themes_dir(self.tmp),
            os.path.join(self.tmp, '06_reports', '05_per_stage'),
        )

    def test_reports_per_session_dir(self):
        self.assertEqual(
            P.reports_per_session_dir(self.tmp),
            os.path.join(self.tmp, '06_reports', '03_per_session'),
        )

    def test_reports_per_participant_dir(self):
        self.assertEqual(
            P.reports_per_participant_dir(self.tmp),
            os.path.join(self.tmp, '06_reports', '04_per_participant'),
        )


class TestProvenanceAndMetaPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_meta_dir(self):
        self.assertEqual(P.meta_dir(self.tmp), os.path.join(self.tmp, '02_meta'))

    def test_auditable_logs_dir(self):
        self.assertEqual(
            P.auditable_logs_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'auditable_logs'),
        )

    def test_llm_raw_dir_is_alias_of_auditable(self):
        self.assertEqual(P.llm_raw_dir(self.tmp), P.auditable_logs_dir(self.tmp))

    def test_llm_classification_log_path(self):
        self.assertEqual(
            P.llm_classification_log_path(self.tmp),
            os.path.join(self.tmp, '02_meta', 'auditable_logs', 'llm_classification_log.txt'),
        )

    def test_llm_prompts_path(self):
        self.assertEqual(
            P.llm_prompts_path(self.tmp),
            os.path.join(self.tmp, '02_meta', 'auditable_logs', 'llm_prompts.txt'),
        )

    def test_llm_checkpoints_dir(self):
        self.assertEqual(
            P.llm_checkpoints_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'auditable_logs', 'checkpoints'),
        )

    def test_theme_definitions_txt_path(self):
        self.assertEqual(
            P.theme_definitions_txt_path(self.tmp),
            os.path.join(self.tmp, '02_meta', 'theme_definitions.txt'),
        )

    def test_anonymization_key_txt_path(self):
        self.assertEqual(
            P.anonymization_key_txt_path(self.tmp),
            os.path.join(self.tmp, '02_meta', 'anonymization_key.txt'),
        )

    def test_codebook_raw_dir(self):
        self.assertEqual(
            P.codebook_raw_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'codebook_raw'),
        )

    def test_mechanism_dir(self):
        self.assertEqual(
            P.mechanism_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'mechanism'),
        )

    def test_efficacy_dir(self):
        self.assertEqual(
            P.efficacy_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'efficacy'),
        )


class TestTranscriptPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_transcripts_diarized_dir(self):
        self.assertEqual(
            P.transcripts_diarized_dir(self.tmp),
            os.path.join(self.tmp, '01_transcripts_inputs'),
        )

    def test_full_transcripts_dir(self):
        self.assertEqual(
            P.full_transcripts_dir(self.tmp),
            os.path.join(self.tmp, '04_validation', 'full_transcripts'),
        )

    def test_transcripts_coded_dir_is_alias_of_full(self):
        self.assertEqual(
            P.transcripts_coded_dir(self.tmp),
            P.full_transcripts_dir(self.tmp),
        )


class TestAnalysisDataPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_analysis_data_dir(self):
        self.assertEqual(
            P.analysis_data_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data'),
        )

    def test_session_stats_dir(self):
        self.assertEqual(
            P.session_stats_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'session_stats'),
        )

    def test_graphing_dir(self):
        self.assertEqual(
            P.graphing_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'graphing'),
        )

    def test_longitudinal_dir(self):
        self.assertEqual(
            P.longitudinal_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data'),
        )

    def test_cumulative_report_dir(self):
        self.assertEqual(
            P.cumulative_report_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data'),
        )

    def test_sessions_json_dir(self):
        self.assertEqual(
            P.sessions_json_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'per_session'),
        )

    def test_participants_json_dir(self):
        self.assertEqual(
            P.participants_json_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'per_participant'),
        )

    def test_themes_json_dir(self):
        self.assertEqual(
            P.themes_json_dir(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'per_theme'),
        )

    def test_session_summaries_json_path(self):
        self.assertEqual(
            P.session_summaries_json_path(self.tmp),
            os.path.join(self.tmp, '03_analysis_data', 'session_summaries.json'),
        )


class TestValidationPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_validation_dir(self):
        self.assertEqual(
            P.validation_dir(self.tmp),
            os.path.join(self.tmp, '04_validation'),
        )

    def test_testsets_dir(self):
        self.assertEqual(
            P.testsets_dir(self.tmp),
            os.path.join(self.tmp, '04_validation', 'testsets'),
        )

    def test_cross_validation_dir(self):
        self.assertEqual(
            P.cross_validation_dir(self.tmp),
            os.path.join(self.tmp, '04_validation', 'cross_validation'),
        )

    def test_content_validity_dir(self):
        self.assertEqual(
            P.content_validity_dir(self.tmp),
            os.path.join(self.tmp, '04_validation', 'content_validity'),
        )

    def test_human_eval_dir(self):
        self.assertEqual(
            P.human_eval_dir(self.tmp),
            os.path.join(self.tmp, '04_validation'),
        )

    def test_figures_dir(self):
        self.assertEqual(
            P.figures_dir(self.tmp),
            os.path.join(self.tmp, '05_figures'),
        )


class TestHumanReportsPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_human_reports_dir(self):
        self.assertEqual(
            P.human_reports_dir(self.tmp),
            os.path.join(self.tmp, '06_reports'),
        )

    def test_executive_summary_path(self):
        self.assertEqual(
            P.executive_summary_path(self.tmp),
            os.path.join(self.tmp, '06_reports', '00_executive_summary.txt'),
        )

    def test_reports_readme_path(self):
        self.assertEqual(
            P.reports_readme_path(self.tmp),
            os.path.join(self.tmp, '06_reports', '00_READ_ME.txt'),
        )

    def test_methods_appendix_path(self):
        self.assertEqual(
            P.methods_appendix_path(self.tmp),
            os.path.join(self.tmp, '06_reports', '07_methods_appendix.txt'),
        )


class TestTrainingDataPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_master_segments_dir(self):
        self.assertEqual(
            P.master_segments_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'training_data'),
        )

    def test_training_data_dir(self):
        self.assertEqual(
            P.training_data_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'training_data'),
        )

    def test_master_segments_and_training_data_same(self):
        self.assertEqual(
            P.master_segments_dir(self.tmp),
            P.training_data_dir(self.tmp),
        )


class TestFrozenSegmentationPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_segmented_sessions_dir(self):
        self.assertEqual(
            P.segmented_sessions_dir(self.tmp),
            os.path.join(self.tmp, '01_transcripts', 'segmented'),
        )

    def test_segmented_session_dir(self):
        self.assertEqual(
            P.segmented_session_dir(self.tmp, 'ses_001'),
            os.path.join(self.tmp, '01_transcripts', 'segmented', 'ses_001'),
        )

    def test_session_segments_path(self):
        self.assertEqual(
            P.session_segments_path(self.tmp, 'ses_001'),
            os.path.join(self.tmp, '01_transcripts', 'segmented', 'ses_001', 'segments.jsonl'),
        )

    def test_segmentation_meta_path(self):
        self.assertEqual(
            P.segmentation_meta_path(self.tmp, 'ses_001'),
            os.path.join(self.tmp, '01_transcripts', 'segmented', 'ses_001', 'segmentation_meta.json'),
        )


class TestClassificationOverlayPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_classifications_dir(self):
        self.assertEqual(
            P.classifications_dir(self.tmp),
            os.path.join(self.tmp, '02_meta', 'classifications'),
        )

    def test_overlay_theme(self):
        self.assertEqual(
            P.classification_overlay_path(self.tmp, 'theme'),
            os.path.join(self.tmp, '02_meta', 'classifications', 'theme_labels.jsonl'),
        )

    def test_overlay_purer(self):
        self.assertEqual(
            P.classification_overlay_path(self.tmp, 'purer'),
            os.path.join(self.tmp, '02_meta', 'classifications', 'purer_labels.jsonl'),
        )

    def test_overlay_codebook(self):
        self.assertEqual(
            P.classification_overlay_path(self.tmp, 'codebook'),
            os.path.join(self.tmp, '02_meta', 'classifications', 'codebook_labels.jsonl'),
        )

    def test_overlay_cv(self):
        self.assertEqual(
            P.classification_overlay_path(self.tmp, 'cv'),
            os.path.join(self.tmp, '02_meta', 'classifications', 'cross_validation_labels.jsonl'),
        )

    def test_classification_manifest_path(self):
        self.assertEqual(
            P.classification_manifest_path(self.tmp),
            os.path.join(self.tmp, '02_meta', 'classifications', 'classification_manifest.json'),
        )

    def test_overlay_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            P.classification_overlay_path(self.tmp, 'nonexistent_key')


class TestFlatTestsetPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_testset_human_flat_path(self):
        self.assertEqual(
            P.testset_human_flat_path(self.tmp, 1),
            os.path.join(self.tmp, '04_validation', 'testsets',
                         'human_classification_testset_worksheet_1.txt'),
        )

    def test_testset_ai_flat_path(self):
        self.assertEqual(
            P.testset_ai_flat_path(self.tmp, 2),
            os.path.join(self.tmp, '04_validation', 'testsets',
                         'AI_classification_testset_worksheet_2.txt'),
        )

    def test_testset_meta_path(self):
        self.assertEqual(
            P.testset_meta_path(self.tmp, 3),
            os.path.join(self.tmp, '02_meta', 'testset_meta',
                         'human_classification_testset_worksheet_3.meta.json'),
        )

    def test_next_testset_number_no_dir(self):
        # testsets dir does not exist -> should return 1
        self.assertEqual(P.next_testset_number(self.tmp), 1)

    def test_next_testset_number_with_existing(self):
        ts_dir = P.testsets_dir(self.tmp)
        os.makedirs(ts_dir)
        # Create two worksheets: 1 and 3 — next should be 4
        for n in (1, 3):
            open(os.path.join(ts_dir,
                              f'human_classification_testset_worksheet_{n}.txt'), 'w').close()
        self.assertEqual(P.next_testset_number(self.tmp), 4)

    def test_count_existing_testsets_no_dir(self):
        self.assertEqual(P.count_existing_testsets(self.tmp), 0)

    def test_count_existing_testsets(self):
        ts_dir = P.testsets_dir(self.tmp)
        os.makedirs(ts_dir)
        for n in (1, 2, 5):
            open(os.path.join(ts_dir,
                              f'human_classification_testset_worksheet_{n}.txt'), 'w').close()
        # Non-matching file should not be counted
        open(os.path.join(ts_dir, 'other_file.txt'), 'w').close()
        self.assertEqual(P.count_existing_testsets(self.tmp), 3)


class TestContentValidityPaths(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_cv_testsets_dir_is_content_validity_dir(self):
        self.assertEqual(
            P.cv_testsets_dir(self.tmp),
            P.content_validity_dir(self.tmp),
        )

    def test_cv_testset_dir(self):
        self.assertEqual(
            P.cv_testset_dir(self.tmp, 'cv_purer_v1'),
            os.path.join(self.tmp, '04_validation', 'content_validity', 'cv_purer_v1'),
        )

    def test_cv_testset_manifest_path(self):
        self.assertEqual(
            P.cv_testset_manifest_path(self.tmp, 'cv_purer_v1'),
            os.path.join(self.tmp, '04_validation', 'content_validity', 'cv_purer_v1', 'manifest.json'),
        )

    def test_cv_testset_items_path(self):
        self.assertEqual(
            P.cv_testset_items_path(self.tmp, 'cv_purer_v1'),
            os.path.join(self.tmp, '04_validation', 'content_validity', 'cv_purer_v1', 'items.jsonl'),
        )

    def test_cv_testset_human_worksheet_path(self):
        self.assertEqual(
            P.cv_testset_human_worksheet_path(self.tmp, 'cv_purer_v1'),
            os.path.join(self.tmp, '04_validation', 'content_validity',
                         'cv_purer_v1', 'human_worksheet.txt'),
        )

    def test_cv_testset_definition_key_path(self):
        self.assertEqual(
            P.cv_testset_definition_key_path(self.tmp, 'cv_purer_v1'),
            os.path.join(self.tmp, '04_validation', 'content_validity',
                         'cv_purer_v1', 'definition_key.txt'),
        )

    def test_cv_testset_answer_key_path(self):
        self.assertEqual(
            P.cv_testset_answer_key_path(self.tmp, 'cv_purer_v1'),
            os.path.join(self.tmp, '04_validation', 'content_validity',
                         'cv_purer_v1', 'AI_answer_key.txt'),
        )


if __name__ == '__main__':
    unittest.main()
