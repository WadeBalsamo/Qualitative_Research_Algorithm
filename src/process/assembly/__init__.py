"""process/assembly — dataset assembly and export functions."""

from .master_dataset import assemble_master_dataset
from .coded_transcripts import export_coded_transcript
from .stats_reports import export_per_transcript_stats, export_cumulative_report
from .training_export import export_theme_definitions, export_theme_definitions_txt, export_content_validity_test_set, export_training_data
from .human_forms import (
    export_human_classification_forms,
    export_flagged_for_review,
    generate_or_refresh_validation_testsets,
    create_frozen_testset,
    refresh_testset_answer_key,
    export_validation_testsets,  # Phase 1 back-compat — remove with legacy_migration.py
    export_content_validity_human_worksheet,
    export_content_validity_definition_key,
    export_content_validity_answer_key,
)
from .content_validity import (
    create_frozen_content_validity_testset,
    refresh_cv_answer_key,
    list_content_validity_testsets,
    generate_or_refresh_content_validity_testsets,
)
from .mindfulbert_dataset import build_mindfulbert_dataset

__all__ = [
    'assemble_master_dataset',
    'export_theme_definitions',
    'export_theme_definitions_txt',
    'export_content_validity_test_set',
    'export_content_validity_human_worksheet',
    'export_content_validity_definition_key',
    'export_content_validity_answer_key',
    'export_coded_transcript',
    'export_per_transcript_stats',
    'export_cumulative_report',
    'export_human_classification_forms',
    'export_flagged_for_review',
    'export_training_data',
    'generate_or_refresh_validation_testsets',
    'create_frozen_testset',
    'refresh_testset_answer_key',
    'export_validation_testsets',  # Phase 1 back-compat — remove with legacy_migration.py
    # Phase 2 content-validity
    'create_frozen_content_validity_testset',
    'refresh_cv_answer_key',
    'list_content_validity_testsets',
    'generate_or_refresh_content_validity_testsets',
    # Track C — MindfulBERT training-set builder
    'build_mindfulbert_dataset',
]
