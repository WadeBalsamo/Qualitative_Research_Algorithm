"""
process/output_paths.py
-----------------------
Single source of truth for all output directory paths.

All functions accept `run_dir` (the top-level pipeline output directory) and
return a path string.
"""
import os


# ── Provenance & configuration ────────────────────────────────────────────

def meta_dir(run_dir: str) -> str:
    """Sensitive provenance: speaker key, theme defs, config, process log."""
    return os.path.join(run_dir, '02_meta')


def auditable_logs_dir(run_dir: str) -> str:
    """LLM classification logs, prompts, checkpoints, and theme definitions."""
    return os.path.join(run_dir, '02_meta', 'auditable_logs')


def llm_raw_dir(run_dir: str) -> str:
    """LLM theme-classification checkpoints and status files (alias for auditable_logs_dir)."""
    return auditable_logs_dir(run_dir)


def llm_classification_log_path(run_dir: str) -> str:
    """Fixed-name append-mode classification status log."""
    return os.path.join(auditable_logs_dir(run_dir), 'llm_classification_log.txt')


def llm_prompts_path(run_dir: str) -> str:
    """Every LLM prompt and response captured across the entire pipeline."""
    return os.path.join(auditable_logs_dir(run_dir), 'llm_prompts.txt')


def llm_checkpoints_dir(run_dir: str) -> str:
    """JSON checkpoint files from classification runs."""
    return os.path.join(auditable_logs_dir(run_dir), 'checkpoints')


def theme_definitions_txt_path(run_dir: str) -> str:
    """Human-readable theme framework reference document."""
    return os.path.join(run_dir, '02_meta', 'theme_definitions.txt')


def anonymization_key_txt_path(run_dir: str) -> str:
    """Human-readable speaker anonymization key table."""
    return os.path.join(run_dir, '02_meta', 'anonymization_key.txt')


def codebook_raw_dir(run_dir: str) -> str:
    """Codebook embedding-classification checkpoints and status files."""
    return os.path.join(run_dir, '02_meta', 'codebook_raw')


# ── Transcripts ───────────────────────────────────────────────────────────

def transcripts_diarized_dir(run_dir: str) -> str:
    """Raw diarized transcript files (provenance)."""
    return os.path.join(run_dir, '01_transcripts', 'diarized')


def transcripts_coded_dir(run_dir: str) -> str:
    """Per-session coded transcript .txt files."""
    return os.path.join(run_dir, '01_transcripts', 'coded')


# ── Machine-readable analysis data ───────────────────────────────────────

def analysis_data_dir(run_dir: str) -> str:
    """Root for machine-readable numeric artifacts (longitudinal_summary, etc.)."""
    return os.path.join(run_dir, '03_analysis_data')


def session_stats_dir(run_dir: str) -> str:
    """Per-session stats JSON files."""
    return os.path.join(run_dir, '03_analysis_data', 'session_stats')


def graphing_dir(run_dir: str) -> str:
    """Graph-ready CSVs for downstream visualization."""
    return os.path.join(run_dir, '03_analysis_data', 'graphing')


def longitudinal_dir(run_dir: str) -> str:
    """Longitudinal CSV outputs (session_stage_progression.csv, etc.)."""
    return os.path.join(run_dir, '03_analysis_data')


def cumulative_report_dir(run_dir: str) -> str:
    """Directory for cumulative_report.json."""
    return os.path.join(run_dir, '03_analysis_data')


def sessions_json_dir(run_dir: str) -> str:
    """Per-session JSON analysis reports."""
    return os.path.join(run_dir, '03_analysis_data', 'per_session')


def participants_json_dir(run_dir: str) -> str:
    """Per-participant JSON analysis reports."""
    return os.path.join(run_dir, '03_analysis_data', 'per_participant')


# ── Validation ────────────────────────────────────────────────────────────

def validation_dir(run_dir: str) -> str:
    """Researcher validation artifacts (forms, test sets, flagged items)."""
    return os.path.join(run_dir, '04_validation')


def testsets_dir(run_dir: str) -> str:
    """Validation test set worksheets."""
    return os.path.join(run_dir, '04_validation', 'testsets')


def cross_validation_dir(run_dir: str) -> str:
    """Theme ↔ codebook cross-validation result files."""
    return os.path.join(run_dir, '04_validation', 'cross_validation')


def content_validity_dir(run_dir: str) -> str:
    """Directory for content_validity_test_set.jsonl."""
    return os.path.join(run_dir, '04_validation')


def human_eval_dir(run_dir: str) -> str:
    """Directory for human_coding_evaluation_set.csv."""
    return os.path.join(run_dir, '04_validation')


# ── Figures ───────────────────────────────────────────────────────────────

def figures_dir(run_dir: str) -> str:
    """All PNG figures."""
    return os.path.join(run_dir, '05_figures')


# ── Human-readable reports ────────────────────────────────────────────────

def human_reports_dir(run_dir: str) -> str:
    """Root for all human-readable .txt reports."""
    return os.path.join(run_dir, '06_reports')


def themes_dir(run_dir: str) -> str:
    """Per-theme/stage text and JSON reports."""
    return os.path.join(run_dir, '06_reports', 'per_theme')


def themes_json_dir(run_dir: str) -> str:
    """Per-theme/stage JSON analysis files."""
    return os.path.join(run_dir, '03_analysis_data', 'per_theme')


# ── Training data ─────────────────────────────────────────────────────────

def master_segments_dir(run_dir: str) -> str:
    """Directory for master_segments.csv / master_segments.jsonl."""
    return os.path.join(run_dir, '02_meta', 'training_data')


def training_data_dir(run_dir: str) -> str:
    """BERT-ready JSONL training files and label_map.json."""
    return os.path.join(run_dir, '02_meta', 'training_data')


def session_summaries_json_path(run_dir: str) -> str:
    """Machine-readable session summaries JSON (analysis data, not reports)."""
    return os.path.join(run_dir, '03_analysis_data', 'session_summaries.json')


# ── Frozen segmentation ───────────────────────────────────────────────────

def segmented_sessions_dir(run_dir: str) -> str:
    """Root for per-session frozen segmentation artifacts."""
    return os.path.join(run_dir, '01_transcripts', 'segmented')


def segmented_session_dir(run_dir: str, session_id: str) -> str:
    """Directory for a single session's frozen segments."""
    return os.path.join(segmented_sessions_dir(run_dir), session_id)


def session_segments_path(run_dir: str, session_id: str) -> str:
    """Frozen segments JSONL for one session."""
    return os.path.join(segmented_session_dir(run_dir, session_id), 'segments.jsonl')


def segmentation_meta_path(run_dir: str, session_id: str) -> str:
    """Segmentation params hash and ingest timestamp for one session."""
    return os.path.join(segmented_session_dir(run_dir, session_id), 'segmentation_meta.json')


# ── Frozen testsets ───────────────────────────────────────────────────────

def testset_dir(run_dir: str, name: str) -> str:
    """Directory for a single named frozen testset."""
    return os.path.join(testsets_dir(run_dir), name)


def testset_manifest_path(run_dir: str, name: str) -> str:
    """Frozen manifest (segment IDs + content SHAs) for a testset."""
    return os.path.join(testset_dir(run_dir, name), 'manifest.json')


def testset_snapshot_path(run_dir: str, name: str) -> str:
    """Frozen segment text snapshot for a testset."""
    return os.path.join(testset_dir(run_dir, name), 'segments_snapshot.jsonl')


def testset_human_worksheet_path(run_dir: str, name: str) -> str:
    """Frozen blind-coding worksheet for a testset."""
    return os.path.join(testset_dir(run_dir, name), 'human_worksheet.txt')


def testset_answer_key_path(run_dir: str, name: str) -> str:
    """Refreshable AI answer key for a testset."""
    return os.path.join(testset_dir(run_dir, name), 'AI_answer_key.txt')


# ── Frozen content-validity testsets ─────────────────────────────────────

def cv_testsets_dir(run_dir: str) -> str:
    """Root for content-validity testset directories."""
    return os.path.join(run_dir, '04_validation', 'content_validity')


def cv_testset_dir(run_dir: str, name: str) -> str:
    """Directory for a single named content-validity testset."""
    return os.path.join(cv_testsets_dir(run_dir), name)


def cv_testset_manifest_path(run_dir: str, name: str) -> str:
    """Frozen manifest for a content-validity testset."""
    return os.path.join(cv_testset_dir(run_dir, name), 'manifest.json')


def cv_testset_items_path(run_dir: str, name: str) -> str:
    """Frozen items JSONL for a content-validity testset."""
    return os.path.join(cv_testset_dir(run_dir, name), 'items.jsonl')


def cv_testset_human_worksheet_path(run_dir: str, name: str) -> str:
    """Frozen blind-coding worksheet for a content-validity testset."""
    return os.path.join(cv_testset_dir(run_dir, name), 'human_worksheet.txt')


def cv_testset_definition_key_path(run_dir: str, name: str) -> str:
    """Frozen framework definition reference for a content-validity testset."""
    return os.path.join(cv_testset_dir(run_dir, name), 'definition_key.txt')


def cv_testset_answer_key_path(run_dir: str, name: str) -> str:
    """Refreshable AI graded report for a content-validity testset."""
    return os.path.join(cv_testset_dir(run_dir, name), 'AI_answer_key.txt')
