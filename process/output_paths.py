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
