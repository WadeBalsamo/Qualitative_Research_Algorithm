"""
process/output_paths.py
-----------------------
Single source of truth for all output directory paths.

All functions accept `run_dir` (the top-level pipeline output directory) and
return a path string.  Phase 1: return values match the existing layout —
changing them here (Phase 2) moves every output without touching individual
writers.
"""
import os


# ── Provenance & configuration ────────────────────────────────────────────

def meta_dir(run_dir: str) -> str:
    """Sensitive provenance: speaker key, theme defs, config, process log."""
    return os.path.join(run_dir, '07_meta')


# ── Transcripts ───────────────────────────────────────────────────────────

def transcripts_coded_dir(run_dir: str) -> str:
    """Per-session coded transcript .txt files."""
    return os.path.join(run_dir, '01_transcripts', 'coded')


# ── Human-readable reports ────────────────────────────────────────────────

def human_reports_dir(run_dir: str) -> str:
    """Root for all human-readable .txt reports."""
    return os.path.join(run_dir, '02_human_reports')


def sessions_json_dir(run_dir: str) -> str:
    """Per-session JSON analysis reports."""
    return os.path.join(run_dir, '02_human_reports', 'per_session')


def participants_json_dir(run_dir: str) -> str:
    """Per-participant JSON analysis reports."""
    return os.path.join(run_dir, '02_human_reports', 'per_participant')


def constructs_dir(run_dir: str) -> str:
    """Per-construct/stage text and JSON reports."""
    return os.path.join(run_dir, '02_human_reports', 'per_construct')


def constructs_json_dir(run_dir: str) -> str:
    """Per-construct/stage JSON files (same folder as constructs_dir)."""
    return constructs_dir(run_dir)


# ── Figures ───────────────────────────────────────────────────────────────

def figures_dir(run_dir: str) -> str:
    """All PNG figures."""
    return os.path.join(run_dir, '03_figures')


# ── Machine-readable analysis data ───────────────────────────────────────

def analysis_data_dir(run_dir: str) -> str:
    """Root for machine-readable numeric artifacts (longitudinal_summary, etc.)."""
    return os.path.join(run_dir, '04_analysis_data')


def session_stats_dir(run_dir: str) -> str:
    """Per-session stats JSON files."""
    return os.path.join(run_dir, '04_analysis_data', 'session_stats')


def graphing_dir(run_dir: str) -> str:
    """Graph-ready CSVs for downstream visualization."""
    return os.path.join(run_dir, '04_analysis_data', 'graphing')


def longitudinal_dir(run_dir: str) -> str:
    """Longitudinal CSV outputs (session_stage_progression.csv, etc.)."""
    return os.path.join(run_dir, '04_analysis_data')


def cumulative_report_dir(run_dir: str) -> str:
    """Directory for cumulative_report.json."""
    return os.path.join(run_dir, '04_analysis_data')


# ── Validation ────────────────────────────────────────────────────────────

def validation_dir(run_dir: str) -> str:
    """Researcher validation artifacts (forms, test sets, flagged items)."""
    return os.path.join(run_dir, '05_validation')


def testsets_dir(run_dir: str) -> str:
    """Validation test set worksheets."""
    return os.path.join(run_dir, '05_validation', 'testsets')


def cross_validation_dir(run_dir: str) -> str:
    """Theme ↔ codebook cross-validation result files."""
    return os.path.join(run_dir, '05_validation', 'cross_validation')


def content_validity_dir(run_dir: str) -> str:
    """Directory for content_validity_test_set.jsonl."""
    return os.path.join(run_dir, '05_validation')


def human_eval_dir(run_dir: str) -> str:
    """Directory for human_coding_evaluation_set.csv."""
    return os.path.join(run_dir, '05_validation')


# ── Training data ─────────────────────────────────────────────────────────

def master_segments_dir(run_dir: str) -> str:
    """Directory for master_segments.csv / master_segments.jsonl."""
    return os.path.join(run_dir, '06_training_data')


def training_data_dir(run_dir: str) -> str:
    """BERT-ready JSONL training files and label_map.json."""
    return os.path.join(run_dir, '06_training_data')
