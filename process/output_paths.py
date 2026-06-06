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


def gnn_model_dir(run_dir: str) -> str:
    """Trained GNN-layer checkpoint bundle (weights + manifest) and cached embeddings."""
    return os.path.join(run_dir, '02_meta', 'gnn')


def gnn_data_dir(run_dir: str) -> str:
    """Machine-readable GNN-layer analysis artifacts (segment positions, motifs, lift, coupling)."""
    return os.path.join(run_dir, '03_analysis_data', 'gnn')


def mechanism_dir(run_dir: str) -> str:
    """Machine-readable mechanistic-analysis artifacts (Δprogression, liminality, trajectory types)."""
    return os.path.join(run_dir, '03_analysis_data', 'mechanism')


def efficacy_dir(run_dir: str) -> str:
    """Machine-readable program-efficacy artifacts (trajectories, slopes, barrier crossing, linkage)."""
    return os.path.join(run_dir, '03_analysis_data', 'efficacy')


# ── Transcripts ───────────────────────────────────────────────────────────

def transcripts_diarized_dir(run_dir: str) -> str:
    """Raw diarized transcript files (provenance input copies)."""
    return os.path.join(run_dir, '01_transcripts_inputs')


def full_transcripts_dir(run_dir: str) -> str:
    """Full-session transcript artifacts: coded transcripts and human classification forms."""
    return os.path.join(run_dir, '04_validation', 'full_transcripts')


def transcripts_coded_dir(run_dir: str) -> str:
    """Per-session coded transcript .txt files (alias for full_transcripts_dir)."""
    return full_transcripts_dir(run_dir)


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
    """Directory for content validity files and frozen CV testset directories."""
    return os.path.join(run_dir, '04_validation', 'content_validity')


def human_eval_dir(run_dir: str) -> str:
    """Directory for human_coding_evaluation_set.csv."""
    return os.path.join(run_dir, '04_validation')


# ── Figures ───────────────────────────────────────────────────────────────

def figures_dir(run_dir: str) -> str:
    """All PNG figures."""
    return os.path.join(run_dir, '05_figures')


# ── Human-readable reports (tiered, numbered) ─────────────────────────────
#
# 06_reports/ is organized into a numbered tier so reading order is obvious:
#   00_READ_ME.txt / 00_executive_summary.txt  — start here
#   01_outcomes/    — did the program work (efficacy, longitudinal, barrier)
#   02_mechanism/   — how it works (transitions, cue→response, PURER, atlas…)
#   03_per_session/ — per-session drill-down
#   04_per_participant/
#   05_per_stage/   — per VAAMR stage + codebook exemplars
#   06_gnn/         — GNN discovery + validation
#   07_methods_appendix.txt — how each metric is computed + caveats

def human_reports_dir(run_dir: str) -> str:
    """Root for all human-readable .txt reports."""
    return os.path.join(run_dir, '06_reports')


def reports_outcomes_dir(run_dir: str) -> str:
    """Program-outcome reports: efficacy, longitudinal, avoidance barrier."""
    return os.path.join(run_dir, '06_reports', '01_outcomes')


def reports_mechanism_dir(run_dir: str) -> str:
    """Mechanism reports: transitions, cue→response, PURER, mechanism, atlas, superposition."""
    return os.path.join(run_dir, '06_reports', '02_mechanism')


def reports_per_session_dir(run_dir: str) -> str:
    """Per-session drill-down reports and the session-distribution overview."""
    return os.path.join(run_dir, '06_reports', '03_per_session')


def reports_per_participant_dir(run_dir: str) -> str:
    """Per-participant longitudinal drill-down reports."""
    return os.path.join(run_dir, '06_reports', '04_per_participant')


def reports_gnn_dir(run_dir: str) -> str:
    """GNN discovery & validation reports (validation, triangulation, motifs, coupling)."""
    return os.path.join(run_dir, '06_reports', '06_gnn')


def executive_summary_path(run_dir: str) -> str:
    """Deterministic program-improvement brief (top-level synthesis)."""
    return os.path.join(human_reports_dir(run_dir), '00_executive_summary.txt')


def reports_readme_path(run_dir: str) -> str:
    """Guide to every report in 06_reports/ with recommended reading order."""
    return os.path.join(human_reports_dir(run_dir), '00_READ_ME.txt')


def methods_appendix_path(run_dir: str) -> str:
    """How each report/metric is computed, with methodological caveats."""
    return os.path.join(human_reports_dir(run_dir), '07_methods_appendix.txt')


def themes_dir(run_dir: str) -> str:
    """Per-theme/stage text reports (and codebook exemplars)."""
    return os.path.join(run_dir, '06_reports', '05_per_stage')


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


# ── Classification overlays (Phase 3) ────────────────────────────────────

def classifications_dir(run_dir: str) -> str:
    """Per-classifier overlay JSONL files and the provenance manifest."""
    return os.path.join(run_dir, '02_meta', 'classifications')


def classification_overlay_path(run_dir: str, key: str) -> str:
    """Overlay JSONL for a specific classifier: theme | purer | codebook | cv."""
    _filenames = {
        'theme': 'theme_labels.jsonl',
        'purer': 'purer_labels.jsonl',
        'codebook': 'codebook_labels.jsonl',
        'cv': 'cross_validation_labels.jsonl',
        'gnn': 'gnn_labels.jsonl',
    }
    return os.path.join(classifications_dir(run_dir), _filenames[key])


def classification_manifest_path(run_dir: str) -> str:
    """Provenance manifest recording model/version for each overlay."""
    return os.path.join(classifications_dir(run_dir), 'classification_manifest.json')


# ── Flat numbered testsets ─────────────────────────────────────────────────

def testset_human_flat_path(run_dir: str, n: int) -> str:
    """Human coding worksheet for flat testset #n."""
    return os.path.join(testsets_dir(run_dir), f'human_classification_testset_worksheet_{n}.txt')


def testset_ai_flat_path(run_dir: str, n: int) -> str:
    """AI answer key for flat testset #n."""
    return os.path.join(testsets_dir(run_dir), f'AI_classification_testset_worksheet_{n}.txt')


def next_testset_number(run_dir: str) -> int:
    """Return the next unused worksheet number (1-based) by scanning testsets_dir."""
    import re
    ts_dir = testsets_dir(run_dir)
    if not os.path.isdir(ts_dir):
        return 1
    pattern = re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    used = [int(m.group(1)) for f in os.listdir(ts_dir) if (m := pattern.match(f))]
    return max(used) + 1 if used else 1


def testset_meta_path(run_dir: str, n: int) -> str:
    """Sidecar metadata for flat testset #n (per-segment SHA256 + legacy flag).

    Kept under 02_meta so it does not clutter the 04_validation worksheet output.
    """
    return os.path.join(meta_dir(run_dir), 'testset_meta',
                        f'human_classification_testset_worksheet_{n}.meta.json')


def count_existing_testsets(run_dir: str) -> int:
    """Count how many flat human worksheets exist in testsets_dir."""
    import re
    ts_dir = testsets_dir(run_dir)
    if not os.path.isdir(ts_dir):
        return 0
    pattern = re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    return sum(1 for f in os.listdir(ts_dir) if pattern.match(f))


# ── Frozen content-validity testsets ─────────────────────────────────────

def cv_testsets_dir(run_dir: str) -> str:
    """Root for frozen content-validity testset directories (alias for content_validity_dir)."""
    return content_validity_dir(run_dir)


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
