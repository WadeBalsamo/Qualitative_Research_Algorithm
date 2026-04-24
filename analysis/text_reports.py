"""Backward-compat shim — public API now lives in analysis/reports/."""
from .reports import (
    generate_comprehensive_session_report,
    generate_stage_text_report,
    generate_all_stage_text_reports,
    generate_transition_explanation,
    generate_longitudinal_text_report,
    generate_therapist_cues_report,
)
__all__ = [
    'generate_comprehensive_session_report',
    'generate_stage_text_report',
    'generate_all_stage_text_reports',
    'generate_transition_explanation',
    'generate_longitudinal_text_report',
    'generate_therapist_cues_report',
]
