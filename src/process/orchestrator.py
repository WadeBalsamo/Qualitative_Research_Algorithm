"""
orchestrator.py
---------------
Top-level 8-stage pipeline orchestrator.

Coordinates the complete QRA classification pipeline across eight stages:

Stage 1: Transcript Ingestion & Segmentation - Loads and segments diarized transcripts
Stage 2: Construct Operationalization - Builds framework definitions and content-validity test sets
Stage 3: Theme Classification (VAAMR) - Zero-shot LLM classification of participant segments
Stage 3b: Codebook Classification - Multi-label codebook classification using embedding + LLM ensemble
Stage 3c: PURER Classification - Therapist cue-unit classification at the dialogue turn level
Stage 4: Cross-Validation - Computes theme-code co-occurrence statistics
Stage 5: Human Validation Set Export - Creates frozen human coding evaluation sets
Stage 6: Dataset Assembly - Joins frozen segments with all overlays into master_segments.csv
Stage 7: Report Generation - Produces coded transcripts, stats reports, and visualization data
Stage 8: Analysis (optional) - Generates longitudinal summaries and figures

Exposes modular stage_* functions for Phase 3 standalone operation. Interacts with every process/ submodule.
"""

import copy
import json
import os
import datetime
from collections import Counter
from dataclasses import asdict
from typing import List, Optional

import pandas as pd

from classification_tools.data_structures import Segment
from classification_tools.llm_client import LLMClient, LLMClientConfig
from classification_tools.validation import create_balanced_evaluation_set
from classification_tools.llm_classifier import (
    classify_segments_zero_shot,
    create_content_validity_test_set,
    LLMCodebookClassifier,
)
from classification_tools.response_parser import parse_all_results, parse_purer_results
from classification_tools.llm_classifier import (
    classify_purer_cue_units as _classify_purer_cue_units,
    _build_context_block as _build_context_block_for_purer,
)
from theme_framework.theme_schema import ThemeFramework

from .config import PipelineConfig
from .transcript_ingestion import (
    ConversationalSegmenter,
    load_diarized_session,
    load_vtt_session,
    discover_session_files,
    parse_session_id_metadata,
)
from .llm_segmentation import LLMSegmentationRefiner
from .process_logger import ProcessLogger
from .assembly import (
    assemble_master_dataset,
    export_theme_definitions,
    export_theme_definitions_txt,
    export_content_validity_test_set,
    export_content_validity_human_worksheet,
    export_content_validity_definition_key,
    export_content_validity_answer_key,
    export_coded_transcript,
    export_per_transcript_stats,
    export_cumulative_report,
    export_human_classification_forms,
    export_flagged_for_review,
    export_training_data,
    generate_or_refresh_validation_testsets,
    export_validation_testsets,  # Phase 1 back-compat — remove with legacy_migration.py
    generate_or_refresh_content_validity_testsets,
)
from .cross_validation import (
    compute_theme_codebook_cooccurrence,
    summarize_theme_code_associations,
    export_cross_validation_results,
)
from . import output_paths as _paths
from . import legacy_migration  # LEGACY-MIGRATION CALL SITE — Phase 3 cleanup target
from . import segments_io
from .speaker_filter import apply_speaker_filter as _apply_speaker_filter
from .speaker_anonymization import load_speaker_map as _load_speaker_map
from .text_anonymization import scrub_segments as _scrub_segments


# Backward-compat alias; prefer segments_io.resolve_session_id in new code.
_resolve_session_id = segments_io.resolve_session_id


class PipelineObserver:
    """Base observer with no-op methods for all pipeline events.

    The optional ``scope_id`` keyword argument on each method lets partial
    (stage-only) runs scope their progress without referencing a non-existent
    prior stage.  Existing observers ignore it for back-compat.
    """

    def on_stage_start(self, stage_name: str, stage_number: str,
                       *, scope_id: Optional[str] = None, **kwargs):
        pass

    def on_stage_progress(self, stage_name: str, message: str,
                          *, scope_id: Optional[str] = None, **kwargs):
        pass

    def on_stage_complete(self, stage_name: str, summary: str,
                          *, scope_id: Optional[str] = None, **kwargs):
        pass

    def on_pipeline_complete(self, output_dir: str, **kwargs):
        pass


class SilentObserver(PipelineObserver):
    """Minimal output: stage headers and summaries."""

    def on_stage_start(self, stage_name: str, stage_number: str,
                       *, scope_id: Optional[str] = None, **kwargs):
        print(f"\n{'=' * 60}")
        print(f"STAGE {stage_number}: {stage_name}")
        print(f"{'=' * 60}")

    def on_stage_progress(self, stage_name: str, message: str,
                          *, scope_id: Optional[str] = None, **kwargs):
        print(f"  {message}")

    def on_stage_complete(self, stage_name: str, summary: str,
                          *, scope_id: Optional[str] = None, **kwargs):
        print(f"  {summary}")

    def on_pipeline_complete(self, output_dir: str, **kwargs):
        print(f"\n{'=' * 60}")
        print("PIPELINE COMPLETE")
        print(f"{'=' * 60}")
        print(f"All outputs in: {output_dir}")

from codebook.embedding_classifier import EmbeddingCodebookClassifier, ensure_embedding_model_ready
from codebook.ensemble import CodebookEnsemble
from . import classifications_io as _cio


# ===========================================================================
# Phase 3 — Modular stage functions
#
# Each function follows this contract:
#   - If segments=None, loads from frozen disk state + overlays (standalone mode).
#   - If config/framework is provided, runs the actual classifier.
#   - Always writes its overlay from the current in-memory segment state.
#   - Returns the (possibly mutated) segments list, or pd.DataFrame for assemble.
#
# output_dir is resolved from the explicit kwarg first, then config.output_dir.
# ===========================================================================

def _resolve_output_dir(output_dir, config):
    if output_dir is not None:
        return output_dir
    if config is not None and hasattr(config, 'output_dir'):
        return config.output_dir
    raise ValueError("output_dir required: pass --output-dir or provide a config with output_dir")


def _build_classifier_manifest_entry(config, classifier_key: str, framework=None, codebook=None, *, n_segments: int) -> dict:
    """Build a manifest entry recording all reproducibility-relevant classifier settings.

    Always records n_segments. Records model / n_runs / temperature from the
    appropriate sub-config when ``config`` is provided. Records framework
    name+version for theme/cv keys when ``framework`` is provided. Records
    codebook name+version for codebook/cv keys when ``codebook`` is provided.
    """
    entry: dict = {'n_segments': n_segments}

    if config is not None:
        sub_attr = {
            'theme': 'theme_classification',
            'purer': 'purer_classification',
            'codebook': 'codebook_llm',
        }.get(classifier_key)
        if sub_attr is not None:
            sub = getattr(config, sub_attr, None)
            if sub is not None:
                model = getattr(sub, 'model', None)
                if classifier_key == 'purer' and not model:
                    model = getattr(config.theme_classification, 'model', None)
                if model:
                    entry['model'] = model
                n_runs = getattr(sub, 'n_runs', None)
                if n_runs is not None:
                    entry['n_runs'] = n_runs
                temperature = getattr(sub, 'temperature', None)
                if temperature is not None:
                    entry['temperature'] = temperature

    if framework is not None and classifier_key in ('theme', 'cv'):
        entry['framework'] = {
            'name': framework.name,
            'version': getattr(framework, 'version', '?'),
        }
    if codebook is not None and classifier_key in ('codebook', 'cv'):
        entry['codebook'] = {
            'name': codebook.name,
            'version': getattr(codebook, 'version', '?'),
        }

    return entry


def stage_classify_theme(
    config,
    framework,
    *,
    segments: Optional[List[Segment]] = None,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
    only_session_ids: Optional[set] = None,
) -> List[Segment]:
    """
    Theme classification stage.

    If config and framework are provided, runs zero-shot LLM theme classification
    against participant segments, then writes/updates theme_labels.jsonl.
    If segments=None, loads from 01_transcripts/segmented/ + non-theme overlays.

    only_session_ids:  When set, classify only segments whose session_id is in this set
                       and merge results into the existing overlay (instead of full rewrite).
    """
    _od = _resolve_output_dir(output_dir, config)
    os.makedirs(_od, exist_ok=True)

    if segments is None:
        segments = segments_io.load_segments_for_stage(
            _od, apply=('purer', 'codebook', 'cv'),
        )

    if only_session_ids is not None:
        subset = [s for s in segments if s.session_id in only_session_ids]
    else:
        subset = segments

    if config is not None and framework is not None:
        _theme_llm_classify(config, framework, subset, _od, observer)

    if only_session_ids is not None:
        _cio.merge_theme_overlay(_od, subset)
    else:
        _cio.write_theme_overlay(_od, segments)
    entry = _build_classifier_manifest_entry(
        config, 'theme', framework=framework, n_segments=len(segments),
    )
    if only_session_ids is not None:
        entry['last_incremental_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry['n_new_segments'] = len(subset)
    _cio.update_classification_manifest(_od, key='theme', entry=entry)

    return segments


def stage_classify_purer(
    config,
    *,
    segments: Optional[List[Segment]] = None,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
    only_session_ids: Optional[set] = None,
) -> List[Segment]:
    """
    PURER cue-unit classification stage.

    Writes purer_labels.jsonl from current in-memory PURER fields.
    Runs LLM classification when config is provided.

    only_session_ids:  When set, classify only segments whose session_id is in this set
                       and merge results into the existing overlay (instead of full rewrite).
    """
    _od = _resolve_output_dir(output_dir, config)
    os.makedirs(_od, exist_ok=True)

    if segments is None:
        segments = segments_io.load_segments_for_stage(
            _od, apply=('theme', 'codebook', 'cv'),
        )

    if only_session_ids is not None:
        subset = [s for s in segments if s.session_id in only_session_ids]
    else:
        subset = segments

    if config is not None:
        _purer_llm_classify(config, subset, _od, observer)

    if only_session_ids is not None:
        _cio.merge_purer_overlay(_od, subset)
    else:
        _cio.write_purer_overlay(_od, segments)
    entry = _build_classifier_manifest_entry(
        config, 'purer', n_segments=len(segments),
    )
    if only_session_ids is not None:
        entry['last_incremental_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry['n_new_segments'] = len(subset)
    _cio.update_classification_manifest(_od, key='purer', entry=entry)

    return segments


def stage_classify_codebook(
    config,
    codebook,
    *,
    segments: Optional[List[Segment]] = None,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
    only_session_ids: Optional[set] = None,
) -> List[Segment]:
    """
    Codebook classification stage.

    Writes codebook_labels.jsonl from current in-memory codebook fields.
    Runs embedding + LLM codebook classification when config and codebook are provided.

    only_session_ids:  When set, classify only segments whose session_id is in this set
                       and merge results into the existing overlay (instead of full rewrite).
    """
    _od = _resolve_output_dir(output_dir, config)
    os.makedirs(_od, exist_ok=True)

    if segments is None:
        segments = segments_io.load_segments_for_stage(
            _od, apply=('theme', 'purer', 'cv'),
        )

    if only_session_ids is not None:
        subset = [s for s in segments if s.session_id in only_session_ids]
    else:
        subset = segments

    if config is not None and codebook is not None:
        _codebook_classify(config, codebook, subset, _od, observer)

    if only_session_ids is not None:
        _cio.merge_codebook_overlay(_od, subset)
    else:
        _cio.write_codebook_overlay(_od, segments)
    entry = _build_classifier_manifest_entry(
        config, 'codebook', codebook=codebook, n_segments=len(segments),
    )
    if only_session_ids is not None:
        entry['last_incremental_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry['n_new_segments'] = len(subset)
    _cio.update_classification_manifest(_od, key='codebook', entry=entry)

    return segments


def stage_cross_validation(
    config,
    framework,
    *,
    segments: Optional[List[Segment]] = None,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
    only_session_ids: Optional[set] = None,
) -> List[Segment]:
    """
    Cross-validation stage (theme × codebook co-occurrence).

    Writes cross_validation_labels.jsonl.  When config and framework are both
    provided, also computes and exports co-occurrence statistics.

    only_session_ids:  When set, classify only segments whose session_id is in this set
                       and merge results into the existing overlay (instead of full rewrite).
    """
    _od = _resolve_output_dir(output_dir, config)
    os.makedirs(_od, exist_ok=True)

    if segments is None:
        segments = segments_io.load_segments_for_stage(
            _od, apply=('theme', 'purer', 'codebook'),
        )

    if only_session_ids is not None:
        subset = [s for s in segments if s.session_id in only_session_ids]
    else:
        subset = segments

    if config is not None and framework is not None and \
            getattr(config, 'run_theme_labeler', False) and \
            getattr(config, 'run_codebook_classifier', False):
        _run_cv_stats(segments, framework, _od, observer)

    if only_session_ids is not None:
        _cio.merge_cross_validation_overlay(_od, subset)
    else:
        _cio.write_cross_validation_overlay(_od, segments)
    entry = _build_classifier_manifest_entry(
        config, 'cv', framework=framework, codebook=None, n_segments=len(segments),
    )
    if only_session_ids is not None:
        entry['last_incremental_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        entry['n_new_segments'] = len(subset)
    _cio.update_classification_manifest(
        _od, key='cv', entry=entry,
    )

    return segments


def stage_assemble(
    config,
    *,
    segments: Optional[List[Segment]] = None,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
):
    """
    Dataset assembly stage.

    Joins frozen segments + all present overlays and writes
    master_segments.csv.  Returns the master pd.DataFrame.
    When segments=None, loads from disk.
    """
    import pandas as pd
    from dataclasses import asdict

    _od = _resolve_output_dir(output_dir, config)
    os.makedirs(_od, exist_ok=True)

    if segments is None:
        segments = segments_io.load_segments_for_stage(
            _od, apply=('theme', 'purer', 'codebook', 'cv', 'gnn'),
        )

    confidence_tiers: dict = {}
    if config is not None and hasattr(config, 'confidence_tiers'):
        confidence_tiers = asdict(config.confidence_tiers)

    _ms_dir = _paths.master_segments_dir(_od)
    os.makedirs(_ms_dir, exist_ok=True)
    _gnn_auth, _gate_ok = _gnn_promotion_flags(config, _od)
    master_df = assemble_master_dataset(
        segments,
        os.path.join(_ms_dir, 'master_segments.csv'),
        confidence_tiers=confidence_tiers,
        gnn_authoritative=_gnn_auth,
        gate_passed=_gate_ok,
    )

    return master_df


# ---------------------------------------------------------------------------
# Private helpers for stage classification logic
# (extracted from run_full_pipeline body to avoid duplication)
# ---------------------------------------------------------------------------

def _gnn_promotion_flags(config, output_dir: str):
    """Resolve (gnn_authoritative, gate_passed) for label-of-record promotion.

    The graph can become the authoritative label of record only when the operator
    opted in (config.gnn_layer.gnn_authoritative) AND the persisted reliability-gate
    verdict says ready_for_scaling. A config flag alone never promotes an un-gated
    graph (Track 0.2). If the operator opted in but the gate is missing/failing, we
    force no-promotion and log a clear warning.
    """
    auth = bool(getattr(getattr(config, 'gnn_layer', None), 'gnn_authoritative', False))
    if not auth:
        return False, False
    try:
        from gnn_layer import validation as _gval
        gate = _gval.gate_ready_for_scaling(output_dir)
    except Exception:
        gate = False
    if not gate:
        print(
            "  Warning: gnn_authoritative=True but the GNN reliability gate has not "
            "passed (missing or failing verdict at 03_analysis_data/gnn/gnn_gate.json). "
            "Keeping LLM-consensus labels of record; the graph will not become "
            "authoritative until the gate reports ready_for_scaling."
        )
    return auth, gate

def resolve_pinned_classifier_config(
    run_dir: str,
    key: str,
    current_config: 'PipelineConfig',
    *,
    framework=None,
    codebook=None,
) -> 'PipelineConfig':
    """Return a PipelineConfig with classifier fields pinned to the manifest entry for ``key``.

    Pins model, n_runs, and temperature on the appropriate sub-config block.
    Also emits warnings if the supplied framework / codebook version differs
    from the version recorded in the manifest. If no manifest entry exists
    for ``key``, returns ``current_config`` unchanged.
    """
    manifest = _cio.read_classification_manifest(run_dir)
    if not manifest or key not in manifest:
        return current_config

    entry = manifest[key]
    sub_attr = {
        'theme': 'theme_classification',
        'purer': 'purer_classification',
        'codebook': 'codebook_llm',
    }.get(key)

    pinned = copy.deepcopy(current_config) if sub_attr is not None else current_config

    if sub_attr is not None:
        sub = getattr(pinned, sub_attr, None)
        if sub is not None:
            pinned_model = entry.get('model')
            pinned_n_runs = entry.get('n_runs')
            pinned_temp = entry.get('temperature')

            current_model = getattr(sub, 'model', None)
            if pinned_model and current_model != pinned_model:
                print(
                    f"  [incremental] pinning {key} classifier model: "
                    f"{pinned_model!r} (current config has {current_model!r})"
                )
                sub.model = pinned_model

            if pinned_n_runs is not None and getattr(sub, 'n_runs', None) != pinned_n_runs:
                print(f"  [incremental] pinning {key} n_runs: {pinned_n_runs}")
                sub.n_runs = pinned_n_runs

            if pinned_temp is not None and getattr(sub, 'temperature', None) != pinned_temp:
                print(f"  [incremental] pinning {key} temperature: {pinned_temp}")
                sub.temperature = pinned_temp

    # Framework version drift warning (theme + cv)
    if framework is not None and key in ('theme', 'cv'):
        fw_entry = entry.get('framework') or {}
        pinned_fw_version = fw_entry.get('version')
        current_fw_version = getattr(framework, 'version', None)
        if pinned_fw_version and current_fw_version and pinned_fw_version != current_fw_version:
            print(
                f"  [incremental] WARNING: {key} framework version drift: "
                f"manifest has version={pinned_fw_version!r}, "
                f"current framework has version={current_fw_version!r}. "
                f"Continuing with current framework."
            )

    # Codebook version drift warning (codebook + cv)
    if codebook is not None and key in ('codebook', 'cv'):
        cb_entry = entry.get('codebook') or {}
        pinned_cb_version = cb_entry.get('version')
        current_cb_version = getattr(codebook, 'version', None)
        if pinned_cb_version and current_cb_version and pinned_cb_version != current_cb_version:
            print(
                f"  [incremental] WARNING: {key} codebook version drift: "
                f"manifest has version={pinned_cb_version!r}, "
                f"current codebook has version={current_cb_version!r}. "
                f"Continuing with current codebook."
            )

    return pinned


def _theme_llm_classify(config, framework, segments, output_dir, observer):
    """Run zero-shot LLM theme classification in-place on segments."""
    theme_config = config.theme_classification
    theme_config.output_dir = _paths.auditable_logs_dir(output_dir)

    _llm_log_path = _paths.llm_prompts_path(output_dir)
    plog = ProcessLogger(None, llm_log_path=_llm_log_path)

    from .speaker_filter import apply_speaker_filter as _apply_sf
    segments_to_classify = _apply_sf(segments, config.speaker_filter)

    results_all, _ = classify_segments_zero_shot(
        segments=segments_to_classify,
        framework=framework,
        config=theme_config,
        resume_from=config.resume_from,
        process_logger=plog,
    )

    name_to_id = framework.build_name_to_id_map()
    updated, _ = parse_all_results(results_all, segments, name_to_id)
    # parse_all_results returns the updated list; splice back into segments
    segments[:] = updated
    plog.close_llm_log()


def _purer_llm_classify(config, segments, output_dir, observer):
    """Run PURER cue-unit classification in-place on therapist segments.

    Long cue blocks (total therapist words > max_cue_words) are split along
    turn boundaries into contiguous sub-cues before classification.  If a
    sub-cue gets an unparseable response the sub-cue is bisected and retried
    (up to MAX_BISECT_DEPTH times) so that single-turn failures are the only
    remaining unlabeled blocks.  A coverage report is written to the reports
    directory when the function returns.
    """
    _has_therapists = any(s.speaker == 'therapist' for s in segments)
    if not getattr(config, 'run_purer_labeler', False) or not _has_therapists:
        return

    from theme_framework.registry import load as _registry_load
    _therapist_fw_name = getattr(config, 'therapist_framework', 'purer')
    purer_framework = _registry_load(_therapist_fw_name or 'purer')
    purer_cfg = config.purer_classification
    purer_cfg.output_dir = _paths.auditable_logs_dir(output_dir)
    purer_cue = getattr(config, 'purer_cue', None)

    # Inherit backend/model from theme classification if not explicitly set
    tc = config.theme_classification
    _default_model = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
    if not purer_cfg.model or purer_cfg.model == _default_model:
        purer_cfg.model = tc.model
        purer_cfg.backend = tc.backend
        purer_cfg.api_key = tc.api_key
        purer_cfg.lmstudio_base_url = getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1')
        purer_cfg.temperature = tc.temperature
    if not getattr(purer_cfg, 'per_run_models', None):
        inherited = list(getattr(tc, 'per_run_models', []))
        if inherited:
            purer_cfg.per_run_models = inherited
            purer_cfg.n_runs = len(inherited)
        else:
            purer_cfg.n_runs = 1

    _llm_log_path = _paths.llm_prompts_path(output_dir)
    plog = ProcessLogger(None, llm_log_path=_llm_log_path)

    skip_lessons = getattr(purer_cue, 'skip_lesson_content', False)
    max_lesson_words = getattr(purer_cue, 'max_lesson_words', 400)
    max_ctx_words = getattr(purer_cue, 'max_context_words', 1000)
    ctx_window = getattr(purer_cfg, 'context_window_segments', 6)
    max_cue_words = getattr(purer_cue, 'max_cue_words', 300)

    from .cue_blocks import (
        cue_blocks_from_segments as _cue_blocks_from_segments,
        split_by_word_budget as _split_by_word_budget,
        format_purer_coverage as _format_purer_coverage,
    )
    sorted_segs, specs = _cue_blocks_from_segments(
        segments, stage_attr='primary_stage', require_stage=True
    )

    # ---------------------------------------------------------------------------
    # Coverage accumulators (per-session and overall)
    # ---------------------------------------------------------------------------
    _overall: dict = {
        'n_blocks': 0,
        'n_skipped_lesson': 0,
        'skipped_lesson_words': 0,
        'n_labeled_segments': 0,
        'labeled_words': 0,
        'n_unparseable': 0,
        'unparseable_words': 0,
        'total_therapist_words': 0,
    }
    _per_session: dict = {}  # session_id → same-keyed dict

    def _session_stats(sid: str) -> dict:
        if sid not in _per_session:
            _per_session[sid] = {
                'n_blocks': 0,
                'n_skipped_lesson': 0,
                'skipped_lesson_words': 0,
                'n_labeled_segments': 0,
                'labeled_words': 0,
                'n_unparseable': 0,
                'unparseable_words': 0,
                'total_therapist_words': 0,
            }
        return _per_session[sid]

    def _add(key: str, value: int, sid: str):
        _overall[key] = _overall.get(key, 0) + value
        ss = _session_stats(sid)
        ss[key] = ss.get(key, 0) + value

    # ---------------------------------------------------------------------------
    # Build initial cue_units list
    # ---------------------------------------------------------------------------
    # Each entry: {'segment': synthetic_Segment, 'from_segment': ...,
    #              'to_segment': ..., 'context_block': ...,
    #              '_constituents': [Segment, ...]}
    # The '_constituents' key is carried alongside (not sent to the LLM).

    cue_units: list = []

    for spec in specs:
        if not spec.therapist_items:
            continue
        from_seg = spec.from_item
        to_seg = spec.to_item
        session_id = spec.session_id
        between = spec.therapist_items
        from_idx = spec.from_index

        cue_text_full = '\n'.join(t.text.strip() for t in between if t.text.strip())
        cue_words = len(cue_text_full.split())

        # Accumulate total therapist words before any skip/split decisions.
        _add('n_blocks', 1, session_id)
        _add('total_therapist_words', cue_words, session_id)

        # --- skip-as-lesson branch ---
        if skip_lessons and cue_words > max_lesson_words:
            # This block is treated as long didactic monologue — do NOT classify.
            _add('n_skipped_lesson', 1, session_id)
            _add('skipped_lesson_words', cue_words, session_id)
            continue
        # When skip_lessons is False, even long/lesson blocks are NOT skipped;
        # they fall through to the split-by-budget logic below, getting chunked
        # and labeled rather than dropped.  (1d: lesson chunking when skip is OFF)

        exchange_ctx = (
            _build_context_block_for_purer(sorted_segs, from_idx, window_size=ctx_window,
                                           max_words=max_ctx_words)
            if from_idx >= 0 else ''
        )

        # --- split into sub-cues if block exceeds per-sub-cue word budget ---
        if cue_words > max_cue_words:
            sub_groups = _split_by_word_budget(between, max_cue_words, lambda s: s.text)
        else:
            # Single group — no splitting needed.
            sub_groups = [between]

        for k, sub_group in enumerate(sub_groups):
            sub_text = '\n'.join(t.text.strip() for t in sub_group if t.text.strip())
            sub_words = len(sub_text.split())
            from_id = from_seg.segment_id
            to_id = to_seg.segment_id
            # Unique synthetic id includes partition index so multi-sub-cue
            # blocks have distinct checkpoint keys.
            synthetic_id = f'purer_cue_{from_id}_to_{to_id}_p{k}'
            synthetic = Segment(
                segment_id=synthetic_id,
                trial_id=from_seg.trial_id,
                session_id=session_id,
                session_number=from_seg.session_number,
                cohort_id=from_seg.cohort_id,
                speaker='therapist',
                text=sub_text,
                word_count=sub_words,
                start_time_ms=sub_group[0].start_time_ms,
                end_time_ms=sub_group[-1].end_time_ms,
            )
            cue_units.append({
                'segment': synthetic,
                'from_segment': from_seg,
                'to_segment': to_seg,
                'context_block': exchange_ctx,
                '_constituents': sub_group,
                '_session_id': session_id,
            })

    if not cue_units:
        plog.close_llm_log()
        _write_purer_coverage_report(output_dir, _overall, _per_session,
                                     _format_purer_coverage)
        return

    # ---------------------------------------------------------------------------
    # Classify + bisect-retry (1c)
    # ---------------------------------------------------------------------------
    MAX_BISECT_DEPTH = 6

    # Strip '_constituents'/'_session_id' before passing to the LLM classifier
    # (it expects only the 4 canonical keys).
    def _strip_internal(units):
        return [
            {k: v for k, v in cu.items() if not k.startswith('_')}
            for cu in units
        ]

    def _is_failure(result: dict) -> bool:
        """Return True if this cue-unit result is missing or has no primary_stage."""
        if not isinstance(result, dict):
            return True
        consensus = result.get('consensus', {})
        if consensus.get('primary_stage') is None:
            return True
        if consensus.get('agreement_level') == 'none':
            return True
        return False

    def _classify_batch(units, resume_from):
        """Run _classify_purer_cue_units on *units*; return results dict."""
        if not units:
            return {}
        try:
            results, _ = _classify_purer_cue_units(
                cue_units=_strip_internal(units),
                framework=purer_framework,
                config=purer_cfg,
                resume_from=resume_from,
                process_logger=plog,
            )
            return results
        except Exception:
            return {}

    def _bisect_and_classify(failing_units: list, depth: int) -> dict:
        """
        Recursively bisect each failing multi-constituent cue-unit, classify
        the halves, and return a mapping synthetic_id → result for all
        (re-)classified sub-cues.  Single-constituent failures are left as-is.
        """
        if depth > MAX_BISECT_DEPTH or not failing_units:
            return {}

        new_units = []
        for cu in failing_units:
            constituents = cu['_constituents']
            if len(constituents) <= 1:
                # Cannot split further; will be recorded as unparseable.
                continue
            mid = len(constituents) // 2
            halves = [constituents[:mid], constituents[mid:]]
            parent_id = cu['segment'].segment_id
            for h_idx, half in enumerate(halves):
                h_text = '\n'.join(t.text.strip() for t in half if t.text.strip())
                h_words = len(h_text.split())
                h_id = f'{parent_id}_b{h_idx}'
                h_seg = Segment(
                    segment_id=h_id,
                    trial_id=cu['segment'].trial_id,
                    session_id=cu['segment'].session_id,
                    session_number=cu['segment'].session_number,
                    cohort_id=cu['segment'].cohort_id,
                    speaker='therapist',
                    text=h_text,
                    word_count=h_words,
                    start_time_ms=half[0].start_time_ms,
                    end_time_ms=half[-1].end_time_ms,
                )
                new_units.append({
                    'segment': h_seg,
                    'from_segment': cu['from_segment'],
                    'to_segment': cu['to_segment'],
                    'context_block': cu['context_block'],
                    '_constituents': half,
                    '_session_id': cu['_session_id'],
                })

        if not new_units:
            return {}

        # Classify this batch — resume_from=None to avoid checkpoint key reuse.
        batch_results = _classify_batch(new_units, resume_from=None)

        # Recurse on any new failures that are still splittable.
        still_failing = [
            nu for nu in new_units
            if _is_failure(batch_results.get(nu['segment'].segment_id))
            and len(nu['_constituents']) > 1
        ]
        deeper = _bisect_and_classify(still_failing, depth + 1)

        # Merge: deeper results override (they are more fine-grained).
        merged = {nu['segment'].segment_id: (nu, batch_results.get(nu['segment'].segment_id))
                  for nu in new_units}
        # Return a flat map: synthetic_id → (unit, result) for all new_units +
        # all recursively produced units.
        all_results: dict = {}
        for nu in new_units:
            sid_k = nu['segment'].segment_id
            all_results[sid_k] = (nu, batch_results.get(sid_k))
        all_results.update(deeper)
        return all_results

    # --- Initial classification pass ---
    initial_results = _classify_batch(cue_units, resume_from=config.resume_from)

    # Identify failures that have more than one constituent (splittable).
    failing_initial = [
        cu for cu in cue_units
        if _is_failure(initial_results.get(cu['segment'].segment_id))
        and len(cu['_constituents']) > 1
    ]

    # --- Bisect-retry pass ---
    bisect_results: dict = _bisect_and_classify(failing_initial, depth=1)
    # bisect_results maps synthetic_id → (unit, result)

    # ---------------------------------------------------------------------------
    # Propagate labels to constituent therapist Segments
    # ---------------------------------------------------------------------------

    def _propagate(unit: dict, result: dict):
        """Apply a successful consensus result to all constituent Segments."""
        if not isinstance(result, dict):
            return False
        consensus = result.get('consensus', {})
        primary = consensus.get('primary_stage')
        if primary is None:
            return False
        session_id = unit['_session_id']
        constituents = unit['_constituents']
        for th_seg in constituents:
            th_seg.purer_primary = primary
            th_seg.purer_secondary = consensus.get('secondary_stage')
            th_seg.purer_confidence_primary = consensus.get('primary_confidence', 0.0)
            th_seg.purer_confidence_secondary = consensus.get('secondary_confidence')
            th_seg.purer_justification = consensus.get('justification', '')
            th_seg.purer_run_consistency = consensus.get('n_agree', 0)
            th_seg.purer_agreement_level = consensus.get('agreement_level')
            n_agree = consensus.get('n_agree', 0)
            n_raters = consensus.get('n_raters', 1) or 1
            th_seg.purer_agreement_fraction = n_agree / n_raters
            th_seg.purer_needs_review = bool(consensus.get('needs_review', False))
            th_seg.purer_rater_ids = result.get('rater_ids') or []
            th_seg.purer_rater_votes = result.get('rater_votes') or []
            seg_words = getattr(th_seg, 'word_count', None) or len((th_seg.text or '').split())
            _add('n_labeled_segments', 1, session_id)
            _add('labeled_words', seg_words, session_id)
        return True

    # Process initial successes.
    for cu in cue_units:
        sid_k = cu['segment'].segment_id
        result = initial_results.get(sid_k)
        if _is_failure(result):
            # Will be handled by bisect results (or recorded as unparseable).
            continue
        _propagate(cu, result)

    # Process bisect results.
    for sid_k, (bisect_unit, bisect_result) in bisect_results.items():
        if not _is_failure(bisect_result):
            _propagate(bisect_unit, bisect_result)
        else:
            # Single-turn sub-cues that still failed after all bisect attempts.
            if len(bisect_unit['_constituents']) <= 1:
                sub_words = bisect_unit['segment'].word_count or 0
                _add('n_unparseable', 1, bisect_unit['_session_id'])
                _add('unparseable_words', sub_words, bisect_unit['_session_id'])

    # Also record initial single-turn failures that were not splittable and had
    # no bisect attempt (they were excluded from failing_initial above).
    for cu in cue_units:
        sid_k = cu['segment'].segment_id
        result = initial_results.get(sid_k)
        if _is_failure(result) and len(cu['_constituents']) <= 1:
            sub_words = cu['segment'].word_count or 0
            _add('n_unparseable', 1, cu['_session_id'])
            _add('unparseable_words', sub_words, cu['_session_id'])

    plog.close_llm_log()

    # ---------------------------------------------------------------------------
    # Write coverage report (1e)
    # ---------------------------------------------------------------------------
    _write_purer_coverage_report(output_dir, _overall, _per_session,
                                 _format_purer_coverage)


def _write_purer_coverage_report(output_dir: str, overall: dict, per_session: dict,
                                  format_fn) -> None:
    """Write report_purer_coverage.txt to the reports directory."""
    try:
        reports_dir = _paths.human_reports_dir(output_dir)
        os.makedirs(reports_dir, exist_ok=True)
        report_path = os.path.join(reports_dir, 'report_purer_coverage.txt')
        stats = dict(overall)
        stats['per_session'] = per_session
        report_text = format_fn(stats)
        with open(report_path, 'w', encoding='utf-8') as fh:
            fh.write(report_text)
    except Exception:
        pass  # Coverage report failure must never crash the pipeline.


def _codebook_classify(config, codebook, segments, output_dir, observer):
    """Run embedding + LLM codebook classification in-place on segments."""
    _llm_log_path = _paths.llm_prompts_path(output_dir)
    plog = ProcessLogger(None, llm_log_path=_llm_log_path)

    codebook_output_dir = _paths.codebook_raw_dir(output_dir)
    os.makedirs(codebook_output_dir, exist_ok=True)
    config.codebook_embedding.exemplar_export_path = os.path.join(
        codebook_output_dir, 'found_exemplar_utterances.json'
    )

    from .speaker_filter import apply_speaker_filter as _apply_sf
    cb_segments = _apply_sf(segments, config.speaker_filter)

    embedding_classifier = EmbeddingCodebookClassifier(config.codebook_embedding)
    embedding_results = embedding_classifier.classify_segments(cb_segments, codebook)

    tc = config.theme_classification
    llm_cfg = LLMClientConfig(
        backend=tc.backend,
        api_key=tc.api_key,
        model=tc.model,
        temperature=tc.temperature,
        lmstudio_base_url=getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
        ollama_host=getattr(tc, 'ollama_host', '0.0.0.0'),
        ollama_port=getattr(tc, 'ollama_port', 11434),
        process_logger=plog,
    )
    llm_client = LLMClient(llm_cfg)
    config.codebook_llm.output_dir = codebook_output_dir
    llm_classifier = LLMCodebookClassifier(llm_client, config.codebook_llm)
    try:
        llm_results = llm_classifier.classify_segments(
            cb_segments, codebook, output_dir=codebook_output_dir,
        )
    except Exception:
        llm_results = {}

    ensemble = CodebookEnsemble(config.codebook_ensemble)
    ensemble_results = ensemble.reconcile(embedding_results, llm_results)

    for seg in segments:
        if seg.segment_id in ensemble_results:
            ens = ensemble_results[seg.segment_id]
            seg.codebook_labels_embedding = sorted(
                a.code_id for a in embedding_results.get(seg.segment_id, [])
            )
            seg.codebook_labels_llm = sorted(
                a.code_id for a in llm_results.get(seg.segment_id, [])
            )
            seg.codebook_labels_ensemble = ens.final_codes
            seg.codebook_disagreements = [d['code_id'] for d in ens.disagreement_details]
            seg.codebook_confidence = {a.code_id: a.confidence for a in ens.final_assignments}

    plog.close_llm_log()


def _run_cv_stats(segments, framework, output_dir, observer):
    """Compute cross-validation co-occurrence stats and export to disk."""
    import pandas as pd
    segments_df = pd.DataFrame([vars(s) for s in segments])
    cooccurrence = compute_theme_codebook_cooccurrence(
        segments_df, framework,
        codebook_label_column='codebook_labels_ensemble',
        theme_label_column='primary_stage',
    )
    _cv_params = {'min_lift': 1.5, 'min_count': 3, 'top_n': 10}
    associations_by_theme = summarize_theme_code_associations(cooccurrence, **_cv_params)
    export_cross_validation_results(cooccurrence, associations_by_theme, _cv_params, output_dir)


def stage_validation_artifacts(
    config,
    framework,
    codebook=None,
    *,
    segments: Optional[List[Segment]] = None,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
    create_missing: bool = True,
) -> None:
    """
    Stage 7 (validation sub-stage) — human classification forms, flagged-for-review,
    validation testsets, and content-validity testset refresh.

    When segments=None, loads from frozen disk state with all overlays applied.
    When create_missing=False, only refreshes existing testsets/CV (does not create new ones).
    """
    if observer is None:
        observer = SilentObserver()

    _od = _resolve_output_dir(output_dir, config)

    if segments is None:
        segments = segments_io.load_segments_for_stage(
            _od, apply=('theme', 'purer', 'codebook', 'cv', 'gnn'),
        )

    if framework is None:
        from theme_framework.registry import load as _registry_load
        _fw_name = getattr(config, 'participant_framework', 'vaamr') if config else 'vaamr'
        framework = _registry_load(_fw_name or 'vaamr')

    # Human classification forms (blind-coding, no results)
    export_human_classification_forms(segments, framework, _od)

    # Dataset-wide flagged-for-review report
    export_flagged_for_review(segments, framework, _od)

    # Cross-session validation testsets
    if config is not None:
        ts_cfg = getattr(config, 'test_sets', None)
        if ts_cfg is not None:
            generate_or_refresh_validation_testsets(
                segments,
                framework,
                _od,
                test_sets_config=ts_cfg,
                codebook_enabled=getattr(config, 'run_codebook_classifier', False),
                codebook=codebook,
                create_missing=create_missing,
            )

        # Content-validity testsets
        cv_cfg = getattr(config, 'content_validity', None)
        if cv_cfg is not None:
            framework_purer = None
            if getattr(getattr(cv_cfg, 'purer', None), 'enabled', False):
                try:
                    from theme_framework.registry import load as _registry_load
                    _fw_name = getattr(config, 'therapist_framework', 'purer') or 'purer'
                    framework_purer = _registry_load(_fw_name)
                except Exception:
                    pass
            generate_or_refresh_content_validity_testsets(
                _od,
                cv_config=cv_cfg,
                framework_vaamr=framework if getattr(getattr(cv_cfg, 'vaamr', None), 'enabled', False) else None,
                framework_purer=framework_purer,
                theme_classification_cfg=getattr(config, 'theme_classification', None),
            )


def stage_ingest(
    config: PipelineConfig,
    *,
    output_dir: Optional[str] = None,
    observer: Optional[PipelineObserver] = None,
    force_reingest: Optional[str] = None,
    force_reingest_all: bool = False,
) -> List[Segment]:
    """
    Stage 1 — Transcript Ingestion and Segmentation.

    Segments every session whose frozen segments are missing or stale.
    Legacy projects are auto-migrated on first call.

    force_reingest: session_id to force re-segmentation for (ignores frozen guard).
    force_reingest_all: if True, re-segments all sessions.

    Returns the full interleaved list of all segments (participant + therapist).
    """
    if observer is None:
        observer = SilentObserver()

    _od = _resolve_output_dir(output_dir, config)
    os.makedirs(_od, exist_ok=True)

    meta_dir = _paths.meta_dir(_od)
    os.makedirs(meta_dir, exist_ok=True)
    _auditable_dir = _paths.auditable_logs_dir(_od)
    os.makedirs(_auditable_dir, exist_ok=True)

    _verbose = getattr(config.segmentation, 'verbose_segmentation', False)
    _plog_path = os.path.join(_auditable_dir, 'segmentation_process_log.txt') if _verbose else None
    _llm_log_path = _paths.llm_prompts_path(_od)
    plog = ProcessLogger(_plog_path, llm_log_path=_llm_log_path)

    _existing_speaker_map, _use_unknown_prefix = _load_speaker_map(meta_dir, config)

    observer.on_stage_start(
        "Transcript Ingestion and Segmentation", "1",
        explanation_key='ingestion',
    )

    sf = config.speaker_filter
    excluded_speakers = sf.speakers if sf.mode == 'exclude' else []

    seg_config = {
        'embedding_model': config.segmentation.embedding_model,
        'silence_threshold_ms': config.segmentation.silence_threshold_ms,
        'semantic_shift_percentile': config.segmentation.semantic_shift_percentile,
        'min_segment_words_conversational': config.segmentation.min_segment_words_conversational,
        'max_segment_words_conversational': config.segmentation.max_segment_words_conversational,
        'max_gap_seconds': getattr(config.segmentation, 'max_gap_seconds', 30.0),
        'min_words_per_sentence': getattr(config.segmentation, 'min_words_per_sentence', 10),
        'max_segment_duration_seconds': getattr(config.segmentation, 'max_segment_duration_seconds', 300.0),
        'excluded_speakers': excluded_speakers,
        'speaker_filter_mode': config.speaker_filter.mode,
        'use_adaptive_threshold': getattr(config.segmentation, 'use_adaptive_threshold', True),
        'min_prominence': getattr(config.segmentation, 'min_prominence', 0.05),
        'broad_window_size': getattr(config.segmentation, 'broad_window_size', 7),
        'use_topic_clustering': getattr(config.segmentation, 'use_topic_clustering', False),
        'process_logger': plog,
        'existing_speaker_map': _existing_speaker_map,
        'use_unknown_prefix': _use_unknown_prefix,
    }
    if legacy_migration.is_legacy_project(_od):
        _n_segs = legacy_migration.migrate_legacy_segments(_od)
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            f"Auto-migrated {_n_segs} sessions from v2.0 legacy layout",
        )
    if legacy_migration.is_v25_layout(_od):  # LEGACY-MIGRATION CALL SITE
        legacy_migration.migrate_v25_to_v3(_od)
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            "Auto-migrated v2.5 project layout to v3 directory structure",
        )
    if legacy_migration.is_jsonl_project(_od):
        _res = legacy_migration.migrate_jsonl_to_sqlite(_od)
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            f"Migrated {_res['segments']} segments from JSONL to SQLite ({_res['sessions']} sessions)",
        )
    if legacy_migration.upgrade_config_file(_od):
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            "Upgraded qra_config.json with defaults for newly-added parameters",
        )

    current_hash = segments_io.params_hash(config.segmentation)

    session_files = discover_session_files(config.transcript_dir)
    if not session_files:
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            f"Warning: No session files found in {config.transcript_dir}",
        )
        observer.on_stage_progress(
            "Transcript Ingestion and Segmentation",
            "Looking for JSON and VTT files directly...",
        )
        import glob as _glob
        session_files = sorted(
            set(_glob.glob(os.path.join(config.transcript_dir, '**/*.json'), recursive=True))
            | set(_glob.glob(os.path.join(config.transcript_dir, '**/*.vtt'), recursive=True))
        )

    # Nothing to segment → write an empty anonymization key and return WITHOUT
    # constructing the segmenter (which would eagerly load the embedding model).
    if not session_files:
        _speaker_key_path = os.path.join(_paths.meta_dir(_od), 'speaker_anonymization_key.json')
        with open(_speaker_key_path, 'w') as _f:
            json.dump({}, _f, indent=2)
        _write_anonymization_key_txt({}, _paths.anonymization_key_txt_path(_od))
        plog.close()
        observer.on_stage_complete(
            "Transcript Ingestion and Segmentation",
            "Produced 0 segments from 0 sessions",
        )
        return []

    # Construct the segmenter (loads the embedding model) only now that we know
    # there is work to do.
    use_llm_refine = getattr(config.segmentation, 'use_llm_refinement', True)
    segmenter = ConversationalSegmenter(seg_config)

    llm_refiner = None
    if use_llm_refine:
        theme_cfg = config.theme_classification
        refiner_llm_cfg = LLMClientConfig(
            backend=theme_cfg.backend,
            api_key=theme_cfg.api_key,
            model=theme_cfg.model,
            lmstudio_base_url=getattr(theme_cfg, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
            no_reasoning=True,
            process_logger=plog,
        )
        llm_refiner = LLMSegmentationRefiner(
            LLMClient(refiner_llm_cfg),
            {
                'mode': getattr(config.segmentation, 'llm_refinement_mode', 'full'),
                'ambiguity_threshold': getattr(config.segmentation, 'llm_ambiguity_threshold', 0.15),
                'batch_size': getattr(config.segmentation, 'llm_batch_size', 5),
                'excluded_speakers': excluded_speakers,
                'max_context_words': config.segmentation.max_segment_words_conversational,
                'max_context_duration_s': getattr(config.segmentation, 'max_segment_duration_seconds', 300.0),
                'max_gap_seconds': getattr(config.segmentation, 'max_gap_seconds', 30.0),
                'embedding_model': config.segmentation.embedding_model,
                'process_logger': plog,
            },
            speaker_normalizer=segmenter.speaker_norm,
        )

    all_segments: List[Segment] = []
    for session_file in session_files:
        sid = _resolve_session_id(session_file)

        _force_this = force_reingest_all or (force_reingest is not None and force_reingest == sid)

        if not _force_this and segments_io.is_segmentation_fresh(_od, sid, current_hash):
            observer.on_stage_progress(
                "Transcript Ingestion and Segmentation",
                f"Reusing frozen segments for {sid}",
            )
            all_segments.extend(segments_io.read_session_segments(_od, sid))
            continue

        if session_file.lower().endswith('.vtt'):
            session_data = load_vtt_session(session_file)
        else:
            session_data = load_diarized_session(session_file)
        metadata = session_data['metadata']
        metadata.setdefault('trial_id', config.trial_id)

        if session_file.lower().endswith('.vtt'):
            stem = os.path.splitext(os.path.basename(session_file))[0]
            parsed = parse_session_id_metadata(stem)
            metadata.setdefault('session_id', stem)
            metadata.setdefault('session_number', parsed['session_number'])
            metadata.setdefault('cohort_id', parsed['cohort_id'])
            metadata.setdefault('session_variant', parsed['session_variant'])
        else:
            default_session_id = os.path.basename(os.path.dirname(session_file))
            parsed = parse_session_id_metadata(default_session_id)
            metadata.setdefault('session_id', default_session_id)
            metadata.setdefault('session_number', parsed['session_number'])
            metadata.setdefault('cohort_id', parsed['cohort_id'])
            metadata.setdefault('session_variant', parsed['session_variant'])
        metadata.setdefault('source_file', session_file)

        if llm_refiner:
            result = segmenter.segment_session(
                session_data['sentences'], metadata,
                return_intermediates=True,
            )
            session_segments = result['segments']
            session_segments = llm_refiner.refine(
                session_segments,
                result['sentences'],
                result['sim_curve'],
                result['embeddings'],
                result.get('boundary_confidence'),
                original_sentences=result.get('original_sentences'),
            )
        else:
            session_segments = segmenter.segment_session(
                session_data['sentences'], metadata
            )

        _th_gap = getattr(getattr(config, 'purer_cue', None), 'therapist_max_gap_seconds', 120.0)
        therapist_segs = segmenter.extract_therapist_segments(
            session_data['sentences'], metadata, max_gap_seconds=_th_gap,
        )
        if therapist_segs:
            combined = sorted(session_segments + therapist_segs, key=lambda s: s.start_time_ms)
            for i, seg in enumerate(combined):
                seg.segment_index = i
            session_segments = combined
        else:
            for i, seg in enumerate(session_segments):
                seg.segment_index = i

        # --- PHI text anonymization (before freeze) ---
        if getattr(config, 'anonymize_transcript_text', True):
            _anon_map = dict(_existing_speaker_map)
            for _orig, (_role, _anon_id) in segmenter.speaker_norm.speaker_map.items():
                if _orig not in _anon_map:
                    _anon_map[_orig] = (_role, _anon_id)
            session_segments, _anon_stats = _scrub_segments(
                session_segments,
                _anon_map,
                use_transformer=True,
                confidence_threshold=getattr(config, 'anonymize_text_confidence_threshold', 0.6),
                model_name=getattr(config, 'anonymize_text_model', 'obi/deid_roberta_i2b2'),
            )
            if _anon_stats['n_known'] or _anon_stats['n_unknown']:
                plog._write(
                    f"[text_anonymization] {sid}: "
                    f"{_anon_stats['n_known']} known + {_anon_stats['n_unknown']} unknown "
                    f"replacements in {_anon_stats['n_segments_modified']} segments "
                    f"(engine: {_anon_stats.get('engine_backend', '?')})\n"
                )

        import shutil as _shutil
        _diar_dir = _paths.transcripts_diarized_dir(_od)
        os.makedirs(_diar_dir, exist_ok=True)
        _diar_dest = os.path.join(_diar_dir, os.path.basename(session_file))
        if not os.path.exists(_diar_dest):
            _shutil.copy2(session_file, _diar_dest)

        segments_io.write_session_segments(_od, sid, session_segments, current_hash,
                                           force=_force_this)
        all_segments.extend(session_segments)

    session_counts = Counter(s.session_id for s in all_segments)
    for seg in all_segments:
        seg.total_segments_in_session = session_counts[seg.session_id]

    plog.close()

    observer.on_stage_complete(
        "Transcript Ingestion and Segmentation",
        f"Produced {len(all_segments)} segments from {len(session_files)} sessions",
    )

    # Write speaker anonymization key while segmenter is still live (speaker_map is in-memory).
    _speaker_key_path = os.path.join(_paths.meta_dir(_od), 'speaker_anonymization_key.json')
    _speaker_key = {
        orig: {'role': role, 'anonymized_id': anon_id}
        for orig, (role, anon_id) in segmenter.speaker_norm.speaker_map.items()
    }
    with open(_speaker_key_path, 'w') as _f:
        json.dump(_speaker_key, _f, indent=2)
    _write_anonymization_key_txt(_speaker_key, _paths.anonymization_key_txt_path(_od))

    # Release GPU memory held by segmenter and refiner before classification stages.
    segmenter.release_gpu_memory()
    if llm_refiner is not None and hasattr(llm_refiner, '_embed_model'):
        llm_refiner._embed_model = None
    try:
        import gc, torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass

    return all_segments


def run_incremental_pipeline(
    config: PipelineConfig,
    framework: ThemeFramework,
    codebook=None,
    observer: Optional[PipelineObserver] = None,
    *,
    walkthrough: bool = False,
) -> pd.DataFrame:
    """Add new transcripts to an existing project without disturbing frozen artifacts.

    Discovers sessions in config.transcript_dir whose frozen segments are missing
    or stale, optionally runs an interactive speaker walkthrough, segments only
    the new sessions, then for each classifier key already recorded in the
    classification manifest, classifies only the new segments and merges results
    into the existing overlay. Re-assembles the master dataset and re-runs
    analysis (if enabled). Frozen testsets and content-validity worksheets are
    never mutated.

    Parameters
    ----------
    config : PipelineConfig
    framework : ThemeFramework
    codebook : optional
    observer : PipelineObserver, optional
    walkthrough : bool
        When True, runs the interactive speaker walkthrough even if zero new
        speakers are detected (used by qra add-data). When False, the
        walkthrough still runs IF new speakers are detected, but exits silently
        otherwise (used by transparent auto-routing from run_full_pipeline).

    Returns
    -------
    pd.DataFrame
        The full (old + new) master dataset.
    """
    if observer is None:
        observer = SilentObserver()

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 0: discover new sessions
    # ------------------------------------------------------------------
    current_hash = segments_io.params_hash(config.segmentation)
    session_files = discover_session_files(config.transcript_dir)
    if not session_files:
        import glob as _glob
        session_files = sorted(
            set(_glob.glob(os.path.join(config.transcript_dir, '**/*.json'), recursive=True))
            | set(_glob.glob(os.path.join(config.transcript_dir, '**/*.vtt'), recursive=True))
        )

    new_session_files: List[str] = []
    new_sids: set = set()
    for sf in session_files:
        sid = _resolve_session_id(sf)
        if not segments_io.is_segmentation_fresh(output_dir, sid, current_hash):
            new_session_files.append(sf)
            new_sids.add(sid)

    print()
    print("=" * 70)
    print("QRA INCREMENTAL PIPELINE")
    print("=" * 70)
    print(f"  Discovered {len(session_files)} session file(s); "
          f"{len(new_sids)} new (un-segmented).")

    # ------------------------------------------------------------------
    # Phase 1: speaker walkthrough (interactive)
    # ------------------------------------------------------------------
    if walkthrough or new_session_files:
        try:
            from .speaker_walkthrough import run_speaker_walkthrough, discover_new_speakers
            meta_dir = _paths.meta_dir(output_dir)
            os.makedirs(meta_dir, exist_ok=True)
            existing_map, _ = _load_speaker_map(meta_dir, config)
            new_speakers = discover_new_speakers(new_session_files, existing_map)
            if walkthrough or new_speakers:
                run_speaker_walkthrough(config, new_session_files, output_dir=output_dir)
        except KeyboardInterrupt:
            print("\n  Walkthrough cancelled by user.")
            raise

    # ------------------------------------------------------------------
    # Phase 2: segment new sessions only (stage_ingest already skips fresh)
    # ------------------------------------------------------------------
    all_segments = stage_ingest(config, output_dir=output_dir, observer=observer)

    # ------------------------------------------------------------------
    # Phase 3: per-key incremental classification
    # ------------------------------------------------------------------
    manifest = _cio.read_classification_manifest(output_dir) or {}
    kinds_applied: List[str] = []

    if not new_sids:
        print("\n  No new sessions to classify. Skipping classification stages.")
    else:
        if 'theme' in manifest and config.run_theme_labeler:
            pinned = resolve_pinned_classifier_config(output_dir, 'theme', config, framework=framework)
            all_segments = stage_classify_theme(
                pinned, framework,
                segments=all_segments, output_dir=output_dir, observer=observer,
                only_session_ids=new_sids,
            )
            kinds_applied.append('theme')

        if 'purer' in manifest and config.run_purer_labeler and any(
                s.speaker == 'therapist' for s in all_segments):
            pinned = resolve_pinned_classifier_config(output_dir, 'purer', config)
            all_segments = stage_classify_purer(
                pinned,
                segments=all_segments, output_dir=output_dir, observer=observer,
                only_session_ids=new_sids,
            )
            kinds_applied.append('purer')

        if 'codebook' in manifest and config.run_codebook_classifier:
            cb = codebook
            if cb is None:
                from codebook.phenomenology_codebook import get_phenomenology_codebook
                cb = get_phenomenology_codebook()
            pinned = resolve_pinned_classifier_config(output_dir, 'codebook', config, codebook=cb)
            all_segments = stage_classify_codebook(
                pinned, cb,
                segments=all_segments, output_dir=output_dir, observer=observer,
                only_session_ids=new_sids,
            )
            kinds_applied.append('codebook')

        if 'cv' in manifest and config.run_theme_labeler and config.run_codebook_classifier:
            pinned = resolve_pinned_classifier_config(output_dir, 'cv', config, framework=framework, codebook=codebook)
            all_segments = stage_cross_validation(
                pinned, framework,
                segments=all_segments, output_dir=output_dir, observer=observer,
                only_session_ids=new_sids,
            )
            kinds_applied.append('cv')

    # ------------------------------------------------------------------
    # Phase 4: re-assemble master dataset
    # ------------------------------------------------------------------
    confidence_tier_config = asdict(config.confidence_tiers)
    _msdir = _paths.master_segments_dir(output_dir)
    os.makedirs(_msdir, exist_ok=True)
    _gnn_auth, _gate_ok = _gnn_promotion_flags(config, output_dir)
    master_df = assemble_master_dataset(
        all_segments,
        os.path.join(_msdir, 'master_segments.csv'),
        confidence_tiers=confidence_tier_config,
        gnn_authoritative=_gnn_auth,
        gate_passed=_gate_ok,
    )

    # ------------------------------------------------------------------
    # Phase 5: refresh validation artifacts (do NOT create new testsets)
    # ------------------------------------------------------------------
    try:
        stage_validation_artifacts(
            config, framework, codebook,
            segments=all_segments, output_dir=output_dir, observer=observer,
            create_missing=False,
        )
    except Exception as e:
        print(f"  Warning: validation refresh failed: {e}")

    # ------------------------------------------------------------------
    # Phase 6: re-run analysis (idempotent — overwrites cleanly)
    # ------------------------------------------------------------------
    if getattr(config, 'auto_analyze', False):
        try:
            from analysis.runner import run_analysis
            observer.on_stage_start("Results Analysis", "8",
                                    explanation_key='results_analysis')
            analysis_result = run_analysis(
                output_dir, verbose=False,
                llm_log_path=_paths.llm_prompts_path(output_dir),
            )
            observer.on_stage_complete(
                "Results Analysis",
                f"Analysis complete: {len(analysis_result['files_generated'])} files written",
            )
        except ImportError:
            pass
        except Exception as e:
            print(f"\n  Warning: results analysis failed: {e}")

    # ------------------------------------------------------------------
    # Closing summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("INCREMENTAL RUN COMPLETE")
    print("=" * 70)
    print(f"  New sessions segmented: {len(new_sids)}")
    print(f"  Classifiers applied:    {', '.join(kinds_applied) if kinds_applied else '(none — no manifest keys)'}")
    print(f"  Total segments:         {len(master_df)}")
    if new_sids:
        print()
        print("  If you want a validation testset that includes the new cohort, run:")
        print("    qra testset create --kind {vaamr|purer|codebook} --name <name> "
              f"--output-dir {output_dir}")

    try:
        from .output_index import write_index
        write_index(output_dir)
    except Exception:
        pass

    return master_df


def run_full_pipeline(
    config: PipelineConfig,
    framework: ThemeFramework,
    codebook=None,
    observer: Optional[PipelineObserver] = None,
) -> pd.DataFrame:
    """
    Execute the complete classification pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Pipeline configuration with paths, model settings, and thresholds.
    framework : ThemeFramework
        The theme/stage framework to use for classification.
    codebook : optional
        Codebook instance for codebook classification.
    observer : PipelineObserver, optional
        Observer for pipeline events (UI feedback). Defaults to SilentObserver.

    Returns
    -------
    pd.DataFrame
        The master segment dataset.
    """
    if observer is None:
        observer = SilentObserver()

    output_dir = config.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Auto-route to incremental mode when an existing project is detected.
    # Force-reprocessing is opt-in via QRA_FORCE_FULL=1 (CLI flag could be added later).
    _existing_sids = segments_io.list_segmented_sessions(config.output_dir)
    _existing_manifest = _cio.read_classification_manifest(config.output_dir) or {}
    if (_existing_sids and _existing_manifest
            and not os.environ.get('QRA_FORCE_FULL')):
        return run_incremental_pipeline(
            config, framework, codebook=codebook, observer=observer,
        )

    # ------------------------------------------------------------------
    # Pre-flight: ensure embedding model is downloaded before the pipeline
    # starts.  Both segmentation and codebook classification use the same
    # model; downloading it now avoids a surprise mid-run pause.
    # ------------------------------------------------------------------
    seg_emb_model = config.segmentation.embedding_model
    print(f"  Checking embedding model: {seg_emb_model}")
    ensure_embedding_model_ready(seg_emb_model)

    # Stage 1 — delegated to extracted stage_ingest function.
    # speaker_key_path is retained here because Stage 7 writes the anonymization key.
    speaker_key_path = os.path.join(_paths.meta_dir(output_dir), 'speaker_anonymization_key.json')
    all_segments = stage_ingest(config, output_dir=output_dir, observer=observer)

    # ------------------------------------------------------------------
    # Stage 2: Construct Operationalization
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Construct Operationalization", "2",
        explanation_key='operationalization',
    )

    content_validity_items = create_content_validity_test_set(framework)
    export_theme_definitions(
        framework,
        os.path.join(_paths.auditable_logs_dir(output_dir), 'theme_definitions.json'),
    )
    export_theme_definitions_txt(
        framework,
        config.theme_classification,
        _paths.theme_definitions_txt_path(output_dir),
    )
    _cv_dir = _paths.content_validity_dir(output_dir)
    os.makedirs(_cv_dir, exist_ok=True)
    export_content_validity_test_set(
        content_validity_items,
        os.path.join(_cv_dir, 'content_validity_test_set.jsonl'),
    )
    export_content_validity_human_worksheet(
        content_validity_items,
        framework,
        output_dir,
    )
    export_content_validity_definition_key(
        framework,
        output_dir,
    )
    export_content_validity_answer_key(
        content_validity_items,
        framework,
        output_dir,
    )

    observer.on_stage_complete(
        "Construct Operationalization",
        f"Built {len(content_validity_items)} content validity test items; "
        "exported content_validity_test_set.jsonl, "
        "content_validity_human_worksheet.txt, "
        "content_validity_definition_key.txt, "
        "and content_validity_answer_key.txt",
    )

    # ------------------------------------------------------------------
    # Stage 3: Zero-Shot LLM Theme Classification
    # ------------------------------------------------------------------
    if config.run_theme_labeler:
        all_segments = stage_classify_theme(
            config, framework,
            segments=all_segments, output_dir=output_dir, observer=observer,
        )
    else:
        observer.on_stage_progress(
            "Zero-Shot LLM Theme Classification",
            "Skipping theme classification (run_theme_labeler=False)",
        )

    # ------------------------------------------------------------------
    # Stage 3c: PURER Cue-Unit Classification (optional)
    # ------------------------------------------------------------------
    _has_therapists = any(s.speaker == 'therapist' for s in all_segments)
    if config.run_purer_labeler and _has_therapists:
        all_segments = stage_classify_purer(
            config, segments=all_segments, output_dir=output_dir, observer=observer,
        )
    elif config.run_purer_labeler and not _has_therapists:
        observer.on_stage_progress(
            "PURER Cue-Unit Classification",
            "Skipping PURER — no therapist speakers found in session data",
        )
    else:
        observer.on_stage_progress(
            "PURER Cue-Unit Classification",
            "Skipping PURER classification (run_purer_labeler=False)",
        )


    # ------------------------------------------------------------------
    # Stage 3b: Codebook Classification (optional)
    # ------------------------------------------------------------------
    if config.run_codebook_classifier:
        if codebook is None:
            from codebook.phenomenology_codebook import get_phenomenology_codebook
            codebook = get_phenomenology_codebook()
        # Persist codebook definitions for downstream `qra analyze` (no equivalent in _codebook_classify).
        _cb_def_path = os.path.join(_paths.meta_dir(output_dir), 'codebook_definitions.json')
        if not os.path.exists(_cb_def_path):
            _cb_defs = {
                'name': codebook.name,
                'version': codebook.version,
                'description': codebook.description,
                'codes': [
                    {
                        'code_id': c.code_id,
                        'category': c.category,
                        'domain': c.domain,
                        'description': c.description,
                        'inclusive_criteria': c.inclusive_criteria,
                        'exclusive_criteria': c.exclusive_criteria,
                        'exemplar_utterances': c.exemplar_utterances,
                    }
                    for c in codebook.codes
                ],
            }
            with open(_cb_def_path, 'w') as _f:
                json.dump(_cb_defs, _f, indent=2)
        all_segments = stage_classify_codebook(
            config, codebook,
            segments=all_segments, output_dir=output_dir, observer=observer,
        )

    # ------------------------------------------------------------------
    # Stage 4: Cross-Validation (optional, when both theme and codebook)
    # ------------------------------------------------------------------
    if config.run_theme_labeler and config.run_codebook_classifier:
        all_segments = stage_cross_validation(
            config, framework,
            segments=all_segments, output_dir=output_dir, observer=observer,
        )

    # ------------------------------------------------------------------
    # Stage 5: Preparing Human Validation Set
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Preparing Human Validation Set", "5",
        explanation_key='human_validation_set',
    )

    if all_segments:
        segments_df = pd.DataFrame([vars(s) for s in all_segments])
        participant_labeled = segments_df[
            (segments_df['speaker'] == 'participant')
            & (segments_df['primary_stage'].notna())
        ]
    else:
        participant_labeled = pd.DataFrame()

    if len(participant_labeled) > 0:
        eval_set = create_balanced_evaluation_set(
            participant_labeled,
            n_per_class=config.validation.n_per_class,
        )
        _hedir = _paths.human_eval_dir(output_dir)
        os.makedirs(_hedir, exist_ok=True)
        eval_set.to_csv(
            os.path.join(_hedir, 'human_coding_evaluation_set.csv'),
            index=False,
        )
        observer.on_stage_complete(
            "Preparing Human Validation Set",
            f"Exported {len(eval_set)} segments for human coding",
        )
    else:
        observer.on_stage_complete(
            "Preparing Human Validation Set",
            "No labeled participant segments available for evaluation set",
        )

    # ------------------------------------------------------------------
    # Stage 6: Dataset Assembly
    # ------------------------------------------------------------------
    observer.on_stage_start(
        "Dataset Assembly", "6",
        explanation_key='dataset_assembly',
    )

    confidence_tier_config = asdict(config.confidence_tiers)
    _msdir = _paths.master_segments_dir(output_dir)
    os.makedirs(_msdir, exist_ok=True)
    _gnn_auth, _gate_ok = _gnn_promotion_flags(config, output_dir)
    master_df = assemble_master_dataset(
        all_segments,
        os.path.join(_msdir, 'master_segments.csv'),
        confidence_tiers=confidence_tier_config,
        gnn_authoritative=_gnn_auth,
        gate_passed=_gate_ok,
    )

    observer.on_stage_complete(
        "Dataset Assembly",
        f"Master dataset assembled with {len(master_df)} segments",
    )

    # ------------------------------------------------------------------
    # Stage 7: Coded Transcript + Statistics Reports
    # ------------------------------------------------------------------
    observer.on_stage_start("Report Generation", "7", explanation_key='report_generation')

    # Speaker anonymization key was written by stage_ingest; just log its presence.
    speaker_key: dict = {}
    if os.path.isfile(speaker_key_path):
        try:
            with open(speaker_key_path) as _f:
                speaker_key = json.load(_f)
        except (OSError, json.JSONDecodeError):
            pass
    observer.on_stage_progress(
        "Report Generation",
        "  Speaker anonymization key: 02_meta/speaker_anonymization_key.json + anonymization_key.txt",
    )

    # Build a lightweight LLM client for rationale summarization (reuses theme config)
    _sum_client = None
    if config.run_theme_labeler:
        tc = config.theme_classification
        try:
            _sum_client = LLMClient(LLMClientConfig(
                backend=tc.backend,
                api_key=tc.api_key,
                model=tc.model,
                temperature=0.0,
                lmstudio_base_url=getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
                ollama_host=getattr(tc, 'ollama_host', '0.0.0.0'),
                ollama_port=getattr(tc, 'ollama_port', 11434),
            ))
        except Exception:
            _sum_client = None

    # Per-session coded transcripts
    for session_id, session_df in (master_df.groupby('session_id')
                                   if not master_df.empty and 'session_id' in master_df.columns
                                   else []):
        segs_for_session = [s for s in all_segments if s.session_id == session_id]
        export_coded_transcript(
            segs_for_session, framework, codebook, output_dir, session_id,
            llm_client=_sum_client,
        )
        observer.on_stage_progress(
            "Report Generation",
            f"  Coded transcript: 04_validation/full_transcripts/coded_transcript_{session_id}.txt",
        )

    # Human forms, flagged-for-review, testsets, CV testsets
    stage_validation_artifacts(
        config, framework, codebook,
        segments=all_segments, output_dir=output_dir, observer=observer,
    )
    observer.on_stage_progress(
        "Report Generation",
        "  Human forms, flagged-for-review, and testset/CV answer keys written.",
    )

    # Per-transcript stats (one JSON per session)
    export_per_transcript_stats(master_df, framework, codebook, output_dir)
    observer.on_stage_progress(
        "Report Generation",
        "  Per-transcript stats: 03_analysis_data/session_stats/stats_<session>.json",
    )

    # Cumulative report across all transcripts
    export_cumulative_report(master_df, framework, codebook, output_dir)
    observer.on_stage_progress(
        "Report Generation",
        "  Cumulative report: 03_analysis_data/cumulative_report.json",
    )

    # BERT training data export
    export_training_data(all_segments, framework, codebook, output_dir)
    observer.on_stage_progress(
        "Report Generation",
        "  Training data: 02_meta/training_data/theme_classification.jsonl + codebook_multilabel.jsonl",
    )

    observer.on_stage_complete(
        "Report Generation",
        f"Reports written to {output_dir}",
    )

    # ------------------------------------------------------------------
    # Pipeline Complete
    # ------------------------------------------------------------------
    observer.on_pipeline_complete(
        output_dir,
        total_segments=len(master_df),
    )

    # ------------------------------------------------------------------
    # Optional: Post-pipeline Results Analysis
    # ------------------------------------------------------------------
    if getattr(config, 'auto_analyze', False):
        try:
            from analysis.runner import run_analysis
            observer.on_stage_start("Results Analysis", "8",
                                    explanation_key='results_analysis')
            analysis_result = run_analysis(output_dir, verbose=False, llm_log_path=_paths.llm_prompts_path(output_dir))
            observer.on_stage_complete(
                "Results Analysis",
                f"Analysis complete: {len(analysis_result['files_generated'])} files "
                f"written to 06_reports/ and 03_analysis_data/",
            )
        except ImportError:
            pass  # analysis module not available — skip silently
        except Exception as e:
            print(f"\n  Warning: results analysis failed: {e}")
            print(f"  Run manually: python qra.py analyze --output-dir {output_dir}")

    # Write directory index last, after all outputs are in place.
    try:
        from .output_index import write_index
        write_index(output_dir)
    except Exception:
        pass

    return master_df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _write_anonymization_key_txt(speaker_key: dict, output_path: str) -> None:
    """Write a human-readable table of speaker anonymization mappings."""
    import datetime as _dt

    col_orig = max((len(k) for k in speaker_key), default=14)
    col_orig = max(col_orig, len('Original Label'))
    col_role = max((len(v.get('role', '')) for v in speaker_key.values()), default=4)
    col_role = max(col_role, len('Role'))
    col_anon = max((len(v.get('anonymized_id', '')) for v in speaker_key.values()), default=13)
    col_anon = max(col_anon, len('Anonymized ID'))

    sep = f"  {'-' * col_orig}  {'-' * col_role}  {'-' * col_anon}"
    header = f"  {'Original Label':<{col_orig}}  {'Role':<{col_role}}  {'Anonymized ID':<{col_anon}}"

    lines = [
        'SPEAKER ANONYMIZATION KEY',
        f'Generated: {_dt.datetime.utcnow().strftime("%Y-%m-%d")}',
        '',
        header,
        sep,
    ]
    for original, entry in sorted(speaker_key.items(), key=lambda x: x[1].get('anonymized_id', '')):
        role = entry.get('role', '')
        anon_id = entry.get('anonymized_id', '')
        lines.append(f"  {original:<{col_orig}}  {role:<{col_role}}  {anon_id:<{col_anon}}")
    lines.append('')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')

