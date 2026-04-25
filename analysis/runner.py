"""
analysis/runner.py
------------------
Top-level orchestrator for the results analysis module.

Called by `python qra.py analyze --output-dir ...` and optionally by the
pipeline after Stage 7 when config.auto_analyze is True.
"""

import json
import os
import traceback


def run_analysis(output_dir: str, verbose: bool = True) -> dict:
    """Execute the full results analysis on an existing pipeline output directory.

    Reads master_segment_dataset.csv and theme_definitions.json from output_dir.
    Writes all reports to output_dir/02_human_reports/ and output_dir/04_analysis_data/.

    Parameters
    ----------
    output_dir : str
        Pipeline output directory (must contain master_segment_dataset.csv
        and theme_definitions.json).
    verbose : bool
        Print progress messages to stdout.

    Returns
    -------
    dict with keys:
        output_dir, n_segments, n_participants, n_sessions, files_generated
    """
    from .loader import load_segments, load_framework, sort_session_ids
    from .participant import generate_all_participant_reports
    from .session import generate_all_session_analyses
    from .construct import generate_all_construct_reports, generate_codebook_text_report
    from .figure_data import export_all_graphing_datasets
    from .longitudinal import generate_longitudinal_summary
    from .stage_progression import compute_session_stage_progression
    from .reports import (
        generate_comprehensive_session_report,
        generate_all_stage_text_reports,
        generate_transition_explanation,
        generate_therapist_cues_report,
        generate_longitudinal_text_report,
    )
    from process import output_paths as _paths

    def log(msg):
        if verbose:
            print(f"  {msg}")

    # ----------------------------------------------------------------
    # 0. Load pipeline config for LLM-backed features (best-effort)
    # ----------------------------------------------------------------
    therapist_cue_config = None
    llm_client = None
    _pipeline_config = None

    for _cfg_path in (
        os.path.join(_paths.meta_dir(output_dir), 'qra_config.json'),
        os.path.join(output_dir, 'qra_config.json'),
        os.path.join(output_dir, 'config.json'),
    ):
        if os.path.isfile(_cfg_path):
            try:
                with open(_cfg_path, encoding='utf-8') as _f:
                    _raw = json.load(_f)
                from process.setup_wizard import build_config_from_wizard_data
                _pipeline_config = build_config_from_wizard_data(_raw)
            except Exception:
                pass
            break

    if _pipeline_config is not None:
        _tc_cfg = _therapist_cue_config_raw = _pipeline_config.therapist_cues
        if _tc_cfg.enabled:
            therapist_cue_config = _tc_cfg
            try:
                from classification_tools.llm_client import LLMClient, LLMClientConfig
                _tc = _pipeline_config.theme_classification
                if _tc and _tc.model:
                    _llm_cfg = LLMClientConfig(
                        backend=_tc.backend,
                        api_key=_tc.api_key,
                        replicate_api_token=_tc.replicate_api_token,
                        model=_tc.model,
                        models=[_tc.model],
                        temperature=_tc.temperature,
                        lmstudio_base_url=getattr(_tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
                    )
                    llm_client = LLMClient(_llm_cfg)
            except Exception as _e:
                log(f"Warning: could not initialize LLM for cue response: {_e}")
    else:
        log("Note: no pipeline config found in output_dir — therapist cue summarization skipped.")

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    log("[1/8] Loading segments...")
    # Full dataset (all speakers, including therapist) for cue collection
    df_all = None
    try:
        df_all = load_segments(output_dir, speaker_filter=None, require_labeled=False)
    except Exception:
        pass

    try:
        df = load_segments(output_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return {'output_dir': output_dir, 'n_segments': 0,
                'n_participants': 0, 'n_sessions': 0, 'files_generated': []}

    try:
        framework = load_framework(output_dir)
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return {'output_dir': output_dir, 'n_segments': 0,
                'n_participants': 0, 'n_sessions': 0, 'files_generated': []}

    n_segments = len(df)
    n_participants = df['participant_id'].nunique()
    n_sessions = df['session_id'].nunique()

    log(f"    {n_segments} segments | {n_participants} participants | {n_sessions} sessions")

    if n_segments == 0:
        print("  No labeled segments found — nothing to analyze.")
        return {'output_dir': output_dir, 'n_segments': 0,
                'n_participants': n_participants, 'n_sessions': n_sessions,
                'files_generated': []}

    if n_sessions < 2 and verbose:
        print("  Note: only 1 session found — longitudinal reports will have limited data.")

    # Create output subdirs
    for _d in (
        _paths.sessions_json_dir(output_dir),
        _paths.participants_json_dir(output_dir),
        _paths.constructs_dir(output_dir),
        _paths.graphing_dir(output_dir),
        _paths.figures_dir(output_dir),
        _paths.longitudinal_dir(output_dir),
        _paths.analysis_data_dir(output_dir),
        _paths.human_reports_dir(output_dir),
    ):
        os.makedirs(_d, exist_ok=True)

    files_generated = []

    # ----------------------------------------------------------------
    # 2. Per-session analyses
    # ----------------------------------------------------------------
    log(f"[2/8] Generating per-session analyses ({n_sessions} sessions)...")
    try:
        session_reports = generate_all_session_analyses(df, framework, output_dir)
        for r in session_reports:
            sid = r.get('session_id', '')
            files_generated.append(
                os.path.join(_paths.sessions_json_dir(output_dir), f'session_{sid}.json')
            )
        log(f"    {len(session_reports)} session reports written.")
    except Exception as e:
        print(f"  Warning: session analyses failed: {e}")
        if verbose:
            traceback.print_exc()
        session_reports = []

    # ----------------------------------------------------------------
    # 3. Per-participant reports
    # ----------------------------------------------------------------
    log(f"[3/8] Generating per-participant reports ({n_participants} participants)...")
    try:
        participant_reports = generate_all_participant_reports(df, framework, output_dir)
        for r in participant_reports:
            pid = r.get('participant_id', '')
            files_generated.append(
                os.path.join(_paths.participants_json_dir(output_dir), f'participant_{pid}.json')
            )
        log(f"    {len(participant_reports)} participant reports written.")
    except Exception as e:
        print(f"  Warning: participant reports failed: {e}")
        if verbose:
            traceback.print_exc()
        participant_reports = []

    # ----------------------------------------------------------------
    # 4. Per-construct reports
    # ----------------------------------------------------------------
    n_stages = len(framework)
    log(f"[4/8] Generating per-construct reports ({n_stages} stages + codebook codes)...")
    stage_reports = []
    try:
        stage_reports = generate_all_construct_reports(df, framework, output_dir) or []
        # Collect written files
        _cjdir = _paths.constructs_json_dir(output_dir)
        if os.path.isdir(_cjdir):
            for fname in os.listdir(_cjdir):
                if fname.endswith('.json'):
                    files_generated.append(os.path.join(_cjdir, fname))
        log(f"    Construct reports written to 02_human_reports/per_construct/.")
    except Exception as e:
        print(f"  Warning: construct reports failed: {e}")
        if verbose:
            traceback.print_exc()

    try:
        ref_path = generate_codebook_text_report(df, framework, output_dir)
        if ref_path:
            files_generated.append(ref_path)
            log("    Codebook exemplars: 02_human_reports/per_construct/codebook_exemplars.txt")
    except Exception as e:
        print(f"  Warning: codebook exemplars report failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 5. Graph-ready CSVs
    # ----------------------------------------------------------------
    log("[5/8] Exporting graph-ready datasets...")
    try:
        csv_paths = export_all_graphing_datasets(df, framework, output_dir)
        files_generated.extend(csv_paths)
        log(f"    {len(csv_paths)} CSV files written to 04_analysis_data/graphing/.")
    except Exception as e:
        print(f"  Warning: graphing exports failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 6. Longitudinal summary
    # ----------------------------------------------------------------
    log("[6/8] Generating longitudinal summary...")
    try:
        generate_longitudinal_summary(df, participant_reports, framework, output_dir)
        files_generated.append(os.path.join(_paths.analysis_data_dir(output_dir), 'longitudinal_summary.json'))
        log("    Longitudinal summary written.")
    except Exception as e:
        print(f"  Warning: longitudinal summary failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 7. Longitudinal outputs: session stage progression + adjacency index
    # ----------------------------------------------------------------
    log("[7/8] Computing session stage progression...")
    try:
        progression_df = compute_session_stage_progression(df, framework, output_dir)
        files_generated.append(os.path.join(_paths.longitudinal_dir(output_dir), 'session_stage_progression.csv'))
        log(f"    {len(progression_df)} progression rows written.")
    except Exception as e:
        print(f"  Warning: session stage progression failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 8. Figures
    # ----------------------------------------------------------------
    log("[8/8] Generating analysis figures...")
    try:
        from .figures import generate_all_figures
        fig_paths = generate_all_figures(df, framework, output_dir)
        files_generated.extend(fig_paths)
        log(f"    {len(fig_paths)} figures written.")
    except Exception as e:
        print(f"  Warning: figure generation failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 9. Human-readable text reports
    # ----------------------------------------------------------------
    log("[9/8] Generating human-readable text reports...")
    try:
        txt_paths = []

        path = generate_comprehensive_session_report(
            df, framework, session_reports, output_dir
        )
        if path:
            txt_paths.append(path)

        stage_txt_paths = generate_all_stage_text_reports(
            df, framework, stage_reports, output_dir
        )
        txt_paths.extend(stage_txt_paths)

        path = generate_transition_explanation(
            df, framework, output_dir,
            therapist_cue_config=therapist_cue_config,
            llm_client=llm_client,
            df_all=df_all,
        )
        if path:
            txt_paths.append(path)

        if therapist_cue_config and therapist_cue_config.enabled:
            try:
                cue_path = generate_therapist_cues_report(
                    df, framework, output_dir, therapist_cue_config, llm_client,
                    df_all=df_all,
                )
                if cue_path:
                    txt_paths.append(cue_path)
            except Exception as _e:
                print(f"  Warning: cue response report failed: {_e}")
                if verbose:
                    traceback.print_exc()

        path = generate_longitudinal_text_report(
            df, participant_reports, framework, output_dir
        )
        if path:
            txt_paths.append(path)

        files_generated.extend(txt_paths)
        log(f"    {len(txt_paths)} text reports written.")
    except Exception as e:
        print(f"  Warning: text report generation failed: {e}")
        if verbose:
            traceback.print_exc()

    try:
        from process.output_index import write_index
        write_index(output_dir)
    except Exception:
        pass

    return {
        'output_dir': output_dir,
        'n_segments': n_segments,
        'n_participants': n_participants,
        'n_sessions': n_sessions,
        'files_generated': files_generated,
    }
