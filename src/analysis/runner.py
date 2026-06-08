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
from dataclasses import dataclass


@dataclass
class _AnalysisContext:
    """Best-effort pipeline config + LLM handles for analysis-time report features."""
    pipeline_config: object = None
    therapist_cue_config: object = None
    session_summaries_config: object = None
    participant_summaries_config: object = None
    llm_client: object = None
    analysis_plog: object = None


def _load_analysis_context(output_dir: str, llm_log_path, log) -> _AnalysisContext:
    """Load the pipeline config (if present) and build LLM-backed report handles.

    Best-effort: reads qra_config.json from the output dir (trying legacy
    locations), and constructs the summarization LLMClient only when therapist-
    cue / session / participant summaries are enabled. Missing or unreadable
    config degrades gracefully — the returned context leaves LLM features unset
    and report generation skips them.
    """
    from process import output_paths as _paths

    ctx = _AnalysisContext()

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
                ctx.pipeline_config = build_config_from_wizard_data(_raw)
            except Exception:
                pass
            break

    if ctx.pipeline_config is None:
        log("Note: no pipeline config found in output_dir — LLM-backed report summarization skipped.")
        return ctx

    _tc_cfg = ctx.pipeline_config.therapist_cues
    if _tc_cfg.enabled:
        ctx.therapist_cue_config = _tc_cfg
    ctx.session_summaries_config = getattr(ctx.pipeline_config, 'session_summaries', None)
    ctx.participant_summaries_config = getattr(ctx.pipeline_config, 'participant_summaries', None)

    _needs_llm = (
        _tc_cfg.enabled
        or (ctx.session_summaries_config and ctx.session_summaries_config.enabled)
        or (ctx.participant_summaries_config and ctx.participant_summaries_config.enabled)
    )
    if _needs_llm:
        try:
            from classification_tools.llm_client import LLMClient, LLMClientConfig
            _tc = ctx.pipeline_config.theme_classification
            _summ_model = getattr(_tc, 'summarization_model', 'nvidia/nemotron-3-nano-4b') or _tc.model
            if _tc and _summ_model:
                _llm_cfg = LLMClientConfig(
                    backend=_tc.backend,
                    api_key=_tc.api_key,
                    model=_summ_model,
                    models=[_summ_model],
                    temperature=_tc.temperature,
                    lmstudio_base_url=getattr(_tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
                )
                ctx.llm_client = LLMClient(_llm_cfg)
                if llm_log_path:
                    from process.process_logger import ProcessLogger
                    ctx.analysis_plog = ProcessLogger(llm_log_path=llm_log_path)
                    ctx.llm_client.config.process_logger = ctx.analysis_plog
        except Exception as _e:
            log(f"Warning: could not initialize LLM for analysis reports: {_e}")

    return ctx


def run_analysis(output_dir: str, verbose: bool = True, llm_log_path: str = None,
                 force_gnn: bool = None, force_classifier: bool = None) -> dict:
    """Execute the full results analysis on an existing pipeline output directory.

    Reads master_segments.{jsonl,csv} and theme_definitions.json from output_dir.
    Writes text reports to output_dir/06_reports/, graphing data to
    output_dir/03_analysis_data/, and figures to output_dir/05_figures/.

    Parameters
    ----------
    output_dir : str
        Pipeline output directory (must contain master_segments.csv
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
    from .theme import generate_all_theme_reports, generate_codebook_text_report
    from .figure_data import export_all_graphing_datasets
    from .longitudinal import generate_longitudinal_summary
    from .stage_progression import compute_session_stage_progression
    from .reports import (
        generate_comprehensive_session_report,
        generate_all_stage_text_reports,
        generate_transition_explanation,
        generate_therapist_cues_report,
        generate_longitudinal_text_report,
        generate_all_session_txt_reports,
        generate_all_participant_txt_reports,
        generate_session_summaries,
    )
    from process import output_paths as _paths

    def log(msg):
        if verbose:
            print(f"  {msg}")

    # ----------------------------------------------------------------
    # 0. Load pipeline config for LLM-backed features (best-effort)
    # ----------------------------------------------------------------
    _ctx = _load_analysis_context(output_dir, llm_log_path, log)
    _pipeline_config = _ctx.pipeline_config
    therapist_cue_config = _ctx.therapist_cue_config
    session_summaries_config = _ctx.session_summaries_config
    participant_summaries_config = _ctx.participant_summaries_config
    llm_client = _ctx.llm_client
    _analysis_plog = _ctx.analysis_plog

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

    # Attach VAAMR stage-mixture (superposition) columns so every downstream
    # report can surface the blend instead of a hard argmax. Additive + graceful
    # (GNN positions → LLM ballots → secondary_stage); never mutates final_label.
    _superpos_cfg = getattr(_pipeline_config, 'superposition', None) if _pipeline_config is not None else None
    if _superpos_cfg is None or getattr(_superpos_cfg, 'enabled', True):
        try:
            from .superposition import attach_superposition
            _nstg = len(framework) if framework else 5
            attach_superposition(df, output_dir, config=_superpos_cfg, n_stages=_nstg)
            if df_all is not None:
                attach_superposition(df_all, output_dir, config=_superpos_cfg, n_stages=_nstg)
            log(f"    Superposition attached (source: {df['mixture_source'].mode().iloc[0] if len(df) else 'n/a'}).")
        except Exception as e:
            print(f"  Warning: superposition attach failed: {e}")
            if verbose:
                traceback.print_exc()

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
        _paths.themes_json_dir(output_dir),
        _paths.themes_dir(output_dir),
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
    log(f"[4/8] Generating per-theme reports ({n_stages} stages + codebook codes)...")
    stage_reports = []
    try:
        stage_reports = generate_all_theme_reports(df, framework, output_dir) or []
        # Collect written files
        _cjdir = _paths.themes_json_dir(output_dir)
        if os.path.isdir(_cjdir):
            for fname in os.listdir(_cjdir):
                if fname.endswith('.json'):
                    files_generated.append(os.path.join(_cjdir, fname))
        log(f"    Theme reports written to 03_analysis_data/per_theme/.")
    except Exception as e:
        print(f"  Warning: theme reports failed: {e}")
        if verbose:
            traceback.print_exc()

    try:
        ref_path = generate_codebook_text_report(df, framework, output_dir)
        if ref_path:
            files_generated.append(ref_path)
            log("    Codebook exemplars: 06_reports/05_per_stage/codebook_exemplars.txt")
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
        log(f"    {len(csv_paths)} CSV files written to 03_analysis_data/graphing/.")
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
        from .figures import generate_all_figures, generate_all_session_stage_timelines
        fig_paths = generate_all_figures(df, framework, output_dir)
        files_generated.extend(fig_paths)
        timeline_paths = generate_all_session_stage_timelines(df, framework, output_dir)
        files_generated.extend(timeline_paths)
        log(f"    {len(fig_paths) + len(timeline_paths)} figures written ({len(timeline_paths)} session timelines).")
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

        # Session summaries (computed once, stored as JSON + txt, reused below)
        session_summaries = {}
        if session_summaries_config and session_summaries_config.enabled:
            try:
                session_summaries = generate_session_summaries(
                    df_all if df_all is not None else df,
                    [r['session_id'] for r in session_reports if r.get('session_id')],
                    output_dir,
                    llm_client=llm_client,
                    max_words=session_summaries_config.max_words_per_session,
                )
                from process import output_paths as _paths2
                txt_paths.append(_paths2.session_summaries_json_path(output_dir))
                txt_paths.append(os.path.join(_paths2.reports_per_session_dir(output_dir), 'session_summaries.txt'))
                log(f"    Session summaries generated for {len(session_summaries)} sessions.")
            except Exception as _e:
                print(f"  Warning: session summaries generation failed: {_e}")
                if verbose:
                    traceback.print_exc()

        # Per-session .txt files
        session_txt_paths = generate_all_session_txt_reports(
            df, session_reports, framework, output_dir,
            llm_client=llm_client,
            therapist_cue_config=therapist_cue_config,
            df_all=df_all,
            session_summaries=session_summaries,
        )
        txt_paths.extend(session_txt_paths)

        # Per-participant .txt files
        participant_txt_paths = generate_all_participant_txt_reports(
            df, participant_reports, framework, output_dir,
            llm_client=llm_client,
            participant_summaries_config=participant_summaries_config,
        )
        txt_paths.extend(participant_txt_paths)

        files_generated.extend(txt_paths)
        log(f"    {len(txt_paths)} text reports written.")
    except Exception as e:
        print(f"  Warning: text report generation failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 10. PURER × VAMMR influence analysis (when PURER labels exist)
    # ----------------------------------------------------------------
    _has_purer = (
        df_all is not None
        and 'purer_primary' in df_all.columns
        and df_all['purer_primary'].notna().any()
    )
    if df_all is not None:
        try:
            from .purer_analysis import run_purer_analysis
            from .purer_figures import generate_purer_figures
            log("[10/8] Running PURER × VAMMR cue-block influence analysis...")
            purer_result = run_purer_analysis(df_all, output_dir, framework=framework)
            files_generated.extend(purer_result.get('files_written', []))
            n_blocks  = purer_result.get('n_cue_blocks', 0)
            n_empty   = purer_result.get('n_empty', 0)
            n_labeled = purer_result.get('purer_labeled', 0)

            if n_labeled > 0:
                log(
                    f"    {n_blocks} cue blocks: {n_empty} empty, "
                    f"{n_labeled} PURER-labeled."
                )
                try:
                    fig_paths = generate_purer_figures(
                        purer_result['influence'], framework, output_dir
                    )
                    files_generated.extend(fig_paths)
                    log(f"    PURER figures: {[os.path.basename(p) for p in fig_paths]}")
                except Exception as _fe:
                    print(f"  Warning: PURER figures failed: {_fe}")
                    if verbose:
                        traceback.print_exc()
            else:
                log(
                    f"    {n_blocks} cue blocks built ({n_empty} empty). "
                    "No PURER labels yet — re-run pipeline with run_purer_labeler=True."
                )
        except Exception as e:
            print(f"  Warning: PURER analysis failed: {e}")
            if verbose:
                traceback.print_exc()

    # ----------------------------------------------------------------
    # 11. GNN representation-and-discovery layer (optional; OFF by default)
    # ----------------------------------------------------------------
    _gnn_cfg = getattr(_pipeline_config, 'gnn_layer', None) if _pipeline_config is not None else None
    _gnn_enabled = getattr(_gnn_cfg, 'enabled', False) if _gnn_cfg is not None else False
    if force_gnn is not None:
        _gnn_enabled = force_gnn
        # If config never loaded but the user forced --gnn, fall back to defaults.
        if _gnn_cfg is None and force_gnn:
            from gnn_layer.config import GnnLayerConfig
            _gnn_cfg = GnnLayerConfig(enabled=True)
    # `qra gnn train` opts into the (default-OFF, H5-refuted) consensus-distillation classifier.
    if force_classifier is not None and _gnn_cfg is not None:
        try:
            _gnn_cfg.gnn_classifier_enabled = bool(force_classifier)
        except Exception:
            pass
    _gnn_ran = False
    if df_all is not None and _gnn_cfg is not None and _gnn_enabled:
        try:
            from gnn_layer.runner import run_gnn_analysis
            # Multi-run-ballot pre-flight: the GNN's consensus targets come from
            # per-segment rater ballots; a single-run LLM project trains on one-hot
            # labels and the gate κ is unreliable. Warn (non-blocking) here; the
            # dedicated `qra gnn train` path can hard-stop instead.
            try:
                from gnn_layer.soft_labels import ballot_coverage
                _cov = ballot_coverage(df_all)
                if _cov['n_participant'] and _cov['multirun_fraction'] < 0.5:
                    log(f"[11/8] GNN: only {round(100 * _cov['multirun_fraction'])}% of "
                        "participant segments carry multi-run LLM ballots — weak training "
                        "signal; the reliability gate κ may be optimistic/unreliable "
                        "(re-run VAAMR with n_runs >= 3 for a trustworthy gate).")
            except Exception:
                pass
            log("[11/8] Running GNN representation-and-discovery layer...")
            gnn_result = run_gnn_analysis(
                df_all, output_dir, framework=framework, config=_gnn_cfg, llm_client=llm_client,
            )
            files_generated.extend(gnn_result.get('files_written', []))
            log(f"    GNN layer status: {gnn_result.get('status', 'ok')}")
            _gnn_ran = gnn_result.get('status', '') != 'error'
        except Exception as e:
            print(f"  Warning: GNN layer failed: {e}")
            if verbose:
                traceback.print_exc()

    # Append GNN motif cross-reference section to the PURER report (after GNN
    # writes cue_motifs.csv / coupling_factors.csv; no-op if files absent).
    if _gnn_ran:
        try:
            from .purer_analysis import append_gnn_motif_section
            gnn_motif_section = append_gnn_motif_section(output_dir)
            if gnn_motif_section is not None:
                log("    GNN motif section appended to 06_reports/02_mechanism/purer.txt.")
        except Exception as e:
            print(f"  Warning: GNN motif section append failed: {e}")
            if verbose:
                traceback.print_exc()

    # ----------------------------------------------------------------
    # 12. Superposition surfacing + mechanistic analysis (additive)
    # ----------------------------------------------------------------
    _run_mech = getattr(_superpos_cfg, 'run_mechanism_analysis', True) if _superpos_cfg is not None else True
    if 'mixture' in df.columns:
        # Soft (expected-count) VAAMR × codebook lift — boundary-expressed codes.
        try:
            from process.cross_validation import (
                compute_soft_theme_codebook_cooccurrence,
                export_soft_cross_validation_results,
            )
            log("[12/8] Computing soft (mixture-weighted) cross-validation lift...")
            soft_cooc = compute_soft_theme_codebook_cooccurrence(df, framework)
            if soft_cooc:
                soft_path = export_soft_cross_validation_results(soft_cooc, output_dir)
                if soft_path:
                    files_generated.append(soft_path)
        except Exception as e:
            print(f"  Warning: soft cross-validation failed: {e}")
            if verbose:
                traceback.print_exc()

        # Superposition human-readable report + figures.
        try:
            from .reports.superposition_report import generate_superposition_report
            sp_path = generate_superposition_report(df, framework, output_dir)
            if sp_path:
                files_generated.append(sp_path)
                log("    Superposition report: 06_reports/02_mechanism/superposition.txt")
        except Exception as e:
            print(f"  Warning: superposition report failed: {e}")
            if verbose:
                traceback.print_exc()

        try:
            from .figures import generate_superposition_figures
            sp_figs = generate_superposition_figures(df, framework, output_dir)
            files_generated.extend(sp_figs)
        except Exception as e:
            print(f"  Warning: superposition figures failed: {e}")
            if verbose:
                traceback.print_exc()

        # Program-efficacy dossier (does it work; links to external outcomes).
        _eff_cfg = getattr(_pipeline_config, 'efficacy', None) if _pipeline_config is not None else None
        if _eff_cfg is None or getattr(_eff_cfg, 'enabled', True):
            try:
                from .efficacy import run_efficacy_analysis
                log("    Running program-efficacy analysis...")
                eff_result = run_efficacy_analysis(df, framework, output_dir, config=_eff_cfg)
                files_generated.extend(eff_result.get('files_written', []))
                log("    Progression summary: 06_reports/01_outcomes/progression_summary.txt")
            except Exception as e:
                print(f"  Warning: efficacy analysis failed: {e}")
                if verbose:
                    traceback.print_exc()

        # Mechanistic FROM→CUE→TO analysis (continuous Δprogression).
        if _run_mech and df_all is not None and 'mixture' in df_all.columns:
            try:
                from .mechanism import run_mechanism_analysis
                log("    Running mechanistic Δprogression analysis...")
                _mech_cfg = getattr(_pipeline_config, 'mechanism', None) if _pipeline_config is not None else None
                mech_result = run_mechanism_analysis(df, df_all, output_dir, framework, config=_mech_cfg)
                files_generated.extend(mech_result.get('files_written', []))
                log(f"    Mechanism: {mech_result.get('n_blocks', 0)} cue blocks analyzed.")
            except Exception as e:
                print(f"  Warning: mechanism analysis failed: {e}")
                if verbose:
                    traceback.print_exc()

            # Therapeutic language atlas (readable patterns behind the statistics).
            try:
                from .reports.language_atlas import generate_language_atlas
                atlas_path = generate_language_atlas(df, df_all, framework, output_dir)
                if atlas_path:
                    files_generated.append(atlas_path)
                    log("    Language atlas: 06_reports/02_mechanism/language_atlas.txt")
            except Exception as e:
                print(f"  Warning: language atlas failed: {e}")
                if verbose:
                    traceback.print_exc()

    # ----------------------------------------------------------------
    # 12b. MindfulBERT training-set builder (Track C; the end-goal artifact)
    #      Runs after mechanism (observed Δprogression) + GNN (counterfactual
    #      augmentation channel). OFF by default; augmentation is gate-gated.
    # ----------------------------------------------------------------
    if df_all is not None and _gnn_cfg is not None and getattr(_gnn_cfg, 'build_mindfulbert_dataset', False):
        try:
            from process.assembly import build_mindfulbert_dataset
            log("[12b/8] Building MindfulBERT training set (observed Δprogression)...")
            mb_result = build_mindfulbert_dataset(df_all, output_dir, config=_gnn_cfg)
            files_generated.extend(mb_result.get('files_written', []))
            log(f"    MindfulBERT dataset: {mb_result.get('n_examples', 0)} examples "
                f"(gate_passed={mb_result.get('gate_passed')}).")
        except Exception as e:
            print(f"  Warning: MindfulBERT dataset build failed: {e}")
            if verbose:
                traceback.print_exc()

    # ----------------------------------------------------------------
    # 12b. Inter-rater reliability — regenerate IFF the project has imported
    #      human codes AND the machine state (LLM/GNN/held-out) has changed,
    #      so IRR always reflects the current models for this validation pass.
    # ----------------------------------------------------------------
    try:
        from . import irr_analysis as _irr
        _irr_res = _irr.maybe_run_irr(output_dir, _pipeline_config, verbose=verbose)
        if _irr_res == 'unchanged':
            log("    IRR: human codes present, machine state unchanged — kept existing report.")
        elif _irr_res is not None:
            files_generated.append(_paths.reports_irr_path(output_dir))
            _drift = _irr_res.get('testset_drift') or []
            if _drift:
                log(f"    ‼ IRR: {len(_drift)} test-set item(s) DRIFTED from the human-coded "
                    "text — see the report header / run `qra irr run` to inspect.")
            log(f"    IRR regenerated (GNN axis: {_irr_res.get('gnn_axis')}): "
                "06_reports/06b_irr_report.txt")
    except Exception as e:
        print(f"  ‼ IRR analysis FAILED (non-fatal — analysis continues): {e}")
        traceback.print_exc()

    # ----------------------------------------------------------------
    # 13. Top-level synthesis: executive summary, methods appendix, READ_ME.
    #     Written LAST so they can read every artifact the run produced.
    # ----------------------------------------------------------------
    log("[13/8] Writing executive summary, methods appendix, and reports READ_ME...")
    try:
        from .reports.executive_summary import generate_executive_summary
        es_path = generate_executive_summary(output_dir, df, framework, df_all=df_all)
        if es_path:
            files_generated.append(es_path)
            log("    Executive summary: 06_reports/00_executive_summary.txt")
    except Exception as e:
        print(f"  Warning: executive summary failed: {e}")
        if verbose:
            traceback.print_exc()
    try:
        from .reports.reports_guide import generate_methods_appendix, generate_reports_readme
        ma_path = generate_methods_appendix(output_dir)
        if ma_path:
            files_generated.append(ma_path)
        rm_path = generate_reports_readme(output_dir)
        if rm_path:
            files_generated.append(rm_path)
    except Exception as e:
        print(f"  Warning: reports guide failed: {e}")
        if verbose:
            traceback.print_exc()

    try:
        from process.output_index import write_index
        write_index(output_dir)
    except Exception:
        pass

    if _analysis_plog is not None:
        _analysis_plog.close_llm_log()

    return {
        'output_dir': output_dir,
        'n_segments': n_segments,
        'n_participants': n_participants,
        'n_sessions': n_sessions,
        'files_generated': files_generated,
    }
