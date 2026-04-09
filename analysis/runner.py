"""
analysis/runner.py
------------------
Top-level orchestrator for the results analysis module.

Called by `python qra.py analyze --output-dir ...` and optionally by the
pipeline after Stage 7 when config.auto_analyze is True.
"""

import os
import traceback


def run_analysis(output_dir: str, verbose: bool = True) -> dict:
    """Execute the full results analysis on an existing pipeline output directory.

    Reads master_segment_dataset.csv and theme_definitions.json from output_dir.
    Writes all reports to output_dir/reports/analysis/.

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
    from .construct import generate_all_construct_reports
    from .graphing import export_all_graphing_datasets
    from .longitudinal import generate_longitudinal_summary

    def log(msg):
        if verbose:
            print(f"  {msg}")

    # ----------------------------------------------------------------
    # 1. Load data
    # ----------------------------------------------------------------
    log("[1/6] Loading segments...")
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
    base_analysis = os.path.join(output_dir, 'reports', 'analysis')
    for sub in ('participants', 'sessions', 'constructs', 'graphing'):
        os.makedirs(os.path.join(base_analysis, sub), exist_ok=True)

    files_generated = []

    # ----------------------------------------------------------------
    # 2. Per-session analyses
    # ----------------------------------------------------------------
    log(f"[2/6] Generating per-session analyses ({n_sessions} sessions)...")
    try:
        session_reports = generate_all_session_analyses(df, framework, output_dir)
        for r in session_reports:
            sid = r.get('session_id', '')
            files_generated.append(
                os.path.join(base_analysis, 'sessions', f'session_{sid}.json')
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
    log(f"[3/6] Generating per-participant reports ({n_participants} participants)...")
    try:
        participant_reports = generate_all_participant_reports(df, framework, output_dir)
        for r in participant_reports:
            pid = r.get('participant_id', '')
            files_generated.append(
                os.path.join(base_analysis, 'participants', f'participant_{pid}.json')
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
    log(f"[4/6] Generating per-construct reports ({n_stages} stages + codebook codes)...")
    try:
        generate_all_construct_reports(df, framework, output_dir)
        # Collect written files
        constructs_dir = os.path.join(base_analysis, 'constructs')
        if os.path.isdir(constructs_dir):
            for fname in os.listdir(constructs_dir):
                if fname.endswith('.json'):
                    files_generated.append(os.path.join(constructs_dir, fname))
        log(f"    Construct reports written to reports/analysis/constructs/.")
    except Exception as e:
        print(f"  Warning: construct reports failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 5. Graph-ready CSVs
    # ----------------------------------------------------------------
    log("[5/6] Exporting graph-ready datasets...")
    try:
        csv_paths = export_all_graphing_datasets(df, framework, output_dir)
        files_generated.extend(csv_paths)
        log(f"    {len(csv_paths)} CSV files written to reports/analysis/graphing/.")
    except Exception as e:
        print(f"  Warning: graphing exports failed: {e}")
        if verbose:
            traceback.print_exc()

    # ----------------------------------------------------------------
    # 6. Longitudinal summary
    # ----------------------------------------------------------------
    log("[6/6] Generating longitudinal summary...")
    try:
        generate_longitudinal_summary(df, participant_reports, framework, output_dir)
        files_generated.append(os.path.join(base_analysis, 'longitudinal_summary.json'))
        log("    Longitudinal summary written.")
    except Exception as e:
        print(f"  Warning: longitudinal summary failed: {e}")
        if verbose:
            traceback.print_exc()

    return {
        'output_dir': output_dir,
        'n_segments': n_segments,
        'n_participants': n_participants,
        'n_sessions': n_sessions,
        'files_generated': files_generated,
    }
