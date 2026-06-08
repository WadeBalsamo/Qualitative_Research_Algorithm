"""
process/interactive_tui.py
--------------------------
Interactive TUI launched when `python qra.py` is run without arguments.

Two top-level paths:
  [1] New Project   — runs the setup wizard then offers immediate pipeline execution
  [2] Open Project  — detects project state and presents a contextual action menu
                      covering all modular pipeline stages, zero-shot options,
                      testset management, content validity, and analysis

Uses only stdlib (no curses / rich / textual).
"""
import json
import os
import sys
from typing import Optional

# Width used for decorative separators
_W = 72


# ---------------------------------------------------------------------------
# Low-level display helpers
# ---------------------------------------------------------------------------

def _hr(char: str = '─') -> None:
    print(char * _W)


def _banner() -> None:
    print()
    _hr('═')
    print('  QRA — Qualitative Research Algorithm')
    print('  Computational Phenomenology Pipeline')
    print('  VAAMR × PURER × VCE Codebook Frameworks')
    _hr('═')
    print()


def _section(title: str) -> None:
    print()
    _hr()
    print(f'  {title}')
    _hr()
    print()


def _info(text: str) -> None:
    for line in text.splitlines():
        print(f'  {line}')


def _ok(text: str) -> None:
    print(f'  ✓ {text}')


def _warn(text: str) -> None:
    print(f'  ! {text}')


def _err(text: str) -> None:
    print(f'  ✗ {text}')


def _pause() -> None:
    input('\n  Press Enter to continue...')


def _ask(prompt: str, default: str = '') -> str:
    display = f'  {prompt} [{default}]: ' if default else f'  {prompt}: '
    raw = input(display).strip()
    return raw if raw else default


def _confirm(prompt: str = 'Proceed?', default: bool = True) -> bool:
    hint = 'Y/n' if default else 'y/N'
    raw = input(f'  {prompt} [{hint}]: ').strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


def _menu(title: str, options: list, back_label: str = 'Back') -> int:
    """
    Print a numbered menu and return the 1-based selection, or 0 for back/quit.
    options: list of (label, description) tuples.
    """
    print()
    _hr()
    print(f'  {title}')
    _hr()
    print()
    for i, (label, desc) in enumerate(options, 1):
        print(f'  [{i}] {label}')
        if desc:
            for line in desc.splitlines():
                print(f'       {line}')
        print()
    print(f'  [0] {back_label}')
    print()
    while True:
        raw = input('  Choice: ').strip()
        if raw == '0':
            return 0
        try:
            n = int(raw)
            if 1 <= n <= len(options):
                return n
        except ValueError:
            pass
        print(f'  Please enter a number between 0 and {len(options)}.')


# ---------------------------------------------------------------------------
# Project state detection
# ---------------------------------------------------------------------------

def _detect_state(output_dir: str) -> dict:
    """
    Return a dict describing what exists in `output_dir`:
      is_legacy        bool  master_segments.jsonl but no 01_transcripts/segmented/
      has_segments     bool  frozen per-session segments exist (SQLite-backed)
      has_theme        bool  theme overlay exists in qra.db (SQLite-backed)
      has_purer        bool  purer overlay exists in qra.db (SQLite-backed)
      has_codebook     bool  codebook overlay exists in qra.db (SQLite-backed)
      has_master       bool  master_segments*.csv exists
      has_testsets     list  names of frozen testsets in 04_validation/testsets/
      has_cv_testsets  list  names of content-validity testsets
      has_analysis     bool  03_analysis_data/ non-empty
      config_path      str|None  path to qra_config.json if found
    """
    from . import output_paths as _paths
    from . import segments_io
    from . import classifications_io
    from .legacy_migration import is_legacy_project
    import glob

    def _dir_non_empty(p: str) -> bool:
        return os.path.isdir(p) and bool(os.listdir(p))

    has_segments = bool(segments_io.list_segmented_sessions(output_dir))
    has_theme = classifications_io.overlay_exists(output_dir, 'theme')
    has_purer = classifications_io.overlay_exists(output_dir, 'purer')
    has_codebook = classifications_io.overlay_exists(output_dir, 'codebook')

    ms_dir = _paths.master_segments_dir(output_dir)
    has_master = bool(glob.glob(os.path.join(ms_dir, 'master_segments*.csv')))

    # Testsets — flat numbered worksheets
    import re as _re
    ts_dir = _paths.testsets_dir(output_dir)
    has_testsets: list = []
    if os.path.isdir(ts_dir):
        _ts_pattern = _re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
        for fname in sorted(os.listdir(ts_dir)):
            m = _ts_pattern.match(fname)
            if m:
                has_testsets.append(int(m.group(1)))

    # Content-validity testsets
    has_cv_testsets: list = []
    try:
        from .assembly.content_validity import list_content_validity_testsets
        has_cv_testsets = sorted(
            t['name'] for t in list_content_validity_testsets(output_dir)
        )
    except Exception:
        has_cv_testsets = []

    has_analysis = _dir_non_empty(_paths.analysis_data_dir(output_dir))
    gnn_status = _gnn_status(output_dir)
    probe_status = _probe_status(output_dir)

    # Look for config
    config_path = None
    candidates = [
        _paths.meta_dir(output_dir) + '/qra_config.json',
        os.path.join(output_dir, 'qra_config.json'),
    ]
    for c in candidates:
        if os.path.isfile(c):
            config_path = c
            break

    return dict(
        is_legacy=is_legacy_project(output_dir),
        has_segments=has_segments,
        has_theme=has_theme,
        has_purer=has_purer,
        has_codebook=has_codebook,
        has_master=has_master,
        has_testsets=has_testsets,
        has_cv_testsets=has_cv_testsets,
        has_analysis=has_analysis,
        gnn_status=gnn_status,
        probe_status=probe_status,
        config_path=config_path,
    )


def _gnn_status(output_dir: str) -> str:
    """Reliability state of the GNN consensus layer.

    'ready'     — passed its out-of-sample gate; LLM-free scaling recommended
    'not_ready' — trained + gate run, but not yet reliable enough
    'trained'   — checkpoint exists but no validation report yet
    'absent'    — GNN layer has not been run

    Prefers the machine-readable gate verdict (gnn_gate.json); falls back to the
    human validation.txt report and finally the trained-weights checkpoint.
    """
    from . import output_paths as _paths
    try:
        from gnn_layer.classifier.validation import read_gate_verdict
        verdict = read_gate_verdict(output_dir)
        if verdict is not None:
            return 'ready' if verdict.get('ready_for_scaling') else 'not_ready'
    except Exception:
        pass
    rep = os.path.join(_paths.reports_gnn_dir(output_dir), 'validation.txt')
    if os.path.isfile(rep):
        try:
            with open(rep, encoding='utf-8') as fh:
                for line in fh:
                    if 'LLM-FREE SCALING?' in line:
                        return 'ready' if 'YES' in line else 'not_ready'
        except OSError:
            pass
        return 'trained'
    weights = os.path.join(_paths.gnn_model_dir(output_dir), 'weights.pt')
    if os.path.isfile(weights):
        return 'trained'
    # Discovery + mechanism layer (default ON; the GraphSAGE classifier is default OFF). These
    # reports exist when the layer has run even though no classifier gate was produced.
    _rep = _paths.reports_gnn_dir(output_dir)
    if any(os.path.isfile(os.path.join(_rep, f)) for f in
           ('discriminant_validity.txt', 'transition_model.txt', 'communities.txt')):
        return 'discovery'
    return 'absent'


def _print_project_state(state: dict) -> None:
    print()
    _info('Project state:')
    print()

    def _flag(val, label):
        mark = '✓' if val else '·'
        print(f'    {mark}  {label}')

    _flag(state['is_legacy'],    'Legacy layout (pre-modular; migration needed)')
    _flag(state['has_segments'], 'Frozen per-session segments')
    _flag(state['has_theme'],    'VAAMR theme classification overlay')
    _flag(state['has_purer'],    'PURER classification overlay')
    _flag(state['has_codebook'], 'VCE codebook classification overlay')
    _flag(state['has_master'],   'Assembled master_segments dataset')
    _flag(bool(state['has_testsets']),    f'Frozen validation testsets ({len(state["has_testsets"])})')
    _flag(bool(state['has_cv_testsets']), f'Content-validity testsets ({len(state["has_cv_testsets"])})')
    _flag(state['has_analysis'], 'Analysis data / reports')
    _gnn_state_labels = {
        'ready': '✓  GNN consensus graph — ready for LLM-free scaling',
        'not_ready': '·  GNN consensus graph — trained, gate not yet reliable',
        'trained': '·  GNN consensus graph — trained, gate not run',
        'discovery': '✓  GNN discovery + mechanism ran (classifier OFF by default)',
        'absent': '·  GNN layer — not run',
    }
    print(f'    {_gnn_state_labels.get(state.get("gnn_status", "absent"))}')
    if state['config_path']:
        print(f'       Config: {state["config_path"]}')
    print()


# ---------------------------------------------------------------------------
# Config loading helpers
# ---------------------------------------------------------------------------

def _load_config(config_path: str):
    """Load PipelineConfig from a saved qra_config.json."""
    from .config import PipelineConfig
    import json as _json

    # Reuse the flattener from qra.py
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import qra as _qra
    with open(config_path) as fh:
        raw = _json.load(fh)
    return PipelineConfig.from_json(_qra._flatten_wizard_config(raw))


def _load_framework_default():
    from constructs.registry import load as _registry_load_fw
    return _registry_load_fw('vaamr')


# ---------------------------------------------------------------------------
# Action helpers — each wraps one pipeline stage with UI framing
# ---------------------------------------------------------------------------

def _action_ingest(config, output_dir: str) -> None:
    from .orchestrator import stage_ingest, SilentObserver
    _section('Stage 1 — Ingest & Freeze Segments')
    _info(
        'Loads transcript files, runs semantic segmentation, and writes frozen\n'
        'per-session segments to the project database (qra.db).\n'
        '\n'
        'Legacy JSONL projects are auto-migrated into qra.db on first ingest.\n'
        '\n'
        'Frozen segments are NEVER re-written by classify/assemble stages.\n'
        'Choose "re-segment from scratch" below to rebuild them deliberately.'
    )
    print()
    from . import segments_io as _sio
    fresh = False
    if _sio.list_segmented_sessions(output_dir):
        fresh = _confirm('Re-segment ALL sessions from scratch? (rebuilds frozen segments)',
                         default=False)
    if not _confirm('Run ingest now?'):
        return
    segments = stage_ingest(config, output_dir=output_dir, observer=SilentObserver(),
                            force_reingest_all=fresh)
    n_sess = len(set(s.session_id for s in segments))
    print()
    _ok(f'{len(segments)} segments across {n_sess} sessions ingested and frozen.')
    _pause()


def _action_classify_theme(config, output_dir: str, framework) -> None:
    from .orchestrator import stage_classify_theme
    from . import classifications_io as _cio
    from . import reclassify_ops as _reclassify
    _section('Stage 3 — VAAMR Theme Classification')
    _info(
        'Runs zero-shot LLM classification on participant segments using the\n'
        'VAAMR framework (Vigilance → Avoidance → Attention → Metacognition\n'
        '→ Reappraisal).  Multi-run ensemble voting produces confidence tiers.\n'
        '\n'
        'Writes: the theme overlay in qra.db\n'
        'Reads:  frozen segments (qra.db)\n'
        'Frozen segments and testsets are NOT affected by this stage.\n'
        '\n'
        'ZERO-SHOT MODE: strips exemplar/subtle/adversarial utterances from\n'
        'the prompt so the model classifies from definitions alone — useful for\n'
        'evaluating framework validity without example anchoring.'
    )
    print()

    # If VAAMR was already classified, offer to start over from scratch (mirrors
    # the PURER "re-run all" option) before falling through to a normal run.
    if _cio.overlay_exists(output_dir, 'theme'):
        _info('VAAMR classification already exists for this project.')
        print()
        opts = [
            ('Re-run (resume from checkpoints)',
             'Continue from existing LLM run checkpoints where possible.'),
            ('Re-run FROM SCRATCH',
             'Clear VAAMR checkpoints + the theme overlay first, then classify anew.'),
        ]
        sub = _menu('VAAMR Reclassify Options', opts)
        if sub == 0:
            return
        if sub == 2:
            if not _confirm('Clear VAAMR checkpoints + overlay and re-run from scratch?'):
                return
            r = _reclassify.reset_for_fresh(output_dir, 'vaamr')
            _ok(f"Cleared {r['checkpoints_removed']} checkpoint(s) + the theme overlay.")
            config.resume_from = None

    # Zero-shot toggle
    zero_shot = _confirm(
        'Use zero-shot prompting? (definitions only, no examples)', default=False
    )
    if zero_shot:
        config.theme_classification.zero_shot_prompt = True
        _ok('Zero-shot mode ON — exemplars will be excluded from prompts.')
    else:
        config.theme_classification.zero_shot_prompt = False
        _ok('Standard mode — exemplars included in prompts.')

    print()
    _info(f'Model:   {config.theme_classification.model}')
    _info(f'Runs:    {config.theme_classification.n_runs}')
    if config.theme_classification.per_run_models:
        for i, m in enumerate(config.theme_classification.per_run_models):
            tag = ' (primary)' if i == 0 else f' (checker {i})'
            _info(f'  Run {i+1}: {m}{tag}')
    print()

    if not _confirm('Run VAAMR classification now?'):
        return
    stage_classify_theme(config, framework, output_dir=output_dir)
    print()
    _ok('Theme overlay written to qra.db. Run Assemble to rebuild master_segments.')
    _pause()


def _gnn_tag(status: str) -> str:
    """Menu suffix reflecting the GNN reliability gate."""
    return {
        'ready': '  ★ ready (recommended)',
        'not_ready': '  (gate: not yet reliable)',
        'trained': '  (trained — gate not run)',
        'discovery': '  (discovery ran; run "gnn train" for the classifier)',
        'absent': '  (needs GNN layer first)',
    }.get(status, '')


def _probe_status(output_dir: str) -> str:
    """Reliability state of the LLM-free probe scaler (methodology §8.6).

    'ready'     — probe↔human κ reached the human band on this project's human subset
    'not_ready' — gate ran but the probe is not yet reliable enough to scale
    'absent'    — the probe has not been trained/gated yet

    Reads the machine-readable verdict at 03_analysis_data/probe/probe_gate.json.
    """
    try:
        from classification_tools.probe.probe_classifier import read_probe_gate
        verdict = read_probe_gate(output_dir)
        if verdict is not None:
            return 'ready' if verdict.get('ready_for_scaling') else 'not_ready'
    except Exception:
        pass
    return 'absent'


def _probe_tag(status: str) -> str:
    """Menu suffix reflecting the probe reliability gate."""
    return {
        'ready': '  ★ gate passed (recommended for bulk/new data)',
        'not_ready': '  (gate: below the human band — assistive only)',
        'absent': '  (run "Probe — train + gate" first)',
    }.get(status, '')


def _action_classify_gnn(config, output_dir: str, framework, state: dict) -> None:
    """LLM-free classification of new segments using the trained graph (scale mode)."""
    import pandas as _pd
    from . import segments_io as _sio
    from . import classifications_io as _cio
    from gnn_layer.runner import run_gnn_classify

    _section('Classify — Graph consensus (LLM-free)')
    status = state.get('gnn_status', 'absent')
    if status == 'absent':
        _warn('No trained graph yet. Run "Analysis & Reports" with the GNN layer enabled')
        _warn('(set gnn_layer.enabled in config, or run: qra analyze --gnn), then return.')
        _pause()
        return
    if status != 'ready':
        _warn('The graph has NOT passed its reliability gate yet')
        _warn('(see 06_reports/06_gnn/validation.txt). Labels may be unreliable.')
        if not _confirm('Classify with the graph anyway?', default=False):
            return
    else:
        _ok('Graph passed its reliability gate — safe to label new data without LLMs.')

    _info('Labels only segments that lack an LLM label; raw LLM ballots are untouched.')
    print()

    segments = _sio.load_segments_for_stage(output_dir, apply=())
    by_id = {s.segment_id: s for s in segments}
    _cio.apply_overlays(output_dir, by_id,
                        keys=('theme', 'purer', 'codebook', 'cv', 'gnn'))
    df_all = _pd.DataFrame([{
        'segment_id': s.segment_id, 'text': s.text, 'speaker': s.speaker,
        'session_id': s.session_id,
        'start_time_ms': s.start_time_ms, 'end_time_ms': s.end_time_ms,
        'final_label': s.final_label, 'primary_stage': s.primary_stage,
        'purer_primary': getattr(s, 'purer_primary', None),
    } for s in segments])

    res = run_gnn_classify(df_all, output_dir, framework=framework,
                           config=config.gnn_layer, verbose=True)
    _ok(f"Status: {res['status']}; classified {res.get('n_classified', 0)} segment(s) "
        "with no LLM calls.")
    _info('Run "Assemble master dataset" to fold the new labels into master_segments.')
    _pause()


def _action_classify_purer(config, output_dir: str) -> None:
    import glob as _glob
    from . import output_paths as _paths
    from .orchestrator import stage_classify_purer
    _section('Stage 3c — PURER Therapist Classification')
    _info(
        'Classifies therapist cue-blocks between consecutive participant turns\n'
        'using the PURER framework (Phenomenological → Utilization → Reframing\n'
        '→ Educate/Expectancy → Reinforcement).\n'
        '\n'
        'Classification is at the cue-block level; labels are propagated back\n'
        'to all constituent therapist segments after classification.\n'
        '\n'
        'Writes: the PURER overlay in qra.db\n'
        'Participant VAAMR labels and frozen segments are NOT modified.'
    )
    print()

    purer_model = (getattr(config.purer_classification, 'model', None)
                   or config.theme_classification.model)
    n_runs = getattr(config.purer_classification, 'n_runs', 1)
    per_run_models = list(getattr(config.purer_classification, 'per_run_models', []) or [])

    _info(f'Model:   {purer_model}')
    _info(f'Runs:    {n_runs}')
    if per_run_models:
        for i, m in enumerate(per_run_models):
            tag = ' (primary)' if i == 0 else f' (checker {i})'
            _info(f'  Run {i+1}: {m}{tag}')
    print()

    config_path = os.path.join(_paths.meta_dir(output_dir), 'qra_config.json')

    def _save_purer_to_disk():
        if not os.path.isfile(config_path):
            return
        try:
            with open(config_path) as _f:
                _raw = json.load(_f)
            pc = _raw.setdefault('purer_classification', {})
            pc['model'] = config.purer_classification.model
            pc['n_runs'] = config.purer_classification.n_runs
            pc['per_run_models'] = list(
                getattr(config.purer_classification, 'per_run_models', []) or []
            )
            with open(config_path, 'w') as _f:
                json.dump(_raw, _f, indent=2)
            _ok('Config updated on disk.')
        except Exception as _e:
            _warn(f'Could not save config: {_e}')

    def _delete_purer_checkpoints():
        ckpt_dir = _paths.llm_checkpoints_dir(output_dir)
        removed = 0
        for _f in _glob.glob(os.path.join(ckpt_dir, 'purer_cue_results_*')):
            os.remove(_f)
            removed += 1
        if removed:
            _ok(f'Removed {removed} PURER checkpoint file(s).')

    from . import classifications_io as _cio
    already_done = _cio.overlay_exists(output_dir, 'purer')

    if already_done:
        _info('PURER classification already exists for this project.')
        print()
        opts = [
            ('Re-run all', 'Full reclassification from scratch with the same models.'),
            ('Re-run all with new model', 'Change the PURER model and re-run everything.'),
            ('Add a new run', 'Append one more run with a different checker model.'),
            ('Reclassify a specific run', 'Re-run just one run, optionally with a new model.'),
        ]
        sub = _menu('PURER Reclassify Options', opts)
        if sub == 0:
            return

        if sub == 1:
            if not _confirm('Delete PURER checkpoints and re-run all classification now?'):
                return
            _delete_purer_checkpoints()
            config.resume_from = None
            stage_classify_purer(config, output_dir=output_dir)
            print()
            _ok('PURER overlay written to qra.db. Run Assemble to rebuild master_segments.')
            _pause()
            return

        elif sub == 2:
            new_model = _ask('New PURER primary model ID', purer_model)
            if not new_model:
                return
            config.purer_classification.model = new_model
            if per_run_models:
                config.purer_classification.per_run_models = [new_model] + per_run_models[1:]
            _save_purer_to_disk()
            if not _confirm('Delete PURER checkpoints and re-run with new model now?'):
                return
            _delete_purer_checkpoints()
            config.resume_from = None
            stage_classify_purer(config, output_dir=output_dir)
            print()
            _ok('PURER overlay written to qra.db. Run Assemble to rebuild master_segments.')
            _pause()
            return

        elif sub == 3:
            new_checker = _ask('Model for the new run', '')
            if not new_checker:
                return
            new_per_run = per_run_models + [new_checker]
            config.purer_classification.per_run_models = new_per_run
            config.purer_classification.n_runs = len(new_per_run)
            _save_purer_to_disk()
            if not _confirm(f'Add run {len(new_per_run)} with model {new_checker!r}?'):
                return
            config.resume_from = None
            stage_classify_purer(config, output_dir=output_dir)
            print()
            _ok('PURER overlay written to qra.db. Run Assemble to rebuild master_segments.')
            _pause()
            return

        elif sub == 4:
            from classification_tools.classification_loop import patch_runs_checkpoint
            ckpt_dir = _paths.llm_checkpoints_dir(output_dir)
            candidates = sorted(
                _glob.glob(os.path.join(ckpt_dir, 'purer_cue_results_*_runs.json'))
            )
            if not candidates:
                _err('No PURER *_runs.json checkpoint found. Cannot reclassify an individual run.')
                _pause()
                return
            checkpoint_path = candidates[-1]
            _info(f'Checkpoint: {os.path.basename(checkpoint_path)}')
            print()
            run_number = int(_ask(f'Which run to reclassify? (1–{n_runs})', '1'))
            if not (1 <= run_number <= n_runs):
                _err(f'Run number must be between 1 and {n_runs}.')
                _pause()
                return
            run_idx = run_number - 1
            current = (per_run_models[run_idx] if run_idx < len(per_run_models) else purer_model)
            new_model = _ask(f'New model for run {run_number} (Enter = keep current)', current)
            new_model_val = new_model if new_model != current else None
            if not _confirm(f'Reclassify PURER run {run_number} now?'):
                return
            patch_runs_checkpoint(checkpoint_path, run_idx, new_model=new_model_val)
            if new_model_val:
                prm = list(getattr(config.purer_classification, 'per_run_models', []) or [])
                if run_idx < len(prm):
                    prm[run_idx] = new_model_val
                    config.purer_classification.per_run_models = prm
                _save_purer_to_disk()
            config.resume_from = checkpoint_path
            stage_classify_purer(config, output_dir=output_dir)
            print()
            _ok('PURER overlay written to qra.db. Run Assemble to rebuild master_segments.')
            _pause()
            return

    # First-time classification
    zero_shot = _confirm(
        'Use zero-shot prompting? (definitions only, no examples)', default=False
    )
    if zero_shot:
        config.purer_classification.zero_shot_prompt = True
        _ok('Zero-shot mode ON.')
    print()

    if _confirm('Change PURER model before running?', default=False):
        new_model = _ask('New PURER model ID', purer_model)
        if new_model:
            config.purer_classification.model = new_model
            _save_purer_to_disk()
    print()

    if not _confirm('Run PURER classification now?'):
        return
    stage_classify_purer(config, output_dir=output_dir)
    print()
    _ok('purer_labels.jsonl written. Run Assemble to rebuild master_segments.')
    _pause()


def _action_resegment_purer(config, output_dir: str) -> None:
    from .orchestrator import stage_resegment_therapist, SilentObserver
    _section('Re-segment PURER / Therapist Content')
    _info(
        'Re-extracts therapist (PURER) segments and re-applies PHI scrubbing\n'
        'WITHOUT touching participant VAAMR segments or frozen testsets.\n'
        '\n'
        'Run this before re-classifying PURER (e.g. after changing the\n'
        'therapist gap threshold or the PURER unit of analysis).\n'
        '\n'
        'Participant VAAMR segments and frozen testsets are preserved.'
    )
    print()

    gap = getattr(config.purer_cue, 'therapist_max_gap_seconds', None)
    _info(f'Therapist max gap (seconds): {gap}')
    print()

    if not _confirm('Re-segment therapist (PURER) content now?'):
        return
    try:
        r = stage_resegment_therapist(config, output_dir, observer=SilentObserver())
        print()
        _ok(f"Re-segmented {r.get('sessions', 0)} session(s).")
        _info(f"Therapist segments: {r.get('old_therapist', 0)} → {r.get('new_therapist', 0)}")
        _info(f"Participant segments preserved: {r.get('participant_preserved', 0)}")
        _info('Participant VAAMR segments and frozen testsets were preserved.')
        _info('Run "PURER Classification" next, then Assemble to rebuild master_segments.')
    except Exception as _e:
        _warn(f'Re-segmentation failed: {_e}')
    _pause()


def _action_assemble(config, output_dir: str, framework) -> None:
    from .orchestrator import stage_assemble
    _section('Stage 6 — Assemble Master Dataset')
    _info(
        'Joins frozen segments + all classification overlays into a single\n'
        'master_segments.csv export with confidence tiers.\n'
        '\n'
        'Also regenerates:\n'
        '  • coded transcripts  (04_validation/full_transcripts/)\n'
        '  • human coding forms (04_validation/)\n'
        '  • AI testset answer keys (SHA-verified — aborts if text drifted)\n'
        '  • flagged_for_review.txt\n'
        '  • per-transcript stats and cumulative report\n'
        '\n'
        'Frozen segments and human worksheets are NEVER modified by assemble.'
    )
    print()
    if not _confirm('Assemble master dataset now?'):
        return
    stage_assemble(config, output_dir=output_dir)
    print()
    _ok('master_segments.csv written and validation artifacts refreshed.')
    _pause()


def _action_analyze(output_dir: str) -> None:
    from analysis.runner import run_analysis
    _section('Stage 8 — Results Analysis')
    _info(
        'Runs all post-hoc analyses on the assembled master_segments dataset:\n'
        '\n'
        '  • Per-participant longitudinal VAAMR trajectories\n'
        '  • Per-session stage distribution summaries\n'
        '  • Per-theme / stage cohort breakdowns\n'
        '  • PURER × VAAMR cue-response influence analysis\n'
        '  • Cross-session lift statistics (if codebook overlay present)\n'
        '  • Graph-ready CSVs and PNG figures\n'
        '  • LLM-generated session and participant narrative summaries\n'
        '\n'
        'Output: 03_analysis_data/, 05_figures/, 06_reports/'
    )
    print()
    opts = [
        ('Use config default', 'Honor gnn_layer.enabled from the project config.'),
        ('Force GNN ON', 'Run the GNN representation + consensus layer this time.'),
        ('Force GNN OFF', 'Skip the GNN layer this time (faster).'),
    ]
    sel = _menu('GNN layer for this analysis run', opts)
    if sel == 0:
        return
    force_gnn = {1: None, 2: True, 3: False}[sel]
    run_analysis(output_dir, force_gnn=force_gnn)
    print()
    _ok('Analysis complete. Reports written to 06_reports/.')
    _pause()


def _action_validate(config, output_dir: str, framework) -> None:
    from .orchestrator import stage_assemble
    _section('Validate — Refresh All Validation Artifacts')
    _info(
        'Refreshes all validation artifacts without re-classifying:\n'
        '\n'
        '  • Human classification forms regenerated from current segments\n'
        '  • AI testset answer keys updated from current overlay labels\n'
        '    (SHA verification ensures frozen segment text is unchanged)\n'
        '  • Content-validity test AI answer keys refreshed\n'
        '  • coded transcripts re-emitted with latest labels\n'
        '\n'
        'Human worksheets and testset manifests are NEVER overwritten.'
    )
    print()
    if not _confirm('Refresh validation artifacts now?'):
        return
    stage_assemble(config, output_dir=output_dir)
    print()
    _ok('Validation artifacts refreshed.')
    _pause()


def _action_testset_menu(config, output_dir: str, state: dict, framework) -> None:
    """Sub-menu for testset management."""
    from . import segments_io as _sio
    from .assembly.human_forms import refresh_testset_answer_key

    while True:
        opts = [
            ('Create new VAAMR testset',
             'Stratified random sample of participant segments → frozen human worksheet\n'
             'and initial AI answer key. Segment text is SHA-locked at creation.'),
            ('Create new PURER testset',
             'Stratified random sample of therapist segments with PURER labels.'),
            ('Refresh AI answer keys (all testsets)',
             'Re-emit AI_answer_key.txt for every frozen testset using current\n'
             'overlay labels. SHA-verifies segment text — aborts if text drifted.'),
            ('List frozen testsets',
             'Show name, kind, segment count, and creation date for each testset.'),
        ]
        choice = _menu('Testset Management', opts)
        if choice == 0:
            return

        if choice == 1:
            _action_testset_create(config, output_dir, framework, kind='vaamr')
        elif choice == 2:
            _action_testset_create(config, output_dir, framework, kind='purer')
        elif choice == 3:
            _action_testset_refresh_all(config, output_dir, framework, state)
        elif choice == 4:
            _action_testset_list(output_dir, state)


def _action_testset_create(config, output_dir: str, framework, kind: str) -> None:
    from . import output_paths as _paths
    from .assembly.human_forms import generate_or_refresh_validation_testsets
    from . import segments_io as _sio

    _section(f'Create New {kind.upper()} Testset')
    next_n = _paths.next_testset_number(output_dir)
    n_sets = int(_ask('Number of parallel testsets (for IRR between coders)', '2'))
    frac = float(_ask('Fraction of eligible segments per set', '0.10'))
    seed = int(_ask('Random seed', '42'))

    _info(f'\nWill create {n_sets} testset(s) (worksheet #{next_n}–#{next_n + n_sets - 1}) '
          f'sampling {frac*100:.0f}% of {kind} segments.')
    if not _confirm('Create now?'):
        return

    from .config import TestSetsConfig, TestSetSpec
    config.test_sets = TestSetsConfig(
        **{kind: TestSetSpec(enabled=True, name=f'{kind}_testset', n_sets=n_sets,
                            fraction_per_set=frac, random_seed=seed)}
    )
    segments = _sio.load_segments_for_stage(output_dir)
    paths = generate_or_refresh_validation_testsets(
        segments, framework, output_dir,
        test_sets_config=config.test_sets,
        create_missing=True, codebook_enabled=config.run_codebook_classifier,
    )
    print()
    for p in paths:
        _ok(f'Created: {os.path.basename(p)}')
    _pause()


def _action_testset_refresh_all(config, output_dir: str, framework, state: dict) -> None:
    from . import segments_io as _sio
    from .assembly.human_forms import refresh_testset_answer_key
    from ._freeze import FrozenArtifactError

    _section('Refresh All Testset AI Answer Keys')
    if not state['has_testsets']:
        _warn('No testsets found in this project.')
        _pause()
        return

    _info(f'Found {len(state["has_testsets"])} testset(s):')
    for n in state['has_testsets']:
        _info(f'  • worksheet #{n}')
    print()
    _info('Each AI answer key will be re-emitted from current overlay labels.\n'
          'Human worksheets are NEVER modified.')
    print()
    if not _confirm('Refresh all AI answer keys now?'):
        return

    segments = _sio.load_segments_for_stage(output_dir)
    by_id = {s.segment_id: s for s in segments}

    for n in state['has_testsets']:
        try:
            path = refresh_testset_answer_key(
                by_id, framework, output_dir, n,
                codebook_enabled=config.run_codebook_classifier,
            )
            _ok(f'Worksheet #{n}: AI answer key updated → {os.path.basename(path)}')
        except FrozenArtifactError as exc:
            _err(f'Worksheet #{n}: {exc}')
        except Exception as exc:
            _err(f'Worksheet #{n}: {exc}')
    print()
    _pause()


def _action_testset_list(output_dir: str, state: dict) -> None:
    from . import output_paths as _paths
    from .assembly.human_forms import _detect_worksheet_kind
    _section('Validation Testsets')
    if not state['has_testsets']:
        _warn('No testsets found.')
        _pause()
        return
    for n in state['has_testsets']:
        human_path = _paths.testset_human_flat_path(output_dir, n)
        ai_path = _paths.testset_ai_flat_path(output_dir, n)
        kind = _detect_worksheet_kind(human_path)
        ai_status = 'AI key present' if os.path.isfile(ai_path) else 'AI key missing'
        print(f'  • Worksheet #{n}  ({kind}, {ai_status})')
    print()
    _pause()


def _action_cv_menu(config, output_dir: str, state: dict, framework) -> None:
    """Sub-menu for content-validity testset management."""
    while True:
        opts = [
            ('Create VAAMR content-validity testset',
             'Builds items from VAAMR exemplar / subtle / adversarial utterances\n'
             'defined in the framework. Freezes human worksheet and definition key.\n'
             'AI answer key is refreshable.'),
            ('Create PURER content-validity testset',
             'Same as above but for PURER therapist move definitions.'),
            ('Refresh CV AI answer keys (all)',
             'Re-grades AI answers for all content-validity testsets.'),
            ('List content-validity testsets',
             'Show existing CV testset names and creation dates.'),
        ]
        choice = _menu('Content-Validity Testsets', opts)
        if choice == 0:
            return

        if choice == 1:
            _action_cv_create(config, output_dir, framework, fw_name='vaamr')
        elif choice == 2:
            _action_cv_create(config, output_dir, framework, fw_name='purer')
        elif choice == 3:
            _action_cv_refresh_all(config, output_dir, state, framework)
        elif choice == 4:
            _action_cv_list(output_dir, state)


def _action_cv_create(config, output_dir: str, framework, fw_name: str) -> None:
    from .assembly.content_validity import generate_or_refresh_cv_testsets
    _section(f'Create {fw_name.upper()} Content-Validity Testset')
    name = _ask('Testset name', f'cv_{fw_name}_v1')
    _info(
        f'\nContent-validity testset "{name}" will be built from the\n'
        f'{fw_name.upper()} framework\'s exemplar / subtle / adversarial utterances.\n'
        'Human worksheet and definition key will be frozen.\n'
        'AI answer key is generated now and refreshable later.'
    )
    if not _confirm('Create now?'):
        return

    from .config import ContentValidityConfig, ContentValiditySpec
    config.content_validity = ContentValidityConfig(
        **{fw_name: ContentValiditySpec(enabled=True, name=name)}
    )
    if fw_name == 'purer':
        from constructs.registry import load as _registry_load_fw
        fw = _registry_load_fw('purer')
    else:
        fw = framework

    generate_or_refresh_cv_testsets(config, fw, output_dir=output_dir)
    print()
    _ok(f'Content-validity testset "{name}" created.')
    _pause()


def _action_cv_refresh_all(config, output_dir: str, state: dict, framework) -> None:
    from .assembly.content_validity import generate_or_refresh_cv_testsets
    _section('Refresh Content-Validity AI Answer Keys')
    if not state['has_cv_testsets']:
        _warn('No content-validity testsets found.')
        _pause()
        return
    for name in state['has_cv_testsets']:
        _info(f'  • {name}')
    print()
    if not _confirm('Refresh all CV AI answer keys now?'):
        return
    generate_or_refresh_cv_testsets(config, framework, output_dir=output_dir)
    _ok('CV answer keys refreshed.')
    _pause()


def _action_cv_list(output_dir: str, state: dict) -> None:
    from .assembly.content_validity import list_content_validity_testsets
    _section('Content-Validity Testsets')
    testsets = list_content_validity_testsets(output_dir)
    if not testsets:
        _warn('No content-validity testsets found.')
        _pause()
        return
    for t in testsets:
        fw = (t.get('framework') or {}).get('name', '?')
        n = t.get('n_items', '?')
        created = (t.get('created_at') or 'unknown')[:10]
        print(f'  • {t["name"]}  ({fw}, {n} items, created {created})')
    print()
    _pause()


def _action_edit_config(config_path: str) -> None:
    _section('Edit Project Configuration')
    if not config_path:
        _warn('No qra_config.json found for this project.')
        _pause()
        return
    _info(f'Config file: {config_path}')
    print()
    _info('You can edit the config directly in your editor, or change individual')
    _info('settings here. Re-running ingest/classify/assemble will pick up changes.')
    print()
    opts = [
        ('Change LM Studio server URL', 'Update the lmstudio_base_url for theme and PURER classifiers.'),
        ('Change classification models', 'Set primary model and per-run checker models.'),
        ('Change number of classification runs', 'n_runs controls the multi-model ensemble size.'),
        ('Toggle codebook (VCE) classifier', 'Enable or disable the 54-code phenomenology codebook.'),
        ('Toggle PURER classifier', 'Enable or disable PURER therapist cue classification.'),
        ('Toggle GNN layer', 'Enable or disable the GNN discovery + consensus-distillation layer.'),
        ('Toggle GNN authoritative labels', 'Make graph-consensus labels the label of record\n'
         '(recommended only after the reliability gate reports the graph is ready).'),
        ('Open config in $EDITOR', 'Open the raw JSON in your system editor.'),
    ]
    choice = _menu('Config Edits', opts)
    if choice == 0:
        return
    try:
        with open(config_path) as fh:
            raw = json.load(fh)
    except Exception as exc:
        _err(f'Cannot read config: {exc}')
        _pause()
        return

    changed = False
    if choice == 1:
        url = _ask('New LM Studio URL', raw.get('theme_classification', {}).get('lmstudio_base_url', 'http://localhost:1234/v1'))
        raw.setdefault('theme_classification', {})['lmstudio_base_url'] = url
        raw.setdefault('purer_classification', {})['lmstudio_base_url'] = url
        changed = True
    elif choice == 2:
        theme_model = raw.get('theme_classification', {}).get('model', '')
        purer_model = raw.get('purer_classification', {}).get('model', theme_model)
        model = _ask('VAAMR (theme) primary model ID', theme_model)
        raw.setdefault('theme_classification', {})['model'] = model
        purer = _ask('PURER primary model ID (Enter = same as VAAMR)', purer_model)
        raw.setdefault('purer_classification', {})['model'] = purer or model
        per_run_raw = _ask('Per-run models (comma-separated, blank = keep current)', '')
        if per_run_raw:
            per_run = [m.strip() for m in per_run_raw.split(',')]
            raw['theme_classification']['per_run_models'] = per_run
            raw['purer_classification']['per_run_models'] = per_run
            raw['theme_classification']['n_runs'] = len(per_run)
            raw['purer_classification']['n_runs'] = len(per_run)
        changed = True
    elif choice == 3:
        n = int(_ask('Number of classification runs', str(raw.get('theme_classification', {}).get('n_runs', 3))))
        raw.setdefault('theme_classification', {})['n_runs'] = n
        raw.setdefault('purer_classification', {})['n_runs'] = n
        changed = True
    elif choice == 4:
        cur = raw.get('pipeline', {}).get('run_codebook_classifier', False)
        new_val = _confirm(f'Enable codebook classifier? (currently {"ON" if cur else "OFF"})', not cur)
        raw.setdefault('pipeline', {})['run_codebook_classifier'] = new_val
        changed = True
    elif choice == 5:
        cur = raw.get('run_purer_labeler', True)
        new_val = _confirm(f'Enable PURER classifier? (currently {"ON" if cur else "OFF"})', not cur)
        raw['run_purer_labeler'] = new_val
        changed = True
    elif choice == 6:
        cur = raw.get('gnn_layer', {}).get('enabled', False)
        new_val = _confirm(f'Enable GNN layer? (currently {"ON" if cur else "OFF"})', not cur)
        raw.setdefault('gnn_layer', {})['enabled'] = new_val
        changed = True
    elif choice == 7:
        cur = raw.get('gnn_layer', {}).get('gnn_authoritative', False)
        if not cur:
            _warn('Recommended only after 06_reports/06_gnn/validation.txt reports YES.')
        new_val = _confirm(
            f'Make GNN labels authoritative? (currently {"ON" if cur else "OFF"})', not cur)
        raw.setdefault('gnn_layer', {})['gnn_authoritative'] = new_val
        changed = True
    elif choice == 8:
        editor = os.environ.get('EDITOR', 'nano')
        os.system(f'{editor} "{config_path}"')
        return

    if changed:
        with open(config_path, 'w') as fh:
            json.dump(raw, fh, indent=2)
        _ok(f'Config saved to {config_path}')
    _pause()


# ---------------------------------------------------------------------------
# Existing Project flow
# ---------------------------------------------------------------------------

_SMALL_ADDITION_SESSIONS = 3  # ≤ this many new sessions → the LLM is affordable; recommend it


def _estimate_new_sessions(output_dir: str, config_path) -> Optional[int]:
    """Best-effort count of input transcript files not yet segmented (the add-data scale)."""
    try:
        import glob
        import json as _json
        from . import segments_io as _sio
        from .config import PipelineConfig
        segmented = set(_sio.list_segmented_sessions(output_dir))
        tdir = None
        if config_path and os.path.isfile(config_path):
            with open(config_path, encoding='utf-8') as f:
                cfg = PipelineConfig.from_json(_json.load(f))
            tdir = cfg.transcript_dir
        if not tdir or not os.path.isdir(tdir):
            return None
        files = glob.glob(os.path.join(tdir, '*.json')) + glob.glob(os.path.join(tdir, '*.vtt'))
        return sum(1 for f in files
                   if os.path.splitext(os.path.basename(f))[0] not in segmented)
    except Exception:
        return None


def _recommend_add_data_mode(probe: dict, gnn: dict, n_new: Optional[int]):
    """(mode, reason) recommendation given the cheap-classifier gates + the addition's scale."""
    probe_ready, gnn_ready = bool(probe.get('ready')), bool(gnn.get('ready'))
    if not (probe_ready or gnn_ready):
        return 'llm', ('no cheap classifier has passed its gate — the multi-run LLM '
                       '(human-level) is the only reliable option.')
    if n_new is not None and n_new <= _SMALL_ADDITION_SESSIONS:
        return 'llm', (f'a small addition (~{n_new} new session(s)) — the LLM is affordable '
                       'here and stays the human-level label of record.')
    if probe_ready:
        pk = probe.get('human_kappa')
        return 'probe', (f'a larger addition and the probe gate passed (↔human κ {pk:.2f}); '
                         'it scales LLM-free, tagged below the LLM.' if pk is not None else
                         'a larger addition and the probe gate passed; it scales LLM-free.')
    return 'gnn', ('a larger addition and the GNN gate passed; it scales LLM-free, '
                   'tagged below the LLM.')


def _add_data_mode_picker(output_dir: str, config_path) -> str:
    """Show the IRR comparison + a scale-based recommendation; return the chosen mode.

    The user's new ask: when adding data, choose how to label the new segments (LLM / probe /
    GNN), seeing the probe's and GNN's reliability vs this project's existing data and a
    recommendation that depends on the addition's scale.
    """
    from analysis.irr_analysis import load_irr_metrics
    m = load_irr_metrics(output_dir)
    probe, gnn, llm = m.get('probe', {}), m.get('gnn', {}), m.get('llm', {})

    def _k(x):
        return 'n/a' if not isinstance(x, (int, float)) else f'{x:.3f}'

    n_new = _estimate_new_sessions(output_dir, config_path)
    existing_labeled = None
    df = _probe_master_df(output_dir)
    if df is not None and 'final_label' in df.columns and 'speaker' in df.columns:
        try:
            existing_labeled = int(((df['speaker'] == 'participant')
                                    & (df['final_label'].notna())).sum())
        except Exception:
            existing_labeled = None

    _section('Classification mode for the NEW data')
    _info('The multi-run LLM is the label of record (human-level). The probe / GNN are')
    _info('LLM-free scalers that FILL below the LLM (tagged lower-confidence) — useful at')
    _info('scale when running the LLM on every new segment is impractical.')
    print()
    band = llm.get('human_human_band')
    bandtxt = f'{band[0]:.2f}–{band[1]:.2f}' if band else 'n/a'
    _info('Reliability vs this project\'s existing data (Cohen κ; higher = better):')
    _info(f'  reference   LLM↔human κ {_k(llm.get("human_kappa"))}   '
          f'(human↔human band α {bandtxt})')
    _info(f'  PROBE       ↔human {_k(probe.get("human_kappa"))}   ↔LLM {_k(probe.get("llm_kappa"))}'
          f'   gate {"PASSED ★" if probe.get("ready") else "not passed"}'
          + (f'   [{probe.get("mode")}]' if probe.get('mode') else ''))
    _info(f'  GNN         ↔human {_k(gnn.get("human_kappa"))}   ↔LLM {_k(gnn.get("llm_kappa"))}'
          f'   gate {"PASSED ★" if gnn.get("ready") else "not passed"}')
    pcr = probe.get('per_class_recall') or {}
    if pcr:
        _info('  probe per-stage recall: '
              + ', '.join(f'{k} {_k(v)}' for k, v in pcr.items() if v is not None))
    if existing_labeled is not None:
        _info(f'  existing LLM/human-labeled participant segments: {existing_labeled}')
    if n_new is not None:
        _info(f'  new sessions to add (estimate): {n_new}')
    print()

    rec, reason = _recommend_add_data_mode(probe, gnn, n_new)
    _ok(f'Recommended: {rec.upper()} — {reason}')
    print()
    opts = [
        ('Multi-run LLM consensus (default, human-level)' + (' ★' if rec == 'llm' else ''),
         'Highest quality; the label of record. Per-segment frontier-LLM cost.'),
        ('Probe — LLM-free, gated' + (' ★' if rec == 'probe' else ''),
         'Per-rater ensemble on cached embeddings; fills below the LLM (probe_consensus).'),
        ('GNN — LLM-free, non-authoritative' + (' ★' if rec == 'gnn' else ''),
         'Graph consensus; fills below the LLM (gnn_consensus).'),
    ]
    choice = _menu('Label the new segments with', opts, back_label='Cancel add-data')
    if choice == 0:
        return ''  # cancel
    return {1: 'llm', 2: 'probe', 3: 'gnn'}.get(choice, 'llm')


def _action_add_data(output_dir: str, config_path) -> None:
    """Incrementally segment + classify only NEW transcripts (longitudinal additions)."""
    _section('Add New Transcripts (incremental)')
    _info(
        'Segments + classifies only NEW transcript files, then re-assembles and\n'
        're-analyzes. Frozen segments and frozen validation testsets are never\n'
        'disturbed. Newly-discovered speakers are walked through interactively.\n'
        '\n'
        'Requires an existing project (frozen segments + classification manifest).'
    )
    print()
    if not _confirm('Run incremental add-data now?'):
        return
    # Mode picker: LLM (default) / probe / GNN, with the IRR comparison + a scale recommendation.
    mode = _add_data_mode_picker(output_dir, config_path)
    if not mode:
        _info('Add-data cancelled.')
        return
    import argparse as _argparse
    import qra as _qra
    fake = _argparse.Namespace(
        config=config_path, transcript_dir=None, output_dir=output_dir, trial_id=None,
        backend=None, model=None, api_key=None, models=None, lmstudio_url=None,
        no_auto_analyze=False, test_zeroshot=False, preset=None,
        resume_from=None, framework=None, codebook=None,
        speaker_filter_mode=None, exclude_speakers=None,
        no_theme_labeler=False, run_codebook_classifier=False,
        no_codebook_classifier=False, verbose_segmentation=False,
        zero_shot=False, no_text_anonymization=False,
        classify_mode=mode,
    )
    try:
        _qra.cmd_add_data(fake)
    except SystemExit as ex:
        if ex.code not in (0, None):
            _warn('No new transcripts found (or add-data exited).')
    except Exception as exc:
        _err(f'add-data failed: {exc}')
    _pause()


def _action_classify_codebook(config, output_dir: str) -> None:
    """Run the VCE phenomenology codebook classifier on participant segments."""
    from .orchestrator import stage_classify_codebook
    from constructs.codebook.phenomenology_codebook import get_phenomenology_codebook
    _section('Classify — VCE Codebook (participant segments)')
    _info(
        'Runs the VCE phenomenology codebook (embedding similarity + LLM ensemble)\n'
        'on participant segments. Writes the codebook overlay to qra.db.\n'
        'Participant VAAMR labels and frozen segments are NOT modified.'
    )
    print()
    if not _confirm('Run codebook classification now?'):
        return
    stage_classify_codebook(config, get_phenomenology_codebook(), output_dir=output_dir)
    print()
    _ok('Codebook overlay written to qra.db. Run Assemble to rebuild master_segments.')
    _pause()


def _action_classify_cv(config, output_dir: str, framework) -> None:
    """Compute VAAMR × VCE co-occurrence lift statistics (cross-validation)."""
    from .orchestrator import stage_cross_validation
    _section('Classify — Cross-Validation (VAAMR × VCE lift)')
    _info(
        'Computes theme × codebook co-occurrence lift statistics implementing\n'
        "Varela's mutual-constraints logic. Requires both VAAMR and codebook\n"
        'overlays. Writes lift statistics to 04_validation/cross_validation/.'
    )
    print()
    if not _confirm('Run cross-validation now?'):
        return
    stage_cross_validation(config, framework, output_dir=output_dir)
    print()
    _ok('Cross-validation lift statistics written.')
    _pause()


def _action_gnn_train(config, output_dir: str, state: dict) -> None:
    """Train the graph + run the reliability gate (also ADDS the GNN to an LLM-only project)."""
    from analysis.runner import run_analysis
    from gnn_layer.classifier.validation import read_gate_verdict, format_gate_verdict
    _section('Train / update GNN consensus layer')
    _info(
        'Trains the graph on the existing LLM/human consensus, runs the reliability\n'
        'gate (kappa vs LLM), and writes the GNN reports + consensus overlay.\n'
        'Use this to ADD the GNN to a project that has only run LLM consensus.\n'
        '\n'
        'A trustworthy reliability gate needs multi-run LLM ballots (n_runs >= 3).'
    )
    print()
    # Multi-run ballot pre-flight (non-blocking warning).
    try:
        import pandas as _pd
        from . import segments_io as _sio
        from . import classifications_io as _cio
        from gnn_layer.soft_labels import ballot_coverage
        segs = _sio.load_segments_for_stage(output_dir, apply=())
        by_id = {s.segment_id: s for s in segs}
        _cio.apply_overlays(output_dir, by_id, keys=('theme',))
        cov = ballot_coverage(_pd.DataFrame([
            {'speaker': s.speaker, 'segment_id': s.segment_id,
             'rater_votes': getattr(s, 'rater_votes', None)} for s in segs]))
        if cov['n_participant'] and cov['multirun_fraction'] < 0.5:
            _warn(f"Only {round(100 * cov['multirun_fraction'])}% of participant segments "
                  "carry multi-run LLM ballots.")
            _warn('The GNN consensus signal needs >=2 runs; the gate kappa may be unreliable.')
            _warn('Consider re-running VAAMR with n_runs >= 3 first.')
            if not _confirm('Train the GNN anyway?', default=False):
                return
    except Exception:
        pass
    if not _confirm('Train the GNN layer now? (this can take a while)'):
        return
    run_analysis(output_dir, verbose=True, force_gnn=True)
    print()
    _info(format_gate_verdict(read_gate_verdict(output_dir), output_dir))
    _pause()


def _action_gnn_status(output_dir: str) -> None:
    """Print the GNN reliability-gate verdict (kappa vs LLM; ready for scaling?)."""
    from gnn_layer.classifier.validation import read_gate_verdict, format_gate_verdict
    _section('GNN Reliability Gate')
    _info(format_gate_verdict(read_gate_verdict(output_dir), output_dir))
    print()
    _info('Inter-rater reliability readout: when the gate reports the graph is')
    _info('READY, its agreement with the LLM/human consensus has reached the')
    _info('target — the graph can label new data without LLM calls (scale mode).')
    _pause()


def _probe_master_df(output_dir: str):
    """Read the assembled master_segments.csv (the probe's data source), or None."""
    import pandas as _pd
    from . import output_paths as _paths
    csv = os.path.join(_paths.master_segments_dir(output_dir), 'master_segments.csv')
    if not os.path.isfile(csv):
        return None
    return _pd.read_csv(csv)


def _print_probe_verdict(v) -> None:
    """Human-readable probe gate summary (shared by the probe TUI actions)."""
    if not v:
        _warn('No probe gate yet — run "Probe — train + gate" first.')
        return
    def _f(x):
        return f'{x:.3f}' if isinstance(x, (int, float)) else 'n/a'
    hci = v.get('probe_human_ci', [None, None])
    lci = v.get('probe_llm_ci', [None, None])
    _info(f"mode: {v.get('mode')}   raters: "
          f"{', '.join(v.get('raters') or []) or '(A1n single probe)'}")
    _info(f"probe ↔ human : κ {_f(v.get('probe_human_kappa'))} "
          f"[{_f(hci[0])}, {_f(hci[1])}]  n={v.get('probe_human_n')}  "
          f"(floor {v.get('irr_human_band_floor')})")
    _info(f"probe ↔ LLM   : κ {_f(v.get('probe_llm_kappa'))} "
          f"[{_f(lci[0])}, {_f(lci[1])}]  n={v.get('probe_llm_n')}")
    pcr = v.get('per_class_recall') or {}
    if pcr:
        _info('per-stage recall: ' + ', '.join(f'{k} {_f(rv)}' for k, rv in pcr.items()))
    if v.get('rare_stage_notes'):
        _warn('rare-stage: ' + '; '.join(v['rare_stage_notes']))
    if v.get('ready_for_scaling'):
        _ok('Gate PASSED — the probe may fill unlabeled segments (tagged probe_consensus).')
    else:
        _warn('Gate NOT passed — assistive only; the LLM stays the label of record.')


def _action_probe_train(config, output_dir: str) -> None:
    """Fit the LLM-free probe scaler + run the participant-grouped reliability gate."""
    from classification_tools.probe import probe_classifier as _pc
    _section('Probe — train + reliability gate (LLM-free scaler)')
    _info(
        'Fits the per-rater ensemble probe on the existing LLM/human labels and runs the\n'
        'participant-grouped gate (probe↔human / probe↔LLM κ). The probe FILLS unlabeled\n'
        'participant segments BELOW the LLM and never overrides an LLM/human label.\n'
        '\n'
        'Needs an assembled master_segments.csv and the Qwen embedding backend reachable.'
    )
    print()
    if not _confirm('Train the probe now?'):
        return
    df = _probe_master_df(output_dir)
    if df is None:
        _warn('No master_segments.csv — run "Assemble master dataset" first.')
        _pause()
        return
    probe_cfg = getattr(config, 'probe', None) or _pc.ProbeConfig()
    try:
        _pc.train_probe(df, output_dir, probe_cfg)
        print()
        _print_probe_verdict(_pc.evaluate_probe(df, output_dir, probe_cfg))
    except Exception as exc:
        _err(f'probe train failed: {exc}')
    _pause()


def _action_probe_status(output_dir: str) -> None:
    """Print the probe reliability-gate verdict (probe↔human/LLM κ; ready to scale?)."""
    from classification_tools.probe import probe_classifier as _pc
    _section('Probe Reliability Gate')
    _print_probe_verdict(_pc.read_probe_gate(output_dir))
    print()
    _info('The multi-run LLM consensus stays the label of record (human-level). The probe')
    _info('is an assistive, gated scaler — useful for bulk/new data the LLM budget cannot reach.')
    _pause()


def _action_probe_classify(config, output_dir: str, state: dict) -> None:
    """LLM-free fill of UNLABELED participant segments with the gated probe."""
    from classification_tools.probe import probe_classifier as _pc
    _section('Probe — classify unlabeled (LLM-free)')
    status = state.get('probe_status', 'absent')
    if status == 'absent':
        _warn('No trained/gated probe yet. Run "Probe — train + gate" first.')
        _pause()
        return
    if status != 'ready':
        _warn('The probe gate has NOT passed (probe↔human κ below the human band).')
        _warn('Probe labels are noisier than the LLM; they are tagged probe_consensus.')
        if not _confirm('Fill with the probe anyway?', default=False):
            return
    else:
        _ok('Probe gate passed — safe to fill unlabeled segments without LLMs.')
    _info('Fills only participant segments the LLM never balloted on; abstains where unsure.')
    print()
    df = _probe_master_df(output_dir)
    if df is None:
        _warn('No master_segments.csv — run "Assemble master dataset" first.')
        _pause()
        return
    probe_cfg = getattr(config, 'probe', None) or _pc.ProbeConfig()
    try:
        n = _pc.classify_with_probe(df, output_dir, probe_cfg)
        _ok(f'Probe labeled {n} previously-unlabeled participant segment(s) → probe_labels overlay.')
        if n:
            # The operator opted in here (gate passed, or confirmed despite a failing gate),
            # so promote this batch's fills on the re-assemble — still BELOW the LLM.
            from .orchestrator import stage_assemble
            stage_assemble(config, output_dir=output_dir, probe_ready=True)
            _ok('master_segments updated — probe_consensus labels folded in (below the LLM).')
    except Exception as exc:
        _err(f'probe classify failed: {exc}')
    _pause()


def _action_migrate(output_dir: str) -> None:
    """Import a legacy JSONL project into qra.db (preview, then confirm)."""
    from . import db as _db
    from .legacy_migration import is_jsonl_project, migrate_jsonl_to_sqlite, preview_counts
    _section('Migrate Legacy JSONL → qra.db')
    if _db.db_exists(output_dir):
        _ok('qra.db already present — nothing to migrate.')
        _pause()
        return
    if not is_jsonl_project(output_dir):
        _warn('No legacy JSONL pipeline files found — nothing to migrate.')
        _pause()
        return
    c = preview_counts(output_dir)
    ov = ', '.join(f'{k}:{n}' for k, n in sorted(c['overlays'].items())) or 'none'
    _info(f"Would import {c['segments']} segments across {c['sessions']} session(s).")
    _info(f"  overlays:    {ov}")
    _info(f"  manifest:    {c['manifest_keys']} key(s)")
    _info(f"  testsets:    {c['testset_worksheets']} worksheet(s)")
    _info(f"  cv testsets: {c['cv_testsets']}")
    _info('Non-destructive: originals are relocated to _legacy_files/.')
    print()
    if not _confirm('Run migration now?'):
        return
    result = migrate_jsonl_to_sqlite(output_dir)
    _ok(f"Imported {result['segments']} segments across {result['sessions']} session(s) into qra.db.")
    _pause()


def _existing_project_flow() -> None:
    _section('Open Existing Project')
    _info(
        'Enter the path to a QRA output directory.  This is the folder that\n'
        'contains 02_meta/, 04_validation/, etc.  It can be a legacy project\n'
        '(created before the modular pipeline) — auto-migration will be offered.'
    )
    print()

    while True:
        output_dir = _ask('Project output directory', './data/MMORE_Processed')
        output_dir = os.path.expanduser(output_dir)
        if os.path.isdir(output_dir):
            break
        _warn(f'Directory not found: {output_dir}')
        if not _confirm('Try a different path?'):
            return

    state = _detect_state(output_dir)

    # Load config if available
    config = None
    framework = _load_framework_default()
    if state['config_path']:
        try:
            config = _load_config(state['config_path'])
            _ok(f'Config loaded from {state["config_path"]}')
        except Exception as exc:
            _warn(f'Could not parse config ({exc}); using defaults.')

    if config is None:
        from .config import PipelineConfig
        config = PipelineConfig()
        config.output_dir = output_dir

    config.output_dir = output_dir

    while True:
        # Refresh state each loop so status marks update after actions.
        state = _detect_state(output_dir)
        _print_project_state(state)

        def _tag(cond, suffix='  ✓ done'):
            return suffix if cond else ''

        def _need_segments(fn):
            """Wrap an action so it guards on frozen segments existing first."""
            def _wrapped():
                if not state['has_segments']:
                    _warn('No frozen segments yet — run Ingest first.')
                    _pause()
                else:
                    fn()
            return _wrapped

        def _edit_config_and_reload():
            nonlocal config
            _action_edit_config(state['config_path'])
            if state['config_path']:
                try:
                    config = _load_config(state['config_path'])
                    config.output_dir = output_dir
                    _ok('Config reloaded into session.')
                except Exception:
                    pass

        def _edit_anonymization():
            from .anonymization_editor import run_anonymization_tui
            run_anonymization_tui(output_dir, config=config)

        gnn_status = state.get('gnn_status', 'absent')
        probe_status = state.get('probe_status', 'absent')

        # Ordered by workflow group: Build → Classify → GNN → Assemble/Analyze →
        # Validation → Maintenance.  Each entry is (label, help, zero-arg action).
        entries = [
            # -- Build --
            ('Ingest & freeze segments' + _tag(state['has_segments']),
             'Segment transcripts and freeze them to qra.db (offers re-segment from scratch).'
             + ('  [LEGACY — run Migrate first if offered]' if state['is_legacy'] else ''),
             lambda: _action_ingest(config, output_dir)),
            ('Re-segment PURER / therapist content (preserve VAAMR + testsets)',
             'Re-extract therapist (PURER) segments and re-apply PHI scrubbing WITHOUT touching\n'
             'participant VAAMR segments or frozen testsets. Run before re-classifying PURER.',
             _need_segments(lambda: _action_resegment_purer(config, output_dir))),
            ('Add new transcripts (incremental)',
             'Segment + classify only NEW sessions, then re-assemble + re-analyze.\n'
             'Frozen segments and frozen testsets are untouched.',
             lambda: _action_add_data(output_dir, state['config_path'])),
            # -- Classify --
            ('Classify — VAAMR (participant stages)' + _tag(state['has_theme']),
             'LLM multi-run VAAMR classification. Zero-shot + re-run-from-scratch options.',
             _need_segments(lambda: _action_classify_theme(config, output_dir, framework))),
            ('Classify — PURER (therapist moves)' + _tag(state['has_purer']),
             'LLM PURER cue-block classification with reclassify options.',
             _need_segments(lambda: _action_classify_purer(config, output_dir))),
            ('Classify — VCE codebook' + _tag(state['has_codebook']),
             'VCE phenomenology codebook (embedding + LLM) on participant segments.',
             _need_segments(lambda: _action_classify_codebook(config, output_dir))),
            ('Classify — cross-validation (VAAMR × VCE lift)',
             'Theme × codebook co-occurrence lift statistics (needs both overlays).',
             _need_segments(lambda: _action_classify_cv(config, output_dir, framework))),
            # -- GNN --
            ('Train / update GNN consensus layer' + _gnn_tag(gnn_status),
             'Train the graph + run the reliability gate. Adds the GNN to an LLM-only project.',
             _need_segments(lambda: _action_gnn_train(config, output_dir, state))),
            ('Classify — Graph consensus (LLM-free, non-authoritative)' + _gnn_tag(gnn_status),
             'Fill unlabeled data with the trained graph — no LLM calls. Demoted: fills BELOW '
             'the LLM, never overrides it (the probe is the recommended scaler).',
             _need_segments(lambda: _action_classify_gnn(config, output_dir, framework, state))),
            ('View GNN reliability / κ status',
             'Gate verdict: κ(graph,LLM) vs target; is the graph ready for LLM-free scaling?',
             lambda: _action_gnn_status(output_dir)),
            # -- Probe (LLM-free scalable classification; methodology §8.6) --
            ('Probe — train + gate (LLM-free scaler)' + _probe_tag(probe_status),
             'Fit the per-rater ensemble probe + run the participant-grouped gate '
             '(probe↔human / probe↔LLM κ).',
             _need_segments(lambda: _action_probe_train(config, output_dir))),
            ('Probe — classify unlabeled (LLM-free)' + _probe_tag(probe_status),
             'Fill participant segments the LLM never labeled — abstains where unsure; '
             'tagged probe_consensus (below the LLM).',
             _need_segments(lambda: _action_probe_classify(config, output_dir, state))),
            ('View probe reliability / κ status',
             'Gate verdict: probe↔human / probe↔LLM κ; is the probe ready for LLM-free scaling?',
             lambda: _action_probe_status(output_dir)),
            # -- Assemble / Analyze --
            ('Assemble master dataset' + _tag(state['has_master']),
             'Join frozen segments + overlays → master_segments.csv + coded transcripts.',
             lambda: _action_assemble(config, output_dir, framework)),
            ('Analysis & reports' + _tag(state['has_analysis']),
             'Trajectories, summaries, PURER×VAAMR influence, figures (prompts GNN on/off).',
             lambda: _action_analyze(output_dir)),
            # -- Validation --
            ('Testset management',
             f'{len(state["has_testsets"])} frozen testset(s).  Create / list / refresh AI keys.',
             lambda: _action_testset_menu(config, output_dir, state, framework)),
            ('Content-validity testsets',
             f'{len(state["has_cv_testsets"])} CV testset(s).  Create / list / refresh.',
             lambda: _action_cv_menu(config, output_dir, state, framework)),
            ('Refresh validation artifacts',
             'Re-emit human forms, coded transcripts, and AI answer keys (no re-classify).',
             lambda: _action_validate(config, output_dir, framework)),
            # -- Maintenance --
            ('Edit configuration',
             'Change models, LM Studio URL, feature flags, or open config in $EDITOR.',
             _edit_config_and_reload),
            ('Edit speaker anonymization key',
             'Rename / merge / relabel speakers and cascade across every artifact.',
             _edit_anonymization),
        ]
        if state['is_legacy']:
            entries.append((
                'Migrate legacy JSONL → qra.db',
                'Import a pre-SQLite project into qra.db (preview, then confirm).',
                lambda: _action_migrate(output_dir)))

        opts = [(label, desc) for label, desc, _ in entries]
        choice = _menu(f'Project: {os.path.basename(output_dir)}', opts,
                       back_label='Back to main menu')
        if choice == 0:
            return

        try:
            entries[choice - 1][2]()
        except KeyboardInterrupt:
            print()
            _warn('Action interrupted.')
            _pause()
        except Exception as exc:
            print()
            _err(f'Error: {exc}')
            import traceback
            traceback.print_exc()
            _pause()


# ---------------------------------------------------------------------------
# New Project flow
# ---------------------------------------------------------------------------

def _new_project_flow() -> None:
    _section('New Project — Setup Wizard')
    _info(
        'The setup wizard will walk you through:\n'
        '  • Transcript directory and output paths\n'
        '  • Speaker anonymization and filtering\n'
        '  • Segmentation parameters (embedding model, thresholds)\n'
        '  • LLM backend and model selection (or a preset)\n'
        '  • VAAMR / PURER framework options\n'
        '  • VCE codebook classifier (optional)\n'
        '  • Classification runs and confidence tiers\n'
        '  • Validation testsets and content-validity testsets\n'
        '  • Analysis and report generation settings\n'
        '\n'
        'Three modes:  [1] Small/Test  [2] Production  [3] Custom (all params)'
    )
    print()
    if not _confirm('Start setup wizard?'):
        return

    from .setup_wizard import SetupWizard, build_config_from_wizard_data
    wizard = SetupWizard()
    result = wizard.run()
    config_path = result['config_path']
    print()
    _ok(f'Configuration saved to: {config_path}')
    print()

    if _confirm('Run the full pipeline now with this config?'):
        import qra as _qra
        import argparse
        fake_args = argparse.Namespace(
            config=config_path,
            transcript_dir=None, output_dir=None, trial_id=None,
            backend=None, model=None, api_key=None,
            no_auto_analyze=False, test_zeroshot=False, preset=None,
            resume_from=None, framework=None, codebook=None,
            speaker_filter_mode=None, exclude_speakers=None,
            no_theme_labeler=False, run_codebook_classifier=False,
            no_codebook_classifier=False, verbose_segmentation=False,
            zero_shot=False,
        )
        try:
            _qra.cmd_run(fake_args)
        except Exception as exc:
            _err(f'Pipeline error: {exc}')
            import traceback
            traceback.print_exc()
    _pause()


# ---------------------------------------------------------------------------
# About screen
# ---------------------------------------------------------------------------

def _about() -> None:
    _section('About QRA')
    _info(
        'QRA applies two classification frameworks bilaterally to therapy\n'
        'transcripts from the Move-MORE Feasibility Trial.\n'
        '\n'
        'VAAMR — participant segments across a five-stage developmental arc:\n'
        '  0  Vigilance       Attentional capture by pain\n'
        '  1  Avoidance       Attentional skill for experiential escape\n'
        '  2  Attention       Stable volitional presence with somatic experience\n'
        '  3  Metacognition   Reflexive observation of mental processes\n'
        '  4  Reappraisal     Transformation of pain\'s meaning / sensory structure\n'
        '\n'
        'PURER — therapist cue-blocks between consecutive participant turns:\n'
        '  P  Phenomenological  Step-by-step elicitation of practice experience\n'
        '  U  Utilization       Forward application to everyday life\n'
        '  R  Reframing         Repositioning report as a MORE concept\n'
        '  E  Educate/Expectancy Psychoeducation about pain / mindfulness\n'
        '  R  Reinforcement     Selective affirmation of adaptive responses\n'
        '\n'
        'VCE Codebook — optional 54-code phenomenology multi-label enrichment\n'
        'across 6 domains.  Applied only to participant segments.\n'
        '\n'
        'On-disk layout:\n'
        '  qra.db                      Per-project SQLite store — FROZEN segments\n'
        '                              + refreshable classification overlays\n'
        '                              (theme / purer / codebook / cv / gnn) + manifests\n'
        '  04_validation/testsets/     FROZEN human worksheets + refreshable AI keys\n'
        '  03_analysis_data/           Graph-ready CSVs, per-session/participant JSON\n'
        '  05_figures/                 PNG visualizations\n'
        '  06_reports/                 Human-readable text reports\n'
        '\n'
        'CLI surface (all features also reachable from the TUI above):\n'
        '  qra ingest     Segment + freeze transcripts (--fresh = re-segment all)\n'
        '  qra classify   --what vaamr|purer|codebook|cross-validation|all\n'
        '                 --zero-shot (no exemplars)   --fresh (re-classify from scratch)\n'
        '  qra gnn        train / classify (LLM-free) / status   (GNN consensus layer)\n'
        '  qra assemble   Join overlays → master_segments.csv + validation artifacts\n'
        '  qra analyze    Post-hoc analysis (--gnn / --no-gnn to force the GNN layer)\n'
        '  qra add-data   Incrementally add new transcripts (longitudinal)\n'
        '  qra validate   Refresh validation artifacts without reclassifying\n'
        '  qra testset    create / refresh / list  frozen validation testsets\n'
        '  qra cv         create / refresh / list  content-validity testsets\n'
        '  qra migrate    Import a legacy JSONL project into qra.db\n'
        '  qra run        Full pipeline (ingest → classify → assemble → analyze)\n'
        '  qra setup      Interactive configuration wizard (also reachable here)\n'
    )
    _pause()


# ---------------------------------------------------------------------------
# Main TUI entry point
# ---------------------------------------------------------------------------

def run_tui() -> None:
    """Main TUI loop — launched when qra.py is run with no arguments."""
    _banner()
    _info(
        'Welcome to QRA.  Use the numbered menu to navigate.\n'
        'All pipeline stages are available from the "Open Project" path.\n'
        'Press Ctrl-C at any time to cancel an action and return to the menu.'
    )

    while True:
        opts = [
            ('New Project',
             'Run the setup wizard to configure a new QRA project, then\n'
             'optionally execute the full pipeline immediately.'),
            ('Open Project',
             'Load an existing (or legacy) project directory and access\n'
             'all pipeline stages, testset management, and analysis tools.'),
            ('About / Help',
             'Framework reference, on-disk layout, and CLI command summary.'),
        ]
        choice = _menu('Main Menu', opts, back_label='Exit')
        if choice == 0:
            print()
            _info('Goodbye.')
            print()
            sys.exit(0)

        try:
            if choice == 1:
                _new_project_flow()
            elif choice == 2:
                _existing_project_flow()
            elif choice == 3:
                _about()
        except KeyboardInterrupt:
            print()
            _warn('Interrupted — returning to main menu.')
