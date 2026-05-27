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
      has_segments     bool  frozen per-session segments exist
      has_theme        bool  02_meta/classifications/theme_labels.jsonl exists
      has_purer        bool  02_meta/classifications/purer_labels.jsonl exists
      has_codebook     bool  02_meta/classifications/codebook_labels.jsonl exists
      has_master       bool  master_segments*.jsonl exists
      has_testsets     list  names of frozen testsets in 04_validation/testsets/
      has_cv_testsets  list  names of content-validity testsets
      has_analysis     bool  03_analysis_data/ non-empty
      config_path      str|None  path to qra_config.json if found
    """
    from . import output_paths as _paths
    from .legacy_migration import is_legacy_project
    import glob

    def _dir_non_empty(p: str) -> bool:
        return os.path.isdir(p) and bool(os.listdir(p))

    has_segments = _dir_non_empty(_paths.segmented_sessions_dir(output_dir))
    has_theme = os.path.isfile(_paths.classification_overlay_path(output_dir, 'theme'))
    has_purer = os.path.isfile(_paths.classification_overlay_path(output_dir, 'purer'))
    has_codebook = os.path.isfile(_paths.classification_overlay_path(output_dir, 'codebook'))

    ms_dir = _paths.master_segments_dir(output_dir)
    has_master = bool(glob.glob(os.path.join(ms_dir, 'master_segments*.jsonl')))

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
    cv_dir = _paths.cv_testsets_dir(output_dir)
    has_cv_testsets: list = []
    if os.path.isdir(cv_dir):
        for name in sorted(os.listdir(cv_dir)):
            manifest = os.path.join(cv_dir, name, 'manifest.json')
            if os.path.isfile(manifest):
                has_cv_testsets.append(name)

    has_analysis = _dir_non_empty(_paths.analysis_data_dir(output_dir))

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
        config_path=config_path,
    )


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
    from theme_framework.registry import load as _registry_load_fw
    return _registry_load_fw('vaamr')


# ---------------------------------------------------------------------------
# Action helpers — each wraps one pipeline stage with UI framing
# ---------------------------------------------------------------------------

def _action_ingest(config, output_dir: str) -> None:
    from .orchestrator import stage_ingest, SilentObserver
    _section('Stage 1 — Ingest & Freeze Segments')
    _info(
        'Loads transcript files, runs semantic segmentation, and writes frozen\n'
        'per-session segments to 01_transcripts/segmented/<session_id>/segments.jsonl.\n'
        '\n'
        'For legacy projects this also migrates existing testset worksheets into\n'
        'per-testset directories with manifest.json + content-SHA snapshots so\n'
        'that future testset refreshes can verify segment text has not drifted.\n'
        '\n'
        'Frozen segments are NEVER re-written by classify/assemble stages.\n'
        'Re-ingestion requires --reingest (which you can do from the CLI).'
    )
    print()
    if not _confirm('Run ingest now?'):
        return
    segments = stage_ingest(config, output_dir=output_dir, observer=SilentObserver())
    n_sess = len(set(s.session_id for s in segments))
    print()
    _ok(f'{len(segments)} segments across {n_sess} sessions ingested and frozen.')
    _pause()


def _action_classify_theme(config, output_dir: str, framework) -> None:
    from .orchestrator import stage_classify_theme
    _section('Stage 3 — VAAMR Theme Classification')
    _info(
        'Runs zero-shot LLM classification on participant segments using the\n'
        'VAAMR framework (Vigilance → Avoidance → Attention → Metacognition\n'
        '→ Reappraisal).  Multi-run ensemble voting produces confidence tiers.\n'
        '\n'
        'Writes: 02_meta/classifications/theme_labels.jsonl\n'
        'Reads:  frozen segments (01_transcripts/segmented/)\n'
        'Frozen segments and testsets are NOT affected by this stage.\n'
        '\n'
        'ZERO-SHOT MODE: strips exemplar/subtle/adversarial utterances from\n'
        'the prompt so the model classifies from definitions alone — useful for\n'
        'evaluating framework validity without example anchoring.'
    )
    print()

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
    _ok('theme_labels.jsonl written. Run Assemble to rebuild master_segments.')
    _pause()


def _action_classify_purer(config, output_dir: str) -> None:
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
        'Writes: 02_meta/classifications/purer_labels.jsonl\n'
        'Participant VAAMR labels and frozen segments are NOT modified.'
    )
    print()

    zero_shot = _confirm(
        'Use zero-shot prompting? (definitions only, no examples)', default=False
    )
    if zero_shot:
        config.purer_classification.zero_shot_prompt = True
        _ok('Zero-shot mode ON.')
    print()

    model = getattr(config.purer_classification, 'model', None) or config.theme_classification.model
    _info(f'Model:   {model}  (n_runs=1, single-run)')
    print()

    if not _confirm('Run PURER classification now?'):
        return
    stage_classify_purer(config, output_dir=output_dir)
    print()
    _ok('purer_labels.jsonl written. Run Assemble to rebuild master_segments.')
    _pause()


def _action_assemble(config, output_dir: str, framework) -> None:
    from .orchestrator import stage_assemble
    _section('Stage 6 — Assemble Master Dataset')
    _info(
        'Joins frozen segments + all classification overlays into a single\n'
        'master_segments.jsonl / master_segments.csv with confidence tiers.\n'
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
    stage_assemble(config, framework, output_dir=output_dir)
    print()
    _ok('master_segments.jsonl written and validation artifacts refreshed.')
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
    if not _confirm('Run analysis now?'):
        return
    run_analysis(output_dir)
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
    stage_assemble(config, framework, output_dir=output_dir)
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
        except Exception as exc:
            _err(f'{name}: {exc}')
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
        from theme_framework.registry import load as _registry_load_fw
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
    from . import output_paths as _paths
    _section('Content-Validity Testsets')
    if not state['has_cv_testsets']:
        _warn('No content-validity testsets found.')
        _pause()
        return
    for name in state['has_cv_testsets']:
        m_path = _paths.cv_testset_manifest_path(output_dir, name)
        try:
            with open(m_path) as fh:
                m = json.load(fh)
            n = m.get('n_items', '?')
            created = m.get('created_at', 'unknown')[:10]
            fw = m.get('framework', '?')
            print(f'  • {name}  ({fw}, {n} items, created {created})')
        except Exception:
            print(f'  • {name}  [unreadable manifest]')
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
        ('Toggle codebook (VCE) classifier', 'Enable or disable the 59-code phenomenology codebook.'),
        ('Toggle PURER classifier', 'Enable or disable PURER therapist cue classification.'),
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
        model = _ask('Primary model ID', raw.get('theme_classification', {}).get('model', ''))
        raw.setdefault('theme_classification', {})['model'] = model
        per_run_raw = _ask('Per-run models (comma-separated, blank = use primary only)', '')
        if per_run_raw:
            raw['theme_classification']['per_run_models'] = [m.strip() for m in per_run_raw.split(',')]
        changed = True
    elif choice == 3:
        n = int(_ask('Number of classification runs', str(raw.get('theme_classification', {}).get('n_runs', 3))))
        raw.setdefault('theme_classification', {})['n_runs'] = n
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
        # Refresh state each loop so status marks update after actions
        state = _detect_state(output_dir)
        _print_project_state(state)

        # Build contextual action list — flag unavailable actions
        def _tag(cond, suffix='  ✓ done'):
            return suffix if cond else ''

        opts = [
            ('Ingest & Freeze Segments' + _tag(state['has_segments']),
             'Segment transcripts, freeze per-session data, migrate legacy testsets.'
             + (' [LEGACY — required first step]' if state['is_legacy'] else '')),
            ('Classify — VAAMR (participant stages)' + _tag(state['has_theme']),
             'Run LLM zero-shot VAAMR classification. Zero-shot mode available.\n'
             'Requires frozen segments (step above).'),
            ('Classify — PURER (therapist moves)' + _tag(state['has_purer']),
             'Run LLM PURER cue-block classification on therapist segments.\n'
             'Requires frozen segments.'),
            ('Assemble master dataset' + _tag(state['has_master']),
             'Join frozen segments + overlays → master_segments.jsonl + coded transcripts.'),
            ('Analysis & Reports' + _tag(state['has_analysis']),
             'Per-participant trajectories, per-session summaries, PURER×VAAMR\n'
             'influence analysis, figures, and LLM narrative summaries.'),
            ('Testset Management',
             f'{len(state["has_testsets"])} frozen testset(s).  Create, list, or refresh AI answer keys.'),
            ('Content-Validity Testsets',
             f'{len(state["has_cv_testsets"])} CV testset(s).  Create, list, or refresh AI graded reports.'),
            ('Refresh Validation Artifacts',
             'Re-emit human forms, coded transcripts, and AI answer keys without re-classifying.'),
            ('Edit Configuration',
             'Change models, LM Studio URL, feature flags, or open config in $EDITOR.'),
        ]

        choice = _menu(f'Project: {os.path.basename(output_dir)}', opts, back_label='Back to main menu')
        if choice == 0:
            return

        try:
            if choice == 1:
                _action_ingest(config, output_dir)
            elif choice == 2:
                if not state['has_segments']:
                    _warn('No frozen segments yet — run Ingest first.')
                    _pause()
                else:
                    _action_classify_theme(config, output_dir, framework)
            elif choice == 3:
                if not state['has_segments']:
                    _warn('No frozen segments yet — run Ingest first.')
                    _pause()
                else:
                    _action_classify_purer(config, output_dir)
            elif choice == 4:
                _action_assemble(config, output_dir, framework)
            elif choice == 5:
                _action_analyze(output_dir)
            elif choice == 6:
                _action_testset_menu(config, output_dir, state, framework)
            elif choice == 7:
                _action_cv_menu(config, output_dir, state, framework)
            elif choice == 8:
                _action_validate(config, output_dir, framework)
            elif choice == 9:
                _action_edit_config(state['config_path'])
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
        'VCE Codebook — optional 59-code phenomenology multi-label enrichment\n'
        'across 7 domains.  Applied only to participant segments.\n'
        '\n'
        'On-disk layout:\n'
        '  01_transcripts/segmented/   FROZEN — one segments.jsonl per session\n'
        '  02_meta/classifications/    Refreshable overlays (theme / purer / codebook)\n'
        '  04_validation/testsets/     FROZEN human worksheets + refreshable AI keys\n'
        '  03_analysis_data/           Graph-ready CSVs, per-session/participant JSON\n'
        '  05_figures/                 PNG visualizations\n'
        '  06_reports/                 Human-readable text reports\n'
        '\n'
        'CLI surface (all features also reachable from the TUI above):\n'
        '  qra ingest     Segment + freeze transcripts\n'
        '  qra classify   Overlay: --what theme|purer|codebook|cross-validation|all\n'
        '                          --zero-shot  (strip exemplars from prompts)\n'
        '  qra assemble   Join overlays → master_segments + validation artifacts\n'
        '  qra analyze    Post-hoc analysis on assembled dataset\n'
        '  qra validate   Refresh validation artifacts without reclassifying\n'
        '  qra testset    create / refresh / list  frozen validation testsets\n'
        '  qra cv         create / refresh / list  content-validity testsets\n'
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
