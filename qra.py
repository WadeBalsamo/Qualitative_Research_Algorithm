#!/usr/bin/env python3
"""
qra.py
------
Unified CLI entry point for the QRA classification pipeline.

Subcommands:
    setup     Interactive wizard that saves a config JSON
    run       Execute pipeline (always auto mode)
    analyze   Post-hoc analysis on an existing output directory
    testsets  Generate validation test set worksheets from existing output
"""

import argparse
import json
import os
import sys
import traceback

# Ensure the package root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =========================================================================
# Shared argument helpers
# =========================================================================

def _add_common_args(parser: argparse.ArgumentParser):
    """Add arguments shared by `run` subcommand."""
    # Config file
    parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to saved config JSON (from `qra setup`)',
    )

    # Input/output
    parser.add_argument('--transcript-dir', default=None)
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--trial-id', default=None)

    # Backend & model
    parser.add_argument(
        '--backend',
        default=None,
        choices=['openrouter', 'replicate', 'huggingface', 'ollama', 'lmstudio'],
    )
    parser.add_argument(
        '--lmstudio-url', default=None,
        help='LM Studio server base URL (default: http://127.0.0.1:1234/v1)',
    )
    parser.add_argument('--model', default=None)
    parser.add_argument(
        '--models', nargs='+', default=None,
        help='Multiple model IDs for cross-referencing',
    )
    parser.add_argument('--api-key', default=None)
    parser.add_argument('--replicate-api-token', default=None)

    # Framework & codebook
    parser.add_argument(
        '--framework', default=None,
        help='Theme framework: "vaamr" (default) or path to custom JSON',
    )
    parser.add_argument(
        '--codebook', default=None,
        help='Codebook: "phenomenology" (default) or path to custom JSON',
    )

    # Classification settings
    parser.add_argument('--n-runs', type=int, default=None)
    parser.add_argument('--temperature', type=float, default=None)

    # Confidence thresholds
    parser.add_argument('--high-confidence-threshold', type=float, default=None)
    parser.add_argument('--medium-confidence-threshold', type=float, default=None)

    # Feature flags
    parser.add_argument('--no-theme-labeler', action='store_true')
    parser.add_argument('--run-codebook-classifier', action='store_true')
    parser.add_argument('--no-codebook-classifier', action='store_true')
    parser.add_argument(
        '--verbose-segmentation', action='store_true',
        help='Write process_log.txt with every LLM prompt/response and segmentation step',
    )

    # Embedding classifier settings
    parser.add_argument('--no-two-pass', action='store_true')
    parser.add_argument(
        '--embedding-model', default=None,
        help='Sentence-transformer model for codebook embedding classification '
             '(default: Qwen/Qwen3-Embedding-8B; use all-MiniLM-L6-v2 for lightweight)',
    )
    parser.add_argument('--exemplar-import-path', default=None)
    parser.add_argument('--criteria-weight', type=float, default=None)
    parser.add_argument('--exemplar-weight', type=float, default=None)
    parser.add_argument('--exemplar-confidence-threshold', type=float, default=None)
    parser.add_argument('--max-exemplar-tokens', type=int, default=None)

    # Speaker filtering
    parser.add_argument(
        '--speaker-filter-mode', default=None,
        choices=['none', 'exclude'],
        help='none: classify all | exclude: drop listed speakers',
    )
    parser.add_argument(
        '--exclude-speakers', nargs='+', default=None, metavar='SPEAKER',
        help='Speaker labels to exclude (use with --speaker-filter-mode exclude)',
    )

    # Checkpoint
    parser.add_argument('--resume-from', default=None)

    # Post-pipeline analysis (on by default; use --no-auto-analyze to skip)
    parser.add_argument(
        '--no-auto-analyze',
        action='store_true',
        help='Skip the results analysis stage after the pipeline completes',
    )

    # Zero-shot test mode
    parser.add_argument(
        '--test-zeroshot',
        action='store_true',
        help='Run zero-shot LLM classification against the content validity test set '
             '(skips full pipeline; writes graded report to 05_validation/)',
    )
    parser.add_argument(
        '--preset',
        choices=['small', 'production'],
        default=None,
        help='Use a predefined model ensemble for --test-zeroshot '
             '(small = nemotron-nano/gemma-e2b/qwen3-8b; '
             'production = qwen3-80b/gemma-31b/nemotron-super). '
             'Requires --lmstudio-url.',
    )


# =========================================================================
# Config building
# =========================================================================

def _load_framework(framework_arg):
    """Load a ThemeFramework from preset name or JSON path."""
    if framework_arg is None or framework_arg == 'vaamr':
        from theme_framework.vaamr import get_vaamr_framework
        return get_vaamr_framework()
    if framework_arg == 'vammr':
        raise ValueError(
            'The "vammr" preset is deprecated. Use "vaamr" or a custom framework JSON.'
        )
    # Custom JSON path
    with open(framework_arg) as f:
        fw_data = json.load(f)
    from theme_framework.theme_schema import ThemeFramework, ThemeDefinition
    themes = []
    for t in fw_data.get('themes', []):
        themes.append(ThemeDefinition(
            theme_id=t['theme_id'],
            key=t['key'],
            name=t['name'],
            short_name=t.get('short_name', t['name']),
            prompt_name=t.get('prompt_name', t['name'].lower()),
            definition=t['definition'],
            prototypical_features=t.get('prototypical_features', []),
            distinguishing_criteria=t.get('distinguishing_criteria', ''),
            exemplar_utterances=t.get('exemplar_utterances', []),
        ))
    return ThemeFramework(
        name=fw_data.get('framework', 'custom'),
        version=fw_data.get('version', '1.0'),
        description=fw_data.get('description', ''),
        themes=themes,
    )


def _load_codebook(codebook_arg):
    """Load a Codebook from preset name or JSON path. Returns None if disabled."""
    if codebook_arg is None or codebook_arg == 'phenomenology':
        from codebook.phenomenology_codebook import get_phenomenology_codebook
        return get_phenomenology_codebook()

    with open(codebook_arg) as f:
        cb_data = json.load(f)
    from codebook.codebook_schema import Codebook, CodeDefinition
    codes = []
    for c in cb_data.get('codes', []):
        codes.append(CodeDefinition(
            code_id=c['code_id'],
            category=c['category'],
            domain=c['domain'],
            description=c['description'],
            subcodes=c.get('subcodes', []),
            inclusive_criteria=c.get('inclusive_criteria', ''),
            exclusive_criteria=c.get('exclusive_criteria', ''),
            exemplar_utterances=c.get('exemplar_utterances', []),
        ))
    return Codebook(
        name=cb_data.get('name', 'custom'),
        version=cb_data.get('version', '1.0'),
        description=cb_data.get('description', ''),
        codes=codes,
    )


def _build_config(args):
    """Build PipelineConfig from CLI args, optionally merging a config file."""
    from process.config import PipelineConfig

    # Start from config file if provided
    if args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        # Use from_json for proper nested reconstruction
        config = PipelineConfig.from_json(_flatten_wizard_config(file_data))
    else:
        config = PipelineConfig()

    # CLI overrides (only override if explicitly provided)
    if args.transcript_dir is not None:
        config.transcript_dir = args.transcript_dir
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    if args.trial_id is not None:
        config.trial_id = args.trial_id
    if args.resume_from is not None:
        config.resume_from = args.resume_from

    # Speaker filter
    sf_mode = getattr(args, 'speaker_filter_mode', None)
    exclude_spk = getattr(args, 'exclude_speakers', None)
    if sf_mode is not None:
        config.speaker_filter.mode = sf_mode
    if exclude_spk is not None:
        config.speaker_filter.mode = 'exclude'
        config.speaker_filter.speakers = exclude_spk

    # Feature flags
    if args.no_theme_labeler:
        config.run_theme_labeler = False
    if args.run_codebook_classifier:
        config.run_codebook_classifier = True
    if args.no_codebook_classifier:
        config.run_codebook_classifier = False
    if getattr(args, 'verbose_segmentation', False):
        config.segmentation.verbose_segmentation = True

    # Backend & model
    tc = config.theme_classification
    if args.backend is not None:
        tc.backend = args.backend
    if args.model is not None:
        tc.model = args.model
    lmstudio_url = getattr(args, 'lmstudio_url', None)
    if lmstudio_url is not None:
        tc.lmstudio_base_url = lmstudio_url
    if args.models is not None:
        tc.models = args.models
    if args.n_runs is not None:
        tc.n_runs = args.n_runs
    if args.temperature is not None:
        tc.temperature = args.temperature

    # API keys: CLI > env > existing config
    if args.api_key is not None:
        tc.api_key = args.api_key
    elif not tc.api_key:
        tc.api_key = os.environ.get('OPENROUTER_API_KEY', '')

    if args.replicate_api_token is not None:
        tc.replicate_api_token = args.replicate_api_token
    elif not tc.replicate_api_token:
        tc.replicate_api_token = os.environ.get('REPLICATE_API_TOKEN', '')

    # Embedding settings
    emb = config.codebook_embedding
    if args.no_two_pass:
        emb.two_pass = False
    if getattr(args, 'embedding_model', None) is not None:
        emb.embedding_model = args.embedding_model
    if args.exemplar_import_path is not None:
        emb.exemplar_import_path = args.exemplar_import_path
    if args.criteria_weight is not None:
        emb.criteria_weight = args.criteria_weight
    if args.exemplar_weight is not None:
        emb.exemplar_weight = args.exemplar_weight
    if args.exemplar_confidence_threshold is not None:
        emb.exemplar_confidence_threshold = args.exemplar_confidence_threshold
    if args.max_exemplar_tokens is not None:
        emb.max_exemplar_tokens = args.max_exemplar_tokens

    # Confidence thresholds
    if args.high_confidence_threshold is not None:
        config.confidence_tiers.high_confidence = args.high_confidence_threshold
    if args.medium_confidence_threshold is not None:
        config.confidence_tiers.medium_min_confidence = args.medium_confidence_threshold

    if getattr(args, 'no_auto_analyze', False):
        config.auto_analyze = False

    return config


def _flatten_wizard_config(data: dict) -> dict:
    """Flatten wizard-format config (nested pipeline/theme_classification keys)
    into PipelineConfig.from_json-compatible dict."""
    result = {}

    # Lift pipeline-level keys to top level
    pipeline = data.get('pipeline', {})
    for key in ('transcript_dir', 'output_dir', 'trial_id',
                'run_theme_labeler', 'run_codebook_classifier'):
        if key in pipeline:
            result[key] = pipeline[key]

    # Pass through sub-config dicts directly
    for key in ('segmentation', 'speaker_filter', 'theme_classification', 'codebook_embedding',
                'codebook_llm', 'codebook_ensemble', 'validation', 'confidence_tiers',
                'test_sets', 'purer_classification', 'purer_cue'):
        if key in data:
            result[key] = data[key]

    # Copy top-level keys that are already flat
    for key in ('resume_from', 'autoresearch_dir', 'run_purer_labeler'):
        if key in data:
            result[key] = data[key]

    return result


# =========================================================================
# Subcommands
# =========================================================================

def cmd_setup(args):
    """Interactive setup wizard."""
    from process.setup_wizard import SetupWizard, build_config_from_wizard_data

    wizard = SetupWizard()
    result = wizard.run()
    config_path = result['config_path']

    if _prompt_yes_no_simple("Run pipeline now?", True):
        config = build_config_from_wizard_data(result['config_data'])
        framework_spec = result['config_data'].get('framework', {})
        framework = _load_framework(framework_spec.get('custom_path') or framework_spec.get('preset', 'vaamr'))

        codebook = None
        if config.run_codebook_classifier:
            codebook_spec = result['config_data'].get('codebook', {})
            codebook = _load_codebook(codebook_spec.get('custom_path') or codebook_spec.get('preset', 'phenomenology'))

        _execute_pipeline(config, framework, codebook)


def cmd_run(args):
    """Execute pipeline."""
    if getattr(args, 'test_zeroshot', False):
        cmd_test_zeroshot(args)
        return

    config = _build_config(args)

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            file_data = json.load(f)
        fw = file_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vaamr')
    framework = _load_framework(framework_arg)

    # Load codebook
    codebook = None
    if config.run_codebook_classifier:
        codebook_arg = args.codebook
        if codebook_arg is None and args.config:
            with open(args.config) as f:
                file_data = json.load(f)
            cb = file_data.get('codebook', {})
            codebook_arg = cb.get('custom_path') or cb.get('preset', 'phenomenology')
        codebook = _load_codebook(codebook_arg)

    # HuggingFace model preloading
    if config.theme_classification.backend == 'huggingface':
        _preload_huggingface_models(config)

    _execute_pipeline(config, framework, codebook)


def _resolve_test_rater_config(args) -> dict:
    """
    Build a rater config dict for --test-zeroshot.

    Preset mode uses setup_wizard presets (lmstudio, 3-model ensemble).
    Custom mode uses CLI --backend/--model/--models/--api-key args.
    """
    from process.setup_wizard import _PRESET_SMALL, _PRESET_PRODUCTION

    preset_name = getattr(args, 'preset', None)
    if preset_name in ('small', 'production'):
        preset = _PRESET_SMALL if preset_name == 'small' else _PRESET_PRODUCTION
        lmstudio_url = getattr(args, 'lmstudio_url', None) or 'http://127.0.0.1:1234/v1'
        api_key = getattr(args, 'api_key', None) or os.environ.get('OPENROUTER_API_KEY', '')
        return {
            'backend': 'lmstudio',
            'model': preset['primary_model'],
            'per_run_models': list(preset['per_run_models']),
            'n_runs': preset['n_runs'],
            'temperature': preset['temperature'],
            'api_key': api_key,
            'replicate_api_token': '',
            'lmstudio_base_url': lmstudio_url,
            'preset_label': preset['label'],
        }

    # Custom: derive from CLI args
    backend = getattr(args, 'backend', None) or 'openrouter'
    model = getattr(args, 'model', None) or 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'
    models = getattr(args, 'models', None) or []
    n_runs = getattr(args, 'n_runs', None)
    temperature = getattr(args, 'temperature', None) or 0.0
    api_key = getattr(args, 'api_key', None) or os.environ.get('OPENROUTER_API_KEY', '')
    replicate_token = getattr(args, 'replicate_api_token', None) or os.environ.get('REPLICATE_API_TOKEN', '')
    lmstudio_url = getattr(args, 'lmstudio_url', None) or 'http://127.0.0.1:1234/v1'

    per_run_models: list = []
    if models and len(models) >= 2:
        per_run_models = list(models)
        resolved_n_runs = n_runs if n_runs is not None else len(per_run_models)
    else:
        per_run_models = []
        resolved_n_runs = 1

    return {
        'backend': backend,
        'model': model,
        'per_run_models': per_run_models,
        'n_runs': resolved_n_runs,
        'temperature': temperature,
        'api_key': api_key,
        'replicate_api_token': replicate_token,
        'lmstudio_base_url': lmstudio_url,
        'preset_label': None,
    }


def _cv_items_to_segments(test_items: list):
    """Convert content validity test item dicts to minimal Segment objects."""
    from classification_tools.data_structures import Segment
    segments = []
    for item in test_items:
        seg = Segment(
            segment_id=item['test_item_id'],
            trial_id='cv_test',
            participant_id='cv_participant',
            session_id='cv_test',
            session_number=1,
            cohort_id=None,
            session_variant='',
            segment_index=int(item['test_item_id'].split('_')[1]),
            start_time_ms=0,
            end_time_ms=0,
            total_segments_in_session=len(test_items),
            speaker='participant',
            text=item['text'],
            word_count=len(item['text'].split()),
        )
        segments.append(seg)
    return segments


def _write_zeroshot_report(
    test_items: list,
    results_all: dict,
    rater_ids: list,
    framework,
    output_dir: str,
) -> str:
    """
    Write a human-readable graded report for the zero-shot content validity test.

    Format mirrors 05_validation/ artifacts (78-char separators, same header style).
    Returns the written file path.
    """
    import datetime as _dt
    import textwrap

    from process import output_paths as _paths

    _W = 78
    id_to_name = {t.theme_id: t.short_name for t in framework.themes} if framework else {}

    def _sname(stage_id):
        if stage_id is None:
            return 'ABSTAIN'
        return id_to_name.get(stage_id, str(stage_id))

    def _fconf(c):
        if c is None:
            return '?'
        return f'{float(c):.2f}'

    # ---------------------------------------------------------------------------
    # Pass-type helper: 'primary', 'secondary', or None.
    # A secondary match still counts as correct — if a rater assigned two
    # stages and the expected one is among them, that should not penalise the
    # model.  This mirrors pipeline behaviour where both primary and secondary
    # labels are recorded and considered meaningful.
    # ---------------------------------------------------------------------------
    def _pass_type(primary, secondary, expected):
        if primary is not None and primary == expected:
            return 'primary'
        if secondary is not None and secondary == expected:
            return 'secondary'
        return None

    # Build per-item result rows
    rows = []
    for item in test_items:
        iid = item['test_item_id']
        expected = item['expected_stage']
        difficulty = item['difficulty']
        text = item['text']

        raw = results_all.get(iid, {})
        rater_votes = raw.get('rater_votes', [])
        consensus = raw.get('consensus', {})

        # vote_single_label: top-level keys are 'primary_stage', 'secondary_stage',
        # 'consensus_vote' (not 'stage'/'vote')
        cons_primary = consensus.get('primary_stage')
        cons_secondary = consensus.get('secondary_stage')
        cons_cv = consensus.get('consensus_vote')  # int | 'ABSTAIN' | None (split)
        cons_pt = _pass_type(cons_primary, cons_secondary, expected)

        # Per-rater votes indexed by rater id
        votes_by_rater = {rv.get('rater', ''): rv for rv in rater_votes}

        rows.append({
            'item_id': iid,
            'expected': expected,
            'difficulty': difficulty,
            'text': text,
            'rater_votes': votes_by_rater,
            'cons_primary': cons_primary,
            'cons_secondary': cons_secondary,
            'cons_cv': cons_cv,
            'cons_pt': cons_pt,           # 'primary' | 'secondary' | None
            'cons_pass': cons_pt is not None,
        })

    # ---------------------------------------------------------------------------
    # Score helpers: count passes (primary OR secondary) and secondary-only passes
    # ---------------------------------------------------------------------------
    tiers = ['clear', 'subtle', 'adversarial']

    def _score(rows_subset, rater=None):
        """Returns (n_pass, n_secondary_only, n_total)."""
        n_pass = 0
        n_sec = 0
        total = len(rows_subset)
        for r in rows_subset:
            if rater is not None:
                rv = r['rater_votes'].get(rater, {})
                vote = rv.get('vote', 'ERROR')
                pt = _pass_type(
                    rv.get('stage') if vote == 'CODED' else None,
                    rv.get('secondary_stage') if vote == 'CODED' else None,
                    r['expected'],
                )
            else:
                pt = r['cons_pt']
            if pt == 'primary':
                n_pass += 1
            elif pt == 'secondary':
                n_pass += 1
                n_sec += 1
        return n_pass, n_sec, total

    def _cell(n_pass, n_sec, total):
        if total == 0:
            return 'N/A'
        pct = 100 * n_pass / total
        base = f'{n_pass}/{total} ({pct:.0f}%)'
        return base + (f' ~{n_sec}' if n_sec else '')

    # Pad rater ids to consistent display width
    max_rater_len = max((len(r) for r in rater_ids), default=10)
    rater_col_w = max(max_rater_len, len('CONSENSUS'))

    vdir = _paths.validation_dir(output_dir)
    os.makedirs(vdir, exist_ok=True)
    out_path = os.path.join(vdir, 'content_validity_zeroshot_results.txt')

    with open(out_path, 'w', encoding='utf-8') as fh:
        # ---- Header ----
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY ZERO-SHOT CLASSIFICATION RESULTS\n')
        fh.write('=' * _W + '\n')
        fh.write(f'Framework: {framework.name}   Version: {framework.version}   '
                 f'Themes: {len(framework.themes)}\n')
        fh.write(f'Generated: {_dt.date.today().isoformat()}\n')
        fh.write('=' * _W + '\n\n')

        # ---- Rater lineup ----
        fh.write('RATER LINEUP\n')
        fh.write('-' * _W + '\n')
        if len(rater_ids) == 1:
            fh.write(f'  Single rater: {rater_ids[0]}\n')
        else:
            for i, rid in enumerate(rater_ids, start=1):
                fh.write(f'  Run {i}: {rid}\n')
        fh.write('\n')

        # ---- Score summary table ----
        fh.write('SCORE SUMMARY\n')
        fh.write('-' * _W + '\n')
        fh.write(
            '  Scoring: [PASS] = expected stage is primary label; '
            '[PASS~] = expected stage\n'
            '  is secondary label (both count as correct). '
            '~N in a cell = secondary-match count.\n\n'
        )
        header_cols = ['Overall'] + tiers
        col_w = 16
        header_line = f'  {"Rater":{rater_col_w + 2}}'
        for h in header_cols:
            header_line += f'  {h:<{col_w}}'
        fh.write(header_line.rstrip() + '\n')
        fh.write('  ' + '-' * (rater_col_w + 2 + (col_w + 2) * len(header_cols)) + '\n')

        all_raters_for_summary = rater_ids + ['CONSENSUS']
        for rater in all_raters_for_summary:
            is_cons = (rater == 'CONSENSUS')
            label = rater if not is_cons else 'CONSENSUS'
            n_p, n_s, t = _score(rows, None if is_cons else rater)
            row_line = f'  {label:{rater_col_w + 2}}  {_cell(n_p, n_s, t):<{col_w}}'
            for tier in tiers:
                tier_rows = [r for r in rows if r['difficulty'] == tier]
                n_p2, n_s2, t2 = _score(tier_rows, None if is_cons else rater)
                row_line += f'  {_cell(n_p2, n_s2, t2):<{col_w}}'
            fh.write(row_line.rstrip() + '\n')

        fh.write('\n')
        fh.write('  By stage (consensus):\n')
        for theme in sorted(framework.themes, key=lambda t: t.theme_id):
            stage_rows = [r for r in rows if r['expected'] == theme.theme_id]
            n_p, n_s, t = _score(stage_rows, None)
            pct = f'{100*n_p/t:.0f}%' if t else '?'
            sec_note = f'  (~{n_s} secondary)' if n_s else ''
            fh.write(
                f'    Stage {theme.theme_id} {theme.short_name:<16} '
                f'{n_p}/{t} ({pct}){sec_note}\n'
            )
        fh.write('\n')

        # ---- Item-by-item ----
        fh.write('=' * _W + '\n')
        fh.write('ITEM-BY-ITEM RESULTS\n')
        fh.write('=' * _W + '\n\n')

        for row in rows:
            exp_name = _sname(row['expected'])
            if row['cons_pt'] == 'primary':
                cons_label = '[PASS]'
            elif row['cons_pt'] == 'secondary':
                cons_label = '[PASS~]'
            else:
                cons_label = '[FAIL]'

            fh.write('=' * _W + '\n')
            fh.write(
                f"[{row['item_id']}]  Tier: {row['difficulty']:<12}  "
                f"Expected: {exp_name}   Consensus: {cons_label}\n"
            )
            fh.write('-' * _W + '\n')
            for line in textwrap.wrap(
                f'"{row["text"]}"', width=_W - 2,
                initial_indent='  ', subsequent_indent='  ',
            ) or ['  ']:
                fh.write(line + '\n')
            fh.write('\n')

            for rid in rater_ids:
                rv = row['rater_votes'].get(rid, {})
                vote = rv.get('vote', 'ERROR')
                primary = rv.get('stage') if vote == 'CODED' else None
                secondary = rv.get('secondary_stage') if vote == 'CODED' else None
                conf = rv.get('confidence')
                sec_conf = rv.get('secondary_confidence')
                just = (rv.get('justification') or '').strip()

                pt = _pass_type(primary, secondary, row['expected'])
                result_tag = '[PASS]' if pt == 'primary' else ('[PASS~]' if pt == 'secondary' else '[FAIL]')

                if vote == 'CODED':
                    stage_str = _sname(primary)
                    if secondary is not None:
                        stage_str += f' / {_sname(secondary)}'
                    conf_str = f'  conf={_fconf(conf)}'
                    if secondary is not None and sec_conf is not None:
                        conf_str += f'/{_fconf(sec_conf)}'
                else:
                    stage_str = vote
                    conf_str = ''

                label_col = f'[{rid}]'
                fh.write(
                    f'  {label_col:{rater_col_w + 2}}  '
                    f'{stage_str:<22}{conf_str}  {result_tag}\n'
                )
                # Show justification when this item has any failure (rater or consensus)
                if just and (pt != 'primary' or not row['cons_pass']):
                    for line in textwrap.wrap(
                        just, width=_W - 8,
                        initial_indent='      → ', subsequent_indent='        ',
                    ):
                        fh.write(line + '\n')

            # Consensus line with primary + secondary
            if row['cons_primary'] is not None:
                cons_stage_str = _sname(row['cons_primary'])
                if row['cons_secondary'] is not None:
                    cons_stage_str += f' / {_sname(row["cons_secondary"])}'
            elif row['cons_cv'] == 'ABSTAIN':
                cons_stage_str = 'ABSTAIN'
            elif row['cons_cv'] is None:
                cons_stage_str = 'SPLIT'
            else:
                cons_stage_str = 'ERROR'

            fh.write(
                f'  {"CONSENSUS":{rater_col_w + 2}}  '
                f'{cons_stage_str:<22}              {cons_label}\n'
            )
            fh.write('\n')

    return out_path


def cmd_test_zeroshot(args):
    """Run zero-shot LLM classification against the content validity test set."""
    from classification_tools.llm_classifier import (
        create_content_validity_test_set,
        classify_segments_zero_shot,
    )
    from theme_framework.config import ThemeClassificationConfig

    framework_arg = getattr(args, 'framework', None)
    framework = _load_framework(framework_arg)

    rater_cfg = _resolve_test_rater_config(args)

    output_dir = getattr(args, 'output_dir', None) or './data/zeroshot_test'
    os.makedirs(output_dir, exist_ok=True)

    print('\n' + '=' * 70)
    print('QRA CONTENT VALIDITY ZERO-SHOT TEST')
    print('=' * 70)
    print(f'  Framework : {framework.name} v{framework.version} ({len(framework.themes)} themes)')
    if rater_cfg.get('preset_label'):
        print(f'  Preset    : {rater_cfg["preset_label"]}')
    if rater_cfg['per_run_models']:
        print(f'  Raters    : {len(rater_cfg["per_run_models"])} models')
        for i, m in enumerate(rater_cfg['per_run_models'], start=1):
            print(f'    Run {i}: {m}')
    else:
        print(f'  Model     : {rater_cfg["model"]}')
    print(f'  Backend   : {rater_cfg["backend"]}')
    print()

    # Build test items and fake segments
    test_items = create_content_validity_test_set(framework)
    print(f'  Test items: {len(test_items)} '
          f'({sum(1 for i in test_items if i["difficulty"]=="clear")} clear, '
          f'{sum(1 for i in test_items if i["difficulty"]=="subtle")} subtle, '
          f'{sum(1 for i in test_items if i["difficulty"]=="adversarial")} adversarial)')
    print()

    segments = _cv_items_to_segments(test_items)

    # Write static companion documents before classification begins
    from process.assembly import (
        export_content_validity_human_worksheet,
        export_content_validity_definition_key,
        export_content_validity_answer_key,
    )
    export_content_validity_human_worksheet(test_items, framework, output_dir)
    export_content_validity_definition_key(framework, output_dir)
    export_content_validity_answer_key(test_items, framework, output_dir)
    print('  Companion documents written to 04_validation/:')
    print('    content_validity_human_worksheet.txt')
    print('    content_validity_definition_key.txt')
    print('    content_validity_answer_key.txt')
    print()

    # Build classification config
    checkpoint_dir = os.path.join(output_dir, '_cv_checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    tc = ThemeClassificationConfig(
        model=rater_cfg['model'],
        per_run_models=rater_cfg['per_run_models'],
        n_runs=rater_cfg['n_runs'],
        temperature=rater_cfg['temperature'],
        backend=rater_cfg['backend'],
        api_key=rater_cfg['api_key'],
        replicate_api_token=rater_cfg['replicate_api_token'],
        lmstudio_base_url=rater_cfg['lmstudio_base_url'],
        output_dir=checkpoint_dir,
        context_window_segments=0,
        min_classifiable_words=0,
        randomize_codebook=False,
    )

    # LM Studio: wait for server before starting
    if tc.backend == 'lmstudio':
        _wait_for_lmstudio(tc.lmstudio_base_url)

    results_all, _ = classify_segments_zero_shot(segments, framework, tc)

    # Determine rater IDs that appear in results
    seen_raters = []
    if rater_cfg['per_run_models']:
        seen_raters = list(rater_cfg['per_run_models'])
    else:
        seen_raters = [rater_cfg['model']]

    report_path = _write_zeroshot_report(
        test_items=test_items,
        results_all=results_all,
        rater_ids=seen_raters,
        framework=framework,
        output_dir=output_dir,
    )

    # Quick summary to stdout
    from process import output_paths as _paths
    n_pass = n_secondary = 0
    for item in test_items:
        raw = results_all.get(item['test_item_id'], {})
        cons = raw.get('consensus', {})
        exp = item['expected_stage']
        primary = cons.get('primary_stage')
        secondary = cons.get('secondary_stage')
        if primary == exp:
            n_pass += 1
        elif secondary is not None and secondary == exp:
            n_pass += 1
            n_secondary += 1
    n_total = len(test_items)
    pct = 100 * n_pass / n_total if n_total else 0
    sec_note = f'  ({n_secondary} as secondary label)' if n_secondary else ''
    print(f'\n  Consensus score: {n_pass}/{n_total} ({pct:.0f}%){sec_note}')
    print(f'  Report: {os.path.relpath(report_path)}')


def cmd_analyze(args):
    """Run post-hoc results analysis on an existing pipeline output directory."""
    from analysis.runner import run_analysis

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("QRA RESULTS ANALYSIS")
    print("=" * 70)
    print(f"  Output dir: {output_dir}")
    print()

    result = run_analysis(output_dir, verbose=True)

    print(f"\nAnalysis complete.")
    print(f"  {result['n_segments']} segments | "
          f"{result['n_participants']} participants | "
          f"{result['n_sessions']} sessions")
    print(f"  Reports: {output_dir}/06_reports/ and {output_dir}/03_analysis_data/")
    print(f"  Files generated: {len(result['files_generated'])}")


def cmd_testsets(args):
    """Generate validation test set worksheets from existing pipeline output."""
    import glob
    from process.assembly import export_validation_testsets
    from process import output_paths as _paths

    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        print(f"Error: output directory not found: {output_dir}")
        sys.exit(1)

    # Find master JSONL in new layout, then fall back to legacy root location
    _ms_dir = _paths.master_segments_dir(output_dir)
    jsonl_files = sorted(glob.glob(os.path.join(_ms_dir, 'master_segments*.jsonl')))
    if not jsonl_files:
        jsonl_files = sorted(glob.glob(os.path.join(output_dir, 'master_segments_*.jsonl')))
    if not jsonl_files:
        print(f"Error: no master_segments*.jsonl found in {output_dir}")
        print("Run the pipeline first: python qra.py run --config <config>")
        sys.exit(1)

    jsonl_path = jsonl_files[-1]
    print(f"\nLoading segments from: {os.path.relpath(jsonl_path)}")

    segments = _load_segments_from_jsonl(jsonl_path)
    participant_segs = [s for s in segments if s.speaker == 'participant']
    print(f"Loaded {len(segments)} segments ({len(participant_segs)} participant)")

    # Load framework
    framework_arg = args.framework
    if framework_arg is None and args.config:
        with open(args.config) as f:
            fw_data = json.load(f)
        fw = fw_data.get('framework', {})
        framework_arg = fw.get('custom_path') or fw.get('preset', 'vaamr')
    framework = _load_framework(framework_arg)

    # Determine codebook status
    codebook_enabled = args.codebook_enabled
    if not codebook_enabled and args.config:
        with open(args.config) as f:
            cfg_data = json.load(f)
        codebook_enabled = cfg_data.get('pipeline', {}).get('run_codebook_classifier', False)

    # Generate
    written = export_validation_testsets(
        segments,
        framework,
        output_dir,
        n_sets=args.n_sets,
        fraction_per_set=args.fraction,
        random_seed=args.seed,
        codebook_enabled=codebook_enabled,
    )

    print(f"\nGenerated {len(written)} worksheet files:")
    for p in written:
        print(f"  {os.path.relpath(p, output_dir)}")


def _load_segments_from_jsonl(jsonl_path: str):
    """Reconstruct Segment objects from a master_segments JSONL file."""
    import pandas as pd
    from classification_tools.data_structures import Segment

    df = pd.read_json(jsonl_path, lines=True)
    segments = []

    def _int_or_none(val):
        try:
            return int(val) if val is not None and str(val) != 'nan' else None
        except (ValueError, TypeError):
            return None

    def _float_or_none(val):
        try:
            return float(val) if val is not None and str(val) != 'nan' else None
        except (ValueError, TypeError):
            return None

    def _list_or_none(val):
        if val is None:
            return None
        if isinstance(val, list):
            return val if val else None
        return None

    for _, row in df.iterrows():
        # rater_ids and rater_votes are stored as JSON strings
        rater_ids = None
        if isinstance(row.get('rater_ids'), str):
            try:
                rater_ids = json.loads(row['rater_ids'])
            except (json.JSONDecodeError, TypeError):
                pass

        rater_votes = None
        if isinstance(row.get('rater_votes'), str):
            try:
                rater_votes = json.loads(row['rater_votes'])
            except (json.JSONDecodeError, TypeError):
                pass

        seg = Segment(
            segment_id=str(row['segment_id']),
            trial_id=str(row.get('trial_id', '')),
            participant_id=str(row.get('participant_id', '')),
            session_id=str(row['session_id']),
            session_number=_int_or_none(row.get('session_number')) or 1,
            cohort_id=_int_or_none(row.get('cohort_id')),
            session_variant=str(row.get('session_variant', '') or ''),
            segment_index=_int_or_none(row.get('segment_index')) or 0,
            start_time_ms=_int_or_none(row.get('start_time_ms')) or 0,
            end_time_ms=_int_or_none(row.get('end_time_ms')) or 0,
            total_segments_in_session=_int_or_none(row.get('total_segments_in_session')) or 1,
            speaker=str(row.get('speaker', '')),
            text=str(row.get('text', '')),
            word_count=_int_or_none(row.get('word_count')) or 0,
            primary_stage=_int_or_none(row.get('primary_stage')),
            secondary_stage=_int_or_none(row.get('secondary_stage')),
            llm_confidence_primary=_float_or_none(row.get('llm_confidence_primary')),
            llm_confidence_secondary=_float_or_none(row.get('llm_confidence_secondary')),
            llm_justification=row.get('llm_justification'),
            llm_run_consistency=_int_or_none(row.get('llm_run_consistency')),
            rater_ids=rater_ids,
            rater_votes=rater_votes,
            agreement_level=str(row.get('agreement_level') or ''),
            agreement_fraction=_float_or_none(row.get('agreement_fraction')),
            needs_review=bool(row.get('needs_review', False)),
            consensus_vote=row.get('consensus_vote'),
            tie_broken_by_confidence=bool(row.get('tie_broken_by_confidence', False)),
            codebook_labels_embedding=_list_or_none(row.get('codebook_labels_embedding')),
            codebook_labels_llm=_list_or_none(row.get('codebook_labels_llm')),
            codebook_labels_ensemble=_list_or_none(row.get('codebook_labels_ensemble')),
            codebook_disagreements=_list_or_none(row.get('codebook_disagreements')),
            codebook_confidence=None,
            human_label=_int_or_none(row.get('human_label')),
            human_secondary_label=_int_or_none(row.get('human_secondary_label')),
            adjudicated_label=_int_or_none(row.get('adjudicated_label')),
            in_human_coded_subset=bool(row.get('in_human_coded_subset', False)),
            label_status=str(row.get('label_status', 'llm_only')),
            final_label=_int_or_none(row.get('final_label')),
            final_label_source=row.get('final_label_source'),
            label_confidence_tier=row.get('label_confidence_tier'),
            speakers_in_segment=None,
            session_file=str(row.get('session_file', '')),
        )
        segments.append(seg)

    return segments


# =========================================================================
# Pipeline execution
# =========================================================================

def _wait_for_lmstudio(base_url: str, timeout_s: int = 32000, poll_s: int = 5) -> None:
    """
    Block until the LM Studio server responds to GET /v1/models, or timeout.
    """
    import time
    import requests

    url = base_url.rstrip('/') + '/models'
    waited = 0
    first_fail = True

    while waited < timeout_s:
        try:
            r = requests.get(url, timeout=4)
            if r.status_code < 500:
                if not first_fail:
                    print(" connected.")
                return
        except Exception:
            pass

        if first_fail:
            print(
                f"\n  LM Studio not yet reachable at {base_url}\n"
                f"  Waiting up to {timeout_s}s for the server to start",
                end='', flush=True,
            )
            first_fail = False
        else:
            print('.', end='', flush=True)

        time.sleep(poll_s)
        waited += poll_s

    print(f"\n  Warning: LM Studio did not respond within {timeout_s}s — proceeding anyway.")


def _execute_pipeline(config, framework, codebook=None):
    """Run the pipeline with the given config and framework."""
    from process.orchestrator import run_full_pipeline

    # LM Studio: wait until the server is reachable before starting
    if config.theme_classification.backend == 'lmstudio':
        _wait_for_lmstudio(
            getattr(config.theme_classification, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1')
        )

    # Print header
    print("\n" + "=" * 70)
    print("QRA CLASSIFICATION PIPELINE")
    print("=" * 70)
    tc = config.theme_classification
    print(f"  Framework: {framework.name} ({framework.num_themes} themes)")
    per_run = getattr(tc, 'per_run_models', []) or []
    use_per_run = len(per_run) >= 2 and len(per_run) == tc.n_runs
    if use_per_run:
        print(f"  Models:    {tc.n_runs} distinct raters (per-run interrater)")
        for i, m in enumerate(per_run):
            print(f"             Run {i + 1}: {m}")
    else:
        print(f"  Model:     {tc.model}")
    print(f"  Backend:   {tc.backend}")
    if tc.backend == 'lmstudio':
        print(f"  LM Studio: {getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1')}")
    if config.run_codebook_classifier:
        print(f"  Codebook:  enabled")
    print()

    master_df = run_full_pipeline(config, framework, codebook=codebook)

    # Summary
    print(f"\nTotal segments: {len(master_df)}")
    if 'primary_stage' in master_df.columns:
        labeled = master_df[master_df['primary_stage'].notna()]
        print(f"Successfully labeled: {len(labeled)}")

    return master_df


def _preload_huggingface_models(config):
    """Download and preload HuggingFace models."""
    from classification_tools.model_loader import ensure_models_ready, load_model

    models = config.theme_classification.models
    if not models:
        models = [config.theme_classification.model]

    print("\nPreparing HuggingFace models...")
    ensure_models_ready(download_if_missing=True)

    for model_id in models:
        try:
            print(f"  Loading {model_id}...")
            load_model(model_id)
            print(f"  Loaded successfully")
        except Exception as e:
            print(f"  Failed to load {model_id}: {e}")
            sys.exit(1)


def _prompt_yes_no_simple(label: str, default: bool = True) -> bool:
    default_str = 'Y/n' if default else 'y/N'
    raw = input(f"\n{label} [{default_str}]: ").strip().lower()
    if not raw:
        return default
    return raw in ('y', 'yes')


# =========================================================================
# Main
# =========================================================================

def main():
    parser = argparse.ArgumentParser(
        prog='qra',
        description='QRA: Qualitative Research Algorithm pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive setup wizard
  python qra.py setup

  # Run pipeline with defaults
  python qra.py run --backend openrouter --model openai/gpt-4o

  # Run with saved config
  python qra.py run --config ./qra_config.json

  # HuggingFace multi-model pipeline
  python qra.py run --backend huggingface

  # Run pipeline and auto-generate analysis reports afterward
  python qra.py run --backend lmstudio --auto-analyze

  # Analyze existing pipeline output (standalone post-hoc)
  python qra.py analyze --output-dir ./data/output/

  # Generate validation test set worksheets from existing output
  python qra.py testsets --output-dir ./data/output/
  python qra.py testsets --output-dir ./data/output/ --n-sets 3 --fraction 0.15

  # Zero-shot content validity test (skips full pipeline)
  python qra.py run --test-zeroshot --preset small --lmstudio-url http://10.0.0.58:1234/v1 --output-dir ./data/output/
  python qra.py run --test-zeroshot --backend openrouter --models model-a model-b model-c --api-key $KEY --output-dir ./data/output/
  python qra.py run --test-zeroshot --backend openrouter --model openai/gpt-4o --api-key $KEY --output-dir ./data/output/
        """,
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # ---- setup ----
    subparsers.add_parser('setup', help='Interactive configuration wizard')

    # ---- run ----
    run_parser = subparsers.add_parser(
        'run', help='Execute the classification pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_args(run_parser)

    # ---- analyze ----
    analyze_parser = subparsers.add_parser(
        'analyze',
        help='Run results analysis on an existing pipeline output directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Reads master_segment_dataset.csv and theme_definitions.json from OUTPUT_DIR\n'
            'and produces per-session, per-participant, per-construct, and longitudinal\n'
            'reports in OUTPUT_DIR/reports/analysis/.'
        ),
    )
    analyze_parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Path to pipeline output directory containing master_segment_dataset.csv',
    )

    # ---- testsets ----
    testsets_parser = subparsers.add_parser(
        'testsets',
        help='Generate validation test set worksheets from existing pipeline output',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=(
            'Loads master_segments_*.jsonl from OUTPUT_DIR and generates cross-session\n'
            'human coding and AI classification worksheets for inter-rater reliability.'
        ),
    )
    testsets_parser.add_argument(
        '--output-dir', '-o',
        required=True,
        help='Path to pipeline output directory (must contain master_segments_*.jsonl)',
    )
    testsets_parser.add_argument(
        '--config', '-c',
        default=None,
        help='Path to config JSON (used to load framework and codebook settings)',
    )
    testsets_parser.add_argument(
        '--framework', default=None,
        help='Theme framework: "vaamr" (default) or path to custom JSON',
    )
    testsets_parser.add_argument(
        '--n-sets', type=int, default=2,
        help='Number of test sets to generate (default: 2)',
    )
    testsets_parser.add_argument(
        '--fraction', type=float, default=0.10,
        help='Fraction of participant segments per set (default: 0.10)',
    )
    testsets_parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for reproducibility (default: 42)',
    )
    testsets_parser.add_argument(
        '--codebook-enabled', action='store_true',
        help='Include codebook classification labels in AI worksheet',
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    try:
        if args.command == 'setup':
            cmd_setup(args)
        elif args.command == 'run':
            cmd_run(args)
        elif args.command == 'analyze':
            cmd_analyze(args)
        elif args.command == 'testsets':
            cmd_testsets(args)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed with error: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
