"""
process/assembly/content_validity.py
-------------------------------------
Frozen content-validity testset creation and refresh.

Lifts the VAAMR content-validity machinery out of human_forms.py and
generalizes it to support PURER. Codebook is deferred (no exemplar
utterances in codebook codes yet).

On-disk layout:
  04_validation/content_validity/<name>/
    manifest.json        (frozen)
    items.jsonl          (frozen)
    human_worksheet.txt  (frozen)
    definition_key.txt   (frozen)
    AI_answer_key.txt    (refreshable)
"""

import datetime
import json
import os
import textwrap
from typing import Dict, List, Optional

from process import output_paths as _paths
from process._freeze import FrozenArtifactError, sha256_text, write_frozen

_W = 78


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_frozen_content_validity_testset(
    framework,
    run_dir: str,
    *,
    name: str,
    kind: str,
    theme_classification_cfg=None,
    force: bool = False,
) -> str:
    """
    Build items from framework exemplar/subtle/adversarial utterances, then write:
      manifest.json, items.jsonl, human_worksheet.txt, definition_key.txt  (frozen)
      AI_answer_key.txt                                                      (refreshable)

    kind must be 'vaamr' or 'purer'. Raises NotImplementedError for 'codebook'
    (no exemplar utterances in codebook codes yet).

    Returns the testset directory path.
    Raises FrozenArtifactError if testset already exists and force=False.
    """
    if kind == 'codebook':
        raise NotImplementedError(
            "codebook content-validity is not yet supported — "
            "codebook codes have no exemplar utterances. "
            "Populate CodeDefinition.exemplar_utterances first."
        )

    ws_path = _paths.cv_testset_human_worksheet_path(run_dir, name)
    if not force and os.path.exists(ws_path):
        raise FrozenArtifactError(
            f"Content-validity testset {name!r} already exists at {ws_path}. "
            "Pass force=True to overwrite."
        )

    items = _enumerate_items(framework)
    testset_dir = _paths.cv_testset_dir(run_dir, name)
    os.makedirs(testset_dir, exist_ok=True)

    fw_name = getattr(framework, 'name', kind)
    fw_version = getattr(framework, 'version', '1')
    now_iso = datetime.datetime.utcnow().isoformat() + 'Z'

    manifest = {
        'kind': kind,
        'name': name,
        'framework': {'name': fw_name, 'version': fw_version},
        'item_ids': [item['id'] for item in items],
        'content_sha256': {item['id']: sha256_text(item['text']) for item in items},
        'created_at': now_iso,
    }

    write_frozen(
        _paths.cv_testset_manifest_path(run_dir, name),
        lambda fh: json.dump(manifest, fh, indent=2),
        force=force,
    )
    write_frozen(
        _paths.cv_testset_items_path(run_dir, name),
        lambda fh: _write_items_jsonl(fh, items),
        force=force,
    )
    write_frozen(
        ws_path,
        lambda fh: _write_cv_human_worksheet(fh, items, framework),
        force=force,
    )
    write_frozen(
        _paths.cv_testset_definition_key_path(run_dir, name),
        lambda fh: _write_cv_definition_key(fh, framework),
        force=force,
    )

    _grade_cv_items(run_dir, name, items, framework, theme_classification_cfg)

    return testset_dir


def refresh_cv_answer_key(
    run_dir: str,
    name: str,
    theme_classification_cfg,
    framework,
) -> str:
    """
    Re-emit AI_answer_key.txt for an existing content-validity testset.

    Reads manifest and items.jsonl. Verifies each item's text SHA-256 against
    the manifest (raises FrozenArtifactError if any item text has drifted).
    Writes a new AI_answer_key.txt. Touches no frozen files.

    Returns the path to the updated AI_answer_key.txt.
    """
    manifest_path = _paths.cv_testset_manifest_path(run_dir, name)
    items_path = _paths.cv_testset_items_path(run_dir, name)

    with open(manifest_path, encoding='utf-8') as fh:
        manifest = json.load(fh)

    stored_sha = manifest.get('content_sha256', {})
    items = []
    with open(items_path, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                items.append(json.loads(line))

    drifted = []
    for item in items:
        item_id = item['id']
        expected_sha = stored_sha.get(item_id, '')
        if sha256_text(item['text']) != expected_sha:
            drifted.append(item_id)

    if drifted:
        raise FrozenArtifactError(
            f"Content-validity testset {name!r}: {len(drifted)} item(s) have drifted "
            f"text since the testset was frozen: {drifted[:5]}"
            + (" (and more)" if len(drifted) > 5 else "")
        )

    _grade_cv_items(run_dir, name, items, framework, theme_classification_cfg)
    return _paths.cv_testset_answer_key_path(run_dir, name)


def list_content_validity_testsets(run_dir: str) -> List[dict]:
    """
    Return one summary dict per content_validity/<name>/ folder.

    Each dict has: name, kind, framework (name + version), n_items, created_at.
    Returns empty list if no content_validity directory exists.
    """
    cv_root = _paths.cv_testsets_dir(run_dir)
    if not os.path.isdir(cv_root):
        return []

    results = []
    for entry in sorted(os.scandir(cv_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        manifest_path = os.path.join(entry.path, 'manifest.json')
        if not os.path.isfile(manifest_path):
            continue
        try:
            with open(manifest_path, encoding='utf-8') as fh:
                m = json.load(fh)
            results.append({
                'name': m.get('name', entry.name),
                'kind': m.get('kind', '?'),
                'framework': m.get('framework', {}),
                'n_items': len(m.get('item_ids', [])),
                'created_at': m.get('created_at', '?'),
            })
        except Exception:
            continue

    return results


def generate_or_refresh_content_validity_testsets(
    run_dir: str,
    *,
    cv_config,
    framework_vaamr,
    framework_purer,
    theme_classification_cfg,
    create_missing: bool = True,
) -> List[str]:
    """
    Coordinator used by orchestrator Stage 7 and by qra cv refresh --all.

    For each enabled spec in cv_config:
      - if testset exists: refresh_cv_answer_key(...)
      - else (and create_missing=True): create_frozen_content_validity_testset(...)

    When create_missing=False, only refreshes testsets that already exist;
    skips kinds whose directory is absent. Used by `qra assemble`.

    Returns list of testset directories touched.
    """
    dirs: List[str] = []

    spec_pairs = [
        (cv_config.vaamr, 'vaamr', framework_vaamr),
        (cv_config.purer, 'purer', framework_purer),
    ]

    for spec, kind, framework in spec_pairs:
        if not spec.enabled:
            continue
        if framework is None:
            continue

        name = spec.name
        manifest_path = _paths.cv_testset_manifest_path(run_dir, name)

        if os.path.isfile(manifest_path):
            refresh_cv_answer_key(run_dir, name, theme_classification_cfg, framework)
        elif create_missing:
            create_frozen_content_validity_testset(
                framework, run_dir,
                name=name,
                kind=kind,
                theme_classification_cfg=theme_classification_cfg,
            )
        else:
            # create_missing=False and testset does not yet exist — skip
            continue

        dirs.append(_paths.cv_testset_dir(run_dir, name))

    return dirs


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _enumerate_items(framework) -> List[dict]:
    """Extract items from framework utterances (exemplar/subtle/adversarial)."""
    items = []
    item_idx = 0
    tier_map = [
        ('clear', 'exemplar_utterances', 'exemplar'),
        ('subtle', 'subtle_utterances', 'subtle'),
        ('adversarial', 'adversarial_utterances', 'adversarial'),
    ]
    for theme in framework.themes:
        for difficulty, field_name, source_field in tier_map:
            for utterance in getattr(theme, field_name, []):
                item_id = f'cv_{item_idx:04d}'
                items.append({
                    'id': item_id,
                    'text': utterance,
                    'expected_stage': theme.theme_id,
                    'difficulty': difficulty,
                    'source_field': source_field,
                    'content_sha256': sha256_text(utterance),
                })
                item_idx += 1
    return items


def _write_items_jsonl(fh, items: List[dict]) -> None:
    for item in items:
        fh.write(json.dumps(item) + '\n')


def _write_cv_human_worksheet(fh, items: List[dict], framework) -> None:
    fw_name = getattr(framework, 'name', '?')
    stage_labels = '?'
    if framework is not None:
        stage_labels = ', '.join(
            f"{t.theme_id}={t.short_name}" for t in framework.themes
        )

    fh.write('=' * _W + '\n')
    fh.write(f'CONTENT VALIDITY TEST SET — HUMAN CODING WORKSHEET\n')
    fh.write(f'Framework: {fw_name}\n')
    fh.write('=' * _W + '\n')
    fh.write(f'Generated: {datetime.date.today().isoformat()}   Items: {len(items)}\n')
    fh.write('Purpose: Rate each item independently. Use the definition key for reference.\n\n')
    fh.write('INSTRUCTIONS\n')
    fh.write('-' * _W + '\n')
    fh.write(f'  Labels: {stage_labels}\n\n')
    fh.write('  Difficulty tiers:\n')
    fh.write('    clear       — prototypical; expected label is evident\n')
    fh.write('    subtle      — requires careful reading and inference\n')
    fh.write('    adversarial — may superficially resemble a different label\n\n')
    fh.write('  For each item record the primary label and a brief rationale.\n\n')

    for item in items:
        item_id = item.get('id', '?')
        text = item.get('text', '')
        difficulty = item.get('difficulty', '?')

        fh.write('=' * _W + '\n')
        fh.write(f'[ITEM {item_id}]  Tier: {difficulty}\n')
        fh.write('-' * _W + '\n')
        for line in textwrap.wrap(
            f'"{text}"', width=_W - 2,
            initial_indent='  ', subsequent_indent='  ',
        ) or ['  ']:
            fh.write(line + '\n')
        fh.write('\n')
        fh.write('  Label: ___   Secondary (optional): ___\n')
        fh.write('  Rationale: ' + '_' * 60 + '\n')
        fh.write('  ' + '_' * 72 + '\n')
        fh.write('\n')


def _write_cv_definition_key(fh, framework) -> None:
    fw_name = getattr(framework, 'name', '?')
    fw_version = getattr(framework, 'version', '?')

    fh.write('=' * _W + '\n')
    fh.write('CONSTRUCT DEFINITION KEY\n')
    fh.write(f'Framework: {fw_name}   Version: {fw_version}\n')
    fh.write(f'Generated: {datetime.date.today().isoformat()}\n')
    fh.write('=' * _W + '\n\n')
    fh.write(
        'Use this key when completing the content validity human coding worksheet.\n\n'
    )

    for theme in sorted(framework.themes, key=lambda t: t.theme_id):
        fh.write('=' * _W + '\n')
        fh.write(f'[{theme.theme_id}]  {theme.name.upper()}\n')
        fh.write('-' * _W + '\n')
        for line in textwrap.wrap(
            theme.definition, width=_W - 2,
            initial_indent='  ', subsequent_indent='  ',
        ):
            fh.write(line + '\n')
        fh.write('\n')
        if theme.prototypical_features:
            fh.write('  Prototypical features:\n')
            for feat in theme.prototypical_features:
                for line in textwrap.wrap(
                    feat, width=_W - 6,
                    initial_indent='    • ', subsequent_indent='      ',
                ):
                    fh.write(line + '\n')
            fh.write('\n')
        if theme.distinguishing_criteria:
            fh.write('  Key distinction:\n')
            for line in textwrap.wrap(
                theme.distinguishing_criteria, width=_W - 4,
                initial_indent='    ', subsequent_indent='    ',
            ):
                fh.write(line + '\n')
            fh.write('\n')
        exemplars = (theme.exemplar_utterances or [])[:3]
        if exemplars:
            fh.write('  Calibration exemplars (not in worksheet):\n')
            for ex in exemplars:
                for line in textwrap.wrap(
                    f'"{ex}"', width=_W - 6,
                    initial_indent='    – ', subsequent_indent='      ',
                ):
                    fh.write(line + '\n')
            fh.write('\n')

    fh.write('=' * _W + '\n')
    fh.write('END OF DEFINITION KEY\n')
    fh.write('=' * _W + '\n')


def _grade_cv_items(
    run_dir: str,
    name: str,
    items: List[dict],
    framework,
    theme_classification_cfg,
) -> None:
    """
    Write AI_answer_key.txt.

    When theme_classification_cfg is provided, runs zero-shot LLM classification.
    When None, writes a PENDING placeholder (used in unit tests and dry runs).
    """
    ai_path = _paths.cv_testset_answer_key_path(run_dir, name)

    if theme_classification_cfg is None or not items:
        _write_cv_answer_key_pending(ai_path, items, framework)
        return

    try:
        _write_cv_answer_key_graded(ai_path, items, framework, theme_classification_cfg)
    except Exception as exc:
        _write_cv_answer_key_pending(ai_path, items, framework, error=str(exc))


def _write_cv_answer_key_pending(ai_path, items, framework, error=None) -> None:
    fw_name = getattr(framework, 'name', '?') if framework else '?'
    id_to_name = {}
    if framework:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    with open(ai_path, 'w', encoding='utf-8') as fh:
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY — AI ANSWER KEY\n')
        fh.write(f'Framework: {fw_name}\n')
        fh.write('=' * _W + '\n')
        if error:
            fh.write(f'STATUS: GRADING FAILED — {error}\n\n')
        else:
            fh.write('STATUS: PENDING — run `qra cv refresh` to grade with an LLM.\n\n')
        fh.write(f'Items: {len(items)}\n')
        fh.write('=' * _W + '\n\n')
        for item in items:
            item_id = item.get('id', '?')
            expected = item.get('expected_stage')
            stage_name = id_to_name.get(expected, str(expected)) if expected is not None else '?'
            fh.write(f'[ITEM {item_id}]  Expected: {expected} — {stage_name}  '
                     f'Tier: {item.get("difficulty", "?")}\n')
        fh.write('\n')


def _write_cv_answer_key_graded(ai_path, items, framework, theme_classification_cfg) -> None:
    """Run zero-shot grading and write the graded report."""
    from classification_tools.llm_classifier import create_content_validity_test_set
    from classification_tools.classification_loop import classify_segments_with_multi_run_consensus
    from classification_tools.llm_client import LLMClient, LLMClientConfig
    from theme_framework.config import ThemeClassificationConfig

    # Convert items to Segment-shaped objects for the classifier
    from classification_tools.data_structures import Segment
    segments = []
    for item in items:
        seg = Segment(
            segment_id=item['id'],
            trial_id='cv_test',
            participant_id='cv_participant',
            session_id='cv_test',
            session_number=1,
            segment_index=int(item['id'].split('_')[1]),
            start_time_ms=0,
            end_time_ms=0,
            total_segments_in_session=len(items),
            speaker='participant',
            text=item['text'],
            word_count=len(item['text'].split()),
        )
        segments.append(seg)

    tc = theme_classification_cfg
    client_cfg = LLMClientConfig(
        backend=tc.backend,
        api_key=tc.api_key,
        replicate_api_token=getattr(tc, 'replicate_api_token', ''),
        model=tc.model,
        temperature=getattr(tc, 'temperature', 0.0),
        lmstudio_base_url=getattr(tc, 'lmstudio_base_url', 'http://127.0.0.1:1234/v1'),
        ollama_host=getattr(tc, 'ollama_host', '0.0.0.0'),
        ollama_port=getattr(tc, 'ollama_port', 11434),
    )
    client = LLMClient(client_cfg)
    classified = classify_segments_with_multi_run_consensus(
        segments, framework, client, n_runs=getattr(tc, 'n_runs', 1),
        temperature=getattr(tc, 'temperature', 0.0),
    )

    id_to_name = {t.theme_id: t.short_name for t in framework.themes} if framework else {}
    fw_name = getattr(framework, 'name', '?') if framework else '?'

    with open(ai_path, 'w', encoding='utf-8') as fh:
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY — AI ANSWER KEY (GRADED)\n')
        fh.write(f'Framework: {fw_name}\n')
        fh.write('=' * _W + '\n')
        fh.write(f'Items: {len(items)}\n\n')

        classified_by_id = {s.segment_id: s for s in classified}
        for item in items:
            item_id = item.get('id', '?')
            expected = item.get('expected_stage')
            stage_name = id_to_name.get(expected, str(expected)) if expected is not None else '?'
            seg = classified_by_id.get(item_id)
            predicted = getattr(seg, 'primary_stage', None) if seg else None
            pred_name = id_to_name.get(predicted, str(predicted)) if predicted is not None else '?'
            correct = (predicted == expected)
            mark = '[PASS]' if correct else '[FAIL]'
            fh.write(
                f'{mark}  [{item_id}]  '
                f'Expected: {expected}={stage_name}  '
                f'Predicted: {predicted}={pred_name}  '
                f'Tier: {item.get("difficulty", "?")}\n'
            )
        fh.write('\n')
