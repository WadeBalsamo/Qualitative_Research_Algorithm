"""Export functions for validation test sets and review forms."""

import json
import os
import textwrap
from collections import defaultdict
from typing import Dict, List, Optional

from classification_tools.data_structures import Segment
from .. import output_paths as _paths
from .._freeze import FrozenArtifactError, write_frozen
from ._shared import _ms_to_hms, _fmt_conf, _theme_name_from

_W = 78  # column width used across all human-readable forms


def export_content_validity_human_worksheet(
    test_items: List[Dict],
    framework,
    run_dir: str,
) -> str:
    """
    Write a blind-coding worksheet for the content validity test set.

    Items are shown without the expected stage so human raters can code
    independently. Difficulty tier is disclosed so raters can calibrate
    effort. Mirrors the format of human_classification_*.txt.

    Returns the written file path.
    """
    import datetime as _dt

    stage_labels = '?'
    if framework is not None:
        stage_labels = ', '.join(
            f"{t.theme_id}={t.short_name}" for t in framework.themes
        )

    validation_dir = _paths.validation_dir(run_dir)
    os.makedirs(validation_dir, exist_ok=True)
    out_path = os.path.join(validation_dir, 'content_validity_human_worksheet.txt')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY TEST SET — HUMAN CODING WORKSHEET\n')
        fh.write('=' * _W + '\n')
        fh.write(
            f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
            f"Items: {len(test_items)}\n"
        )
        fh.write(
            'Purpose: Rate each item independently. '
            'Use the definition key for reference.\n\n'
        )
        fh.write('INSTRUCTIONS\n')
        fh.write('-' * _W + '\n')
        fh.write(f'  Stage labels: {stage_labels}\n\n')
        fh.write('  Difficulty tiers:\n')
        fh.write('    clear       — prototypical utterances; expected stage is evident\n')
        fh.write('    subtle      — items requiring careful reading and inference\n')
        fh.write('    adversarial — items that may superficially resemble another stage\n\n')
        fh.write('  For each item record the primary stage and a brief rationale.\n')
        fh.write('  A secondary stage is optional when two stages are clearly present.\n\n')

        for item in test_items:
            item_id = item.get('test_item_id', '?')
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
            fh.write('  Stage: ___   Secondary (optional): ___\n')
            fh.write('  Rationale: ' + '_' * 60 + '\n')
            fh.write('  ' + '_' * 72 + '\n')
            fh.write('\n')

    return out_path


def export_content_validity_definition_key(
    framework,
    run_dir: str,
) -> Optional[str]:
    """
    Write a human-readable construct definition key for use alongside the
    content validity worksheet. Shows each stage's definition, prototypical
    features, distinguishing criterion, and calibration exemplars.

    Returns the written file path, or None if framework is None.
    """
    import datetime as _dt

    if framework is None:
        return None

    validation_dir = _paths.validation_dir(run_dir)
    os.makedirs(validation_dir, exist_ok=True)
    out_path = os.path.join(validation_dir, 'content_validity_definition_key.txt')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('=' * _W + '\n')
        fh.write('CONSTRUCT DEFINITION KEY\n')
        fh.write(f'Framework: {framework.name}   Version: {framework.version}\n')
        fh.write(f'Generated: {_dt.datetime.utcnow().strftime("%Y-%m-%d")}\n')
        fh.write('=' * _W + '\n\n')
        fh.write(
            'Use this key when completing the content validity human coding worksheet.\n'
            'Each stage is defined by its core construct, prototypical features,\n'
            'key distinction from adjacent stages, and calibration exemplars.\n'
            'The worksheet items are different from the exemplars listed here.\n\n'
        )

        for theme in sorted(framework.themes, key=lambda t: t.theme_id):
            fh.write('=' * _W + '\n')
            fh.write(f'[STAGE {theme.theme_id}]  {theme.name.upper()}\n')
            fh.write('-' * _W + '\n')

            # Definition
            for line in textwrap.wrap(
                theme.definition, width=_W - 2,
                initial_indent='  ', subsequent_indent='  ',
            ):
                fh.write(line + '\n')
            fh.write('\n')

            # Prototypical features
            if theme.prototypical_features:
                fh.write('  Prototypical features:\n')
                for feat in theme.prototypical_features:
                    for line in textwrap.wrap(
                        feat, width=_W - 6,
                        initial_indent='    • ', subsequent_indent='      ',
                    ):
                        fh.write(line + '\n')
                fh.write('\n')

            # Distinguishing criterion
            if theme.distinguishing_criteria:
                fh.write('  Key distinction:\n')
                for line in textwrap.wrap(
                    theme.distinguishing_criteria, width=_W - 4,
                    initial_indent='    ', subsequent_indent='    ',
                ):
                    fh.write(line + '\n')
                fh.write('\n')

            # Calibration exemplars (clear tier only; not the worksheet items)
            exemplars = (theme.exemplar_utterances or [])[:4]
            if exemplars:
                fh.write(
                    '  Calibration exemplars '
                    '(for orientation — not present in the worksheet):\n'
                )
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

    return out_path


def export_content_validity_answer_key(
    test_items: List[Dict],
    framework,
    run_dir: str,
) -> str:
    """
    Write the answer key for the content validity human coding worksheet.

    Shows each item's text with its expected stage revealed. Mirrors the
    human_classification_*.txt format but with Stage pre-filled. Intended
    for post-hoc comparison only — complete the worksheet before consulting
    this document.

    Returns the written file path.
    """
    import datetime as _dt

    id_to_name: dict = {}
    stage_labels: str = '?'
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}
        stage_labels = ', '.join(
            f"{t.theme_id}={t.short_name}" for t in framework.themes
        )

    validation_dir = _paths.validation_dir(run_dir)
    os.makedirs(validation_dir, exist_ok=True)
    out_path = os.path.join(validation_dir, 'content_validity_answer_key.txt')

    with open(out_path, 'w', encoding='utf-8') as fh:
        fh.write('=' * _W + '\n')
        fh.write('CONTENT VALIDITY TEST SET — ANSWER KEY\n')
        fh.write('=' * _W + '\n')
        if framework is not None:
            fh.write(
                f'Framework: {framework.name}   Version: {framework.version}   '
                f'Items: {len(test_items)}\n'
            )
        fh.write(f'Generated: {_dt.date.today().isoformat()}\n')
        fh.write('=' * _W + '\n\n')
        fh.write(
            'IMPORTANT: Complete the human coding worksheet independently\n'
            'before consulting this document. The expected stage for each item\n'
            'is revealed below for post-hoc comparison and calibration only.\n\n'
        )
        fh.write('STAGE LABELS\n')
        fh.write('-' * _W + '\n')
        fh.write(f'  {stage_labels}\n\n')
        fh.write('  Difficulty tiers:\n')
        fh.write('    clear       — prototypical utterances; stage is clearly present\n')
        fh.write('    subtle      — requires careful reading; stage is present but muted\n')
        fh.write(
            '    adversarial — superficially resembles a different stage; '
            'expected stage\n'
            '                  is still correct despite surface-level misdirection\n\n'
        )

        # Group by difficulty tier for readability
        tier_order = ['clear', 'subtle', 'adversarial']
        by_tier: Dict[str, List[Dict]] = {t: [] for t in tier_order}
        for item in test_items:
            by_tier.setdefault(item.get('difficulty', 'clear'), []).append(item)

        for tier in tier_order:
            tier_items = by_tier.get(tier, [])
            if not tier_items:
                continue
            fh.write('=' * _W + '\n')
            fh.write(f'TIER: {tier.upper()}\n')
            fh.write('=' * _W + '\n\n')

            for item in tier_items:
                item_id = item.get('test_item_id', '?')
                expected = item.get('expected_stage')
                stage_name = id_to_name.get(expected, str(expected)) if expected is not None else '?'
                text = item.get('text', '')

                fh.write('=' * _W + '\n')
                fh.write(
                    f'[ITEM {item_id}]  '
                    f'Expected: {expected} — {stage_name}\n'
                )
                fh.write('-' * _W + '\n')
                for line in textwrap.wrap(
                    f'"{text}"', width=_W - 2,
                    initial_indent='  ', subsequent_indent='  ',
                ) or ['  ']:
                    fh.write(line + '\n')
                fh.write('\n')

    return out_path


def export_human_classification_forms(
    segments: List[Segment],
    framework,
    run_dir: str,
) -> List[str]:
    """
    Write one blank-form .txt per session into <output_dir>/validation/.

    Each file shows segments in order with empty rating fields so human
    coders can classify blind, without any LLM results present.
    Only participant segments are included (therapist segments omitted).
    Returns list of written file paths.
    """
    import datetime as _dt

    id_to_name: dict = {}
    stage_labels: str = '?'
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}
        stage_labels = ', '.join(
            f"{t.theme_id}={t.short_name}" for t in framework.themes
        )

    def _theme_name(stage) -> str:
        return _theme_name_from(stage, id_to_name)

    validation_dir = _paths.validation_dir(run_dir)
    os.makedirs(validation_dir, exist_ok=True)

    # Group by session
    by_session: Dict[str, List[Segment]] = defaultdict(list)
    for seg in segments:
        by_session[seg.session_id].append(seg)

    written: List[str] = []

    for session_id, segs in sorted(by_session.items()):
        sorted_segs = sorted(segs, key=lambda s: s.segment_index)

        # Only participant segments (skip therapist-only context turns)
        participant_segs = [
            s for s in sorted_segs
            if s.speaker and s.speaker.lower() not in ('therapist', 't', 'interviewer')
        ]
        if not participant_segs:
            participant_segs = sorted_segs  # fallback: include all if no match

        if not participant_segs:
            continue

        first = participant_segs[0]
        last = participant_segs[-1]
        duration_s = max(0, (last.end_time_ms - first.start_time_ms) // 1000)
        dur_h, dur_rem = divmod(duration_s, 3600)
        dur_m, dur_s = divmod(dur_rem, 60)

        all_pids = sorted({s.participant_id for s in participant_segs if s.participant_id})
        participants_str = ", ".join(all_pids) if all_pids else first.participant_id

        out_path = os.path.join(validation_dir, f'human_classification_{session_id}.txt')
        with open(out_path, 'w', encoding='utf-8') as fh:
            fh.write("=" * 78 + "\n")
            fh.write(f"SESSION: {session_id}\n")
            fh.write(f"Participants: {participants_str}   Trial: {first.trial_id}\n")
            fh.write(f"Duration: {dur_h}:{dur_m:02d}:{dur_s:02d}   "
                     f"Segments: {len(participant_segs)}\n")
            fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}\n")
            fh.write("=" * 78 + "\n\n")
            fh.write("INSTRUCTIONS\n")
            fh.write("-" * 78 + "\n")
            fh.write(f"  Stage labels: {stage_labels}\n")
            fh.write("  For each segment, record the primary stage, an optional\n")
            fh.write("  secondary stage, and a brief rationale.\n\n")

            for seg in participant_segs:
                speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker or '?'])
                fh.write("=" * 78 + "\n")
                fh.write(
                    f"[SEGMENT {seg.segment_index + 1:03d}]  "
                    f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                    f"{seg.word_count}w   Speaker(s): {speakers_str}\n"
                )
                fh.write("-" * 78 + "\n")
                for line in textwrap.wrap(seg.text, width=76,
                                          initial_indent="  ",
                                          subsequent_indent="  ") or ["  "]:
                    fh.write(line + "\n")
                fh.write("\n")
                fh.write("  Stage: ___   Secondary (optional): ___\n")
                fh.write("  Rationale: " + "_" * 60 + "\n")
                fh.write("  " + "_" * 72 + "\n")
                fh.write("\n")

        written.append(out_path)

    return written


def export_flagged_for_review(
    segments: List[Segment],
    framework,
    run_dir: str,
) -> None:
    """
    Write a single .txt listing every needs_review segment across all sessions.

    Includes full rater ballot detail, combined justifications, coded result
    and confidence (primary and secondary), and agreement metadata.
    """
    import datetime as _dt

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    def _theme_name(stage) -> str:
        return _theme_name_from(stage, id_to_name)

    flagged = [s for s in segments if getattr(s, 'needs_review', False)]
    flagged.sort(key=lambda s: (s.session_id, s.segment_index))

    session_ids = sorted({s.session_id for s in flagged})

    _vdir = _paths.validation_dir(run_dir)
    os.makedirs(_vdir, exist_ok=True)
    output_path = os.path.join(_vdir, 'flagged_for_review.txt')

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write(
            f"FLAGGED FOR REVIEW — {len(flagged)} segment(s) "
            f"across {len(session_ids)} session(s)\n"
        )
        fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}\n")
        fh.write("=" * 78 + "\n\n")

        for seg in flagged:
            speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker or '?'])
            fh.write(
                f"[Session {seg.session_id}  Segment {seg.segment_index + 1:03d}]  "
                f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                f"{seg.word_count}w   Speaker(s): {speakers_str}\n"
            )
            fh.write("-" * 78 + "\n")
            for line in textwrap.wrap(seg.text, width=76,
                                      initial_indent="  ",
                                      subsequent_indent="  ") or ["  "]:
                fh.write(line + "\n")
            fh.write("\n")

            # Rater ballots
            rater_votes = getattr(seg, 'rater_votes', None) or []
            if rater_votes:
                fh.write("  RATER BALLOTS\n")
                for rv in rater_votes:
                    rid = rv.get('rater', '?')
                    vote = rv.get('vote', '?')
                    stage = rv.get('stage')
                    conf = rv.get('confidence')
                    just = (rv.get('justification') or '').strip()
                    sec = rv.get('secondary_stage')
                    sec_conf = rv.get('secondary_confidence')
                    if vote == 'CODED':
                        if sec is not None:
                            sec_conf_str = f", conf={_fmt_conf(sec_conf)}" if sec_conf is not None else ""
                            sec_part = f"  (2nd: {_theme_name(sec)}{sec_conf_str})"
                        else:
                            sec_part = ""
                        fh.write(f"    [{rid}]  {_theme_name(stage)}  "
                                 f"conf={_fmt_conf(conf)}{sec_part}\n")
                    elif vote == 'ABSTAIN':
                        fh.write(f"    [{rid}]  ABSTAIN\n")
                    else:
                        fh.write(f"    [{rid}]  ERROR\n")
                    if just:
                        for line in textwrap.wrap(just, width=68,
                                                  initial_indent="        → ",
                                                  subsequent_indent="          "):
                            fh.write(line + "\n")
                fh.write("\n")

            # Coded result summary
            agreement = getattr(seg, 'agreement_level', None) or '?'
            _rids = getattr(seg, 'rater_ids', None) or []
            n_agree = int(round(
                (getattr(seg, 'agreement_fraction', None) or 0.0) * max(len(_rids), 1)
            ))
            n_raters = len(_rids)

            if seg.primary_stage is not None:
                primary_str = (
                    f"{_theme_name(seg.primary_stage)} (conf {_fmt_conf(seg.llm_confidence_primary)})"
                )
                if seg.secondary_stage is not None:
                    sec_str = (
                        f"  |  Secondary: {_theme_name(seg.secondary_stage)} "
                        f"(conf {_fmt_conf(seg.llm_confidence_secondary)})"
                    )
                else:
                    sec_str = ""
                fh.write(f"  CODED RESULT: {primary_str}{sec_str}\n")
            else:
                if getattr(seg, 'consensus_vote', None) == 'ABSTAIN':
                    fh.write("  CODED RESULT: UNCLASSIFIED — consensus ABSTAIN\n")
                elif agreement == 'split':
                    fh.write("  CODED RESULT: UNCLASSIFIED — split vote\n")
                elif agreement == 'none':
                    fh.write("  CODED RESULT: UNCLASSIFIED — all raters failed\n")
                else:
                    fh.write("  CODED RESULT: UNCLASSIFIED\n")

            fh.write(f"  AGREEMENT: {agreement}  ({n_agree}/{n_raters})\n")
            if getattr(seg, 'tie_broken_by_confidence', False):
                fh.write("  Tie-broken by confidence: Yes\n")

            fh.write("-" * 78 + "\n\n")



def generate_or_refresh_validation_testsets(
    segments: List[Segment],
    framework,
    run_dir: str,
    *,
    test_sets_config=None,
    codebook_enabled: bool = False,
    codebook=None,
    create_missing: bool = True,
    # Legacy keyword args (Phase 1 back-compat — remove with legacy_migration.py)
    n_sets: Optional[int] = None,
    fraction_per_set: Optional[float] = None,
    random_seed: Optional[int] = None,
) -> List[str]:
    """
    Coordinator for multi-kind validation testset generation/refresh.

    Creates flat numbered worksheet files in testsets_dir. Returns list of
    human worksheet paths written or refreshed.

    When create_missing=False, only refreshes AI keys for existing flat worksheets;
    does not create new ones. Used by `qra assemble`.
    """
    import re as _re
    from process.config import TestSetsConfig, TestSetSpec

    # Legacy call-site back-compat: if TestSetsConfig not provided, build from old args
    if test_sets_config is None:
        _n = n_sets if n_sets is not None else 2
        _f = fraction_per_set if fraction_per_set is not None else 0.10
        _s = random_seed if random_seed is not None else 42
        test_sets_config = TestSetsConfig(
            vaamr=TestSetSpec(enabled=True, name='vaamr_testset', n_sets=_n,
                              fraction_per_set=_f, random_seed=_s),
        )

    segments_by_id = {s.segment_id: s for s in segments}
    paths: List[str] = []

    if not create_missing:
        # Refresh AI keys for all existing flat worksheets
        ts_dir = _paths.testsets_dir(run_dir)
        pattern = _re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
        nums = sorted(
            int(m.group(1)) for f in (os.listdir(ts_dir) if os.path.isdir(ts_dir) else [])
            if (m := pattern.match(f))
        )
        n_total = len(nums)
        for n in nums:
            refresh_testset_answer_key(
                segments_by_id, framework, run_dir, n,
                codebook_enabled=codebook_enabled, codebook=codebook,
                n_total=n_total,
            )
            paths.append(_paths.testset_human_flat_path(run_dir, n))
        return paths

    # Create new flat numbered worksheets
    for kind, spec in [
        ('vaamr', test_sets_config.vaamr),
        ('purer', test_sets_config.purer),
        ('codebook', test_sets_config.codebook),
    ]:
        if not spec.enabled:
            continue
        for i in range(1, spec.n_sets + 1):
            human_path = create_frozen_testset(
                segments, framework, run_dir,
                kind=kind,
                n_sets=spec.n_sets,
                set_index=i,
                fraction_per_set=spec.fraction_per_set,
                random_seed=spec.random_seed,
                codebook_enabled=codebook_enabled,
                codebook=codebook,
            )
            paths.append(human_path)

    return paths


def create_frozen_testset(
    segments: List[Segment],
    framework,
    run_dir: str,
    *,
    name: str = '',  # kept for call-site back-compat; unused for output paths
    kind: str = 'vaamr',
    n_sets: int,
    set_index: int,
    fraction_per_set: float,
    random_seed: int,
    codebook_enabled: bool,
    codebook=None,
    force: bool = False,
    exclude_segment_ids: Optional[set] = None,
) -> str:
    """
    Compute the stratified sample for set_index and write flat numbered worksheet files:
      human_classification_testset_worksheet_N.txt  (frozen — immutable once written)
      AI_classification_testset_worksheet_N.txt     (refreshable)

    N is auto-assigned as the next available number in testsets_dir.
    kind must be 'vaamr', 'purer', or 'codebook'.
    exclude_segment_ids: optional set of (session_id, seg_num_1based) to exclude from the
      sampling pool — used by the CLI to prevent overlap with existing testsets.
    Returns path to the human worksheet.
    Raises FrozenArtifactError if the worksheet already exists and force=False.
    """
    # Build pool by kind
    if kind == 'purer':
        pool = _pool_purer(segments)
        _key_fn = lambda s: getattr(s, 'purer_primary', None)
    elif kind == 'codebook':
        pool = _pool_codebook(segments)
        _key_fn = lambda s: s.cohort_id if s.cohort_id is not None else 'none'
    else:
        pool = _pool_vaamr(segments)
        _key_fn = lambda s: s.cohort_id if s.cohort_id is not None else 'none'

    # Exclude already-used segments when caller provides the set (e.g. CLI incremental adds)
    if exclude_segment_ids:
        pool = [s for s in pool if (s.session_id, s.segment_index + 1) not in exclude_segment_ids]
        if not pool:
            raise ValueError(
                "No unused segments remain for a new testset. "
                "All pool segments are already covered by existing testsets."
            )

    segs = _stratified_sample(pool, n_sets, set_index, fraction_per_set, random_seed, _key_fn)

    os.makedirs(_paths.testsets_dir(run_dir), exist_ok=True)
    n = _paths.next_testset_number(run_dir)
    human_path = _paths.testset_human_flat_path(run_dir, n)
    ai_path = _paths.testset_ai_flat_path(run_dir, n)

    if not force and os.path.isfile(human_path):
        raise FrozenArtifactError(
            f"Testset worksheet #{n} already exists at {human_path}. "
            "Pass force=True to overwrite."
        )

    # Dispatch human worksheet and AI key by kind
    if kind == 'purer':
        write_frozen(
            human_path,
            lambda fh: _write_purer_human_worksheet_to_handle(segs, framework, fh, set_index, n_sets),
            force=force,
        )
        _write_purer_answer_key(segs, framework, ai_path, set_index, n_sets)
    elif kind == 'codebook':
        write_frozen(
            human_path,
            lambda fh: _write_codebook_human_worksheet_to_handle(segs, codebook, fh, set_index, n_sets),
            force=force,
        )
        _write_codebook_answer_key(segs, codebook, ai_path, set_index, n_sets)
    else:
        write_frozen(
            human_path,
            lambda fh: _write_human_testset_to_handle(segs, framework, fh, set_index, n_sets),
            force=force,
        )
        _write_ai_testset(segs, framework, ai_path, set_index, n_sets, codebook_enabled)

    # Write per-segment content SHA sidecar for drift detection on future refreshes
    _write_testset_meta(run_dir, n, segs)

    return human_path


def refresh_testset_answer_key(
    segments_by_id: Dict[str, Segment],
    framework,
    run_dir: str,
    n: int,
    *,
    codebook_enabled: bool,
    codebook=None,
    n_total: int = 0,
    confirm_fn=None,
) -> str:
    """
    Re-emit AI_classification_testset_worksheet_N.txt from current labels.

    Parses the frozen human worksheet to identify which segments are in the set,
    then regenerates the AI key. Also updates the 'N of M' header in the human
    worksheet to reflect n_total (auto-computed if n_total=0).

    Raises FrozenArtifactError if segment content has changed and the user
    declines to continue. confirm_fn(prompt) -> bool overrides the interactive
    stdin prompt (useful in tests or --yes CLI mode).

    Returns path to the updated AI answer key.
    """
    import hashlib

    human_path = _paths.testset_human_flat_path(run_dir, n)
    if not os.path.isfile(human_path):
        raise FrozenArtifactError(
            f"No human worksheet found for testset #{n} at {human_path}."
        )

    if n_total == 0:
        n_total = _paths.count_existing_testsets(run_dir)

    items = _parse_worksheet_items(human_path)
    # Build lookup by (session_id, segment_index) for fast access
    by_session_seg = {
        (s.session_id, s.segment_index): s for s in segments_by_id.values()
    }
    segs = []
    missing = []
    for session_id, seg_num_1based in items:
        seg = by_session_seg.get((session_id, seg_num_1based - 1))
        if seg is None:
            missing.append(f'{session_id}:{seg_num_1based}')
        else:
            segs.append(seg)
    if missing:
        raise FrozenArtifactError(
            f"Testset #{n}: {len(missing)} segment(s) not found in current master segments: "
            f"{missing[:5]}" + (" (and more)" if len(missing) > 5 else "")
        )

    # Validate segment content has not changed since testset was created
    meta_path = _paths.testset_meta_path(run_dir, n)
    if os.path.isfile(meta_path):
        with open(meta_path, encoding='utf-8') as fh:
            meta = json.load(fh)
        by_key = {(e['session_id'], e['seg_num']): e['sha256'] for e in meta['segments']}
        changed = []
        for seg in segs:
            key = (seg.session_id, seg.segment_index + 1)
            expected = by_key.get(key)
            if expected is not None:
                actual = hashlib.sha256(seg.text.encode()).hexdigest()
                if actual != expected:
                    changed.append(f'  {seg.session_id} seg {seg.segment_index + 1}')
        if changed:
            print(f'WARNING: Testset #{n} — {len(changed)} segment(s) have changed content:')
            for c in changed[:10]:
                print(c)
            if len(changed) > 10:
                print(f'  ... and {len(changed) - 10} more')
            _confirm = confirm_fn if confirm_fn is not None else _default_confirm
            if not _confirm(f'Segment content changed in testset #{n}. Continue? [y/N] '):
                raise FrozenArtifactError(
                    f'Refresh aborted: segment content changed in testset #{n}.'
                )

    kind = _detect_worksheet_kind(human_path)
    ai_path = _paths.testset_ai_flat_path(run_dir, n)

    if kind == 'purer':
        _write_purer_answer_key(segs, framework, ai_path, n, n_total)
    elif kind == 'codebook':
        _write_codebook_answer_key(segs, codebook, ai_path, n, n_total)
    else:
        _write_ai_testset(segs, framework, ai_path, n, n_total, codebook_enabled)

    # Update the human worksheet header to reflect testset number and current total
    _sync_worksheet_header(human_path, n, n_total)

    return ai_path


def _parse_worksheet_items(worksheet_path: str) -> List[tuple]:
    """Parse (session_id, segment_number_1based) pairs from [ITEM NNN] lines."""
    import re as _re
    pattern = _re.compile(r'\[ITEM\s+\d+\]\s+Session:\s+(\S+)\s+Segment\s+(\d+)')
    items = []
    with open(worksheet_path, encoding='utf-8') as fh:
        for line in fh:
            m = pattern.search(line)
            if m:
                items.append((m.group(1), int(m.group(2))))
    return items


def _detect_worksheet_kind(worksheet_path: str) -> str:
    """Detect testset kind ('purer', 'codebook', or 'vaamr') from worksheet header."""
    with open(worksheet_path, encoding='utf-8') as fh:
        for line in fh:
            if 'PURER' in line and 'VALIDATION TEST SET' in line:
                return 'purer'
            if 'CODEBOOK' in line and 'VALIDATION TEST SET' in line:
                return 'codebook'
            if 'VALIDATION TEST SET' in line:
                return 'vaamr'
    return 'vaamr'


import re as _re_module
_HEADER_RE = _re_module.compile(
    r'((?:PURER |CODEBOOK )?VALIDATION TEST SET\s+)\d+(\s+of\s+)\d+(\s*—)'
)


def _worksheet_is_legacy_import(meta_path: str) -> bool:
    """True if a worksheet's .meta.json marks it as an imported legacy (validated) testset."""
    try:
        with open(meta_path, encoding='utf-8') as fh:
            return bool(json.load(fh).get('legacy_import'))
    except (OSError, ValueError):
        return False


def _sync_worksheet_header(path: str, n: int, n_total: int) -> None:
    """Replace the 'K of M' in a worksheet header (human or AI key) with 'n of n_total'.

    Only the header line changes; segment selection, text and item order are untouched.
    """
    with open(path, encoding='utf-8') as fh:
        content = fh.read()
    updated = _HEADER_RE.sub(
        lambda m: f'{m.group(1)}{n}{m.group(2)}{n_total}{m.group(3)}',
        content, count=1,
    )
    if updated != content:
        with open(path, 'w', encoding='utf-8') as fh:
            fh.write(updated)


def _collect_used_segment_ids(run_dir: str) -> set:
    """Return set of (session_id, seg_num_1based) for all segments in existing testsets."""
    import re as _re
    ts_dir = _paths.testsets_dir(run_dir)
    if not os.path.isdir(ts_dir):
        return set()
    pattern = _re.compile(r'^human_classification_testset_worksheet_(\d+)\.txt$')
    used: set = set()
    for f in os.listdir(ts_dir):
        if pattern.match(f):
            used.update(_parse_worksheet_items(os.path.join(ts_dir, f)))
    return used


def _write_testset_meta(run_dir: str, n: int, segs: List[Segment]) -> None:
    """Write per-segment SHA256 sidecar for human worksheet #n (under 02_meta)."""
    import hashlib
    meta = {'segments': [
        {
            'session_id': s.session_id,
            'seg_num': s.segment_index + 1,
            'sha256': hashlib.sha256(s.text.encode()).hexdigest(),
        }
        for s in segs
    ]}
    meta_path = _paths.testset_meta_path(run_dir, n)
    os.makedirs(os.path.dirname(meta_path), exist_ok=True)
    with open(meta_path, 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=2)


def _default_confirm(prompt: str) -> bool:
    """Prompt the user interactively; return True if they confirm."""
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ('y', 'yes')


# Phase 1 back-compat — remove with legacy_migration.py
export_validation_testsets = generate_or_refresh_validation_testsets


# ---------------------------------------------------------------------------
# Pool builders
# ---------------------------------------------------------------------------

def _pool_vaamr(segments: List[Segment]) -> List[Segment]:
    """Participant segments — the VAAMR testset pool."""
    return [
        s for s in segments
        if s.speaker and s.speaker.lower() not in ('therapist', 't', 'interviewer')
    ]


def _pool_purer(segments: List[Segment]) -> List[Segment]:
    """Therapist segments that have a non-null purer_primary label."""
    return [
        s for s in segments
        if s.speaker and s.speaker.lower() in ('therapist', 't', 'interviewer')
        and getattr(s, 'purer_primary', None) is not None
    ]


def _pool_codebook(segments: List[Segment]) -> List[Segment]:
    """Participant segments that have at least one codebook_labels_ensemble entry."""
    return [
        s for s in segments
        if s.speaker and s.speaker.lower() not in ('therapist', 't', 'interviewer')
        and getattr(s, 'codebook_labels_ensemble', None)
    ]


def _stratified_sample(
    pool: List[Segment],
    n_sets: int,
    set_index: int,
    fraction_per_set: float,
    random_seed: int,
    key_fn,
) -> List[Segment]:
    """
    Stratify pool by key_fn, then produce the set_index-th split of n_sets.
    Returns segments for the requested set, sorted by (session_id, segment_index).
    """
    import random

    by_key: Dict[object, List[Segment]] = defaultdict(list)
    for seg in pool:
        by_key[key_fn(seg)].append(seg)

    rng = random.Random(random_seed)
    sets: List[List[Segment]] = [[] for _ in range(n_sets)]
    for group in by_key.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        n_per_set = max(1, round(len(shuffled) * fraction_per_set))
        for i in range(n_sets):
            start = i * n_per_set
            end = min(start + n_per_set, len(shuffled))
            if start < len(shuffled):
                sets[i].extend(shuffled[start:end])

    result = sets[set_index - 1]
    result.sort(key=lambda x: (x.session_id, x.segment_index))
    return result


# ---------------------------------------------------------------------------
# PURER worksheet and answer-key writers
# ---------------------------------------------------------------------------

def _write_purer_human_worksheet_to_handle(segs, framework, fh, set_idx, n_sets):
    import datetime as _dt

    move_labels: str = '?'
    if framework is not None:
        move_labels = ', '.join(f"{t.theme_id}={t.short_name}" for t in framework.themes)

    fh.write("=" * _W + "\n")
    fh.write(f"PURER VALIDATION TEST SET {set_idx} of {n_sets} — HUMAN CODING WORKSHEET\n")
    fh.write("=" * _W + "\n")
    fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
             f"Segments: {len(segs)}\n")
    fh.write("Instructions: Code each therapist segment independently. "
             "Do not consult other sources.\n")
    fh.write(f"PURER move labels: {move_labels}\n\n")

    for item_num, seg in enumerate(segs, start=1):
        fh.write("=" * _W + "\n")
        fh.write(
            f"[ITEM {item_num:03d}]  Session: {seg.session_id}   "
            f"Segment {seg.segment_index + 1:03d}\n"
        )
        fh.write(
            f"            "
            f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
            f"{seg.word_count}w\n"
        )
        fh.write("-" * _W + "\n")
        for line in textwrap.wrap(seg.text, width=76,
                                  initial_indent="  ",
                                  subsequent_indent="  ") or ["  "]:
            fh.write(line + "\n")
        fh.write("\n")
        fh.write("  Primary PURER move: ___   Secondary (optional): ___\n")
        fh.write("  Rationale: " + "_" * 60 + "\n")
        fh.write("  " + "_" * 72 + "\n")
        fh.write("\n")

    # Append compact PURER definitions at the bottom for reference
    if framework is not None:
        fh.write("=" * _W + "\n")
        fh.write("PURER MOVE REFERENCE\n")
        fh.write("=" * _W + "\n")
        for theme in sorted(framework.themes, key=lambda t: t.theme_id):
            fh.write(f"  [{theme.theme_id}] {theme.key} — {theme.name}\n")
            for line in textwrap.wrap(theme.definition, width=_W - 4,
                                      initial_indent="    ", subsequent_indent="    "):
                fh.write(line + "\n")
            fh.write("\n")


def _write_purer_answer_key(segs, framework, output_path, set_idx, n_sets):
    import datetime as _dt

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    def _mname(move_id) -> str:
        return _theme_name_from(move_id, id_to_name)

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("=" * _W + "\n")
        fh.write(f"PURER VALIDATION TEST SET {set_idx} of {n_sets} — AI ANSWER KEY\n")
        fh.write("=" * _W + "\n")
        fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
                 f"Segments: {len(segs)}\n")
        fh.write("Use for inter-rater reliability comparison against the human coding "
                 f"worksheet {set_idx}.\n\n")

        for item_num, seg in enumerate(segs, start=1):
            fh.write("=" * _W + "\n")
            fh.write(
                f"[ITEM {item_num:03d}]  Session: {seg.session_id}   "
                f"Segment {seg.segment_index + 1:03d}\n"
            )
            fh.write(
                f"            "
                f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                f"{seg.word_count}w\n"
            )
            fh.write("-" * _W + "\n")
            for line in textwrap.wrap(seg.text, width=76,
                                      initial_indent="  ",
                                      subsequent_indent="  ") or ["  "]:
                fh.write(line + "\n")
            fh.write("\n")

            # PURER rater ballots
            purer_votes = getattr(seg, 'purer_rater_votes', None) or []
            if purer_votes:
                fh.write("PURER RATER BALLOTS\n")
                for rv in purer_votes:
                    rid = rv.get('rater', '?')
                    vote = rv.get('vote', '?')
                    stage = rv.get('stage')
                    conf = rv.get('confidence')
                    just = (rv.get('justification') or '').strip()
                    sec = rv.get('secondary_stage')
                    sec_conf = rv.get('secondary_confidence')
                    if vote == 'CODED':
                        sec_part = ""
                        if sec is not None:
                            sec_conf_str = f", conf={_fmt_conf(sec_conf)}" if sec_conf is not None else ""
                            sec_part = f"  (2nd: {_mname(sec)}{sec_conf_str})"
                        fh.write(f"  [{rid}]  {_mname(stage)}  conf={_fmt_conf(conf)}{sec_part}\n")
                    elif vote == 'ABSTAIN':
                        fh.write(f"  [{rid}]  ABSTAIN\n")
                    else:
                        fh.write(f"  [{rid}]  ERROR\n")
                    if just:
                        for line in textwrap.wrap(just, width=70,
                                                  initial_indent="      → ",
                                                  subsequent_indent="        "):
                            fh.write(line + "\n")
                fh.write("\n")
            elif seg.purer_primary is not None:
                fh.write("PURER CONSENSUS\n")
                sec_part = ""
                if seg.purer_secondary is not None:
                    sec_part = f"  (2nd: {_mname(seg.purer_secondary)}, conf={_fmt_conf(seg.purer_confidence_secondary)})"
                fh.write(f"  {_mname(seg.purer_primary)}  conf={_fmt_conf(seg.purer_confidence_primary)}{sec_part}\n")
                agreement = getattr(seg, 'purer_agreement_level', None) or '?'
                fh.write(f"  Agreement: {agreement}\n")
                fh.write("\n")


# ---------------------------------------------------------------------------
# Codebook worksheet and answer-key writers
# ---------------------------------------------------------------------------

def _write_codebook_human_worksheet_to_handle(segs, codebook, fh, set_idx, n_sets):
    import datetime as _dt

    fh.write("=" * _W + "\n")
    fh.write(f"CODEBOOK VALIDATION TEST SET {set_idx} of {n_sets} — HUMAN CODING WORKSHEET\n")
    fh.write("=" * _W + "\n")
    fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
             f"Segments: {len(segs)}\n")
    fh.write("Instructions: Select all applicable codebook codes for each segment.\n\n")

    for item_num, seg in enumerate(segs, start=1):
        fh.write("=" * _W + "\n")
        fh.write(
            f"[ITEM {item_num:03d}]  Session: {seg.session_id}   "
            f"Segment {seg.segment_index + 1:03d}\n"
        )
        fh.write(
            f"            "
            f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
            f"{seg.word_count}w   Participant: {seg.participant_id or '?'}\n"
        )
        fh.write("-" * _W + "\n")
        for line in textwrap.wrap(seg.text, width=76,
                                  initial_indent="  ",
                                  subsequent_indent="  ") or ["  "]:
            fh.write(line + "\n")
        fh.write("\n")
        fh.write("  Codes (select all that apply): _" + "_" * 60 + "\n")
        fh.write("  Rationale: " + "_" * 60 + "\n")
        fh.write("  " + "_" * 72 + "\n")
        fh.write("\n")

    if codebook is not None:
        fh.write("=" * _W + "\n")
        fh.write("CODEBOOK REFERENCE\n")
        fh.write("=" * _W + "\n")
        for code in getattr(codebook, 'codes', []):
            fh.write(f"  [{code.code_id}] {code.category}\n")
            desc = (code.description or '').strip()
            if desc:
                for line in textwrap.wrap(desc, width=_W - 6,
                                          initial_indent="    ", subsequent_indent="    "):
                    fh.write(line + "\n")
            fh.write("\n")


def _write_codebook_answer_key(segs, codebook, output_path, set_idx, n_sets):
    import datetime as _dt

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("=" * _W + "\n")
        fh.write(f"CODEBOOK VALIDATION TEST SET {set_idx} of {n_sets} — AI ANSWER KEY\n")
        fh.write("=" * _W + "\n")
        fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
                 f"Segments: {len(segs)}\n")
        fh.write("Use for inter-rater reliability comparison against the human coding "
                 f"worksheet {set_idx}.\n\n")

        for item_num, seg in enumerate(segs, start=1):
            fh.write("=" * _W + "\n")
            fh.write(
                f"[ITEM {item_num:03d}]  Session: {seg.session_id}   "
                f"Segment {seg.segment_index + 1:03d}\n"
            )
            fh.write(
                f"            "
                f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                f"{seg.word_count}w   Participant: {seg.participant_id or '?'}\n"
            )
            fh.write("-" * _W + "\n")
            for line in textwrap.wrap(seg.text, width=76,
                                      initial_indent="  ",
                                      subsequent_indent="  ") or ["  "]:
                fh.write(line + "\n")
            fh.write("\n")

            ensemble = getattr(seg, 'codebook_labels_ensemble', None) or []
            llm = getattr(seg, 'codebook_labels_llm', None) or []
            emb = getattr(seg, 'codebook_labels_embedding', None) or []
            if ensemble:
                fh.write(f"CODES (ensemble): {', '.join(str(c) for c in ensemble)}\n")
            if llm and not ensemble:
                fh.write(f"CODES (LLM): {', '.join(str(c) for c in llm)}\n")
            if emb and not ensemble:
                fh.write(f"CODES (embedding): {', '.join(str(c) for c in emb)}\n")

            disagreements = getattr(seg, 'codebook_disagreements', None) or []
            if disagreements:
                fh.write(f"DISAGREEMENTS: {', '.join(str(c) for c in disagreements)}\n")
            fh.write("\n")


# ---------------------------------------------------------------------------
# Snapshot writer
# ---------------------------------------------------------------------------


def _write_human_testset(
    segs: List[Segment],
    framework,
    output_path: str,
    set_idx: int,
    n_sets: int,
) -> None:
    with open(output_path, 'w', encoding='utf-8') as fh:
        _write_human_testset_to_handle(segs, framework, fh, set_idx, n_sets)


def _write_human_testset_to_handle(
    segs: List[Segment],
    framework,
    fh,
    set_idx: int,
    n_sets: int,
) -> None:
    import datetime as _dt

    stage_labels: str = '?'
    if framework is not None:
        stage_labels = ', '.join(f"{t.theme_id}={t.short_name}" for t in framework.themes)

    fh.write("=" * 78 + "\n")
    fh.write(f"VALIDATION TEST SET {set_idx} of {n_sets} — HUMAN CODING WORKSHEET\n")
    fh.write("=" * 78 + "\n")
    fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
             f"Segments: {len(segs)}\n")
    fh.write("Instructions: Code each segment independently. "
             "Do not consult other sources.\n")
    fh.write(f"Stage labels: {stage_labels}\n\n")

    for item_num, seg in enumerate(segs, start=1):
        speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker or '?'])
        fh.write("=" * 78 + "\n")
        fh.write(
            f"[ITEM {item_num:03d}]  Session: {seg.session_id}   "
            f"Segment {seg.segment_index + 1:03d}\n"
        )
        fh.write(
            f"            "
            f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
            f"{seg.word_count}w   Participant: {seg.participant_id or speakers_str}\n"
        )
        fh.write("-" * 78 + "\n")
        for line in textwrap.wrap(seg.text, width=76,
                                  initial_indent="  ",
                                  subsequent_indent="  ") or ["  "]:
            fh.write(line + "\n")
        fh.write("\n")
        fh.write("  Primary stage: ___   Secondary (optional): ___\n")
        fh.write("  Rationale: " + "_" * 60 + "\n")
        fh.write("  " + "_" * 72 + "\n")
        fh.write("\n")


def _write_ai_testset(
    segs: List[Segment],
    framework,
    output_path: str,
    set_idx: int,
    n_sets: int,
    codebook_enabled: bool,
) -> None:
    import datetime as _dt

    id_to_name: dict = {}
    if framework is not None:
        id_to_name = {t.theme_id: t.short_name for t in framework.themes}

    def _tname(stage) -> str:
        return _theme_name_from(stage, id_to_name)

    with open(output_path, 'w', encoding='utf-8') as fh:
        fh.write("=" * 78 + "\n")
        fh.write(f"VALIDATION TEST SET {set_idx} of {n_sets} — AI CLASSIFICATION WORKSHEET\n")
        fh.write("=" * 78 + "\n")
        fh.write(f"Generated: {_dt.datetime.utcnow().strftime('%Y-%m-%d')}   "
                 f"Segments: {len(segs)}\n")
        fh.write("Use for inter-rater reliability comparison against the human coding "
                 f"worksheet {set_idx}.\n\n")

        for item_num, seg in enumerate(segs, start=1):
            speakers_str = ", ".join(seg.speakers_in_segment or [seg.speaker or '?'])
            fh.write("=" * 78 + "\n")
            fh.write(
                f"[ITEM {item_num:03d}]  Session: {seg.session_id}   "
                f"Segment {seg.segment_index + 1:03d}\n"
            )
            fh.write(
                f"            "
                f"{_ms_to_hms(seg.start_time_ms)}–{_ms_to_hms(seg.end_time_ms)}   "
                f"{seg.word_count}w   Participant: {seg.participant_id or speakers_str}\n"
            )
            fh.write("-" * 78 + "\n")
            for line in textwrap.wrap(seg.text, width=76,
                                      initial_indent="  ",
                                      subsequent_indent="  ") or ["  "]:
                fh.write(line + "\n")
            fh.write("\n")

            # Rater ballots
            rater_votes = getattr(seg, 'rater_votes', None) or []
            if rater_votes:
                fh.write("RATER BALLOTS\n")
                for rv in rater_votes:
                    rid = rv.get('rater', '?')
                    vote = rv.get('vote', '?')
                    stage = rv.get('stage')
                    conf = rv.get('confidence')
                    just = (rv.get('justification') or '').strip()
                    sec = rv.get('secondary_stage')
                    sec_conf = rv.get('secondary_confidence')
                    if vote == 'CODED':
                        sec_part = ""
                        if sec is not None:
                            sec_conf_str = f", conf={_fmt_conf(sec_conf)}" if sec_conf is not None else ""
                            sec_part = f"  (2nd: {_tname(sec)}{sec_conf_str})"
                        fh.write(f"  [{rid}]  {_tname(stage)}  conf={_fmt_conf(conf)}{sec_part}\n")
                    elif vote == 'ABSTAIN':
                        fh.write(f"  [{rid}]  ABSTAIN (irrelevant to framework)\n")
                    else:
                        fh.write(f"  [{rid}]  ERROR (no parseable response)\n")
                    if just:
                        for line in textwrap.wrap(just, width=70,
                                                  initial_indent="      → ",
                                                  subsequent_indent="        "):
                            fh.write(line + "\n")
                fh.write("\n")
            elif seg.primary_stage is not None:
                # Single-rater fallback (no interrater runs)
                fh.write("RATER BALLOTS\n")
                just = (getattr(seg, 'llm_justification', None) or '').strip()
                sec = seg.secondary_stage
                sec_part = ""
                if sec is not None:
                    sec_part = f"  (2nd: {_tname(sec)}, conf={_fmt_conf(seg.llm_confidence_secondary)})"
                fh.write(f"  [rater_0]  {_tname(seg.primary_stage)}  "
                         f"conf={_fmt_conf(seg.llm_confidence_primary)}{sec_part}\n")
                if just:
                    for line in textwrap.wrap(just, width=70,
                                              initial_indent="      → ",
                                              subsequent_indent="        "):
                        fh.write(line + "\n")
                fh.write("\n")

            # Consensus
            agreement = getattr(seg, 'agreement_level', None) or '?'
            rater_ids = getattr(seg, 'rater_ids', None) or []
            n_raters = len(rater_ids)
            agreement_frac = getattr(seg, 'agreement_fraction', None) or 0.0
            n_agree = int(round(agreement_frac * max(n_raters, 1)))

            fh.write("CONSENSUS: ")
            if seg.primary_stage is not None:
                agree_str = f"{n_agree}/{n_raters}" if n_raters else agreement
                fh.write(f"{_tname(seg.primary_stage)} (stage {seg.primary_stage})  "
                         f"[{agreement}, {agree_str} agreement]\n")
                fh.write(f"  Mean confidence: {_fmt_conf(seg.llm_confidence_primary)}")
                if seg.secondary_stage is not None:
                    fh.write(f"   Secondary: {_tname(seg.secondary_stage)} "
                             f"(conf {_fmt_conf(seg.llm_confidence_secondary)})")
                fh.write("\n")
            else:
                consensus_vote = getattr(seg, 'consensus_vote', None)
                if consensus_vote == 'ABSTAIN':
                    fh.write(f"UNCLASSIFIED — consensus ABSTAIN ({agreement})\n")
                elif agreement == 'split':
                    fh.write("UNCLASSIFIED — SPLIT VOTE (no majority); flagged for review\n")
                else:
                    fh.write(f"UNCLASSIFIED ({agreement})\n")

            # Codebook codes
            if codebook_enabled:
                codes = getattr(seg, 'codebook_labels_ensemble', None) or \
                        getattr(seg, 'codebook_labels_llm', None) or []
                if codes:
                    label = "CODES" if getattr(seg, 'codebook_labels_ensemble', None) else "CODES (LLM)"
                    fh.write(f"{label}: {', '.join(str(c) for c in codes)}\n")

            fh.write("\n")
