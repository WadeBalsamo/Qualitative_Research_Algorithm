"""Export functions for validation test sets and review forms."""

import os
import textwrap
from collections import defaultdict
from typing import List, Dict

from classification_tools.data_structures import Segment
from .. import output_paths as _paths
from ._shared import _ms_to_hms, _fmt_conf, _theme_name_from


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



def export_validation_testsets(
    segments: List[Segment],
    framework,
    run_dir: str,
    n_sets: int = 2,
    fraction_per_set: float = 0.10,
    random_seed: int = 42,
    codebook_enabled: bool = False,
) -> List[str]:
    """
    Generate n_sets pairs of validation worksheets (human blind + AI-disclosed)
    from a stratified random sample of participant segments across all sessions
    and cohorts.  Each set is non-overlapping.

    Outputs to <output_dir>/validation/testsets/.
    Returns list of written file paths.
    """
    import random

    pool = [
        s for s in segments
        if s.speaker and s.speaker.lower() not in ('therapist', 't', 'interviewer')
    ]

    by_cohort: Dict[object, List[Segment]] = defaultdict(list)
    for seg in pool:
        by_cohort[seg.cohort_id if seg.cohort_id is not None else 'none'].append(seg)

    rng = random.Random(random_seed)
    sets: List[List[Segment]] = [[] for _ in range(n_sets)]

    for cohort_segs in by_cohort.values():
        shuffled = list(cohort_segs)
        rng.shuffle(shuffled)
        n_per_set = max(1, round(len(shuffled) * fraction_per_set))
        for i in range(n_sets):
            start = i * n_per_set
            end = min(start + n_per_set, len(shuffled))
            if start < len(shuffled):
                sets[i].extend(shuffled[start:end])

    for s in sets:
        s.sort(key=lambda x: (x.session_id, x.segment_index))

    testsets_dir = _paths.testsets_dir(run_dir)
    os.makedirs(testsets_dir, exist_ok=True)
    written: List[str] = []

    for idx, segs in enumerate(sets, start=1):
        human_path = os.path.join(
            testsets_dir, f'human_classification_testset_worksheet_{idx}.txt'
        )
        ai_path = os.path.join(
            testsets_dir, f'AI_classification_testset_worksheet_{idx}.txt'
        )
        _write_human_testset(segs, framework, human_path, idx, n_sets)
        _write_ai_testset(segs, framework, ai_path, idx, n_sets, codebook_enabled)
        written.extend([human_path, ai_path])

    return written


def _write_human_testset(
    segs: List[Segment],
    framework,
    output_path: str,
    set_idx: int,
    n_sets: int,
) -> None:
    import datetime as _dt

    stage_labels: str = '?'
    if framework is not None:
        stage_labels = ', '.join(f"{t.theme_id}={t.short_name}" for t in framework.themes)

    with open(output_path, 'w', encoding='utf-8') as fh:
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
