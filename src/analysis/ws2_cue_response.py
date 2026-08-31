"""
analysis/ws2_cue_response.py
----------------------------
WS2 — turn-accurate PURER cue->response re-derivation for the EXISTING
cohorts 1-3 corpus (data/MMORE_Processed), recovered from the RAW VTT
transcripts.

Why this module exists
======================
The frozen ``segments`` table collapses the therapist prompting between
participant turns (a segmentation bug fixed only for cohort 4+).  The cue
units the production PURER pass can build on the frozen segments are therefore
SPARSE: the FROM->CUE->TO transition table has only ~11 populated
(from_stage x move) cells, and most same-participant transitions carry no
therapist cue at all.

The RAW transcripts (``01_transcripts_inputs/*.vtt``) still contain the
therapist's prompt-before-share 95-100% of the time, so the evidence is
recoverable.  This module:

  1. Parses the raw VTTs (``Name: text`` cues) and collapses consecutive
     same-speaker cues into turns (:func:`parse_vtt_turns`).
  2. Maps speakers to roles via the project speaker key.
  3. INHERITS the validated VAAMR stage onto each participant turn from the
     READ-ONLY frozen ``segments``/``theme_labels`` (max temporal overlap,
     same participant) -- it never recodes VAAMR.
  4. Builds turn-accurate, WITHIN-participant cue units (FROM turn -> therapist
     speech in the gap -> the SAME participant's NEXT OWN staged turn) by
     reusing :func:`process.cue_blocks.build_cue_blocks` with
     ``require_same_participant=True`` (the canonical use-(B) unit, identical
     in spirit to ``analysis/reports/transition_report.py``).
  5. Re-classifies PURER in cue-block mode (whole therapist run between the two
     participant turns classified ONCE) using the existing PURER prompt /
     parser / framework and the LM Studio backend.
  6. Computes the cue->response mechanism (FROM_stage -> dominant move ->
     TO_stage; mean delta and per-cell table) under the same-participant
     constraint and writes a text report.

ALL outputs go to a SEPARATE namespace
(``data/MMORE_Processed/ws2_rederivation/``: ``ws2.db`` + ``reports/``).
The live ``qra.db`` (``segments`` / ``theme_labels`` / ``purer_labels`` /
``classification_runs`` / ``label_ballots``) is opened READ-ONLY and is never
written, cleared, or rebuilt.

The classification loop is resumable: each cue unit's label is written to
``ws2.db`` immediately, and an already-labeled unit is skipped on restart.
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Dict, List, Optional, Tuple

# First-party imports (require src/ on sys.path; see _ensure_src_path()).
from classification_tools.data_structures import Segment
from classification_tools.theme_llm.llm_classifier import (
    PURER_CUE_PROMPT_TEMPLATE,
    _build_context_block,
    _parse_single_run,
)
from constructs.purer import get_purer_framework, PURER_TIE_BREAK_ORDER
from process.cue_blocks import (
    cue_blocks_from_segments,
    MIN_SAME_PARTICIPANT_BLOCKS,
    insufficient_blocks_banner,
)
from process.transcript_ingestion import load_vtt_session


# Move id -> short tag (mirrors analysis/reports/_formatting._PURER_SHORT).
PURER_SHORT = {0: 'P', 1: 'U', 2: 'R', 3: 'E', 4: 'R2'}
PURER_NAME = {0: 'Phenomenological', 1: 'Utilization', 2: 'Reframing',
              3: 'Educate/Expectancy', 4: 'Reinforcement'}


# ---------------------------------------------------------------------------
# Speaker key
# ---------------------------------------------------------------------------

def load_speaker_key(path: str) -> Dict[str, Dict[str, str]]:
    """Load ``speaker_anonymization_key.json`` -> {name: {role, anonymized_id}}."""
    with open(path, 'r', encoding='utf-8') as fh:
        return json.load(fh)


def _role_to_speaker(role: str) -> str:
    """Map a speaker key role to a cue-block speaker class.

    ``participant`` and ``therapist`` map through; everything else
    (``staff``, unknown) becomes ``other`` so it is excluded from both the
    participant anchors AND the therapist cue (it still occupies the timeline
    so it never gets folded into a cue window).
    """
    if role == 'participant':
        return 'participant'
    if role == 'therapist':
        return 'therapist'
    return 'other'


# ---------------------------------------------------------------------------
# Raw VTT -> turns
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    turn_id: str
    session_id: str
    speaker_name: str
    role: str
    speaker: str          # 'participant' | 'therapist' | 'other'
    participant_id: str   # anonymized id for participants, '' otherwise
    start_ms: int
    end_ms: int
    text: str
    primary_stage: Optional[int] = None  # inherited later


def parse_vtt_turns(vtt_path: str, session_id: str,
                    speaker_key: Dict[str, Dict[str, str]]):
    """Parse a raw VTT into TURNS (consecutive same-speaker cues collapsed).

    Returns ``(turns, n_cues, unknown_speakers)`` where ``turns`` is a list of
    :class:`Turn` in start-time order, ``n_cues`` is the number of raw VTT cues
    parsed, and ``unknown_speakers`` is a ``{name: count}`` map of speaker
    prefixes not present in the speaker key.
    """
    loaded = load_vtt_session(vtt_path)
    cues = loaded['sentences']  # [{text, speaker, start, end}, ...] in file order
    n_cues = len(cues)

    unknown: Dict[str, int] = defaultdict(int)
    turns: List[Turn] = []
    cur: Optional[Turn] = None
    seq = 0

    for cue in cues:
        name = (cue.get('speaker') or '').strip()
        text = (cue.get('text') or '').strip()
        if not text:
            continue
        info = speaker_key.get(name)
        if info is None:
            unknown[name] += 1
            role = 'unknown'
            anon = ''
        else:
            role = info.get('role', 'unknown')
            anon = info.get('anonymized_id', '')
        speaker = _role_to_speaker(role)
        participant_id = anon if speaker == 'participant' else ''
        start_ms = int(round(float(cue.get('start', 0.0)) * 1000))
        end_ms = int(round(float(cue.get('end', 0.0)) * 1000))

        # Collapse consecutive same-speaker cues into one turn.
        if cur is not None and cur.speaker_name == name:
            cur.text = (cur.text + ' ' + text).strip()
            cur.end_ms = max(cur.end_ms, end_ms)
            continue

        cur = Turn(
            turn_id=f'{session_id}_turn{seq:04d}',
            session_id=session_id,
            speaker_name=name,
            role=role,
            speaker=speaker,
            participant_id=participant_id,
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
        )
        turns.append(cur)
        seq += 1

    return turns, n_cues, dict(unknown)


# ---------------------------------------------------------------------------
# Read-only frozen VAAMR (segments + theme_labels)
# ---------------------------------------------------------------------------

@dataclass
class FrozenParticipantSeg:
    participant_id: str
    session_id: str
    start_ms: int
    end_ms: int
    primary_stage: Optional[int]


def load_frozen_participant_segments(live_db_path: str
                                     ) -> Dict[str, List[FrozenParticipantSeg]]:
    """Load READ-ONLY frozen participant segments + their VAAMR stage.

    Joins ``segments`` (frozen) to ``theme_labels`` (validated VAAMR overlay)
    for ``speaker='participant'``.  Returns ``{session_id: [FrozenParticipantSeg]}``.
    The DB is opened in ``mode=ro`` -- this function NEVER writes.
    """
    uri = f'file:{live_db_path}?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            """
            SELECT s.participant_id, s.session_id, s.start_time_ms, s.end_time_ms,
                   t.primary_stage
            FROM segments s
            LEFT JOIN theme_labels t ON s.segment_id = t.segment_id
            WHERE s.speaker = 'participant'
            """
        ).fetchall()
    finally:
        conn.close()

    by_session: Dict[str, List[FrozenParticipantSeg]] = defaultdict(list)
    for pid, sid, st, en, stage in rows:
        by_session[sid].append(FrozenParticipantSeg(
            participant_id=str(pid or ''),
            session_id=str(sid or ''),
            start_ms=int(st or 0),
            end_ms=int(en or 0),
            primary_stage=(int(stage) if stage is not None else None),
        ))
    return by_session


def inherit_stages(turns: List[Turn],
                   frozen_by_session: Dict[str, List[FrozenParticipantSeg]]):
    """Inherit the validated VAAMR primary_stage onto each participant turn.

    For each participant turn, among frozen participant segments of the SAME
    participant in the SAME session, pick the one with MAXIMAL temporal overlap
    with the turn's span; if none overlaps, fall back to the segment whose span
    CONTAINS the turn's midpoint.  The inherited stage may be ``None`` (the
    overlapping frozen segment was itself unstaged, or no match exists).

    Returns ``(n_participant_turns, n_inherited)`` where ``n_inherited`` is how
    many participant turns received a non-null stage.

    Documented approximation: where a merged frozen segment hid an INTERNAL
    stage change, all raw turns overlapping it inherit that single stage.
    """
    n_part = 0
    n_inherited = 0
    for t in turns:
        if t.speaker != 'participant':
            continue
        n_part += 1
        candidates = [
            fs for fs in frozen_by_session.get(t.session_id, [])
            if fs.participant_id == t.participant_id
        ]
        if not candidates:
            continue
        best = None
        best_overlap = 0
        for fs in candidates:
            overlap = min(t.end_ms, fs.end_ms) - max(t.start_ms, fs.start_ms)
            if overlap > best_overlap:
                best_overlap = overlap
                best = fs
        if best is None:
            # No positive overlap -> containment-of-midpoint fallback.
            mid = (t.start_ms + t.end_ms) // 2
            for fs in candidates:
                if fs.start_ms <= mid <= fs.end_ms:
                    best = fs
                    break
        if best is not None:
            t.primary_stage = best.primary_stage
            if best.primary_stage is not None:
                n_inherited += 1
    return n_part, n_inherited


# ---------------------------------------------------------------------------
# Turns -> Segment objects -> cue blocks
# ---------------------------------------------------------------------------

def turns_to_segments(turns: List[Turn]) -> List[Segment]:
    """Convert :class:`Turn` objects into pipeline :class:`Segment` objects.

    ``primary_stage`` carries the inherited VAAMR stage so that
    ``cue_blocks_from_segments(..., stage_attr='primary_stage')`` anchors on it.
    """
    segs: List[Segment] = []
    for t in turns:
        segs.append(Segment(
            segment_id=t.turn_id,
            session_id=t.session_id,
            participant_id=t.participant_id,
            speaker=t.speaker,
            text=t.text,
            word_count=len(t.text.split()),
            start_time_ms=t.start_ms,
            end_time_ms=t.end_ms,
            primary_stage=t.primary_stage,
        ))
    return segs


@dataclass
class CueUnit:
    cue_id: str
    session_id: str
    participant_id: str
    from_turn_id: str
    to_turn_id: str
    from_stage: int
    to_stage: int
    from_text: str
    cue_text: str
    cue_words: int
    context_block: str
    n_therapist_turns: int


def build_cue_units(turn_segments: List[Segment],
                    *, max_lesson_words: int = 400,
                    skip_lessons: bool = True,
                    context_window: int = 6,
                    max_context_words: int = 1000):
    """Build within-participant cue units from turn segments.

    Reuses :func:`process.cue_blocks.cue_blocks_from_segments` with
    ``require_same_participant=True`` and ``require_stage=True`` so FROM and TO
    are the SAME participant's consecutive STAGED own turns and the cue is the
    therapist speech strictly between them.

    Long didactic monologues (therapist words > ``max_lesson_words``) are
    skipped when ``skip_lessons`` is True (mirrors the production
    ``purer_cue.skip_lesson_content`` default), and reported separately.

    Returns ``(cue_units, stats)``.
    """
    sorted_segs, specs = cue_blocks_from_segments(
        turn_segments,
        stage_attr='primary_stage',
        require_stage=True,
        require_same_participant=True,
    )

    stats = {
        'n_specs': len(specs),
        'n_with_cue': 0,
        'n_skipped_lesson': 0,
        'n_empty_cue': 0,
    }

    cue_units: List[CueUnit] = []
    for spec in specs:
        if not spec.therapist_items:
            stats['n_empty_cue'] += 1
            continue
        cue_text = '\n'.join(
            ti.text.strip() for ti in spec.therapist_items if ti.text.strip()
        )
        cue_words = len(cue_text.split())
        if not cue_text:
            stats['n_empty_cue'] += 1
            continue
        if skip_lessons and cue_words > max_lesson_words:
            stats['n_skipped_lesson'] += 1
            continue

        from_seg = spec.from_item
        to_seg = spec.to_item
        ctx = (
            _build_context_block(sorted_segs, spec.from_index,
                                 window_size=context_window,
                                 max_words=max_context_words)
            if spec.from_index >= 0 else ''
        )
        cue_id = f'ws2_cue_{from_seg.segment_id}__to__{to_seg.segment_id}'
        cue_units.append(CueUnit(
            cue_id=cue_id,
            session_id=spec.session_id,
            participant_id=str(spec.from_participant or ''),
            from_turn_id=from_seg.segment_id,
            to_turn_id=to_seg.segment_id,
            from_stage=int(spec.from_stage),
            to_stage=int(spec.to_stage),
            from_text=from_seg.text,
            cue_text=cue_text,
            cue_words=cue_words,
            context_block=ctx,
            n_therapist_turns=len(spec.therapist_items),
        ))
        stats['n_with_cue'] += 1

    return cue_units, stats


# ---------------------------------------------------------------------------
# ws2.db (private store)
# ---------------------------------------------------------------------------

WS2_SCHEMA = """
CREATE TABLE IF NOT EXISTS ws2_cue_units (
    cue_id TEXT PRIMARY KEY,
    session_id TEXT,
    participant_id TEXT,
    from_turn_id TEXT,
    to_turn_id TEXT,
    from_stage INTEGER,
    to_stage INTEGER,
    cue_words INTEGER,
    n_therapist_turns INTEGER,
    cue_text TEXT,
    from_text TEXT
);
CREATE TABLE IF NOT EXISTS ws2_purer_labels (
    cue_id TEXT PRIMARY KEY,
    primary_move INTEGER,
    secondary_move INTEGER,
    primary_confidence REAL,
    vote TEXT,
    justification TEXT,
    model TEXT,
    raw_response TEXT
);
CREATE TABLE IF NOT EXISTS ws2_turns (
    turn_id TEXT PRIMARY KEY,
    session_id TEXT,
    speaker TEXT,
    role TEXT,
    participant_id TEXT,
    start_ms INTEGER,
    end_ms INTEGER,
    primary_stage INTEGER,
    text TEXT
);
"""


def open_ws2_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(WS2_SCHEMA)
    conn.commit()
    return conn


def persist_turns(conn: sqlite3.Connection, turns: List[Turn]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO ws2_turns
           (turn_id, session_id, speaker, role, participant_id,
            start_ms, end_ms, primary_stage, text)
           VALUES (?,?,?,?,?,?,?,?,?)""",
        [(t.turn_id, t.session_id, t.speaker, t.role, t.participant_id,
          t.start_ms, t.end_ms, t.primary_stage, t.text) for t in turns],
    )
    conn.commit()


def persist_cue_units(conn: sqlite3.Connection, units: List[CueUnit]) -> None:
    conn.executemany(
        """INSERT OR REPLACE INTO ws2_cue_units
           (cue_id, session_id, participant_id, from_turn_id, to_turn_id,
            from_stage, to_stage, cue_words, n_therapist_turns, cue_text, from_text)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        [(u.cue_id, u.session_id, u.participant_id, u.from_turn_id, u.to_turn_id,
          u.from_stage, u.to_stage, u.cue_words, u.n_therapist_turns,
          u.cue_text, u.from_text) for u in units],
    )
    conn.commit()


def already_labeled(conn: sqlite3.Connection) -> set:
    """Cue ids with a SETTLED label (CODED or ABSTAIN).

    ERROR ballots are intentionally excluded so a resumable re-run re-attempts
    the units that failed (e.g. backend timeouts under load) without redoing the
    ones that already succeeded.
    """
    return {r[0] for r in conn.execute(
        "SELECT cue_id FROM ws2_purer_labels WHERE vote IN ('CODED','ABSTAIN')")}


# ---------------------------------------------------------------------------
# PURER classification (resumable)
# ---------------------------------------------------------------------------

def _build_purer_prompt(unit: CueUnit, framework, *, compact: bool = False) -> str:
    """Build the canonical PURER cue prompt.

    ``compact=True`` uses a trimmed framework prompt (zero-shot definitions, no
    subtle/adversarial exemplars) so the whole prompt fits inside a small
    (e.g. 8192-token) loaded context window alongside a bounded completion
    budget.  The prompt template, fields, and parser are otherwise identical to
    the production PURER path.
    """
    if compact:
        codebook_string = framework.to_prompt_string(
            zero_shot=True, include_subtle=False, include_adversarial=False,
            n_exemplars=0)
    else:
        codebook_string = framework.to_prompt_string(zero_shot=False)
    return PURER_CUE_PROMPT_TEMPLATE.format(
        framework_name=framework.name,
        framework_description=framework.description,
        codebook_string=codebook_string,
        num_themes=framework.num_themes,
        context_block=unit.context_block or '',
        from_participant_text=unit.from_text.strip(),
        text=unit.cue_text,
    )


def _lmstudio_chat_bounded(base_url: str, model: str, prompt: str,
                           *, temperature: float, max_tokens: int,
                           read_timeout: int) -> str:
    """Single LM Studio chat call with an EXPLICIT bounded ``max_tokens``.

    The shared :class:`LLMClient` requests ``max_tokens = full_context`` which a
    strict server rejects ("Context size has been exceeded") when the prompt is
    large and the model is loaded at a small context.  This helper instead caps
    the completion budget so ``prompt + max_tokens <= context``.

    Returns the best text to parse: the assistant ``content`` if it carries
    JSON, else the ``reasoning_content`` (reasoning models sometimes emit the
    JSON only there).  Raises on transport / server error.
    """
    import requests

    resp = requests.post(
        f'{base_url.rstrip("/")}/chat/completions',
        headers={'Authorization': 'Bearer lm-studio',
                 'Content-Type': 'application/json'},
        json={
            'model': model,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'messages': [{'role': 'user', 'content': prompt}],
        },
        timeout=(30, read_timeout),
    )
    data = resp.json()
    if not isinstance(data, dict) or not data.get('choices'):
        err = data.get('error') if isinstance(data, dict) else str(data)[:200]
        raise ValueError(f'LM Studio returned no choices: {err}')
    msg = data['choices'][0].get('message', {})
    content = (msg.get('content') or '').strip()
    reasoning = (msg.get('reasoning_content') or '').strip()
    # Prefer content when it looks like it contains JSON; else fall back to the
    # reasoning channel (where small reasoning models often place the JSON).
    if content and '{' in content:
        return content
    if reasoning and '{' in reasoning:
        return reasoning
    return content or reasoning


def classify_cue_units(conn: sqlite3.Connection,
                       units: List[CueUnit],
                       *, model: str, backend: str,
                       lmstudio_base_url: str, temperature: float,
                       read_timeout: int = 1200, max_retries: int = 1,
                       context_tokens: int = 8192, completion_margin: int = 300,
                       on_log=None) -> Dict[str, int]:
    """Classify each cue unit's PURER move ONCE (resumable via ws2.db).

    For each not-yet-labeled cue unit, send the canonical PURER cue prompt to
    the LM Studio backend with a COMPACT framework prompt and a BOUNDED
    ``max_tokens`` (so ``prompt + completion <= context_tokens`` -- the loaded
    gemma model exposes only an 8192-token window), parse with the shared
    :func:`_parse_single_run`, and write the result to ``ws2_purer_labels``
    immediately.  Returns counters.

    The loop is GENTLE on a shared backend: a single serial request at a time,
    ``max_retries`` bounded, and any failure recorded as an ERROR ballot (the
    unit is simply re-attempted on the next resumable run) rather than blocking.
    """
    framework = get_purer_framework()
    name_to_id = framework.build_name_to_id_map()

    done = already_labeled(conn)
    pending = [u for u in units if u.cue_id not in done]
    counters = {'total': len(units), 'already': len(done & {u.cue_id for u in units}),
                'classified': 0, 'coded': 0, 'abstain': 0, 'error': 0}

    for i, unit in enumerate(pending):
        prompt = _build_purer_prompt(unit, framework, compact=True)
        # Budget the completion so prompt + max_tokens fits the loaded context.
        prompt_tokens_est = len(prompt) // 4
        max_tokens = max(256, context_tokens - prompt_tokens_est - completion_margin)
        primary = secondary = None
        conf = None
        vote = 'ERROR'
        justification = ''
        raw_text = ''
        attempts = 0
        while attempts < max(1, max_retries):
            attempts += 1
            try:
                raw_text = _lmstudio_chat_bounded(
                    lmstudio_base_url, model, prompt,
                    temperature=temperature, max_tokens=max_tokens,
                    read_timeout=read_timeout) or ''
                break
            except Exception as exc:  # network / backend failure
                justification = f'[exception] {exc}'
                raw_text = ''
        try:
            parsed = _parse_single_run(raw_text, name_to_id) if raw_text else None
            if parsed is None:
                vote = 'ERROR'
                counters['error'] += 1
            else:
                vote = parsed.get('vote', 'ERROR')
                primary = parsed.get('primary_stage')
                secondary = parsed.get('secondary_stage')
                conf = parsed.get('primary_confidence')
                justification = parsed.get('justification', '') or ''
                if vote == 'CODED':
                    counters['coded'] += 1
                elif vote == 'ABSTAIN':
                    counters['abstain'] += 1
                else:
                    counters['error'] += 1
        except Exception as exc:
            vote = 'ERROR'
            justification = f'[parse exception] {exc}'
            counters['error'] += 1

        conn.execute(
            """INSERT OR REPLACE INTO ws2_purer_labels
               (cue_id, primary_move, secondary_move, primary_confidence,
                vote, justification, model, raw_response)
               VALUES (?,?,?,?,?,?,?,?)""",
            (unit.cue_id, primary, secondary, conf, vote,
             justification, model, raw_text[:4000]),
        )
        conn.commit()
        counters['classified'] += 1
        if on_log and (counters['classified'] % 10 == 0 or i == len(pending) - 1):
            on_log(f"  classified {counters['classified']}/{len(pending)} "
                   f"(coded={counters['coded']} abstain={counters['abstain']} "
                   f"error={counters['error']})")

    return counters


# ---------------------------------------------------------------------------
# Mechanism + report
# ---------------------------------------------------------------------------

def compute_mechanism(conn: sqlite3.Connection) -> Dict:
    """Compute the cue->response mechanism over LABELED same-participant cue units.

    Returns a dict with:
      ``rows``         : per labeled+coded unit (from_stage, move, to_stage, delta)
      ``per_move``     : {move: {'n', 'mean_delta', 'deltas'}}
      ``cell_counts``  : {(from_stage, move): n}
      ``n_labeled``    : count of CODED cue units
      ``n_same_participant`` : same as n_labeled (all are within-participant)
    """
    rows = []
    q = """
        SELECT u.from_stage, u.to_stage, l.primary_move
        FROM ws2_cue_units u
        JOIN ws2_purer_labels l ON u.cue_id = l.cue_id
        WHERE l.vote = 'CODED' AND l.primary_move IS NOT NULL
    """
    for from_stage, to_stage, move in conn.execute(q):
        rows.append({
            'from_stage': int(from_stage),
            'to_stage': int(to_stage),
            'move': int(move),
            'delta': int(to_stage) - int(from_stage),
        })

    per_move: Dict[int, Dict] = {}
    for r in rows:
        m = r['move']
        per_move.setdefault(m, {'n': 0, 'deltas': []})
        per_move[m]['n'] += 1
        per_move[m]['deltas'].append(r['delta'])
    for m, d in per_move.items():
        d['mean_delta'] = (sum(d['deltas']) / len(d['deltas'])) if d['deltas'] else 0.0

    cell_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    for r in rows:
        cell_counts[(r['from_stage'], r['move'])] += 1

    return {
        'rows': rows,
        'per_move': per_move,
        'cell_counts': dict(cell_counts),
        'n_labeled': len(rows),
        'n_same_participant': len(rows),
    }


STAGE_NAMES = {0: 'Vigilance', 1: 'Avoidance', 2: 'AttentionReg',
               3: 'Metacognition', 4: 'Reappraisal'}


def render_report(mech: Dict, build_stats: Dict) -> str:
    lines: List[str] = []
    lines.append('WS2 — TURN-ACCURATE PURER CUE->RESPONSE MECHANISM (re-derived from raw VTTs)')
    lines.append('=' * 78)
    lines.append(f'Generated: {date.today().isoformat()}')
    lines.append('')
    lines.append('Unit of analysis: WITHIN-participant cue block — participant P\'s staged')
    lines.append('turn (FROM) -> therapist speech in the gap (the cue) -> P\'s NEXT OWN staged')
    lines.append('turn (TO). VAAMR stages are INHERITED (read-only) from the frozen segments;')
    lines.append('they are never recoded here. PURER moves are classified once per cue block.')
    lines.append('')
    lines.append('BUILD SUMMARY')
    lines.append('-' * 78)
    for k in ('n_sessions', 'raw_cues_total', 'raw_turns_total',
              'therapist_turns_raw', 'frozen_therapist_segments',
              'participant_turns', 'participant_turns_staged',
              'same_participant_specs', 'cue_units_with_cue',
              'cue_units_skipped_lesson', 'cue_units_empty',
              'cue_units_labeled_coded', 'cue_units_abstain',
              'cue_units_error'):
        if k in build_stats:
            lines.append(f'  {k:32}: {build_stats[k]}')
    lines.append('')

    n = mech['n_labeled']
    lines.append('SAME-PARTICIPANT LABELED CUE BLOCKS (use-B unit)')
    lines.append('-' * 78)
    lines.append(f'  CODED same-participant cue blocks: {n}  '
                 f'(threshold MIN_SAME_PARTICIPANT_BLOCKS={MIN_SAME_PARTICIPANT_BLOCKS})')
    lines.append('')

    if n < MIN_SAME_PARTICIPANT_BLOCKS:
        lines.append(insufficient_blocks_banner(
            n, 'within-participant cue->progression'))
        lines.append('')

    # Per-move delta-progression table.
    lines.append('PER-MOVE delta-PROGRESSION  (delta = TO_stage - FROM_stage)')
    lines.append('-' * 78)
    lines.append(f'  {"move":<18}{"n":>6}{"mean delta":>14}')
    for m in sorted(mech['per_move'].keys()):
        d = mech['per_move'][m]
        label = f'{PURER_SHORT.get(m, m)} {PURER_NAME.get(m, "")}'
        lines.append(f'  {label:<18}{d["n"]:>6}{d["mean_delta"]:>14.3f}')
    lines.append('')

    # Per-(from_stage x move) cell table.
    lines.append('CELL TABLE: populated (FROM_stage x move) cells')
    lines.append('-' * 78)
    cells = mech['cell_counts']
    n_cells = len(cells)
    lines.append(f'  populated cells: {n_cells} / 25 possible')
    lines.append('  (production turn-mode PURER on the COLLAPSED frozen segments populated '
                 '~11 cells;')
    lines.append('   this turn-accurate same-participant re-derivation recovers a denser table.)')
    lines.append('')
    header = '  FROM\\move      ' + ''.join(f'{PURER_SHORT[m]:>6}' for m in range(5))
    lines.append(header)
    for fs in range(5):
        row = f'  {fs}:{STAGE_NAMES.get(fs, ""):<11}'
        for m in range(5):
            c = cells.get((fs, m), 0)
            row += f'{(c if c else "."):>6}'
        lines.append(row)
    lines.append('')
    lines.append('Construct key: P=Phenomenological U=Utilization R=Reframing '
                 'E=Educate/Expectancy R2=Reinforcement')
    lines.append('Caveat: stages inherited from merged frozen segments may hide an internal')
    lines.append('stage change (all raw turns overlapping one frozen segment share its stage).')
    lines.append('Therapist speech in a group gap may also address other participants. These are')
    lines.append('weaker confounds than mixing speakers across the FROM->TO transition itself.')
    unk = build_stats.get('unknown_speakers') or {}
    if unk:
        lines.append('')
        lines.append('Unknown VTT speakers (not in speaker key, treated as non-cue "other"): '
                     + ', '.join(f'{k}x{v}' for k, v in sorted(unk.items())))
    return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_ws2(*, processed_dir: str, classify: bool = True,
            skip_lessons: bool = True, max_lesson_words: int = 400,
            model: str = 'google/gemma-4-31b', backend: str = 'lmstudio',
            lmstudio_base_url: str = 'http://10.0.0.58:1234/v1',
            temperature: float = 0.1, read_timeout: int = 1200,
            max_retries: int = 1, context_tokens: int = 8192,
            on_log=None) -> Dict:
    """Full WS2 pipeline. Returns the build_stats dict (also used by the report)."""
    def log(msg):
        if on_log:
            on_log(msg)
        else:
            print(msg, flush=True)

    live_db = os.path.join(processed_dir, 'qra.db')
    vtt_dir = os.path.join(processed_dir, '01_transcripts_inputs')
    speaker_key_path = os.path.join(
        processed_dir, '02_meta', 'speaker_anonymization_key.json')
    ws2_dir = os.path.join(processed_dir, 'ws2_rederivation')
    os.makedirs(os.path.join(ws2_dir, 'reports'), exist_ok=True)
    ws2_db_path = os.path.join(ws2_dir, 'ws2.db')

    speaker_key = load_speaker_key(speaker_key_path)
    frozen_by_session = load_frozen_participant_segments(live_db)
    frozen_therapist = _count_frozen_therapist(live_db)

    vtt_files = sorted(
        f for f in os.listdir(vtt_dir) if f.lower().endswith('.vtt'))
    log(f'Found {len(vtt_files)} VTT files.')

    all_turns: List[Turn] = []
    all_units: List[CueUnit] = []
    unknown_speakers: Dict[str, int] = defaultdict(int)
    raw_cues_total = 0
    therapist_turns_raw = 0
    participant_turns = 0
    participant_turns_staged = 0
    same_participant_specs = 0
    skipped_lesson = 0
    empty_cue = 0

    for vf in vtt_files:
        session_id = os.path.splitext(vf)[0]
        turns, n_cues, unknown = parse_vtt_turns(
            os.path.join(vtt_dir, vf), session_id, speaker_key)
        raw_cues_total += n_cues
        for name, c in unknown.items():
            unknown_speakers[name] += c
        n_part, n_inh = inherit_stages(turns, frozen_by_session)
        participant_turns += n_part
        participant_turns_staged += n_inh
        therapist_turns_raw += sum(1 for t in turns if t.speaker == 'therapist')
        all_turns.extend(turns)

        segs = turns_to_segments(turns)
        units, stats = build_cue_units(
            segs, max_lesson_words=max_lesson_words, skip_lessons=skip_lessons)
        same_participant_specs += stats['n_specs']
        skipped_lesson += stats['n_skipped_lesson']
        empty_cue += stats['n_empty_cue']
        all_units.extend(units)
        log(f'  {session_id}: cues={n_cues} turns={len(turns)} '
            f'part_turns={n_part} staged={n_inh} '
            f'specs={stats["n_specs"]} cue_units={stats["n_with_cue"]}')

    conn = open_ws2_db(ws2_db_path)
    persist_turns(conn, all_turns)
    persist_cue_units(conn, all_units)
    log(f'Persisted {len(all_turns)} turns and {len(all_units)} cue units to ws2.db.')

    counters = {'coded': 0, 'abstain': 0, 'error': 0}
    if classify:
        log(f'Classifying {len(all_units)} cue units with {model} '
            f'@ {lmstudio_base_url} (resumable)...')
        counters = classify_cue_units(
            conn, all_units, model=model, backend=backend,
            lmstudio_base_url=lmstudio_base_url, temperature=temperature,
            read_timeout=read_timeout, max_retries=max_retries,
            context_tokens=context_tokens, on_log=log)
        log(f'Classification done: {counters}')

    mech = compute_mechanism(conn)

    build_stats = {
        'n_sessions': len(vtt_files),
        'raw_cues_total': raw_cues_total,
        'raw_turns_total': len(all_turns),
        'therapist_turns_raw': therapist_turns_raw,
        'frozen_therapist_segments': frozen_therapist,
        'participant_turns': participant_turns,
        'participant_turns_staged': participant_turns_staged,
        'same_participant_specs': same_participant_specs,
        'cue_units_with_cue': len(all_units),
        'cue_units_skipped_lesson': skipped_lesson,
        'cue_units_empty': empty_cue,
        'cue_units_labeled_coded': mech['n_labeled'],
        'cue_units_abstain': _count_vote(conn, 'ABSTAIN'),
        'cue_units_error': _count_vote(conn, 'ERROR'),
        'unknown_speakers': dict(unknown_speakers),
        'n_populated_cells': len(mech['cell_counts']),
    }

    report = render_report(mech, build_stats)
    report_path = os.path.join(ws2_dir, 'reports', 'cue_response_ws2.txt')
    with open(report_path, 'w', encoding='utf-8') as fh:
        fh.write(report)
    log(f'Report written to {report_path}')
    conn.close()
    build_stats['report_path'] = report_path
    build_stats['ws2_db_path'] = ws2_db_path
    build_stats['mechanism'] = mech
    return build_stats


def _count_frozen_therapist(live_db_path: str) -> int:
    uri = f'file:{live_db_path}?mode=ro'
    conn = sqlite3.connect(uri, uri=True)
    try:
        (n,) = conn.execute(
            "SELECT COUNT(*) FROM segments WHERE speaker='therapist'").fetchone()
    finally:
        conn.close()
    return int(n)


def _count_vote(conn: sqlite3.Connection, vote: str) -> int:
    (n,) = conn.execute(
        'SELECT COUNT(*) FROM ws2_purer_labels WHERE vote=?', (vote,)).fetchone()
    return int(n)
