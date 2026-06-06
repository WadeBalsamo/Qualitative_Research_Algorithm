"""
process/cue_blocks.py
---------------------
Shared cue-block builder that unifies three formerly independent implementations:

  1. process/orchestrator.py   _purer_llm_classify        — Segment dataclass objects
  2. analysis/purer_analysis.py compute_cue_block_purer_profiles — pandas DataFrame rows
  3. gnn_layer/inference.py    build_cue_blocks_with_segments   — pandas DataFrame rows

All three built cue blocks (the run of therapist turns between two consecutive
participant turns) independently and had diverged.  The analysis.py and
gnn_layer versions shared the same bug: when participant end timestamps were 0
or touching (``from_end == to_start``), they emitted empty blocks instead of
falling back to a sorted-index window.  The orchestrator version had the correct
fallback.  This module canonicalises the orchestrator logic and fixes the bug.

Canonical window logic
----------------------
1. Sort ALL items in a session by ``start_time_ms`` (stable sort).  Build a
   global position index keyed by item id.
2. Participants = items with ``speaker == 'participant'`` (and, when
   ``require_stage=True``, a non-null stage).  Therapists = items with
   ``speaker == 'therapist'``.
3. For each consecutive participant pair (from, to):
   - ``fe = end_time_ms(from)``, ``ts = start_time_ms(to)``
   - If ``fe > 0``: timestamp-overlap window →
       therapist items where ``start_time_ms < ts AND end_time_ms > fe``
   - Else (fe == 0, missing / unset): index-fallback →
       therapist items whose global sorted position is strictly between
       from's and to's positions.
4. Preserve therapist order (by sorted position).
5. YIELD ALL blocks including EMPTY ones — callers decide whether to skip.

Public API
----------
@dataclass CueBlockSpec
    session_id, from_item, to_item, from_index, to_index,
    from_stage, to_stage, transition_type, therapist_items

build_cue_blocks(items, *, get_session, get_speaker, get_start, get_end,
                 get_stage, get_id, require_stage=True)
    -> (sorted_items, list[CueBlockSpec])
    Generic core; all accessors are callables so it works on any item type.

cue_blocks_from_segments(segments, *, stage_attr='primary_stage',
                         require_stage=True)
    -> (sorted_items, list[CueBlockSpec])
    Thin wrapper for ``Segment`` dataclass objects.

cue_blocks_from_records(records, *, stage_key='final_label',
                        require_stage=True)
    -> list[CueBlockSpec]
    Thin wrapper for list-of-dicts (e.g. from ``df.to_dict('records')``).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _coerce_int_or_none(value) -> Optional[int]:
    """Convert *value* to int, returning None for None / NaN / non-numeric."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    # duck-typed NaN check (works without importing math at module level)
    if f != f:  # NaN
        return None
    return int(f)


# ---------------------------------------------------------------------------
# CueBlockSpec
# ---------------------------------------------------------------------------

@dataclass
class CueBlockSpec:
    """
    One cue block: the therapist items between two consecutive participant items.

    Attributes
    ----------
    session_id : str
    from_item : Any
        The participant item that opens the window.
    to_item : Any
        The participant item that closes the window.
    from_index : int
        Global sorted position of *from_item* in the full sorted item list
        returned by :func:`build_cue_blocks`.  Use this to index into that list
        for context-window look-ups (e.g. ``_build_context_block_for_purer``).
    to_index : int
        Global sorted position of *to_item*.
    from_stage : int
        Stage label of *from_item* (or 0 when ``require_stage=False`` and the
        stage is null).
    to_stage : int
        Stage label of *to_item* (same note).
    transition_type : str
        ``'forward'`` if from_stage < to_stage, ``'backward'`` if
        from_stage > to_stage, ``'lateral'`` if equal.
    therapist_items : list
        The actual item objects (Segment or dict) in sorted order.
    """
    session_id: str
    from_item: Any
    to_item: Any
    from_index: int
    to_index: int
    from_stage: int
    to_stage: int
    transition_type: str
    therapist_items: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Generic core
# ---------------------------------------------------------------------------

def build_cue_blocks(
    items,
    *,
    get_session: Callable[[Any], str],
    get_speaker: Callable[[Any], str],
    get_start: Callable[[Any], int],
    get_end: Callable[[Any], int],
    get_stage: Callable[[Any], Optional[int]],
    get_id: Callable[[Any], str],
    require_stage: bool = True,
) -> Tuple[List[Any], List[CueBlockSpec]]:
    """
    Build cue-block specs from an arbitrary list of items.

    Parameters
    ----------
    items : iterable
        Any collection of items (Segment objects, dicts, …).
    get_session : callable(item) -> str
        Extract the session identifier.
    get_speaker : callable(item) -> str
        Return ``'participant'`` or ``'therapist'`` (other values are ignored).
    get_start : callable(item) -> int
        Return start_time_ms.
    get_end : callable(item) -> int
        Return end_time_ms (0 means missing/unset → triggers index fallback).
    get_stage : callable(item) -> int or None
        Return the stage label (None means unlabelled).
    get_id : callable(item) -> str
        Return the unique segment identifier.
    require_stage : bool, default True
        When True, participant items with ``get_stage(item) is None`` are
        excluded from the participant list (they cannot anchor a cue window).
        When False, all participants are included; stage 0 is substituted for
        null stages when computing ``from_stage``/``to_stage``.

    Returns
    -------
    sorted_items : list
        All items sorted by ``start_time_ms`` (stable sort — items with equal
        start times preserve original order).
    specs : list[CueBlockSpec]
        One spec per consecutive participant pair, in session-order.  Specs
        for empty blocks (no therapist items between the pair) ARE included.
    """
    # 1. Sort all items globally by start_time_ms
    sorted_items: List[Any] = sorted(items, key=lambda it: get_start(it))

    # 2. Build global position index
    item_id_to_idx: dict = {}
    segs_by_session: dict = defaultdict(list)   # session → participant items
    th_by_session: dict = defaultdict(list)     # session → therapist items

    for i, item in enumerate(sorted_items):
        item_id_to_idx[get_id(item)] = i
        spk = get_speaker(item)
        sid = get_session(item)
        if spk == 'participant':
            stage = get_stage(item)
            if require_stage and stage is None:
                continue
            segs_by_session[sid].append(item)
        elif spk == 'therapist':
            th_by_session[sid].append(item)

    # 3. Build specs per session
    specs: List[CueBlockSpec] = []

    for session_id, p_items in segs_by_session.items():
        th_items = th_by_session.get(session_id, [])

        for i in range(len(p_items) - 1):
            from_item = p_items[i]
            to_item = p_items[i + 1]

            from_idx = item_id_to_idx.get(get_id(from_item), -1)
            to_idx = item_id_to_idx.get(get_id(to_item), -1)

            fe = get_end(from_item)   # from_end_ms
            ts = get_start(to_item)   # to_start_ms

            if fe > 0:
                # Canonical timestamp-overlap window.
                # Matches orchestrator logic; fixes the purer_analysis bug where
                # touching timestamps (fe == ts) produced an empty slice.
                between = [
                    t for t in th_items
                    if get_start(t) < ts and get_end(t) > fe
                ]
            else:
                # Index fallback: from_seg.end_time_ms == 0 means the field is
                # unset / missing — use sorted-position window instead.
                between = [
                    t for t in th_items
                    if from_idx < item_id_to_idx.get(get_id(t), -1) < to_idx
                ]

            # Derive stage values
            from_stage_val = get_stage(from_item)
            to_stage_val = get_stage(to_item)
            from_stage = int(from_stage_val) if from_stage_val is not None else 0
            to_stage = int(to_stage_val) if to_stage_val is not None else 0

            if from_stage < to_stage:
                transition_type = 'forward'
            elif from_stage > to_stage:
                transition_type = 'backward'
            else:
                transition_type = 'lateral'

            spec = CueBlockSpec(
                session_id=str(session_id),
                from_item=from_item,
                to_item=to_item,
                from_index=from_idx,
                to_index=to_idx,
                from_stage=from_stage,
                to_stage=to_stage,
                transition_type=transition_type,
                therapist_items=between,
            )
            specs.append(spec)

    return sorted_items, specs


# ---------------------------------------------------------------------------
# Wrapper: Segment dataclass objects
# ---------------------------------------------------------------------------

def cue_blocks_from_segments(
    segments,
    *,
    stage_attr: str = 'primary_stage',
    require_stage: bool = True,
) -> Tuple[List[Any], List[CueBlockSpec]]:
    """
    Build cue-block specs from a list of ``Segment`` dataclass objects.

    Parameters
    ----------
    segments : list[Segment]
    stage_attr : str, default ``'primary_stage'``
        Attribute name to read as the stage label.  Use ``'primary_stage'``
        during PURER classification (Stage 3c) and ``'final_label'`` for
        post-hoc analysis when that overlay is available.
    require_stage : bool, default True

    Returns
    -------
    (sorted_segments, specs)
        ``sorted_segments`` is the globally sorted list (same objects, new
        ordering) — pass it directly to ``_build_context_block_for_purer``.
    """
    return build_cue_blocks(
        segments,
        get_session=lambda s: s.session_id,
        get_speaker=lambda s: s.speaker,
        get_start=lambda s: s.start_time_ms,
        get_end=lambda s: s.end_time_ms,
        get_stage=lambda s: getattr(s, stage_attr, None),
        get_id=lambda s: s.segment_id,
        require_stage=require_stage,
    )


# ---------------------------------------------------------------------------
# Wrapper: list-of-dicts (DataFrame rows)
# ---------------------------------------------------------------------------

def cue_blocks_from_records(
    records,
    *,
    stage_key: str = 'final_label',
    require_stage: bool = True,
) -> List[CueBlockSpec]:
    """
    Build cue-block specs from a list of dicts (e.g. ``df.to_dict('records')``).

    Parameters
    ----------
    records : list[dict]
    stage_key : str, default ``'final_label'``
        Dict key for the stage label.  Values are coerced to int via
        :func:`_coerce_int_or_none` (NaN / None → None).
    require_stage : bool, default True

    Returns
    -------
    list[CueBlockSpec]
        The ``therapist_items`` in each spec are the raw dict objects.
    """
    _, specs = build_cue_blocks(
        records,
        get_session=lambda r: str(r.get('session_id', '')),
        get_speaker=lambda r: str(r.get('speaker', '')),
        get_start=lambda r: int(r.get('start_time_ms', 0) or 0),
        get_end=lambda r: int(r.get('end_time_ms', 0) or 0),
        get_stage=lambda r: _coerce_int_or_none(r.get(stage_key)),
        get_id=lambda r: str(r.get('segment_id', id(r))),
        require_stage=require_stage,
    )
    return specs


# ---------------------------------------------------------------------------
# Sub-cue splitting helper
# ---------------------------------------------------------------------------

def split_by_word_budget(items, max_words: int, text_of) -> list:
    """
    Greedily pack *items* into contiguous sub-lists whose combined word count
    stays at or below *max_words*.

    Parameters
    ----------
    items : list
        Ordered sequence of items to pack (e.g. therapist Segment objects).
    max_words : int
        Maximum combined word count allowed in a single sub-list.
    text_of : callable(item) -> str
        Returns the text string for an item.  Word count is
        ``len(text_of(item).split())``.

    Rules
    -----
    - Items are NEVER split across sub-lists.
    - If a single item alone exceeds *max_words* it becomes its own singleton
      sub-list (the budget is violated for that sub-list only — unavoidable).
    - Empty / whitespace items contribute 0 words.
    - The flattened output list is identical in order and membership to *items*.
    - Returns at least one sub-list when *items* is non-empty.
    - Returns an empty list when *items* is empty.

    Returns
    -------
    list[list]
        Sub-lists of items.
    """
    if not items:
        return []

    groups: list = []
    current_group: list = []
    current_words: int = 0

    for item in items:
        w = len(text_of(item).split())
        if not current_group:
            # Always start a new group with the first item, even if it alone
            # exceeds the budget (singleton over-budget sub-list).
            current_group.append(item)
            current_words = w
        elif current_words + w <= max_words:
            current_group.append(item)
            current_words += w
        else:
            # Flush current group and start a new one.
            groups.append(current_group)
            current_group = [item]
            current_words = w

    if current_group:
        groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# PURER coverage report formatter
# ---------------------------------------------------------------------------

def format_purer_coverage(stats: dict) -> str:
    """
    Render a human-readable PURER coverage report from a *stats* dict.

    Expected keys (all optional; missing values treated as 0):
      per_session : dict[str, dict]
          Mapping session_id → per-session counts (same keys as below).
      n_blocks : int
          Total non-empty cue blocks encountered.
      n_skipped_lesson : int
          Blocks skipped because skip_lesson_content was True and the block
          exceeded max_lesson_words.
      skipped_lesson_words : int
          Therapist word total across skipped-as-lesson blocks.
      n_labeled_segments : int
          Therapist segment objects that received a PURER label.
      labeled_words : int
          Word total across labeled therapist segments.
      n_unparseable : int
          Single-turn sub-cues that could not be parsed even after bisect.
      unparseable_words : int
          Word total across unparseable sub-cues.
      total_therapist_words : int
          Word total across ALL therapist turns in non-empty cue blocks.

    Returns
    -------
    str
        Multi-line human-readable report string.
    """
    def _pct(numerator: int, denominator: int) -> str:
        if denominator == 0:
            return 'n/a'
        return f'{100.0 * numerator / denominator:.1f}%'

    def _session_block(sid: str, s: dict) -> str:
        n_blk = s.get('n_blocks', 0)
        n_skip = s.get('n_skipped_lesson', 0)
        skip_w = s.get('skipped_lesson_words', 0)
        n_lbl = s.get('n_labeled_segments', 0)
        lbl_w = s.get('labeled_words', 0)
        n_unp = s.get('n_unparseable', 0)
        unp_w = s.get('unparseable_words', 0)
        tot_w = s.get('total_therapist_words', 0)
        lines = [
            f'  Session: {sid}',
            f'    Cue blocks (non-empty):  {n_blk}',
            f'    Skipped as lesson:       {n_skip}  ({skip_w} therapist words)',
            f'    Labeled segments:        {n_lbl}  ({lbl_w} words)',
            f'    Unparseable (bisect):    {n_unp}  ({unp_w} words)',
            f'    Total therapist words:   {tot_w}',
            f'    Coverage (labeled/total):{_pct(lbl_w, tot_w)}',
        ]
        return '\n'.join(lines)

    lines = ['=' * 62, 'PURER CLASSIFICATION COVERAGE REPORT', '=' * 62, '']

    per_session = stats.get('per_session', {})
    if per_session:
        lines.append('Per-session breakdown:')
        lines.append('')
        for sid in sorted(per_session.keys()):
            lines.append(_session_block(sid, per_session[sid]))
            lines.append('')

    # Overall totals
    n_blk   = stats.get('n_blocks', 0)
    n_skip  = stats.get('n_skipped_lesson', 0)
    skip_w  = stats.get('skipped_lesson_words', 0)
    n_lbl   = stats.get('n_labeled_segments', 0)
    lbl_w   = stats.get('labeled_words', 0)
    n_unp   = stats.get('n_unparseable', 0)
    unp_w   = stats.get('unparseable_words', 0)
    tot_w   = stats.get('total_therapist_words', 0)

    lines += [
        'Overall totals:',
        f'  Cue blocks (non-empty):  {n_blk}',
        f'  Skipped as lesson:       {n_skip}  ({skip_w} therapist words)',
        f'  Labeled segments:        {n_lbl}  ({lbl_w} words)',
        f'  Unparseable (bisect):    {n_unp}  ({unp_w} words)',
        f'  Total therapist words:   {tot_w}',
        f'  Coverage (labeled/total):{_pct(lbl_w, tot_w)}',
        '',
    ]

    return '\n'.join(lines)
