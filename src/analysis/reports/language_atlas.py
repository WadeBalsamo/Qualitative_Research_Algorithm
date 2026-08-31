"""
analysis/reports/language_atlas.py
----------------------------------
Therapeutic language atlas — the readable "key language patterns" deliverable.

Organized around SAME-PARTICIPANT VAAMR transitions: the participant expresses
a stage, hears the therapist, and next expresses a different stage. For each
transition type the atlas gives:

  0. its CUE PROTOTYPE — the centroid proximal cue (the real cue nearest the
     embedding centroid of all proximal cues that preceded that transition),
     distilled to its most characteristic sentences; and
  1. every instance in full: FROM quote → key cue sentences → TO quote, with
     provenance (cohort · session · participant · timecode).

Cues are PROXIMAL (last ≤250 words of therapist-led dialogue immediately
before the TO turn, reconstructed from raw VTT caption timing, de-identified
with the production scrubber) and distilled EXTRACTIVELY — the displayed
sentences are real quotes selected by embedding similarity to the transition's
cue centroid, never paraphrase. Directional/associational, like the mechanism
dossier it draws on (see 03_mechanism/mechanism.txt for CIs).
"""

import os
import re
from typing import Dict, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths
from ._formatting import _wrap_quote

_SPEAKER_LINE_RE = re.compile(r'^\[([^\]]+)\]:\s*(.*)$')

# Tail size of the PROXIMAL cue (therapist words immediately before the TO
# turn, sliced from VTT caption timing, scrubbed).
CUE_PROXIMAL_MAX_WORDS = 250

# Sentences of each proximal cue shown (selected by similarity to the
# transition's cue centroid; real quotes, extractive).
CUE_KEY_SENTENCES = 3


# ---------------------------------------------------------------------------
# Segment lookup + small formatting helpers
# ---------------------------------------------------------------------------

def _text_lookup(df_all: pd.DataFrame) -> Dict[str, dict]:
    out = {}
    for _, r in df_all.iterrows():
        out[str(r.get('segment_id', ''))] = {
            'text': str(r.get('text', '')),
            'mixture': r.get('mixture'),
            'speaker': r.get('speaker'),
            'participant_id': r.get('participant_id'),
            'session_id': r.get('session_id'),
            'session_number': r.get('session_number'),
            'cohort_id': r.get('cohort_id'),
            'start_time_ms': r.get('start_time_ms'),
            'end_time_ms': r.get('end_time_ms'),
        }
    return out


def _fmt_mixture(mix, framework) -> str:
    if mix is None:
        return ''
    vec = np.asarray(mix, dtype=float)
    order = np.argsort(vec)[::-1]
    stage_ids = sorted(framework.keys())
    parts = []
    for k in order[:2]:
        nm = framework.get(stage_ids[k] if k < len(stage_ids) else int(k), {}).get('short_name', str(k))
        parts.append(f"{nm} {vec[k]:.2f}")
    return ' / '.join(parts)


def _stage_name(framework, stage) -> str:
    try:
        return framework.get(int(stage), {}).get('short_name', str(stage))
    except (TypeError, ValueError):
        return str(stage)


def _fmt_timecode(ms) -> str:
    """Session-relative H:MM:SS from milliseconds; '--:--:--' when unavailable."""
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return '--:--:--'
    if not np.isfinite(ms) or ms <= 0:
        return '--:--:--'
    s = int(ms // 1000)
    return f"{s // 3600:d}:{(s // 60) % 60:02d}:{s % 60:02d}"


def _fmt_gap(from_end_ms, to_start_ms) -> str:
    """FROM→TO gap as 'gap Xm YYs'; 'gap n/a' when either bound is missing."""
    try:
        a, b = float(from_end_ms), float(to_start_ms)
    except (TypeError, ValueError):
        return 'gap n/a'
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0 or b <= 0 or b < a:
        return 'gap n/a'
    s = int((b - a) // 1000)
    return f"gap {s // 60}m {s % 60:02d}s"


def _fmt_cohort(cohort_id) -> str:
    try:
        if cohort_id is None or (isinstance(cohort_id, float) and not np.isfinite(cohort_id)):
            return 'Cohort ?'
        return f"Cohort {int(float(cohort_id))}"
    except (TypeError, ValueError):
        return f"Cohort {cohort_id}" if str(cohort_id).strip() else 'Cohort ?'


def _fmt_session(info: dict) -> str:
    sid = str(info.get('session_id') or '?')
    sn = info.get('session_number')
    try:
        if sn is not None and np.isfinite(float(sn)):
            return f"Session {int(float(sn))} ({sid})"
    except (TypeError, ValueError):
        pass
    return f"Session {sid}"


def _speaker_lines(text: str, default_speaker: str):
    """Split segment text into (speaker_tag, utterance) pairs.

    Segment text can contain multiple embedded '[Speaker]: …' lines
    (newline-separated, multi-speaker within one segment is real). Unprefixed
    lines continue the previous speaker; leading unprefixed text belongs to the
    segment's own speaker.
    """
    lines = []
    for raw in str(text).replace('\r\n', '\n').split('\n'):
        raw = raw.strip()
        if not raw:
            continue
        m = _SPEAKER_LINE_RE.match(raw)
        if m:
            lines.append([m.group(1), m.group(2)])
        elif lines:
            lines[-1][1] = (lines[-1][1] + ' ' + raw).strip()
        else:
            lines.append([str(default_speaker or 'unknown'), raw])
    return [(spk, txt) for spk, txt in lines if txt]


def _is_own_speech(speaker_tag: str, participant_id) -> bool:
    pid = str(participant_id or '').strip()
    return bool(pid) and pid in str(speaker_tag)


def _render_attributed_quote(seg_info: dict, participant_id, L, indent='      '):
    """Render one segment's text as attributed direct quotes.

    The block participant's own words are primary; embedded speech by anyone
    else is explicitly marked as context so the direct quote is unambiguous.
    """
    lines = _speaker_lines(seg_info.get('text', ''),
                           seg_info.get('participant_id') or seg_info.get('speaker'))
    if not lines:
        L.append(_wrap_quote('', indent=len(indent)))
        return
    for spk, txt in lines:
        if participant_id is not None and not _is_own_speech(spk, participant_id):
            L.append(f"{indent}(context — {spk}):")
        else:
            L.append(f"{indent}{spk}:")
        L.append(_wrap_quote(txt, indent=len(indent) + 2))


def _same_participant(b) -> bool:
    fp, tp = b.get('from_participant'), b.get('to_participant')
    return bool(fp) and bool(tp) and str(fp) == str(tp)


def _cue_speaker_label(prox) -> str:
    spks = prox.get('speakers') or []
    return ' + '.join(spks) if spks else 'therapist'


def _render_cue_sentences(sents, elided, prox, L, indent='      '):
    """Render extracted cue sentences as one quoted passage."""
    approx = '≈' if prox.get('approx') else ''
    tc = _fmt_timecode(prox.get('start_ms'))
    L.append(f"{indent}{_cue_speaker_label(prox)}  @ {approx}{tc}:")
    joined = ' […] '.join(sents) if elided and len(sents) > 1 else ' '.join(sents)
    if elided and len(sents) == 1:
        joined = '[…] ' + joined
    L.append(_wrap_quote(joined, indent=len(indent) + 2))


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def generate_language_atlas(df, df_all, framework, output_dir, llm_client=None) -> Optional[str]:
    """Write 03_mechanism/language_atlas.txt. Returns path, or None if inputs
    are missing. (`llm_client` is accepted for interface stability but the
    atlas is fully extractive — no LLM prose.)"""
    from gnn_layer.cue_features import build_cue_blocks_with_segments
    from ..mechanism import _seg_lookup, _load_block_motifs, _enrich_blocks
    from ..proximal_cue import ProximalCueExtractor
    from ..cue_distill import CueDistiller

    blocks = build_cue_blocks_with_segments(df_all)
    if not blocks:
        return None
    lookup = _seg_lookup(df_all)
    enriched = _enrich_blocks(blocks, lookup, _load_block_motifs(output_dir))
    if not enriched:
        return None
    tlu = _text_lookup(df_all)
    extractor = ProximalCueExtractor(output_dir, df_all, max_words=CUE_PROXIMAL_MAX_WORDS)
    distiller = CueDistiller()

    def _prox_for(b):
        fr = tlu.get(b['from_seg_id'], {})
        to = tlu.get(b['to_seg_id'], {})
        return extractor.extract(b.get('session_id'),
                                 fr.get('end_time_ms'), to.get('start_time_ms'))

    # ---- same-participant stage-changing blocks, grouped by transition ------
    inv = [b for b in enriched if _same_participant(b)
           and b.get('to_stage') is not None and b.get('from_stage') is not None
           and b['to_stage'] != b['from_stage']]

    def _pair_key(b):
        return (int(b['from_stage']), int(b['to_stage']))

    pair_order = sorted({_pair_key(b) for b in inv},
                        key=lambda p: (-(p[1] - p[0]), p[0])) if inv else []

    groups = {}
    for pair in pair_order:
        group = [b for b in inv if _pair_key(b) == pair]
        group.sort(key=lambda b: -(b.get('delta_prog') if isinstance(b.get('delta_prog'), (int, float))
                                   and np.isfinite(b.get('delta_prog')) else 0.0))
        entries = []
        for b in group:
            p = _prox_for(b)
            entries.append((b, p))
        cue_texts = [p['text'] for _, p in entries
                     if p and not p.get('withheld') and p.get('text')]
        groups[pair] = (entries, cue_texts)

    # Forward/backward cue pools + the effectiveness contrast axis.
    fwd_cues = [t for pair in pair_order if pair[1] > pair[0] for t in groups[pair][1]]
    bwd_cues = [t for pair in pair_order if pair[1] < pair[0] for t in groups[pair][1]]
    axis = distiller.contrast_axis(fwd_cues, bwd_cues)

    def _group_stats(pair):
        entries, _ = groups[pair]
        deltas = [b.get('delta_prog') for b, _ in entries
                  if isinstance(b.get('delta_prog'), (int, float)) and np.isfinite(b.get('delta_prog'))]
        mean_d = f"mean Δ{np.mean(deltas):+.2f}" if deltas else 'Δ n/a'
        fname = _stage_name(framework, pair[0])
        tname = _stage_name(framework, pair[1])
        direction = 'FORWARD' if pair[1] > pair[0] else 'BACKWARD'
        return fname, tname, direction, mean_d, len(entries)

    L = []
    L.append("=" * 78)
    L.append("THERAPEUTIC LANGUAGE ATLAS")
    L.append("=" * 78)
    L.append("")
    L.append("Same-participant VAAMR transitions and the THERAPIST language immediately")
    L.append(f"preceding them. CUES are PROXIMAL — the last ≤{CUE_PROXIMAL_MAX_WORDS} therapist words")
    L.append("before the TO turn, reconstructed from VTT caption timing with caption-label")
    L.append("speaker attribution (therapist-labeled captions only; participant and staff")
    L.append("speech excluded; unmapped labels fall back to a time-span heuristic), and")
    L.append("de-identified with the production scrubber. '≈' = interpolated timecode.")
    L.append("Distillation is EXTRACTIVE along a forward-vs-backward CONTRAST AXIS")
    L.append("(embedding centroid of forward-preceding cues minus backward-preceding):")
    L.append("high-scoring sentences are the therapist language most characteristic of")
    L.append("cues that preceded forward movement. Every displayed sentence is a real")
    L.append("quote ([…] marks elision); nothing is paraphrased.")
    L.append("Every cue is attributed to its anonymized therapist ID from the VTT caption")
    L.append("labels. NOTE: therapist FIRST-PERSON sentences (\"I…\", \"my pain…\") are")
    L.append("typically the therapist voicing a worksheet vignette or modeling a")
    L.append("reappraisal aloud — they are therapist-spoken; verify context in section 1/2.")
    L.append("Directional/associational, n≈20 pilot (see 03_mechanism/mechanism.txt for")
    L.append("CIs). Read as candidate teachable patterns, not proof.")
    L.append("[distillation method: %METHOD%]")
    L.append("")

    def _cite(b, p):
        fr = tlu.get(b['from_seg_id'], {})
        fname = _stage_name(framework, b.get('from_stage'))
        tname = _stage_name(framework, b.get('to_stage'))
        delta = b.get('delta_prog')
        delta_s = (f"Δ{delta:+.2f}" if isinstance(delta, (int, float))
                   and np.isfinite(delta) else 'Δ n/a')
        approx = '≈' if p and p.get('approx') else ''
        tc = _fmt_timecode(p.get('start_ms')) if p else '--:--:--'
        spk = _cue_speaker_label(p) if p else 'therapist'
        return (f"{fname} → {tname} · {_fmt_cohort(fr.get('cohort_id'))} · "
                f"{_fmt_session(fr)} · {b.get('from_participant')} · {delta_s} · "
                f"{spk} @ {approx}{tc}")

    # ---- 0. Most discriminative therapist language --------------------------
    L.append("-" * 78)
    L.append("0. MOST DISCRIMINATIVE THERAPIST LANGUAGE — forward vs backward")
    L.append(f"   (all sentences of all proximal cues ({len(fwd_cues)} forward-preceding /")
    L.append(f"    {len(bwd_cues)} backward-preceding cues), ranked by contrast-axis score;")
    L.append("    the strongest available signal for WHICH therapist language patterns")
    L.append("    accompany forward vs backward VAAMR movement — associational, not causal)")
    L.append("-" * 78)
    if axis is None:
        L.append("  (contrast axis unavailable — needs both forward and backward cues and a")
        L.append("   working sentence embedder; see per-transition sections below)")
    else:
        scored = {1.0: [], -1.0: []}  # sign → [(score, sentence, b, p)]
        seen = {}
        for pair in pair_order:
            sign = 1.0 if pair[1] > pair[0] else -1.0
            for b, p in groups[pair][0]:
                if not (p and not p.get('withheld') and p.get('text')):
                    continue
                pairs_scored = distiller.score_sentences(p['text'], axis)
                if pairs_scored is None:
                    continue
                for sent, sc in pairs_scored:
                    if len(sent.split()) < 5:
                        continue
                    key = sent.strip().lower()
                    if key in seen and seen[key] >= abs(sc):
                        continue
                    seen[key] = abs(sc)
                    scored[sign].append((sc, sent, b, p))
        for sign, title, top_n in ((1.0, 'FORWARD-ASSOCIATED', 12), (-1.0, 'BACKWARD-ASSOCIATED', 8)):
            pool = [x for x in scored[sign] if (x[0] > 0) == (sign > 0)]
            pool.sort(key=lambda x: -abs(x[0]))
            L.append(f"\n  {title} therapist language "
                     f"({'spoken before forward shifts' if sign > 0 else 'spoken before backward shifts'}):")
            if not pool:
                L.append("    (none crossed the axis in this direction)")
            for sc, sent, b, p in pool[:top_n]:
                L.append("")
                L.append(_wrap_quote(sent, indent=4))
                L.append(f"      — {_cite(b, p)}  (score {sc:+.2f})")

    # ---- 1. Cue prototypes per transition type ------------------------------
    L.append("")
    L.append("-" * 78)
    L.append("1. TRANSITION CUE PROTOTYPES — the centroid proximal cue per transition type")
    L.append("   (for each VAAMR transition: the real therapist cue nearest the embedding")
    L.append("    centroid of that transition's proximal cues, shown IN FULL for context)")
    L.append("-" * 78)
    if not inv:
        L.append("  (no same-participant stage-changing blocks found)")
    for pair in pair_order:
        entries, cue_texts = groups[pair]
        fname, tname, direction, mean_d, n = _group_stats(pair)
        L.append(f"\n  {fname} → {tname}  ({direction} · {n} instance(s) · {mean_d})")
        if not cue_texts:
            L.append("    (no usable proximal cues for this transition)")
            continue
        proto_i = distiller.prototype(cue_texts)
        proto_text = cue_texts[proto_i]
        proto_entry = next((bp for bp in entries if bp[1] and bp[1].get('text') == proto_text), None)
        if proto_entry is not None:
            b, p = proto_entry
            fr = tlu.get(b['from_seg_id'], {})
            L.append(f"    prototype: {_fmt_cohort(fr.get('cohort_id'))} · {_fmt_session(fr)} · "
                     f"before {b.get('from_participant')}'s shift")
            approx = '≈' if p.get('approx') else ''
            L.append(f"    {_cue_speaker_label(p)}  @ {approx}{_fmt_timecode(p.get('start_ms'))} "
                     "(full proximal cue):")
            L.append(_wrap_quote(proto_text, indent=6))

    # ---- 2. Transition inventory (every instance, in full) ------------------
    L.append("")
    L.append("-" * 78)
    L.append("2. TRANSITION INVENTORY — every same-participant stage change")
    L.append("   (FROM quote → key cue sentences (contrast-axis ranked) → TO quote;")
    L.append("    forward transitions first, largest Δ first within each type)")
    L.append("-" * 78)
    if not inv:
        L.append("  (no same-participant stage-changing blocks found)")
    else:
        n_fwd = sum(1 for b in inv if b['to_stage'] > b['from_stage'])
        L.append(f"\n  {len(inv)} same-participant stage changes total "
                 f"({n_fwd} forward, {len(inv) - n_fwd} backward) across "
                 f"{len({b.get('from_participant') for b in inv})} participants.")
        for pair in pair_order:
            entries, cue_texts = groups[pair]
            fname, tname, direction, mean_d, n = _group_stats(pair)
            L.append(f"\n  ── {fname} → {tname}  ({direction} · {n} instance(s) · {mean_d}) "
                     + "─" * 10)
            for b, p in entries:
                fr = tlu.get(b['from_seg_id'], {})
                to = tlu.get(b['to_seg_id'], {})
                delta = b.get('delta_prog')
                delta_s = (f"Δprog {delta:+.2f}" if isinstance(delta, (int, float))
                           and np.isfinite(delta) else 'Δprog n/a')
                gap = _fmt_gap(fr.get('end_time_ms'), to.get('start_time_ms'))
                pid = b.get('from_participant') or fr.get('participant_id')
                L.append(f"\n    {fname} → {tname}   ({delta_s} · {gap})")
                L.append(f"    {_fmt_cohort(fr.get('cohort_id'))} · {_fmt_session(fr)} · "
                         f"Participant {pid}")
                L.append(f"      FROM  [{_fmt_mixture(fr.get('mixture'), framework)}]  "
                         f"@ {_fmt_timecode(fr.get('start_time_ms'))}")
                _render_attributed_quote(fr, pid, L, indent='        ')
                L.append("      CUE   (proximal, key sentences)")
                if p is None:
                    L.append("        (no VTT timing — cue unavailable)")
                elif p.get('withheld'):
                    L.append("        (proximal cue withheld — anonymizer offline)")
                elif not p.get('text'):
                    L.append("        (no therapist speech in window)")
                else:
                    sign = 1.0 if pair[1] > pair[0] else -1.0
                    sents, elided = distiller.key_sentences(p['text'], cue_texts,
                                                            k=CUE_KEY_SENTENCES,
                                                            axis=axis, sign=sign)
                    _render_cue_sentences(sents, elided, p, L, indent='        ')
                L.append(f"      TO    [{_fmt_mixture(to.get('mixture'), framework)}]  "
                         f"@ {_fmt_timecode(to.get('start_time_ms'))}")
                _render_attributed_quote(to, b.get('to_participant') or pid, L, indent='        ')

    out = "\n".join(L).replace('%METHOD%', distiller.method)
    rep_dir = _paths.reports_mechanism_dir(output_dir)
    os.makedirs(rep_dir, exist_ok=True)
    path = os.path.join(rep_dir, 'language_atlas.txt')
    with open(path, 'w', encoding='utf-8') as f:
        f.write(out)
    return path
