"""
analysis/reports/justification_grounding.py
--------------------------------------------
LLM Justification-Grounding Audit — a trust instrument.

For every labeled segment, the classifier returns a free-text justification that
is *supposed* to cite the participant's (or therapist's) own words. This audit
measures, per segment, whether the quoted spans in that justification actually
appear in the segment text — turning "the model cites the text" into a reported
rate and flagging the ungrounded minority.

What it does NOT do: it does not check whether the LLM's *label* is correct.
A faithfully-quoted segment can still be mis-staged. Grounding bounds
CONFABULATION, not correctness; it complements (does not replace) the human↔LLM
inter-rater reliability of §5.3–5.4.

Outputs
-------
  04_validation/justification_grounding.csv   — per-segment audit rows
  04_validation/justification_grounding.json  — aggregate rates
  06_reports/06_classifier/justification_grounding.txt — human-readable report

Stdlib-only (re, difflib, json) — no network, no model downloads, no new deps.
Default-ON: wired into analysis/runner.py right after segments are loaded.
"""

import json
import os
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from process import output_paths as _paths


# ── Quote extraction + normalization ──────────────────────────────────────

# Straight and curly quote pairs.  Each group captures the inner span.
# The straight-single pattern is BOUNDARY-AWARE: the opening ' must follow a
# string-edge / whitespace / opening-punctuation char and the closing ' must
# precede a string-edge / whitespace / punctuation char, so a bare apostrophe
# contraction ("don't ... can't") is NOT mis-read as a quoted span. Crucially the
# inner span MAY itself contain mid-word contraction apostrophes (a ' with a word
# char on BOTH sides), so a genuine single-quoted phrase that happens to contain a
# contraction — "'I'm a walking miracle'", "'no one's ever taught me that'" — is
# still captured (the earlier `[^']{2,}` form silently dropped these). Double +
# curly quotes are unambiguous and matched plainly.
_QUOTE_PATTERNS = [
    re.compile(r'"([^"]{2,})"'),                       # straight double
    # straight single (boundary-aware; inner allows mid-word apostrophes):
    #   open '  ← string-edge / ws / ( [ — – -
    #   inner   ← >=2 chars, no edge whitespace; a ' counts only when flanked by \w
    #   close ' → string-edge / ws / ) . , ; : ! ? ] — – -
    re.compile(r"(?:^|(?<=[\s(\[—–-]))'([^'\s](?:[^']|(?<=\w)'(?=\w))*[^'\s])'(?=$|[\s).,;:!?\]—–-])"),
    re.compile(r'“([^”]{2,})”'),                  # curly double  “ … ”
    re.compile(r'‘([^’]{2,})’'),                  # curly single  ‘ … ’
]

# Content-token splitter: words of length >= 3 (drops most stopword noise for
# the no-quote lexical-overlap fallback).  Casefolded.
_WORD_RE = re.compile(r"[a-z0-9']+")

_STOPWORDS = {
    'the', 'and', 'that', 'this', 'with', 'for', 'are', 'was', 'were', 'has',
    'have', 'had', 'not', 'but', 'she', 'her', 'his', 'him', 'they', 'them',
    'their', 'you', 'your', 'its', 'our', 'who', 'which', 'what', 'when',
    'where', 'how', 'why', 'from', 'into', 'about', 'than', 'then', 'there',
    'here', 'been', 'being', 'because', 'while', 'would', 'could', 'should',
    'will', 'can', 'may', 'might', 'does', 'did', 'doing', 'done',
}


def _normalize(s: str) -> str:
    """Casefold, collapse whitespace, strip edge punctuation/quotes."""
    if not s:
        return ''
    s = str(s).casefold()
    s = re.sub(r'\s+', ' ', s).strip()
    # Strip leading/trailing punctuation and stray quote glyphs.
    s = s.strip(' \t\n\r"\'“”‘’.,;:!?-—–()[]')
    return s


def _extract_quotes(justification: str) -> List[str]:
    """Return the list of quoted spans found in a justification (straight+curly)."""
    if not justification:
        return []
    spans: List[str] = []
    for pat in _QUOTE_PATTERNS:
        for m in pat.finditer(justification):
            inner = m.group(1).strip()
            if inner:
                spans.append(inner)
    return spans


def _content_tokens(s: str) -> set:
    toks = _WORD_RE.findall(str(s).casefold())
    return {t for t in toks if len(t) >= 3 and t not in _STOPWORDS}


def _lexical_overlap(justification: str, text: str) -> float:
    """Content-token Jaccard(justification, text) — the no-quote fallback signal."""
    a = _content_tokens(justification)
    b = _content_tokens(text)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _span_grounded(span: str, norm_text: str, fuzzy_threshold: float = 0.90,
                   min_fuzzy_chars: int = 8) -> bool:
    """True if `span` is grounded in `norm_text` (exact substring, else fuzzy window).

    Exact path: normalized span is a substring of normalized text (any length).
    Fuzzy fallback: slide a window of len(span) over norm_text and accept when any
    window's SequenceMatcher ratio >= fuzzy_threshold (handles paraphrase / minor
    transcription drift). Stdlib `difflib` only. The fuzzy path is GATED to spans of
    >= `min_fuzzy_chars` normalized characters: for very short spans the ratio>=0.90
    window saturates (a handful of shared letters clears the bar), so a tiny span
    must match EXACTLY to count as grounded — otherwise short fragments get
    spuriously grounded against unrelated text.
    """
    nspan = _normalize(span)
    if not nspan:
        return False
    if nspan in norm_text:
        return True
    if not norm_text:
        return False
    if len(nspan) < min_fuzzy_chars:
        # Too short to fuzzy-match safely; exact-substring already failed above.
        return False
    # Fuzzy sliding window. Step by a fraction of the span so this stays cheap on
    # long segments; the matcher tolerates small offsets at each anchor.
    w = len(nspan)
    n = len(norm_text)
    if w > n:
        # Span longer than the whole text — compare against the whole text once.
        return SequenceMatcher(None, nspan, norm_text).ratio() >= fuzzy_threshold
    step = max(1, w // 4)
    sm = SequenceMatcher(None)
    sm.set_seq1(nspan)
    for start in range(0, n - w + 1, step):
        sm.set_seq2(norm_text[start:start + w])
        if sm.ratio() >= fuzzy_threshold:
            return True
    return False


# ── Per-segment audit ──────────────────────────────────────────────────────

def _audit_one(justification: str, text: str,
               quote_grounded_threshold: float = 0.5,
               overlap_threshold: float = 0.15) -> Optional[dict]:
    """Audit a single (justification, text) pair. Returns a record or None if no justification."""
    if justification is None or (isinstance(justification, float) and pd.isna(justification)):
        return None
    justification = str(justification).strip()
    if not justification:
        return None

    text = '' if (text is None or (isinstance(text, float) and pd.isna(text))) else str(text)
    norm_text = _normalize(text)

    spans = _extract_quotes(justification)
    n_quotes = len(spans)
    if n_quotes > 0:
        n_grounded = sum(1 for s in spans if _span_grounded(s, norm_text))
        grounded_frac = n_grounded / n_quotes
        overlap = _lexical_overlap(justification, text)
        ungrounded = grounded_frac < quote_grounded_threshold
    else:
        n_grounded = 0
        grounded_frac = float('nan')   # undefined when nothing is quoted
        overlap = _lexical_overlap(justification, text)
        ungrounded = overlap < overlap_threshold

    return {
        'has_quotes': n_quotes > 0,
        'n_quotes': n_quotes,
        'n_grounded': n_grounded,
        'grounded_frac': grounded_frac,
        'overlap': round(float(overlap), 4),
        'ungrounded': bool(ungrounded),
    }


def _parse_votes(raw) -> List[dict]:
    """Parse a rater_votes cell (JSON string, list, or NaN) into a list of dicts."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return []
    if isinstance(raw, list):
        return [v for v in raw if isinstance(v, dict)]
    if isinstance(raw, str):
        s = raw.strip()
        if not s:
            return []
        try:
            parsed = json.loads(s)
        except (ValueError, TypeError):
            try:
                import ast
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                return []
        return [v for v in parsed if isinstance(v, dict)] if isinstance(parsed, list) else []
    return []


# ── Aggregation over a frame ────────────────────────────────────────────────

def _audit_frame(df: pd.DataFrame, framework: dict,
                 justification_col: str, votes_col: str,
                 stage_col: str, label_kind: str) -> dict:
    """Run the audit over one frame (VAAMR participant or PURER therapist).

    Returns {'rows': [...], 'aggregate': {...}, 'per_segment_records': [...]} or
    {'rows': [], ...} when the justification column is absent/empty.
    """
    empty = {'rows': [], 'aggregate': None, 'records': []}
    if df is None or len(df) == 0 or justification_col not in df.columns:
        return empty
    if not df[justification_col].notna().any():
        return empty

    stage_names = {}
    if framework:
        for sid, meta in framework.items():
            try:
                stage_names[int(sid)] = meta.get('short_name') or meta.get('name') or str(sid)
            except (ValueError, TypeError):
                pass

    rows: List[dict] = []
    # Per-rater accumulators: rater -> [n_quotes_total, n_grounded_total, n_just_with_quotes, n_just]
    per_rater: Dict[str, List[float]] = {}
    # Per-stage accumulators: stage -> [grounded_span_sum, span_total, n_seg_with_quote, n_seg_grounded_ge1, n_seg]
    per_stage: Dict[int, List[float]] = {}

    for _, r in df.iterrows():
        rec = _audit_one(r.get(justification_col), r.get('text'))
        if rec is None:
            continue
        seg_id = str(r.get('segment_id', ''))

        stage_val = r.get(stage_col)
        try:
            stage_int = int(stage_val) if pd.notna(stage_val) else None
        except (ValueError, TypeError):
            stage_int = None

        row = {
            'segment_id': seg_id,
            'kind': label_kind,
            'stage': stage_int if stage_int is not None else '',
            'stage_name': stage_names.get(stage_int, '') if stage_int is not None else '',
            'has_quotes': rec['has_quotes'],
            'n_quotes': rec['n_quotes'],
            'n_grounded': rec['n_grounded'],
            'grounded_frac': ('' if rec['grounded_frac'] != rec['grounded_frac']
                              else round(rec['grounded_frac'], 4)),
            'overlap': rec['overlap'],
            'ungrounded': rec['ungrounded'],
        }
        rows.append(row)

        # Per-stage roll-up.
        if stage_int is not None:
            acc = per_stage.setdefault(stage_int, [0.0, 0.0, 0.0, 0.0, 0.0])
            acc[4] += 1
            if rec['has_quotes']:
                acc[1] += rec['n_quotes']
                acc[0] += rec['n_grounded']
                acc[2] += 1
                if rec['n_grounded'] >= 1:
                    acc[3] += 1

        # Per-rater roll-up: each rater's own ballot justification audited vs the
        # same segment text (which checker model confabulates most).
        for v in _parse_votes(r.get(votes_col)):
            rater = str(v.get('rater', '') or 'unknown')
            vrec = _audit_one(v.get('justification'), r.get('text'))
            if vrec is None:
                continue
            racc = per_rater.setdefault(rater, [0.0, 0.0, 0.0, 0.0])
            racc[3] += 1
            if vrec['has_quotes']:
                racc[0] += vrec['n_quotes']
                racc[1] += vrec['n_grounded']
                racc[2] += 1

    if not rows:
        return empty

    rdf = pd.DataFrame(rows)
    n_seg = len(rdf)
    quoting = rdf[rdf['has_quotes']]
    n_quoting = len(quoting)
    total_spans = int(rdf['n_quotes'].sum())
    total_grounded = int(rdf['n_grounded'].sum())
    n_seg_ge1_grounded = int((rdf['n_grounded'] >= 1).sum())

    per_stage_out = {}
    for sid, acc in sorted(per_stage.items()):
        gsum, span_total, n_q, n_g1, n_s = acc
        per_stage_out[sid] = {
            'stage': sid,
            'stage_name': stage_names.get(sid, str(sid)),
            'n_segments': int(n_s),
            'n_quoting': int(n_q),
            'spans_total': int(span_total),
            'spans_grounded': int(gsum),
            'pct_spans_grounded': (round(100.0 * gsum / span_total, 1) if span_total else None),
            'pct_seg_grounded_quote': (round(100.0 * n_g1 / n_s, 1) if n_s else None),
        }

    per_rater_out = {}
    for rater, racc in sorted(per_rater.items()):
        spans_total, spans_grounded, n_q, n_j = racc
        per_rater_out[rater] = {
            'rater': rater,
            'n_justifications': int(n_j),
            'n_quoting': int(n_q),
            'pct_quoting': (round(100.0 * n_q / n_j, 1) if n_j else None),
            'spans_total': int(spans_total),
            'spans_grounded': int(spans_grounded),
            'pct_spans_grounded': (round(100.0 * spans_grounded / spans_total, 1) if spans_total else None),
        }

    aggregate = {
        'label_kind': label_kind,
        'n_segments_with_justification': n_seg,
        'n_segments_quoting': n_quoting,
        'pct_segments_quoting': round(100.0 * n_quoting / n_seg, 1) if n_seg else None,
        'spans_total': total_spans,
        'spans_grounded': total_grounded,
        'pct_spans_grounded': round(100.0 * total_grounded / total_spans, 1) if total_spans else None,
        'pct_segments_with_grounded_quote': round(100.0 * n_seg_ge1_grounded / n_seg, 1) if n_seg else None,
        'n_ungrounded_flagged': int(rdf['ungrounded'].sum()),
        'pct_ungrounded_flagged': round(100.0 * int(rdf['ungrounded'].sum()) / n_seg, 1) if n_seg else None,
        'per_stage': per_stage_out,
        'per_rater': per_rater_out,
    }
    return {'rows': rows, 'aggregate': aggregate, 'records': rows}


# ── Report writer ───────────────────────────────────────────────────────────

def _fmt_pct(v) -> str:
    return '   n/a' if v is None else f'{v:5.1f}%'


def _write_report(vaamr: dict, purer: dict, df_lookup: Dict[str, str],
                  output_dir: str, top_n: int = 15) -> str:
    L: List[str] = []
    L.append("=" * 78)
    L.append("LLM JUSTIFICATION-GROUNDING AUDIT")
    L.append("=" * 78)
    L.append("")
    L.append("Does each classifier justification demonstrably CITE the segment it labels?")
    L.append("This bounds CONFABULATION (citing words that aren't there), NOT correctness —")
    L.append("a faithfully-quoted segment can still be mis-staged. A flagged item means the")
    L.append("justification did not DEMONSTRABLY cite the segment text (a review queue), not")
    L.append("that the model lied: an honest abstractive/paraphrased rationale that shares")
    L.append("no surface tokens lands here too. Read alongside the human↔LLM IRR")
    L.append("(06b_irr_report.txt).")
    L.append("")

    for title, audit in (("VAAMR (participant segments)", vaamr),
                         ("PURER (therapist cue blocks)", purer)):
        agg = audit.get('aggregate')
        L.append("-" * 78)
        L.append(title)
        L.append("-" * 78)
        if agg is None:
            L.append("  (no justifications present — skipped)")
            L.append("")
            continue
        L.append(f"  Segments with a justification ............ {agg['n_segments_with_justification']}")
        L.append(f"  Justifications that quote at all ......... {_fmt_pct(agg['pct_segments_quoting'])} "
                 f"({agg['n_segments_quoting']}/{agg['n_segments_with_justification']})")
        L.append(f"  Quoted spans GROUNDED in the text ........ {_fmt_pct(agg['pct_spans_grounded'])} "
                 f"({agg['spans_grounded']}/{agg['spans_total']} spans)")
        L.append(f"  Segments with >=1 grounded quote ......... {_fmt_pct(agg['pct_segments_with_grounded_quote'])}")
        L.append(f"  Did-not-demonstrably-cite (review queue) . {_fmt_pct(agg['pct_ungrounded_flagged'])} "
                 f"({agg['n_ungrounded_flagged']} segments; rule below)")
        L.append("")

        # Per-stage / per-move table.
        ps = agg.get('per_stage') or {}
        if ps:
            L.append("  Grounding by stage/move:")
            L.append(f"    {'stage':<22}{'n':>5}{'quoting':>9}{'spans_grnd':>12}{'seg_grnd':>10}")
            for sid in sorted(ps.keys()):
                row = ps[sid]
                nm = f"{sid} {row['stage_name']}"[:21]
                L.append(f"    {nm:<22}{row['n_segments']:>5}{row['n_quoting']:>9}"
                         f"{_fmt_pct(row['pct_spans_grounded']):>12}{_fmt_pct(row['pct_seg_grounded_quote']):>10}")
            L.append("")

        # Per-rater (model) table — which checker model confabulates most.
        pr = agg.get('per_rater') or {}
        if pr:
            L.append("  Grounding by rater/model (per-ballot justifications):")
            L.append(f"    {'model':<34}{'n_just':>8}{'quoting':>9}{'spans_grnd':>12}")
            # Sort worst grounding first (the confabulators), then by volume.
            def _rank(item):
                d = item[1]
                g = d['pct_spans_grounded']
                return (g if g is not None else 999, -d['n_justifications'])
            for rater, d in sorted(pr.items(), key=_rank):
                nm = rater[:33]
                L.append(f"    {nm:<34}{d['n_justifications']:>8}{_fmt_pct(d['pct_quoting']):>9}"
                         f"{_fmt_pct(d['pct_spans_grounded']):>12}")
            L.append("")

    # Flag rule + caveat.
    L.append("-" * 78)
    L.append("FLAG RULE  (did-not-demonstrably-cite → review queue, NOT a lie detector)")
    L.append("-" * 78)
    L.append("  A justification lands in the review queue when:")
    L.append("    - it quotes the text but < 50% of its quoted spans appear in the segment, OR")
    L.append("    - it quotes nothing AND content-token Jaccard(justification, text) < 0.15.")
    L.append("  Grounding = exact normalized-substring match, with a difflib fuzzy fallback")
    L.append("  (SequenceMatcher ratio >= 0.90 over a sliding window, spans >= 8 chars) for")
    L.append("  paraphrase/drift. The no-quote branch also catches honest ABSTRACTIVE /")
    L.append("  paraphrased rationales (no shared surface tokens), so a flag is a 'verify")
    L.append("  this cite' prompt, not evidence of confabulation. Grounding bounds")
    L.append("  CONFABULATION, not correctness.")
    L.append("")

    # Ranked dossier of the least-grounded items (worst first), text vs justification.
    L.append("-" * 78)
    L.append(f"LEAST-GROUNDED ITEMS (worst {top_n}; review candidates — verify the cite)")
    L.append("-" * 78)
    L.append("")

    all_rows = list(vaamr.get('rows', [])) + list(purer.get('rows', []))
    # Worst = ungrounded first; among those, lowest grounded_frac (quoting) then
    # lowest overlap (non-quoting).
    def _badness(row):
        gf = row.get('grounded_frac')
        gf = 1.0 if gf == '' else float(gf)
        ov = float(row.get('overlap', 0.0) or 0.0)
        # Primary key: not-flagged sinks to the bottom.
        return (0 if row.get('ungrounded') else 1, gf, ov)

    ranked = sorted(all_rows, key=_badness)
    shown = 0
    for row in ranked:
        if shown >= top_n:
            break
        seg_id = row['segment_id']
        text = df_lookup.get(seg_id, {}).get('text', '')
        just = df_lookup.get(seg_id, {}).get(
            'purer_justification' if row['kind'] == 'PURER' else 'llm_justification', '')
        gf = row.get('grounded_frac')
        gf_str = 'no-quote' if gf == '' else f"grounded {float(gf)*100:.0f}%"
        L.append(f"  [{row['kind']}] {seg_id}  stage={row.get('stage')}  "
                 f"({gf_str}, overlap={row.get('overlap')})")
        L.append("    SEGMENT TEXT:")
        L.append(_wrap(text, indent=6))
        L.append("    JUSTIFICATION:")
        L.append(_wrap(just, indent=6))
        L.append("")
        shown += 1
    if shown == 0:
        L.append("  (none flagged — all justifications are grounded)")
        L.append("")

    path = os.path.join(_paths.reports_classifier_dir(output_dir), 'justification_grounding.txt')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(L))
    return path


def _wrap(text: str, indent: int = 6, max_width: int = 78) -> str:
    """Lightweight word-wrap with indent (local copy to avoid cross-module coupling)."""
    if not text:
        return ' ' * indent + '(empty)'
    prefix = ' ' * indent
    out, current = [], prefix
    for word in str(text).split():
        if len(current) + len(word) + 1 > max_width and current != prefix:
            out.append(current)
            current = prefix + word
        else:
            current = current + word if current == prefix else current + ' ' + word
    out.append(current)
    return '\n'.join(out)


# ── Public entry point ──────────────────────────────────────────────────────

def generate_justification_grounding_report(df: pd.DataFrame, framework: dict,
                                            output_dir: str,
                                            df_all: Optional[pd.DataFrame] = None) -> Optional[str]:
    """Audit LLM/PURER justification grounding; write CSV + JSON + text report.

    Parameters
    ----------
    df : pd.DataFrame
        Participant (VAAMR) segments — carries `llm_justification`, `rater_votes`,
        `primary_stage`, `text`, `segment_id`.
    framework : dict
        VAAMR framework dict (int stage_id → {short_name, ...}) for stage labels.
    output_dir : str
        Pipeline output directory.
    df_all : pd.DataFrame, optional
        Full frame (all speakers). When it carries therapist `purer_justification`,
        the PURER audit is produced too; skipped gracefully when absent.

    Returns the report path, or None when there is nothing to audit.
    """
    if df is None or len(df) == 0:
        return None

    # VAAMR audit on participant segments.
    vaamr = _audit_frame(df, framework, 'llm_justification', 'rater_votes',
                         'primary_stage', 'VAAMR')

    # PURER audit on therapist segments (only if present in df_all).
    purer = {'rows': [], 'aggregate': None, 'records': []}
    purer_framework = {}
    if df_all is not None and 'purer_justification' in df_all.columns:
        ther = df_all[df_all.get('speaker') == 'therapist'].copy() if 'speaker' in df_all.columns else df_all
        try:
            from constructs.registry import load as _load_fw
            pf = _load_fw('purer')
            for d in (pf.themes if pf is not None else []):
                purer_framework[int(d.theme_id)] = {'short_name': d.short_name, 'name': d.name}
        except Exception:
            purer_framework = {0: {'short_name': 'P'}, 1: {'short_name': 'U'},
                               2: {'short_name': 'R'}, 3: {'short_name': 'E'},
                               4: {'short_name': 'R2'}}
        purer = _audit_frame(ther, purer_framework, 'purer_justification',
                             'purer_rater_votes', 'purer_primary', 'PURER')

    if vaamr.get('aggregate') is None and purer.get('aggregate') is None:
        return None

    # Build a segment_id → {text, justifications} lookup for the dossier (from the
    # widest frame available so therapist text is reachable).
    src = df_all if df_all is not None else df
    df_lookup: Dict[str, dict] = {}
    for _, r in src.iterrows():
        df_lookup[str(r.get('segment_id', ''))] = {
            'text': str(r.get('text', '') or ''),
            'llm_justification': str(r.get('llm_justification', '') or ''),
            'purer_justification': str(r.get('purer_justification', '') or ''),
        }

    # ── Write per-segment CSV ──
    val_dir = _paths.validation_dir(output_dir)
    os.makedirs(val_dir, exist_ok=True)
    all_rows = list(vaamr.get('rows', [])) + list(purer.get('rows', []))
    csv_path = os.path.join(val_dir, 'justification_grounding.csv')
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    # ── Write aggregate JSON ──
    json_path = os.path.join(val_dir, 'justification_grounding.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({'vaamr': vaamr.get('aggregate'),
                   'purer': purer.get('aggregate')}, f, indent=2)

    # ── Write human report ──
    report_path = _write_report(vaamr, purer, df_lookup, output_dir)
    return report_path
