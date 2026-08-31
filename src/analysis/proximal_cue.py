"""
analysis/proximal_cue.py
------------------------
Proximal therapist-cue extraction for the language atlas.

The cue-block "window" (all therapist speech between two stage-bearing
participant turns) routinely spans thousands of words because therapist
segments can cover an entire lesson. The mechanistically interesting language
is the therapist speech IMMEDIATELY before the participant's next (TO) turn.

This module reconstructs that proximal speech from the raw session VTT
captions (5–8 s granularity in most sessions), using caption timestamps for
slicing. Speaker attribution is caption-label based: each caption's raw VTT
speaker label is resolved to a role via the project speaker key — only
role == 'therapist' captions enter the cue (participant and staff speech is
excluded). Labels absent from the key fall back to a time-span heuristic
against the frozen therapist segments.

PHI: raw VTT text contains real names, so every extracted cue is passed
through the production de-identification ensemble
(process/text_anonymization.scrub_text) with the known-name patterns from the
project speaker key. If the scrub engine cannot load, the cue is WITHHELD —
raw text is never returned.
"""

import os
from typing import Dict, List, Optional, Tuple

# Captions longer than this are considered coarse: boundary clipping inside
# them uses word-proportion interpolation, so times/word-cuts are approximate.
_COARSE_CAPTION_MS = 15_000

# When the FROM turn's end is unusable (missing, or the FROM segment overlaps
# the TO turn — long diarized participant spans do this), the proximal window
# falls back to this lookback before the TO turn.
_FALLBACK_LOOKBACK_MS = 300_000

# Speaker-key name tokens that are ordinary words, never scrubbed as names by
# the belt-and-braces pass (they'd destroy normal prose).
_GENERIC_NAME_TOKENS = {'speaker', 'study', 'move', 'more', 'movemore',
                        'mindfulness', 'coordinator', 'anon', 'unknown'}


class ProximalCueExtractor:
    """Extract the last `max_words` therapist words before a TO turn.

    Parameters
    ----------
    output_dir : str
        Project output dir (locates `01_transcripts_inputs/` + speaker key).
    df_all : pd.DataFrame
        Assembled corpus (needs session_id / speaker / start_time_ms /
        end_time_ms) — supplies therapist segment spans per session.
    max_words : int
        Tail size of the proximal cue.
    """

    def __init__(self, output_dir: str, df_all, max_words: int = 250,
                 confidence_threshold: float = 0.6,
                 model_name: str = 'obi/deid_roberta_i2b2'):
        self.output_dir = output_dir
        self.max_words = int(max_words)
        self.confidence_threshold = confidence_threshold
        self.model_name = model_name
        self._captions: Dict[str, Optional[List[Tuple[float, float, str, str]]]] = {}
        self._cache: Dict[tuple, Optional[dict]] = {}
        self._patterns = None
        self._engine_backend = None  # lazy: None = not tried yet
        self._roles: Optional[Dict[str, str]] = None  # lazy: raw label (lower) → role
        self._anon: Dict[str, str] = {}  # raw label (lower) → anonymized_id
        self._name_token_re = None  # lazy: belt-and-braces known-name-token regex

        # Therapist segment spans per session, from the frozen (scrubbed) store.
        self._spans: Dict[str, List[Tuple[float, float]]] = {}
        if df_all is not None and 'speaker' in getattr(df_all, 'columns', []):
            th = df_all[df_all['speaker'] == 'therapist']
            for sid, g in th.groupby('session_id'):
                spans = []
                for s, e in zip(g['start_time_ms'], g['end_time_ms']):
                    try:
                        s, e = float(s), float(e)
                    except (TypeError, ValueError):
                        continue
                    if e > s and e > 0 and s >= 0:
                        spans.append((s, e))
                self._spans[str(sid)] = sorted(spans)

    # -- lazy loaders --------------------------------------------------------

    def _ensure_scrubber(self) -> bool:
        """Load name patterns + NER engine once. False → must withhold."""
        if self._engine_backend is not None:
            return self._engine_backend != 'none'
        try:
            from process.text_anonymization import build_name_patterns, init_engine
            from process.speaker_anonymization import load_speaker_map
            from process import output_paths as _paths
            speaker_map, _ = load_speaker_map(_paths.meta_dir(self.output_dir), config=None)
            self._patterns = build_name_patterns(speaker_map)
            self._engine_backend = init_engine(self.model_name) or 'none'
        except Exception:
            self._engine_backend = 'none'
        return self._engine_backend != 'none'

    def _session_captions(self, session_id: str):
        """[(start_ms, end_ms, speaker, text)] from the session VTT, or None."""
        sid = str(session_id)
        if sid in self._captions:
            return self._captions[sid]
        path = os.path.join(self.output_dir, '01_transcripts_inputs', f'{sid}.vtt')
        caps = None
        if os.path.isfile(path):
            try:
                from process.transcript_ingestion import load_vtt_session
                sentences = load_vtt_session(path).get('sentences', [])
                caps = [(float(s['start']) * 1000.0, float(s['end']) * 1000.0,
                         str(s.get('speaker', '') or ''), str(s['text']))
                        for s in sentences if str(s.get('text', '')).strip()]
                caps.sort(key=lambda c: c[0])
            except Exception:
                caps = None
        self._captions[sid] = caps
        return caps

    def _role_map(self) -> Dict[str, str]:
        """Speaker-key raw names (lowercased) → role ('therapist'/'participant'/'staff').
        Also fills self._anon (raw name lower → anonymized_id)."""
        if self._roles is None:
            roles: Dict[str, str] = {}
            anon: Dict[str, str] = {}
            try:
                from process.speaker_anonymization import load_speaker_map
                from process import output_paths as _paths
                speaker_map, _ = load_speaker_map(_paths.meta_dir(self.output_dir), config=None)
                for raw_name, entry in speaker_map.items():
                    if isinstance(entry, (list, tuple)):
                        role, aid = entry[0], entry[1] if len(entry) > 1 else None
                    elif isinstance(entry, dict):
                        role, aid = entry.get('role'), entry.get('anonymized_id')
                    else:
                        continue
                    if role:
                        k = str(raw_name).strip().lower()
                        roles[k] = str(role).strip().lower()
                        if aid:
                            anon[k] = str(aid)
            except Exception:
                pass
            self._roles = roles
            self._anon = anon
        return self._roles

    def _scrub_name_tokens(self, text: str) -> str:
        """Belt-and-braces pass AFTER the production scrub: replace any bare
        speaker-key name token that survived (e.g. a given name the key holds
        under two spellings, which the known-name patterns skip as ambiguous
        and the NER may miss). Generic replacement '(NAME)' needs no identity
        resolution, so ambiguity is irrelevant; common-word tokens are
        excluded via _GENERIC_NAME_TOKENS."""
        import re as _re
        if self._name_token_re is None:
            toks = set()
            try:
                from process.speaker_anonymization import load_speaker_map
                from process import output_paths as _paths
                speaker_map, _ = load_speaker_map(_paths.meta_dir(self.output_dir), config=None)
                for raw_name in speaker_map:
                    for t in _re.split(r'[\s\-_.()]+', str(raw_name)):
                        if len(t) >= 3 and not t.isdigit() \
                                and t.lower() not in _GENERIC_NAME_TOKENS:
                            toks.add(_re.escape(t))
            except Exception:
                pass
            self._name_token_re = _re.compile(
                r'\b(' + '|'.join(sorted(toks)) + r')\b', _re.IGNORECASE) if toks else False
        if not self._name_token_re:
            return text
        return self._name_token_re.sub('(NAME)', text)

    # -- core ----------------------------------------------------------------

    def _span_overlap_therapist(self, session_id: str, cs: float, ce: float) -> bool:
        """Fallback heuristic: >50% of the caption lies in a therapist span."""
        dur = max(ce - cs, 1.0)
        covered = 0.0
        for s, e in self._spans.get(str(session_id), []):
            lo, hi = max(cs, s), min(ce, e)
            if hi > lo:
                covered += hi - lo
        return covered / dur > 0.5

    def _is_therapist(self, session_id: str, speaker: str, cs: float, ce: float) -> bool:
        """Caption-label role lookup first; span heuristic only for unmapped labels."""
        role = self._role_map().get(str(speaker or '').strip().lower())
        if role is not None:
            return role == 'therapist'
        return self._span_overlap_therapist(session_id, cs, ce)

    def extract(self, session_id: str, fe_ms, ts_ms) -> Optional[dict]:
        """Proximal cue for the window [fe_ms, ts_ms] (FROM end → TO start).

        Returns None when the TO time is invalid or no VTT exists. When the
        FROM end is unusable (missing or >= TO start — overlapping diarized
        spans), the window falls back to the last `_FALLBACK_LOOKBACK_MS`
        before TO, flagged `window_fallback=True`. Otherwise:
        {'text', 'start_ms', 'end_ms', 'window_words', 'approx', 'withheld',
         'window_fallback'} — text is '' when no therapist speech falls in
        the window.
        """
        try:
            ts = float(ts_ms)
        except (TypeError, ValueError):
            return None
        if not ts > 0:
            return None
        fallback = False
        try:
            fe = float(fe_ms)
        except (TypeError, ValueError):
            fe = 0.0
        if not (fe > 0 and ts > fe):
            fe = max(0.0, ts - _FALLBACK_LOOKBACK_MS)
            fallback = True
        key = (str(session_id), round(fe), round(ts))
        if key in self._cache:
            return self._cache[key]

        caps = self._session_captions(session_id)
        if caps is None:
            self._cache[key] = None
            return None

        words: List[Tuple[str, float, str]] = []  # (word, approx_start_ms, anon_speaker)
        approx = False
        for cs, ce, speaker, text in caps:
            if ce <= fe or cs >= ts:
                continue
            if not self._is_therapist(session_id, speaker, cs, ce):
                continue
            spk_id = self._anon.get(str(speaker or '').strip().lower(), 'therapist')
            toks = text.split()
            if not toks:
                continue
            n = len(toks)
            dur = max(ce - cs, 1.0)
            lo_i, hi_i = 0, n
            if cs < fe:   # caption starts before the window — clip its head
                lo_i = min(n, int(round(n * (fe - cs) / dur)))
            if ce > ts:   # caption runs past the TO turn — clip its tail
                hi_i = max(lo_i, int(round(n * (ts - cs) / dur)))
            if (cs < fe or ce > ts) and dur > _COARSE_CAPTION_MS:
                approx = True
            for i in range(lo_i, hi_i):
                words.append((toks[i], cs + (i / n) * dur, spk_id))

        window_words = len(words)
        if window_words == 0:
            out = {'text': '', 'start_ms': None, 'end_ms': None, 'speakers': [],
                   'window_words': 0, 'approx': approx, 'withheld': False, 'window_fallback': fallback}
            self._cache[key] = out
            return out

        tail = words[-self.max_words:]
        speakers = []
        for _, _, spk in tail:
            if spk not in speakers:
                speakers.append(spk)
        if not self._ensure_scrubber():
            out = {'text': None, 'start_ms': tail[0][1], 'end_ms': ts, 'speakers': speakers,
                   'window_words': window_words, 'approx': approx, 'withheld': True, 'window_fallback': fallback}
            self._cache[key] = out
            return out

        from process.text_anonymization import scrub_text
        raw = ' '.join(w for w, _, _ in tail)
        scrubbed, _, _ = scrub_text(
            raw, self._patterns,
            confidence_threshold=self.confidence_threshold)
        scrubbed = self._scrub_name_tokens(scrubbed)
        # The tail's first word may start mid-caption; its timestamp is
        # interpolated, which only matters for coarse captions.
        first_ms = tail[0][1]
        if len(words) > len(tail):
            approx = approx or any(
                cs <= first_ms <= ce and (ce - cs) > _COARSE_CAPTION_MS
                for cs, ce, _, _t in caps)
        out = {'text': scrubbed, 'start_ms': first_ms, 'end_ms': ts, 'speakers': speakers,
               'window_words': window_words, 'approx': approx, 'withheld': False, 'window_fallback': fallback}
        self._cache[key] = out
        return out
