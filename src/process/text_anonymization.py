"""
process/text_anonymization.py
------------------------------
Span-based ensemble PHI de-identification for therapy transcript segments.

Three-tier detection pipeline, all operating on *candidate spans* before any text
is mutated:

  Tier 1 — Known-name dictionary (speaker_anonymization_key.json)
            Deterministic, confidence = 1.0.  Replacements: {anonymized_id}.

  Tier 2 — High-precision transcript rules (greeting/vocative/direct-address
            patterns common in clinical conversation transcripts).
            Replacements: (NAME).  Confidence 0.60–0.85.

  Tier 3 — Presidio NLP engine (spaCy PERSON tagger OR HF transformer de-id
            model).  Replacements: (NAME).  Confidence from model score.

After collection, spans pass through:
  • Allowlist filter  — protect known non-PHI proper nouns (locations, practice
                        terms, days/months)
  • Context filter    — suppress spans preceded by location-preposition context
  • Confidence gate   — drop model hits below threshold (key_match always kept)
  • Overlap resolver  — deduplicate; prefer key_match > rule > model

Replacements applied right-to-left to preserve character offsets.

Public API:
  build_name_patterns(speaker_map)           → NamePatterns
  scrub_text(text, name_patterns, **kwargs)  → (scrubbed_text, n_known, n_unknown)
  scrub_segments(segments, speaker_map, ...) → (segments, stats_dict)
  init_engine(model_name)                    → backend_name_str

Engine state (lazy-loaded, module-level singleton):
  _engine.backend   # 'presidio_transformer' | 'presidio_spacy' | 'hf_transformer' | 'none'
"""
import re
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# DeIdSpan — core data structure
# ---------------------------------------------------------------------------

@dataclass
class DeIdSpan:
    """A candidate PHI span with source attribution and confidence."""
    start: int
    end: int
    text: str
    source: str          # 'key_match' | 'greeting_rule' | 'vocative_rule' | ... | 'presidio' | 'transformer'
    confidence: float    # 0.0–1.0; key_match is always 1.0
    replacement: str     # '{anonymized_id}' for known, '(NAME)' for unknown
    blocked_reason: Optional[str] = field(default=None, repr=False)


# ---------------------------------------------------------------------------
# NLP engine singleton (lazy-loaded)
# ---------------------------------------------------------------------------

_SOURCE_PRIORITY = {
    'key_match': 4,
    'greeting_rule': 2,
    'farewell_rule': 2,
    'trailing_vocative': 2,
    'vocative_rule': 1,
    'presidio': 2,
    'transformer': 2,
}

_NAME_LABELS = frozenset([
    'person', 'name', 'patient', 'doctor', 'username', 'per',
    'b-per', 'i-per',
])


class _DeIdEngine:
    """
    Lazy-loaded NLP engine with three-tier fallback:
      1. Presidio + HF TransformersNlpEngine  (best recall, requires model download)
      2. Presidio + SpacyNlpEngine            (good precision, en_core_web_sm)
      3. Direct HF transformers pipeline      (if Presidio not installed)
      4. None (regex rules only)
    """

    def __init__(self):
        self._presidio_engine = None
        self._hf_pipeline = None
        self.backend: str = 'none'
        self._loaded_model: Optional[str] = None

    def load(self, model_name: str = 'obi/deid_roberta_i2b2') -> str:
        """
        Load the best available backend for model_name. Idempotent — safe to
        call multiple times; re-loads only if the model name has changed.
        Returns the backend name string.
        """
        if self._loaded_model == model_name and self.backend != 'none':
            return self.backend

        self._loaded_model = model_name

        # ---- Tier 1: Presidio + TransformersNlpEngine ----------------------
        if model_name:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import TransformersNlpEngine
                nlp_config = {
                    'lang_code': 'en',
                    'model_name': {
                        'spacy': 'en_core_web_sm',
                        'transformers': model_name,
                    },
                }
                nlp_engine = TransformersNlpEngine(models=[nlp_config])
                self._presidio_engine = AnalyzerEngine(
                    nlp_engine=nlp_engine, supported_languages=['en']
                )
                self.backend = 'presidio_transformer'
                return self.backend
            except Exception:
                pass

        # ---- Tier 2: Presidio + SpacyNlpEngine -----------------------------
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_analyzer.nlp_engine import SpacyNlpEngine
            nlp_engine = SpacyNlpEngine(
                models=[{'lang_code': 'en', 'model_name': 'en_core_web_sm'}]
            )
            self._presidio_engine = AnalyzerEngine(
                nlp_engine=nlp_engine, supported_languages=['en']
            )
            self.backend = 'presidio_spacy'
            return self.backend
        except Exception:
            pass

        # ---- Tier 3: direct HF transformers pipeline -----------------------
        if model_name:
            try:
                from transformers import pipeline as hf_pipeline
                self._hf_pipeline = hf_pipeline(
                    'token-classification',
                    model=model_name,
                    aggregation_strategy='simple',
                    device=-1,
                )
                self.backend = 'hf_transformer'
                return self.backend
            except Exception:
                pass

        # ---- Tier 4: regex-only fallback -----------------------------------
        self.backend = 'none'
        if not getattr(self, '_warned', False):
            warnings.warn(
                "No NLP engine available for unknown-name detection "
                "(Presidio/spaCy not installed and HF model failed to load). "
                "Unknown names will be caught by regex heuristics only. "
                "Install with: pip install presidio-analyzer presidio-anonymizer spacy "
                "&& python -m spacy download en_core_web_sm",
                stacklevel=3,
            )
            self._warned = True
        return self.backend

    def predict_spans(self, text: str, fallback: str = '(NAME)') -> List[DeIdSpan]:
        """Return PERSON-entity spans detected by the loaded engine."""
        if self._presidio_engine is not None:
            return self._presidio_predict(text, fallback)
        if self._hf_pipeline is not None:
            return self._hf_predict(text, fallback)
        return []

    def _presidio_predict(self, text: str, fallback: str) -> List[DeIdSpan]:
        try:
            results = self._presidio_engine.analyze(
                text=text, language='en', entities=['PERSON']
            )
            return [
                DeIdSpan(
                    start=r.start, end=r.end,
                    text=text[r.start:r.end],
                    source=self.backend,
                    confidence=float(r.score),
                    replacement=fallback,
                )
                for r in results
            ]
        except Exception:
            return []

    def _hf_predict(self, text: str, fallback: str) -> List[DeIdSpan]:
        try:
            results = self._hf_pipeline(text)
            spans = []
            for r in results:
                if r['entity_group'].lower() in _NAME_LABELS:
                    spans.append(DeIdSpan(
                        start=r['start'], end=r['end'],
                        text=r['word'],
                        source='transformer',
                        confidence=float(r['score']),
                        replacement=fallback,
                    ))
            return spans
        except Exception:
            return []


# Module-level engine singleton
_engine = _DeIdEngine()


def init_engine(model_name: str = 'obi/deid_roberta_i2b2') -> str:
    """
    Explicitly initialize the NLP engine.  Safe to call at pipeline start to
    pay the load cost upfront rather than on the first segment.
    Returns the backend name.
    """
    return _engine.load(model_name)


# ---------------------------------------------------------------------------
# NamePatterns — known-name replacement lookup
# ---------------------------------------------------------------------------

class NamePatterns:
    """Pre-compiled patterns for known speaker names."""

    def __init__(self, pattern: Optional[re.Pattern], replacements: Dict[str, str]):
        self.pattern = pattern          # combined re.Pattern, or None if map is empty
        self.replacements = replacements  # {name_lower: '{anonymized_id}'}


def build_name_patterns(speaker_map: dict) -> NamePatterns:
    """
    Build NamePatterns from a speaker map  {raw_name: (role, anonymized_id)}.

    Multi-word names (e.g. "Michelle Berg") also generate single-word tokens
    for each component ≥ 3 chars, unless the token is shared across entries
    with different anonymized_ids (ambiguous → skip).  Sorted longest-first to
    prevent partial-match shadowing.
    """
    if not speaker_map:
        return NamePatterns(pattern=None, replacements={})

    full_replacements: Dict[str, str] = {}
    for raw_name, entry in speaker_map.items():
        if isinstance(entry, (list, tuple)):
            _, anon_id = entry
        elif isinstance(entry, dict):
            anon_id = entry.get('anonymized_id', raw_name)
        else:
            continue
        full_replacements[raw_name.lower()] = f'{{{anon_id}}}'

    token_to_anon: Dict[str, str] = {}
    token_ambiguous: set = set()

    for raw_name, entry in speaker_map.items():
        if isinstance(entry, (list, tuple)):
            _, anon_id = entry
        elif isinstance(entry, dict):
            anon_id = entry.get('anonymized_id', raw_name)
        else:
            continue

        parts = raw_name.split()
        if len(parts) < 2:
            continue

        for token in parts:
            if len(token) < 3:
                continue
            tok_lower = token.lower()
            if tok_lower in full_replacements:
                continue
            if tok_lower in token_to_anon:
                if token_to_anon[tok_lower] != anon_id:
                    token_ambiguous.add(tok_lower)
            else:
                token_to_anon[tok_lower] = anon_id

    for tok in token_ambiguous:
        token_to_anon.pop(tok, None)

    replacements = dict(full_replacements)
    for tok_lower, anon_id in token_to_anon.items():
        if tok_lower not in replacements:
            replacements[tok_lower] = f'{{{anon_id}}}'

    if not replacements:
        return NamePatterns(pattern=None, replacements={})

    sorted_names = sorted(replacements.keys(), key=len, reverse=True)
    pattern = re.compile(
        r'\b(?:' + '|'.join(re.escape(n) for n in sorted_names) + r')\b',
        re.IGNORECASE | re.UNICODE,
    )
    return NamePatterns(pattern=pattern, replacements=replacements)


# ---------------------------------------------------------------------------
# Lexicons
# ---------------------------------------------------------------------------

# Non-name tokens — filter rule-engine capture groups.
_NON_NAMES: frozenset = frozenset([
    'hi', 'hey', 'hello', 'thanks', 'thank', 'okay', 'ok', 'yes', 'no',
    'so', 'well', 'now', 'just', 'good', 'great', 'sure', 'right', 'oh',
    'ah', 'morning', 'afternoon', 'evening', 'night', 'bye', 'goodbye',
    'goodnight', 'what', 'when', 'where', 'why', 'how', 'who', 'this',
    'that', 'these', 'those', 'here', 'there', 'then', 'than', 'but',
    'and', 'the', 'for', 'with', 'from', 'not', 'are', 'was', 'were',
    'has', 'have', 'had', 'all', 'any', 'can', 'could', 'would', 'should',
    'pain', 'body', 'mind', 'more', 'move', 'study', 'session',
    'everyone', 'everybody', 'someone', 'somebody', 'anyone', 'anybody',
    # Mindfulness / Buddhist practice terms
    'metta', 'dharma', 'karma', 'yoga', 'buddha', 'zen', 'chi', 'qi',
    'loving', 'kindness', 'compassion', 'wisdom', 'awareness', 'presence',
    'breath', 'breathe', 'breathing', 'meditation', 'mindfulness',
    # Day / month names (common sentence-initial capitalisation)
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday',
    'january', 'february', 'march', 'april', 'june', 'july', 'august',
    'september', 'october', 'november', 'december',
])

# Allowlist — protect capitalised proper nouns that are not PHI from the
# model/Presidio tier.  These are added here (not _NON_NAMES) because the
# model tier captures spans, not single tokens, so a frozenset check on the
# full span text is cleaner.
_ALLOWLIST: frozenset = frozenset([
    # Mindfulness / Buddhist terms (capitalised forms the model may tag)
    'Metta', 'Dharma', 'Karma', 'Yoga', 'Buddha', 'Buddhism', 'Buddhist',
    'Zen', 'Vipassana', 'Samadhi', 'Nirvana', 'Loving-Kindness', "Eric Garland", "Jack Kornfield"
    # Geographic / cultural
    'India', 'China', 'Tibet', 'Japan', 'Korea', 'Nepal', 'Thailand',
    'America', 'American', 'English', 'Spanish', 'French', 'German',
    # Common product / brand names in sessions
    'YouTube', 'Zoom', 'Google',
    # Days and months — model tier may tag "Monday" as a PERSON in some contexts
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday',
    'January', 'February', 'March', 'April', 'May', 'June', 'July',
    'August', 'September', 'October', 'November', 'December',
])

# Context prepositions that strongly suggest a location/time span follows,
# not a person name.
_LOCATION_CTX_RE = re.compile(r'\b(?:in|from|at|near|of|to)\s*$', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Regex rule patterns for Tier 2
# ---------------------------------------------------------------------------

_GREETING_RE = re.compile(
    r'\b(?:[Hh]i|[Hh]ey|[Hh]ello|[Tt]hanks|[Tt]hank\s+[Yy]ou)'
    r'\s*,?\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b',
)
_FAREWELL_RE = re.compile(
    r'\b(?:[Bb]ye|[Gg]oodbye|[Gg]oodnight|[Gg]ood\s+[Nn]ight|[Ss]ee\s+[Yy]ou)'
    r'\s*,?\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)\b',
)
_VOCATIVE_RE = re.compile(
    r'(?<![{\w])([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)(?=\s*[,!?])',
)
_TRAILING_VOCATIVE_RE = re.compile(
    r',\s+([A-Z][a-z]{2,}(?:\s+[A-Z][a-z]{2,})?)(?=[.!?]|\s*$)',
    re.MULTILINE,
)

_RULE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    (_GREETING_RE,         'greeting_rule',    0.85),
    (_FAREWELL_RE,         'farewell_rule',    0.85),
    (_TRAILING_VOCATIVE_RE,'trailing_vocative', 0.75),
    (_VOCATIVE_RE,         'vocative_rule',    0.60),
]


# ---------------------------------------------------------------------------
# Span collection functions
# ---------------------------------------------------------------------------

def _known_key_spans(text: str, name_patterns: NamePatterns) -> List[DeIdSpan]:
    """Collect known-name spans from the speaker_anonymization_key."""
    if name_patterns.pattern is None:
        return []
    spans = []
    for m in name_patterns.pattern.finditer(text):
        replacement = name_patterns.replacements.get(m.group(0).lower())
        if replacement:
            spans.append(DeIdSpan(
                start=m.start(), end=m.end(),
                text=m.group(0),
                source='key_match',
                confidence=1.0,
                replacement=replacement,
            ))
    return spans


def _rule_spans(
    text: str,
    protected: List[Tuple[int, int]],
    fallback: str = '(NAME)',
) -> List[DeIdSpan]:
    """Collect PERSON candidate spans from high-precision transcript rules."""
    spans = []
    for pattern, source, confidence in _RULE_PATTERNS:
        for m in pattern.finditer(text):
            grp = m.group(1) if m.lastindex else None
            if not grp:
                continue
            if not grp[0].isupper():
                continue
            if grp.lower() in _NON_NAMES:
                continue
            if any(w.lower() in _NON_NAMES for w in grp.split()):
                continue
            # Locate the group span in the original text
            try:
                grp_start = text.index(grp, m.start())
            except ValueError:
                continue
            grp_end = grp_start + len(grp)
            if _overlaps_any(grp_start, grp_end, protected):
                continue
            spans.append(DeIdSpan(
                start=grp_start, end=grp_end,
                text=grp,
                source=source,
                confidence=confidence,
                replacement=fallback,
            ))
    return spans


def _model_spans(
    text: str,
    engine: _DeIdEngine,
    fallback: str = '(NAME)',
) -> List[DeIdSpan]:
    """Collect PERSON spans from the NLP engine (Presidio or direct HF)."""
    if engine.backend == 'none':
        return []
    return engine.predict_spans(text, fallback)


# ---------------------------------------------------------------------------
# Filter + resolution functions
# ---------------------------------------------------------------------------

def _overlaps_any(start: int, end: int, spans: List[Tuple[int, int]]) -> bool:
    return any(s < end and e > start for s, e in spans)


def _filter_spans(
    spans: List[DeIdSpan],
    text: str,
    confidence_threshold: float,
) -> List[DeIdSpan]:
    """Apply allowlist, context, and confidence filters."""
    kept = []
    for span in spans:
        if span.source == 'key_match':
            kept.append(span)
            continue
        # Allowlist: protect known non-PHI proper nouns
        if span.text in _ALLOWLIST:
            span.blocked_reason = 'allowlist'
            continue
        # Context filter: location-preposition precedes span
        prefix = text[max(0, span.start - 12): span.start]
        if _LOCATION_CTX_RE.search(prefix):
            span.blocked_reason = 'location_context'
            continue
        # Confidence gate
        if span.confidence < confidence_threshold:
            span.blocked_reason = f'low_confidence({span.confidence:.2f})'
            continue
        kept.append(span)
    return kept


def _resolve_overlaps(spans: List[DeIdSpan]) -> List[DeIdSpan]:
    """
    Deduplicate overlapping spans.  key_match always wins.  Within the same
    priority tier, prefer higher confidence.  Sweep left-to-right.
    """
    # Sort: higher priority first, then higher confidence
    spans.sort(
        key=lambda s: (_SOURCE_PRIORITY.get(s.source, 0), s.confidence),
        reverse=True,
    )
    resolved: List[DeIdSpan] = []
    for span in spans:
        if not _overlaps_any(span.start, span.end, [(s.start, s.end) for s in resolved]):
            resolved.append(span)
    return resolved


def _apply_replacements(text: str, spans: List[DeIdSpan]) -> str:
    """Apply span replacements right-to-left to preserve character offsets."""
    for span in sorted(spans, key=lambda s: s.start, reverse=True):
        text = text[:span.start] + span.replacement + text[span.end:]
    return text


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scrub_text(
    text: str,
    name_patterns: NamePatterns,
    *,
    use_transformer: bool = True,
    confidence_threshold: float = 0.6,
    fallback: str = '(NAME)',
) -> Tuple[str, int, int]:
    """
    Scrub PHI names from a single text string using the span-based ensemble.

    Returns:
        (scrubbed_text, n_known_replaced, n_unknown_replaced)
    """
    # Tier 1: known-name spans
    key_spans = _known_key_spans(text, name_patterns)
    protected = [(s.start, s.end) for s in key_spans]

    # Tier 2: rule-based spans (skip regions already covered by key_match)
    rule = _rule_spans(text, protected, fallback)

    # Tier 3: model spans (skip already-covered regions)
    all_protected = protected + [(s.start, s.end) for s in rule]
    model = (
        [s for s in _model_spans(text, _engine, fallback)
         if not _overlaps_any(s.start, s.end, all_protected)]
        if use_transformer
        else []
    )

    all_spans = key_spans + rule + model

    # Filter + resolve
    all_spans = _filter_spans(all_spans, text, confidence_threshold)
    final = _resolve_overlaps(all_spans)

    n_known = sum(1 for s in final if s.source == 'key_match')
    n_unknown = len(final) - n_known

    return _apply_replacements(text, final), n_known, n_unknown


def scrub_segments(
    segments: list,
    speaker_map: dict,
    *,
    use_transformer: bool = True,
    confidence_threshold: float = 0.6,
    fallback: str = '(NAME)',
    model_name: str = 'obi/deid_roberta_i2b2',
) -> Tuple[list, dict]:
    """
    Apply PHI name scrubbing to the text field of each segment in-place.

    Initialises the NLP engine once before the segment loop.
    Returns (segments, stats) with keys n_known, n_unknown, n_segments_modified.
    """
    if use_transformer and _engine.backend == 'none':
        _engine.load(model_name)

    name_patterns = build_name_patterns(speaker_map)
    total_known = 0
    total_unknown = 0
    n_modified = 0

    for seg in segments:
        if not seg.text:
            continue
        scrubbed, n_k, n_u = scrub_text(
            seg.text, name_patterns,
            use_transformer=use_transformer,
            confidence_threshold=confidence_threshold,
            fallback=fallback,
        )
        if scrubbed != seg.text:
            seg.text = scrubbed
            seg.word_count = len(scrubbed.split())
            n_modified += 1
        total_known += n_k
        total_unknown += n_u

    return segments, {
        'n_known': total_known,
        'n_unknown': total_unknown,
        'n_segments_modified': n_modified,
        'engine_backend': _engine.backend,
    }
