"""
tests/unit/test_proximal_cue.py
-------------------------------
Hermetic tests for analysis/proximal_cue.ProximalCueExtractor.

Uses a synthetic VTT + synthetic therapist segment spans; the PHI scrubber is
always mocked (never loads the NER model). Covers: window slicing, tail-words
cap, boundary clipping + coarse-caption approx flag, non-therapist exclusion,
invalid/missing inputs, and the withheld-when-scrubber-unavailable guarantee.
"""

import os
import sys
import tempfile
import unittest
from unittest.mock import patch

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))

import pandas as pd


def _write_vtt(out_dir, session_id, cues):
    """cues: list of (start_s, end_s, text)."""
    d = os.path.join(out_dir, '01_transcripts_inputs')
    os.makedirs(d, exist_ok=True)
    lines = ['WEBVTT', '']

    def ts(sec):
        h, rem = divmod(sec, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02d}:{int(m):02d}:{s:06.3f}"

    for i, (s, e, text) in enumerate(cues, 1):
        lines += [str(i), f"{ts(s)} --> {ts(e)}", text, '']
    with open(os.path.join(d, f'{session_id}.vtt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


def _df_all(session_id, therapist_spans_ms):
    rows = []
    for i, (s, e) in enumerate(therapist_spans_ms):
        rows.append({'segment_id': f'th_{i}', 'session_id': session_id,
                     'speaker': 'therapist', 'start_time_ms': s, 'end_time_ms': e})
    # one participant row so the groupby has mixed speakers
    rows.append({'segment_id': 'p_0', 'session_id': session_id,
                 'speaker': 'participant', 'start_time_ms': 0, 'end_time_ms': 1000})
    return pd.DataFrame(rows)


def _unscrubbed(extractor):
    """Bypass the real scrubber: pretend engine loaded, identity scrub."""
    extractor._engine_backend = 'mock'
    extractor._patterns = None
    return patch('process.text_anonymization.scrub_text',
                 side_effect=lambda text, patterns, **kw: (text, 0, 0))


class TestProximalSlicing(unittest.TestCase):

    def _extractor(self, tmp, cues, spans_ms, max_words=250):
        from analysis.proximal_cue import ProximalCueExtractor
        _write_vtt(tmp, 's1', cues)
        return ProximalCueExtractor(tmp, _df_all('s1', spans_ms), max_words=max_words)

    def test_window_slice_and_tail(self):
        # therapist speaks 10-20s ("alpha beta"), 20-30s ("gamma delta");
        # window covers both fully → all 4 words, tail capped at 3.
        cues = [(10, 20, 'Therapist: alpha beta'), (20, 30, 'Therapist: gamma delta')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(9_000, 31_000)], max_words=3)
            with _unscrubbed(ex):
                out = ex.extract('s1', 9_000, 31_000)
            self.assertIsNotNone(out)
            self.assertFalse(out['withheld'])
            self.assertEqual(out['window_words'], 4)
            self.assertEqual(out['text'], 'beta gamma delta')  # last 3 of 4
            self.assertFalse(out['approx'])

    def test_boundary_clipping_and_coarse_approx(self):
        # One 60s caption of 6 words; window covers only its second half
        # → proportional clip keeps last 3 words, coarse (>15s) → approx.
        cues = [(0, 60, 'T: w1 w2 w3 w4 w5 w6')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(0, 60_000)])
            with _unscrubbed(ex):
                out = ex.extract('s1', 30_000, 60_000)
            self.assertEqual(out['text'], 'w4 w5 w6')
            self.assertTrue(out['approx'])

    def test_non_therapist_captions_excluded(self):
        # second caption lies outside every therapist span → excluded
        cues = [(10, 20, 'T: kept words'), (20, 30, 'P: dropped words')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(10_000, 20_000)])
            with _unscrubbed(ex):
                out = ex.extract('s1', 5_000, 35_000)
            self.assertEqual(out['text'], 'kept words')

    def test_empty_window_and_invalid_inputs(self):
        cues = [(10, 20, 'T: something here')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(10_000, 20_000)])
            with _unscrubbed(ex):
                empty = ex.extract('s1', 40_000, 50_000)   # after all speech
                self.assertEqual(empty['text'], '')
                self.assertEqual(empty['window_words'], 0)
                self.assertFalse(empty['window_fallback'])
                self.assertIsNone(ex.extract('s1', 10_000, 0))       # ts invalid
                self.assertIsNone(ex.extract('s1', 10_000, None))    # ts missing
                self.assertIsNone(ex.extract('missing', 10_000, 20_000))  # no VTT

    def test_fallback_window_when_from_end_unusable(self):
        """fe missing or fe >= ts → lookback window before TO, flagged."""
        cues = [(10, 20, 'T: alpha beta gamma')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(9_000, 21_000)])
            with _unscrubbed(ex):
                a = ex.extract('s1', 0, 25_000)         # fe invalid
                b = ex.extract('s1', 30_000, 25_000)    # fe >= ts (overlap)
            for out in (a, b):
                self.assertTrue(out['window_fallback'])
                self.assertEqual(out['text'], 'alpha beta gamma')

    def test_withheld_when_scrubber_unavailable(self):
        cues = [(10, 20, 'T: real name secret')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(10_000, 20_000)])
            with patch.object(type(ex), '_ensure_scrubber', return_value=False):
                out = ex.extract('s1', 9_000, 21_000)
            self.assertTrue(out['withheld'])
            self.assertIsNone(out['text'])  # raw text NEVER returned

    def test_caption_label_attribution_overrides_spans(self):
        """A participant-labeled caption inside a therapist span is EXCLUDED;
        staff excluded; therapist-labeled included; unmapped labels use spans."""
        cues = [(10, 20, 'T: kept therapist words'),
                (20, 30, 'P: leaked participant words'),
                (30, 40, 'S: staff logistics words')]
        with tempfile.TemporaryDirectory() as tmp:
            # therapist span covers EVERYTHING (the real-data failure mode)
            ex = self._extractor(tmp, cues, [(0, 60_000)])
            ex._roles = {'t': 'therapist', 'p': 'participant', 's': 'staff'}
            with _unscrubbed(ex):
                out = ex.extract('s1', 5_000, 45_000)
            self.assertEqual(out['text'], 'kept therapist words')

    def test_unmapped_label_falls_back_to_spans(self):
        cues = [(10, 20, 'X: unmapped inside span'), (25, 35, 'Y: unmapped outside span')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(10_000, 20_000)])
            ex._roles = {}
            with _unscrubbed(ex):
                out = ex.extract('s1', 5_000, 45_000)
            self.assertEqual(out['text'], 'unmapped inside span')

    def test_belt_and_braces_name_token_scrub(self):
        """A known-name token that survives the production scrub is replaced
        with (NAME) by the post-pass, regardless of key ambiguity."""
        import re
        cues = [(10, 20, 'T: probably Wade will fix it tomorrow')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(9_000, 21_000)])
            ex._name_token_re = re.compile(r'\b(Wade|Krista)\b', re.IGNORECASE)
            with _unscrubbed(ex):
                out = ex.extract('s1', 9_000, 21_000)
            self.assertEqual(out['text'], 'probably (NAME) will fix it tomorrow')

    def test_speakers_reported_as_anonymized_ids(self):
        cues = [(10, 20, 'T: first chunk'), (20, 30, 'U: second chunk')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(9_000, 31_000)])
            ex._roles = {'t': 'therapist', 'u': 'therapist'}
            ex._anon = {'t': 'therapist_1', 'u': 'therapist_2'}
            with _unscrubbed(ex):
                out = ex.extract('s1', 9_000, 31_000)
            self.assertEqual(out['speakers'], ['therapist_1', 'therapist_2'])
            self.assertEqual(out['text'], 'first chunk second chunk')

    def test_cache_returns_same_object(self):
        cues = [(10, 20, 'T: alpha beta')]
        with tempfile.TemporaryDirectory() as tmp:
            ex = self._extractor(tmp, cues, [(9_000, 21_000)])
            with _unscrubbed(ex):
                a = ex.extract('s1', 9_000, 21_000)
                b = ex.extract('s1', 9_000, 21_000)
            self.assertIs(a, b)


if __name__ == '__main__':
    unittest.main()
