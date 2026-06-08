"""
tests/unit/test_justification_grounding.py
------------------------------------------
Hermetic unit tests for the LLM justification-grounding audit (Feature 1).
No network, no model downloads, stdlib-only code path.

Covers:
  - a justification that quotes real segment text  -> GROUNDED, not flagged
  - a justification quoting text NOT in the segment -> ungrounded -> flagged
  - a justification with no quotes + low overlap    -> flagged
  - aggregate rates (% spans grounded, flag count) are computed correctly
  - per-rater roll-up from rater_votes (JSON-string ballots) works
  - fuzzy substring fallback grounds a paraphrase within threshold
  - PURER audit is produced from df_all, and skipped gracefully when absent
  - the three output artifacts are written (CSV, JSON, report .txt)
"""

import json
import os
import sys
import tempfile
import unittest

import pandas as pd

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))
if _QRA_ROOT not in sys.path:
    sys.path.insert(1, _QRA_ROOT)

from analysis.reports import justification_grounding as JG


_FRAMEWORK = {
    0: {'key': 'vigilance', 'short_name': 'Vigilance', 'name': 'Vigilance'},
    1: {'key': 'avoidance', 'short_name': 'Avoidance', 'name': 'Avoidance'},
    2: {'key': 'attn_reg', 'short_name': 'AttnReg', 'name': 'Attention Regulation'},
    3: {'key': 'metacog', 'short_name': 'Metacog', 'name': 'Metacognition'},
    4: {'key': 'reappraisal', 'short_name': 'Reappraisal', 'name': 'Reappraisal'},
}


def _participant_df():
    """A tiny fixture: 3 participant segments with engineered grounding cases."""
    return pd.DataFrame([
        # (i) GROUNDED: justification quotes language present in the segment.
        {
            'segment_id': 'seg_grounded',
            'speaker': 'participant',
            'text': "I notice the tightness in my shoulders and I just watch it come and go.",
            'primary_stage': 3,
            'final_label': 3,
            'llm_justification': 'The participant says they "watch it come and go", reflecting '
                                 'metacognitive observation of the sensation.',
            'rater_votes': json.dumps([
                {'rater': 'model-good', 'vote': 3, 'stage': 3,
                 'justification': 'They "watch it come and go" — observing the process.'},
                {'rater': 'model-bad', 'vote': 3, 'stage': 3,
                 'justification': 'The participant mentions "deep ocean meditation tapes" at length.'},
            ]),
        },
        # (ii) UNGROUNDED quote: cites a phrase NOT in the segment text.
        {
            'segment_id': 'seg_ungrounded_quote',
            'speaker': 'participant',
            'text': "Yeah. Right. Okay.",
            'primary_stage': 2,
            'final_label': 2,
            'llm_justification': 'The participant describes how they "redirected attention back to '
                                 'the breath whenever the pain spiked", showing attention regulation.',
            'rater_votes': json.dumps([
                {'rater': 'model-bad', 'vote': 2, 'stage': 2,
                 'justification': 'They "redirected attention back to the breath" repeatedly.'},
            ]),
        },
        # (iii) NO quotes + low lexical overlap -> flagged via overlap rule.
        {
            'segment_id': 'seg_no_quote_low',
            'speaker': 'participant',
            'text': "Cat dog tree river mountain.",
            'primary_stage': 0,
            'final_label': 0,
            'llm_justification': 'The participant exhibits hypervigilant somatic monitoring with '
                                 'catastrophic appraisal of nociceptive signals throughout.',
            'rater_votes': None,
        },
    ])


class TestPerSpanGrounding(unittest.TestCase):
    def test_exact_substring_grounded(self):
        text = "I just watch it come and go without judging it."
        self.assertTrue(JG._span_grounded("watch it come and go", JG._normalize(text)))

    def test_absent_span_not_grounded(self):
        text = "Yeah. Right. Okay."
        self.assertFalse(JG._span_grounded("redirected attention back to the breath",
                                           JG._normalize(text)))

    def test_fuzzy_fallback_grounds_paraphrase(self):
        # One-word transcription drift should still ground via the >=0.90 fuzzy window.
        text = "I redirected my attention back to the breath each time."
        self.assertTrue(JG._span_grounded("redirected my attention back to the breath",
                                          JG._normalize(text)))

    def test_short_span_not_fuzzy_grounded(self):
        # F1-B: a tiny span (< 8 normalized chars) must NOT be fuzzy-grounded against
        # unrelated text — the ratio>=0.90 window saturates for short fragments.
        # "the cat" is 7 chars; it is absent from the text and must stay ungrounded.
        self.assertFalse(JG._span_grounded("the cat",
                                           JG._normalize("a completely unrelated sentence here")))
        # But an exact substring of any length is still grounded (exact path unchanged).
        self.assertTrue(JG._span_grounded("the cat",
                                          JG._normalize("I saw the cat run away.")))

    def test_audit_one_no_quote_uses_overlap(self):
        rec = JG._audit_one("hypervigilant catastrophic nociceptive monitoring",
                            "cat dog tree river")
        self.assertFalse(rec['has_quotes'])
        self.assertTrue(rec['ungrounded'])     # overlap ~ 0 < 0.15


class TestQuoteExtraction(unittest.TestCase):
    def test_double_and_curly_quotes_extracted(self):
        self.assertEqual(JG._extract_quotes('She said "watch the breath" now.'),
                         ['watch the breath'])
        self.assertEqual(JG._extract_quotes('The phrase “come and go” matters.'),
                         ['come and go'])

    def test_real_single_quote_extracted(self):
        self.assertEqual(JG._extract_quotes("They state 'I just watch it' clearly."),
                         ['I just watch it'])

    def test_single_quoted_span_with_apostrophe_extracted(self):
        # F1-A regression: a single-quoted span that CONTAINS a contraction must
        # still be captured (the old `[^']{2,}` form silently dropped these, which
        # undercounted the denominator on real participant speech).
        self.assertEqual(JG._extract_quotes("She said 'I'm a walking miracle' today."),
                         ["I'm a walking miracle"])
        self.assertEqual(
            JG._extract_quotes("They felt 'no one's ever taught me that' before."),
            ["no one's ever taught me that"])
        # And a sentence mixing a bare contraction with a real single-quoted phrase
        # that itself contains a contraction.
        self.assertEqual(
            JG._extract_quotes("He's calmer now: 'I'm fine with it' he reported."),
            ["I'm fine with it"])

    def test_apostrophe_contractions_not_mis_extracted(self):
        # The single-quote pattern must be boundary-aware: contractions are NOT quotes.
        self.assertEqual(JG._extract_quotes("They don't want to and can't cope."), [])

    def test_mixed_contraction_and_quote(self):
        self.assertEqual(JG._extract_quotes("It doesn't matter, 'it just passes' anyway."),
                         ['it just passes'])


class TestAuditFrame(unittest.TestCase):
    def setUp(self):
        self.df = _participant_df()
        self.res = JG._audit_frame(self.df, _FRAMEWORK, 'llm_justification',
                                   'rater_votes', 'primary_stage', 'VAAMR')

    def test_three_rows_produced(self):
        self.assertEqual(len(self.res['rows']), 3)

    def test_grounded_segment_not_flagged(self):
        row = next(r for r in self.res['rows'] if r['segment_id'] == 'seg_grounded')
        self.assertTrue(row['has_quotes'])
        self.assertEqual(row['grounded_frac'], 1.0)
        self.assertFalse(row['ungrounded'])

    def test_ungrounded_quote_flagged(self):
        row = next(r for r in self.res['rows'] if r['segment_id'] == 'seg_ungrounded_quote')
        self.assertTrue(row['has_quotes'])
        self.assertEqual(row['grounded_frac'], 0.0)
        self.assertTrue(row['ungrounded'])

    def test_no_quote_low_overlap_flagged(self):
        row = next(r for r in self.res['rows'] if r['segment_id'] == 'seg_no_quote_low')
        self.assertFalse(row['has_quotes'])
        self.assertEqual(row['grounded_frac'], '')   # undefined when nothing quoted
        self.assertTrue(row['ungrounded'])

    def test_aggregate_flag_count_and_rate(self):
        agg = self.res['aggregate']
        # Exactly 2 of the 3 are flagged ungrounded.
        self.assertEqual(agg['n_ungrounded_flagged'], 2)
        self.assertEqual(agg['n_segments_with_justification'], 3)
        # 1 grounded span out of 2 total quoted spans across the two quoting segments.
        self.assertEqual(agg['spans_total'], 2)
        self.assertEqual(agg['spans_grounded'], 1)
        self.assertEqual(agg['pct_spans_grounded'], 50.0)

    def test_per_rater_rollup_distinguishes_models(self):
        pr = self.res['aggregate']['per_rater']
        # 'model-good' grounds its quote; 'model-bad' does not.
        self.assertIn('model-good', pr)
        self.assertIn('model-bad', pr)
        self.assertEqual(pr['model-good']['pct_spans_grounded'], 100.0)
        self.assertEqual(pr['model-bad']['pct_spans_grounded'], 0.0)


class TestPublicEntryPointAndArtifacts(unittest.TestCase):
    def test_writes_three_artifacts_and_purer_skips_gracefully(self):
        df = _participant_df()
        with tempfile.TemporaryDirectory() as out:
            # No df_all -> PURER audit must be skipped, VAAMR still produced.
            path = JG.generate_justification_grounding_report(df, _FRAMEWORK, out, df_all=None)
            self.assertIsNotNone(path)
            self.assertTrue(os.path.isfile(path))

            csv_path = os.path.join(out, '04_validation', 'justification_grounding.csv')
            json_path = os.path.join(out, '04_validation', 'justification_grounding.json')
            self.assertTrue(os.path.isfile(csv_path))
            self.assertTrue(os.path.isfile(json_path))

            cdf = pd.read_csv(csv_path)
            self.assertEqual(len(cdf), 3)
            self.assertEqual(int(cdf['ungrounded'].sum()), 2)

            with open(json_path) as f:
                agg = json.load(f)
            self.assertIsNotNone(agg['vaamr'])
            self.assertIsNone(agg['purer'])        # gracefully absent

            with open(path, encoding='utf-8') as _tf:
                txt = _tf.read()
            self.assertIn('JUSTIFICATION-GROUNDING AUDIT', txt)
            self.assertIn('LEAST-GROUNDED ITEMS', txt)

    def test_purer_audit_runs_when_df_all_has_purer(self):
        df = _participant_df()
        df_all = pd.concat([
            df.assign(speaker='participant'),
            pd.DataFrame([{
                'segment_id': 'th1', 'speaker': 'therapist',
                'text': "Okay.",
                'purer_primary': 4,
                'purer_justification': 'The therapist affirms with "great work this week", a '
                                       'reinforcement move.',
                'purer_rater_votes': json.dumps([
                    {'rater': 'purer-model', 'vote': 4,
                     'justification': 'Affirms with "great work this week".'},
                ]),
            }]),
        ], ignore_index=True)
        with tempfile.TemporaryDirectory() as out:
            JG.generate_justification_grounding_report(df, _FRAMEWORK, out, df_all=df_all)
            with open(os.path.join(out, '04_validation', 'justification_grounding.json')) as f:
                agg = json.load(f)
            self.assertIsNotNone(agg['purer'])
            self.assertEqual(agg['purer']['n_segments_with_justification'], 1)
            # The quote "great work this week" is NOT in "Okay." -> ungrounded.
            self.assertEqual(agg['purer']['n_ungrounded_flagged'], 1)

    def test_empty_frame_returns_none(self):
        with tempfile.TemporaryDirectory() as out:
            self.assertIsNone(
                JG.generate_justification_grounding_report(pd.DataFrame(), _FRAMEWORK, out))

    def test_no_justification_column_returns_none(self):
        df = pd.DataFrame([{'segment_id': 's1', 'text': 'hello', 'primary_stage': 1}])
        with tempfile.TemporaryDirectory() as out:
            self.assertIsNone(
                JG.generate_justification_grounding_report(df, _FRAMEWORK, out))


if __name__ == '__main__':
    unittest.main()
