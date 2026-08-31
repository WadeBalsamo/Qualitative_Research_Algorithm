"""
tests/unit/test_cue_distill.py
------------------------------
Hermetic tests for analysis/cue_distill.CueDistiller — the embedder is always
mocked (or forced into fallback); no model download.
"""

import os
import sys
import unittest
from unittest.mock import patch

_QRA_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_QRA_ROOT, 'src'))

import numpy as np


def _mock_embed(distiller, table):
    """Patch distiller._embed with a lookup table {text: vector}."""
    def fake(texts):
        return np.stack([np.asarray(table[t], dtype=float) for t in texts])
    return patch.object(distiller, '_embed', side_effect=fake)


class TestSplitSentences(unittest.TestCase):

    def test_basic_split(self):
        from analysis.cue_distill import split_sentences
        s = split_sentences("First one. Second one! Third… Fourth?")
        self.assertEqual(s, ['First one.', 'Second one!', 'Third…', 'Fourth?'])

    def test_empty(self):
        from analysis.cue_distill import split_sentences
        self.assertEqual(split_sentences(''), [])
        self.assertEqual(split_sentences(None), [])


class TestPrototype(unittest.TestCase):

    def test_centroid_nearest(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        cues = ['a', 'b', 'c']
        # centroid ≈ (0.33, 0.33); 'b' at 45° is nearest
        table = {'a': [1, 0], 'b': [0.7, 0.7], 'c': [0, 1]}
        with _mock_embed(d, table):
            self.assertEqual(d.prototype(cues), 1)

    def test_fallback_median_length(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        d._failed = True
        cues = ['one', 'one two three', 'one two']  # word counts 1, 3, 2
        self.assertEqual(d.prototype(cues), 2)      # median length = 2 words
        self.assertEqual(d.method, 'tail-fallback')

    def test_single_cue(self):
        from analysis.cue_distill import CueDistiller
        self.assertEqual(CueDistiller().prototype(['only']), 0)


class TestKeySentences(unittest.TestCase):

    def test_selects_most_central_in_order(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        cue = "Alpha alpha. Beta beta. Gamma gamma. Delta delta."
        group = [cue]
        table = {
            cue: [1, 0],
            'Alpha alpha.': [1, 0], 'Beta beta.': [0, 1],
            'Gamma gamma.': [0.9, 0.1], 'Delta delta.': [0.8, 0.2],
        }
        with _mock_embed(d, table):
            sents, elided = d.key_sentences(cue, group, k=2)
        # top-2 by similarity to centroid [1,0]: Alpha, Gamma — original order
        self.assertEqual(sents, ['Alpha alpha.', 'Gamma gamma.'])
        self.assertTrue(elided)

    def test_short_cue_returned_whole(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        sents, elided = d.key_sentences("One. Two.", ["One. Two."], k=3)
        self.assertEqual(sents, ['One.', 'Two.'])
        self.assertFalse(elided)

    def test_fallback_takes_tail(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        d._failed = True
        cue = "S1. S2. S3. S4. S5."
        sents, elided = d.key_sentences(cue, [cue], k=2)
        self.assertEqual(sents, ['S4.', 'S5.'])
        self.assertTrue(elided)


class TestContrastAxis(unittest.TestCase):

    def test_axis_direction_and_scoring(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        table = {'fwd cue': [1.0, 0.0], 'bwd cue': [0.0, 1.0],
                 'Forward-ish sentence one.': [0.9, 0.1],
                 'Backward-ish sentence two.': [0.1, 0.9]}
        with _mock_embed(d, table):
            axis = d.contrast_axis(['fwd cue'], ['bwd cue'])
            self.assertIsNotNone(axis)
            scored = d.score_sentences(
                'Forward-ish sentence one. Backward-ish sentence two.', axis)
        self.assertGreater(scored[0][1], 0)   # forward-ish → positive
        self.assertLess(scored[1][1], 0)      # backward-ish → negative

    def test_axis_none_when_pool_empty_or_fallback(self):
        from analysis.cue_distill import CueDistiller
        d = CueDistiller()
        self.assertIsNone(d.contrast_axis([], ['x']))
        self.assertIsNone(d.contrast_axis(['x'], []))
        d._failed = True
        self.assertIsNone(d.contrast_axis(['x'], ['y']))

    def test_key_sentences_axis_ranking_with_sign(self):
        from analysis.cue_distill import CueDistiller
        import numpy as np
        d = CueDistiller()
        cue = "Fwd fwd one. Neutral middle two. Bwd bwd three."
        table = {'Fwd fwd one.': [1.0, 0.0], 'Neutral middle two.': [0.5, 0.5],
                 'Bwd bwd three.': [0.0, 1.0]}
        axis = np.array([1.0, -1.0]) / np.sqrt(2)
        with _mock_embed(d, table):
            fwd_pick, _ = d.key_sentences(cue, [cue], k=1, axis=axis, sign=1.0)
            bwd_pick, _ = d.key_sentences(cue, [cue], k=1, axis=axis, sign=-1.0)
        self.assertEqual(fwd_pick, ['Fwd fwd one.'])
        self.assertEqual(bwd_pick, ['Bwd bwd three.'])


if __name__ == '__main__':
    unittest.main()
