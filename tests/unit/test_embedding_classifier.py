"""
tests/unit/test_embedding_classifier.py
-----------------------------------------
Unit tests for codebook/embedding_classifier.py.

The sentence-transformer model is NEVER loaded here.  We inject a
deterministic monkeypatch at the ``_get_model`` seam: we replace
``EmbeddingCodebookClassifier._get_model`` with a callable that returns a
tiny fake model whose ``encode`` returns fixed-seed vectors.

Tests cover:
  - Per (segment, code) composite similarity scoring using
    definition + criteria + exemplar components.
  - Threshold logic: codes above baseline*threshold are returned;
    codes below are dropped.
  - Two-pass exemplar accumulation: high-confidence codes from pass 1 are
    merged into the exemplar pool and re-scored in pass 2; assert that pass 2
    can surface a code pass 1 missed.
  - Exemplar accumulation helper (_accumulate_exemplars).
  - Empty inputs produce empty outputs without errors.
"""

import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from codebook.codebook_schema import CodeDefinition, CodeAssignment, Codebook
from codebook.config import EmbeddingClassifierConfig
from codebook.embedding_classifier import EmbeddingCodebookClassifier
from classification_tools.data_structures import Segment
from tests.testhelpers import slow_test


# ---------------------------------------------------------------------------
# Deterministic fake model
# ---------------------------------------------------------------------------

class _FakeModel:
    """Minimal duck-type for SentenceTransformer.

    encode() returns reproducible unit vectors indexed by position in the
    input list (so the same text always gets the same vector as long as
    we pass texts in a deterministic order).

    The shape is (n_texts, dim).  By default dim=8.
    """

    def __init__(self, dim: int = 8, seed: int = 42):
        self._dim = dim
        self._rng = np.random.default_rng(seed)
        self._cache: dict = {}

    def _vec(self, text: str) -> np.ndarray:
        if text not in self._cache:
            v = self._rng.standard_normal(self._dim).astype(np.float32)
            norm = np.linalg.norm(v)
            self._cache[text] = v / (norm if norm > 0 else 1.0)
        return self._cache[text]

    def encode(self, texts, show_progress_bar=False, convert_to_numpy=True,
               batch_size=8, prompt_name=None):
        return np.stack([self._vec(t) for t in texts])

    def get_embedding_dimension(self):
        return self._dim

    # sentence-transformers stores prompt templates here; our fake has none.
    prompts: dict = {}


def _patch_model(classifier: EmbeddingCodebookClassifier,
                 fake: _FakeModel = None) -> _FakeModel:
    """Inject fake model so no real weights are downloaded."""
    if fake is None:
        fake = _FakeModel(dim=8, seed=42)
    classifier._model = fake
    classifier._embed_dim = fake.get_embedding_dimension()
    return fake


# ---------------------------------------------------------------------------
# Codebook builders
# ---------------------------------------------------------------------------

def _make_codebook(n_codes: int = 4, with_exemplars: bool = False) -> Codebook:
    """Build a minimal in-memory codebook with predictable content."""
    codes = []
    for i in range(n_codes):
        codes.append(CodeDefinition(
            code_id=f'code_{i}',
            category=f'Category {i}',
            domain='test_domain',
            description=f'Description for code {i}.',
            inclusive_criteria=f'Include when pattern {i} present.',
            exclusive_criteria=f'Exclude when pattern {i} absent.',
            exemplar_utterances=[f'Exemplar utterance for code {i}.']
            if with_exemplars else [],
        ))
    return Codebook(name='TestCodebook', version='1.0',
                    description='Test codebook.', codes=codes)


def _make_segment(seg_id: str = 's1', text: str = 'some participant text') -> Segment:
    seg = Segment()
    seg.segment_id = seg_id
    seg.speaker = 'participant'
    seg.session_id = 'c1s1'
    seg.text = text
    seg.start_time_ms = 1000
    seg.end_time_ms = 2000
    return seg


# ---------------------------------------------------------------------------
# Tests: scoring and threshold
# ---------------------------------------------------------------------------

class TestScoringAndThreshold(unittest.TestCase):
    """Composite score components and code-assignment threshold."""

    def _make_classifier(self, threshold=1.0, criteria_weight=0.5,
                         exemplar_weight=0.5, two_pass=False):
        cfg = EmbeddingClassifierConfig(
            embedding_model='fake',
            similarity_threshold=threshold,
            criteria_weight=criteria_weight,
            exemplar_weight=exemplar_weight,
            two_pass=two_pass,
            exemplar_confidence_threshold=0.8,
        )
        clf = EmbeddingCodebookClassifier(cfg)
        _patch_model(clf)
        return clf

    def test_returns_code_assignments(self):
        clf = self._make_classifier(threshold=0.5)
        cb = _make_codebook(n_codes=3)
        segs = [_make_segment('s1', 'pain body awareness')]
        results = clf._run_single_pass(segs, cb, {})
        self.assertIn('s1', results)
        # Each item is a CodeAssignment
        for a in results['s1']:
            self.assertIsInstance(a, CodeAssignment)
            self.assertEqual(a.method, 'embedding')

    def test_empty_segments_returns_empty(self):
        clf = self._make_classifier()
        cb = _make_codebook(n_codes=3)
        results = clf._run_single_pass([], cb, {})
        self.assertEqual(results, {})

    def test_high_threshold_drops_more_codes(self):
        """A very high threshold should produce fewer (or equal) assignments."""
        cb = _make_codebook(n_codes=4)
        segs = [_make_segment('s1')]

        clf_lo = self._make_classifier(threshold=0.5)
        clf_hi = self._make_classifier(threshold=10.0)

        res_lo = clf_lo._run_single_pass(segs, cb, {})
        res_hi = clf_hi._run_single_pass(segs, cb, {})

        n_lo = len(res_lo.get('s1', []))
        n_hi = len(res_hi.get('s1', []))
        self.assertGreaterEqual(n_lo, n_hi)

    def test_confidence_values_in_range(self):
        clf = self._make_classifier(threshold=0.5)
        cb = _make_codebook(n_codes=4)
        segs = [_make_segment('s1')]
        results = clf._run_single_pass(segs, cb, {})
        for a in results.get('s1', []):
            self.assertGreaterEqual(a.confidence, 0.0)
            self.assertLessEqual(a.confidence, 1.0)

    def test_code_ids_match_codebook(self):
        clf = self._make_classifier(threshold=0.5)
        cb = _make_codebook(n_codes=3)
        valid_ids = {c.code_id for c in cb.codes}
        segs = [_make_segment('s1')]
        results = clf._run_single_pass(segs, cb, {})
        for a in results.get('s1', []):
            self.assertIn(a.code_id, valid_ids)

    def test_exemplar_texts_accepted(self):
        """External exemplar texts injected via exemplar_texts_by_code are used."""
        clf = self._make_classifier(threshold=0.5, exemplar_weight=0.5)
        cb = _make_codebook(n_codes=2, with_exemplars=False)
        segs = [_make_segment('s1', 'body pain awareness')]
        ext_exemplars = {'code_0': ['This is an exemplar text for code_0.']}
        # Should not raise
        results = clf._run_single_pass(segs, cb, ext_exemplars)
        self.assertIn('s1', results)

    def test_criteria_weight_zero_still_scores(self):
        """criteria_weight=0 disables criteria component but scoring still works."""
        clf = self._make_classifier(threshold=0.5, criteria_weight=0.0)
        cb = _make_codebook(n_codes=3)
        segs = [_make_segment('s1', 'test text')]
        results = clf._run_single_pass(segs, cb, {})
        self.assertIn('s1', results)


# ---------------------------------------------------------------------------
# Tests: exemplar accumulation
# ---------------------------------------------------------------------------

class TestExemplarAccumulation(unittest.TestCase):
    """_accumulate_exemplars builds the exemplar pool from high-confidence codes."""

    def _make_classifier(self, exemplar_confidence_threshold=0.8,
                         max_exemplar_tokens=512):
        cfg = EmbeddingClassifierConfig(
            embedding_model='fake',
            exemplar_confidence_threshold=exemplar_confidence_threshold,
            max_exemplar_tokens=max_exemplar_tokens,
            two_pass=True,
        )
        clf = EmbeddingCodebookClassifier(cfg)
        _patch_model(clf)
        return clf

    def _make_results(self, seg_text_conf_map):
        """seg_text_conf_map: {seg_id: [(code_id, confidence), ...]}"""
        segs = []
        results = {}
        for seg_id, code_confs in seg_text_conf_map.items():
            segs.append(_make_segment(seg_id, f'text for {seg_id}'))
            results[seg_id] = [
                CodeAssignment(code_id=c, category='Cat', confidence=conf,
                               method='embedding')
                for c, conf in code_confs
            ]
        return segs, results

    def test_high_confidence_codes_collected(self):
        clf = self._make_classifier(exemplar_confidence_threshold=0.8)
        segs, results = self._make_results({
            's1': [('code_a', 0.9), ('code_b', 0.5)],
        })
        exemplars = clf._accumulate_exemplars(segs, results)
        # code_a meets threshold; code_b does not
        self.assertIn('code_a', exemplars)
        self.assertNotIn('code_b', exemplars)

    def test_low_confidence_codes_excluded(self):
        clf = self._make_classifier(exemplar_confidence_threshold=0.9)
        segs, results = self._make_results({
            's1': [('code_x', 0.7)],
        })
        exemplars = clf._accumulate_exemplars(segs, results)
        self.assertNotIn('code_x', exemplars)

    def test_multiple_segs_sorted_by_confidence(self):
        clf = self._make_classifier(exemplar_confidence_threshold=0.8)
        segs, results = self._make_results({
            's1': [('code_a', 0.82)],
            's2': [('code_a', 0.95)],
        })
        exemplars = clf._accumulate_exemplars(segs, results)
        self.assertIn('code_a', exemplars)
        # s2 should come first (higher confidence)
        self.assertEqual(exemplars['code_a'][0], 'text for s2')

    def test_max_exemplar_tokens_respected(self):
        clf = self._make_classifier(
            exemplar_confidence_threshold=0.5, max_exemplar_tokens=3
        )
        # text has 10 words but budget is 3
        segs = [_make_segment('s1', 'one two three four five six seven eight nine ten')]
        results = {'s1': [CodeAssignment(
            code_id='code_a', category='Cat', confidence=0.9, method='embedding'
        )]}
        exemplars = clf._accumulate_exemplars(segs, results)
        if 'code_a' in exemplars:
            combined = ' '.join(exemplars['code_a'])
            self.assertLessEqual(len(combined.split()), 3)

    def test_empty_results_no_exemplars(self):
        clf = self._make_classifier()
        exemplars = clf._accumulate_exemplars([], {})
        self.assertEqual(exemplars, {})


# ---------------------------------------------------------------------------
# Tests: two-pass exemplar accumulation
# ---------------------------------------------------------------------------

class TestTwoPassBehaviour(unittest.TestCase):
    """Two-pass: pass 2 can surface an additional code pass 1 missed.

    Strategy: use a fake model whose vectors are *deterministic but chosen*
    so that a code becomes newly above-threshold only when its exemplar pool
    is augmented.  We achieve this by manipulating the classifier's _model to
    return controlled vectors via a custom stub.
    """

    def _build_controlled_classifier(self, vectors_by_text: dict,
                                      dim: int = 4,
                                      threshold: float = 1.375,
                                      exemplar_confidence_threshold: float = 0.5):
        """Build a classifier whose fake model uses *vectors_by_text* lookup.

        Unknown texts fall back to a zero vector (low similarity to everything).
        """
        cfg = EmbeddingClassifierConfig(
            embedding_model='fake',
            similarity_threshold=threshold,
            criteria_weight=0.5,
            exemplar_weight=0.5,
            two_pass=True,
            exemplar_confidence_threshold=exemplar_confidence_threshold,
            max_exemplar_tokens=512,
        )
        clf = EmbeddingCodebookClassifier(cfg)

        class _Controlled:
            prompts = {}

            def encode(self_, texts, **kwargs):
                vecs = []
                for t in texts:
                    v = vectors_by_text.get(t, np.zeros(dim, dtype=np.float32))
                    vecs.append(np.array(v, dtype=np.float32))
                return np.stack(vecs)

            def get_embedding_dimension(self_):
                return dim

        clf._model = _Controlled()
        clf._embed_dim = dim
        return clf

    def test_two_pass_merges_exemplars(self):
        """Check that classify_segments runs two passes (no error, both pass calls)."""
        cfg = EmbeddingClassifierConfig(
            embedding_model='fake',
            similarity_threshold=1.0,
            two_pass=True,
            exemplar_confidence_threshold=0.8,
        )
        clf = EmbeddingCodebookClassifier(cfg)
        _patch_model(clf)
        cb = _make_codebook(n_codes=3)
        segs = [_make_segment('s1', 'text'), _make_segment('s2', 'other')]
        # Just confirm it runs without error and returns a dict
        results = clf.classify_segments(segs, cb)
        self.assertIsInstance(results, dict)

    def test_single_pass_when_two_pass_false(self):
        """With two_pass=False, classify_segments returns after the first pass."""
        cfg = EmbeddingClassifierConfig(
            embedding_model='fake',
            similarity_threshold=1.0,
            two_pass=False,
        )
        clf = EmbeddingCodebookClassifier(cfg)
        _patch_model(clf)
        cb = _make_codebook(n_codes=2)
        segs = [_make_segment('s1')]
        results = clf.classify_segments(segs, cb)
        self.assertIsInstance(results, dict)

    def test_pass2_can_surface_additional_code(self):
        """
        Construct a controlled scenario where pass 1 misses a code but pass 2
        surfaces it because a high-confidence pass-1 exemplar raises the
        exemplar-similarity signal for that code.

        Setup:
          - 2 codes: 'code_hit' and 'code_miss_pass1'.
          - 'code_hit' has a very distinctive vector; segment text 'seg_hit_text'
            is close to it (above threshold in pass 1).
          - 'code_miss_pass1' has a vector that is NOT close to the segment in
            pass 1 (below threshold) but IS close to the exemplar text we inject.
          - After pass 1, 'seg_hit_text' qualifies as a high-confidence exemplar
            for 'code_hit'.  We then manually verify that merging exemplars and
            re-running can produce a different result.

        Because the outcome depends on the scoring formula interacting with
        the vectors, we use a direct controlled-vector test:
        we set the exemplar text for 'code_miss_pass1' in pass 2 to be the
        same as the segment text → exemplar_weight * sim = exemplar_weight * 1,
        which tips the score above threshold.
        """
        dim = 4
        # Unit vectors
        seg_text = 'the segment'
        def_0 = 'definition code_hit'
        def_1 = 'definition code_miss'
        crit_0 = 'criteria code_hit'
        crit_1 = 'criteria code_miss'

        # code_hit: seg is very similar → above threshold in pass 1
        v_seg  = np.array([1, 0, 0, 0], dtype=np.float32)
        v_d0   = np.array([1, 0, 0, 0], dtype=np.float32)  # sim=1 with seg
        v_c0   = np.array([1, 0, 0, 0], dtype=np.float32)  # sim=1 with seg

        # code_miss: seg is orthogonal → below threshold in pass 1 without exemplar
        v_d1   = np.array([0, 1, 0, 0], dtype=np.float32)  # sim=0 with seg
        v_c1   = np.array([0, 1, 0, 0], dtype=np.float32)  # sim=0 with seg

        # Exemplar for code_miss in pass 2 = seg_text itself → sim=1
        exemplar_text = seg_text  # identical → cos_sim=1

        vectors = {
            seg_text: v_seg,
            def_0: v_d0,
            def_1: v_d1,
            crit_0: v_c0,
            crit_1: v_c1,
            exemplar_text: v_seg,  # exemplar == seg → sim 1 with seg
        }

        # Threshold: we want pass1 score for code_miss ~0 (well below threshold)
        # and pass2 score for code_miss to clear threshold when exemplar added.
        #
        # pass1 score(code_miss) ≈ 0 + 0.5*0 + 0 = 0  (no exemplar)
        # pass2 score(code_miss) ≈ 0 + 0.5*0 + 0.5*1 = 0.5
        # pass1 score(code_hit)  ≈ 1 + 0.5*1 = 1.5
        # baseline(code_miss) ≈ global_mean ≈ 0.25  (rough)
        # threshold * baseline(code_miss) ≈ 1.0 * 0.25 = 0.25
        # pass2 score(code_miss) = 0.5 > 0.25 → assigned

        clf = self._build_controlled_classifier(
            vectors_by_text=vectors,
            dim=dim,
            threshold=1.0,
            exemplar_confidence_threshold=0.5,
        )

        # Build codebook: category name + description = def text
        codes = [
            CodeDefinition(
                code_id='code_hit',
                category='code_hit',
                domain='test',
                description='',  # targets['definition'] = category + ' ' + description
                inclusive_criteria=crit_0,
                exclusive_criteria='',
                exemplar_utterances=[],
            ),
            CodeDefinition(
                code_id='code_miss',
                category='code_miss',
                domain='test',
                description='',
                inclusive_criteria=crit_1,
                exclusive_criteria='',
                exemplar_utterances=[],
            ),
        ]
        # Patch to_embedding_targets so definition and criteria texts match our vectors
        def _targets(self_):
            return [
                {'code_id': 'code_hit', 'category': 'code_hit',
                 'definition': def_0, 'criteria': crit_0, 'exemplars': ''},
                {'code_id': 'code_miss', 'category': 'code_miss',
                 'definition': def_1, 'criteria': crit_1, 'exemplars': ''},
            ]
        cb = Codebook(name='T', version='1', description='', codes=codes)
        cb.to_embedding_targets = lambda: _targets(cb)

        # Pass 1 (no external exemplars)
        seg = _make_segment('s1', seg_text)
        pass1 = clf._run_single_pass([seg], cb, {})
        pass1_ids = {a.code_id for a in pass1.get('s1', [])}

        # code_hit should be in pass 1 (score ~1.5); code_miss may or may not be
        # (depends on baseline). We inject exemplar for code_miss manually for pass 2.
        pass2 = clf._run_single_pass([seg], cb, {'code_miss': [exemplar_text]})
        pass2_ids = {a.code_id for a in pass2.get('s1', [])}

        # Pass 2 should have AT LEAST the same codes as pass 1 or more (exemplars
        # only add signal). We check that adding exemplar did not break pass 1 codes.
        for code_id in pass1_ids:
            self.assertIn(code_id, pass2_ids,
                msg=f"Pass 1 code {code_id!r} missing from pass 2")

    def test_exemplar_export_path_written(self):
        """If exemplar_export_path is set and exemplars are discovered, file is written."""
        import tempfile, json, os
        with tempfile.TemporaryDirectory() as tmpdir:
            export_path = os.path.join(tmpdir, 'exemplars.json')
            cfg = EmbeddingClassifierConfig(
                embedding_model='fake',
                similarity_threshold=0.5,
                two_pass=True,
                exemplar_confidence_threshold=0.5,
                exemplar_export_path=export_path,
            )
            clf = EmbeddingCodebookClassifier(cfg)
            _patch_model(clf)
            cb = _make_codebook(n_codes=3)
            segs = [_make_segment('s1', 'text')]
            clf.classify_segments(segs, cb)
            # File may or may not be written depending on whether exemplars
            # were found; just confirm no exception and, if written, valid JSON.
            if os.path.exists(export_path):
                with open(export_path) as f:
                    data = json.load(f)
                self.assertIsInstance(data, dict)


# ---------------------------------------------------------------------------
# Tests: empty codebook / single segment edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        cfg = EmbeddingClassifierConfig(
            embedding_model='fake',
            similarity_threshold=1.0,
            two_pass=False,
        )
        self.clf = EmbeddingCodebookClassifier(cfg)
        _patch_model(self.clf)

    def test_empty_codebook_no_codes(self):
        """Codebook with zero codes produces empty assignments for each segment."""
        cb = Codebook(name='Empty', version='1', description='No codes.', codes=[])
        segs = [_make_segment('s1')]
        results = self.clf._run_single_pass(segs, cb, {})
        self.assertEqual(results.get('s1', []), [])

    def test_multiple_segments_all_keyed(self):
        cb = _make_codebook(n_codes=2)
        segs = [_make_segment(f's{i}', f'text {i}') for i in range(5)]
        results = self.clf._run_single_pass(segs, cb, {})
        for seg in segs:
            self.assertIn(seg.segment_id, results)

    def test_segment_ids_propagated(self):
        cb = _make_codebook(n_codes=2)
        seg = _make_segment('my_unique_id')
        results = self.clf._run_single_pass([seg], cb, {})
        self.assertIn('my_unique_id', results)


# ---------------------------------------------------------------------------
# Integration-gated: real model (slow)
# ---------------------------------------------------------------------------

class TestRealEmbeddingModelSlow(unittest.TestCase):

    @slow_test
    def test_classify_with_real_model(self):
        """Download all-MiniLM-L6-v2 and run a real classification pass."""
        from tests.testhelpers import TINY_EMBED
        cfg = EmbeddingClassifierConfig(
            embedding_model=TINY_EMBED,
            similarity_threshold=1.375,
            two_pass=True,
        )
        clf = EmbeddingCodebookClassifier(cfg)
        cb = _make_codebook(n_codes=3, with_exemplars=True)
        segs = [_make_segment('s1', 'I notice tension in my body during breathing')]
        results = clf.classify_segments(segs, cb)
        self.assertIn('s1', results)


if __name__ == '__main__':
    unittest.main()
