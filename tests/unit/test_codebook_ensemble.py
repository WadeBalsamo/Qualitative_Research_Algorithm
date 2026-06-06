"""
tests/unit/test_codebook_ensemble.py
-------------------------------------
Unit tests for codebook/ensemble.py — reconciliation of embedding + LLM
label sets.

Methodology §4.4 specifies:
  - UNION (preferred_method='both') is the production behaviour: codes from
    either method above threshold are retained.
  - INTERSECTION (require_agreement=True) is reconstructable.
  - Composite confidence: embedding 0.6 / LLM 0.4  NOTE: the production
    ensemble does NOT blend confidences — it prefers LLM metadata where
    present and falls back to embedding.  These tests verify the actual
    source-code behaviour (LLM-preferred, embedding as fallback) rather than
    an assumed weighted blend.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from codebook.codebook_schema import CodeAssignment
from codebook.config import EnsembleConfig
from codebook.ensemble import CodebookEnsemble, EnsembleResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ca(code_id, confidence=0.9, method='embedding', category=None):
    return CodeAssignment(
        code_id=code_id,
        category=category or code_id.upper(),
        confidence=confidence,
        justification='test',
        method=method,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsembleUnionMode(unittest.TestCase):
    """UNION (preferred_method='both') — the production default."""

    def setUp(self):
        self.cfg = EnsembleConfig(
            require_agreement=False,
            flag_disagreements=True,
            preferred_method='both',
        )
        self.ens = CodebookEnsemble(self.cfg)

    def _reconcile(self, emb_codes, llm_codes):
        emb = {'seg1': [_ca(c, method='embedding') for c in emb_codes]}
        llm = {'seg1': [_ca(c, method='llm')       for c in llm_codes]}
        return self.ens.reconcile(emb, llm)['seg1']

    def test_union_contains_all_codes(self):
        r = self._reconcile(['body_scan', 'breath'], ['breath', 'pain_reframe'])
        self.assertEqual(sorted(r.final_codes), ['body_scan', 'breath', 'pain_reframe'])

    def test_agreed_codes_tracked(self):
        r = self._reconcile(['a', 'b'], ['b', 'c'])
        self.assertIn('b', r.agreed_codes)
        self.assertNotIn('a', r.agreed_codes)
        self.assertNotIn('c', r.agreed_codes)

    def test_embedding_only_tracked(self):
        r = self._reconcile(['emb_only', 'shared'], ['shared'])
        self.assertIn('emb_only', r.embedding_only_codes)
        self.assertNotIn('emb_only', r.llm_only_codes)

    def test_llm_only_tracked(self):
        r = self._reconcile(['shared'], ['shared', 'llm_only'])
        self.assertIn('llm_only', r.llm_only_codes)
        self.assertNotIn('llm_only', r.embedding_only_codes)

    def test_disagreement_flagged(self):
        r = self._reconcile(['a'], ['b'])
        self.assertTrue(r.needs_human_review)

    def test_no_disagreement_when_agreed(self):
        r = self._reconcile(['a'], ['a'])
        # Both methods agree → no disagreement
        self.assertFalse(r.needs_human_review)
        self.assertEqual(r.embedding_only_codes, [])
        self.assertEqual(r.llm_only_codes, [])

    def test_final_assignments_prefer_llm_metadata(self):
        """When a code appears in both, ensemble assignment carries LLM confidence."""
        emb = {'s1': [_ca('code_x', confidence=0.5, method='embedding')]}
        llm = {'s1': [_ca('code_x', confidence=0.95, method='llm')]}
        r = self.ens.reconcile(emb, llm)['s1']
        assignment = next(a for a in r.final_assignments if a.code_id == 'code_x')
        # LLM metadata preferred → confidence is LLM's 0.95
        self.assertAlmostEqual(assignment.confidence, 0.95)
        self.assertEqual(assignment.method, 'ensemble')

    def test_embedding_only_falls_back_to_embedding_metadata(self):
        """Codes from embedding only fall back to embedding confidence."""
        emb = {'s1': [_ca('emb_code', confidence=0.7, method='embedding')]}
        llm = {'s1': []}
        r = self.ens.reconcile(emb, llm)['s1']
        assignment = next(a for a in r.final_assignments if a.code_id == 'emb_code')
        self.assertAlmostEqual(assignment.confidence, 0.7)
        self.assertEqual(assignment.method, 'ensemble')


class TestEnsembleIntersectionMode(unittest.TestCase):
    """INTERSECTION (require_agreement=True) — must appear in both to be final."""

    def setUp(self):
        cfg = EnsembleConfig(
            require_agreement=True,
            flag_disagreements=True,
            preferred_method='both',  # irrelevant when require_agreement=True
        )
        self.ens = CodebookEnsemble(cfg)

    def test_intersection_only_agreed(self):
        emb = {'s1': [_ca('shared'), _ca('emb_only')]}
        llm = {'s1': [_ca('shared'), _ca('llm_only')]}
        r = self.ens.reconcile(emb, llm)['s1']
        self.assertEqual(r.final_codes, ['shared'])

    def test_intersection_empty_when_no_overlap(self):
        emb = {'s1': [_ca('a')]}
        llm = {'s1': [_ca('b')]}
        r = self.ens.reconcile(emb, llm)['s1']
        self.assertEqual(r.final_codes, [])
        self.assertTrue(r.needs_human_review)


class TestEnsembleLLMPreferredMode(unittest.TestCase):
    """preferred_method='llm' — final set is exactly the LLM codes."""

    def setUp(self):
        cfg = EnsembleConfig(
            require_agreement=False,
            flag_disagreements=False,
            preferred_method='llm',
        )
        self.ens = CodebookEnsemble(cfg)

    def test_final_is_llm_only_set(self):
        emb = {'s1': [_ca('emb_a'), _ca('shared')]}
        llm = {'s1': [_ca('llm_a'), _ca('shared')]}
        r = self.ens.reconcile(emb, llm)['s1']
        self.assertEqual(sorted(r.final_codes), ['llm_a', 'shared'])


class TestEnsembleEmbeddingPreferredMode(unittest.TestCase):
    """preferred_method='embedding' — final set is exactly the embedding codes."""

    def setUp(self):
        cfg = EnsembleConfig(
            require_agreement=False,
            flag_disagreements=False,
            preferred_method='embedding',
        )
        self.ens = CodebookEnsemble(cfg)

    def test_final_is_embedding_only_set(self):
        emb = {'s1': [_ca('emb_a'), _ca('shared')]}
        llm = {'s1': [_ca('llm_a'), _ca('shared')]}
        r = self.ens.reconcile(emb, llm)['s1']
        self.assertEqual(sorted(r.final_codes), ['emb_a', 'shared'])


class TestEnsembleEmptyInputs(unittest.TestCase):
    """Degenerate / edge cases: empty result dicts and empty assignment lists."""

    def setUp(self):
        self.ens = CodebookEnsemble(EnsembleConfig(preferred_method='both'))

    def test_empty_both(self):
        results = self.ens.reconcile({}, {})
        self.assertEqual(results, {})

    def test_segment_in_only_embedding(self):
        emb = {'s1': [_ca('only_emb')]}
        r = self.ens.reconcile(emb, {})['s1']
        self.assertIn('only_emb', r.final_codes)
        self.assertIn('only_emb', r.embedding_only_codes)

    def test_segment_in_only_llm(self):
        llm = {'s1': [_ca('only_llm', method='llm')]}
        r = self.ens.reconcile({}, llm)['s1']
        self.assertIn('only_llm', r.final_codes)
        self.assertIn('only_llm', r.llm_only_codes)

    def test_empty_code_lists_no_review(self):
        emb = {'s1': []}
        llm = {'s1': []}
        r = self.ens.reconcile(emb, llm)['s1']
        self.assertEqual(r.final_codes, [])
        self.assertFalse(r.needs_human_review)

    def test_result_is_ensemble_result_instance(self):
        r = self.ens.reconcile({'s1': []}, {'s1': []})
        self.assertIsInstance(r['s1'], EnsembleResult)


class TestEnsembleDisagreementDetails(unittest.TestCase):
    """Disagreement detail records are populated correctly."""

    def setUp(self):
        self.ens = CodebookEnsemble(EnsembleConfig(
            preferred_method='both',
            flag_disagreements=True,
        ))

    def test_disagreement_details_types(self):
        emb = {'s1': [_ca('emb_x', confidence=0.6, method='embedding', category='EMB_X')]}
        llm = {'s1': [_ca('llm_y', confidence=0.8, method='llm', category='LLM_Y')]}
        r = self.ens.reconcile(emb, llm)['s1']
        types = {d['type'] for d in r.disagreement_details}
        self.assertIn('embedding_only', types)
        self.assertIn('llm_only', types)

    def test_embedding_only_detail_has_embedding_confidence(self):
        emb = {'s1': [_ca('emb_x', confidence=0.6, method='embedding')]}
        llm = {'s1': []}
        r = self.ens.reconcile(emb, llm)['s1']
        detail = next(d for d in r.disagreement_details if d['code_id'] == 'emb_x')
        self.assertIn('embedding_confidence', detail)
        self.assertAlmostEqual(detail['embedding_confidence'], 0.6)

    def test_llm_only_detail_has_llm_confidence(self):
        emb = {'s1': []}
        llm = {'s1': [_ca('llm_y', confidence=0.85, method='llm')]}
        r = self.ens.reconcile(emb, llm)['s1']
        detail = next(d for d in r.disagreement_details if d['code_id'] == 'llm_y')
        self.assertIn('llm_confidence', detail)
        self.assertAlmostEqual(detail['llm_confidence'], 0.85)

    def test_agreed_codes_not_in_disagreement_details(self):
        emb = {'s1': [_ca('shared')]}
        llm = {'s1': [_ca('shared', method='llm')]}
        r = self.ens.reconcile(emb, llm)['s1']
        ids = [d['code_id'] for d in r.disagreement_details]
        self.assertNotIn('shared', ids)

    def test_flag_disagreements_false_suppresses_review(self):
        cfg = EnsembleConfig(preferred_method='both', flag_disagreements=False)
        ens = CodebookEnsemble(cfg)
        emb = {'s1': [_ca('a')]}
        llm = {'s1': [_ca('b', method='llm')]}
        r = ens.reconcile(emb, llm)['s1']
        # No flagging even though there is genuine disagreement
        self.assertFalse(r.needs_human_review)


class TestEnsembleMultipleSegments(unittest.TestCase):
    """Union of segment IDs from both result dicts is covered."""

    def setUp(self):
        self.ens = CodebookEnsemble(EnsembleConfig(preferred_method='both'))

    def test_all_segment_ids_covered(self):
        emb = {
            's1': [_ca('x')],
            's2': [_ca('y')],
        }
        llm = {
            's2': [_ca('y', method='llm')],
            's3': [_ca('z', method='llm')],
        }
        results = self.ens.reconcile(emb, llm)
        self.assertIn('s1', results)
        self.assertIn('s2', results)
        self.assertIn('s3', results)

    def test_segment_ids_match_in_result(self):
        emb = {'seg_alpha': [_ca('a')]}
        llm = {'seg_beta': [_ca('b', method='llm')]}
        results = self.ens.reconcile(emb, llm)
        self.assertEqual(results['seg_alpha'].segment_id, 'seg_alpha')
        self.assertEqual(results['seg_beta'].segment_id, 'seg_beta')


class TestEnsembleDefaultConfig(unittest.TestCase):
    """EnsembleConfig defaults produce the expected production behaviour."""

    def test_default_preferred_method(self):
        cfg = EnsembleConfig()
        # Default is 'llm' per config.py — final set = LLM codes
        ens = CodebookEnsemble(cfg)
        emb = {'s1': [_ca('emb_only')]}
        llm = {'s1': [_ca('llm_code', method='llm')]}
        r = ens.reconcile(emb, llm)['s1']
        # With preferred_method='llm' and require_agreement=False:
        # final = LLM set only
        self.assertIn('llm_code', r.final_codes)
        self.assertNotIn('emb_only', r.final_codes)

    def test_default_flag_disagreements_true(self):
        cfg = EnsembleConfig()
        self.assertTrue(cfg.flag_disagreements)

    def test_default_require_agreement_false(self):
        cfg = EnsembleConfig()
        self.assertFalse(cfg.require_agreement)


if __name__ == '__main__':
    unittest.main()
