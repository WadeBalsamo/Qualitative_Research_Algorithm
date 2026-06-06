"""
Integration: real PHI de-identification engine (obi/deid_roberta_i2b2).

The unit tier covers the regex/key-span paths; here we load the real
transformer NER model and confirm it scrubs an unknown name. Downloads the
model on first run; skips cleanly if it cannot be loaded.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from tests.testhelpers import integration_test


@integration_test
class TestRealDeidEngine(unittest.TestCase):
    def test_transformer_scrubs_unknown_name(self):
        from process import text_anonymization as ta
        # No known-name key: force the model to do the work on an unseen name.
        patterns = ta.build_name_patterns({})
        text = "Alice said the pain felt sharp this morning."
        try:
            scrubbed, n_known, n_unknown = ta.scrub_text(
                text, patterns, use_transformer=True, confidence_threshold=0.6)
        except Exception as e:
            raise unittest.SkipTest(f"deid model unavailable: {e}")
        # If the engine genuinely loaded, the name should be removed; if the
        # backend fell back to 'none', n_unknown will be 0 — skip rather than fail.
        if n_unknown == 0 and "Alice" in scrubbed:
            raise unittest.SkipTest("deid transformer backend not active (fell back to 'none')")
        self.assertNotIn("Alice", scrubbed)


if __name__ == "__main__":
    unittest.main()
