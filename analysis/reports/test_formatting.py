import unittest

from ._formatting import _summarize_cue, _summarize_participant_text, _wrap_quote


class TestFormattingHelpers(unittest.TestCase):
    def test_wrap_quote_preserves_full_text_and_newlines(self):
        text = (
            "This is a long quote that should be wrapped across multiple lines "
            "without truncation.\nIt also contains an explicit newline that should "
            "remain present in the output."
        )
        wrapped = _wrap_quote(text, indent=4, max_width=40)
        self.assertIn('This is a long quote', wrapped)
        self.assertIn('explicit', wrapped)
        self.assertIn('newline', wrapped)
        self.assertNotIn('...', wrapped)
        self.assertTrue(wrapped.startswith('    "'))
        self.assertIn('\n    "It also contains', wrapped)

    def test_summarize_cue_returns_full_text_when_no_llm(self):
        text = ' '.join(str(i) for i in range(100))
        result, was_summarized = _summarize_cue(text, llm_client=None, max_words=10)
        self.assertEqual(result, text)
        self.assertFalse(was_summarized)

    def test_summarize_participant_text_returns_full_text_when_no_llm(self):
        text = ' '.join(str(i) for i in range(100))
        result, was_summarized = _summarize_participant_text(text, llm_client=None, max_words=10)
        self.assertEqual(result, text)
        self.assertFalse(was_summarized)


if __name__ == '__main__':
    unittest.main()
