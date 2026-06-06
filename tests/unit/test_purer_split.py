"""
tests/test_purer_split.py
-------------------------
Unit tests for the PURE helpers added in Phase 1b–1f:

  process.cue_blocks.split_by_word_budget
  process.cue_blocks.format_purer_coverage

No LLM, no network, no file-system writes required.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from process.cue_blocks import split_by_word_budget, format_purer_coverage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _item(text: str) -> dict:
    """Minimal dict item for split_by_word_budget tests."""
    return {'text': text}


def _text_of(item) -> str:
    return item['text']


# ---------------------------------------------------------------------------
# split_by_word_budget
# ---------------------------------------------------------------------------

class TestSplitByWordBudget(unittest.TestCase):

    # ── basic splitting ──────────────────────────────────────────────────────

    def test_empty_input_returns_empty_list(self):
        result = split_by_word_budget([], max_words=10, text_of=_text_of)
        self.assertEqual(result, [])

    def test_single_small_item_returns_one_group(self):
        items = [_item('one two three')]  # 3 words
        result = split_by_word_budget(items, max_words=10, text_of=_text_of)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], items)

    def test_all_items_fit_in_one_group(self):
        items = [_item('a b'), _item('c d'), _item('e f')]  # 6 words total
        result = split_by_word_budget(items, max_words=10, text_of=_text_of)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], items)

    def test_items_split_into_two_groups(self):
        # Budget=3: [a b c] fits (3), then [d e f] starts new group.
        items = [_item('a b c'), _item('d e f')]
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [items[0]])
        self.assertEqual(result[1], [items[1]])

    def test_items_packed_greedy_until_budget_exhausted(self):
        # Budget=5: item0(3w)+item1(2w)=5 → fits; item2(3w) → new group.
        items = [_item('a b c'), _item('d e'), _item('f g h')]
        result = split_by_word_budget(items, max_words=5, text_of=_text_of)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], items[:2])
        self.assertEqual(result[1], [items[2]])

    def test_total_items_conserved(self):
        """Flatten(result) must equal input in order and membership."""
        items = [_item(f'word_{i} extra') for i in range(7)]
        result = split_by_word_budget(items, max_words=4, text_of=_text_of)
        flat = [item for grp in result for item in grp]
        self.assertEqual(flat, items)

    def test_order_preserved(self):
        texts = ['alpha', 'beta gamma', 'delta epsilon zeta', 'eta', 'theta iota']
        items = [_item(t) for t in texts]
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        flat = [item for grp in result for item in grp]
        self.assertEqual([i['text'] for i in flat], texts)

    # ── single over-budget item becomes singleton ────────────────────────────

    def test_single_over_budget_item_is_singleton_group(self):
        # The one item has 10 words but budget is 3 → singleton sub-list.
        items = [_item('a b c d e f g h i j')]
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], items)

    def test_over_budget_item_followed_by_small_item(self):
        # item0 = 10 words (over budget=3) → singleton group
        # item1 = 2 words → new group (budget not exceeded)
        items = [_item('a b c d e f g h i j'), _item('x y')]
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [items[0]])
        self.assertEqual(result[1], [items[1]])

    def test_multiple_over_budget_items_each_is_singleton(self):
        items = [_item('a b c d e'), _item('f g h i j'), _item('k l m n o')]
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        self.assertEqual(len(result), 3)
        for grp, orig_item in zip(result, items):
            self.assertEqual(grp, [orig_item])

    # ── budget exactly met ───────────────────────────────────────────────────

    def test_exact_budget_fit_stays_in_one_group(self):
        items = [_item('one'), _item('two'), _item('three')]  # each 1 word, budget=3
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        self.assertEqual(len(result), 1)

    def test_exact_budget_then_one_more_word_splits(self):
        # budget=3: 'a b c' fits exactly, 'd' triggers new group.
        items = [_item('a b c'), _item('d')]
        result = split_by_word_budget(items, max_words=3, text_of=_text_of)
        self.assertEqual(len(result), 2)

    # ── empty / whitespace items ─────────────────────────────────────────────

    def test_empty_text_items_count_as_zero_words(self):
        # All whitespace / empty → 0 words each; should all go into one group
        items = [_item(''), _item('   '), _item('\t')]
        result = split_by_word_budget(items, max_words=1, text_of=_text_of)
        # 0 + 0 + 0 ≤ 1 → greedy packs all into one group
        self.assertEqual(len(result), 1)
        flat = [item for grp in result for item in grp]
        self.assertEqual(flat, items)

    # ── many items ──────────────────────────────────────────────────────────

    def test_many_items_flat_equal_input(self):
        import random
        random.seed(42)
        words = ['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']
        items = [_item(' '.join(random.choices(words, k=random.randint(1, 8))))
                 for _ in range(20)]
        result = split_by_word_budget(items, max_words=10, text_of=_text_of)
        flat = [item for grp in result for item in grp]
        self.assertEqual(flat, items)

    def test_each_group_word_count_respects_budget_unless_singleton(self):
        """Every group with >1 item must have a combined word count ≤ max_words."""
        import random
        random.seed(7)
        words = 'the quick brown fox jumped over lazy dog'.split()
        max_w = 6
        items = [_item(' '.join(random.choices(words, k=random.randint(1, 5))))
                 for _ in range(15)]
        result = split_by_word_budget(items, max_words=max_w, text_of=_text_of)
        for grp in result:
            if len(grp) > 1:
                total = sum(len(_text_of(it).split()) for it in grp)
                self.assertLessEqual(
                    total, max_w,
                    f'Group with {len(grp)} items has {total} words, budget={max_w}',
                )


# ---------------------------------------------------------------------------
# format_purer_coverage
# ---------------------------------------------------------------------------

class TestFormatPurerCoverage(unittest.TestCase):

    def _simple_stats(self, **overrides) -> dict:
        base = {
            'n_blocks': 10,
            'n_skipped_lesson': 2,
            'skipped_lesson_words': 500,
            'n_labeled_segments': 20,
            'labeled_words': 600,
            'n_unparseable': 1,
            'unparseable_words': 50,
            'total_therapist_words': 700,
            'per_session': {},
        }
        base.update(overrides)
        return base

    # ── return type and non-empty ────────────────────────────────────────────

    def test_returns_string(self):
        result = format_purer_coverage(self._simple_stats())
        self.assertIsInstance(result, str)

    def test_non_empty_output(self):
        result = format_purer_coverage(self._simple_stats())
        self.assertGreater(len(result), 50)

    # ── arithmetic: labeled / total percentage ───────────────────────────────

    def test_coverage_percentage_correct(self):
        # labeled_words=600, total_therapist_words=700 → 85.7%
        stats = self._simple_stats(labeled_words=600, total_therapist_words=700)
        result = format_purer_coverage(stats)
        self.assertIn('85.7%', result)

    def test_coverage_100_percent(self):
        stats = self._simple_stats(labeled_words=400, total_therapist_words=400)
        result = format_purer_coverage(stats)
        self.assertIn('100.0%', result)

    def test_coverage_zero_percent(self):
        stats = self._simple_stats(labeled_words=0, total_therapist_words=400)
        result = format_purer_coverage(stats)
        self.assertIn('0.0%', result)

    def test_divide_by_zero_total_returns_na(self):
        # total_therapist_words=0 → coverage should render 'n/a'
        stats = self._simple_stats(labeled_words=0, total_therapist_words=0)
        result = format_purer_coverage(stats)
        self.assertIn('n/a', result)

    def test_missing_total_words_key_treated_as_zero_na(self):
        stats = {
            'n_blocks': 5,
            'labeled_words': 100,
            # 'total_therapist_words' intentionally absent
        }
        result = format_purer_coverage(stats)
        self.assertIn('n/a', result)

    # ── all fields present in output ─────────────────────────────────────────

    def test_skipped_lesson_word_count_appears(self):
        stats = self._simple_stats(n_skipped_lesson=3, skipped_lesson_words=750)
        result = format_purer_coverage(stats)
        self.assertIn('750', result)

    def test_unparseable_count_appears(self):
        stats = self._simple_stats(n_unparseable=4, unparseable_words=88)
        result = format_purer_coverage(stats)
        self.assertIn('88', result)

    def test_n_blocks_appears(self):
        stats = self._simple_stats(n_blocks=17)
        result = format_purer_coverage(stats)
        self.assertIn('17', result)

    # ── per-session breakdown ─────────────────────────────────────────────────

    def test_per_session_data_appears_in_output(self):
        stats = self._simple_stats(per_session={
            'session_01': {
                'n_blocks': 3,
                'labeled_words': 120,
                'total_therapist_words': 150,
            }
        })
        result = format_purer_coverage(stats)
        self.assertIn('session_01', result)

    def test_per_session_coverage_percentage_correct(self):
        # labeled=90, total=150 → 60.0%
        stats = self._simple_stats(per_session={
            'sess_A': {
                'labeled_words': 90,
                'total_therapist_words': 150,
            }
        })
        result = format_purer_coverage(stats)
        self.assertIn('60.0%', result)

    def test_per_session_zero_total_shows_na(self):
        stats = self._simple_stats(per_session={
            'sess_empty': {
                'labeled_words': 0,
                'total_therapist_words': 0,
            }
        })
        result = format_purer_coverage(stats)
        self.assertIn('n/a', result)

    # ── empty stats dict ─────────────────────────────────────────────────────

    def test_empty_stats_does_not_crash(self):
        result = format_purer_coverage({})
        self.assertIsInstance(result, str)
        self.assertIn('n/a', result)

    # ── word accounting invariant ─────────────────────────────────────────────

    def test_labeled_plus_unparseable_plus_skipped_le_total(self):
        """
        labeled_words + unparseable_words + skipped_lesson_words should be
        ≤ total_therapist_words (they are drawn from non-overlapping subsets).
        This is a soft invariant — we verify the report reflects consistent numbers.
        """
        lbl = 300
        unp = 50
        skp = 100
        tot = lbl + unp + skp  # exact partition

        stats = {
            'n_blocks': 10,
            'n_skipped_lesson': 2,
            'skipped_lesson_words': skp,
            'n_labeled_segments': 15,
            'labeled_words': lbl,
            'n_unparseable': 1,
            'unparseable_words': unp,
            'total_therapist_words': tot,
            'per_session': {},
        }
        result = format_purer_coverage(stats)
        # When exactly partitioned, coverage = labeled / total = 300/450 ≈ 66.7%
        self.assertIn('66.7%', result)


if __name__ == '__main__':
    unittest.main()
