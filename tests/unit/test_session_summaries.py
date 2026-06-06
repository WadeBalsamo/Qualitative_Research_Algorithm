"""
tests/unit/test_session_summaries.py
--------------------------------------
Tests for analysis/reports/session_summaries.py.

generate_session_summaries(df_all, session_ids, output_dir,
                            llm_client=None, max_words=500) -> dict

Covers:
  - Returns dict keyed by session_id
  - Writes session_summaries.json  (03_analysis_data/)
  - Writes session_summaries.txt   (06_reports/03_per_session/)
  - Per-session entries appear in the txt file
  - When llm_client=None, short text used verbatim (no summarization)
  - Monkeypatched LLM client: stub that returns deterministic summary
  - When a session has no therapist segments, summary is '[no therapist segments]'
  - Empty session_ids list → empty dict + files written
  - txt header contains 'SESSION THERAPIST SUMMARIES'
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from process import output_paths as _paths
from analysis.reports.session_summaries import generate_session_summaries


# ── fixtures ──────────────────────────────────────────────────────────────────

def _make_df_all(session_ids, include_therapist=True):
    """Minimal interleaved df with therapist and participant rows."""
    rows = []
    t = 1000
    for sid in session_ids:
        for j in range(6):
            spk = 'therapist' if (j % 2 == 1) else 'participant'
            if not include_therapist and spk == 'therapist':
                continue
            rows.append({
                'segment_id': f'{sid}_{j}',
                'session_id': sid,
                'speaker': spk,
                'text': f'Therapist cue text for {sid} seg {j}' if spk == 'therapist'
                        else f'Participant text {sid} {j}',
                'start_time_ms': t,
                'end_time_ms': t + 800,
                'final_label': (j % 3) if spk == 'participant' else np.nan,
                'purer_primary': (j % 5) if spk == 'therapist' else np.nan,
            })
            t += 1000
    return pd.DataFrame(rows)


class _StubLLMClient:
    """Deterministic stub: returns 'STUB SUMMARY' for any prompt."""
    def __init__(self, response='STUB SUMMARY'):
        self.response = response
        self.call_count = 0

    def request(self, prompt):
        self.call_count += 1
        return self.response, None


# ── test classes ──────────────────────────────────────────────────────────────

class TestGenerateSessionSummariesBasic(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.session_ids = ['c1s1', 'c1s2', 'c1s3']
        self.df_all = _make_df_all(self.session_ids)

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_dict_keyed_by_session(self):
        result = generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        self.assertIsInstance(result, dict)
        for sid in self.session_ids:
            self.assertIn(sid, result)

    def test_json_file_written(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        json_path = os.path.join(_paths.analysis_data_dir(self.tmp), 'session_summaries.json')
        self.assertTrue(os.path.isfile(json_path))

    def test_json_file_valid_and_keyed(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        json_path = os.path.join(_paths.analysis_data_dir(self.tmp), 'session_summaries.json')
        with open(json_path, encoding='utf-8') as f:
            data = json.load(f)
        self.assertIsInstance(data, dict)
        for sid in self.session_ids:
            self.assertIn(sid, data)

    def test_txt_file_written(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        txt_path = os.path.join(_paths.reports_per_session_dir(self.tmp), 'session_summaries.txt')
        self.assertTrue(os.path.isfile(txt_path))

    def test_txt_header_present(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        txt_path = os.path.join(_paths.reports_per_session_dir(self.tmp), 'session_summaries.txt')
        with open(txt_path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('SESSION THERAPIST SUMMARIES', content)

    def test_txt_has_session_blocks(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        txt_path = os.path.join(_paths.reports_per_session_dir(self.tmp), 'session_summaries.txt')
        with open(txt_path, encoding='utf-8') as f:
            content = f.read()
        for sid in self.session_ids:
            self.assertIn(sid, content)

    def test_txt_has_generated_date(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp)
        txt_path = os.path.join(_paths.reports_per_session_dir(self.tmp), 'session_summaries.txt')
        with open(txt_path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('Generated:', content)

    def test_txt_has_max_words_note(self):
        generate_session_summaries(self.df_all, self.session_ids, self.tmp, max_words=300)
        txt_path = os.path.join(_paths.reports_per_session_dir(self.tmp), 'session_summaries.txt')
        with open(txt_path, encoding='utf-8') as f:
            content = f.read()
        self.assertIn('300', content)


class TestNoTherapistSegments(unittest.TestCase):
    def test_no_therapist_returns_sentinel(self):
        session_ids = ['c1s1', 'c1s2']
        df_all = _make_df_all(session_ids, include_therapist=False)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_session_summaries(df_all, session_ids, tmp)
        for sid in session_ids:
            self.assertEqual(result[sid], '[no therapist segments]')

    def test_mixed_presence(self):
        """One session has therapist text, one doesn't."""
        session_ids = ['c1s1', 'c1s2']
        df_only_s1_therapist = pd.concat([
            _make_df_all(['c1s1'], include_therapist=True),
            _make_df_all(['c1s2'], include_therapist=False),
        ], ignore_index=True)
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_session_summaries(df_only_s1_therapist, session_ids, tmp)
        self.assertNotEqual(result['c1s1'], '[no therapist segments]')
        self.assertEqual(result['c1s2'], '[no therapist segments]')


class TestShortTextVerbatim(unittest.TestCase):
    """When therapist text is short and llm_client=None, text is used verbatim."""

    def test_short_text_verbatim_no_client(self):
        session_ids = ['c1s1']
        df = _make_df_all(session_ids)
        # Limit max_words high so summarization is never triggered
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_session_summaries(df, session_ids, tmp, llm_client=None, max_words=999)
        # The text should contain real therapist words, not LLM output
        self.assertIn('Therapist cue text', result['c1s1'])


class TestLLMClientMonkeypatched(unittest.TestCase):
    """Monkeypatched LLM client returns stub; only called when text exceeds max_words."""

    def test_stub_called_when_text_long(self):
        """With a very low max_words, the stub should be called and its response stored."""
        session_ids = ['c1s1']
        df = _make_df_all(session_ids)
        stub = _StubLLMClient(response='STUB SUMMARY OUTPUT')
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_session_summaries(
                df, session_ids, tmp, llm_client=stub, max_words=2
            )
        self.assertEqual(result['c1s1'], 'STUB SUMMARY OUTPUT')
        self.assertGreater(stub.call_count, 0)

    def test_stub_not_called_when_text_short(self):
        """With a high max_words, the stub should not be called at all."""
        session_ids = ['c1s1']
        df = _make_df_all(session_ids)
        stub = _StubLLMClient()
        with tempfile.TemporaryDirectory() as tmp:
            generate_session_summaries(df, session_ids, tmp, llm_client=stub, max_words=9999)
        self.assertEqual(stub.call_count, 0)

    def test_stub_result_in_txt(self):
        """When stub is called, its output appears in the txt file per session."""
        session_ids = ['c1s1', 'c1s2']
        df = _make_df_all(session_ids)
        stub = _StubLLMClient(response='DETERMINISTIC STUB TEXT')
        with tempfile.TemporaryDirectory() as tmp:
            generate_session_summaries(df, session_ids, tmp, llm_client=stub, max_words=2)
            txt_path = os.path.join(_paths.reports_per_session_dir(tmp), 'session_summaries.txt')
            with open(txt_path, encoding='utf-8') as f:
                content = f.read()
        self.assertIn('DETERMINISTIC STUB TEXT', content)


class TestEmptySessionIds(unittest.TestCase):
    def test_empty_session_list(self):
        df = _make_df_all(['c1s1'])
        with tempfile.TemporaryDirectory() as tmp:
            result = generate_session_summaries(df, [], tmp)
        self.assertEqual(result, {})

    def test_empty_session_list_still_writes_files(self):
        df = _make_df_all(['c1s1'])
        with tempfile.TemporaryDirectory() as tmp:
            generate_session_summaries(df, [], tmp)
            json_path = os.path.join(_paths.analysis_data_dir(tmp), 'session_summaries.json')
            txt_path = os.path.join(_paths.reports_per_session_dir(tmp), 'session_summaries.txt')
            self.assertTrue(os.path.isfile(json_path))
            self.assertTrue(os.path.isfile(txt_path))


class TestSortOrder(unittest.TestCase):
    """Summaries follow the provided session_ids order in the txt output."""

    def test_session_order_in_txt(self):
        session_ids = ['c1s3', 'c1s1', 'c1s2']
        df = _make_df_all(session_ids)
        with tempfile.TemporaryDirectory() as tmp:
            generate_session_summaries(df, session_ids, tmp)
            txt_path = os.path.join(_paths.reports_per_session_dir(tmp), 'session_summaries.txt')
            with open(txt_path, encoding='utf-8') as f:
                content = f.read()
        pos = {sid: content.index(sid) for sid in session_ids}
        self.assertLess(pos['c1s3'], pos['c1s1'])
        self.assertLess(pos['c1s1'], pos['c1s2'])


if __name__ == '__main__':
    unittest.main()
