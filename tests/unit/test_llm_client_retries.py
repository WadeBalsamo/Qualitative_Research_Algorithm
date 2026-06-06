"""
test_llm_client_retries.py
--------------------------
Unit tests for the shared retry loop introduced in
``classification_tools.llm_client.LLMClient._make_request_with_retries``.

These tests are hermetic: ``requests.post`` / ``requests.get`` are
monkeypatched and ``time.sleep`` is patched so no real network calls or
real sleeps occur.  They lock in the behavior-preservation contract:

  * first-try success returns the expected ``(text, metadata)``
  * a transient failure followed by success retries the right number of times
  * all attempts failing exhausts after exactly ``max_retries`` attempts and
    returns ``(None, None)`` WITHOUT extra retries
  * ``time.sleep`` is called with the expected exponential-backoff schedule
  * the "exhausted context window" path returns ``(None, metadata)`` and is
    terminal (no retry)
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from classification_tools.llm_client import LLMClient, LLMClientConfig


def _make_response(payload):
    """Build a fake ``requests`` response whose ``.json()`` returns *payload*."""
    resp = MagicMock()
    resp.json.return_value = payload
    return resp


# An OpenAI-compatible completion that _extract_content reads as a clean answer.
_OK_OPENAI_PAYLOAD = {
    "choices": [
        {
            "message": {"content": '{"answer": "ok"}'},
            "finish_reason": "stop",
        }
    ]
}

# An Ollama-native flat message payload (no choices[]).
_OK_OLLAMA_PAYLOAD = {
    "message": {"content": '{"answer": "ok"}'},
    "done": True,
    "done_reason": "stop",
    "total_duration": 12345,
}


def _make_client(backend, max_retries=3, retry_base_delay=2.0):
    cfg = LLMClientConfig(
        backend=backend,
        api_key="test-key",
        model="test/model",
        temperature=0.0,
        max_retries=max_retries,
        retry_base_delay=retry_base_delay,
    )
    client = LLMClient(cfg)
    # Avoid any network in the context-length fetcher; pin a fixed budget.
    client._get_context_length = lambda: 8192
    return client


class _RetryLoopMixin:
    """Shared scenarios parameterized per backend."""

    backend = None          # 'openrouter' | 'ollama' | 'lmstudio'
    ok_payload = None        # payload whose .json() yields a clean answer
    expected_text = '{"answer": "ok"}'

    def _call(self, client):
        if self.backend == "openrouter":
            return client._openrouter_request("hello")
        if self.backend == "ollama":
            return client._ollama_request("hello")
        if self.backend == "lmstudio":
            return client._lmstudio_request("hello")
        raise AssertionError(f"unknown backend {self.backend}")

    # -- (a) first-try success ----------------------------------------------
    def test_first_try_success(self):
        client = _make_client(self.backend)
        with (
            patch("requests.post", return_value=_make_response(self.ok_payload)) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        self.assertEqual(text, self.expected_text)
        self.assertIsNotNone(metadata)
        self.assertIsInstance(metadata, dict)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()

    # -- (b) transient failure then success ---------------------------------
    def test_transient_failure_then_success(self):
        client = _make_client(self.backend, max_retries=3, retry_base_delay=2.0)
        side_effects = [
            ConnectionError("boom"),
            _make_response(self.ok_payload),
        ]
        with (
            patch("requests.post", side_effect=side_effects) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        self.assertEqual(text, self.expected_text)
        self.assertIsNotNone(metadata)
        # Two attempts: one failed, one succeeded.
        self.assertEqual(post.call_count, 2)
        # Exactly one backoff sleep between the two attempts: 2.0 * 2**0 == 2.0
        sleep.assert_called_once_with(2.0)

    # -- (c) all attempts fail: exhausts after exactly max_retries ----------
    def test_all_attempts_fail_returns_none_none(self):
        client = _make_client(self.backend, max_retries=3, retry_base_delay=2.0)
        with (
            patch("requests.post", side_effect=ConnectionError("down")) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        # All three backends return (None, None) on exhausted retries.
        self.assertIsNone(text)
        self.assertIsNone(metadata)
        # Exactly max_retries attempts, no more.
        self.assertEqual(post.call_count, 3)
        # Sleep happens between attempts only: after attempt 1 and 2, not 3.
        self.assertEqual(sleep.call_count, 2)

    # -- (d) backoff schedule -----------------------------------------------
    def test_backoff_schedule(self):
        client = _make_client(self.backend, max_retries=4, retry_base_delay=2.0)
        with (
            patch("requests.post", side_effect=ConnectionError("down")),
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            self._call(client)

        # retry_base_delay * 2**attempt for attempts 0,1,2 (no sleep after last)
        delays = [c.args[0] for c in sleep.call_args_list]
        self.assertEqual(delays, [2.0, 4.0, 8.0])


class TestOpenRouterRetries(_RetryLoopMixin, unittest.TestCase):
    backend = "openrouter"
    ok_payload = _OK_OPENAI_PAYLOAD

    def test_exhausted_context_returns_none_metadata_terminal(self):
        """finish_reason='length' with no content => (None, metadata), no retry."""
        client = _make_client(self.backend)
        payload = {
            "id": "or-1",
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        }
        with (
            patch("requests.post", return_value=_make_response(payload)) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        self.assertIsNone(text)
        # OpenRouter exhausted-path returns the metadata dict (NOT None).
        self.assertEqual(metadata, payload)
        self.assertEqual(post.call_count, 1)  # terminal: no retry
        sleep.assert_not_called()


class TestOllamaRetries(_RetryLoopMixin, unittest.TestCase):
    backend = "ollama"
    ok_payload = _OK_OLLAMA_PAYLOAD

    def test_success_metadata_is_synthesized(self):
        """Ollama success returns a synthesized metadata dict (not raw data)."""
        client = _make_client(self.backend)
        with (
            patch("requests.post", return_value=_make_response(self.ok_payload)),
            patch("classification_tools.llm_client.time.sleep"),
        ):
            text, metadata = self._call(client)

        self.assertEqual(text, self.expected_text)
        self.assertEqual(metadata["model"], "test/model")
        self.assertTrue(metadata["done"])
        self.assertEqual(metadata["total_duration"], 12345)

    def test_exhausted_context_returns_raw_data_terminal(self):
        """done_reason='length' with no content => (None, data), no retry."""
        client = _make_client(self.backend)
        payload = {
            "message": {"content": ""},
            "done": True,
            "done_reason": "length",
        }
        with (
            patch("requests.post", return_value=_make_response(payload)) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        self.assertIsNone(text)
        # Ollama exhausted-path returns the raw `data` dict.
        self.assertEqual(metadata, payload)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()


class TestLMStudioRetries(_RetryLoopMixin, unittest.TestCase):
    backend = "lmstudio"
    ok_payload = _OK_OPENAI_PAYLOAD

    def test_success_returns_raw_data(self):
        """LM Studio success returns the raw response data as metadata."""
        client = _make_client(self.backend)
        with (
            patch("requests.post", return_value=_make_response(self.ok_payload)),
            patch("classification_tools.llm_client.time.sleep"),
        ):
            text, metadata = self._call(client)

        self.assertEqual(text, self.expected_text)
        self.assertEqual(metadata, self.ok_payload)

    def test_missing_choices_is_retryable(self):
        """A response with no choices raises ValueError, which is retried."""
        client = _make_client(self.backend, max_retries=3, retry_base_delay=2.0)
        bad = {"error": {"message": "model not loaded"}}
        side_effects = [
            _make_response(bad),       # ValueError -> retry
            _make_response(self.ok_payload),
        ]
        with (
            patch("requests.post", side_effect=side_effects) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        self.assertEqual(text, self.expected_text)
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once_with(2.0)

    def test_exhausted_context_returns_raw_data_terminal(self):
        """finish_reason='length' with no content => (None, data), no retry."""
        client = _make_client(self.backend)
        payload = {
            "choices": [{"message": {"content": ""}, "finish_reason": "length"}],
        }
        with (
            patch("requests.post", return_value=_make_response(payload)) as post,
            patch("classification_tools.llm_client.time.sleep") as sleep,
        ):
            text, metadata = self._call(client)

        self.assertIsNone(text)
        self.assertEqual(metadata, payload)
        self.assertEqual(post.call_count, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
