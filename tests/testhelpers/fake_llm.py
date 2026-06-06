"""
tests.testhelpers.fake_llm
--------------------------
Deterministic, offline stand-in for ``classification_tools.llm_client.LLMClient``.

Mirrors the public surface the classification code relies on:

    text, metadata = client.request(prompt)
    results = client.multi_model_request(prompt, models=[...])
    client.check_loaded_model(name) -> True

``request`` returns a ``(json_text, metadata)`` tuple exactly like the real
client. The default responder inspects the prompt for any known VAAMR/PURER
construct name and echoes the first match back as ``primary_stage`` so the
real parser (which maps names -> ids via the framework) yields a concrete,
deterministic label with no network and no model download.

Pass a custom ``responder(prompt) -> dict`` to drive specific behaviours
(malformed JSON, abstentions, codebook coding, truncation, ...).
"""
from __future__ import annotations

import json
from typing import Callable, Dict, List, Optional, Tuple

# Canonical construct names for both frameworks (lowercased keys also matched).
# Order matters: the first name found in a prompt is the one echoed back.
_VAAMR_NAMES = [
    "Vigilance", "Avoidance", "Attention Regulation", "Metacognition", "Reappraisal",
]
_PURER_NAMES = [
    "Phenomenological", "Utilization", "Reframing", "Educate/Expectancy",
    "Education", "Reinforcement",
]


def _default_responder(prompt: str, default_name: str) -> Dict:
    """Echo the first recognized construct name found in the prompt.

    Falls back to ``default_name`` when none is found so the caller always
    gets a parseable, schema-valid VAAMR/PURER response.
    """
    lowered = prompt.lower()
    chosen = None
    # Prefer the longest names first so "Attention Regulation" wins over a bare
    # substring of another construct.
    for name in sorted(_VAAMR_NAMES + _PURER_NAMES, key=len, reverse=True):
        if name.lower() in lowered:
            chosen = name
            break
    if chosen is None:
        chosen = default_name
    return {
        "primary_stage": chosen,
        "primary_confidence": 0.85,
        "secondary_stage": None,
        "secondary_confidence": None,
        "justification": "deterministic test response",
        "evidence_phrase": "test evidence",
    }


class FakeLLMClient:
    """Offline LLMClient replacement.

    Parameters
    ----------
    responder:
        Optional ``callable(prompt) -> dict``. The dict is JSON-encoded and
        returned as the response text. If omitted, a default VAAMR/PURER
        responder is used.
    default_name:
        Construct name returned when the default responder finds nothing in
        the prompt (defaults to ``"Attention Regulation"``).
    raw_text:
        If set, this exact string is returned verbatim as the response text
        (use to inject malformed JSON).
    finish_reason:
        Value placed at ``metadata['choices'][0]['finish_reason']`` — set to
        ``'length'`` to exercise truncation handling.
    config:
        Optional object exposed as ``.config`` (mirrors ``LLMClient.config``);
        a tiny default with a ``model`` attribute is created otherwise.
    """

    def __init__(self,
                 responder: Optional[Callable[[str], Dict]] = None,
                 default_name: str = "Attention Regulation",
                 raw_text: Optional[str] = None,
                 finish_reason: str = "stop",
                 config=None):
        self._responder = responder
        self._default_name = default_name
        self._raw_text = raw_text
        self._finish_reason = finish_reason
        self.calls: List[str] = []   # every prompt seen, for assertions
        self.config = config if config is not None else _FakeConfig()

    # -- public interface ---------------------------------------------------
    def request(self, prompt: str) -> Tuple[Optional[str], Optional[Dict]]:
        self.calls.append(prompt)
        if self._raw_text is not None:
            text = self._raw_text
        else:
            payload = (self._responder(prompt) if self._responder
                       else _default_responder(prompt, self._default_name))
            text = json.dumps(payload)
        metadata = {
            "choices": [{
                "finish_reason": self._finish_reason,
                "message": {"content": text, "reasoning_content": ""},
            }],
            "model": getattr(self.config, "model", "fake-model"),
        }
        return text, metadata

    def multi_model_request(
        self, prompt: str, models: Optional[List[str]] = None
    ) -> List[Tuple[str, Optional[str], Optional[Dict]]]:
        if models is None:
            models = getattr(self.config, "models", None) or [getattr(self.config, "model", "fake-model")]
        out = []
        for m in models:
            text, meta = self.request(prompt)
            out.append((m, text, meta))
        return out

    def check_loaded_model(self, expected_model: str) -> bool:
        return True


class _FakeConfig:
    backend = "fake"
    model = "fake-model"
    models: List[str] = []
    temperature = 0.0
    process_logger = None
