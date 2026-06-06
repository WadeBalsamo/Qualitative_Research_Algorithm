"""
tests.testhelpers.ollama_helper
-------------------------------
Set up the smallest available Ollama model for the integration tier.

``ensure_ollama_model()`` checks the daemon is reachable, pulls the tiny model
if needed, and returns an ``LLMClientConfig(backend='ollama', model=...)`` ready
to hand to the real classifier. If Ollama is not installed, the daemon is down,
or the pull fails, it raises ``unittest.SkipTest`` so integration tests degrade
gracefully instead of failing.

Override the model with ``QRA_TEST_OLLAMA_MODEL`` (default qwen2.5:0.5b-instruct).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import unittest
import urllib.error
import urllib.request

TINY_OLLAMA = os.environ.get("QRA_TEST_OLLAMA_MODEL", "qwen2.5:0.5b-instruct")
_DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "127.0.0.1")
_DEFAULT_PORT = int(os.environ.get("OLLAMA_PORT", "11434"))


def _daemon_base_url(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> str:
    return f"http://{host}:{port}"


def daemon_is_up(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> bool:
    try:
        with urllib.request.urlopen(f"{_daemon_base_url(host, port)}/api/tags", timeout=5) as r:
            return r.status == 200
    except (urllib.error.URLError, OSError, ValueError):
        return False


def _installed_models(host: str = _DEFAULT_HOST, port: int = _DEFAULT_PORT) -> list:
    try:
        with urllib.request.urlopen(f"{_daemon_base_url(host, port)}/api/tags", timeout=5) as r:
            data = json.loads(r.read().decode("utf-8"))
        return [m.get("name", "") for m in data.get("models", [])]
    except Exception:
        return []


def ensure_ollama_model(model: str = TINY_OLLAMA,
                        host: str = _DEFAULT_HOST,
                        port: int = _DEFAULT_PORT):
    """Ensure ``model`` is available via a running Ollama daemon.

    Returns an ``LLMClientConfig`` pointed at it. Raises ``unittest.SkipTest``
    if Ollama/the daemon/the model is unavailable.
    """
    if shutil.which("ollama") is None:
        raise unittest.SkipTest("ollama CLI not installed")
    if not daemon_is_up(host, port):
        raise unittest.SkipTest(f"ollama daemon not reachable at {_daemon_base_url(host, port)}")

    installed = _installed_models(host, port)
    if not any(m == model or m.startswith(model.split(":")[0] + ":") for m in installed):
        try:
            subprocess.run(["ollama", "pull", model], check=True,
                           capture_output=True, text=True, timeout=900)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            raise unittest.SkipTest(f"could not pull ollama model {model}: {e}")

    from classification_tools.llm_client import LLMClientConfig
    return LLMClientConfig(
        backend="ollama",
        model=model,
        ollama_host=host,
        ollama_port=port,
        temperature=0.0,
    )
