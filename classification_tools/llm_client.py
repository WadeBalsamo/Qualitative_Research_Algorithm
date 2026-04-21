"""
llm_client.py
-------------
Unified LLM API client for OpenRouter, Replicate, Ollama, LM Studio, and
Hugging Face backends, plus JSON extraction utilities.

Supports multi-model cross-referencing for consensus-based classification.

Token budget strategy
---------------------
For local backends (LM Studio, Ollama) and OpenRouter, ``max_tokens`` is set
to the model's full context window on every request.  The server automatically
clamps completion length to ``context_length - prompt_tokens``, so this is
effectively "use all remaining space" without needing to know the prompt size
in advance.  Context lengths are fetched once per (backend, model) pair and
cached for the lifetime of the process.

Reasoning-model handling
------------------------
Models that emit chain-of-thought (e.g. Qwen-thinking, DeepSeek-R1) place
thinking tokens in a separate ``reasoning_content`` field and the actual
answer in ``content``.  If ``content`` is empty after a call, ``_extract_content``
attempts to salvage a JSON object from ``reasoning_content`` before giving up.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, List


@dataclass
class LLMClientConfig:
    """Configuration for the LLM API client."""
    backend: str = 'huggingface'  # 'openrouter', 'replicate', 'ollama', 'lmstudio', or 'huggingface'
    api_key: str = ''
    replicate_api_token: str = ''
    model: str = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'  # Single model or list for multi-model
    models: List[str] = field(default_factory=list)  # For multi-model cross-referencing
    temperature: float = 0.0
    max_new_tokens: int = 512  # Used by Replicate and HuggingFace only
    timeout: int = 600
    max_retries: int = 3
    retry_base_delay: float = 2.0
    ollama_host: str = '0.0.0.0'
    ollama_port: int = 11434
    lmstudio_base_url: str = 'http://127.0.0.1:1234/v1'  # LM Studio OpenAI-compatible endpoint
    use_gpu: bool = True  # Use GPU if available for Hugging Face models
    batch_size: int = 1  # For future batch processing
    no_reasoning: bool = False  # Disable chain-of-thought tokens (LM Studio, Ollama thinking models)
    process_logger: Optional[Any] = field(default=None, compare=False, repr=False)  # ProcessLogger instance for I/O tracing


class LLMClient:
    """Unified LLM client dispatching to OpenRouter, Replicate, Ollama, LM Studio, or HuggingFace."""

    # Class-level cache: (backend_key, model) -> context_length
    # backend_key is a stable string identifying the endpoint (URL or backend name).
    _context_length_cache: Dict[Tuple[str, str], int] = {}

    _CONTEXT_LENGTH_FALLBACK = 8192

    def __init__(self, config: LLMClientConfig):
        self.config = config

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def request(self, prompt: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to the configured backend and return (text, metadata)."""
        plog = self.config.process_logger
        if plog:
            plog.log_llm_prompt(self.config.model, prompt)

        if self.config.backend == 'replicate':
            result_text, metadata = self._replicate_request(prompt)
        elif self.config.backend == 'ollama':
            result_text, metadata = self._ollama_request(prompt)
        elif self.config.backend == 'huggingface':
            result_text, metadata = self._huggingface_request(prompt)
        elif self.config.backend == 'lmstudio':
            result_text, metadata = self._lmstudio_request(prompt)
        else:
            result_text, metadata = self._openrouter_request(prompt)

        if plog:
            reasoning = ''
            if metadata and isinstance(metadata, dict):
                choices = metadata.get('choices', [])
                if choices:
                    reasoning = choices[0].get('message', {}).get('reasoning_content') or ''
            plog.log_llm_response(result_text or '(empty)', reasoning)

        return result_text, metadata

    def multi_model_request(
        self, prompt: str, models: Optional[List[str]] = None
    ) -> List[Tuple[str, Optional[str], Optional[Dict]]]:
        """
        Send a prompt to multiple models and return all responses.

        Returns a list of (model_id, text, metadata) tuples.
        """
        if models is None:
            models = self.config.models if self.config.models else [self.config.model]

        results = []
        for model_id in models:
            original_model = self.config.model
            self.config.model = model_id
            text, metadata = self.request(prompt)
            results.append((model_id, text, metadata))
            self.config.model = original_model

        return results

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _get_context_length(self) -> int:
        """
        Return the context window size for the configured model and backend.

        Results are cached per (backend_key, model).  Falls back to
        ``_CONTEXT_LENGTH_FALLBACK`` if the metadata endpoint is unavailable.
        """
        backend = self.config.backend
        model = self.config.model

        if backend == 'lmstudio':
            backend_key = self.config.lmstudio_base_url.rstrip('/')
            cache_key = (backend_key, model)
            if cache_key not in LLMClient._context_length_cache:
                LLMClient._context_length_cache[cache_key] = (
                    self._fetch_lmstudio_context_length(backend_key)
                )
            return LLMClient._context_length_cache[cache_key]

        elif backend == 'ollama':
            backend_key = f"ollama://{self.config.ollama_host}:{self.config.ollama_port}"
            cache_key = (backend_key, model)
            if cache_key not in LLMClient._context_length_cache:
                LLMClient._context_length_cache[cache_key] = (
                    self._fetch_ollama_context_length()
                )
            return LLMClient._context_length_cache[cache_key]

        elif backend == 'openrouter':
            backend_key = 'openrouter'
            cache_key = (backend_key, model)
            if cache_key not in LLMClient._context_length_cache:
                LLMClient._context_length_cache[cache_key] = (
                    self._fetch_openrouter_context_length()
                )
            return LLMClient._context_length_cache[cache_key]

        # HuggingFace context length is read at generation time from the model config.
        # Replicate doesn't expose metadata; callers use max_new_tokens directly.
        return self._CONTEXT_LENGTH_FALLBACK

    @staticmethod
    def _extract_content(choice: Dict) -> Tuple[Optional[str], bool]:
        """
        Extract the response text from an OpenAI-compatible chat completion choice.

        Handles reasoning models (DeepSeek-R1, Qwen-thinking, etc.) that emit
        chain-of-thought in ``reasoning_content`` and the actual answer in
        ``content``.  If ``content`` is empty, attempts to salvage a JSON
        object from ``reasoning_content``.

        Returns (text, exhausted) where ``exhausted`` is True when
        ``finish_reason == 'length'`` and no usable content could be extracted —
        meaning the prompt consumed the full context window and retrying won't help.
        """
        message = choice.get('message', {})
        result_text = message.get('content') or ''

        if result_text.strip():
            return result_text, False

        finish_reason = choice.get('finish_reason', '')
        reasoning = message.get('reasoning_content', '') or ''

        if reasoning:
            start = reasoning.find('{')
            end = reasoning.rfind('}') + 1
            if start != -1 and end > start:
                return reasoning[start:end], False

        if finish_reason == 'length':
            return None, True  # context exhausted — no point retrying

        return result_text or None, False

    # ------------------------------------------------------------------
    # Context-length fetchers (one per provider)
    # ------------------------------------------------------------------

    def _fetch_lmstudio_context_length(self, base_url: str) -> int:
        """Query LM Studio's /v1/models for the loaded model's context window."""
        import requests
        try:
            resp = requests.get(
                f"{base_url}/models",
                headers={"Authorization": "Bearer lm-studio"},
                timeout=10,
            )
            for info in resp.json().get('data', []):
                ctx = (
                    info.get('context_length')
                    or info.get('max_context_length')
                    or info.get('max_position_embeddings')
                )
                if ctx:
                    return int(ctx)
        except Exception:
            pass
        return self._CONTEXT_LENGTH_FALLBACK

    def _fetch_ollama_context_length(self) -> int:
        """
        Query Ollama's /api/show for the model's context window.

        Ollama returns model metadata under ``model_info`` with keys like
        ``llama.context_length``, and also exposes ``num_ctx`` in the
        ``parameters`` string.
        """
        import requests
        base_url = f"http://{self.config.ollama_host}:{self.config.ollama_port}"
        try:
            resp = requests.post(
                f"{base_url}/api/show",
                json={"name": self.config.model},
                timeout=10,
            )
            data = resp.json()

            # Prefer structured model_info fields (Ollama ≥ 0.2)
            model_info = data.get('model_info', {})
            for key in ('llama.context_length', 'context_length', 'max_position_embeddings'):
                if key in model_info:
                    return int(model_info[key])

            # Fall back to parsing the parameters string: "num_ctx 4096\n..."
            params_str = data.get('parameters', '')
            for line in params_str.splitlines():
                parts = line.strip().split()
                if len(parts) == 2 and parts[0] == 'num_ctx':
                    return int(parts[1])
        except Exception:
            pass
        return self._CONTEXT_LENGTH_FALLBACK

    def _fetch_openrouter_context_length(self) -> int:
        """
        Query OpenRouter's /api/v1/models to find context_length for the model.
        """
        import requests
        try:
            resp = requests.get(
                "https://openrouter.ai/api/v1/models",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=15,
            )
            for info in resp.json().get('data', []):
                if info.get('id') == self.config.model:
                    ctx = info.get('context_length')
                    if ctx:
                        return int(ctx)
        except Exception:
            pass
        return self._CONTEXT_LENGTH_FALLBACK

    # ------------------------------------------------------------------
    # Backend request methods
    # ------------------------------------------------------------------

    def _openrouter_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to OpenRouter API and return the response text + metadata."""
        import requests

        max_tokens = self._get_context_length()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url="https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.config.api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": self.config.model,
                        "temperature": self.config.temperature,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=self.config.timeout,
                )

                metadata = response.json()
                choice = metadata['choices'][0]
                result_text, exhausted = self._extract_content(choice)

                if exhausted:
                    print(
                        f"  Warning: OpenRouter context window exhausted "
                        f"(max_tokens={max_tokens}). Prompt may be too long."
                    )
                    return None, metadata

                return result_text, metadata
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"  API error (attempt {attempt + 1}): {e}, retrying in {delay:.0f}s")
                    time.sleep(delay)

        print(f"  API error after {self.config.max_retries} attempts: {last_error}")
        return None, None

    def _replicate_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to Replicate API and return the response text."""
        try:
            import replicate
            client = replicate.Client(api_token=self.config.replicate_api_token)

            output = client.run(
                self.config.model,
                input={
                    "prompt": prompt,
                    "temperature": self.config.temperature,
                    "max_new_tokens": self.config.max_new_tokens,
                },
            )
            result_text = ''.join(output)
            return result_text, None
        except Exception as e:
            print(f"  Replicate API error: {e}")
            return None, None

    def _ollama_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to local Ollama API and return the response text."""
        import requests

        base_url = f"http://{self.config.ollama_host}:{self.config.ollama_port}"
        # num_ctx tells Ollama how large a context window to allocate for this
        # request.  Setting it to the model's reported context length means
        # "use the full window"; the server caps generation at whatever tokens
        # remain after the prompt.
        num_ctx = self._get_context_length()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                ollama_options = {
                    "temperature": self.config.temperature,
                    "num_ctx": num_ctx,
                    # num_predict -1 means "generate until stop or context full"
                    "num_predict": -1,
                }
                if self.config.no_reasoning:
                    ollama_options["think"] = False  # Qwen3/thinking model: disable CoT
                response = requests.post(
                    url=f"{base_url}/api/chat",
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "format": "json",
                        "options": ollama_options,
                    },
                    timeout=self.config.timeout,
                )

                data = response.json()

                # Ollama uses a flat message structure, not OpenAI choices[].
                # Build a synthetic choice dict so _extract_content works uniformly.
                message = data.get('message', {})
                synthetic_choice = {
                    'message': message,
                    'finish_reason': 'length' if data.get('done_reason') == 'length' else 'stop',
                }
                result_text, exhausted = self._extract_content(synthetic_choice)

                if exhausted:
                    print(
                        f"  Warning: Ollama context window exhausted "
                        f"(num_ctx={num_ctx}). Prompt may be too long."
                    )
                    return None, data

                metadata = {
                    'model': self.config.model,
                    'done': data.get('done', False),
                    'total_duration': data.get('total_duration'),
                }
                return result_text, metadata
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"  Ollama error (attempt {attempt + 1}): {e}, retrying in {delay:.0f}s")
                    time.sleep(delay)

        print(f"  Ollama error after {self.config.max_retries} attempts: {last_error}")
        return None, None

    def _lmstudio_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """
        Send a prompt to a local LM Studio server (OpenAI-compatible API).

        LM Studio exposes the same /v1/chat/completions interface as OpenAI.
        No real API key is needed — a dummy value satisfies the header check.
        The base URL is configurable via ``lmstudio_base_url``
        (default: ``http://127.0.0.1:1234/v1``).

        Token budget is set to the model's full context window so the server
        clamps completion length to whatever space remains after the prompt.
        """
        import requests

        base_url = self.config.lmstudio_base_url.rstrip('/')
        max_tokens = self._get_context_length()
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                json_body = {
                    "model": self.config.model,
                    "temperature": self.config.temperature,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                }
                if self.config.no_reasoning:
                    json_body["include_reasoning"] = False
                response = requests.post(
                    url=f"{base_url}/chat/completions",
                    headers={
                        "Authorization": "Bearer lm-studio",
                        "Content-Type": "application/json",
                    },
                    json=json_body,
                    timeout=self.config.timeout,
                )
                data = response.json()

                if not isinstance(data, dict) or 'choices' not in data or not data['choices']:
                    if isinstance(data, dict):
                        err_obj = data.get('error') or {}
                        err_msg = (err_obj.get('message', '') if isinstance(err_obj, dict) else str(err_obj)) or str(data)[:200]
                    else:
                        err_msg = str(data)[:200]
                    raise ValueError(f"LM Studio returned no choices: {err_msg}")

                result_text, exhausted = self._extract_content(data['choices'][0])

                if exhausted:
                    print(
                        f"  Warning: model context window exhausted "
                        f"(max_tokens={max_tokens}). Prompt may be too long."
                    )
                    return None, data

                return result_text, data
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    delay = self.config.retry_base_delay * (2 ** attempt)
                    print(f"  LM Studio error (attempt {attempt + 1}): {e}, retrying in {delay:.0f}s")
                    time.sleep(delay)

        print(f"  LM Studio error after {self.config.max_retries} attempts: {last_error}")
        return None, None

    def _huggingface_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to a locally-loaded Hugging Face model and return the response."""
        try:
            import torch
            from .model_loader import load_model

            model, tokenizer = load_model(self.config.model)

            # Derive the model's actual context window from its config.
            model_max_ctx = getattr(model.config, 'max_position_embeddings', None)

            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_max_ctx or 2048,
            )
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            prompt_len = inputs['input_ids'].shape[1]
            # Generate up to whatever space remains in the context window.
            if model_max_ctx:
                max_new = model_max_ctx - prompt_len
            else:
                max_new = self.config.max_new_tokens

            with torch.no_grad():
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max(max_new, 1),
                    temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                end_time = time.time()

            new_tokens = outputs[0][prompt_len:]
            result_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

            metadata = {
                'model': self.config.model,
                'backend': 'huggingface',
                'generation_time': end_time - start_time,
                'device': str(model.device),
            }

            return result_text.strip(), metadata

        except Exception as e:
            print(f"  Hugging Face model error: {e}")
            import traceback
            traceback.print_exc()
            return None, None


def extract_json(output_str: str) -> Dict:
    """
    Extract JSON from LLM output, handling extra text around the JSON block.

    Tries direct parsing first, then falls back to finding the outermost
    { ... } pair in the output string.
    """
    try:
        return json.loads(output_str)
    except json.JSONDecodeError:
        start = output_str.find('{')
        end = output_str.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(output_str[start:end])
        raise ValueError("Could not extract JSON from LLM output.")
