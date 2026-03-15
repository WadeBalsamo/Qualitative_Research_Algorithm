"""
llm_client.py
-------------
Unified LLM API client for OpenRouter, Replicate, and Ollama backends,
plus JSON extraction utilities.

Supports multi-model cross-referencing for consensus-based classification.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List


@dataclass
class LLMClientConfig:
    """Configuration for the LLM API client."""
    backend: str = 'huggingface'  # 'openrouter', 'replicate', 'ollama', or 'huggingface'
    api_key: str = ''
    replicate_api_token: str = ''
    model: str = 'meta-llama/Llama-4-Maverick-17B-128E-Instruct'  # Single model or list for multi-model
    models: List[str] = field(default_factory=list)  # For multi-model cross-referencing
    temperature: float = 0.0
    max_new_tokens: int = 512
    timeout: int = 120
    max_retries: int = 3
    retry_base_delay: float = 2.0
    ollama_host: str = '0.0.0.0'
    ollama_port: int = 11434
    use_gpu: bool = True  # Use GPU if available for Hugging Face models
    batch_size: int = 1  # For future batch processing


class LLMClient:
    """Unified LLM client dispatching to OpenRouter, Replicate, or Ollama."""

    def __init__(self, config: LLMClientConfig):
        self.config = config

    def request(self, prompt: str) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to the configured backend and return (text, metadata)."""
        if self.config.backend == 'replicate':
            return self._replicate_request(prompt)
        elif self.config.backend == 'ollama':
            return self._ollama_request(prompt)
        elif self.config.backend == 'huggingface':
            return self._huggingface_request(prompt)
        return self._openrouter_request(prompt)

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
            # Temporarily override the model in config
            original_model = self.config.model
            self.config.model = model_id

            text, metadata = self.request(prompt)
            results.append((model_id, text, metadata))

            # Restore original model
            self.config.model = original_model

        return results

    def _openrouter_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to OpenRouter API and return the response text + metadata."""
        import requests

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
                        "response_format": {"type": "json_object"},
                        "messages": [{"role": "user", "content": prompt}],
                    },
                    timeout=self.config.timeout,
                )

                metadata = response.json()
                result_text = metadata['choices'][0]['message']['content']
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
        last_error = None

        for attempt in range(self.config.max_retries):
            try:
                response = requests.post(
                    url=f"{base_url}/api/chat",
                    json={
                        "model": self.config.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                        "options": {
                            "temperature": self.config.temperature,
                            "num_predict": self.config.max_new_tokens,
                        },
                    },
                    timeout=self.config.timeout,
                )

                data = response.json()
                result_text = data.get('message', {}).get('content', '')
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

    def _huggingface_request(
        self, prompt: str,
    ) -> Tuple[Optional[str], Optional[Dict]]:
        """Send a prompt to a locally-loaded Hugging Face model and return the response."""
        try:
            import torch
            from shared.model_loader import load_model

            # Load the model and tokenizer
            model, tokenizer = load_model(self.config.model)

            # Prepare the input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)

            # Move to the same device as the model
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                start_time = time.time()
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    temperature=self.config.temperature if self.config.temperature > 0 else 1.0,
                    do_sample=self.config.temperature > 0,
                    pad_token_id=tokenizer.eos_token_id,
                )
                end_time = time.time()

            # Decode the output
            # Only get the new tokens (skip the input prompt)
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
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
