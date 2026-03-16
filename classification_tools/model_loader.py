"""
model_loader.py
---------------
Setup and validation for locally-hosted Hugging Face models.

Manages three models for cross-referencing:
- LLAMA 4 Maverick 17B from Hugging Face
- Mixtral 8x7B Instruct
- Qwen 3 Next 80B from Hugging Face
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import torch

# ---------- Configuration ----------
# Model identifiers from Hugging Face
LLAMA_MAVERICK_MODEL = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
MIXTRAL_MODEL = "mistralai/Mixtral-8x7B-Instruct-v0.1"
QWEN_MODEL = "Qwen/Qwen3-Next-80B-A3B-Instruct"

REQUIRED_MODELS = [LLAMA_MAVERICK_MODEL, MIXTRAL_MODEL, QWEN_MODEL]

MODEL_DISPLAY_NAMES = {
    LLAMA_MAVERICK_MODEL: "LLAMA 4 Maverick 17B",
    MIXTRAL_MODEL: "Mixtral 8x7B",
    QWEN_MODEL: "Qwen 3 Next 80B",
}

# Cache directory for models
CACHE_DIR = os.environ.get('HF_HOME', os.path.expanduser('~/.cache/huggingface'))

# Global model cache
_model_cache: Dict[str, Tuple] = {}


def get_required_models() -> List[str]:
    """Return the list of required model identifiers."""
    return REQUIRED_MODELS.copy()


def get_model_display_name(model_id: str) -> str:
    """Return human-readable name for a model."""
    return MODEL_DISPLAY_NAMES.get(model_id, model_id)


def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")

    try:
        import transformers
        print(f"  ✓ transformers {transformers.__version__}")
    except ImportError:
        print("  ✗ transformers not found")
        print("\nInstalling transformers...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "transformers", "accelerate"], check=True)
        print("  ✓ transformers installed")

    try:
        import torch
        print(f"  ✓ torch {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ CUDA not available, using CPU (will be slower)")
    except ImportError:
        print("  ✗ torch not found")
        print("\nPlease install PyTorch: pip install torch")
        sys.exit(1)


def download_model(model_id: str) -> bool:
    """
    Download a model from Hugging Face if not cached locally.

    Returns True if model is ready, False otherwise.
    """
    print(f"\nDownloading model: {get_model_display_name(model_id)}")
    print(f"  Hugging Face ID: {model_id}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        print("  Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        print("  Downloading model weights (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Use FP16 to save memory
            low_cpu_mem_usage=True,
            device_map="auto",  # Automatically distribute across GPUs/CPU
        )

        print(f"  ✓ Model {get_model_display_name(model_id)} downloaded successfully")

        # Clean up to save memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True

    except Exception as e:
        print(f"  ✗ Error downloading model {model_id}: {e}")
        return False


def verify_model(model_id: str) -> bool:
    """Verify a model is available locally."""
    try:
        from transformers import AutoTokenizer

        # Try to load tokenizer - if it's cached, this is fast
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            local_files_only=True,  # Only check local cache
        )
        return True
    except Exception:
        return False


def load_model(model_id: str, force_reload: bool = False):
    """
    Load a model and tokenizer into memory.

    Uses caching to avoid reloading the same model multiple times.
    """
    global _model_cache

    # Return cached model if available
    if not force_reload and model_id in _model_cache:
        print(f"  Using cached model: {get_model_display_name(model_id)}")
        return _model_cache[model_id]

    print(f"  Loading model: {get_model_display_name(model_id)}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=CACHE_DIR,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        model.eval()  # Set to evaluation mode

        _model_cache[model_id] = (model, tokenizer)
        print(f"  ✓ Model loaded successfully")

        return model, tokenizer

    except Exception as e:
        print(f"  ✗ Error loading model {model_id}: {e}")
        raise


def unload_model(model_id: str):
    """Unload a model from memory to free up resources."""
    global _model_cache

    if model_id in _model_cache:
        del _model_cache[model_id]
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print(f"  Unloaded model: {get_model_display_name(model_id)}")


def ensure_models_ready(download_if_missing: bool = True):
    """
    Ensure all required models are downloaded and ready.

    This function:
    1. Checks if required packages are installed
    2. Verifies models are cached locally
    3. Downloads missing models from Hugging Face
    4. Optionally loads models into memory

    Args:
        download_if_missing: If True, download models that aren't cached
    """
    print("=" * 70)
    print("MODEL LOADER: Ensuring Hugging Face models are ready")
    print("=" * 70)

    # Check dependencies
    check_dependencies()

    print("\nChecking model availability...")

    # Check and download models
    all_ready = True
    for model_id in REQUIRED_MODELS:
        if verify_model(model_id):
            print(f"✓ {get_model_display_name(model_id)} is cached locally")
        else:
            print(f"✗ {get_model_display_name(model_id)} not found in cache")
            if download_if_missing:
                success = download_model(model_id)
                if not success:
                    all_ready = False
            else:
                all_ready = False

    if not all_ready:
        print("\n⚠ Some models are not ready")
        if not download_if_missing:
            print("Run with download_if_missing=True to download them")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("All models ready for classification")
    print("=" * 70 + "\n")


def test_models():
    """Run a quick test on all models."""
    print("\nTesting models...")

    for model_id in REQUIRED_MODELS:
        print(f"\nTesting {get_model_display_name(model_id)}...")
        try:
            model, tokenizer = load_model(model_id)

            prompt = "Reply with just your model name."
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            print("  Generating response...")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.7,
                    do_sample=True,
                )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"  Response: {response[:100]}")

            # Unload to save memory
            unload_model(model_id)

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()


def get_model_info():
    """Print information about the configured models."""
    print("\n" + "=" * 70)
    print("CONFIGURED MODELS")
    print("=" * 70)

    for model_id in REQUIRED_MODELS:
        cached = "✓ Cached" if verify_model(model_id) else "✗ Not cached"
        print(f"\n{get_model_display_name(model_id)}")
        print(f"  ID: {model_id}")
        print(f"  Status: {cached}")
        print(f"  Cache: {CACHE_DIR}")


if __name__ == '__main__':
    ensure_models_ready()
    get_model_info()

    # Optionally test models
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_models()
