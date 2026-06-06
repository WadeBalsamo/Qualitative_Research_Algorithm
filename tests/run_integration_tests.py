#!/usr/bin/env python3
"""
run_integration_tests.py — real-model integration tier.

Runs the tests under tests/integration/, which exercise the pipeline with the
smallest REAL models: all-MiniLM-L6-v2 embeddings and a tiny Ollama LLM. These
download model weights on first run and write outputs under
tests/testrun-outputs/ (gitignored).

Before discovery it tries to pre-pull the tiny Ollama model so the
LLM-dependent tests run; if Ollama is unavailable those tests skip cleanly
(the embedding/GNN tests still run). Integration tests are NEVER collected by
run_unit_tests.py.

    python tests/run_integration_tests.py
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Integration tier runs the heavy/real-model paths.
os.environ["QRA_RUN_SLOW"] = "1"
os.environ["QRA_RUN_INTEGRATION"] = "1"


def _prepull_ollama() -> None:
    try:
        from tests.testhelpers.ollama_helper import ensure_ollama_model
        cfg = ensure_ollama_model()
        print(f"Ollama ready: {cfg.model}")
    except Exception as e:  # unittest.SkipTest or anything else
        print(f"Ollama not available ({e}); LLM-dependent tests will skip.")


def main() -> int:
    os.makedirs(os.path.join(_HERE, "testrun-outputs"), exist_ok=True)
    integ_dir = os.path.join(_HERE, "integration")
    out_path = os.path.join(_HERE, "integration_results.txt")

    print("=" * 80)
    print("QRA integration tier (real tiny models)")
    print("=" * 80)
    _prepull_ollama()

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=integ_dir, pattern="test_*.py", top_level_dir=_ROOT)

    with open(out_path, "w", encoding="utf-8") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2, buffer=False)
        result = runner.run(suite)
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Failures:  {len(result.failures)}\n")
        f.write(f"Errors:    {len(result.errors)}\n")
        f.write(f"Skipped:   {len(result.skipped)}\n")
        f.write(f"Success:   {result.wasSuccessful()}\n")

    print(f"Tests run: {result.testsRun}  Failures: {len(result.failures)}  "
          f"Errors: {len(result.errors)}  Skipped: {len(result.skipped)}")
    print(f"Full results: {out_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
