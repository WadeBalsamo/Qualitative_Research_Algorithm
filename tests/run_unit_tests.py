#!/usr/bin/env python3
"""
run_unit_tests.py — hermetic unit-tier runner.

Discovers and runs every test under tests/unit/ (no network, no model
downloads, no Ollama). Writes a full transcript to tests/unit_results.txt and
mirrors a summary to the console. Exit code is 0 only when the suite is green.

    python tests/run_unit_tests.py
    python tests/run_unit_tests.py -v          # extra console verbosity
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Unit tier is the QUICK tier: force slow/full-model tests OFF so a default run
# never trains a full GNN, runs the full pipeline, or downloads model weights.
os.environ["QRA_RUN_SLOW"] = "0"
os.environ["QRA_RUN_INTEGRATION"] = "0"


def main() -> int:
    unit_dir = os.path.join(_HERE, "unit")
    out_path = os.path.join(_HERE, "unit_results.txt")

    loader = unittest.TestLoader()
    suite = loader.discover(start_dir=unit_dir, pattern="test_*.py", top_level_dir=_ROOT)

    print(f"Discovering unit tests in {unit_dir}")
    print(f"Results -> {out_path}")
    print("=" * 80)

    with open(out_path, "w", encoding="utf-8") as f:
        runner = unittest.TextTestRunner(stream=f, verbosity=2, buffer=False)
        result = runner.run(suite)
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Failures:  {len(result.failures)}\n")
        f.write(f"Errors:    {len(result.errors)}\n")
        f.write(f"Skipped:   {len(result.skipped)}\n")
        f.write(f"Success:   {result.wasSuccessful()}\n")

    print(f"Tests run: {result.testsRun}")
    print(f"Failures:  {len(result.failures)}")
    print(f"Errors:    {len(result.errors)}")
    print(f"Skipped:   {len(result.skipped)}")
    print(f"Success:   {result.wasSuccessful()}")
    print(f"Full results: {out_path}")
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
