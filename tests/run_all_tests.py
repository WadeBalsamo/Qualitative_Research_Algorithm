#!/usr/bin/env python3
"""
Comprehensive test runner for QRA test suite.
Runs all tests without output truncation and writes results to test_results.txt.
"""

import sys
import unittest
import io
import os
from contextlib import redirect_stdout, redirect_stderr

def run_all_tests():
    """Discover and run all tests in the tests directory."""

    # Create output file
    output_file = os.path.join(os.path.dirname(__file__), 'test_results.txt')

    with open(output_file, 'w', encoding='utf-8') as f:
        # Discover all tests in current directory
        loader = unittest.TestLoader()
        start_dir = os.path.dirname(__file__)
        suite = loader.discover(start_dir, pattern='test_*.py')

        # Create a custom test runner with maximum verbosity
        runner = unittest.TextTestRunner(
            stream=f,
            verbosity=2,
            buffer=False  # Don't buffer output
        )

        # Also print to console
        print(f"Running all tests from {start_dir}")
        print(f"Results will be written to {output_file}")
        print("=" * 80)

        # Run tests
        result = runner.run(suite)

        # Write summary
        f.write("\n" + "=" * 80 + "\n")
        f.write("TEST SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Tests run: {result.testsRun}\n")
        f.write(f"Failures: {len(result.failures)}\n")
        f.write(f"Errors: {len(result.errors)}\n")
        f.write(f"Skipped: {len(result.skipped)}\n")
        f.write(f"Success: {result.wasSuccessful()}\n")

        if result.failures:
            f.write("\n" + "=" * 80 + "\n")
            f.write("FAILURES DETAIL\n")
            f.write("=" * 80 + "\n")
            for test, traceback in result.failures:
                f.write(f"\n{test}:\n")
                f.write(f"{traceback}\n")

        if result.errors:
            f.write("\n" + "=" * 80 + "\n")
            f.write("ERRORS DETAIL\n")
            f.write("=" * 80 + "\n")
            for test, traceback in result.errors:
                f.write(f"\n{test}:\n")
                f.write(f"{traceback}\n")

        # Print summary to console
        print("=" * 80)
        print(f"Tests run: {result.testsRun}")
        print(f"Failures: {len(result.failures)}")
        print(f"Errors: {len(result.errors)}")
        print(f"Skipped: {len(result.skipped)}")
        print(f"Success: {result.wasSuccessful()}")
        print(f"\nFull results written to: {output_file}")

        return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_all_tests())
