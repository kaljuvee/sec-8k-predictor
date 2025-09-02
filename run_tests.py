#!/usr/bin/env python3
"""
Test runner for SEC 8-K Predictor

Runs all unit tests and generates a test report.
"""

import unittest
import sys
import os
from pathlib import Path
import time

# Add src directory to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_tests():
    """Run all tests and generate report"""
    print("=" * 60)
    print("SEC 8-K Predictor - Test Suite")
    print("=" * 60)
    
    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = Path(__file__).parent / "tests"
    suite = loader.discover(start_dir, pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(
        verbosity=2,
        stream=sys.stdout,
        buffer=True
    )
    
    start_time = time.time()
    result = runner.run(suite)
    end_time = time.time()
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"Execution time: {end_time - start_time:.2f} seconds")
    
    # Print detailed failure information
    if result.failures:
        print("\n" + "-" * 60)
        print("FAILURES")
        print("-" * 60)
        for test, traceback in result.failures:
            print(f"\nFAILED: {test}")
            print(traceback)
    
    if result.errors:
        print("\n" + "-" * 60)
        print("ERRORS")
        print("-" * 60)
        for test, traceback in result.errors:
            print(f"\nERROR: {test}")
            print(traceback)
    
    if result.skipped:
        print("\n" + "-" * 60)
        print("SKIPPED TESTS")
        print("-" * 60)
        for test, reason in result.skipped:
            print(f"SKIPPED: {test} - {reason}")
    
    print("\n" + "=" * 60)
    
    # Return success status
    return len(result.failures) == 0 and len(result.errors) == 0

def run_specific_test(test_module):
    """Run a specific test module"""
    print(f"Running tests for {test_module}...")
    
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromName(f'tests.{test_module}')
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test module
        test_module = sys.argv[1]
        success = run_specific_test(test_module)
    else:
        # Run all tests
        success = run_tests()
    
    sys.exit(0 if success else 1)

