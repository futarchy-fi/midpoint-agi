#!/usr/bin/env python
"""
Test runner for tool-related tests.

This script runs all the tests related to tools and the GoalDecomposer's interaction with tools.
"""

import os
import sys
import unittest
import warnings

# Suppress urllib3 warnings about OpenSSL/LibreSSL version
warnings.filterwarnings('ignore', message='.*OpenSSL 1.1.1\\+.*')

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if __name__ == "__main__":
    # Find and run all tests
    test_loader = unittest.TestLoader()
    
    # Discover tests based on patterns
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    test_suite = unittest.TestSuite()
    
    for pattern in ["test_*.py"]:
        suite = test_loader.discover(tests_dir, pattern=pattern)
        test_suite.addTests(suite)
    
    # Run tests with reduced verbosity to avoid noisy logs
    result = unittest.TextTestRunner(verbosity=1).run(test_suite)
    
    # Return non-zero exit code if tests failed
    if not result.wasSuccessful():
        sys.exit(1) 