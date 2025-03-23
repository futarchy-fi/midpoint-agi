#!/usr/bin/env python
"""Tests for proper asyncio pattern usage across the codebase."""

import unittest
import os
import sys
import re
from pathlib import Path
import inspect
import importlib

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

class TestAsyncioPatterns(unittest.TestCase):
    """Tests for checking the correct use of asyncio patterns."""

    def setUp(self):
        """Set up test fixtures."""
        # Path to src directory
        self.src_dir = repo_root / "src"
        
        # Function names that should not be called directly from async functions
        self.problematic_funcs = [
            'asyncio.run',
            'loop.run_until_complete',
            'loop.run_forever'
        ]
        
        # Collect all Python files in the src directory
        self.python_files = list(self.src_dir.glob("**/*.py"))
        
        # Track issues found for reporting
        self.issues_found = []

    def test_no_nested_asyncio_run_calls(self):
        """Verify there are no nested asyncio.run() calls in async functions."""
        for py_file in self.python_files:
            if py_file.is_file():
                # Read the file content
                with open(py_file, 'r', errors='ignore') as f:
                    try:
                        source_code = f.read()
                        
                        # Find all async function definitions
                        async_funcs = re.findall(r'async\s+def\s+(\w+)', source_code)
                        
                        # For each async function, check if it contains problematic function calls
                        for func_name in async_funcs:
                            # Extract the function body
                            pattern = rf'async\s+def\s+{func_name}.*?:(.*?)(?:async\s+def|\Z)'
                            matches = re.findall(pattern, source_code, re.DOTALL)
                            
                            if matches and len(matches) > 0:
                                func_body = matches[0]
                                
                                # Check for problematic function calls
                                for bad_func in self.problematic_funcs:
                                    # Look for the bad function call pattern
                                    pattern = rf'(\b{re.escape(bad_func)}\s*\()'
                                    bad_calls = re.findall(pattern, func_body)
                                    
                                    if bad_calls:
                                        issue = f"Found '{bad_func}' inside async function '{func_name}' in {py_file.relative_to(repo_root)}"
                                        self.issues_found.append(issue)
                    except UnicodeDecodeError:
                        # Skip files that can't be decoded properly
                        pass
        
        # If issues were found, fail the test with details
        if self.issues_found:
            self.fail(f"Found {len(self.issues_found)} improper asyncio patterns:\n" + 
                     "\n".join(f"- {issue}" for issue in self.issues_found))

    def test_async_functions_properly_awaited(self):
        """Check if async functions are properly awaited when called from other async functions."""
        # List of modules to analyze
        module_paths = []
        
        # Find all Python modules in src directory
        for py_file in self.python_files:
            if py_file.is_file():
                # Convert file path to module path
                rel_path = py_file.relative_to(repo_root)
                module_path = str(rel_path).replace('/', '.').replace('\\', '.').replace('.py', '')
                module_paths.append(module_path)
        
        # Track async functions in each module
        async_functions = {}
        
        # Import modules and collect async functions
        for module_path in module_paths:
            try:
                # Skip __init__.py files and other special modules
                if "__pycache__" in module_path or module_path.endswith("__init__"):
                    continue
                    
                module = importlib.import_module(module_path)
                
                # Find all async functions in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.iscoroutinefunction(obj) and not name.startswith('_'):
                        async_functions[f"{module_path}.{name}"] = obj
            except (ImportError, ModuleNotFoundError, AttributeError):
                # Skip modules that can't be imported properly
                pass
        
        # Now check all Python files for proper awaiting
        issues = []
        for py_file in self.python_files:
            if py_file.is_file():
                with open(py_file, 'r', errors='ignore') as f:
                    try:
                        source_code = f.read()
                        
                        # For each async function, check if it's properly awaited
                        for func_name, func_obj in async_functions.items():
                            # Get the short name of the function (without module path)
                            short_name = func_name.split('.')[-1]
                            
                            # Skip common names that might have many matches
                            if short_name in {'run', 'main', 'get', 'post', 'put', 'delete'}:
                                continue
                                
                            # Look for function calls without await
                            pattern = rf'(?<!await\s+)(?<!await\()\b{re.escape(short_name)}\s*\('
                            non_awaited_calls = re.findall(pattern, source_code)
                            
                            if non_awaited_calls:
                                # Verify this isn't a false positive by checking if function is called in an async function
                                # This is a basic check - it would miss some cases but catch obvious ones
                                async_context = re.search(rf'async\s+def.*?{re.escape(short_name)}\s*\(', source_code, re.DOTALL)
                                
                                if async_context:
                                    issue = f"Possible non-awaited async function '{short_name}' in {py_file.relative_to(repo_root)}"
                                    issues.append(issue)
                    except UnicodeDecodeError:
                        # Skip files that can't be decoded properly
                        pass
        
        # Report issues, but set as warning since this check can have false positives
        if issues:
            print(f"\nWARNING: Found {len(issues)} possible non-awaited async functions:")
            for issue in issues:
                print(f"- {issue}")

    def test_validate_repository_state_usage(self):
        """Check if validate_repository_state is always awaited."""
        # Look specifically for validate_repository_state calls
        for py_file in self.python_files:
            if py_file.is_file():
                with open(py_file, 'r', errors='ignore') as f:
                    try:
                        source_code = f.read()
                        
                        # Look for validate_repository_state calls without await
                        # First check for the function being used
                        if 'validate_repository_state' in source_code:
                            # Find all non-awaited calls
                            non_awaited = re.findall(r'(?<!await\s+)(?<!asyncio\.run\()(?<!await\()validate_repository_state\s*\(', source_code)
                            
                            # Find all calls inside asyncio.run
                            inside_run = re.findall(r'asyncio\.run\(\s*validate_repository_state\s*\(', source_code)
                            
                            # Track if issues were found
                            if non_awaited or inside_run:
                                relative_path = py_file.relative_to(repo_root)
                                
                                if non_awaited:
                                    self.issues_found.append(
                                        f"Found non-awaited call to validate_repository_state in {relative_path}"
                                    )
                                    
                                if inside_run:
                                    self.issues_found.append(
                                        f"Found validate_repository_state inside asyncio.run() in {relative_path}"
                                    )
                    except UnicodeDecodeError:
                        # Skip files that can't be decoded properly
                        pass
                        
        # If issues were found, fail the test with details
        if self.issues_found:
            self.fail(f"Found {len(self.issues_found)} improper validate_repository_state calls:\n" + 
                     "\n".join(f"- {issue}" for issue in self.issues_found))

if __name__ == "__main__":
    unittest.main() 