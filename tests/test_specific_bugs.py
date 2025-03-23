#!/usr/bin/env python
"""Tests for specific bugs that have been encountered."""

import unittest
import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
import tempfile
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import test helpers
from tests.test_helpers import async_test

class TestSpecificBugs(unittest.TestCase):
    """Tests for specific bugs that have been encountered."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Create a git repository
        self._init_git_repo()
        
        # Create logs directory
        self.logs_dir = self.repo_path / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        # Path to the goal_decomposer.py script
        self.script_path = repo_root / "src" / "midpoint" / "agents" / "goal_decomposer.py"

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _init_git_repo(self):
        """Initialize a git repository in the temp directory."""
        subprocess.run(["git", "init"], cwd=self.temp_dir, check=True, capture_output=True)
        
        # Create a test file
        test_file = self.repo_path / "test_file.txt"
        test_file.write_text("Test content")
        
        # Commit the file
        subprocess.run(["git", "config", "user.name", "Test User"], cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "add", "."], cwd=self.temp_dir, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=self.temp_dir, check=True, capture_output=True)

    def test_asyncio_nesting_bug(self):
        """
        Test specifically for the asyncio nesting bug.
        
        This reproduces the error: 
        "asyncio.run() cannot be called from a running event loop"
        """
        # Create a sample subgoal file with the same structure as the one causing the issue
        subgoal_file = self.logs_dir / "subgoal_20250322_204836_d8145d.json"
        subgoal_content = {
            "next_step": "Test to reproduce the asyncio nesting bug",
            "validation_criteria": ["Test passes", "No asyncio errors"],
            "reasoning": "We need to test the specific bug that occurs when validate_repository_state is called with asyncio.run() inside main()",
            "requires_further_decomposition": False,
            "relevant_context": {},
            "metadata": {}
        }
        with open(subgoal_file, 'w') as f:
            json.dump(subgoal_content, f)
        
        # Run the script with parameters that would trigger the issue
        result = subprocess.run(
            [sys.executable, str(self.script_path), 
             "--repo-path", str(self.repo_path),
             "--input-file", str(subgoal_file),
             "--memory-hash", "a9c9c63eadd4b64b2191b78f811750953c1aa53b"],
            capture_output=True,
            text=True,
            env={**os.environ, "OPENAI_API_KEY": "dummy_key"}  # Provide a dummy API key
        )
        
        # Check for the specific error message
        self.assertNotIn("asyncio.run() cannot be called from a running event loop", result.stderr,
                        f"Script had asyncio nesting error: {result.stderr}")
        self.assertNotIn("coroutine 'validate_repository_state' was never awaited", result.stderr,
                        f"Script had coroutine never awaited error: {result.stderr}")
        
    def test_fix_for_validate_repository_state(self):
        """Test that the fix for validate_repository_state in main() works correctly."""
        # Import the goal_decomposer module
        import src.midpoint.agents.goal_decomposer as goal_decomposer
        
        # Get the line of code that contains the call to validate_repository_state
        with open(self.script_path, 'r') as f:
            source_code = f.readlines()
        
        # Find the line that calls validate_repository_state in main()
        problem_line = None
        main_function_lines = []
        in_main_function = False
        for i, line in enumerate(source_code):
            if "async def main():" in line:
                in_main_function = True
            
            if in_main_function:
                main_function_lines.append((i+1, line))  # 1-indexed line numbers
                
                # Look for the problematic call
                if "asyncio.run(validate_repository_state" in line:
                    problem_line = i+1  # 1-indexed line number
            
            if in_main_function and line.startswith("if __name__ =="):
                in_main_function = False
        
        # If we found the problem line, we should verify it's been fixed
        if problem_line:
            print(f"\nFound problematic validate_repository_state call at line {problem_line}:")
            print(f"  {source_code[problem_line-1].strip()}")
            
            # Recommend the fix
            print("\nRecommended fix:")
            print("  await validate_repository_state(args.repo_path, git_hash=current_hash, skip_clean_check=True)")
            
            # Provide a warning if not fixed
            self.fail(f"Found problematic asyncio.run(validate_repository_state) call at line {problem_line}. " +
                      "This should be changed to 'await validate_repository_state(...)' to avoid the nested event loop error.")
        else:
            # Either the problem was fixed or the code structure changed
            # Try to find any await validate_repository_state call in main()
            found_await = False
            for line_num, line in main_function_lines:
                if "await validate_repository_state" in line:
                    found_await = True
                    print(f"\nFound fixed validate_repository_state call at line {line_num}:")
                    print(f"  {line.strip()}")
                    break
            
            # The code should have the proper await pattern
            self.assertTrue(found_await, 
                          "Could not find 'await validate_repository_state' in main(). " +
                          "Check if the code structure has changed or if the fix was applied differently.")

if __name__ == "__main__":
    unittest.main() 