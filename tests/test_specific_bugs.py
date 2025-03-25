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
        self.script_path = Path(__file__).parent.parent / "src" / "midpoint" / "agents" / "goal_decomposer.py"

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
             str(self.repo_path),  # repo_path as positional argument
             "Test asyncio nesting bug",  # goal as positional argument
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

if __name__ == "__main__":
    unittest.main() 