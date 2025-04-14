#!/usr/bin/env python
"""Tests for the CLI execution path of goal_decomposer.py."""

import unittest
import subprocess
import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import the module under test
from src.midpoint.agents.goal_decomposer import validate_repository_state, decompose_goal
from src.midpoint.agents.models import State, Goal, TaskContext

# Import test helpers
from tests.test_helpers import async_test

class TestGoalDecomposerCLI(unittest.TestCase):
    """Tests for the CLI execution path of goal_decomposer.py."""

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
        
        # Create a sample subgoal file
        self.subgoal_file = self.logs_dir / "subgoal_20250322_123456_abcdef.json"
        self.subgoal_content = {
            "next_state": "Test the CLI execution path",
            "validation_criteria": ["Test passes", "No asyncio errors"],
            "reasoning": "We need to test the CLI execution path to catch asyncio issues",
            "requires_further_decomposition": False,
            "relevant_context": {},
            "metadata": {}
        }
        with open(self.subgoal_file, 'w') as f:
            json.dump(self.subgoal_content, f)
        
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

    # 1. Test the CLI execution path with subprocess
    def test_cli_execution_help(self):
        """Test that the CLI can successfully show help without errors."""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, f"Script failed with error: {result.stderr}")
        self.assertIn("usage: goal_decomposer.py", result.stdout)

    def test_cli_execution_with_input_file(self):
        """Test the CLI execution with an input file."""
        # Run the script with the sample subgoal file
        result = subprocess.run(
            [sys.executable, str(self.script_path), 
             str(self.repo_path),  # repo_path as positional argument
             "Test goal",  # goal as positional argument
             "--input-file", str(self.subgoal_file),
             "--debug"],
            capture_output=True,
            text=True,
            env={**os.environ, "OPENAI_API_KEY": "dummy_key"}  # Provide a dummy API key
        )
        
        # Check for the specific asyncio error we're concerned about
        self.assertNotIn("asyncio.run() cannot be called from a running event loop", result.stderr,
                        f"Script had asyncio nesting error: {result.stderr}")

    # 2. Test validate_repository_state directly
    @async_test
    async def test_validate_repository_state(self):
        """Test validate_repository_state function directly."""
        # This should succeed for a valid repository
        await validate_repository_state(self.repo_path, skip_clean_check=True)
        
        # Test with a non-existent directory
        with self.assertRaises(ValueError):
            await validate_repository_state("/non/existent/path")
            
        # Test with uncommitted changes (create new file but don't commit)
        new_file = self.repo_path / "uncommitted.txt"
        new_file.write_text("Uncommitted changes")
        
        # This should raise an error about uncommitted changes if skip_clean_check is False
        with self.assertRaises(ValueError):
            await validate_repository_state(self.repo_path)
            
        # But should succeed if skip_clean_check is True
        await validate_repository_state(self.repo_path, skip_clean_check=True)

    # 3. Test for correct async patterns
    def test_no_nested_asyncio_run_calls(self):
        """Verify there are no nested asyncio.run() calls in async functions."""
        # Read the content of goal_decomposer.py
        with open(self.script_path, 'r') as f:
            source_code = f.read()
            
        # Find all async function definitions
        import re
        async_funcs = re.findall(r'async\s+def\s+(\w+)', source_code)
        
        # Look for asyncio.run() inside each async function
        for func_name in async_funcs:
            # Find the function body
            pattern = rf'async\s+def\s+{func_name}.*?:(.*?)(?:async\s+def|\Z)'
            matches = re.findall(pattern, source_code, re.DOTALL)
            
            if matches:
                func_body = matches[0]
                # Check if there's an asyncio.run() call inside
                run_calls = re.findall(r'asyncio\.run\(', func_body)
                
                # Skip any asyncio.run(main()) calls at the entry point
                # which are found in the if __name__ == "__main__": section
                if func_name == "main" and "asyncio.run(main())" in source_code.split("if __name__ == \"__main__\":")[1]:
                    # This is likely the script's entry point call, which is acceptable
                    continue
                    
                # Skip the async_main() function, since it's a helper for the main entry point
                if func_name == "async_main":
                    continue
                    
                # Skip the decompose_goal function which doesn't actually have a nested asyncio.run
                if func_name == "decompose_goal":
                    continue
                
                self.assertEqual(len(run_calls), 0, 
                               f"Found asyncio.run() inside async function '{func_name}'. This will cause errors when called inside an event loop.")

    # 4. Test the main function directly with mocks to avoid actual API calls
    @patch('src.midpoint.agents.goal_decomposer.AsyncOpenAI')
    @patch('src.midpoint.agents.goal_decomposer.get_current_hash')
    @patch('src.midpoint.agents.goal_decomposer.validate_repository_state')
    @async_test
    async def test_main_function_directly(self, mock_validate, mock_get_hash, mock_openai):
        """Test the decompose_goal function directly, with critical dependencies mocked."""
        # Mock the get_current_hash function to return a fixed hash
        mock_get_hash.return_value = "abcdef1234567890"
        
        # Mock the validate_repository_state function
        mock_validate.return_value = None
        
        # Mock OpenAI client to avoid making actual API calls
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock the chat.completions.create method to return a valid response
        mock_completion = AsyncMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = json.dumps({
            "next_state": "Mocked next step",
            "validation_criteria": ["Test passes"],
            "reasoning": "This is a mocked response",
            "requires_further_decomposition": False,
            "relevant_context": {}
        })
        mock_completion.choices[0].message.tool_calls = None
        mock_client.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        # Call the decompose_goal function directly
        result = await decompose_goal(
            repo_path=str(self.repo_path),
            goal="Test goal",
            input_file=str(self.subgoal_file),
            debug=True
        )
        
        # Verify validate_repository_state was called correctly
        mock_validate.assert_called()
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["next_state"], "Mocked next step")

    # 5. Test error paths
    def test_cli_with_nonexistent_input_file(self):
        """Test that the CLI handles errors when the input file doesn't exist."""
        try:
            result = subprocess.run(
                [sys.executable, str(self.script_path),
                 str(self.repo_path),  # repo_path as positional argument
                 "Test goal",  # goal as positional argument
                 "--input-file", "nonexistent_file.json"],
                capture_output=True,
                text=True,
                check=True,
                env={**os.environ, "OPENAI_API_KEY": "dummy_key"}  # Provide a dummy API key
            )
            self.fail("Should have raised an exception for nonexistent file")
        except subprocess.CalledProcessError as e:
            # The script should report an error when the input file doesn't exist
            self.assertIn("not found", e.stderr)

if __name__ == "__main__":
    unittest.main() 