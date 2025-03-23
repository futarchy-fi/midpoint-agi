#!/usr/bin/env python
"""Unit tests for the TaskExecutor CLI script."""

import unittest
import os
import sys
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import the script under test
debug_dir = repo_root / "debug"
sys.path.append(str(debug_dir))
import run_task_executor

# Import test helpers
from tests.test_helpers import async_test
from tests.test_integration_fixtures import (
    setup_test_repository,
    cleanup_test_fixtures
)

from src.midpoint.agents.models import ExecutionResult, TaskContext
from src.midpoint.agents.task_executor import TaskExecutor

class TestTaskExecutorCLI(unittest.TestCase):
    """Tests for the TaskExecutor CLI script."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up test repository
        self.repo_path = setup_test_repository(with_dummy_files=True)
        
        # Create test output directory
        self.output_dir = self.repo_path / "test_output"
        self.output_dir.mkdir(exist_ok=True)
        
        # Create a test subgoal file
        self.subgoal_file = self.repo_path / "test_subgoal.json"
        with open(self.subgoal_file, 'w') as f:
            json.dump({
                "next_step": "Add a print statement to main.py",
                "validation_criteria": ["main.py contains a print statement"],
                "reasoning": "This is a test task",
                "requires_further_decomposition": False,
                "relevant_context": {}
            }, f, indent=2)
        
        # Create an invalid subgoal file (missing required fields)
        self.invalid_subgoal_file = self.repo_path / "invalid_subgoal.json"
        with open(self.invalid_subgoal_file, 'w') as f:
            json.dump({
                "reasoning": "This is a test task with missing fields",
                "relevant_context": {}
            }, f, indent=2)
        
        # Create a complex subgoal file (with multiple steps)
        self.complex_subgoal_file = self.repo_path / "complex_subgoal.json"
        with open(self.complex_subgoal_file, 'w') as f:
            json.dump({
                "next_step": "Implement a logger class with multiple log levels",
                "validation_criteria": [
                    "Create a Logger class with INFO, WARNING, ERROR log levels",
                    "Include timestamp and log level in log messages",
                    "Add option to log to file or console",
                    "Implement method to configure log level at runtime"
                ],
                "reasoning": "A more complex task that requires multiple implementation steps",
                "requires_further_decomposition": True,
                "relevant_context": {
                    "existing_files": ["utils.py", "config.py"],
                    "dependencies": ["datetime", "sys"]
                }
            }, f, indent=2)
            
        # Path to the CLI script
        self.script_path = repo_root / "debug" / "run_task_executor.py"
        
        # Dummy API key for testing
        self.dummy_api_key = "dummy_key_for_testing_only"
        
        # Environment with API key for subprocess calls
        self.test_env = {**os.environ, "OPENAI_API_KEY": self.dummy_api_key}

    def tearDown(self):
        """Tear down test fixtures."""
        # Clean up is handled by the global cleanup function
        pass

    def test_cli_help_command(self):
        """Test that the CLI can show help without errors."""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, 
                         f"Script failed to show help: {result.stderr}")
        self.assertIn("Run the TaskExecutor with specified parameters", result.stdout)

    def test_missing_arguments(self):
        """Test that the CLI properly errors on missing arguments."""
        # Test without repo path
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--task", "Test task"],
            capture_output=True,
            text=True
        )
        
        self.assertNotEqual(result.returncode, 0, 
                           "Script should fail when repo path is missing")
        self.assertIn("error: the following arguments are required: --repo-path", result.stderr)
        
        # Test without task or input file
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--repo-path", str(self.repo_path)],
            capture_output=True,
            text=True
        )
        
        self.assertNotEqual(result.returncode, 0,
                           "Script should fail when both task and input file are missing")
        self.assertIn("one of the arguments --task --input-file is required", result.stderr)

    @patch('debug.run_task_executor.TaskExecutor')
    @patch('debug.run_task_executor.get_current_hash')
    @patch('debug.run_task_executor.get_current_branch')
    @async_test
    async def test_invalid_subgoal_file(self, mock_get_branch, mock_get_hash, mock_task_executor_class):
        """Test handling of invalid subgoal file (missing required fields)."""
        # Set up mocks
        mock_get_branch.return_value = "main"
        mock_get_hash.return_value = "abc123"
        
        # Create an invalid subgoal file
        with open(self.invalid_subgoal_file, 'r') as f:
            invalid_data = json.load(f)
            
        # Verify the test file is actually invalid (missing next_step)
        self.assertNotIn("next_step", invalid_data)
        
        # Test with subprocess directly so we can capture exit code
        process = subprocess.Popen(
            [
                sys.executable, 
                str(self.script_path), 
                "--repo-path", str(self.repo_path),
                "--input-file", str(self.invalid_subgoal_file)
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        # Check that the process exited with an error code
        self.assertNotEqual(process.returncode, 0, "Script should fail with invalid input file")
        
        # Check that we got the expected error message
        self.assertIn("missing 'next_step' field", stdout + stderr)

    @patch('midpoint.agents.task_executor.TaskExecutor.execute_task')
    @patch('debug.run_task_executor.get_current_hash')
    @patch('debug.run_task_executor.get_current_branch')
    @async_test
    async def test_complex_task_execution(self, mock_get_branch, mock_get_hash, mock_execute_task):
        """Test executing a complex task with multiple validation criteria."""
        # Set up mocks
        mock_get_branch.return_value = "main"
        mock_get_hash.return_value = "abc123"
        
        # Set up the mock result
        mock_execution_result = ExecutionResult(
            success=True,
            branch_name={'branch_name': 'task-0', 'message': ''},
            git_hash="def456",
            error_message=None,
            execution_time=10.5,
            repository_path=str(self.repo_path),
            validation_results=[
                "Created Logger class with multiple log levels",
                "Added timestamp and log level formatting",
                "Implemented file and console logging",
                "Added runtime log level configuration"
            ]
        )
        mock_execute_task.return_value = mock_execution_result
        
        # Create arguments for the run_executor function
        class Args:
            repo_path = str(self.repo_path)
            task = None
            input_file = str(self.complex_subgoal_file)
            goal = "Implement better logging"
            output_dir = str(self.output_dir)
            debug = True
            quiet = False
            no_commit = False
        
        # Run the executor function
        result = await run_task_executor.run_executor(Args())
        
        # Check that the task completed successfully
        self.assertEqual(result.success, mock_execution_result.success)
        
        # Verify that execute_task was called
        mock_execute_task.assert_called_once()
        
        # Extract the arguments passed to execute_task and verify them
        task_context = mock_execute_task.call_args[0][0]
        task_description = mock_execute_task.call_args[0][1]
        
        # Check that the task description was extracted from the file
        with open(self.complex_subgoal_file, 'r') as f:
            complex_data = json.load(f)
        self.assertEqual(task_description, complex_data["next_step"])
        
        # Check that the validation criteria were extracted
        self.assertEqual(
            task_context.goal.validation_criteria, 
            complex_data["validation_criteria"]
        )
        
        # Check that the goal description was set from args
        self.assertEqual(task_context.goal.description, "Implement better logging")

    @patch('midpoint.agents.task_executor.TaskExecutor.execute_task')
    @patch('debug.run_task_executor.get_current_hash')
    @patch('debug.run_task_executor.get_current_branch')
    @async_test
    async def test_executor_function(self, mock_get_branch, mock_get_hash, mock_execute_task):
        """Test that the executor function works correctly."""
        # Set up mocks
        mock_get_branch.return_value = "main"
        mock_get_hash.return_value = "abc123"
        
        # Set up the mock result
        mock_result = ExecutionResult(
            success=True,
            branch_name={'branch_name': 'task-0', 'message': ''},
            git_hash="def456",
            error_message=None,
            execution_time=10.5,
            repository_path=str(self.repo_path),
            validation_results=["Test validation passed"]
        )
        mock_execute_task.return_value = mock_result
        
        # Create arguments for the run_executor function
        class Args:
            repo_path = str(self.repo_path)
            task = "Test task"
            input_file = None
            goal = "Test goal"
            output_dir = str(self.output_dir)
            debug = False
            quiet = False
            no_commit = False
        
        # Run the executor function
        result = await run_task_executor.run_executor(Args())
        
        # Check the success status
        self.assertEqual(result.success, mock_result.success)
        
        # Verify that execute_task was called
        mock_execute_task.assert_called_once()
        
        # Extract the arguments passed to execute_task
        task_context = mock_execute_task.call_args[0][0]
        task_description = mock_execute_task.call_args[0][1]
        
        # Check that the task context was created correctly
        self.assertEqual(task_context.state.repository_path, str(self.repo_path))
        self.assertEqual(task_context.goal.description, "Test goal")
        
        # Check that the task description was passed correctly
        self.assertEqual(task_description, "Test task")

    @patch('midpoint.agents.task_executor.TaskExecutor.execute_task')
    @patch('debug.run_task_executor.get_current_hash')
    @patch('debug.run_task_executor.get_current_branch')
    @async_test
    async def test_loading_from_input_file(self, mock_get_branch, mock_get_hash, mock_execute_task):
        """Test that the executor function properly loads from an input file."""
        # Set up mocks
        mock_get_branch.return_value = "main"
        mock_get_hash.return_value = "abc123"
        
        # Set up the mock result
        mock_result = ExecutionResult(
            success=True,
            branch_name={'branch_name': 'task-0', 'message': ''},
            git_hash="def456",
            error_message=None,
            execution_time=10.5,
            repository_path=str(self.repo_path),
            validation_results=["Test validation passed"]
        )
        mock_execute_task.return_value = mock_result
        
        # Create arguments for the run_executor function
        class Args:
            repo_path = str(self.repo_path)
            task = None
            input_file = str(self.subgoal_file)
            goal = None
            output_dir = str(self.output_dir)
            debug = False
            quiet = False
            no_commit = False
        
        # Run the executor function
        result = await run_task_executor.run_executor(Args())
        
        # Check the success status
        self.assertEqual(result.success, mock_result.success)
        
        # Verify that execute_task was called once
        mock_execute_task.assert_called_once()
        
        # Extract the arguments passed to execute_task
        task_context = mock_execute_task.call_args[0][0]
        task_description = mock_execute_task.call_args[0][1]
        
        # Read the expected task from the subgoal file
        with open(self.subgoal_file, 'r') as f:
            subgoal_data = json.load(f)
        
        # Check that the task description was extracted correctly
        self.assertEqual(task_description, subgoal_data['next_step'])
        
        # Check that the validation criteria were added to the context
        self.assertEqual(
            task_context.goal.validation_criteria, 
            subgoal_data['validation_criteria']
        )

if __name__ == "__main__":
    unittest.main() 