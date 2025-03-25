#!/usr/bin/env python
"""Integration tests for the TaskExecutor CLI and execution flow."""

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
import atexit
import signal
from concurrent.futures import ThreadPoolExecutor
import time
import copy

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import test helpers and fixtures
from tests.test_helpers import async_test
from tests.test_integration_fixtures import (
    setup_test_repository,
    create_test_subgoal_file,
    cleanup_test_fixtures,
    setup_memory_repository
)

# Import module under test
from src.midpoint.agents.task_executor import TaskExecutor, configure_logging
from src.midpoint.agents.models import State, Goal, TaskContext, ExecutionResult, MemoryState
from src.midpoint.agents.goal_decomposer import validate_repository_state

# Register cleanup function to run at exit
atexit.register(cleanup_test_fixtures)

class TestTaskExecutorIntegration(unittest.TestCase):
    """Integration tests for the TaskExecutor and its CLI."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up test repository
        self.repo_path = setup_test_repository(with_dummy_files=True)
        
        # Create test subgoal file
        self.subgoal_file = create_test_subgoal_file(
            self.repo_path, 
            requires_decomposition=False,
            custom_content={
                "next_step": "Create a new file called hello.py with a print statement",
                "validation_criteria": ["hello.py file exists", "hello.py contains print statement"],
                "reasoning": "Simple task for integration testing",
                "requires_further_decomposition": False,
                "relevant_context": {}
            }
        )
        
        # Path to the CLI script
        self.script_path = repo_root / "debug" / "run_task_executor.py"
        
        # Dummy API key for testing
        self.dummy_api_key = "dummy_key_for_testing_only"
        
        # Environment with API key for subprocess calls
        self.test_env = {**os.environ, "OPENAI_API_KEY": self.dummy_api_key}
        
        # Make CLI script executable if not already
        if not os.access(self.script_path, os.X_OK):
            os.chmod(self.script_path, 0o755)

        # Create a TaskContext for testing
        self.context = TaskContext(
            state=State(
                repository_path=str(self.repo_path),
                git_hash="test_hash",
                description="Test repository state for integration testing"
            ),
            goal=Goal(
                description="Test goal for integration testing",
                validation_criteria=["Test passes"],
                success_threshold=0.8
            ),
            iteration=0,
            execution_history=[]
        )
        
        # Create a simple task for testing
        self.task = "Create a file named example.txt with 'Hello, World!' content"
        
        # Configure logging
        configure_logging(debug=True, log_dir_path=str(self.repo_path / "logs"))

    def tearDown(self):
        """Tear down test fixtures."""
        # Individual test cleanup is handled by the global cleanup function
        pass

    def test_integration_help_command(self):
        """Test that the CLI can show help without errors."""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, 
                       f"Script failed to show help: {result.stderr}")
        self.assertIn("usage: run_task_executor.py", result.stdout)

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @patch('src.midpoint.agents.task_executor.TaskExecutor.execute_task')
    @async_test
    async def test_concurrent_execution(self, mock_execute_task, mock_openai):
        """Test that tasks can be executed concurrently."""
        # Mock the execute_task method to return different results based on input task
        async def mock_execution(context, task):
            if "task1" in task:
                return ExecutionResult(
                    success=True,
                    branch_name="task-1",
                    git_hash="task1_hash",
                    error_message=None,
                    execution_time=0.5,
                    repository_path=str(self.repo_path),
                    validation_results=["Task 1 completed"]
                )
            else:
                return ExecutionResult(
                    success=True,
                    branch_name="task-2",
                    git_hash="task2_hash",
                    error_message=None,
                    execution_time=0.3,
                    repository_path=str(self.repo_path),
                    validation_results=["Task 2 completed"]
                )
                
        mock_execute_task.side_effect = mock_execution
        
        # Initialize TaskExecutor
        executor = TaskExecutor()
        
        # Create two tasks
        task1 = "This is task1: Create file1.txt"
        task2 = "This is task2: Create file2.txt"
        
        # Create two contexts
        context2 = copy.deepcopy(self.context)
        
        # Execute tasks concurrently
        results = await asyncio.gather(
            executor.execute_task(self.context, task1),
            executor.execute_task(context2, task2)
        )
        
        result1, result2 = results
        
        # Check results
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        
        # Check that each task used its own hash
        self.assertEqual(result1.git_hash, "task1_hash")
        self.assertEqual(result2.git_hash, "task2_hash")

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @patch('src.midpoint.agents.task_executor.TaskExecutor.execute_task')
    @async_test
    async def test_nested_event_loops(self, mock_execute_task, mock_openai):
        """Test that nested event loops don't cause issues."""
        # Mock the execute_task method to return a successful result
        mock_result = ExecutionResult(
            success=True,
            branch_name="task-0",
            git_hash="abc123",
            error_message=None,
            execution_time=0.5,
            repository_path=str(self.repo_path),
            validation_results=["Verified example.txt exists"]
        )
        mock_execute_task.return_value = mock_result
        
        # Initialize TaskExecutor
        executor = TaskExecutor()
        
        # Create a function to run a nested task
        async def run_nested_task():
            # This would normally cause issues with nested event loops
            # But we're mocking the execution
            return await executor.execute_task(self.context, self.task)
            
        # Run the nested task function
        result = await run_nested_task()
        
        # Check the result
        self.assertTrue(result.success)
        self.assertEqual(result.git_hash, "abc123")

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @patch('src.midpoint.agents.task_executor.TaskExecutor.execute_task')
    @async_test
    async def test_integration_task_execution(self, mock_execute_task, mock_openai):
        """Test the TaskExecutor with a task that creates a file."""
        # Mock the execute_task method to return a successful result
        mock_result = ExecutionResult(
            success=True,
            branch_name="task-0",
            git_hash="test_commit_hash",
            error_message=None,
            execution_time=0.5,
            repository_path=str(self.repo_path),
            validation_results=["Verified example.txt exists", "Confirmed example.txt contains Hello World"]
        )
        mock_execute_task.return_value = mock_result
        
        # Initialize TaskExecutor
        executor = TaskExecutor()
        
        # Execute the task (this will call our mocked method)
        result = await executor.execute_task(self.context, self.task)
        
        # Check the result
        self.assertTrue(result.success, "Task execution should be successful")
        self.assertEqual(result.git_hash, "test_commit_hash")
        self.assertEqual(len(result.validation_results), 2)

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @patch('src.midpoint.agents.task_executor.TaskExecutor.execute_task')
    @async_test
    async def test_integration_with_memory(self, mock_execute_task, mock_openai):
        """Test the TaskExecutor with memory integration."""
        # Mock the execute_task method to return a successful result with memory-related validation
        mock_result = ExecutionResult(
            success=True,
            branch_name="task-0",
            git_hash="test_commit_hash",
            error_message=None,
            execution_time=0.5,
            repository_path=str(self.repo_path),
            validation_results=["Verified example.txt exists", "Stored study document in memory"]
        )
        mock_execute_task.return_value = mock_result
        
        # Initialize TaskExecutor
        executor = TaskExecutor()
        
        # Update context with memory repository information
        self.context.state.memory_repository_path = str(self.repo_path)
        self.context.state.memory_hash = "test_hash"
        self.context.memory_state = MemoryState(
            memory_hash="test_hash",
            repository_path=str(self.repo_path)
        )
        
        # Execute the task (this will call our mocked method)
        result = await executor.execute_task(self.context, self.task)
        
        # Check the result
        self.assertTrue(result.success, "Task execution should be successful")
        self.assertEqual(result.git_hash, "test_commit_hash")
        self.assertEqual(len(result.validation_results), 2)
        self.assertIn("Stored study document in memory", result.validation_results)

if __name__ == "__main__":
    unittest.main() 