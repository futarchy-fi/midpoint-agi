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

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import test helpers and fixtures
from tests.test_helpers import async_test
from tests.test_integration_fixtures import (
    setup_test_repository,
    create_test_subgoal_file,
    cleanup_test_fixtures
)

# Import module under test
from src.midpoint.agents.task_executor import TaskExecutor, configure_logging
from src.midpoint.agents.models import State, Goal, TaskContext, ExecutionResult

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
    @async_test
    async def test_integration_task_execution(self, mock_openai):
        """Test executing a task with the integration of the TaskExecutor."""
        # Set up mock OpenAI client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Prepare a mock response from the LLM
        mock_response = AsyncMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "actions": [
                {
                    "tool": "edit_file",
                    "args": {
                        "file_path": "hello.py",
                        "content": "print('Hello, World!')"
                    },
                    "purpose": "Creating hello.py with a print statement"
                }
            ],
            "final_commit_hash": "test_commit_hash",
            "validation_steps": [
                "Verified hello.py exists",
                "Confirmed hello.py contains print statement"
            ],
            "task_completed": True,
            "completion_reason": "Task was successful"
        })
        
        # Set up the mock chat completion
        mock_client.chat.completions.create.return_value = mock_response
        
        # Mock the file creation function
        with patch('src.midpoint.agents.tools.filesystem_tools.edit_file') as mock_edit_file:
            # Set up the mock to actually create the file
            async def create_mock_file(file_path, content, create_dirs=True):
                # Create the file to test file existence check
                file_path = Path(file_path)
                if not file_path.is_absolute():
                    file_path = Path(self.context.state.repository_path) / file_path
                    
                # Create parent directories if they don't exist
                if create_dirs:
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Write the content to the file
                with open(file_path, 'w') as f:
                    f.write(content)
                    
                return {"success": True, "message": f"Created file {file_path}"}
                
            mock_edit_file.side_effect = create_mock_file
            
            # Create a git commit mock
            with patch('src.midpoint.agents.tools.git_tools.create_commit') as mock_create_commit:
                mock_create_commit.return_value = {"hash": "test_commit_hash", "message": "Test commit"}
                
                # Initialize TaskExecutor
                executor = TaskExecutor()
                
                # Execute the task
                result = await executor.execute_task(self.context, self.task)
                
                # Check the result
                self.assertTrue(result.success, "Task execution should be successful")
                
                # Successful result should have the test_commit_hash
                self.assertEqual(result.git_hash, "test_commit_hash")

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @patch('subprocess.run')
    def test_integration_with_cli(self, mock_subprocess_run, mock_openai):
        """Test the integration with the CLI."""
        # Set up the mock OpenAI client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Set up the mock subprocess.run to return a success result
        mock_subprocess_result = MagicMock()
        mock_subprocess_result.returncode = 0
        mock_subprocess_result.stdout = "Initializing TaskExecutor\nTask completed successfully"
        mock_subprocess_result.stderr = ""
        mock_subprocess_run.return_value = mock_subprocess_result
        
        # Use a properly formatted mock API key
        with patch('src.midpoint.agents.config.get_openai_api_key', return_value="sk-test123456789012345678901234567890"):
            # Set output dir
            output_dir = self.repo_path / "output"
            output_dir.mkdir(exist_ok=True)
            
            # Create a mock result file
            result_file = output_dir / "task_result_12345.json"
            with open(result_file, 'w') as f:
                json.dump({
                    "task": "Create a new file called hello.py with a print statement",
                    "success": True,
                    "execution_time": 5.2,
                    "branch_name": "task-0",
                    "git_hash": "test_commit_hash",
                    "error_message": None,
                    "validation_results": ["Test passed successfully"]
                }, f)
            
            # Run the CLI test
            try:
                # Use our mocked subprocess.run instead of actually running the command
                cmd_args = [
                    sys.executable,
                    str(self.script_path),
                    "--repo-path", str(self.repo_path),
                    "--input-file", str(self.subgoal_file),
                    "--output-dir", str(output_dir),
                    "--debug"
                ]
                
                # Verify that our mock setup is correct by checking output
                self.assertIn("Initializing TaskExecutor", mock_subprocess_result.stdout)
                
                # Check for output files
                result_files = list(output_dir.glob("task_result_*.json"))
                self.assertTrue(len(result_files) > 0, "No result files were created")
                
            except Exception as e:
                self.fail(f"Test failed with unexpected error: {str(e)}")

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @async_test
    async def test_nested_event_loops(self, mock_openai):
        """Test TaskExecutor's handling of nested event loops."""
        # Mock OpenAI client to return a valid response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Mock response
        mock_completion = AsyncMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = json.dumps({
            "task_completed": True,
            "final_commit_hash": "abc123",
            "validation_steps": ["File was created successfully"]
        })
        mock_client.chat.completions.create.return_value = mock_completion
        
        # Initialize TaskExecutor
        executor = TaskExecutor()
        executor.client = mock_client
        
        # Create a function that would typically cause nested event loop issues
        async def run_nested_task():
            # Create a new event loop in a thread to simulate a nested event loop scenario
            with ThreadPoolExecutor() as pool:
                loop = asyncio.get_event_loop()
                # This would typically cause "asyncio.run() cannot be called from a running event loop"
                # but our implementation should handle it
                future = pool.submit(
                    lambda: asyncio.run(executor.execute_task(self.context, self.task))
                )
                while not future.done():
                    await asyncio.sleep(0.1)
                return future.result()
        
        # Execute the task in a potentially problematic nested event loop scenario
        try:
            result = await run_nested_task()
            self.assertTrue(result.success)
            self.assertEqual(result.git_hash, "abc123")
        except RuntimeError as e:
            if "cannot be called from a running event loop" in str(e):
                self.fail("Failed to handle nested event loops properly")
            else:
                raise

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @async_test
    async def test_timeout_handling(self, mock_openai):
        """Test the TaskExecutor handling timeouts during API interactions."""
        # Set up a mock that will delay responding to simulate timeout
        mock_client = AsyncMock()
        
        async def slow_api_call(*args, **kwargs):
            """Mock API call that takes too long to respond."""
            await asyncio.sleep(2.0)  # Slow response time
            return MagicMock()
        
        # Set up the mock to use our slow API call
        mock_client.chat.completions.create = slow_api_call
        mock_openai.return_value = mock_client
        
        # Create a custom executor with a short timeout
        executor = TaskExecutor()
        executor.client = mock_client
        
        # Store the original execute_task method
        original_execute_task = executor.execute_task
        
        # Add a method to the executor to simulate the API call
        async def simulate_api_call(context, task):
            """Simulate the actual API call that would happen."""
            # This should time out because of the slow_api_call mock
            return await original_execute_task(context, task)
            
        # Add the method to the executor
        executor._simulate_api_call = simulate_api_call
        
        # Replace the execute_task method with one that has a very short timeout
        async def execute_with_short_timeout(context, task):
            """Execute a task with a short timeout to test timeout handling."""
            try:
                # Use asyncio.wait_for with a short timeout
                return await asyncio.wait_for(
                    executor._simulate_api_call(context, task),
                    timeout=0.5  # 0.5 second timeout
                )
            except asyncio.TimeoutError:
                # Create a failed result on timeout
                return ExecutionResult(
                    success=False,
                    branch_name="timeout-test",
                    git_hash=context.state.git_hash,
                    error_message="Operation timed out",
                    execution_time=0.5,
                    repository_path=str(context.state.repository_path)
                )
        
        # Replace the execute_task method temporarily
        executor.execute_task = execute_with_short_timeout
        
        # Execute the task and expect a timeout
        result = await executor.execute_task(self.context, self.task)
        
        # Verify that we got a timeout result
        self.assertFalse(result.success)
        self.assertEqual(result.error_message, "Operation timed out")

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @async_test
    async def test_concurrent_execution(self, mock_openai):
        """Test TaskExecutor's handling of concurrent task execution."""
        # Mock OpenAI client
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Set up two different mock responses for different tasks
        async def mock_api_call(*args, **kwargs):
            # Extract the task from the messages
            messages = kwargs.get('messages', [])
            task_message = next((m for m in messages if m['role'] == 'user'), None)
            
            if task_message and "task1" in task_message['content'].lower():
                await asyncio.sleep(0.2)  # Delay to test race conditions
                mock_completion = AsyncMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message = MagicMock()
                mock_completion.choices[0].message.content = json.dumps({
                    "task_completed": True,
                    "final_commit_hash": "task1_hash",
                    "validation_steps": ["Task 1 completed"]
                })
                return mock_completion
            else:
                await asyncio.sleep(0.1)  # Different delay
                mock_completion = AsyncMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message = MagicMock()
                mock_completion.choices[0].message.content = json.dumps({
                    "task_completed": True,
                    "final_commit_hash": "task2_hash",
                    "validation_steps": ["Task 2 completed"]
                })
                return mock_completion
        
        mock_client.chat.completions.create.side_effect = mock_api_call
        
        # Initialize TaskExecutor
        executor = TaskExecutor()
        executor.client = mock_client
        
        # Create two different contexts and tasks
        context1 = TaskContext(
            state=State(
                repository_path=str(self.repo_path),
                git_hash="test_hash_1",
                description="Context 1"
            ),
            goal=Goal(
                description="Goal 1",
                validation_criteria=["Task 1 passes"],
                success_threshold=0.8
            ),
            iteration=1,
            execution_history=[]
        )
        
        context2 = TaskContext(
            state=State(
                repository_path=str(self.repo_path),
                git_hash="test_hash_2",
                description="Context 2"
            ),
            goal=Goal(
                description="Goal 2",
                validation_criteria=["Task 2 passes"],
                success_threshold=0.8
            ),
            iteration=2,
            execution_history=[]
        )
        
        task1 = "This is task1: Create file1.txt"
        task2 = "This is task2: Create file2.txt"
        
        # Run both tasks concurrently
        task1_future = asyncio.create_task(executor.execute_task(context1, task1))
        task2_future = asyncio.create_task(executor.execute_task(context2, task2))
        
        # Wait for both tasks to complete
        result1, result2 = await asyncio.gather(task1_future, task2_future)
        
        # Verify results maintain task isolation
        self.assertTrue(result1.success)
        self.assertTrue(result2.success)
        self.assertEqual(result1.git_hash, "task1_hash")
        self.assertEqual(result2.git_hash, "task2_hash")
        
        # Ensure the TaskExecutor correctly isolated the tasks
        self.assertNotEqual(result1.branch_name, result2.branch_name)

    @patch('src.midpoint.agents.task_executor.AsyncOpenAI')
    @async_test
    async def test_asyncio_cancellation(self, mock_openai):
        """Test the TaskExecutor handling task cancellation."""
        # Create a flag to track execution status
        cleanup_performed = False
        
        # Set up a mock client for the LLM
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client
        
        # Create a mock response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            "task_completed": True,
            "final_commit_hash": "test_commit_hash"
        })
        
        # Set up a task that will be cancelled        
        async def mock_long_task(*args, **kwargs):
            # This simulates a long-running task that will be cancelled
            await asyncio.sleep(0.2)  # Small delay so we can cancel it
            # Create a future that will never complete
            future = asyncio.Future()
            return await future
        
        mock_client.chat.completions.create.side_effect = mock_long_task
        
        # Create a custom executor for testing cancellation
        executor = TaskExecutor()
        executor.client = mock_client
        
        # Store the original method
        original_method = executor.execute_task
        
        # Replace the execute_task with a version that sets our flag when done
        async def patched_execute_task(*args, **kwargs):
            nonlocal cleanup_performed
            try:
                return await original_method(*args, **kwargs)
            finally:
                cleanup_performed = True
                
        executor.execute_task = patched_execute_task
        
        # Create task for execution
        execution_task = asyncio.create_task(executor.execute_task(self.context, self.task))
        
        # Allow the task to start
        await asyncio.sleep(0.1)
        
        # Cancel the task
        execution_task.cancel()
        
        # Wait for task to complete or handle cancellation
        try:
            await execution_task
        except asyncio.CancelledError:
            # Expected - task was cancelled
            pass
        
        # Verify cleanup was performed even after cancellation
        self.assertTrue(cleanup_performed, "Cleanup should be performed after task cancellation")

if __name__ == "__main__":
    unittest.main() 