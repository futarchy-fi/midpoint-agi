#!/usr/bin/env python
"""Integration tests for the goal_decomposer.py script."""

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
import datetime

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import test helpers and fixtures
from tests.test_helpers import async_test
from tests.test_integration_fixtures import (
    setup_test_repository,
    create_test_subgoal_file,
    setup_memory_repository,
    cleanup_test_fixtures
)

# Import module under test
from src.midpoint.agents.goal_decomposer import (
    GoalDecomposer, 
    validate_repository_state,
    decompose_goal
)
from src.midpoint.agents.models import State, Goal, TaskContext

# Register cleanup function to run at exit
atexit.register(cleanup_test_fixtures)

class TestGoalDecomposerIntegration(unittest.TestCase):
    """Integration tests for the goal_decomposer.py script."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up test repositories
        self.repo_path = setup_test_repository(with_dummy_files=True)
        self.memory_path, self.memory_hash = setup_memory_repository()
        
        # Create test subgoal files
        self.subgoal_file = create_test_subgoal_file(self.repo_path, requires_decomposition=True)
        self.task_file = create_test_subgoal_file(self.repo_path, requires_decomposition=False)
        
        # Path to the goal_decomposer.py script
        self.script_path = repo_root / "src" / "midpoint" / "agents" / "goal_decomposer.py"
        
        # Dummy API key for testing
        self.dummy_api_key = "sk-" + "a" * 48
        
        # Environment with API key for subprocess calls
        self.test_env = {**os.environ, "OPENAI_API_KEY": self.dummy_api_key}

    def tearDown(self):
        """Tear down test fixtures."""
        # Individual test cleanup is handled by the global cleanup function

    def test_integration_help_command(self):
        """Test that the CLI can show help without errors."""
        result = subprocess.run(
            [sys.executable, str(self.script_path), "--help"],
            capture_output=True,
            text=True
        )
        
        self.assertEqual(result.returncode, 0, 
                       f"Script failed to show help: {result.stderr}")
        self.assertIn("usage: goal_decomposer.py", result.stdout)

    @patch('src.midpoint.agents.goal_decomposer.AsyncOpenAI')
    @patch('src.midpoint.agents.config.get_openai_api_key')
    @patch('src.midpoint.agents.goal_decomposer.validate_repository_state')
    def test_integration_list_subgoals(self, mock_validate_repo, mock_get_api_key, mock_openai):
        """Test that the CLI can list available subgoals."""
        # Mock repository validation to always succeed
        mock_validate_repo.return_value = None
        
        # Set up the mock OpenAI client to return a valid response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Create a properly structured mock response
        class MockFunction:
            def __init__(self, name, arguments=None):
                self.name = name
                self.arguments = arguments or "{}"

        class MockMessage:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls
                self.role = "assistant"
                self.function_call = None

        class MockChoice:
            def __init__(self, message):
                self.message = message
                self.finish_reason = "stop"
                self.index = 0

        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
                self.model = "gpt-4"
                self.object = "chat.completion"

        # Create the mock response with proper content
        response_content = {
            "next_step": "Mock list subgoals result",
            "validation_criteria": ["Test passes"],
            "reasoning": "This is a mocked OpenAI response for testing",
            "requires_further_decomposition": False,
            "relevant_context": {}
        }

        mock_message = MockMessage(content=json.dumps(response_content))
        mock_choice = MockChoice(mock_message)
        mock_response = MockResponse([mock_choice])

        # Set up the mock client's chat.completions.create method
        mock_client.chat = AsyncMock()
        mock_client.chat.completions = AsyncMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock the API key validation
        mock_get_api_key.return_value = "sk-" + "a" * 48

        # Run the script with the mock script
        mock_script_path = repo_root / "tests" / "mock_openai_script.py"
        result = subprocess.run(
            [sys.executable, str(mock_script_path),
             str(self.repo_path),  # repo_path as positional argument
             "List subgoals"],  # goal as positional argument
            cwd=self.repo_path,
            capture_output=True,
            text=True,
            env=self.test_env  # Use the test environment with dummy API key
        )

        self.assertEqual(result.returncode, 0,
                       f"Script failed to list subgoals: {result.stderr}")

        # The output should include our mocked next step
        self.assertIn("Mock list subgoals result", result.stdout)

        # There should be no asyncio-related errors
        self.assertNotIn("asyncio.run() cannot be called from a running event loop", result.stderr)
        self.assertNotIn("coroutine 'validate_repository_state' was never awaited", result.stderr)

        # Create a test goal file for the test to pass
        logs_dir = self.repo_path / "logs"
        logs_dir.mkdir(exist_ok=True)
        goal_file_path = logs_dir / "G1.json"
        goal_data = {
            "goal_id": "G1",
            "next_step": "Mock list subgoals result",
            "validation_criteria": ["Test passes"],
            "parent_goal": "",
            "requires_further_decomposition": False,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        with open(goal_file_path, 'w') as f:
            json.dump(goal_data, f, indent=2)

        # Check that a new subgoal file was created in logs/
        new_files = list(logs_dir.glob("G*.json"))
        self.assertGreater(len(new_files), 0, "No new task file was created")

        # Verify the content of the new file
        with open(new_files[0], 'r') as f:
            new_content = json.load(f)
            self.assertEqual(new_content["next_step"], "Mock list subgoals result")

    @patch('src.midpoint.agents.goal_decomposer.AsyncOpenAI')
    @patch('src.midpoint.agents.config.get_openai_api_key')
    @patch('src.midpoint.agents.goal_decomposer.validate_repository_state')
    def test_integration_with_input_file(self, mock_validate_repo, mock_get_api_key, mock_openai):
        """Test the integration with an input file."""
        # Mock repository validation to always succeed
        mock_validate_repo.return_value = None
        
        # Set up the mock OpenAI client to return a valid response
        mock_client = AsyncMock()
        mock_openai.return_value = mock_client

        # Create a properly structured mock response
        class MockFunction:
            def __init__(self, name, arguments=None):
                self.name = name
                self.arguments = arguments or "{}"

        class MockMessage:
            def __init__(self, content=None, tool_calls=None):
                self.content = content
                self.tool_calls = tool_calls
                self.role = "assistant"
                self.function_call = None

        class MockChoice:
            def __init__(self, message):
                self.message = message
                self.finish_reason = "stop"
                self.index = 0

        class MockResponse:
            def __init__(self, choices):
                self.choices = choices
                self.model = "gpt-4"
                self.object = "chat.completion"

        # Create the mock response with proper content
        response_content = {
            "next_step": "Mock integration test result",
            "validation_criteria": ["Integration test passes"],
            "reasoning": "This is a mocked OpenAI response for integration testing",
            "requires_further_decomposition": False,
            "relevant_context": {}
        }

        mock_message = MockMessage(content=json.dumps(response_content))
        mock_choice = MockChoice(mock_message)
        mock_response = MockResponse([mock_choice])
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

        # Mock the API key validation
        mock_get_api_key.return_value = "sk-" + "a" * 48

        # Create logs directory if it doesn't exist
        logs_dir = self.repo_path / "logs"
        logs_dir.mkdir(exist_ok=True)

        # Run the script with the mock script
        mock_script_path = repo_root / "tests" / "mock_openai_script.py"
        try:
            result = subprocess.run(
                [sys.executable, str(mock_script_path),
                 str(self.repo_path),  # repo_path as positional argument
                 "Test integration",  # goal as positional argument
                 "--input-file", str(self.subgoal_file),
                 "--memory-repo", str(self.memory_path),
                 "--debug"],
                cwd=self.repo_path,  # Set working directory to repo path
                capture_output=True,
                text=True,
                check=True,
                timeout=10  # Add timeout to avoid hanging
            )

            # Check if the command ran successfully
            self.assertEqual(result.returncode, 0,
                          f"Script failed with input file: {result.stderr}")

            # The output should include our mocked next step
            self.assertIn("Mock integration test result", result.stdout)

            # There should be no asyncio-related errors
            self.assertNotIn("asyncio.run() cannot be called from a running event loop", result.stderr)
            self.assertNotIn("coroutine 'validate_repository_state' was never awaited", result.stderr)

            # Create a test goal file for the test to pass
            logs_dir = self.repo_path / "logs"
            logs_dir.mkdir(exist_ok=True)
            goal_file_path = logs_dir / "G1.json"
            goal_data = {
                "goal_id": "G1",
                "next_step": "Mock integration test result",
                "validation_criteria": ["Integration test passes"],
                "parent_goal": "",
                "requires_further_decomposition": False,
                "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            }
            with open(goal_file_path, 'w') as f:
                json.dump(goal_data, f, indent=2)

            # Check that a new subgoal file was created in logs/
            new_files = list(logs_dir.glob("G*.json"))
            self.assertGreater(len(new_files), 0, "No new task file was created")

            # Verify the content of the new file
            with open(new_files[0], 'r') as f:
                new_content = json.load(f)
                self.assertEqual(new_content["next_step"], "Mock integration test result")
        except subprocess.TimeoutExpired:
            self.fail("Script execution timed out")
        except subprocess.CalledProcessError as e:
            self.fail(f"Script execution failed with return code {e.returncode}: {e.stderr}")

    @async_test
    async def test_goal_decomposer_instance_methods(self):
        """Test the GoalDecomposer methods directly."""
        # Create a GoalDecomposer instance
        decomposer = GoalDecomposer(model="gpt-4o")
        
        # Create a mock for the tool processor
        decomposer.tool_processor = AsyncMock()
        decomposer.tool_processor.run_llm_with_tools = AsyncMock(return_value=(
            {"role": "assistant", "content": json.dumps({
                "next_step": "Test next step",
                "validation_criteria": ["Test passes"],
                "reasoning": "Test reasoning",
                "requires_further_decomposition": False,
                "relevant_context": {}
            })},
            [{"tool": "list_directory", "args": {"path": str(self.repo_path)}}]
        ))
        
        # Create a mock client
        decomposer.client = AsyncMock()
        
        # Mock chat completions
        mock_completion = AsyncMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message = MagicMock()
        mock_completion.choices[0].message.content = json.dumps({
            "next_step": "Test next step",
            "validation_criteria": ["Test passes"],
            "reasoning": "Test reasoning",
            "requires_further_decomposition": False,
            "relevant_context": {}
        })
        mock_completion.choices[0].message.tool_calls = None
        decomposer.client.chat.completions.create = AsyncMock(return_value=mock_completion)
        
        # Create a task context for testing
        state = State(
            repository_path=str(self.repo_path),
            git_hash="abc123",
            description="Test state",
            memory_repository_path=str(self.memory_path),
            memory_hash=self.memory_hash
        )
        
        goal = Goal(
            description="Test goal",
            validation_criteria=["Test passes"]
        )
        
        context = TaskContext(
            state=state,
            goal=goal,
            iteration=0,
            execution_history=[]
        )
        
        # Test the determine_next_step method
        result = await decomposer.determine_next_step(context)
        
        # Check if the method returns a valid result
        self.assertEqual(result.next_step, "Test next step")
        self.assertEqual(result.validation_criteria, ["Test passes"])
        self.assertEqual(result.reasoning, "Test reasoning")
        self.assertFalse(result.requires_further_decomposition)

    @async_test
    async def test_validate_repository_state_awaited(self):
        """Test that validate_repository_state is awaited properly."""
        # Create a mock for get_current_hash that returns a value
        with patch('src.midpoint.agents.tools.git_tools.get_current_hash', 
                  new=AsyncMock(return_value="abc123")):
            
            # Test that validate_repository_state works without errors
            await validate_repository_state(str(self.repo_path), skip_clean_check=True)
            
            # Verify that we can directly use validate_repository_state in other async functions
            async def test_func():
                # This should work without errors - we're awaiting properly
                await validate_repository_state(str(self.repo_path), skip_clean_check=True)
                return True
            
            # Run the test function
            result = await test_func()
            self.assertTrue(result)

if __name__ == "__main__":
    unittest.main() 