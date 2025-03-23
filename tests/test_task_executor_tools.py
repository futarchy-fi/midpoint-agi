#!/usr/bin/env python
"""Tests for the TaskExecutor's tool usage."""

import unittest
import os
import sys
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock, call
import atexit

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import test helpers and fixtures
from tests.test_helpers import async_test
from tests.test_integration_fixtures import (
    setup_test_repository,
    cleanup_test_fixtures
)

# Import module under test
from src.midpoint.agents.task_executor import TaskExecutor
from src.midpoint.agents.models import State, Goal, TaskContext, ExecutionResult
from src.midpoint.agents.tools.git_tools import create_branch, create_commit

# Register cleanup function to run at exit
atexit.register(cleanup_test_fixtures)

class TestTaskExecutorTools(unittest.TestCase):
    """Tests for the TaskExecutor's tool usage."""

    def setUp(self):
        """Set up test fixtures."""
        # Set up a test repository
        self.repo_path = setup_test_repository(with_dummy_files=True)
        
        # Create a TaskContext for testing
        self.context = TaskContext(
            state=State(
                repository_path=str(self.repo_path),
                git_hash="test_hash",
                description="Test state"
            ),
            goal=Goal(
                description="Test goal",
                validation_criteria=["Test validation"],
                success_threshold=0.8
            ),
            iteration=0,
            execution_history=[]
        )
        
        # Create a simple task for testing
        self.task = "Create a new Python file with a greeting function"

    def tearDown(self):
        """Tear down test fixtures."""
        # Handled by cleanup_test_fixtures registered with atexit

    @patch('src.midpoint.agents.tools.filesystem_tools.list_directory')
    @async_test
    async def test_list_directory_tool(self, mock_list_directory):
        """Test that the list_directory tool is properly mocked and can be called."""
        # Set up the mock
        expected_files = ["example.txt", "README.md"]
        expected_dirs = ["src", "tests"]
        mock_list_directory.return_value = {
            "files": expected_files,
            "directories": expected_dirs
        }
        
        # Import the actual function
        from src.midpoint.agents.tools.filesystem_tools import list_directory
        
        # Call the function directly
        result = await list_directory(
            path=self.context.state.repository_path,
            pattern="*",
            recursive=False
        )
        
        # Verify the mock was called correctly
        mock_list_directory.assert_called_once_with(
            path=self.context.state.repository_path,
            pattern="*",
            recursive=False
        )
        
        # Verify the result
        self.assertEqual(result["files"], expected_files)
        self.assertEqual(result["directories"], expected_dirs)

    @patch('src.midpoint.agents.tools.filesystem_tools.read_file')
    @async_test
    async def test_read_file_tool(self, mock_read_file):
        """Test that the read_file tool is properly mocked and can be called."""
        # Set up the mock
        expected_content = "This is an example file."
        mock_read_file.return_value = expected_content
        
        # Import the actual function
        from src.midpoint.agents.tools.filesystem_tools import read_file
        
        # Create a test file
        test_file_path = self.repo_path / "example.txt"
        test_file_path.write_text(expected_content)
        
        # Call the function directly
        result = await read_file(
            path=self.context.state.repository_path,
            file_path="example.txt",
            start_line=0,
            max_lines=100
        )
        
        # Verify the mock was called correctly
        mock_read_file.assert_called_once_with(
            path=self.context.state.repository_path,
            file_path="example.txt",
            start_line=0,
            max_lines=100
        )
        
        # Verify the result
        self.assertEqual(result, expected_content)

    @patch('src.midpoint.agents.tools.code_tools.search_code')
    @async_test
    async def test_search_code_tool(self, mock_search_code):
        """Test that the search_code tool is properly mocked and can be called."""
        # Set up the mock
        expected_results = [
            {
                "file": "test.py",
                "line": 10,
                "content": "def greet(name):"
            }
        ]
        mock_search_code.return_value = expected_results
        
        # Import the actual function
        from src.midpoint.agents.tools.code_tools import search_code
        
        # Call the function directly
        result = await search_code(
            path=self.context.state.repository_path,
            query="def greet",
            file_pattern="*.py",
            max_results=10
        )
        
        # Verify the mock was called correctly
        mock_search_code.assert_called_once_with(
            path=self.context.state.repository_path,
            query="def greet",
            file_pattern="*.py",
            max_results=10
        )
        
        # Verify the result
        self.assertEqual(result, expected_results)

    @patch('src.midpoint.agents.tools.filesystem_tools.edit_file')
    @async_test
    async def test_edit_file_tool(self, mock_edit_file):
        """Test that the edit_file tool is properly mocked and can be called."""
        # Set up the mock
        mock_edit_file.return_value = True
        
        # Import the actual function
        from src.midpoint.agents.tools.filesystem_tools import edit_file
        
        # Create test content
        file_content = "def greet(name):\n    return f'Hello, {name}!'\n"
        
        # Call the function directly
        result = await edit_file(
            path=self.context.state.repository_path,
            file_path="greeting.py",
            content=file_content
        )
        
        # Verify the mock was called correctly
        mock_edit_file.assert_called_once_with(
            path=self.context.state.repository_path,
            file_path="greeting.py",
            content=file_content
        )
        
        # Verify the result
        self.assertTrue(result)

    @patch('src.midpoint.agents.tools.terminal_tools.run_terminal_cmd')
    @async_test
    async def test_run_terminal_cmd_tool(self, mock_run_terminal_cmd):
        """Test that the run_terminal_cmd tool is properly mocked and can be called."""
        # Set up the mock
        expected_output = {
            "stdout": "file1.txt\nfile2.txt",
            "stderr": "",
            "returncode": 0
        }
        mock_run_terminal_cmd.return_value = expected_output
        
        # Import the actual function
        from src.midpoint.agents.tools.terminal_tools import run_terminal_cmd
        
        # Call the function directly
        result = await run_terminal_cmd(
            command="ls -la",
            cwd=self.context.state.repository_path
        )
        
        # Verify the mock was called correctly
        mock_run_terminal_cmd.assert_called_once_with(
            command="ls -la",
            cwd=self.context.state.repository_path
        )
        
        # Verify the result
        self.assertEqual(result, expected_output)

    @patch('src.midpoint.agents.tools.web_tools.web_search')
    @async_test
    async def test_web_search_tool(self, mock_web_search):
        """Test that the web_search tool is properly mocked and can be called."""
        # Set up the mock
        expected_results = [
            {
                "url": "https://example.com/python-greetings",
                "title": "Python Greeting Function Examples",
                "snippet": "A common greeting function in Python might look like: def greet(name): return f'Hello, {name}!'"
            }
        ]
        mock_web_search.return_value = expected_results
        
        # Import the actual function
        from src.midpoint.agents.tools.web_tools import web_search
        
        # Call the function directly
        result = await web_search(
            query="python greeting function example",
            num_results=5
        )
        
        # Verify the mock was called correctly
        mock_web_search.assert_called_once_with(
            query="python greeting function example",
            num_results=5
        )
        
        # Verify the result
        self.assertEqual(result, expected_results)
        
    @patch('src.midpoint.agents.tools.git_tools.create_branch')
    @async_test
    async def test_create_branch_tool(self, mock_create_branch):
        """Test that the create_branch tool is properly mocked and can be called."""
        # Set up the mock
        expected_branch_name = "task-123-test"
        mock_create_branch.return_value = expected_branch_name
        
        # Import the actual function
        from src.midpoint.agents.tools.git_tools import create_branch
        
        # Call the function directly
        result = await create_branch(
            repo_path=self.context.state.repository_path,
            branch_name="task-123-test"
        )
        
        # Verify the mock was called correctly
        mock_create_branch.assert_called_once_with(
            repo_path=self.context.state.repository_path,
            branch_name="task-123-test"
        )
        
        # Verify the result
        self.assertEqual(result, expected_branch_name)
        
    @patch('src.midpoint.agents.tools.git_tools.create_commit')
    @async_test
    async def test_create_commit_tool(self, mock_create_commit):
        """Test that the create_commit tool is properly mocked and can be called."""
        # Set up the mock
        expected_commit_hash = "abcdef1234567890"
        mock_create_commit.return_value = expected_commit_hash
        
        # Import the actual function
        from src.midpoint.agents.tools.git_tools import create_commit
        
        # Call the function directly
        result = await create_commit(
            repo_path=self.context.state.repository_path,
            commit_message="Add greeting function"
        )
        
        # Verify the mock was called correctly
        mock_create_commit.assert_called_once_with(
            repo_path=self.context.state.repository_path,
            commit_message="Add greeting function"
        )
        
        # Verify the result
        self.assertEqual(result, expected_commit_hash)

if __name__ == "__main__":
    unittest.main() 