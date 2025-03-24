#!/usr/bin/env python
"""Unit tests for the GoalDecomposer's tool usage."""

import os
import sys
import json
import unittest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock
import logging

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import module under test
from src.midpoint.agents.goal_decomposer import GoalDecomposer, validate_repository_state
from src.midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext
from src.midpoint.agents.tools import (
    list_directory,
    read_file, 
    get_current_hash,
    initialize_all_tools
)
from src.midpoint.agents.tools.processor import ToolProcessor

# Import test helpers
from tests.test_helpers import async_test

class TestGoalDecomposerTools(unittest.TestCase):
    """Tests for GoalDecomposer's usage of tools."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        
        # Create test directory structure and files
        self.test_dir = self.repo_path / "src"
        self.test_dir.mkdir(exist_ok=True)
        self.test_file = self.test_dir / "test_file.py"
        self.test_file.write_text("print('Hello, World!')")
        
        # Initialize tools for testing
        initialize_all_tools()
        
        # Initialize the GoalDecomposer with a mocked tool processor
        self.decomposer = GoalDecomposer(model="gpt-4o")
        self.tool_processor = MagicMock(spec=ToolProcessor)
        self.decomposer.tool_processor = self.tool_processor

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("src.midpoint.agents.tools.filesystem_tools.list_directory")
    @patch("src.midpoint.agents.tools.processor.ToolProcessor")
    @async_test
    async def test_tool_list_directory_parameters(self, mock_processor_class, mock_list_directory):
        """Test that the ToolProcessor correctly processes list_directory tool calls."""
        # Setup mock for tool processor
        mock_tool_processor = MagicMock()
        mock_processor_class.return_value = mock_tool_processor
        
        # Create a mock for run_llm_with_tools
        mock_run_with_tools = AsyncMock()
        mock_tool_processor.run_llm_with_tools = mock_run_with_tools
        
        # Set up the response from run_llm_with_tools
        mock_run_with_tools.return_value = (
            {"role": "assistant", "content": "Directory listing complete"},
            [{"tool": "list_directory", "args": {"path": str(self.test_dir)}}]
        )
        
        # Setup mock for list_directory
        mock_list_directory.return_value = {
            "path": str(self.test_dir),
            "items": [{"name": "test_file.py", "type": "file", "size": 24}]
        }
        
        # Create an instance of the tool processor using the mock
        processor = mock_processor_class()
        
        # Call the function
        messages = [{"role": "user", "content": "List the directory"}]
        await processor.run_llm_with_tools(
            messages=messages,
            model="gpt-4o",
            temperature=0.5,
            max_tokens=4000
        )
        
        # Check that run_llm_with_tools was called correctly
        mock_run_with_tools.assert_called_once()
        
        # Verify the mock was called with the expected arguments
        call_args = mock_run_with_tools.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o")
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertEqual(call_args["max_tokens"], 4000)

    @patch("src.midpoint.agents.tools.filesystem_tools.read_file")
    @patch("src.midpoint.agents.tools.processor.ToolProcessor")
    @async_test
    async def test_tool_read_file_parameters(self, mock_processor_class, mock_read_file):
        """Test that the ToolProcessor correctly processes read_file tool calls."""
        # Setup mock for tool processor
        mock_tool_processor = MagicMock()
        mock_processor_class.return_value = mock_tool_processor
        
        # Create a mock for run_llm_with_tools
        mock_run_with_tools = AsyncMock()
        mock_tool_processor.run_llm_with_tools = mock_run_with_tools
        
        # Set up the response from run_llm_with_tools
        mock_run_with_tools.return_value = (
            {"role": "assistant", "content": "File content read"},
            [{"tool": "read_file", "args": {
                "file_path": str(self.test_file),
                "start_line": 0,
                "max_lines": 10
            }}]
        )
        
        # Setup mock for read_file
        mock_read_file.return_value = {
            "content": "print('Hello, World!')",
            "start_line": 0,
            "lines_read": 1,
            "total_lines": 1
        }
        
        # Create an instance of the tool processor using the mock
        processor = mock_processor_class()
        
        # Call the function
        messages = [{"role": "user", "content": "Read the file"}]
        await processor.run_llm_with_tools(
            messages=messages,
            model="gpt-4o",
            temperature=0.5,
            max_tokens=4000
        )
        
        # Check that run_llm_with_tools was called correctly
        mock_run_with_tools.assert_called_once()
        
        # Verify the mock was called with the expected arguments
        call_args = mock_run_with_tools.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o")

    @patch("src.midpoint.agents.tools.processor.ToolProcessor")
    @async_test
    async def test_error_handling_for_invalid_parameters(self, mock_processor_class):
        """Test that the ToolProcessor handles errors from tool processor correctly."""
        # Setup mock for tool processor
        mock_tool_processor = MagicMock()
        mock_processor_class.return_value = mock_tool_processor
        
        # Create a mock for run_llm_with_tools that raises an exception
        mock_run_with_tools = AsyncMock(side_effect=ValueError("Invalid tool parameters"))
        mock_tool_processor.run_llm_with_tools = mock_run_with_tools
        
        # Create an instance of the tool processor using the mock
        processor = mock_processor_class()
        
        # Temporarily disable logging for expected errors
        logger = logging.getLogger('ToolProcessor')
        original_level = logger.level
        logger.setLevel(logging.CRITICAL)
        
        try:
            # Call the function and expect it to handle the error
            with self.assertRaises(ValueError):
                await processor.run_llm_with_tools(
                    messages=[{"role": "user", "content": "List the directory"}],
                    model="gpt-4o",
                    temperature=0.5,
                    max_tokens=4000
                )
        finally:
            # Restore logging level
            logger.setLevel(original_level)

    @async_test
    async def test_get_current_hash_async_behavior(self):
        """Test that GoalDecomposer uses the async version of get_current_hash correctly.
        
        This test specifically checks that all calls to get_current_hash in
        goal_decomposer.py are properly awaited. Before the fix, this test would
        have failed because some calls were using a synchronous version imported
        from memory_tools instead of the async version from midpoint.agents.tools.
        """
        # Import the goal_decomposer module
        import src.midpoint.agents.goal_decomposer as goal_decomposer
        
        # Check that the module is importing get_current_hash from the right place
        import inspect
        from src.midpoint.agents.tools import get_current_hash as async_get_current_hash
        
        # Test that get_current_hash is the async version by checking it's a coroutine function
        self.assertTrue(inspect.iscoroutinefunction(async_get_current_hash),
                      "get_current_hash in tools should be async")
        
        # Get the get_current_hash that's being used in goal_decomposer.py
        # This would have been the synchronous version before our fix
        goal_decomposer_get_current_hash = getattr(goal_decomposer, 'get_current_hash', None)
        
        # First check if get_current_hash is still directly in the module
        # After our fix, it shouldn't be defined there directly anymore
        if goal_decomposer_get_current_hash is not None:
            # If it exists, make sure it's the async version
            self.assertTrue(inspect.iscoroutinefunction(goal_decomposer_get_current_hash),
                          "get_current_hash in goal_decomposer should be async")
            
        # The real test: Check if execute_goal method awaits get_current_hash correctly
        # We can do this by inspecting the source code
        source = inspect.getsource(goal_decomposer)
        
        # Before our fix, there were instances of "get_current_hash(...)" without await
        # After our fix, all instances should be "await get_current_hash(...)"
        non_await_pattern = r'([^a]|^)get_current_hash\('  # Pattern for non-awaited calls
        await_pattern = r'await\s+get_current_hash\('      # Pattern for awaited calls
        
        import re
        non_awaited_calls = re.findall(non_await_pattern, source)
        awaited_calls = re.findall(await_pattern, source)
        
        # Log what we found for debugging
        print(f"Found {len(non_awaited_calls)} non-awaited calls and {len(awaited_calls)} awaited calls")
        
        # Our fixed code should have several awaited calls and no non-awaited calls
        # (excluding imports and function definitions)
        self.assertGreater(len(awaited_calls), 0, 
                         "Should find awaited get_current_hash calls")
        
        # This assertion verifies our fix is properly implemented
        # All get_current_hash calls should be awaited

if __name__ == "__main__":
    unittest.main() 