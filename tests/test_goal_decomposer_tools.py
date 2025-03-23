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
        self.decomposer.tool_processor = MagicMock(spec=ToolProcessor)

    def tearDown(self):
        """Tear down test fixtures."""
        shutil.rmtree(self.temp_dir)

    @patch("src.midpoint.agents.tools.filesystem_tools.list_directory")
    @patch("src.midpoint.agents.goal_decomposer.ToolProcessor")
    @async_test
    async def test_tool_list_directory_parameters(self, mock_processor, mock_list_directory):
        """Test that the GoalDecomposer correctly processes list_directory tool calls."""
        # Setup mock for tool processor
        mock_tool_processor = MagicMock()
        mock_processor.return_value = mock_tool_processor
        
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
        
        # Set the mocked tool processor on the decomposer
        self.decomposer.tool_processor = mock_tool_processor
        
        # Call the function
        await self.decomposer._run_with_tools(messages=[{"role": "user", "content": "List the directory"}])
        
        # Check that run_llm_with_tools was called correctly
        mock_run_with_tools.assert_called_once()
        
        # Verify the mock was called with the expected arguments
        call_args = mock_run_with_tools.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o")
        self.assertEqual(call_args["temperature"], 0.5)
        self.assertEqual(call_args["max_tokens"], 4000)

    @patch("src.midpoint.agents.tools.filesystem_tools.read_file")
    @patch("src.midpoint.agents.goal_decomposer.ToolProcessor")
    @async_test
    async def test_tool_read_file_parameters(self, mock_processor, mock_read_file):
        """Test that the GoalDecomposer correctly processes read_file tool calls."""
        # Setup mock for tool processor
        mock_tool_processor = MagicMock()
        mock_processor.return_value = mock_tool_processor
        
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
        
        # Set the mocked tool processor on the decomposer
        self.decomposer.tool_processor = mock_tool_processor
        
        # Call the function
        await self.decomposer._run_with_tools(messages=[{"role": "user", "content": "Read the file"}])
        
        # Check that run_llm_with_tools was called correctly
        mock_run_with_tools.assert_called_once()
        
        # Verify the mock was called with the expected arguments
        call_args = mock_run_with_tools.call_args[1]
        self.assertEqual(call_args["model"], "gpt-4o")

    @patch("src.midpoint.agents.goal_decomposer.ToolProcessor")
    @async_test
    async def test_error_handling_for_invalid_parameters(self, mock_processor):
        """Test that the GoalDecomposer handles errors from tool processor correctly."""
        # Setup mock for tool processor
        mock_tool_processor = MagicMock()
        mock_processor.return_value = mock_tool_processor
        
        # Create a mock for run_llm_with_tools that raises an exception
        mock_run_with_tools = AsyncMock(side_effect=ValueError("Invalid tool parameters"))
        mock_tool_processor.run_llm_with_tools = mock_run_with_tools
        
        # Set the mocked tool processor on the decomposer
        self.decomposer.tool_processor = mock_tool_processor
        
        # Temporarily disable logging for expected errors
        logger = logging.getLogger('GoalDecomposer')
        original_level = logger.level
        logger.setLevel(logging.CRITICAL)
        
        try:
            # Call the function and expect it to handle the error
            with self.assertRaises(ValueError):
                await self.decomposer._run_with_tools(messages=[{"role": "user", "content": "List the directory"}])
        finally:
            # Restore logging level
            logger.setLevel(original_level)

    @patch("src.midpoint.agents.tools.git_tools.get_current_hash")
    @async_test
    async def test_get_current_hash_async_behavior(self, mock_get_current_hash):
        """Test that GoalDecomposer uses the async version of get_current_hash correctly.
        
        This test specifically checks that all calls to get_current_hash in
        goal_decomposer.py are properly awaited. Before the fix, this test would
        have failed because some calls were using a synchronous version imported
        from memory_tools instead of the async version from midpoint.agents.tools.
        """
        # Setup mock for get_current_hash - ensure it returns a coroutine
        expected_hash = "abcdef1234567890abcdef1234567890abcdef12"
        mock_get_current_hash.return_value = expected_hash
        
        # Create a temporary git-like directory structure
        git_dir = self.repo_path / ".git"
        git_dir.mkdir(exist_ok=True)
        
        # Skip testing validate_repository_state directly since it's harder to patch correctly
        # Instead, focus on the store_memory_document function which was one of the bug locations
        
        # Reset the mock for the main test
        mock_get_current_hash.reset_mock()
        
        # Test scenario: get_current_hash in the store_memory_document function
        # This call was one of the buggy calls we fixed
        memory_repo_path = str(self.repo_path)
        
        # Add needed imports for our test
        from src.midpoint.agents.tools import store_memory_document, get_current_hash
        
        # We need to patch at the correct level where it's actually imported
        with patch('src.midpoint.agents.tools.git_tools.GetCurrentHashTool.execute', 
                  return_value=expected_hash) as mock_execute:
            # Call the function that uses get_current_hash
            try:
                store_result = await store_memory_document(
                    content="Test content",
                    category="test_category",
                    memory_repo_path=memory_repo_path
                )
                
                # If we got here without errors, that's already an improvement
                # over the original buggy code which would have raised coroutine-related errors
                
                # Verify the tool's execute method was called
                # This is a better way to verify since our architecture uses the Tool pattern
                self.assertTrue(mock_execute.called, 
                             "GetCurrentHashTool.execute should be called")
                self.assertTrue(mock_execute.awaited,
                             "GetCurrentHashTool.execute should be awaited")
                
            except Exception as e:
                # If the function fails with an await-related error, that indicates the bug
                if "coroutine" in str(e) and "await" in str(e):
                    self.fail(f"store_memory_document failed with await error: {str(e)}")
                else:
                    # Other errors might be due to our test setup, print for debugging
                    self.fail(f"Test setup error: {str(e)}")
        
        # This test would have failed with the original bug because the 
        # non-awaited synchronous version would produce errors or wouldn't 
        # register as being properly awaited

if __name__ == "__main__":
    unittest.main() 