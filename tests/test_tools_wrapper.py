#!/usr/bin/env python
"""Unit tests for the tools wrapper functions in tools.py."""

import os
import sys
import unittest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock, AsyncMock

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import module under test
from src.midpoint.agents.tools import (
    list_directory,
    read_file,
    edit_file,
    search_code,
    web_search,
    web_scrape,
    get_current_hash
)
from src.midpoint.agents.tools.filesystem_tools import (
    ListDirectoryTool,
    ReadFileTool,
    EditFileTool
)

# Import test helpers
from tests.test_helpers import async_test

class TestToolsWrapper(unittest.TestCase):
    """Tests for tools wrapper module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = Path(self.temp_dir)
        self.test_dir = self.repo_path / "test_dir"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create test files
        self.test_file = self.test_dir / "test_file.txt"
        self.test_file.write_text("This is a test file.\nIt has two lines.")
        
        # Create a subdirectory with a file
        subdir = self.test_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        (subdir / "subfile.txt").write_text("Subfile content")

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    @patch.object(ListDirectoryTool, "execute")
    @async_test
    async def test_list_directory_wrapper(self, mock_execute):
        """Test list_directory wrapper function."""
        # Setup mock
        mock_execute.return_value = {
            "path": str(self.test_dir),
            "items": [{"name": "test_file.txt", "type": "file", "size": 30}]
        }
        
        # Call the wrapper function
        result = await list_directory(path=str(self.test_dir))
        
        # Check that the execute method was called with correct parameters
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        self.assertEqual(call_args["path"], str(self.test_dir))
        
        # Test with all parameters
        mock_execute.reset_mock()
        await list_directory(path=str(self.test_dir), pattern="*.txt", recursive=True)
        
        # Check all parameters were passed correctly
        call_args = mock_execute.call_args[1]
        self.assertEqual(call_args["path"], str(self.test_dir))
        self.assertEqual(call_args["pattern"], "*.txt")
        self.assertEqual(call_args["recursive"], True)
        
        # Verify the function doesn't accept repo_path or directory parameters
        with self.assertRaises(TypeError):
            await list_directory(repo_path=str(self.repo_path), directory="test_dir")

    @patch.object(ReadFileTool, "execute")
    @async_test
    async def test_read_file_wrapper(self, mock_execute):
        """Test read_file wrapper function."""
        # Setup mock
        mock_execute.return_value = {
            "content": "Test file content\nLine 2\nLine 3",
            "start_line": 0,
            "lines_read": 3,
            "total_lines": 3
        }
        
        # Call the wrapper function
        result = await read_file(file_path=str(self.test_file))
        
        # Check that the execute method was called with correct parameters
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        self.assertEqual(call_args["file_path"], str(self.test_file))
        
        # Test with all parameters
        mock_execute.reset_mock()
        await read_file(file_path=str(self.test_file), start_line=1, max_lines=2)
        
        # Check all parameters were passed correctly
        call_args = mock_execute.call_args[1]
        self.assertEqual(call_args["file_path"], str(self.test_file))
        self.assertEqual(call_args["start_line"], 1)
        self.assertEqual(call_args["max_lines"], 2)
        
        # Verify the function doesn't accept repo_path parameter
        with self.assertRaises(TypeError):
            await read_file(repo_path=str(self.repo_path), file_path=str(self.test_file))

    @patch.object(EditFileTool, "execute")
    @async_test
    async def test_edit_file_wrapper(self, mock_execute):
        """Test edit_file wrapper function."""
        # Setup mock
        mock_execute.return_value = {"success": True}
        
        # Call the wrapper function
        result = await edit_file(file_path=str(self.test_file), content="New content")
        
        # Check that the execute method was called with correct parameters
        mock_execute.assert_called_once()
        call_args = mock_execute.call_args[1]
        self.assertEqual(call_args["file_path"], str(self.test_file))
        self.assertEqual(call_args["content"], "New content")
        
        # Test with all parameters
        mock_execute.reset_mock()
        await edit_file(file_path=str(self.test_file), content="New content", create_dirs=True)
        
        # Check all parameters were passed correctly
        call_args = mock_execute.call_args[1]
        self.assertEqual(call_args["file_path"], str(self.test_file))
        self.assertEqual(call_args["content"], "New content")
        self.assertEqual(call_args["create_dirs"], True)

if __name__ == "__main__":
    unittest.main() 