#!/usr/bin/env python
"""Unit tests for the filesystem tools module."""

import os
import sys
import unittest
import tempfile
import shutil
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import module under test
from src.midpoint.agents.tools.filesystem_tools import (
    list_directory,
    read_file,
    edit_file,
    ListDirectoryTool,
    ReadFileTool,
    EditFileTool
)

# Import test helpers
from tests.test_helpers import async_test

class TestFilesystemTools(unittest.TestCase):
    """Tests for filesystem tools module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_dir = Path(self.temp_dir) / "test_dir"
        self.test_dir.mkdir(exist_ok=True)
        
        # Create some test files and directories
        (self.test_dir / "file1.txt").write_text("Test file 1 content")
        (self.test_dir / "file2.py").write_text("def test_function():\n    return 'test'")
        
        # Create a subdirectory with files
        subdir = self.test_dir / "subdir"
        subdir.mkdir(exist_ok=True)
        (subdir / "subfile.txt").write_text("Subfile content")

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)

    @async_test
    async def test_list_directory(self):
        """Test list_directory function with path parameter."""
        # Test with path parameter
        result = await list_directory(path=str(self.test_dir))
        
        # Check that the function returns the expected structure
        self.assertIn("items", result)
        self.assertIn("path", result)
        
        # Check that it found our test files and directory
        items = [item["name"] for item in result["items"]]
        self.assertIn("file1.txt", items)
        self.assertIn("file2.py", items)
        self.assertIn("subdir", items)
        
        # Check file types
        for item in result["items"]:
            if item["name"] == "subdir":
                self.assertEqual(item["type"], "directory")
            else:
                self.assertEqual(item["type"], "file")

    @async_test
    async def test_list_directory_with_pattern(self):
        """Test list_directory function with pattern parameter."""
        # Test with pattern parameter to filter files
        result = await list_directory(path=str(self.test_dir), pattern="*.txt")
        
        # Check that it only found text files
        items = [item["name"] for item in result["items"]]
        self.assertIn("file1.txt", items)
        self.assertNotIn("file2.py", items)

    @async_test
    async def test_list_directory_recursive(self):
        """Test list_directory function with recursive parameter."""
        # Test with recursive parameter
        result = await list_directory(path=str(self.test_dir), recursive=True)
        
        # Check that it found files in subdirectories
        items = [item["name"] for item in result["items"]]
        self.assertTrue(any("subdir" in item for item in items))

    @async_test
    async def test_read_file(self):
        """Test read_file function."""
        # Create a test file with multiple lines
        test_file = self.test_dir / "multiline.txt"
        test_file.write_text("Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        
        # Test reading the entire file
        result = await read_file(file_path=str(test_file))
        
        # Check that the function returns the expected structure
        self.assertIn("content", result)
        self.assertIn("start_line", result)
        self.assertIn("lines_read", result)
        self.assertIn("total_lines", result)
        
        # Check content
        self.assertEqual(result["content"], "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        self.assertEqual(result["total_lines"], 5)
        
        # Test reading with start_line and max_lines
        result = await read_file(file_path=str(test_file), start_line=1, max_lines=2)
        self.assertEqual(result["content"], "Line 2\nLine 3\n")
        self.assertEqual(result["start_line"], 1)
        self.assertEqual(result["lines_read"], 2)

    @async_test
    async def test_edit_file(self):
        """Test edit_file function."""
        # Create a test file
        test_file = self.test_dir / "edit_test.txt"
        test_file.write_text("Original content")
        
        # Test editing an existing file
        result = await edit_file(file_path=str(test_file), content="New content")
        
        # Check success
        self.assertNotIn("error", result)
        
        # Verify file was updated
        self.assertEqual(test_file.read_text(), "New content")
        
        # Test creating a new file
        new_file = self.test_dir / "new_file.txt"
        result = await edit_file(file_path=str(new_file), content="Brand new content")
        
        # Check success and verify file was created
        self.assertNotIn("error", result)
        self.assertEqual(new_file.read_text(), "Brand new content")

if __name__ == "__main__":
    unittest.main() 