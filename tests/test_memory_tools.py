#!/usr/bin/env python
"""Unit tests for the memory tools module."""

import os
import sys
import unittest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import module under test
from scripts.memory_tools import (
    get_repo_path,
    get_current_hash,
    store_document,
    retrieve_documents,
    update_cross_reference,
    get_memory_for_code_hash
)

class TestMemoryTools(unittest.TestCase):
    """Tests for memory tools module."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the test repository
        self.temp_dir = tempfile.mkdtemp()
        self.memory_repo_path = Path(self.temp_dir) / "memory_repo"
        self.memory_repo_path.mkdir(exist_ok=True)
        
        # Create basic directory structure
        (self.memory_repo_path / "reasoning").mkdir(exist_ok=True)
        (self.memory_repo_path / "observations").mkdir(exist_ok=True)
        (self.memory_repo_path / "decisions").mkdir(exist_ok=True)
        (self.memory_repo_path / "cross_references").mkdir(exist_ok=True)
        
        # Create a .git directory to simulate a real git repository
        (self.memory_repo_path / ".git").mkdir(exist_ok=True)
        
        # Set up environment variables
        self.original_repo_path = os.environ.get("MEMORY_REPO_PATH")
        os.environ["MEMORY_REPO_PATH"] = str(self.memory_repo_path)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Restore environment variables
        if self.original_repo_path:
            os.environ["MEMORY_REPO_PATH"] = self.original_repo_path
        else:
            os.environ.pop("MEMORY_REPO_PATH", None)

    def test_get_repo_path(self):
        """Test that get_repo_path returns the correct path."""
        # Test with environment variable
        self.assertEqual(get_repo_path(), str(self.memory_repo_path))
        
        # Test with explicit path
        explicit_path = "/path/to/explicit/repo"
        self.assertEqual(get_repo_path(explicit_path), explicit_path)

    @patch("scripts.memory_tools.subprocess.run")
    def test_get_current_hash(self, mock_run):
        """Test that get_current_hash calls git correctly."""
        # Set up the mock
        mock_process = MagicMock()
        mock_process.stdout = "abcdef1234567890\n"
        mock_run.return_value = mock_process
        
        # Call the function
        result = get_current_hash(str(self.memory_repo_path))
        
        # Verify the result
        self.assertEqual(result, "abcdef1234567890")
        
        # Verify git was called correctly
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertEqual(args[0][0], "git")
        self.assertEqual(args[0][1], "rev-parse")
        self.assertEqual(args[0][2], "HEAD")
        self.assertEqual(kwargs["cwd"], str(self.memory_repo_path))

    @patch("scripts.memory_tools.subprocess.run")
    def test_store_document(self, mock_run):
        """Test that store_document creates the file and commits it."""
        # Set up the mock
        mock_run.return_value = MagicMock(returncode=0)
        
        # Sample content and metadata
        content = "This is a test document"
        category = "reasoning"
        metadata = {
            "id": "test_doc_1",
            "code_hash": "1234567890abcdef",
            "commit_message": "Add test document"
        }
        
        # Call the function
        result = store_document(
            content=content,
            category=category,
            metadata=metadata,
            repo_path=str(self.memory_repo_path)
        )
        
        # Verify the file was created with the correct content
        expected_path = self.memory_repo_path / category / f"{metadata['id']}.md"
        self.assertTrue(expected_path.exists())
        
        with open(expected_path, "r") as f:
            file_content = f.read()
        self.assertIn(content, file_content)
        
        # Verify git commands were called correctly
        self.assertEqual(mock_run.call_count, 2)  # add and commit
        
        # Verify the return value
        self.assertEqual(result["path"], str(expected_path))
        self.assertEqual(result["category"], category)
        self.assertEqual(result["id"], metadata["id"])

    def test_retrieve_documents(self):
        """Test that retrieve_documents returns the correct documents."""
        # Create some test documents
        category = "observations"
        for i in range(5):
            doc_path = self.memory_repo_path / category / f"test_doc_{i}.md"
            with open(doc_path, "w") as f:
                f.write(f"This is test document {i}")
        
        # Call the function
        documents = retrieve_documents(
            category=category,
            limit=3,
            repo_path=str(self.memory_repo_path)
        )
        
        # Verify the results
        self.assertEqual(len(documents), 3)
        for path, content in documents:
            self.assertTrue(Path(path).exists())
            self.assertTrue(content.startswith("This is test document"))

    @patch("scripts.memory_tools.subprocess.run")
    def test_update_cross_reference(self, mock_run):
        """Test that update_cross_reference creates the correct reference file."""
        # Set up the mock
        mock_run.return_value = MagicMock(returncode=0)
        
        # Test data
        code_hash = "1234567890abcdef"
        memory_hash = "abcdef1234567890"
        
        # Call the function
        result = update_cross_reference(
            code_hash=code_hash,
            memory_hash=memory_hash,
            repo_path=str(self.memory_repo_path)
        )
        
        # Verify the file was created
        expected_path = self.memory_repo_path / "cross_references" / f"{code_hash}.txt"
        self.assertTrue(expected_path.exists())
        
        with open(expected_path, "r") as f:
            file_content = f.read().strip()
        self.assertEqual(file_content, memory_hash)
        
        # Verify git commands were called
        self.assertEqual(mock_run.call_count, 2)  # add and commit
        
        # Verify the return value
        self.assertEqual(result["code_hash"], code_hash)
        self.assertEqual(result["memory_hash"], memory_hash)

    def test_get_memory_for_code_hash(self):
        """Test that get_memory_for_code_hash returns the correct hash."""
        # Create a test cross-reference file
        code_hash = "1234567890abcdef"
        memory_hash = "abcdef1234567890"
        
        ref_path = self.memory_repo_path / "cross_references" / f"{code_hash}.txt"
        with open(ref_path, "w") as f:
            f.write(memory_hash)
        
        # Call the function
        result = get_memory_for_code_hash(
            code_hash=code_hash,
            repo_path=str(self.memory_repo_path)
        )
        
        # Verify the result
        self.assertEqual(result, memory_hash)
        
        # Test with a non-existent code hash
        non_existent = get_memory_for_code_hash(
            code_hash="nonexistentcodehashabcdef",
            repo_path=str(self.memory_repo_path)
        )
        self.assertIsNone(non_existent)

if __name__ == "__main__":
    unittest.main() 