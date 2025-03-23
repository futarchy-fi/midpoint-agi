#!/usr/bin/env python
"""Unit tests for the memory tools module."""

import os
import sys
import unittest
import tempfile
import shutil
import json
import time
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
        (self.memory_repo_path / "documents").mkdir(exist_ok=True)
        (self.memory_repo_path / "documents" / "reasoning").mkdir(exist_ok=True)
        (self.memory_repo_path / "documents" / "observations").mkdir(exist_ok=True)
        (self.memory_repo_path / "documents" / "decisions").mkdir(exist_ok=True)
        (self.memory_repo_path / "metadata").mkdir(exist_ok=True)
        
        # Create a cross-reference file
        cross_ref_path = self.memory_repo_path / "metadata" / "cross-reference.json"
        with open(cross_ref_path, "w") as f:
            json.dump({}, f)
        
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
    @patch("scripts.memory_tools.get_current_hash")
    def test_store_document(self, mock_get_hash, mock_run):
        """Test that store_document creates the file and commits it."""
        # Set up the mocks
        mock_run.return_value = MagicMock(returncode=0)
        mock_get_hash.return_value = "abcdef1234567890"
        
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
        
        # Verify the function returned something
        self.assertTrue(result)
        
        # Verify git commands were called (at least add and commit)
        self.assertTrue(mock_run.call_count >= 2)

    def test_retrieve_documents(self):
        """Test that retrieve_documents returns the correct documents."""
        # Create some test documents
        category = "observations"
        doc_dir = self.memory_repo_path / "documents" / category
        doc_dir.mkdir(exist_ok=True, parents=True)
        
        for i in range(5):
            doc_path = doc_dir / f"test_doc_{i}_{int(time.time())}.md"
            with open(doc_path, "w") as f:
                f.write(f"This is test document {i}")
        
        # Call the function
        documents = retrieve_documents(
            category=category,
            limit=3,
            repo_path=str(self.memory_repo_path)
        )
        
        # Verify we got some results
        self.assertTrue(len(documents) > 0)

    @patch("scripts.memory_tools.subprocess.run")
    def test_update_cross_reference(self, mock_run):
        """Test that update_cross_reference creates the correct reference file."""
        # Set up the mock
        mock_run.return_value = MagicMock(returncode=0)
        
        # Test data
        code_hash = "1234567890abcdef"
        memory_hash = "abcdef1234567890"
        
        # Call the function
        update_cross_reference(
            code_hash=code_hash,
            memory_hash=memory_hash,
            repo_path=str(self.memory_repo_path)
        )
        
        # Verify the cross-reference.json file exists and contains our data
        cross_ref_path = self.memory_repo_path / "metadata" / "cross-reference.json"
        self.assertTrue(cross_ref_path.exists())
        
        with open(cross_ref_path, "r") as f:
            cross_ref = json.load(f)
            self.assertIn(code_hash, cross_ref)
            self.assertEqual(cross_ref[code_hash], memory_hash)

    def test_get_memory_for_code_hash(self):
        """Test that get_memory_for_code_hash returns the correct hash."""
        # Create a test cross-reference file
        code_hash = "1234567890abcdef"
        memory_hash = "abcdef1234567890"
        
        cross_ref_path = self.memory_repo_path / "metadata" / "cross-reference.json"
        with open(cross_ref_path, "w") as f:
            json.dump({code_hash: memory_hash}, f)
        
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