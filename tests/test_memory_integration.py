#!/usr/bin/env python
"""Integration tests for the memory system."""

import os
import sys
import unittest
import tempfile
import shutil
import subprocess
from pathlib import Path

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import from the scripts
from scripts.memory_tools import (
    get_repo_path,
    store_document,
    retrieve_documents,
    update_cross_reference,
    get_memory_for_code_hash
)

from scripts.init_memory_repo import init_memory_repo

class TestMemoryIntegration(unittest.TestCase):
    """Integration tests for the memory system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a temporary directory for the test repository
        self.temp_dir = tempfile.mkdtemp()
        self.memory_repo_path = os.path.join(self.temp_dir, "memory_repo")
        
        # Store the original MEMORY_REPO_PATH environment variable
        self.original_repo_path = os.environ.get("MEMORY_REPO_PATH")
        os.environ["MEMORY_REPO_PATH"] = self.memory_repo_path
        
        # Initialize the memory repository
        init_memory_repo(self.memory_repo_path, None, None)

    def tearDown(self):
        """Tear down test fixtures."""
        # Remove the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Restore the original MEMORY_REPO_PATH environment variable
        if self.original_repo_path:
            os.environ["MEMORY_REPO_PATH"] = self.original_repo_path
        else:
            if "MEMORY_REPO_PATH" in os.environ:
                del os.environ["MEMORY_REPO_PATH"]

    def test_memory_workflow(self):
        """Test the complete memory workflow."""
        # Create a code hash for testing
        code_hash = "1234567890abcdef"
        
        # Store documents in different categories
        reasoning_doc = store_document(
            content="This is my reasoning for the current task.",
            category="reasoning",
            metadata={
                "id": "reasoning_1",
                "code_hash": code_hash,
                "commit_message": "Add reasoning document"
            },
            repo_path=self.memory_repo_path
        )
        
        observation_doc = store_document(
            content="I observed that the system behavior changed when I modified this file.",
            category="observations",
            metadata={
                "id": "observation_1",
                "code_hash": code_hash,
                "commit_message": "Add observation document"
            },
            repo_path=self.memory_repo_path
        )
        
        decision_doc = store_document(
            content="I decided to use a factory pattern for this implementation.",
            category="decisions",
            metadata={
                "id": "decision_1",
                "code_hash": code_hash,
                "commit_message": "Add decision document"
            },
            repo_path=self.memory_repo_path
        )
        
        # Get the current memory hash
        memory_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.memory_repo_path,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        
        # Update the cross-reference
        update_cross_reference(
            code_hash=code_hash,
            memory_hash=memory_hash,
            repo_path=self.memory_repo_path
        )
        
        # Verify that documents can be retrieved
        reasoning_docs = retrieve_documents(
            category="reasoning",
            limit=10,
            repo_path=self.memory_repo_path
        )
        
        observation_docs = retrieve_documents(
            category="observations",
            limit=10,
            repo_path=self.memory_repo_path
        )
        
        decision_docs = retrieve_documents(
            category="decisions",
            limit=10,
            repo_path=self.memory_repo_path
        )
        
        # Check that there's at least one document in each category
        self.assertGreaterEqual(len(reasoning_docs), 1)
        self.assertGreaterEqual(len(observation_docs), 1)
        self.assertGreaterEqual(len(decision_docs), 1)
        
        # Check that the documents contain the expected content
        reasoning_content = reasoning_docs[0][1]
        observation_content = observation_docs[0][1]
        decision_content = decision_docs[0][1]
        
        self.assertIn("This is my reasoning", reasoning_content)
        self.assertIn("I observed that", observation_content)
        self.assertIn("I decided to use", decision_content)
        
        # Verify that the memory hash can be looked up using the code hash
        retrieved_memory_hash = get_memory_for_code_hash(
            code_hash=code_hash,
            repo_path=self.memory_repo_path
        )
        
        self.assertEqual(retrieved_memory_hash, memory_hash)

    def test_memory_update_workflow(self):
        """Test updating memory with new documents."""
        # Initial setup
        code_hash = "1234567890abcdef"
        
        # Store an initial document
        initial_doc = store_document(
            content="Initial observation.",
            category="observations",
            metadata={
                "id": "observation_initial",
                "code_hash": code_hash,
                "commit_message": "Initial observation"
            },
            repo_path=self.memory_repo_path
        )
        
        # Get the current memory hash
        initial_memory_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.memory_repo_path,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        
        # Update the cross-reference
        update_cross_reference(
            code_hash=code_hash,
            memory_hash=initial_memory_hash,
            repo_path=self.memory_repo_path
        )
        
        # Store a new document
        updated_doc = store_document(
            content="Updated observation with more details.",
            category="observations",
            metadata={
                "id": "observation_updated",
                "code_hash": code_hash,
                "commit_message": "Updated observation"
            },
            repo_path=self.memory_repo_path
        )
        
        # Get the new memory hash
        updated_memory_hash = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=self.memory_repo_path,
            capture_output=True,
            text=True,
            check=True
        ).stdout.strip()
        
        # Verify hashes are different
        self.assertNotEqual(initial_memory_hash, updated_memory_hash)
        
        # Update the cross-reference
        update_cross_reference(
            code_hash=code_hash,
            memory_hash=updated_memory_hash,
            repo_path=self.memory_repo_path
        )
        
        # Verify that the most recent memory hash is returned
        retrieved_memory_hash = get_memory_for_code_hash(
            code_hash=code_hash,
            repo_path=self.memory_repo_path
        )
        
        self.assertEqual(retrieved_memory_hash, updated_memory_hash)
        
        # Retrieve all observation documents
        observation_docs = retrieve_documents(
            category="observations",
            limit=10,
            repo_path=self.memory_repo_path
        )
        
        # Verify that both documents are present
        self.assertEqual(len(observation_docs), 2)
        
        # Check document contents
        doc_contents = [content for _, content in observation_docs]
        self.assertTrue(any("Initial observation" in content for content in doc_contents))
        self.assertTrue(any("Updated observation" in content for content in doc_contents))

if __name__ == "__main__":
    unittest.main() 