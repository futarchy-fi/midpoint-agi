#!/usr/bin/env python
"""Integration test fixtures for testing with real files."""

import os
import tempfile
import shutil
import subprocess
from pathlib import Path
import json
import time

# Base directory for all temporary test files
TEST_FIXTURES_ROOT = tempfile.mkdtemp(prefix="midpoint_test_")

def setup_test_repository(with_dummy_files=True):
    """
    Set up a test git repository with necessary structure.
    
    Args:
        with_dummy_files: Whether to include dummy files in the repository.
        
    Returns:
        Path to the repository.
    """
    # Create a new temp directory for the repo
    repo_path = Path(tempfile.mkdtemp(dir=TEST_FIXTURES_ROOT, prefix="repo_"))
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=repo_path, check=True, capture_output=True)
    
    # Set up git config
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=repo_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo_path, check=True, capture_output=True)
    
    # Create logs directory
    logs_dir = repo_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create src directory structure if needed
    if with_dummy_files:
        src_dir = repo_path / "src"
        src_dir.mkdir(exist_ok=True)
        
        # Create a dummy file
        test_file = src_dir / "test_file.py"
        test_file.write_text('"""Test file for repository."""\n\nprint("Hello, world!")\n')
        
        # Create another file
        main_file = src_dir / "main.py"
        main_file.write_text('"""Main module."""\n\nif __name__ == "__main__":\n    print("Main module")\n')
        
        # Commit the files
        subprocess.run(["git", "add", "."], cwd=repo_path, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-m", "Initial commit"], cwd=repo_path, check=True, capture_output=True)
    
    return repo_path

def create_test_subgoal_file(repo_path, requires_decomposition=False, custom_content=None):
    """
    Create a test subgoal file in the logs directory.
    
    Args:
        repo_path: Path to the repository.
        requires_decomposition: Whether the subgoal requires further decomposition.
        custom_content: Optional dict to override the default content of the subgoal file.
        
    Returns:
        Path to the created subgoal file.
    """
    logs_dir = repo_path / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create a unique timestamp-based filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    hash_suffix = os.urandom(3).hex()
    
    subgoal_file = logs_dir / f"test_subgoal_{timestamp}_{hash_suffix}.json"
    
    # Default content
    content = {
        "goal": "Test goal for integration testing",
        "next_step": "Add a print statement to main.py",
        "validation_criteria": ["Test validation passes"],
        "reasoning": "This is a test task for integration testing.",
        "requires_further_decomposition": requires_decomposition,
        "relevant_context": {}
    }
    
    # Override with custom content if provided
    if custom_content:
        content.update(custom_content)
    
    with open(subgoal_file, 'w') as f:
        json.dump(content, f, indent=2)
    
    return subgoal_file

def setup_memory_repository():
    """
    Set up a test memory repository.
    
    Returns:
        Tuple of (path to memory repository, current memory hash).
    """
    # Create a new temp directory for the memory repo
    memory_path = Path(tempfile.mkdtemp(dir=TEST_FIXTURES_ROOT, prefix="memory_"))
    
    # Initialize git repository
    subprocess.run(["git", "init"], cwd=memory_path, check=True, capture_output=True)
    
    # Set up git config
    subprocess.run(["git", "config", "user.name", "Test User"], cwd=memory_path, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=memory_path, check=True, capture_output=True)
    
    # Create basic memory structure
    docs_dir = memory_path / "documents"
    docs_dir.mkdir(exist_ok=True)
    
    for category in ["reasoning", "observations", "decisions", "study"]:
        category_dir = docs_dir / category
        category_dir.mkdir(exist_ok=True)
    
    # Create a README file
    readme_path = memory_path / "README.md"
    readme_path.write_text("# Test Memory Repository\n\nThis is a test memory repository for integration tests.\n")
    
    # Commit the files
    subprocess.run(["git", "add", "."], cwd=memory_path, check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initialize memory repository"], cwd=memory_path, check=True, capture_output=True)
    
    # Get the current hash
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], 
        cwd=memory_path, 
        check=True, 
        capture_output=True, 
        text=True
    )
    memory_hash = result.stdout.strip()
    
    return memory_path, memory_hash

def cleanup_test_fixtures():
    """Clean up all test fixtures."""
    try:
        if os.path.exists(TEST_FIXTURES_ROOT):
            shutil.rmtree(TEST_FIXTURES_ROOT)
    except Exception as e:
        print(f"Warning: Failed to clean up test fixtures: {e}")

if __name__ == "__main__":
    # Test the fixtures
    print("Setting up test repository...")
    repo_path = setup_test_repository()
    print(f"Test repository created at: {repo_path}")
    
    print("\nCreating test subgoal file...")
    subgoal_file = create_test_subgoal_file(repo_path)
    print(f"Test subgoal file created at: {subgoal_file}")
    
    print("\nSetting up memory repository...")
    memory_path, memory_hash = setup_memory_repository()
    print(f"Test memory repository created at: {memory_path}")
    print(f"Memory hash: {memory_hash}")
    
    print("\nCleanup...")
    cleanup_test_fixtures()
    print("Done.") 