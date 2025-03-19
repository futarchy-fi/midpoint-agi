#!/usr/bin/env python
"""
Script to fix and run the test_goal_decomposer.py test with the actual git hash.
"""

import os
import sys
import asyncio
from pathlib import Path

async def get_current_hash(repo_path: str) -> str:
    """Get the current git hash."""
    import asyncio
    result = await asyncio.create_subprocess_exec(
        "git", "rev-parse", "HEAD",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    if result.returncode != 0:
        raise RuntimeError(f"Failed to get current hash: {stderr.decode()}")
        
    return stdout.decode().strip()

async def main():
    """Main function."""
    # Path to test repository
    test_repo_path = os.path.join(os.getcwd(), "test-repo")
    print(f"Using test repository at: {test_repo_path}")
    
    # Get the actual git hash
    try:
        git_hash = await get_current_hash(test_repo_path)
        print(f"Actual git hash: {git_hash}")
    except Exception as e:
        print(f"Error getting git hash: {str(e)}")
        return
    
    # Run the test file with pytest
    print("\nRunning the test_goal_decomposer.py file...")
    
    # We'll now run the pytest command
    result = await asyncio.create_subprocess_exec(
        "python", "-m", "pytest", "tests/test_goal_decomposer.py", "-v",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, stderr = await result.communicate()
    
    # Print the output
    print("\nTest output:")
    print(stdout.decode())
    
    if stderr:
        print("\nErrors:")
        print(stderr.decode())
    
    print(f"\nTest exit code: {result.returncode}")

if __name__ == "__main__":
    asyncio.run(main()) 