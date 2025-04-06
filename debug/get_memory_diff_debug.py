#!/usr/bin/env python
"""
Script to debug memory diff issues for goal T1.

This script verifies the memory diff functionality is working correctly
by getting diffs between the initial and final memory hashes from T1.json.
"""

import os
import sys
import json
import logging
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Try to import the memory tools
try:
    from scripts.memory_tools import get_repo_path as system_get_repo_path
    print("Successfully imported system get_repo_path")
except ImportError as e:
    print(f"Error importing system get_repo_path: {e}")
    
    def system_get_repo_path():
        """Fallback function for getting memory repo path."""
        return os.environ.get("MEMORY_REPO_PATH", os.path.join(os.path.expanduser("~"), ".midpoint", "memory"))

# Import the get_memory_diff function
try:
    from midpoint.agents.tools.memory_tools import get_memory_diff, HAS_MEMORY_TOOLS
    print(f"Successfully imported get_memory_diff. HAS_MEMORY_TOOLS = {HAS_MEMORY_TOOLS}")
except ImportError as e:
    print(f"Error importing get_memory_diff: {e}")
    sys.exit(1)

async def direct_git_diff(initial_hash: str, final_hash: str, repo_path: str) -> Dict[str, Any]:
    """
    Get a diff directly using git command.
    
    Args:
        initial_hash: Initial commit hash
        final_hash: Final commit hash
        repo_path: Path to the git repository
        
    Returns:
        Dictionary with diff results
    """
    try:
        print(f"Running direct git diff between {initial_hash} and {final_hash}")
        
        # Get list of changed files
        cmd = ["git", "diff", "--name-only", initial_hash, final_hash]
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        changed_files = [
            file for file in proc.stdout.strip().split("\n") 
            if file.strip()
        ]
        print(f"Changed files: {changed_files}")
        
        # Get diff summary
        cmd = ["git", "diff", "--stat", initial_hash, final_hash]
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        diff_summary = proc.stdout.strip()
        print(f"Diff summary: {diff_summary}")
        
        # Get diff content
        cmd = ["git", "diff", initial_hash, final_hash]
        proc = subprocess.run(
            cmd,
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        diff_content = proc.stdout
        print(f"Diff content length: {len(diff_content)}")
        
        return {
            "changed_files": changed_files,
            "diff_summary": diff_summary,
            "diff_content": diff_content
        }
        
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {e}")
        print(f"Command: {e.cmd}")
        print(f"Return code: {e.returncode}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return {"error": str(e)}
    except Exception as e:
        print(f"Error in direct_git_diff: {e}")
        return {"error": str(e)}

async def test_memory_diff(goal_id: str = "T1"):
    """
    Test memory diff functionality for the specified goal.
    
    Args:
        goal_id: ID of the goal to test
    """
    try:
        # Load goal data
        goal_path = repo_root / ".goal" / f"{goal_id}.json"
        if not goal_path.exists():
            print(f"Goal file not found: {goal_path}")
            return None
            
        with open(goal_path, "r") as f:
            goal_data = json.load(f)
            
        # Extract memory hashes
        initial_memory_hash = goal_data.get("initial_state", {}).get("memory_hash")
        final_memory_hash = goal_data.get("current_state", {}).get("memory_hash")
        memory_repo_path = goal_data.get("initial_state", {}).get("memory_repository_path")
        
        if not initial_memory_hash or not final_memory_hash:
            print(f"Memory hashes not found in goal data: {goal_data}")
            return None
            
        if not memory_repo_path:
            memory_repo_path = system_get_repo_path()
            
        # Convert memory_repo_path to Path object if it's a string
        if isinstance(memory_repo_path, str):
            memory_repo_path = Path(memory_repo_path)
            
        print(f"Memory repository path: {memory_repo_path}")
        print(f"Initial memory hash: {initial_memory_hash}")
        print(f"Final memory hash: {final_memory_hash}")
        
        # Verify both hashes exist
        try:
            for hash_val in [initial_memory_hash, final_memory_hash]:
                subprocess.run(
                    ["git", "cat-file", "-e", hash_val],
                    cwd=memory_repo_path,
                    check=True
                )
            print("Both memory hashes are valid git objects")
        except subprocess.CalledProcessError as e:
            print(f"Error verifying memory hashes: {e}")
            return None
        
        # Method 1: Use get_memory_diff function
        print("\n=== Using get_memory_diff function ===")
        try:
            diff_result = await get_memory_diff(
                initial_hash=initial_memory_hash,
                final_hash=final_memory_hash,
                memory_repo_path=str(memory_repo_path)
            )
            
            print("\nMemory Diff Results:")
            print(f"Changed Files: {json.dumps(diff_result['changed_files'], indent=2)}")
            print(f"Diff Summary:\n{diff_result['diff_summary']}")
            print(f"Diff Content (truncated to 500 chars):")
            print(diff_result['diff_content'][:500])
            if diff_result.get('truncated'):
                print("... [truncated]")
        except Exception as e:
            print(f"Error using get_memory_diff: {e}")
        
        # Method 2: Direct git commands
        print("\n=== Using direct git commands ===")
        try:
            direct_diff = await direct_git_diff(
                initial_hash=initial_memory_hash,
                final_hash=final_memory_hash,
                repo_path=str(memory_repo_path)
            )
            
            if "error" in direct_diff:
                print(f"Error in direct git diff: {direct_diff['error']}")
            else:
                print("\nDirect Git Diff Results:")
                print(f"Changed Files: {json.dumps(direct_diff['changed_files'], indent=2)}")
                print(f"Diff Summary:\n{direct_diff['diff_summary']}")
                print(f"Diff Content (truncated to 500 chars):")
                print(direct_diff['diff_content'][:500])
        except Exception as e:
            print(f"Error in direct git diff: {e}")
            
    except Exception as e:
        print(f"Unexpected error in test_memory_diff: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    goal_id = sys.argv[1] if len(sys.argv) > 1 else "T1"
    print(f"Testing memory diff for goal {goal_id}")
    asyncio.run(test_memory_diff(goal_id)) 