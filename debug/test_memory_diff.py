#!/usr/bin/env python
"""
Example script to test get_memory_diff functionality.

This script demonstrates how to use the get_memory_diff function from memory_tools.
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import the get_memory_diff function
try:
    from midpoint.agents.tools.memory_tools import get_memory_diff, HAS_MEMORY_TOOLS
    print(f"Successfully imported get_memory_diff. HAS_MEMORY_TOOLS = {HAS_MEMORY_TOOLS}")
except ImportError as e:
    print(f"Error importing get_memory_diff: {e}")
    sys.exit(1)

# Import system get_repo_path if available
try:
    from scripts.memory_tools import get_repo_path as system_get_repo_path
    print(f"Successfully imported system_get_repo_path")
except ImportError as e:
    print(f"Error importing system_get_repo_path: {e}")
    system_get_repo_path = lambda: os.environ.get("MEMORY_REPO_PATH", os.path.join(repo_root, ".memory"))

async def test_memory_diff():
    """Test the get_memory_diff function with current memory repository."""
    try:
        # Get memory repository path
        memory_repo_path = system_get_repo_path()
        
        if not memory_repo_path:
            print("WARNING: Failed to get memory repository path")
            print("Using current directory as fallback")
            memory_repo_path = os.getcwd()
            
        print(f"Using memory repository path: {memory_repo_path}")
        
        # Verify git repository
        import subprocess
        
        try:
            # Check if git repository exists
            subprocess.run(
                ["git", "rev-parse", "--is-inside-work-tree"],
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Confirmed {memory_repo_path} is a git repository")
        except subprocess.CalledProcessError:
            print(f"ERROR: {memory_repo_path} is not a git repository")
            return None
            
        # Get current hash
        try:
            current_hash = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            print(f"Current git hash: {current_hash}")
        except subprocess.CalledProcessError as e:
            print(f"Failed to get current hash: {e}")
            print(f"Output: {e.output}")
            print(f"Stderr: {e.stderr}")
            return None
        
        # Get previous hash (HEAD~1)
        try:
            previous_hash = subprocess.run(
                ["git", "rev-parse", "HEAD~1"],
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            ).stdout.strip()
            print(f"Previous git hash: {previous_hash}")
        except subprocess.CalledProcessError:
            print("Failed to get previous commit hash. The repository might have only one commit.")
            print("Creating a dummy memory document to create a diff...")
            
            # Create a test document to generate a diff
            test_doc_path = Path(memory_repo_path) / "documents" / "test"
            test_doc_path.mkdir(exist_ok=True, parents=True)
            test_file = test_doc_path / f"test_memory_diff_{os.getpid()}.md"
            
            with open(test_file, "w") as f:
                f.write("# Test Memory Document\n\nThis is a test document created to generate a diff.")
            
            # Commit the test document
            try:
                subprocess.run(["git", "add", "."], cwd=memory_repo_path, check=True)
                subprocess.run(
                    ["git", "commit", "-m", "Add test document for memory diff test"],
                    cwd=memory_repo_path,
                    env={"GIT_AUTHOR_NAME": "Test Script", "GIT_AUTHOR_EMAIL": "test@example.com", 
                        "GIT_COMMITTER_NAME": "Test Script", "GIT_COMMITTER_EMAIL": "test@example.com"},
                    check=True
                )
                
                # Now get the updated hashes
                current_hash = subprocess.run(
                    ["git", "rev-parse", "HEAD"],
                    cwd=memory_repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                ).stdout.strip()
                
                previous_hash = subprocess.run(
                    ["git", "rev-parse", "HEAD~1"],
                    cwd=memory_repo_path,
                    capture_output=True,
                    text=True,
                    check=True
                ).stdout.strip()
            except subprocess.CalledProcessError as e:
                print(f"Failed to create test document: {e}")
                print(f"Output: {e.output if hasattr(e, 'output') else 'N/A'}")
                print(f"Stderr: {e.stderr if hasattr(e, 'stderr') else 'N/A'}")
                return None
        
        print(f"Comparing memory diff between:")
        print(f"  Previous hash: {previous_hash}")
        print(f"  Current hash:  {current_hash}")
        
        # Verify hashes exist
        try:
            for hash_val in [previous_hash, current_hash]:
                subprocess.run(
                    ["git", "cat-file", "-e", hash_val],
                    cwd=memory_repo_path,
                    check=True
                )
            print("Both git hashes are valid objects")
        except subprocess.CalledProcessError:
            print(f"ERROR: One or both git hashes are invalid")
            return None
        
        # Get the memory diff
        try:
            print(f"Calling get_memory_diff with:")
            print(f"  initial_hash: {previous_hash}")
            print(f"  final_hash: {current_hash}")
            print(f"  memory_repo_path: {memory_repo_path}")
            
            diff_result = await get_memory_diff(
                initial_hash=previous_hash,
                final_hash=current_hash,
                memory_repo_path=memory_repo_path
            )
            
            # Print the results
            print("\nMemory Diff Results:")
            print("===================")
            print(f"Changed Files: {json.dumps(diff_result['changed_files'], indent=2)}")
            print("\nDiff Summary:")
            print(diff_result['diff_summary'])
            print("\nDiff Content (truncated):")
            print(diff_result['diff_content'][:500])
            if len(diff_result['diff_content']) > 500:
                print("... (truncated)")
            
            return diff_result
        except Exception as e:
            print(f"Error getting memory diff: {e}")
            return None
        
    except Exception as e:
        print(f"Unexpected error in test_memory_diff: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Testing memory diff functionality...")
    diff_result = asyncio.run(test_memory_diff())
    if diff_result:
        print("\nTest completed successfully.")
    else:
        print("\nTest failed.")
        sys.exit(1) 