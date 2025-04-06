#!/usr/bin/env python
"""
Script to compare specific memory hashes from G1 and S1 goals.
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

async def compare_specific_hashes():
    """Compare specific memory hashes from G1 and S1/T1 goals."""
    try:
        # The specific memory hashes we want to compare
        g1_memory_hash = "30ce45829825b9d679760ca0fd5526866e7877d5"  # reverted G1.json hash
        s1_memory_hash = "a00babccb37b6ddbf70d7b178bbabaca221a33e4"  # from S1.json and T1.json
        
        # Get memory repository path
        memory_repo_path = system_get_repo_path()
        
        if not memory_repo_path:
            print("WARNING: Failed to get memory repository path")
            print("Using .midpoint/memory as fallback")
            memory_repo_path = os.path.expanduser("~/.midpoint/memory")
            
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
        
        # Verify hashes exist
        try:
            for hash_val, hash_name in [(g1_memory_hash, "G1 (reverted)"), (s1_memory_hash, "S1/T1")]:
                try:
                    subprocess.run(
                        ["git", "cat-file", "-e", hash_val],
                        cwd=memory_repo_path,
                        check=True
                    )
                    print(f"Verified {hash_name} hash ({hash_val}) is a valid git object")
                except subprocess.CalledProcessError:
                    print(f"ERROR: {hash_name} hash ({hash_val}) is not a valid git object")
                    return None
        except Exception as e:
            print(f"Error verifying hashes: {e}")
            return None
        
        # Check if G1 memory state is empty
        try:
            print(f"\nChecking if G1 memory state (30ce4582) is empty...")
            # List all files in the repo at this commit
            cmd = ["git", "ls-tree", "-r", "--name-only", g1_memory_hash]
            proc = subprocess.run(
                cmd,
                cwd=memory_repo_path,
                capture_output=True,
                text=True,
                check=True
            )
            
            files = [file for file in proc.stdout.strip().split("\n") if file.strip()]
            
            if not files:
                print("G1 memory state appears to be empty (no files found)")
            else:
                print(f"G1 memory state contains {len(files)} files:")
                for file in files[:10]:  # Show first 10 files
                    print(f"  - {file}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more")
        except Exception as e:
            print(f"Error checking G1 memory contents: {e}")
        
        print(f"\nComparing memory diff between:")
        print(f"  G1 (reverted) hash: {g1_memory_hash}")
        print(f"  S1/T1 hash:        {s1_memory_hash}")
        
        # Get the memory diff
        try:
            print(f"Calling get_memory_diff...")
            
            diff_result = await get_memory_diff(
                initial_hash=g1_memory_hash,
                final_hash=s1_memory_hash,
                memory_repo_path=memory_repo_path
            )
            
            # Print the results
            print("\nMemory Diff Results:")
            print("===================")
            print(f"Changed Files: {json.dumps(diff_result['changed_files'], indent=2)}")
            print("\nDiff Summary:")
            print(diff_result['diff_summary'])
            print("\nDiff Content (truncated):")
            content_preview = diff_result['diff_content'][:2000]  # Show first 2000 chars
            print(content_preview)
            if len(diff_result['diff_content']) > 2000:
                print("... [content truncated, see full output in memory_diff.txt] ...")
                
                # Save full diff to a file
                diff_file = os.path.join(repo_root, "debug", "memory_diff.txt")
                with open(diff_file, "w") as f:
                    f.write(diff_result['diff_content'])
                print(f"Full diff content saved to {diff_file}")
            
            return diff_result
        except Exception as e:
            print(f"Error getting memory diff: {e}")
            import traceback
            traceback.print_exc()
            
            # Try alternate approach - direct git diff
            try:
                print("\nTrying direct git diff...")
                cmd = ["git", "diff", g1_memory_hash, s1_memory_hash]
                proc = subprocess.run(
                    cmd,
                    cwd=memory_repo_path,
                    capture_output=True,
                    text=True
                )
                
                if proc.returncode == 0:
                    print("\nDirect Git Diff Results (truncated):")
                    print("=======================")
                    direct_diff = proc.stdout
                    print(direct_diff[:2000])
                    if len(direct_diff) > 2000:
                        print("... [content truncated] ...")
                        # Save full diff to a file
                        diff_file = os.path.join(repo_root, "debug", "direct_git_diff.txt")
                        with open(diff_file, "w") as f:
                            f.write(direct_diff)
                        print(f"Full direct git diff saved to {diff_file}")
                else:
                    print(f"Direct git diff failed with error: {proc.stderr}")
            except Exception as e2:
                print(f"Direct git diff also failed: {e2}")
                
            return None
        
    except Exception as e:
        print(f"Unexpected error in compare_specific_hashes: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Comparing memory hashes from reverted G1 and S1/T1 goals...")
    diff_result = asyncio.run(compare_specific_hashes())
    if diff_result:
        print("\nComparison completed successfully.")
    else:
        print("\nComparison failed or produced no results.")
        sys.exit(1) 