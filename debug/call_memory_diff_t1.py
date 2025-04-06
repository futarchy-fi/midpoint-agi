#!/usr/bin/env python
"""
Script to call get_memory_diff with T1 goal memory hashes.
"""

import os
import sys
import json
import asyncio
from pathlib import Path

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import the get_memory_diff function
try:
    from midpoint.agents.tools.memory_tools import get_memory_diff
    print("Successfully imported get_memory_diff")
except ImportError as e:
    print(f"Error importing get_memory_diff: {e}")
    sys.exit(1)

async def call_t1_memory_diff():
    """Call get_memory_diff with T1 goal memory hashes."""
    try:
        # T1 goal memory hashes from .goal/T1.json
        initial_hash = "0f442c923b56352ac8ec944db2ff75df3cd868f5"
        final_hash = "1f8cb055a453cd68533e9033bdabefa1050fc2ae"
        memory_repo_path = "/Users/kas/.midpoint/memory"
        
        print(f"Calling get_memory_diff with:")
        print(f"  initial_hash: {initial_hash}")
        print(f"  final_hash: {final_hash}")
        print(f"  memory_repo_path: {memory_repo_path}")
        
        # Get the memory diff
        diff_result = await get_memory_diff(
            initial_hash=initial_hash,
            final_hash=final_hash,
            memory_repo_path=memory_repo_path
        )
        
        # Print the results
        print("\nMemory Diff Results:")
        print("===================")
        print(f"Changed Files: {json.dumps(diff_result['changed_files'], indent=2)}")
        print("\nDiff Summary:")
        print(diff_result['diff_summary'])
        print("\nDiff Content:")
        print(diff_result['diff_content'])
        
        return diff_result
        
    except Exception as e:
        print(f"Error calling get_memory_diff: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Calling memory_diff for T1 goal...")
    diff_result = asyncio.run(call_t1_memory_diff())
    if diff_result:
        print("\nFunction call completed successfully.")
    else:
        print("\nFunction call failed.")
        sys.exit(1) 