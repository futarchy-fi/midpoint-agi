#!/usr/bin/env python
"""
Example of reverting to a previous state using the cross-reference system.

This script demonstrates how to:
1. View the cross-reference history
2. Select a specific state to revert to
3. Perform the reversion
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

from scripts.memory_tools import get_repo_path, get_memory_for_code_hash

def view_history(limit=10):
    """Display the cross-reference history."""
    # Get repository path
    repo_path = get_repo_path()
    cross_ref_path = Path(repo_path) / "metadata" / "cross-reference.json"
    
    if not cross_ref_path.exists():
        print(f"Cross-reference file not found: {cross_ref_path}")
        return []
    
    # Load cross-reference
    with open(cross_ref_path, "r") as f:
        cross_ref = json.load(f)
    
    if "mappings" not in cross_ref:
        print("No history found in cross-reference file")
        return []
    
    # Sort by timestamp (newest first)
    mappings = sorted(cross_ref["mappings"], key=lambda m: m["timestamp"], reverse=True)
    
    # Limit results
    mappings = mappings[:limit]
    
    print("\nCross-reference history:")
    for i, mapping in enumerate(mappings):
        timestamp = mapping["timestamp"]
        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))
        print(f"{i+1}. {time_str}: Code {mapping['code_hash'][:7]} -> Memory {mapping['memory_hash'][:7]}")
    
    return mappings

def revert_to_state(code_hash, memory_hash=None, timestamp=None):
    """
    Revert both code and memory repositories to a specific state.
    
    Args:
        code_hash: The code repository hash to revert to
        memory_hash: The memory repository hash to revert to (optional)
        timestamp: The timestamp to find the closest memory hash (optional)
    """
    # Get memory repository path
    memory_repo_path = get_repo_path()
    
    # Find the memory hash if not provided
    if not memory_hash and code_hash:
        memory_hash = get_memory_for_code_hash(code_hash, timestamp=timestamp)
        if not memory_hash:
            print(f"No memory hash found for code hash: {code_hash}")
            return False
    
    # Get code repository path (default to parent of memory repo)
    code_repo_path = os.environ.get("CODE_REPO_PATH", str(Path(memory_repo_path).parent))
    
    print(f"\nReverting to state:")
    print(f"  Code hash: {code_hash}")
    print(f"  Memory hash: {memory_hash}")
    
    # Check if repos exist
    if not Path(code_repo_path).exists():
        print(f"Code repository not found: {code_repo_path}")
        return False
    
    if not Path(memory_repo_path).exists():
        print(f"Memory repository not found: {memory_repo_path}")
        return False
    
    # Perform the reversion
    try:
        print("\nReverting code repository...")
        subprocess.run(["git", "checkout", code_hash], cwd=code_repo_path, check=True)
        
        print("Reverting memory repository...")
        subprocess.run(["git", "checkout", memory_hash], cwd=memory_repo_path, check=True)
        
        print("\nReversion successful!")
        print(f"Both repositories are now at state: {code_hash[:7]} / {memory_hash[:7]}")
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"Error during reversion: {e}")
        return False

def interactive_revert():
    """Run an interactive reversion session."""
    print("Memory System - Interactive Reversion Tool")
    print("=========================================")
    
    # Show history
    mappings = view_history(limit=20)
    if not mappings:
        return
    
    # Get user selection
    try:
        selection = int(input("\nEnter the number of the state to revert to (0 to exit): "))
        if selection == 0:
            return
        
        if selection < 1 or selection > len(mappings):
            print(f"Invalid selection. Please enter a number between 1 and {len(mappings)}")
            return
        
        # Get the selected mapping
        mapping = mappings[selection - 1]
        code_hash = mapping["code_hash"]
        memory_hash = mapping["memory_hash"]
        
        # Confirm
        confirmation = input(f"\nRevert to state from {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(mapping['timestamp']))}? (y/n): ")
        if confirmation.lower() != 'y':
            print("Reversion cancelled.")
            return
        
        # Perform reversion
        revert_to_state(code_hash, memory_hash)
    
    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\nReversion cancelled.")

def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description="Memory state reversion tool")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # History command
    history_parser = subparsers.add_parser("history", help="View cross-reference history")
    history_parser.add_argument("--limit", type=int, default=10, help="Maximum entries to show")
    
    # Revert command
    revert_parser = subparsers.add_parser("revert", help="Revert to a specific state")
    revert_parser.add_argument("code_hash", help="Code repository hash")
    revert_parser.add_argument("--memory-hash", help="Memory repository hash (optional)")
    revert_parser.add_argument("--timestamp", type=int, help="Find memory hash closest to this timestamp")
    
    # Interactive command
    subparsers.add_parser("interactive", help="Interactive reversion mode")
    
    args = parser.parse_args()
    
    if args.command == "history":
        view_history(args.limit)
    
    elif args.command == "revert":
        revert_to_state(args.code_hash, args.memory_hash, args.timestamp)
    
    elif args.command == "interactive":
        interactive_revert()
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 