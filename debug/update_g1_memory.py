#!/usr/bin/env python
"""
Script to update G1.json to use the initial memory commit as its state.
"""

import os
import sys
import json
from pathlib import Path

# Initial commit hash we want to use
ROOT_MEMORY_HASH = "228836718f5d96e36b3548da90dbb980b7cd4176"

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

def update_g1_memory():
    """Update G1.json to use the initial memory commit."""
    
    # Path to G1.json
    g1_path = repo_root / ".goal" / "G1.json"
    
    if not g1_path.exists():
        print(f"ERROR: G1.json not found at {g1_path}")
        return False
    
    # Load G1.json
    try:
        with open(g1_path, 'r') as f:
            g1_data = json.load(f)
    except Exception as e:
        print(f"ERROR: Failed to load G1.json: {e}")
        return False
    
    # Create a backup of the original file
    backup_path = repo_root / ".goal" / "G1.json.bak"
    try:
        with open(backup_path, 'w') as f:
            json.dump(g1_data, f, indent=2)
        print(f"Created backup of G1.json at {backup_path}")
    except Exception as e:
        print(f"WARNING: Failed to create backup: {e}")
    
    # Current memory hash in initial_state
    current_initial_hash = g1_data.get("initial_state", {}).get("memory_hash")
    print(f"Current initial memory hash: {current_initial_hash}")
    
    # Current memory hash in current_state
    current_final_hash = g1_data.get("current_state", {}).get("memory_hash")
    print(f"Current current_state memory hash: {current_final_hash}")
    
    # Update initial_state memory_hash
    if "initial_state" in g1_data:
        g1_data["initial_state"]["memory_hash"] = ROOT_MEMORY_HASH
        print(f"Updated initial_state memory_hash to {ROOT_MEMORY_HASH}")
    else:
        print("WARNING: initial_state not found in G1.json")
    
    # Update current_state memory_hash
    if "current_state" in g1_data:
        g1_data["current_state"]["memory_hash"] = ROOT_MEMORY_HASH
        print(f"Updated current_state memory_hash to {ROOT_MEMORY_HASH}")
    else:
        print("WARNING: current_state not found in G1.json")

    # Update description in current_state
    if "current_state" in g1_data:
        # Copy description from initial_state
        if "initial_state" in g1_data and "description" in g1_data["initial_state"]:
            g1_data["current_state"]["description"] = g1_data["initial_state"]["description"]
            print(f"Updated current_state description to match initial_state")
    
    # If decomposed flag exists, set it to False
    if "decomposed" in g1_data:
        original_decomposed = g1_data["decomposed"]
        g1_data["decomposed"] = False
        print(f"Updated decomposed flag from {original_decomposed} to False")
    
    # Update completed_tasks to empty list if it exists
    if "completed_tasks" in g1_data:
        original_tasks = g1_data["completed_tasks"]
        g1_data["completed_tasks"] = []
        print(f"Cleared completed_tasks (was: {original_tasks})")
    
    # Remove next_step, reasoning and relevant_context if they exist
    for field in ["next_step", "reasoning", "relevant_context"]:
        if field in g1_data:
            print(f"Removed {field} field")
            del g1_data[field]
    
    # Save the updated G1.json
    try:
        with open(g1_path, 'w') as f:
            json.dump(g1_data, f, indent=2)
        print(f"Successfully updated G1.json")
        return True
    except Exception as e:
        print(f"ERROR: Failed to save updated G1.json: {e}")
        return False

if __name__ == "__main__":
    print("Updating G1.json to use the initial memory commit...")
    success = update_g1_memory()
    if success:
        print("\nG1.json updated successfully to use initial memory commit.")
        print(f"Memory hash set to: {ROOT_MEMORY_HASH}")
        print("Decomposed flag set to: False")
        print("Completed tasks cleared")
        print("Removed extra fields (next_step, reasoning, relevant_context)")
    else:
        print("\nFailed to update G1.json.")
        sys.exit(1) 