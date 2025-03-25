"""
Command-line interface for goal branch management.
"""

import os
import json
import argparse
import logging
import datetime
import subprocess
import asyncio
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple

from .agents.models import Goal, SubgoalPlan
from .agents.goal_decomposer import decompose_goal as agent_decompose_goal

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

# Constants
GOAL_DIR = ".goal"
CHECKPOINT_DIR = f"{GOAL_DIR}/checkpoints"
VISUALIZATION_DIR = f"{GOAL_DIR}/visualization"


def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path


def ensure_checkpoint_dir():
    """Ensure the checkpoints directory exists."""
    checkpoint_path = Path(CHECKPOINT_DIR)
    if not checkpoint_path.exists():
        checkpoint_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created checkpoint directory: {CHECKPOINT_DIR}")
    return checkpoint_path


def ensure_visualization_dir():
    """Ensure the visualization directory exists."""
    vis_path = Path(VISUALIZATION_DIR)
    if not vis_path.exists():
        vis_path.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created visualization directory: {VISUALIZATION_DIR}")
    return vis_path


def generate_goal_id(parent_id=None):
    """Generate a goal ID in format G1 or G1-S1."""
    goal_path = ensure_goal_dir()
    
    if not parent_id:
        # Find next available top-level goal number
        existing = [f for f in goal_path.glob("G*.json")]
        next_num = len(existing) + 1
        return f"G{next_num}"
    else:
        # Find next available subgoal number for parent
        parent_base = parent_id.split('.')[0]  # Remove .json extension if present
        existing = [f for f in goal_path.glob(f"{parent_base}-S*.json")]
        next_num = len(existing) + 1
        return f"{parent_base}-S{next_num}"


def create_goal_file(goal_id, description, parent_id=None):
    """Create a goal file with the given ID and description."""
    goal_path = ensure_goal_dir()
    
    # Prepare the goal content
    goal_content = {
        "goal_id": goal_id,
        "description": description,
        "parent_goal": parent_id or "",
        "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    }
    
    # Write the goal file
    output_file = goal_path / f"{goal_id}.json"
    with open(output_file, 'w') as f:
        json.dump(goal_content, f, indent=2)
        
    logging.info(f"Created goal file: {output_file}")
    return str(output_file)


def list_goals():
    """List all goals and subgoals in tree format."""
    goal_path = ensure_goal_dir()
    
    # Get all goal files
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    # Find top-level goals
    top_goals = {k: v for k, v in goal_files.items() if not v["parent_goal"]}
    
    # Build tree structure
    def print_goal_tree(goal_id, depth=0):
        if goal_id not in goal_files:
            return
            
        goal = goal_files[goal_id]
        indent = "  " * depth
        print(f"{indent}• {goal_id}: {goal['description']}")
        
        # Find and print children
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal") == goal_id or 
                      v.get("parent_goal") == f"{goal_id}.json"}
        
        for child_id in sorted(children.keys()):
            print_goal_tree(child_id, depth + 1)
    
    # Print all top-level goals and their subgoals
    if top_goals:
        print("Goal Tree:")
        for goal_id in sorted(top_goals.keys()):
            print_goal_tree(goal_id)
    else:
        print("No goals found.")


def create_new_goal(description):
    """Create a new top-level goal."""
    goal_id = generate_goal_id()
    create_goal_file(goal_id, description)
    print(f"Created new goal {goal_id}: {description}")
    return goal_id


def create_new_subgoal(parent_id, description):
    """Create a new subgoal under the specified parent."""
    # Verify parent exists
    goal_path = ensure_goal_dir()
    parent_file = goal_path / f"{parent_id}.json"
    
    if not parent_file.exists():
        logging.error(f"Parent goal {parent_id} not found")
        return None
    
    # Generate subgoal ID
    subgoal_id = generate_goal_id(parent_id)
    
    # Create subgoal file
    create_goal_file(subgoal_id, description, parent_id)
    print(f"Created new subgoal {subgoal_id} under {parent_id}: {description}")
    return subgoal_id


def get_current_hash():
    """Get the current git hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current git hash: {e}")
        return None


def get_current_branch():
    """Get the current git branch."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get current branch: {e}")
        return None


def get_goal_id_from_branch(branch_name):
    """Extract goal ID from branch name."""
    # Branch naming convention: goal-G1-S1-abcdef (goal prefix, ID, random suffix)
    parts = branch_name.split('-')
    
    # Check if it's a goal branch
    if len(parts) < 3 or parts[0] != "goal":
        return None
    
    # Remove the 'goal' prefix and the random suffix
    if parts[1].startswith('G') and parts[1][1:].isdigit():
        if len(parts) >= 3 and parts[2].startswith('S') and parts[2][1:].isdigit():
            return f"{parts[1]}-{parts[2]}"
        return parts[1]
    
    return None


def get_parent_goal_id(goal_id):
    """Get the parent goal ID for a given goal ID."""
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal file not found: {goal_file}")
        return None
    
    try:
        with open(goal_file, 'r') as f:
            data = json.load(f)
            parent_id = data.get("parent_goal", "")
            if parent_id.endswith('.json'):
                parent_id = parent_id[:-5]  # Remove .json extension
            return parent_id
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return None


def find_branch_for_goal(goal_id):
    """Find the git branch for a specific goal ID."""
    try:
        # Get all branches
        result = subprocess.run(
            ["git", "branch"],
            check=True,
            capture_output=True,
            text=True
        )
        
        branches = result.stdout.strip().split('\n')
        branches = [b.strip() for b in branches]
        branches = [b[2:] if b.startswith('* ') else b for b in branches]  # Remove the * from current branch
        
        # Look for branches with the goal ID
        for branch in branches:
            if branch.startswith(f"goal-{goal_id}-") or branch == f"goal-{goal_id}":
                return branch
        
        return None
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to list branches: {e}")
        return None


def get_recent_commits(count=10):
    """Get recent commits."""
    try:
        result = subprocess.run(
            ["git", "log", f"-{count}", "--oneline"],
            check=True,
            capture_output=True,
            text=True
        )
        return result.stdout.strip().split('\n')
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to get recent commits: {e}")
        return []


def go_back_commits(steps=1):
    """Go back N commits on the current branch."""
    if steps < 1:
        logging.error("Number of steps must be at least 1")
        return False
    
    try:
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Get current hash before we move
        current_hash = get_current_hash()
        
        # Reset to N commits back
        reset_point = f"HEAD~{steps}"
        subprocess.run(
            ["git", "reset", "--hard", reset_point],
            check=True,
            capture_output=True,
            text=True
        )
        
        new_hash = get_current_hash()
        print(f"Moved back {steps} commit(s) on branch {current_branch}")
        print(f"From: {current_hash[:8]} -> To: {new_hash[:8]}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to go back commits: {e}")
        return False


def reset_to_commit(commit_id):
    """Reset to a specific commit on the current branch."""
    try:
        # Verify commit exists
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_id],
            check=True,
            capture_output=True,
            text=True
        )
        
        if "commit" not in result.stdout:
            logging.error(f"Invalid commit ID: {commit_id}")
            return False
        
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Get current hash before we move
        current_hash = get_current_hash()
        
        # Reset to the specified commit
        subprocess.run(
            ["git", "reset", "--hard", commit_id],
            check=True,
            capture_output=True,
            text=True
        )
        
        new_hash = get_current_hash()
        print(f"Reset to commit {commit_id} on branch {current_branch}")
        print(f"From: {current_hash[:8]} -> To: {new_hash[:8]}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to reset to commit: {e}")
        return False


def create_checkpoint(message):
    """Create a labeled checkpoint for easy navigation."""
    try:
        # Get current hash
        current_hash = get_current_hash()
        if not current_hash:
            return False
        
        # Get current branch
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        # Create checkpoint file
        checkpoint_path = ensure_checkpoint_dir()
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Sanitize message for filename
        safe_message = "".join(c if c.isalnum() else "_" for c in message)
        
        checkpoint_file = checkpoint_path / f"{timestamp}_{safe_message}.json"
        
        # Prepare checkpoint data
        checkpoint_data = {
            "timestamp": timestamp,
            "message": message,
            "git_hash": current_hash,
            "branch": current_branch
        }
        
        # Save checkpoint file
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        print(f"Created checkpoint: {message}")
        print(f"Branch: {current_branch}, Commit: {current_hash[:8]}")
        return True
    except Exception as e:
        logging.error(f"Failed to create checkpoint: {e}")
        return False


def list_checkpoints():
    """List all checkpoints."""
    checkpoint_path = ensure_checkpoint_dir()
    
    checkpoint_files = list(checkpoint_path.glob("*.json"))
    if not checkpoint_files:
        print("No checkpoints found.")
        return []
    
    checkpoints = []
    for file_path in sorted(checkpoint_files, reverse=True):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                checkpoints.append(data)
        except:
            logging.warning(f"Failed to read checkpoint file: {file_path}")
    
    if checkpoints:
        print("Checkpoints:")
        for i, cp in enumerate(checkpoints):
            print(f"{i+1}. [{cp['timestamp']}] {cp['message']} (Branch: {cp['branch']}, Commit: {cp['git_hash'][:8]})")
    else:
        print("No valid checkpoints found.")
    
    return checkpoints


def go_to_parent_goal():
    """Go to the parent goal branch."""
    # Get current branch
    current_branch = get_current_branch()
    if not current_branch:
        return False
    
    # Extract goal ID from branch
    goal_id = get_goal_id_from_branch(current_branch)
    if not goal_id:
        logging.error(f"Current branch is not a goal branch: {current_branch}")
        return False
    
    # Get the parent goal ID
    parent_id = get_parent_goal_id(goal_id)
    if not parent_id:
        logging.error(f"No parent goal found for {goal_id}")
        return False
    
    # Find the branch for the parent goal
    parent_branch = find_branch_for_goal(parent_id)
    if not parent_branch:
        logging.error(f"No branch found for parent goal {parent_id}")
        return False
    
    # Switch to the parent branch
    try:
        subprocess.run(
            ["git", "checkout", parent_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to parent goal: {parent_id}")
        print(f"Branch: {parent_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to parent branch: {e}")
        return False


def go_to_subgoal(subgoal_id):
    """Go to a specific subgoal branch."""
    # Verify subgoal exists
    goal_path = ensure_goal_dir()
    subgoal_file = goal_path / f"{subgoal_id}.json"
    
    if not subgoal_file.exists():
        logging.error(f"Subgoal not found: {subgoal_id}")
        return False
    
    # Find the branch for the subgoal
    subgoal_branch = find_branch_for_goal(subgoal_id)
    if not subgoal_branch:
        logging.error(f"No branch found for subgoal {subgoal_id}")
        return False
    
    # Switch to the subgoal branch
    try:
        subprocess.run(
            ["git", "checkout", subgoal_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to subgoal: {subgoal_id}")
        print(f"Branch: {subgoal_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to subgoal branch: {e}")
        return False


def go_to_root_goal():
    """Go to the top-level goal branch."""
    # Get current branch
    current_branch = get_current_branch()
    if not current_branch:
        return False
    
    # Extract goal ID from branch
    goal_id = get_goal_id_from_branch(current_branch)
    if not goal_id:
        logging.error(f"Current branch is not a goal branch: {current_branch}")
        return False
    
    # If this is already a top-level goal (G1, G2, etc.), we're already there
    if '-' not in goal_id:
        print(f"Already at root goal: {goal_id}")
        return True
    
    # Extract the root goal ID (G part)
    root_id = goal_id.split('-')[0]
    
    # Find the branch for the root goal
    root_branch = find_branch_for_goal(root_id)
    if not root_branch:
        logging.error(f"No branch found for root goal {root_id}")
        return False
    
    # Switch to the root branch
    try:
        subprocess.run(
            ["git", "checkout", root_branch],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"Switched to root goal: {root_id}")
        print(f"Branch: {root_branch}")
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to switch to root branch: {e}")
        return False


def list_subgoals(goal_id=None):
    """List subgoals for a specific goal ID or current branch."""
    # If no goal ID provided, use the current branch
    if not goal_id:
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        goal_id = get_goal_id_from_branch(current_branch)
        if not goal_id:
            logging.error(f"Current branch is not a goal branch: {current_branch}")
            return False
    
    # Get all goal files
    goal_path = ensure_goal_dir()
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            continue
    
    # Find direct subgoals
    subgoals = {k: v for k, v in goal_files.items() 
               if v.get("parent_goal") == goal_id or 
                  v.get("parent_goal") == f"{goal_id}.json"}
    
    if not subgoals:
        print(f"No subgoals found for goal {goal_id}")
        return []
    
    print(f"Subgoals for {goal_id}:")
    for subgoal_id in sorted(subgoals.keys()):
        subgoal = subgoals[subgoal_id]
        print(f"• {subgoal_id}: {subgoal['description']}")
    
    return list(subgoals.keys())


def mark_goal_complete(goal_id=None):
    """Mark a goal as complete."""
    # If no goal ID provided, use the current branch
    if not goal_id:
        current_branch = get_current_branch()
        if not current_branch:
            return False
        
        goal_id = get_goal_id_from_branch(current_branch)
        if not goal_id:
            logging.error(f"Current branch is not a goal branch: {current_branch}")
            return False
    
    # Verify goal exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal not found: {goal_id}")
        return False
    
    try:
        # Update goal file with completion status
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
        
        # Mark as complete with timestamp
        goal_data["complete"] = True
        goal_data["completion_time"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save updated goal data
        with open(goal_file, 'w') as f:
            json.dump(goal_data, f, indent=2)
        
        print(f"Marked goal {goal_id} as complete")
        
        # Get current git hash for reference
        current_hash = get_current_hash()
        if current_hash:
            print(f"Current commit: {current_hash[:8]}")
        
        return True
    except Exception as e:
        logging.error(f"Failed to mark goal as complete: {e}")
        return False


def merge_subgoal(subgoal_id, testing=False):
    """Merge a specific subgoal into the current goal."""
    # Verify subgoal exists
    goal_path = ensure_goal_dir()
    subgoal_file = goal_path / f"{subgoal_id}.json"
    
    if not subgoal_file.exists():
        logging.error(f"Subgoal not found: {subgoal_id}")
        return False
    
    # Verify that the subgoal is marked as complete
    try:
        with open(subgoal_file, 'r') as f:
            subgoal_data = json.load(f)
        
        if not subgoal_data.get("complete", False):
            logging.warning(f"Subgoal {subgoal_id} is not marked as complete. Merging incomplete goals is not recommended.")
            response = input("Are you sure you want to continue? (y/N): ")
            if response.lower() != 'y':
                print("Merge cancelled.")
                return False
    except Exception as e:
        logging.error(f"Failed to read subgoal file: {e}")
        return False
    
    # Get the parent goal ID
    parent_id = subgoal_data.get("parent_goal", "")
    if not parent_id:
        logging.error(f"Subgoal {subgoal_id} doesn't have a parent goal")
        return False
    
    if parent_id.endswith('.json'):
        parent_id = parent_id[:-5]  # Remove .json extension
    
    # Get current branch and verify we're on the parent branch
    current_branch = get_current_branch()
    parent_branch = find_branch_for_goal(parent_id)
    
    if not parent_branch:
        logging.error(f"No branch found for parent goal {parent_id}")
        return False
    
    if current_branch != parent_branch and not testing:
        logging.warning(f"Current branch ({current_branch}) is not the parent goal branch ({parent_branch})")
        response = input("Would you like to switch to the parent branch first? (Y/n): ")
        if response.lower() != 'n':
            try:
                subprocess.run(
                    ["git", "checkout", parent_branch],
                    check=True,
                    capture_output=True,
                    text=True
                )
                print(f"Switched to parent branch: {parent_branch}")
                current_branch = parent_branch
            except subprocess.CalledProcessError as e:
                logging.error(f"Failed to switch to parent branch: {e}")
                return False
    
    # Find the subgoal branch
    subgoal_branch = find_branch_for_goal(subgoal_id)
    if not subgoal_branch:
        logging.error(f"No branch found for subgoal {subgoal_id}")
        return False
    
    # Try to merge the subgoal branch into the current (parent) branch
    try:
        # First, try a merge with --no-commit to see if there are conflicts
        result = subprocess.run(
            ["git", "merge", "--no-commit", "--no-ff", subgoal_branch],
            capture_output=True,
            text=True
        )
        
        if "CONFLICT" in result.stderr or "CONFLICT" in result.stdout:
            print("Merge conflicts detected:")
            print(result.stderr)
            print(result.stdout)
            
            # Abort the merge
            subprocess.run(
                ["git", "merge", "--abort"],
                check=True,
                capture_output=True
            )
            
            print("Merge aborted due to conflicts.")
            print("Please resolve conflicts manually:")
            print(f"1. git checkout {parent_branch}")
            print(f"2. git merge {subgoal_branch}")
            print("3. Resolve conflicts")
            print("4. git add <resolved-files>")
            print("5. git commit -m 'Merge subgoal {subgoal_id}'")
            
            return False
        
        # If we got here, the merge can proceed without conflicts
        # Abort the --no-commit merge first
        subprocess.run(
            ["git", "reset", "--hard", "HEAD"],
            check=True,
            capture_output=True
        )
        
        # Now do the actual merge
        message = f"Merge subgoal {subgoal_id} into {parent_id}"
        subprocess.run(
            ["git", "merge", "--no-ff", "-m", message, subgoal_branch],
            check=True,
            capture_output=True
        )
        
        print(f"Successfully merged subgoal {subgoal_id} into {parent_id}")
        
        # Get the new commit hash
        new_hash = get_current_hash()
        if new_hash:
            print(f"Merge commit: {new_hash[:8]}")
        
        # Update parent goal with completion information
        parent_file = goal_path / f"{parent_id}.json"
        if parent_file.exists():
            try:
                with open(parent_file, 'r') as f:
                    parent_data = json.load(f)
                
                # Add merged subgoal to the list of merged subgoals
                if "merged_subgoals" not in parent_data:
                    parent_data["merged_subgoals"] = []
                
                merge_info = {
                    "subgoal_id": subgoal_id,
                    "merge_time": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "merge_commit": new_hash
                }
                
                parent_data["merged_subgoals"].append(merge_info)
                
                # Save updated parent data
                with open(parent_file, 'w') as f:
                    json.dump(parent_data, f, indent=2)
            except Exception as e:
                logging.error(f"Failed to update parent goal metadata: {e}")
        
        return True
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to merge subgoal: {e}")
        return False


def show_goal_status():
    """Show completion status of all goals."""
    goal_path = ensure_goal_dir()
    
    # Get all goal files
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    # Find top-level goals
    top_goals = {k: v for k, v in goal_files.items() if not v["parent_goal"]}
    
    if not top_goals:
        print("No goals found.")
        return
    
    print("Goal Status:")
    
    # Define a function to print goal status recursively
    def print_goal_status(goal_id, depth=0):
        if goal_id not in goal_files:
            return
            
        goal = goal_files[goal_id]
        indent = "  " * depth
        
        # Determine status symbol
        if goal.get("complete", False):
            status = "✅"
        else:
            # Check if all subgoals are complete
            subgoals = {k: v for k, v in goal_files.items() 
                       if v.get("parent_goal") == goal_id or 
                          v.get("parent_goal") == f"{goal_id}.json"}
            
            if not subgoals:
                status = "🔘"  # No subgoals
            elif all(sg.get("complete", False) for sg in subgoals.values()):
                status = "🟠"  # All subgoals complete but not merged
            else:
                status = "⚪"  # Some subgoals incomplete
        
        print(f"{indent}{status} {goal_id}: {goal['description']}")
        
        # Print completion timestamp if available
        if goal.get("complete", False) and "completion_time" in goal:
            completion_time = goal["completion_time"]
            print(f"{indent}   Completed: {completion_time}")
        
        # Print merged subgoals if available
        if "merged_subgoals" in goal and goal["merged_subgoals"]:
            for merge_info in goal["merged_subgoals"]:
                subgoal_id = merge_info.get("subgoal_id", "")
                merge_time = merge_info.get("merge_time", "")
                print(f"{indent}   Merged: {subgoal_id} at {merge_time}")
        
        # Find and print children
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal") == goal_id or 
                      v.get("parent_goal") == f"{goal_id}.json"}
        
        for child_id in sorted(children.keys()):
            print_goal_status(child_id, depth + 1)
    
    # Print all top-level goals and their subgoals
    for goal_id in sorted(top_goals.keys()):
        print_goal_status(goal_id)
    
    # Print legend
    print("\nStatus Legend:")
    print("✅ Complete")
    print("🟠 All subgoals complete (ready to merge)")
    print("⚪ Incomplete (some subgoals pending)")
    print("🔘 No subgoals")


def show_goal_tree():
    """Show visual representation of goal hierarchy."""
    goal_path = ensure_goal_dir()
    
    # Get all goal files
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    # Find top-level goals
    top_goals = {k: v for k, v in goal_files.items() if not v["parent_goal"]}
    
    if not top_goals:
        print("No goals found.")
        return
    
    print("Goal Tree:")
    
    # Define a function to print goal tree recursively with better formatting
    def print_goal_tree(goal_id, prefix="", is_last=True, depth=0):
        if goal_id not in goal_files:
            return
            
        goal = goal_files[goal_id]
        
        # Determine status symbol
        if goal.get("complete", False):
            status = "✅"
        else:
            # Check if all subgoals are complete
            subgoals = {k: v for k, v in goal_files.items() 
                       if v.get("parent_goal") == goal_id or 
                          v.get("parent_goal") == f"{goal_id}.json"}
            
            if not subgoals:
                status = "🔘"  # No subgoals
            elif all(sg.get("complete", False) for sg in subgoals.values()):
                status = "🟠"  # All subgoals complete but not merged
            else:
                status = "⚪"  # Some subgoals incomplete
        
        # Determine branch characters
        branch = "└── " if is_last else "├── "
        
        # Print the current goal
        print(f"{prefix}{branch}{status} {goal_id}: {goal['description']}")
        
        # Find children
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal") == goal_id or 
                      v.get("parent_goal") == f"{goal_id}.json"}
        
        # Sort children by ID
        sorted_children = sorted(children.keys())
        
        # Determine new prefix for children
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        # Print children
        for i, child_id in enumerate(sorted_children):
            is_last_child = (i == len(sorted_children) - 1)
            print_goal_tree(child_id, new_prefix, is_last_child, depth + 1)
    
    # Print all top-level goals and their subgoals
    sorted_top_goals = sorted(top_goals.keys())
    for i, goal_id in enumerate(sorted_top_goals):
        is_last = (i == len(sorted_top_goals) - 1)
        print_goal_tree(goal_id, "", is_last)
    
    # Print legend
    print("\nStatus Legend:")
    print("✅ Complete")
    print("🟠 All subgoals complete (ready to merge)")
    print("⚪ Incomplete (some subgoals pending)")
    print("🔘 No subgoals")


def show_goal_history():
    """Show timeline of goal exploration."""
    goal_path = ensure_goal_dir()
    
    # Get all goal files and sort by timestamp
    goals = []
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goals.append(data)
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    # Sort goals by timestamp
    goals.sort(key=lambda x: x.get("timestamp", ""))
    
    if not goals:
        print("No goals found.")
        return
    
    print("Goal History:")
    print("=============")
    
    # Print goals in chronological order
    for goal in goals:
        goal_id = goal.get("goal_id", "Unknown")
        desc = goal.get("description", "No description")
        timestamp = goal.get("timestamp", "Unknown time")
        parent = goal.get("parent_goal", "")
        
        # Format timestamp
        if len(timestamp) >= 8:
            try:
                timestamp = f"{timestamp[:4]}-{timestamp[4:6]}-{timestamp[6:8]} {timestamp[9:11]}:{timestamp[11:13]}:{timestamp[13:15]}"
            except:
                pass  # Keep original format if parsing fails
        
        # Print basic goal info
        print(f"[{timestamp}] {goal_id}: {desc}")
        
        # Print parent info if available
        if parent:
            print(f"  └── Subgoal of: {parent}")
        
        # Print completion info if available
        if goal.get("complete", False):
            completion_time = goal.get("completion_time", "Unknown time")
            if len(completion_time) >= 8:
                try:
                    completion_time = f"{completion_time[:4]}-{completion_time[4:6]}-{completion_time[6:8]} {completion_time[9:11]}:{completion_time[11:13]}:{completion_time[13:15]}"
                except:
                    pass  # Keep original format if parsing fails
            print(f"  └── Completed: {completion_time}")
        
        # Print merged subgoals if available
        if "merged_subgoals" in goal and goal["merged_subgoals"]:
            for merge_info in goal["merged_subgoals"]:
                subgoal_id = merge_info.get("subgoal_id", "")
                merge_time = merge_info.get("merge_time", "")
                if len(merge_time) >= 8:
                    try:
                        merge_time = f"{merge_time[:4]}-{merge_time[4:6]}-{merge_time[6:8]} {merge_time[9:11]}:{merge_time[11:13]}:{merge_time[13:15]}"
                    except:
                        pass  # Keep original format if parsing fails
                print(f"  └── Merged: {subgoal_id} at {merge_time}")
        
        print("")  # Empty line between goals
    
    # Print summary
    total_goals = len(goals)
    complete_goals = sum(1 for g in goals if g.get("complete", False))
    print(f"Summary: {complete_goals}/{total_goals} goals completed")


def generate_graph():
    """Generate a graphical visualization of the goal hierarchy."""
    try:
        # Check if graphviz is installed
        subprocess.run(["dot", "-V"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: Graphviz is not installed. Please install Graphviz to use this feature.")
        print("Installation instructions: https://graphviz.org/download/")
        return False
    
    goal_path = ensure_goal_dir()
    visualization_path = ensure_visualization_dir()
    
    # Get all goal files
    goal_files = {}
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except:
            logging.warning(f"Failed to read goal file: {file_path}")
    
    if not goal_files:
        print("No goals found.")
        return False
    
    # Create DOT file content
    dot_content = [
        "digraph Goals {",
        "  rankdir=TB;",
        "  node [shape=box, style=filled, fontname=Arial];",
        "  edge [fontname=Arial];",
        ""
    ]
    
    # Add nodes (goals)
    for goal_id, goal_data in goal_files.items():
        # Determine node color based on status
        if goal_data.get("complete", False):
            color = "green"
        else:
            # Check if all subgoals are complete
            subgoals = {k: v for k, v in goal_files.items() 
                      if v.get("parent_goal") == goal_id or 
                         v.get("parent_goal") == f"{goal_id}.json"}
            
            if not subgoals:
                color = "gray"  # No subgoals
            elif all(sg.get("complete", False) for sg in subgoals.values()):
                color = "orange"  # All subgoals complete but not merged
            else:
                color = "lightblue"  # Some subgoals incomplete
        
        # Escape special characters
        desc = goal_data.get("description", "").replace('"', '\\"')
        
        # Add node
        dot_content.append(f'  "{goal_id}" [label="{goal_id}\\n{desc}", fillcolor={color}];')
    
    # Add edges (parent-child relationships)
    for goal_id, goal_data in goal_files.items():
        parent_id = goal_data.get("parent_goal", "")
        if parent_id:
            if parent_id.endswith('.json'):
                parent_id = parent_id[:-5]  # Remove .json extension
            dot_content.append(f'  "{parent_id}" -> "{goal_id}";')
    
    # Close the graph
    dot_content.append("}")
    
    # Write DOT file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    dot_file = visualization_path / f"goals_{timestamp}.dot"
    pdf_file = visualization_path / f"goals_{timestamp}.pdf"
    png_file = visualization_path / f"goals_{timestamp}.png"
    
    with open(dot_file, 'w') as f:
        f.write("\n".join(dot_content))
    
    # Generate PDF
    try:
        subprocess.run(
            ["dot", "-Tpdf", str(dot_file), "-o", str(pdf_file)],
            check=True,
            capture_output=True
        )
        print(f"PDF graph generated: {pdf_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate PDF: {e}")
    
    # Generate PNG
    try:
        subprocess.run(
            ["dot", "-Tpng", str(dot_file), "-o", str(png_file)],
            check=True,
            capture_output=True
        )
        print(f"PNG graph generated: {png_file}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to generate PNG: {e}")
    
    return True


async def decompose_existing_goal(goal_id, debug=False, quiet=False, bypass_validation=False):
    """Decompose an existing goal into subgoals using the GoalDecomposer."""
    # Verify goal exists
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        logging.error(f"Goal {goal_id} not found")
        return False
    
    # Load goal content
    try:
        with open(goal_file, 'r') as f:
            goal_content = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read goal file: {e}")
        return False
    
    # Get repository path (current directory)
    repo_path = os.getcwd()
    
    # Call the goal decomposer
    try:
        result = await agent_decompose_goal(
            repo_path=repo_path,
            goal=goal_content["description"],
            parent_goal=goal_id,
            goal_id=goal_id,  # Pass the goal_id explicitly
            debug=debug,
            quiet=quiet,
            bypass_validation=bypass_validation
        )
        
        if result["success"]:
            print(f"\nGoal {goal_id} successfully decomposed into subgoals")
            print(f"\nNext step: {result['next_step']}")
            print("\nValidation criteria:")
            for criterion in result["validation_criteria"]:
                print(f"- {criterion}")
            
            if result["requires_further_decomposition"]:
                print("\nRequires further decomposition: Yes")
            else:
                print("\nRequires further decomposition: No")
            
            print(f"\nGoal file: {result['goal_file']}")
            return True
        else:
            logging.error(f"Failed to decompose goal: {result.get('error', 'Unknown error')}")
            return False
    except Exception as e:
        logging.error(f"Error during goal decomposition: {str(e)}")
        return False


async def async_main(args):
    """Async entry point for CLI commands."""
    # Handle commands
    if args.command == "decompose":
        return await decompose_existing_goal(args.goal_id, args.debug, args.quiet, args.bypass_validation)
    
    # All other commands are synchronous, so just call them directly
    if args.command == "new":
        return create_new_goal(args.description)
    elif args.command == "sub":
        return create_new_subgoal(args.parent_id, args.description)
    elif args.command == "list":
        return list_goals()
    elif args.command == "back":
        return go_back_commits(args.steps)
    elif args.command == "reset":
        return reset_to_commit(args.commit_id)
    elif args.command == "checkpoint":
        return create_checkpoint(args.message)
    elif args.command == "checkpoints":
        return list_checkpoints()
    elif args.command == "up":
        return go_to_parent_goal()
    elif args.command == "down":
        return go_to_subgoal(args.subgoal_id)
    elif args.command == "root":
        return go_to_root_goal()
    elif args.command == "subs":
        return list_subgoals()
    elif args.command == "complete":
        return mark_goal_complete()
    elif args.command == "merge":
        return merge_subgoal(args.subgoal_id)
    elif args.command == "status":
        return show_goal_status()
    elif args.command == "tree":
        return show_goal_tree()
    elif args.command == "history":
        return show_goal_history()
    elif args.command == "graph":
        return generate_graph()
    else:
        return None


def main():
    """Main CLI entry point."""
    import asyncio
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Goal branch management commands")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Goal Management Commands
    # -----------------------
    # goal new <description>
    new_parser = subparsers.add_parser("new", help="Create a new top-level goal")
    new_parser.add_argument("description", help="Description of the goal")
    
    # goal sub <parent-id> <description>
    sub_parser = subparsers.add_parser("sub", help="Create a subgoal under the specified parent")
    sub_parser.add_argument("parent_id", help="Parent goal ID")
    sub_parser.add_argument("description", help="Description of the subgoal")
    
    # goal list
    subparsers.add_parser("list", help="List all goals and subgoals in tree format")
    
    # goal decompose <goal-id>
    decompose_parser = subparsers.add_parser("decompose", help="Decompose a goal into subgoals")
    decompose_parser.add_argument("goal_id", help="Goal ID to decompose")
    decompose_parser.add_argument("--debug", action="store_true", help="Show debug output")
    decompose_parser.add_argument("--quiet", action="store_true", help="Only show warnings and result")
    decompose_parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    
    # State Navigation Commands
    # ------------------------
    # goal back [steps]
    back_parser = subparsers.add_parser("back", help="Go back N commits on current goal branch")
    back_parser.add_argument("steps", nargs="?", type=int, default=1, help="Number of commits to go back")
    
    # goal reset <commit-id>
    reset_parser = subparsers.add_parser("reset", help="Reset to specific commit on current branch")
    reset_parser.add_argument("commit_id", help="Commit ID to reset to")
    
    # goal checkpoint "message"
    checkpoint_parser = subparsers.add_parser("checkpoint", help="Create labeled checkpoint for easy navigation")
    checkpoint_parser.add_argument("message", help="Checkpoint message")
    
    # goal checkpoints
    subparsers.add_parser("checkpoints", help="List all checkpoints")
    
    # Hierarchy Navigation Commands
    # ---------------------------
    # goal up
    subparsers.add_parser("up", help="Go to parent goal branch")
    
    # goal down <subgoal-id>
    down_parser = subparsers.add_parser("down", help="Go to specific subgoal branch")
    down_parser.add_argument("subgoal_id", help="Subgoal ID to navigate to")
    
    # goal root
    subparsers.add_parser("root", help="Go to top-level goal")
    
    # goal subs
    subparsers.add_parser("subs", help="List available subgoals for current goal")
    
    # Result Incorporation Commands
    # ----------------------------
    # goal complete
    subparsers.add_parser("complete", help="Mark current goal as complete")
    
    # goal merge <subgoal-id>
    merge_parser = subparsers.add_parser("merge", help="Merge specific subgoal into current goal")
    merge_parser.add_argument("subgoal_id", help="Subgoal ID to merge")
    
    # goal status
    subparsers.add_parser("status", help="Show completion status of all goals")
    
    # Visualization Tools
    # ------------------
    # goal tree
    subparsers.add_parser("tree", help="Show visual representation of goal hierarchy")
    
    # goal history
    subparsers.add_parser("history", help="Show timeline of goal exploration")
    
    # goal graph
    subparsers.add_parser("graph", help="Generate graphical visualization")
    
    args = parser.parse_args()
    
    # Handle async commands with a single asyncio.run call
    if args.command == "decompose":
        asyncio.run(async_main(args))
    else:
        # Handle synchronous commands directly
        if args.command == "new":
            create_new_goal(args.description)
        elif args.command == "sub":
            create_new_subgoal(args.parent_id, args.description)
        elif args.command == "list":
            list_goals()
        elif args.command == "back":
            go_back_commits(args.steps)
        elif args.command == "reset":
            reset_to_commit(args.commit_id)
        elif args.command == "checkpoint":
            create_checkpoint(args.message)
        elif args.command == "checkpoints":
            list_checkpoints()
        elif args.command == "up":
            go_to_parent_goal()
        elif args.command == "down":
            go_to_subgoal(args.subgoal_id)
        elif args.command == "root":
            go_to_root_goal()
        elif args.command == "subs":
            list_subgoals()
        elif args.command == "complete":
            mark_goal_complete()
        elif args.command == "merge":
            merge_subgoal(args.subgoal_id)
        elif args.command == "status":
            show_goal_status()
        elif args.command == "tree":
            show_goal_tree()
        elif args.command == "history":
            show_goal_history()
        elif args.command == "graph":
            generate_graph()
        else:
            parser.print_help()


if __name__ == "__main__":
    main() 