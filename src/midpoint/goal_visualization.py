"""
Functions for visualizing goals and their relationships.
"""

import json
import logging
import datetime
import subprocess
from pathlib import Path

# Constants
GOAL_DIR = ".goal"
VISUALIZATION_DIR = f"{GOAL_DIR}/visualization"

def ensure_goal_dir():
    """Ensure the .goal directory exists."""
    goal_path = Path(GOAL_DIR)
    if not goal_path.exists():
        goal_path.mkdir()
        logging.info(f"Created goal directory: {GOAL_DIR}")
    return goal_path

def ensure_visualization_dir():
    """Ensure the visualization directory exists."""
    vis_path = Path(VISUALIZATION_DIR)
    if not vis_path.exists():
        vis_path.mkdir()
        logging.info(f"Created visualization directory: {VISUALIZATION_DIR}")
    return vis_path

def get_all_goal_files():
    """Get all goal files as a dictionary of id: data."""
    goal_path = ensure_goal_dir()
    goal_files = {}
    
    for file_path in goal_path.glob("*.json"):
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                goal_files[data["goal_id"]] = data
        except Exception as e:
            logging.warning(f"Failed to read goal file {file_path}: {e}")
    
    return goal_files

def show_goal_status():
    """Show completion status of all goals."""
    goal_files = get_all_goal_files()
    
    if not goal_files:
        print("No goals found")
        return
    
    def print_goal_status(goal_id, depth=0):
        if goal_id not in goal_files:
            return
        
        goal = goal_files[goal_id]
        indent = "  " * depth
        
        status = "" # Initialize status symbol

        # --- Determine Status Symbol --- 
        # Priority 1: Give Up from Analysis
        if goal.get('last_analysis', {}).get('suggested_action') == 'give_up':
            status = "❌"
        
        # Priority 2: Last Execution Failure (if not already marked give_up)
        elif "last_execution" in goal and not goal["last_execution"].get("success", True):
            status = "❌" 
        
        # Priority 3: Completion / Validation Status
        elif goal.get("complete", False) or goal.get("completed", False):
            if "validation_status" in goal and goal["validation_status"].get("last_score", 0) >= goal.get("success_threshold", 0.8):
                status = "✅"  # Complete and validated
            else:
                status = "🔷"  # Complete but not validated

        # Priority 4: Check for completed subgoals first
        elif (("completed_subgoals" in goal and goal["completed_subgoals"]) or 
              ("completed_tasks" in goal and goal["completed_tasks"]) or
              ("merged_subgoals" in goal and goal["merged_subgoals"])):
            status = "🔷"  # Has at least one completed subgoal (shown with blue diamond)

        # Priority 5: Pending/Decomposed Status (if no completed subgoals)
        else:
            # Goal-specific status
            subgoals = {k: v for k, v in goal_files.items() 
                      if v.get("parent_goal", "").upper() == goal_id.upper()}
            all_subgoals_complete = all(sg.get("complete", False) or sg.get("completed", False) for sg in subgoals.values())
            
            if not subgoals:
                status = "🔘"  # No subgoals (not yet decomposed)
            elif all_subgoals_complete:
                 # This state means goal should likely be complete or needs validation
                 status = "🔷" # Default to complete but not validated if somehow reached here
            else:
                status = "⚪"  # Some subgoals incomplete
        
        # Fallback
        if not status:
            status = "❔" # Default/Unknown status

        # Get progress text
        progress_text = ""
        
        # Use completed_subgoal_count and total_subgoal_count if available
        if "completed_subgoal_count" in goal and "total_subgoal_count" in goal:
            completed = goal["completed_subgoal_count"]
            total = goal["total_subgoal_count"]
            if total > 0:
                progress_text = f" ({completed}/{total})"
        # Fallback to completed_task_count if the new fields aren't available
        elif "completed_task_count" in goal and "total_task_count" in goal:
            completed = goal["completed_task_count"]
            total = goal["total_task_count"]
            if total > 0:
                progress_text = f" ({completed}/{total})"
        
        # Get memory hash information from current_state
        memory_hash = goal.get("current_state", {}).get("memory_hash", "")
        memory_hash_display = f" [mem:{memory_hash[:8]}]" if memory_hash else ""
        
        # Print goal line with memory hash
        print(f"{indent}{status} {goal_id}{progress_text}{memory_hash_display}: {goal['description']}")
        
        # Print completion timestamp if available
        if (goal.get("complete", False) or goal.get("completed", False)) and \
           ("completion_time" in goal or "completion_timestamp" in goal):
            completion_time = goal.get("completion_time") or goal.get("completion_timestamp")
            print(f"{indent}   Completed: {completion_time}")
        
        # Print current state if available
        if "current_state" in goal and not goal.get("complete", False):
            current_state = goal["current_state"]
            last_updated = current_state.get("last_updated", "")
            last_task = current_state.get("last_task", "")
            if last_updated and last_task:
                print(f"{indent}   Last updated: {last_updated} by task {last_task}")
        
        # Print last analysis recommendation if available
        if "last_analysis" in goal:
            last_analysis = goal["last_analysis"]
            action = last_analysis.get("suggested_action", "")
            timestamp = last_analysis.get("timestamp", "")
            cmd = last_analysis.get("suggested_command", "")
            
            if action:
                print(f"{indent}   Recommended action: {action}")
                if cmd:
                    print(f"{indent}   Suggested command: {cmd}")
        
        # Print completed subgoals if available (consolidated from previously separate sections)
        if "completed_subgoals" in goal and goal["completed_subgoals"] and not goal.get("complete", False):
            print(f"{indent}   Completed subgoals:")
            for subgoal in goal["completed_subgoals"]:
                subgoal_id = subgoal.get("subgoal_id", "")
                completion_time = subgoal.get("completion_time", "")
                subgoal_memory_hash = subgoal.get("final_state", {}).get("memory_hash", "")
                memory_hash_display = f" [mem:{subgoal_memory_hash[:8]}]" if subgoal_memory_hash else ""
                
                # Get validation score if available
                validation_score = ""
                if subgoal.get("validation_status") and isinstance(subgoal["validation_status"], dict):
                    score = subgoal["validation_status"].get("last_score")
                    if score is not None:
                        validation_score = f" (Validated: {score:.2%})"
                
                print(f"{indent}     - {subgoal_id}{memory_hash_display}{validation_score}: {subgoal.get('description', '')}")
        # Fallback to the old fields for backward compatibility
        elif not goal.get("complete", False):
            # Check and display completed_tasks (old format)
            if "completed_tasks" in goal and goal["completed_tasks"]:
                print(f"{indent}   Completed tasks:")
                for task in goal["completed_tasks"]:
                    task_id = task.get("task_id", "")
                    timestamp = task.get("timestamp", "")
                    task_memory_hash = task.get("final_state", {}).get("memory_hash", "")
                    memory_hash_display = f" [mem:{task_memory_hash[:8]}]" if task_memory_hash else ""
                    print(f"{indent}     - {task_id}{memory_hash_display} at {timestamp}")
            
            # Check and display merged_subgoals (old format)
            if "merged_subgoals" in goal and goal["merged_subgoals"]:
                print(f"{indent}   Merged subgoals:")
                for merge_info in goal["merged_subgoals"]:
                    subgoal_id = merge_info.get("subgoal_id", "")
                    merge_time = merge_info.get("merge_time", "")
                    print(f"{indent}     - {subgoal_id} at {merge_time}")
        
        # Print validation status if available
        if "validation_status" in goal:
            validation = goal["validation_status"]
            last_validated = validation.get("last_validated", "")
            last_score = validation.get("last_score", 0.0)
            last_validated_by = validation.get("last_validated_by", "")
            last_git_hash = validation.get("last_git_hash", "")
            
            if last_validated:
                print(f"{indent}   Last validated: {last_validated} by {last_validated_by}")
                print(f"{indent}   Validation score: {last_score:.2%}")
                if last_git_hash:
                    print(f"{indent}   Git hash: {last_git_hash[:8]}")
        elif goal.get("complete", False) or goal.get("completed", False):
            print(f"{indent}   ⚠️  Not yet validated")
        
        # Find and print children
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal", "").upper() == goal_id.upper()}
        
        for child_id in sorted(children.keys()):
            print_goal_status(child_id, depth + 1)
    
    # Find root goals (those without parents)
    root_goals = {k: v for k, v in goal_files.items() 
                  if not v.get("parent_goal")}
    
    # Print status for each root goal
    for root_id in sorted(root_goals.keys()):
        print_goal_status(root_id)

def show_goal_tree():
    """Show visual representation of goal hierarchy."""
    goal_files = get_all_goal_files()
    
    # Find top-level goals
    top_goals = {k: v for k, v in goal_files.items() if not v.get("parent_goal")}
    
    if not top_goals:
        print("No goals found.")
        return
    
    print("Goal Tree:")
    
    # Define a function to print goal tree recursively with better formatting
    def print_goal_tree(goal_id, prefix="", is_last=True, depth=0):
        if goal_id not in goal_files:
            return
            
        goal = goal_files[goal_id]
        
        # Check for state equality with parent
        states_equal = True
        parent_id = goal.get("parent_goal", "")
        if parent_id in goal_files:
            parent = goal_files[parent_id]
            
            # Check if this goal or task has an initial_state
            if "initial_state" in goal and "current_state" in parent:
                # Compare initial git hash with parent's current git hash
                goal_initial_hash = goal.get("initial_state", {}).get("git_hash", "")
                parent_current_hash = parent.get("current_state", {}).get("git_hash", "")
                
                # Compare initial memory hash with parent's current memory hash
                goal_initial_memory_hash = goal.get("initial_state", {}).get("memory_hash", "")
                parent_current_memory_hash = parent.get("current_state", {}).get("memory_hash", "")
                
                # Check git hash equality
                git_hash_equal = not (goal_initial_hash and parent_current_hash and goal_initial_hash != parent_current_hash)
                
                # Check memory hash equality
                memory_hash_equal = not (goal_initial_memory_hash and parent_current_memory_hash 
                                        and goal_initial_memory_hash != parent_current_memory_hash)
                
                # States are equal only if both git hash and memory hash are equal
                states_equal = git_hash_equal and memory_hash_equal
        
        # Determine status symbol
        if goal.get("complete", False) or goal.get("completed", False):
            status = "✅"
        else:
            # Special handling for tasks
            if goal.get("is_task", False):
                if "execution_result" in goal and goal["execution_result"].get("success"):
                    status = "✅"  # Completed task
                else:
                    if not states_equal:
                        status = "🔺"  # Branch task
                    else:
                        status = "🔷"  # Directly executable task
            else:
                # Check if all subgoals are complete
                # Use case-insensitive comparison
                subgoals = {k: v for k, v in goal_files.items() 
                           if v.get("parent_goal", "").upper() == goal_id.upper() or 
                              v.get("parent_goal", "").upper() == f"{goal_id.upper()}.json"}
                
                if not subgoals:
                    # Check if goal is a task
                    if goal.get("is_task", False):
                        if not states_equal:
                            status = "🔺"  # Branch task
                        else:
                            status = "🔷"  # Directly executable task
                    else:
                        status = "🔘"  # No subgoals (not yet decomposed)
                elif all(sg.get("complete", False) or sg.get("completed", False) for sg in subgoals.values()):
                    status = "⚪"  # All subgoals complete but needs explicit completion
                else:
                    if not states_equal:
                        status = "🔸"  # Branch subgoal
                    else:
                        status = "⚪"  # Some subgoals incomplete
        
        # Determine branch characters
        branch = "└── " if is_last else "├── "
        
        # Print the current goal
        print(f"{prefix}{branch}{status} {goal_id}: {goal['description']}")
        
        # Find children
        # Use case-insensitive comparison
        children = {k: v for k, v in goal_files.items() 
                   if v.get("parent_goal", "").upper() == goal_id.upper() or 
                      v.get("parent_goal", "").upper() == f"{goal_id.upper()}.json"}
        
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
    print("🟡 Partially completed")
    print("⚪ Incomplete")
    print("🔷 Directly executable task")
    print("🔘 Not yet decomposed")
    print("🔺 Branch task")
    print("🔸 Branch subgoal")

def show_goal_diffs(goal_id, show_code=True, show_memory=False):
    """Show code and memory diffs for a goal.
    
    Args:
        goal_id: The ID of the goal to show diffs for
        show_code: Whether to show code diff (default True)
        show_memory: Whether to show memory diff (default False)
    """
    from .goal_git import run_diff_command
    
    goal_path = ensure_goal_dir()
    goal_file = goal_path / f"{goal_id}.json"
    
    if not goal_file.exists():
        print(f"Goal {goal_id} not found")
        return False
    
    try:
        # Load goal data
        with open(goal_file, 'r') as f:
            goal_data = json.load(f)
    except Exception as e:
        print(f"Failed to read goal file: {e}")
        return False
    
    # Extract initial and current state information
    initial_state = goal_data.get("initial_state", {})
    current_state = goal_data.get("current_state", {})
    
    initial_hash = initial_state.get("git_hash")
    current_hash = current_state.get("git_hash")
    
    initial_memory_hash = initial_state.get("memory_hash")
    current_memory_hash = current_state.get("memory_hash")
    memory_repo_path = current_state.get("memory_repository_path")
    
    # Show code diff if requested
    if show_code and initial_hash and current_hash and initial_hash != current_hash:
        print(f"\n=== CODE DIFF for {goal_id} ===")
        print(f"Showing changes from {initial_hash[:8]} to {current_hash[:8]}")
        run_diff_command(initial_hash, current_hash)
    elif show_code:
        if not initial_hash or not current_hash:
            print("Missing git hash information for code diff")
        elif initial_hash == current_hash:
            print("No code changes (initial and current hash are the same)")
    
    # Show memory diff if requested
    if show_memory and memory_repo_path and initial_memory_hash and current_memory_hash and initial_memory_hash != current_memory_hash:
        print(f"\n=== MEMORY DIFF for {goal_id} ===")
        print(f"Showing memory changes from {initial_memory_hash[:8]} to {current_memory_hash[:8]}")
        run_diff_command(initial_memory_hash, current_memory_hash, repo_path=memory_repo_path)
    elif show_memory:
        if not memory_repo_path:
            print("No memory repository path specified")
        elif not initial_memory_hash or not current_memory_hash:
            print("Missing memory hash information for memory diff")
        elif initial_memory_hash == current_memory_hash:
            print("No memory changes (initial and current hash are the same)")
    
    return True 