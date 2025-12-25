# Midpoint Goal Management System

This directory contains the Midpoint Goal Management System, a hierarchical goal and task management tool designed for AI agents.

## Code Organization

The code has been refactored to improve maintainability and make it easier to understand. Below is the organization of the main modules:

### Core Modules

- `goal_commands.py`: Main CLI command dispatcher and argument parser
- `goal_file_management.py`: Functions for managing goal files (listing, ID generation)
- `goal_git.py`: Git-related operations (branches, commits, diffs)
- `goal_state.py`: Goal state management (creation, completion, merging)
- `goal_visualization.py`: Visualization functions (status, tree, history, graph)
- `goal_decompose_command.py`: Goal decomposition implementation
- `goal_execute_command.py`: Task execution implementation
- `goal_revert.py`: Goal reversion functionality

### Agent Modules

- `agents/goal_analyzer.py`: Analyzes goals and recommends next actions
- `agents/goal_decomposer.py`: Decomposes goals into subgoals and tasks
- `agents/goal_validator.py`: Validates goal completion criteria
- `agents/task_executor.py`: Executes individual tasks
- `agents/models.py`: Data models for the agent system
- `agents/tools/`: Various tools used by agents

### Goal Operations

- `goal_operations/goal_update.py`: Functions for updating parent goals based on child outcomes

## Main Functions and Flow

1. **Goal Creation & Management**
   - `create_new_goal`: Creates a top-level goal with its own branch
   - `create_new_child_goal`: Creates a child goal (subgoal or task) under a parent goal
   - `mark_goal_complete`: Marks a goal as complete

2. **Goal Execution**
   - `decompose_existing_goal`: Breaks down a goal into subgoals and tasks
   - `execute_task`: Executes a specific task

3. **Visualization & Status**
   - `show_goal_status`: Shows status of all goals
   - `show_goal_tree`: Displays hierarchical tree of goals
   - `show_goal_history`: Shows timeline of goal exploration
   - `generate_graph`: Creates graphical visualization
   - `show_goal_diffs`: Shows code and memory diffs for a goal

## Development Guidelines

When modifying this codebase:

1. Place new functionality in the appropriate module
2. Update imports in `goal_commands.py` for new functionality
3. Keep each module focused on a single responsibility
4. Add proper docstrings to all functions and classes
5. Use typing annotations for better code clarity
6. Test new functionality with simple end-to-end tests

## Recent Refactoring

The codebase was recently refactored to improve maintainability:

1. Visualization functions were moved to `goal_visualization.py`
2. State management functions were moved to `goal_state.py`
3. File operations were consolidated in `goal_file_management.py`
4. Git operations were moved to `goal_git.py`
5. Command dispatching was updated in `goal_commands.py`

The `goal_cli.py` module now serves primarily as a backwards compatibility layer and will eventually be deprecated in favor of the new modular structure. 