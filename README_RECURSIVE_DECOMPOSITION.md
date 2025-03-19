# Recursive Goal Decomposition

This document explains how to use the recursive goal decomposition feature of the Midpoint system.

## Overview

Recursive goal decomposition is a powerful approach that automatically breaks down complex goals into smaller, more manageable subgoals until reaching directly executable tasks. It follows a depth-first search approach, decomposing each goal into the most promising next subgoal until reaching tasks that can be directly executed by a TaskExecutor.

## How It Works

1. The system starts with a high-level goal
2. The GoalDecomposer agent determines the most promising next step
3. The agent decides if this step needs further decomposition or can be directly executed
4. If decomposition is needed, the system creates a new subgoal and repeats from step 2
5. If the task is executable, it marks it as ready for execution
6. The system continues until all branches of the goal tree reach executable tasks

## Key Features

- **Hierarchical Decomposition**: Complex goals are broken down through multiple levels
- **Selective Context Passing**: Only relevant information is passed to child subgoals
- **Real-time Visualization**: Goals and subgoals are logged in a tree-like structure
- **Repository Validation**: Ensures the repository is in the expected state before decomposition
- **Resource Management**: Tracks points consumed for budgeting

## Running Recursive Decomposition

Use the `run_recursive_decomposition.py` script to demonstrate the recursive decomposition:

```bash
python run_recursive_decomposition.py <repository_path> <goal_description>
```

For example:

```bash
python run_recursive_decomposition.py /path/to/repo "Implement a user authentication system with registration, login, and profile management"
```

During execution, you can monitor progress with:

```bash
tail -f goal_hierarchy.log
```

## Output

The goal hierarchy log will show the decomposition process:

```
Starting recursive goal decomposition for: Implement a user authentication system
Repository: /path/to/repo
Git Hash: abc123...

Goal: Implement a user authentication system
└── Subgoal: Set up the basic authentication framework

  Goal: Set up the basic authentication framework
  └── Subgoal: Create user model and database schema

    Goal: Create user model and database schema
    └── Subgoal: Define user table with email, password hash, and profile fields
      ✓ READY FOR EXECUTION: Define user table with email, password hash, and profile fields

  Goal: Set up the basic authentication framework
  └── Subgoal: Implement password hashing utilities
    ✓ READY FOR EXECUTION: Implement password hashing utilities

Goal: Implement a user authentication system
└── Subgoal: Create user registration endpoint
  ✓ READY FOR EXECUTION: Create user registration endpoint

...

Recursive Decomposition Complete
Total subgoals: 8
Executable tasks: 5
```

## Programmatic Usage

You can also use the recursive decomposition in your code:

```python
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer, validate_repository_state
from midpoint.agents.tools import get_current_hash

# Create goal
goal = Goal(
    description="Implement feature X",
    validation_criteria=["X works", "All tests pass"],
    success_threshold=0.8
)

# Create context
context = TaskContext(
    state=State(
        git_hash=git_hash,
        description="Initial state",
        repository_path=repo_path
    ),
    goal=goal,
    iteration=0,
    points_consumed=0,
    total_budget=1000,
    execution_history=[]
)

# Run recursive decomposition
goal_decomposer = GoalDecomposer()
subgoals = await goal_decomposer.decompose_recursively(context)

# Find executable tasks
executable_tasks = [s for s in subgoals if not s.requires_further_decomposition]
```

## Extending the System

To extend the recursive decomposition system:

1. **Task Execution**: Implement the TaskExecutor agent to execute the leaf nodes (executable tasks)
2. **Failure Handling**: Add functionality to handle failures by backtracking to earlier decomposition points
3. **Parallel Decomposition**: Explore multiple decomposition paths simultaneously for efficiency
4. **Dynamic Adjustment**: Allow re-decomposition if execution fails or context changes

## Best Practices

- Provide clear, specific high-level goals with well-defined validation criteria
- Ensure the repository is in a clean state before starting decomposition
- Allocate sufficient budget for complex decomposition trees
- Use meaningful commit messages in the repository for better context
- Consider pruning the decomposition tree if it becomes too deep or broad 