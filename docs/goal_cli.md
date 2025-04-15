# Goal Management CLI

The goal management CLI provides a set of commands for managing goals and subgoals in your project. It helps you track progress, navigate between different tasks, and visualize the hierarchy of goals.

## Installation

The goal CLI is automatically installed when you install the Midpoint package:

```bash
pip install -e .
```

## Core Concepts

### Goal IDs
Goals are identified using a simple flat ID system:
- Top-level goals: `G1`, `G2`, etc.
- Subgoals: `S1`, `S2`, etc.
- Tasks: `T1`, `T2`, etc.

Each goal or task has a unique ID regardless of its position in the hierarchy. Parent-child relationships are tracked internally through metadata.

### Goal Metadata
Goal metadata is stored in `.goal` directory with JSON files named after the goal ID.

### Branch Naming
Each goal is associated with a git branch, typically named `goal-<ID>-<random_suffix>`.

## Commands

### Goal Management

#### Create a new top-level goal
```bash
goal new "Description of the goal"
```

#### Create a subgoal under a parent goal
```bash
goal sub <parent-id> "Description of the subgoal"
```

#### List all goals and subgoals
```bash
goal list
```

### State Navigation

#### Go back N commits on current branch
```bash
goal back [steps]
```
Default is 1 step if not specified.

#### Reset to a specific commit on the current branch
```bash
goal reset <commit-id>
```


### Hierarchy Navigation

#### Go to the parent goal branch
```bash
goal up
```

#### Go to a specific subgoal branch
```bash
goal down <subgoal-id>
```

#### Go to the top-level goal branch
```bash
goal root
```

#### List available subgoals for current goal
```bash
goal subs
```

### Result Incorporation

#### Mark current goal as complete
```bash
goal complete
```

#### Merge a specific subgoal into the current goal
```bash
goal merge <subgoal-id>
```

#### Validate a goal's completion criteria
```bash
goal validate <goal-id>
```
This command will:
1. Display the goal's validation criteria
2. Guide you through validating each criterion
3. Record your validation decisions and reasoning
4. Calculate an overall validation score
5. Save the validation history
6. Update the goal's validation status

Options:
- `--debug`: Show debug output
- `--quiet`: Only show warnings and result

#### Show completion status of all goals
```bash
goal status
```

#### Show code and memory diffs for a goal
```bash
goal diff <goal-id>
```
This command displays the `git diff` output between the initial and current states recorded for the specified goal, separately for the code repository and the memory repository (if configured and available in the goal's state). This is useful for understanding the exact changes made during the goal's execution.

### Visualization Tools

#### Show visual representation of goal hierarchy
```bash
goal tree
```

#### Show timeline of goal exploration
```bash
goal history
```

#### Generate graphical visualization
```bash
goal graph
```
Requires [Graphviz](https://graphviz.org/download/) to be installed.

For detailed examples of the visualization outputs, please see the [Goal CLI Visualization Guide](goal_cli_visuals.md).

#### Convert existing hierarchical goal IDs to the new flat ID system
```bash
goal convert
```
This command converts any remaining goal files that might be using the legacy hierarchical ID system (like G1-S1-S1) to the new flat ID system (G1, S1, T1). This is primarily for backward compatibility and should only be needed if upgrading from an older version of the tool.

#### Revert a goal's current state back to its initial state
```bash
goal revert <goal-id>
```
This command will reset all fields in the goal's current state to match its initial state. If the goal has any child goals or tasks, you will be prompted for confirmation before proceeding.

## Status Indicators

The goal management system uses the following status indicators:

- ✅ Complete: Goal is marked as complete
- 🟠 Ready to merge: All subgoals are complete but not yet merged
- ⚪ Incomplete: Some subgoals are still incomplete
- 🔘 No subgoals: Goal has no subgoals

## Examples

### Create a goal hierarchy

```bash
# Create a top-level goal
goal new "Implement authentication system"
# G1 is created

# Create subgoals
goal sub G1 "Implement user registration"
# S1 is created
goal sub G1 "Implement login/logout"
# S2 is created
goal sub G1 "Implement password reset"
# S3 is created

# Create a task under a subgoal
goal sub S1 "Design registration form"
# S4 is created (or goal task S1 "Design registration form" would create T1)
```

### Navigate the goal hierarchy

```bash
# Go to a specific subgoal
goal down S1

# Go back to parent
goal up

# Go to root goal from anywhere in the hierarchy
goal root

# List available subgoals
goal subs
```

### Track progress

```bash
# Mark current goal as complete
goal complete

# Show status of all goals
goal status

# Merge a completed subgoal into its parent
goal merge S1
```

### Visualize the goal hierarchy

```bash
# Show a text-based tree view
goal tree

# Show goal history
goal history

# Generate a graphical visualization (requires Graphviz)
goal graph
```

## Advanced Usage


### Conflict Resolution

When merging subgoals, if conflicts occur:

1. The system will automatically abort the merge
2. You'll need to manually resolve conflicts:
   ```bash
   git merge <subgoal-branch>
   # Resolve conflicts
   git add <resolved-files>
   git commit
   ```

## Tips

1. Always mark a goal as complete (`goal complete`) before merging it into its parent

3. Use `goal tree` to quickly visualize the current state of your goals
4. Use `goal graph` to generate diagrams for documentation or presentations

## Integration with Git Workflow

The goal management system is designed to work seamlessly with Git:

1. Each goal and subgoal is associated with its own branch
2. Changes for a specific goal should be made on that goal's branch
3. When a goal is complete, merge it into its parent goal's branch
4. The system preserves the full history of changes through the branch hierarchy

### Typical Workflow

1. Create a top-level goal: `goal new "Feature X"`
2. Create subgoals: `goal sub G1 "Component A"`
3. Navigate to a subgoal branch: `goal down S1`
4. Make changes on the subgoal branch
5. Mark subgoal as complete: `goal complete`
6. Navigate to parent: `goal up`
7. Merge the subgoal: `goal merge S1`
8. Repeat until the top-level goal is complete 