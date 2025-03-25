# Goal Decomposition Command

The `goal decompose` command allows you to decompose an existing goal into subgoals using the GoalDecomposer.

## Usage

```bash
goal decompose <goal-id> [--debug] [--quiet]
```

### Arguments

- `goal-id`: The ID of the goal to decompose (e.g., G1)

### Options

- `--debug`: Show detailed debug output
- `--quiet`: Only show warnings and the final result

## Description

The `goal decompose` command uses the GoalDecomposer to break down a complex goal into more manageable subgoals. This helps in creating a hierarchical structure of tasks that can be addressed individually.

When you run this command:

1. It verifies that the goal with the provided ID exists
2. It extracts the goal description from the goal file
3. It calls the GoalDecomposer to break down the goal
4. It displays the next step, validation criteria, and other relevant information
5. It creates a new subgoal file with the generated information

## Example

```bash
$ goal decompose G1
```

Example output:

```
Goal G1 successfully decomposed into subgoals

Next step: Implement user authentication system

Validation criteria:
- User can register with email and password
- User can log in with correct credentials
- Invalid login attempts are rejected
- Passwords are securely hashed
- User sessions are maintained with tokens

Requires further decomposition: Yes

Goal file: .goal/G1-S1.json
```

## Integration with Other Commands

After decomposing a goal, you can:

1. Use `goal list` to see the hierarchy of goals and subgoals
2. Use `goal down <subgoal-id>` to navigate to a specific subgoal branch
3. Decompose a subgoal further with `goal decompose <subgoal-id>`
4. Mark a goal as complete with `goal complete` when finished

## Notes

- The GoalDecomposer uses LLM technology to intelligently break down goals
- It analyzes the repository content to suggest context-appropriate subgoals
- The decomposition is not recursive by default (each subgoal needs to be manually decomposed if needed)
- For complex projects, you might need to decompose goals multiple levels deep 