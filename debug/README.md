# Debug Tools

This directory contains debugging and testing tools for the Midpoint system. These tools are used for development and testing individual components of the system.

## Contents

### Runner Scripts
- `run_goal_decomposer.py`: Tests the goal decomposition component in isolation
- `run_midpoint_goal_decomposer.py`: Alternative implementation of goal decomposition
- `run_recursive_decomposition.py`: Tests recursive goal decomposition
- `run_orchestrator.py`: Tests the orchestrator component (currently under development)

### Example Implementations
- `example_goal_decomposer.py`: Basic example of goal decomposition
- `example_goal_decomposer_custom.py`: Custom implementation example
- `custom_goal_decomposer.py`: Another custom implementation

### Testing
- `real_api_test.py`: Tests the API integration

## Task Executor CLI

The `run_task_executor.py` script provides a command-line interface for the TaskExecutor agent, allowing you to execute tasks directly.

### Usage

```bash
python debug/run_task_executor.py --repo-path <repository_path> [options]
```

#### Required Arguments

- `--repo-path`: Path to the repository to work with

#### Task Specification (one of these is required)

- `--task`: Task description to execute
- `--input-file`: Path to subgoal JSON file containing the task

#### Optional Arguments

- `--goal`: Overall goal context for the task
- `--output-dir`: Directory to save output files (default: 'output')
- `--debug`: Enable debug mode with detailed logging
- `--no-commit`: Prevent automatic commits

### Examples

Execute a task directly:
```bash
python debug/run_task_executor.py --repo-path /path/to/repo --task "Create a new file called hello.py"
```

Execute a task from a subgoal file:
```bash
python debug/run_task_executor.py --repo-path /path/to/repo --input-file logs/subgoal_1234567890.json
```

### Integration with Goal Decomposer

You can use the TaskExecutor CLI to execute tasks produced by the Goal Decomposer:

1. Run the goal decomposer to generate a task:
   ```bash
   python debug/run_goal_decomposer.py --repo /path/to/repo --objective "Implement feature X"
   ```

2. Use the generated subgoal file as input to the task executor:
   ```bash
   python debug/run_task_executor.py --repo-path /path/to/repo --input-file logs/subgoal_1234567890.json
   ```

## Usage

These tools are primarily for development and debugging purposes. The main CLI interface (`src/midpoint/cli.py`) is currently under development and not yet functional.

To use these debug tools:

1. Ensure you have the required environment variables set up (see main README)
2. Run the desired debug script with appropriate arguments
3. Check the output and logs for debugging information

## Development Status

The main orchestrator and CLI interface are currently under development. These debug tools can be used to test and develop individual components while the main system is being built.

## Contributing

When adding new debug tools:
1. Place them in this directory
2. Update this README with a description
3. Include usage instructions in the script's docstring 