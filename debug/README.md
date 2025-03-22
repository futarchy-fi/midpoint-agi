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