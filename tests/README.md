# Goal Decomposer Test Suite

This directory contains tests for the GoalDecomposer agent and related functionality.

## Testing Strategy

The test suite has been designed to cover different levels of testing, including:

1. **Unit Tests**: Testing individual components in isolation
2. **Integration Tests**: Testing how components work together
3. **CLI Tests**: Testing the command-line interface
4. **Asyncio Pattern Tests**: Testing proper asyncio usage across the codebase
5. **Specific Bug Tests**: Tests that verify fixes for specific bugs

### Key Test Files

- `test_goal_decomposer_tools.py`: Tests the tools used by the GoalDecomposer
- `test_goal_decomposer_imports.py`: Tests the import behavior of the GoalDecomposer module
- `test_goal_decomposer_cli.py`: Tests the CLI execution path
- `test_asyncio_patterns.py`: Checks for proper asyncio pattern usage
- `test_specific_bugs.py`: Tests for specific bugs, including the asyncio nesting issue
- `test_goal_decomposer_integration.py`: Integration tests for the GoalDecomposer
- `test_integration_fixtures.py`: Fixtures for integration tests

### Asyncio Testing Focus

A key aspect of the testing strategy is ensuring proper asyncio usage to prevent issues like:

- Nested `asyncio.run()` calls
- Non-awaited coroutines
- Improper event loop management

### Running the Tests

Run all tests:
```bash
python -m unittest discover tests
```

Run a specific test:
```bash
python -m unittest tests/test_specific_bugs.py
```

## Test Fixtures

The test suite includes fixtures for creating:

- Test git repositories
- Sample subgoal files
- Memory repositories

These fixtures make it easier to write comprehensive tests without duplicating setup code. 