# Midpoint AGI

An advanced AI orchestration system for solving complex problems through iterative planning, execution, and validation.

## Features

- Goal decomposition and planning
- Git-based state management
- Task execution with validation
- Human-in-the-loop supervision
- Detailed operation logging

## Development Setup

### Quick Setup

For a guided setup experience, run:

```bash
python setup_dev.py
```

This script will:
1. Create a virtual environment
2. Install development dependencies
3. Offer to set up a test repository
4. Offer to configure your OpenAI API key

### Manual Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/midpoint.git
cd midpoint
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate
```

3. Install development dependencies:
```bash
pip install -e ".[dev]"
```

4. Set up your environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

5. Create a test repository:
```bash
# Create in a custom location (recommended)
python examples/setup_test_repo.py --path ~/my-test-repo

# Or use the default location (~/midpoint-test-repo)
python examples/setup_test_repo.py
```

6. Verify your setup:
```bash
python examples/verify_setup.py
```

## Testing the Goal Decomposer

To test the GoalDecomposer agent:

```bash
# If using custom repository location
MIDPOINT_TEST_REPO=~/my-test-repo python examples/test_goal_decomposer.py

# Or with default location
python examples/test_goal_decomposer.py
```

## Testing

1. Run the test suite:
```bash
pytest
```

2. Run specific test files:
```bash
pytest tests/test_goal_decomposer.py
```

3. Run with coverage:
```bash
pytest --cov=src/midpoint
```

## Development Workflow

1. Create a new branch for your feature:
```bash
git checkout -b feature/your-feature
```

2. Make your changes and run tests:
```bash
pytest
```

3. Commit your changes:
```bash
git add .
git commit -m "Your commit message"
```

4. Push to your fork and create a pull request

## Test Repository Management

The system uses git repositories for testing. You can:

1. Create a new test repository:
```bash
python examples/setup_test_repo.py --path ~/my-test-repo
```

2. Test git operations:
```bash
python examples/test_repo_operations.py ~/my-test-repo --state
python examples/test_repo_operations.py ~/my-test-repo --branch feature/new-feature
```

3. Clean up test repositories:
```bash
# Remove the test repository
rm -rf ~/my-test-repo
```

## Project Structure

```
midpoint/
├── src/
│   └── midpoint/         # Main package
│       ├── __init__.py
│       └── agents/       # Agent implementations
│           ├── __init__.py
│           ├── goal_decomposer.py  # Goal decomposition agent
│           ├── models.py          # Shared data models
│           ├── tools.py           # Git and utility functions
│           └── config.py          # Configuration management
├── examples/              # Example scripts and tests
│   ├── setup_test_repo.py # Test repository setup
│   └── verify_setup.py    # Setup verification
├── tests/                 # Test suite
│   ├── test_goal_decomposer.py
│   └── test_repo_context.py
└── docs/                  # Documentation
    ├── FEATURES.md
    └── VISION.md
```

## Troubleshooting

If you encounter issues, run the verification script to diagnose problems:

```bash
python examples/verify_setup.py
```

Common issues:
1. **Import errors**: Make sure you've installed the package with `pip install -e ".[dev]"`
2. **OpenAI API key**: Ensure your API key is set in your `.env` file and loaded in the environment
3. **Conflicting packages**: Check for any conflicting `agents` packages in your Python environment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 