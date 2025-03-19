# Midpoint

A tool for decomposing complex goals into manageable steps.

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd midpoint
```

2. Set up the development environment:
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

3. Verify your environment:
```bash
python check_env.py
```

4. Run tests:
```bash
python -m pytest tests/
```

## Development Setup

The project requires a Python virtual environment and certain environment variables to be set up. The setup script (`setup.sh`) handles most of this automatically, but here's what it does:

1. Creates a virtual environment (`.venv`)
2. Activates the virtual environment
3. Installs the package in development mode
4. Creates an environment check script

### Environment Checks

The project includes strict environment checks to ensure proper setup. These checks verify:
- Virtual environment is active
- Package is installed correctly
- Required environment variables are set
- Project structure is correct
- Git repository is initialized

Run the environment check:
```bash
python check_env.py
```

#### Force Running (Not Recommended)

If you need to run tests or scripts outside the intended environment, you can use force flags:

```bash
# Force run despite environment issues
python check_env.py --force

# Skip specific checks
python check_env.py --skip venv,package,api_key

# Run demo with validation skipped
python simple_test.py --no-validation
```

⚠️ **Warning**: Using force flags or skipping checks is not recommended and may lead to unexpected behavior.

### Manual Setup

If you prefer to set up manually:

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install the package in development mode:
```bash
python setup_dev.py
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Running Tests

Before running tests, ensure your environment is set up correctly:

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Run the environment check:
```bash
python check_env.py
```

3. Run the tests:
```bash
python -m pytest tests/
```

## Troubleshooting

If you encounter import errors or other issues:

1. Make sure you're in the virtual environment:
```bash
source .venv/bin/activate
```

2. Verify the package is installed:
```bash
python check_env.py
```

3. If issues persist, try reinstalling:
```bash
python setup_dev.py
```

## Features

- Goal decomposition and planning
- Git-based state management
- Task execution with validation
- Human-in-the-loop supervision
- Detailed operation logging

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

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 