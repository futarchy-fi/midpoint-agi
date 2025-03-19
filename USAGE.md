# Midpoint Usage Guide

This document provides quick reference commands for using the Midpoint system.

## Setup

```bash
# Clone the repository
git clone <repository-url>
cd midpoint

# Set up environment
./setup.sh
# OR manual setup:
python -m venv .venv
source .venv/bin/activate
python setup_dev.py

# Create .env file with API keys
cp .env.example .env
# Edit .env with your OPENAI_API_KEY
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run specific test
python -m pytest tests/test_goal_decomposer.py
```

## Using the System

### Full Flow (Decomposition + Execution + Validation)

```bash
python examples/test_deep_flow.py test-repo "Create a task management system with basic functionality for adding, listing, and completing tasks"
```

### Recursive Decomposition Only

```bash
python run_recursive_decomposition.py test-repo "Create a task management system with basic functionality for adding, listing, and completing tasks"
```

### Creating Test Repository

```bash
# Create a new test repository
mkdir test-repo
cd test-repo
git init
echo "# Test Repository" > README.md
git add README.md
git commit -m "Initial commit"
cd ..
```

## Common Examples

### Simple Task Example

```bash
# Create a simple hello world file
python examples/test_deep_flow.py test-repo "Create a simple hello world file"
```

### Complex Task Example

```bash
# Create a more complex application
python examples/test_deep_flow.py test-repo "Create a task management system with the ability to add, list, and complete tasks"
```

### Very Complex Task Example

```bash
# Create a sophisticated system
python examples/test_deep_flow.py test-repo "Create a web API server with three endpoints: one to add items to a database, one to list all items, and one to search for specific items"
```

## Troubleshooting

- If you encounter environment issues, check that your virtual environment is activated
- Ensure your API key is set in the `.env` file
- Make sure the repository is in a clean state before running tests
- Check for potential merge conflicts or uncommitted changes 