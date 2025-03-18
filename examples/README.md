# Midpoint Test Examples

This directory contains scripts for testing the Midpoint AGI system's git operations.

## Setup

1. First, create a test repository:
```bash
python setup_test_repo.py
```
This will create a test repository at `~/midpoint-test-repo` with some initial content and branches.

2. Install the Midpoint package in development mode:
```bash
pip install -e ..
```

## Testing Git Operations

Use the test client to try out different git operations:

```bash
# Check repository state
python test_repo_operations.py ~/midpoint-test-repo --state

# Create a new branch
python test_repo_operations.py ~/midpoint-test-repo --branch feature/new-feature

# Create a commit
python test_repo_operations.py ~/midpoint-test-repo --commit "Add new changes"

# Checkout a branch
python test_repo_operations.py ~/midpoint-test-repo --checkout feature/test-feature

# Revert to a specific hash
python test_repo_operations.py ~/midpoint-test-repo --hash <commit-hash>
```

## Test Repository Structure

The test repository will have the following structure:
```
├── README.md
├── src/
│   ├── main.py
│   └── feature.py (in feature/test-feature branch)
└── tests/
    └── test_main.py
```

And the following branches:
- `main` - Main branch with initial content
- `feature/test-feature` - Feature branch with additional content

## Safety Features

The test client includes several safety features:
- All operations are scoped to the specified repository
- No remote operations are allowed by default
- State is tracked and can be restored if operations fail
- All operations are logged for audit purposes 