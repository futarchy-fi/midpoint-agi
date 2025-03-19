#!/bin/bash

# Script to properly run tests with the correct virtual environment

# Exit immediately if a command exits with a non-zero status
set -e

# Print commands before execution
set -x

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Go to project root directory
cd "$SCRIPT_DIR"

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install the package in development mode if not already installed
if ! python -c "import midpoint" &> /dev/null; then
    echo "Installing package in development mode..."
    pip install -e ".[dev]"
fi

# Make sure we have the required test repository
if [ ! -d "test-repo" ]; then
    echo "Creating test repository..."
    mkdir -p test-repo
    cd test-repo
    git init
    echo 'def hello_world():
    return "Hello, World!"' > hello.py
    git add hello.py
    git config --local user.email "test@example.com"
    git config --local user.name "Test User"
    git commit -m "Initial commit"
    cd ..
fi

# Run our custom example script that uses the actual git hash
echo "Running example script with correct environment..."
python example_goal_decomposer_custom_2.py 