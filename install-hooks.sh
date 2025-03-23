#!/bin/bash

echo "Installing git pre-commit hook..."

# Copy the pre-commit hook to the .git/hooks directory
cp pre-commit-hook.sh .git/hooks/pre-commit

# Make it executable
chmod +x .git/hooks/pre-commit

echo "Pre-commit hook installed successfully!"
echo "The hook will run memory tests before each commit." 