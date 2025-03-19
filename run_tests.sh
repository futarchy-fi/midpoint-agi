#!/bin/bash

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Run environment check
python check_env.py --force

# Run tests
python -m pytest tests/ -v 