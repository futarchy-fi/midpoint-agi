#!/bin/bash

echo "Running memory tests before commit..."
# Define color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Run pytest on our memory tests
python -m pytest tests/test_memory_*.py -v

# Check if tests passed
if [ $? -eq 0 ]; then
    echo -e "${GREEN}All tests passed! Proceeding with commit.${NC}"
    exit 0
else
    echo -e "${RED}Tests failed! Please fix the tests before committing.${NC}"
    exit 1
fi 