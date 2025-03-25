#!/bin/bash

echo "Running critical tests before commit..."
# Define color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Run memory tests first
echo -e "${YELLOW}Running memory tests...${NC}"
PYTHONWARNINGS=ignore python -m pytest tests/test_memory_*.py -v

# Check if memory tests passed
if [ $? -ne 0 ]; then
    echo -e "${RED}Memory tests failed! Please fix the tests before committing.${NC}"
    exit 1
fi

# Run filesystem tools tests
echo -e "${YELLOW}Running filesystem tools tests...${NC}"
PYTHONWARNINGS=ignore python -m pytest tests/test_filesystem_tools.py -v

# Check if filesystem tools tests passed
if [ $? -ne 0 ]; then
    echo -e "${RED}Filesystem tools tests failed! Please fix the tests before committing.${NC}"
    exit 1
fi

# Run specific bugs tests
echo -e "${YELLOW}Running specific bugs tests...${NC}"
PYTHONWARNINGS=ignore python -m pytest tests/test_specific_bugs.py -v

# Check if specific bugs tests passed
if [ $? -ne 0 ]; then
    echo -e "${RED}Specific bugs tests failed! Please fix the tests before committing.${NC}"
    exit 1
fi

# Run goal decomposer tools tests
echo -e "${YELLOW}Running goal decomposer tools tests...${NC}"
PYTHONWARNINGS=ignore python -m pytest tests/test_goal_decomposer_tools.py -v

# Check if goal decomposer tools tests passed
if [ $? -ne 0 ]; then
    echo -e "${RED}Goal decomposer tools tests failed! Please fix the tests before committing.${NC}"
    exit 1
fi

# Run tools wrapper tests
echo -e "${YELLOW}Running tools wrapper tests...${NC}"
PYTHONWARNINGS=ignore python -m pytest tests/test_tools_wrapper.py -v

# Check if tools wrapper tests passed
if [ $? -ne 0 ]; then
    echo -e "${RED}Tools wrapper tests failed! Please fix the tests before committing.${NC}"
    exit 1
fi

echo -e "${GREEN}All tests passed! Proceeding with commit.${NC}"
exit 0 