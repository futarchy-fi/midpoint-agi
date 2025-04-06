#!/usr/bin/env python
"""
Test script for isolating and testing the JSON extraction functionality 
from the GoalValidator class.

This script tests the _extract_validation_json method with various mock LLM responses
to debug parsing issues.
"""

import json
import logging
import sys
import os
from pathlib import Path
import re

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import the GoalValidator class
try:
    from midpoint.agents.goal_validator import GoalValidator
    print("Successfully imported GoalValidator")
except ImportError as e:
    print(f"Failed to import GoalValidator: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()

# Sample LLM responses for testing
MOCK_RESPONSES = [
    # Response 1: Plain JSON
    """{
  "criteria_results": [
    {
      "criterion": "A list of all files in the repository is obtained.",
      "passed": true,
      "reasoning": "The list of files in the repository was successfully retrieved.",
      "evidence": ["Directory listing showing README.md, Makefile, and Python scripts"]
    },
    {
      "criterion": "Potential files containing hardcoded LLM model configurations are identified.",
      "passed": true,
      "reasoning": "Files with potential hardcoded model configurations were identified.",
      "evidence": ["Found relevant files in the config directory"]
    }
  ]
}""",

    # Response 2: JSON inside markdown code block
    """```json
{
  "criteria_results": [
    {
      "criterion": "A list of all files in the repository is obtained.",
      "passed": true,
      "reasoning": "The list of files in the repository was successfully retrieved.",
      "evidence": ["Directory listing showing README.md, Makefile, and Python scripts"]
    },
    {
      "criterion": "Potential files containing hardcoded LLM model configurations are identified.",
      "passed": true,
      "reasoning": "Files with potential hardcoded model configurations were identified.",
      "evidence": ["Found relevant files in the config directory"]
    }
  ]
}
```""",

    # Response 3: Messy response with JSON embedded
    """I've evaluated the goal criteria and here's my assessment:

```json
{
  "criteria_results": [
    {
      "criterion": "A list of all files in the repository is obtained.",
      "passed": true,
      "reasoning": "The list of files in the repository was successfully retrieved.",
      "evidence": ["Directory listing showing README.md, Makefile, and Python scripts"]
    },
    {
      "criterion": "Potential files containing hardcoded LLM model configurations are identified.",
      "passed": true,
      "reasoning": "Files with potential hardcoded model configurations were identified.",
      "evidence": ["Found relevant files in the config directory"]
    }
  ]
}
```

I hope this helps!""",

    # Response 4: Raw LLM response (from debug output)
    """```json
{
  "criteria_results": [
    {
      "criterion": "A list of all files in the repository is obtained.",
      "passed": true,
      "reasoning": "The list of files in the repository was successfully retrieved, showing a comprehensive directory structure with various files and directories.",
      "evidence": [
        "The directory listing includes files such as 'README.md', 'Makefile', and multiple Python scripts in the 'agents' directory."
      ]
    },
    {
      "criterion": "Pot...
```""",

    # Response 5: No code block, just direct JSON
    """{
  "criteria_results": [
    {
      "criterion": "A list of all files in the repository is obtained.",
      "passed": true,
      "reasoning": "The list of files in the repository was successfully retrieved.",
      "evidence": ["Directory listing showing README.md, Makefile, and Python scripts"]
    }
  ]
}""",

    # Response 6: Debug log format (the exact format we saw)
    """root: DEBUG: Raw LLM response content: ```json
{
  "criteria_results": [
    {
      "criterion": "A list of all files in the repository is obtained.",
      "passed": true,
      "reasoning": "The list of files in the repository was successfully retrieved, showing a comprehensive directory structure with various files and directories.",
      "evidence": [
        "The directory listing includes files such as 'README.md', 'Makefile', and multiple Python scripts in the 'agents' directory."
      ]
    },
    {
      "criterion": "Potential files containing hardcoded LLM model configurations are identified based on file names and extensions.",
      "passed": true,
      "reasoning": "Files that might contain hardcoded LLM model configurations were identified by examining file names and extensions.",
      "evidence": [
        "Files like 'config.py', 'llm_config.py', and others with configuration-related names were identified as potential locations for hardcoded LLM model settings."
      ]
    }
  ],
  "score": 1.0,
  "reasoning": "The goal was successfully achieved. All files in the repository were listed, and potential files containing hardcoded LLM model configurations were identified based on their names and extensions. The evidence shows that the system correctly identified configuration files that might contain LLM model settings."
}
```"""
]

def extract_validation_json_simple(content):
    """A simplified version of the extraction function to debug the issues"""
    logger.debug(f"Content length: {len(content)}")
    logger.debug(f"Content starts with: {content[:100]}")
    
    # Check if the content starts with log prefix and strip it
    if content.startswith("root: DEBUG: Raw LLM response content:"):
        content = content.replace("root: DEBUG: Raw LLM response content:", "", 1).strip()
        logger.debug(f"After stripping log prefix, content starts with: {content[:100]}")
    
    # Extract JSON from markdown code blocks
    json_block_match = re.search(r"```json\s*([\s\S]*?)\s*```", content)
    if json_block_match:
        json_str = json_block_match.group(1).strip()
        logger.debug(f"Extracted JSON from markdown: {json_str[:100]}...")
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON from markdown: {e}")
    
    # Try direct JSON parsing
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        logger.debug("Direct JSON parsing failed")
    
    # Look for raw JSON starting with criteria_results
    if '"criteria_results"' in content:
        start_idx = content.find('"criteria_results"')
        # Find the opening brace before criteria_results
        opening_brace_idx = content.rfind('{', 0, start_idx)
        if opening_brace_idx >= 0:
            # Count braces to find the matching closing brace
            brace_count = 1
            for i in range(opening_brace_idx + 1, len(content)):
                if content[i] == '{':
                    brace_count += 1
                elif content[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        json_str = content[opening_brace_idx:i+1]
                        logger.debug(f"Extracted raw JSON: {json_str[:100]}...")
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.error(f"Error parsing raw JSON: {e}")
                        break
    
    logger.error("Could not extract valid JSON from content")
    return None

def main():
    """Test the _extract_validation_json method with various mock responses."""
    # Create a GoalValidator instance
    validator = GoalValidator()
    # Add a logger attribute to the validator instance
    validator.logger = logger

    # Set up a file for saving the test results
    result_file = Path("json_extraction_test_results.txt")
    with open(result_file, "w") as f:
        f.write("JSON Extraction Test Results\n")
        f.write("==========================\n\n")

        # Test each mock response
        for i, response in enumerate(MOCK_RESPONSES):
            f.write(f"Test {i+1}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Mock Response:\n{response[:200]}...\n\n")
            
            try:
                # Extract JSON from the response
                extracted_json = validator._extract_validation_json(response)
                
                # Pretty print the extracted JSON
                formatted_json = json.dumps(extracted_json, indent=2)
                f.write(f"Extracted JSON:\n{formatted_json}\n\n")
                
                # Check if criteria_results is present
                if "criteria_results" in extracted_json:
                    criteria_results = extracted_json["criteria_results"]
                    f.write(f"Found {len(criteria_results)} criteria results\n")
                    for j, cr in enumerate(criteria_results):
                        f.write(f"  Criterion {j+1}: {cr.get('criterion', 'N/A')}\n")
                        f.write(f"    Passed: {cr.get('passed', 'N/A')}\n")
                else:
                    f.write("No criteria_results found in extracted JSON\n")
            except Exception as e:
                f.write(f"Error extracting JSON: {e}\n")
            
            f.write("\n\n")

    print(f"Test results saved to {result_file}")
    
    # Also print results to console
    with open(result_file, "r") as f:
        print(f.read())

if __name__ == "__main__":
    main() 