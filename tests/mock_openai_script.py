#!/usr/bin/env python
"""Mock script for OpenAI client in subprocess tests."""

import json
import sys
import os
import logging
from unittest.mock import AsyncMock, patch
from pathlib import Path

# Add the parent directory to the Python path
repo_root = Path(__file__).parent.parent
sys.path.append(str(repo_root))

# Import the goal_decomposer module
from src.midpoint.agents.goal_decomposer import decompose_goal, validate_repository_state
from src.midpoint.agents.config import get_openai_api_key

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Create mock response classes
class MockFunction:
    def __init__(self, name, arguments=None):
        self.name = name
        self.arguments = arguments or "{}"

class MockMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls
        self.role = "assistant"
        self.function_call = None

class MockChoice:
    def __init__(self, message):
        self.message = message
        self.finish_reason = "stop"
        self.index = 0

class MockResponse:
    def __init__(self, choices):
        self.choices = choices
        self.model = "gpt-4"
        self.object = "chat.completion"

# Set up the mock client
mock_client = AsyncMock()
mock_client.chat = AsyncMock()
mock_client.chat.completions = AsyncMock()

# Mock the OpenAI client and API key
with patch('src.midpoint.agents.goal_decomposer.AsyncOpenAI', return_value=mock_client):
    with patch('src.midpoint.agents.config.get_openai_api_key', return_value="sk-" + "a" * 48):
        # Mock validate_repository_state to always succeed
        with patch('src.midpoint.agents.goal_decomposer.validate_repository_state', return_value=None):
            # Mock the ToolProcessor's run_llm_with_tools method
            with patch('src.midpoint.agents.goal_decomposer.ToolProcessor') as mock_tool_processor:
                mock_processor_instance = AsyncMock()
                mock_tool_processor.return_value = mock_processor_instance
                
                # Run the goal decomposer
                if __name__ == '__main__':
                    # Save original arguments
                    original_args = sys.argv.copy()
                    
                    # Replace the script path with the goal_decomposer.py path
                    sys.argv[0] = str(repo_root / "src" / "midpoint" / "agents" / "goal_decomposer.py")
                    
                    # Parse arguments
                    repo_path = sys.argv[1]
                    goal = sys.argv[2]
                    input_file = None
                    memory_repo = None
                    debug = False
                    
                    # Parse optional arguments
                    for i in range(3, len(sys.argv)):
                        if sys.argv[i] == "--input-file" and i + 1 < len(sys.argv):
                            input_file = sys.argv[i + 1]
                        elif sys.argv[i] == "--memory-repo" and i + 1 < len(sys.argv):
                            memory_repo = sys.argv[i + 1]
                        elif sys.argv[i] == "--debug":
                            debug = True
                    
                    # Create response content based on the goal
                    if goal == "List subgoals":
                        response_content = {
                            "next_step": "Mock list subgoals result",
                            "validation_criteria": ["Test passes"],
                            "reasoning": "This is a mocked OpenAI response for testing",
                            "requires_further_decomposition": False,
                            "relevant_context": {},
                            "parent_goal": None,
                            "goal_id": "G1",
                            "timestamp": "20240324_000001"
                        }
                    else:
                        # Default response for other goals
                        response_content = {
                            "next_step": "Mock integration test result",
                            "validation_criteria": ["Integration test passes"],
                            "reasoning": "This is a mocked OpenAI response for testing",
                            "requires_further_decomposition": False,
                            "relevant_context": {},
                            "parent_goal": None,
                            "goal_id": "G1",
                            "timestamp": "20240324_000001"
                        }
                    
                    # Create mock message and response
                    mock_message = MockMessage(content=json.dumps(response_content))
                    mock_choice = MockChoice(mock_message)
                    mock_response = MockResponse([mock_choice])
                    
                    # Set up the mock client's response
                    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
                    mock_processor_instance.run_llm_with_tools = AsyncMock(return_value=(mock_message, []))
                    
                    # Create logs directory if it doesn't exist
                    logs_dir = Path(repo_path) / "logs"
                    logs_dir.mkdir(exist_ok=True)
                    
                    # Run the goal decomposer
                    import asyncio
                    try:
                        result = asyncio.run(decompose_goal(
                            repo_path=repo_path,
                            goal=goal,
                            input_file=input_file,
                            memory_repo=memory_repo,
                            debug=debug
                        ))
                        
                        # Update the goal file with our mocked response content
                        if 'goal_file' in result and result['goal_file']:
                            goal_file_path = Path(repo_path) / "logs" / result['goal_file']
                            if goal_file_path.exists():
                                try:
                                    with open(goal_file_path, 'r') as f:
                                        goal_file_content = json.load(f)
                                    
                                    # Update the next_step to match our mocked response
                                    goal_file_content['next_step'] = response_content['next_step']
                                    
                                    with open(goal_file_path, 'w') as f:
                                        json.dump(goal_file_content, f, indent=2)
                                        
                                    # Also update the result to be returned
                                    result['next_step'] = response_content['next_step']
                                except Exception as e:
                                    print(f"Error updating goal file: {str(e)}", file=sys.stderr)
                        
                        print(json.dumps(result, indent=2))
                    except Exception as e:
                        print(f"Error: {str(e)}", file=sys.stderr)
                        sys.exit(1)
                    
                    # Restore original arguments
                    sys.argv = original_args 