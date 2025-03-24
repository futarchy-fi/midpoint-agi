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
from src.midpoint.agents.goal_decomposer import main
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

# Create mock response content
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

# Set up the mock client
mock_client = AsyncMock()
mock_client.chat = AsyncMock()
mock_client.chat.completions = AsyncMock()
mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

# Mock the OpenAI client and API key
with patch('src.midpoint.agents.goal_decomposer.AsyncOpenAI', return_value=mock_client):
    with patch('src.midpoint.agents.config.get_openai_api_key', return_value="sk-" + "a" * 48):
        # Mock the ToolProcessor's run_llm_with_tools method
        with patch('src.midpoint.agents.goal_decomposer.ToolProcessor') as mock_tool_processor:
            mock_processor_instance = AsyncMock()
            mock_tool_processor.return_value = mock_processor_instance
            mock_processor_instance.run_llm_with_tools = AsyncMock(return_value=(mock_message, []))
            
            # Run the main function
            if __name__ == '__main__':
                # Save original arguments
                original_args = sys.argv.copy()
                
                # Replace the script path with the goal_decomposer.py path
                sys.argv[0] = str(repo_root / "src" / "midpoint" / "agents" / "goal_decomposer.py")
                
                # Create logs directory if it doesn't exist
                repo_path = Path(sys.argv[1])
                logs_dir = repo_path / "logs"
                logs_dir.mkdir(exist_ok=True)
                
                # Run the main function
                import asyncio
                try:
                    asyncio.run(main())
                    print(f"✅ Next task: {response_content['next_step']}")
                except Exception as e:
                    print(f"Error: {str(e)}", file=sys.stderr)
                    sys.exit(1)
                
                # Restore original arguments
                sys.argv = original_args 