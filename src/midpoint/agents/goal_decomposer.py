"""
Goal Decomposition agent implementation.

This module implements the GoalDecomposer agent that determines the next step
toward achieving a complex goal.
"""

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext
from midpoint.agents.tools import (
    list_directory,
    read_file,
    search_code,
    track_points
)
from midpoint.agents.config import get_openai_api_key

class GoalDecomposer:
    """Agent responsible for determining the next step toward a complex goal."""
    
    def __init__(self):
        """Initialize the GoalDecomposer agent."""
        # Initialize OpenAI client with API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Define the system prompt focusing on determining the next step
        self.system_prompt = """You are an expert software architect and project planner.
Your task is to determine the SINGLE NEXT STEP toward a complex software development goal.
Follow the OODA loop: Observe, Orient, Decide, Act.

1. OBSERVE: Explore the repository to understand its current state. Use the available tools.
2. ORIENT: Analyze how the current state relates to the final goal.
3. DECIDE: Determine the most promising SINGLE NEXT STEP.
4. OUTPUT: Provide a structured response with the next step and validation criteria.

For complex goals, consider if the best next step is exploration, research, or a "study session" 
rather than immediately jumping to implementation.

You have access to these tools:
- list_directory: List files and directories in the repository
- read_file: Read the contents of a file
- search_code: Search the codebase for patterns

You MUST provide a structured response in JSON format with these fields:
- next_step: A clear description of the single next step to take
- validation_criteria: List of measurable criteria to validate this step's completion
- reasoning: Explanation of why this is the most promising next action"""
        
        # Define tool schema for the OpenAI API
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "List the contents of a directory in the repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "directory": {
                                "type": "string",
                                "description": "Directory to list within the repository",
                                "default": "."
                            }
                        },
                        "required": ["repo_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "Read the contents of a file in the repository",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file within the repository"
                            },
                            "start_line": {
                                "type": "integer",
                                "description": "First line to read (0-indexed)",
                                "default": 0
                            },
                            "max_lines": {
                                "type": "integer",
                                "description": "Maximum number of lines to read",
                                "default": 100
                            }
                        },
                        "required": ["repo_path", "file_path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "search_code",
                    "description": "Search the codebase for a pattern",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "pattern": {
                                "type": "string",
                                "description": "Regular expression pattern to search for"
                            },
                            "file_pattern": {
                                "type": "string",
                                "description": "Pattern for files to include (e.g., '*.py')",
                                "default": "*"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 20
                            }
                        },
                        "required": ["repo_path", "pattern"]
                    }
                }
            }
        ]

    async def determine_next_step(self, context: TaskContext) -> SubgoalPlan:
        """
        Determine the next step toward achieving the goal.
        
        Args:
            context: The current task context containing the goal and state
            
        Returns:
            A SubgoalPlan containing the next step and validation criteria
            
        Raises:
            ValueError: If the goal or context is invalid
            Exception: For other errors during execution
        """
        # Validate inputs
        if not context.goal:
            raise ValueError("No goal provided in context")
        if not context.state.repository_path:
            raise ValueError("Repository path not provided in state")
            
        # Prepare the user prompt
        user_prompt = self._create_user_prompt(context)
        
        # Initialize messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Track an API call for determining the next step
        await track_points("goal_decomposition", 10)
        
        # Track tool usage for metadata
        tool_usage = []
        
        # Chat completion with tool use
        try:
            final_output = None
            
            # Loop until we get a final output
            while final_output is None:
                # Call OpenAI API
                response = await self.client.chat.completions.create(
                    model="gpt-4o",
                    messages=messages,
                    tools=self.tools,
                    tool_choice="auto",
                    temperature=0.1,
                    max_tokens=4000
                )
                
                # Get the model's message
                message = response.choices[0].message
                
                # Add the message to our conversation
                messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})
                
                # Check if the model wants to use tools
                if message.tool_calls:
                    # Process each tool call
                    for tool_call in message.tool_calls:
                        # Get function details
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                        
                        # Track the tool usage
                        tool_usage.append(f"{function_name}: {json.dumps(function_args)}")
                        
                        # Execute the appropriate function
                        if function_name == "list_directory":
                            result = await list_directory(**function_args)
                            result_str = json.dumps(result, indent=2)
                        elif function_name == "read_file":
                            result_str = await read_file(**function_args)
                        elif function_name == "search_code":
                            result_str = await search_code(**function_args)
                        else:
                            result_str = f"Error: Unknown function {function_name}"
                            
                        # Add the function result to our messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result_str
                        })
                        
                        # Track points for tool use
                        await track_points("tool_use", 5)
                else:
                    # If no tool calls, check if we have a final output
                    if message.content:
                        try:
                            # Attempt to parse the response as JSON
                            content = message.content.strip()
                            
                            # Handle the case where the JSON is embedded in a code block
                            if "```json" in content:
                                parts = content.split("```json")
                                if len(parts) > 1:
                                    json_part = parts[1].split("```")[0].strip()
                                    output_data = json.loads(json_part)
                                else:
                                    output_data = json.loads(content)
                            elif "```" in content:
                                parts = content.split("```")
                                if len(parts) > 1:
                                    json_part = parts[1].strip()
                                    output_data = json.loads(json_part)
                                else:
                                    output_data = json.loads(content)
                            else:
                                output_data = json.loads(content)
                                
                            # Check if the output has the required fields
                            if all(key in output_data for key in ["next_step", "validation_criteria", "reasoning"]):
                                final_output = SubgoalPlan(
                                    next_step=output_data["next_step"],
                                    validation_criteria=output_data["validation_criteria"],
                                    reasoning=output_data["reasoning"],
                                    metadata={
                                        "raw_response": message.content,
                                        "tool_usage": tool_usage
                                    }
                                )
                            else:
                                # Ask for a properly formatted response
                                messages.append({
                                    "role": "user",
                                    "content": "Please provide a valid JSON response with the fields: next_step, validation_criteria, and reasoning."
                                })
                        except json.JSONDecodeError:
                            # If not valid JSON, ask for a properly formatted response
                            messages.append({
                                "role": "user",
                                "content": "Please provide your response in valid JSON format with the fields: next_step, validation_criteria, and reasoning."
                            })
            
            # Validate the subgoal plan
            self._validate_subgoal(final_output, context)
            
            return final_output
            
        except Exception as e:
            raise Exception(f"Error during next step determination: {str(e)}")
    
    def _create_user_prompt(self, context: TaskContext) -> str:
        """Create the user prompt for the agent."""
        return f"""Goal: {context.goal.description}

Validation Criteria for Final Goal:
{chr(10).join(f"- {criterion}" for criterion in context.goal.validation_criteria)}

Current State:
- Git Hash: {context.state.git_hash}
- Description: {context.state.description}
- Repository Path: {context.state.repository_path}

Context:
- Iteration: {context.iteration}
- Previous Steps: {len(context.execution_history) if context.execution_history else 0}

Your task is to explore the repository and determine the SINGLE NEXT STEP toward achieving the goal.
Focus on providing a clear next step with measurable validation criteria.
"""
    
    def _validate_subgoal(self, subgoal: SubgoalPlan, context: TaskContext) -> None:
        """Validate the generated subgoal plan."""
        if not subgoal.next_step:
            raise ValueError("Subgoal has no next step defined")
            
        if not subgoal.validation_criteria:
            raise ValueError("Subgoal has no validation criteria")
            
        if not subgoal.reasoning:
            raise ValueError("Subgoal has no reasoning")
            
        # Additional validation can be added here 