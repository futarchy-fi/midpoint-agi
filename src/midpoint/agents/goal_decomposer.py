"""
Goal Decomposition agent implementation.

This module implements the GoalDecomposer agent that determines the next step
toward achieving a complex goal.
"""

import os
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext
from midpoint.agents.tools import (
    list_directory,
    read_file,
    search_code,
    get_current_hash
)
from midpoint.agents.config import get_openai_api_key
from ..utils.logging import log_manager

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

As part of your analysis, you MUST determine whether the next step requires further decomposition:
- If the next step is still complex and would benefit from being broken down further, set 'requires_further_decomposition' to TRUE
- If the next step is simple enough to be directly implemented by a TaskExecutor agent, set 'requires_further_decomposition' to FALSE

Also identify any relevant context that should be passed to child subgoals:
- Include ONLY information that will be directly helpful for understanding and implementing the subgoal
- DO NOT include high-level strategic information that isn't directly relevant to the subgoal
- Structure this as key-value pairs in the 'relevant_context' field

You have access to these tools:
- list_directory: List files and directories in the repository
- read_file: Read the contents of a file
- search_code: Search the codebase for patterns

You MUST provide a structured response in JSON format with these fields:
- next_step: A clear description of the single next step to take
- validation_criteria: List of measurable criteria to validate this step's completion
- reasoning: Explanation of why this is the most promising next action
- requires_further_decomposition: Boolean indicating if this step needs further breakdown (true) or can be directly executed (false)
- relevant_context: Object containing relevant information to pass to child subgoals"""
        
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
                    "description": "Search the codebase for patterns",
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
            },
            {
                "type": "function",
                "function": {
                    "name": "web_search",
                    "description": "Search the web using DuckDuckGo's API",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query"
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Maximum number of results to return",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "web_scrape",
                    "description": "Scrape content from a webpage",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "url": {
                                "type": "string",
                                "description": "The URL to scrape"
                            }
                        },
                        "required": ["url"]
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
                        elif function_name == "web_search":
                            result_str = await web_search(**function_args)
                        elif function_name == "web_scrape":
                            result_str = await web_scrape(**function_args)
                        else:
                            result_str = f"Error: Unknown function {function_name}"
                            
                        # Add the function result to our messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": function_name,
                            "content": result_str
                        })
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
                                # Extract requires_further_decomposition (default to True if not provided)
                                requires_further_decomposition = output_data.get("requires_further_decomposition", True)
                                
                                # Extract relevant_context (default to empty dict if not provided)
                                relevant_context = output_data.get("relevant_context", {})
                                
                                final_output = SubgoalPlan(
                                    next_step=output_data["next_step"],
                                    validation_criteria=output_data["validation_criteria"],
                                    reasoning=output_data["reasoning"],
                                    requires_further_decomposition=requires_further_decomposition,
                                    relevant_context=relevant_context,
                                    metadata={
                                        "raw_response": message.content,
                                        "tool_usage": tool_usage
                                    }
                                )
                            else:
                                # Ask for a properly formatted response
                                messages.append({
                                    "role": "user",
                                    "content": "Please provide a valid JSON response with the fields: next_step, validation_criteria, reasoning, requires_further_decomposition, and relevant_context."
                                })
                        except json.JSONDecodeError:
                            # If not valid JSON, ask for a properly formatted response
                            messages.append({
                                "role": "user",
                                "content": "Please provide your response in valid JSON format with the fields: next_step, validation_criteria, reasoning, requires_further_decomposition, and relevant_context."
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

    async def decompose_recursively(self, context: TaskContext, log_file: str = "goal_hierarchy.log") -> List[SubgoalPlan]:
        """
        Recursively decompose a goal until reaching directly executable tasks.
        
        Args:
            context: The current task context containing the goal and state
            log_file: Path to log file for visualizing the goal hierarchy (deprecated)
            
        Returns:
            List of SubgoalPlan objects representing the decomposition hierarchy
        """
        # Validate repository state
        await validate_repository_state(
            context.state.repository_path, 
            context.state.git_hash
        )
        
        # Get the decomposition depth for logging
        depth = self._get_decomposition_depth(context)
        
        # Get the next subgoal
        subgoal = await self.determine_next_step(context)
        
        # Log this decomposition step with branch and git info
        log_manager.log_goal_decomposition(
            depth=depth,
            parent_goal=context.goal.description,
            subgoal=subgoal.next_step,
            branch_name=context.state.branch_name if hasattr(context.state, 'branch_name') else None,
            git_hash=context.state.git_hash
        )
        
        # If subgoal doesn't need further decomposition, return it
        if not subgoal.requires_further_decomposition:
            log_manager.log_execution_ready(
                depth=depth + 1,
                task=subgoal.next_step,
                branch_name=context.state.branch_name if hasattr(context.state, 'branch_name') else None,
                git_hash=context.state.git_hash
            )
            return [subgoal]
        
        # Create a new context with this subgoal as the goal
        new_context = TaskContext(
            state=context.state,
            goal=Goal(
                description=subgoal.next_step,
                validation_criteria=subgoal.validation_criteria,
                success_threshold=context.goal.success_threshold
            ),
            iteration=0,  # Reset for the new subgoal
            execution_history=context.execution_history,
            metadata={
                **(context.metadata if hasattr(context, 'metadata') else {}),
                "parent_goal": context.goal.description,
                "parent_context": subgoal.relevant_context
            }
        )
        
        # Recursively decompose and collect all resulting subgoals
        sub_subgoals = await self.decompose_recursively(new_context)
        
        # Return the current subgoal and all its sub-subgoals
        return [subgoal] + sub_subgoals
    
    def _get_decomposition_depth(self, context: TaskContext) -> int:
        """Determine the current decomposition depth from context metadata."""
        # If there's no metadata or no parent_goal, we're at the root (depth 0)
        if not hasattr(context, 'metadata') or not context.metadata:
            return 0
            
        # Count the number of parent_goal references in metadata
        depth = 0
        current_metadata = context.metadata
        while current_metadata and 'parent_goal' in current_metadata:
            depth += 1
            current_metadata = current_metadata.get('parent_context', {})
            
        return depth

async def validate_repository_state(repo_path: str, expected_hash: str) -> bool:
    """
    Validate repository is in expected state before decomposition.
    
    Args:
        repo_path: Path to the git repository
        expected_hash: Expected git hash of the repository
        
    Returns:
        True if repository is in expected state
        
    Raises:
        ValueError: If repository is not in expected state
    """
    # Check if repo exists
    if not os.path.exists(os.path.join(repo_path, ".git")):
        raise ValueError(f"Not a git repository: {repo_path}")
    
    # Check current hash
    current_hash = await get_current_hash(repo_path)
    if current_hash != expected_hash:
        raise ValueError(f"Repository hash mismatch. Expected: {expected_hash}, Got: {current_hash}")
    
    # Check for uncommitted changes
    process = await asyncio.create_subprocess_exec(
        "git", "status", "--porcelain",
        cwd=repo_path,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await process.communicate()
    
    if stdout.strip():
        raise ValueError(f"Repository has uncommitted changes: {stdout.decode()}")
    
    return True 