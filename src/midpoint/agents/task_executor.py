"""
Task Executor - Generic task execution agent for the Midpoint system.

IMPORTANT: This module implements a generic task execution system that uses LLM to interpret
and execute tasks. It MUST NOT contain any task-specific logic or hardcoded implementations.
All task-specific decisions and implementations should be handled by the LLM at runtime.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import os
import logging

from .models import TaskContext, ExecutionTrace, State, Goal, ExecutionResult
from .goal_decomposer import validate_repository_state
from .tools import (
    check_repo_state,
    create_branch,
    create_commit,
    get_current_hash,
    get_current_branch,
    checkout_branch,
    list_directory,
    read_file,
    search_code,
    edit_file,
    run_terminal_cmd,
    validate_repository_state,
    web_search,
    web_scrape
)
from .config import get_openai_api_key
from openai import AsyncOpenAI

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('TaskExecutor')

class TaskExecutor:
    """
    Generic task execution agent that uses LLM to implement changes.
    
    This class MUST:
    - Remain completely task-agnostic
    - Not contain any hardcoded task implementations
    - Delegate all task-specific decisions to the LLM
    - Use the provided tools to implement changes based on LLM decisions
    
    The workflow is:
    1. LLM analyzes the task and current repository state
    2. LLM decides what changes are needed
    3. LLM uses the available tools to implement those changes
    4. LLM validates the changes
    5. Changes are committed if valid
    """
    
    def __init__(self):
        """Initialize the TaskExecutor agent."""
        logger.info("Initializing TaskExecutor")
        # Initialize OpenAI client with API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        logger.info("TaskExecutor initialized successfully")
        
        # System prompt for the LLM that will make all task execution decisions
        self.system_prompt = """You are a task execution agent responsible for implementing code changes.
Your role is to:
1. Execute tasks using available tools
2. Track progress and report status
3. Handle errors gracefully
4. Maintain clean git state

IMPORTANT: A new branch has been created for you to work on. You are responsible for:
1. Making all necessary changes to implement the task
2. Creating commits with meaningful messages
3. Ensuring all changes are committed before completing
4. Returning the final commit hash in your output

Available tools:
- list_directory: List contents of a directory
- read_file: Read contents of a file
- search_code: Search for code patterns
- create_commit: Create a git commit
- run_terminal_cmd: Run a terminal command
- edit_file: Edit the contents of a file
- web_search: Search the web using DuckDuckGo's API
- web_scrape: Scrape content from a webpage

IMPORTANT TOOL USAGE NOTES:
1. When using tools, provide only the name without any prefixes (e.g., use "list_directory" not "functions.list_directory")
2. The create_commit tool will return the new commit hash - you must save this hash and use it as the final_commit_hash in your response
3. Always make your changes and verify them before creating a commit
4. After creating a commit, store the returned hash and use it in your final output

For each task:
1. Analyze the current repository state
2. Determine what changes are needed
3. Use the available tools to implement changes
4. Validate the changes
5. Create appropriate commits
6. Return the final commit hash from the create_commit call

Your response must be in JSON format with these fields:
{
    "actions": [
        {
            "tool": "string",  # Name of the tool to use (without any prefixes)
            "args": {},  # Arguments for the tool
            "purpose": "string"  # Why this action is needed
        }
    ],
    "final_commit_hash": "string",  # Hash returned from the create_commit call
    "validation_steps": [  # Steps to validate the changes
        "string"
    ],
    "task_completed": boolean,  # Whether the task was successfully completed
    "completion_reason": "string"  # Explanation if task was not completed
}"""

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
                    "name": "edit_file",
                    "description": "Edit the contents of a file",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file to edit"
                            },
                            "content": {
                                "type": "string",
                                "description": "New content for the file"
                            }
                        },
                        "required": ["repo_path", "file_path", "content"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_terminal_cmd",
                    "description": "Run a terminal command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "The command to run"
                            },
                            "cwd": {
                                "type": "string",
                                "description": "Working directory for the command"
                            }
                        },
                        "required": ["command", "cwd"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_commit",
                    "description": "Create a git commit with the given message",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "repo_path": {
                                "type": "string",
                                "description": "Path to the git repository"
                            },
                            "message": {
                                "type": "string",
                                "description": "Commit message"
                            }
                        },
                        "required": ["repo_path", "message"]
                    }
                }
            }
        ]

    async def execute_task(self, context: TaskContext, task: str) -> ExecutionResult:
        """
        Execute a task using LLM-driven decision making.
        
        This method MUST NOT contain any task-specific logic.
        All task-specific decisions and implementations should be made by the LLM.
        
        The LLM will:
        1. Analyze the task and repository state
        2. Decide what changes are needed
        3. Use the available tools to implement changes
        4. Validate the changes
        5. Create appropriate commits
        
        Args:
            context: The current task context
            task: The task description to execute
            
        Returns:
            ExecutionResult containing the execution outcome
        """
        logger.info(f"Starting task execution: {task}")
        logger.info(f"Repository: {context.state.repository_path}")
        logger.info(f"Current Git Hash: {context.state.git_hash}")
        
        # Initialize execution
        start_time = time.time()
        base_name = f"task-{context.iteration}"
        
        try:
            # Create a new branch for this execution
            logger.info(f"Creating new branch: {base_name}")
            branch_name = await create_branch(context.state.repository_path, base_name)
            logger.info(f"Created branch: {branch_name}")
            
            # Create the user prompt for the LLM
            user_prompt = f"""Task: {task}

Repository Path: {context.state.repository_path}
Branch: {branch_name} (newly created for this task)
Git Hash: {context.state.git_hash}

Please analyze the task and determine what changes are needed.
Use the available tools to implement the changes and validate them.
Remember to commit your changes and return the final commit hash."""

            logger.debug(f"User prompt: {user_prompt}")

            # Initialize messages
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Chat completion with tool use
            try:
                final_output = None
                iteration = 0
                
                # Loop until we get a final output
                while final_output is None:
                    iteration += 1
                    logger.info(f"Starting LLM iteration {iteration}")
                    
                    # Call OpenAI API
                    response = await self.client.chat.completions.create(
                        model="gpt-4",
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.1,
                        max_tokens=4000
                    )
                    
                    # Get the model's message
                    message = response.choices[0].message
                    logger.debug(f"LLM response: {message.content}")
                    
                    # Add the message to our conversation
                    messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})
                    
                    # If the model wants to use tools
                    if message.tool_calls:
                        logger.info(f"LLM requested {len(message.tool_calls)} tool calls")
                        # Handle each tool call
                        for tool_call in message.tool_calls:
                            # Get the function name and arguments
                            func_name = tool_call.function.name
                            func_args = json.loads(tool_call.function.arguments)
                            
                            # Handle "functions." prefix in tool names
                            actual_func_name = func_name
                            if func_name.startswith("functions."):
                                actual_func_name = func_name.split(".", 1)[1]
                                logger.info(f"Handling functions prefix: {func_name} -> {actual_func_name}")
                            
                            # Define a mapping of tool names to actual functions
                            tool_functions = {
                                "list_directory": list_directory,
                                "read_file": read_file,
                                "search_code": search_code,
                                "edit_file": edit_file,
                                "run_terminal_cmd": run_terminal_cmd,
                                "create_commit": create_commit,
                                "web_search": web_search,
                                "web_scrape": web_scrape
                            }
                            
                            if actual_func_name in tool_functions:
                                # Get the function to execute
                                tool_function = tool_functions[actual_func_name]
                                logger.info(f"Executing tool: {actual_func_name}")
                                logger.debug(f"Tool arguments: {json.dumps(func_args, indent=2)}")
                                
                                try:
                                    # Execute the function with appropriate arguments
                                    logger.info(f"Calling tool function {tool_function.__name__} with arguments")
                                    
                                    # Call different functions based on the tool name
                                    if actual_func_name == "list_directory":
                                        result = await list_directory(func_args["repo_path"], func_args.get("directory", "."))
                                    elif actual_func_name == "read_file":
                                        result = await read_file(
                                            func_args["repo_path"],
                                            func_args["file_path"],
                                            func_args.get("start_line", 0),
                                            func_args.get("max_lines", 100)
                                        )
                                    elif actual_func_name == "search_code":
                                        result = await search_code(
                                            func_args["repo_path"],
                                            func_args["pattern"],
                                            func_args.get("file_pattern", "*"),
                                            func_args.get("max_results", 20)
                                        )
                                    elif actual_func_name == "edit_file":
                                        result = await edit_file(
                                            func_args["repo_path"],
                                            func_args["file_path"],
                                            func_args["content"]
                                        )
                                    elif actual_func_name == "run_terminal_cmd":
                                        result = await run_terminal_cmd(
                                            command=func_args["command"],
                                            cwd=func_args["cwd"]
                                        )
                                    elif actual_func_name == "create_commit":
                                        result = await create_commit(
                                            func_args["repo_path"],
                                            func_args["message"]
                                        )
                                    elif actual_func_name == "web_search":
                                        result = await web_search(func_args["query"])
                                    elif actual_func_name == "web_scrape":
                                        result = await web_scrape(func_args["url"])
                                    
                                    # Log the result
                                    logger.info(f"Tool {actual_func_name} executed successfully")
                                    logger.debug(f"Tool result: {str(result)}")
                                    
                                    # For specific tools, add extra logging about repository state
                                    if actual_func_name in ['create_commit', 'edit_file', 'run_terminal_cmd']:
                                        try:
                                            repo_path = func_args.get('repo_path', context.state.repository_path)
                                            logger.info(f"Checking repository state after {actual_func_name}")
                                            repo_status = await check_repo_state(repo_path)
                                            logger.debug(f"Repository state after {actual_func_name}: {repo_status}")
                                            current_hash = await get_current_hash(repo_path)
                                            logger.info(f"Current hash after {actual_func_name}: {current_hash}")
                                        except Exception as state_e:
                                            logger.warning(f"Failed to check repository state: {str(state_e)}")
                                
                                except Exception as e:
                                    logger.error(f"Error executing tool {actual_func_name}: {str(e)}")
                                    logger.error(f"Exception type: {type(e).__name__}")
                                    logger.error(f"Exception args: {e.args}")
                                    result = f"Error: {str(e)}"
                            else:
                                logger.warning(f"Tool {actual_func_name} not found in available tools")
                                result = f"Error: Tool {actual_func_name} not found in available tools"
                            
                            # Add the result to messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": actual_func_name,
                                "content": str(result)
                            })
                    else:
                        # Try to parse the final output
                        try:
                            # Parse the final output
                            if not isinstance(message.content, str):
                                logger.error(f"Unexpected message content type: {type(message.content)}")
                                continue
                            
                            try:
                                final_output = json.loads(message.content)
                                logger.info("Received final output from LLM")
                                logger.debug(f"Final output: {json.dumps(final_output, indent=2)}")
                                
                                # Validate required fields
                                required_fields = ["task_completed", "final_commit_hash"]
                                if not all(field in final_output for field in required_fields):
                                    logger.warning("Final output missing required fields")
                                    logger.debug(f"Missing fields: {[field for field in required_fields if field not in final_output]}")
                                    continue
                            except json.JSONDecodeError as e:
                                logger.error(f"Failed to parse LLM output as JSON: {str(e)}")
                                logger.debug(f"Raw content: {message.content[:500]}...")
                                iteration += 1
                                continue
                            
                            # Check if final commit hash matches the current hash
                            current_hash = await get_current_hash(context.state.repository_path)
                            logger.info(f"Final hash check - Expected: {final_output['final_commit_hash']}, Got: {current_hash}")
                            
                            # Validate the output format
                            if not all(key in final_output for key in ["actions", "final_commit_hash", "validation_steps", "task_completed", "completion_reason"]):
                                logger.warning("Final output missing required fields")
                                final_output = None
                                messages.append({
                                    "role": "user",
                                    "content": "Please provide your response in the correct JSON format with all required fields."
                                })
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse LLM output as JSON: {str(e)}")
                            messages.append({
                                "role": "user",
                                "content": "Please provide your response in valid JSON format."
                            })
                
                # Validate the final commit hash
                current_hash = await get_current_hash(context.state.repository_path)
                if final_output["final_commit_hash"] != current_hash:
                    logger.error(f"Final commit hash mismatch. Expected: {final_output['final_commit_hash']}, Got: {current_hash}")
                    return ExecutionResult(
                        success=False,
                        branch_name=branch_name,
                        git_hash=current_hash,
                        error_message="Final commit hash mismatch",
                        execution_time=time.time() - start_time,
                        repository_path=context.state.repository_path
                    )
                
                # Return result based on task completion
                return ExecutionResult(
                    success=final_output["task_completed"],
                    branch_name=branch_name,
                    git_hash=final_output["final_commit_hash"],
                    error_message=None if final_output["task_completed"] else final_output["completion_reason"],
                    execution_time=time.time() - start_time,
                    repository_path=context.state.repository_path
                )
                
            except Exception as e:
                logger.error(f"Error during execution: {str(e)}")
                return ExecutionResult(
                    success=False,
                    branch_name=branch_name,
                    git_hash=context.state.git_hash,
                    error_message=f"Error during execution: {str(e)}",
                    execution_time=time.time() - start_time,
                    repository_path=context.state.repository_path
                )
            
        except Exception as e:
            logger.error(f"Fatal error during task execution: {str(e)}")
            # If execution fails, clean up the branch and go back to main
            try:
                await run_terminal_cmd(
                    command=["git", "checkout", "main"], 
                    cwd=context.state.repository_path
                )
                # Don't assume we know the branch name if create_branch failed
                if 'branch_name' in locals():
                    await run_terminal_cmd(
                        command=["git", "branch", "-D", branch_name],
                        cwd=context.state.repository_path
                    )
            except Exception as cleanup_error:
                logger.error(f"Error during cleanup: {str(cleanup_error)}")
                
            return ExecutionResult(
                success=False,
                branch_name=base_name,  # Use base_name since we don't have actual branch name
                git_hash=context.state.git_hash,
                error_message=str(e),
                execution_time=time.time() - start_time,
                repository_path=context.state.repository_path
            ) 