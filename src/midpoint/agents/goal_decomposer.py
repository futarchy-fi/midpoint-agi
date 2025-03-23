"""
Goal Decomposition agent implementation.

This module implements the GoalDecomposer agent that determines the next step
toward achieving a complex goal.
"""

import os
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext
from midpoint.agents.tools import (
    list_directory,
    read_file,
    search_code,
    get_current_hash,
    web_search,
    web_scrape
)
from midpoint.agents.config import get_openai_api_key
from midpoint.utils.logging import log_manager
from dotenv import load_dotenv
import logging
from pathlib import Path
import sys

load_dotenv()

# Create a function to configure logging so it only happens when needed
def configure_logging(debug=False, quiet=False, log_dir_path="logs"):
    """
    Configure logging for the goal decomposer.
    
    Args:
        debug: Whether to show debug messages on console
        quiet: Whether to only show warnings and the final result
        log_dir_path: Directory to store log files
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir_path)
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file name with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"goal_decomposer_{timestamp}.log"
    
    # Create file handler for full logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Create console handler with a more concise format
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    
    # Set console handler level based on arguments
    if debug:
        console_handler.setLevel(logging.DEBUG)
    elif quiet:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs
    
    # Remove existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add our custom handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Set up a filter for console output to make it more concise
    class ConsoleFormatFilter(logging.Filter):
        def filter(self, record):
            # Only process INFO level logs for formatting
            if record.levelno == logging.INFO:
                # Make emojis and messages more concise
                if '📂 Listing directory:' in record.msg:
                    record.msg = record.msg.replace('📂 Listing directory:', '📂')
                elif '📄 Reading:' in record.msg:
                    record.msg = record.msg.replace('📄 Reading:', '📄')
                elif '🔍 Searching code:' in record.msg:
                    record.msg = record.msg.replace('🔍 Searching code:', '🔍')
                elif '🤖 API call completed' in record.msg:
                    return False  # Don't show API calls in console
                elif '✅ Next step:' in record.msg:
                    # Show this message only in standalone mode
                    if 'main' not in sys._getframe().f_back.f_code.co_name:
                        return True
                    return False  # Don't show in main() since we have better formatting there
                elif 'Validation criteria:' in record.msg:
                    # Hide validation criteria messages in main() since we have better formatting there
                    if 'main' not in sys._getframe().f_back.f_code.co_name:
                        return True
                    return False
                elif 'Requires further decomposition:' in record.msg:
                    # Hide decomposition info in main() since we have better formatting there
                    if 'main' not in sys._getframe().f_back.f_code.co_name:
                        return True
                    return False
                elif 'Determining next step for goal:' in record.msg:
                    try:
                        # Try to safely extract the goal description
                        if record.args and len(record.args) > 0:
                            goal_desc = str(record.args[0])
                            record.msg = f"🎯 Goal: {goal_desc}"
                        else:
                            record.msg = "🎯 Processing goal"
                        record.args = ()  # Clear arguments to avoid formatting errors
                    except:
                        record.msg = "🎯 Processing goal"
                        record.args = ()
                elif '🚀 Starting GoalDecomposer' in record.msg:
                    record.msg = '🚀 Starting'
                elif 'HTTP Request:' in record.msg or 'API' in record.msg:
                    return False  # Don't show HTTP requests in console
                elif 'Validating repository state' in record.msg:
                    return False  # Hide validation message in console
            return True
    
    # Apply the filter only to the console handler
    console_handler.addFilter(ConsoleFormatFilter())
    
    # Log the configuration
    if quiet:
        print("Running in quiet mode - only showing result and errors...", file=sys.stderr)
    
    # Return log file path for reference
    return log_file

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

    async def determine_next_step(self, context: TaskContext, setup_logging=False, debug=False, quiet=False) -> SubgoalPlan:
        """
        Determine the next step toward achieving the goal.
        
        Args:
            context: The current task context containing the goal and state
            setup_logging: Whether to set up logging for this invocation (should be False when called from orchestrator)
            debug: Whether to enable debug logging on console
            quiet: Whether to minimize console output
            
        Returns:
            A SubgoalPlan containing the next step and validation criteria
            
        Raises:
            ValueError: If the goal or context is invalid
            Exception: For other errors during execution
        """
        # Set up logging if requested (only when called directly, not from orchestrator)
        if setup_logging:
            configure_logging(debug, quiet)
            
        logging.info("Determining next step for goal: %s", context.goal.description)
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
                # Log BEFORE making the API call (at DEBUG level now)
                logging.debug(f"Calling OpenAI API with model: gpt-4o (iteration {len(messages)//2})")
                logging.debug("Request messages: %s", json.dumps(self._serialize_messages(messages), indent=2))
                
                try:
                    response = await self.client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        tools=self.tools,
                        tool_choice="auto",
                        temperature=0.1,
                        max_tokens=4000
                    )
                    logging.info(f"🤖 API call completed ({len(messages)//2})")
                except Exception as e:
                    logging.error("Error calling OpenAI API: %s", str(e))
                    raise
                
                # Get the model's message
                message = response.choices[0].message
                
                # Add the message to our conversation
                messages.append({"role": "assistant", "content": message.content, "tool_calls": message.tool_calls})
                
                # Check if the model wants to use tools
                if message.tool_calls:
                    # Create a human-friendly summary of tool calls for INFO level
                    tool_summary = []
                    for tc in message.tool_calls:
                        args = json.loads(tc.function.arguments)
                        if tc.function.name == "list_directory":
                            dir_path = args.get("directory", ".")
                            tool_summary.append(f"📂 Listing directory: {dir_path}")
                        elif tc.function.name == "read_file":
                            file_path = args.get("file_path", "unknown")
                            tool_summary.append(f"📄 Reading: {file_path}")
                        elif tc.function.name == "search_code":
                            pattern = args.get("pattern", "unknown")
                            tool_summary.append(f"🔍 Searching code: {pattern}")
                        elif tc.function.name == "web_search":
                            query = args.get("query", "unknown")
                            tool_summary.append(f"🌐 Web search: {query}")
                        elif tc.function.name == "web_scrape":
                            url = args.get("url", "unknown")
                            tool_summary.append(f"🌐 Scraping: {url}")
                        else:
                            tool_summary.append(f"{tc.function.name}")
                    
                    # Log the summary at INFO level - only the first tool to keep it concise
                    if tool_summary:
                        logging.info(f"{tool_summary[0]}")
                        # Log additional tools at DEBUG level if there are multiple
                        if len(tool_summary) > 1:
                            logging.debug(f"Additional tools: {', '.join(tool_summary[1:])}")
                    
                    # Process each tool call
                    for tool_call in message.tool_calls:
                        try:
                            # Get function details
                            function_name = tool_call.function.name
                            function_args = json.loads(tool_call.function.arguments)
                            
                            # Log details at DEBUG level
                            logging.debug("Executing tool: %s with arguments: %s", function_name, json.dumps(function_args))
                            
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
                            
                            # Log completion at DEBUG level
                            logging.debug(f"Completed executing tool: {function_name}")
                                
                            # Add the function result to our messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": result_str
                            })
                        except ValueError as ve:
                            # Handle specific errors with more user-friendly messages
                            error_message = str(ve)
                            logging.debug("Tool execution error: %s", error_message)
                            if "Directory does not exist" in error_message:
                                dir_name = error_message.split(': ')[1] if ': ' in error_message else ''
                                logging.info(f"❌ Directory not found: {dir_name}")                              
                            
                            # Add the error message to our messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": f"Error: {error_message}"
                            })
                        except Exception as e:
                            # Handle any other exceptions (keep as ERROR)
                            error_message = f"Unexpected error during tool execution: {str(e)}"
                            logging.error(error_message)
                            
                            # Add the error message to our messages
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": function_name,
                                "content": f"Error: {error_message}"
                            })
                else:
                    # If no tool calls, this should be the final response
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
            
            # Log tool usage at debug level
            logging.debug("Tool usage: %s", tool_usage)

            # Add logging for API call details - use serialized messages at debug level
            try:
                serialized_messages = self._serialize_messages(messages)
                logging.debug("API call history: %s", json.dumps(serialized_messages, indent=2))
            except Exception as e:
                logging.error("Failed to serialize messages for logging: %s", str(e))

            # Log successful outcome at info level with a more complete message
            logging.info(f"✅ Next step: {final_output.next_step}")
            
            # Add more detailed info at INFO level for better visibility
            logging.info("Validation criteria:")
            for i, criterion in enumerate(final_output.validation_criteria, 1):
                logging.info(f"  {i}. {criterion}")
            
            logging.info(f"Requires further decomposition: {final_output.requires_further_decomposition}")
            
            # Add logging for the final output at debug level
            try:
                logging.debug("Final output details: %s", json.dumps(final_output.__dict__, indent=2))
            except Exception as e:
                logging.error("Failed to serialize final output for logging: %s", str(e))

            # Fix the NoneType error - check if tool_calls exists and is not None
            if hasattr(message, 'tool_calls') and message.tool_calls is not None:
                logging.debug("Tool calls in final message: %d", len(message.tool_calls))
            else:
                logging.debug("No tool calls in final message")

            return final_output
            
        except Exception as e:
            # Let the main function handle the specific error types
            raise
    
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

    def _serialize_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert OpenAI message objects to serializable dictionaries."""
        serialized = []
        for msg in messages:
            # Create a base serialized message with the role
            if isinstance(msg, dict) and "role" in msg:
                serialized_msg = {"role": msg["role"]}
                
                # Handle content
                if isinstance(msg, dict) and "content" in msg:
                    serialized_msg["content"] = msg["content"]
                
                # Handle tool calls
                if isinstance(msg, dict) and "tool_calls" in msg and msg["tool_calls"]:
                    serialized_tool_calls = []
                    
                    for tool_call in msg["tool_calls"]:
                        # For OpenAI objects with attribute-based access
                        if hasattr(tool_call, "function") and hasattr(tool_call, "id"):
                            try:
                                serialized_tool_call = {
                                    "id": tool_call.id,
                                    "type": "function",
                                    "function": {
                                        "name": tool_call.function.name,
                                        "arguments": tool_call.function.arguments
                                    }
                                }
                                serialized_tool_calls.append(serialized_tool_call)
                            except AttributeError:
                                # Skip if attributes aren't accessible
                                logging.debug("Skipping tool call due to missing attributes")
                                continue
                        # For dictionary-based tool calls
                        elif isinstance(tool_call, dict) and "function" in tool_call:
                            serialized_tool_calls.append(tool_call)
                    
                    # Only add tool_calls if we have any
                    if serialized_tool_calls:
                        serialized_msg["tool_calls"] = serialized_tool_calls
                
                serialized.append(serialized_msg)
            else:
                # If not a dict with role, just add a simplified version
                serialized.append({"role": "unknown", "content": str(msg)})
        
        return serialized

    async def decompose_recursively(self, context: TaskContext, log_file: str = "goal_hierarchy.log") -> List[SubgoalPlan]:
        """
        Recursively decompose a goal until reaching directly executable tasks.
        
        Args:
            context: The current task context containing the goal and state
            log_file: Path to log file for visualizing the goal hierarchy (deprecated)
            
        Returns:
            List of SubgoalPlan objects representing the decomposition hierarchy
        """
        logging.info("Starting recursive decomposition for the goal: %s", context.goal.description)
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

async def validate_repository_state(repo_path, git_hash=None, skip_clean_check=False):
    """Validate that the repository is in a good state for goal decomposition."""
    logging.info("Validating repository state for path: %s", repo_path)
    
    if not os.path.isdir(repo_path):
        raise ValueError(f"Repository path does not exist: {repo_path}")
    
    # Check if this is a git repository
    git_dir = os.path.join(repo_path, ".git")
    if not os.path.isdir(git_dir):
        logging.warning("Path does not appear to be a git repository: %s", repo_path)
        return
    
    # Skip further checks if requested
    if skip_clean_check:
        return
    
    # Check if the repository has uncommitted changes
    try:
        import subprocess
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            check=True
        )
        
        if result.stdout.strip():
            raise ValueError(f"Repository has uncommitted changes: {repo_path}")
    except subprocess.CalledProcessError as e:
        logging.error("Failed to check git status: %s", str(e))
        raise ValueError(f"Failed to check git status: {str(e)}")
    
    # If git_hash is provided, check that it matches the current hash
    if git_hash:
        try:
            current_hash = await get_current_hash(repo_path)
            if current_hash != git_hash:
                logging.warning("Repository hash mismatch: expected %s, got %s", git_hash, current_hash)
        except Exception as e:
            logging.error("Failed to check git hash: %s", str(e))
            raise ValueError(f"Failed to check git hash: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description="Run GoalDecomposer to determine the next step.")
    parser.add_argument('--repo-path', required=True, help='Path to the git repository')
    parser.add_argument('--goal-description', required=True, help='Description of the goal')
    parser.add_argument('--iteration', type=int, default=0, help='Iteration number')
    parser.add_argument('--execution-history', type=str, default='[]', help='Execution history as JSON string')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging on console (all logs are always written to file)')
    parser.add_argument('--quiet', action='store_true', help='Only show warnings and errors on console')

    args = parser.parse_args()

    # Set up logging based on command line arguments
    log_file = configure_logging(args.debug, args.quiet)

    try:
        logging.info("🚀 Starting GoalDecomposer")
        # Create the TaskContext
        state = State(repository_path=args.repo_path, git_hash="", description="Current state description")
        goal = Goal(description=args.goal_description)
        context = TaskContext(state=state, goal=goal, iteration=args.iteration, execution_history=[])

        # Validate repository state and get current hash
        current_hash = asyncio.run(get_current_hash(args.repo_path))
        state.git_hash = current_hash
        asyncio.run(validate_repository_state(
            args.repo_path,
            git_hash=current_hash,  # Use the new parameter name
            skip_clean_check=True
        ))

        # Initialize GoalDecomposer and determine the next step
        decomposer = GoalDecomposer()
        next_step = asyncio.run(decomposer.determine_next_step(
            context,
            setup_logging=True,  # This configures logging
            debug=args.debug,
            quiet=args.quiet
        ))

        # Output the result - print detailed information to console
        # Show more complete information about the next step
        print("\n\n====================")
        print("=== NEXT STEP ===")
        print("====================")
        print(f"{next_step.next_step}")
        
        print("\n====================")
        print("=== VALIDATION CRITERIA ===")
        print("====================")
        for i, criterion in enumerate(next_step.validation_criteria, 1):
            print(f"{i}. {criterion}")
        
        print("\n====================")
        print("=== REASONING ===")
        print("====================")
        print(f"{next_step.reasoning}")
        
        print("\n====================")
        print("=== REQUIRES FURTHER DECOMPOSITION ===")
        print("====================")
        print(f"{next_step.requires_further_decomposition}")
        
        # If there's relevant context, display it too
        if next_step.relevant_context:
            print("\n====================")
            print("=== RELEVANT CONTEXT ===")
            print("====================")
            for key, value in next_step.relevant_context.items():
                if isinstance(value, list):
                    print(f"{key}:")
                    for item in value:
                        print(f"  - {item}")
                else:
                    print(f"{key}: {value}")
        
        # Log the detailed output to the log file
        logging.debug(f"Validation Criteria: {next_step.validation_criteria}")
        logging.debug(f"Reasoning: {next_step.reasoning}")
    except TypeError as e:
        if "'NoneType' object is not iterable" in str(e):
            print("Error: The agent response processing failed. This typically happens when:")
            print("1. The agent provided a final response without tool calls")
            print("2. The response format wasn't properly handled")
            print("Try running with --debug to see more details about the API interactions.")
            print("Technical details:", str(e))
            logging.debug("NoneType iteration error: %s", str(e))
            return
        else:
            print("Error: Missing required argument. Please ensure all required fields are provided.")
            print("Details:", str(e))
    except ValueError as e:
        if "uncommitted changes" in str(e):
            print("Error: The repository has uncommitted changes.")
            print("Please commit or stash your changes before proceeding.")
            print("Details:", str(e).split(':', 1)[1].strip())
            return
        elif "Directory does not exist" in str(e):
            print(f"Warning: {str(e)}")
            print("The agent tried to access a directory that doesn't exist.")
            print("This may be because the component is not yet implemented or is in a different location.")
            logging.debug("Failed directory access: %s", str(e))
            return
        else:
            print("Error:", str(e))
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
        import traceback
        logging.error("Traceback: %s", traceback.format_exc())
        print(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    main() 