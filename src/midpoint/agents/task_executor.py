"""
Task Executor - Generic task execution agent for the Midpoint system.

IMPORTANT: This module implements a generic task execution system that uses LLM to interpret
and execute tasks. It MUST NOT contain any task-specific logic or hardcoded implementations.
All task-specific decisions and implementations should be handled by the LLM at runtime.
"""

import asyncio
import time
import json
import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import re
import os
import logging
from pathlib import Path
import subprocess

from .models import TaskContext, ExecutionTrace, State, Goal, ExecutionResult
from .tools import initialize_all_tools
from .tools.processor import ToolProcessor
from .tools.registry import ToolRegistry
from .tools.git_tools import create_branch, create_commit, get_current_hash, get_current_branch
from .tools.filesystem_tools import list_directory, read_file
from .tools.code_tools import search_code
from .tools.web_tools import web_search, web_scrape
from .tools.terminal_tools import run_terminal_cmd
from .tools.memory_tools import store_memory_document, retrieve_memory_documents
from .config import get_openai_api_key
from openai import OpenAI

# Set up logging
logger = logging.getLogger('TaskExecutor')

def configure_logging(debug=False, quiet=False, log_dir_path="logs"):
    """
    Configure logging for the TaskExecutor.
    
    Args:
        debug: Whether to show debug messages on console
        quiet: Whether to only show warnings and errors
        log_dir_path: Directory to store log files
    """
    # Create log directory if it doesn't exist
    log_dir = Path(log_dir_path)
    log_dir.mkdir(exist_ok=True)
    
    # Create a unique log file name with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"task_executor_{timestamp}.log"
    
    # Create file handler for full logging
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
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
    
    # Create a special formatter for task execution events
    class TaskExecutorFilter(logging.Filter):
        def filter(self, record):
            # Format specific log records for better clarity
            if hasattr(record, 'taskstep') and record.taskstep:
                record.msg = f"📋 Task Step: {record.msg}"
            if hasattr(record, 'tool') and record.tool:
                record.msg = f"🔧 Tool: {record.tool} - {record.msg}"
            if hasattr(record, 'validation') and record.validation:
                record.msg = f"✓ Validation: {record.msg}"
            if hasattr(record, 'memory') and record.memory:
                record.msg = f"📚 Memory: {record.msg}"
            return True
    
    console_handler.addFilter(TaskExecutorFilter())
    logger.info(f"Logging configured. Full logs will be saved to: {log_file}")
    return log_file

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
        self.client = OpenAI(api_key=api_key)
        
        # Initialize tool registry and processor
        self.tool_registry = ToolRegistry()
        initialize_all_tools()
        self.tool_processor = ToolProcessor(self.client)
        
        # Create a lock for concurrent execution
        self._execution_lock = asyncio.Lock()
        
        # Generate system prompt after tool registry is initialized
        self.system_prompt = self._generate_system_prompt()
        
        # Initialize conversation buffer for memory storage
        self.conversation_buffer = []
        
        logger.info("TaskExecutor initialized successfully")

    def _save_interaction_to_memory(self, context: TaskContext):
        """
        Save the conversation to memory, following the GoalDecomposer's pattern.
        """
        if not context.state.memory_repository_path or not context.state.memory_hash:
            logger.info("No memory repo or hash available, skipping _save_interaction_to_memory")
            return

        try:
            # Get repository path and memory hash
            memory_repo_path = context.state.memory_repository_path
            memory_hash = context.state.memory_hash
            
            # Create timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Format conversation content
            content = f"# Task Execution Conversation\n\nTimestamp: {timestamp}\n\n"
            
            # Add each relevant entry from the conversation buffer
            for entry in self.conversation_buffer:
                entry_type = entry.get("type", "unknown")
                
                # Format based on entry type
                if entry_type == "user_message":
                    content += f"## User Message\n\n{entry.get('user_prompt', '')}\n\n"
                elif entry_type == "assistant_response":
                    content += f"## Assistant Response\n\n{entry.get('final_content', '')}\n\n"
                elif entry_type == "tool_usage":
                    tool_name = entry.get('tool_name', 'unknown_tool')
                    tool_args = entry.get('tool_args', {})
                    content += f"## Tool Usage: {tool_name}\n\n"
                    content += f"Arguments: {json.dumps(tool_args, indent=2)}\n\n"
            
            # Add metadata
            content += "## Metadata\n\n"
            content += f"- Task ID: {context.metadata.get('task_id', 'unknown')}\n"
            content += f"- Parent Goal: {context.metadata.get('parent_goal', 'none')}\n"
            content += f"- Repository: {context.state.repository_path}\n"
            content += f"- Branch: {context.state.branch_name}\n"
            content += f"- Git Hash: {context.state.git_hash}\n"
            content += f"- Execution Timestamp: {timestamp}\n"
            
            # Get task name from context
            task_name = context.goal.description
            
            # Create category with task name
            category = "task_execution"
            if task_name:
                # Sanitize task name for use in category
                safe_task_name = re.sub(r'[^a-zA-Z0-9_-]', '_', task_name)[:50]  # Limit length
                category = f"task_execution_{safe_task_name}"
            
            # Store in memory
            result = store_memory_document(
                content=content,
                category=category,
                metadata={
                    "interaction_type": "task_execution",
                    "task_id": context.metadata.get("task_id"),
                    "parent_goal": context.metadata.get("parent_goal"),
                    "repository": context.state.repository_path,
                    "branch": context.state.branch_name,
                    "git_hash": context.state.git_hash,
                    "timestamp": timestamp,
                    "task_name": task_name,
                    "commit_message": f"Add task execution for: {task_name}" if task_name else "Add task execution"
                },
                memory_repo_path=memory_repo_path,
                memory_hash=memory_hash
            )
            
            if result.get("success", False):
                logger.info(f"Successfully saved task execution conversation to memory: {result.get('document_path', 'unknown')}", extra={'memory': True})
            else:
                logger.error(f"Failed to save conversation to memory: {result.get('error', 'Unknown error')}")
            
        except Exception as e:
            logger.error(f"Failed to store conversation to memory: {str(e)}")
            logger.info("Memory operation details:")
            logger.info(f"  Repository path: {context.state.memory_repository_path}")
            logger.info(f"  Memory hash: {context.state.memory_hash[:8] if context.state.memory_hash else 'None'}")
            logger.info(f"  Task: {context.goal.description}")

    def _generate_system_prompt(self) -> str:
        """
        Generate the system prompt with tool descriptions.
        """
        # Get all available tools (excluding memory tools)
        available_tools = []
        # Use ToolRegistry.get_tool_schemas() which now filters out memory tools
        for schema in ToolRegistry.get_tool_schemas():
            tool_info = schema.get("function", {})
            available_tools.append({
                "name": tool_info.get("name", ""),
                "description": tool_info.get("description", ""),
                "parameters": tool_info.get("parameters", {})
            })
        
        # Create a formatted list of tool descriptions
        tool_descriptions_text = "\n".join([
            f"- {tool['name']}: {tool['description']}"
            for tool in available_tools
        ])
        
        # Create the system prompt with the tool descriptions
        system_prompt = f"""You are a task execution agent responsible for implementing changes.
Your role is to:
1. Analyze the task and repository state.
2. Assess task feasibility: Determine if the task is clear enough and possible with your available tools.
3. Execute tasks using available tools if feasible.
4. Track progress and report status.
5. Handle errors gracefully.
6. Maintain clean git state.

You have the following tools available:
{tool_descriptions_text}

For each task:
1. **Assess Feasibility First:** Before executing, determine if the task's core intent is achievable with your tools. Is the task description specific and clear?
2. **Execute if Feasible:** If the task is clear and achievable, use the available tools to implement the necessary changes (e.g., code modifications).
3. **Validate Changes:** If changes were made, attempt to validate them.
4. **Commit Changes:** If changes were made and validated, create appropriate commits.

**Critical Failure Reporting:**
- If the core action requested by the task is impossible due to tool limitations (e.g., requires GUI interaction, opening an editor), you MUST report failure (`success: false`). Explain *why* it's impossible in the summary, even if you performed related preparatory steps like locating a file.
- If the task description is too ambiguous, lacks sufficient detail, or requires information you cannot obtain with your tools, you MUST report failure (`success: false`). Explain the ambiguity or missing information clearly in the summary.
- Report `success: true` ONLY if the core intent of the task was successfully achieved using your tools.

When you've completed the task or determined it cannot be completed, provide a final response with:
1. A summary of what you did, OR a clear explanation of why the task failed (see Critical Failure Reporting).
2. Whether the task was completed successfully (`success: true/false`).
3. Any validation steps that can be taken to verify the changes (if successful).
4. The git hash of the final commit, if you made code changes.

IMPORTANT: Your final response MUST be in valid JSON format with these fields:
{{
  "summary": "A clear summary of what you did or why the task failed",
  "success": true/false,
  "validation_steps": ["List of validation steps (empty if failed)"],
  "git_hash": "Optional git hash if code was changed"
}}

For exploratory or study tasks, focus on analyzing the codebase and documenting your findings. If the task is purely analysis, report success if the analysis was completed."""
        
        return system_prompt

    def execute_task(self, context: TaskContext, task_description: Optional[str] = None) -> ExecutionResult:
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
        6. Store relevant information in memory repository
        
        Args:
            context: The current task context
            task_description: Optional task description to override context.goal.description
            
        Returns:
            ExecutionResult containing the execution outcome
        """
        try:
            # Validate memory state
            if not context.memory_state:
                raise ValueError("Memory state is required for task execution")
            
            # Validate memory state attributes
            memory_hash = getattr(context.memory_state, "memory_hash", None)
            memory_path = getattr(context.memory_state, "repository_path", None)
            if not memory_hash or not memory_path:
                raise ValueError("Memory state must have both memory_hash and repository_path")
            
            # Use provided task description if available, otherwise use goal description
            task = task_description or context.goal.description
            
            # Log task information
            logger.info(f"Executing task: {task}")
            logger.info(f"Repository: {context.state.repository_path}")
            logger.info(f"Branch: {context.state.branch_name}")
            
            # Log memory state if available
            if context.memory_state:
                logger.info(f"Memory repository: {memory_path}")
                if memory_hash:
                    logger.info(f"Initial memory hash: {memory_hash[:8]}")
            elif context.state.memory_repository_path:
                logger.info(f"Memory repository: {context.state.memory_repository_path}")
                if context.state.memory_hash:
                    logger.info(f"Initial memory hash: {context.state.memory_hash[:8]}")
            
            # Create the user prompt for the LLM
            user_prompt = f"""Task: {task}

Repository Path: {context.state.repository_path}
Branch: {context.state.branch_name}
Git Hash: {context.state.git_hash}"""

            # Add memory information to prompt if available
            if context.memory_state:
                user_prompt += f"""

Memory Repository: {memory_path}
Memory Hash: {memory_hash}"""
            
            # Execute the task
            result_json_str = self._execute_task_with_llm(user_prompt, context)
            
            # Parse the result JSON to get the actual success status
            execution_success = False
            error_message = "Unknown execution outcome"
            try:
                result_data = json.loads(result_json_str)
                execution_success = result_data.get("success", False)
                if not execution_success:
                     error_message = result_data.get("summary") or result_data.get("error", "Task execution failed internally")
            except json.JSONDecodeError as parse_error:
                logger.error(f"Failed to parse final result JSON from _execute_task_with_llm: {parse_error}")
                logger.error(f"Result string was: {result_json_str}")
                error_message = "Failed to parse internal execution result"
                execution_success = False # Ensure failure if JSON is bad

            # Get the current memory state if available
            current_memory_hash = None
            memory_repo_path = None
            
            if context.memory_state:
                memory_repo_path = getattr(context.memory_state, "repository_path", None)
            if not memory_repo_path:
                memory_repo_path = context.state.memory_repository_path
            
            if memory_repo_path:
                try:
                    current_memory_hash = get_current_hash(memory_repo_path)
                    logger.info(f"Updated memory hash: {current_memory_hash[:8]}")
                except Exception as e:
                    logger.warning(f"Failed to get current memory hash: {e}")
            
            # Create final state with updated memory hash
            final_state = State(
                git_hash=context.state.git_hash, # Note: This might be stale if commit failed
                repository_path=context.state.repository_path,
                description=context.state.description,
                branch_name=context.state.branch_name,
                memory_hash=current_memory_hash,
                memory_repository_path=memory_repo_path
            )
            
            # Return the result using the PARSED success status and include error message if failed
            return ExecutionResult(
                success=execution_success,
                branch_name=context.state.branch_name,
                git_hash=context.state.git_hash,
                repository_path=context.state.repository_path,
                final_state=final_state,
                error_message=None if execution_success else error_message
            )
            
        except Exception as e:
            # This outer exception block catches errors *before* _execute_task_with_llm finishes
            logger.error(f"High-level error executing task: {str(e)}", exc_info=True)

    def _execute_task_with_llm(self, user_prompt: str, context: TaskContext) -> str:
        """Execute a task using the LLM."""
        # Initialize messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Get memory repository path if available
        memory_repo_path = context.state.memory_repository_path if hasattr(context.state, "memory_repository_path") else None
        
        # Get memory hash if available
        memory_hash = context.state.memory_hash if hasattr(context.state, "memory_hash") else None
        if memory_hash:
            logger.info(f"Using memory hash from context state: {memory_hash[:8]}")
        
        # Reset conversation buffer for this new execution
        self.conversation_buffer = []
        
        # Add memory context as separate messages if available
        if memory_hash and memory_repo_path:
            try:
                # Use the new memory retrieval function
                from midpoint.agents.tools.memory_tools import retrieve_recent_memory
                
                # Get approximately 10000 characters of recent memory
                total_chars, memory_documents = retrieve_recent_memory(
                    memory_hash=memory_hash,
                    char_limit=10000,
                    repo_path=memory_repo_path
                )
                
                if memory_documents:
                    # Add each memory document as a separate message
                    for path, content, timestamp in memory_documents:
                        filename = os.path.basename(path)
                        timestamp_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        
                        messages.append({
                            "role": "system",
                            "content": f"Memory document from {filename} ({timestamp_str}):\n{content}"
                        })
                    
                    # Log memory context stats
                    logger.info(f"Added {len(memory_documents)} memory documents to conversation")
            except Exception as e:
                logger.error(f"Error retrieving memory context: {str(e)}")
        
        # Add the user prompt as the final message
        messages.append({"role": "user", "content": user_prompt})
        
        # Record user prompt in conversation buffer
        self.conversation_buffer.append({
            "type": "user_message",
            "user_prompt": user_prompt,
            "timestamp": datetime.datetime.now().isoformat()
        })
        
        # Track tool usage for metadata
        tool_usage = []
        
        try:
            # Get the task execution plan from the model
            message, tool_calls = self.tool_processor.run_llm_with_tools(
                messages,
                model="gpt-4o-mini",
                validate_json_format=True,
                max_tokens=3000
            )
            
            # Process tool calls and update tool usage
            if tool_calls:
                for tool_call in tool_calls:
                    tool_usage.append(tool_call)
                    
                    # Record tool usage in conversation buffer
                    self.conversation_buffer.append({
                        "type": "tool_usage",
                        "tool_name": tool_call.get("name", "unknown_tool"),
                        "tool_args": tool_call.get("arguments", {}),
                        "timestamp": datetime.datetime.now().isoformat()
                    })
            
            # Get the final assistant message content
            if isinstance(message, list):
                # If message is a list, get the last message's content
                final_content = message[-1].get('content', '')
            else:
                # If message is a dict or object, get its content
                final_content = message.get('content') if isinstance(message, dict) else message.content

            # Log the final raw response for debugging
            logger.debug(f"Final Raw model response: {final_content}")
                
            # Record final assistant response in conversation buffer
            self.conversation_buffer.append({
                "type": "assistant_response",
                "final_content": final_content,
                "timestamp": datetime.datetime.now().isoformat()
            })
            
            # --- Aligned Error Handling --- 
            output_data = None
            try:
                output_data = json.loads(final_content)
                logger.debug("Successfully parsed final response as JSON.")
            except json.JSONDecodeError as json_err:
                logger.warning(f"Failed to parse final response as JSON: {json_err}. Trying markdown extraction...")
                # Try to extract JSON from markdown code blocks if present (similar to GoalDecomposer)
                json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', final_content, re.DOTALL)
                if json_match:
                    try:
                        output_data = json.loads(json_match.group(1))
                        logger.info("Successfully extracted JSON from markdown code block.")
                    except json.JSONDecodeError as inner_err:
                        logger.error(f"Failed to parse extracted JSON: {inner_err}")
                        output_data = None # Ensure it's None if extraction parsing fails
                else:
                    logger.error("No JSON code block found for extraction.")
                    output_data = None # Ensure it's None if no block found
            
            # Check if parsing/extraction succeeded and required fields are present
            if output_data and all(key in output_data for key in ["summary", "success", "validation_steps"]):
                logger.info(f"✅ Task execution summary: {output_data['summary']}")
                if output_data["validation_steps"]:
                    logger.info("Validation steps:")
                    for i, step in enumerate(output_data["validation_steps"], 1):
                        logger.info(f"  {i}. {step}")
                # Save conversation BEFORE returning successfully
                self._save_interaction_to_memory(context) # Sync call
                return json.dumps(output_data)
            else:
                # Handle failure (JSON invalid or missing fields)
                error_reason = "LLM response missing required fields or invalid JSON"
                if not output_data:
                    error_reason = "Failed to parse LLM response as valid JSON"
                logger.error(f"Task failed: {error_reason}. Final content was: {final_content[:500]}...")
                default_response = {
                    "summary": error_reason,
                    "success": False,
                    "validation_steps": ["Task execution failed due to invalid LLM response format"],
                    "error": error_reason
                }
                # Save conversation BEFORE returning failure
                self._save_interaction_to_memory(context) # Sync call
                return json.dumps(default_response)
            
        except Exception as e:
            # Log the general error
            logger.error(f"Error during task execution: {str(e)}", exc_info=True) # Add traceback logging
            
            # Always attempt to save the conversation, even on error
            try:
                self._save_interaction_to_memory(context) # Sync call
            except Exception as save_error:
                logger.error(f"Failed to save conversation on error: {str(save_error)}")

    def is_git_ancestor(self, repo_path: str, ancestor_hash: str, descendant_hash: str) -> bool:
        """
        Check if one hash is an ancestor of another in a git repository.
        """
        try:
            # Use git merge-base --is-ancestor to check if one hash is an ancestor of another
            result = subprocess.run(
                ["git", "merge-base", "--is-ancestor", ancestor_hash, descendant_hash],
                cwd=repo_path,
                capture_output=True,
                check=False
            )
            
            # Return True if exit_code is 0, indicating ancestor relationship
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Error checking git ancestry: {str(e)}")
            return False 