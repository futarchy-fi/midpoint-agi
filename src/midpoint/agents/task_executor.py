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
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import re
import os
import logging
from pathlib import Path
import subprocess
import traceback

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
        Save the conversation buffer to memory.
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
            
            # Format conversation content from the buffer
            content = f"# Task Execution Conversation\n\nTimestamp: {timestamp}\n\n"
            
            # Add each entry from the conversation buffer based on role
            for entry in self.conversation_buffer:
                role = entry.get("role", "unknown")
                entry_content = entry.get("content", "") # Default to empty string if content is None or missing
                if entry_content is None: # Ensure entry_content is never None for formatting
                    entry_content = "" 

                if role == "user":
                    content += f"## User Message\n\n{entry_content}\n\n"
                elif role == "assistant":
                    # Check for tool calls within the assistant message
                    tool_calls = entry.get("tool_calls")
                    if tool_calls and isinstance(tool_calls, list):
                         content += f"## Assistant Action (Tool Call)\n\n"
                         if entry_content: # Include any text assistant said before the call
                              content += f"{entry_content}\n\n" 
                         for i, tc in enumerate(tool_calls):
                              func = tc.get("function", {})
                              tool_name = func.get("name", "unknown_tool")
                              tool_args = func.get("arguments", "{}")
                              content += f"**Tool Call {i+1}:** `{tool_name}`\n"
                              # Format arguments nicely, maybe as a code block
                              try:
                                   # Attempt to parse and pretty-print JSON arguments
                                   parsed_args = json.loads(tool_args)
                                   formatted_args = json.dumps(parsed_args, indent=2)
                                   content += f"```json\n{formatted_args}\n```\n"
                              except json.JSONDecodeError:
                                   # Fallback for non-JSON or malformed arguments
                                   content += f"```\n{tool_args}\n```\n"
                         content += "\n" # Add newline after tool calls section
                    else:
                         # Regular assistant text response
                         content += f"## Assistant Response\n\n{entry_content}\n\n"
                elif role == "tool":
                    tool_name = entry.get("name", "unknown_tool")
                    tool_call_id = entry.get("tool_call_id", "unknown_id")
                    content += f"## Tool Result (`{tool_name}`, ID: {tool_call_id})\n\n"
                    content += f"```\n{entry_content}\n```\n\n" # Put tool result in code block
                elif role == "system" and "[Error serializing message:" in entry_content:
                     # Handle the placeholder error message we added earlier
                     content += f"## System Error\n\n{entry_content}\n\n"
                # Note: We generally skip system messages unless it's our error placeholder

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
            # Use provided task description if available, otherwise use goal description
            task = task_description or context.goal.description
            
            # Log task information
            logger.info(f"Executing task: {task}")
            logger.info(f"Repository: {context.state.repository_path}")
            logger.info(f"Branch: {context.state.branch_name}")
            
            # Log memory state if available
            if context.memory_state:
                logger.info(f"Memory repository: {context.state.memory_repository_path}")
                if context.state.memory_hash:
                    logger.info(f"Initial memory hash: {context.state.memory_hash[:8]}")
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

Memory Repository: {context.state.memory_repository_path}
Memory Hash: {context.state.memory_hash}"""
            
            # Execute the task
            result_json_str, tool_usage_list = self._execute_task_with_llm(user_prompt, context)
            
            # Parse the result JSON to get the actual success status and summary
            execution_success = False
            summary = "Task execution failed internally."
            validation_steps = []
            error_message = "Unknown execution outcome" # Default error
            final_git_hash = context.state.git_hash # Use initial hash as fallback

            try:
                result_data = json.loads(result_json_str)
                execution_success = result_data.get("success", False)
                summary = result_data.get("summary", "No summary provided.")
                # Ensure validation_steps is always a list
                validation_steps = result_data.get("validation_steps") 
                if validation_steps is None: # Handle null case
                   validation_steps = []
                elif not isinstance(validation_steps, list):
                    # Attempt to handle non-list (e.g., string) - might need refinement
                    validation_steps = [str(validation_steps)] 
                    logger.warning("Validation steps were not a list, converted to list of strings.")
                
                # Get the final git hash reported by the LLM, if any
                # This might represent the hash *after* a successful commit within the LLM flow
                final_git_hash = result_data.get("git_hash") or context.state.git_hash

                if not execution_success:
                    # Use summary as error message if specifically marked as failure
                    error_message = summary 
                else:
                    error_message = None # Clear error if success is true

            except json.JSONDecodeError as parse_error:
                logger.error(f"Failed to parse final result JSON from _execute_task_with_llm: {parse_error}")
                logger.error(f"Result string was: {result_json_str}")
                summary = "Failed to parse LLM response JSON."
                error_message = summary # Use parse failure as error message
                execution_success = False # Ensure failure if JSON is bad
                validation_steps = ["Failed to parse LLM response"] # Indicate failure reason
            except Exception as e:
                # Catch other potential errors during parsing/processing result_data
                logger.error(f"Error processing LLM result data: {e}", exc_info=True)
                summary = f"Error processing LLM result: {str(e)}"
                error_message = summary
                execution_success = False
                validation_steps = ["Error processing LLM result"]

            # Get the current memory state if available
            current_memory_hash = None
            memory_repo_path = None
            
            # Determine memory path consistently
            if hasattr(context, 'memory_state') and context.memory_state:
                memory_repo_path = getattr(context.memory_state, "repository_path", None)
            if not memory_repo_path:
                memory_repo_path = context.state.memory_repository_path

            # Get final memory hash
            if memory_repo_path:
                try:
                    current_memory_hash = get_current_hash(memory_repo_path)
                    logger.info(f"Final memory hash after execution: {current_memory_hash[:8]}")
                except Exception as e:
                    logger.warning(f"Failed to get final memory hash: {e}")
                    # Use initial memory hash as fallback if final fetch fails
                    current_memory_hash = context.state.memory_hash
            else:
                # If no memory repo path, use initial hash
                current_memory_hash = context.state.memory_hash
            
            # Create final state using potentially updated hashes
            final_state = State(
                git_hash=final_git_hash, # Use hash reported by LLM or initial if missing
                repository_path=context.state.repository_path,
                description="State after task execution attempt", # Generic description
                branch_name=context.state.branch_name,
                memory_hash=current_memory_hash, # Use the hash obtained after execution attempt
                memory_repository_path=memory_repo_path
            )
            
            # Return the result using the PARSED success status and include error message if failed
            # Populate the new summary and validation_steps fields
            result_metadata = {"tool_usage": tool_usage_list}
            return ExecutionResult(
                success=execution_success,
                summary=summary, # Pass the summary from LLM
                suggested_validation_steps=validation_steps, # Pass the validation steps
                branch_name=context.state.branch_name,
                git_hash=final_git_hash, # Pass the potentially updated git hash
                repository_path=context.state.repository_path,
                final_state=final_state, # Include the final state object
                error_message=error_message, # Use derived error message
                metadata=result_metadata # Add the metadata field
            )
            
        except Exception as e:
            # This outer exception block catches errors *before* _execute_task_with_llm finishes
            logger.error(f"High-level error executing task: {str(e)}", exc_info=True)
            # === EDIT: Return failure ExecutionResult with metadata if possible ===
            # We might not have tool_usage if the error happened before _execute_task_with_llm finished
            # Default to empty list if tool_usage_list is not defined in this scope
            error_tool_usage = locals().get("tool_usage_list", [])
            error_metadata = {"tool_usage": error_tool_usage, "error_details": traceback.format_exc()}

            # Try to construct a final state, may fail if context is incomplete
            final_state = None
            try:
                current_memory_hash = context.state.memory_hash # Fallback
                memory_repo_path = context.state.memory_repository_path # Fallback
                if memory_repo_path:
                     current_memory_hash = get_current_hash(memory_repo_path) or context.state.memory_hash
                final_state = State(
                    git_hash=context.state.git_hash, # Use initial hash
                    repository_path=context.state.repository_path,
                    description="State after high-level task execution error",
                    branch_name=context.state.branch_name,
                    memory_hash=current_memory_hash,
                    memory_repository_path=memory_repo_path
                )
            except Exception as state_err:
                 logger.error(f"Could not determine final state during high-level error handling: {state_err}")

            return ExecutionResult(
                success=False,
                summary=f"High-level error during task execution: {str(e)}",
                suggested_validation_steps=[],
                branch_name=context.state.branch_name if context and context.state else "unknown",
                git_hash=context.state.git_hash if context and context.state else "unknown",
                repository_path=context.state.repository_path if context and context.state else "unknown",
                final_state=final_state, # May be None
                error_message=str(e),
                metadata=error_metadata
            )
            # === END EDIT ===

    def _execute_task_with_llm(self, user_prompt: str, context: TaskContext) -> Tuple[str, List[Any]]:
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
        
        # Track tool usage for metadata
        tool_usage = []
        
        try:
            # Get the task execution plan from the model
            final_messages, tool_calls = self.tool_processor.run_llm_with_tools(
                messages,
                model="gpt-4o-mini",
                validate_json_format=True,
                max_tokens=3000
            )
            
            # === POPULATE conversation_buffer FROM final_messages ===
            # Ensure final_messages is a list
            if not isinstance(final_messages, list):
                logger.error(f"ToolProcessor did not return a list of messages: {type(final_messages)}")
                # Handle error case appropriately, maybe raise or return default error
                final_messages = [] # Default to empty list to prevent downstream errors

            # Clear and repopulate buffer from the actual final conversation history
            self.conversation_buffer = []
            # Skip initial system message(s) when adding to buffer for saving
            system_message_count = sum(1 for msg in messages if msg.get('role') == 'system')
            relevant_messages = final_messages[system_message_count:] # Get messages after system prompts
            
            # Convert message objects/dicts to a serializable format for the buffer
            for msg in relevant_messages:
                if isinstance(msg, dict):
                    # Already a dictionary, ensure content is present
                    if 'content' not in msg:
                         msg['content'] = None # Ensure content key exists, even if null
                    self.conversation_buffer.append(msg)
                else:
                    # Attempt to convert potential message objects (like from OpenAI response)
                    try:
                        msg_dict = {
                            "role": getattr(msg, 'role', 'unknown'),
                            "content": getattr(msg, 'content', None),
                        }
                        # Handle tool calls if present on the object
                        if hasattr(msg, 'tool_calls') and msg.tool_calls:
                            msg_dict['tool_calls'] = [
                                {
                                    "id": tc.id,
                                    "type": tc.type,
                                    "function": {
                                        "name": tc.function.name,
                                        "arguments": tc.function.arguments
                                    }
                                } for tc in msg.tool_calls
                            ]
                        # Handle tool call ID if present (for tool role messages)
                        if hasattr(msg, 'tool_call_id'):
                             msg_dict['tool_call_id'] = msg.tool_call_id
                        # Handle tool name if present (for tool role messages)
                        if hasattr(msg, 'name'):
                             msg_dict['name'] = msg.name
                             
                        self.conversation_buffer.append(msg_dict)
                    except Exception as e:
                        logger.error(f"Could not serialize message object for buffer: {e} - {msg}")
                        # Append a placeholder if serialization fails
                        self.conversation_buffer.append({"role": "system", "content": f"[Error serializing message: {e}]"})

            # Process tool calls and update tool usage (for ExecutionResult metadata)
            if tool_calls:
                logger.debug(f"Processing tool_calls: {tool_calls}") # DEBUG: Log the raw tool_calls list
                for tool_call in tool_calls:
                    tool_usage.append(tool_call) # Keep populating tool_usage for metadata
            
            # Get the final assistant message content (for parsing the final JSON response)
            # Find the *last* message from the assistant in the actual final_messages list
            final_assistant_message = next((msg for msg in reversed(final_messages) if isinstance(msg, dict) and msg.get("role") == "assistant"), None)
            
            if final_assistant_message:
                final_content = final_assistant_message.get('content', '')
            else:
                 # Handle case where no final assistant message content is found
                 logger.warning("No final assistant message content found in the conversation history.")
                 final_content = "{}" # Default to empty JSON to avoid errors

            # Log the final raw response for debugging
            logger.debug(f"Final Raw model response (for JSON parsing): {final_content}")
            
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
                return final_content, tool_usage
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
                return json.dumps(default_response), tool_usage
            
        except Exception as e:
            # Log the general error
            logger.error(f"Error during task execution: {str(e)}", exc_info=True) # Add traceback logging
            
            # Always attempt to save the conversation, even on error
            try:
                self._save_interaction_to_memory(context) # Sync call
            except Exception as save_error:
                logger.error(f"Failed to save conversation on error: {str(save_error)}")

            # Return a default error JSON structure if an exception occurs
            default_error_response = {
                 "summary": f"Task failed with exception: {str(e)}",
                 "success": False,
                 "validation_steps": [f"Task execution failed due to exception: {str(e)}"],
                 "error": str(e)
            }
            return json.dumps(default_error_response), tool_usage

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