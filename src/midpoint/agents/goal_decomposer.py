"""
Goal Decomposition agent implementation.

This module implements the GoalDecomposer agent that determines the next step
toward achieving a complex goal.
"""

# Fix the memory tools imports - this must be at the top of the file
import sys
import os
from pathlib import Path
import logging
import traceback
import re

# Early initialization of logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Set up the script location and repository root
script_location = Path(__file__).resolve()
repo_root = script_location.parent.parent.parent.parent

# Add repo root and scripts directory to path
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "scripts"))

# Debug information
if os.environ.get("DEBUG"):
    print(f"Script location: {script_location}")
    print(f"Repository root: {repo_root}")
    print(f"Python path: {sys.path}")
    scripts_dir = repo_root / "scripts"
    print(f"Scripts directory exists: {scripts_dir.exists()}")
    memory_tools_file = scripts_dir / "memory_tools.py"
    print(f"Memory tools file exists: {memory_tools_file.exists()}")

# Import the memory tools functions directly
try:
    # Try direct import first
    import memory_tools
    
    # Create direct references to the functions we need (except get_current_hash)
    get_repo_path = memory_tools.get_repo_path
    store_document = memory_tools.store_document
    retrieve_documents = memory_tools.retrieve_documents
    update_cross_reference = memory_tools.update_cross_reference
    
    # Log success
    logging.debug("Successfully imported memory_tools directly")
except Exception as e:
    logging.warning(f"Memory tools import failed. Using fallback implementations. Error: {e}")
    if os.environ.get("DEBUG"):
        print("Exception traceback:")
        traceback.print_exc()
    
    # Define fallback implementations
    def get_repo_path():
        """Fallback implementation to get memory repository path."""
        if os.environ.get("DEBUG"):
            print("Using fallback get_repo_path")
        return os.environ.get("MEMORY_REPO_PATH", os.path.expanduser("~/.midpoint/memory"))
    
    def store_document(content, category, metadata=None, repo_path=None):
        """Fallback implementation to store a document."""
        logging.warning("Using fallback store_document implementation.")
        # Get repository path
        repo_path = repo_path or get_repo_path()
        repo_path = Path(repo_path)
        
        # Create basic directories
        docs_dir = repo_path / "documents" / category
        docs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a simple filename
        filename = f"doc_{int(time.time())}.md"
        doc_path = docs_dir / filename
        
        # Write content
        with open(doc_path, "w") as f:
            f.write(content)
            
        logging.info(f"Stored document at: {doc_path} (fallback implementation)")
        return str(doc_path.relative_to(repo_path) if repo_path in doc_path.parents else doc_path)
        
    def retrieve_documents(category=None, limit=10, repo_path=None):
        """Fallback implementation to retrieve documents."""
        logging.warning("Using fallback retrieve_documents implementation.")
        # Get repository path
        repo_path = repo_path or get_repo_path()
        repo_path = Path(repo_path)
        
        # Set search path
        if category:
            search_path = repo_path / "documents" / category
        else:
            search_path = repo_path / "documents"
        
        results = []
        
        if search_path.exists():
            # Find all .md files
            files = list(search_path.glob("**/*.md"))
            
            # Sort by modification time (newest first)
            files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Limit results
            files = files[:limit]
            
            # Read contents
            for file in files:
                try:
                    with open(file, "r") as f:
                        results.append((str(file.relative_to(repo_path) if repo_path in file.parents else file), f.read()))
                except:
                    pass
        
        return results
    
    def update_cross_reference(code_hash, memory_hash, repo_path=None):
        """Fallback implementation to update cross-reference between code and memory."""
        logging.warning("Using fallback update_cross_reference implementation.")
        # Get repository path
        repo_path = repo_path or get_repo_path()
        repo_path = Path(repo_path)
        
        # Create cross-reference directory
        xref_dir = repo_path / "references"
        xref_dir.mkdir(parents=True, exist_ok=True)
        
        # Create file for cross-reference
        xref_path = xref_dir / f"{code_hash}.txt"
        
        # Write memory hash to the file
        with open(xref_path, "w") as f:
            f.write(memory_hash)
            
        logging.info(f"Updated cross-reference at: {xref_path} (fallback implementation)")
        return True

# Now import the rest of the modules needed
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext
from midpoint.agents.tools import initialize_all_tools
from midpoint.agents.tools.processor import ToolProcessor
from midpoint.agents.tools.registry import ToolRegistry
from midpoint.agents.tools import (
    list_directory,
    read_file,
    search_code,
    get_current_hash,
    web_search,
    web_scrape,
    store_memory_document,
    retrieve_memory_documents,
)
from midpoint.agents.config import get_openai_api_key
from midpoint.utils.logging import log_manager
from dotenv import load_dotenv
import datetime
import subprocess
import time

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
                # List of patterns to hide from console output
                hide_patterns = [
                    'Added relevant context from input file',
                    'Using next_step from input file',
                    'Validation criteria:',
                    'validation_criteria',
                    'Requires further decomposition',
                    'relevant_context',
                    'Determining next step for goal',
                    '===',
                    '====================',
                    '\n====================',
                    'NEXT STEP',
                    'VALIDATION CRITERIA',
                    'REASONING',
                    'REQUIRES FURTHER DECOMPOSITION',
                    'RELEVANT CONTEXT',
                    '✅ Next step:',    # Hide the old next step format
                ]
                
                # Check if any hide patterns are in the message
                for pattern in hide_patterns:
                    if pattern in record.msg:
                        return False
                
                # Also hide the individual validation criteria lines from the default output
                if record.msg.startswith('  ') and any(c.isdigit() for c in record.msg) and '. ' in record.msg:
                    return False
                
                # Make emojis and messages more concise
                if '📂 Listing directory:' in record.msg:
                    record.msg = record.msg.replace('📂 Listing directory:', '📂')
                elif '📄 Reading:' in record.msg:
                    record.msg = record.msg.replace('📄 Reading:', '📄')
                elif '🔍 Searching code:' in record.msg:
                    record.msg = record.msg.replace('🔍 Searching code:', '🔍 Searching for pattern:')
                elif '🤖 API call completed' in record.msg:
                    return False  # Don't show API calls in console
                elif '✅ Next step:' in record.msg:
                    # Show this message only in standalone mode
                    if 'main' not in sys._getframe().f_back.f_code.co_name:
                        return True
                    return False  # Don't show in main() since we have better formatting there
                # Allow our new emoji formats to pass through
                elif any(emoji in record.msg for emoji in ['📂', '📄', '🔍', '🌐', '💾', '🔄', '🔗', '✅', '📝', '➕', '➖', '🔀']):
                    return True
                elif ('🔄 Next subgoal:' in record.msg) or ('✅ Next task:' in record.msg):
                    return True
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
    """
    Goal decomposition agent implementation.
    
    This agent is responsible for determining the next step toward achieving a complex goal.
    """
    
    def __init__(self, model: str = "gpt-4", max_history_entries: int = 5):
        """
        Initialize the goal decomposer.
        
        Args:
            model: The model to use for generation
            max_history_entries: Maximum number of history entries to consider
        """
        self.logger = logging.getLogger('GoalDecomposer')
        self.model = model
        self.max_history_entries = max_history_entries
        
        # Get API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Initialize tool processor
        initialize_all_tools()
        self.tool_processor = ToolProcessor(self.client)
        
        # Get tools from registry
        self.tools = ToolRegistry.get_tool_schemas()
        
        # Generate system prompt with dynamic tool descriptions
        self.system_prompt = self._generate_system_prompt()
        
    async def _save_interaction_to_memory(self, 
                                        interaction_type: str, 
                                        content: str, 
                                        metadata: Dict[str, Any] = None, 
                                        memory_repo_path: str = None) -> Dict[str, Any]:
        """
        Save an interaction to the memory system.
        
        Args:
            interaction_type: Type of interaction (e.g., 'conversation', 'reasoning', 'tool_use', 'final_output')
            content: Content of the interaction
            metadata: Additional metadata about the interaction (must include memory_hash)
            memory_repo_path: Path to the memory repository (optional)
            
        Returns:
            Result of the memory storage operation
        """
        # Create a timestamped filename
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save directly to general_memory folder without creating subfolders
        category = "../general_memory"
        
        # Create metadata if not provided
        if metadata is None:
            metadata = {}
        
        # Add timestamp and interaction type to metadata
        metadata["timestamp"] = timestamp
        metadata["interaction_type"] = interaction_type
        metadata["agent_type"] = "goal_decomposer"
        
        # Create a custom filename with format: TIMESTAMP_goal_decomposer_TYPE.md
        filename = f"{timestamp}_goal_decomposer_{interaction_type}.md"
        
        # Set the id field in metadata (used by store_document to create the filename)
        metadata["id"] = f"{timestamp}_goal_decomposer_{interaction_type}"
        
        # Extract the memory hash from metadata - this is now REQUIRED
        memory_hash = metadata.get("memory_hash")
        if not memory_hash:
            logging.error(f"Cannot save to memory: memory_hash is required in metadata")
            return {"success": False, "error": "memory_hash is required in metadata"}
        
        logging.info(f"Saving to memory with hash: {memory_hash}")
        
        # Store the document
        try:
            result = await store_memory_document(
                content=content,
                category=category,
                metadata=metadata,
                memory_repo_path=memory_repo_path,
                memory_hash=memory_hash  # Explicitly pass the memory_hash
            )
            # Log more detailed information about the memory result
            logging.info(f"Saved {interaction_type} to memory: {result}")
            
            # Handle the nested document_path structure if present
            doc_path_info = result.get("document_path", {})
            if isinstance(doc_path_info, dict):
                logging.info(f"Memory details from document_path - path: {doc_path_info.get('path')}, hash: {doc_path_info.get('memory_hash')}, category: {doc_path_info.get('category')}, filename: {doc_path_info.get('filename')}")
            else:
                logging.info(f"Memory details (direct) - path: {result.get('path')}, hash: {result.get('memory_hash')}, category: {result.get('category')}, filename: {result.get('filename')}")
            
            return result
        except Exception as e:
            logging.error(f"Failed to save {interaction_type} to memory: {str(e)}")
            return {"success": False, "error": str(e)}

    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt that includes dynamically generated tool descriptions.
        
        Returns:
            The system prompt with tool descriptions.
        """
        # Generate list of available tools and their descriptions
        tool_descriptions = []
        for tool_schema in self.tools:
            if tool_schema['type'] == 'function' and 'function' in tool_schema:
                function = tool_schema['function']
                name = function.get('name', '')
                description = function.get('description', '')
                tool_descriptions.append(f"- {name}: {description}")
        
        # Join the tool descriptions with newlines
        tool_descriptions_text = "\n".join(tool_descriptions)
        
        # Create the system prompt with the tool descriptions
        system_prompt = f"""You are an expert software architect and project planner.
Your task is to determine the SINGLE NEXT STEP toward a complex software development goal.
Follow the OODA loop: Observe, Orient, Decide, Act.

1. OBSERVE: Explore the repository to understand its current state. Use the available tools.
2. ORIENT: Analyze how the current state relates to the final goal.
3. DECIDE: Determine the most promising SINGLE NEXT STEP.
4. OUTPUT: Provide a structured response with the next step and validation criteria.

For complex goals, consider if the best next step is exploration, research, or a "study session" 
rather than immediately jumping to implementation.

You have access to a memory repository where you can store and retrieve information across sessions.
For intellectual tasks (such as studying, analyzing, or understanding code), you should consider
updating the memory repository as a valid next step. Tasks are considered "done" when the memory
has been properly updated, even if no code changes were made.

Memory categories:
- reasoning: Documents capturing your reasoning process
- observations: Documents recording observations about the codebase
- decisions: Documents recording decisions made
- study: Documents capturing in-depth study of code or concepts

As part of your analysis, you MUST determine whether the next step requires further decomposition:
- If the next step is still complex and would benefit from being broken down further, set 'requires_further_decomposition' to TRUE
- If the next step is simple enough to be directly implemented by a TaskExecutor agent, set 'requires_further_decomposition' to FALSE

Also identify any relevant context that should be passed to child subgoals:
- Include ONLY information that will be directly helpful for understanding and implementing the subgoal
- DO NOT include high-level strategic information that isn't directly relevant to the subgoal
- Structure this as key-value pairs in the 'relevant_context' field

You have access to these tools:
{tool_descriptions_text}

You MUST provide a structured response in JSON format with these fields:
- next_step: A clear description of the single next step to take
- validation_criteria: List of measurable criteria to validate this step's completion
- reasoning: Explanation of why this is the most promising next action
- requires_further_decomposition: Boolean indicating if this step needs further breakdown (true) or can be directly executed (false)
- relevant_context: Object containing relevant information to pass to child subgoals

IMPORTANT: Return ONLY raw JSON without any markdown formatting or code blocks. Do not wrap the JSON in ```json ... ``` or any other formatting."""
        
        return system_prompt

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
        
        # Get memory repository path if available
        memory_repo_path = context.state.memory_repository_path if hasattr(context.state, "memory_repository_path") else None
        
        # Get memory hash if available - we'll use this to ensure we save to the correct memory state
        memory_hash = context.state.memory_hash if hasattr(context.state, "memory_hash") else None
        if memory_hash:
            logging.info(f"Using existing memory hash from context state: {memory_hash}")
        
        # Initialize messages
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # Track tool usage for metadata
        tool_usage = []
        
        try:
            # Get the next step from the model
            message, tool_calls = await self.tool_processor.run_llm_with_tools(
                messages,
                model=self.model,
                validate_json_format=True
            )
            
            # Process tool calls and update tool usage
            if tool_calls:
                for tool_call in tool_calls:
                    tool_usage.append(tool_call)
            
            # Parse the model's response
            try:
                content = message.get('content') if isinstance(message, dict) else message.content
                output_data = json.loads(content)
            except json.JSONDecodeError:
                logging.error("Failed to parse model response as JSON")
                raise ValueError("Model response is not valid JSON")
            
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
                        "raw_response": content,  # Use the content we already extracted
                        "tool_usage": tool_usage
                    }
                )
            else:
                # Ask for a properly formatted response
                messages.append({
                    "role": "user",
                    "content": "Please provide a valid JSON response with the fields: next_step, validation_criteria, reasoning, requires_further_decomposition, and relevant_context."
                })
            
            # Validate the subgoal plan
            self._validate_subgoal(final_output, context)
            
            # Log tool usage at debug level
            logging.debug("Tool usage: %s", tool_usage)

            # Add logging for API call details - use serialized messages at debug level
            try:
                serialized_messages = self._serialize_messages(messages)
                logging.debug("API call history: %s", json.dumps(serialized_messages, indent=2))
                
                # Save only the complete conversation to memory
                await self._save_interaction_to_memory(
                    interaction_type="conversation",
                    content=json.dumps(serialized_messages, indent=2),
                    metadata={
                        "goal": context.goal.description, 
                        "message_count": len(serialized_messages),
                        "next_step": final_output.next_step,
                        "requires_further_decomposition": final_output.requires_further_decomposition,
                        "memory_hash": memory_hash  # Pass the memory hash to ensure we save to the correct state
                    },
                    memory_repo_path=memory_repo_path
                )
            except Exception as e:
                logging.error("Failed to serialize messages for logging: %s", str(e))

            # Log successful outcome at info level with a more complete message
            logging.info(f"✅ Next step: {final_output.next_step}")
            
            # Only log detailed validation criteria at debug level to avoid duplication
            logging.debug("Validation criteria:")
            for i, criterion in enumerate(final_output.validation_criteria, 1):
                logging.debug(f"  {i}. {criterion}")
            
            logging.debug(f"Requires further decomposition: {final_output.requires_further_decomposition}")
            
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

            # Prepare and store the conversation in memory
            try:
                # Prepare the conversation for memory
                conversation_data = {
                    "system": self.system_prompt,
                    "user": user_prompt,
                    "assistant": content,
                    "tool_usage": tool_usage
                }
                
                # Save the conversation to memory
                memory_result = await self._save_interaction_to_memory(
                    interaction_type="conversation",
                    content=json.dumps(conversation_data, indent=2),
                    metadata={
                        "timestamp": int(time.time()),
                        "memory_hash": memory_hash  # Pass the memory hash to ensure we save to the correct state
                    },
                    memory_repo_path=memory_repo_path
                )
                
                # Update context state with memory information if available
                if memory_result and memory_result.get("success"):
                    logging.info(f"Saved conversation to memory: {memory_result}")
                    
                    # Handle the case where document_path is a dictionary
                    doc_path_info = memory_result.get("document_path", {})
                    if isinstance(doc_path_info, dict):
                        # Update memory hash from the document_path dictionary
                        if "memory_hash" in doc_path_info:
                            context.state.memory_hash = doc_path_info.get("memory_hash")
                        
                        # Extract filename from path to create a memory reference ID
                        if "filename" in doc_path_info:
                            # Use the filename as a clean memory reference
                            context.state.memory_reference = doc_path_info.get("filename")
                        elif "path" in doc_path_info:
                            # Extract just the filename from the path as fallback
                            from pathlib import Path
                            path = doc_path_info.get("path")
                            context.state.memory_reference = Path(path).name
                            # Still store the full path in memory_document_path for backward compatibility
                            context.state.memory_document_path = path
                    # Handle the case where the fields are directly in memory_result
                    else:
                        # Update memory hash if available at top level
                        if "memory_hash" in memory_result:
                            context.state.memory_hash = memory_result.get("memory_hash")
                        
                        # Extract filename from path to create a memory reference ID
                        if "path" in memory_result:
                            from pathlib import Path
                            path = memory_result.get("path")
                            context.state.memory_reference = Path(path).name
                            # Still store the full path for backward compatibility
                            context.state.memory_document_path = path
                        
                        # Set memory timestamp
                        context.state.memory_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            except Exception as e:
                logging.error(f"Failed to save conversation to memory: {str(e)}")
                
            return final_output
            
        except Exception as e:
            # Log the error to memory
            await self._save_interaction_to_memory(
                interaction_type="error",
                content=f"Error in determine_next_step: {str(e)}\n\n{traceback.format_exc()}",
                metadata={
                    "error_type": type(e).__name__, 
                    "goal": context.goal.description,
                    "memory_hash": memory_hash  # Pass the memory hash to ensure we save to the correct state
                },
                memory_repo_path=memory_repo_path
            )
            # Let the main function handle the specific error types
            raise
    
    def _create_user_prompt(self, context: TaskContext) -> str:
        """Create the user prompt for the agent."""
        prompt = f"""Goal: {context.goal.description}

Validation Criteria for Final Goal:
{chr(10).join(f"- {criterion}" for criterion in context.goal.validation_criteria)}

Current State:
- Git Hash: {context.state.git_hash}
- Description: {context.state.description}
- Repository Path: {context.state.repository_path}
"""

        # Add memory information if available
        if context.state.memory_hash and context.state.memory_repository_path:
            prompt += f"""
Memory State:
- Memory Hash: {context.state.memory_hash}
- Memory Repository Path: {context.state.memory_repository_path}
"""
        elif context.memory_state:
            prompt += f"""
Memory State:
- Memory Hash: {context.memory_state.memory_hash}
- Memory Repository Path: {context.memory_state.repository_path}
"""

        prompt += f"""
Context:
- Iteration: {context.iteration}
- Previous Steps: {len(context.execution_history) if context.execution_history else 0}

Your task is to explore the repository and determine the SINGLE NEXT STEP toward achieving the goal.
Focus on providing a clear next step with measurable validation criteria.

NOTE: For intellectual tasks that involve studying or understanding code rather than modifying it,
you can consider updating the memory repository as a valid next step. Memory operations are
appropriate when the goal involves gaining knowledge or understanding without changing code.
"""
        return prompt
    
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

    def generate_goal_id(self, parent_id=None, logs_dir="logs", is_task=False):
        """
        Generate a goal ID in format G1, S1, or T1
        
        NOTE: This method is maintained for backward compatibility and testing.
        In production, goal_cli.py is responsible for ID generation.
        """
        logging.warning("Using goal_decomposer.generate_goal_id which is deprecated. Goal IDs should be generated by goal_cli.py")
        logs_path = Path(logs_dir)
        
        # If a parent_id is provided, this is a subgoal or task
        if parent_id:
            if is_task:
                # Find next available task number
                max_num = 0
                for file_path in logs_path.glob("T*.json"):
                    # Match only files with pattern T followed by digits and .json
                    match = re.match(r"T(\d+)\.json$", file_path.name)
                    if match:
                        num = int(match.group(1))
                        max_num = max(max_num, num)
                
                # Next task number is one more than the maximum found
                next_num = max_num + 1
                return f"T{next_num}"
            else:
                # Find next available subgoal number
                max_num = 0
                for file_path in logs_path.glob("S*.json"):
                    # Match only files with pattern S followed by digits and .json
                    match = re.match(r"S(\d+)\.json$", file_path.name)
                    if match:
                        num = int(match.group(1))
                        max_num = max(max_num, num)
                
                # Next subgoal number is one more than the maximum found
                next_num = max_num + 1
                return f"S{next_num}"
        else:
            # For new top-level goals, find next available number
            max_num = 0
            for file_path in logs_path.glob("G*.json"):
                # Match only files with pattern G followed by digits and .json
                match = re.match(r"G(\d+)\.json$", file_path.name)
                if match:
                    num = int(match.group(1))
                    max_num = max(max_num, num)
            
            # Next goal number is one more than the maximum found
            next_num = max_num + 1
            return f"G{next_num}"

    def list_subgoal_files(self, logs_dir="logs"):
        """
        List all subgoal files in the logs directory.
        
        NOTE: This method is maintained for backward compatibility and testing.
        In production, goal files should be managed by goal_cli.py.
        """
        logging.warning("Using GoalDecomposer.list_subgoal_files which is deprecated. Goal files should be managed by goal_cli.py")
        return list_subgoal_files(logs_dir)

    def create_top_goal_file(self, context: TaskContext, logs_dir="logs") -> str:
        """
        [DEPRECATED] Create a subgoal file for a top-level goal.
        
        This function is deprecated. Goal file creation should now be handled
        by goal_cli.py.
        
        Args:
            context: The current task context containing the goal
            logs_dir: Directory to store the goal file
            
        Returns:
            The filename of the created subgoal file
        """
        logging.warning("Using GoalDecomposer.create_top_goal_file which is deprecated. Goal files should be created by goal_cli.py")
        
        # Use goal_id from context metadata if available, otherwise generate one
        if hasattr(context, 'metadata') and context.metadata and 'goal_id' in context.metadata:
            goal_id = context.metadata['goal_id']
        else:
            # Generate a goal ID
            goal_id = self.generate_goal_id(logs_dir=logs_dir)
        
        # Create the file for test purposes
        filename = f"{goal_id}.json"
        file_path = Path(logs_dir) / filename
        
        # Create a simple goal data structure
        goal_data = {
            "goal_id": goal_id,
            "parent_goal": "",
            "description": context.goal.description,
            "next_step": context.goal.description,
            "validation_criteria": context.goal.validation_criteria,
            "requires_further_decomposition": True,
            "iteration": context.iteration,
            "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        }
        
        # Save the file
        with open(file_path, 'w') as f:
            json.dump(goal_data, f, indent=2)
        
        return filename

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
        process = await asyncio.create_subprocess_exec(
            "git", "status", "--porcelain",
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise ValueError(f"Failed to check git status: {stderr.decode()}")
            
        if stdout.decode().strip():
            raise ValueError(f"Repository has uncommitted changes: {repo_path}")
    except Exception as e:
        logging.error("Failed to check git status: %s", str(e))
        raise ValueError(f"Failed to check git status: {str(e)}")
    
    # If git_hash is provided, check that it matches the current hash
    if git_hash:
        try:
            current_hash = await get_current_hash(repo_path)
            if current_hash != git_hash:
                raise ValueError(f"Repository hash mismatch: expected {git_hash}, got {current_hash}")
        except Exception as e:
            logging.warning(f"Failed to validate repository hash: {str(e)}")
            raise ValueError(f"Failed to validate repository hash: {str(e)}")

def list_subgoal_files(logs_dir="logs"):
    """
    List all subgoal files in the logs directory.
    
    NOTE: This method is maintained for backward compatibility and testing.
    In production, goal files should be managed by goal_cli.py.
    """
    logging.warning("Using goal_decomposer.list_subgoal_files which is deprecated. Goal files should be managed by goal_cli.py")
    if not os.path.exists(logs_dir):
        return []
        
    subgoal_files = []
    for filename in os.listdir(logs_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(logs_dir, filename)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", "")
                    next_step = data.get("next_step", "")
                    parent_goal = data.get("parent_goal", "")
                    goal_id = data.get("goal_id", "")
                    subgoal_files.append((file_path, timestamp, next_step, parent_goal, goal_id))
            except:
                continue
                
    # Sort by timestamp if available
    subgoal_files.sort(key=lambda x: x[1] if x[1] else "", reverse=True)
    return subgoal_files

async def load_input_file(input_file: str, context: TaskContext) -> None:
    """Load goal details from an input file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
        
    try:
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Update context with data from input file
        if "next_step" in data:
            context.goal.description = data["next_step"]
            logging.info(f"Using next_step from input file: {data['next_step']}")
            
        if "validation_criteria" in data:
            context.goal.validation_criteria = data["validation_criteria"]
            logging.info("Added validation criteria from input file")
            
        if "relevant_context" in data:
            context.metadata["relevant_context"] = data["relevant_context"]
            logging.info("Added relevant context from input file")
            
        # Add memory information from current_state if available
        if "current_state" in data:
            current_state = data["current_state"]
            
            # Update git_hash from current_state
            if "git_hash" in current_state:
                context.state.git_hash = current_state["git_hash"]
                logging.info(f"Using git_hash from input file: {current_state['git_hash']}")
                
            # Update memory_hash from current_state
            if "memory_hash" in current_state:
                context.state.memory_hash = current_state["memory_hash"]
                logging.info(f"Using memory_hash from input file: {current_state['memory_hash']}")
                
            # Update memory_repository_path from current_state
            if "memory_repository_path" in current_state:
                context.state.memory_repository_path = current_state["memory_repository_path"]
                logging.info(f"Using memory_repository_path from input file: {current_state['memory_repository_path']}")
                
            # Update other memory-related fields if available
            if "memory_reference" in current_state:
                context.state.memory_reference = current_state["memory_reference"]
                logging.info(f"Using memory_reference from input file: {current_state['memory_reference']}")
                
            if "memory_document_path" in current_state:
                context.state.memory_document_path = current_state["memory_document_path"]
            
            if "memory_timestamp" in current_state:
                context.state.memory_timestamp = current_state["memory_timestamp"]
            
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON in input file: {input_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to load input file: {str(e)}")

async def is_git_ancestor(repo_path: str, ancestor_hash: str, descendant_hash: str) -> bool:
    """
    Check if one hash is an ancestor of another in a git repository.
    
    Args:
        repo_path: Path to the git repository
        ancestor_hash: The hash to check if it's an ancestor
        descendant_hash: The hash to check if it's a descendant
        
    Returns:
        True if ancestor_hash is an ancestor of descendant_hash, False otherwise
    """
    if ancestor_hash == descendant_hash:
        return True
        
    try:
        # Use git merge-base --is-ancestor to check if one hash is an ancestor of another
        process = await asyncio.create_subprocess_exec(
            "git", "merge-base", "--is-ancestor", ancestor_hash, descendant_hash,
            cwd=repo_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Wait for the command to complete
        await process.communicate()
        
        # The command returns 0 if it's an ancestor, 1 if not
        return process.returncode == 0
    except Exception as e:
        logging.error(f"Error checking git ancestry: {str(e)}")
        return False

async def decompose_goal(
    repo_path: str,
    goal: str,
    input_file: Optional[str] = None,
    parent_goal: Optional[str] = None,
    goal_id: Optional[str] = None,
    memory_repo: Optional[str] = None,
    debug: bool = False,
    quiet: bool = False,
    bypass_validation: bool = False,
    logs_dir: str = "logs"
) -> Dict[str, Any]:
    """
    Asynchronous function to decompose a goal into subgoals.
    
    Args:
        repo_path: Path to the target repository
        goal: Description of the goal to achieve
        input_file: Optional path to input file with goal context
        parent_goal: Optional parent goal ID
        goal_id: Optional goal ID
        memory_repo: Optional path to memory repository
        debug: Whether to show debug output
        quiet: Whether to only show warnings and final result
        bypass_validation: Whether to bypass repository validation (for testing)
        logs_dir: Directory to store log files (default: "logs")
        
    Returns:
        Dictionary containing the decomposition result
    """
    # Configure logging - always use logs_dir for actual logs
    configure_logging(debug, quiet, logs_dir)
    
    # Initialize state and context
    state = State(
        git_hash=await get_current_hash(repo_path),
        repository_path=repo_path,
        description="Initial state before goal decomposition"
    )
    
    # Create context before loading input file
    context = TaskContext(
        state=state,
        goal=Goal(description=goal),
        iteration=0,
        execution_history=[],
        metadata={}
    )
    
    # Load input file if specified - this will override state with values from the file
    if input_file:
        await load_input_file(input_file, context)
    else:
        # Only set memory_repository_path if memory_repo is provided and we don't have input file
        # This ensures memory_repo is NEVER used when input_file is provided
        if memory_repo:
            logging.info(f"Setting memory repository path to: {memory_repo}")
            state.memory_repository_path = memory_repo
            # We don't set memory_hash here because we don't want to initialize memory
            # without explicit instructions (this would be a new goal with no memory yet)
    
    # Add metadata for parent goals and goal IDs
    if parent_goal:
        context.metadata["parent_goal"] = parent_goal
    if goal_id:
        context.metadata["goal_id"] = goal_id
    
    # Store the initial memory hash and git hash for sanity checking later
    initial_memory_hash = context.state.memory_hash
    initial_git_hash = context.state.git_hash
    
    # Validate repository state
    if not bypass_validation:
        try:
            await validate_repository_state(repo_path)
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    # Initialize goal decomposer
    decomposer = GoalDecomposer()
    
    # Determine next step
    subgoal_plan = await decomposer.determine_next_step(context)
    
    # Get current git hash
    git_hash = await get_current_hash(repo_path)
    
    # Use memory_hash from context state (which was either loaded from input_file or never set)
    memory_hash = context.state.memory_hash
    
    # Sanity check: Verify that the final git hash is a descendant of the initial git hash
    if initial_git_hash and git_hash and initial_git_hash != git_hash:
        is_descendant = await is_git_ancestor(
            repo_path,
            initial_git_hash,
            git_hash
        )
        if not is_descendant:
            logging.warning(f"REPO ANCESTRY CHECK FAILED: Final git hash {git_hash} is not a descendant of initial git hash {initial_git_hash}")
            # Don't fail the operation, but log the warning
        else:
            logging.info(f"Repo ancestry check passed: {git_hash} is a descendant of {initial_git_hash}")
    
    # Sanity check: Verify that the final memory hash is a descendant of the initial memory hash
    if initial_memory_hash and memory_hash and initial_memory_hash != memory_hash:
        if context.state.memory_repository_path:
            is_descendant = await is_git_ancestor(
                context.state.memory_repository_path, 
                initial_memory_hash, 
                memory_hash
            )
            if not is_descendant:
                logging.warning(f"MEMORY ANCESTRY CHECK FAILED: Final memory hash {memory_hash} is not a descendant of initial memory hash {initial_memory_hash}")
                # Don't fail the operation, but log the warning
            else:
                logging.info(f"Memory ancestry check passed: {memory_hash} is a descendant of {initial_memory_hash}")
    
    # Return the decomposition result without creating any files
    return {
        "success": True,
        "next_step": subgoal_plan.next_step,
        "validation_criteria": subgoal_plan.validation_criteria,
        "reasoning": subgoal_plan.reasoning,
        "requires_further_decomposition": subgoal_plan.requires_further_decomposition,
        "relevant_context": subgoal_plan.relevant_context,
        "git_hash": git_hash,
        "memory_hash": memory_hash,
        "is_task": not subgoal_plan.requires_further_decomposition,
        "goal_file": f"{goal_id or 'G1'}.json",  # Add goal_file for test compatibility with simple naming
        # Include memory reference ID (preferred) and other memory information
        "memory_reference": getattr(context.state, "memory_reference", None),
        "memory_document_path": getattr(context.state, "memory_document_path", None),
        "memory_timestamp": getattr(context.state, "memory_timestamp", None),
        # Include initial memory hash for reference
        "initial_memory_hash": initial_memory_hash,
        # Include initial git hash for reference
        "initial_git_hash": initial_git_hash
    }

# Create a separate async entry point for CLI to avoid nesting asyncio.run() calls
async def async_main():
    """Async entry point for CLI"""
    parser = argparse.ArgumentParser(description="Decompose a goal into subgoals")
    parser.add_argument("repo_path", help="Path to the target repository")
    parser.add_argument("goal", help="Description of the goal to achieve")
    parser.add_argument("--input-file", help="Path to input file with goal context")
    parser.add_argument("--parent-goal", help="Parent goal ID")
    parser.add_argument("--goal-id", help="Goal ID")
    parser.add_argument("--memory-repo", help="Path to memory repository")
    parser.add_argument("--debug", action="store_true", help="Show debug output")
    parser.add_argument("--quiet", action="store_true", help="Only show warnings and final result")
    parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
    parser.add_argument("--logs-dir", default="logs", help="Directory to store log files")
    
    args = parser.parse_args()
    
    # Call the async function directly
    result = await decompose_goal(
        repo_path=args.repo_path,
        goal=args.goal,
        input_file=args.input_file,
        parent_goal=args.parent_goal,
        goal_id=args.goal_id,
        memory_repo=args.memory_repo,
        debug=args.debug,
        quiet=args.quiet,
        bypass_validation=args.bypass_validation,
        logs_dir=args.logs_dir
    )
    
    # Print result as JSON
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    # Only use asyncio.run at the top level
    asyncio.run(async_main()) 