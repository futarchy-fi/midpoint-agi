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
import datetime

# Early initialization of logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Set up the script location and repository root
script_location = Path(__file__).resolve()
repo_root = script_location.parent.parent.parent.parent

# Add repo root, src directory, and scripts directory to path
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))
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
    retrieve_documents = memory_tools.retrieve_documents
    update_cross_reference = memory_tools.update_cross_reference
    
    # Import the high-level store_memory_document
    from midpoint.agents.tools.memory_tools import store_memory_document
    
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
    
    async def store_memory_document(content, category, metadata=None, repo_path=None, memory_hash=None):
        """Fallback implementation to store a document."""
        logging.warning("Using fallback store_memory_document implementation.")
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
        return {
            "success": True,
            "document_path": str(doc_path.relative_to(repo_path) if repo_path in doc_path.parents else doc_path)
        }

# Now import the rest of the modules needed
import json
import asyncio
import argparse
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from midpoint.agents.models import State, Goal, SubgoalPlan, TaskContext, MemoryState
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
    retrieve_memory_documents,
)
from midpoint.agents.config import get_openai_api_key
from midpoint.utils.logging import log_manager
from dotenv import load_dotenv
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
    task_summary_file = log_dir / f"task_summary_{timestamp}.log"
    llm_responses_file = log_dir / f"llm_responses_{timestamp}.log"
    
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
    
    # Create task summary file with header
    with open(task_summary_file, "w") as f:
        f.write(f"Task Summary Log - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
    
    # Create LLM responses file with header
    with open(llm_responses_file, "w") as f:
        f.write(f"LLM Responses Log - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
    
    # Set up a filter for console output to make it more concise
    class ConsoleFormatFilter(logging.Filter):
        def filter(self, record):
            # Process both INFO and DEBUG level logs for formatting
            if record.levelno in [logging.INFO, logging.DEBUG]:
                # Convert record.msg to string if it's not already
                msg = str(record.msg)
                
                # Truncate long lines to 300 characters
                if len(msg) > 300:
                    # Find the last space before 300 characters to avoid cutting words
                    last_space = msg[:300].rfind(' ')
                    if last_space > 0:
                        msg = msg[:last_space] + "..."
                    else:
                        msg = msg[:300] + "..."
                
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
                    if pattern in msg:
                        return False
                
                # Also hide the individual validation criteria lines from the default output
                if msg.startswith('  ') and any(c.isdigit() for c in msg) and '. ' in msg:
                    return False
                
                # Make emojis and messages more concise
                if '📂 Listing directory:' in msg:
                    msg = msg.replace('📂 Listing directory:', '📂')
                elif '📄 Reading:' in msg:
                    msg = msg.replace('📄 Reading:', '📄')
                elif '🔍 Searching code:' in msg:
                    msg = msg.replace('🔍 Searching code:', '🔍 Searching for pattern:')
                elif '🤖 API call completed' in msg:
                    return False  # Don't show API calls in console
                elif '✅ Next step:' in msg:
                    # Show this message only in standalone mode
                    if 'main' not in sys._getframe().f_back.f_code.co_name:
                        return True
                    return False  # Don't show in main() since we have better formatting there
                # Allow our new emoji formats to pass through
                elif any(emoji in msg for emoji in ['📂', '📄', '🔍', '🌐', '💾', '🔄', '🔗', '✅', '📝', '➕', '➖', '🔀']):
                    return True
                elif ('🔄 Next subgoal:' in msg) or ('✅ Next task:' in msg):
                    return True
                elif 'Determining next step for goal:' in msg:
                    try:
                        # Try to safely extract the goal description
                        if record.args and len(record.args) > 0:
                            goal_desc = str(record.args[0])
                            msg = f"🎯 Goal: {goal_desc}"
                        else:
                            msg = "🎯 Processing goal"
                        record.args = ()  # Clear arguments to avoid formatting errors
                    except:
                        msg = "🎯 Processing goal"
                        record.args = ()
                elif '🚀 Starting GoalDecomposer' in msg:
                    msg = '🚀 Starting'
                elif 'HTTP Request:' in msg or 'API' in msg:
                    return False  # Don't show HTTP requests in console
                elif 'Validating repository state' in msg:
                    return False  # Hide validation message in console
                
                # Update the record's message with our formatted version
                record.msg = msg
                
            return True
    
    # Apply the filter only to the console handler
    console_handler.addFilter(ConsoleFormatFilter())
    
    # Log the configuration
    if quiet:
        print("Running in quiet mode - only showing result and errors...", file=sys.stderr)
    
    # Return log file paths for reference
    return log_file, task_summary_file, llm_responses_file

def log_task_summary(task_summary_file: Path, context: TaskContext):
    """
    Log a summary of tasks and metadata to the task summary file.
    
    Args:
        task_summary_file: Path to the task summary log file
        context: The current task context
    """
    try:
        with open(task_summary_file, "a") as f:
            f.write("\n=== Task Context Summary ===\n")
            f.write(f"Goal: {context.goal.description}\n")
            f.write(f"Iteration: {context.iteration}\n")
            f.write(f"Git Hash: {context.state.git_hash}\n")
            
            if hasattr(context, 'metadata') and context.metadata:
                f.write("\nMetadata:\n")
                for key, value in context.metadata.items():
                    if key == 'completed_tasks':
                        f.write(f"\nCompleted Tasks ({len(value)}):\n")
                        for i, task in enumerate(value, 1):
                            f.write(f"\nTask {i}:\n")
                            f.write(f"  Description: {task.get('description', 'No description')}\n")
                            if task.get('validation_criteria'):
                                f.write("  Validation criteria:\n")
                                for criterion in task['validation_criteria']:
                                    f.write(f"    - {criterion}\n")
                            if task.get('final_state'):
                                f.write(f"  Final state: {task.get('final_state', {}).get('description', 'No final state')}\n")
                    else:
                        f.write(f"  {key}: {value}\n")
            
            f.write("\n" + "=" * 50 + "\n")
            f.flush()
    except Exception as e:
        logging.error(f"Failed to write task summary: {str(e)}")

class GoalDecomposer:
    """
    Goal decomposition agent implementation.
    
    This agent is responsible for determining the next step toward achieving a complex goal.
    """
    
    def __init__(self, model: str = "gpt-4o-mini", max_history_entries: int = 5):
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
        
    async def _save_interaction_to_memory(
            self,
            interaction_type: str,
            content: str,
            metadata: Optional[Dict[str, Any]] = None,
            memory_hash: Optional[str] = None,
            repo_path: Optional[str] = None,
            goal_name: Optional[str] = None
        ) -> Optional[str]:
        """Save interaction to memory repository."""
        try:
            # Get memory repository path
            if not repo_path:
                repo_path = get_repo_path()
                logging.info(f"Using default memory repository path: {repo_path}")
            
            # Require memory hash to be provided
            if not memory_hash:
                logging.error("No memory hash provided")
                logging.info("Current memory state:")
                logging.info(f"  Repository path: {repo_path}")
                logging.info(f"  Metadata: {metadata}")
                return None
            
            logging.info(f"Attempting to save {interaction_type} with memory hash: {memory_hash[:8]}...")
            
            # Create timestamped filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{interaction_type}_{timestamp}.md"
            
            # Create document content
            doc_content = f"""# {interaction_type.title()}

Timestamp: {timestamp}

{content}
"""
            
            # Add metadata if provided
            if metadata:
                doc_content += "\n## Metadata\n\n"
                for key, value in metadata.items():
                    doc_content += f"- {key}: {value}\n"
            
            # Create category with goal name if available
            category = "goal_decomposition"
            if goal_name:
                # Sanitize goal name for use in category
                safe_goal_name = re.sub(r'[^a-zA-Z0-9_-]', '_', goal_name)[:50]  # Limit length
                category = f"goal_decomposition_{safe_goal_name}"
            
            # Store document using high-level store_memory_document
            result = await store_memory_document(
                content=doc_content,
                category=category,
                metadata={
                    "interaction_type": interaction_type,
                    "timestamp": timestamp,
                    "goal_name": goal_name,
                    "commit_message": f"Add {interaction_type} for goal: {goal_name}" if goal_name else f"Add {interaction_type}",
                    **(metadata or {})
                },
                memory_hash=memory_hash,
                memory_repo_path=repo_path
            )
            
            if result["success"]:
                logging.info(f"Successfully saved {interaction_type} to memory: {result['document_path']}")
                return result["document_path"]
            else:
                logging.error(f"Failed to save {interaction_type} to memory: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logging.error(f"Error saving {interaction_type} to memory: {str(e)}")
            logging.info("Memory operation details:")
            logging.info(f"  Interaction type: {interaction_type}")
            logging.info(f"  Memory hash: {memory_hash[:8] if memory_hash else 'None'}")
            logging.info(f"  Repository path: {repo_path}")
            return None

    async def _save_conversation_to_memory(
        self,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        memory_hash: Optional[str] = None,
        repo_path: Optional[str] = None,
        goal_name: Optional[str] = None
    ) -> Optional[str]:
        """Save conversation to memory, excluding memory messages."""
        try:
            # Filter out memory messages and system prompt
            filtered_messages = [
                msg for msg in messages 
                if msg["role"] not in ["system"]
            ]
            
            # Format conversation content
            content = "## Conversation\n\n"
            for msg in filtered_messages:
                content += f"### {msg['role'].title()}\n\n{msg['content']}\n\n"
            
            # Add metadata if provided
            if metadata:
                content += "## Metadata\n\n"
                for key, value in metadata.items():
                    content += f"- {key}: {value}\n"
            
            # Get memory repository path if not provided
            if not repo_path:
                repo_path = get_repo_path()
            
            # Get memory hash from metadata or use provided memory_hash
            if not memory_hash and metadata and "memory_hash" in metadata:
                memory_hash = metadata["memory_hash"]
                logging.info(f"Using memory hash from metadata: {memory_hash[:8]}")
            elif memory_hash:
                logging.info(f"Using provided memory hash: {memory_hash[:8]}")
            else:
                logging.error("No memory hash provided")
                return None
            
            # Create category with goal name if available
            category = "goal_decomposition"
            if goal_name:
                # Sanitize goal name for use in category
                safe_goal_name = re.sub(r'[^a-zA-Z0-9_-]', '_', goal_name)[:50]  # Limit length
                category = f"goal_decomposition_{safe_goal_name}"
            
            # Store document using high-level store_memory_document
            result = await store_memory_document(
                content=content,
                category=category,
                metadata={
                    "interaction_type": "goal_decomposition",
                    "timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "goal_name": goal_name,
                    "commit_message": f"Add conversation for goal: {goal_name}" if goal_name else "Add conversation",
                    **(metadata or {})
                },
                memory_hash=memory_hash,
                memory_repo_path=repo_path
            )
            
            if result["success"]:
                logging.info(f"Saved conversation to memory: {result['document_path']}")
                return result["document_path"]
            else:
                logging.error(f"Failed to save conversation to memory: {result.get('error', 'Unknown error')}")
                return None
                
        except Exception as e:
            logging.error(f"Error saving conversation to memory: {str(e)}")
            return None

    def _generate_system_prompt(self) -> str:
        """
        Generate a system prompt that includes dynamically generated tool descriptions.
        
        Returns:
            The system prompt string.
        """
        tool_descriptions_text = self._get_tool_descriptions()
        
        # Create the system prompt with the tool descriptions
        system_prompt = f"""You are an expert software architect and project planner.
Your task is to determine if a goal is complete and if not, determine the SINGLE NEXT STEP toward that goal.
Follow the OODA loop: Observe, Orient, Decide, Act.

1. OBSERVE: Explore the repository to understand its current state. Use the available tools.
2. ORIENT: Analyze how the current state relates to the final goal.
3. DECIDE: First determine if the goal is complete, then if not, determine the most promising SINGLE NEXT STEP.
4. OUTPUT: Provide a structured response indicating goal completion status and next steps if needed.

For complex goals, consider if the best next step is exploration, research, or a "study session" 
rather than immediately jumping to implementation.

You have access to a memory repository where you can store and retrieve information across sessions.
For intellectual tasks (such as studying, analyzing, or understanding code), you should consider
updating the memory repository as a valid next step.

For incomplete goals:
- Determine whether the next step requires further decomposition
- If the next step is still complex and would benefit from being broken down further, set 'requires_further_decomposition' to TRUE
- If the next step is simple enough to be directly implemented by a TaskExecutor agent, set 'requires_further_decomposition' to FALSE
- Identify relevant context that should be passed to child subgoals

You have access to these tools:
{tool_descriptions_text}
"""
        
        return system_prompt

    def _get_tool_descriptions(self):
        """
        Generate a formatted string of tool descriptions.
        
        Returns:
            A string containing formatted tool descriptions.
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
        return "\n".join(tool_descriptions)

    async def _get_recent_memory_context(self, memory_hash, memory_repo_path, limit=5):
        """
        Retrieve recent memory documents from the general_memory folder and format them for context.
        
        Args:
            memory_hash: The memory hash to use for retrieval
            memory_repo_path: Path to the memory repository
            limit: Maximum number of documents to retrieve
            
        Returns:
            A formatted string containing the recent memory context
        """
        try:
            # Import the retrieve_documents function
            from midpoint.agents.tools.memory_tools import retrieve_documents
            
            # Get recent documents from the general_memory folder
            documents = retrieve_documents(
                category="../general_memory",  # Path is relative to documents/ in the repo
                limit=limit,
                repo_path=memory_repo_path,
                memory_hash=memory_hash
            )
            
            if not documents:
                logging.info("No recent memory documents found")
                return ""
            
            # Format the documents into a context string
            memory_context = "## RECENT MEMORY\n\n"
            for i, (path, content) in enumerate(documents, 1):
                # Extract filename from path
                filename = os.path.basename(path)
                # Try to parse timestamp from filename (format: TIMESTAMP_goal_decomposer_TYPE.md)
                timestamp_str = filename.split('_')[0] if '_' in filename else 'unknown'
                
                # Format the document with timestamp and truncate content if too long
                max_content_length = 500  # Limit content length to avoid excessive tokens
                truncated_content = content[:max_content_length] + "..." if len(content) > max_content_length else content
                
                memory_context += f"### Memory Document {i}: {filename}\n"
                memory_context += f"Timestamp: {timestamp_str}\n"
                memory_context += f"```\n{truncated_content}\n```\n\n"
            
            return memory_context
        except Exception as e:
            logging.error(f"Error retrieving memory context: {str(e)}")
            return ""

    async def determine_next_step(self, context: TaskContext, setup_logging=False, debug=False, quiet=False) -> SubgoalPlan:
        """Determine the next step toward achieving the goal."""
        # Set up logging if requested (only when called directly, not from orchestrator)
        if setup_logging:
            log_file, task_summary_file, llm_responses_file = configure_logging(debug, quiet)
            
        logging.info("Determining next step for goal: %s", context.goal.description)
        
        # Log task summary at the start
        if setup_logging:
            log_task_summary(task_summary_file, context)
        
        # Validate memory state
        if not context.memory_state:
            raise ValueError("Memory state is required for goal decomposition")
        
        # Validate memory state attributes
        memory_hash = getattr(context.memory_state, "memory_hash", None)
        memory_path = getattr(context.memory_state, "repository_path", None)
        if not memory_hash or not memory_path:
            raise ValueError("Memory state must have both memory_hash and repository_path")
        
        # Log memory state information
        logging.info(f"Memory state: hash={memory_hash[:8]}, path={memory_path}")
        
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
            logging.info(f"Using memory hash from context state: {memory_hash[:8]}")
        
        # Initialize messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
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
                            "content": content
                        })
                    
                    # Log memory context stats
                    logging.info(f"Added {len(memory_documents)} memory documents to conversation")
            except Exception as e:
                logging.error(f"Error retrieving memory context: {str(e)}")
        
        # Add the user prompt as the final message
        messages.append({"role": "user", "content": user_prompt})
        
        # Track tool usage for metadata
        tool_usage = []
        
        try:
            # Get the next step from the model
            message, tool_calls = await self.tool_processor.run_llm_with_tools(
                messages,
                model=self.model,
                validate_json_format=True,
                max_tokens=3000
            )
            
            # Process tool calls and update tool usage
            if tool_calls:
                for tool_call in tool_calls:
                    tool_usage.append(tool_call)
            
            # Parse the model's response
            try:
                if isinstance(message, list):
                    # If message is a list, get the last message's content
                    content = message[-1].get('content', '')
                else:
                    # If message is a dict or object, get its content
                    content = message.get('content') if isinstance(message, dict) else message.content
                output_data = json.loads(content)
            except json.JSONDecodeError:
                logging.error("Failed to parse model response as JSON")
                raise ValueError("Model response is not valid JSON")
            
            # Log LLM response to dedicated file
            if setup_logging:
                with open(llm_responses_file, "a") as f:
                    f.write("\n=== LLM Response ===\n")
                    f.write(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(f"Goal: {context.goal.description}\n")
                    f.write(f"Model: {self.model}\n")
                    f.write("\nMessages:\n")
                    for msg in messages:
                        f.write(f"\n{msg['role'].upper()}:\n{msg['content']}\n")
                    f.write("\nResponse:\n")
                    f.write(content)
                    f.write("\n\n" + "=" * 50 + "\n")
                    f.flush()
            
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
            self._validate_subgoal(final_output)
            
            # Log tool usage at debug level
            logging.debug("Tool usage: %s", tool_usage)

            # Add logging for API call details - use serialized messages at debug level
            try:
                serialized_messages = self._serialize_messages(messages)
                logging.debug("API call history: %s", json.dumps(serialized_messages, indent=2))
                
                # Save only the complete conversation to memory
                await self._save_conversation_to_memory(
                    messages,
                    metadata={
                        "goal": context.goal.description, 
                        "message_count": len(serialized_messages),
                        "next_step": final_output.next_step,
                        "requires_further_decomposition": final_output.requires_further_decomposition,
                        "memory_hash": memory_hash  # Pass the memory hash to ensure we save to the correct state
                    },
                    memory_hash=memory_hash,
                    repo_path=memory_repo_path,
                    goal_name=context.goal.description  # Pass the goal name
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

            # Log final task summary
            if setup_logging:
                with open(task_summary_file, "a") as f:
                    f.write("\n=== Final Output ===\n")
                    f.write(f"Next step: {final_output.next_step}\n")
                    f.write(f"Requires further decomposition: {final_output.requires_further_decomposition}\n")
                    f.write("\nValidation criteria:\n")
                    for i, criterion in enumerate(final_output.validation_criteria, 1):
                        f.write(f"  {i}. {criterion}\n")
                    f.write("\n" + "=" * 50 + "\n")
                    f.flush()

            return final_output
            
        except Exception as e:
            # Log the error to memory
            await self._save_conversation_to_memory(
                messages,
                metadata={
                    "error_type": type(e).__name__, 
                    "goal": context.goal.description,
                    "memory_hash": memory_hash  # Pass the memory hash to ensure we save to the correct state
                },
                memory_hash=memory_hash,
                repo_path=memory_repo_path,
                goal_name=context.goal.description  # Pass the goal name
            )
            
            # Log error to task summary
            if setup_logging:
                with open(task_summary_file, "a") as f:
                    f.write("\n=== Error ===\n")
                    f.write(f"Error type: {type(e).__name__}\n")
                    f.write(f"Error message: {str(e)}\n")
                    f.write("\n" + "=" * 50 + "\n")
                    f.flush()
            
            # Let the main function handle the specific error types
            raise
    
    def _create_user_prompt(self, context: TaskContext) -> str:
        """Create the user prompt for the agent."""
        prompt = f"""Goal: {context.goal.description}

Validation Criteria for Final Goal:
{chr(10).join(f"- {criterion}" for criterion in context.goal.validation_criteria)}

Current State:
- Git Hash: {context.state.git_hash}
"""
        # Add memory state information if available
        if context.memory_state:
            memory_hash = getattr(context.memory_state, "memory_hash", None)
            memory_path = getattr(context.memory_state, "repository_path", None)
            prompt += f"""
Memory State:
- Memory Hash: {memory_hash}
- Memory Repository Path: {memory_path}
"""

        # Add completed tasks information if available
        if hasattr(context, 'metadata') and context.metadata and 'completed_tasks' in context.metadata:
            logging.info("Found completed tasks in metadata:")
            logging.info(f"Number of completed tasks: {len(context.metadata['completed_tasks'])}")
            prompt += "\nCompleted Tasks:\n"
            for i, task in enumerate(context.metadata['completed_tasks'], 1):
                logging.info(f"Task {i}:")
                logging.info(f"  Description: {task.get('description', 'No description')}")
                logging.info(f"  Validation criteria: {task.get('validation_criteria', [])}")
                logging.info(f"  Final state: {task.get('final_state', {}).get('description', 'No final state')}")
                
                prompt += f"{i}. {task['description']}\n"
                if task.get('validation_criteria'):
                    prompt += "   Validation criteria:\n"
                    for criterion in task['validation_criteria']:
                        prompt += f"   - {criterion}\n"
                if task.get('final_state'):
                    prompt += f"   Final state: {task['final_state'].get('description', '')}\n"
                prompt += "\n"  # Add blank line between tasks
        else:
            logging.info("No completed tasks found in metadata")

        prompt += f"""
Context:
- Iteration: {context.iteration}
- Previous Steps: {len(context.metadata.get('completed_tasks', [])) if context.metadata else 0}

Your task is to determine based on the completed tasks whether the validation criteria for the goal have been met.

For uncomplated goals, you must explore the repository, reason and provide a SINGLE NEXT STEP toward achieving the goal.

For complex goals, consider if the best next step is exploration, research, or a "study session" 
rather than immediately jumping to implementation.

You MUST provide a structured response in JSON format with these fields:
For completed goals:
- goal_completed: true
- completion_summary: A summary of what was accomplished and which task(s) satisfied the goal's validation criteria
- reasoning: Explanation of why the goal is considered complete

For incomplete goals:
- goal_completed: false
- next_step: A clear description of the single next step to take
- validation_criteria: List of measurable criteria to validate this step's completion
- reasoning: Explanation of why this is the most promising next action
- requires_further_decomposition: Boolean indicating if this step needs further breakdown
- relevant_context: Object containing relevant information to pass to child subgoals

IMPORTANT: Return ONLY raw JSON without any markdown formatting or code blocks. Do not wrap the JSON in ```json ... ``` or any other formatting.
"""
        return prompt
    
    def _validate_subgoal(self, subgoal: SubgoalPlan) -> bool:
        """
        Validate that a subgoal plan has all required fields.
        
        Args:
            subgoal: The subgoal plan to validate
            
        Returns:
            True if valid, False otherwise
            
        Raises:
            ValueError: If the subgoal is invalid
        """
        # Reasoning is always required
        if not subgoal.reasoning:
            raise ValueError("Subgoal must have reasoning")

        # Check goal completion status
        if subgoal.goal_completed:
            # For completed goals, we need a completion summary but not next_step or validation_criteria
            if not subgoal.completion_summary:
                raise ValueError("Completed goals must have a completion_summary")
            if subgoal.next_step or subgoal.validation_criteria:
                raise ValueError("Completed goals should not have next_step or validation_criteria")
            # Completed goals should not require further decomposition
            if subgoal.requires_further_decomposition:
                raise ValueError("Completed goals cannot require further decomposition")
        else:
            # For incomplete goals, we need next_step and validation_criteria
            if not subgoal.next_step:
                raise ValueError("Incomplete goals must have next_step")
            if not subgoal.validation_criteria:
                raise ValueError("Incomplete goals must have validation_criteria")
            # Completion summary should not be present for incomplete goals
            if subgoal.completion_summary:
                raise ValueError("Incomplete goals should not have completion_summary")

        return True

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
    try:
        # Log the start of file loading
        logging.info(f"Loading input file: {input_file}")
        
        # Load and parse the input file
        with open(input_file, 'r') as f:
            data = json.load(f)
            
        # Extract goal description if available
        if "goal" in data:
            context.goal.description = data["goal"]
            logging.info(f"Using goal from input file: {data['goal']}")
            
        # Extract validation criteria if available
        if "validation_criteria" in data:
            context.goal.validation_criteria = data["validation_criteria"]
            logging.info(f"Found {len(data['validation_criteria'])} validation criteria in input file")
            
        # Extract success threshold if available
        if "success_threshold" in data:
            context.goal.success_threshold = data["success_threshold"]
            logging.info(f"Using success threshold from input file: {data['success_threshold']}")
            
        # Extract metadata if available
        if "metadata" in data:
            context.metadata.update(data["metadata"])
            logging.info(f"Loaded metadata from input file: {data['metadata']}")
            
        # Extract completed tasks if available
        if "completed_tasks" in data:
            completed_tasks = data["completed_tasks"]
            context.metadata["completed_tasks"] = completed_tasks
            
            # Log detailed information about completed tasks
            logging.info(f"\n=== Completed Tasks Summary ===")
            logging.info(f"Total completed tasks: {len(completed_tasks)}")
            
            for i, task in enumerate(completed_tasks, 1):
                logging.info(f"\nTask {i}:")
                logging.info(f"  Description: {task.get('description', 'No description')}")
                
                if task.get('validation_criteria'):
                    logging.info("  Validation criteria:")
                    for criterion in task['validation_criteria']:
                        logging.info(f"    - {criterion}")
                
                if task.get('final_state'):
                    logging.info(f"  Final state: {task.get('final_state', {}).get('description', 'No final state')}")
                
                # Log any additional task metadata
                for key, value in task.items():
                    if key not in ['description', 'validation_criteria', 'final_state']:
                        logging.info(f"  {key}: {value}")
            
            logging.info("\n" + "=" * 50)
        else:
            logging.info("No completed tasks found in input file")
            
        # Add memory information from current_state if available
        if "current_state" in data:
            current_state = data["current_state"]
            logging.info("\n=== Current State Information ===")
            
            # Update git_hash from current_state
            if "git_hash" in current_state:
                context.state.git_hash = current_state["git_hash"]
                logging.info(f"Git hash: {current_state['git_hash'][:8]}")
            else:
                raise ValueError("No git_hash found in current_state")
                
            # Update memory_hash from current_state
            if "memory_hash" in current_state:
                context.state.memory_hash = current_state["memory_hash"]
                logging.info(f"Memory hash: {current_state['memory_hash'][:8]}")
            else:
                raise ValueError("No memory_hash found in current_state")
                
            # Update memory_repository_path from current_state
            if "memory_repository_path" in current_state:
                context.state.memory_repository_path = current_state["memory_repository_path"]
                logging.info(f"Memory repository path: {current_state['memory_repository_path']}")
            else:
                raise ValueError("No memory_repository_path found in current_state")
            
            # Create memory state - this is required
            if not current_state.get("memory_hash") or not current_state.get("memory_repository_path"):
                raise ValueError("Both memory_hash and memory_repository_path are required in current_state")
            
            context.memory_state = MemoryState(
                memory_hash=current_state["memory_hash"],
                repository_path=current_state["memory_repository_path"]
            )
            logging.info(f"Created memory state with hash {current_state['memory_hash'][:8]} and path {current_state['memory_repository_path']}")
            logging.info("=" * 50)
        else:
            raise ValueError("No current_state found in input file")
            
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in input file: {input_file}")
        raise ValueError(f"Invalid JSON in input file: {input_file}")
    except Exception as e:
        logging.error(f"Failed to load input file: {str(e)}")
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
    """
    # Configure logging - always use logs_dir for actual logs
    configure_logging(debug, quiet, logs_dir)
    
    # Initialize state and context
    state = State(
        repository_path=repo_path,
        description="Initial state before goal decomposition",
        memory_hash=None,  # Will be set from parent goal if available
        memory_repository_path=memory_repo
    )
    
    # Create context before loading input file
    context = TaskContext(
        state=state,
        goal=Goal(description=goal),
        memory_state=MemoryState(
            memory_hash=state.memory_hash or "",  # Use empty string as default if None
            repository_path=state.memory_repository_path or ""  # Use empty string as default if None
        ),
        iteration=0,
        execution_history=[],
        metadata={}
    )
    
    # Load input file if specified - this will override state with values from the file
    if input_file:
        await load_input_file(input_file, context)
    else:
        # Only set memory_repository_path if memory_repo is provided and we don't have input file
        if memory_repo:
            logging.info(f"Setting memory repository path to: {memory_repo}")
            state.memory_repository_path = memory_repo
    
    # Add metadata for parent goals and goal IDs
    if parent_goal:
        context.metadata["parent_goal"] = parent_goal
    if goal_id:
        context.metadata["goal_id"] = goal_id
    
    # Store the initial git hash for sanity checking later
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
    
    # Determine next step with logging enabled
    subgoal_plan = await decomposer.determine_next_step(context, setup_logging=True, debug=debug, quiet=quiet)
    
    # Get current git hash
    git_hash = await get_current_hash(repo_path)
    
    # Sanity check: Verify that the final git hash is a descendant of the initial git hash
    if initial_git_hash and git_hash and initial_git_hash != git_hash:
        is_descendant = await is_git_ancestor(
            repo_path,
            initial_git_hash,
            git_hash
        )
        if not is_descendant:
            logging.warning(f"REPO ANCESTRY CHECK FAILED: Final git hash {git_hash} is not a descendant of initial git hash {initial_git_hash}")
        else:
            logging.info(f"Repo ancestry check passed: {git_hash} is a descendant of {initial_git_hash}")
    
    # Return the decomposition result without creating any files
    return {
        "success": True,
        "next_step": subgoal_plan.next_step,
        "validation_criteria": subgoal_plan.validation_criteria,
        "reasoning": subgoal_plan.reasoning,
        "requires_further_decomposition": subgoal_plan.requires_further_decomposition,
        "relevant_context": subgoal_plan.relevant_context,
        "git_hash": git_hash,
        "is_task": not subgoal_plan.requires_further_decomposition,
        "goal_file": f"{goal_id or 'G1'}.json",  # Add goal_file for test compatibility with simple naming
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