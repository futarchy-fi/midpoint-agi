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
        import time
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
    """Agent responsible for determining the next step toward a complex goal."""
    
    def __init__(self, model: str = "gpt-4o", max_history_entries: int = 5):
        """Initialize the GoalDecomposer."""
        
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
- relevant_context: Object containing relevant information to pass to child subgoals"""
        
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
            
        # Check if this is a top-level goal without a parent, if so create a top-level goal file
        parent_goal_filename = None
        if not hasattr(context, 'metadata') or not context.metadata or 'parent_goal' not in context.metadata:
            # This is a top-level goal, create a goal file for it
            parent_goal_filename = self.create_top_goal_file(context)
            logging.info(f"Created top-level goal file: {parent_goal_filename}")
        else:
            # This is a subgoal, parent is already set
            parent_goal_filename = context.metadata.get('parent_goal_file', '')
            logging.info(f"Using parent goal: {parent_goal_filename}")
            
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
                            tool_summary.append(f"📂 {dir_path}")
                        elif tc.function.name == "read_file":
                            file_path = args.get("file_path", "unknown")
                            tool_summary.append(f"📄 {file_path}")
                        elif tc.function.name == "search_code":
                            pattern = args.get("pattern", "unknown")
                            tool_summary.append(f"🔍 {pattern}")
                        elif tc.function.name == "web_search":
                            query = args.get("query", "unknown")
                            tool_summary.append(f"🌐 {query}")
                        elif tc.function.name == "web_scrape":
                            url = args.get("url", "unknown")
                            tool_summary.append(f"🌐 {url}")
                        elif tc.function.name == "store_memory_document":
                            category = args.get("category", "unknown")
                            content_preview = args.get("content", "")[:30] + "..." if len(args.get("content", "")) > 30 else args.get("content", "")
                            tool_summary.append(f"💾 Storing document in {category}: \"{content_preview}\"")
                        elif tc.function.name == "retrieve_memory_documents":
                            category = args.get("category", "all")
                            limit = args.get("limit", 5)
                            tool_summary.append(f"🔍 Retrieving {limit} documents from {category}")
                        else:
                            tool_summary.append(f"🛠️ {tc.function.name}")
                    
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
                                # Handle parameter mapping for list_directory
                                # Convert repo_path/directory to path if present
                                if "repo_path" in function_args or "directory" in function_args:
                                    path = function_args.pop("repo_path", context.state.repository_path)
                                    directory = function_args.pop("directory", ".")
                                    function_args["path"] = os.path.join(path, directory)
                                
                                try:
                                    result = await list_directory(**function_args)
                                    result_str = json.dumps(result, indent=2)
                                except TypeError as e:
                                    error_msg = f"Unexpected error during tool execution: {str(e)}"
                                    logging.error(error_msg)
                                    result_str = {"error": error_msg}
                                    result_str = json.dumps(result_str, indent=2)
                            elif function_name == "read_file":
                                # Handle parameter mapping for read_file
                                # Convert repo_path/file_path if needed
                                if "repo_path" in function_args and "file_path" in function_args:
                                    repo_path = function_args.pop("repo_path")
                                    file_path = function_args.pop("file_path")
                                    function_args["file_path"] = os.path.join(repo_path, file_path)
                                
                                try:
                                    result = await read_file(**function_args)
                                    result_str = json.dumps(result, indent=2)
                                except TypeError as e:
                                    error_msg = f"Unexpected error during tool execution: {str(e)}"
                                    logging.error(error_msg)
                                    result_str = {"error": error_msg}
                                    result_str = json.dumps(result_str, indent=2)
                            elif function_name == "search_code":
                                try:
                                    result_str = await search_code(**function_args)
                                except TypeError as e:
                                    error_msg = f"Unexpected error during tool execution: {str(e)}"
                                    logging.error(error_msg)
                                    result_str = {"error": error_msg}
                                    result_str = json.dumps(result_str, indent=2)
                            elif function_name == "web_search":
                                try:
                                    result_str = await web_search(**function_args)
                                except TypeError as e:
                                    error_msg = f"Unexpected error during tool execution: {str(e)}"
                                    logging.error(error_msg)
                                    result_str = {"error": error_msg}
                                    result_str = json.dumps(result_str, indent=2)
                            elif function_name == "web_scrape":
                                try:
                                    result_str = await web_scrape(**function_args)
                                except TypeError as e:
                                    error_msg = f"Unexpected error during tool execution: {str(e)}"
                                    logging.error(error_msg)
                                    result_str = {"error": error_msg}
                                    result_str = json.dumps(result_str, indent=2)
                            elif function_name == "store_memory_document":
                                # Get memory repo path from context, function args, or default
                                memory_repo_path = function_args.get("memory_repo_path")
                                if not memory_repo_path:
                                    if context.state.memory_repository_path:
                                        memory_repo_path = context.state.memory_repository_path
                                    elif context.memory_state and context.memory_state.repository_path:
                                        memory_repo_path = context.memory_state.repository_path
                                    else:
                                        memory_repo_path = get_repo_path()
                                
                                # Prepare metadata with code hash if available
                                metadata = function_args.get("metadata", {})
                                if context.state.git_hash and "code_hash" not in metadata:
                                    metadata["code_hash"] = context.state.git_hash
                                
                                # Get the current memory hash before storing the document
                                old_memory_hash = None
                                try:
                                    old_memory_hash = await get_current_hash(memory_repo_path)
                                except Exception as e:
                                    logging.debug(f"Failed to get memory hash before storing document: {str(e)}")
                                
                                # Store the document
                                document_result = store_document(
                                    content=function_args["content"],
                                    category=function_args["category"],
                                    metadata=metadata,
                                    repo_path=memory_repo_path
                                )
                                
                                # Extract information from the result
                                if isinstance(document_result, dict):
                                    document_path = document_result["path"]
                                    document_filename = document_result["filename"]
                                    # If store_document returned the memory hash, use it
                                    if "memory_hash" in document_result:
                                        new_memory_hash = document_result["memory_hash"]
                                else:
                                    # Handle old return format (just a string path)
                                    document_path = document_result
                                    document_filename = os.path.basename(document_path)
                                
                                # Log that we stored a document
                                logging.info(f"💾 Stored document in {function_args['category']}: {document_filename}")
                                
                                # Get the new memory hash after storing the document
                                if not new_memory_hash:
                                    try:
                                        new_memory_hash = await get_current_hash(memory_repo_path)
                                    except Exception as e:
                                        logging.debug(f"Failed to get memory hash after storing document: {str(e)}")
                                
                                # Log memory hash change if it occurred
                                if old_memory_hash and new_memory_hash and old_memory_hash != new_memory_hash:
                                    logging.info(f"🔄 Memory hash changed: {old_memory_hash[:7]} → {new_memory_hash[:7]}")
                                    
                                    # Show what changed in the memory repository
                                    try:
                                        # Get details of what changed in the memory repo
                                        result = subprocess.run(
                                            ["git", "diff", "--name-status", f"{old_memory_hash}..{new_memory_hash}"],
                                            cwd=memory_repo_path,
                                            capture_output=True,
                                            text=True,
                                            check=True
                                        )
                                        if result.stdout.strip():
                                            # Parse the git diff output and log each changed file
                                            for line in result.stdout.strip().split('\n'):
                                                if not line.strip():
                                                    continue
                                                parts = line.split()
                                                if len(parts) >= 2:
                                                    change_type, file_path = parts[0], ' '.join(parts[1:])
                                                    change_emoji = "📝" if change_type == "M" else "➕" if change_type == "A" else "➖" if change_type == "D" else "🔀"
                                                    logging.info(f"{change_emoji} Memory {file_path} was {'modified' if change_type == 'M' else 'added' if change_type == 'A' else 'deleted' if change_type == 'D' else 'changed'}")
                                    except Exception as e:
                                        logging.debug(f"Failed to get memory diff: {str(e)}")
                                    
                                    # Update state's memory hash
                                    if hasattr(context, 'state') and hasattr(context.state, 'memory_hash'):
                                        context.state.memory_hash = new_memory_hash
                                    
                                    # Link code and memory hashes if code hash is available
                                    if context.state.git_hash:
                                        try:
                                            update_cross_reference(context.state.git_hash, new_memory_hash, memory_repo_path)
                                            logging.debug(f"🔗 Linked code hash {context.state.git_hash[:7]} to memory hash {new_memory_hash[:7]}")
                                        except Exception as e:
                                            logging.warning(f"Failed to link code hash to memory hash: {str(e)}")
                                
                                result_str = json.dumps({
                                    "success": True,
                                    "document_path": document_path,
                                    "message": f"Document stored in {function_args['category']} category"
                                })
                                
                                logging.info(f"💾 Stored memory document: {document_path} in {function_args['category']} category")
                            elif function_name == "retrieve_memory_documents":
                                # Get memory repo path from context, function args, or default
                                memory_repo_path = function_args.get("memory_repo_path")
                                if not memory_repo_path:
                                    if context.state.memory_repository_path:
                                        memory_repo_path = context.state.memory_repository_path
                                    elif context.memory_state and context.memory_state.repository_path:
                                        memory_repo_path = context.memory_state.repository_path
                                    else:
                                        memory_repo_path = get_repo_path()
                                
                                # Retrieve documents
                                documents = retrieve_documents(
                                    category=function_args.get("category"),
                                    limit=function_args.get("limit", 5),
                                    repo_path=memory_repo_path
                                )
                                
                                # Format the result
                                result_docs = []
                                for path, content in documents:
                                    result_docs.append({
                                        "path": path,
                                        "content": content[:1000] + ("..." if len(content) > 1000 else "")
                                    })
                                
                                result_str = json.dumps({
                                    "success": True,
                                    "documents": result_docs,
                                    "total": len(documents)
                                }, indent=2)
                                
                                logging.info(f"🔍 Retrieved {len(documents)} memory documents")
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

            return final_output
            
        except Exception as e:
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

    def create_top_goal_file(self, context: TaskContext) -> str:
        """
        Create a subgoal file for a top-level goal.
        
        Args:
            context: The current task context containing the goal
            
        Returns:
            The filename of the created subgoal file
        """
        logging.info("Creating top-level goal file")
        
        # Create the output data for the top-level goal
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = os.urandom(3).hex()
        
        # Create a unique file name
        filename = f"subgoal_{timestamp}_{hash_suffix}.json"
        goal_id = f"subgoal_{timestamp}_{hash_suffix}"
        
        # Prepare the goal content
        goal_content = {
            "description": context.goal.description,
            "next_step": context.goal.description,  # For top-level goals, these are the same
            "validation_criteria": context.goal.validation_criteria,
            "reasoning": "Top-level goal created for lineage tracking",
            "requires_further_decomposition": True,  # Top-level goals always need decomposition
            "relevant_context": {},
            "parent_goal": "",  # Empty for top-level goals
            "goal_id": goal_id,
            "timestamp": timestamp,
            "iteration": context.iteration
        }
        
        # Ensure logs directory exists
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        
        # Write the goal file
        output_file = os.path.join(logs_dir, filename)
        with open(output_file, 'w') as f:
            json.dump(goal_content, f, indent=2)
            
        logging.info(f"💾 Created top-level goal file: {output_file}")
        
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
            logging.warning(f"Failed to validate repository hash: {str(e)}")

def list_subgoal_files(logs_dir="logs"):
    """List available subgoal JSON files in the logs directory.
    
    Args:
        logs_dir: Path to the logs directory containing subgoal files
        
    Returns:
        List of tuples (file_path, timestamp, next_step, parent_goal) sorted by timestamp (newest first)
    """
    subgoal_files = []
    
    # Ensure logs directory exists
    os.makedirs(logs_dir, exist_ok=True)
    
    # Find all subgoal JSON files
    for file in os.listdir(logs_dir):
        file_path = os.path.join(logs_dir, file)
        
        if (file.startswith("subgoal_") or file.startswith("task_")) and file.endswith(".json"):
            try:
                # Extract timestamp from filename (format: subgoal_TIMESTAMP_HASH.json)
                parts = file.split('_')
                if len(parts) >= 3:
                    timestamp_str = parts[1]
                    try:
                        timestamp = datetime.datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                        formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                    except ValueError:
                        formatted_time = "Unknown time"
                else:
                    formatted_time = "Unknown time"
                
                # Read file to extract next step and parent goal
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    next_step = data.get("next_step", "Unknown")
                    parent_goal = data.get("parent_goal", "")
                
                # Add to list
                subgoal_files.append((file_path, formatted_time, next_step, parent_goal))
            except Exception as e:
                logging.debug(f"Error reading subgoal file {file}: {str(e)}")
    
    # Sort by timestamp (newest first)
    subgoal_files.sort(key=lambda x: os.path.getmtime(x[0]), reverse=True)
    
    return subgoal_files

async def main():
    """
    Main entry point for running the GoalDecomposer to determine the next step.
    """
    parser = argparse.ArgumentParser(description="Run GoalDecomposer to determine the next step.")
    parser.add_argument('--repo-path', required=True, help='Path to the git repository')
    parser.add_argument('--goal-description', help='Description of the goal')
    parser.add_argument('--input-file', help='Path to subgoal JSON file to use as input')
    parser.add_argument('--input-index', type=int, help='Index of subgoal file from list to use as input (1-based)')
    parser.add_argument('--append-to', help='Append the new subgoal to the chain from this file (creates a chain file)')
    parser.add_argument('--iteration', type=int, default=0, help='Iteration number')
    parser.add_argument('--execution-history', type=str, default='[]', help='Execution history as JSON string')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging on console (all logs are always written to file)')
    parser.add_argument('--quiet', action='store_true', help='Only show warnings and errors on console')
    parser.add_argument('--list-subgoals', action='store_true', help='List available subgoal files in the logs directory')
    parser.add_argument('--memory-hash', help='Hash of the memory repository state')
    parser.add_argument('--memory-repo-path', help='Path to the memory repository')

    args = parser.parse_args()

    # List available subgoal files if requested
    if args.list_subgoals:
        subgoal_files = list_subgoal_files()
        if not subgoal_files:
            logging.info("No saved subgoal files found in the logs directory")
            return
            
        logging.info("\nAvailable Subgoal Files:")
        logging.info("========================")
        for i, (file_path, timestamp, next_step, parent_goal) in enumerate(subgoal_files, 1):
            # Check if the file is a subgoal that requires further decomposition
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                requires_decomposition = data.get("requires_further_decomposition", True)
                emoji = "🔄" if requires_decomposition else "✅"
                step_type = "Subgoal" if requires_decomposition else "Task"
            except:
                # Default to subgoal if we can't determine
                emoji = "🔄"
                step_type = "Subgoal"
            
            logging.info(f"{i}. {os.path.basename(file_path)}")
            logging.info(f"   Time: {timestamp}")
            logging.info(f"   {emoji} {step_type}: {next_step}")
            logging.info(f"   Parent Goal: {parent_goal}")
            logging.info("")
        return

    # Set up logging based on command line arguments
    log_file = configure_logging(args.debug, args.quiet)

    # Handle input file by index if provided
    if args.input_index is not None:
        subgoal_files = list_subgoal_files()
        if not subgoal_files:
            logging.warning("No saved subgoal files found in the logs directory")
            return
            
        if args.input_index < 1 or args.input_index > len(subgoal_files):
            logging.error(f"Invalid index {args.input_index}. Please specify a value between 1 and {len(subgoal_files)}")
            return
            
        args.input_file = subgoal_files[args.input_index - 1][0]  # Get file path from the selected index
        logging.info(f"Using subgoal file: {os.path.basename(args.input_file)}")

    # Validate input arguments
    if not args.goal_description and not args.input_file:
        logging.error("Either --goal-description or --input-file or --input-index must be provided")
        return
    
    if args.goal_description and args.input_file:
        logging.warning("Both --goal-description and --input-file provided, using --input-file")
    
    # Load from input file if provided
    if args.input_file:
        try:
            with open(args.input_file, 'r') as f:
                input_data = json.load(f)
            
            if not isinstance(input_data, dict):
                logging.error(f"Error: Input file {args.input_file} does not contain a valid JSON object")
                return

            # Extract goal description from the input file
            if "next_step" in input_data:
                goal_description = input_data["next_step"]
                # Only log this information, don't print to console
                logging.info(f"Using next_step from input file: {goal_description}")
            else:
                logging.error(f"Input file {args.input_file} does not contain a 'next_step' field")
                return
                
            # Extract validation criteria if available
            validation_criteria = input_data.get("validation_criteria", [])
            
            # Extract relevant context if available
            relevant_context = input_data.get("relevant_context", {})
            # Only log this information, don't print to stdout
            logging.info(f"Added relevant context from input file: {relevant_context}")
            
            # Extract goal_id if available
            goal_id = input_data.get("goal_id", "")
            
            # Extract parent_goal if available
            parent_goal = input_data.get("parent_goal", "")
            
            # Extract timestamp if available
            timestamp = input_data.get("timestamp", "")
            
            # Extract iteration if available
            iteration = input_data.get("iteration", 0)
            
            # Extract memory state information if available in the input data
            if "memory_hash" in input_data:
                state_memory_hash = input_data["memory_hash"]
                if not args.memory_hash:  # Command line arg takes precedence
                    args.memory_hash = state_memory_hash
                    logging.info(f"Using memory hash from input file: {state_memory_hash[:7] if state_memory_hash else 'None'}")
            
            if "memory_repository_path" in input_data:
                state_memory_repo_path = input_data["memory_repository_path"]
                if not args.memory_repo_path:  # Command line arg takes precedence
                    args.memory_repo_path = state_memory_repo_path
                    logging.debug(f"Using memory repository path from input file: {state_memory_repo_path}")
        except json.JSONDecodeError:
            logging.error(f"Input file {args.input_file} does not contain valid JSON")
            raise ValueError(f"Input file {args.input_file} does not contain valid JSON")
        except FileNotFoundError:
            logging.error(f"Input file {args.input_file} not found")
            raise FileNotFoundError(f"Input file {args.input_file} not found")
        except Exception as e:
            logging.error(f"Error reading input file: {str(e)}")
            raise
    else:
        # Use the goal description from command line
        goal_description = args.goal_description
        validation_criteria = []
        relevant_context = {}
        goal_id = ""
        parent_goal = ""
        timestamp = ""
        iteration = args.iteration

    try:
        logging.info("🚀 Starting GoalDecomposer")
        # Create the TaskContext
        state = State(
            repository_path=args.repo_path, 
            git_hash="", 
            description="Current state description",
            memory_repository_path=args.memory_repo_path,
            memory_hash=args.memory_hash
        )
        goal = Goal(description=goal_description, validation_criteria=validation_criteria)
        context = TaskContext(state=state, goal=goal, iteration=iteration, execution_history=[])
        
        # Add relevant context to metadata if loaded from input file
        if args.input_file and (relevant_context or parent_goal):
            if not hasattr(context, "metadata"):
                context.metadata = {}
            if relevant_context:
                context.metadata["parent_context"] = relevant_context
                # Only log this information, don't print to stdout
                logging.info(f"Added relevant context from input file: {relevant_context}")
            if parent_goal:
                context.metadata["parent_goal_file"] = parent_goal
                logging.info(f"Added parent goal from input file: {parent_goal}")
            if goal_id:
                context.metadata["goal_id"] = goal_id
                logging.info(f"Added goal ID from input file: {goal_id}")

        # Validate repository state and get current hash
        current_hash = await get_current_hash(args.repo_path)
        state.git_hash = current_hash
        
        # Get memory hash if not provided but memory repo path is
        if not state.memory_hash and state.memory_repository_path:
            try:
                memory_hash = await get_current_hash(state.memory_repository_path)
                state.memory_hash = memory_hash
                logging.info(f"Setting memory hash to current hash: {memory_hash[:7] if memory_hash else 'None'}")
            except Exception as e:
                logging.warning(f"Failed to get memory hash: {str(e)}")
        
        # Look up memory hash based on code hash if no memory hash is available
        if not state.memory_hash and state.repository_path and state.git_hash:
            try:
                # Import memory tools - ensure get_memory_for_code_hash is available
                from scripts.memory_tools import get_memory_for_code_hash
                default_memory_path = os.path.expanduser("~/.midpoint/memory")
                memory_repo_path = state.memory_repository_path or default_memory_path
                
                # Check if memory repo exists, create if not
                memory_repo = Path(memory_repo_path)
                if not memory_repo.exists():
                    logging.info(f"Creating memory repository at {memory_repo_path}")
                    memory_repo.mkdir(parents=True, exist_ok=True)
                    # Create basic structure
                    (memory_repo / "documents").mkdir(exist_ok=True)
                    (memory_repo / "documents" / "reasoning").mkdir(exist_ok=True)
                    (memory_repo / "documents" / "observations").mkdir(exist_ok=True)
                    (memory_repo / "documents" / "decisions").mkdir(exist_ok=True)
                    (memory_repo / "documents" / "study").mkdir(exist_ok=True)
                    (memory_repo / "metadata").mkdir(exist_ok=True)
                    
                    # Create cross-reference file
                    cross_ref_path = memory_repo / "metadata" / "cross-reference.json"
                    with open(cross_ref_path, "w") as f:
                        json.dump({"mappings": [], "latest": {}}, f, indent=2)
                    
                    # Initialize git repo if it doesn't exist
                    try:
                        subprocess.run(["git", "init"], cwd=memory_repo, check=True)
                        # Create an initial commit
                        with open(memory_repo / "README.md", "w") as f:
                            f.write("# Agent Memory Repository\n\nThis repository stores memory documents for the agent.\n")
                        
                        # Create .gitignore to exclude cross-reference.json
                        with open(memory_repo / ".gitignore", "w") as f:
                            f.write("# Ignore cross-reference file which changes frequently\n")
                            f.write("metadata/cross-reference.json\n")
                            
                        subprocess.run(["git", "add", "."], cwd=memory_repo, check=True)
                        subprocess.run(["git", "commit", "-m", "Initialize memory repository"], cwd=memory_repo, check=True)
                        logging.info("Initialized git repository in memory repository")
                    except Exception as e:
                        logging.warning(f"Failed to initialize git repository in memory repository: {str(e)}")
                
                # Try to get memory hash for code hash
                try:
                    memory_hash = get_memory_for_code_hash(state.git_hash, repo_path=memory_repo_path)
                    
                    if memory_hash:
                        state.memory_hash = memory_hash
                        if not state.memory_repository_path:
                            state.memory_repository_path = memory_repo_path
                        logging.info(f"Found memory hash for code hash {state.git_hash[:7]}: {memory_hash[:7]}")
                    else:
                        logging.info(f"No memory hash found for code hash {state.git_hash[:7]}")
                except Exception as e:
                    logging.warning(f"Failed to get memory hash for code hash: {str(e)}")
            except Exception as e:
                logging.warning(f"Failed to process memory repository: {str(e)}")
        
        await validate_repository_state(
            args.repo_path,
            git_hash=current_hash,  # Use the new parameter name
            skip_clean_check=True
        )

        # Initialize GoalDecomposer and determine the next step
        decomposer = GoalDecomposer()
        next_step = await decomposer.determine_next_step(
            context,
            setup_logging=True,  # This configures logging
            debug=args.debug,
            quiet=args.quiet
        )

        # Get the final memory hash if it might have changed
        final_memory_hash = None
        if state.memory_repository_path:
            try:
                final_memory_hash = await get_current_hash(state.memory_repository_path)
                
                # Log memory hash changes only if it changed since the last tool operation
                # (we already log changes during tool operations)
                if state.memory_hash != final_memory_hash:
                    # Check if there were actual changes from the current state.memory_hash
                    # (which would have been updated during tool operations)
                    try:
                        # Get details of what changed in the memory repo
                        result = subprocess.run(
                            ["git", "diff", "--name-status", f"{state.memory_hash}..{final_memory_hash}"],
                            cwd=state.memory_repository_path,
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        if result.stdout.strip():
                            # There were additional changes after tool operations
                            logging.debug(f"🔄 Memory hash changed after operations: {state.memory_hash[:7]} → {final_memory_hash[:7]}")
                            
                            # Parse the git diff output and log each changed file
                            for line in result.stdout.strip().split('\n'):
                                if not line.strip():
                                    continue
                                parts = line.split()
                                if len(parts) >= 2:
                                    change_type, file_path = parts[0], ' '.join(parts[1:])
                                    change_emoji = "📝" if change_type == "M" else "➕" if change_type == "A" else "➖" if change_type == "D" else "🔀"
                                    logging.info(f"{change_emoji} Memory {file_path} was {'modified' if change_type == 'M' else 'added' if change_type == 'A' else 'deleted' if change_type == 'D' else 'changed'}")
                        else:
                            # No actual changes on disk, just updating the final hash
                            logging.debug(f"Memory hash updated: {state.memory_hash[:7]} → {final_memory_hash[:7]} (no file changes)")
                    except Exception as e:
                        # Fallback to just logging the hash change
                        logging.debug(f"🔄 Memory hash changed after operations: {state.memory_hash[:7]} → {final_memory_hash[:7]}")
                        logging.debug(f"Failed to get memory diff: {str(e)}")
                
                # Link the code and memory hashes if they've both changed
                if state.git_hash and final_memory_hash and state.memory_hash != final_memory_hash:
                    try:
                        update_cross_reference(state.git_hash, final_memory_hash, state.memory_repository_path)
                        logging.debug(f"🔗 Linked code hash {state.git_hash[:7]} to memory hash {final_memory_hash[:7]}")
                    except Exception as e:
                        logging.warning(f"Failed to link code hash to memory hash: {str(e)}")
            except Exception as e:
                logging.warning(f"Failed to get final memory hash: {str(e)}")
        
        # Create the output data
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        hash_suffix = os.urandom(3).hex()
        
        # Choose file prefix and emoji based on whether this needs further decomposition
        file_prefix = "subgoal" if next_step.requires_further_decomposition else "task"
        status_emoji = "🔄" if next_step.requires_further_decomposition else "✅"
        step_type = "Next subgoal" if next_step.requires_further_decomposition else "Next task"
        
        # Create a unique file name and goal_id
        output_filename = f"{file_prefix}_{timestamp}_{hash_suffix}.json"
        goal_id = f"{file_prefix}_{timestamp}_{hash_suffix}"
        
        # Determine parent goal from context or input file
        parent_goal = ""
        if args.input_file:
            parent_goal = os.path.basename(args.input_file)
        elif hasattr(context, 'metadata') and context.metadata and 'parent_goal_file' in context.metadata:
            parent_goal = context.metadata['parent_goal_file']
        
        # Set the goal_id and parent_goal in the SubgoalPlan
        next_step.goal_id = goal_id
        next_step.parent_goal = parent_goal
        next_step.timestamp = timestamp
        next_step.iteration = context.iteration
        
        output_file = os.path.join("logs", output_filename)
        
        # Ensure logs directory exists
        os.makedirs("logs", exist_ok=True)
        
        # Create the output data structure
        output_data = {
            "next_step": next_step.next_step,
            "validation_criteria": next_step.validation_criteria,
            "reasoning": next_step.reasoning,
            "requires_further_decomposition": next_step.requires_further_decomposition,
            "relevant_context": next_step.relevant_context,
            "memory_hash": final_memory_hash or state.memory_hash,  # Use final hash if available
            "memory_repository_path": state.memory_repository_path,
            "parent_goal": next_step.parent_goal,  # Add parent goal reference
            "goal_id": next_step.goal_id,          # Add goal ID
            "timestamp": next_step.timestamp,      # Add timestamp
            "iteration": next_step.iteration,      # Add iteration
            "metadata": next_step.metadata
        }
        
        # Write the output to the file
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Log the result to console with appropriate emoji and step type
        logging.info(f"{status_emoji} {step_type}: {next_step.next_step}")
        
        # Only log the file location
        logging.info(f"💾 Saved {file_prefix} to {output_file}")
        
        # Print the final memory hash change if it occurred (specifically at the end)
        if state.memory_hash != final_memory_hash and final_memory_hash:
            logging.info(f"🔄 Memory hash changed after operations: {state.memory_hash[:7]} → {final_memory_hash[:7]}")
        
        # If append-to option is provided, create a chain file
        if args.append_to:
            try:
                # Load the chain from the specified file
                chain_data = None
                if os.path.exists(args.append_to):
                    with open(args.append_to, 'r') as f:
                        chain_data = json.load(f)
                
                # Create a new chain file if it doesn't exist
                if chain_data is None:
                    chain_data = {
                        "chain_name": f"Subgoal Chain {timestamp}",
                        "created_at": timestamp,
                        "steps": []
                    }
                
                # Add current subgoal to the chain
                new_step = {
                    "step_number": len(chain_data["steps"]) + 1,
                    "subgoal_file": os.path.basename(output_file),
                    "timestamp": timestamp,
                    "next_step": next_step.next_step,
                    "requires_further_decomposition": next_step.requires_further_decomposition
                }
                chain_data["steps"].append(new_step)
                
                # Write the updated chain to the file
                with open(args.append_to, 'w') as f:
                    json.dump(chain_data, f, indent=2)
                
                logging.info(f"Appended subgoal to chain file: {args.append_to}")
                logging.info(f"Subgoal appended to chain: {args.append_to}")
                logging.info(f"Chain now has {len(chain_data['steps'])} steps")
            except Exception as e:
                logging.error(f"Error appending to chain file: {str(e)}")
        
        # Log all detailed information but don't print to stdout
        logging.debug("\n\n====================")
        logging.debug("=== NEXT STEP ===")
        logging.debug("====================")
        logging.debug(next_step.next_step)
        
        logging.debug("\n====================")
        logging.debug("=== VALIDATION CRITERIA ===")
        logging.debug("====================")
        for i, criteria in enumerate(next_step.validation_criteria, 1):
            logging.debug(f"{i}. {criteria}")
        
        logging.debug("\n====================")
        logging.debug("=== REASONING ===")
        logging.debug("====================")
        logging.debug(next_step.reasoning)
        
        logging.debug("\n====================")
        logging.debug("=== REQUIRES FURTHER DECOMPOSITION ===")
        logging.debug("====================")
        logging.debug(str(next_step.requires_further_decomposition))
        
        logging.debug("\n====================")
        logging.debug("=== RELEVANT CONTEXT ===")
        logging.debug("====================")
        for k, v in next_step.relevant_context.items():
            logging.debug(f"{k}: {v}")
    except TypeError as e:
        if "'NoneType' object is not iterable" in str(e):
            logging.error("The agent response processing failed. This typically happens when:")
            logging.error("1. The agent provided a final response without tool calls")
            logging.error("2. The response format wasn't properly handled")
            logging.error("Try running with --debug to see more details about the API interactions.")
            logging.error(f"Technical details: {str(e)}")
            logging.debug("NoneType iteration error: %s", str(e))
            return
        else:
            logging.error("Missing required argument. Please ensure all required fields are provided.")
            logging.error(f"Details: {str(e)}")
    except ValueError as e:
        if "uncommitted changes" in str(e):
            logging.error("The repository has uncommitted changes.")
            logging.error("Please commit or stash your changes before proceeding.")
            logging.error(f"Details: {str(e).split(':', 1)[1].strip()}")
            return
        elif "Directory does not exist" in str(e):
            logging.warning(f"{str(e)}")
            logging.warning("The agent tried to access a directory that doesn't exist.")
            logging.warning("This may be because the component is not yet implemented or is in a different location.")
            logging.debug("Failed directory access: %s", str(e))
            return
        else:
            logging.error(f"Error: {str(e)}")
    except Exception as e:
        logging.error("An unexpected error occurred: %s", str(e))
        import traceback
        logging.error("Traceback: %s", traceback.format_exc())
        logging.error(f"Error: {str(e)}")
        return

if __name__ == "__main__":
    # Set up argument parser and configure using main
    asyncio.run(main()) 