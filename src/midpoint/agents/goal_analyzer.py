"""
Goal Analysis agent implementation.

This module implements the GoalAnalyzer agent that analyzes the status of a goal
and suggests the next appropriate action based on its state and context.
"""

# Fix the memory tools imports - this must be at the top of the file
import sys # Added sys import
import os # Added os import
from pathlib import Path
import logging
import traceback # Added traceback import
import re
import datetime
import json
from typing import List, Dict, Any, Optional, Tuple # Added Tuple

# Assuming goal_cli is in src/midpoint relative to this file's location
# (Path setup remains the same)
script_dir = Path(__file__).parent
src_dir = script_dir.parent
midpoint_dir = src_dir # Adjusted path assuming goal_cli is directly in src/midpoint
if str(midpoint_dir) not in sys.path:
    sys.path.insert(0, str(midpoint_dir))
    
# Assuming scripts dir is two levels up from src
repo_root_from_agents = script_dir.parent.parent
scripts_dir_path = repo_root_from_agents / "scripts"
if str(scripts_dir_path) not in sys.path:
     sys.path.insert(0, str(scripts_dir_path))

# Early logging init
logging.basicConfig(level=logging.INFO, format='%(message)s')

# --- Robust Memory Tools Import --- 
try:
    # Import from the correct path with correct function names
    from midpoint.agents.tools.memory_tools import system_get_repo_path as get_repo_path
    from midpoint.agents.tools.memory_tools import retrieve_recent_memory
    from midpoint.agents.tools.memory_tools import system_store_document as store_memory_document
    
    # Log success
    logging.debug("Successfully imported memory_tools functions")
except Exception as e:
    # Log error and exit - no fallbacks
    error_msg = f"Memory tools import failed: {e}"
    logging.error(error_msg)
    if os.environ.get("DEBUG"):
        print("Exception traceback:")
        traceback.print_exc()
    # We don't want fallbacks, so we will raise the error
    raise ImportError(error_msg)
# --- End Memory Tools Import ---

# Import necessary types and classes
from openai import OpenAI
from .models import State, Goal, TaskContext, MemoryState # Removed SubgoalPlan, AnalysisResult not defined yet
from .tools import initialize_all_tools
from .tools.processor import ToolProcessor
from .tools.registry import ToolRegistry
# Keep tool imports if analyzer LLM might use them for observation
from .tools import (
    list_directory,
    read_file,
    search_code,
    get_current_hash,
    web_search,
    web_scrape,
    retrieve_memory_documents, # Might be redundant if memory_tools import worked
    run_terminal_cmd
)
from .config import get_openai_api_key
# from .utils.logging import log_manager # Commented out if not used
from dotenv import load_dotenv
import subprocess
import time # Added time import for fallback

load_dotenv()

# Global variables to store log file paths
log_file = None 
task_summary_file = None
llm_responses_file = None

def configure_logging(debug=False, quiet=False, log_dir_path="logs"):
    """
    Configure logging for the goal analyzer.
    
    Args:
        debug: Whether to show debug messages
        quiet: Whether to show only warnings and final result
        log_dir_path: Directory to store log files
    """
    # Access global variables
    global log_file, task_summary_file, llm_responses_file
    
    # Ensure log_dir_path is a Path object
    log_dir = Path(log_dir_path)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"goal_analyzer_{timestamp}.log"
    task_summary_file = log_dir / f"task_summary_{timestamp}.log"
    llm_responses_file = log_dir / f"llm_responses_{timestamp}.log"
    
    # Ensure all log files are writable
    ensure_file_writable(log_file)
    ensure_file_writable(task_summary_file)
    ensure_file_writable(llm_responses_file)
    
    # Set up root logger with file and console handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # Capture all logs at the root level
    
    # Remove any existing handlers to prevent duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create file handler for full logging with immediate flush
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    file_handler.setLevel(logging.DEBUG)  # Log everything to file
    
    # Create console handler with appropriate level
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(message)s'))
    if debug:
        console_handler.setLevel(logging.DEBUG)
    elif quiet:
        console_handler.setLevel(logging.WARNING)
    else:
        console_handler.setLevel(logging.INFO)
    
    # Add handlers to root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Create task summary file with header
    with open(task_summary_file, "w") as f:
        f.write(f"Task Summary Log - {timestamp}\n")
        f.write("=" * 50 + "\n\n")
        f.flush()  # Force immediate write
    
    # Configure the dedicated LLM interactions logger
    llm_logger = logging.getLogger('llm_interactions')
    llm_logger.setLevel(logging.DEBUG)  # Capture all LLM interaction details
    
    # Remove any existing handlers
    for handler in llm_logger.handlers[:]:
        llm_logger.removeHandler(handler)
    
    # Create file handler specifically for LLM responses with immediate flush
    llm_file_handler = logging.FileHandler(llm_responses_file)
    llm_file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    llm_file_handler.setLevel(logging.DEBUG)
    
    # Enable immediate flushing of log messages for more reliable logging
    class ImmediateFlushingHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
    
    # Replace with immediate flushing handler
    llm_immediate_handler = ImmediateFlushingHandler(llm_responses_file)
    llm_immediate_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    llm_immediate_handler.setLevel(logging.DEBUG)
    
    # Add the handler to the LLM logger ONLY
    llm_logger.addHandler(llm_immediate_handler)
    
    # Prevent LLM logs from propagating to the root logger's handlers
    llm_logger.propagate = False
    
    # Write an initial message to verify the file is writable
    try:
        with open(llm_responses_file, "a") as f:
            f.write("LLM Responses Log initialized at " + timestamp + "\n")
            f.flush()  # Force immediate write
    except Exception as e:
        logging.error(f"Error writing to LLM responses file: {e}")
    
    # Test log message directly to the LLM logger
    llm_logger.debug("LLM logger initialized")
    
    logging.info("Logging configured")
    return log_file, task_summary_file, llm_responses_file

class GoalAnalyzer:
    """
    Goal Analysis agent implementation.
    
    This agent analyzes the status of a goal and suggests the next appropriate
    action (e.g., decompose, create task, mark complete, give up).
    """
    
    def __init__(self, model: str = "gpt-4o-mini", max_history_entries: int = 5):
        """
        Initialize the goal analyzer.
        
        Args:
            model: The model to use for generation
            max_history_entries: Maximum number of history entries to consider
        """
        self.logger = logging.getLogger('GoalAnalyzer')
        self.model = model
        self.max_history_entries = max_history_entries
        # Re-add ToolProcessor initialization here
        api_key = get_openai_api_key() # Assuming this is available
        if not api_key: raise ValueError("OpenAI API key not found")
        self.client = OpenAI(api_key=api_key)
        initialize_all_tools() # Ensure tools are initialized
        self.tool_processor = ToolProcessor(self.client)
        self.tools = ToolRegistry.get_tool_schemas() # Also re-add tool loading
        # Generate the new analysis-specific system prompt
        self.system_prompt = self._generate_system_prompt()
        self.logger.info("GoalAnalyzer initialized.")
        
        # Ensure memory_state is created even if hash/path are initially None
        self.memory_state = MemoryState(memory_hash=None, repository_path=None)
        
    def _generate_system_prompt(self) -> str:
        """Generate the system prompt for the Goal Analyzer."""
        tool_descriptions_text = self._get_tool_descriptions()

        # New system prompt focused on analysis and suggesting next commands
        system_prompt = f"""You are an expert project manager AI responsible for analyzing the status of development goals.
Your task is to review the provided context for a specific goal and decide the most appropriate next action.

Context Provided:
- Goal details (description, validation criteria, parent)
- Current state (repository path, git hash, memory hash)
- Status of child goals/tasks (if any)
- History of completed tasks and merged subgoals for this goal
- Recent memory context relevant to this goal
- Last execution status (if any)
- Failed attempts history for this goal (if any)
- Merged subgoals history and their outcomes

Your Analysis Steps:
1. OBSERVE: Carefully review all the provided context. Pay special attention to:
   - Last execution status and any failures
   - Validation status and any failed validation criteria
   - Failed attempts history
   - Available tools and their capabilities
   - Parent goal context and requirements

2. ORIENT: Assess the goal's progress and feasibility. Consider:
   - Is the goal complete?
   - Did the goal pass validation? If not, what specific criteria failed and why?
   - Is the current state significantly different from the initial state?
   - Are children complete?
   - Does the memory suggest recent failures or successes?
   - Are required tools or capabilities available?
   - Have similar attempts failed before?

3. DECIDE: Based on your analysis, choose ONE of the following actions:
    - "decompose": Recommended when the goal needs new subgoals or subtasks. Use this when:
       a) The goal has no children tasks/subgoals yet and is too complex for a single task
       b) The goal has children tasks that are completed but failed validation, requiring new approaches
       c) Some children tasks are completed and validated, but remaining work is still complex
       d) Previous decomposition attempts were unsuccessful and a new approach is needed
    
    - "execute": Recommended only when the remaining work is very specific, straightforward, and can be completed in a single atomic step.
    
    - "validate": Recommended only when sufficient children tasks have been completed AND successfully validated, to potentially satisfy all the requirements for the goal.
    
    - "mark_complete": Recommended when the goal is already validated or clearly finished based on context.
    
    - "update_parent": Recommended when a child was completed but the parent state doesn't reflect it.
    
    - "give_up": Recommended when:
       a) The goal is ill-defined or ambiguous
       b) The goal is fundamentally impossible with current tools/capabilities
       c) The history of failed attempts shows a lack of variability and latest attempts have been repetitive and uncreative

4. OUTPUT: Provide your decision as a JSON object containing ONLY the "action" (string) and "justification" (string) fields.
   The justification MUST be detailed. For `decompose`, explain *what specific aspects* make the goal too complex for a single task.
   For `execute`, explain *why* it's simple enough.
   For `give_up`, explain *what went wrong or why the goal is impossible/irrelevant* (e.g., required tools unavailable, repeated failures, description ambiguous).

Available Tools (for observation only):
{tool_descriptions_text}

IMPORTANT: Return ONLY raw JSON, like {{\"action\": \"decompose\", \"justification\": \"The goal is complex and requires further breakdown based on X.\"}}. Do not wrap it in markdown.
"""
        return system_prompt

    def _get_tool_descriptions(self):
        return ""

    def analyze_goal_state(self, context: TaskContext, setup_logging: bool = False, debug: bool = False, quiet: bool = False) -> Dict[str, Any]:
        """Analyze the current state of the goal and suggest the next action."""
        # Global variables to ensure we're using the same log files throughout
        global log_file, task_summary_file, llm_responses_file
        
        if setup_logging:
            # Configure logging only once and store the file paths
            log_file, task_summary_file, llm_responses_file = configure_logging(debug, quiet)
            
        # Create dedicated LLM logger - ensure we're always using the same logger instance
        llm_logger = logging.getLogger('llm_interactions')
        
        # Ensure handler is set up even if configure_logging wasn't called
        # This is important because sometimes the analyze_goal_state might be called 
        # without setting up logging first
        if not llm_logger.handlers and llm_responses_file:
            handler = logging.FileHandler(llm_responses_file)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            llm_logger.handlers = [handler]
            llm_logger.propagate = False
        
        # Force a test log message to verify the logger is working
        log_with_timestamp("========== STARTING GOAL ANALYSIS ==========", llm_logger)
        log_with_timestamp(f"Goal: {context.goal.description}", llm_logger)

        logging.info("Analyzing state for goal: %s", context.goal.description)
        # Log task summary? Maybe rename log_task_summary or create log_analysis_summary

        # Validate memory state
        if not context.memory_state: raise ValueError("Memory state is required for goal analysis")
        memory_hash = getattr(context.memory_state, "memory_hash", None)
        memory_path = getattr(context.memory_state, "repository_path", None)
        if not memory_hash or not memory_path: raise ValueError("Memory state must have hash and path")
        logging.info(f"Memory state: hash={memory_hash[:8] if memory_hash else None}, path={memory_path}")
        log_with_timestamp(f"Memory state: hash={memory_hash[:8] if memory_hash else None}, path={memory_path}", llm_logger)

        # Validate inputs
        if not context.goal: raise ValueError("No goal provided")
        if not context.state.repository_path: raise ValueError("Repo path not provided")
        goal_id = context.metadata.get("goal_id")
        if not goal_id: raise ValueError("goal_id missing from context metadata")

        llm_logger.debug(f"==== CONTEXT ====")
        llm_logger.debug(f"Goal: {context.goal.description}")
        llm_logger.debug(f"Goal ID: {goal_id}")
        llm_logger.debug(f"Repository path: {context.state.repository_path}")
        llm_logger.debug(f"Validation criteria: {context.goal.validation_criteria}")
        llm_logger.debug(f"Metadata: {json.dumps(context.metadata, default=str)}")
        
        # Prepare the user prompt using the new function
        # Pass necessary context elements directly
        try:
            log_with_timestamp("Creating analysis user prompt...", llm_logger)
            user_prompt = self._create_analysis_user_prompt(
                context, 
                goal_id, 
                memory_hash, 
                memory_path
            )
            # Log the complete user prompt that will be sent to the LLM
            log_with_timestamp("==== FULL USER PROMPT ====", llm_logger)
            llm_logger.debug(user_prompt)
            log_with_timestamp("==== END USER PROMPT ====", llm_logger)
        except ValueError as ve:
            error_msg = f"Failed to create analysis user prompt: {str(ve)}"
            logging.error(error_msg)
            llm_logger.error(error_msg)
            return {
                "action": "error",
                "justification": f"Analysis failed during preparation: {str(ve)}",
                "metadata": {"error_type": "preparation_error", "details": str(ve)}
            }
        except Exception as e:
            error_msg = f"Unexpected error creating analysis user prompt: {str(e)}"
            logging.error(error_msg)
            llm_logger.error(error_msg)
            return {
                "action": "error",
                "justification": "Unexpected error during analysis preparation",
                "metadata": {"error_type": "unexpected_error", "details": str(e)}
            }

        messages = [{"role": "system", "content": self._generate_system_prompt()}]
        # Add memory retrieved in _create_analysis_user_prompt (how to pass it here?)
        # Let's assume _create_analysis_user_prompt returns prompt string AND memory docs string
        # For now, just add the prompt. Memory integration needs refinement.
        # The prompt generation now handles adding memory context internally
        messages.append({"role": "user", "content": user_prompt})

        # Log the system prompt as well
        log_with_timestamp("==== SYSTEM PROMPT ====", llm_logger)
        llm_logger.debug(self._generate_system_prompt())
        log_with_timestamp("==== END SYSTEM PROMPT ====", llm_logger)

        tool_usage = []
        max_retries = 1 # Allow one retry on JSON parsing failure
        attempts = 0
        final_output = None
        raw_content = ""

        while attempts <= max_retries:
            attempts += 1
            # --- Start Try Block for this attempt ---
            try:
                # Log request messages
                try:
                    log_with_timestamp(f"==== FINAL LLM MESSAGES (Attempt {attempts}) ====", llm_logger)
                    llm_logger.debug(json.dumps(self._serialize_messages(messages), indent=2))
                    log_with_timestamp("==== END FINAL LLM MESSAGES ====", llm_logger)
                except Exception as log_e:
                    log_with_timestamp(f"Failed to serialize request messages for logging: {str(log_e)}", llm_logger)

                # Get analysis from the model, allowing tool use
                log_with_timestamp("Sending request to LLM...", llm_logger)
                message, tool_calls = self.tool_processor.run_llm_with_tools(
                    messages, model=self.model,
                    validate_json_format=True, # Expecting JSON output
                    max_tokens=1000 # Analysis output should be shorter
                )
                log_with_timestamp("Received response from LLM", llm_logger)

                # Record tool usage if any occurred in this attempt
                if tool_calls:
                    log_with_timestamp("==== TOOL USAGE ====", llm_logger)
                    for tool_call in tool_calls: 
                        tool_usage.append(str(tool_call))
                        try:
                            tool_name = tool_call.function.name if hasattr(tool_call, 'function') else tool_call.get('function', {}).get('name', 'unknown_tool')
                            tool_args = tool_call.function.arguments if hasattr(tool_call, 'function') else tool_call.get('function', {}).get('arguments', '{}')
                            log_with_timestamp(f"Tool: {tool_name}", llm_logger)
                            log_with_timestamp(f"Arguments: {tool_args}", llm_logger)
                        except Exception as tool_e:
                            log_with_timestamp(f"Failed to log tool call: {str(tool_e)}", llm_logger)
                    log_with_timestamp("==== END TOOL USAGE ====", llm_logger)
                
                # Parse the model's final response (expecting JSON)
                if isinstance(message, list): content = message[-1].get('content', '')
                else: content = message.get('content') if isinstance(message, dict) else message.content
                raw_content = content # Store raw content for potential error message

                logging.debug(f"Raw model response (Attempt {attempts}): {content}")
                log_with_timestamp(f"==== LLM RAW RESPONSE CONTENT (Attempt {attempts}) ====", llm_logger)
                llm_logger.debug(content) # Log the raw string content
                log_with_timestamp(f"==== END LLM RAW RESPONSE CONTENT (Attempt {attempts}) ====", llm_logger)
                
                # --- Start Try Block for JSON Parsing ---
                try:
                    log_with_timestamp("Parsing LLM response...", llm_logger)
                    output_data = json.loads(content)
                     # Validate expected fields
                    if not all(key in output_data for key in ["action", "justification"]):
                        # If missing, try extracting from markdown code blocks
                        log_with_timestamp("Missing expected fields, trying to extract from code blocks...", llm_logger)
                        json_match = re.search(r'```(?:json)?\s*(\{.+?\})?\s*```', content, re.DOTALL)
                        if json_match:
                            try:
                                output_data = json.loads(json_match.group(1))
                                if not all(key in output_data for key in ["action", "justification"]):
                                    raise ValueError("Extracted JSON missing 'action' or 'justification'")
                            except (json.JSONDecodeError, ValueError) as inner_e:
                                 logging.error(f"Failed to parse extracted JSON or validate fields: {inner_e}")
                                 log_with_timestamp(f"Failed to parse extracted JSON or validate fields: {inner_e}", llm_logger)
                                 raise ValueError("LLM response JSON invalid or missing fields even after extraction.")
                        else:
                             raise ValueError("LLM response missing 'action' or 'justification' and no JSON block found.")

                    # Handle give_up action
                    if output_data.get("action") == "give_up":
                        log_with_timestamp("Goal analyzer recommended giving up", llm_logger)
                        
                        # Load the goal file to get parent information
                        goal_path = Path(".goal")
                        goal_file = goal_path / f"{goal_id}.json"
                        if goal_file.exists():
                            with open(goal_file, 'r') as f:
                                goal_data = json.load(f)
                            
                            # Get parent goal ID
                            parent_goal_id = goal_data.get("parent_goal")
                            if parent_goal_id:
                                # Load parent goal file
                                parent_file = goal_path / f"{parent_goal_id}.json"
                                if parent_file.exists():
                                    with open(parent_file, 'r') as f:
                                        parent_data = json.load(f)
                                    
                                    # Initialize failed_attempts_history if it doesn't exist
                                    if "failed_attempts_history" not in parent_data:
                                        parent_data["failed_attempts_history"] = []
                                    
                                    # Add failure record
                                    failure_record = {
                                        "attempted_goal_id": goal_id,
                                        "attempted_goal_description": goal_data.get("description", "N/A"),
                                        "failure_reason": output_data.get("justification", "No reason provided"),
                                        "failure_timestamp": datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                                    }
                                    parent_data["failed_attempts_history"].append(failure_record)
                                    
                                    # Save updated parent data
                                    with open(parent_file, 'w') as f:
                                        json.dump(parent_data, f, indent=2)
                                    
                                    log_with_timestamp(f"Updated parent goal {parent_goal_id} with failure history", llm_logger)
                        
                        # Update current goal file to mark it as given up
                        goal_data["status"] = "given_up"
                        if "last_analysis" not in goal_data:
                            goal_data["last_analysis"] = {}
                        goal_data["last_analysis"]["action"] = "give_up"
                        goal_data["last_analysis"]["justification"] = output_data.get("justification", "No justification provided")
                        goal_data["last_analysis"]["timestamp"] = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        
                        with open(goal_file, 'w') as f:
                            json.dump(goal_data, f, indent=2)
                        
                        log_with_timestamp(f"Marked goal {goal_id} as given up", llm_logger)
                    
                    # Create the final result dictionary
                    final_output = {
                        "action": output_data["action"],
                        "justification": output_data["justification"],
                        "metadata": { "raw_response": content, "tool_usage": tool_usage }
                    }
                    logging.info(f"✅ Analysis complete. Suggested action: {final_output['action']}")
                    logging.debug(f"Justification: {final_output['justification']}")
                    log_with_timestamp("==== PARSED OUTPUT ====", llm_logger)
                    log_with_timestamp(f"Action: {final_output['action']}", llm_logger)
                    log_with_timestamp(f"Justification: {final_output['justification']}", llm_logger)
                    log_with_timestamp("==== END PARSED OUTPUT ====", llm_logger)
                    break # Success, exit retry loop
                
                # --- Handle JSON Parsing Errors ---
                except (json.JSONDecodeError, ValueError) as e:
                    error_msg = f"Error processing model response (Attempt {attempts}): {str(e)}"
                    logging.error(error_msg)
                    log_with_timestamp(error_msg, llm_logger)
                    log_with_timestamp(f"Raw response was: {content}", llm_logger)
                    
                    if attempts > max_retries:
                        log_with_timestamp("Max retries reached. Failing analysis.", llm_logger)
                        # Handle error - return a default "error" action
                        final_output = {
                            "action": "error",
                            "justification": f"Failed to process LLM analysis response after {attempts} attempts: {str(e)}",
                            "metadata": { "raw_response": content, "tool_usage": tool_usage }
                        }
                        break # Exit loop after final failure
                    else:
                        # Prepare for retry: Add error message to prompt history
                        log_with_timestamp("Preparing to retry LLM call...", llm_logger)
                        # Add the failed assistant response AND the user error message for context
                        messages.append({"role": "assistant", "content": raw_content}) 
                        messages.append({"role": "user", "content": f"Your previous response could not be parsed or was invalid: {str(e)}. Please provide ONLY the required JSON object with 'action' and 'justification' fields. Do not include any other text or markdown formatting."})
                        # No 'continue' needed here, loop will naturally continue

            # --- Handle Unexpected Errors during LLM call/processing for this attempt ---
            except Exception as e:
                 error_msg = f"Unexpected error during LLM call or processing (Attempt {attempts}): {str(e)}"
                 logging.error(error_msg)
                 log_with_timestamp(error_msg, llm_logger)
                 log_with_timestamp(f"Traceback: {traceback.format_exc()}", llm_logger)
                 # Return error action immediately on unexpected failure
                 final_output = {
                     "action": "error",
                     "justification": f"Goal analysis failed with unexpected error: {str(e)}",
                     "metadata": {"error_type": "llm_call_error", "details": str(e), "tool_usage": tool_usage}
                 }
                 break # Exit loop immediately on unexpected error
        
        # --- After the loop --- 
        # (Memory saving call removed previously)
            
        log_with_timestamp("========== GOAL ANALYSIS COMPLETE ==========", llm_logger)

        # Ensure final_output has a value (should always be set by the loop)
        if final_output is None:
            logging.error("Final output was unexpectedly None after retry loop.")
            final_output = {
                "action": "error",
                "justification": "Analysis failed unexpectedly after retry loop.",
                "metadata": { "raw_response": raw_content, "tool_usage": tool_usage }
            }

        return final_output # Return the dictionary

    def _create_analysis_user_prompt(self, context: TaskContext, goal_id: str, memory_hash: str, memory_repo_path: str) -> str:
        """Create the user prompt for the Goal Analyzer LLM."""
        llm_logger = logging.getLogger('llm_interactions')
        llm_logger.debug(f"==== BUILDING ANALYSIS USER PROMPT ====")
        llm_logger.debug(f"Goal ID: {goal_id}")
        llm_logger.debug(f"Memory hash: {memory_hash[:8] if memory_hash else None}")
        llm_logger.debug(f"Memory repo path: {memory_repo_path}")
        
        prompt_lines = []
        goal_data = {}
        children_details = []
        memory_content = "No recent memory retrieved."

        # --- 1. Load Goal Data ---
        goal_path = Path(".goal") # Assuming .goal dir exists
        goal_file = goal_path / f"{goal_id}.json"
        if goal_file.exists():
            try:
                with open(goal_file, 'r') as f:
                    goal_data = json.load(f)
                logging.info(f"Loaded goal data for {goal_id}")
                llm_logger.debug(f"Loaded goal data from {goal_file}")
                llm_logger.debug(f"Goal data: {json.dumps(goal_data, indent=2, default=str)[:1000]}...")
            except Exception as e:
                error_msg = f"Failed to load goal file {goal_file}: {e}"
                logging.error(error_msg)
                llm_logger.error(error_msg)
                prompt_lines.append(f"Error: Could not load goal file {goal_file}")
        else:
            warning_msg = f"Goal file not found: {goal_file}"
            logging.warning(warning_msg)
            llm_logger.warning(warning_msg)
            prompt_lines.append(f"Warning: Goal file {goal_file} not found.")

        # --- 2. Get Children Details ---
        try:
            # Local function to get children details - avoid circular imports
            def get_children_details(parent_goal_id: str) -> List[Dict[str, Any]]:
                """Get details of direct children (subgoals and tasks) for a given parent ID."""
                llm_logger.debug(f"Looking for children of parent ID: {parent_goal_id}")
                goal_path = Path(".goal")
                children_details = []
                try:
                    for file_path in goal_path.glob("*.json"):
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                        
                        # Check parent match (case-insensitive)
                        parent_match = False
                        file_parent = data.get("parent_goal", "")
                        if file_parent:
                            if file_parent.upper() == parent_goal_id.upper() or \
                              file_parent.upper() == f"{parent_goal_id.upper()}.json":
                                parent_match = True

                        if parent_match:
                            child_info = {
                                "goal_id": data.get("goal_id", "Unknown"),
                                "description": data.get("description", "N/A"),
                                "is_task": data.get("is_task", False),
                                "complete": data.get("complete", False) or data.get("completed", False),
                                # Add other relevant fields if needed, e.g., last validation score
                                "validation_score": data.get("validation_status", {}).get("last_score")
                            }
                            children_details.append(child_info)
                            llm_logger.debug(f"Found child: {child_info['goal_id']} - {child_info['description'][:50]}...")
                            
                except Exception as e:
                    error_msg = f"Error getting children details for {parent_goal_id}: {e}"
                    logging.error(error_msg)
                    llm_logger.error(error_msg)
                    raise ValueError(f"Failed to get children details for {parent_goal_id}: {e}")

                # Sort by goal ID for consistent order
                children_details.sort(key=lambda x: x["goal_id"])
                return children_details
                
            # Use the local function
            children_details = get_children_details(goal_id)
            
            # Validate children_details is a valid list
            if children_details is None:
                raise ValueError(f"get_children_details returned None for goal {goal_id}")
            if not isinstance(children_details, list):
                raise ValueError(f"get_children_details returned non-list type: {type(children_details)}")
                
            logging.info(f"Found {len(children_details)} children for goal {goal_id}")
            llm_logger.debug(f"Found {len(children_details)} children for goal {goal_id}")
        except Exception as e:
            error_msg = f"Failed to get children details for {goal_id}: {e}"
            logging.error(error_msg)
            llm_logger.error(error_msg)
            prompt_lines.append("Error: Could not get children details.")
            raise ValueError(f"Failed to get children details for goal {goal_id}: {e}")

        # --- 3. Retrieve Recent Memory ---
        if memory_hash and memory_repo_path:
            try:
                # Import here to avoid potential top-level import issues
                from midpoint.agents.tools.memory_tools import retrieve_recent_memory
                llm_logger.debug(f"Retrieving recent memory with hash={memory_hash[:8]} from {memory_repo_path}")
                total_chars, memory_documents = retrieve_recent_memory(
                    memory_hash=memory_hash,
                    char_limit=10000, # Configurable?
                    repo_path=memory_repo_path
                )
                if memory_documents:
                    memory_context_lines = ["## Recent Memory Context"]
                    for path, content, timestamp in memory_documents:
                        filename = os.path.basename(path)
                        ts_str = datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        memory_context_lines.append(f"### Memory: {filename} ({ts_str})")
                        # Limit content length per document shown to LLM
                        max_len = 1000
                        truncated_content = content[:max_len] + "..." if len(content) > max_len else content
                        memory_context_lines.append(f"```\n{truncated_content}\n```") # Added missing newline
                        llm_logger.debug(f"Added memory document: {filename} ({ts_str}) - {len(content)} chars")
                    memory_content = "\n".join(memory_context_lines)
                    logging.info(f"Retrieved {len(memory_documents)} memory documents ({total_chars} chars).")
                    llm_logger.debug(f"Retrieved {len(memory_documents)} memory documents ({total_chars} chars).")
                else:
                    logging.info("No recent memory documents found.")
                    llm_logger.debug("No recent memory documents found.")
                    memory_content = "No recent memory documents found."

            except ImportError:
                 error_msg = "Could not import retrieve_recent_memory from midpoint.agents.tools.memory_tools."
                 logging.error(error_msg)
                 llm_logger.error(error_msg)
                 memory_content = "Error: Could not retrieve memory context (import failed)."
            except Exception as e:
                error_msg = f"Error retrieving memory context: {str(e)}"
                logging.error(error_msg)
                llm_logger.error(error_msg)
                memory_content = f"Error retrieving memory context: {str(e)}"
        else:
            warning_msg = "Memory hash or repo path missing, cannot retrieve memory context."
            logging.warning(warning_msg)
            llm_logger.warning(warning_msg)
            memory_content = "Memory context unavailable (missing hash or path)."

        # --- 4. Format Prompt ---
        llm_logger.debug(f"Formatting final analysis prompt")
        prompt_lines.append(f"Goal Analysis Request for Goal [{goal_id}]")
        prompt_lines.append("="*20)

        # Goal Details
        prompt_lines.append("## Goal Details")
        prompt_lines.append(f"ID: {goal_id}")
        prompt_lines.append(f"Description ({goal_id}): {goal_data.get('description', 'N/A')}")
        prompt_lines.append(f"Parent Goal ID: {goal_data.get('parent_goal', 'None')}")
        prompt_lines.append(f"Type: {'Task' if goal_data.get('is_task') else 'Goal'}")
        val_criteria = goal_data.get('validation_criteria', [])
        prompt_lines.append("Validation Criteria:")
        if val_criteria:
             for i, crit in enumerate(val_criteria, 1): prompt_lines.append(f"  {i}. {crit}")
        else: prompt_lines.append("  None")

        # Current State
        prompt_lines.append("\n## Current State")
        current_state = goal_data.get('current_state', {})
        initial_state = goal_data.get('initial_state', {}) # For comparison
        prompt_lines.append(f"Repository Path: {current_state.get('repository_path', 'N/A')}")
        prompt_lines.append(f"Git Hash: {current_state.get('git_hash', 'N/A')}")
        prompt_lines.append(f"Memory Hash: {current_state.get('memory_hash', 'N/A')}")
        prompt_lines.append(f"Memory Repository Path: {current_state.get('memory_repository_path', 'N/A')}")
        # Add comparison to initial state if helpful
        if initial_state.get('git_hash') != current_state.get('git_hash'):
            prompt_lines.append(f"(Initial Git Hash: {initial_state.get('git_hash', 'N/A')})")
        if initial_state.get('memory_hash') != current_state.get('memory_hash'):
             prompt_lines.append(f"(Initial Memory Hash: {initial_state.get('memory_hash', 'N/A')})")

        # Children Status
        prompt_lines.append("\n## Children Status")
        if children_details:
            for child in children_details:
                status = "Complete" if child.get('complete') else "Incomplete"
                val_score = child.get('validation_score')
                score_str = f" (Validation: {val_score:.1%})" if val_score is not None else ""
                prompt_lines.append(f"- {child.get('goal_id')}: {child.get('description', 'N/A')} [{status}{score_str}]")
        else:
            prompt_lines.append("No children found.")

        # History
        prompt_lines.append("\n## History for this Goal")
        completed_tasks = goal_data.get('completed_tasks', [])
        merged_subgoals = goal_data.get('merged_subgoals', [])
        failed_attempts = goal_data.get('failed_attempts_history', []) # Load failed attempts history here

        if completed_tasks:
            prompt_lines.append(f"Completed Tasks ({len(completed_tasks)}):")
            for task in completed_tasks[:5]: # Limit history shown
                prompt_lines.append(f"- {task.get('task_id')}: {task.get('description', 'N/A')} (at {task.get('timestamp')})")
            if len(completed_tasks) > 5: prompt_lines.append("  ...")
        else: prompt_lines.append("No completed tasks recorded.")

        if merged_subgoals:
            prompt_lines.append(f"Merged Subgoals ({len(merged_subgoals)}):")
            for merge in merged_subgoals[:5]: # Limit history shown
                 prompt_lines.append(f"- {merge.get('subgoal_id')} (at {merge.get('merge_time')})")
            if len(merged_subgoals) > 5: prompt_lines.append("  ...")
        else: prompt_lines.append("No subgoals merged.")

        # Now this check is safe
        if failed_attempts:
            prompt_lines.append(f"Failed Attempts History ({len(failed_attempts)}):")
            # Display the most recent failures first
            for attempt in reversed(failed_attempts[-50:]): # Show last 5 failures
                ts = attempt.get('failure_timestamp', 'N/A')
                reason = attempt.get('failure_reason', 'No reason provided')
                
                # Determine the ID of the failed attempt
                failed_id = attempt.get('failed_child_goal_id') or attempt.get('attempted_goal_id') or "UnknownID"
                
                # Extract description and criteria (new fields)
                description = attempt.get('failed_child_description', 'No description provided')
                criteria = attempt.get('failed_child_validation_criteria', [])
                criteria_summary = f"{len(criteria)} defined" if isinstance(criteria, list) else "Unknown criteria format"

                # Format the output lines
                prompt_lines.append(f"- Failure Record for {failed_id} at {ts}:")
                prompt_lines.append(f"    Description: {description}")
                prompt_lines.append(f"    Criteria: {criteria_summary}")
                prompt_lines.append(f"    Reason: {reason[:1000]}...")
                
            if len(failed_attempts) > 50: 
                prompt_lines.append("  ... (more past failures exist)")
        else:
            prompt_lines.append("No failed attempts recorded for this goal.")

        # Last Execution Status from goal data
        prompt_lines.append("\n## Last Execution Status")
        last_execution = goal_data.get('last_execution')
        if last_execution and isinstance(last_execution, dict):
            timestamp = last_execution.get("timestamp", "N/A")
            success = last_execution.get("success", "Unknown")
            summary = last_execution.get("summary", "No summary provided.")
            prompt_lines.append(f"Timestamp: {timestamp}")
            prompt_lines.append(f"Success: {success}")
            prompt_lines.append(f"Summary: {summary}")
        else:
            prompt_lines.append("No specific last execution status recorded.")

        # Validation Status - Add detailed validation information for better analysis
        prompt_lines.append("\n## Validation Status")
        validation_status = goal_data.get('validation_status')
        if validation_status and isinstance(validation_status, dict):
            prompt_lines.append(f"Last Validated: {validation_status.get('last_validated', 'Unknown')}")
            
            # Get score and format as percentage
            score = validation_status.get('last_score')
            if score is not None:
                score_pct = score * 100  # Convert to percentage
                prompt_lines.append(f"Validation Score: {score_pct:.1f}% (Threshold: {goal_data.get('success_threshold', 80):.1f}%)")
            else:
                prompt_lines.append("Validation Score: Not available")
                
            prompt_lines.append(f"Validated By: {validation_status.get('last_validated_by', 'Unknown')}")
            prompt_lines.append(f"Git Hash at Validation: {validation_status.get('last_git_hash', 'Unknown')}")
            
            # Include overall reasoning
            reasoning = validation_status.get('reasoning')
            if reasoning:
                prompt_lines.append(f"Validation Reasoning: {reasoning}")
            
            # Include detailed criteria results
            criteria_results = validation_status.get('criteria_results')
            if criteria_results and isinstance(criteria_results, list):
                prompt_lines.append("\nDetailed Validation Criteria Results:")
                for i, result in enumerate(criteria_results, 1):
                    passed = result.get('passed', False)
                    status_icon = "✅" if passed else "❌"
                    criterion = result.get('criterion', f"Criterion {i}")
                    prompt_lines.append(f"  {status_icon} {criterion}")
                    
                    # For failed criteria, show reasoning and evidence
                    if not passed:
                        reasoning = result.get('reasoning')
                        if reasoning:
                            prompt_lines.append(f"    Reason for failure: {reasoning}")
                        
                        evidence = result.get('evidence')
                        if evidence and isinstance(evidence, list):
                            prompt_lines.append(f"    Evidence:")
                            for item in evidence:
                                prompt_lines.append(f"      - {item}")
            else:
                prompt_lines.append("No detailed validation criteria results available.")
        else:
            prompt_lines.append("No validation results available for this goal.")

        # --- 5. Code Inspection ---
        llm_logger.debug("Performing code inspection for relevant files")
        prompt_lines.append("\n## Code Inspection")
        prompt_lines.append("Automatic code inspection is disabled. Use available tools (e.g., read_file, search_code) if you need to inspect files.")

        # Memory Context
        prompt_lines.append("\n" + memory_content)

        # Final Instruction
        prompt_lines.append("\n" + "="*20)
        prompt_lines.append("Based on ALL the context above, analyze the goal's status.")
        prompt_lines.append("If necessary, use tools to observe the current state further.")
        prompt_lines.append("IMPORTANT: For validation or completion decisions, verify that code implementation correctly meets requirements.")
        
        # Add specific instructions for handling failed validations
        validation_status = goal_data.get('validation_status', {})
        validation_score = validation_status.get('last_score')
        success_threshold = goal_data.get('success_threshold', 0.8)
        
        if validation_score is not None and validation_score < success_threshold:
            prompt_lines.append("\nIMPORTANT VALIDATION FAILURE INSTRUCTIONS:")
            prompt_lines.append("This goal has FAILED VALIDATION with a score below the success threshold.")
            prompt_lines.append("Review the detailed Validation Status section above.")
            prompt_lines.append("Focus specifically on the failed criteria and their reasoning to understand what needs to be fixed.")
            prompt_lines.append("If the issues are specific and straightforward to fix, recommend 'execute' with a clear justification of what needs to be fixed.")
            prompt_lines.append("If the validation failure indicates fundamental flaws requiring a new approach, recommend 'decompose' with justification.")
        
        prompt_lines.append("\nFor goals with no children, carefully consider if `execute` might be suitable if the goal represents a single, concrete action, even if it involves multiple small steps.")
        prompt_lines.append("Then, provide your suggested next action and a DETAILED justification in the required JSON format.")

        final_prompt = "\n".join(prompt_lines)
        llm_logger.debug(f"==== FINAL PROMPT STATS ====")
        llm_logger.debug(f"Prompt length: {len(final_prompt)} characters, ~{len(prompt_lines)} lines")
        llm_logger.debug(f"==== END PROMPT BUILDING ====")
        
        return final_prompt

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

    def create_top_goal_file(self, context: TaskContext, logs_dir="logs") -> str:
        """
        [DEPRECATED] Create a subgoal file for a top-level goal.
        Args:
            context: The current task context containing the goal
            logs_dir: Directory to store the goal file
            
        Returns:
            The filename of the created subgoal file
            
        NOTE: This method is maintained for backward compatibility and testing.
        In production, goal files should be managed by goal_cli.py.
        """
        logging.warning("Using GoalAnalyzer.create_top_goal_file which is deprecated. Goal files should be created by goal_cli.py")
        # Simplified implementation for stub
        goal_id = self.generate_goal_id(logs_dir=logs_dir)
        filename = f"{goal_id}.json"
        file_path = Path(logs_dir) / filename
        try:
            with open(file_path, 'w') as f:
                 json.dump({"goal_id": goal_id, "description": context.goal.description}, f)
            return filename
        except Exception as e:
             logging.error(f"Stub create_top_goal_file failed: {e}")
             return "error.json"

# Other helper functions (validate_repository_state, list_subgoal_files, load_input_file, is_git_ancestor)
# remain the same for now.
# Need to import these or ensure they are available if used:
from .models import State, Goal, TaskContext, MemoryState # Ensure models are imported
# Assuming these helpers exist in the same directory or are properly imported:
def validate_repository_state(repo_path, git_hash=None, skip_clean_check=False): 
    logging.debug(f"Stub validate_repository_state called for {repo_path}")
    pass 
def list_subgoal_files(logs_dir="logs"): 
    logging.debug("Stub list_subgoal_files called")
    return []
def load_input_file(input_file: str, context: TaskContext) -> None: 
    logging.debug(f"Stub load_input_file called for {input_file}")
    # Needs actual implementation to modify context based on file
    pass 
def is_git_ancestor(repo_path: str, ancestor_hash: str, descendant_hash: str) -> bool: 
    logging.debug("Stub is_git_ancestor called")
    return True # Assume true for stub

# Wrapper function (re-adding the definition)
def analyze_goal(
    repo_path: str, goal: str, validation_criteria: List[str] = None,
    parent_goal_id: str = None, goal_id: str = None, memory_hash: str = None,
    memory_repo_path: str = None, debug: bool = False, quiet: bool = False,
    bypass_validation: bool = False, logs_dir: str = "logs",
    input_file: Optional[str] = None
) -> Dict[str, Any]:
    """Analyze a goal's state using the GoalAnalyzer agent."""
    # Access global variables
    global log_file, task_summary_file, llm_responses_file
    
    # Always configure logging at the start to ensure log files are properly set up
    configure_logging(debug, quiet, logs_dir)
    logging.info(f"Analyzing goal: {goal}")
    
    # Additional test message to LLM logger
    llm_logger = logging.getLogger('llm_interactions')
    llm_logger.debug(f"Goal analysis initiated for: {goal}")
    llm_logger.debug(f"Using log files: {log_file}, {task_summary_file}, {llm_responses_file}")

    # Initial state setup - don't catch ImportError
    from .tools.git_tools import get_current_hash 
    current_git_hash = get_current_hash(repo_path) if repo_path else get_current_hash()
    if not current_git_hash:
         raise RuntimeError("Failed to get current git hash.")
    logging.info(f"Current git hash: {current_git_hash[:8]}")

    # Initialize state and context objects
    state = State(
        git_hash=current_git_hash, repository_path=repo_path,
        description="Initial state before goal analysis",
        memory_hash=memory_hash, memory_repository_path=memory_repo_path
    )
    memory_state = MemoryState(memory_hash=memory_hash, repository_path=memory_repo_path)

    context = TaskContext(
        iteration=0, 
        goal=Goal(description=goal, validation_criteria=validation_criteria or []),
        state=state, memory_state=memory_state,
        execution_history=[], metadata={}
    )

    # Load context from input file if provided
    if input_file:
        try:
            load_input_file(input_file, context) 
            current_git_hash = context.state.git_hash
            memory_hash = context.memory_state.memory_hash
            memory_repo_path = context.memory_state.repository_path
            goal_id = context.metadata.get("goal_id") 
            logging.info(f"Context loaded from {input_file}")
            llm_logger.debug(f"Context loaded from {input_file}")
        except Exception as e:
             error_msg = f"Failed to load or process input file {input_file}: {e}"
             logging.error(error_msg)
             llm_logger.error(error_msg)
             return {"success": False, "error": f"Failed to load input file: {e}"}

    # Add goal_id to metadata if provided via argument (and not already loaded)
    if goal_id and not context.metadata.get("goal_id"):
         context.metadata["goal_id"] = goal_id
    elif not context.metadata.get("goal_id"):
        raise ValueError("Goal ID must be provided via --goal-id argument or within the --input-file")
    
    final_goal_id = context.metadata.get("goal_id")
    llm_logger.debug(f"Using goal ID: {final_goal_id}")

    # Handle missing memory state with defaults instead of failing
    if not context.memory_state:
        logging.warning("Memory state is missing, initializing with defaults")
        context.memory_state = MemoryState(memory_hash=None, repository_path=None)
        
    # If memory path is missing, set default
    if not context.memory_state.repository_path:
        default_memory_path = os.path.expanduser("~/.midpoint/memory")
        logging.warning(f"Memory repository path is missing, using default: {default_memory_path}")
        context.memory_state.repository_path = default_memory_path
        # Create directory if it doesn't exist
        os.makedirs(default_memory_path, exist_ok=True)
        
    # If memory hash is missing but repository exists, try to get it
    if not context.memory_state.memory_hash and os.path.exists(context.memory_state.repository_path):
        try:
            from .tools.git_tools import get_current_hash
            memory_hash = get_current_hash(context.memory_state.repository_path)
            if memory_hash:
                logging.info(f"Retrieved memory hash: {memory_hash[:8]}")
                context.memory_state.memory_hash = memory_hash
            else:
                # Fallback to a dummy hash if necessary
                logging.warning("Could not retrieve memory hash, using repository hash as fallback")
                context.memory_state.memory_hash = current_git_hash
        except Exception as e:
            logging.warning(f"Error getting memory hash: {e}, using repository hash as fallback")
            context.memory_state.memory_hash = current_git_hash

    # Validate repository state (don't catch exceptions)
    if not bypass_validation:
        validate_repository_state(repo_path)

    # Create and run the analyzer (don't catch exceptions)
    analyzer = GoalAnalyzer()
    # Pass the potentially updated memory_repo_path to the analyzer instance method
    # Make sure we're passing setup_logging=False here since we already configured logging
    analysis_result_dict = analyzer.analyze_goal_state(context, setup_logging=False, debug=debug, quiet=quiet)

    # Get updated hashes after analysis
    from .tools.git_tools import get_current_hash 
    updated_git_hash = get_current_hash(repo_path) if repo_path else get_current_hash()
    if not updated_git_hash:
         logging.warning("Failed to get updated git hash after analysis.")
         updated_git_hash = current_git_hash 
         
    updated_memory_hash = None
    if memory_repo_path and os.path.exists(memory_repo_path):
         updated_memory_hash = get_current_hash(memory_repo_path)
         if not updated_memory_hash:
              logging.warning(f"Failed to get updated memory hash from {memory_repo_path}.")
    else:
         logging.info("Memory repo path not specified or doesn't exist, cannot get updated memory hash.")

    # Construct Final Result 
    final_result = {
        "success": analysis_result_dict.get("action") != "error",
        "action": analysis_result_dict.get("action", "error"),
        "justification": analysis_result_dict.get("justification", "Analysis failed or produced invalid output"),
        "metadata": analysis_result_dict.get("metadata", {}), 
        "git_hash": updated_git_hash, 
        "initial_git_hash": current_git_hash, 
        "memory_hash": updated_memory_hash, 
        "memory_repository_path": memory_repo_path, 
        "goal_id": final_goal_id 
    }

    # Log final results to the LLM logger as well for completeness
    llm_logger.debug(f"Analysis for goal {final_goal_id} finished with action: {final_result['action']}")
    llm_logger.debug(f"Final analysis result: {json.dumps(final_result, indent=2)}")

    logging.info(f"Analysis for goal {final_goal_id} finished with action: {final_result['action']}")
    logging.debug(f"Final analysis result: {json.dumps(final_result, indent=2)}")

    return final_result

# Correctly commented out CLI entry points
# # Create a separate async entry point for CLI to avoid nesting asyncio.run() calls
# async def async_main():
#     """Async entry point for CLI"""
#     parser = argparse.ArgumentParser(description="Analyze a goal's state")
#     parser.add_argument("repo_path", help="Path to the target repository")
#     parser.add_argument("goal", help="Description of the goal to analyze")
#     parser.add_argument("--input-file", help="Path to input file with goal context")
#     parser.add_argument("--parent-goal", help="Parent goal ID")
#     parser.add_argument("--goal-id", help="Goal ID")
#     parser.add_argument("--memory-repo", help="Path to memory repository")
#     parser.add_argument("--debug", action="store_true", help="Show debug output")
#     parser.add_argument("--quiet", action="store_true", help="Only show warnings and final result")
#     parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
#     parser.add_argument("--logs-dir", default="logs", help="Directory to store log files")
#     
#     args = parser.parse_args()
#     
#     # Call the async function directly - NOTE: analyze_goal is currently sync
#     result = analyze_goal(
#         repo_path=args.repo_path,
#         goal=args.goal,
#         input_file=args.input_file,
#         parent_goal_id=args.parent_goal,
#         goal_id=args.goal_id,
#         memory_repo_path=args.memory_repo,
#         debug=args.debug,
#         quiet=args.quiet,
#         bypass_validation=args.bypass_validation,
#         logs_dir=args.logs_dir
#     )
#     
#     # Print result as JSON
#     print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     # Only use asyncio.run at the top level
#     # asyncio.run(async_main()) # Keep commented
#     pass # Added pass to avoid syntax error on empty block

# # Convert to synchronous
# # Create a CLI entry point
# def main():
#     """Entry point for CLI"""
#     parser = argparse.ArgumentParser(description="Analyze a goal's state")
#     parser.add_argument("repo_path", help="Path to the target repository")
#     parser.add_argument("goal", help="Description of the goal to analyze")
#     parser.add_argument("--input-file", help="Path to input file with goal context")
#     parser.add_argument("--parent-goal", help="Parent goal ID")
#     parser.add_argument("--goal-id", help="Goal ID")
#     parser.add_argument("--memory-repo", help="Path to memory repository")
#     parser.add_argument("--debug", action="store_true", help="Show debug output")
#     parser.add_argument("--quiet", action="store_true", help="Only show warnings and final result")
#     parser.add_argument("--bypass-validation", action="store_true", help="Skip repository validation (for testing)")
#     parser.add_argument("--logs-dir", default="logs", help="Directory to store log files")
#     
#     args = parser.parse_args()
#     
#     # Call the function directly
#     result = analyze_goal(
#         repo_path=args.repo_path,
#         goal=args.goal,
#         input_file=args.input_file,
#         parent_goal_id=args.parent_goal,
#         goal_id=args.goal_id,
#         memory_repo_path=args.memory_repo,
#         debug=args.debug,
#         quiet=args.quiet,
#         bypass_validation=args.bypass_validation
#     )
#     
#     # Print result as JSON
#     print(json.dumps(result, indent=2))

# if __name__ == "__main__":
#     # main() # Keep commented
#     pass # Added pass to avoid syntax error on empty block 

# Helper function to ensure file can be written
def ensure_file_writable(file_path):
    """Ensure the file exists and is writable."""
    try:
        # Create the file if it doesn't exist
        if not os.path.exists(file_path):
            with open(file_path, 'w') as f:
                f.write("")
        
        # Check if file is writable
        if not os.access(file_path, os.W_OK):
            # Try to fix permissions
            os.chmod(file_path, 0o644)
            
        # Final test - try to write to it
        with open(file_path, 'a') as f:
            f.write("")
            
        return True
    except Exception as e:
        logging.error(f"Failed to ensure file is writable: {file_path}, error: {e}")
        return False 

# Helper function for LLM logger debugging
def log_with_timestamp(message, logger=None):
    """Add a timestamp to a log message and ensure it gets written immediately."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    full_message = f"[{timestamp}] {message}"
    
    if logger:
        logger.debug(full_message)
    
    # Also try direct file write as a fallback
    global llm_responses_file
    if llm_responses_file and os.path.exists(llm_responses_file):
        try:
            with open(llm_responses_file, "a") as f:
                f.write(full_message + "\n")
                f.flush()
        except Exception as e:
            # Can't log this error since that might cause a loop
            pass
            
    return full_message 