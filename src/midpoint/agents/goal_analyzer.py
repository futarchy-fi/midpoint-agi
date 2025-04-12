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
    # Try direct import first
    import memory_tools
    
    # Create direct references to the functions we need
    get_repo_path = memory_tools.get_repo_path
    retrieve_recent_memory = memory_tools.retrieve_recent_memory # Assuming this exists now
    store_memory_document = memory_tools.store_memory_document
    
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
    
    def retrieve_recent_memory(memory_hash: str, char_limit: int, repo_path: str) -> Tuple[int, List]:
        """Fallback implementation for retrieving recent memory."""
        logging.warning("Using fallback retrieve_recent_memory - returns empty.")
        return 0, []
        
    def store_memory_document(content, category, metadata=None, repo_path=None, memory_hash=None):
        """Fallback implementation to store a document."""
        # Simplified fallback from goal_decomposer
        logging.warning("Using fallback store_memory_document implementation.")
        repo_path = repo_path or get_repo_path()
        docs_dir = Path(repo_path) / "documents" / category
        docs_dir.mkdir(parents=True, exist_ok=True)
        filename = f"doc_{int(time.time())}.md" # Need time import
        doc_path = docs_dir / filename
        try:
            with open(doc_path, "w") as f: f.write(content)
            logging.info(f"Stored document at: {doc_path} (fallback implementation)")
            return {"success": True, "document_path": str(doc_path)}
        except Exception as write_e:
             logging.error(f"Fallback store_memory_document failed: {write_e}")
             return {"success": False, "error": str(write_e)}
# --- End Robust Memory Tools Import ---

# Import other necessary modules
try:
    # Import necessary functions from goal_cli
    # Adjust path assumption if needed
    # Assuming goal_cli.py is in src/midpoint/ 
    from goal_cli import _get_children_details # Function to get children status
except ImportError as e:
    logging.error(f"Could not import _get_children_details from goal_cli: {e}")
    # Define a fallback if import fails
    def _get_children_details(goal_id: str) -> List[Dict[str, Any]]:
        logging.warning("Using fallback _get_children_details - returns empty list.")
        return []

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

def configure_logging(debug=False, quiet=False, log_dir_path="logs"):
    """
    Configure logging for the goal analyzer.
    
    Args:
        debug: Whether to show debug messages
        quiet: Whether to show only warnings and final result
        log_dir_path: Directory to store log files
    """
    # Ensure log_dir_path is a Path object
    log_dir = Path(log_dir_path)
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"goal_analyzer_{timestamp}.log"
    task_summary_file = log_dir / f"task_summary_{timestamp}.log"
    llm_responses_file = log_dir / f"llm_responses_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.DEBUG if debug else logging.WARNING if quiet else logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
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

Your Analysis Steps:
1. OBSERVE: Carefully review all the provided context. If necessary context is missing or unclear, use the available tools (like read_file, search_code, list_directory) to gather more information about the repository state.
2. ORIENT: Assess the goal's progress. Consider: Is the goal complete? Is the current state significantly different from the initial state? Are children complete? Does the memory suggest recent failures or successes?
3. DECIDE: Based on your analysis, choose ONE of the following actions:
    - "decompose": If the goal is complex, incomplete, and needs further breakdown.
    - "create_task": If the goal is well-defined, actionable, and ready for direct execution.
    - "validate": If the goal seems complete (e.g., all children done) but needs validation.
    - "mark_complete": If the goal is already validated or clearly finished based on context.
    - "update_parent": If a child was completed but the parent state doesn't reflect it.
    - "give_up": If the goal seems impossible, stuck, or no longer relevant based on context/memory.
4. OUTPUT: Provide your decision as a JSON object containing ONLY the "action" (string) and "justification" (string) fields.

Available Tools (for observation only):
{tool_descriptions_text}

IMPORTANT: Return ONLY raw JSON, like {{\"action\": \"decompose\", \"justification\": \"The goal is complex and requires further breakdown based on X.\"}}. Do not wrap it in markdown.
"""
        return system_prompt

    def _get_tool_descriptions(self):
        return ""

    def analyze_goal_state(self, context: TaskContext, setup_logging: bool = False, debug: bool = False, quiet: bool = False) -> Dict[str, Any]:
        """Analyze the current state of the goal and suggest the next action."""
        if setup_logging:
            # Assuming configure_logging is defined elsewhere and imported
            log_file, task_summary_file, llm_responses_file = configure_logging(debug, quiet)

        logging.info("Analyzing state for goal: %s", context.goal.description)
        # Log task summary? Maybe rename log_task_summary or create log_analysis_summary

        # Validate memory state
        if not context.memory_state: raise ValueError("Memory state is required for goal analysis")
        memory_hash = getattr(context.memory_state, "memory_hash", None)
        memory_path = getattr(context.memory_state, "repository_path", None)
        if not memory_hash or not memory_path: raise ValueError("Memory state must have hash and path")
        logging.info(f"Memory state: hash={memory_hash[:8]}, path={memory_path}")

        # Validate inputs
        if not context.goal: raise ValueError("No goal provided")
        if not context.state.repository_path: raise ValueError("Repo path not provided")
        goal_id = context.metadata.get("goal_id")
        if not goal_id: raise ValueError("goal_id missing from context metadata")

        # Prepare the user prompt using the new function
        # Pass necessary context elements directly
        user_prompt = self._create_analysis_user_prompt(
            context, 
            goal_id, 
            memory_hash, 
            memory_path
        )

        messages = [{"role": "system", "content": self._generate_system_prompt()}]
        # Add memory retrieved in _create_analysis_user_prompt (how to pass it here?)
        # Let's assume _create_analysis_user_prompt returns prompt string AND memory docs string
        # For now, just add the prompt. Memory integration needs refinement.
        # The prompt generation now handles adding memory context internally
        messages.append({"role": "user", "content": user_prompt})

        tool_usage = []
        llm_logger = logging.getLogger('llm_interactions')

        try:
            # Log request
            try: llm_logger.debug("LLM Request:\n%s", json.dumps(self._serialize_messages(messages), indent=2))
            except Exception as log_e: llm_logger.error("Failed to serialize request: %s", str(log_e))

            # Get analysis from the model, allowing tool use
            message, tool_calls = self.tool_processor.run_llm_with_tools(
                messages, model=self.model,
                validate_json_format=True, # Expecting JSON output
                max_tokens=1000 # Analysis output should be shorter
            )

            if tool_calls:
                # We might need to handle the case where tools update the state
                # before the final analysis JSON is produced. 
                # For now, just record usage.
                for tool_call in tool_calls: 
                    # Basic serialization for logging/metadata
                    tool_usage.append(str(tool_call)) 

            # Parse the model's final response (expecting JSON)
            if isinstance(message, list): content = message[-1].get('content', '')
            else: content = message.get('content') if isinstance(message, dict) else message.content

            logging.debug(f"Raw model response: {content}")
            llm_logger.debug("LLM Raw Response:\n%s", content)

            try:
                output_data = json.loads(content)
                 # Validate expected fields
                if not all(key in output_data for key in ["action", "justification"]):
                    # If missing, try extracting from markdown code blocks
                    json_match = re.search(r'```(?:json)?\s*(\{.+?\})?\s*```', content, re.DOTALL)
                    if json_match:
                        try:
                            output_data = json.loads(json_match.group(1))
                            if not all(key in output_data for key in ["action", "justification"]):
                                raise ValueError("Extracted JSON missing 'action' or 'justification'")
                        except (json.JSONDecodeError, ValueError) as inner_e:
                             logging.error(f"Failed to parse extracted JSON or validate fields: {inner_e}")
                             raise ValueError("LLM response JSON invalid or missing fields even after extraction.")
                    else:
                         raise ValueError("LLM response missing 'action' or 'justification' and no JSON block found.")

                # Create the final result dictionary
                final_output = {
                    "action": output_data["action"],
                    "justification": output_data["justification"],
                    "metadata": { "raw_response": content, "tool_usage": tool_usage }
                }
                logging.info(f"✅ Analysis complete. Suggested action: {final_output['action']}")
                logging.debug(f"Justification: {final_output['justification']}")

            except (json.JSONDecodeError, ValueError) as e:
                logging.error(f"Error processing model response: {str(e)}")
                llm_logger.error("Error processing model response: %s\nRaw response was: %s", str(e), content)
                # Handle error - return a default "error" action
                final_output = {
                    "action": "error",
                    "justification": f"Failed to process LLM analysis response: {str(e)}",
                    "metadata": { "raw_response": content, "tool_usage": tool_usage }
                }

            # Save conversation to memory
            # Make sure get_repo_path() is available or handled
            try: 
                repo_path_for_saving = memory_path or get_repo_path() 
            except NameError: # Handle if get_repo_path wasn't imported
                 repo_path_for_saving = None
                 logging.warning("get_repo_path not available, cannot determine default memory path for saving.")
                 
            if repo_path_for_saving and memory_hash: # Only save if path and hash are valid
                 self._save_conversation_to_memory(
                     messages, 
                     metadata={
                         "goal": context.goal.description,
                         "goal_id": goal_id,
                         "suggested_action": final_output["action"],
                         "memory_hash": memory_hash # Use the initial hash for saving context
                     },
                     memory_hash=memory_hash,
                     repo_path=repo_path_for_saving,
                     goal_name=context.goal.description
                 )
            else:
                logging.warning("Skipping saving conversation to memory due to missing repo path or hash.")

            return final_output # Return the dictionary

        except Exception as e:
            logging.error(f"Error retrieving memory context: {str(e)}")
            raise
    
    def _create_analysis_user_prompt(self, context: TaskContext, goal_id: str, memory_hash: str, memory_repo_path: str) -> str:
        """Create the user prompt for the Goal Analyzer LLM."""
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
            except Exception as e:
                logging.error(f"Failed to load goal file {goal_file}: {e}")
                prompt_lines.append(f"Error: Could not load goal file {goal_file}")
        else:
            logging.warning(f"Goal file not found: {goal_file}")
            prompt_lines.append(f"Warning: Goal file {goal_file} not found.")

        # --- 2. Get Children Details ---
        try:
            # Import here to avoid potential top-level import issues if goal_cli changes
            from goal_cli import _get_children_details
            children_details = _get_children_details(goal_id)
            logging.info(f"Found {len(children_details)} children for goal {goal_id}")
        except ImportError:
             logging.error("Could not import _get_children_details from goal_cli.")
             prompt_lines.append("Error: Could not get children details (import failed).")
        except Exception as e:
            logging.error(f"Failed to get children details for {goal_id}: {e}")
            prompt_lines.append("Error: Could not get children details.")

        # --- 3. Retrieve Recent Memory ---
        if memory_hash and memory_repo_path:
            try:
                # Import here to avoid potential top-level import issues
                from tools.memory_tools import retrieve_recent_memory
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
                    memory_content = "\n".join(memory_context_lines)
                    logging.info(f"Retrieved {len(memory_documents)} memory documents ({total_chars} chars).")
                else:
                    logging.info("No recent memory documents found.")
                    memory_content = "No recent memory documents found."

            except ImportError:
                 logging.error("Could not import retrieve_recent_memory from memory_tools.")
                 memory_content = "Error: Could not retrieve memory context (import failed)."
            except Exception as e:
                logging.error(f"Error retrieving memory context: {str(e)}")
                memory_content = f"Error retrieving memory context: {str(e)}"
        else:
            logging.warning("Memory hash or repo path missing, cannot retrieve memory context.")
            memory_content = "Memory context unavailable (missing hash or path)."


        # --- 4. Format Prompt ---
        prompt_lines.append(f"Goal Analysis Request: {goal_id}")
        prompt_lines.append("="*20)

        # Goal Details
        prompt_lines.append("## Goal Details")
        prompt_lines.append(f"ID: {goal_id}")
        prompt_lines.append(f"Description: {goal_data.get('description', 'N/A')}")
        prompt_lines.append(f"Parent: {goal_data.get('parent_goal', 'None')}")
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

        # Memory Context
        prompt_lines.append("\n" + memory_content)

        # Final Instruction
        prompt_lines.append("\n" + "="*20)
        prompt_lines.append("Based on ALL the context above, analyze the goal's status.")
        prompt_lines.append("If necessary, use tools to observe the current state further.")
        prompt_lines.append("Then, provide your suggested next action and justification in the required JSON format.")

        return "\n".join(prompt_lines)

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

    def _save_conversation_to_memory(self, messages: List[Dict[str, Any]], metadata: Dict[str, Any], memory_hash: str, repo_path: str, goal_name: str):
        """Save the conversation and metadata to memory."""
        # Implement the logic to save the conversation and metadata to memory
        # This is a placeholder and should be replaced with the actual implementation
        # For now, just log that it was called
        logging.info(f"Placeholder: _save_conversation_to_memory called for goal {goal_name}")
        # Example of calling the actual tool (if needed later and imported)
        # try:
        #     from .tools.memory_tools import store_memory_document # Ensure import
        #     # ... format content ...
        #     store_memory_document(content=..., category=..., metadata=metadata, memory_hash=memory_hash, memory_repo_path=repo_path)
        # except Exception as e:
        #      logging.error(f"Actual save_conversation_to_memory failed: {e}")
        pass

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
    configure_logging(debug, quiet, logs_dir)
    logging.info(f"Analyzing goal: {goal}")

    # Initial state setup
    try: 
        from .tools.git_tools import get_current_hash 
        current_git_hash = get_current_hash(repo_path) if repo_path else get_current_hash()
        if not current_git_hash:
             raise RuntimeError("Failed to get current git hash.")
        logging.info(f"Current git hash: {current_git_hash[:8]}")
    except Exception as e:
        logging.error(f"Error getting initial git hash: {e}")
        return {"success": False, "error": f"Failed to get initial git hash: {e}"}

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
        except Exception as e:
             logging.error(f"Failed to load or process input file {input_file}: {e}")
             return {"success": False, "error": f"Failed to load input file: {e}"}

    # Add goal_id to metadata if provided via argument (and not already loaded)
    if goal_id and not context.metadata.get("goal_id"):
         context.metadata["goal_id"] = goal_id
    elif not context.metadata.get("goal_id"):
        raise ValueError("Goal ID must be provided via --goal-id argument or within the --input-file")
    
    final_goal_id = context.metadata.get("goal_id")

    # Re-validate memory state after potential input file loading
    if not context.memory_state or not context.memory_state.memory_hash or not context.memory_state.repository_path:
         logging.warning("Memory state (hash or path) is missing after context setup. Analysis might be impaired.")
         memory_hash = None 
         memory_repo_path = context.memory_state.repository_path if context.memory_state else None 

    # Validate repository state (optional)
    if not bypass_validation:
        try: 
            validate_repository_state(repo_path)
        except Exception as e: 
            logging.error(f"Repository validation failed: {e}")
            return {"success": False, "error": f"Repository validation failed: {e}"}

    # Create and run the analyzer
    try:
        analyzer = GoalAnalyzer()
        # Pass the potentially updated memory_repo_path to the analyzer instance method
        analysis_result_dict = analyzer.analyze_goal_state(context, setup_logging=True, debug=debug, quiet=quiet)
    except Exception as e:
        logging.error(f"GoalAnalyzer failed during execution: {e}", exc_info=debug)
        return {"success": False, "error": f"Goal analysis agent failed: {e}"}

    # Get updated hashes after analysis
    try: 
        from .tools.git_tools import get_current_hash 
        updated_git_hash = get_current_hash(repo_path) if repo_path else get_current_hash()
        if not updated_git_hash:
             logging.warning("Failed to get updated git hash after analysis.")
             updated_git_hash = current_git_hash 
             
        updated_memory_hash = None
        if memory_repo_path and os.path.exists(memory_repo_path):
             try:
                 updated_memory_hash = get_current_hash(memory_repo_path)
                 if not updated_memory_hash:
                      logging.warning(f"Failed to get updated memory hash from {memory_repo_path}.")
             except Exception as mem_e:
                  logging.warning(f"Error getting updated memory hash from {memory_repo_path}: {mem_e}")
        else:
             logging.info("Memory repo path not specified or doesn't exist, cannot get updated memory hash.")
             
    except Exception as e:
        logging.error(f"Error getting updated state hashes: {e}")
        updated_git_hash = current_git_hash
        updated_memory_hash = memory_hash 

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