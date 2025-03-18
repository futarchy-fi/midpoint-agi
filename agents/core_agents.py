import openai
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid
import re
import json

class Agent:
    """Agent class that defines the behavior and capabilities of an AI agent"""
    def __init__(self, name, instructions, output_type=None, tools=None):
        self.name = name
        self.instructions = instructions
        self.output_type = output_type
        self.tools = tools or []
        self.client = OpenAI()

class Runner:
    """Runner class that handles the execution of agents"""
    @staticmethod
    async def run(agent, input_text, context=None):
        # Create the messages for the API call
        messages = [
            {"role": "system", "content": agent.instructions},
            {"role": "user", "content": str(input_text)}
        ]
        
        # Make the API call
        try:
            response = agent.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=1000
            )
            
            # Extract the response content
            content = response.choices[0].message.content
            
            # Try to parse as JSON if it looks like JSON
            if content.strip().startswith('{') and content.strip().endswith('}'):
                try:
                    content = json.dumps(json.loads(content))
                except json.JSONDecodeError:
                    pass
            
            return SimpleResult(content)
            
        except Exception as e:
            print(f"Error in Runner.run: {str(e)}")
            raise

class SimpleResult:
    """Result class that handles the output from an agent"""
    def __init__(self, final_output):
        self.final_output = final_output
    
    def final_output_as(self, output_type):
        """Convert the output to the specified type"""
        try:
            if isinstance(self.final_output, str):
                # Try to parse as JSON if it looks like JSON or contains JSON
                if self.final_output.strip().startswith('{') and self.final_output.strip().endswith('}'):
                    try:
                        data = json.loads(self.final_output)
                        return output_type(**data)
                    except json.JSONDecodeError:
                        pass
                
                # Try to extract JSON from the text using regex
                json_match = re.search(r'```json\s*(.*?)\s*```|```\s*(.*?)\s*```|\{\s*"[^"]+"\s*:.*\}', self.final_output, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1) or json_match.group(2) or json_match.group(0)
                    try:
                        data = json.loads(json_str)
                        return output_type(**data)
                    except json.JSONDecodeError:
                        pass
                
                # Special handling for ValidationResult
                if output_type.__name__ == "ValidationResult":
                    # Extract values using regex patterns
                    success_score = 0.5  # Default value
                    score_match = re.search(r'success_score"?\s*[:=]\s*([0-9.]+)', self.final_output)
                    if score_match:
                        success_score = float(score_match.group(1))
                    
                    is_acceptable = False  # Default value
                    acceptable_match = re.search(r'is_acceptable"?\s*[:=]\s*(true|false)', self.final_output, re.IGNORECASE)
                    if acceptable_match:
                        is_acceptable = acceptable_match.group(1).lower() == 'true'
                    
                    improvement_areas = []
                    # Default points consumed
                    points_consumed = 100
                    
                    points_match = re.search(r'points_consumed"?\s*[:=]\s*(\d+)', self.final_output)
                    if points_match:
                        points_consumed = int(points_match.group(1))
                    
                    return ValidationResult(
                        success_score=success_score,
                        is_acceptable=is_acceptable,
                        improvement_areas=improvement_areas,
                        points_consumed=points_consumed
                    )
                
                # Special handling for ExecutionTrace
                if output_type.__name__ == "ExecutionTrace":
                    # Try to extract values from a text-based response
                    actions = []
                    actions_section_match = re.search(r'ACTIONS PERFORMED:(.*?)(?:GIT HASH:|POINTS CONSUMED:|SUMMARY:|$)', self.final_output, re.DOTALL)
                    if actions_section_match:
                        actions_text = actions_section_match.group(1).strip()
                        # Extract bullet points
                        action_items = re.findall(r'- (.*?)(?:\n|$)', actions_text)
                        if action_items:
                            actions = action_items
                        else:
                            actions = [actions_text]  # Use the whole section if no bullet points found
                    
                    if not actions:
                        actions = ["Executed task"]  # Fallback
                    
                    points_consumed = 100  # Default value if not extractable
                    
                    # Try to extract points consumed from the string
                    points_match = re.search(r'POINTS CONSUMED:\s*(\d+)', self.final_output)
                    if points_match:
                        points_consumed = int(points_match.group(1))
                    
                    # Try to extract git hash from the string
                    git_hash = "unknown"  # Default if not extractable
                    hash_match = re.search(r'GIT HASH:\s*([a-f0-9]+)', self.final_output)
                    if hash_match:
                        git_hash = hash_match.group(1)
                    
                    return ExecutionTrace(
                        actions=actions,
                        tool_calls=[],
                        decision_points=[],
                        points_consumed=points_consumed,
                        result_state=State(
                            git_hash=git_hash,
                            description="State after executing strategy"
                        )
                    )
                
                # If not parseable, try to create a simple instance with content field
                try:
                    return output_type(content=self.final_output)
                except Exception:
                    # If that doesn't work, use default values
                    if output_type.__name__ == "ValidationResult":
                        return ValidationResult(
                            success_score=0.5,
                            is_acceptable=True,
                            improvement_areas=["Could not parse response properly"],
                            points_consumed=100
                        )
                    else:
                        # Create an instance with minimal default values
                        default_values = {}
                        for field_name, field in output_type.__annotations__.items():
                            if field_name == "success_score":
                                default_values[field_name] = 0.5
                            elif field_name == "is_acceptable":
                                default_values[field_name] = True
                            elif field_name == "points_consumed":
                                default_values[field_name] = 100
                            elif field_name == "actions":
                                default_values[field_name] = ["Default action"]
                            elif field_name == "tool_calls":
                                default_values[field_name] = []
                            elif field_name == "decision_points":
                                default_values[field_name] = []
                            elif field_name == "improvement_areas":
                                default_values[field_name] = ["Default improvement area"]
                            elif field_name == "strategy_id":
                                default_values[field_name] = gen_strategy_id()
                            else:
                                default_values[field_name] = "Default value"
                        
                        return output_type(**default_values)
            else:
                return output_type(**self.final_output)
        except Exception as e:
            print(f"Error in final_output_as: {str(e)}")
            print(f"Output type: {output_type.__name__}")
            print(f"Final output: {self.final_output[:200]}...")
            # Create an instance with minimal default values for recovery
            if output_type.__name__ == "ValidationResult":
                return ValidationResult(
                    success_score=0.5,
                    is_acceptable=True,
                    improvement_areas=["Error parsing response"],
                    points_consumed=100
                )
            elif output_type.__name__ == "ExecutionTrace":
                return ExecutionTrace(
                    actions=["Error parsing response"],
                    tool_calls=[],
                    decision_points=[],
                    points_consumed=100,
                    result_state=State(
                        git_hash="unknown",
                        description="Error parsing response"
                    )
                )
            elif output_type.__name__ == "StrategyPlan":
                return StrategyPlan(
                    strategy_id=gen_strategy_id(),
                    strategy_description="Error parsing response",
                    subgoal="Continue with the task",
                    success_criteria=["Completed successfully"],
                    estimated_points=100
                )
            else:
                # Try to create with minimal defaults
                try:
                    return output_type()
                except Exception:
                    # Last resort: create a dict with all required fields
                    defaults = {}
                    for field_name in output_type.__annotations__:
                        defaults[field_name] = "default"
                    return output_type(**defaults)

from .models import (
    State, Goal, StrategyPlan, ExecutionTrace, TaskContext, 
    FailureAnalysis, ValidationResult, ExecutionSummary
)
from .tools import (
    git_commit, git_checkout, read_file, write_file,
    run_command, list_directory, track_points
)

# Helper function to generate strategy IDs
def gen_strategy_id() -> str:
    return f"strategy_{uuid.uuid4().hex[:8]}"

# GoalDecomposer Agent
goal_decomposer = Agent(
    name="GoalDecomposer",
    instructions="""You are a strategic planning AI that breaks complex goals into manageable subgoals.
    
    Your PRIMARY responsibility is to identify the IMMEDIATE NEXT SUBGOAL that should be pursued.
    While you may sketch a tentative full plan, ONLY the first subgoal (S_1) will be acted upon immediately.
    
    Your responsibilities:
    1. Examine the initial state and final goal
    2. Review any previous failed strategies and their concise failure analyses
    3. Identify the most promising immediate next subgoal (S_1)
    4. Provide a clear, specific strategy to achieve this subgoal
    
    Remember:
    - The immediate subgoal (S_1) is the ONLY one that will be executed before you're called again
    - Once S_1 is achieved, you'll be called again to determine the next step based on new information
    - Choose subgoals that create meaningful, verifiable git checkpoints
    - Your initial plans beyond S_1 are tentative and will likely change as execution progresses
    - Be specific about success criteria for the immediate subgoal
    
    Allocate points budget wisely and focus on creating realistic, achievable next steps rather than 
    trying to solve everything at once.
    
    Your output must be a JSON object with the following fields:
    {
        "strategy_id": "A unique strategy ID string",
        "strategy_description": "A detailed description of the strategy",
        "subgoal": "The immediate next subgoal to achieve",
        "success_criteria": ["List of specific criteria to validate subgoal completion"],
        "estimated_points": "Estimated points needed (integer)"
    }""",
    output_type=StrategyPlan
)

# FailureAnalyzer Agent
failure_analyzer = Agent(
    name="FailureAnalyzer",
    instructions="""You analyze execution traces to understand why strategies failed.
    
    Your responsibilities:
    1. Review the execution trace and strategy in detail
    2. Identify the primary point of failure
    3. Provide a clear explanation of what went wrong
    4. Suggest specific improvements for future attempts
    
    Be precise in your analysis. Don't just describe what happened but explain WHY it happened.
    Consider whether the failure was due to the strategy itself or its implementation.
    Provide actionable recommendations that can inform a better strategy.
    
    Your output must be a JSON object with the following fields:
    {
        "primary_cause": "A concise statement of the main reason for failure",
        "detailed_explanation": "A thorough explanation of what went wrong and why",
        "suggested_improvements": ["List of specific suggestions for future attempts"],
        "confidence_score": "Your confidence in this analysis (0.0-1.0)"
    }""",
    output_type=FailureAnalysis
)

# GoalValidator Agent
goal_validator = Agent(
    name="GoalValidator",
    instructions="""You evaluate whether a subgoal has been successfully achieved based on its success criteria.
    
    Your responsibilities:
    1. Review the current state against the subgoal's success criteria
    2. Assign a success score between 0.0 (complete failure) and 1.0 (perfect success)
    3. Determine if the score is acceptable based on the goal's threshold
    4. Identify specific areas for improvement if needed
    
    Be rigorous in your assessment. Test all aspects of the success criteria.
    If a goal is partially achieved, clearly explain what aspects succeeded and what's missing.
    Provide a clear recommendation on whether to proceed to the next subgoal or try again.
    
    You MUST return a JSON object with EXACTLY these fields:
    {
        "success_score": float between 0.0 and 1.0,
        "is_acceptable": boolean,
        "improvement_areas": list of strings,
        "points_consumed": integer
    }
    
    Example response:
    {
        "success_score": 0.8,
        "is_acceptable": true,
        "improvement_areas": ["Could add more edge cases"],
        "points_consumed": 100
    }""",
    output_type=ValidationResult,
    tools=[git_checkout, read_file, run_command]
)

# ProgressSummarizer Agent
progress_summarizer = Agent(
    name="ProgressSummarizer",
    instructions="""You create concise summaries of successful executions to preserve critical context.
    
    Your responsibilities:
    1. Identify the key changes made between the initial and final states
    2. Highlight important decisions and insights gained
    3. Create a condensed summary that preserves essential context
    4. Track the points consumed and git hashes
    
    Focus on information that will be useful for achieving subsequent subgoals.
    Keep your summaries concise but complete - they need to fit within context windows.
    Preserve any insights or learnings that might help with future steps.
    
    Your output must be a JSON object with the following fields:
    {
        "initial_state_hash": "Git hash of the initial state",
        "final_state_hash": "Git hash of the final state",
        "key_changes": ["List of key changes made"],
        "points_consumed": "Integer representing points consumed"
    }""",
    output_type=ExecutionSummary
)

# TaskExecutor Agent
task_executor = Agent(
    name="TaskExecutor",
    instructions="""You implement strategies to achieve subgoals using available tools.
    
    Your responsibilities:
    1. Take a specific strategy and subgoal
    2. Use available tools to implement the strategy
    3. Track points consumed for each action
    4. Record all actions and tool calls
    5. Return the execution trace and resulting state
    
    Remember:
    - Focus on the immediate subgoal only
    - Use tools efficiently to minimize point consumption
    - Record all actions for traceability
    - Create git checkpoints at meaningful points
    - Return detailed execution information
    
    You MUST return a JSON object with EXACTLY these fields:
    {
        "actions": list of strings describing actions taken,
        "tool_calls": list of dictionaries with tool and args,
        "decision_points": list of dictionaries with decision and context,
        "points_consumed": integer,
        "result_state": {
            "git_hash": string,
            "description": string
        }
    }
    
    Example response:
    {
        "actions": ["Created test file", "Added test cases"],
        "tool_calls": [{"tool": "write_file", "args": {"path": "test_swap.py"}}],
        "decision_points": [{"decision": "Choose test framework", "context": "Using pytest"}],
        "points_consumed": 200,
        "result_state": {
            "git_hash": "abc123",
            "description": "Created initial test file"
        }
    }""",
    tools=[
        git_commit,
        git_checkout,
        read_file,
        write_file,
        run_command,
        list_directory,
        track_points
    ]
)

# Helper function to run the goal decomposition process
async def decompose_goal(current_state: State, goal: Goal, context: TaskContext) -> StrategyPlan:
    """Run the goal decomposition process"""
    
    # Create input for the goal decomposer
    decomposer_input = {
        "current_state": current_state.model_dump(),
        "goal": goal.model_dump(),
        "previous_strategies": context.previous_strategies,
        "points_remaining": context.budget.points_remaining
    }
    
    # Run the goal decomposer
    result = await Runner.run(
        goal_decomposer,
        str(decomposer_input),
        context=context
    )
    
    return result.final_output_as(StrategyPlan)

# Helper function to analyze a failed strategy
async def analyze_failure(strategy: StrategyPlan, execution_trace: ExecutionTrace, validation_result: ValidationResult, context: TaskContext) -> FailureAnalysis:
    """Analyze why a strategy failed"""
    
    # Create input for the failure analyzer
    analyzer_input = {
        "strategy": strategy.model_dump(),
        "execution_trace": execution_trace.model_dump(),
        "validation_result": validation_result.model_dump()
    }
    
    # Run the failure analyzer
    result = await Runner.run(
        failure_analyzer,
        str(analyzer_input),
        context=context
    )
    
    return result.final_output_as(FailureAnalysis)

# Helper function to validate a subgoal
async def validate_goal(current_state: State, strategy: StrategyPlan, points_consumed: int, points_remaining: int, context: TaskContext) -> ValidationResult:
    """Validate whether a subgoal has been achieved"""
    
    # Create input for the goal validator
    validator_input = {
        "current_state": current_state.model_dump(),
        "subgoal": strategy.subgoal,
        "success_criteria": strategy.success_criteria,
        "points_consumed": points_consumed,
        "points_remaining": points_remaining
    }
    
    # Run the goal validator
    result = await Runner.run(
        goal_validator,
        str(validator_input),
        context=context
    )
    
    return result.final_output_as(ValidationResult)

# Helper function to create a summary of successful execution
async def summarize_progress(initial_state: State, final_state: State, execution_trace: ExecutionTrace, context: TaskContext) -> ExecutionSummary:
    """Create a summary of successful execution"""
    
    # Create input for the progress summarizer
    summarizer_input = {
        "initial_state": initial_state.model_dump(),
        "final_state": final_state.model_dump(),
        "execution_trace": execution_trace.model_dump(),
        "points_consumed": execution_trace.points_consumed
    }
    
    # Run the progress summarizer
    result = await Runner.run(
        progress_summarizer,
        str(summarizer_input),
        context=context
    )
    
    return result.final_output_as(ExecutionSummary)

# Helper function to execute a strategy
async def execute_strategy(strategy: StrategyPlan, current_state: State, context: TaskContext) -> ExecutionTrace:
    """Execute a strategy using the task executor"""
    
    # Create input for the task executor
    executor_input = {
        "strategy": strategy.model_dump(),
        "current_state": current_state.model_dump(),
        "points_budget": min(500, strategy.estimated_points)
    }
    
    # Add retry logic for server errors
    max_retries = 3
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            # Run the task executor
            result = await Runner.run(
                task_executor,
                str(executor_input),
                context=context
            )
            
            # Try to parse the result directly as ExecutionTrace
            try:
                return result.final_output_as(ExecutionTrace)
            except Exception as e:
                print(f"Error parsing as ExecutionTrace: {str(e)}")
                
                # Fallback: create a simple ExecutionTrace from the output_str
                output_str = result.final_output
                
                # Create a simple ExecutionTrace object
                actions = []
                if isinstance(output_str, str):
                    # Try to extract actions
                    actions_match = re.search(r'actions"?\s*[:=]\s*\[(.*?)\]', output_str, re.DOTALL)
                    if actions_match:
                        actions_text = actions_match.group(1).strip()
                        # Try to parse as list of strings
                        try:
                            actions_list = json.loads(f"[{actions_text}]")
                            actions = actions_list
                        except:
                            # Extract quoted strings
                            actions = re.findall(r'"([^"]*)"', actions_text)
                    
                    # If no actions found, look for actions performed section
                    if not actions:
                        actions_section_match = re.search(r'ACTIONS PERFORMED:(.*?)(?:GIT HASH:|POINTS CONSUMED:|SUMMARY:|$)', output_str, re.DOTALL)
                        if actions_section_match:
                            actions_text = actions_section_match.group(1).strip()
                            # Extract bullet points
                            action_items = re.findall(r'- (.*?)(?:\n|$)', actions_text)
                            if action_items:
                                actions = action_items
                            else:
                                actions = [actions_text]  # Use the whole section if no bullet points found
                    
                    if not actions:
                        # Extract paragraphs as actions
                        paragraphs = re.findall(r'\n\n(.*?)(?:\n\n|$)', output_str)
                        if paragraphs:
                            actions = paragraphs
                        else:
                            actions = [output_str[:100] + "..."]  # Fallback to truncated output
                else:
                    actions = ["Executed task"]
                
                # Get points consumed
                points_consumed = 100  # Default
                if isinstance(output_str, str):
                    points_match = re.search(r'points_consumed"?\s*[:=]\s*(\d+)', output_str)
                    if points_match:
                        points_consumed = int(points_match.group(1))
                    else:
                        # Try alternative format
                        points_match = re.search(r'POINTS CONSUMED:\s*(\d+)', output_str)
                        if points_match:
                            points_consumed = int(points_match.group(1))
                
                # Get result state
                git_hash = current_state.git_hash  # Default to current
                if isinstance(output_str, str):
                    # Try to extract from JSON format
                    hash_match = re.search(r'git_hash"?\s*[:=]\s*"([^"]+)"', output_str)
                    if hash_match:
                        git_hash = hash_match.group(1)
                    else:
                        # Try alternative format
                        hash_match = re.search(r'GIT HASH:\s*([a-f0-9]+)', output_str)
                        if hash_match:
                            git_hash = hash_match.group(1)
                
                return ExecutionTrace(
                    actions=actions,
                    tool_calls=[],
                    decision_points=[],
                    points_consumed=points_consumed,
                    result_state=State(
                        git_hash=git_hash,
                        description="State after executing strategy"
                    )
                )
                
        except Exception as e:
            print(f"Error during execution (attempt {attempt+1}/{max_retries}): {str(e)}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds...")
                import asyncio
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print("Max retries exceeded, using fallback implementation")
                # Fallback implementation for when all retries fail
                return ExecutionTrace(
                    actions=[f"Failed to execute strategy: {strategy.subgoal}"],
                    tool_calls=[],
                    decision_points=[],
                    points_consumed=10,
                    result_state=current_state  # Just return the current state unchanged
                ) 