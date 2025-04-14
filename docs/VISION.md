# Advanced AGI System with Recursive Goal Decomposition

This document outlines the vision and architecture for an advanced AGI system that can recursively follow goal-decomposition and planning when applied to the task of modifying repositories using git.

## Core Vision

The system is designed to overcome a fundamental limitation of current AI systems: the inability to effectively reason over long chains of thought and manage complex objectives that exceed their context window or reasoning capabilities. By decomposing complex goals into manageable subgoals and using a coordinated multi-agent approach, the system can tackle problems that would be intractable for a single agent.

Key capabilities include:
- Strategic planning with OODA loop decision-making
- Recursive goal decomposition using depth-first search
- Tactical execution with practical tool use
- Detailed failure analysis and learning
- Validation and verification of results
- Efficient context management
- Budget-aware execution
- Intelligent failure handling with parent goal propagation

## Core Principles

1. **Focus on the Next Step**: Rather than planning all steps at once, the system prioritizes determining the most promising single next step. After each step is executed and validated, the system reassesses from the new state.

2. **OODA Loop Decision Making**: The system follows the Observe-Orient-Decide-Act loop:
   - **Observe**: Explore the current repository state
   - **Orient**: Analyze how the current state relates to the goal
   - **Decide**: Determine the most promising next step
   - **Act**: Execute that step or further decompose it

3. **Recursive Decomposition**: Complex goals are recursively broken down into progressively simpler subgoals until reaching directly executable tasks, using a depth-first search approach.

4. **Selective Context Passing**: Only relevant information is passed to child subgoals, avoiding overwhelming the model with unnecessary high-level details.

5. **Repository as State**: The git repository serves as the concrete state representation, with commits providing verifiable checkpoints.

6. **Failure Propagation**: When a goal is determined to be impossible or no longer relevant, the system intelligently propagates this information to parent goals, allowing for strategic adaptation.

## System Architecture

The system consists of five specialized agents, each with a distinct role in the problem-solving process:

### 1. GoalAnalyzer (Strategic Planning)

**Purpose**: Analyzes the current state of a goal and determines the most appropriate next action, including whether to decompose, execute, validate, or give up.

**Inputs**:
- Current goal state and description
- Repository state (git hash, path)
- Child goals/tasks status
- Previous execution attempts and analyses
- Failure history from previous attempts

**Outputs**:
- Analysis result containing:
  - Recommended action (decompose, create_task, validate, mark_complete, update_parent, give_up)
  - Detailed justification for the action
  - Relevant context for the next step

**Responsibilities**:
- Analyze the current state using available tools
- Follow OODA loop to determine next steps
- Identify when a goal is impossible or no longer relevant
- Provide detailed justification for all decisions
- Consider failure history when making decisions

### 2. TaskExecutor (Implementation)

**Purpose**: Implements directly executable tasks identified by the GoalAnalyzer.

**Inputs**:
- Current state
- Executable task specifications
- Validation criteria
- Points budget for this execution

**Outputs**:
- Execution trace with:
  - Actions performed
  - Tool calls made
  - Points consumed
  - Resulting state (typically a new git commit)
  - Success/failure status

**Responsibilities**:
- Use tools effectively (git, filesystem, web search, etc.)
- Track point consumption
- Create meaningful git checkpoints
- Maintain detailed execution records
- Store failure information when execution fails

### 3. GoalValidator (Verification)

**Purpose**: Evaluates whether a subgoal has been successfully achieved.

**Inputs**:
- Current state (after execution)
- Subgoal and success criteria
- Points consumed and remaining

**Outputs**:
- Validation result with:
  - Success score (0.0 to 1.0)
  - Whether this is acceptable
  - Areas for improvement
  - Points accounting

**Responsibilities**:
- Rigorously test against success criteria
- Provide clear success/failure determination
- Identify specific improvement areas if needed

### 4. FailureAnalyzer (Diagnostics)

**Purpose**: Analyzes failed executions to understand why they didn't succeed.

**Inputs**:
- Task that failed
- Execution trace
- Validation result
- Failure history from previous attempts

**Outputs**:
- Failure analysis with:
  - Primary cause of failure
  - Detailed explanation
  - Suggested improvements
  - Confidence in analysis

**Responsibilities**:
- Identify root causes of failures
- Provide actionable feedback for future attempts
- Distinguish between strategic and tactical failures
- Track patterns of failure across multiple attempts

### 5. ProgressSummarizer (Context Management)

**Purpose**: Creates concise summaries of successful executions to preserve critical context.

**Inputs**:
- Initial state (before execution)
- Final state (after successful execution)
- Execution trace

**Outputs**:
- Execution summary with:
  - Key changes made
  - Important decisions
  - Points consumed
  - Git hash references

**Responsibilities**:
- Condense execution details to essential information
- Preserve context for subsequent subgoals
- Track progress toward the final goal

## Workflow Orchestration

The system follows this general workflow:

1. **Planning Phase**:
   - GoalAnalyzer analyzes the problem using the OODA loop
   - Determines the single most promising next step
   - Recursively decomposes until reaching an executable task
   - Decides when to give up on a goal and propagate failure to parent

2. **Execution Phase**:
   - TaskExecutor implements the executable task
   - Creates a detailed trace of its actions
   - Stores failure information when execution fails

3. **Validation Phase**:
   - GoalValidator assesses if the task was achieved
   - If successful:
     - ProgressSummarizer condenses the execution trace
     - System returns to Planning Phase with the new state
   - If unsuccessful:
     - FailureAnalyzer diagnoses what went wrong
     - System returns to Planning Phase with the analysis

4. **Failure Handling Phase**:
   - If a goal is determined to be impossible or no longer relevant:
     - The goal is marked as "given_up"
     - Failure information is propagated to the parent goal
     - The parent goal is analyzed in the next iteration
     - If no parent exists, the overall process is marked as unsuccessful

5. **Recursion**:
   - This process continues until the final goal is achieved, the budget is exhausted, or all paths are determined to be impossible

## Recursive Decomposition Approach

The system employs a depth-first search approach to goal decomposition:

1. Start with a high-level goal
2. Determine the most promising next step
3. Check if this step requires further decomposition
   - If yes: Create a new subgoal and recursively decompose it
   - If no: Mark it as ready for execution
4. Execute the task and validate its completion
5. Continue with the next step based on the new state
6. If a goal is determined to be impossible, propagate failure to parent

This approach ensures that:
- The system always works on the most promising path first
- Each step is broken down to an appropriate level of detail
- Execution always occurs on directly implementable tasks
- The system can adapt to changing circumstances after each step
- The system can intelligently abandon impossible paths

## Points Budget System

To constrain and prioritize agent activities, the system uses a points budget:

- Each agent action consumes points:
  - Text generation: 1 point per paragraph
  - Tool use: 5 points per tool call 
  - API calls: 10 points
  - Code execution: 20 points

- Budget allocation:
  - Overall budget for the entire goal (e.g., 50,000 points)
  - GoalAnalyzer consumes points for exploration and planning
  - TaskExecutor is limited to specific points per execution

- Points tracking:
  - Each agent reports points consumed
  - System maintains a budget ledger
  - Helps prioritize efficient approaches

## State Management with Git

The system uses git as its primary state management tool:

- Each meaningful state corresponds to a git commit
- Provides versioning and rollback capabilities
- Enables verification through concrete artifacts
- Creates checkpoints for resuming work

## Goal Identification System

The system uses a unified naming convention for all goals and tasks:

- All nodes (goals, subgoals, and tasks) use the 'G' prefix followed by a unique number
- This simplifies tracking and visualization
- Eliminates confusion between different types of nodes
- Makes the system more maintainable and extensible

## Tracing and Observation

For transparency and debugging, the system includes:

- Real-time tracing of agent activities
- Hierarchical goal visualization
- Points consumption tracking
- Decision points with reasoning
- Tool usage statistics
- Failure history tracking and visualization

## Implementation Considerations

Current implementation uses:
- OpenAI API for agent intelligence
- Pydantic models for structured data exchange
- Async I/O for efficient execution
- Function tools for repository interactions

## Future Enhancements

Potential future improvements include:
1. Parallel strategy exploration
2. Improved learning across sessions
3. Human-in-the-loop collaboration
4. Dynamic tool selection
5. Meta-learning to improve the overall process
6. Enhanced failure pattern recognition
7. Adaptive goal prioritization based on failure history

By combining strategic planning with tactical execution and learning from failures, this system aims to create an advanced AGI capable of tackling complex, long-horizon tasks in a transparent and controllable manner. 