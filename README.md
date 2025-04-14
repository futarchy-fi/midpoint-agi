# Midpoint

Midpoint is an advanced AGI system that can recursively follow goal-decomposition and planning when applied to the task of modifying repositories using git.

## Current Development Status

This project is currently in active development. The core architecture is implemented, but many features are still being refined and enhanced.

## Project Structure

```
midpoint/
├── agents/                 # Agent implementations
│   ├── goal_analyzer.py    # Strategic planning agent
│   ├── task_executor.py    # Implementation agent
│   ├── goal_validator.py   # Verification agent
│   ├── failure_analyzer.py # Diagnostics agent
│   └── progress_summarizer.py # Context management agent
├── models/                 # Data models
│   ├── goal.py             # Goal and subgoal models
│   ├── task.py             # Task models
│   ├── execution.py        # Execution trace models
│   ├── validation.py       # Validation result models
│   ├── failure.py          # Failure analysis models
│   └── summary.py          # Execution summary models
├── tools/                  # Tool implementations
│   ├── git_tools.py        # Git operations
│   ├── file_tools.py       # File system operations
│   ├── search_tools.py     # Web search capabilities
│   └── code_tools.py       # Code analysis and modification
├── memory/                 # Agent memory system
│   ├── memory_manager.py   # Memory management
│   ├── memory_store.py     # Memory storage
│   └── memory_retriever.py # Memory retrieval
├── utils/                  # Utility functions
│   ├── tracing.py          # Tracing utilities
│   ├── visualization.py    # Visualization utilities
│   └── config.py           # Configuration utilities
├── tests/                  # Test suite
├── docs/                   # Documentation
│   ├── VISION.md           # System vision and architecture
│   └── API.md              # API documentation
├── examples/               # Example usage
├── requirements.txt        # Dependencies
└── main.py                 # Entry point
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/midpoint.git
cd midpoint

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
from midpoint import Midpoint

# Initialize the system
midpoint = Midpoint(
    openai_api_key="your-api-key",
    repository_path="/path/to/repo",
    initial_commit="abc123"
)

# Define a goal
goal = {
    "description": "Add a new feature to the codebase",
    "validation_criteria": ["Tests pass", "Documentation updated"],
    "budget": 50000
}

# Execute the goal
result = midpoint.execute(goal)

# Check the result
if result.success:
    print(f"Goal achieved! Final commit: {result.final_commit}")
else:
    print(f"Goal not achieved: {result.failure_reason}")
```

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_goal_analyzer.py

# Run with coverage
pytest --cov=midpoint
```

## System Architecture

Midpoint uses a multi-agent architecture with five specialized agents:

1. **GoalAnalyzer**: Analyzes the current state and determines the most appropriate next action, including whether to decompose, execute, validate, or give up on a goal.

2. **TaskExecutor**: Implements directly executable tasks identified by the GoalAnalyzer.

3. **GoalValidator**: Evaluates whether a subgoal has been successfully achieved.

4. **FailureAnalyzer**: Analyzes failed executions to understand why they didn't succeed.

5. **ProgressSummarizer**: Creates concise summaries of successful executions to preserve critical context.

The system follows an OODA loop (Observe-Orient-Decide-Act) approach to problem-solving, with recursive goal decomposition using a depth-first search strategy.

### Workflow

1. **Planning Phase**: GoalAnalyzer determines the most promising next step
2. **Execution Phase**: TaskExecutor implements the task
3. **Validation Phase**: GoalValidator assesses if the task was achieved
4. **Failure Handling Phase**: If a goal is impossible, failure is propagated to parent goals
5. **Recursion**: Process continues until goal is achieved, budget is exhausted, or all paths are impossible

### Goal Identification System

The system uses a unified naming convention for all goals and tasks:

- All nodes (goals, subgoals, and tasks) use the 'G' prefix followed by a unique number
- This simplifies tracking and visualization
- Eliminates confusion between different types of nodes

### Failure Handling

When a goal is determined to be impossible or no longer relevant:

- The goal is marked as "given_up"
- Failure information is propagated to the parent goal
- The parent goal is analyzed in the next iteration
- If no parent exists, the overall process is marked as unsuccessful

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Agent Memory System

The Midpoint system includes a sophisticated memory system that allows agents to store and retrieve information across multiple executions. This enables the system to learn from past experiences and make more informed decisions.

### Key Components

1. **Memory Manager**: Coordinates memory operations across agents
2. **Memory Store**: Persists memory data to disk
3. **Memory Retriever**: Retrieves relevant memories based on context

### Memory Types

- **Short-term Memory**: Temporary storage for the current execution
- **Long-term Memory**: Persistent storage across multiple executions
- **Episodic Memory**: Records of past executions and their outcomes
- **Semantic Memory**: Knowledge about the codebase and domain

### Usage

```python
# Store a memory
midpoint.memory.store(
    agent="goal_analyzer",
    memory_type="episodic",
    content={
        "goal_id": "G1",
        "action": "decompose",
        "reasoning": "The goal requires multiple steps",
        "timestamp": "2023-01-01T12:00:00Z"
    }
)

# Retrieve memories
memories = midpoint.memory.retrieve(
    agent="goal_analyzer",
    memory_type="episodic",
    filter={"goal_id": "G1"}
)
```

### Memory Visualization

The system includes tools for visualizing the memory graph, showing relationships between different memories and how they influence decision-making. 