# Midpoint

A recursive goal decomposition and execution system for complex repository management tasks.

## Project Overview

Midpoint is an advanced AI system designed to overcome the fundamental limitation of current AI systems: the inability to effectively reason over long chains of thought and manage complex objectives. By decomposing complex goals into manageable subgoals and using a coordinated multi-agent approach, the system can tackle problems that would be intractable for a single agent.

The system follows these key principles:
- **Recursive Goal Decomposition**: Complex goals are broken down into progressively simpler subgoals until reaching directly executable tasks.
- **OODA Loop Decision Making**: Each step follows Observe-Orient-Decide-Act to determine the best next action.
- **Repository-based State Management**: The system uses git repositories as its state representation, with commits providing verifiable checkpoints.
- **Intelligent Agent Execution**: Tasks are executed by specialized agents capable of making multiple tool calls and adapting to the repository state.
- **Smart Validation**: The system intelligently validates work against success criteria by exploring the repository and running tests.

## Current Implementation Status

### ✅ Fully Implemented Features

1. **Git Repository State Management**:
   - Branch creation and management
   - State checking for uncommitted changes
   - Hash-based state tracking
   - Commit management
   
2. **Goal Decomposition Agent**:
   - Recursive decomposition of complex goals
   - OODA loop implementation
   - Repository state exploration
   - Selective context passing
   - Executable task identification

3. **Task Execution Agent**:
   - Smart agent that can understand and execute tasks
   - Multiple tool support (file editing, searching, command execution)
   - Git branch and commit management
   - Error handling and recovery
   
4. **Goal Validation Agent**:
   - Intelligent validation of task results
   - Evidence collection using repository tools
   - File existence and content verification
   - Test execution verification
   - Detailed validation reporting

### 🔄 In Progress Features

1. **Orchestration System**:
   - Full integration of all agent components
   - End-to-end workflow management
   - Complex multi-step task handling
   
2. **Human Supervision System**:
   - Interactive mode with approval steps
   - Progress visualization
   - Command inspection and modification

### 📋 Planned Features

1. **Failure Analysis Agent**:
   - Advanced diagnostics for failed executions
   - Root cause identification
   - Improvement suggestions

2. **Progress Summarization**:
   - Concise summaries of execution steps
   - Context management across multiple steps
   - Key decision tracking

## Quick Start

1. Clone the repository:
```bash
git clone <repository-url>
cd midpoint
```

2. Set up the development environment:
```bash
# Make the setup script executable
chmod +x setup.sh

# Run the setup script
./setup.sh
```

3. Verify your environment:
```bash
python check_env.py
```

4. Run tests:
```bash
python -m pytest tests/
```

## Using the System

### Running the Full Flow Test

The full flow test demonstrates the end-to-end process from goal decomposition to execution and validation:

```bash
python examples/test_deep_flow.py <repository_path> "<goal_description>"
```

Example:
```bash
python examples/test_deep_flow.py test-repo "Create a task management system with basic functionality for adding, listing, and completing tasks"
```

This will:
1. Recursively decompose the goal into manageable subgoals
2. Execute the most concrete subgoal using the TaskExecutor
3. Validate the execution using the GoalValidator

### Running Only the Recursive Decomposition

If you only want to see the goal decomposition without execution:

```bash
python run_recursive_decomposition.py <repository_path> "<goal_description>"
```

This will show the recursive decomposition process and output a hierarchical log of the decomposed goals.

### Creating a Test Repository

For testing, you can use the provided test repository or create a new one:

```bash
# Create a new test repository
mkdir test-repo
cd test-repo
git init
echo "# Test Repository" > README.md
git add README.md
git commit -m "Initial commit"
cd ..
```

## Development Setup

The project requires a Python virtual environment and certain environment variables to be set up. The setup script (`setup.sh`) handles most of this automatically, but here's what it does:

1. Creates a virtual environment (`.venv`)
2. Activates the virtual environment
3. Installs the package in development mode
4. Creates an environment check script

### Environment Variables

Create a `.env` file in the project root with the following variables:

```
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4-1106-preview
```

You can copy the example file and modify it:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Manual Setup

If you prefer to set up manually:

1. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install the package in development mode:
```bash
python setup_dev.py
```

## Running Tests

Before running tests, ensure your environment is set up correctly:

1. Activate the virtual environment:
```bash
source .venv/bin/activate
```

2. Run the environment check:
```bash
python check_env.py
```

3. Run the tests:
```bash
python -m pytest tests/
```

## System Architecture

The system consists of specialized agents, each with a distinct role:

### 1. GoalDecomposer (Strategic Planning)
- Breaks complex goals into manageable subgoals
- Focuses on the single most important next step
- Recursively decomposes until reaching executable tasks

### 2. TaskExecutor (Implementation)
- Implements directly executable tasks 
- Uses multiple tools (file editing, search, commands)
- Creates git checkpoints with appropriate commits

### 3. GoalValidator (Verification)
- Evaluates whether a subgoal has been successfully achieved
- Tests against specific success criteria
- Provides detailed validation reports

## Workflow

The typical workflow is:

1. **Planning Phase**:
   - GoalDecomposer analyzes the problem using the OODA loop
   - Determines the single most promising next step
   - Recursively decomposes until reaching an executable task

2. **Execution Phase**:
   - TaskExecutor implements the executable task
   - Creates a detailed trace of its actions

3. **Validation Phase**:
   - GoalValidator assesses if the task was achieved
   - If successful, system returns to Planning Phase with the new state
   - If unsuccessful, system retries or provides error information

## Project Structure

```
midpoint/
├── src/
│   └── midpoint/         # Main package
│       ├── __init__.py
│       └── agents/       # Agent implementations
│           ├── __init__.py
│           ├── goal_decomposer.py  # Goal decomposition agent
│           ├── task_executor.py    # Task execution agent
│           ├── goal_validator.py   # Result validation agent
│           ├── models.py          # Shared data models
│           ├── tools.py           # Git and utility functions
│           └── config.py          # Configuration management
├── examples/              # Example scripts and tests
│   ├── test_deep_flow.py  # End-to-end test
│   ├── setup_test_repo.py # Test repository setup
│   └── verify_setup.py    # Setup verification
├── tests/                 # Test suite
│   ├── test_goal_decomposer.py
│   ├── test_task_executor.py
│   └── test_repo_context.py
└── docs/                  # Documentation
    ├── FEATURES.md        # Feature implementation status
    └── VISION.md          # System vision and architecture
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 