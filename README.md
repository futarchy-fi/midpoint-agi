# Midpoint

A recursive goal decomposition and execution system for complex repository management tasks.

## Project Overview

Midpoint is an advanced AI system designed to overcome the fundamental limitation of current AI systems: the inability to effectively reason over long chains of thought and manage complex objectives. By decomposing complex goals into manageable subgoals and using a coordinated multi-agent approach, the system can tackle problems that would be intractable for a single agent.

## Current Development Status

⚠️ **Important**: The main orchestrator and CLI interface are currently under development. The system is not yet ready for production use.

### Working Components
- Goal Decomposition System
- Task Execution System
- Goal Validation System
- Repository State Management
- Memory Management System

### In Development
- Main Orchestrator (coordination between components)
- CLI Interface
- End-to-end workflow

### Testing and Development
Development and testing tools are available in the `tests/` directory. These tools can be used to test individual components while the main system is being developed.

## Project Structure

```
midpoint/
├── src/                    # Main source code
│   └── midpoint/          # Core package
│       ├── agents/        # Specialized AI agents
│       ├── config/        # Configuration files
│       ├── utils/         # Utility functions
│       └── orchestrator.py # Main orchestrator (in development)
├── scripts/               # Utility scripts
│   ├── hooks/             # Git hooks
│   └── memory_tools.py    # Memory system utilities
├── tests/                 # Test suite and development tools
├── docs/                  # Documentation
└── logs/                  # Log files
```

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -e .
   ```
   For development and testing:
   ```
   pip install -e ".[dev]"
   ```
3. Install git hooks (optional):
   ```
   make install-hooks
   ```

### Usage

Use the Makefile for common operations:

```
# Run the Midpoint system
make run

# Run tests
make test

# Run memory-specific tests
make test-memory

# Run critical tests (those included in pre-commit hook)
make test-critical

# Clean temporary files
make clean

# See all available commands
make help
```

### Testing

We use several mechanisms to ensure code quality:

1. **Automated Tests**: Run tests via the Makefile
   ```bash
   # Run all tests
   make test
   
   # Run memory-specific tests
   make test-memory
   ```

2. **Manual Testing**: Run specific test suites with pytest
   ```bash
   # Run specific test file
   python -m pytest tests/test_memory_tools.py
   
   # Run with verbose output
   python -m pytest tests/test_memory_*.py -v
   ```

3. **Pre-commit Hooks**: Automatically run tests before each commit
   - Install with `make install-hooks`
   - Prevents commits if tests fail
   - Currently runs:
     - Memory tests
     - Filesystem tools tests
     - Specific bugs tests 
     - Goal decomposer tools tests
     - Tools wrapper tests

Our test suite includes:
- **Unit Tests**: Test individual functions in isolation
- **Integration Tests**: Test multiple components working together

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
   - LLM-driven task execution
   - Multiple tool support (file editing, searching, command execution)
   - Git branch and commit management
   - Comprehensive logging and debugging
   - Error handling and recovery

### 🔄 In Progress Features

1. **Orchestration System**:
   - Integration of GoalDecomposer and TaskExecutor
   - Iterative execution with state feedback
   - Sequential subgoal implementation
   - Task dependency management
   
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
- Uses LLM-driven decision making
- Leverages multiple tools (file editing, search, commands)
- Creates git checkpoints with appropriate commits
- Provides detailed logging of all operations

### 3. GoalValidator (Verification)
- Evaluates whether a subgoal has been successfully achieved
- Tests against specific success criteria
- Provides detailed validation reports

### 4. Orchestrator (Coordination)
- Coordinates the workflow between agents
- Manages the state transitions between decomposition and execution
- Tracks overall progress toward the high-level goal
- Handles error recovery and retries

## Workflow

The typical workflow is:

1. **Planning Phase**:
   - GoalDecomposer analyzes the problem using the OODA loop
   - Determines the single most promising next step
   - Recursively decomposes until reaching an executable task

2. **Execution Phase**:
   - TaskExecutor implements the executable task
   - Creates branch for isolated execution
   - Uses LLM-driven decision making to implement changes
   - Commits changes when complete

3. **Validation Phase**:
   - GoalValidator assesses if the task was achieved
   - Provides detailed validation report

4. **Orchestration Loop**:
   - If task was successful, returns to Planning Phase with the new state
   - Continues until the high-level goal is achieved
   - Handles failures through retries or alternative approaches

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Agent Memory System

The system now includes an agent memory feature, allowing agents to retain and recall information across sessions. The memory system is organized as a separate Git repository storing memory documents categorized by type.

#### Key Components:

- **scripts/init_memory_repo.py**: Initializes a memory repository with appropriate structure
- **scripts/memory_tools.py**: Utility functions for storing, retrieving memories, and managing cross-references
- **scripts/memory_example.py**: Simple example demonstrating memory usage
- **scripts/memory_integration.py**: Example showing integration with agent systems
- **scripts/memory_revert_example.py**: Tool for reverting to previous states with synchronized code and memory

#### Features:

- **Document-based Memory**: Store and retrieve memories as documents
- **Category Organization**: Organize memories by type (reasoning, observations, decisions)
- **Cross-referencing System**: Track relationships between code and memory states
- **Historical Tracking**: Maintain complete history of all code-memory mappings
- **State Reversion**: Ability to revert to any previous state with matching code and memory

#### Usage:

1. Initialize the memory repository:
   ```bash
   python scripts/init_memory_repo.py ~/.midpoint/memory
   ```

2. Store and retrieve memories:
   ```bash
   # Store a memory document
   python -m scripts.memory_tools store reasoning --content "This is important reasoning"
   
   # Retrieve recent memory documents
   python -m scripts.memory_tools retrieve --category observations
   ```

3. View cross-reference history:
   ```bash
   python -m scripts.memory_tools history
   ```

4. Revert to previous states:
   ```bash
   python -m scripts.memory_revert_example interactive
   ```

See the [Memory System Documentation](docs/memory_system.md) for complete details. 