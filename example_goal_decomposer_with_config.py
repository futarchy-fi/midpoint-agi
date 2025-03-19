#!/usr/bin/env python
"""
Example of using the GoalDecomposer with configuration.
"""

import os
import asyncio
from midpoint.agents.config import get_openai_api_key
from midpoint.agents.models import State, Goal, TaskContext
from midpoint.agents.goal_decomposer import GoalDecomposer

async def main():
    """Run the example."""
    # Get API key from environment
    api_key = get_openai_api_key()
    if not api_key:
        print("Error: OPENAI_API_KEY not set in environment")
        return

    # Create a test repository state
    state = State(
        repository_path=".",
        current_branch="main",
        current_hash="abc123"
    )

    # Create a test goal
    goal = Goal(
        description="Add a new feature to calculate Fibonacci numbers",
        validation_criteria=[
            "New file fibonacci.py exists",
            "File contains a function to calculate Fibonacci numbers",
            "Function has proper documentation",
            "Tests exist and pass"
        ]
    )

    # Create task context
    context = TaskContext(
        state=state,
        goal=goal,
        execution_history=[]
    )

    # Create and run the decomposer
    decomposer = GoalDecomposer()
    result = await decomposer.determine_next_step(context)

    # Print the result
    print("\nNext step plan:")
    print(f"Description: {result.next_step}")
    print("\nValidation criteria:")
    for criterion in result.validation_criteria:
        print(f"- {criterion}")
    print(f"\nRequires further decomposition: {result.requires_further_decomposition}")
    print("\nRelevant context:")
    for key, value in result.relevant_context.items():
        print(f"- {key}: {value}")

if __name__ == "__main__":
    asyncio.run(main()) 