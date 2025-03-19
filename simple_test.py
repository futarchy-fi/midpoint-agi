import asyncio
import os
from midpoint.agents.models import State, Goal, TaskContext, StrategyPlan
from midpoint.agents.goal_decomposer import GoalDecomposer
from unittest.mock import patch

# Set dummy OpenAI API key
os.environ["OPENAI_API_KEY"] = "sk-" + "a" * 48

async def run_demo():
    """Run a simple demonstration of the GoalDecomposer without external API calls."""
    # Create a sample goal
    goal = Goal(
        description="Add a new feature to handle user authentication",
        validation_criteria=[
            "User can register with email and password",
            "User can login with credentials",
            "User can logout",
            "User session is maintained"
        ],
        success_threshold=0.8
    )
    
    # Create a sample context
    context = TaskContext(
        state=State(
            git_hash="dummy-hash",
            description="Initial state with test file"
        ),
        goal=goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Create a sample response that would normally come from the API
    sample_response = """Strategy: Implement authentication system

Steps:
- Create user model with email and password fields
- Implement user registration endpoint
- Implement user login endpoint
- Implement session management
- Add logout functionality
- Secure protected routes

Reasoning: This approach follows best practices for authentication systems by separating concerns and implementing each feature incrementally.

Points: 500"""

    # Create a GoalDecomposer instance
    decomposer = GoalDecomposer()
    
    # Use the _parse_response method directly to parse the sample response
    strategy = decomposer._parse_response(sample_response)
    
    # Print the results
    print("\n=== Strategy ===")
    print(f"Description: {strategy.metadata.get('strategy_description', 'No description')}")
    print("\n=== Steps ===")
    for i, step in enumerate(strategy.steps, 1):
        print(f"{i}. {step}")
    print(f"\n=== Reasoning ===\n{strategy.reasoning}")
    print(f"\n=== Estimated Points ===\n{strategy.estimated_points}")
    print("\n=== Validation ===")
    try:
        decomposer._validate_strategy(strategy, context)
        print("Strategy validation: PASSED")
    except ValueError as e:
        print(f"Strategy validation: FAILED - {str(e)}")

if __name__ == "__main__":
    asyncio.run(run_demo()) 