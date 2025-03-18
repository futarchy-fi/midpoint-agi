"""
Example script to test the GoalDecomposer functionality.
"""

import asyncio
import os
from pathlib import Path
import json
from unittest.mock import patch, AsyncMock
from midpoint.agents.models import State, Goal, TaskContext, StrategyPlan
from midpoint.agents.goal_decomposer import GoalDecomposer
from midpoint.agents.tools import get_current_hash

# Sample mock response
MOCK_RESPONSE = """
Strategy for Adding User Authentication

Steps:
1. Set up a user model with email, password (hashed), and session data
2. Create registration endpoint to allow new users to sign up with email and password
3. Implement password hashing using bcrypt or similar library for secure password storage
4. Create login endpoint for user authentication with credentials
5. Implement session management and token generation to maintain user sessions
6. Add logout functionality to invalidate sessions when users logout
7. Implement middleware for protected routes
8. Create password reset functionality (optional)
9. Add frontend forms for registration and login
10. Implement proper error handling and validation

Reasoning: The strategy breaks down authentication into logical components starting with data models, then core functionality (register/login/logout), followed by session management and finally frontend integration. This approach ensures each piece can be tested independently before moving to the next step.

Points: 500
"""

async def main():
    """Run a test of the GoalDecomposer."""
    # Get the test repository path
    repo_path = os.getenv("MIDPOINT_TEST_REPO", os.path.expanduser("~/midpoint-test-repo"))
    repo_path = Path(repo_path)
    
    if not repo_path.exists():
        print(f"Error: Test repository not found at {repo_path}")
        print("Please run: python examples/setup_test_repo.py")
        return
        
    # Create a test goal
    goal = Goal(
        description="Add user authentication to the application",
        validation_criteria=[
            "Users can register with email and password",
            "Users can login with credentials",
            "Users can logout",
            "User sessions are maintained",
            "Passwords are securely hashed"
        ],
        success_threshold=0.8
    )
    
    # Get current repository state
    current_hash = await get_current_hash(str(repo_path))
    
    # Create task context
    context = TaskContext(
        state=State(
            git_hash=current_hash,
            description="Initial state before adding authentication"
        ),
        goal=goal,
        iteration=0,
        points_consumed=0,
        total_budget=1000,
        execution_history=[]
    )
    
    # Create and run the GoalDecomposer
    decomposer = GoalDecomposer()
    
    try:
        # Check if we have a valid API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key or api_key == "sk-..." or not api_key.startswith("sk-"):
            print("\nUsing mock response as no valid API key was found.")
            # Use mock data instead of calling the OpenAI API
            with patch.object(decomposer, "_validate_strategy", return_value=None):
                with patch.object(decomposer, "_parse_response", return_value=create_mock_strategy()):
                    with patch.object(decomposer.client.chat.completions, "create", new_callable=AsyncMock) as mock_create:
                        mock_create.return_value.choices = [AsyncMock()]
                        mock_create.return_value.choices[0].message.content = MOCK_RESPONSE
                        strategy = await decomposer.decompose_goal(context)
        else:
            print("\nDecomposing goal using OpenAI API...")
            strategy = await decomposer.decompose_goal(context)
        
        print("\nGenerated Strategy:")
        print("-" * 50)
        print(f"Description: {strategy.metadata['strategy_description']}")
        print("\nSteps:")
        for i, step in enumerate(strategy.steps, 1):
            print(f"{i}. {step}")
        print(f"\nReasoning: {strategy.reasoning}")
        print(f"Estimated Points: {strategy.estimated_points}")
        print("-" * 50)
        
    except Exception as e:
        print(f"\nError: {e}")
        print("\nPlease check:")
        print("1. Your OpenAI API key is set in the environment")
        print("2. The test repository exists and is a valid git repository")
        print("3. You have an active internet connection")

def create_mock_strategy():
    """Create a mock strategy for testing without API calls."""
    return StrategyPlan(
        steps=[
            "Set up a user model with email, password (hashed), and session data",
            "Create registration endpoint to allow new users to sign up with email and password",
            "Implement password hashing using bcrypt or similar library for secure password storage",
            "Create login endpoint for user authentication with credentials",
            "Implement session management and token generation to maintain user sessions",
            "Add logout functionality to invalidate sessions when users logout",
            "Implement middleware for protected routes",
            "Create password reset functionality (optional)",
            "Add frontend forms for registration and login",
            "Implement proper error handling and validation"
        ],
        reasoning="The strategy breaks down authentication into logical components starting with data models, then core functionality (register/login/logout), followed by session management and finally frontend integration. This approach ensures each piece can be tested independently before moving to the next step.",
        estimated_points=500,
        metadata={
            "strategy_description": "Strategy for Adding User Authentication",
            "raw_response": MOCK_RESPONSE
        }
    )

if __name__ == "__main__":
    asyncio.run(main()) 