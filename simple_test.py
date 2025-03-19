import asyncio
import os
import sys
import argparse
from typing import Optional
from check_env import check_environment, EnvironmentError

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run the GoalDecomposer demo")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force run even with environment issues (not recommended)"
    )
    parser.add_argument(
        "--skip-checks",
        type=str,
        help="Comma-separated list of environment checks to skip"
    )
    parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip strategy validation (not recommended)"
    )
    return parser.parse_args()

async def run_demo(force: bool = False, skip_checks: Optional[str] = None, no_validation: bool = False):
    """Run a simple demonstration of the GoalDecomposer without external API calls."""
    # Run environment check first
    try:
        skip_checks_list = skip_checks.split(",") if skip_checks else None
        check_environment(force=force, skip_checks=skip_checks_list)
    except EnvironmentError as e:
        if not force:
            print(f"\033[1;31mError: {str(e)}\033[0m")
            print("\033[1;33mTo force run (not recommended):\033[0m")
            print("python simple_test.py --force")
            print("\033[1;33mTo skip specific checks:\033[0m")
            print("python simple_test.py --skip-checks venv,package,api_key")
            sys.exit(1)
    
    # Import after environment check
    from midpoint.agents.models import State, Goal, TaskContext, StrategyPlan
    from midpoint.agents.goal_decomposer import GoalDecomposer
    from unittest.mock import patch
    
    # Set dummy OpenAI API key
    os.environ["OPENAI_API_KEY"] = "sk-" + "a" * 48
    
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
    
    if not no_validation:
        print("\n=== Validation ===")
        try:
            decomposer._validate_strategy(strategy, context)
            print("Strategy validation: PASSED")
        except ValueError as e:
            print(f"Strategy validation: FAILED - {str(e)}")
            if not force:
                print("\n\033[1;33mTo skip validation (not recommended):\033[0m")
                print("python simple_test.py --no-validation")
                sys.exit(1)
    else:
        print("\n\033[1;33m⚠️  Strategy validation skipped (not recommended)\033[0m")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_demo(
        force=args.force,
        skip_checks=args.skip_checks,
        no_validation=args.no_validation
    )) 