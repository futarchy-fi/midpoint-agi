#!/usr/bin/env python
"""
Example script demonstrating the GoalDecomposer in action with a custom parser.

This script creates a sample task context and uses the GoalDecomposer
with a custom parser to handle the detailed API response format.
"""

import asyncio
import os
import json
import re
from pathlib import Path
from midpoint.agents.models import State, Goal, TaskContext, StrategyPlan
from midpoint.agents.goal_decomposer import GoalDecomposer
from openai import AsyncOpenAI
from midpoint.agents.config import get_openai_api_key

# Override the API key with the one from config.json
def load_api_key_from_config():
    """Load the OpenAI API key from the config.json file."""
    config_path = Path.home() / ".midpoint" / "config.json"
    if config_path.exists():
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            api_key = config.get('openai', {}).get('api_key')
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
                return True, api_key
        except (json.JSONDecodeError, FileNotFoundError, KeyError):
            pass
    return False, None

# Custom version of the GoalDecomposer with an improved parser
class CustomGoalDecomposer(GoalDecomposer):
    """A custom version of the GoalDecomposer with an improved parser."""
    
    def __init__(self):
        """Initialize with a modified system prompt to encourage a specific format."""
        # Initialize OpenAI client with API key from config
        api_key = os.environ["OPENAI_API_KEY"]
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        if not api_key.startswith("sk-"):
            raise ValueError("Invalid OpenAI API key format")
            
        # Initialize OpenAI client
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Define a modified system prompt that encourages bullet points for steps
        self.system_prompt = """You are an expert software architect and project planner.
Your task is to break down complex software development goals into clear, actionable steps.
For each goal, you should:
1. Analyze the requirements and validation criteria
2. Break down the goal into logical subgoals
3. Create a detailed execution plan with concrete steps
4. Estimate the points needed for each step
5. Ensure the plan is feasible within the given budget

Your output MUST include:
1. A "Strategy" section with a brief description
2. A "Steps" section with a bullet point list of concrete steps (start each with '- ')
3. A "Reasoning" section explaining your approach
4. A "Points" section with your estimate of total points needed

Example format:
Strategy: Brief description of the strategy

Steps:
- Step 1: Description of first step
- Step 2: Description of second step
- Step 3: Description of third step
...etc.

Reasoning: Explanation of the chosen approach...

Points: Total estimated points (e.g., 500)"""
    
    async def decompose_goal(self, context):
        """Override to add custom parsing."""
        print("\nSending request to OpenAI API...")
        
        # Prepare the user prompt
        user_prompt = self._create_user_prompt(context)
        
        # Call OpenAI API
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=2000
            )
            
            # Get raw response
            raw_response = response.choices[0].message.content
            print("\nReceived raw response from API:\n")
            print("-" * 80)
            print(raw_response)
            print("-" * 80)
            print("\nParsing response...")
            
            # Use custom parsing
            strategy = self.custom_parse_response(raw_response)
            
            # Skip validation for this example
            return strategy
            
        except Exception as e:
            print(f"\nError during API call or processing: {str(e)}")
            raise

    def custom_parse_response(self, response_text):
        """
        Parse the response from the OpenAI API.
        
        This parser handles various formats, looking first for the expected format
        and falling back to extracting information from other formats.
        """
        # Initialize variables
        strategy_desc = ""
        steps = []
        reasoning = ""
        points = 0
        
        # Try to extract strategy
        if "Strategy:" in response_text:
            strategy_parts = response_text.split("Strategy:", 1)[1].split("\n", 1)
            if strategy_parts:
                strategy_desc = "Strategy: " + strategy_parts[0].strip()
        
        # Try to extract steps section
        if "Steps:" in response_text:
            steps_section = response_text.split("Steps:", 1)[1]
            if "Reasoning:" in steps_section:
                steps_section = steps_section.split("Reasoning:", 1)[0]
            
            # Find all bullet points in the steps section
            bullet_points = re.findall(r'- (.*?)(?=\n-|\n\n|$)', steps_section)
            if bullet_points:
                steps = [bp.strip() for bp in bullet_points if bp.strip()]
        
        # If no steps found, try alternative patterns
        if not steps:
            # Look for numbered steps
            numbered_steps = re.findall(r'[0-9]+\.\s+(.*?)(?=\n[0-9]+\.|\n\n|$)', response_text)
            if numbered_steps:
                steps = [step.strip() for step in numbered_steps if step.strip()]
            else:
                # Fall back to any bullet points in the text
                bullet_points = re.findall(r'- (.*?)(?=\n-|\n\n|$)', response_text)
                steps = [bp.strip() for bp in bullet_points if bp.strip()]
        
        # Try to extract reasoning
        if "Reasoning:" in response_text:
            reasoning_section = response_text.split("Reasoning:", 1)[1]
            if "Points:" in reasoning_section:
                reasoning = reasoning_section.split("Points:", 1)[0].strip()
            else:
                reasoning = reasoning_section.strip()
        
        # Try to extract total points directly
        total_points_patterns = [
            r'Total Estimated Points:\s*([0-9]+)',
            r'Total Points:\s*([0-9]+)',
            r'Total Point Estimate:\s*([0-9]+)',
            r'Estimated Points:\s*([0-9]+)',
            r'Points:\s*([0-9]+)'
        ]
        
        for pattern in total_points_patterns:
            matches = re.findall(pattern, response_text, re.IGNORECASE)
            if matches:
                try:
                    points = int(matches[0])
                    break
                except (ValueError, IndexError):
                    pass
        
        # If total points not found directly, try to extract and sum individual step points
        if points == 0:
            # Look for patterns like "**Step X:** Y points" or "Step X: Y points"
            step_points = re.findall(r'Step [0-9]+:\*?\*?\s+[0-9]+\s+points', response_text, re.IGNORECASE)
            if step_points:
                point_values = []
                for step in step_points:
                    # Extract the numeric value
                    match = re.search(r'([0-9]+)\s+points', step, re.IGNORECASE)
                    if match:
                        try:
                            point_values.append(int(match.group(1)))
                        except (ValueError, IndexError):
                            pass
                if point_values:
                    points = sum(point_values)
        
        # Default values if not found
        if not strategy_desc and steps:
            strategy_desc = "Strategy: " + steps[0]
        if not reasoning:
            reasoning = "Not provided"
        if points == 0:
            points = 500  # Default
        
        return StrategyPlan(
            steps=steps,
            reasoning=reasoning,
            estimated_points=points,
            metadata={
                "strategy_description": strategy_desc if strategy_desc else "Not specified",
                "raw_response": response_text
            }
        )

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