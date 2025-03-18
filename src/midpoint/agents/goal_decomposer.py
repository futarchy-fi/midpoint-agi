"""
Goal Decomposition agent implementation.

This module implements the GoalDecomposer agent that breaks down complex goals
into manageable subgoals and execution steps.
"""

import os
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from midpoint.agents.models import State, Goal, StrategyPlan, TaskContext
from midpoint.agents.tools import track_points
from midpoint.agents.config import get_openai_api_key

class GoalDecomposer:
    """Agent responsible for decomposing complex goals into manageable steps."""
    
    def __init__(self):
        """Initialize the GoalDecomposer agent."""
        # Initialize OpenAI client with API key from config
        api_key = get_openai_api_key()
        if not api_key:
            raise ValueError("OpenAI API key not found in config or environment")
        self.client = AsyncOpenAI(api_key=api_key)
        
        # Define the system prompt for goal decomposition
        self.system_prompt = """You are an expert software architect and project planner.
Your task is to break down complex software development goals into clear, actionable steps.
For each goal, you should:
1. Analyze the requirements and validation criteria
2. Break down the goal into logical subgoals
3. Create a detailed execution plan with concrete steps
4. Estimate the points needed for each step
5. Ensure the plan is feasible within the given budget

Your output should be structured and include:
- A clear strategy description
- A list of concrete execution steps
- Reasoning for the chosen approach
- Point estimates for each step
- Total points needed"""

    async def decompose_goal(self, context: TaskContext) -> StrategyPlan:
        """
        Decompose a goal into a strategy plan.
        
        Args:
            context: The current task context containing the goal and state
            
        Returns:
            A StrategyPlan containing the decomposed steps and estimates
            
        Raises:
            ValueError: If the goal or context is invalid
            Exception: For other errors during decomposition
        """
        # Validate inputs
        if not context.goal:
            raise ValueError("No goal provided in context")
        if context.total_budget <= 0:
            raise ValueError("Invalid points budget")
            
        # Track points for this operation
        await track_points("goal_decomposition", 10)
        
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
            
            # Parse the response
            strategy = self._parse_response(response.choices[0].message.content)
            
            # Validate the strategy
            self._validate_strategy(strategy, context)
            
            return strategy
            
        except Exception as e:
            raise Exception(f"Error during goal decomposition: {str(e)}")
    
    def _create_user_prompt(self, context: TaskContext) -> str:
        """Create the user prompt for the OpenAI API."""
        return f"""Goal: {context.goal.description}

Validation Criteria:
{chr(10).join(f"- {criterion}" for criterion in context.goal.validation_criteria)}

Current State:
- Git Hash: {context.state.git_hash}
- Description: {context.state.description}

Context:
- Iteration: {context.iteration}
- Points Consumed: {context.points_consumed}
- Total Budget: {context.total_budget}

Please provide a detailed strategy plan for achieving this goal."""
    
    def _parse_response(self, response: str) -> StrategyPlan:
        """Parse the OpenAI API response into a StrategyPlan."""
        # Split response into sections
        sections = response.split("\n\n")
        
        # Extract strategy description
        strategy_desc = sections[0].strip()
        
        # Extract steps
        steps = []
        for section in sections:
            if section.lower().startswith("steps:"):
                steps = [step.strip("- ").strip() 
                        for step in section.split("\n")[1:]
                        if step.strip()]
                break
        
        # Extract reasoning
        reasoning = ""
        for section in sections:
            if section.lower().startswith("reasoning:"):
                reasoning = section.split(":", 1)[1].strip()
                break
        
        # Extract point estimates
        points = 0
        for section in sections:
            if "points" in section.lower():
                try:
                    points = int(section.split(":")[1].strip())
                except:
                    pass
                break
        
        return StrategyPlan(
            steps=steps,
            reasoning=reasoning,
            estimated_points=points,
            metadata={
                "strategy_description": strategy_desc,
                "raw_response": response
            }
        )
    
    def _validate_strategy(self, strategy: StrategyPlan, context: TaskContext) -> None:
        """Validate the generated strategy."""
        if not strategy.steps:
            raise ValueError("Strategy has no steps")
            
        if not strategy.reasoning:
            raise ValueError("Strategy has no reasoning")
            
        if strategy.estimated_points <= 0:
            raise ValueError("Invalid point estimate")
            
        if strategy.estimated_points > context.total_budget:
            raise ValueError("Strategy exceeds available budget")
            
        # Verify all validation criteria are addressed
        criteria_covered = set()
        for step in strategy.steps:
            for criterion in context.goal.validation_criteria:
                if criterion.lower() in step.lower():
                    criteria_covered.add(criterion)
        
        if len(criteria_covered) < len(context.goal.validation_criteria) * context.goal.success_threshold:
            raise ValueError("Strategy does not cover enough validation criteria") 