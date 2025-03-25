"""
Test case for ensuring goals and subgoals are stored in the same directory.
"""

import os
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import pytest

from midpoint.goal_cli import create_new_goal, decompose_existing_goal, GOAL_DIR


class TestGoalDirectoryConsistency(unittest.TestCase):
    """Test case for ensuring goals and subgoals are stored in the same directory."""
    
    def setUp(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create fake git repository
        os.system("git init")
        os.system("git config user.email 'test@example.com'")
        os.system("git config user.name 'Test User'")
        os.system("touch README.md")
        os.system("git add README.md")
        os.system("git commit -m 'Initial commit'")
    
    def tearDown(self):
        """Clean up after the test."""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    @patch('midpoint.agents.goal_decomposer.AsyncOpenAI')
    @patch('midpoint.agents.goal_decomposer.ToolProcessor')
    @patch('midpoint.agents.goal_decomposer.initialize_all_tools')
    @pytest.mark.asyncio
    async def test_goal_and_subgoal_directories(self, mock_init_tools, mock_tool_processor, mock_openai):
        """Test that goals and subgoals are stored in the .goal directory while logs are in logs directory."""
        # Setup mocks for the goal decomposer
        mock_openai_instance = MagicMock()
        mock_openai.return_value = mock_openai_instance
        
        # Mock the OpenAI response to include next_step and validation_criteria
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.tool_calls = []
        mock_response.choices[0].message.content = json.dumps({
            "next_step": "Create a basic implementation of feature X",
            "validation_criteria": ["Code compiles", "Tests pass"],
            "reasoning": "Feature X is a core requirement",
            "requires_further_decomposition": False,
            "relevant_context": {}
        })
        mock_openai_instance.chat.completions.create.return_value = mock_response
        
        # Create a new top-level goal
        goal_id = create_new_goal("Implement feature X")
        
        # Verify the goal was created in the .goal directory
        goal_file = Path(f"{GOAL_DIR}/{goal_id}.json")
        self.assertTrue(goal_file.exists(), f"Goal file {goal_file} not found")
        
        # Decompose the goal
        result = await decompose_existing_goal(goal_id, bypass_validation=True)
        self.assertTrue(result, "Goal decomposition failed")
        
        # List all JSON files in the .goal directory
        goal_files = list(Path(GOAL_DIR).glob("*.json"))
        
        # Find subgoal files (files with pattern {goal_id}-S*.json)
        subgoal_pattern = f"{goal_id}-S*.json"
        subgoal_files = list(Path(GOAL_DIR).glob(subgoal_pattern))
        
        # We should have at least one subgoal in the .goal directory
        self.assertTrue(len(subgoal_files) > 0, f"No subgoals found matching pattern {subgoal_pattern}")
        
        # Make sure no goal files are in the logs directory
        logs_dir = Path("logs")
        if logs_dir.exists():
            logs_goal_files = list(logs_dir.glob("*.json"))
            self.assertEqual(len(logs_goal_files), 0, 
                            f"Found goal files in logs directory: {logs_goal_files}")
            
            # Verify that log files exist in the logs directory
            log_files = list(logs_dir.glob("goal_decomposer_*.log"))
            self.assertTrue(len(log_files) > 0, "No log files found in logs directory")
        
        # Verify the top-level goal and each subgoal is in the .goal directory
        for file in goal_files:
            with open(file, 'r') as f:
                data = json.load(f)
                # If this is a subgoal, its parent_goal should match goal_id
                if 'parent_goal' in data and data['parent_goal']:
                    self.assertEqual(data['parent_goal'], goal_id,
                                    f"Subgoal {file} has incorrect parent_goal: {data['parent_goal']}")


if __name__ == '__main__':
    unittest.main() 