"""
Basic tests for the 'goal decompose' command functionality.
"""

import os
import json
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
import tempfile

import sys
# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import the decompose_goal function directly for testing
from midpoint.agents.goal_decomposer import decompose_goal


class TestGoalDecompose(unittest.TestCase):
    """Test the goal decompose functionality directly."""
    
    def setUp(self):
        """Set up a test environment."""
        # Create a temp directory
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = self.temp_dir
        
        # Create a simple README file in the temp repo
        readme_path = Path(self.temp_dir) / "README.md"
        with open(readme_path, 'w') as f:
            f.write("# Test Repository\nThis is a test repository for goal decomposition.")
    
    def tearDown(self):
        """Clean up after tests."""
        # Clean up temp directory
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
    
    @patch('midpoint.agents.goal_decomposer.GoalDecomposer.determine_next_step')
    @patch('midpoint.agents.goal_decomposer.GoalDecomposer.create_top_goal_file')
    @patch('midpoint.agents.goal_decomposer.get_current_hash')
    def test_decompose_goal_basic(self, mock_get_hash, mock_create_file, mock_determine_next_step):
        """Test basic goal decomposition with validation bypassed."""
        # Setup mocks
        mock_get_hash.return_value = "abc123"
        mock_create_file.return_value = "G1-S1.json"
        
        # Create a mock subgoal plan
        from midpoint.agents.models import SubgoalPlan
        mock_plan = SubgoalPlan(
            next_step="Implement feature X",
            validation_criteria=["Test passes", "Feature works"],
            reasoning="This is a reasonable next step",
            requires_further_decomposition=True,
            relevant_context={}
        )
        mock_determine_next_step.return_value = mock_plan
        
        # Call the decompose_goal function directly with bypass_validation
        import asyncio
        result = asyncio.run(decompose_goal(
            repo_path=self.repo_path,
            goal="Create a new feature",
            input_file=None,
            parent_goal=None,
            goal_id=None,
            memory_repo=None,
            debug=False,
            quiet=True,
            bypass_validation=True
        ))
        
        # Verify the result
        self.assertTrue(result["success"])
        self.assertEqual(result["next_step"], "Implement feature X")
        self.assertEqual(result["validation_criteria"], ["Test passes", "Feature works"])
        self.assertTrue(result["requires_further_decomposition"])
        self.assertEqual(result["git_hash"], "abc123")
        self.assertEqual(result["goal_file"], "G1-S1.json")
    
    @patch('midpoint.agents.goal_decomposer.validate_repository_state')
    @patch('midpoint.agents.goal_decomposer.get_current_hash')
    def test_decompose_goal_with_invalid_repo(self, mock_get_hash, mock_validate):
        """Test goal decomposition with an invalid repository."""
        # Setup mocks
        mock_get_hash.return_value = "abc123"
        mock_validate.side_effect = ValueError("Invalid repository")
        
        # Call the decompose_goal function directly
        import asyncio
        result = asyncio.run(decompose_goal(
            repo_path=self.repo_path,
            goal="Create a new feature",
            input_file=None,
            parent_goal=None,
            goal_id=None,
            memory_repo=None,
            debug=False,
            quiet=True,
            bypass_validation=False  # Don't bypass to test validation
        ))
        
        # Verify the result
        self.assertFalse(result["success"])
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Invalid repository")


if __name__ == "__main__":
    unittest.main() 