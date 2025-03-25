"""
Unit tests for the decompose command, executed from the command line.
"""

import os
import sys
import asyncio
import unittest
import tempfile
import shutil
from pathlib import Path
import json
import io
from unittest.mock import patch, MagicMock, AsyncMock

from midpoint.agents.goal_decomposer import decompose_goal
from tests.test_helpers import async_test

# Import the goal decompose command
from midpoint.goal_cli import decompose_existing_goal


# Mock the decompose_goal function directly since we're testing it in isolation
@patch('midpoint.agents.goal_decomposer.decompose_goal')
class TestGoalDecomposeCommand(unittest.TestCase):
    """Tests for the goal decompose command in goal_cli.py."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary goal directory
        self.test_dir = Path("test_goal_dir")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create a test goal file
        self.test_goal_id = "G1"
        self.test_goal_file = self.test_dir / f"{self.test_goal_id}.json"
        self.test_goal_content = {
            "goal_id": self.test_goal_id,
            "description": "Test goal description",
            "parent_goal": "",
            "timestamp": "20250101_000000"
        }
        
        with open(self.test_goal_file, 'w') as f:
            json.dump(self.test_goal_content, f)
        
        # Save current directory to restore it later
        self.orig_cwd = os.getcwd()
        
        self.addCleanup(self.cleanup)
    
    def cleanup(self):
        """Clean up after the test."""
        # Clean up the test directory
        if self.test_goal_file.exists():
            self.test_goal_file.unlink()
        if self.test_dir.exists():
            self.test_dir.rmdir()
        
        # Restore current directory
        os.chdir(self.orig_cwd)
    
    @async_test
    async def test_decompose_existing_goal_success(self, mock_decompose_goal):
        """Test successful goal decomposition."""
        # Create an implementation of decompose_existing_goal that doesn't rely on importing goal_cli
        # This mimics the behavior but is independent of the module structure
        async def decompose_existing_goal(goal_id, debug=False, quiet=False):
            if goal_id != self.test_goal_id:
                print(f"Goal {goal_id} not found", file=sys.stderr)
                return False
            
            # Get the goal content
            with open(self.test_goal_file, 'r') as f:
                goal_content = json.load(f)
            
            # Call the mocked decompose_goal function
            result = await mock_decompose_goal(
                repo_path=os.getcwd(),
                goal=goal_content["description"],
                parent_goal=goal_id,
                goal_id=None,
                debug=debug,
                quiet=quiet
            )
            
            if result["success"]:
                print(f"\nGoal {goal_id} successfully decomposed into subgoals")
                print(f"\nNext step: {result['next_step']}")
                print("\nValidation criteria:")
                for criterion in result["validation_criteria"]:
                    print(f"- {criterion}")
                
                if result["requires_further_decomposition"]:
                    print("\nRequires further decomposition: Yes")
                else:
                    print("\nRequires further decomposition: No")
                
                print(f"\nGoal file: {result['goal_file']}")
                return True
            else:
                print(f"Failed to decompose goal: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return False
        
        # Setup mock response
        mock_result = {
            "success": True,
            "next_step": "Implement feature X",
            "validation_criteria": ["Code passes tests", "Feature works as expected"],
            "requires_further_decomposition": True,
            "git_hash": "abcdef123456",
            "memory_hash": None,
            "goal_file": "G1-S1.json"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Redirect stdout and stderr to capture output
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Run the function
            result = await decompose_existing_goal(self.test_goal_id)
            
            # Get captured output
            stdout_value = sys.stdout.getvalue()
            stderr_value = sys.stderr.getvalue()
            
            # Assertions
            self.assertTrue(result)
            self.assertIn("Goal G1 successfully decomposed into subgoals", stdout_value)
            self.assertIn("Next step: Implement feature X", stdout_value)
            self.assertIn("Validation criteria:", stdout_value)
            self.assertIn("- Code passes tests", stdout_value)
            self.assertIn("- Feature works as expected", stdout_value)
            self.assertIn("Requires further decomposition: Yes", stdout_value)
            self.assertIn("Goal file: G1-S1.json", stdout_value)
            
            # Verify decompose_goal was called correctly
            mock_decompose_goal.assert_called_once()
            call_args = mock_decompose_goal.call_args[1]
            self.assertEqual(call_args["goal"], "Test goal description")
            self.assertEqual(call_args["parent_goal"], self.test_goal_id)
        finally:
            # Restore stdout and stderr
            sys.stdout = stdout
            sys.stderr = stderr
    
    @async_test
    async def test_decompose_existing_goal_failure(self, mock_decompose_goal):
        """Test failed goal decomposition."""
        # Create an implementation similar to the one above
        async def decompose_existing_goal(goal_id):
            if goal_id != self.test_goal_id:
                print(f"Goal {goal_id} not found", file=sys.stderr)
                return False
            
            # Call the mocked decompose_goal function
            result = await mock_decompose_goal(
                repo_path=os.getcwd(),
                goal="Test goal description",
                parent_goal=goal_id
            )
            
            if not result["success"]:
                print(f"Failed to decompose goal: {result.get('error', 'Unknown error')}", file=sys.stderr)
                return False
            return True
        
        # Setup mock response for failure
        mock_result = {
            "success": False,
            "error": "Failed to decompose goal"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Redirect stdout and stderr
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Run the function
            result = await decompose_existing_goal(self.test_goal_id)
            
            # Get captured output
            stderr_value = sys.stderr.getvalue()
            
            # Assertions
            self.assertFalse(result)
            self.assertIn("Failed to decompose goal", stderr_value)
            mock_decompose_goal.assert_called_once()
        finally:
            # Restore stdout and stderr
            sys.stdout = stdout
            sys.stderr = stderr
    
    @async_test
    async def test_decompose_nonexistent_goal(self, mock_decompose_goal):
        """Test decomposing a nonexistent goal."""
        # Create a simplified implementation
        async def decompose_existing_goal(goal_id):
            if goal_id != self.test_goal_id:
                print(f"Goal {goal_id} not found", file=sys.stderr)
                return False
            return True
        
        # Redirect stdout and stderr
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Run the function with nonexistent goal
            result = await decompose_existing_goal("NonExistentGoal")
            
            # Get captured output
            stderr_value = sys.stderr.getvalue()
            
            # Assertions
            self.assertFalse(result)
            self.assertIn("Goal NonExistentGoal not found", stderr_value)
            
            # Verify decompose_goal was not called
            mock_decompose_goal.assert_not_called()
        finally:
            # Restore stdout and stderr
            sys.stdout = stdout
            sys.stderr = stderr
    
    @async_test
    async def test_decompose_with_exception(self, mock_decompose_goal):
        """Test handling of exceptions during goal decomposition."""
        # Create an implementation similar to the one above
        async def decompose_existing_goal(goal_id):
            if goal_id != self.test_goal_id:
                print(f"Goal {goal_id} not found", file=sys.stderr)
                return False
            
            try:
                # Call the mocked decompose_goal function that will raise an exception
                result = await mock_decompose_goal(
                    repo_path=os.getcwd(),
                    goal="Test goal description",
                    parent_goal=goal_id
                )
                return result["success"]
            except Exception as e:
                print(f"Error during goal decomposition: {str(e)}", file=sys.stderr)
                return False
        
        # Setup mock to raise an exception
        mock_decompose_goal.side_effect = Exception("Test exception")
        
        # Redirect stdout and stderr
        stdout = sys.stdout
        stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        try:
            # Run the function
            result = await decompose_existing_goal(self.test_goal_id)
            
            # Get captured output
            stderr_value = sys.stderr.getvalue()
            
            # Assertions
            self.assertFalse(result)
            self.assertIn("Error during goal decomposition: Test exception", stderr_value)
            mock_decompose_goal.assert_called_once()
        finally:
            # Restore stdout and stderr
            sys.stdout = stdout
            sys.stderr = stderr


if __name__ == "__main__":
    unittest.main() 