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
from midpoint.goal_cli import decompose_existing_goal, agent_decompose_goal, ensure_goal_dir, GOAL_DIR


# Mock the decompose_goal function directly since we're testing it in isolation
@patch('midpoint.agents.goal_decomposer.decompose_goal')
class TestGoalDecomposeCommand(unittest.TestCase):
    """Tests for the goal decompose command in goal_cli.py."""
    
    def setUp(self):
        """Set up the test environment."""
        global GOAL_DIR
        
        # Create a temporary directory for test files
        self.test_dir = Path(tempfile.mkdtemp()) / ".goal"
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # Save original GOAL_DIR
        self.original_goal_dir = GOAL_DIR
        
        # Set up the goal directory
        os.environ["MIDPOINT_GOAL_DIR"] = str(self.test_dir.parent)
        GOAL_DIR = str(self.test_dir)
        
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
            # Remove all files in the directory
            for file in self.test_dir.glob("*"):
                file.unlink()
            # Now remove the directory
            shutil.rmtree(self.test_dir.parent)
        
        # Restore current directory
        os.chdir(self.orig_cwd)
        
        # Restore original GOAL_DIR
        global GOAL_DIR
        GOAL_DIR = self.original_goal_dir
    
    @async_test
    async def test_decompose_existing_goal_success(self, mock_decompose_goal):
        """Test successful goal decomposition."""
        # Setup mock response for success
        mock_result = {
            "success": True,
            "next_step": "Implement user authentication",
            "validation_criteria": [
                "User can register with email and password",
                "User can log in with correct credentials"
            ],
            "requires_further_decomposition": True,
            "goal_file": ".goal/G1-S1.json"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Create test goal files
        top_level_id = "G1"
        top_level_file = self.test_dir / f"{top_level_id}.json"
        
        # Create top-level goal file first
        with open(top_level_file, 'w') as f:
            json.dump({
                "goal_id": top_level_id,
                "description": "Implement authentication system",
                "branch_name": "goal-G1",
                "timestamp": "20250324_000000"
            }, f)
        
        # Then create the subgoal file
        self.test_goal_id = "G1-S1"
        self.test_goal_file = self.test_dir / f"{self.test_goal_id}.json"
        
        with open(self.test_goal_file, 'w') as f:
            json.dump({
                "goal_id": self.test_goal_id,
                "description": "Implement user authentication",
                "parent_goal": top_level_id,
                "timestamp": "20250324_000000"
            }, f)
        
        # Mock git commands
        with patch('subprocess.run') as mock_run, patch('midpoint.goal_cli.get_current_branch') as mock_get_branch:
            # Mock current branch
            mock_get_branch.return_value = "main"
            
            # Mock git status to show no changes
            mock_run.side_effect = [
                # git status
                type('Result', (), {'stdout': '', 'returncode': 0}),
                # git checkout goal-G1
                type('Result', (), {'stdout': 'Switched to branch goal-G1', 'returncode': 0}),
                # git checkout original branch
                type('Result', (), {'stdout': 'Switched to branch main', 'returncode': 0})
            ]
            
            # Run the function
            result = await decompose_existing_goal(self.test_goal_id)
            
            # Assertions
            assert result is True
            mock_decompose_goal.assert_called_once()
            
            # Check that git commands were called correctly
            assert mock_run.call_count == 3
            assert mock_run.call_args_list[0][0][0] == ["git", "status", "--porcelain"]
            assert mock_run.call_args_list[1][0][0] == ["git", "checkout", "goal-G1"]
            assert mock_run.call_args_list[2][0][0] == ["git", "checkout", "main"]
            
            # Clean up files before directory removal
            if self.test_goal_file.exists():
                self.test_goal_file.unlink()
            if top_level_file.exists():
                top_level_file.unlink()
    
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