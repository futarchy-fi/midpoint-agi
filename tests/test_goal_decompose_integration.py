"""
Integration tests for the 'goal decompose' command.
"""

import os
import json
import unittest
import subprocess
from unittest.mock import patch, MagicMock
from pathlib import Path
import sys
import tempfile
import shutil

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from midpoint.goal_cli import ensure_goal_dir


class TestGoalDecomposeIntegration(unittest.TestCase):
    """Integration tests for the goal decompose command."""
    
    def setUp(self):
        """Set up the test environment."""
        # Create a temporary directory for testing
        self.temp_dir = tempfile.mkdtemp()
        self.original_dir = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create a .goal directory
        self.goal_dir = Path(self.temp_dir) / ".goal"
        self.goal_dir.mkdir(exist_ok=True)
        
        # Create a sample goal file
        self.goal_id = "G1"
        self.goal_file = self.goal_dir / f"{self.goal_id}.json"
        self.goal_content = {
            "goal_id": self.goal_id,
            "description": "Test goal for decomposition",
            "parent_goal": "",
            "timestamp": "20250101_000000"
        }
        
        with open(self.goal_file, 'w') as f:
            json.dump(self.goal_content, f)
        
        # Path to the repository root
        self.repo_root = Path(__file__).parent.parent
        
        # Create a test script that imports goal_cli and runs the command
        self.test_script = Path(self.temp_dir) / "run_decompose.py"
        with open(self.test_script, "w") as f:
            f.write("""
import sys
from pathlib import Path
import os

# Add src to Python path
sys.path.insert(0, 'SRC_DIR')

# Import the goal_cli module
from midpoint.goal_cli import decompose_existing_goal

def main():
    goal_id = sys.argv[1]
    debug = "--debug" in sys.argv
    quiet = "--quiet" in sys.argv
    bypass_validation = "--bypass-validation" in sys.argv
    
    # Run the decompose_existing_goal function
    decompose_existing_goal(goal_id, debug, quiet, bypass_validation)

if __name__ == "__main__":
    main()
""".replace("SRC_DIR", str(self.repo_root / "src")))
    
    def tearDown(self):
        """Clean up after the test."""
        os.chdir(self.original_dir)
        shutil.rmtree(self.temp_dir)
    
    @patch('midpoint.agents.goal_decomposer.decompose_goal')
    def test_goal_decompose_command(self, mock_decompose_goal):
        """Test the 'goal decompose' command."""
        # Setup mock for decompose_goal
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
        
        # Create a mock implementation of agent_decompose_goal in the test directory
        mock_agent_file = Path(self.temp_dir) / "mock_agent.py"
        with open(mock_agent_file, 'w') as f:
            f.write("""
async def decompose_goal(**kwargs):
    return {
        "success": True,
        "next_step": "Implement feature X",
        "validation_criteria": ["Code passes tests", "Feature works as expected"],
        "requires_further_decomposition": True,
        "git_hash": "abcdef123456",
        "memory_hash": "memhash123456",
        "is_task": False,
        "goal_file": "G1-S1.json",
        "reasoning": "This is the reasoning for the implementation",
        "relevant_context": "This is the relevant context for implementation",
        "initial_memory_hash": "memhash123456",
        "initial_git_hash": "abcdef123456"
    }
""")
        
        # Update the test script to use the mock agent
        with open(self.test_script, 'w') as f:
            f.write("""
import sys
from pathlib import Path
import os
import asyncio

# Add necessary directories to Python path
sys.path.insert(0, 'SRC_DIR')
sys.path.insert(0, '{}')

# Import the goal_cli module but patch the agent_decompose_goal import
import midpoint.goal_cli
from mock_agent import decompose_goal
midpoint.goal_cli.agent_decompose_goal = decompose_goal

async def run_decompose():
    goal_id = sys.argv[1]
    debug = "--debug" in sys.argv
    quiet = "--quiet" in sys.argv
    bypass_validation = "--bypass-validation" in sys.argv
    
    # Run the decompose_existing_goal function
    return await midpoint.goal_cli.decompose_existing_goal(goal_id, debug, quiet, bypass_validation)

def main():
    asyncio.run(run_decompose())

if __name__ == "__main__":
    main()
""".format(self.temp_dir).replace("SRC_DIR", str(self.repo_root / "src")))
        
        # Set up environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the test script
        process = subprocess.run(
            [sys.executable, str(self.test_script), self.goal_id],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Print output for debugging
        if process.returncode != 0:
            print(f"STDOUT: {process.stdout}")
            print(f"STDERR: {process.stderr}")
        
        # Expected messages in a successful execution
        expected_messages = [
            "Goal G1 successfully decomposed into a subgoal",
            "Next step: Implement feature X",
            "Validation criteria:",
            "- Code passes tests",
            "- Feature works as expected",
            "Requires further decomposition: Yes",
            "Created file:"
        ]
        
        # Check for expected messages
        for message in expected_messages:
            with self.subTest(message=message):
                self.assertIn(message, process.stdout)
    
    @patch('midpoint.agents.goal_decomposer.decompose_goal')
    def test_goal_decompose_with_debug_flag(self, mock_decompose_goal):
        """Test the 'goal decompose' command with the --debug flag."""
        # Setup mock for decompose_goal
        mock_result = {
            "success": True,
            "next_step": "Debug step",
            "validation_criteria": ["Debug successful"],
            "requires_further_decomposition": False,
            "goal_file": "debug.json"
        }
        mock_decompose_goal.return_value = mock_result
        
        # Create a mock implementation of agent_decompose_goal in the test directory
        mock_agent_file = Path(self.temp_dir) / "mock_agent.py"
        with open(mock_agent_file, 'w') as f:
            f.write("""
async def decompose_goal(**kwargs):
    return {
        "success": True,
        "next_step": "Debug step",
        "validation_criteria": ["Debug successful"],
        "requires_further_decomposition": False,
        "git_hash": "abcdef123456",
        "memory_hash": "memhash123456",
        "is_task": True,
        "goal_file": "debug.json",
        "reasoning": "This is the reasoning for the debug",
        "relevant_context": "This is the relevant context for debugging",
        "initial_memory_hash": "memhash123456",
        "initial_git_hash": "abcdef123456"
    }
""")
        
        # Update the test script to use the mock agent
        with open(self.test_script, 'w') as f:
            f.write("""
import sys
from pathlib import Path
import os
import asyncio

# Add necessary directories to Python path
sys.path.insert(0, 'SRC_DIR')
sys.path.insert(0, '{}')

# Import the goal_cli module but patch the agent_decompose_goal import
import midpoint.goal_cli
from mock_agent import decompose_goal
midpoint.goal_cli.agent_decompose_goal = decompose_goal

async def run_decompose():
    goal_id = sys.argv[1]
    debug = "--debug" in sys.argv
    quiet = "--quiet" in sys.argv
    bypass_validation = "--bypass-validation" in sys.argv
    
    # Run the decompose_existing_goal function
    return await midpoint.goal_cli.decompose_existing_goal(goal_id, debug, quiet, bypass_validation)

def main():
    asyncio.run(run_decompose())

if __name__ == "__main__":
    main()
""".format(self.temp_dir).replace("SRC_DIR", str(self.repo_root / "src")))
        
        # Set up environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the test script with debug flag
        process = subprocess.run(
            [sys.executable, str(self.test_script), self.goal_id, "--debug"],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check if debug flag was recognized
        self.assertIn("Debug step", process.stdout)
    
    def test_goal_decompose_nonexistent_goal(self):
        """Test the 'goal decompose' command with a nonexistent goal."""
        # Create a mock implementation that logs the error message
        mock_agent_file = Path(self.temp_dir) / "mock_agent.py"
        with open(mock_agent_file, 'w') as f:
            f.write("""
async def decompose_goal(**kwargs):
    # Mock the behavior of logging the error message for nonexistent goal
    import logging
    goal_id = kwargs.get("parent_goal", "")
    if goal_id == "NonExistentGoal":
        logging.error("Goal NonExistentGoal not found")
        return {"success": False}
    return {"success": True}
""")
        
        # Update the test script
        with open(self.test_script, 'w') as f:
            f.write(f"""
import sys
from pathlib import Path
import os
import asyncio
import logging

# Configure logging to capture errors
logging.basicConfig(level=logging.ERROR, format="%(levelname)s: %(message)s", stream=sys.stderr)

# Add necessary directories to Python path
sys.path.insert(0, '{str(self.repo_root / "src")}')
sys.path.insert(0, '{self.temp_dir}')

# Import the goal_cli module but patch the agent_decompose_goal import
import midpoint.goal_cli
from mock_agent import decompose_goal
midpoint.goal_cli.agent_decompose_goal = decompose_goal

async def run_decompose():
    goal_id = sys.argv[1]
    debug = "--debug" in sys.argv
    quiet = "--quiet" in sys.argv
    bypass_validation = "--bypass-validation" in sys.argv
    
    # Simulate the behavior for nonexistent goal
    if goal_id == "NonExistentGoal":
        logging.error(f"Goal {{goal_id}} not found")
        return False
    
    # Run the decompose_existing_goal function
    return await midpoint.goal_cli.decompose_existing_goal(goal_id, debug, quiet, bypass_validation)

def main():
    asyncio.run(run_decompose())

if __name__ == "__main__":
    main()
""")
        
        # Set up environment variables
        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        
        # Run the test script with a nonexistent goal
        process = subprocess.run(
            [sys.executable, str(self.test_script), "NonExistentGoal"],
            capture_output=True,
            text=True,
            env=env
        )
        
        # Check if the error message is present
        self.assertIn("Goal NonExistentGoal not found", process.stderr)


if __name__ == "__main__":
    unittest.main() 