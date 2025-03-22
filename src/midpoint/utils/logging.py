import os
import time
import uuid
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LogSession:
    """Represents a logging session with unique ID and metadata."""
    session_id: str
    start_time: float
    repository_path: str
    initial_git_hash: str
    goal_description: str

class LogManager:
    """Manages logging sessions and provides consistent log formatting."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize the log manager with a directory for log files."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.current_session: Optional[LogSession] = None
    
    def start_session(self, repository_path: str, git_hash: str, goal_description: str) -> LogSession:
        """Start a new logging session."""
        session_id = str(uuid.uuid4())[:8]  # Use first 8 chars of UUID for readability
        self.current_session = LogSession(
            session_id=session_id,
            start_time=time.time(),
            repository_path=repository_path,
            initial_git_hash=git_hash,
            goal_description=goal_description
        )
        return self.current_session
    
    def get_session_log_path(self, log_type: str) -> str:
        """Get the path for a session-specific log file."""
        if not self.current_session:
            raise RuntimeError("No active logging session")
        return os.path.join(self.log_dir, f"{log_type}_{self.current_session.session_id}.log")
    
    def write_log_header(self, log_type: str, additional_info: dict = None):
        """Write the header information to a log file."""
        if not self.current_session:
            raise RuntimeError("No active logging session")
            
        log_path = self.get_session_log_path(log_type)
        with open(log_path, "w") as f:
            f.write(f"Session ID: {self.current_session.session_id}\n")
            f.write(f"Start Time: {datetime.fromtimestamp(self.current_session.start_time)}\n")
            f.write(f"Repository: {self.current_session.repository_path}\n")
            f.write(f"Initial Git Hash: {self.current_session.initial_git_hash}\n")
            f.write(f"Goal: {self.current_session.goal_description}\n")
            if additional_info:
                for key, value in additional_info.items():
                    f.write(f"{key}: {value}\n")
            f.write("\n")
    
    def log_goal_decomposition(self, depth: int, parent_goal: str, subgoal: str, 
                             branch_name: Optional[str] = None, git_hash: Optional[str] = None):
        """Log a goal decomposition step with branch and git information."""
        if not self.current_session:
            raise RuntimeError("No active logging session")
            
        log_path = self.get_session_log_path("goal_hierarchy")
        indent = "  " * depth
        with open(log_path, "a") as f:
            f.write(f"{indent}Goal: {parent_goal}\n")
            f.write(f"{indent}└── Subgoal: {subgoal}\n")
            if branch_name:
                f.write(f"{indent}    Branch: {branch_name}\n")
            if git_hash:
                f.write(f"{indent}    Git Hash: {git_hash}\n")
            f.flush()
    
    def log_execution_ready(self, depth: int, task: str, 
                          branch_name: Optional[str] = None, git_hash: Optional[str] = None):
        """Log when a task is ready for execution."""
        if not self.current_session:
            raise RuntimeError("No active logging session")
            
        log_path = self.get_session_log_path("goal_hierarchy")
        indent = "  " * depth
        with open(log_path, "a") as f:
            f.write(f"{indent}✓ READY FOR EXECUTION: {task}\n")
            if branch_name:
                f.write(f"{indent}    Branch: {branch_name}\n")
            if git_hash:
                f.write(f"{indent}    Git Hash: {git_hash}\n")
            f.flush()
    
    def log_execution_result(self, iteration: int, subgoal: str, git_hash: str, 
                           branch_name: str, validation_score: float, execution_time: float):
        """Log an execution result."""
        if not self.current_session:
            raise RuntimeError("No active logging session")
            
        log_path = self.get_session_log_path("execution")
        with open(log_path, "a") as f:
            f.write(f"\nExecution {iteration}:\n")
            f.write(f"Subgoal: {subgoal}\n")
            f.write(f"Branch: {branch_name}\n")
            f.write(f"Git Hash: {git_hash}\n")
            f.write(f"Validation Score: {validation_score:.2f}\n")
            f.write(f"Execution Time: {execution_time:.2f}s\n")
            f.write(f"Timestamp: {datetime.now()}\n")
            f.flush()

# Create a global log manager instance
log_manager = LogManager() 