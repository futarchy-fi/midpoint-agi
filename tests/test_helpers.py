"""Helper functions for tests."""

import asyncio
import functools
import os
from pathlib import Path
from typing import Any, Callable
from unittest.mock import patch
from midpoint.utils.logging import LogManager

def setup_test_logging(test_dir: str) -> tuple[LogManager, patch]:
    """Set up logging for tests.
    
    Args:
        test_dir: The test directory where logs should be stored.
        
    Returns:
        A tuple containing the LogManager instance and the patcher.
    """
    logs_dir = Path(test_dir) / "logs"
    logs_dir.mkdir(exist_ok=True)
    log_manager = LogManager(str(logs_dir))
    patcher = patch('midpoint.utils.logging.log_manager', new=log_manager)
    return log_manager, patcher

def async_test(coroutine_function: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator for async test functions."""
    @functools.wraps(coroutine_function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Always create a new event loop for each test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(coroutine_function(*args, **kwargs))
        finally:
            loop.close()
    
    return wrapper 