"""Helper utilities for testing."""

import asyncio
import unittest
from typing import Any, Callable, Coroutine

def async_test(coroutine_function: Callable[..., Coroutine]) -> Callable:
    """
    Decorator to run async test methods.
    
    Usage:
        class MyTestCase(unittest.TestCase):
            @async_test
            async def test_async_function(self):
                result = await some_async_function()
                self.assertEqual(result, expected_value)
    """
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        # Always create a new event loop for each test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(coroutine_function(*args, **kwargs))
        finally:
            # Clean up properly
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
    
    return wrapper 