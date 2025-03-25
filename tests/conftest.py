"""
Pytest configuration file for handling warnings and fixtures.
"""

import warnings
import pytest
from urllib3.exceptions import NotOpenSSLWarning
from _pytest.warning_types import PytestDeprecationWarning

# Create a pytest plugin to capture urllib3 warnings
def pytest_runtest_setup():
    """Set up for each test - suppress specific warnings."""
    # Suppress the NotOpenSSLWarning specifically
    warnings.filterwarnings("ignore", category=NotOpenSSLWarning)
    # Also suppress the asyncio warning
    warnings.filterwarnings("ignore", message=".*asyncio_default_fixture_loop_scope.*")

# Add a default function-scoped event loop fixture
@pytest.fixture(scope="function")
def event_loop():
    """Create a function-scoped event loop for asyncio tests."""
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

def pytest_configure(config):
    """Set up pytest configuration."""
    # Suppress the NotOpenSSLWarning
    warnings.filterwarnings(
        "ignore",
        category=Warning,
        message="urllib3 v2 only supports OpenSSL 1.1.1\\+.*"
    )
    
    # Set up default fixture loop scope for asyncio
    # This is explicitly set in pyproject.toml now
    pass
