import pytest
import asyncio
from midpoint.agents.tools import web_search, web_scrape
import os

@pytest.mark.asyncio
async def test_web_search():
    """Test the web search functionality."""
    query = "What is the exa browser?"
    result = await web_search(query, max_results=3)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print("\nWeb Search Results:")
    print(result)

@pytest.mark.asyncio
async def test_web_scrape():
    """Test the web scraping functionality."""
    # Using a known stable URL for testing
    url = "https://github.com/ogham/exa"
    result = await web_scrape(url)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    print("\nWeb Scrape Results:")
    print(result[:500] + "...")  # Print first 500 chars for readability

@pytest.mark.asyncio
async def test_tavily_search():
    """Test Tavily search functionality"""
    # Skip test if API key is not set
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")
        
    result = await tavily_search("Python programming language", max_results=3)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "Title:" in result
    assert "Content:" in result
    assert "URL:" in result

@pytest.mark.asyncio
async def test_web_search_with_tavily():
    """Test combined web search with both DuckDuckGo and Tavily"""
    # Skip test if API key is not set
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")
        
    result = await web_search("Python programming language", max_results=3)
    assert isinstance(result, str)
    assert len(result) > 0
    assert "=== DuckDuckGo Results ===" in result
    assert "=== Tavily Results ===" in result

if __name__ == "__main__":
    asyncio.run(test_web_search())
    asyncio.run(test_web_scrape())
    asyncio.run(test_tavily_search())
    asyncio.run(test_web_search_with_tavily()) 