"""
Web tools for the Midpoint agent system.

This module provides tools for interacting with the web,
including web search and web scraping capabilities.
"""

import re
import json
import logging
import urllib.parse
from typing import List, Dict, Any, Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    requests = None
    BeautifulSoup = None

from .base import Tool
from .registry import ToolRegistry

class WebSearchTool(Tool):
    """Tool for searching the web."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
        
    async def execute(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo's API."""
        try:
            results = await self._duckduckgo_search(query, max_results)
            return results
        except Exception as e:
            logging.error(f"Error in DuckDuckGo search: {str(e)}")
            # Fallback to a simpler search message
            return f"Web search for '{query}' failed: {str(e)}"
    
    async def _duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo."""
        # Quote the query for URL inclusion
        quoted_query = urllib.parse.quote_plus(query)
        
        # DuckDuckGo API endpoint 
        url = f"https://api.duckduckgo.com/?q={quoted_query}&format=json&pretty=1"
        
        # Create SSL context with certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl=ssl_context) as response:
                    if response.status != 200:
                        raise RuntimeError(f"DuckDuckGo API returned status {response.status}")
                        
                    # Parse the response
                    data = await response.json()
                    
                    # Format the results
                    results = []
                    
                    # Handle abstract text
                    if data.get("AbstractText"):
                        results.append({
                            "title": data.get("Heading", "Abstract"),
                            "snippet": data.get("AbstractText"),
                            "url": data.get("AbstractURL")
                        })
                        
                    # Handle related topics
                    for topic in data.get("RelatedTopics", [])[:max_results - len(results)]:
                        if "Topics" in topic:
                            # This is a category
                            for subtopic in topic.get("Topics", [])[:max_results - len(results)]:
                                if subtopic.get("Text") and subtopic.get("FirstURL"):
                                    results.append({
                                        "title": subtopic.get("Text").split(" - ")[0],
                                        "snippet": subtopic.get("Text"),
                                        "url": subtopic.get("FirstURL")
                                    })
                        elif topic.get("Text") and topic.get("FirstURL"):
                            results.append({
                                "title": topic.get("Text").split(" - ")[0],
                                "snippet": topic.get("Text"),
                                "url": topic.get("FirstURL")
                            })
                            
                    # Format the results as a string
                    if not results:
                        return f"No results found for '{query}'"
                        
                    formatted_results = f"Web search results for '{query}':\n\n"
                    for i, result in enumerate(results, 1):
                        formatted_results += f"{i}. {result['title']}\n"
                        formatted_results += f"   {result['snippet']}\n"
                        formatted_results += f"   URL: {result['url']}\n\n"
                        
                    return formatted_results
        except Exception as e:
            logging.error(f"Error in DuckDuckGo search: {str(e)}")
            raise

class WebScrapeTool(Tool):
    """Tool for scraping web pages."""
    
    @property
    def name(self) -> str:
        return "web_scrape"
    
    @property
    def description(self) -> str:
        return "Scrape content from a web page"
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to scrape"
                }
            },
            "required": ["url"]
        }
    
    async def execute(self, url: str) -> str:
        """Scrape content from a webpage."""
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
            
        # Create SSL context with certifi
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        
        try:
            # Make the request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, ssl=ssl_context) as response:
                    if response.status != 200:
                        raise RuntimeError(f"Failed to scrape URL: {url}, status code: {response.status}")
                        
                    # Get the content
                    html = await response.text()
                    
                    # Extract text from HTML using a simple regex-based approach
                    # This is very simplistic - a real implementation would use a proper HTML parser
                    text = self._extract_text_from_html(html)
                    
                    # Limit the length of the response
                    max_length = 5000
                    if len(text) > max_length:
                        text = text[:max_length] + f"\n\n[Content truncated, {len(text) - max_length} more characters...]"
                        
                    return f"Content from {url}:\n\n{text}"
        except Exception as e:
            logging.error(f"Error scraping URL {url}: {str(e)}")
            return f"Failed to scrape URL: {url}\nError: {str(e)}"
    
    def _extract_text_from_html(self, html: str) -> str:
        """Extract readable text from HTML."""
        # Remove script and style elements
        html = re.sub(r'<script[^>]*>.*?</script>', ' ', html, flags=re.DOTALL)
        html = re.sub(r'<style[^>]*>.*?</style>', ' ', html, flags=re.DOTALL)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        
        # Handle entities
        text = re.sub(r'&nbsp;', ' ', text)
        text = re.sub(r'&lt;', '<', text)
        text = re.sub(r'&gt;', '>', text)
        text = re.sub(r'&amp;', '&', text)
        text = re.sub(r'&quot;', '"', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Join paragraphs with newlines
        return '\n\n'.join(paragraphs)

# Instantiate and register the tools
web_search_tool = WebSearchTool()
web_scrape_tool = WebScrapeTool()

ToolRegistry.register_tool(web_search_tool)
ToolRegistry.register_tool(web_scrape_tool)

# Export tool functions
async def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo's API."""
    return await web_search_tool.execute(query=query, max_results=max_results)

async def web_scrape(url: str) -> str:
    """Scrape content from a webpage."""
    return await web_scrape_tool.execute(url=url) 