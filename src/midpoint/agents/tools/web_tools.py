"""
Web tools for Midpoint agents.

This module provides web tools for searching and scraping web content.
"""

import re
import urllib.parse
from typing import List, Dict, Any, Optional

try:
    import requests
    from bs4 import BeautifulSoup
    HAS_WEB_DEPS = True
except ImportError:
    HAS_WEB_DEPS = False

from midpoint.agents.tools.base import Tool
from midpoint.agents.tools.registry import ToolRegistry

class WebSearchTool(Tool):
    """Tool for searching the web."""
    
    @property
    def name(self) -> str:
        return "web_search"
    
    @property
    def description(self) -> str:
        return "Search the web for information on a query"
    
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
    
    @property
    def required_parameters(self) -> List[str]:
        return ["query"]
    
    def execute(self, query: str, max_results: int = 5) -> str:
        """Execute the web search."""
        if not HAS_WEB_DEPS:
            return "Web search unavailable: Missing required packages (requests, bs4)"
            
        try:
            results = self._duckduckgo_search(query, max_results)
            return results
        except Exception as e:
            return f"Web search for '{query}' failed: {str(e)}"
    
    def _duckduckgo_search(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo."""
        # Quote the query for URL inclusion
        quoted_query = urllib.parse.quote(query)
        
        # Construct the URL
        url = f"https://html.duckduckgo.com/html/?q={quoted_query}"
        
        # Fetch the search results page
        try:
            response = requests.get(
                url, 
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
                }
            )
            response.raise_for_status()
        except Exception as e:
            raise RuntimeError(f"Failed to fetch search results: {str(e)}")
        
        # Parse the HTML
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract search results
        results = []
        result_elements = soup.select(".result")[:max_results]
        
        for element in result_elements:
            title_el = element.select_one(".result__title")
            snippet_el = element.select_one(".result__snippet")
            url_el = element.select_one(".result__url")
            
            if title_el and url_el:
                title = title_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                url = url_el.get_text(strip=True)
                
                results.append({
                    "title": title,
                    "snippet": snippet,
                    "url": url
                })
        
        # Format the results as text
        if not results:
            return f"No results found for '{query}'"
        
        output = f"Search results for '{query}':\n\n"
        
        for i, result in enumerate(results, 1):
            output += f"{i}. {result['title']}\n"
            output += f"   URL: {result['url']}\n"
            if result['snippet']:
                output += f"   {result['snippet']}\n"
            output += "\n"
            
        return output

class WebScrapeTool(Tool):
    """Tool for scraping web content."""
    
    @property
    def name(self) -> str:
        return "web_scrape"
    
    @property
    def description(self) -> str:
        return "Scrape content from a webpage"
    
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
    
    @property
    def required_parameters(self) -> List[str]:
        return ["url"]
    
    def execute(self, url: str) -> str:
        """Scrape content from the given URL."""
        if not HAS_WEB_DEPS:
            return "Web scraping unavailable: Missing required packages (requests, bs4)"
            
        if not url.startswith(('http://', 'https://')):
            raise ValueError("URL must start with http:// or https://")
            
        try:
            # Fetch the URL
            html_content = self._fetch_url(url)
            
            # Parse the HTML
            text_content = self._parse_html(html_content)
            
            # Clean up and truncate if needed
            text_content = text_content.strip()
            if len(text_content) > 8000:
                text_content = text_content[:8000] + "...\n[Content truncated due to length]"
                
            return f"Content from {url}:\n\n{text_content}"
        except Exception as e:
            return f"Error scraping {url}: {str(e)}"
    
    def _fetch_url(self, url: str, params=None) -> str:
        """Fetch content from a URL."""
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            raise RuntimeError(f"Failed to fetch URL: {str(e)}")
    
    def _parse_html(self, html: str) -> str:
        """Parse HTML content to extract text."""
        return self._process_html(html)
    
    def _process_html(self, html: str) -> str:
        """Process HTML content to extract text content."""
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
            
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Remove blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

# Instantiate and register the tools
web_search_tool = WebSearchTool()
web_scrape_tool = WebScrapeTool()

ToolRegistry.register_tool(web_search_tool)
ToolRegistry.register_tool(web_scrape_tool)

# Export tool functions
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web using DuckDuckGo's API."""
    return web_search_tool.execute(query=query, max_results=max_results)

def web_scrape(url: str) -> str:
    """Scrape content from a webpage."""
    return web_scrape_tool.execute(url=url) 