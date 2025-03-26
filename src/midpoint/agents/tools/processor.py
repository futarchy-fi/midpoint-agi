"""
Processor for handling LLM tool calls.

This module provides a processor for handling LLM tool calls and executing the appropriate tools.
"""

import json
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI

from midpoint.agents.tools.registry import ToolRegistry

class ToolProcessor:
    """Handles LLM tool calls and executes appropriate tools."""
    
    def __init__(self, client: AsyncOpenAI):
        self.client = client
        
    async def process_tool_calls(self, message: Dict[str, Any], 
                                context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Process tool calls from an LLM message and return results."""
        tool_responses = []
        
        # Extract tool_calls from message dict or object
        tool_calls = []
        if isinstance(message, dict) and "tool_calls" in message:
            tool_calls = message["tool_calls"]
        elif hasattr(message, "tool_calls") and message.tool_calls:
            tool_calls = message.tool_calls
        
        if not tool_calls:
            return tool_responses
            
        # Process each tool call
        for tool_call in tool_calls:
            try:
                # Extract function name and args from dict or object
                function_name = ""
                function_args = {}
                tool_call_id = ""
                
                if isinstance(tool_call, dict):
                    if "function" in tool_call:
                        function_name = tool_call["function"]["name"]
                        function_args = json.loads(tool_call["function"]["arguments"])
                    tool_call_id = tool_call.get("id", "")
                else:
                    if hasattr(tool_call, "function"):
                        function_name = tool_call.function.name
                        function_args = json.loads(tool_call.function.arguments)
                    tool_call_id = tool_call.id if hasattr(tool_call, "id") else ""
                
                # Get the tool from registry
                tool = ToolRegistry.get_tool(function_name)
                if not tool:
                    raise ValueError(f"Unknown tool: {function_name}")
                
                # Log the tool execution
                logging.info(f"Executing tool: {function_name}")
                
                # Execute the tool
                result = await tool.execute(**function_args)
                
                # Format the result
                if isinstance(result, dict):
                    result_str = json.dumps(result, indent=2)
                else:
                    result_str = str(result)
                
                # Add to responses
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "name": function_name,
                    "content": result_str
                })
                
            except ValueError as ve:
                # Handle specific errors with more user-friendly messages
                error_message = str(ve)
                logging.error(f"Tool execution error: {error_message}")
                
                # Format the error response
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id if "tool_call_id" in locals() else "",
                    "name": function_name if "function_name" in locals() else "unknown",
                    "content": f"Error: {error_message}"
                })
            except Exception as e:
                # Handle any other exceptions
                error_message = f"Unexpected error during tool execution: {str(e)}"
                logging.error(error_message)
                
                # Format the error response
                tool_responses.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id if "tool_call_id" in locals() else "",
                    "name": function_name if "function_name" in locals() else "unknown",
                    "content": f"Error: {error_message}"
                })
                
        return tool_responses
    
    async def run_llm_with_tools(self, 
                               messages: List[Dict[str, Any]], 
                               model: str = "gpt-4",
                               temperature: float = 0.1,
                               max_tokens: int = 4000,
                               validate_json_format: bool = False) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Run LLM with tools until a final response is generated.
        
        Args:
            messages: Initial messages (system prompt, user prompt, etc.)
            model: The LLM model to use
            temperature: Temperature for LLM generation
            max_tokens: Maximum tokens for LLM generation
            validate_json_format: Whether to validate JSON in responses
            
        Returns:
            Tuple of:
            - message_history: Complete list of all messages (system, user, assistant, tool results)
            - tool_usage: List of tool calls and their results
        """
        current_messages = messages.copy()
        tool_usage = []
        
        # Make sure we have initialized tools
        if not ToolRegistry._initialized:
            from midpoint.agents.tools import initialize_all_tools
            initialize_all_tools()
        
        # Maximum number of iterations to prevent infinite loops
        max_iterations = 10
        current_iteration = 0
        
        while current_iteration < max_iterations:
            current_iteration += 1
            try:
                # Call the LLM
                try:
                    response = await self.client.chat.completions.create(
                        model=model,
                        messages=current_messages,
                        tools=ToolRegistry.get_tool_schemas(),
                        tool_choice="auto",
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                except Exception as e:
                    logging.error(f"Error in LLM API call: {str(e)}")
                    raise
                
                # Get the assistant's message
                assistant_message = response.choices[0].message
                
                # Log the raw content for debugging (truncated if too long)
                raw_content = assistant_message.content or ""
                if len(raw_content) > 500:
                    log_content = raw_content[:500] + "..."
                else:
                    log_content = raw_content
                logging.debug(f"Raw LLM response content: {log_content}")
                
                # Create a serializable copy of the message for our message history
                assistant_message_dict = {
                    "role": "assistant",
                    "content": assistant_message.content
                }
                
                # Add tool calls if present
                if assistant_message.tool_calls:
                    tool_calls_serialized = []
                    for tc in assistant_message.tool_calls:
                        tool_calls_serialized.append({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        })
                    assistant_message_dict["tool_calls"] = tool_calls_serialized
                
                # Add to messages
                current_messages.append(assistant_message_dict)
                
                # Track tool usage
                if assistant_message.tool_calls:
                    for tc in assistant_message.tool_calls:
                        tool_name = tc.function.name
                        tool_args = json.loads(tc.function.arguments)
                        tool_usage.append({
                            "tool": tool_name,
                            "args": tool_args
                        })
                    
                    # Process tool calls
                    tool_results = await self.process_tool_calls(assistant_message)
                    
                    # Add tool results to messages
                    current_messages.extend(tool_results)
                else:
                    # If the model isn't making a tool call, this is our final response
                    # Only validate JSON format if requested
                    if validate_json_format:
                        try:
                            # Extract potential JSON from the response
                            content = self._extract_json_from_response(assistant_message.content)
                            
                            # If extraction was successful, set as final response
                            if content:
                                # Create a custom response with the extracted content
                                class ExtractedResponse:
                                    def __init__(self, original_message, extracted_content):
                                        self.content = json.dumps(extracted_content)
                                        self.tool_calls = original_message.tool_calls
                                
                                final_response = ExtractedResponse(assistant_message, content)
                            else:
                                final_response = assistant_message
                        except json.JSONDecodeError:
                            # If JSON extraction fails, use the original response
                            final_response = assistant_message
                    else:
                        final_response = assistant_message
                    
                    # Break the loop since we have our final response
                    break
            except Exception as e:
                logging.error(f"Error in LLM tool processing: {str(e)}")
                raise
        
        return current_messages, tool_usage
        
    def _extract_json_from_response(self, content: str) -> Optional[str]:
        """
        Extract valid JSON from various response formats.
        
        Handles:
        1. Raw JSON
        2. JSON in code blocks (both with and without language specifiers)
        3. Partially valid JSON that can be repaired
        
        Returns:
            Valid JSON string if successful, None otherwise
        """
        if not content or not isinstance(content, str):
            return None
            
        content = content.strip()
        
        # Try direct parsing first
        try:
            json.loads(content)
            return content  # Already valid JSON
        except json.JSONDecodeError:
            pass
        
        # Try extracting from code blocks
        # Case 1: ```json ... ```
        if "```json" in content:
            try:
                json_blocks = content.split("```json")
                if len(json_blocks) > 1:
                    for block in json_blocks[1:]:
                        if "```" in block:
                            potential_json = block.split("```")[0].strip()
                            try:
                                json.loads(potential_json)
                                return potential_json
                            except json.JSONDecodeError:
                                pass
            except Exception:
                pass
        
        # Case 2: ``` ... ``` (generic code block)
        if "```" in content:
            try:
                blocks = content.split("```")
                for i in range(1, len(blocks), 2):  # Check blocks inside backticks (odd indexes)
                    potential_json = blocks[i].strip()
                    # Skip language identifiers like 'json' at the start
                    lines = potential_json.split('\n', 1)
                    if len(lines) > 1 and not lines[0].strip().startswith('{'):
                        potential_json = lines[1].strip()
                    
                    try:
                        json.loads(potential_json)
                        return potential_json
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass
        
        # Case 3: Look for json-like structures with curly braces
        try:
            start_idx = content.find('{')
            end_idx = content.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                potential_json = content[start_idx:end_idx+1].strip()
                try:
                    json.loads(potential_json)
                    return potential_json
                except json.JSONDecodeError:
                    pass
        except Exception:
            pass
            
        # If all extraction methods failed, log the content for debugging
        logging.debug(f"Failed to extract JSON from response: {content[:200]}...")
        return None 

    async def _handle_tool_invocation(self, data: Dict[str, Any]):
        """Handle tool invocation events from the ToolProcessor."""
        self.conversation_buffer.append({
            "type": "tool_invocation",
            "tool_invocation": data
        })

    async def _handle_intermediate_response(self, data: Dict[str, Any]):
        """Handle intermediate response events from the ToolProcessor."""
        self.conversation_buffer.append({
            "type": "intermediate_response",
            "partial_response": data["content"]
        }) 