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
                               model: str = "gpt-4o",
                               temperature: float = 0.1,
                               max_tokens: int = 4000,
                               validate_json_format: bool = False) -> Tuple[Any, List[Dict[str, Any]]]:
        """Run LLM with tools until a final response is generated."""
        current_messages = messages.copy()
        tool_usage = []
        final_response = None
        
        # Make sure we have initialized tools
        if not ToolRegistry._initialized:
            from midpoint.agents.tools import initialize_all_tools
            initialize_all_tools()
        
        # Maximum number of iterations to prevent infinite loops
        max_iterations = 10
        current_iteration = 0
        
        while final_response is None and current_iteration < max_iterations:
            current_iteration += 1
            try:
                # Call the LLM
                response = await self.client.chat.completions.create(
                    model=model,
                    messages=current_messages,
                    tools=ToolRegistry.get_tool_schemas(),
                    tool_choice="auto",
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Get the assistant's message
                assistant_message = response.choices[0].message
                
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
                            # Try to parse the content as JSON
                            content = assistant_message.content.strip()
                            
                            # Check if the JSON is valid, and if not, ask for valid JSON
                            try:
                                # If JSON is embedded in code blocks, extract it
                                if "```json" in content:
                                    parts = content.split("```json")
                                    if len(parts) > 1:
                                        json_part = parts[1].split("```")[0].strip()
                                        json.loads(json_part)  # Just testing if it's valid
                                elif "```" in content:
                                    parts = content.split("```")
                                    if len(parts) > 1:
                                        json_part = parts[1].strip()
                                        json.loads(json_part)  # Just testing if it's valid
                                else:
                                    # Try parsing directly
                                    json.loads(content)
                                    
                                # If we got here, the JSON is valid, set as final response
                                final_response = assistant_message
                            except json.JSONDecodeError:
                                # This was not a valid JSON, ask for a valid JSON
                                if current_iteration < max_iterations:
                                    logging.warning(f"Invalid JSON response on iteration {current_iteration}, asking for valid JSON")
                                    current_messages.append({
                                        "role": "user",
                                        "content": "Please provide your response in valid JSON format with fields: next_step, validation_criteria, reasoning, requires_further_decomposition, and relevant_context."
                                    })
                                else:
                                    # If we've reached max iterations, just return what we have
                                    logging.warning("Max iterations reached, returning whatever we have")
                                    final_response = assistant_message
                        except Exception as e:
                            logging.error(f"Error processing final response: {str(e)}")
                            final_response = assistant_message
                    else:
                        # Skip JSON validation, just use the response as is
                        final_response = assistant_message
            
            except Exception as e:
                logging.error(f"Error in LLM tool processing: {str(e)}")
                raise
                
        # If we somehow didn't get a final response, return the last message
        if final_response is None and current_messages:
            for msg in reversed(current_messages):
                if msg.get("role") == "assistant":
                    logging.warning("No final response set, using last assistant message")
                    # Create a basic object with content from the last message
                    class LastMessage:
                        def __init__(self, content):
                            self.content = content
                    
                    final_response = LastMessage(msg.get("content", ""))
                    break
        
        return final_response, tool_usage 