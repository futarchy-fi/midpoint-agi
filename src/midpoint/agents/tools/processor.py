"""
Processor for handling LLM tool calls.

This module provides a processor for handling LLM tool calls and executing the appropriate tools.
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from openai import AsyncOpenAI
from openai import OpenAI

from midpoint.agents.tools.registry import ToolRegistry

# Add a dedicated logger for LLM responses
llm_logger = logging.getLogger("llm_responses")
llm_logger.setLevel(logging.DEBUG)

# Ensure we have a logs directory
if not os.path.exists("logs"):
    os.makedirs("logs")

# Create a file handler for LLM responses
llm_log_filename = f"logs/llm_responses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
llm_file_handler = logging.FileHandler(llm_log_filename)
llm_file_handler.setLevel(logging.DEBUG)
llm_file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
llm_file_handler.setFormatter(llm_file_formatter)
llm_logger.addHandler(llm_file_handler)

class ToolProcessor:
    """Handles LLM tool calls and executes appropriate tools."""
    
    # Model-specific context windows
    MODEL_CONTEXT_WINDOWS = {
        "gpt-4": 128000,
        "gpt-4-turbo-preview": 128000,
        "gpt-4o-mini": 128000,
        "gpt-3.5-turbo": 16385,
    }
    
    def __init__(self, client: OpenAI):
        self.client = client
        # Rough estimate: 4 characters per token
        self.chars_per_token = 4
        # Reserve some tokens for the response
        self.response_token_buffer = 1000
        # Cache for tool schemas token count
        self._tool_schemas_token_count = None
        
    def estimate_token_count(self, messages: List[Dict[str, Any]]) -> int:
        """Estimate the token count of a list of messages."""
        total = 0
        for msg in messages:
            content = msg.get('content', '')
            if content is None:
                content = ''
            total += len(content) // self.chars_per_token
        return total
    
    def get_tool_schemas_token_count(self) -> int:
        """Get the token count of tool schemas, caching the result."""
        if self._tool_schemas_token_count is None:
            schemas = ToolRegistry.get_tool_schemas()
            self._tool_schemas_token_count = sum(
                len(json.dumps(schema)) // self.chars_per_token 
                for schema in schemas
            )
        return self._tool_schemas_token_count
    
    def get_model_context_window(self, model: str) -> int:
        """Get the context window size for a given model."""
        return self.MODEL_CONTEXT_WINDOWS.get(model, 128000)  # Default to 128k if unknown
    
    def get_available_tokens(self, model: str, max_tokens: int) -> int:
        """Calculate available tokens for conversation history."""
        context_window = self.get_model_context_window(model)
        tool_schemas_tokens = self.get_tool_schemas_token_count()
        # Reserve space for:
        # - Tool schemas
        # - Response buffer
        # - Output tokens
        available = context_window - tool_schemas_tokens - self.response_token_buffer - max_tokens
        return max(0, available)  # Ensure we don't return negative
    
    def truncate_conversation(self, messages: List[Dict[str, Any]], model: str, max_tokens: int) -> List[Dict[str, Any]]:
        """Truncate conversation history to fit within token limit."""
        available_tokens = self.get_available_tokens(model, max_tokens)
        
        # Keep system prompt and user message
        system_msg = next((msg for msg in messages if msg['role'] == 'system'), None)
        user_msg = next((msg for msg in messages if msg['role'] == 'user'), None)
        
        # Keep most recent tool results and assistant messages
        recent_messages = []
        current_tokens = 0
        
        # Add system message if present
        if system_msg:
            content = system_msg.get('content', '')
            if content is None:
                content = ''
            system_tokens = len(content) // self.chars_per_token
            current_tokens += system_tokens
            recent_messages.append(system_msg)
            logging.info(f"System message tokens: {system_tokens}")
        
        # Add user message if present
        if user_msg:
            content = user_msg.get('content', '')
            if content is None:
                content = ''
            user_tokens = len(content) // self.chars_per_token
            current_tokens += user_tokens
            recent_messages.append(user_msg)
            logging.info(f"User message tokens: {user_tokens}")
        
        # Process messages in pairs (assistant + tool responses)
        message_pairs = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            if msg['role'] == 'assistant':
                # Get all tool responses that follow this assistant message
                tool_responses = []
                j = i + 1
                while j < len(messages) and messages[j]['role'] == 'tool':
                    tool_responses.append(messages[j])
                    j += 1
                message_pairs.append((msg, tool_responses))
                i = j
            else:
                i += 1
        
        logging.info(f"Found {len(message_pairs)} message pairs to process")
        
        # Add most recent message pairs until we hit the limit
        for assistant_msg, tool_responses in reversed(message_pairs):
            # Calculate tokens for this pair
            pair_tokens = 0
            
            # Count assistant message tokens
            content = assistant_msg.get('content', '')
            if content is None:
                content = ''
            assistant_tokens = len(content) // self.chars_per_token
            pair_tokens += assistant_tokens
            
            # Count tool response tokens
            tool_tokens = 0
            for tool_msg in tool_responses:
                content = tool_msg.get('content', '')
                if content is None:
                    content = ''
                tool_tokens += len(content) // self.chars_per_token
            
            pair_tokens += tool_tokens
            logging.info(f"Message pair tokens - Assistant: {assistant_tokens}, Tools: {tool_tokens}, Total: {pair_tokens}")
            
            # Check if we can add this pair
            if current_tokens + pair_tokens > available_tokens:
                logging.info(f"Reached token limit. Current: {current_tokens}, Pair would add: {pair_tokens}")
                break
                
            # Add the pair
            current_tokens += pair_tokens
            recent_messages.append(assistant_msg)
            recent_messages.extend(tool_responses)
        
        logging.info(f"Final message count: {len(recent_messages)}")
        logging.info(f"Final token count: {current_tokens}")
        return recent_messages
    
    def process_tool_calls(self, message: Dict[str, Any], 
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
                        try:
                            function_args = json.loads(tool_call["function"]["arguments"])
                        except json.JSONDecodeError as json_err:
                            # Attempt to repair common JSON issues
                            try:
                                # Log the original error position for debugging
                                logging.error(f"JSON parse error at position {json_err.pos}: {json_err}")
                                
                                # Log a snippet of the problematic JSON for context
                                error_pos = json_err.pos
                                start_pos = max(0, error_pos - 20)
                                end_pos = min(len(tool_call["function"]["arguments"]), error_pos + 20)
                                context_str = tool_call["function"]["arguments"][start_pos:end_pos]
                                logging.error(f"JSON context around error: '{context_str}'")
                                
                                # Try to repair common issues
                                fixed_json = tool_call["function"]["arguments"]
                                
                                # If missing colon, try to fix
                                if "Expecting ':' delimiter" in str(json_err):
                                    # Insert a colon at the error position
                                    fixed_json = fixed_json[:error_pos] + ":" + fixed_json[error_pos:]
                                
                                # Attempt to parse the fixed JSON
                                function_args = json.loads(fixed_json)
                                logging.info(f"Successfully repaired malformed JSON for tool: {function_name}")
                            except Exception as repair_err:
                                # If repair fails, use empty dict and log detailed error
                                logging.error(f"Failed to repair JSON for tool {function_name}: {repair_err}")
                                logging.error(f"Original arguments: {tool_call['function']['arguments']}")
                                function_args = {}
                    tool_call_id = tool_call.get("id", "")
                else:
                    if hasattr(tool_call, "function"):
                        function_name = tool_call.function.name
                        try:
                            function_args = json.loads(tool_call.function.arguments)
                        except json.JSONDecodeError as json_err:
                            # Attempt to repair common JSON issues
                            try:
                                # Log the original error position for debugging
                                logging.error(f"JSON parse error at position {json_err.pos}: {json_err}")
                                
                                # Log a snippet of the problematic JSON for context
                                error_pos = json_err.pos
                                start_pos = max(0, error_pos - 20)
                                end_pos = min(len(tool_call.function.arguments), error_pos + 20)
                                context_str = tool_call.function.arguments[start_pos:end_pos]
                                logging.error(f"JSON context around error: '{context_str}'")
                                
                                # Try to repair common issues
                                fixed_json = tool_call.function.arguments
                                
                                # If missing colon, try to fix
                                if "Expecting ':' delimiter" in str(json_err):
                                    # Insert a colon at the error position
                                    fixed_json = fixed_json[:error_pos] + ":" + fixed_json[error_pos:]
                                
                                # Attempt to parse the fixed JSON
                                function_args = json.loads(fixed_json)
                                logging.info(f"Successfully repaired malformed JSON for tool: {function_name}")
                            except Exception as repair_err:
                                # If repair fails, use empty dict and log detailed error
                                logging.error(f"Failed to repair JSON for tool {function_name}: {repair_err}")
                                logging.error(f"Original arguments: {tool_call.function.arguments}")
                                function_args = {}
                    tool_call_id = tool_call.id if hasattr(tool_call, "id") else ""
                
                # Get the tool from registry
                tool = ToolRegistry.get_tool(function_name)
                if not tool:
                    raise ValueError(f"Unknown tool: {function_name}")
                
                # Log the tool execution
                logging.info(f"Executing tool: {function_name}")
                
                # Execute the tool
                result = tool.execute(**function_args)
                
                # Format the result
                if isinstance(result, dict):
                    result_str = json.dumps(result, indent=2)
                else:
                    result_str = str(result)
                
                # === START: Truncate search_code output ===
                MAX_OUTPUT_LENGTH = 10000 # Approx 2500 tokens
                if function_name == "search_code" and len(result_str) > MAX_OUTPUT_LENGTH:
                    truncated_output = result_str[:MAX_OUTPUT_LENGTH] + "\n\n[...truncated...]"
                    logging.warning(f"Truncated {function_name} output from {len(result_str)} to {len(truncated_output)} characters.")
                    result_str = truncated_output
                # === END: Truncate search_code output ===
                
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
    
    def run_llm_with_tools(self, 
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
                # Proactively manage conversation length
                estimated_tokens = self.estimate_token_count(current_messages)
                available_tokens = self.get_available_tokens(model, max_tokens)
                
                if estimated_tokens > available_tokens:
                    logging.info(f"Conversation too long ({estimated_tokens} tokens), truncating...")
                    current_messages = self.truncate_conversation(current_messages, model, max_tokens)
                    logging.info(f"Truncated to {self.estimate_token_count(current_messages)} tokens")
                
                # Call the LLM
                try:
                    response = self.client.chat.completions.create(
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
                
                # Log the full response to the dedicated LLM log file
                if assistant_message.tool_calls:
                    llm_logger.debug(f"[TOOL CALL RESPONSE] Model: {model}")
                    for i, tc in enumerate(assistant_message.tool_calls):
                        tool_name = tc.function.name
                        tool_args = tc.function.arguments
                        llm_logger.debug(f"Tool Call #{i+1}: {tool_name}")
                        llm_logger.debug(f"Arguments: {tool_args}")
                else:
                    llm_logger.debug(f"[TEXT RESPONSE] Model: {model}")
                    llm_logger.debug(f"Content: {assistant_message.content}")
                
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
                        try:
                            tool_args = json.loads(tc.function.arguments)
                        except json.JSONDecodeError as json_err:
                            # Attempt to repair common JSON issues
                            try:
                                # Log the original error position for debugging
                                logging.error(f"JSON parse error at position {json_err.pos}: {json_err}")
                                
                                # Log a snippet of the problematic JSON for context
                                error_pos = json_err.pos
                                start_pos = max(0, error_pos - 20)
                                end_pos = min(len(tc.function.arguments), error_pos + 20)
                                context_str = tc.function.arguments[start_pos:end_pos]
                                logging.error(f"JSON context around error: '{context_str}'")
                                
                                # Try to repair common issues
                                fixed_json = tc.function.arguments
                                
                                # If missing colon, try to fix
                                if "Expecting ':' delimiter" in str(json_err):
                                    # Insert a colon at the error position
                                    fixed_json = fixed_json[:error_pos] + ":" + fixed_json[error_pos:]
                                
                                # Attempt to parse the fixed JSON
                                tool_args = json.loads(fixed_json)
                                logging.info(f"Successfully repaired malformed JSON for tool: {tool_name}")
                            except Exception as repair_err:
                                # If repair fails, use empty dict and log detailed error
                                logging.error(f"Failed to repair JSON for tool {tool_name}: {repair_err}")
                                logging.error(f"Original arguments: {tc.function.arguments}")
                                tool_args = {}
                        
                        tool_usage.append({
                            "tool": tool_name,
                            "args": tool_args
                        })
                    
                    # Process tool calls
                    tool_results = self.process_tool_calls(assistant_message)
                    
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
                
                # Log the exception details to the dedicated LLM log file
                llm_logger.error(f"Error processing LLM response: {str(e)}")
                
                # For JSONDecodeError, log more details about the error
                if isinstance(e, json.JSONDecodeError):
                    # Get the most recent assistant message with tool calls if available
                    recent_assistant_msg = next((msg for msg in reversed(current_messages) 
                                               if msg.get('role') == 'assistant' and 'tool_calls' in msg), None)
                    
                    if recent_assistant_msg and 'tool_calls' in recent_assistant_msg:
                        llm_logger.error(f"JSON error position: {e.pos}")
                        for i, tc in enumerate(recent_assistant_msg['tool_calls']):
                            if 'function' in tc:
                                func_name = tc['function'].get('name', 'unknown')
                                args = tc['function'].get('arguments', '')
                                
                                # Log the problematic JSON with context around the error
                                error_pos = e.pos
                                start_pos = max(0, error_pos - 100)
                                end_pos = min(len(args), error_pos + 100)
                                
                                llm_logger.error(f"Tool #{i+1}: {func_name}")
                                llm_logger.error(f"Error location (position {error_pos}):")
                                llm_logger.error(f"JSON context: {args[start_pos:end_pos]}")
                                
                                # Save the full problematic JSON to the log file
                                llm_logger.error(f"Full JSON arguments:")
                                llm_logger.error(args)
                    
                    # Create a generic error message for the LLM
                    error_message = {
                        "role": "tool",
                        "tool_call_id": "error",
                        "name": "error",
                        "content": f"Error in processing tool call: {str(e)}"
                    }
                    current_messages.append(error_message)
                    continue
                else:
                    # For other types of errors, re-raise
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