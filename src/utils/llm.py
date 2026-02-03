"""LiteLLM wrapper for flexible model provider support."""

import os
from typing import List, Dict, Any, Optional
from litellm import completion
from dotenv import load_dotenv

load_dotenv()

class LLMWrapper:
    """Wrapper for LiteLLM to support multiple providers."""
    
    def __init__(
        self,
        model_provider: Optional[str] = None,
        model_name: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.model_provider = model_provider or os.getenv("MODEL_PROVIDER", "ollama")
        self.model_name = model_name or os.getenv("MODEL_NAME", "llama3.2:3b")
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Set up the full model identifier for LiteLLM
        if self.model_provider == "ollama":
            self.model = f"ollama/{self.model_name}"
        elif self.model_provider == "openai":
            self.model = self.model_name  # e.g., "gpt-4"
        elif self.model_provider == "anthropic":
            self.model = self.model_name  # e.g., "claude-3-sonnet-20240229"
        else:
            self.model = f"{self.model_provider}/{self.model_name}"
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a completion using LiteLLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: Optional list of tool definitions
            tool_choice: Optional tool choice directive
            
        Returns:
            Response dict from LiteLLM
        """
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            
            # Add tools if provided
            if tools:
                kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
            
            response = completion(**kwargs)
            return response
            
        except Exception as e:
            print(f"Error in LLM generation: {e}")
            raise
    
    def generate_text(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate text completion (no tool calling).
        
        Args:
            messages: List of message dicts
            
        Returns:
            Generated text string
        """
        response = self.generate(messages)
        return response.choices[0].message.content
    
    def format_tools_for_litellm(self, tool_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Format tool definitions for LiteLLM (OpenAI function calling format).
        
        Args:
            tool_definitions: List of tool dicts with name, description, function
            
        Returns:
            Formatted tools for LiteLLM
        """
        formatted_tools = []
        
        for tool in tool_definitions:
            formatted_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The query or identifier to search/lookup"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
            formatted_tools.append(formatted_tool)
        
        return formatted_tools
