import os
from typing import List, Dict, Optional
from .base_llm import BaseLLM
import anthropic

class AnthropicLLM(BaseLLM):
    def __init__(self, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize the Anthropic LLM class.

        Args:
            model (str): The name of the model (e.g., claude-3-5-sonnet-20241022).
            api_key (Optional[str]): The Anthropic API key. If not provided, it fetches from the environment.
        """
        self.model = model
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("Anthropic API key is required. Set ANTHROPIC_API_KEY environment variable or pass it explicitly.")
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def generate(self, prompt: str, context: Optional[List[Dict]] = None, memory: Optional[List[Dict]] = None) -> str:
        """
        Generate text using Anthropic's Claude model.

        Args:
            prompt (str): The user prompt.
            context (Optional[List[Dict]]): Context to include in the conversation.
            memory (Optional[List[Dict]]): Memory from previous interactions.

        Returns:
            str: The generated response from the model.
        """
        try:
            # Prepare messages for the Anthropic API
            messages = []
            if memory:
                messages.extend(memory)
            if context:
                messages.append({"role": "system", "content": "Context: " + str(context)})
            messages.append({"role": "user", "content": prompt})

            # Call the Anthropic API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=messages,
            )

            # Extract and return the response
            return response.content
        except Exception as e:
            raise ValueError(f"Error while generating response with Anthropic Claude: {e}")
