import os
from typing import List, Dict, Optional
from .base_llm import BaseLLM
from google import genai

class GeminiLLM(BaseLLM):
    def __init__(self, model: str = "gemini-1.5-flash", api_key: Optional[str] = None):
        """
        Initialize the Gemini LLM class.

        Args:
            model (str): The name of the Gemini model (e.g., 'gemini-1.5-flash').
            api_key (Optional[str]): The Gemini API key. If not provided, it fetches from the environment.
        """
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key is required. Set GEMINI_API_KEY environment variable or pass it explicitly.")
        
        # Initialize the client using the API key
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str, context: Optional[List[Dict]] = None, memory: Optional[List[Dict]] = None) -> str:
        """
        Generate text using Google's Gemini model.

        Args:
            prompt (str): The user prompt.
            context (Optional[List[Dict]]): Context to include in the conversation.
            memory (Optional[List[Dict]]): Memory from previous interactions.

        Returns:
            str: The generated response from the model.
        """
        try:
            # Prepare the chat history (optional context and memory)
            history = memory if memory else []
            if context:
                history.append({"role": "system", "content": str(context)})

            # Generate the content using the specified Gemini model
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            # Return the response text
            return response.text
        except Exception as e:
            raise ValueError(f"Error while generating response with Gemini: {e}")
