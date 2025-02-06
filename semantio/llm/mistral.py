from typing import List, Dict, Optional
from .base_llm import BaseLLM
from mistralai import Mistral
import os

class MistralLLM(BaseLLM):
    def __init__(self, model: str = "mistral-large-latest", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("Mistral API key is required. Set MISTRAL_API_KEY environment variable or pass it explicitly.")
        self.client = Mistral(api_key=self.api_key)

    def generate(self, prompt: str, context: Optional[List[Dict]] = None, memory: Optional[List[Dict]] = None) -> str:
        messages = []
        if memory:
            messages.extend(memory)
        if context:
            messages.append({"role": "system", "content": "Context: " + str(context)})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.complete(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content
