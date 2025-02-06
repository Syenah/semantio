from typing import List, Dict, Optional
from .base_llm import BaseLLM
from openai import OpenAI
import os

class DeepSeekLLM(BaseLLM):
    def __init__(self, model: str = "deepseek-chat", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key is required. Set DEEPSEEK_API_KEY environment variable or pass it explicitly.")
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")

    def generate(self, prompt: str, context: Optional[List[Dict]] = None, memory: Optional[List[Dict]] = None) -> str:
        messages = []
        if memory:
            messages.extend(memory)
        if context:
            messages.append({"role": "system", "content": "Context: " + str(context)})
        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        return response.choices[0].message.content
