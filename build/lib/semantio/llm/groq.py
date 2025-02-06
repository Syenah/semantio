import base64
from typing import Optional, List, Dict
from .base_llm import BaseLLM
import groq
import os

class GroqLlm(BaseLLM):
    def __init__(
        self,
        model: str = "mixtral-8x7b-32768",  # Default Groq model
        api_key: Optional[str] = None,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable or pass it explicitly.")
        self.client = groq.Client(api_key=self.api_key)

    @property
    def supports_vision(self) -> bool:
        """
        Check if the model supports vision tasks.
        """
        # List of Groq models that support vision
        vision_models = [
            "llama-3.2-11b-vision-preview",
            "llama-3.2-90b-vision-preview"
        ]
        return self.model in vision_models

    def generate(self, prompt: str, context: Optional[List[Dict]] = None, memory: Optional[List[Dict]] = None) -> str:
        """
        Generate a response to a text-based prompt.
        """
        # Prepare messages for the Groq API
        messages = []
        if memory:
            messages.extend(memory)
        if context:
            messages.append({"role": "system", "content": "Context: " + str(context)})
        messages.append({"role": "user", "content": prompt})

        # Call Groq API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )

        # Extract and return the response
        return response.choices[0].message.content

    def generate_from_image(self, prompt: str, image_bytes: bytes, **kwargs) -> str:
        """
        Process an image and generate a response if the model supports vision.
        """
        if not self.supports_vision:
            raise ValueError(f"Model '{self.model}' does not support vision tasks.")

        try:
            # Convert the image bytes to base64
            image_base64 = base64.b64encode(image_bytes).decode("utf-8")
            
            # Construct the message payload
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_base64}",
                            },
                        },
                    ],
                }
            ]

            # Call the Groq API with the base64-encoded image
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **kwargs,
            )

            # Extract and return the response text
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error while processing image with Groq vision model: {e}")


    def generate_from_image_url(self, prompt: str, image_url: str, **kwargs) -> str:
        """
        Process an image URL and generate a response if the model supports vision.
        """
        if not self.supports_vision:
            raise ValueError(f"Model '{self.model}' does not support vision tasks.")

        try:
            # Call the Groq API with the image URL
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url,
                                },
                            },
                        ],
                    }
                ],
                **kwargs,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise ValueError(f"Error while processing image URL with Groq vision model: {e}")