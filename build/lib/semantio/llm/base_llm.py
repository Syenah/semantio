from abc import ABC, abstractmethod
from typing import List, Dict, Optional

class BaseLLM(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: Optional[List[Dict]] = None,
        memory: Optional[List[Dict]] = None,
    ) -> str:
        pass

    @property
    def supports_vision(self) -> bool:
        """Return True if the LLM supports vision tasks."""
        return False

    def generate_from_image(self, image_bytes: bytes, **kwargs) -> str:
        """Process an image if vision is supported. Default implementation raises an error."""
        raise NotImplementedError("This LLM does not support vision tasks.")
