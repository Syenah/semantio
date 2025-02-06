# base_tool.py
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field

class BaseTool(BaseModel):
    name: str = Field(..., description="The name of the tool.")
    description: str = Field(..., description="A brief description of the tool's functionality.")
    llm: Optional[Any] = Field(None, description="The LLM instance to use for tool execution.")

    def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the tool's functionality."""
        raise NotImplementedError("Subclasses must implement this method.")