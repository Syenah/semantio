from pydantic import BaseModel, Field
from datetime import datetime
from typing import Dict

class MemoryEntry(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict = Field(default_factory=dict)