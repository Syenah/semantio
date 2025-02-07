# hashai/storage/in_memory_storage.py
from typing import List, Optional
from ..models import MemoryEntry
from .base_storage import BaseMemoryStorage

class InMemoryStorage(BaseMemoryStorage):
    def __init__(self):
        self.history: List[MemoryEntry] = []
    
    def store(self, entry: MemoryEntry):
        self.history.append(entry)
    
    def retrieve(self, query: Optional[str] = None, limit: int = 10) -> List[MemoryEntry]:
        return self.history[-limit:]