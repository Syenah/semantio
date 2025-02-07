from abc import ABC, abstractmethod
from typing import List, Optional
from ..models import MemoryEntry

class BaseMemoryStorage(ABC):
    @abstractmethod
    def store(self, entry: MemoryEntry):
        pass
    
    @abstractmethod
    def retrieve(self, query: Optional[str] = None, limit: int = 20) -> List[MemoryEntry]:
        pass