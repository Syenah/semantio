import json
from typing import List, Optional
from ..models import MemoryEntry
from .base_storage import BaseMemoryStorage

class FileStorage(BaseMemoryStorage):
    def __init__(self, file_path: str = "memory.json"):
        self.file_path = file_path
        self.history = self._load_from_file()

    def _load_from_file(self) -> List[MemoryEntry]:
        try:
            with open(self.file_path, "r") as f:
                data = json.load(f)
                return [MemoryEntry(**entry) for entry in data]
        except (FileNotFoundError, json.JSONDecodeError):
            return []

    def _save_to_file(self):
        with open(self.file_path, "w") as f:
            data = [entry.dict() for entry in self.history]
            json.dump(data, f, default=str)

    def store(self, entry: MemoryEntry):
        self.history.append(entry)
        self._save_to_file()

    def retrieve(self, query: Optional[str] = None, limit: int = 20) -> List[MemoryEntry]:
        return self.history[-limit:]