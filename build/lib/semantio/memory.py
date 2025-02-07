from .models import MemoryEntry
from .storage import BaseMemoryStorage, InMemoryStorage, FileStorage
from typing import List, Dict, Optional
from .llm.base_llm import BaseLLM
class Memory:
    def __init__(
        self,
        storage: BaseMemoryStorage = InMemoryStorage(),
        max_context_length: int = 4000,
        summarization_threshold: int = 3000
    ):
        self.storage = storage
        self.max_context_length = max_context_length
        self.summarization_threshold = summarization_threshold
        self._current_context = ""

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None):
        entry = MemoryEntry(
            role=role,
            content=content,
            metadata=metadata or {}
        )
        self.storage.store(entry)
        self._manage_context()

    def get_context(self, llm: Optional[BaseLLM] = None) -> str:
        if len(self._current_context) < self.summarization_threshold:
            return self._current_context
        
        # Automatic summarization when context grows too large
        if llm:
            return self.summarize(llm)
        return self._current_context[:self.max_context_length]
    def _manage_context(self):
        # Include roles in the conversation history
        full_history = "\n".join([f"{e.role}: {e.content}" for e in self.storage.retrieve()])
        if len(full_history) > self.max_context_length:
            self._current_context = full_history[-self.max_context_length:]
        else:
            self._current_context = full_history

    def summarize(self, llm: BaseLLM) -> str:
        # Include roles in the history for summarization
        history = "\n".join([f"{e.role}: {e.content}" for e in self.storage.retrieve()])
        prompt = f"""
        Summarize this conversation history maintaining key details and references:
        {history[-self.summarization_threshold:]}
        """
        self._current_context = llm.generate(prompt)
        return self._current_context

    def clear(self):
        self.storage = InMemoryStorage()
        self._current_context = ""