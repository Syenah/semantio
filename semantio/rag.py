from typing import List, Dict
from .knowledge_base.retriever import Retriever

class RAG:
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def retrieve(self, query: str) -> List[Dict]:
        """
        Retrieve relevant context for a given query.

        Args:
            query (str): The query string.

        Returns:
            List[Dict]: List of relevant documents.
        """
        return self.retriever.retrieve(query)