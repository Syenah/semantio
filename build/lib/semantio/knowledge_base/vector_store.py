import numpy as np
import faiss

class VectorStore:
    def __init__(self, dimension: int = 768):
        """
        Initialize a vector store.

        Args:
            dimension (int): Dimensionality of the embeddings.
        """
        self.index = faiss.IndexFlatL2(dimension)

    def add_embeddings(self, embeddings: np.ndarray):
        """
        Add embeddings to the vector store.

        Args:
            embeddings (np.ndarray): Array of embeddings to add.
        """
        self.index.add(embeddings)

    def search(self, query_embedding: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Search for similar embeddings.

        Args:
            query_embedding (np.ndarray): The query embedding.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            np.ndarray: Indices of the nearest neighbors.
        """
        distances, indices = self.index.search(query_embedding.reshape(1, -1), k)
        return indices[0]