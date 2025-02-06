from typing import List, Dict, Any
from pathlib import Path

class DocumentLoader:
    """
    A class to load documents from various sources (e.g., files, URLs) into the knowledge base.
    """

    def __init__(self):
        """
        Initialize the DocumentLoader.
        """
        pass

    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load documents from a file.

        Args:
            file_path (str): The path to the file.

        Returns:
            List[Dict[str, Any]]: A list of documents, where each document is a dictionary.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        # Example: Load a JSON file
        if file_path.suffix == ".json":
            import json
            with open(file_path, "r") as f:
                return json.load(f)
        # Example: Load a text file
        elif file_path.suffix == ".txt":
            with open(file_path, "r") as f:
                return [{"text": f.read()}]
        else:
            raise ValueError(f"Unsupported file type: {file_path.suffix}")

    def load_from_url(self, url: str) -> List[Dict[str, Any]]:
        """
        Load documents from a URL.

        Args:
            url (str): The URL to load documents from.

        Returns:
            List[Dict[str, Any]]: A list of documents, where each document is a dictionary.
        """
        import requests
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"Failed to fetch data from URL: {url}")

        # Example: Load JSON data from a URL
        if "application/json" in response.headers.get("Content-Type", ""):
            return response.json()
        # Example: Load text data from a URL
        else:
            return [{"text": response.text}]