import os
import json
from pathlib import Path
from typing import Any, Dict, Optional

class FileUtils:
    @staticmethod
    def read_file(file_path: str) -> str:
        """
        Read the contents of a file.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return file.read()
        except Exception as e:
            raise ValueError(f"Failed to read file {file_path}: {e}")

    @staticmethod
    def write_file(file_path: str, content: str) -> None:
        """
        Write content to a file.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(content)
        except Exception as e:
            raise ValueError(f"Failed to write to file {file_path}: {e}")

    @staticmethod
    def read_json(file_path: str) -> Dict[str, Any]:
        """
        Read a JSON file and return its contents as a dictionary.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                return json.load(file)
        except Exception as e:
            raise ValueError(f"Failed to read JSON file {file_path}: {e}")

    @staticmethod
    def write_json(file_path: str, data: Dict[str, Any]) -> None:
        """
        Write a dictionary to a JSON file.
        """
        try:
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, indent=2)
        except Exception as e:
            raise ValueError(f"Failed to write JSON file {file_path}: {e}")

    @staticmethod
    def ensure_directory_exists(directory: str) -> None:
        """
        Ensure that a directory exists. If not, create it.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)