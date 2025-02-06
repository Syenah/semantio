import os
from dotenv import load_dotenv
from pathlib import Path
from typing import Any, Optional, Type, TypeVar, List, Dict

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to load and manage environment variables.
    """

    @staticmethod
    def get(key: str, default: Optional[str] = None) -> str:
        """
        Get an environment variable. Raise an error if it's not found and no default is provided.
        """
        value = os.getenv(key, default)
        if value is None:
            raise ValueError(f"Environment variable {key} is not set.")
        return value

    @staticmethod
    def get_bool(key: str, default: bool = False) -> bool:
        """
        Get an environment variable as a boolean.
        """
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes")

    @staticmethod
    def get_int(key: str, default: int = 0) -> int:
        """
        Get an environment variable as an integer.
        """
        return int(os.getenv(key, str(default)))

# Example usage:
# GROQ_API_KEY = Config.get("GROQ_API_KEY")
# MAX_RETRIES = Config.get_int("MAX_RETRIES", 5)