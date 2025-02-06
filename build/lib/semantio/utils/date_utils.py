from datetime import datetime, timedelta
from typing import Optional

class DateUtils:
    @staticmethod
    def get_current_time() -> str:
        """
        Get the current time in ISO format.
        """
        return datetime.now().isoformat()

    @staticmethod
    def format_time(timestamp: str, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """
        Format a timestamp string into a custom format.
        """
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt.strftime(format)
        except Exception as e:
            raise ValueError(f"Failed to format timestamp {timestamp}: {e}")

    @staticmethod
    def add_time(timestamp: str, days: int = 0, hours: int = 0, minutes: int = 0) -> str:
        """
        Add days, hours, or minutes to a timestamp.
        """
        try:
            dt = datetime.fromisoformat(timestamp)
            dt += timedelta(days=days, hours=hours, minutes=minutes)
            return dt.isoformat()
        except Exception as e:
            raise ValueError(f"Failed to add time to timestamp {timestamp}: {e}")

    @staticmethod
    def is_future_time(timestamp: str) -> bool:
        """
        Check if a timestamp is in the future.
        """
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt > datetime.now()
        except Exception as e:
            raise ValueError(f"Failed to check if timestamp {timestamp} is in the future: {e}")