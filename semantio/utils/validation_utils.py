import re
from typing import Any, Optional

class ValidationUtils:
    @staticmethod
    def validate_email(email: str) -> bool:
        """
        Validate an email address.
        """
        regex = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        return re.match(regex, email) is not None

    @staticmethod
    def validate_phone_number(phone: str) -> bool:
        """
        Validate a phone number (basic validation for US numbers).
        """
        regex = r"^\+?1?\d{10}$"
        return re.match(regex, phone) is not None

    @staticmethod
    def validate_not_empty(value: Any) -> bool:
        """
        Validate that a value is not empty.
        """
        if isinstance(value, str):
            return bool(value.strip())
        elif isinstance(value, (list, dict, set)):
            return bool(value)
        return value is not None

    @staticmethod
    def validate_numeric(value: Any) -> bool:
        """
        Validate that a value is numeric (int or float).
        """
        return isinstance(value, (int, float))

    @staticmethod
    def validate_in_range(value: float, min: float, max: float) -> bool:
        """
        Validate that a numeric value is within a specified range.
        """
        return min <= value <= max