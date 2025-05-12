"""
BaseParser: Abstract base class for language-specific syntax parsers.

This module defines the interface that all language-specific syntax parsers
should implement for consistent syntax validation across different languages.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseParser(ABC):
    """
    Abstract base class for language-specific syntax parsers.
    
    All language parsers should inherit from this class and implement
    the validate method, which determines if code has valid syntax.
    """
    
    @abstractmethod
    def validate(self, code: str) -> bool:
        """
        Validate the syntax of the provided code.
        
        Args:
            code: String containing code to validate
            
        Returns:
            True if the syntax is valid, False otherwise
        """
        pass
    
    def get_errors(self, code: str) -> List[str]:
        """
        Get a list of syntax errors in the code.
        
        Args:
            code: String containing code to validate
            
        Returns:
            List of error messages
        """
        # Default implementation - can be overridden by subclasses
        return [] if self.validate(code) else ["Syntax error"]