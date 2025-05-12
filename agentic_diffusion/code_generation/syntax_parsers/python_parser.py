"""
PythonParser: Language-specific syntax parser for Python code.

This module provides syntax validation capabilities for Python code.
"""

import ast
from typing import List

from agentic_diffusion.code_generation.syntax_parsers.base_parser import BaseParser


class PythonParser(BaseParser):
    """
    Python-specific syntax parser.
    
    Uses Python's ast module to check for syntax correctness.
    """
    
    def validate(self, code: str) -> bool:
        """
        Validate Python code syntax using ast.parse().
        
        Args:
            code: Python code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def get_errors(self, code: str) -> List[str]:
        """
        Get Python syntax error messages.
        
        Args:
            code: Python code to validate
            
        Returns:
            List of error messages
        """
        errors = []
        try:
            ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Line {e.lineno}: {e.msg}")
        except Exception as e:
            errors.append(str(e))
        
        return errors