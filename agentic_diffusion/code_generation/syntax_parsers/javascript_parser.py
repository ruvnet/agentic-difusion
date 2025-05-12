"""
JavaScriptParser: Language-specific syntax parser for JavaScript code.

This module provides syntax validation capabilities for JavaScript code.
"""

import re
import os
import tempfile
import subprocess
from typing import List

from agentic_diffusion.code_generation.syntax_parsers.base_parser import BaseParser


class JavaScriptParser(BaseParser):
    """
    JavaScript-specific syntax parser.
    
    Provides basic validation for JavaScript syntax through rule-based checks
    and external validation using Node.js when available.
    """
    
    def validate(self, code: str) -> bool:
        """
        Validate JavaScript code syntax.
        
        Args:
            code: JavaScript code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        # Check for basic syntax errors
        try:
            # Look for unmatched brackets, parentheses, etc.
            brackets = {'{': '}', '[': ']', '(': ')'}
            stack = []
            for char in code:
                if char in brackets.keys():
                    stack.append(char)
                elif char in brackets.values():
                    if not stack or brackets[stack.pop()] != char:
                        return False
            
            # Check for invalid JS syntax patterns
            invalid_patterns = [
                r'for\s*\(\s*.*\s*in\s*.*\s*\)\s*\{',  # Missing var/let/const in for-in loop
                r'return\s+\n',  # Return with newline
                r'const\s+[a-zA-Z_$][a-zA-Z0-9_$]*\s*;'  # const without initialization
            ]
            
            for pattern in invalid_patterns:
                if re.search(pattern, code):
                    return False
            
            return len(stack) == 0
        except Exception:
            return False
    
    def get_errors(self, code: str) -> List[str]:
        """
        Get JavaScript syntax error messages.
        
        Args:
            code: JavaScript code to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Try to run Node.js if available
        try:
            with tempfile.NamedTemporaryFile(suffix='.js', delete=False) as temp:
                temp.write(code.encode())
                temp.flush()
                
                result = subprocess.run(
                    ['node', '--check', temp.name],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            errors.append(line)
                
                os.unlink(temp.name)
        except (subprocess.SubprocessError, FileNotFoundError):
            # Node.js not available, use simple checks
            brackets = {'{': '}', '[': ']', '(': ')'}
            stack = []
            
            for i, char in enumerate(code):
                if char in brackets.keys():
                    stack.append((char, i))
                elif char in brackets.values():
                    if not stack or brackets[stack[-1][0]] != char:
                        errors.append(f"Mismatched bracket at position {i}")
                    else:
                        stack.pop()
            
            for bracket, pos in stack:
                errors.append(f"Unclosed bracket '{bracket}' at position {pos}")
        
        return errors