"""
GoParser: Language-specific syntax parser for Go code.

This module provides syntax validation capabilities for Go code.
"""

import re
import os
import tempfile
import subprocess
from typing import List

from agentic_diffusion.code_generation.syntax_parsers.base_parser import BaseParser


class GoParser(BaseParser):
    """
    Go-specific syntax parser.
    
    Provides basic validation for Go syntax through rule-based checks
    and external validation using the Go compiler when available.
    """
    
    def validate(self, code: str) -> bool:
        """
        Validate Go code syntax.
        
        Args:
            code: Go code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        # Simple Go syntax checks
        try:
            # Check for basic syntax patterns
            must_have_patterns = [
                r'package\s+[a-zA-Z0-9_]+',  # Must have a package declaration
                r'func\s+',   # Must have at least one function
            ]
            
            for pattern in must_have_patterns:
                if not re.search(pattern, code):
                    return False
            
            # Check for balanced braces
            brackets = {'{': '}', '[': ']', '(': ')'}
            stack = []
            for char in code:
                if char in brackets.keys():
                    stack.append(char)
                elif char in brackets.values():
                    if not stack or brackets[stack.pop()] != char:
                        return False
            
            return len(stack) == 0
        except Exception:
            return False
    
    def get_errors(self, code: str) -> List[str]:
        """
        Get Go syntax error messages.
        
        Args:
            code: Go code to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Try to compile with go if available
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                file_path = os.path.join(tmpdir, "main.go")
                
                with open(file_path, 'w') as f:
                    f.write(code)
                
                result = subprocess.run(
                    ['go', 'build', file_path],
                    capture_output=True,
                    text=True,
                    cwd=tmpdir
                )
                
                if result.returncode != 0:
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            errors.append(line)
        except (subprocess.SubprocessError, FileNotFoundError):
            # Go not available, use simple checks
            # Check for package declaration
            if not re.search(r'package\s+[a-zA-Z0-9_]+', code):
                errors.append("Missing package declaration")
            
            # Check for unbalanced braces
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