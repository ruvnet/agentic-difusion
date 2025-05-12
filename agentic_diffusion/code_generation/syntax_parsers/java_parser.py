"""
JavaParser: Language-specific syntax parser for Java code.

This module provides syntax validation capabilities for Java code.
"""

import re
import os
import tempfile
import subprocess
from typing import List

from agentic_diffusion.code_generation.syntax_parsers.base_parser import BaseParser


class JavaParser(BaseParser):
    """
    Java-specific syntax parser.
    
    Provides basic validation for Java syntax through rule-based checks
    and external validation using javac when available.
    """
    
    def validate(self, code: str) -> bool:
        """
        Validate Java code syntax.
        
        Args:
            code: Java code to validate
            
        Returns:
            True if syntax is valid, False otherwise
        """
        # Simple Java syntax checks
        try:
            # Check for common syntax issues
            must_have_patterns = [
                r'(class|interface|enum)\s+[A-Za-z0-9_]+',  # Must have a class, interface, or enum
                r'\{',  # Must have at least one opening brace
                r'\}'   # Must have at least one closing brace
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
        Get Java syntax error messages.
        
        Args:
            code: Java code to validate
            
        Returns:
            List of error messages
        """
        errors = []
        
        # Try to compile with javac if available
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                # Extract class name from code
                class_match = re.search(r'class\s+([A-Za-z0-9_]+)', code)
                if not class_match:
                    errors.append("No class declaration found")
                    return errors
                
                class_name = class_match.group(1)
                file_path = os.path.join(tmpdir, f"{class_name}.java")
                
                with open(file_path, 'w') as f:
                    f.write(code)
                
                result = subprocess.run(
                    ['javac', file_path],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode != 0:
                    for line in result.stderr.split('\n'):
                        if line.strip():
                            errors.append(line)
        except (subprocess.SubprocessError, FileNotFoundError):
            # javac not available, use simple checks
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
            
            # Check for missing semicolons
            lines = code.split('\n')
            for i, line in enumerate(lines):
                if line.strip() and not line.strip().endswith('{') and not line.strip().endswith('}') and \
                   not line.strip().endswith(';') and not line.strip().endswith('//') and not line.strip().startswith('//'):
                    errors.append(f"Line {i+1}: Missing semicolon")
        
        return errors