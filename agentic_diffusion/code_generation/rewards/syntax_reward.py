"""
SyntaxReward: Computes a syntax correctness score for generated code.

This module provides a sophisticated reward model for evaluating
syntax correctness in generated code across different programming languages.
"""

import ast
import logging
import re
from typing import Dict, Optional, Union, Any
import subprocess
import tempfile
import os

from agentic_diffusion.code_generation.syntax_parsers.base_parser import BaseParser
from agentic_diffusion.code_generation.syntax_parsers.python_parser import PythonParser
from agentic_diffusion.code_generation.syntax_parsers.javascript_parser import JavaScriptParser
from agentic_diffusion.code_generation.syntax_parsers.java_parser import JavaParser
from agentic_diffusion.code_generation.syntax_parsers.go_parser import GoParser

logger = logging.getLogger(__name__)

class SyntaxReward:
    """
    Computes a reward score for syntax correctness.
    
    This reward model evaluates the syntactic correctness of generated code
    using language-specific parsers and validation methods.
    """

    def __init__(self, parsers: Optional[Dict[str, BaseParser]] = None):
        """
        Initialize the syntax reward model.
        
        Args:
            parsers: Optional dictionary of language-specific parsers
        """
        # Initialize default parsers if not provided
        self.parsers = parsers or {
            "python": PythonParser(),
            "javascript": JavaScriptParser(),
            "java": JavaParser(),
            "go": GoParser()
        }
        
        # Fallback validation methods
        self.fallback_validators = {
            "python": self._validate_python,
            "javascript": self._validate_javascript,
            "java": self._validate_java,
            "go": self._validate_go
        }
        
        logger.info("Initialized SyntaxReward with parsers for: %s", 
                   ", ".join(self.parsers.keys()))
    
    def __call__(self, code: str, language: str = "python") -> float:
        """
        Compute syntax correctness score for the given code.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Syntax correctness score between 0.0 and 1.0
        """
        return self.evaluate(code, language)
    
    def evaluate(self, code: str, language: str = "python") -> float:
        """
        Evaluate syntax correctness.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Syntax correctness score between 0.0 and 1.0
        """
        # Normalize language string
        language = language.lower().strip()
        
        # Check if code is empty or whitespace
        if not code or code.isspace():
            return 0.0
        
        try:
            # Try using the appropriate parser
            if language in self.parsers:
                parser = self.parsers[language]
                return 1.0 if parser.validate(code) else 0.0
            
            # Try using fallback validation
            if language in self.fallback_validators:
                validator = self.fallback_validators[language]
                return 1.0 if validator(code) else 0.0
            
            # Default case - assume valid for unknown languages
            logger.warning("No syntax validator available for language: %s", language)
            return 0.5
            
        except Exception as e:
            logger.error("Error evaluating syntax: %s", str(e))
            return 0.0
    
    def get_errors(self, code: str, language: str = "python") -> list:
        """
        Get a list of syntax errors in the code.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            List of syntax error messages
        """
        errors = []
        
        # Normalize language string
        language = language.lower().strip()
        
        # Handle by language
        if language == "python":
            errors = self._get_python_errors(code)
        elif language == "javascript":
            errors = self._get_javascript_errors(code)
        elif language == "java":
            errors = self._get_java_errors(code)
        elif language == "go":
            errors = self._get_go_errors(code)
        
        return errors
    
    def _validate_python(self, code: str) -> bool:
        """
        Validate Python code syntax.
        
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
    
    def _get_python_errors(self, code: str) -> list:
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
    
    def _validate_javascript(self, code: str) -> bool:
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
    
    def _get_javascript_errors(self, code: str) -> list:
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
    
    def _validate_java(self, code: str) -> bool:
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
    
    def _get_java_errors(self, code: str) -> list:
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
    
    def _validate_go(self, code: str) -> bool:
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
    
    def _get_go_errors(self, code: str) -> list:
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