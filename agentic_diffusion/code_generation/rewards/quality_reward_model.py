"""
QualityRewardModel: Advanced model for computing code quality scores with detailed metrics.
"""

import re
from typing import Dict, Any

from agentic_diffusion.code_generation.syntax_model import SyntaxModel

class QualityRewardModel:
    """
    Advanced model for computing a reward score for code quality.
    This model provides detailed metrics beyond basic syntax checking.
    """

    def __init__(self):
        self.syntax_model = SyntaxModel()
        
    def evaluate(self, code: str, language: str = "python") -> float:
        """
        Evaluate code quality with multiple metrics.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # Empty or minimal code gets a low score
        if not code or len(code.strip()) < 10:
            return 0.1
        
        # Get detailed metrics
        metrics = self.get_detailed_metrics(code, language)
        
        # Calculate weighted score based on metrics
        weights = {
            "syntax": 0.3,
            "complexity": 0.2,
            "documentation": 0.15,
            "readability": 0.15,
            "standards": 0.2
        }
        
        weighted_score = sum(metrics[key] * weights[key] for key in weights)
        return min(1.0, max(0.0, weighted_score))  # Ensure it's between 0 and 1
    
    def get_detailed_metrics(self, code: str, language: str) -> Dict[str, float]:
        """
        Get detailed quality metrics for code.
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Dictionary of metrics with values between 0.0 and 1.0
        """
        metrics = {}
        
        # Syntax correctness
        metrics["syntax"] = 1.0 if self.syntax_model.validate(code, language) else 0.4
        
        # Complexity (lower is better, higher score)
        metrics["complexity"] = self._evaluate_complexity(code, language)
        
        # Documentation quality
        metrics["documentation"] = self._evaluate_documentation(code, language)
        
        # Readability
        metrics["readability"] = self._evaluate_readability(code, language)
        
        # Adherence to language standards
        metrics["standards"] = self._evaluate_standards(code, language)
        
        return metrics
    
    def _evaluate_complexity(self, code: str, language: str) -> float:
        """
        Evaluate code complexity (lower complexity = higher score).
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Float score between 0.0 and 1.0
        """
        lines = code.split('\n')
        
        # Length factors
        line_count = len(lines)
        avg_line_length = sum(len(line) for line in lines) / max(1, line_count)
        
        # Nesting level detection
        max_indent = 0
        for line in lines:
            if line.strip():  # Skip empty lines
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        
        # Long lines penalty
        long_lines = sum(1 for line in lines if len(line.strip()) > 100)
        long_lines_ratio = long_lines / max(1, line_count)
        
        # Complexity factors (higher = more complex = lower score)
        length_factor = min(1.0, line_count / 300)  # Normalize by 300 lines
        indent_factor = min(1.0, max_indent / 40)   # Normalize by 40 spaces
        line_length_factor = min(1.0, avg_line_length / 80)  # Normalize by 80 chars
        
        # Calculate complexity score (higher is better = less complex)
        complexity_score = 1.0 - (0.4 * length_factor + 
                                 0.3 * indent_factor + 
                                 0.2 * line_length_factor + 
                                 0.1 * long_lines_ratio)
        
        return max(0.0, min(1.0, complexity_score))
    
    def _evaluate_documentation(self, code: str, language: str) -> float:
        """
        Evaluate documentation quality.
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Float score between 0.0 and 1.0
        """
        lines = code.split('\n')
        total_lines = len([l for l in lines if l.strip()])
        if total_lines == 0:
            return 0.0
        
        # Count comment/docstring lines based on language
        comment_count = 0
        
        if language.lower() == "python":
            # Python comments and docstrings
            in_docstring = False
            for line in lines:
                line = line.strip()
                if '"""' in line or "'''" in line:
                    in_docstring = not in_docstring
                    comment_count += 1
                elif in_docstring or line.startswith('#'):
                    comment_count += 1
                    
        elif language.lower() in ["javascript", "typescript", "java", "c", "cpp"]:
            # C-style comments
            in_block_comment = False
            for line in lines:
                line = line.strip()
                if line.startswith('/*') and '*/' in line:
                    comment_count += 1
                elif line.startswith('/*'):
                    in_block_comment = True
                    comment_count += 1
                elif '*/' in line and in_block_comment:
                    in_block_comment = False
                    comment_count += 1
                elif in_block_comment or line.startswith('//') or line.startswith('*'):
                    comment_count += 1
        
        # Calculate documentation ratio
        doc_ratio = comment_count / total_lines
        
        # Ideal ratio is around 0.2-0.4 (too many comments can also be bad)
        if doc_ratio > 0.5:
            # Penalty for too many comments
            doc_score = 1.0 - ((doc_ratio - 0.5) * 0.5)
        else:
            # Reward for good comment ratio
            doc_score = min(1.0, doc_ratio * 2.5)
            
        return max(0.0, min(1.0, doc_score))
    
    def _evaluate_readability(self, code: str, language: str) -> float:
        """
        Evaluate code readability.
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Float score between 0.0 and 1.0
        """
        lines = code.split('\n')
        
        # Check for consistent indentation
        indent_sizes = set()
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip(' '))
                if leading_spaces > 0:
                    indent_sizes.add(leading_spaces)
        
        # Consistent indentation gets higher score
        if len(indent_sizes) <= 2:  # Allow top-level and one indent level
            indent_score = 1.0
        else:
            indent_score = 1.0 - (min(1.0, (len(indent_sizes) - 2) * 0.25))
            
        # Check for meaningful variable names (longer than 1-2 chars)
        if language.lower() == "python":
            # Look for variable assignments
            var_pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*='
            var_names = re.findall(var_pattern, code)
            
            # Count variables with good names (length > 2)
            good_var_count = sum(1 for name in var_names if len(name) > 2)
            var_score = good_var_count / max(1, len(var_names)) if var_names else 0.5
            
        elif language.lower() in ["javascript", "typescript", "java"]:
            # Look for variable declarations
            var_pattern = r'\b(?:var|let|const|int|float|double|String)\s+([a-zA-Z_][a-zA-Z0-9_]*)'
            var_names = re.findall(var_pattern, code)
            
            # Count variables with good names (length > 2)
            good_var_count = sum(1 for name in var_names if len(name) > 2)
            var_score = good_var_count / max(1, len(var_names)) if var_names else 0.5
            
        else:
            # Default for other languages
            var_score = 0.7  # Default reasonable score
            
        # Calculate overall readability
        readability_score = 0.6 * indent_score + 0.4 * var_score
        
        return max(0.0, min(1.0, readability_score))
    
    def _evaluate_standards(self, code: str, language: str) -> float:
        """
        Evaluate adherence to language standards and best practices.
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Float score between 0.0 and 1.0
        """
        # Language-specific standards checking
        if language.lower() == "python":
            return self._check_python_standards(code)
        elif language.lower() == "javascript":
            return self._check_javascript_standards(code)
        elif language.lower() == "java":
            return self._check_java_standards(code)
        else:
            # Default score for languages without specific checks
            return 0.8
    
    def _check_python_standards(self, code: str) -> float:
        """Check Python code standards (PEP 8 approximation)."""
        issues = 0
        
        # Check for common PEP 8 violations
        lines = code.split('\n')
        for line in lines:
            # Line too long
            if len(line) > 100:
                issues += 1
                
            # Mixed tabs and spaces
            if '\t' in line and ' ' in line:
                issues += 2
                
            # Missing whitespace around operators
            if re.search(r'[a-zA-Z0-9](==|!=|<=|>=|<|>|\+|-|\*|\/)[a-zA-Z0-9]', line):
                issues += 1
        
        # Normalize issues score (0 issues = 1.0, 10+ issues = 0.0)
        standards_score = max(0.0, 1.0 - (issues / 10))
        
        return standards_score
    
    def _check_javascript_standards(self, code: str) -> float:
        """Check JavaScript code standards."""
        issues = 0
        
        # Check for common JS standards violations
        lines = code.split('\n')
        for line in lines:
            # Missing semicolons
            if re.search(r'(var|let|const|return|throw)\s+.+[^;]$', line):
                issues += 1
                
            # Unused var instead of let/const
            if 'var ' in line:
                issues += 1
                
            # Missing spacing in functions
            if re.search(r'function\([^)]*\)', line):
                issues += 1
        
        # Normalize issues score
        standards_score = max(0.0, 1.0 - (issues / 10))
        
        return standards_score
    
    def _check_java_standards(self, code: str) -> float:
        """Check Java code standards."""
        issues = 0
        
        # Check for common Java standards violations
        lines = code.split('\n')
        for line in lines:
            # Line too long
            if len(line) > 120:
                issues += 1
                
            # Missing braces
            if re.search(r'(if|for|while)\s*\([^)]*\)[^{]*$', line):
                issues += 1
                
            # Variable naming (camelCase)
            if re.search(r'\b(int|float|double|String)\s+[A-Z]', line):
                issues += 1
        
        # Normalize issues score
        standards_score = max(0.0, 1.0 - (issues / 10))
        
        return standards_score