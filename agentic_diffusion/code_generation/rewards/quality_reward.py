"""
QualityReward: Computes a quality score for generated code based on syntax, complexity,
readability, documentation, and best practices.

This module provides a comprehensive reward model for evaluating code quality.
"""

import re
import logging
from typing import Dict, Optional, List

from agentic_diffusion.code_generation.syntax_model import SyntaxModel
from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward

logger = logging.getLogger(__name__)

class QualityReward:
    """
    Computes a reward score for code quality.
    
    This reward model evaluates multiple aspects of code quality including
    syntax correctness, complexity, readability, and adherence to best practices.
    """

    def __init__(
        self,
        syntax_reward=None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the quality reward model.
        
        Args:
            syntax_reward: Optional syntax reward model
            weights: Optional custom weights for different quality aspects
        """
        self.syntax_reward = syntax_reward or SyntaxReward()
        
        # Default weights for quality metrics
        self.weights = weights or {
            "syntax": 0.40,
            "complexity": 0.20,
            "readability": 0.20,
            "documentation": 0.20
        }
        
        # Normalize weights to sum to 1.0
        total_weight = sum(self.weights.values())
        if total_weight != 1.0:
            for key in self.weights:
                self.weights[key] /= total_weight
        
        logger.info("Initialized QualityReward with weights: %s", self.weights)
    
    def __call__(self, code: str, language: str = "python", reference: Optional[str] = None) -> float:
        """
        Compute quality score for the given code.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            reference: Optional reference code or specification
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        return self.evaluate(code, language, reference)
    
    def evaluate(self, code: str, language: str = "python", reference: Optional[str] = None) -> float:
        """
        Evaluate code quality.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            reference: Optional reference code or specification
            
        Returns:
            Quality score between 0.0 and 1.0
        """
        # Check if code is empty or whitespace
        if not code or code.isspace():
            return 0.0
        
        try:
            # Evaluate different quality aspects
            syntax_score = self.syntax_reward.evaluate(code, language)
            complexity_score = self._evaluate_complexity(code, language)
            readability_score = self._evaluate_readability(code, language)
            documentation_score = self._evaluate_documentation(code, language)
            
            # Compute weighted score
            weighted_score = (
                self.weights["syntax"] * syntax_score +
                self.weights["complexity"] * complexity_score +
                self.weights["readability"] * readability_score +
                self.weights["documentation"] * documentation_score
            )
            
            return weighted_score
            
        except Exception as e:
            logger.error("Error evaluating quality: %s", str(e))
            return 0.0
    
    def get_detailed_scores(self, code: str, language: str = "python") -> Dict[str, float]:
        """
        Get detailed scores for different quality aspects.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Dictionary of scores for different quality aspects
        """
        syntax_score = self.syntax_reward.evaluate(code, language)
        complexity_score = self._evaluate_complexity(code, language)
        readability_score = self._evaluate_readability(code, language)
        documentation_score = self._evaluate_documentation(code, language)
        
        return {
            "syntax": syntax_score,
            "complexity": complexity_score,
            "readability": readability_score,
            "documentation": documentation_score
        }
    
    def _evaluate_complexity(self, code: str, language: str) -> float:
        """
        Evaluate code complexity. Lower complexity scores mean better quality.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Complexity score between 0.0 and 1.0 (higher is better, meaning less complex)
        """
        lines = code.split('\n')
        if not lines:
            return 0.0
            
        # Count control flow structures (if, for, while, etc.)
        control_flow_patterns = {
            "python": r'\b(if|elif|else|for|while|try|except|with)\b',
            "javascript": r'\b(if|else|for|while|do|switch|try|catch|finally)\b',
            "java": r'\b(if|else|for|while|do|switch|try|catch|finally)\b',
            "go": r'\b(if|else|for|switch|select|case|defer)\b'
        }
        
        pattern = control_flow_patterns.get(language.lower(), r'\b(if|else|for|while)\b')
        control_flow_count = sum(1 for line in lines if re.search(pattern, line))
        
        # Calculate nesting depth
        max_indent = 0
        for line in lines:
            if not line.strip():
                continue
            indent = len(line) - len(line.lstrip())
            max_indent = max(max_indent, indent)
        
        # Convert to nesting levels (assuming 4 spaces or 1 tab per level)
        nesting_depth = max_indent // 4
        
        # Calculate complexity penalty
        complexity_penalty = 0.0
        
        # Penalize excessive control flow
        if control_flow_count > len(lines) * 0.3:  # More than 30% of lines have control flow
            complexity_penalty += min(0.5, (control_flow_count - len(lines) * 0.3) * 0.1)
        
        # Penalize deep nesting
        if nesting_depth > 3:
            complexity_penalty += min(0.5, (nesting_depth - 3) * 0.15)
        
        return max(0.0, 1.0 - complexity_penalty)
    
    def _evaluate_readability(self, code: str, language: str) -> float:
        """
        Evaluate code readability.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Readability score between 0.0 and 1.0
        """
        lines = code.split('\n')
        if not lines:
            return 0.0
            
        readability_penalty = 0.0
        
        # Check line length
        long_line_limit = 80 if language.lower() == "python" else 100
        long_lines = sum(1 for line in lines if len(line.rstrip()) > long_line_limit)
        if long_lines > 0:
            readability_penalty += min(0.3, (long_lines / len(lines)) * 0.5)
        
        # Check for mixed indentation
        has_tabs = '\t' in code
        has_spaces = '    ' in code
        if has_tabs and has_spaces:
            readability_penalty += 0.2
            
        # Check for very short or unclear variable names
        short_var_pattern = r'\b[a-z]{1,2}\b'
        short_vars = len(re.findall(short_var_pattern, code))
        if short_vars > 5:
            readability_penalty += min(0.2, (short_vars - 5) * 0.02)
            
        return max(0.0, 1.0 - readability_penalty)
    
    def _evaluate_documentation(self, code: str, language: str) -> float:
        """
        Evaluate code documentation.
        
        Args:
            code: Code to evaluate
            language: Programming language of the code
            
        Returns:
            Documentation score between 0.0 and 1.0
        """
        lines = code.split('\n')
        if not lines:
            return 0.0
            
        # Count comment lines
        comment_patterns = {
            "python": r'^\s*(#|""")',
            "javascript": r'^\s*(//|/\*|\*)',
            "java": r'^\s*(//|/\*|\*)',
            "go": r'^\s*//'
        }
        
        pattern = comment_patterns.get(language.lower(), r'^\s*(//|#|/\*|\*)')
        comment_lines = sum(1 for line in lines if re.search(pattern, line.strip()))
        
        # Calculate comment ratio
        comment_ratio = comment_lines / max(1, len(lines))
        
        # Score based on comment ratio - aim for 15-25% comments
        if comment_ratio < 0.05:  # Less than 5% comments
            return max(0.0, comment_ratio * 10)  # Scale up to 0.5 max
        elif comment_ratio < 0.15:  # Between 5% and 15%
            return 0.5 + (comment_ratio - 0.05) * 5  # Scale from 0.5 to 1.0
        elif comment_ratio <= 0.25:  # Optimal range: 15-25%
            return 1.0
        else:  # More than 25% comments
            return max(0.5, 1.0 - (comment_ratio - 0.25) * 2)  # Penalize excessive comments