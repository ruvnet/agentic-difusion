"""
Enhanced Reward Model for Diffusion-based Code Generation.

This module implements an advanced reward model that combines multiple metrics
to provide better guidance for the diffusion process during code generation.
"""

import re
import ast
import logging
from typing import Dict, Any, List, Optional, Union, Tuple

from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward
from agentic_diffusion.code_generation.rewards.quality_reward import QualityReward
from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward
from agentic_diffusion.code_generation.syntax_model import SyntaxModel

logger = logging.getLogger(__name__)

class EnhancedRewardModel:
    """
    Advanced reward model for diffusion-based code generation.
    
    This model combines multiple metrics to provide a comprehensive
    assessment of code quality, with specific adaptations for the
    diffusion process requirements.
    """
    
    def __init__(
        self,
        syntax_reward: Optional[SyntaxReward] = None,
        quality_reward: Optional[QualityReward] = None,
        relevance_reward: Optional[RelevanceReward] = None,
        weights: Optional[Dict[str, float]] = None,
        metrics_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the enhanced reward model.
        
        Args:
            syntax_reward: Model for syntax correctness evaluation
            quality_reward: Model for code quality evaluation
            relevance_reward: Model for relevance evaluation
            weights: Dictionary of weights for different metrics
            metrics_config: Configuration for additional metrics
        """
        # Initialize reward components
        self.syntax_reward = syntax_reward or SyntaxReward()
        self.quality_reward = quality_reward or QualityReward()
        self.relevance_reward = relevance_reward or RelevanceReward()
        
        # Default weights give priority to syntax and relevance
        self.weights = weights or {
            "syntax": 0.35,
            "quality": 0.25,
            "relevance": 0.30,
            "complexity": 0.05,
            "documentation": 0.05
        }
        
        # Configuration for metrics
        self.metrics_config = metrics_config or {
            "complexity": {
                "max_line_length": 100,
                "max_indentation": 8,
                "max_nesting": 5
            },
            "documentation": {
                "min_docstring_ratio": 0.1,  # At least 10% of code should be documentation
                "function_doc_required": True
            }
        }
        
        # Optional syntax analyzer
        self.syntax_analyzer = SyntaxModel()
    
    def evaluate(
        self, 
        code: str, 
        specification: Optional[str] = None,
        language: str = "python",
        diffusion_timestep: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate code with multiple metrics, adapted for diffusion process.
        
        Args:
            code: Code to evaluate
            specification: Original specification (for relevance)
            language: Programming language of the code
            diffusion_timestep: Current timestep in diffusion process
            
        Returns:
            Dictionary of metrics and scores
        """
        metrics = {}
        
        # Skip evaluation if code is too short
        if not code or len(code.strip()) < 5:
            return self._default_metrics()
        
        try:
            # Basic metrics from components
            metrics["syntax"] = self.syntax_reward.evaluate(code, language)
            metrics["quality"] = self.quality_reward.evaluate(code, language)
            
            if specification:
                metrics["relevance"] = self.relevance_reward.evaluate(
                    code, reference=specification, language=language
                )
            else:
                metrics["relevance"] = 0.5  # Default when no specification
            
            # Add more advanced metrics
            metrics["complexity"] = self._evaluate_complexity(code, language)
            metrics["documentation"] = self._evaluate_documentation(code, language)
            
            # Apply diffusion timestep-specific adjustments if provided
            if diffusion_timestep is not None:
                metrics = self._adjust_for_diffusion(metrics, diffusion_timestep)
            
            # Calculate overall score
            metrics["overall"] = self._calculate_overall_score(metrics)
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error in reward evaluation: {e}")
            return self._default_metrics()
    
    def _default_metrics(self) -> Dict[str, float]:
        """Return default metrics for error cases."""
        return {
            "syntax": 0.0,
            "quality": 0.0,
            "relevance": 0.0,
            "complexity": 0.0,
            "documentation": 0.0,
            "overall": 0.0
        }
    
    def _calculate_overall_score(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted average of all metrics."""
        total_weight = sum(
            self.weights.get(k, 0.0) for k in metrics.keys() 
            if k != "overall"
        )
        
        if total_weight <= 0:
            return 0.0
            
        weighted_sum = sum(
            metrics[k] * self.weights.get(k, 0.0) 
            for k in metrics.keys() 
            if k != "overall" and k in self.weights
        )
        
        return weighted_sum / total_weight
    
    def _adjust_for_diffusion(
        self, 
        metrics: Dict[str, float], 
        timestep: int,
        max_timesteps: int = 1000
    ) -> Dict[str, float]:
        """
        Adjust metrics based on diffusion timestep.
        
        Early in diffusion (high timestep), we're more lenient.
        Late in diffusion (low timestep), we're more strict.
        
        Args:
            metrics: Current metrics
            timestep: Current diffusion timestep
            max_timesteps: Maximum number of timesteps
            
        Returns:
            Adjusted metrics
        """
        # Calculate progress (0.0 to 1.0)
        # 0.0 = start of diffusion (noisy)
        # 1.0 = end of diffusion (clean)
        progress = 1.0 - (timestep / max_timesteps)
        
        adjusted_metrics = metrics.copy()
        
        # Early in diffusion (noisy): be more lenient on syntax
        # Late in diffusion (clean): be strict on syntax
        if "syntax" in adjusted_metrics:
            if progress < 0.5:
                # Be more lenient in early stages
                leniency = 0.5 * (1.0 - progress * 2)
                adjusted_metrics["syntax"] = min(
                    1.0, 
                    metrics["syntax"] + leniency
                )
            else:
                # Be more strict in later stages
                strictness = 0.2 * ((progress - 0.5) * 2)
                adjusted_metrics["syntax"] = max(
                    0.0, 
                    metrics["syntax"] - strictness
                )
        
        # Similarly adjust other metrics
        for metric in ["quality", "complexity", "documentation"]:
            if metric in adjusted_metrics:
                # Less important in early diffusion, more important in late diffusion
                if progress < 0.3:
                    # Very early stages - these metrics matter less
                    adjusted_metrics[metric] = max(0.3, adjusted_metrics[metric])
                elif progress > 0.7:
                    # Late stages - stricter evaluation
                    scaled_strictness = (progress - 0.7) / 0.3 * 0.2
                    adjusted_metrics[metric] = adjusted_metrics[metric] * (1.0 - scaled_strictness)
        
        return adjusted_metrics
    
    def _evaluate_complexity(self, code: str, language: str) -> float:
        """
        Evaluate code complexity. Lower complexity scores higher.
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Complexity score (0.0 to 1.0, higher is better - meaning less complex)
        """
        lines = code.split('\n')
        if not lines:
            return 0.0
            
        # Calculate metrics
        line_count = len(lines)
        max_line_length = max((len(line) for line in lines if line.strip()), default=0)
        
        # Count indentation levels
        indentation_levels = []
        for line in lines:
            if line.strip():
                indent = len(line) - len(line.lstrip())
                indentation_levels.append(indent)
        
        max_indent = max(indentation_levels, default=0)
        avg_indent = sum(indentation_levels) / max(1, len(indentation_levels))
        
        # Calculate cyclomatic complexity approximation
        if language == "python":
            try:
                # Count control structures
                control_structures = 0
                for line in lines:
                    if re.search(r'\bif\b|\bfor\b|\bwhile\b|\belif\b|\bexcept\b', line):
                        control_structures += 1
                
                # Calculate cyclomatic complexity
                cyclomatic_complexity = 1 + control_structures
                
                # Normalize cyclomatic complexity (1-10 is reasonable)
                norm_complexity = max(0.0, min(1.0, 1.0 - (cyclomatic_complexity - 1) / 9))
            except Exception:
                norm_complexity = 0.5  # Default
        else:
            # Simplified for other languages
            control_count = sum(1 for line in lines if re.search(
                r'\bif\b|\bfor\b|\bwhile\b|\bcatch\b|\bswitch\b', line))
            norm_complexity = max(0.0, min(1.0, 1.0 - control_count / 10))
        
        # Normalize metrics
        config = self.metrics_config["complexity"]
        norm_line_length = max(0.0, min(1.0, 1.0 - max_line_length / config["max_line_length"]))
        norm_max_indent = max(0.0, min(1.0, 1.0 - max_indent / config["max_indentation"]))
        
        # Weight the different aspects of complexity
        weights = {
            "norm_line_length": 0.3,
            "norm_max_indent": 0.3,
            "norm_complexity": 0.4
        }
        
        complexity_score = (
            weights["norm_line_length"] * norm_line_length +
            weights["norm_max_indent"] * norm_max_indent +
            weights["norm_complexity"] * norm_complexity
        )
        
        return complexity_score
    
    def _evaluate_documentation(self, code: str, language: str) -> float:
        """
        Evaluate code documentation quality.
        
        Args:
            code: Code to evaluate
            language: Programming language
            
        Returns:
            Documentation score (0.0 to 1.0)
        """
        lines = code.split('\n')
        total_lines = len(lines)
        if total_lines < 3:
            return 0.0
        
        # Count comment and docstring lines
        if language == "python":
            comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
            
            # Check for docstrings
            docstring_count = 0
            in_docstring = False
            for i, line in enumerate(lines):
                if '"""' in line or "'''" in line:
                    if not in_docstring:
                        in_docstring = True
                    else:
                        in_docstring = False
                    docstring_count += 1
                elif in_docstring:
                    docstring_count += 1
            
            # Check for function/class docstrings
            try:
                tree = ast.parse(code)
                functions = [node for node in ast.walk(tree) 
                           if isinstance(node, (ast.FunctionDef, ast.ClassDef))]
                
                documented_functions = 0
                for func in functions:
                    # Check if function has a docstring
                    if (isinstance(func.body[0], ast.Expr) and 
                        isinstance(func.body[0].value, ast.Str)):
                        documented_functions += 1
                
                function_doc_ratio = documented_functions / max(1, len(functions))
            except SyntaxError:
                function_doc_ratio = 0.0
                
        elif language in ["javascript", "typescript", "java"]:
            comment_lines = sum(1 for line in lines 
                              if line.strip().startswith('//') or
                                 line.strip().startswith('/*') or
                                 line.strip().startswith('*') or
                                 '*/' in line)
            
            # Simplified function doc detection
            function_pattern = r'(function|class|method)'
            jsdoc_pattern = r'\/\*\*[\s\S]*?\*\/'
            
            function_matches = re.findall(function_pattern, code)
            jsdoc_matches = re.findall(jsdoc_pattern, code)
            
            docstring_count = sum(len(match.split('\n')) for match in jsdoc_matches)
            
            # Approximation of function documentation ratio
            function_count = len(function_matches)
            documented_functions = len(jsdoc_matches)
            function_doc_ratio = documented_functions / max(1, function_count)
        else:
            # Generic fallback for other languages
            comment_lines = sum(1 for line in lines if '//' in line or '/*' in line or '*/' in line)
            docstring_count = 0
            function_doc_ratio = 0.0
        
        # Calculate documentation ratio
        doc_lines = comment_lines + docstring_count
        doc_ratio = doc_lines / max(1, total_lines)
        
        # Documentation quality based on different factors
        config = self.metrics_config["documentation"]
        
        # Documentation coverage
        coverage_score = min(1.0, doc_ratio / config["min_docstring_ratio"])
        
        # Function documentation
        func_doc_score = function_doc_ratio if config["function_doc_required"] else 1.0
        
        # Combined score with weights
        return 0.6 * coverage_score + 0.4 * func_doc_score
    
    def compare_codes(
        self, 
        original_code: str, 
        modified_code: str,
        specification: Optional[str] = None,
        language: str = "python"
    ) -> Dict[str, Any]:
        """
        Compare two code versions to determine if modified is better.
        
        This is useful for the diffusion process to evaluate whether
        a denoising step has improved the code quality.
        
        Args:
            original_code: Original code
            modified_code: Modified code
            specification: Original specification
            language: Programming language
            
        Returns:
            Dictionary with comparison results
        """
        # Evaluate both versions
        original_metrics = self.evaluate(original_code, specification, language)
        modified_metrics = self.evaluate(modified_code, specification, language)
        
        # Calculate improvements
        improvements = {
            k: modified_metrics.get(k, 0.0) - original_metrics.get(k, 0.0)
            for k in original_metrics.keys()
        }
        
        # Determine if modification is better overall
        is_better = modified_metrics.get("overall", 0.0) > original_metrics.get("overall", 0.0)
        
        return {
            "original_metrics": original_metrics,
            "modified_metrics": modified_metrics,
            "improvements": improvements,
            "is_better": is_better,
            "overall_improvement": improvements.get("overall", 0.0)
        }