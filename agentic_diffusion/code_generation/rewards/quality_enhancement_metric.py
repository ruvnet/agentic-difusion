"""
Quality Enhancement Metric for the Hybrid LLM + Diffusion approach.

This module provides metrics to measure the quality enhancements achieved by
the hybrid approach compared to the standard diffusion-only approach.
"""

import time
import logging
from typing import Dict, Any, List, Tuple, Optional, Union

from agentic_diffusion.code_generation.rewards.quality_reward import QualityReward
from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward
from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward

# Configure logging
logger = logging.getLogger(__name__)


class QualityEnhancementMetric:
    """
    Metrics to measure quality improvements from the hybrid approach.
    
    This class provides methods to quantify the quality improvements achieved
    by the hybrid LLM + diffusion approach compared to the diffusion-only approach.
    """
    
    def __init__(
        self,
        quality_reward: Optional[QualityReward] = None,
        relevance_reward: Optional[RelevanceReward] = None,
        syntax_reward: Optional[SyntaxReward] = None,
        weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize the quality enhancement metric.
        
        Args:
            quality_reward: Quality reward evaluator (created if None)
            relevance_reward: Relevance reward evaluator (created if None)
            syntax_reward: Syntax reward evaluator (created if None)
            weights: Weights for each quality aspect (defaults to even distribution)
        """
        self.quality_reward = quality_reward or QualityReward()
        self.relevance_reward = relevance_reward or RelevanceReward()
        self.syntax_reward = syntax_reward or SyntaxReward()
        
        # Default weights if not provided
        self.weights = weights or {
            "syntax": 0.4,
            "quality": 0.3,
            "relevance": 0.3
        }
    
    def evaluate_code_quality(
        self, 
        code: str, 
        language: str = "python",
        specification: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate code quality using multiple metrics.
        
        Args:
            code: The code to evaluate
            language: Programming language of the code
            specification: Original specification (for relevance evaluation)
            
        Returns:
            Dictionary of quality metrics
        """
        try:
            metrics = {}
            
            # Basic metrics
            metrics["syntax"] = self.syntax_reward.evaluate(code, language=language)
            metrics["quality"] = self.quality_reward.evaluate(code, language=language)
            
            # Relevance requires a specification
            if specification:
                metrics["relevance"] = self.relevance_reward.evaluate(
                    code, reference=specification, language=language
                )
            
            # Compute overall score (weighted average)
            total_weight = sum(
                self.weights[k] for k in metrics.keys() if k in self.weights
            )
            
            if total_weight > 0:
                overall_score = sum(
                    metrics[k] * self.weights[k] 
                    for k in metrics.keys() if k in self.weights
                ) / total_weight
            else:
                overall_score = 0.0
                
            metrics["overall"] = overall_score
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating code quality: {e}")
            return {"error": str(e), "overall": 0.0}
    
    def calculate_enhancement(
        self,
        initial_code: str,
        enhanced_code: str,
        language: str = "python",
        specification: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate quality enhancement between initial and enhanced code.
        
        Args:
            initial_code: Initial code (from standard approach)
            enhanced_code: Enhanced code (from hybrid approach)
            language: Programming language of the code
            specification: Original specification (for relevance evaluation)
            
        Returns:
            Dictionary of enhancement metrics
        """
        # Evaluate both code versions
        initial_metrics = self.evaluate_code_quality(
            initial_code, language, specification
        )
        enhanced_metrics = self.evaluate_code_quality(
            enhanced_code, language, specification
        )
        
        # Calculate improvements for each metric
        improvements = {}
        improvement_percentages = {}
        
        for key in initial_metrics.keys():
            if key in enhanced_metrics:
                improvements[key] = enhanced_metrics[key] - initial_metrics[key]
                # Avoid division by zero
                if initial_metrics[key] > 0:
                    improvement_percentages[key] = (
                        improvements[key] / initial_metrics[key]
                    ) * 100
                else:
                    improvement_percentages[key] = 0.0
        
        return {
            "initial_metrics": initial_metrics,
            "enhanced_metrics": enhanced_metrics,
            "absolute_improvements": improvements,
            "improvement_percentages": improvement_percentages,
            "overall_improvement": improvements.get("overall", 0.0),
            "overall_improvement_percentage": improvement_percentages.get("overall", 0.0)
        }
    
    def batch_evaluation(
        self,
        code_pairs: List[Tuple[str, str]],
        language: str = "python",
        specifications: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate quality enhancements across a batch of code pairs.
        
        Args:
            code_pairs: List of (initial_code, enhanced_code) tuples
            language: Programming language of the code
            specifications: Optional list of specifications (for relevance)
            
        Returns:
            Aggregate enhancement metrics
        """
        if specifications and len(specifications) != len(code_pairs):
            raise ValueError(
                "Number of specifications must match number of code pairs"
            )
        
        enhancements = []
        total_improvement = 0.0
        total_improvement_percentage = 0.0
        
        for i, (initial, enhanced) in enumerate(code_pairs):
            spec = specifications[i] if specifications else None
            
            result = self.calculate_enhancement(
                initial, enhanced, language, spec
            )
            
            enhancements.append(result)
            total_improvement += result["overall_improvement"]
            total_improvement_percentage += result["overall_improvement_percentage"]
        
        count = len(code_pairs)
        return {
            "individual_enhancements": enhancements,
            "average_improvement": total_improvement / count if count > 0 else 0.0,
            "average_improvement_percentage": (
                total_improvement_percentage / count if count > 0 else 0.0
            ),
            "sample_count": count
        }


def measure_code_enhancement(
    initial_code: str,
    enhanced_code: str,
    language: str = "python",
    specification: Optional[str] = None
) -> Dict[str, Any]:
    """
    Utility function to measure enhancement between two code versions.
    
    Args:
        initial_code: Initial code version
        enhanced_code: Enhanced code version
        language: Programming language
        specification: Original specification
        
    Returns:
        Enhancement metrics
    """
    metric = QualityEnhancementMetric()
    return metric.calculate_enhancement(
        initial_code, enhanced_code, language, specification
    )