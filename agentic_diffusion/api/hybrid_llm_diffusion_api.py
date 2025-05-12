"""
API for the Hybrid LLM + Diffusion Code Generation approach.

This module provides an API interface for the hybrid code generation approach
that combines Large Language Models with diffusion models for improved quality.
"""

import logging
import time
from typing import Dict, Any, Tuple, Optional, Union

from agentic_diffusion.code_generation.hybrid_llm_diffusion_generator import HybridLLMDiffusionGenerator
from agentic_diffusion.code_generation.rewards.quality_reward import QualityReward
from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward
from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward

# Configure logging
logger = logging.getLogger(__name__)


class HybridLLMDiffusionAPI:
    """
    API interface for the Hybrid LLM + Diffusion code generation approach.
    
    This API provides methods to generate code using the hybrid approach
    and evaluate code quality with various metrics.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the hybrid API.
        
        Args:
            config: Configuration dictionary (optional)
        """
        # Default configuration
        self.config = config or {}
        
        # Initialize the generator
        self.generator = HybridLLMDiffusionGenerator(
            llm_provider=self.config.get("llm_provider", "openai"),
            llm_model=self.config.get("llm_model", "gpt-4"),
            refinement_iterations=self.config.get("refinement_iterations", 3),
            temperature=self.config.get("temperature", 0.7),
            max_tokens=self.config.get("max_tokens", 2048)
        )
        
        # Initialize reward evaluators
        self.quality_reward = QualityReward()
        self.syntax_reward = SyntaxReward()
        self.relevance_reward = RelevanceReward()
        
        logger.info("Initialized Hybrid LLM + Diffusion API")
    
    def generate_code(
        self,
        specification: str,
        language: str = "python",
        **kwargs
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Generate code using the hybrid LLM + diffusion approach.
        
        Args:
            specification: Natural language specification
            language: Target programming language
            **kwargs: Additional parameters for generation
            
        Returns:
            Tuple of (generated code, metadata)
        """
        start_time = time.time()
        logger.info(f"Generating {language} code from specification")
        
        try:
            # Generate code using the hybrid generator
            code, metadata = self.generator.generate(
                specification=specification,
                language=language,
                **kwargs
            )
            
            # Add API-level timing information
            metadata["timing"]["api_time"] = time.time() - start_time
            
            return code, metadata
        except Exception as e:
            logger.error(f"Error generating code: {e}")
            # Return error message and metadata
            return f"# Error: {str(e)}", {"error": str(e)}
    
    def evaluate_code(
        self,
        code: str,
        language: str = "python",
        specification: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate code quality using multiple metrics.
        
        Args:
            code: Code to evaluate
            language: Programming language
            specification: Original specification (for relevance evaluation)
            
        Returns:
            Dictionary of quality metrics
        """
        metrics = {}
        
        try:
            # Basic metrics
            metrics["syntax"] = self.syntax_reward.evaluate(code, language=language)
            metrics["quality"] = self.quality_reward.evaluate(code, language=language)
            
            # Relevance requires a specification
            if specification:
                metrics["relevance"] = self.relevance_reward.evaluate(
                    code, reference=specification, language=language
                )
            
            # Compute overall score (weighted average)
            weights = {"syntax": 0.4, "quality": 0.3, "relevance": 0.3}
            available_weights = {k: v for k, v in weights.items() if k in metrics}
            total_weight = sum(available_weights.values())
            
            if total_weight > 0:
                overall_score = sum(
                    metrics[k] * weights[k] 
                    for k in metrics.keys() if k in weights
                ) / total_weight
            else:
                overall_score = 0.0
                
            metrics["overall"] = overall_score
            
            return metrics
        except Exception as e:
            logger.error(f"Error evaluating code quality: {e}")
            return {"error": str(e), "overall": 0.0}
    
    def compare_approaches(
        self,
        specification: str,
        language: str = "python",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compare the hybrid approach with standard diffusion.
        
        Args:
            specification: Natural language specification
            language: Target programming language
            **kwargs: Additional parameters
            
        Returns:
            Comparison results
        """
        try:
            from agentic_diffusion.api.code_generation_api import create_code_generation_api
            from agentic_diffusion.code_generation.code_diffusion import CodeDiffusion
            
            # Initialize standard diffusion API
            diffusion_model = CodeDiffusion()
            diffusion_api = create_code_generation_api(diffusion_model)
            
            # Generate code with standard diffusion
            diffusion_start = time.time()
            diffusion_code, diffusion_metadata = diffusion_api.generate_code(
                specification=specification,
                language=language
            )
            diffusion_time = time.time() - diffusion_start
            
            # Generate code with hybrid approach
            hybrid_start = time.time()
            hybrid_code, hybrid_metadata = self.generate_code(
                specification=specification,
                language=language,
                **kwargs
            )
            hybrid_time = time.time() - hybrid_start
            
            # Evaluate both versions
            diffusion_quality = self.evaluate_code(
                diffusion_code, language, specification
            )
            
            hybrid_quality = self.evaluate_code(
                hybrid_code, language, specification
            )
            
            # Calculate improvement
            if diffusion_quality["overall"] > 0:
                quality_improvement = (
                    (hybrid_quality["overall"] - diffusion_quality["overall"]) /
                    diffusion_quality["overall"]
                ) * 100
            else:
                quality_improvement = 0.0
            
            # Return comparison results
            return {
                "diffusion": {
                    "code": diffusion_code,
                    "quality": diffusion_quality,
                    "time": diffusion_time
                },
                "hybrid": {
                    "code": hybrid_code,
                    "quality": hybrid_quality,
                    "time": hybrid_time
                },
                "comparison": {
                    "quality_improvement_percent": quality_improvement,
                    "time_ratio": hybrid_time / diffusion_time if diffusion_time > 0 else 0,
                    "specification": specification,
                    "language": language
                }
            }
        except Exception as e:
            logger.error(f"Error comparing approaches: {e}")
            return {"error": str(e)}


def create_hybrid_llm_diffusion_api(config: Optional[Dict[str, Any]] = None) -> HybridLLMDiffusionAPI:
    """
    Factory function to create a hybrid LLM + diffusion API.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized API instance
    """
    return HybridLLMDiffusionAPI(config)