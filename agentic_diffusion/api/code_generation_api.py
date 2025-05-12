"""
Code Generation API module.

This module provides API endpoints for code generation using
the Agentic Diffusion system components. It integrates all the code
generation components (CodeTokenizer, SyntaxModel, CodeGenerator,
CodeAdaptationModel) and adaptation mechanisms.
"""

import os
import json
import time
import tracemalloc
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.syntax_model import SyntaxModel
from agentic_diffusion.code_generation.code_generator import CodeGenerator
from agentic_diffusion.code_generation.code_adaptation_model import CodeAdaptationModel
from agentic_diffusion.adaptation.gradient_adaptation import GradientBasedAdaptation
from agentic_diffusion.adaptation.memory_adaptation import MemoryAdaptation
from agentic_diffusion.adaptation.hybrid_adaptation import HybridAdaptation
from agentic_diffusion.code_generation.rewards.quality_reward import QualityReward
from agentic_diffusion.code_generation.rewards.relevance_reward import RelevanceReward
from agentic_diffusion.code_generation.rewards.syntax_reward import SyntaxReward


class CodeGenerationAPI:
    """
    API for code generation and adaptation.
    
    This class integrates all code generation components and provides
    a unified interface for code generation, adaptation, and refinement.
    """
    
    def __init__(self, diffusion_model, config=None):
        """
        Initialize the code generation API.
        
        Args:
            diffusion_model: The diffusion model to use for code generation
            config (dict, optional): Configuration for the API components
        """
        self.config = config or {}
        self.diffusion_model = diffusion_model
        
        # Extract performance-related config
        self.batch_size = self.config.get("batch_size", 4)  # Increased default for better sample selection
        self.precision = self.config.get("precision", "float32")
        self.device = self.config.get("device", None)
        
        # Extract diffusion model parameters
        self.guidance_scale = self.config.get("guidance_scale", 1.5)
        self.temperature = self.config.get("temperature", 0.7)
        self.use_rewards = self.config.get("use_rewards", True)
        self.max_length = self.config.get("max_length", 512)
        self.num_iterations = self.config.get("num_iterations", 1)
        
        # Initialize the code generation components
        self._init_components()
        
    def _init_components(self):
        """Initialize all the code generation components."""
        # Create the tokenizer
        self.tokenizer = CodeTokenizer(
            language=self.config.get("default_language", "python")
        )
        
        # Create the syntax model
        self.syntax_model = SyntaxModel()
        
        # Create the code generator
        self.code_generator = CodeGenerator(
            tokenizer=self.tokenizer,
            syntax_model=self.syntax_model,
            diffusion_model=self.diffusion_model
        )
        
        # Create the reward models
        self.quality_reward = QualityReward()
        self.relevance_reward = RelevanceReward()
        self.syntax_reward = SyntaxReward()
        
        reward_models = [
            self.quality_reward,
            self.relevance_reward,
            self.syntax_reward
        ]
        
        # Create the adaptation mechanisms
        adaptation_type = self.config.get("adaptation_type", "hybrid")
        
        if adaptation_type == "gradient":
            self.adaptation_mechanism = GradientBasedAdaptation(
                diffusion_model=self.diffusion_model,
                adaptation_rate=self.config.get("adaptation_rate", 0.1),
                memory_capacity=self.config.get("memory_capacity", 5)
            )
        elif adaptation_type == "memory":
            self.adaptation_mechanism = MemoryAdaptation(
                code_generator=self.code_generator,
                memory_size=self.config.get("memory_size", 100),
                similarity_threshold=self.config.get("similarity_threshold", 0.7)
            )
        else:  # hybrid (default)
            self.adaptation_mechanism = HybridAdaptation(
                diffusion_model=self.diffusion_model,
                code_generator=self.code_generator,
                gradient_weight=self.config.get("gradient_weight", 0.5),
                memory_weight=self.config.get("memory_weight", 0.5),
                adaptation_rate=self.config.get("adaptation_rate", 0.1),
                memory_size=self.config.get("memory_size", 100)
            )
            
        # Create the code adaptation model
        self.code_adaptation_model = CodeAdaptationModel(
            adaptation_mechanism=self.adaptation_mechanism,
            code_generator=self.code_generator,
            reward_models=reward_models
        )
        
    def generate_code(self, specification, language=None, partial_code=None,
                     custom_parameters=None):
        """
        Generate code from a specification.
        
        Args:
            specification (str): The code specification
            language (str, optional): The programming language. Defaults to None.
            partial_code (str, optional): Partial code to complete. Defaults to None.
            custom_parameters (dict, optional): Custom parameters to override defaults.
            
        Returns:
            tuple: (generated_code, metadata)
        """
        # Prepare parameters with custom overrides
        params = {
            "specification": specification,
            "language": language,
            "partial_code": partial_code,
            "batch_size": self.batch_size,
            "precision": self.precision,
            "device": self.device,
            "guidance_scale": self.guidance_scale,
            "temperature": self.temperature,
            "use_rewards": self.use_rewards,
            "max_length": self.max_length,
            "num_iterations": self.num_iterations
        }
        
        # Override with custom parameters if provided
        if custom_parameters:
            params.update(custom_parameters)
        
        # Performance profiling
        start_time = time.perf_counter()
        tracemalloc.start()
        
        # Generate code with proper error handling
        try:
            code = self.code_generator.generate_code(**params)
        except Exception as e:
            # Stop memory tracking
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            elapsed = time.perf_counter() - start_time
            
            # Determine error type and create detailed metadata
            error_type = type(e).__name__
            error_msg = str(e).lower()
            
            # Create error metadata with more detailed information
            error_metadata = {
                "performance": {
                    "elapsed_time_sec": elapsed,
                    "memory_current_bytes": current,
                    "memory_peak_bytes": peak
                },
                "error": {
                    "type": error_type,
                    "message": str(e),
                    "details": "Code generation failed with an error. This may be due to model issues or invalid input."
                },
                "generation_parameters": {
                    "language": language or self.tokenizer.language,
                    "batch_size": params["batch_size"],
                    "precision": params["precision"],
                    "guidance_scale": params["guidance_scale"],
                    "temperature": params["temperature"],
                    "max_length": params["max_length"],
                    "num_iterations": params["num_iterations"]
                }
            }
            
            # Provide more specific error details based on error message patterns
            if "dimension mismatch" in error_msg or "shape" in error_msg or "size mismatch" in error_msg:
                error_metadata["error"]["category"] = "dimension_mismatch"
                error_metadata["error"]["details"] = (
                    "Dimension mismatch detected in the diffusion model. This may be caused by incompatible "
                    "dimensions in the model's condition blocks. Please update the CodeUNet implementation "
                    "to handle dynamic dimensions across different layers."
                )
                error_metadata["error"]["suggested_fix"] = (
                    "Check the embedding dimensions in code_unet.py and ensure consistent dimensions "
                    "across all blocks in the model architecture."
                )
            elif "empty code" in error_msg:
                error_metadata["error"]["category"] = "empty_output"
                error_metadata["error"]["details"] = (
                    "The diffusion model generated empty code. This may be due to insufficient "
                    "training or incorrect configuration parameters."
                )
                error_metadata["error"]["suggested_fix"] = (
                    "Try adjusting temperature or guidance_scale parameters to improve generation quality."
                )
            elif "failed to generate valid code" in error_msg:
                error_metadata["error"]["category"] = "generation_failure"
                error_metadata["error"]["details"] = (
                    "The diffusion model failed to generate valid code after maximum attempts. "
                    "This may indicate issues with model compatibility or configuration."
                )
                error_metadata["error"]["suggested_fix"] = (
                    "Check for model compatibility issues and verify the input specification is valid."
                )
            
            # Log the detailed error
            print(f"Code generation error: {error_type} - {str(e)}")
            print(f"Error details: {error_metadata['error']['details']}")
                
            return None, error_metadata
            
        # Capture performance metrics for successful generation
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - start_time
        
        # Evaluate code quality
        quality_metrics = self.evaluate_code(code, specification, language)
        
        # Create metadata object with detailed information
        metadata = {
            "performance": {
                "elapsed_time_sec": elapsed,
                "memory_current_bytes": current,
                "memory_peak_bytes": peak
            },
            "quality": quality_metrics,
            "generation_parameters": {
                "language": language or self.tokenizer.language,
                "batch_size": params["batch_size"],
                "precision": params["precision"],
                "guidance_scale": params["guidance_scale"],
                "temperature": params["temperature"],
                "use_rewards": params["use_rewards"],
                "max_length": params["max_length"],
                "num_iterations": params["num_iterations"]
            }
        }
        
        self.last_profile = metadata["performance"]
        return code, metadata
        
    def adapt_code(self, code, language=None, feedback=None, max_iterations=1):
        """
        Adapt code based on feedback.
        
        Args:
            code (str): The code to adapt
            language (str, optional): The programming language. Defaults to None.
            feedback (dict, optional): Feedback for adaptation. Defaults to None.
            max_iterations (int, optional): Maximum adaptation iterations. Defaults to 1.
            
        Returns:
            str: The adapted code
        """
        start_time = time.perf_counter()
        tracemalloc.start()
        adapted = self.code_adaptation_model.adapt(
            code=code,
            language=language,
            feedback=feedback,
            max_iterations=max_iterations
        )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - start_time
        self.last_profile = {
            "elapsed_time_sec": elapsed,
            "memory_current_bytes": current,
            "memory_peak_bytes": peak
        }
        return adapted
        
    def improve_code(self, code, feedback, language=None):
        """
        Improve code based on specific feedback.
        
        Args:
            code (str): The code to improve
            feedback (dict): Feedback for improvement
            language (str, optional): The programming language. Defaults to None.
            
        Returns:
            str: The improved code
        """
        start_time = time.perf_counter()
        tracemalloc.start()
        improved = self.code_adaptation_model.improve(
            code=code,
            feedback=feedback,
            language=language
        )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - start_time
        self.last_profile = {
            "elapsed_time_sec": elapsed,
            "memory_current_bytes": current,
            "memory_peak_bytes": peak
        }
        return improved
        
    def refine_code(self, code, language=None, iterations=1):
        """
        Refine code iteratively.
        
        Args:
            code (str): The code to refine
            language (str, optional): The programming language. Defaults to None.
            iterations (int, optional): Number of refinement iterations. Defaults to 1.
            
        Returns:
            str: The refined code
        """
        start_time = time.perf_counter()
        tracemalloc.start()
        refined_code = code
        for _ in range(iterations):
            refined_code = self.code_adaptation_model.refine(
                code=refined_code,
                language=language
            )
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        elapsed = time.perf_counter() - start_time
        self.last_profile = {
            "elapsed_time_sec": elapsed,
            "memory_current_bytes": current,
            "memory_peak_bytes": peak
        }
        return refined_code
        
    def evaluate_code(self, code, specification=None, language=None):
        """
        Evaluate code quality metrics.
        
        Args:
            code (str): The code to evaluate
            specification (str, optional): The original specification for relevance evaluation
            language (str, optional): The programming language. Defaults to None.
            
        Returns:
            dict: Code quality metrics
        """
        metrics = {}
        
        # Use the actual language or default to Python
        lang = language or self.tokenizer.language
        
        # Evaluate syntax correctness
        metrics["syntax_score"] = self.syntax_reward.evaluate(code, language=lang)
        
        # Evaluate code quality
        metrics["quality_score"] = self.quality_reward.evaluate(code, language=lang)
        
        # Evaluate relevance to specification if provided
        if specification:
            metrics["relevance_score"] = self.relevance_reward.evaluate(
                code, reference=specification, language=lang
            )
        else:
            # If no specification provided, use a default value
            metrics["relevance_score"] = self.relevance_reward.evaluate(code, language=lang)
        
        # Compute overall score (weighted average)
        weights = {
            "syntax_score": 0.4,
            "quality_score": 0.3,
            "relevance_score": 0.3
        }
        
        # Calculate overall score
        overall_score = sum(metrics[k] * weights[k] for k in metrics) / sum(weights.values())
        metrics["overall_score"] = overall_score
        
        # Add boolean flag for convenience
        metrics["syntax_correct"] = metrics["syntax_score"] > 0.7
        
        # Add complexity rating
        if metrics["quality_score"] < 0.3:
            metrics["complexity"] = "high"
        elif metrics["quality_score"] < 0.7:
            metrics["complexity"] = "medium"
        else:
            metrics["complexity"] = "low"
        
        return metrics
        
    def save_state(self, directory):
        """
        Save the state of all components.
        
        Args:
            directory (str): Directory to save the state
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            os.makedirs(directory, exist_ok=True)
            
            # Save the configuration
            with open(os.path.join(directory, "config.json"), "w") as f:
                json.dump(self.config, f)
                
            # Save the adaptation mechanism state
            adaptation_path = os.path.join(directory, "adaptation_mechanism.pkl")
            self.adaptation_mechanism.save_state(adaptation_path)
            
            return True
        except Exception as e:
            print(f"Error saving state: {e}")
            return False
            
    def load_state(self, directory):
        """
        Load the state of all components.
        
        Args:
            directory (str): Directory to load the state from
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Load the configuration
            config_path = os.path.join(directory, "config.json")
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.config = json.load(f)
                
            # Re-initialize components with the loaded config
            self._init_components()
            
            # Load the adaptation mechanism state
            adaptation_path = os.path.join(directory, "adaptation_mechanism.pkl")
            if os.path.exists(adaptation_path):
                self.adaptation_mechanism.load_state(adaptation_path)
                
            return True
        except Exception as e:
            print(f"Error loading state: {e}")
            return False


# Factory function to create the API
def create_code_generation_api(diffusion_model, config=None):
    """
    Create a code generation API instance.
    
    Args:
        diffusion_model: The diffusion model to use
        config (dict, optional): Configuration for the API. Defaults to None.
        
    Returns:
        CodeGenerationAPI: The code generation API
    """
    return CodeGenerationAPI(diffusion_model, config)