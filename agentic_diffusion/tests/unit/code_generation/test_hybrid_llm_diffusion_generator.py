"""
Unit tests for the HybridLLMDiffusionGenerator class.
"""

import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.code_generation.hybrid_llm_diffusion_generator import HybridLLMDiffusionGenerator


@pytest.fixture
def mock_code_generator():
    """Create a mock code generator."""
    generator = MagicMock()
    generator.generate_code.return_value = "def test_function():\n    return 42"
    return generator


@pytest.fixture
def mock_diffusion_model():
    """Create a mock diffusion model."""
    diffusion = MagicMock()
    diffusion.refine_code.return_value = "def refined_function():\n    return 'refined'"
    return diffusion


@pytest.fixture
def mock_adaptation_model():
    """Create a mock adaptation model."""
    adaptation = MagicMock()
    return adaptation


@pytest.fixture
def mock_enhancement_metric():
    """Create a mock enhancement metric."""
    metric = MagicMock()
    metric.evaluate_code_quality.side_effect = [
        {"overall": 0.65, "syntax": 0.7},  # Initial code metrics
        {"overall": 0.85, "syntax": 0.9}   # Refined code metrics
    ]
    metric.calculate_enhancement.return_value = {
        "overall_improvement": 0.2,
        "overall_improvement_percentage": 25.0
    }
    return metric


class TestHybridLLMDiffusionGenerator:
    """Tests for the HybridLLMDiffusionGenerator class."""
    
    def test_initialization(self):
        """Test that the generator initializes correctly with default values."""
        # Patch the dependencies to avoid actual initialization
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator"):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion"):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel"):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric"):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Check initialization values
                        assert generator.llm_provider == "openai"
                        assert generator.llm_model == "gpt-4"
                        assert generator.refinement_iterations == 3
                        assert generator.temperature == 0.7
                        assert generator.max_tokens == 2048
    
    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        # Patch the dependencies
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator"):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion"):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel"):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric"):
                        generator = HybridLLMDiffusionGenerator(
                            llm_provider="anthropic",
                            llm_model="claude-3",
                            refinement_iterations=5,
                            temperature=0.5,
                            max_tokens=4096
                        )
                        
                        # Check custom values
                        assert generator.llm_provider == "anthropic"
                        assert generator.llm_model == "claude-3"
                        assert generator.refinement_iterations == 5
                        assert generator.temperature == 0.5
                        assert generator.max_tokens == 4096
    
    def test_generate_with_llm(self, mock_code_generator):
        """Test the LLM code generation step."""
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator", 
                   return_value=mock_code_generator):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion"):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel"):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric"):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Test the internal method
                        code = generator._generate_with_llm(
                            "Write a function that returns 42",
                            "python"
                        )
                        
                        # Check the result
                        assert code == "def test_function():\n    return 42"
                        
                        # Verify the call to CodeGenerator
                        mock_code_generator.generate_code.assert_called_once()
    
    def test_refine_with_diffusion(self, mock_diffusion_model, mock_adaptation_model):
        """Test the diffusion refinement step."""
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator"):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion", 
                       return_value=mock_diffusion_model):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel", 
                           return_value=mock_adaptation_model):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric"):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Test the internal method
                        refined_code = generator._refine_with_diffusion(
                            initial_code="def test_function():\n    return 42",
                            specification="Write a function that returns something refined",
                            language="python"
                        )
                        
                        # Check the result
                        assert refined_code == "def refined_function():\n    return 'refined'"
                        
                        # Verify the calls
                        mock_adaptation_model.set_context.assert_called_once()
                        mock_diffusion_model.refine_code.assert_called_once()
    
    def test_compute_quality_metrics(self, mock_enhancement_metric):
        """Test the quality metrics computation."""
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator"):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion"):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel"):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric", 
                               return_value=mock_enhancement_metric):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Test the internal method
                        metrics = generator._compute_quality_metrics(
                            code="def test_function():\n    return 42",
                            language="python",
                            specification="Write a test function"
                        )
                        
                        # Check the result
                        assert metrics["overall"] == 0.65
                        assert metrics["syntax"] == 0.7
                        
                        # Verify the call
                        mock_enhancement_metric.evaluate_code_quality.assert_called_once()
    
    def test_create_llm_prompt(self):
        """Test prompt creation for different languages."""
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator"):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion"):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel"):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric"):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Test Python prompt
                        python_prompt = generator._create_llm_prompt(
                            "Write a test function",
                            "python"
                        )
                        assert "Python" in python_prompt
                        assert "PEP 8" in python_prompt
                        
                        # Test JavaScript prompt
                        js_prompt = generator._create_llm_prompt(
                            "Write a test function",
                            "javascript"
                        )
                        assert "JavaScript" in js_prompt
                        assert "ES6+" in js_prompt
                        
                        # Test Java prompt
                        java_prompt = generator._create_llm_prompt(
                            "Write a test function",
                            "java"
                        )
                        assert "Java" in java_prompt
                        
                        # Test generic language
                        go_prompt = generator._create_llm_prompt(
                            "Write a test function",
                            "go"
                        )
                        assert "Go" in go_prompt
    
    def test_create_fallback_code(self):
        """Test fallback code generation for different languages."""
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator"):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion"):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel"):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric"):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Test Python fallback
                        python_fallback = generator._create_fallback_code("python")
                        assert "def main" in python_fallback
                        assert "__name__ == '__main__'" in python_fallback
                        
                        # Test JavaScript fallback
                        js_fallback = generator._create_fallback_code("javascript")
                        assert "function main" in js_fallback
                        
                        # Test Java fallback
                        java_fallback = generator._create_fallback_code("java")
                        assert "public class Main" in java_fallback
                        
                        # Test generic language
                        go_fallback = generator._create_fallback_code("go")
                        assert "TODO" in go_fallback
                        assert "go" in go_fallback
    
    def test_generate_end_to_end(self, mock_code_generator, mock_diffusion_model, 
                                  mock_adaptation_model, mock_enhancement_metric):
        """Test the complete generation pipeline."""
        # Patch all dependencies
        with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator", 
                   return_value=mock_code_generator):
            with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeDiffusion", 
                       return_value=mock_diffusion_model):
                with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeAdaptationModel", 
                           return_value=mock_adaptation_model):
                    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.QualityEnhancementMetric", 
                               return_value=mock_enhancement_metric):
                        generator = HybridLLMDiffusionGenerator()
                        
                        # Call the generate method
                        code, metadata = generator.generate(
                            specification="Write a function that returns 42",
                            language="python"
                        )
                        
                        # Check the results
                        assert code == "def refined_function():\n    return 'refined'"
                        
                        # Check metadata structure
                        assert "quality" in metadata
                        assert "timing" in metadata
                        assert "performance" in metadata
                        assert "config" in metadata
                        
                        # Check quality metrics
                        assert "quality_improvement_percentage" in metadata["quality"]
                        assert metadata["quality"]["quality_improvement_percentage"] == 25.0
                        
                        # Check timing information
                        assert "total_time" in metadata["timing"]
                        assert "llm_generation_time" in metadata["timing"]
                        assert "diffusion_refinement_time" in metadata["timing"]
                        
                        # Verify all the calls
                        mock_code_generator.generate_code.assert_called_once()
                        mock_adaptation_model.set_context.assert_called_once()
                        mock_diffusion_model.refine_code.assert_called_once()