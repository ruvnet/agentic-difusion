"""
Unit tests for the HybridLLMDiffusionAPI class.
"""

import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.api.hybrid_llm_diffusion_api import (
    HybridLLMDiffusionAPI,
    create_hybrid_llm_diffusion_api
)


@pytest.fixture
def mock_generator():
    """Create a mock hybrid generator."""
    generator = MagicMock()
    generator.generate.return_value = (
        "def test_function():\n    return 'test'",
        {
            "quality": {
                "overall": 0.85,
                "syntax": 0.9,
                "quality_improvement": 0.2,
                "quality_improvement_percentage": 25.0
            },
            "timing": {
                "total_time": 2.5,
                "llm_generation_time": 1.0,
                "diffusion_refinement_time": 1.5
            },
            "performance": {
                "tokens_generated": 10,
                "iterations": 3
            },
            "config": {
                "llm_provider": "openai",
                "llm_model": "gpt-4",
                "temperature": 0.7,
                "refinement_iterations": 3
            }
        }
    )
    
    return generator


@pytest.fixture
def mock_reward_evaluators():
    """Create mock reward evaluators."""
    quality_reward = MagicMock()
    quality_reward.evaluate.return_value = 0.8
    
    syntax_reward = MagicMock()
    syntax_reward.evaluate.return_value = 0.9
    
    relevance_reward = MagicMock()
    relevance_reward.evaluate.return_value = 0.7
    
    return quality_reward, syntax_reward, relevance_reward


class TestHybridLLMDiffusionAPI:
    """Tests for the HybridLLMDiffusionAPI class."""
    
    def test_initialization(self):
        """Test that the API initializes with default values."""
        # Patch the hybrid generator
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator"):
            # Initialize the API
            api = HybridLLMDiffusionAPI()
            
            # Check that the generator was created
            assert hasattr(api, "generator")
            assert hasattr(api, "quality_reward")
            assert hasattr(api, "syntax_reward")
            assert hasattr(api, "relevance_reward")
    
    def test_initialization_with_config(self):
        """Test initialization with custom configuration."""
        config = {
            "llm_provider": "anthropic",
            "llm_model": "claude-3",
            "refinement_iterations": 5,
            "temperature": 0.5
        }
        
        # Patch the hybrid generator
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator") as mock_generator_class:
            # Initialize the API
            api = HybridLLMDiffusionAPI(config)
            
            # Verify that the generator was created with the config
            mock_generator_class.assert_called_once_with(
                llm_provider="anthropic",
                llm_model="claude-3",
                refinement_iterations=5,
                temperature=0.5,
                max_tokens=2048
            )
    
    def test_generate_code(self, mock_generator):
        """Test code generation through the API."""
        # Patch the generator class
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator", 
                   return_value=mock_generator):
            # Initialize the API
            api = HybridLLMDiffusionAPI()
            
            # Generate code
            code, metadata = api.generate_code(
                specification="Write a test function",
                language="python"
            )
            
            # Check the results
            assert code == "def test_function():\n    return 'test'"
            assert "quality" in metadata
            assert "timing" in metadata
            assert "api_time" in metadata["timing"]
            
            # Verify the call to the generator
            mock_generator.generate.assert_called_once_with(
                specification="Write a test function",
                language="python"
            )
    
    def test_generate_code_with_error(self):
        """Test error handling in code generation."""
        # Create a mock generator that raises an exception
        mock_generator = MagicMock()
        mock_generator.generate.side_effect = Exception("Test error")
        
        # Patch the generator class
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator", 
                   return_value=mock_generator):
            # Initialize the API
            api = HybridLLMDiffusionAPI()
            
            # Generate code (should handle the error)
            code, metadata = api.generate_code(
                specification="Write a test function",
                language="python"
            )
            
            # Check that error information is returned
            assert code.startswith("# Error:")
            assert "error" in metadata
            assert metadata["error"] == "Test error"
    
    def test_evaluate_code(self, mock_reward_evaluators):
        """Test code quality evaluation."""
        quality_reward, syntax_reward, relevance_reward = mock_reward_evaluators
        
        # Patch the reward classes
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.QualityReward", 
                   return_value=quality_reward):
            with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.SyntaxReward", 
                       return_value=syntax_reward):
                with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.RelevanceReward", 
                           return_value=relevance_reward):
                    with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator"):
                        # Initialize the API
                        api = HybridLLMDiffusionAPI()
                        
                        # Evaluate code
                        metrics = api.evaluate_code(
                            code="def test():\n    return 42",
                            language="python",
                            specification="Write a function that returns 42"
                        )
                        
                        # Check the metrics
                        assert "syntax" in metrics
                        assert "quality" in metrics
                        assert "relevance" in metrics
                        assert "overall" in metrics
                        
                        assert metrics["syntax"] == 0.9
                        assert metrics["quality"] == 0.8
                        assert metrics["relevance"] == 0.7
                        
                        # Check the overall score (weighted average)
                        expected_overall = (0.9 * 0.4 + 0.8 * 0.3 + 0.7 * 0.3)
                        assert metrics["overall"] == expected_overall
                        
                        # Verify the calls to reward evaluators
                        syntax_reward.evaluate.assert_called_once()
                        quality_reward.evaluate.assert_called_once()
                        relevance_reward.evaluate.assert_called_once()
    
    def test_evaluate_code_without_specification(self, mock_reward_evaluators):
        """Test code evaluation without a specification."""
        quality_reward, syntax_reward, relevance_reward = mock_reward_evaluators
        
        # Patch the reward classes
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.QualityReward", 
                   return_value=quality_reward):
            with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.SyntaxReward", 
                       return_value=syntax_reward):
                with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.RelevanceReward", 
                           return_value=relevance_reward):
                    with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator"):
                        # Initialize the API
                        api = HybridLLMDiffusionAPI()
                        
                        # Evaluate code without specification
                        metrics = api.evaluate_code(
                            code="def test():\n    return 42",
                            language="python"
                        )
                        
                        # Check the metrics
                        assert "syntax" in metrics
                        assert "quality" in metrics
                        assert "relevance" not in metrics
                        assert "overall" in metrics
                        
                        # Verify that relevance was not evaluated
                        relevance_reward.evaluate.assert_not_called()
    
    def test_compare_approaches(self, mock_generator):
        """Test comparison between diffusion and hybrid approaches."""
        # Mock for standard diffusion API
        mock_diffusion_api = MagicMock()
        mock_diffusion_api.generate_code.return_value = (
            "def standard_function():\n    return 'standard'",
            {"quality": {"overall": 0.7}}
        )
        
        # Patch the create_code_generation_api function
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.create_code_generation_api", 
                   return_value=mock_diffusion_api):
            # Patch the CodeDiffusion class
            with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.CodeDiffusion"):
                # Patch the HybridLLMDiffusionGenerator
                with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator", 
                           return_value=mock_generator):
                    # Patch the evaluate_code method
                    with patch.object(HybridLLMDiffusionAPI, "evaluate_code") as mock_evaluate:
                        # Set up mock evaluation results
                        mock_evaluate.side_effect = [
                            {"overall": 0.7},  # Diffusion quality
                            {"overall": 0.85}  # Hybrid quality
                        ]
                        
                        # Initialize the API
                        api = HybridLLMDiffusionAPI()
                        
                        # Compare the approaches
                        results = api.compare_approaches(
                            specification="Write a test function",
                            language="python"
                        )
                        
                        # Check the structure of the results
                        assert "diffusion" in results
                        assert "hybrid" in results
                        assert "comparison" in results
                        
                        # Check diffusion results
                        assert "code" in results["diffusion"]
                        assert "quality" in results["diffusion"]
                        assert "time" in results["diffusion"]
                        
                        # Check hybrid results
                        assert "code" in results["hybrid"]
                        assert "quality" in results["hybrid"]
                        assert "time" in results["hybrid"]
                        
                        # Check comparison metrics
                        assert "quality_improvement_percent" in results["comparison"]
                        assert results["comparison"]["quality_improvement_percent"] == pytest.approx(21.43, 0.01)
                        
                        # Verify the calls
                        mock_diffusion_api.generate_code.assert_called_once()
                        mock_generator.generate.assert_called_once()
                        assert mock_evaluate.call_count == 2
    
    def test_factory_function(self):
        """Test the factory function for creating the API."""
        config = {"test_config": True}
        
        # Patch the API class
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionAPI") as mock_api_class:
            # Call the factory function
            api = create_hybrid_llm_diffusion_api(config)
            
            # Verify that the API class was instantiated with the config
            mock_api_class.assert_called_once_with(config)