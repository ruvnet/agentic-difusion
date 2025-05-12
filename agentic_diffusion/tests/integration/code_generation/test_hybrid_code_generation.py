"""
Integration tests for the hybrid LLM + diffusion code generation approach.
"""

import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.code_generation.hybrid_llm_diffusion_generator import HybridLLMDiffusionGenerator
from agentic_diffusion.api.hybrid_llm_diffusion_api import HybridLLMDiffusionAPI, create_hybrid_llm_diffusion_api


@pytest.fixture
def mock_code_generator():
    """Create a mock code generator."""
    generator = MagicMock()
    generator.generate_code.return_value = "def test_function():\n    return 42"
    return generator


@pytest.fixture
def hybrid_generator(mock_code_generator):
    """Create a hybrid generator with a mock code generator."""
    with patch("agentic_diffusion.code_generation.hybrid_llm_diffusion_generator.CodeGenerator", 
               return_value=mock_code_generator):
        yield HybridLLMDiffusionGenerator(
            llm_provider="mock",
            refinement_iterations=2
        )


@pytest.fixture
def hybrid_api(hybrid_generator):
    """Create a hybrid API with a mock generator."""
    with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionGenerator", 
               return_value=hybrid_generator):
        api = HybridLLMDiffusionAPI(config={
            "llm_provider": "mock",
            "refinement_iterations": 2
        })
        yield api


class TestHybridCodeGeneration:
    """Integration tests for hybrid code generation."""
    
    def test_api_generate_code(self, hybrid_api):
        """Test that the API can generate code."""
        # Mock the generator's generate method
        with patch.object(
            hybrid_api.generator, "generate", 
            return_value=("def api_test():\n    return 'success'", {"syntax_score": 1.0})
        ):
            code, metadata = hybrid_api.generate_code(
                "Write a function called api_test that returns 'success'",
                "python"
            )
            
            # Verify the results
            assert code == "def api_test():\n    return 'success'"
            assert "performance" in metadata
            assert "quality" in metadata
            assert "syntax_score" in metadata["quality"]
    
    def test_evaluate_code(self, hybrid_api):
        """Test that the API can evaluate code quality."""
        # Mock the reward evaluations
        with patch.object(hybrid_api.syntax_reward, "evaluate", return_value=1.0):
            with patch.object(hybrid_api.quality_reward, "evaluate", return_value=0.8):
                with patch.object(hybrid_api.relevance_reward, "evaluate", return_value=0.9):
                    metrics = hybrid_api.evaluate_code(
                        "def test():\n    return 42",
                        "python",
                        "Write a function that returns 42"
                    )
                    
                    # Verify metrics
                    assert metrics["syntax"] == 1.0
                    assert metrics["quality"] == 0.8
                    assert metrics["relevance"] == 0.9
                    assert "overall" in metrics
    
    def test_full_pipeline_integration(self, hybrid_api):
        """Test the full code generation pipeline."""
        # Mock _generate_with_llm to return initial code
        with patch.object(
            hybrid_api.generator, "_generate_with_llm",
            return_value="def initial():\n    # TODO: implement"
        ):
            # Mock _refine_with_diffusion to return improved code
            with patch.object(
                hybrid_api.generator, "_refine_with_diffusion",
                return_value="def refined():\n    return 'refined result'"
            ):
                # Mock quality metrics calculation
                with patch.object(
                    hybrid_api.generator, "_compute_quality_metrics",
                    side_effect=[
                        {"syntax_score": 1.0, "overall_quality": 0.9},  # Final code metrics
                        {"syntax_score": 0.7, "overall_quality": 0.6}   # Initial code metrics
                    ]
                ):
                    # Generate code
                    code, metadata = hybrid_api.generate_code(
                        "Write a function that returns a refined result",
                        "python"
                    )
                    
                    # Verify the code
                    assert code == "def refined():\n    return 'refined result'"
                    
                    # Verify the quality improvement metrics
                    assert "quality" in metadata
                    assert "quality_improvement" in metadata["quality"]
                    assert metadata["quality"]["quality_improvement"] > 0
                    
                    # Verify timing information
                    assert "timing" in metadata
                    assert "llm_generation_time" in metadata["timing"]
                    assert "diffusion_refinement_time" in metadata["timing"]
    
    def test_factory_function(self):
        """Test the factory function for creating the API."""
        with patch("agentic_diffusion.api.hybrid_llm_diffusion_api.HybridLLMDiffusionAPI") as mock_api_class:
            mock_api_class.return_value = "mock_api_instance"
            
            # Call the factory function
            api = create_hybrid_llm_diffusion_api({"test_config": True})
            
            # Verify that the API class was instantiated with the config
            mock_api_class.assert_called_once_with({"test_config": True})
            assert api == "mock_api_instance"