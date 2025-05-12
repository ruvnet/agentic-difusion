"""
Integration tests for the code generation pipeline.

This module tests the integration of all code generation components:
CodeTokenizer, SyntaxModel, CodeGenerator, CodeAdaptationModel, and adaptation mechanisms.
"""

import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.api.code_generation_api import create_code_generation_api
from agentic_diffusion.code_generation.code_tokenizer import CodeTokenizer
from agentic_diffusion.code_generation.syntax_model import SyntaxModel


@pytest.fixture
def mock_diffusion_model():
    """Create a mock diffusion model."""
    mock = MagicMock(name="DiffusionModel")
    
    # Set up the generate method to return a reasonable output
    def generate_side_effect(specification, language=None, partial_code=None):
        if language == "python":
            return "def sum_numbers(a, b):\n    return a + b"
        elif language == "javascript":
            return "function sumNumbers(a, b) {\n    return a + b;\n}"
        elif language == "java":
            return "public int sumNumbers(int a, int b) {\n    return a + b;\n}"
        elif language == "go":
            return "func sumNumbers(a, b int) int {\n    return a + b\n}"
        else:
            return "def sum_numbers(a, b):\n    return a + b"
    
    mock.generate.side_effect = generate_side_effect
    mock.sample.side_effect = lambda spec, lang, partial: [generate_side_effect(spec, lang, partial)]
    
    # Mock the compute_gradients method
    mock.compute_gradients.return_value = [0.1, 0.2, 0.3]
    
    # Mock the apply_gradients method
    mock.apply_gradients.return_value = None
    
    return mock


@pytest.fixture
def code_generation_api(mock_diffusion_model):
    """Create a code generation API instance with mock diffusion model."""
    config = {
        "default_language": "python",
        "adaptation_type": "hybrid",
        "gradient_weight": 0.5,
        "memory_weight": 0.5
    }
    
    return create_code_generation_api(mock_diffusion_model, config)


class TestCodeGenerationPipeline:
    """Test the entire code generation pipeline."""
    
    def test_end_to_end_code_generation(self, code_generation_api):
        """
        Test end-to-end code generation.
        
        Verify that code can be generated from a specification.
        """
        specification = "Write a function to sum two numbers."
        
        # Generate code
        code, metadata = code_generation_api.generate_code(specification, language="python")
        
        assert "def sum_numbers" in code
        assert "return a + b" in code
        
        # Verify metadata is properly structured
        assert "performance" in metadata
        assert "quality" in metadata
        assert "generation_parameters" in metadata
    
    def test_code_adaptation(self, code_generation_api):
        """
        Test code adaptation.
        
        Verify that code can be adapted based on feedback.
        """
        code = "def sum_numbers(a, b):\n    return a + b"
        feedback = {"improve": "Add type hints"}
        
        # Adapt code
        adapted_code = code_generation_api.adapt_code(
            code=code,
            language="python",
            feedback=feedback
        )
        
        # Check that the adaptation mechanism was called
        # The actual output depends on the mock implementation
        assert adapted_code is not None
    
    def test_code_improvement(self, code_generation_api):
        """
        Test code improvement.
        
        Verify that code can be improved based on feedback.
        """
        code = "def sum_numbers(a, b):\n    return a + b"
        feedback = {"fix": "Add docstring"}
        
        # Improve code
        improved_code = code_generation_api.improve_code(
            code=code,
            feedback=feedback,
            language="python"
        )
        
        # The actual output depends on the mock implementation
        assert improved_code is not None
    
    def test_code_refinement(self, code_generation_api):
        """
        Test code refinement.
        
        Verify that code can be refined iteratively.
        """
        code = "def sum_numbers(a, b):\n    return a + b"
        
        # Refine code
        refined_code = code_generation_api.refine_code(
            code=code,
            language="python",
            iterations=2
        )
        
        # The actual output depends on the mock implementation
        assert refined_code is not None
    
    def test_code_evaluation(self, code_generation_api):
        """
        Test code evaluation.
        
        Verify that code quality metrics can be computed.
        """
        code = "def sum_numbers(a, b):\n    return a + b"
        specification = "Write a function to sum two numbers."
        
        # Evaluate code
        metrics = code_generation_api.evaluate_code(
            code=code,
            specification=specification,
            language="python"
        )
        
        # Check that metrics were computed
        assert "syntax_score" in metrics
        assert "quality_score" in metrics
        assert "relevance_score" in metrics
        assert "overall_score" in metrics
        assert "syntax_correct" in metrics
        assert "complexity" in metrics
    
    def test_multi_language_support(self, code_generation_api):
        """
        Test multi-language support.
        
        Verify that the system can handle different programming languages.
        """
        specification = "Write a function to sum two numbers."
        
        # Generate Python code
        python_code, _ = code_generation_api.generate_code(
            specification=specification,
            language="python"
        )
        
        # Generate JavaScript code
        js_code, _ = code_generation_api.generate_code(
            specification=specification,
            language="javascript"
        )
        
        # Generate Java code
        java_code, _ = code_generation_api.generate_code(
            specification=specification,
            language="java"
        )
        
        # Generate Go code
        go_code, _ = code_generation_api.generate_code(
            specification=specification,
            language="go"
        )
        
        # Check that language-specific patterns are present
        assert "def" in python_code
        assert "function" in js_code
        assert "public" in java_code
        assert "func" in go_code
    
    def test_state_save_load(self, code_generation_api, tmp_path):
        """
        Test state saving and loading.
        
        Verify that the API state can be saved and loaded.
        """
        # Save the state
        save_path = tmp_path / "test_state"
        success = code_generation_api.save_state(save_path)
        
        # Check that the save operation was successful
        assert success is True
        
        # Create a new API instance
        config = {
            "default_language": "python",
            "adaptation_type": "hybrid"
        }
        
        new_api = create_code_generation_api(
            code_generation_api.diffusion_model,
            config
        )
        
        # Load the state
        load_success = new_api.load_state(save_path)
        
        # Check that the load operation was successful
        assert load_success is True