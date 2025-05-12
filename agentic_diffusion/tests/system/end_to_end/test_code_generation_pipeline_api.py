"""
System-level end-to-end tests for the unified CodeGenerationAPI pipeline.

This module verifies that the full code generation pipeline—including diffusion,
adaptation, recursive refinement, reward-driven improvement, and multi-language
support—is correctly orchestrated and exposed via the modular API.
"""

import os
import pytest
from unittest.mock import MagicMock

from agentic_diffusion.api.code_generation_api import create_code_generation_api

@pytest.fixture
def mock_diffusion_model():
    """Minimal diffusion model mock for API system tests."""
    mock = MagicMock(name="DiffusionModel")
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
    mock.compute_gradients.return_value = [0.1, 0.2, 0.3]
    mock.apply_gradients.return_value = None
    return mock

@pytest.fixture
def code_generation_api(mock_diffusion_model):
    config = {
        "default_language": "python",
        "adaptation_type": "hybrid",
        "gradient_weight": 0.5,
        "memory_weight": 0.5
    }
    return create_code_generation_api(mock_diffusion_model, config)

def test_end_to_end_code_generation_api(code_generation_api):
    """Test end-to-end code generation via the API."""
    specification = "Write a function to sum two numbers."
    code, metadata = code_generation_api.generate_code(specification, language="python")
    assert "def sum_numbers" in code
    assert "return a + b" in code
    
    # Verify metadata
    assert "performance" in metadata
    assert "quality" in metadata
    assert "generation_parameters" in metadata

def test_code_adaptation_api(code_generation_api):
    """Test code adaptation via the API."""
    code = "def sum_numbers(a, b):\n    return a + b"
    feedback = {"improve": "Add type hints"}
    adapted_code = code_generation_api.adapt_code(
        code=code,
        language="python",
        feedback=feedback
    )
    assert adapted_code is not None

def test_code_improvement_api(code_generation_api):
    """Test code improvement via the API."""
    code = "def sum_numbers(a, b):\n    return a + b"
    feedback = {"fix": "Add docstring"}
    improved_code = code_generation_api.improve_code(
        code=code,
        feedback=feedback,
        language="python"
    )
    assert improved_code is not None

def test_code_refinement_api(code_generation_api):
    """Test recursive code refinement via the API."""
    code = "def sum_numbers(a, b):\n    return a + b"
    refined_code = code_generation_api.refine_code(
        code=code,
        language="python",
        iterations=2
    )
    assert refined_code is not None

def test_code_evaluation_api(code_generation_api):
    """Test code evaluation via the API."""
    code = "def sum_numbers(a, b):\n    return a + b"
    specification = "Write a function to sum two numbers."
    metrics = code_generation_api.evaluate_code(
        code=code,
        specification=specification,
        language="python"
    )
    assert "syntax_score" in metrics
    assert "quality_score" in metrics
    assert "relevance_score" in metrics
    assert "overall_score" in metrics
    assert "syntax_correct" in metrics
    assert "complexity" in metrics

def test_multi_language_support_api(code_generation_api):
    """Test multi-language code generation via the API."""
    specification = "Write a function to sum two numbers."
    python_code, _ = code_generation_api.generate_code(
        specification=specification,
        language="python"
    )
    js_code, _ = code_generation_api.generate_code(
        specification=specification,
        language="javascript"
    )
    java_code, _ = code_generation_api.generate_code(
        specification=specification,
        language="java"
    )
    go_code, _ = code_generation_api.generate_code(
        specification=specification,
        language="go"
    )
    assert "def" in python_code
    assert "function" in js_code
    assert "public" in java_code
    assert "func" in go_code

def test_state_save_load_api(code_generation_api, tmp_path):
    """Test API state save and load functionality."""
    save_path = tmp_path / "test_state"
    success = code_generation_api.save_state(str(save_path))
    assert success is True

    config = {
        "default_language": "python",
        "adaptation_type": "hybrid"
    }
    from agentic_diffusion.api.code_generation_api import create_code_generation_api
    new_api = create_code_generation_api(
        code_generation_api.diffusion_model,
        config
    )
    load_success = new_api.load_state(str(save_path))
    assert load_success is True