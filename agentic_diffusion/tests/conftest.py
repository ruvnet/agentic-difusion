"""
Pytest configuration file for Agentic Diffusion test suite.

This file contains shared fixtures and configuration for all tests
in the Agentic Diffusion project, ensuring 90% test coverage as required.
"""

import os
import sys
from pathlib import Path
import pytest
from typing import Dict, Any

# Add the project root to sys.path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# Global fixtures available to all tests
@pytest.fixture(scope="session")
def project_root_path() -> Path:
    """Return the project root path."""
    return project_root


@pytest.fixture(scope="session")
def test_data_path() -> Path:
    """Return the path to test data directory."""
    return project_root / "agentic_diffusion" / "tests" / "data"


@pytest.fixture(scope="session")
def ensure_test_data_dir(test_data_path: Path) -> Path:
    """Ensure the test data directory exists."""
    os.makedirs(test_data_path, exist_ok=True)
    return test_data_path


@pytest.fixture
def default_config() -> Dict[str, Any]:
    """
    Return a default configuration for testing.
    
    This configuration provides sensible defaults for components
    to facilitate testing without requiring full configurations.
    """
    return {
        "diffusion": {
            "model_type": "test_model",
            "num_diffusion_steps": 10,
            "device": "cpu",
            "precision": "float32",
            "noise_schedule": {
                "type": "linear",
                "beta_start": 0.0001,
                "beta_end": 0.02,
            },
            "denoiser": {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 2,
                "num_heads": 4,
                "dropout": 0.1,
            },
            "trajectory_buffer": {
                "capacity": 100,
            },
        },
        "adaptation": {
            "mechanism": "gradient",
            "learning_rate": 1e-4,
            "num_steps": 10,
            "batch_size": 4,
            "patience": 5,
        },
        "code_generation": {
            "languages": ["python", "javascript", "java", "go"],
            "max_sequence_length": 256,
            "token_vocabulary_size": 10000,
            "syntax_guidance_weight": 0.5,
            "prompt_guidance_weight": 0.8,
        },
        "testing": {
            "coverage_target": 0.9,
            "parallel": True,
            "test_types": ["unit", "integration", "system"],
        },
    }


# Mock fixtures for components
@pytest.fixture
def mock_diffusion_model():
    """Return a mock diffusion model for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.generate.return_value = (None, [])
    mock.train_step.return_value = {"loss": 0.1}
    mock.adapt.return_value = True
    
    return mock


@pytest.fixture
def mock_code_tokenizer():
    """Return a mock code tokenizer for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.encode.return_value = [1, 2, 3, 4, 5]
    mock.decode.return_value = "def test(): pass"
    mock.encode_language.return_value = 1
    
    return mock


@pytest.fixture
def mock_syntax_parser():
    """Return a mock syntax parser for testing."""
    from unittest.mock import MagicMock
    
    mock = MagicMock()
    mock.check_syntax.return_value = (True, None)
    mock.fix_syntax.return_value = "def test(): pass"
    
    return mock


# Test data fixtures
@pytest.fixture
def sample_python_code() -> str:
    """Return a sample Python code snippet for testing."""
    return """
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n-1) + fibonacci(n-2)

def main():
    result = fibonacci(10)
    print(f"Fibonacci(10) = {result}")

if __name__ == "__main__":
    main()
"""


@pytest.fixture(scope="session")
def adaptdiffuser_api_client():
    """
    Fixture for AdaptDiffuser API test client.
    Uses FastAPI TestClient to provide a test client for integration tests.
    """
    from fastapi.testclient import TestClient
    from agentic_diffusion.api import adapt_diffuser_api
    app = getattr(adapt_diffuser_api, "app", None)
    if app is None:
        raise RuntimeError("AdaptDiffuser API app not found in adapt_diffuser_api.py")
    client = TestClient(app)
    yield client
    """Return a sample JavaScript code snippet for testing."""
    return """
function fibonacci(n) {
    if (n <= 0) {
        return 0;
    } else if (n === 1) {
        return 1;
    } else {
        return fibonacci(n-1) + fibonacci(n-2);
    }
}

function main() {
    const result = fibonacci(10);
    console.log(`Fibonacci(10) = ${result}`);
}

main();
"""