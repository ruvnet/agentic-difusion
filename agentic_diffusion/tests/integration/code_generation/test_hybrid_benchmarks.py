"""
Integration tests for the hybrid approach benchmarking.
"""

import pytest
from unittest.mock import MagicMock, patch
import os
import tempfile
import json

from agentic_diffusion.examples.code_generation.benchmark_hybrid_approach import (
    load_dataset, save_results, SAMPLE_DATASET
)


@pytest.fixture
def mock_diffusion_api():
    """Create a mock diffusion API."""
    api = MagicMock()
    api.generate_code.return_value = ("def test():\n    pass", {})
    api.evaluate_code.return_value = {"overall": 0.7, "syntax": 0.8, "relevance": 0.6}
    return api


@pytest.fixture
def mock_hybrid_api():
    """Create a mock hybrid API."""
    api = MagicMock()
    api.generate_code.return_value = (
        "def test():\n    return 42", 
        {"quality": {"quality_improvement_percentage": 15.0}}
    )
    api.evaluate_code.return_value = {"overall": 0.85, "syntax": 0.9, "relevance": 0.8}
    return api


class TestHybridBenchmarks:
    """Tests for the hybrid benchmarking functionality."""
    
    def test_load_dataset_default(self):
        """Test that the default dataset is loaded when no path is provided."""
        dataset = load_dataset()
        assert dataset == SAMPLE_DATASET
        assert len(dataset) == 5  # Check the number of samples in the default dataset
    
    def test_load_dataset_custom(self):
        """Test loading a custom dataset from a file."""
        # Create a temporary dataset file
        custom_dataset = [{"id": "custom", "prompt": "Custom prompt", "language": "python"}]
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp:
            json.dump(custom_dataset, temp)
            temp_path = temp.name
        
        try:
            # Load the custom dataset
            dataset = load_dataset(temp_path)
            assert dataset == custom_dataset
            assert len(dataset) == 1
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_load_dataset_invalid(self):
        """Test that the default dataset is used when the file is invalid."""
        # Create a temporary file with invalid JSON
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as temp:
            temp.write("This is not valid JSON")
            temp_path = temp.name
        
        try:
            # Should fall back to default dataset
            dataset = load_dataset(temp_path)
            assert dataset == SAMPLE_DATASET
        finally:
            # Clean up
            os.unlink(temp_path)
    
    def test_save_results(self):
        """Test saving benchmark results to a file."""
        results = {
            "timestamp": "2025-05-11T05:00:00",
            "samples": 5,
            "results": {
                "diffusion": {
                    "avg_quality": 0.7,
                    "avg_time": 1.5
                },
                "hybrid": {
                    "avg_quality": 0.85,
                    "avg_time": 2.0,
                    "avg_improvement": 15.0
                }
            },
            "comparison": {
                "quality_improvement_percent": 21.4,
                "hybrid_vs_diffusion_time_ratio": 1.33,
                "sample_improvements": {
                    "1": 15.0,
                    "2": 20.0,
                    "3": 25.0,
                    "4": 18.0,
                    "5": 28.0
                }
            }
        }
        
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the results
            save_results(results, temp_dir)
            
            # Check that a file was created
            files = os.listdir(temp_dir)
            assert len(files) == 1
            assert files[0].startswith("benchmark_results_")
            assert files[0].endswith(".json")
            
            # Check the file content
            with open(os.path.join(temp_dir, files[0]), 'r') as f:
                saved_results = json.load(f)
                assert saved_results == results
    
    @patch('agentic_diffusion.examples.code_generation.benchmark_hybrid_approach.create_code_generation_api')
    @patch('agentic_diffusion.examples.code_generation.benchmark_hybrid_approach.create_hybrid_llm_diffusion_api')
    def test_benchmark_comparison(self, mock_hybrid_factory, mock_diffusion_factory, mock_diffusion_api, mock_hybrid_api):
        """Test the benchmark comparison between diffusion and hybrid approaches."""
        from agentic_diffusion.examples.code_generation.benchmark_hybrid_approach import main
        
        # Set up mocks
        mock_diffusion_factory.return_value = mock_diffusion_api
        mock_hybrid_factory.return_value = mock_hybrid_api
        
        # Mock command line arguments
        with patch('sys.argv', ['benchmark_hybrid_approach.py', '--output-dir', 'test_output']):
            with patch('agentic_diffusion.examples.code_generation.benchmark_hybrid_approach.save_results') as mock_save:
                with patch('agentic_diffusion.examples.code_generation.benchmark_hybrid_approach.print_summary') as mock_print:
                    # Run the benchmark
                    main()
                    
                    # Check that APIs were called
                    assert mock_diffusion_api.generate_code.call_count == len(SAMPLE_DATASET)
                    assert mock_diffusion_api.evaluate_code.call_count == len(SAMPLE_DATASET)
                    assert mock_hybrid_api.generate_code.call_count == len(SAMPLE_DATASET)
                    assert mock_hybrid_api.evaluate_code.call_count == len(SAMPLE_DATASET)
                    
                    # Check that results were saved
                    assert mock_save.call_count == 1
                    saved_results = mock_save.call_args[0][0]
                    
                    # Verify the structure of results
                    assert "results" in saved_results
                    assert "diffusion" in saved_results["results"]
                    assert "hybrid" in saved_results["results"]
                    assert "comparison" in saved_results
                    
                    # Check quality improvement calculation
                    assert "quality_improvement_percent" in saved_results["comparison"]
                    improvement = saved_results["comparison"]["quality_improvement_percent"]
                    assert improvement > 0  # Should show improvement