"""
Unit tests for the QualityEnhancementMetric class.
"""

import pytest
from unittest.mock import MagicMock, patch

from agentic_diffusion.code_generation.rewards.quality_enhancement_metric import (
    QualityEnhancementMetric,
    measure_code_enhancement
)


class TestQualityEnhancementMetric:
    """Tests for the QualityEnhancementMetric class."""
    
    def test_initialization(self):
        """Test that the metric initializes with default components."""
        metric = QualityEnhancementMetric()
        assert metric.quality_reward is not None
        assert metric.relevance_reward is not None
        assert metric.syntax_reward is not None
        assert "syntax" in metric.weights
        assert "quality" in metric.weights
        assert "relevance" in metric.weights
    
    def test_custom_weights(self):
        """Test initialization with custom weights."""
        custom_weights = {
            "syntax": 0.5,
            "quality": 0.3,
            "relevance": 0.2
        }
        
        metric = QualityEnhancementMetric(weights=custom_weights)
        assert metric.weights == custom_weights
    
    def test_evaluate_code_quality(self):
        """Test the evaluate_code_quality method."""
        # Create mock reward evaluators
        mock_quality_reward = MagicMock()
        mock_relevance_reward = MagicMock()
        mock_syntax_reward = MagicMock()
        
        # Set return values
        mock_quality_reward.evaluate.return_value = 0.8
        mock_relevance_reward.evaluate.return_value = 0.7
        mock_syntax_reward.evaluate.return_value = 0.9
        
        # Create the metric with mock evaluators
        metric = QualityEnhancementMetric(
            quality_reward=mock_quality_reward,
            relevance_reward=mock_relevance_reward,
            syntax_reward=mock_syntax_reward,
            weights={"syntax": 0.4, "quality": 0.3, "relevance": 0.3}
        )
        
        # Test with specification
        metrics = metric.evaluate_code_quality(
            "def test(): pass",
            "python",
            "Write a test function"
        )
        
        # Check the individual metrics
        assert metrics["syntax"] == 0.9
        assert metrics["quality"] == 0.8
        assert metrics["relevance"] == 0.7
        
        # Check the overall score calculation: (0.9*0.4 + 0.8*0.3 + 0.7*0.3) / 1.0 = 0.81
        assert metrics["overall"] == 0.81
        
        # Check that evaluate methods were called
        mock_syntax_reward.evaluate.assert_called_once()
        mock_quality_reward.evaluate.assert_called_once()
        mock_relevance_reward.evaluate.assert_called_once()
    
    def test_evaluate_without_specification(self):
        """Test evaluation without a specification."""
        # Create mock reward evaluators
        mock_quality_reward = MagicMock()
        mock_relevance_reward = MagicMock()
        mock_syntax_reward = MagicMock()
        
        # Set return values
        mock_quality_reward.evaluate.return_value = 0.8
        mock_syntax_reward.evaluate.return_value = 0.9
        
        # Create the metric with mock evaluators
        metric = QualityEnhancementMetric(
            quality_reward=mock_quality_reward,
            relevance_reward=mock_relevance_reward,
            syntax_reward=mock_syntax_reward
        )
        
        # Test without specification
        metrics = metric.evaluate_code_quality("def test(): pass", "python")
        
        # Check metrics
        assert metrics["syntax"] == 0.9
        assert metrics["quality"] == 0.8
        assert "relevance" not in metrics
        
        # Check that relevance was not called
        mock_relevance_reward.evaluate.assert_not_called()
    
    def test_calculate_enhancement(self):
        """Test the calculation of enhancement between two code versions."""
        # Create a mock metric with controlled output
        metric = MagicMock()
        metric.evaluate_code_quality.side_effect = [
            # Initial metrics
            {"syntax": 0.7, "quality": 0.6, "overall": 0.65},
            # Enhanced metrics
            {"syntax": 0.9, "quality": 0.8, "overall": 0.85}
        ]
        
        # Patch the initialization
        with patch.object(QualityEnhancementMetric, 'evaluate_code_quality', 
                          side_effect=metric.evaluate_code_quality):
            test_metric = QualityEnhancementMetric()
            
            result = test_metric.calculate_enhancement(
                initial_code="def test(): pass",
                enhanced_code="def test():\n    return 'enhanced'",
                language="python",
                specification="Write a test function"
            )
            
            # Check the structure of the result
            assert "initial_metrics" in result
            assert "enhanced_metrics" in result
            assert "absolute_improvements" in result
            assert "improvement_percentages" in result
            assert "overall_improvement" in result
            assert "overall_improvement_percentage" in result
            
            # Check the improvement calculations
            assert result["absolute_improvements"]["syntax"] == 0.2
            assert result["absolute_improvements"]["quality"] == 0.2
            assert result["absolute_improvements"]["overall"] == 0.2
            
            # Check the improvement percentages
            assert result["improvement_percentages"]["syntax"] == pytest.approx(28.57, 0.01)
            assert result["improvement_percentages"]["quality"] == pytest.approx(33.33, 0.01)
            assert result["improvement_percentages"]["overall"] == pytest.approx(30.77, 0.01)
    
    def test_batch_evaluation(self):
        """Test batch evaluation of multiple code pairs."""
        # Create a metric with mocked enhancement calculation
        metric = MagicMock()
        metric.calculate_enhancement.side_effect = [
            {
                "overall_improvement": 0.2,
                "overall_improvement_percentage": 30.0
            },
            {
                "overall_improvement": 0.3,
                "overall_improvement_percentage": 40.0
            }
        ]
        
        # Patch the initialization
        with patch.object(QualityEnhancementMetric, 'calculate_enhancement', 
                          side_effect=metric.calculate_enhancement):
            test_metric = QualityEnhancementMetric()
            
            # Test batch evaluation
            code_pairs = [
                ("def test1(): pass", "def test1():\n    return 'enhanced'"),
                ("def test2(): pass", "def test2():\n    return 'enhanced'")
            ]
            
            result = test_metric.batch_evaluation(
                code_pairs=code_pairs,
                language="python",
                specifications=["Write test1", "Write test2"]
            )
            
            # Check the structure of the result
            assert "individual_enhancements" in result
            assert "average_improvement" in result
            assert "average_improvement_percentage" in result
            assert "sample_count" in result
            
            # Check the calculations
            assert result["sample_count"] == 2
            assert result["average_improvement"] == 0.25  # (0.2 + 0.3) / 2
            assert result["average_improvement_percentage"] == 35.0  # (30.0 + 40.0) / 2
    
    def test_utility_function(self):
        """Test the utility function for measuring code enhancement."""
        # Create a mock metric
        mock_metric = MagicMock()
        mock_metric.calculate_enhancement.return_value = {"overall_improvement": 0.2}
        
        # Patch the QualityEnhancementMetric class
        with patch('agentic_diffusion.code_generation.rewards.quality_enhancement_metric.QualityEnhancementMetric', 
                  return_value=mock_metric):
            result = measure_code_enhancement(
                initial_code="def test(): pass",
                enhanced_code="def test():\n    return 'enhanced'",
                language="python",
                specification="Write a test function"
            )
            
            # Check that the result is returned correctly
            assert result == {"overall_improvement": 0.2}
            
            # Check that calculate_enhancement was called with the correct parameters
            mock_metric.calculate_enhancement.assert_called_once_with(
                "def test(): pass",
                "def test():\n    return 'enhanced'",
                "python",
                "Write a test function"
            )