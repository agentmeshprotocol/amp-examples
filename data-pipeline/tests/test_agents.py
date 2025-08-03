"""
Test suite for AutoGen AMP agents.

This module contains unit tests for individual agents
and their capabilities.
"""

import pytest
import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

# Add project modules to path
test_dir = Path(__file__).parent
project_root = test_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'agents'))
sys.path.append(str(project_root / '../shared-lib'))

from agents.data_collector import DataCollectorAgent
from agents.data_cleaner import DataCleanerAgent
from agents.statistical_analyst import StatisticalAnalystAgent
from agents.ml_analyst import MLAnalystAgent
from agents.visualization_agent import VisualizationAgent
from agents.quality_assurance import QualityAssuranceAgent

from amp_client import AMPClientConfig
from amp_types import TransportType


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'id': range(1, n_samples + 1),
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(2, 1.5, n_samples),
        'category': np.random.choice(['A', 'B', 'C'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(n_samples, size=5, replace=False)
    df.loc[missing_indices, 'feature1'] = np.nan
    
    return df


@pytest.fixture
def amp_config():
    """Create test AMP configuration."""
    return AMPClientConfig(
        agent_id="test_agent",
        agent_name="Test Agent",
        framework="autogen",
        version="1.0.0",
        transport_type=TransportType.HTTP,
        endpoint="http://localhost:8000",
        log_level="WARNING"  # Reduce noise in tests
    )


@pytest.fixture
def llm_config():
    """Create test LLM configuration."""
    return {
        "config_list": [
            {
                "model": "gpt-4",
                "api_key": "test-key",
                "api_type": "openai"
            }
        ]
    }


class TestDataCollectorAgent:
    """Test cases for Data Collector Agent."""
    
    def test_agent_initialization(self, amp_config, llm_config):
        """Test agent initialization."""
        agent = DataCollectorAgent(amp_config, llm_config)
        
        assert agent.name == "DataCollector"
        assert len(agent.capabilities) == 4
        assert "data-ingestion-file" in agent.capabilities
        assert "data-ingestion-database" in agent.capabilities
        assert "data-ingestion-api" in agent.capabilities
        assert "data-validation-source" in agent.capabilities
    
    def test_conversation_handling(self, amp_config, llm_config):
        """Test conversation message processing."""
        agent = DataCollectorAgent(amp_config, llm_config)
        
        # Test data collection request
        response = agent._process_conversation_message("collect data from file", None, [])
        assert "collect data" in response.lower()
        
        # Test status request
        response = agent._process_conversation_message("status", None, [])
        assert "collection metrics" in response.lower()
    
    def test_file_type_detection(self, amp_config, llm_config):
        """Test file type detection."""
        agent = DataCollectorAgent(amp_config, llm_config)
        
        assert agent._detect_file_type("data.csv") == "csv"
        assert agent._detect_file_type("data.json") == "json"
        assert agent._detect_file_type("data.xlsx") == "excel"
        assert agent._detect_file_type("data.parquet") == "parquet"
    
    def test_artifact_storage(self, amp_config, llm_config, sample_data):
        """Test artifact storage and retrieval."""
        agent = DataCollectorAgent(amp_config, llm_config)
        
        # Store artifact
        agent.store_artifact("test_data", sample_data, {"test": True})
        
        # Check artifact exists
        assert "test_data" in agent.list_artifacts()
        
        # Retrieve artifact
        retrieved = agent.get_artifact("test_data")
        pd.testing.assert_frame_equal(retrieved, sample_data)


class TestDataCleanerAgent:
    """Test cases for Data Cleaner Agent."""
    
    def test_agent_initialization(self, amp_config, llm_config):
        """Test agent initialization."""
        agent = DataCleanerAgent(amp_config, llm_config)
        
        assert agent.name == "DataCleaner"
        assert len(agent.capabilities) == 5
        assert "data-cleaning-missing-values" in agent.capabilities
        assert "data-cleaning-outliers" in agent.capabilities
    
    def test_missing_value_analysis(self, amp_config, llm_config, sample_data):
        """Test missing value analysis."""
        agent = DataCleanerAgent(amp_config, llm_config)
        
        # Store sample data
        agent.store_artifact("test_data", sample_data)
        
        # Analyze missing values
        missing_summary = agent._analyze_missing_values(sample_data)
        
        assert "_overall" in missing_summary
        assert missing_summary["feature1"]["missing_count"] == 5
        assert missing_summary["feature2"]["missing_count"] == 0
    
    def test_imputation_methods(self, amp_config, llm_config):
        """Test different imputation methods."""
        agent = DataCleanerAgent(amp_config, llm_config)
        
        # Create series with missing values
        series = pd.Series([1, 2, np.nan, 4, 5])
        
        # Test mean imputation
        imputed = agent._impute_column(series, "mean")
        assert not imputed.isnull().any()
        assert imputed.iloc[2] == 3.0  # Mean of [1,2,4,5]
        
        # Test median imputation
        imputed = agent._impute_column(series, "median")
        assert imputed.iloc[2] == 2.5  # Median of [1,2,4,5]


class TestStatisticalAnalystAgent:
    """Test cases for Statistical Analyst Agent."""
    
    def test_agent_initialization(self, amp_config, llm_config):
        """Test agent initialization."""
        agent = StatisticalAnalystAgent(amp_config, llm_config)
        
        assert agent.name == "StatisticalAnalyst"
        assert len(agent.capabilities) == 5
        assert "statistical-descriptive-analysis" in agent.capabilities
        assert "statistical-correlation-analysis" in agent.capabilities
    
    def test_distribution_analysis(self, amp_config, llm_config):
        """Test distribution analysis."""
        agent = StatisticalAnalystAgent(amp_config, llm_config)
        
        # Create normal distribution
        series = pd.Series(np.random.normal(0, 1, 100))
        
        analysis = agent._analyze_distribution(series)
        
        assert "mean" in analysis
        assert "std" in analysis
        assert "skewness" in analysis
        assert "kurtosis" in analysis
        assert abs(analysis["mean"]) < 0.5  # Should be close to 0
    
    def test_correlation_calculation(self, amp_config, llm_config, sample_data):
        """Test correlation calculation."""
        agent = StatisticalAnalystAgent(amp_config, llm_config)
        
        # Calculate correlations
        numeric_data = sample_data[['feature1', 'feature2']].dropna()
        p_values = agent._calculate_correlation_pvalues(numeric_data, "pearson")
        
        assert p_values.shape == (2, 2)
        assert p_values.loc['feature1', 'feature1'] == 0.0  # Diagonal should be 0


class TestMLAnalystAgent:
    """Test cases for ML Analyst Agent."""
    
    def test_agent_initialization(self, amp_config, llm_config):
        """Test agent initialization."""
        agent = MLAnalystAgent(amp_config, llm_config)
        
        assert agent.name == "MLAnalyst"
        assert len(agent.capabilities) == 5
        assert "ml-feature-engineering" in agent.capabilities
        assert "ml-model-training" in agent.capabilities
    
    def test_feature_selection(self, amp_config, llm_config, sample_data):
        """Test feature selection methods."""
        agent = MLAnalystAgent(amp_config, llm_config)
        
        X = sample_data[['feature1', 'feature2']].fillna(0)
        y = sample_data['target']
        
        # Test feature selection
        selected_features, scores = agent._select_features(X, y, "importance", 2, "target")
        
        assert len(selected_features) <= 2
        assert isinstance(scores, dict)
        assert len(scores) == len(X.columns)
    
    def test_algorithm_availability(self, amp_config, llm_config):
        """Test algorithm availability."""
        agent = MLAnalystAgent(amp_config, llm_config)
        
        # Check classification algorithms
        assert "random_forest" in agent.classification_algorithms
        assert "logistic_regression" in agent.classification_algorithms
        
        # Check regression algorithms
        assert "random_forest" in agent.regression_algorithms
        assert "linear_regression" in agent.regression_algorithms


class TestVisualizationAgent:
    """Test cases for Visualization Agent."""
    
    def test_agent_initialization(self, amp_config, llm_config):
        """Test agent initialization."""
        agent = VisualizationAgent(amp_config, llm_config)
        
        assert agent.name == "VisualizationAgent"
        assert len(agent.capabilities) == 5
        assert "visualization-statistical-plots" in agent.capabilities
        assert "visualization-dashboard" in agent.capabilities
    
    def test_visualization_config(self, amp_config, llm_config):
        """Test visualization configuration."""
        viz_config = {
            "default_figure_size": (12, 8),
            "default_color_palette": "viridis",
            "interactive_default": False
        }
        
        agent = VisualizationAgent(amp_config, llm_config, viz_config)
        
        assert agent.viz_config["default_figure_size"] == (12, 8)
        assert agent.viz_config["default_color_palette"] == "viridis"
        assert agent.viz_config["interactive_default"] == False
    
    @patch('matplotlib.pyplot.figure')
    def test_plot_creation(self, mock_figure, amp_config, llm_config, sample_data):
        """Test basic plot creation functionality."""
        agent = VisualizationAgent(amp_config, llm_config)
        
        # Test plot data preparation
        plot_data, plot_config, insights = agent._create_statistical_plot(
            sample_data, "histogram", ["feature1"], None, None, "Test Plot", False
        )
        
        assert plot_config["title"] == "Test Plot"
        assert plot_config["interactive"] == False
        assert isinstance(insights, list)


class TestQualityAssuranceAgent:
    """Test cases for Quality Assurance Agent."""
    
    def test_agent_initialization(self, amp_config, llm_config):
        """Test agent initialization."""
        agent = QualityAssuranceAgent(amp_config, llm_config)
        
        assert agent.name == "QualityAssurance"
        assert len(agent.capabilities) == 5
        assert "qa-data-validation" in agent.capabilities
        assert "qa-model-validation" in agent.capabilities
    
    def test_data_completeness_check(self, amp_config, llm_config, sample_data):
        """Test data completeness checking."""
        agent = QualityAssuranceAgent(amp_config, llm_config)
        
        thresholds = {"missing_data_threshold": 0.1}
        result = agent._check_data_completeness(sample_data, thresholds)
        
        assert "passed" in result
        assert "score" in result
        assert "overall_missing_ratio" in result
        assert "column_completeness" in result
    
    def test_data_consistency_check(self, amp_config, llm_config, sample_data):
        """Test data consistency checking."""
        agent = QualityAssuranceAgent(amp_config, llm_config)
        
        validation_rules = {
            "feature1_type": "float64",
            "feature2_min": -10,
            "feature2_max": 10
        }
        
        result = agent._check_data_consistency(sample_data, validation_rules)
        
        assert "passed" in result
        assert "score" in result
        assert "consistency_checks" in result
    
    def test_qa_standards(self, amp_config, llm_config):
        """Test QA standards configuration."""
        qa_config = {
            "data_quality_threshold": 0.9,
            "model_performance_threshold": 0.8
        }
        
        agent = QualityAssuranceAgent(amp_config, llm_config, qa_config)
        
        assert agent.qa_config["data_quality_threshold"] == 0.9
        assert agent.qa_config["model_performance_threshold"] == 0.8


class TestAgentIntegration:
    """Integration tests for agent interactions."""
    
    def test_artifact_sharing(self, amp_config, llm_config, sample_data):
        """Test artifact sharing between agents."""
        collector = DataCollectorAgent(amp_config, llm_config)
        cleaner = DataCleanerAgent(amp_config, llm_config)
        
        # Collector stores data
        collector.store_artifact("shared_data", sample_data, {"source": "test"})
        
        # Cleaner can access the data (simulated)
        # In real scenario, this would happen through AMP protocol
        shared_data = collector.get_artifact("shared_data")
        cleaner.store_artifact("received_data", shared_data, {"received_from": "collector"})
        
        assert "received_data" in cleaner.list_artifacts()
        retrieved = cleaner.get_artifact("received_data")
        pd.testing.assert_frame_equal(retrieved, sample_data)
    
    def test_conversation_routing(self, amp_config, llm_config):
        """Test conversation message routing."""
        agents = {
            "collector": DataCollectorAgent(amp_config, llm_config),
            "cleaner": DataCleanerAgent(amp_config, llm_config),
            "analyst": StatisticalAnalystAgent(amp_config, llm_config)
        }
        
        # Test different message types route to appropriate responses
        responses = {}
        
        responses["collect"] = agents["collector"]._process_conversation_message("collect data", None, [])
        responses["clean"] = agents["cleaner"]._process_conversation_message("clean data", None, [])
        responses["analyze"] = agents["analyst"]._process_conversation_message("analyze correlation", None, [])
        
        # Each agent should provide relevant response
        assert "collect" in responses["collect"].lower() or "ingest" in responses["collect"].lower()
        assert "clean" in responses["clean"].lower() or "preprocess" in responses["clean"].lower()
        assert "correlation" in responses["analyze"].lower() or "statistical" in responses["analyze"].lower()


@pytest.mark.asyncio
class TestAsyncOperations:
    """Test asynchronous operations (mocked)."""
    
    async def test_capability_invocation_structure(self, amp_config, llm_config, sample_data):
        """Test capability invocation structure (without actual AMP calls)."""
        agent = DataCollectorAgent(amp_config, llm_config)
        
        # Store test data for capability testing
        agent.store_artifact("test_dataset", sample_data)
        
        # Test capability parameter validation
        capability = agent.capabilities["data-ingestion-file"]
        
        # Check required parameters
        required_params = capability.input_schema["required"]
        assert "file_path" in required_params
        
        # Check input schema structure
        properties = capability.input_schema["properties"]
        assert "file_path" in properties
        assert "file_type" in properties


def test_agent_metrics_tracking(amp_config, llm_config, sample_data):
    """Test agent metrics tracking."""
    agent = DataCollectorAgent(amp_config, llm_config)
    
    # Initial metrics should be zero
    assert agent.collection_metrics["files_processed"] == 0
    assert agent.collection_metrics["total_rows_collected"] == 0
    
    # Simulate some activity
    agent.collection_metrics["files_processed"] += 1
    agent.collection_metrics["total_rows_collected"] += len(sample_data)
    
    assert agent.collection_metrics["files_processed"] == 1
    assert agent.collection_metrics["total_rows_collected"] == len(sample_data)


def test_agent_configuration_validation(amp_config, llm_config):
    """Test agent configuration validation."""
    # Test invalid configuration
    invalid_llm_config = {"invalid": "config"}
    
    # Should still initialize but may have limited functionality
    agent = DataCollectorAgent(amp_config, invalid_llm_config)
    assert agent.name == "DataCollector"
    
    # Test custom configuration
    custom_config = {
        "default_missing_strategy": "custom_strategy",
        "outlier_threshold": 5.0
    }
    
    cleaner = DataCleanerAgent(amp_config, llm_config, custom_config)
    assert cleaner.cleaning_config["outlier_threshold"] == 5.0


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])