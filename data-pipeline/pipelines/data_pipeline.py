"""
Data Analysis Pipeline.

Complete data analysis pipeline that orchestrates multiple AutoGen agents
to perform end-to-end data analysis workflows.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from pathlib import Path

# Add agents to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../agents'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared-lib'))

from agents.data_collector import DataCollectorAgent
from agents.data_cleaner import DataCleanerAgent  
from agents.statistical_analyst import StatisticalAnalystAgent
from agents.ml_analyst import MLAnalystAgent
from agents.visualization_agent import VisualizationAgent
from agents.quality_assurance import QualityAssuranceAgent
from agents.base_agent import DataAnalysisOrchestrator

from amp_client import AMPClientConfig
from amp_types import TransportType


@dataclass
class PipelineConfig:
    """Configuration for data analysis pipeline."""
    
    # Pipeline settings
    pipeline_id: str
    pipeline_name: str = "Data Analysis Pipeline"
    max_execution_time: int = 3600  # 1 hour
    
    # Agent configurations
    enable_data_collection: bool = True
    enable_data_cleaning: bool = True
    enable_statistical_analysis: bool = True
    enable_ml_analysis: bool = True
    enable_visualization: bool = True
    enable_quality_assurance: bool = True
    
    # Quality thresholds
    data_quality_threshold: float = 0.8
    model_performance_threshold: float = 0.7
    
    # Output settings
    generate_report: bool = True
    create_dashboard: bool = True
    save_artifacts: bool = True
    
    # AMP network settings
    amp_endpoint: str = "http://localhost:8000"
    transport_type: TransportType = TransportType.HTTP
    
    # LLM configuration
    llm_config: Dict[str, Any] = None


class DataAnalysisPipeline:
    """
    Complete data analysis pipeline using AutoGen agents with AMP protocol.
    
    Orchestrates multiple specialized agents to perform:
    1. Data collection and ingestion
    2. Data cleaning and preprocessing
    3. Statistical analysis
    4. Machine learning modeling
    5. Visualization and reporting
    6. Quality assurance validation
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize data analysis pipeline.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.pipeline_id = config.pipeline_id
        
        # Initialize agents
        self.agents: Dict[str, Any] = {}
        self.orchestrator: Optional[DataAnalysisOrchestrator] = None
        
        # Pipeline state
        self.is_running = False
        self.current_step = ""
        self.pipeline_results: Dict[str, Any] = {}
        self.execution_log: List[Dict[str, Any]] = []
        
        # Set up logging
        self.logger = logging.getLogger(f"amp.pipeline.{config.pipeline_id}")
        self.logger.setLevel(logging.INFO)
        
        # Default LLM config if not provided
        if config.llm_config is None:
            config.llm_config = {
                "config_list": [
                    {
                        "model": "gpt-4",
                        "api_key": os.environ.get("OPENAI_API_KEY", "your-api-key-here"),
                        "api_type": "openai"
                    }
                ]
            }
        
        # Initialize agents based on configuration
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize all enabled agents."""
        
        base_amp_config = {
            "agent_name": f"Pipeline-{self.pipeline_id}",
            "framework": "autogen",
            "version": "1.0.0",
            "transport_type": self.config.transport_type,
            "endpoint": self.config.amp_endpoint,
            "auto_reconnect": True,
            "log_level": "INFO"
        }
        
        agents_list = []
        
        # Data Collector Agent
        if self.config.enable_data_collection:
            collector_config = AMPClientConfig(
                agent_id=f"{self.pipeline_id}-data-collector",
                **base_amp_config
            )
            self.agents["data_collector"] = DataCollectorAgent(
                amp_config=collector_config,
                llm_config=self.config.llm_config
            )
            agents_list.append(self.agents["data_collector"])
        
        # Data Cleaner Agent
        if self.config.enable_data_cleaning:
            cleaner_config = AMPClientConfig(
                agent_id=f"{self.pipeline_id}-data-cleaner",
                **base_amp_config
            )
            self.agents["data_cleaner"] = DataCleanerAgent(
                amp_config=cleaner_config,
                llm_config=self.config.llm_config
            )
            agents_list.append(self.agents["data_cleaner"])
        
        # Statistical Analyst Agent
        if self.config.enable_statistical_analysis:
            stats_config = AMPClientConfig(
                agent_id=f"{self.pipeline_id}-statistical-analyst",
                **base_amp_config
            )
            self.agents["statistical_analyst"] = StatisticalAnalystAgent(
                amp_config=stats_config,
                llm_config=self.config.llm_config
            )
            agents_list.append(self.agents["statistical_analyst"])
        
        # ML Analyst Agent
        if self.config.enable_ml_analysis:
            ml_config = AMPClientConfig(
                agent_id=f"{self.pipeline_id}-ml-analyst",
                **base_amp_config
            )
            self.agents["ml_analyst"] = MLAnalystAgent(
                amp_config=ml_config,
                llm_config=self.config.llm_config
            )
            agents_list.append(self.agents["ml_analyst"])
        
        # Visualization Agent
        if self.config.enable_visualization:
            viz_config = AMPClientConfig(
                agent_id=f"{self.pipeline_id}-visualization",
                **base_amp_config
            )
            self.agents["visualization"] = VisualizationAgent(
                amp_config=viz_config,
                llm_config=self.config.llm_config
            )
            agents_list.append(self.agents["visualization"])
        
        # Quality Assurance Agent
        if self.config.enable_quality_assurance:
            qa_config = AMPClientConfig(
                agent_id=f"{self.pipeline_id}-quality-assurance",
                **base_amp_config
            )
            self.agents["quality_assurance"] = QualityAssuranceAgent(
                amp_config=qa_config,
                llm_config=self.config.llm_config
            )
            agents_list.append(self.agents["quality_assurance"])
        
        # Create orchestrator
        if agents_list:
            self.orchestrator = DataAnalysisOrchestrator(agents_list)
        
        self.logger.info(f"Initialized {len(agents_list)} agents for pipeline {self.pipeline_id}")
    
    async def run_pipeline(self, data_source: Union[str, Dict[str, Any]], 
                          analysis_request: str, 
                          context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run the complete data analysis pipeline.
        
        Args:
            data_source: Data source (file path, database connection, API endpoint)
            analysis_request: Natural language description of analysis requirements
            context: Additional context for the analysis
            
        Returns:
            Pipeline results including all analysis outputs
        """
        if self.is_running:
            raise RuntimeError("Pipeline is already running")
        
        self.is_running = True
        self.current_step = "initializing"
        self.pipeline_results = {"pipeline_id": self.pipeline_id}
        self.execution_log = []
        
        try:
            self.logger.info(f"Starting pipeline {self.pipeline_id}")
            
            # Connect all agents
            await self._connect_agents()
            
            # Step 1: Data Collection
            dataset_key = await self._collect_data(data_source)
            
            # Step 2: Data Cleaning
            cleaned_dataset_key = await self._clean_data(dataset_key)
            
            # Step 3: Statistical Analysis
            stats_results = await self._perform_statistical_analysis(cleaned_dataset_key, analysis_request)
            
            # Step 4: Machine Learning Analysis
            ml_results = await self._perform_ml_analysis(cleaned_dataset_key, analysis_request)
            
            # Step 5: Visualization
            viz_results = await self._create_visualizations(cleaned_dataset_key, stats_results, ml_results)
            
            # Step 6: Quality Assurance
            qa_results = await self._perform_quality_assurance(cleaned_dataset_key, ml_results)
            
            # Step 7: Generate Final Report
            final_report = await self._generate_final_report()
            
            # Compile results
            self.pipeline_results.update({
                "status": "completed",
                "dataset_key": dataset_key,
                "cleaned_dataset_key": cleaned_dataset_key,
                "statistical_analysis": stats_results,
                "ml_analysis": ml_results,
                "visualizations": viz_results,
                "quality_assurance": qa_results,
                "final_report": final_report,
                "execution_log": self.execution_log
            })
            
            self.logger.info(f"Pipeline {self.pipeline_id} completed successfully")
            return self.pipeline_results
            
        except Exception as e:
            self.logger.error(f"Pipeline {self.pipeline_id} failed: {e}")
            self.pipeline_results.update({
                "status": "failed",
                "error": str(e),
                "execution_log": self.execution_log
            })
            raise
        
        finally:
            self.is_running = False
            self.current_step = "completed"
            
            # Disconnect agents
            await self._disconnect_agents()
    
    async def _connect_agents(self):
        """Connect all agents to AMP network."""
        self.current_step = "connecting_agents"
        self._log_step("Connecting agents to AMP network")
        
        if self.orchestrator:
            success = await self.orchestrator.connect_all_agents()
            if not success:
                raise RuntimeError("Failed to connect all agents to AMP network")
        
        self._log_step("All agents connected successfully")
    
    async def _disconnect_agents(self):
        """Disconnect all agents from AMP network."""
        self.current_step = "disconnecting_agents"
        
        if self.orchestrator:
            await self.orchestrator.disconnect_all_agents()
        
        self._log_step("All agents disconnected")
    
    async def _collect_data(self, data_source: Union[str, Dict[str, Any]]) -> str:
        """Collect data using Data Collector Agent."""
        self.current_step = "data_collection"
        self._log_step(f"Starting data collection from: {data_source}")
        
        if "data_collector" not in self.agents:
            raise RuntimeError("Data Collector Agent not enabled")
        
        collector = self.agents["data_collector"]
        
        # Determine data source type and prepare parameters
        if isinstance(data_source, str):
            if data_source.startswith("http"):
                # API endpoint
                result = await collector.invoke_capability(
                    None, "data-ingestion-api", 
                    {"url": data_source}
                )
            elif os.path.exists(data_source):
                # File path
                result = await collector.invoke_capability(
                    None, "data-ingestion-file",
                    {"file_path": data_source}
                )
            else:
                # Database connection string
                result = await collector.invoke_capability(
                    None, "data-ingestion-database",
                    {"connection_string": data_source, "query": "SELECT * FROM main_table LIMIT 1000"}
                )
        else:
            # Dictionary with specific parameters
            if "url" in data_source:
                result = await collector.invoke_capability(
                    None, "data-ingestion-api", data_source
                )
            elif "file_path" in data_source:
                result = await collector.invoke_capability(
                    None, "data-ingestion-file", data_source
                )
            elif "connection_string" in data_source:
                result = await collector.invoke_capability(
                    None, "data-ingestion-database", data_source
                )
            else:
                raise ValueError("Invalid data source format")
        
        dataset_key = result["dataset_key"]
        self.pipeline_results["data_collection"] = result
        
        self._log_step(f"Data collection completed. Dataset: {dataset_key}")
        return dataset_key
    
    async def _clean_data(self, dataset_key: str) -> str:
        """Clean data using Data Cleaner Agent."""
        self.current_step = "data_cleaning"
        self._log_step(f"Starting data cleaning for dataset: {dataset_key}")
        
        if "data_cleaner" not in self.agents:
            return dataset_key  # Skip cleaning if not enabled
        
        cleaner = self.agents["data_cleaner"]
        
        # Perform comprehensive data cleaning
        cleaning_steps = []
        
        # Handle missing values
        missing_result = await cleaner.invoke_capability(
            None, "data-cleaning-missing-values",
            {"dataset_key": dataset_key, "strategy": "median"}
        )
        cleaning_steps.append(missing_result)
        current_key = missing_result["cleaned_dataset_key"]
        
        # Handle outliers
        outlier_result = await cleaner.invoke_capability(
            None, "data-cleaning-outliers",
            {"dataset_key": current_key, "method": "iqr", "action": "cap"}
        )
        cleaning_steps.append(outlier_result)
        current_key = outlier_result["cleaned_dataset_key"]
        
        # Remove duplicates
        duplicate_result = await cleaner.invoke_capability(
            None, "data-cleaning-duplicates",
            {"dataset_key": current_key, "keep": "first"}
        )
        cleaning_steps.append(duplicate_result)
        cleaned_dataset_key = duplicate_result["cleaned_dataset_key"]
        
        # Data quality assessment
        quality_result = await cleaner.invoke_capability(
            None, "data-quality-assessment",
            {"dataset_key": cleaned_dataset_key}
        )
        
        self.pipeline_results["data_cleaning"] = {
            "cleaning_steps": cleaning_steps,
            "quality_assessment": quality_result,
            "final_dataset_key": cleaned_dataset_key
        }
        
        self._log_step(f"Data cleaning completed. Quality score: {quality_result['quality_score']:.3f}")
        return cleaned_dataset_key
    
    async def _perform_statistical_analysis(self, dataset_key: str, analysis_request: str) -> Dict[str, Any]:
        """Perform statistical analysis using Statistical Analyst Agent."""
        self.current_step = "statistical_analysis"
        self._log_step(f"Starting statistical analysis for dataset: {dataset_key}")
        
        if "statistical_analyst" not in self.agents:
            return {}
        
        analyst = self.agents["statistical_analyst"]
        results = {}
        
        # Descriptive statistics
        descriptive_result = await analyst.invoke_capability(
            None, "statistical-descriptive-analysis",
            {"dataset_key": dataset_key, "include_distribution": True}
        )
        results["descriptive_analysis"] = descriptive_result
        
        # Correlation analysis
        correlation_result = await analyst.invoke_capability(
            None, "statistical-correlation-analysis",
            {"dataset_key": dataset_key, "method": "pearson", "significance_test": True}
        )
        results["correlation_analysis"] = correlation_result
        
        self._log_step("Statistical analysis completed")
        return results
    
    async def _perform_ml_analysis(self, dataset_key: str, analysis_request: str) -> Dict[str, Any]:
        """Perform machine learning analysis using ML Analyst Agent."""
        self.current_step = "ml_analysis"
        self._log_step(f"Starting ML analysis for dataset: {dataset_key}")
        
        if "ml_analyst" not in self.agents:
            return {}
        
        ml_analyst = self.agents["ml_analyst"]
        results = {}
        
        # Determine target variable (simplified - would need more sophisticated logic)
        # For demo purposes, assume last column is target
        dataset = ml_analyst.get_artifact(dataset_key)
        if dataset is not None and hasattr(dataset, 'columns'):
            target_variable = dataset.columns[-1]
            
            # Feature engineering
            feature_result = await ml_analyst.invoke_capability(
                None, "ml-feature-engineering",
                {
                    "dataset_key": dataset_key,
                    "target_variable": target_variable,
                    "feature_operations": ["scaling", "encoding"],
                    "selection_method": "importance",
                    "n_features": 10
                }
            )
            results["feature_engineering"] = feature_result
            
            # Model training
            # Determine if classification or regression
            if dataset[target_variable].dtype in ['object', 'category'] or dataset[target_variable].nunique() < 10:
                model_type = "classification"
                algorithms = ["random_forest", "logistic_regression", "gradient_boosting"]
            else:
                model_type = "regression"
                algorithms = ["random_forest", "linear_regression", "gradient_boosting"]
            
            model_result = await ml_analyst.invoke_capability(
                None, "ml-model-training",
                {
                    "dataset_key": feature_result["transformed_dataset_key"],
                    "target_variable": target_variable,
                    "model_type": model_type,
                    "algorithms": algorithms,
                    "cross_validation": True,
                    "hyperparameter_tuning": True
                }
            )
            results["model_training"] = model_result
            
            # Feature importance
            importance_result = await ml_analyst.invoke_capability(
                None, "ml-feature-importance",
                {
                    "model_key": model_result["trained_models_key"],
                    "dataset_key": feature_result["transformed_dataset_key"],
                    "importance_method": "built-in",
                    "top_n_features": 10
                }
            )
            results["feature_importance"] = importance_result
        
        self._log_step("ML analysis completed")
        return results
    
    async def _create_visualizations(self, dataset_key: str, stats_results: Dict[str, Any], 
                                   ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create visualizations using Visualization Agent."""
        self.current_step = "visualization"
        self._log_step("Starting visualization creation")
        
        if "visualization" not in self.agents:
            return {}
        
        viz_agent = self.agents["visualization"]
        results = {}
        
        # Statistical plots
        hist_result = await viz_agent.invoke_capability(
            None, "visualization-statistical-plots",
            {
                "dataset_key": dataset_key,
                "plot_type": "histogram",
                "interactive": True
            }
        )
        results["histogram"] = hist_result
        
        # Correlation visualization
        if stats_results.get("correlation_analysis"):
            corr_result = await viz_agent.invoke_capability(
                None, "visualization-correlation-analysis",
                {
                    "dataset_key": dataset_key,
                    "correlation_method": "pearson",
                    "plot_type": "heatmap",
                    "annotation": True
                }
            )
            results["correlation_heatmap"] = corr_result
        
        # Model performance visualization
        if ml_results.get("model_training"):
            model_viz_result = await viz_agent.invoke_capability(
                None, "visualization-model-performance",
                {
                    "model_results_key": ml_results["model_training"]["trained_models_key"],
                    "plot_type": "feature_importance",
                    "interactive": True
                }
            )
            results["model_performance"] = model_viz_result
        
        # Dashboard
        if self.config.create_dashboard:
            dashboard_result = await viz_agent.invoke_capability(
                None, "visualization-dashboard",
                {
                    "dataset_key": dataset_key,
                    "dashboard_type": "exploratory",
                    "layout": "grid",
                    "title": f"Analysis Dashboard - {self.pipeline_id}"
                }
            )
            results["dashboard"] = dashboard_result
        
        self._log_step("Visualization creation completed")
        return results
    
    async def _perform_quality_assurance(self, dataset_key: str, ml_results: Dict[str, Any]) -> Dict[str, Any]:
        """Perform quality assurance using QA Agent."""
        self.current_step = "quality_assurance"
        self._log_step("Starting quality assurance checks")
        
        if "quality_assurance" not in self.agents:
            return {}
        
        qa_agent = self.agents["quality_assurance"]
        results = {}
        
        # Data validation
        data_validation = await qa_agent.invoke_capability(
            None, "qa-data-validation",
            {
                "dataset_key": dataset_key,
                "quality_thresholds": {
                    "missing_data_threshold": self.config.data_quality_threshold,
                    "outlier_threshold": 0.05
                }
            }
        )
        results["data_validation"] = data_validation
        
        # Model validation
        if ml_results.get("model_training"):
            model_validation = await qa_agent.invoke_capability(
                None, "qa-model-validation",
                {
                    "model_key": ml_results["model_training"]["trained_models_key"],
                    "performance_thresholds": {
                        "model_performance_threshold": self.config.model_performance_threshold
                    }
                }
            )
            results["model_validation"] = model_validation
        
        self._log_step("Quality assurance completed")
        return results
    
    async def _generate_final_report(self) -> Dict[str, Any]:
        """Generate final comprehensive report."""
        self.current_step = "report_generation"
        self._log_step("Generating final report")
        
        if not self.config.generate_report or "visualization" not in self.agents:
            return {}
        
        viz_agent = self.agents["visualization"]
        
        # Collect all analysis artifacts
        analysis_keys = []
        for result_type in ["data_collection", "data_cleaning", "statistical_analysis", "ml_analysis"]:
            if result_type in self.pipeline_results:
                analysis_keys.append(f"pipeline_{result_type}")
        
        # Generate comprehensive report
        report_result = await viz_agent.invoke_capability(
            None, "visualization-report",
            {
                "analysis_keys": analysis_keys,
                "report_type": "full_analysis",
                "include_recommendations": True,
                "format": "html"
            }
        )
        
        self._log_step("Final report generated")
        return report_result
    
    def _log_step(self, message: str):
        """Log a pipeline step."""
        log_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "step": self.current_step,
            "message": message
        }
        self.execution_log.append(log_entry)
        self.logger.info(f"[{self.current_step}] {message}")
    
    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        status = {
            "pipeline_id": self.pipeline_id,
            "is_running": self.is_running,
            "current_step": self.current_step,
            "agents_count": len(self.agents),
            "execution_log_entries": len(self.execution_log)
        }
        
        if self.orchestrator:
            agent_health = await self.orchestrator.health_check_all()
            status["agent_health"] = agent_health
        
        return status
    
    async def get_pipeline_metrics(self) -> Dict[str, Any]:
        """Get pipeline performance metrics."""
        metrics = {
            "pipeline_id": self.pipeline_id,
            "agent_metrics": {}
        }
        
        for agent_name, agent in self.agents.items():
            try:
                agent_metrics = await agent.get_metrics()
                metrics["agent_metrics"][agent_name] = agent_metrics
            except Exception as e:
                self.logger.warning(f"Failed to get metrics for {agent_name}: {e}")
        
        return metrics
    
    def save_pipeline_results(self, output_dir: str):
        """Save pipeline results to files."""
        if not self.config.save_artifacts:
            return
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline results as JSON
        import json
        results_file = output_path / f"pipeline_results_{self.pipeline_id}.json"
        with open(results_file, 'w') as f:
            # Convert results to JSON-serializable format
            serializable_results = self._make_json_serializable(self.pipeline_results)
            json.dump(serializable_results, f, indent=2)
        
        self.logger.info(f"Pipeline results saved to {results_file}")
    
    def _make_json_serializable(self, obj: Any) -> Any:
        """Convert object to JSON-serializable format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return str(obj) if not isinstance(obj, (str, int, float, bool, type(None))) else obj