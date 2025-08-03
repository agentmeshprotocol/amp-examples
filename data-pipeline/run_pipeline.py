"""
Data Analysis Pipeline Runner.

Main application for running the complete data analysis pipeline
with AutoGen agents and AMP protocol integration.
"""

import asyncio
import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'pipelines'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../shared-lib'))

from pipelines.data_pipeline import DataAnalysisPipeline, PipelineConfig
from amp_types import TransportType


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """Set up logging configuration."""
    
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[]
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # Reduce noise from external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


async def run_sample_analysis():
    """Run a sample data analysis workflow."""
    
    # Create sample configuration
    config = PipelineConfig(
        pipeline_id="sample_analysis_001",
        pipeline_name="Sample Data Analysis Pipeline",
        
        # Enable all components
        enable_data_collection=True,
        enable_data_cleaning=True,
        enable_statistical_analysis=True,
        enable_ml_analysis=True,
        enable_visualization=True,
        enable_quality_assurance=True,
        
        # Quality settings
        data_quality_threshold=0.8,
        model_performance_threshold=0.7,
        
        # Output settings
        generate_report=True,
        create_dashboard=True,
        save_artifacts=True,
        
        # LLM configuration (placeholder - update with your API key)
        llm_config={
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": os.environ.get("OPENAI_API_KEY", "your-openai-api-key"),
                    "api_type": "openai"
                }
            ]
        }
    )
    
    # Create sample dataset
    import pandas as pd
    import numpy as np
    
    # Generate synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'education_years': np.random.randint(8, 20, n_samples),
        'experience_years': np.random.randint(0, 40, n_samples),
        'satisfaction_score': np.random.uniform(1, 10, n_samples),
        'department': np.random.choice(['Engineering', 'Sales', 'Marketing', 'HR'], n_samples),
        'performance_rating': np.random.choice(['Low', 'Medium', 'High'], n_samples)
    }
    
    # Introduce some missing values and outliers for cleaning demo
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    data['income'][missing_indices] = np.nan
    
    # Add some outliers
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    data['income'][outlier_indices] = np.random.uniform(200000, 500000, len(outlier_indices))
    
    sample_df = pd.DataFrame(data)
    
    # Save sample dataset
    os.makedirs("data", exist_ok=True)
    sample_file = "data/sample_employee_data.csv"
    sample_df.to_csv(sample_file, index=False)
    
    print(f"Created sample dataset: {sample_file}")
    print(f"Dataset shape: {sample_df.shape}")
    print(f"Dataset preview:\n{sample_df.head()}")
    
    # Initialize and run pipeline
    print("\n" + "="*80)
    print("STARTING DATA ANALYSIS PIPELINE")
    print("="*80)
    
    pipeline = DataAnalysisPipeline(config)
    
    analysis_request = """
    Analyze the employee dataset to understand:
    1. Relationship between age, education, experience and income
    2. Performance patterns across different departments
    3. Factors that predict employee satisfaction
    4. Build a model to predict performance rating based on other features
    """
    
    try:
        # Run the complete pipeline
        results = await pipeline.run_pipeline(
            data_source=sample_file,
            analysis_request=analysis_request,
            context={
                "business_context": "Employee performance and satisfaction analysis",
                "target_metric": "performance_rating",
                "key_stakeholders": ["HR", "Management"]
            }
        )
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETED")
        print("="*80)
        
        # Display results summary
        print(f"Pipeline Status: {results['status']}")
        print(f"Dataset processed: {results.get('dataset_key', 'N/A')}")
        
        if 'statistical_analysis' in results:
            stats = results['statistical_analysis']
            if 'descriptive_analysis' in stats:
                insights = stats['descriptive_analysis'].get('summary_insights', [])
                print(f"\nKey Statistical Insights:")
                for insight in insights[:3]:
                    print(f"- {insight}")
        
        if 'ml_analysis' in results:
            ml = results['ml_analysis']
            if 'model_training' in ml:
                best_model = ml['model_training'].get('best_model', 'Unknown')
                performance = ml['model_training'].get('performance_metrics', {})
                print(f"\nML Analysis Results:")
                print(f"- Best model: {best_model}")
                if 'accuracy' in performance:
                    print(f"- Model accuracy: {performance['accuracy']:.3f}")
                elif 'r2_score' in performance:
                    print(f"- Model R²: {performance['r2_score']:.3f}")
        
        if 'quality_assurance' in results:
            qa = results['quality_assurance']
            if 'data_validation' in qa:
                quality_score = qa['data_validation'].get('quality_score', 0)
                print(f"\nQuality Assurance:")
                print(f"- Data quality score: {quality_score:.3f}")
        
        # Save results
        pipeline.save_pipeline_results("output")
        print(f"\nPipeline results saved to: output/pipeline_results_{config.pipeline_id}.json")
        
        # Display execution log
        print(f"\nExecution Log ({len(results['execution_log'])} steps):")
        for log_entry in results['execution_log'][-5:]:  # Last 5 steps
            print(f"  [{log_entry['step']}] {log_entry['message']}")
        
        return results
        
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        
        # Get pipeline status for debugging
        status = await pipeline.get_pipeline_status()
        print(f"Pipeline status: {status}")
        
        raise


async def run_custom_analysis(data_source: str, analysis_request: str, 
                            config_overrides: Dict[str, Any] = None):
    """Run custom data analysis with user-provided data and requirements."""
    
    # Default configuration
    config = PipelineConfig(
        pipeline_id=f"custom_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}",
        pipeline_name="Custom Data Analysis Pipeline",
        
        # Default settings
        enable_data_collection=True,
        enable_data_cleaning=True,
        enable_statistical_analysis=True,
        enable_ml_analysis=True,
        enable_visualization=True,
        enable_quality_assurance=True,
        
        llm_config={
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": os.environ.get("OPENAI_API_KEY", "your-openai-api-key"),
                    "api_type": "openai"
                }
            ]
        }
    )
    
    # Apply configuration overrides
    if config_overrides:
        for key, value in config_overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
    
    print(f"Starting custom analysis pipeline: {config.pipeline_id}")
    print(f"Data source: {data_source}")
    print(f"Analysis request: {analysis_request}")
    
    pipeline = DataAnalysisPipeline(config)
    
    try:
        results = await pipeline.run_pipeline(
            data_source=data_source,
            analysis_request=analysis_request
        )
        
        print(f"\nCustom analysis completed successfully!")
        print(f"Results saved to: output/pipeline_results_{config.pipeline_id}.json")
        
        return results
        
    except Exception as e:
        print(f"Custom analysis failed: {e}")
        raise


def main():
    """Main entry point for the pipeline runner."""
    
    parser = argparse.ArgumentParser(
        description="Run AutoGen Data Analysis Pipeline with AMP Protocol",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sample analysis
  python run_pipeline.py --sample
  
  # Analyze custom CSV file
  python run_pipeline.py --data data/my_data.csv --request "Predict sales performance"
  
  # Custom analysis with specific configuration
  python run_pipeline.py --data data/my_data.csv --request "Customer segmentation" --no-ml --no-qa
        """
    )
    
    # Data source options
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--sample", action="store_true",
                           help="Run with sample synthetic dataset")
    data_group.add_argument("--data", type=str,
                           help="Path to data file (CSV, JSON, Excel) or database connection")
    
    # Analysis configuration
    parser.add_argument("--request", type=str,
                       help="Natural language description of analysis requirements")
    
    # Pipeline component toggles
    parser.add_argument("--no-collection", action="store_true",
                       help="Disable data collection (use pre-loaded data)")
    parser.add_argument("--no-cleaning", action="store_true",
                       help="Disable data cleaning")
    parser.add_argument("--no-stats", action="store_true",
                       help="Disable statistical analysis")
    parser.add_argument("--no-ml", action="store_true",
                       help="Disable machine learning analysis")
    parser.add_argument("--no-viz", action="store_true",
                       help="Disable visualization")
    parser.add_argument("--no-qa", action="store_true",
                       help="Disable quality assurance")
    
    # Output options
    parser.add_argument("--no-report", action="store_true",
                       help="Disable final report generation")
    parser.add_argument("--no-dashboard", action="store_true",
                       help="Disable dashboard creation")
    parser.add_argument("--output-dir", type=str, default="output",
                       help="Output directory for results")
    
    # Logging options
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       default="INFO", help="Logging level")
    parser.add_argument("--log-file", type=str,
                       help="Log file path (in addition to console)")
    
    # Advanced options
    parser.add_argument("--pipeline-id", type=str,
                       help="Custom pipeline ID")
    parser.add_argument("--amp-endpoint", type=str, default="http://localhost:8000",
                       help="AMP network endpoint")
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.log_level, args.log_file)
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Build configuration overrides
    config_overrides = {
        "enable_data_collection": not args.no_collection,
        "enable_data_cleaning": not args.no_cleaning,
        "enable_statistical_analysis": not args.no_stats,
        "enable_ml_analysis": not args.no_ml,
        "enable_visualization": not args.no_viz,
        "enable_quality_assurance": not args.no_qa,
        "generate_report": not args.no_report,
        "create_dashboard": not args.no_dashboard,
        "amp_endpoint": args.amp_endpoint
    }
    
    if args.pipeline_id:
        config_overrides["pipeline_id"] = args.pipeline_id
    
    try:
        if args.sample:
            # Run sample analysis
            print("Running sample analysis workflow...")
            asyncio.run(run_sample_analysis())
        
        else:
            # Run custom analysis
            if not args.request:
                args.request = "Perform comprehensive data analysis including statistical analysis and predictive modeling"
            
            print("Running custom analysis workflow...")
            asyncio.run(run_custom_analysis(
                data_source=args.data,
                analysis_request=args.request,
                config_overrides=config_overrides
            ))
        
        print("\n" + "="*80)
        print("PIPELINE EXECUTION SUCCESSFUL")
        print("="*80)
        
    except KeyboardInterrupt:
        print("\nPipeline execution interrupted by user")
        sys.exit(1)
    
    except Exception as e:
        print(f"\nPipeline execution failed: {e}")
        if args.log_level == "DEBUG":
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()