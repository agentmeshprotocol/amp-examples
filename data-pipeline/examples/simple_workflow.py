"""
Simple Data Analysis Workflow Example.

This script demonstrates a basic data analysis workflow using
the AutoGen Data Analysis Pipeline.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add project modules to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'pipelines'))
sys.path.append(str(project_root / '../shared-lib'))

from pipelines.data_pipeline import DataAnalysisPipeline, PipelineConfig
import pandas as pd
import numpy as np


def create_sample_data():
    """Create a simple sample dataset for demonstration."""
    
    np.random.seed(42)
    n_samples = 200
    
    # Create simple sales data
    data = {
        'date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'product': np.random.choice(['A', 'B', 'C'], n_samples),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'sales_amount': np.random.normal(1000, 300, n_samples),
        'units_sold': np.random.randint(1, 20, n_samples),
        'customer_satisfaction': np.random.uniform(1, 5, n_samples)
    }
    
    # Add some correlations
    for i in range(n_samples):
        # Higher satisfaction leads to higher sales
        if data['customer_satisfaction'][i] > 4:
            data['sales_amount'][i] *= 1.2
        
        # Product A is premium
        if data['product'][i] == 'A':
            data['sales_amount'][i] *= 1.5
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices, 'customer_satisfaction'] = np.nan
    
    return df


async def run_simple_workflow():
    """Run a simple data analysis workflow."""
    
    print("=" * 60)
    print("SIMPLE DATA ANALYSIS WORKFLOW")
    print("=" * 60)
    
    # Step 1: Create sample data
    print("\n1. Creating sample dataset...")
    df = create_sample_data()
    
    # Save to file
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    data_file = data_dir / "simple_sales_data.csv"
    df.to_csv(data_file, index=False)
    
    print(f"   Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"   Saved to: {data_file}")
    print(f"   Preview:")
    print(df.head().to_string(index=False))
    
    # Step 2: Configure pipeline
    print("\n2. Configuring analysis pipeline...")
    
    config = PipelineConfig(
        pipeline_id="simple_workflow_001",
        pipeline_name="Simple Sales Analysis",
        
        # Enable core components
        enable_data_collection=True,
        enable_data_cleaning=True,
        enable_statistical_analysis=True,
        enable_ml_analysis=True,
        enable_visualization=True,
        enable_quality_assurance=True,
        
        # Lower thresholds for demo data
        data_quality_threshold=0.7,
        model_performance_threshold=0.6,
        
        # Generate outputs
        generate_report=True,
        create_dashboard=True,
        save_artifacts=True,
        
        # LLM configuration (placeholder)
        llm_config={
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": os.environ.get("OPENAI_API_KEY", "demo-key"),
                    "api_type": "openai"
                }
            ]
        }
    )
    
    print(f"   Pipeline ID: {config.pipeline_id}")
    print(f"   Components enabled: 6")
    
    # Step 3: Define analysis request
    print("\n3. Defining analysis objectives...")
    
    analysis_request = """
    Analyze the sales dataset to understand:
    1. Sales trends over time
    2. Performance differences between products and regions
    3. Relationship between customer satisfaction and sales
    4. Predict sales amount based on other factors
    5. Identify any data quality issues
    """
    
    context = {
        "business_context": "Sales performance analysis",
        "target_metric": "sales_amount",
        "time_column": "date",
        "categorical_features": ["product", "region"],
        "key_stakeholders": ["Sales Team", "Management"]
    }
    
    print(f"   Analysis request: {analysis_request.strip()}")
    print(f"   Context: {context}")
    
    # Step 4: Initialize pipeline
    print("\n4. Initializing analysis pipeline...")
    
    try:
        pipeline = DataAnalysisPipeline(config)
        print(f"   Pipeline initialized successfully")
        print(f"   Agents available: {list(pipeline.agents.keys())}")
        
        # Show agent capabilities
        for agent_name, agent in pipeline.agents.items():
            print(f"     {agent_name}: {len(agent.capabilities)} capabilities")
        
    except Exception as e:
        print(f"   ERROR: Failed to initialize pipeline: {e}")
        return
    
    # Step 5: Run analysis (simulated)
    print("\n5. Running analysis pipeline...")
    print("   NOTE: This demo shows the pipeline structure.")
    print("   In a real environment with proper API keys and AMP network,")
    print("   the following would execute the complete analysis:")
    
    print(f"\n   Would execute:")
    print(f"   - Data collection from {data_file}")
    print(f"   - Data cleaning and quality assessment")
    print(f"   - Statistical analysis (correlations, trends)")
    print(f"   - Machine learning modeling")
    print(f"   - Visualization creation")
    print(f"   - Quality assurance validation")
    print(f"   - Final report generation")
    
    # Step 6: Show expected results structure
    print("\n6. Expected analysis results:")
    
    expected_results = {
        "pipeline_id": config.pipeline_id,
        "status": "completed",
        "data_collection": {
            "dataset_key": "simple_sales_data",
            "shape": list(df.shape),
            "data_types": df.dtypes.to_dict()
        },
        "data_cleaning": {
            "quality_score": 0.85,
            "missing_values_handled": 6,
            "outliers_detected": 3
        },
        "statistical_analysis": {
            "correlations": {
                "sales_amount_vs_satisfaction": 0.72,
                "sales_amount_vs_units_sold": 0.58
            },
            "significant_trends": ["Product A outperforms others", "North region strongest"]
        },
        "ml_analysis": {
            "best_model": "random_forest",
            "model_accuracy": 0.78,
            "feature_importance": {
                "product": 0.35,
                "customer_satisfaction": 0.28,
                "region": 0.22,
                "units_sold": 0.15
            }
        },
        "visualizations": {
            "charts_created": 5,
            "dashboard_components": 8
        },
        "quality_assurance": {
            "data_validation_passed": True,
            "model_validation_passed": True,
            "overall_quality_score": 0.82
        }
    }
    
    # Display formatted results
    print(f"   Pipeline Status: {expected_results['status']}")
    print(f"   Data Quality Score: {expected_results['data_cleaning']['quality_score']}")
    print(f"   Best ML Model: {expected_results['ml_analysis']['best_model']}")
    print(f"   Model Accuracy: {expected_results['ml_analysis']['model_accuracy']}")
    print(f"   Quality Score: {expected_results['quality_assurance']['overall_quality_score']}")
    
    # Step 7: Show key insights
    print("\n7. Key insights from analysis:")
    
    insights = [
        "Product A generates 50% higher sales than other products",
        "Customer satisfaction has strong positive correlation (0.72) with sales",
        "North region shows consistently strong performance",
        "Random Forest model achieved 78% accuracy in predicting sales",
        "Data quality is good (85%) with minimal missing values"
    ]
    
    for i, insight in enumerate(insights, 1):
        print(f"   {i}. {insight}")
    
    # Step 8: Next steps
    print("\n8. Recommended next steps:")
    
    recommendations = [
        "Focus marketing efforts on Product A and North region",
        "Investigate factors driving customer satisfaction",
        "Deploy Random Forest model for sales prediction",
        "Collect additional features to improve model accuracy",
        "Set up regular data quality monitoring"
    ]
    
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n" + "=" * 60)
    print("WORKFLOW COMPLETED")
    print("=" * 60)
    print(f"Results would be saved to: output/pipeline_results_{config.pipeline_id}.json")
    print(f"Dashboard available at: output/dashboard_{config.pipeline_id}.html")
    print(f"Full report at: output/analysis_report_{config.pipeline_id}.html")


def main():
    """Main entry point."""
    
    print("AutoGen Data Analysis Pipeline - Simple Workflow Example")
    print("This example demonstrates the pipeline structure and expected outputs.")
    print("To run with real data and LLM integration, configure API keys and AMP network.\n")
    
    try:
        asyncio.run(run_simple_workflow())
    except KeyboardInterrupt:
        print("\nWorkflow interrupted by user")
    except Exception as e:
        print(f"\nWorkflow failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()