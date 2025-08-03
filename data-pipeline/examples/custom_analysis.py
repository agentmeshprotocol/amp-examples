"""
Custom Analysis Example.

This script demonstrates how to create custom analysis workflows
by selectively enabling agents and customizing their behavior.
"""

import asyncio
import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# Add project modules to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'pipelines'))
sys.path.append(str(project_root / '../shared-lib'))

from pipelines.data_pipeline import DataAnalysisPipeline, PipelineConfig


def create_customer_data():
    """Create customer behavior dataset."""
    
    np.random.seed(123)
    n_customers = 500
    
    # Generate customer demographics
    ages = np.random.normal(35, 12, n_customers).astype(int)
    ages = np.clip(ages, 18, 75)
    
    data = {
        'customer_id': range(1, n_customers + 1),
        'age': ages,
        'income': np.random.lognormal(10.5, 0.5, n_customers),
        'months_as_customer': np.random.exponential(24, n_customers),
        'total_purchases': np.random.gamma(2, 500, n_customers),
        'support_tickets': np.random.poisson(2, n_customers),
        'satisfaction_score': np.random.beta(2, 1, n_customers) * 5,  # 0-5 scale
        'email_engagement': np.random.beta(1.5, 2, n_customers),  # 0-1 scale
        'mobile_app_usage': np.random.gamma(1, 10, n_customers),  # hours per month
        'segment': np.random.choice(['Basic', 'Premium', 'VIP'], n_customers, p=[0.6, 0.3, 0.1])
    }
    
    # Add realistic correlations
    for i in range(n_customers):
        # Higher income customers tend to:
        if data['income'][i] > 75000:
            data['total_purchases'][i] *= 1.5
            data['satisfaction_score'][i] *= 1.1
            data['segment'][i] = np.random.choice(['Premium', 'VIP'], p=[0.7, 0.3])
        
        # Longer tenure customers:
        if data['months_as_customer'][i] > 36:
            data['satisfaction_score'][i] *= 1.2
            data['email_engagement'][i] *= 0.8  # Email fatigue
        
        # Premium customers
        if data['segment'][i] == 'Premium':
            data['mobile_app_usage'][i] *= 1.4
        elif data['segment'][i] == 'VIP':
            data['mobile_app_usage'][i] *= 1.8
            data['support_tickets'][i] = max(0, data['support_tickets'][i] - 1)  # Better service
    
    # Clip values to realistic ranges
    for key in data:
        if key in ['satisfaction_score']:
            data[key] = np.clip(data[key], 0, 5)
        elif key in ['email_engagement']:
            data[key] = np.clip(data[key], 0, 1)
        elif key in ['months_as_customer']:
            data[key] = np.clip(data[key], 1, 120)  # Max 10 years
        elif key in ['income']:
            data[key] = np.clip(data[key], 20000, 300000)
    
    df = pd.DataFrame(data)
    
    # Introduce some missing values
    missing_indices = np.random.choice(n_customers, size=int(n_customers * 0.05), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['satisfaction_score', 'email_engagement'])
        df.at[idx, col] = np.nan
    
    return df


async def run_statistical_only_analysis():
    """Run analysis with only statistical components."""
    
    print("=" * 60)
    print("CUSTOM ANALYSIS: STATISTICAL ONLY")
    print("=" * 60)
    
    # Create customer data
    df = create_customer_data()
    data_file = project_root / "data" / "customer_behavior.csv"
    data_file.parent.mkdir(exist_ok=True)
    df.to_csv(data_file, index=False)
    
    print(f"Customer dataset created: {df.shape[0]} customers, {df.shape[1]} features")
    
    # Configure pipeline with only statistical analysis
    config = PipelineConfig(
        pipeline_id="statistical_analysis_001",
        pipeline_name="Customer Behavior Statistical Analysis",
        
        # Enable only specific components
        enable_data_collection=True,
        enable_data_cleaning=True,
        enable_statistical_analysis=True,
        enable_ml_analysis=False,  # Skip ML
        enable_visualization=True,
        enable_quality_assurance=True,
        
        # Statistical-focused settings
        data_quality_threshold=0.85,
        generate_report=True,
        create_dashboard=False,  # Focus on statistical plots
        
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
    
    analysis_request = """
    Perform comprehensive statistical analysis of customer behavior:
    1. Analyze customer demographics and spending patterns
    2. Examine relationships between satisfaction, engagement, and spending
    3. Compare behavior across customer segments (Basic, Premium, VIP)
    4. Identify significant correlations and trends
    5. Test hypotheses about customer loyalty and satisfaction
    """
    
    print(f"Pipeline configuration: Statistical Analysis Focus")
    print(f"Enabled components: Collection, Cleaning, Statistics, Visualization, QA")
    print(f"Analysis focus: Customer behavior patterns and relationships")
    
    return config, str(data_file), analysis_request


async def run_ml_focused_analysis():
    """Run analysis focused on machine learning."""
    
    print("=" * 60)
    print("CUSTOM ANALYSIS: ML FOCUSED")
    print("=" * 60)
    
    # Create different dataset for ML
    np.random.seed(456)
    n_samples = 800
    
    # Generate features for churn prediction
    data = {
        'customer_id': range(1, n_samples + 1),
        'tenure_months': np.random.exponential(18, n_samples),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.gamma(2, 1000, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.5, 0.1]),
        'tech_support': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        'online_security': np.random.choice(['Yes', 'No'], n_samples, p=[0.35, 0.65]),
        'streaming_services': np.random.randint(0, 4, n_samples),  # Number of streaming services
        'customer_service_calls': np.random.poisson(3, n_samples),
        'payment_method': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_samples)
    }
    
    # Generate churn based on features
    churn_probability = np.zeros(n_samples)
    for i in range(n_samples):
        prob = 0.1  # Base probability
        
        # Month-to-month contracts have higher churn
        if data['contract_type'][i] == 'Month-to-month':
            prob += 0.3
        elif data['contract_type'][i] == 'Two year':
            prob -= 0.15
        
        # High charges increase churn probability
        if data['monthly_charges'][i] > 80:
            prob += 0.2
        
        # Poor service increases churn
        if data['customer_service_calls'][i] > 5:
            prob += 0.25
        
        # Good services reduce churn
        if data['tech_support'][i] == 'Yes':
            prob -= 0.1
        if data['online_security'][i] == 'Yes':
            prob -= 0.08
        
        # Tenure reduces churn
        if data['tenure_months'][i] > 24:
            prob -= 0.2
        
        churn_probability[i] = np.clip(prob, 0, 0.8)
    
    # Generate actual churn
    data['churned'] = np.random.binomial(1, churn_probability, n_samples)
    
    df = pd.DataFrame(data)
    data_file = project_root / "data" / "customer_churn.csv"
    df.to_csv(data_file, index=False)
    
    print(f"Churn dataset created: {df.shape[0]} customers, {df.shape[1]} features")
    print(f"Churn rate: {df['churned'].mean():.2%}")
    
    # Configure ML-focused pipeline
    config = PipelineConfig(
        pipeline_id="ml_churn_analysis_001",
        pipeline_name="Customer Churn ML Analysis",
        
        # ML-focused components
        enable_data_collection=True,
        enable_data_cleaning=True,
        enable_statistical_analysis=False,  # Skip detailed statistics
        enable_ml_analysis=True,
        enable_visualization=True,
        enable_quality_assurance=True,
        
        # ML-optimized settings
        data_quality_threshold=0.8,
        model_performance_threshold=0.75,  # Higher threshold for churn prediction
        generate_report=True,
        create_dashboard=True,
        
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
    
    analysis_request = """
    Build machine learning models to predict customer churn:
    1. Perform feature engineering on customer data
    2. Train multiple classification models (Random Forest, Gradient Boosting, Logistic Regression)
    3. Evaluate model performance using accuracy, precision, recall, and AUC
    4. Identify most important features for churn prediction
    5. Provide actionable insights for churn prevention
    """
    
    print(f"Pipeline configuration: ML Analysis Focus")
    print(f"Enabled components: Collection, Cleaning, ML, Visualization, QA")
    print(f"Analysis focus: Churn prediction and feature importance")
    
    return config, str(data_file), analysis_request


async def run_quality_focused_analysis():
    """Run analysis focused on data quality and validation."""
    
    print("=" * 60)
    print("CUSTOM ANALYSIS: QUALITY FOCUSED")
    print("=" * 60)
    
    # Create dataset with intentional quality issues
    np.random.seed(789)
    n_samples = 300
    
    data = {
        'record_id': range(1, n_samples + 1),
        'timestamp': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'sensor_reading': np.random.normal(25, 5, n_samples),
        'location': np.random.choice(['A', 'B', 'C', 'D'], n_samples),
        'operator_id': np.random.randint(1, 10, n_samples),
        'measurement_type': np.random.choice(['Temperature', 'Pressure', 'Humidity'], n_samples),
        'quality_flag': np.random.choice(['Good', 'Fair', 'Poor'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    # Introduce various quality issues
    
    # 1. Missing values (10%)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
    for idx in missing_indices:
        col = np.random.choice(['sensor_reading', 'operator_id'])
        data[col][idx] = np.nan
    
    # 2. Outliers (5%)
    outlier_indices = np.random.choice(n_samples, size=int(n_samples * 0.05), replace=False)
    for idx in outlier_indices:
        data['sensor_reading'][idx] = np.random.choice([-50, 100])  # Extreme values
    
    # 3. Duplicates (3%)
    duplicate_indices = np.random.choice(n_samples-50, size=int(n_samples * 0.03), replace=False)
    for i, idx in enumerate(duplicate_indices):
        if idx + 1 < n_samples:
            # Copy previous record
            for key in data:
                if key != 'record_id':
                    data[key][idx + 1] = data[key][idx]
    
    # 4. Invalid values
    invalid_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    for idx in invalid_indices:
        data['operator_id'][idx] = -1  # Invalid operator ID
    
    df = pd.DataFrame(data)
    data_file = project_root / "data" / "sensor_quality_test.csv"
    df.to_csv(data_file, index=False)
    
    print(f"Quality test dataset created: {df.shape[0]} records, {df.shape[1]} fields")
    print(f"Intentional quality issues: missing values, outliers, duplicates, invalid values")
    
    # Configure quality-focused pipeline
    config = PipelineConfig(
        pipeline_id="quality_validation_001",
        pipeline_name="Data Quality Validation Analysis",
        
        # Quality-focused components
        enable_data_collection=True,
        enable_data_cleaning=True,
        enable_statistical_analysis=False,
        enable_ml_analysis=False,
        enable_visualization=True,  # For quality visualizations
        enable_quality_assurance=True,
        
        # Strict quality settings
        data_quality_threshold=0.95,  # Very high threshold
        model_performance_threshold=0.9,
        generate_report=True,
        create_dashboard=False,
        
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
    
    analysis_request = """
    Perform comprehensive data quality validation:
    1. Assess data completeness, consistency, accuracy, and uniqueness
    2. Identify and catalog all data quality issues
    3. Provide detailed recommendations for data quality improvement
    4. Create quality scorecards and validation reports
    5. Establish data quality monitoring framework
    """
    
    print(f"Pipeline configuration: Quality Validation Focus")
    print(f"Enabled components: Collection, Cleaning, Visualization, QA")
    print(f"Analysis focus: Data quality assessment and validation")
    
    return config, str(data_file), analysis_request


async def run_all_custom_analyses():
    """Run all custom analysis examples."""
    
    print("AutoGen Data Analysis Pipeline - Custom Analysis Examples")
    print("This demonstrates different pipeline configurations for specific use cases.\n")
    
    analyses = [
        ("Statistical Analysis", run_statistical_only_analysis),
        ("ML-Focused Analysis", run_ml_focused_analysis),
        ("Quality Validation", run_quality_focused_analysis)
    ]
    
    for analysis_name, analysis_func in analyses:
        try:
            print(f"\n{'=' * 20} {analysis_name.upper()} {'=' * 20}")
            config, data_file, analysis_request = await analysis_func()
            
            print(f"\nConfiguration Summary:")
            print(f"  Pipeline ID: {config.pipeline_id}")
            print(f"  Data Source: {data_file}")
            print(f"  Components: {sum([config.enable_data_collection, config.enable_data_cleaning, config.enable_statistical_analysis, config.enable_ml_analysis, config.enable_visualization, config.enable_quality_assurance])} enabled")
            print(f"  Quality Threshold: {config.data_quality_threshold}")
            
            # Show what would be executed
            print(f"\nWould execute analysis:")
            print(f"  {analysis_request.strip()}")
            
            print(f"\nExpected outputs:")
            if config.enable_statistical_analysis:
                print(f"  - Descriptive statistics and correlation analysis")
                print(f"  - Hypothesis testing results")
            if config.enable_ml_analysis:
                print(f"  - ML model training and evaluation")
                print(f"  - Feature importance analysis")
            if config.enable_visualization:
                print(f"  - Statistical plots and visualizations")
            if config.enable_quality_assurance:
                print(f"  - Data quality validation report")
            if config.generate_report:
                print(f"  - Comprehensive analysis report")
            
            print(f"\nTo run this analysis with real LLM integration:")
            print(f"  python run_pipeline.py --data {data_file} --request '{analysis_request.split(':')[0]}' \\")
            if not config.enable_statistical_analysis:
                print(f"    --no-stats \\")
            if not config.enable_ml_analysis:
                print(f"    --no-ml \\")
            if not config.create_dashboard:
                print(f"    --no-dashboard \\")
            print(f"    --pipeline-id {config.pipeline_id}")
            
            print(f"\n{'=' * (42 + len(analysis_name))}")
            
        except Exception as e:
            print(f"Error in {analysis_name}: {e}")


def main():
    """Main entry point."""
    
    try:
        asyncio.run(run_all_custom_analyses())
        
        print(f"\n{'=' * 80}")
        print("CUSTOM ANALYSIS EXAMPLES COMPLETED")
        print("=" * 80)
        print("\nKey Takeaways:")
        print("1. Pipeline components can be selectively enabled based on analysis needs")
        print("2. Different configurations optimize for different analytical goals")
        print("3. Quality thresholds can be adjusted for specific requirements")
        print("4. Each configuration produces tailored outputs and insights")
        print("\nNext Steps:")
        print("- Configure API keys and AMP network for full execution")
        print("- Customize agent configurations for your specific domain")
        print("- Extend with additional custom agents as needed")
        
    except KeyboardInterrupt:
        print("\nCustom analysis examples interrupted by user")
    except Exception as e:
        print(f"\nCustom analysis examples failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()