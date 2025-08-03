"""
Statistical Analyst Agent.

Specialized AutoGen agent for statistical analysis including descriptive statistics,
hypothesis testing, correlation analysis, and statistical modeling.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

from .base_agent import AutoGenAMPAgent, AutoGenConfig
from amp_types import Capability, CapabilityConstraints
from amp_client import AMPClientConfig

warnings.filterwarnings('ignore')


class StatisticalAnalystAgent(AutoGenAMPAgent):
    """
    Agent specialized in statistical analysis and hypothesis testing.
    
    Capabilities:
    - Descriptive statistics and distribution analysis
    - Hypothesis testing (t-tests, ANOVA, chi-square)
    - Correlation and association analysis
    - Regression analysis and model diagnostics
    - Time series analysis
    - Statistical significance testing
    """
    
    def __init__(
        self,
        amp_config: AMPClientConfig,
        llm_config: Dict[str, Any],
        analysis_config: Dict[str, Any] = None
    ):
        """
        Initialize Statistical Analyst Agent.
        
        Args:
            amp_config: AMP client configuration
            llm_config: LLM configuration for AutoGen
            analysis_config: Configuration for statistical analysis
        """
        # Define capabilities
        capabilities = [
            Capability(
                id="statistical-descriptive-analysis",
                version="1.0",
                description="Comprehensive descriptive statistical analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "group_by": {"type": "string"},
                        "include_distribution": {"type": "boolean"}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "descriptive_stats": {"type": "object"},
                        "distribution_analysis": {"type": "object"},
                        "summary_insights": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=8000
                ),
                category="analysis"
            ),
            Capability(
                id="statistical-hypothesis-testing",
                version="1.0",
                description="Statistical hypothesis testing (t-tests, ANOVA, chi-square)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "test_type": {"type": "string", "enum": ["ttest_1samp", "ttest_ind", "ttest_rel", "anova", "chi2", "mann_whitney", "wilcoxon"]},
                        "variable1": {"type": "string"},
                        "variable2": {"type": "string"},
                        "alpha": {"type": "number", "minimum": 0, "maximum": 1},
                        "hypothesis": {"type": "string"}
                    },
                    "required": ["dataset_key", "test_type", "variable1"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "test_statistic": {"type": "number"},
                        "p_value": {"type": "number"},
                        "critical_value": {"type": "number"},
                        "conclusion": {"type": "string"},
                        "effect_size": {"type": "number"},
                        "confidence_interval": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=5000
                ),
                category="analysis"
            ),
            Capability(
                id="statistical-correlation-analysis",
                version="1.0",
                description="Correlation and association analysis between variables",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "variables": {"type": "array", "items": {"type": "string"}},
                        "method": {"type": "string", "enum": ["pearson", "spearman", "kendall"]},
                        "significance_test": {"type": "boolean"}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "correlation_matrix": {"type": "object"},
                        "p_values": {"type": "object"},
                        "significant_correlations": {"type": "array"},
                        "correlation_insights": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=6000
                ),
                category="analysis"
            ),
            Capability(
                id="statistical-regression-analysis",
                version="1.0",
                description="Linear and logistic regression analysis with diagnostics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "dependent_variable": {"type": "string"},
                        "independent_variables": {"type": "array", "items": {"type": "string"}},
                        "model_type": {"type": "string", "enum": ["linear", "logistic", "polynomial"]},
                        "include_diagnostics": {"type": "boolean"}
                    },
                    "required": ["dataset_key", "dependent_variable", "independent_variables"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "model_summary": {"type": "object"},
                        "coefficients": {"type": "object"},
                        "model_metrics": {"type": "object"},
                        "diagnostics": {"type": "object"},
                        "predictions": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=12000
                ),
                category="analysis"
            ),
            Capability(
                id="statistical-distribution-testing",
                version="1.0",
                description="Test data distributions and normality",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "column": {"type": "string"},
                        "distribution": {"type": "string", "enum": ["normal", "exponential", "uniform", "gamma", "beta"]},
                        "normality_tests": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset_key", "column"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "distribution_parameters": {"type": "object"},
                        "goodness_of_fit": {"type": "object"},
                        "normality_tests": {"type": "object"},
                        "recommended_distribution": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=7000
                ),
                category="analysis"
            )
        ]
        
        # AutoGen configuration
        autogen_config = AutoGenConfig(
            name="StatisticalAnalyst",
            system_message="""You are a Statistical Analyst Agent specialized in statistical analysis and hypothesis testing.

Your responsibilities:
1. Perform descriptive statistical analysis and summarize data distributions
2. Conduct hypothesis testing using appropriate statistical tests
3. Analyze correlations and associations between variables
4. Build and validate regression models with proper diagnostics
5. Test data distributions and assess normality assumptions

You work collaboratively with other agents in the data analysis pipeline. Always provide statistically sound interpretations and clearly explain the assumptions, limitations, and practical significance of your analyses.

When performing statistical analysis, focus on:
- Selecting appropriate statistical tests based on data characteristics
- Checking assumptions before applying statistical methods
- Providing clear interpretation of results with practical significance
- Reporting confidence intervals and effect sizes when relevant
- Identifying potential confounding variables and limitations""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        super().__init__(
            autogen_config=autogen_config,
            amp_config=amp_config,
            capabilities=capabilities,
            description="Agent specialized in statistical analysis and hypothesis testing",
            tags=["statistics", "hypothesis-testing", "regression", "correlation", "autogen"]
        )
        
        # Analysis configuration
        self.analysis_config = analysis_config or {
            "default_alpha": 0.05,
            "confidence_level": 0.95,
            "min_sample_size": 30,
            "correlation_threshold": 0.3
        }
        
        # Analysis metrics
        self.analysis_metrics = {
            "descriptive_analyses": 0,
            "hypothesis_tests": 0,
            "correlation_analyses": 0,
            "regression_models": 0,
            "distribution_tests": 0
        }
    
    def _process_conversation_message(
        self,
        message: str,
        sender,
        conversation_history: List[Dict]
    ) -> str:
        """Process conversation messages for statistical analysis requests."""
        
        message_lower = message.lower()
        
        if "descriptive" in message_lower or "summary statistics" in message_lower:
            return self._handle_descriptive_request(message)
        elif "hypothesis test" in message_lower or "statistical test" in message_lower:
            return self._handle_hypothesis_request(message)
        elif "correlation" in message_lower or "association" in message_lower:
            return self._handle_correlation_request(message)
        elif "regression" in message_lower or "model" in message_lower:
            return self._handle_regression_request(message)
        elif "distribution" in message_lower or "normality" in message_lower:
            return self._handle_distribution_request(message)
        elif "status" in message_lower or "metrics" in message_lower:
            return self._handle_status_request()
        else:
            return """I'm the Statistical Analyst Agent. I can help you with:

1. **Descriptive Statistics**: Summary statistics, distributions, and data exploration
2. **Hypothesis Testing**: t-tests, ANOVA, chi-square, and non-parametric tests
3. **Correlation Analysis**: Pearson, Spearman, and Kendall correlations
4. **Regression Analysis**: Linear, logistic, and polynomial regression with diagnostics
5. **Distribution Testing**: Normality tests and distribution fitting

What type of statistical analysis would you like me to perform?"""
    
    def _handle_descriptive_request(self, message: str) -> str:
        """Handle descriptive statistics requests."""
        artifacts = self.list_artifacts()
        if artifacts:
            return f"""I can perform descriptive analysis on: {', '.join(artifacts)}

Descriptive analysis includes:
- Central tendency (mean, median, mode)
- Variability (standard deviation, variance, range)
- Distribution shape (skewness, kurtosis)
- Quartiles and percentiles
- Missing value analysis

Example: "Analyze descriptive statistics for [dataset] grouped by [column]" """
        else:
            return "No datasets available for analysis. Please collect and clean data first."
    
    def _handle_hypothesis_request(self, message: str) -> str:
        """Handle hypothesis testing requests."""
        return """I can perform these statistical tests:

**Parametric Tests:**
- One-sample t-test: Compare sample mean to population value
- Independent t-test: Compare means of two groups
- Paired t-test: Compare means of paired observations
- ANOVA: Compare means across multiple groups

**Non-parametric Tests:**
- Mann-Whitney U: Non-parametric alternative to independent t-test
- Wilcoxon: Non-parametric alternative to paired t-test
- Chi-square: Test association between categorical variables

Example: "Test if [variable1] differs between [variable2] groups using t-test" """
    
    def _handle_correlation_request(self, message: str) -> str:
        """Handle correlation analysis requests."""
        return """I can analyze correlations using:

1. **Pearson**: Linear correlation (continuous variables)
2. **Spearman**: Rank correlation (ordinal or non-linear)
3. **Kendall**: Rank correlation (small samples, tied values)

Analysis includes:
- Correlation matrix with significance testing
- Identification of strong/weak associations
- Statistical significance assessment

Example: "Analyze correlations between [variables] using Pearson method" """
    
    def _handle_regression_request(self, message: str) -> str:
        """Handle regression analysis requests."""
        return """I can build regression models:

**Model Types:**
- Linear regression: Continuous dependent variable
- Logistic regression: Binary/categorical dependent variable
- Polynomial regression: Non-linear relationships

**Analysis includes:**
- Model coefficients and significance
- R-squared and model fit metrics
- Residual analysis and diagnostics
- Multicollinearity assessment (VIF)
- Prediction capabilities

Example: "Build linear regression predicting [Y] from [X1, X2, X3]" """
    
    def _handle_distribution_request(self, message: str) -> str:
        """Handle distribution testing requests."""
        return """I can test data distributions:

**Normality Tests:**
- Shapiro-Wilk (small samples)
- Kolmogorov-Smirnov (large samples)
- Anderson-Darling
- D'Agostino-Pearson

**Distribution Fitting:**
- Normal, exponential, uniform, gamma, beta
- Goodness-of-fit testing
- Parameter estimation

Example: "Test normality of [variable] and fit best distribution" """
    
    def _handle_status_request(self) -> str:
        """Handle status and metrics requests."""
        metrics = self.analysis_metrics
        artifacts = self.list_artifacts()
        
        return f"""Statistical Analysis Status:

**Analysis Metrics:**
- Descriptive analyses: {metrics['descriptive_analyses']}
- Hypothesis tests: {metrics['hypothesis_tests']}
- Correlation analyses: {metrics['correlation_analyses']}
- Regression models: {metrics['regression_models']}
- Distribution tests: {metrics['distribution_tests']}

**Analysis Results:** {len(artifacts)}
{', '.join(artifacts) if artifacts else 'None'}

Ready for new statistical analysis tasks."""
    
    # AMP Capability Handlers
    
    async def _handle_statistical_descriptive_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle descriptive statistical analysis capability."""
        try:
            dataset_key = parameters["dataset_key"]
            columns = parameters.get("columns")
            group_by = parameters.get("group_by")
            include_distribution = parameters.get("include_distribution", True)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Select columns for analysis
            if columns is None:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            else:
                numeric_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
                categorical_cols = [col for col in columns if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])]
            
            descriptive_stats = {}
            
            # Numeric variables analysis
            if numeric_cols:
                if group_by and group_by in df.columns:
                    # Grouped analysis
                    numeric_stats = df.groupby(group_by)[numeric_cols].agg([
                        'count', 'mean', 'median', 'std', 'min', 'max', 
                        lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)
                    ])
                    numeric_stats.columns = ['_'.join(col).strip() for col in numeric_stats.columns]
                    descriptive_stats["numeric_grouped"] = numeric_stats.to_dict()
                else:
                    # Overall analysis
                    numeric_stats = df[numeric_cols].describe()
                    
                    # Add additional statistics
                    for col in numeric_cols:
                        series = df[col].dropna()
                        if len(series) > 0:
                            numeric_stats.loc['skewness', col] = series.skew()
                            numeric_stats.loc['kurtosis', col] = series.kurtosis()
                            numeric_stats.loc['variance', col] = series.var()
                            numeric_stats.loc['range', col] = series.max() - series.min()
                    
                    descriptive_stats["numeric_overall"] = numeric_stats.to_dict()
            
            # Categorical variables analysis
            if categorical_cols:
                categorical_stats = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    categorical_stats[col] = {
                        "unique_values": df[col].nunique(),
                        "most_frequent": value_counts.index[0] if len(value_counts) > 0 else None,
                        "frequency_distribution": value_counts.head(10).to_dict(),
                        "missing_count": df[col].isnull().sum()
                    }
                descriptive_stats["categorical"] = categorical_stats
            
            # Distribution analysis
            distribution_analysis = {}
            if include_distribution and numeric_cols:
                for col in numeric_cols:
                    series = df[col].dropna()
                    if len(series) > 0:
                        distribution_analysis[col] = self._analyze_distribution(series)
            
            # Generate insights
            summary_insights = self._generate_descriptive_insights(df, descriptive_stats, group_by)
            
            # Store analysis results
            results_key = f"{dataset_key}_descriptive_analysis"
            metadata = {
                "original_dataset": dataset_key,
                "analysis_type": "descriptive_statistics",
                "columns_analyzed": numeric_cols + categorical_cols,
                "group_by": group_by,
                "include_distribution": include_distribution
            }
            
            analysis_results = {
                "descriptive_stats": descriptive_stats,
                "distribution_analysis": distribution_analysis,
                "summary_insights": summary_insights
            }
            
            self.store_artifact(results_key, analysis_results, metadata)
            
            # Update metrics
            self.analysis_metrics["descriptive_analyses"] += 1
            
            return {
                "descriptive_stats": descriptive_stats,
                "distribution_analysis": distribution_analysis,
                "summary_insights": summary_insights
            }
            
        except Exception as e:
            self.logger.error(f"Descriptive analysis error: {e}")
            raise
    
    async def _handle_statistical_hypothesis_testing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hypothesis testing capability."""
        try:
            dataset_key = parameters["dataset_key"]
            test_type = parameters["test_type"]
            variable1 = parameters["variable1"]
            variable2 = parameters.get("variable2")
            alpha = parameters.get("alpha", self.analysis_config["default_alpha"])
            hypothesis = parameters.get("hypothesis", "two-sided")
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Perform the specified test
            test_results = self._perform_hypothesis_test(
                df, test_type, variable1, variable2, alpha, hypothesis
            )
            
            # Store test results
            results_key = f"{dataset_key}_hypothesis_test_{test_type}"
            metadata = {
                "original_dataset": dataset_key,
                "analysis_type": "hypothesis_testing",
                "test_type": test_type,
                "variable1": variable1,
                "variable2": variable2,
                "alpha": alpha,
                "hypothesis": hypothesis
            }
            
            self.store_artifact(results_key, test_results, metadata)
            
            # Update metrics
            self.analysis_metrics["hypothesis_tests"] += 1
            
            return test_results
            
        except Exception as e:
            self.logger.error(f"Hypothesis testing error: {e}")
            raise
    
    async def _handle_statistical_correlation_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle correlation analysis capability."""
        try:
            dataset_key = parameters["dataset_key"]
            variables = parameters.get("variables")
            method = parameters.get("method", "pearson")
            significance_test = parameters.get("significance_test", True)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Select variables for correlation
            if variables is None:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Filter to available numeric columns
            available_vars = [var for var in variables if var in df.columns and pd.api.types.is_numeric_dtype(df[var])]
            
            if len(available_vars) < 2:
                raise ValueError("Need at least 2 numeric variables for correlation analysis")
            
            # Calculate correlation matrix
            corr_data = df[available_vars]
            
            if method == "pearson":
                correlation_matrix = corr_data.corr(method='pearson')
            elif method == "spearman":
                correlation_matrix = corr_data.corr(method='spearman')
            elif method == "kendall":
                correlation_matrix = corr_data.corr(method='kendall')
            else:
                raise ValueError(f"Unknown correlation method: {method}")
            
            # Calculate p-values if requested
            p_values = None
            if significance_test:
                p_values = self._calculate_correlation_pvalues(corr_data, method)
            
            # Find significant correlations
            significant_correlations = self._find_significant_correlations(
                correlation_matrix, p_values, self.analysis_config["correlation_threshold"]
            )
            
            # Generate insights
            correlation_insights = self._generate_correlation_insights(
                correlation_matrix, significant_correlations
            )
            
            # Store analysis results
            results_key = f"{dataset_key}_correlation_analysis"
            metadata = {
                "original_dataset": dataset_key,
                "analysis_type": "correlation_analysis",
                "method": method,
                "variables": available_vars,
                "significance_test": significance_test
            }
            
            analysis_results = {
                "correlation_matrix": correlation_matrix.to_dict(),
                "p_values": p_values.to_dict() if p_values is not None else None,
                "significant_correlations": significant_correlations,
                "correlation_insights": correlation_insights
            }
            
            self.store_artifact(results_key, analysis_results, metadata)
            
            # Update metrics
            self.analysis_metrics["correlation_analyses"] += 1
            
            return analysis_results
            
        except Exception as e:
            self.logger.error(f"Correlation analysis error: {e}")
            raise
    
    async def _handle_statistical_regression_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle regression analysis capability."""
        try:
            dataset_key = parameters["dataset_key"]
            dependent_variable = parameters["dependent_variable"]
            independent_variables = parameters["independent_variables"]
            model_type = parameters.get("model_type", "linear")
            include_diagnostics = parameters.get("include_diagnostics", True)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Build and analyze regression model
            regression_results = self._build_regression_model(
                df, dependent_variable, independent_variables, model_type, include_diagnostics
            )
            
            # Store model results
            results_key = f"{dataset_key}_regression_{model_type}"
            metadata = {
                "original_dataset": dataset_key,
                "analysis_type": "regression_analysis",
                "model_type": model_type,
                "dependent_variable": dependent_variable,
                "independent_variables": independent_variables,
                "include_diagnostics": include_diagnostics
            }
            
            self.store_artifact(results_key, regression_results, metadata)
            
            # Update metrics
            self.analysis_metrics["regression_models"] += 1
            
            return regression_results
            
        except Exception as e:
            self.logger.error(f"Regression analysis error: {e}")
            raise
    
    async def _handle_statistical_distribution_testing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle distribution testing capability."""
        try:
            dataset_key = parameters["dataset_key"]
            column = parameters["column"]
            distribution = parameters.get("distribution")
            normality_tests = parameters.get("normality_tests", ["shapiro", "kstest"])
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            if column not in df.columns:
                raise ValueError(f"Column not found: {column}")
            
            series = df[column].dropna()
            if len(series) == 0:
                raise ValueError(f"No valid data in column: {column}")
            
            # Perform distribution testing
            distribution_results = self._test_distribution(series, distribution, normality_tests)
            
            # Store test results
            results_key = f"{dataset_key}_distribution_test_{column}"
            metadata = {
                "original_dataset": dataset_key,
                "analysis_type": "distribution_testing",
                "column": column,
                "distribution": distribution,
                "normality_tests": normality_tests
            }
            
            self.store_artifact(results_key, distribution_results, metadata)
            
            # Update metrics
            self.analysis_metrics["distribution_tests"] += 1
            
            return distribution_results
            
        except Exception as e:
            self.logger.error(f"Distribution testing error: {e}")
            raise
    
    # Helper methods
    
    def _analyze_distribution(self, series: pd.Series) -> Dict[str, Any]:
        """Analyze the distribution of a numeric series."""
        return {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q25": float(series.quantile(0.25)),
            "q75": float(series.quantile(0.75)),
            "iqr": float(series.quantile(0.75) - series.quantile(0.25))
        }
    
    def _generate_descriptive_insights(self, df: pd.DataFrame, stats: Dict[str, Any], group_by: str = None) -> List[str]:
        """Generate insights from descriptive statistics."""
        insights = []
        
        # Dataset overview
        insights.append(f"Dataset contains {len(df)} observations with {len(df.columns)} variables")
        
        # Missing data insights
        missing_total = df.isnull().sum().sum()
        if missing_total > 0:
            missing_pct = (missing_total / (len(df) * len(df.columns))) * 100
            insights.append(f"Dataset has {missing_pct:.1f}% missing values overall")
        
        # Numeric variable insights
        if "numeric_overall" in stats:
            numeric_stats = stats["numeric_overall"]
            for col in numeric_stats:
                col_stats = {key: numeric_stats[col][key] for key in numeric_stats[col]}
                if "skewness" in col_stats:
                    skew = col_stats["skewness"]
                    if abs(skew) > 1:
                        direction = "right" if skew > 0 else "left"
                        insights.append(f"{col} shows {direction}-skewed distribution (skewness: {skew:.2f})")
        
        # Categorical variable insights
        if "categorical" in stats:
            for col, col_stats in stats["categorical"].items():
                unique_ratio = col_stats["unique_values"] / len(df)
                if unique_ratio < 0.05:
                    insights.append(f"{col} has low cardinality ({col_stats['unique_values']} unique values)")
                elif unique_ratio > 0.8:
                    insights.append(f"{col} has high cardinality ({col_stats['unique_values']} unique values)")
        
        return insights
    
    def _perform_hypothesis_test(self, df: pd.DataFrame, test_type: str, var1: str, var2: str = None, 
                                alpha: float = 0.05, hypothesis: str = "two-sided") -> Dict[str, Any]:
        """Perform the specified hypothesis test."""
        
        if test_type == "ttest_1samp":
            # One-sample t-test
            sample = df[var1].dropna()
            pop_mean = df[var2].mean() if var2 else 0  # Use var2 as population mean if provided
            statistic, pvalue = stats.ttest_1samp(sample, pop_mean)
            
            # Calculate effect size (Cohen's d)
            effect_size = (sample.mean() - pop_mean) / sample.std()
            
            # Confidence interval
            ci = stats.t.interval(1-alpha, len(sample)-1, loc=sample.mean(), 
                                scale=sample.sem())
            
        elif test_type == "ttest_ind":
            # Independent samples t-test
            if var2 is None:
                raise ValueError("Independent t-test requires two variables")
            
            group1 = df[var1].dropna()
            group2 = df[var2].dropna()
            statistic, pvalue = stats.ttest_ind(group1, group2)
            
            # Cohen's d
            pooled_std = np.sqrt(((len(group1)-1)*group1.var() + (len(group2)-1)*group2.var()) / 
                               (len(group1) + len(group2) - 2))
            effect_size = (group1.mean() - group2.mean()) / pooled_std
            
            ci = [None, None]  # Simplified for this example
            
        elif test_type == "ttest_rel":
            # Paired samples t-test
            if var2 is None:
                raise ValueError("Paired t-test requires two variables")
            
            paired_data = df[[var1, var2]].dropna()
            statistic, pvalue = stats.ttest_rel(paired_data[var1], paired_data[var2])
            
            # Effect size for paired t-test
            differences = paired_data[var1] - paired_data[var2]
            effect_size = differences.mean() / differences.std()
            
            ci = stats.t.interval(1-alpha, len(differences)-1, loc=differences.mean(), 
                                scale=differences.sem())
            
        elif test_type == "anova":
            # One-way ANOVA
            groups = [df[df[var2] == group][var1].dropna() for group in df[var2].unique()]
            groups = [group for group in groups if len(group) > 0]  # Remove empty groups
            
            if len(groups) < 2:
                raise ValueError("ANOVA requires at least 2 groups")
            
            statistic, pvalue = stats.f_oneway(*groups)
            
            # Eta squared (effect size for ANOVA)
            overall_mean = df[var1].mean()
            ss_between = sum(len(group) * (group.mean() - overall_mean)**2 for group in groups)
            ss_total = sum((df[var1] - overall_mean)**2)
            effect_size = ss_between / ss_total if ss_total > 0 else 0
            
            ci = [None, None]
            
        elif test_type == "chi2":
            # Chi-square test of independence
            if var2 is None:
                raise ValueError("Chi-square test requires two variables")
            
            contingency_table = pd.crosstab(df[var1], df[var2])
            statistic, pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            
            # Cramér's V (effect size for chi-square)
            n = contingency_table.sum().sum()
            effect_size = np.sqrt(statistic / (n * min(contingency_table.shape) - 1))
            
            ci = [None, None]
            
        elif test_type == "mann_whitney":
            # Mann-Whitney U test
            if var2 is None:
                raise ValueError("Mann-Whitney test requires two variables")
            
            group1 = df[var1].dropna()
            group2 = df[var2].dropna()
            statistic, pvalue = stats.mannwhitneyu(group1, group2, alternative=hypothesis)
            
            # Effect size (r = Z / sqrt(N))
            z_score = stats.norm.ppf(pvalue/2) if hypothesis == "two-sided" else stats.norm.ppf(pvalue)
            effect_size = abs(z_score) / np.sqrt(len(group1) + len(group2))
            
            ci = [None, None]
            
        elif test_type == "wilcoxon":
            # Wilcoxon signed-rank test
            if var2 is None:
                raise ValueError("Wilcoxon test requires two variables")
            
            paired_data = df[[var1, var2]].dropna()
            statistic, pvalue = stats.wilcoxon(paired_data[var1], paired_data[var2])
            
            # Effect size
            z_score = stats.norm.ppf(pvalue/2) if hypothesis == "two-sided" else stats.norm.ppf(pvalue)
            effect_size = abs(z_score) / np.sqrt(len(paired_data))
            
            ci = [None, None]
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        # Determine conclusion
        if pvalue < alpha:
            conclusion = f"Reject null hypothesis (p = {pvalue:.4f} < {alpha})"
        else:
            conclusion = f"Fail to reject null hypothesis (p = {pvalue:.4f} >= {alpha})"
        
        return {
            "test_statistic": float(statistic),
            "p_value": float(pvalue),
            "critical_value": None,  # Could be calculated based on test type
            "conclusion": conclusion,
            "effect_size": float(effect_size) if effect_size is not None else None,
            "confidence_interval": [float(ci[0]), float(ci[1])] if ci[0] is not None else None
        }
    
    def _calculate_correlation_pvalues(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """Calculate p-values for correlation matrix."""
        n = len(data)
        p_values = pd.DataFrame(index=data.columns, columns=data.columns, dtype=float)
        
        for i, col1 in enumerate(data.columns):
            for j, col2 in enumerate(data.columns):
                if i == j:
                    p_values.loc[col1, col2] = 0.0
                elif i < j:
                    if method == "pearson":
                        _, p_val = stats.pearsonr(data[col1].dropna(), data[col2].dropna())
                    elif method == "spearman":
                        _, p_val = stats.spearmanr(data[col1].dropna(), data[col2].dropna())
                    elif method == "kendall":
                        _, p_val = stats.kendalltau(data[col1].dropna(), data[col2].dropna())
                    else:
                        p_val = np.nan
                    
                    p_values.loc[col1, col2] = p_val
                    p_values.loc[col2, col1] = p_val
        
        return p_values
    
    def _find_significant_correlations(self, corr_matrix: pd.DataFrame, p_values: pd.DataFrame, 
                                     threshold: float) -> List[Dict[str, Any]]:
        """Find statistically significant correlations above threshold."""
        significant = []
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr_val = corr_matrix.loc[col1, col2]
                    p_val = p_values.loc[col1, col2] if p_values is not None else None
                    
                    if abs(corr_val) >= threshold:
                        significant.append({
                            "variable1": col1,
                            "variable2": col2,
                            "correlation": float(corr_val),
                            "p_value": float(p_val) if p_val is not None else None,
                            "strength": self._interpret_correlation_strength(abs(corr_val))
                        })
        
        return sorted(significant, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def _interpret_correlation_strength(self, abs_corr: float) -> str:
        """Interpret correlation strength."""
        if abs_corr >= 0.7:
            return "strong"
        elif abs_corr >= 0.3:
            return "moderate"
        else:
            return "weak"
    
    def _generate_correlation_insights(self, corr_matrix: pd.DataFrame, 
                                     significant_corrs: List[Dict[str, Any]]) -> List[str]:
        """Generate insights from correlation analysis."""
        insights = []
        
        if significant_corrs:
            strongest = significant_corrs[0]
            insights.append(
                f"Strongest correlation: {strongest['variable1']} and {strongest['variable2']} "
                f"(r = {strongest['correlation']:.3f}, {strongest['strength']})"
            )
            
            strong_corrs = [c for c in significant_corrs if c['strength'] == 'strong']
            if strong_corrs:
                insights.append(f"Found {len(strong_corrs)} strong correlations (|r| >= 0.7)")
            
            positive_corrs = [c for c in significant_corrs if c['correlation'] > 0]
            negative_corrs = [c for c in significant_corrs if c['correlation'] < 0]
            
            insights.append(f"Direction: {len(positive_corrs)} positive, {len(negative_corrs)} negative correlations")
        else:
            insights.append("No significant correlations found above the threshold")
        
        return insights
    
    def _build_regression_model(self, df: pd.DataFrame, y_var: str, x_vars: List[str], 
                              model_type: str, include_diagnostics: bool) -> Dict[str, Any]:
        """Build and analyze regression model."""
        
        # Prepare data
        model_data = df[[y_var] + x_vars].dropna()
        
        if len(model_data) == 0:
            raise ValueError("No valid data for regression model")
        
        y = model_data[y_var]
        X = model_data[x_vars]
        
        # Add intercept for statsmodels
        X_with_const = sm.add_constant(X)
        
        # Build model based on type
        if model_type == "linear":
            model = sm.OLS(y, X_with_const).fit()
        elif model_type == "logistic":
            model = sm.Logit(y, X_with_const).fit()
        else:
            raise ValueError(f"Model type {model_type} not implemented")
        
        # Extract model information
        results = {
            "model_summary": {
                "r_squared": float(model.rsquared) if hasattr(model, 'rsquared') else None,
                "adj_r_squared": float(model.rsquared_adj) if hasattr(model, 'rsquared_adj') else None,
                "f_statistic": float(model.fvalue) if hasattr(model, 'fvalue') else None,
                "f_pvalue": float(model.f_pvalue) if hasattr(model, 'f_pvalue') else None,
                "aic": float(model.aic),
                "bic": float(model.bic),
                "n_observations": int(model.nobs)
            },
            "coefficients": {},
            "model_metrics": {},
            "diagnostics": {},
            "predictions": model.fittedvalues.tolist()
        }
        
        # Extract coefficients
        for i, param in enumerate(model.params.index):
            results["coefficients"][param] = {
                "coefficient": float(model.params[param]),
                "std_error": float(model.bse[param]),
                "t_statistic": float(model.tvalues[param]),
                "p_value": float(model.pvalues[param]),
                "conf_int_lower": float(model.conf_int().iloc[i, 0]),
                "conf_int_upper": float(model.conf_int().iloc[i, 1])
            }
        
        # Model metrics
        if model_type == "linear":
            results["model_metrics"]["rmse"] = float(np.sqrt(model.mse_resid))
            results["model_metrics"]["mae"] = float(np.mean(np.abs(model.resid)))
        
        # Diagnostics
        if include_diagnostics and model_type == "linear":
            # Residual analysis
            residuals = model.resid
            results["diagnostics"]["residual_stats"] = {
                "mean": float(residuals.mean()),
                "std": float(residuals.std()),
                "skewness": float(residuals.skew()),
                "kurtosis": float(residuals.kurtosis())
            }
            
            # Heteroscedasticity test (Breusch-Pagan)
            try:
                bp_stat, bp_pvalue, _, _ = het_breuschpagan(residuals, X_with_const)
                results["diagnostics"]["breusch_pagan"] = {
                    "statistic": float(bp_stat),
                    "p_value": float(bp_pvalue)
                }
            except:
                pass
            
            # Durbin-Watson test for autocorrelation
            try:
                dw_stat = durbin_watson(residuals)
                results["diagnostics"]["durbin_watson"] = float(dw_stat)
            except:
                pass
            
            # Multicollinearity (VIF)
            if len(x_vars) > 1:
                try:
                    vif_data = []
                    for i, var in enumerate(x_vars):
                        vif = variance_inflation_factor(X.values, i)
                        vif_data.append({"variable": var, "vif": float(vif)})
                    results["diagnostics"]["vif"] = vif_data
                except:
                    pass
        
        return results
    
    def _test_distribution(self, series: pd.Series, distribution: str = None, 
                         normality_tests: List[str] = None) -> Dict[str, Any]:
        """Test data distribution and normality."""
        
        results = {
            "distribution_parameters": {},
            "goodness_of_fit": {},
            "normality_tests": {},
            "recommended_distribution": None
        }
        
        # Normality tests
        if normality_tests:
            for test in normality_tests:
                try:
                    if test == "shapiro" and len(series) <= 5000:
                        stat, pval = stats.shapiro(series)
                        results["normality_tests"]["shapiro_wilk"] = {
                            "statistic": float(stat),
                            "p_value": float(pval),
                            "is_normal": pval > 0.05
                        }
                    elif test == "kstest":
                        stat, pval = stats.kstest(series, 'norm', args=(series.mean(), series.std()))
                        results["normality_tests"]["kolmogorov_smirnov"] = {
                            "statistic": float(stat),
                            "p_value": float(pval),
                            "is_normal": pval > 0.05
                        }
                    elif test == "anderson":
                        result = stats.anderson(series, dist='norm')
                        results["normality_tests"]["anderson_darling"] = {
                            "statistic": float(result.statistic),
                            "critical_values": result.critical_values.tolist(),
                            "significance_levels": result.significance_level.tolist()
                        }
                except Exception as e:
                    self.logger.warning(f"Failed to perform {test} test: {e}")
        
        # Distribution fitting
        if distribution:
            try:
                if distribution == "normal":
                    params = stats.norm.fit(series)
                    ks_stat, ks_pval = stats.kstest(series, lambda x: stats.norm.cdf(x, *params))
                elif distribution == "exponential":
                    params = stats.expon.fit(series)
                    ks_stat, ks_pval = stats.kstest(series, lambda x: stats.expon.cdf(x, *params))
                elif distribution == "gamma":
                    params = stats.gamma.fit(series)
                    ks_stat, ks_pval = stats.kstest(series, lambda x: stats.gamma.cdf(x, *params))
                else:
                    params = None
                    ks_stat, ks_pval = None, None
                
                if params:
                    results["distribution_parameters"][distribution] = {
                        "parameters": [float(p) for p in params],
                        "ks_statistic": float(ks_stat) if ks_stat else None,
                        "ks_pvalue": float(ks_pval) if ks_pval else None
                    }
            except Exception as e:
                self.logger.warning(f"Failed to fit {distribution} distribution: {e}")
        
        # Recommend best distribution based on normality tests
        normal_count = sum(1 for test_result in results["normality_tests"].values() 
                          if test_result.get("is_normal", False))
        
        if normal_count >= len(results["normality_tests"]) / 2:
            results["recommended_distribution"] = "normal"
        else:
            # Simple heuristic based on skewness
            skewness = series.skew()
            if abs(skewness) < 0.5:
                results["recommended_distribution"] = "normal"
            elif skewness > 1:
                results["recommended_distribution"] = "exponential"
            else:
                results["recommended_distribution"] = "gamma"
        
        return results
    
    def get_analysis_summary(self) -> Dict[str, Any]:
        """Get summary of all statistical analyses."""
        summary = {
            "metrics": self.analysis_metrics.copy(),
            "analysis_results": {}
        }
        
        for artifact_key in self.list_artifacts():
            artifact = self._data_artifacts[artifact_key]
            metadata = artifact["metadata"]
            
            if metadata.get("analysis_type"):
                summary["analysis_results"][artifact_key] = {
                    "analysis_type": metadata["analysis_type"],
                    "timestamp": artifact["timestamp"]
                }
        
        return summary