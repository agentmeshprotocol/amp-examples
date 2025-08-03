"""
Data Cleaner Agent.

Specialized AutoGen agent for data cleaning and preprocessing
including handling missing values, outliers, and normalization.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
import warnings

from .base_agent import AutoGenAMPAgent, AutoGenConfig
from amp_types import Capability, CapabilityConstraints
from amp_client import AMPClientConfig

warnings.filterwarnings('ignore')


class DataCleanerAgent(AutoGenAMPAgent):
    """
    Agent specialized in data cleaning and preprocessing.
    
    Capabilities:
    - Missing value detection and imputation
    - Outlier detection and handling
    - Data type optimization and conversion
    - Normalization and scaling
    - Duplicate detection and removal
    - Data quality assessment
    """
    
    def __init__(
        self,
        amp_config: AMPClientConfig,
        llm_config: Dict[str, Any],
        cleaning_config: Dict[str, Any] = None
    ):
        """
        Initialize Data Cleaner Agent.
        
        Args:
            amp_config: AMP client configuration
            llm_config: LLM configuration for AutoGen
            cleaning_config: Configuration for cleaning operations
        """
        # Define capabilities
        capabilities = [
            Capability(
                id="data-cleaning-missing-values",
                version="1.0",
                description="Handle missing values through detection and imputation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "strategy": {"type": "string", "enum": ["drop", "mean", "median", "mode", "knn", "forward_fill", "backward_fill"]},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "threshold": {"type": "number", "minimum": 0, "maximum": 1}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "missing_summary": {"type": "object"},
                        "imputation_report": {"type": "object"},
                        "cleaned_dataset_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=10000
                ),
                category="data-processing"
            ),
            Capability(
                id="data-cleaning-outliers",
                version="1.0",
                description="Detect and handle outliers using statistical methods",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "method": {"type": "string", "enum": ["iqr", "zscore", "isolation_forest"]},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "threshold": {"type": "number"},
                        "action": {"type": "string", "enum": ["remove", "cap", "transform", "flag"]}
                    },
                    "required": ["dataset_key", "method"]
                },
                output_schema={
                    "type": "object", 
                    "properties": {
                        "outliers_detected": {"type": "integer"},
                        "outlier_summary": {"type": "object"},
                        "cleaned_dataset_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=15000
                ),
                category="data-processing"
            ),
            Capability(
                id="data-cleaning-normalization",
                version="1.0",
                description="Normalize and scale numerical data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "method": {"type": "string", "enum": ["standard", "minmax", "robust", "quantile"]},
                        "columns": {"type": "array", "items": {"type": "string"}},
                        "feature_range": {"type": "array", "items": {"type": "number"}}
                    },
                    "required": ["dataset_key", "method"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "scaling_summary": {"type": "object"},
                        "normalized_dataset_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=8000
                ),
                category="data-processing"
            ),
            Capability(
                id="data-quality-assessment",
                version="1.0",
                description="Comprehensive data quality assessment",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "quality_metrics": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "quality_score": {"type": "number"},
                        "quality_report": {"type": "object"},
                        "recommendations": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=12000
                ),
                category="analysis"
            ),
            Capability(
                id="data-cleaning-duplicates",
                version="1.0",
                description="Detect and remove duplicate records",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "subset": {"type": "array", "items": {"type": "string"}},
                        "keep": {"type": "string", "enum": ["first", "last", "false"]},
                        "method": {"type": "string", "enum": ["exact", "fuzzy"]}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "duplicates_found": {"type": "integer"},
                        "duplicate_summary": {"type": "object"},
                        "cleaned_dataset_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=10000
                ),
                category="data-processing"
            )
        ]
        
        # AutoGen configuration
        autogen_config = AutoGenConfig(
            name="DataCleaner",
            system_message="""You are a Data Cleaner Agent specialized in data preprocessing and quality improvement.

Your responsibilities:
1. Detect and handle missing values using appropriate imputation strategies
2. Identify and manage outliers through statistical methods
3. Normalize and scale numerical data for analysis
4. Remove duplicate records and ensure data uniqueness
5. Assess overall data quality and provide improvement recommendations

You work collaboratively with other agents in the data analysis pipeline. Always provide detailed reports about the cleaning operations performed and the impact on data quality.

When cleaning data, focus on:
- Preserving data integrity and meaningful information
- Choosing appropriate methods based on data characteristics
- Documenting all transformations for reproducibility
- Balancing data quality with information retention
- Providing clear recommendations for further improvements""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        super().__init__(
            autogen_config=autogen_config,
            amp_config=amp_config,
            capabilities=capabilities,
            description="Agent specialized in data cleaning and preprocessing operations",
            tags=["data-cleaning", "preprocessing", "quality", "autogen"]
        )
        
        # Cleaning configuration
        self.cleaning_config = cleaning_config or {
            "default_missing_strategy": "median",
            "outlier_threshold": 3.0,
            "default_scaling": "standard",
            "duplicate_threshold": 0.95
        }
        
        # Cleaning metrics
        self.cleaning_metrics = {
            "datasets_cleaned": 0,
            "missing_values_imputed": 0,
            "outliers_handled": 0,
            "duplicates_removed": 0,
            "quality_assessments": 0
        }
    
    def _process_conversation_message(
        self,
        message: str,
        sender,
        conversation_history: List[Dict]
    ) -> str:
        """Process conversation messages for data cleaning requests."""
        
        message_lower = message.lower()
        
        if "clean data" in message_lower or "preprocess" in message_lower:
            return self._handle_cleaning_request(message)
        elif "missing values" in message_lower or "impute" in message_lower:
            return self._handle_missing_values_request(message)
        elif "outliers" in message_lower:
            return self._handle_outliers_request(message)
        elif "normalize" in message_lower or "scale" in message_lower:
            return self._handle_normalization_request(message)
        elif "duplicates" in message_lower:
            return self._handle_duplicates_request(message)
        elif "quality" in message_lower or "assessment" in message_lower:
            return self._handle_quality_request(message)
        elif "status" in message_lower or "metrics" in message_lower:
            return self._handle_status_request()
        else:
            return """I'm the Data Cleaner Agent. I can help you with:

1. **Missing Values**: Detection and imputation using various strategies
2. **Outlier Handling**: Statistical outlier detection and treatment
3. **Data Normalization**: Scaling and normalization of numerical features
4. **Duplicate Removal**: Finding and removing duplicate records
5. **Quality Assessment**: Comprehensive data quality evaluation

What type of data cleaning would you like me to perform?"""
    
    def _handle_cleaning_request(self, message: str) -> str:
        """Handle general data cleaning requests."""
        artifacts = self.list_artifacts()
        if artifacts:
            return f"""I can clean the following datasets: {', '.join(artifacts)}

Available cleaning operations:
1. **Missing Values**: Handle missing data with imputation
2. **Outliers**: Detect and treat statistical outliers  
3. **Normalization**: Scale numerical features
4. **Duplicates**: Remove duplicate records
5. **Quality Check**: Assess overall data quality

Which dataset would you like me to clean and what specific operations should I perform?"""
        else:
            return "No datasets available for cleaning. Please collect data first using the Data Collector Agent."
    
    def _handle_missing_values_request(self, message: str) -> str:
        """Handle missing values requests."""
        return """I can handle missing values using these strategies:

1. **Drop**: Remove rows/columns with missing values
2. **Mean/Median/Mode**: Simple statistical imputation
3. **KNN**: K-Nearest Neighbors imputation
4. **Forward/Backward Fill**: Time series imputation

Specify the dataset and strategy: "Handle missing values in [dataset] using [strategy]" """
    
    def _handle_outliers_request(self, message: str) -> str:
        """Handle outlier detection requests.""" 
        return """I can detect outliers using these methods:

1. **IQR**: Interquartile Range method
2. **Z-Score**: Standard deviation based detection
3. **Isolation Forest**: Machine learning approach

Actions for outliers:
- **Remove**: Delete outlier records
- **Cap**: Limit to threshold values
- **Transform**: Apply mathematical transformation
- **Flag**: Mark but keep in dataset

Specify: "Detect outliers in [dataset] using [method] and [action]" """
    
    def _handle_normalization_request(self, message: str) -> str:
        """Handle normalization requests."""
        return """I can normalize data using these methods:

1. **Standard**: Z-score standardization (mean=0, std=1)
2. **MinMax**: Scale to specified range (default 0-1)
3. **Robust**: Use median and IQR (less sensitive to outliers)
4. **Quantile**: Transform to uniform distribution

Specify: "Normalize [dataset] using [method]" """
    
    def _handle_duplicates_request(self, message: str) -> str:
        """Handle duplicate removal requests."""
        return """I can handle duplicates with these options:

1. **Exact**: Find exact duplicate rows
2. **Fuzzy**: Find similar rows (configurable threshold)

Keep options:
- **First**: Keep first occurrence
- **Last**: Keep last occurrence  
- **False**: Remove all duplicates

Specify: "Remove duplicates from [dataset] keeping [first/last]" """
    
    def _handle_quality_request(self, message: str) -> str:
        """Handle quality assessment requests."""
        return """I can assess data quality across multiple dimensions:

1. **Completeness**: Missing values analysis
2. **Uniqueness**: Duplicate detection
3. **Validity**: Data type and format validation
4. **Consistency**: Cross-field validation
5. **Accuracy**: Statistical distribution analysis

Specify: "Assess quality of [dataset]" """
    
    def _handle_status_request(self) -> str:
        """Handle status and metrics requests."""
        metrics = self.cleaning_metrics
        artifacts = self.list_artifacts()
        
        return f"""Data Cleaning Status:

**Cleaning Metrics:**
- Datasets cleaned: {metrics['datasets_cleaned']}
- Missing values imputed: {metrics['missing_values_imputed']}
- Outliers handled: {metrics['outliers_handled']}
- Duplicates removed: {metrics['duplicates_removed']}
- Quality assessments: {metrics['quality_assessments']}

**Processed Datasets:** {len(artifacts)}
{', '.join(artifacts) if artifacts else 'None'}

Ready for new cleaning tasks."""
    
    # AMP Capability Handlers
    
    async def _handle_data_cleaning_missing_values(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle missing values cleaning capability."""
        try:
            dataset_key = parameters["dataset_key"]
            strategy = parameters.get("strategy", self.cleaning_config["default_missing_strategy"])
            columns = parameters.get("columns")
            threshold = parameters.get("threshold", 0.5)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            df_clean = df.copy()
            
            # Analyze missing values
            missing_summary = self._analyze_missing_values(df)
            
            # Select columns to process
            if columns is None:
                columns = df.columns.tolist()
            
            imputation_report = {}
            
            for col in columns:
                if col not in df.columns:
                    continue
                    
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue
                
                missing_ratio = missing_count / len(df)
                
                # Apply strategy based on threshold
                if missing_ratio > threshold and strategy == "drop":
                    df_clean = df_clean.drop(columns=[col])
                    imputation_report[col] = {"action": "dropped_column", "missing_count": missing_count}
                elif strategy == "drop":
                    df_clean = df_clean.dropna(subset=[col])
                    imputation_report[col] = {"action": "dropped_rows", "rows_dropped": missing_count}
                else:
                    # Apply imputation
                    original_count = df_clean[col].isnull().sum()
                    df_clean[col] = self._impute_column(df_clean[col], strategy)
                    imputed_count = original_count - df_clean[col].isnull().sum()
                    
                    imputation_report[col] = {
                        "action": "imputed",
                        "strategy": strategy,
                        "values_imputed": imputed_count
                    }
            
            # Store cleaned dataset
            cleaned_key = f"{dataset_key}_missing_cleaned"
            metadata = {
                "original_dataset": dataset_key,
                "cleaning_operation": "missing_values",
                "strategy": strategy,
                "threshold": threshold,
                "imputation_report": imputation_report,
                "shape_before": list(df.shape),
                "shape_after": list(df_clean.shape)
            }
            
            self.store_artifact(cleaned_key, df_clean, metadata)
            
            # Update metrics
            self.cleaning_metrics["datasets_cleaned"] += 1
            self.cleaning_metrics["missing_values_imputed"] += sum(
                report.get("values_imputed", 0) for report in imputation_report.values()
            )
            
            return {
                "missing_summary": missing_summary,
                "imputation_report": imputation_report,
                "cleaned_dataset_key": cleaned_key
            }
            
        except Exception as e:
            self.logger.error(f"Missing values cleaning error: {e}")
            raise
    
    async def _handle_data_cleaning_outliers(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle outlier detection and treatment capability."""
        try:
            dataset_key = parameters["dataset_key"]
            method = parameters["method"]
            columns = parameters.get("columns")
            threshold = parameters.get("threshold", self.cleaning_config["outlier_threshold"])
            action = parameters.get("action", "flag")
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            df_clean = df.copy()
            
            # Select numerical columns if not specified
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            outlier_summary = {}
            total_outliers = 0
            
            for col in columns:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                # Detect outliers
                outlier_mask = self._detect_outliers(df[col], method, threshold)
                outlier_count = outlier_mask.sum()
                total_outliers += outlier_count
                
                if outlier_count > 0:
                    # Apply action
                    if action == "remove":
                        df_clean = df_clean[~outlier_mask]
                    elif action == "cap":
                        if method == "iqr":
                            q1, q3 = df[col].quantile([0.25, 0.75])
                            iqr = q3 - q1
                            lower_bound = q1 - 1.5 * iqr
                            upper_bound = q3 + 1.5 * iqr
                            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                        elif method == "zscore":
                            mean_val = df[col].mean()
                            std_val = df[col].std()
                            lower_bound = mean_val - threshold * std_val
                            upper_bound = mean_val + threshold * std_val
                            df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)
                    elif action == "transform":
                        # Apply log transformation for positive skewed data
                        if df[col].min() > 0:
                            df_clean[col] = np.log1p(df[col])
                    elif action == "flag":
                        # Add outlier flag column
                        df_clean[f"{col}_outlier_flag"] = outlier_mask
                
                outlier_summary[col] = {
                    "outliers_detected": outlier_count,
                    "outlier_percentage": (outlier_count / len(df)) * 100,
                    "method": method,
                    "action": action
                }
            
            # Store cleaned dataset
            cleaned_key = f"{dataset_key}_outliers_cleaned"
            metadata = {
                "original_dataset": dataset_key,
                "cleaning_operation": "outliers",
                "method": method,
                "threshold": threshold,
                "action": action,
                "outlier_summary": outlier_summary,
                "shape_before": list(df.shape),
                "shape_after": list(df_clean.shape)
            }
            
            self.store_artifact(cleaned_key, df_clean, metadata)
            
            # Update metrics
            self.cleaning_metrics["datasets_cleaned"] += 1
            self.cleaning_metrics["outliers_handled"] += total_outliers
            
            return {
                "outliers_detected": total_outliers,
                "outlier_summary": outlier_summary,
                "cleaned_dataset_key": cleaned_key
            }
            
        except Exception as e:
            self.logger.error(f"Outlier cleaning error: {e}")
            raise
    
    async def _handle_data_cleaning_normalization(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data normalization capability."""
        try:
            dataset_key = parameters["dataset_key"]
            method = parameters["method"]
            columns = parameters.get("columns")
            feature_range = parameters.get("feature_range", [0, 1])
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            df_norm = df.copy()
            
            # Select numerical columns if not specified
            if columns is None:
                columns = df.select_dtypes(include=[np.number]).columns.tolist()
            
            scaling_summary = {}
            
            for col in columns:
                if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]):
                    continue
                
                original_stats = {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
                
                # Apply normalization
                if method == "standard":
                    scaler = StandardScaler()
                    df_norm[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                elif method == "minmax":
                    scaler = MinMaxScaler(feature_range=tuple(feature_range))
                    df_norm[col] = scaler.fit_transform(df[col].values.reshape(-1, 1)).flatten()
                elif method == "robust":
                    # Use median and IQR for robust scaling
                    median = df[col].median()
                    q1, q3 = df[col].quantile([0.25, 0.75])
                    iqr = q3 - q1
                    if iqr > 0:
                        df_norm[col] = (df[col] - median) / iqr
                elif method == "quantile":
                    # Quantile transformation
                    ranks = df[col].rank(method='average')
                    df_norm[col] = (ranks - 1) / (len(df) - 1)
                
                normalized_stats = {
                    "mean": df_norm[col].mean(),
                    "std": df_norm[col].std(),
                    "min": df_norm[col].min(),
                    "max": df_norm[col].max()
                }
                
                scaling_summary[col] = {
                    "method": method,
                    "original_stats": original_stats,
                    "normalized_stats": normalized_stats
                }
            
            # Store normalized dataset
            normalized_key = f"{dataset_key}_normalized"
            metadata = {
                "original_dataset": dataset_key,
                "cleaning_operation": "normalization",
                "method": method,
                "feature_range": feature_range,
                "scaling_summary": scaling_summary,
                "columns_normalized": columns
            }
            
            self.store_artifact(normalized_key, df_norm, metadata)
            
            # Update metrics
            self.cleaning_metrics["datasets_cleaned"] += 1
            
            return {
                "scaling_summary": scaling_summary,
                "normalized_dataset_key": normalized_key
            }
            
        except Exception as e:
            self.logger.error(f"Normalization error: {e}")
            raise
    
    async def _handle_data_quality_assessment(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data quality assessment capability."""
        try:
            dataset_key = parameters["dataset_key"]
            quality_metrics = parameters.get("quality_metrics", [
                "completeness", "uniqueness", "validity", "consistency"
            ])
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            quality_report = self._assess_data_quality(df, quality_metrics)
            
            # Calculate overall quality score
            metric_scores = [score for score in quality_report.values() if isinstance(score, (int, float))]
            quality_score = np.mean(metric_scores) if metric_scores else 0
            
            # Generate recommendations
            recommendations = self._generate_quality_recommendations(quality_report)
            
            # Update metrics
            self.cleaning_metrics["quality_assessments"] += 1
            
            return {
                "quality_score": float(quality_score),
                "quality_report": quality_report,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Quality assessment error: {e}")
            raise
    
    async def _handle_data_cleaning_duplicates(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle duplicate detection and removal capability."""
        try:
            dataset_key = parameters["dataset_key"]
            subset = parameters.get("subset")
            keep = parameters.get("keep", "first")
            method = parameters.get("method", "exact")
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Find duplicates
            if method == "exact":
                if subset:
                    duplicate_mask = df.duplicated(subset=subset, keep=False)
                else:
                    duplicate_mask = df.duplicated(keep=False)
                
                duplicates_found = duplicate_mask.sum()
                
                # Remove duplicates
                if keep == "false":
                    df_clean = df[~duplicate_mask]
                else:
                    if subset:
                        df_clean = df.drop_duplicates(subset=subset, keep=keep)
                    else:
                        df_clean = df.drop_duplicates(keep=keep)
            else:
                # Fuzzy matching not implemented in this example
                raise NotImplementedError("Fuzzy duplicate detection not yet implemented")
            
            duplicates_removed = len(df) - len(df_clean)
            
            duplicate_summary = {
                "original_rows": len(df),
                "duplicates_found": duplicates_found,
                "duplicates_removed": duplicates_removed,
                "final_rows": len(df_clean),
                "method": method,
                "keep_strategy": keep
            }
            
            # Store cleaned dataset
            cleaned_key = f"{dataset_key}_duplicates_cleaned"
            metadata = {
                "original_dataset": dataset_key,
                "cleaning_operation": "duplicates",
                "method": method,
                "keep": keep,
                "subset": subset,
                "duplicate_summary": duplicate_summary
            }
            
            self.store_artifact(cleaned_key, df_clean, metadata)
            
            # Update metrics
            self.cleaning_metrics["datasets_cleaned"] += 1
            self.cleaning_metrics["duplicates_removed"] += duplicates_removed
            
            return {
                "duplicates_found": duplicates_found,
                "duplicate_summary": duplicate_summary,
                "cleaned_dataset_key": cleaned_key
            }
            
        except Exception as e:
            self.logger.error(f"Duplicate cleaning error: {e}")
            raise
    
    # Helper methods
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in dataset."""
        missing_summary = {}
        
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / len(df)) * 100
            
            missing_summary[col] = {
                "missing_count": missing_count,
                "missing_percentage": float(missing_percentage),
                "data_type": str(df[col].dtype)
            }
        
        # Overall statistics
        total_missing = df.isnull().sum().sum()
        total_cells = df.shape[0] * df.shape[1]
        
        missing_summary["_overall"] = {
            "total_missing": total_missing,
            "total_cells": total_cells,
            "overall_missing_percentage": float((total_missing / total_cells) * 100)
        }
        
        return missing_summary
    
    def _impute_column(self, series: pd.Series, strategy: str) -> pd.Series:
        """Impute missing values in a column."""
        if strategy == "mean" and pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(series):
            return series.fillna(series.median())
        elif strategy == "mode":
            return series.fillna(series.mode().iloc[0] if not series.mode().empty else series.iloc[0])
        elif strategy == "forward_fill":
            return series.fillna(method='ffill')
        elif strategy == "backward_fill":
            return series.fillna(method='bfill')
        elif strategy == "knn" and pd.api.types.is_numeric_dtype(series):
            # Simple KNN imputation
            imputer = KNNImputer(n_neighbors=5)
            return pd.Series(imputer.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)
        else:
            # Default to mode for categorical, median for numerical
            if pd.api.types.is_numeric_dtype(series):
                return series.fillna(series.median())
            else:
                return series.fillna(series.mode().iloc[0] if not series.mode().empty else "Unknown")
    
    def _detect_outliers(self, series: pd.Series, method: str, threshold: float) -> pd.Series:
        """Detect outliers in a series."""
        if method == "iqr":
            q1, q3 = series.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == "zscore":
            z_scores = np.abs(stats.zscore(series.dropna()))
            outlier_mask = pd.Series(False, index=series.index)
            outlier_mask[series.dropna().index] = z_scores > threshold
            return outlier_mask
        
        elif method == "isolation_forest":
            from sklearn.ensemble import IsolationForest
            clf = IsolationForest(contamination=0.1, random_state=42)
            outlier_mask = pd.Series(False, index=series.index)
            valid_indices = series.dropna().index
            if len(valid_indices) > 10:  # Minimum samples for isolation forest
                predictions = clf.fit_predict(series.dropna().values.reshape(-1, 1))
                outlier_mask[valid_indices] = predictions == -1
            return outlier_mask
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
    
    def _assess_data_quality(self, df: pd.DataFrame, metrics: List[str]) -> Dict[str, Any]:
        """Assess data quality across multiple dimensions."""
        quality_report = {}
        
        if "completeness" in metrics:
            missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
            quality_report["completeness"] = float(1 - missing_ratio)
        
        if "uniqueness" in metrics:
            duplicate_ratio = df.duplicated().sum() / len(df)
            quality_report["uniqueness"] = float(1 - duplicate_ratio)
        
        if "validity" in metrics:
            # Basic validity checks
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            valid_numeric = 0
            total_numeric = 0
            
            for col in numeric_cols:
                total_numeric += len(df[col].dropna())
                valid_numeric += len(df[col].dropna())  # Assuming non-null numeric values are valid
            
            quality_report["validity"] = float(valid_numeric / total_numeric) if total_numeric > 0 else 1.0
        
        if "consistency" in metrics:
            # Basic consistency checks (e.g., data types)
            consistent_types = 0
            total_columns = len(df.columns)
            
            for col in df.columns:
                # Check if column has consistent data type (no mixed types)
                if df[col].apply(type).nunique() <= 1:
                    consistent_types += 1
            
            quality_report["consistency"] = float(consistent_types / total_columns)
        
        return quality_report
    
    def _generate_quality_recommendations(self, quality_report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on quality assessment."""
        recommendations = []
        
        if "completeness" in quality_report and quality_report["completeness"] < 0.8:
            recommendations.append("Consider imputing missing values or removing incomplete records")
        
        if "uniqueness" in quality_report and quality_report["uniqueness"] < 0.9:
            recommendations.append("Remove duplicate records to improve data uniqueness")
        
        if "validity" in quality_report and quality_report["validity"] < 0.9:
            recommendations.append("Validate data formats and remove invalid entries")
        
        if "consistency" in quality_report and quality_report["consistency"] < 0.8:
            recommendations.append("Standardize data types and formats across columns")
        
        if not recommendations:
            recommendations.append("Data quality appears good - consider advanced validation rules")
        
        return recommendations
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning operations."""
        summary = {
            "metrics": self.cleaning_metrics.copy(),
            "cleaned_datasets": {}
        }
        
        for artifact_key in self.list_artifacts():
            artifact = self._data_artifacts[artifact_key]
            metadata = artifact["metadata"]
            
            if metadata.get("cleaning_operation"):
                summary["cleaned_datasets"][artifact_key] = {
                    "operation": metadata["cleaning_operation"],
                    "original_dataset": metadata.get("original_dataset"),
                    "timestamp": artifact["timestamp"]
                }
        
        return summary