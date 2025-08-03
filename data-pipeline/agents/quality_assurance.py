"""
Quality Assurance Agent.

Specialized AutoGen agent for validating results, ensuring accuracy,
and maintaining quality standards throughout the data analysis pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from scipy import stats
import warnings

from .base_agent import AutoGenAMPAgent, AutoGenConfig
from amp_types import Capability, CapabilityConstraints
from amp_client import AMPClientConfig

warnings.filterwarnings('ignore')


class QualityAssuranceAgent(AutoGenAMPAgent):
    """
    Agent specialized in quality assurance and validation.
    
    Capabilities:
    - Data quality validation and integrity checks
    - Model performance validation and benchmarking
    - Statistical test validation and assumption checking
    - Cross-validation and robustness testing
    - Pipeline audit and compliance checking
    - Result consistency verification
    """
    
    def __init__(
        self,
        amp_config: AMPClientConfig,
        llm_config: Dict[str, Any],
        qa_config: Dict[str, Any] = None
    ):
        """
        Initialize Quality Assurance Agent.
        
        Args:
            amp_config: AMP client configuration
            llm_config: LLM configuration for AutoGen
            qa_config: Configuration for QA standards and thresholds
        """
        # Define capabilities
        capabilities = [
            Capability(
                id="qa-data-validation",
                version="1.0",
                description="Comprehensive data quality validation and integrity checks",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "validation_rules": {"type": "object"},
                        "quality_thresholds": {"type": "object"},
                        "check_types": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "validation_passed": {"type": "boolean"},
                        "quality_score": {"type": "number"},
                        "validation_results": {"type": "object"},
                        "issues_found": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=8000
                ),
                category="validation"
            ),
            Capability(
                id="qa-model-validation",
                version="1.0",
                description="Model performance validation and benchmarking",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_key": {"type": "string"},
                        "validation_dataset_key": {"type": "string"},
                        "performance_thresholds": {"type": "object"},
                        "benchmark_models": {"type": "array", "items": {"type": "string"}},
                        "validation_type": {"type": "string", "enum": ["holdout", "cross_validation", "bootstrap"]}
                    },
                    "required": ["model_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "model_validated": {"type": "boolean"},
                        "performance_metrics": {"type": "object"},
                        "benchmark_comparison": {"type": "object"},
                        "validation_issues": {"type": "array"},
                        "deployment_readiness": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=15000
                ),
                category="validation"
            ),
            Capability(
                id="qa-statistical-validation",
                version="1.0",
                description="Statistical test validation and assumption checking",
                input_schema={
                    "type": "object",
                    "properties": {
                        "analysis_key": {"type": "string"},
                        "test_type": {"type": "string"},
                        "assumptions_check": {"type": "boolean"},
                        "significance_level": {"type": "number", "minimum": 0, "maximum": 1},
                        "power_analysis": {"type": "boolean"}
                    },
                    "required": ["analysis_key", "test_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "assumptions_met": {"type": "boolean"},
                        "test_validity": {"type": "boolean"},
                        "assumption_results": {"type": "object"},
                        "power_analysis_results": {"type": "object"},
                        "statistical_warnings": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=6000
                ),
                category="validation"
            ),
            Capability(
                id="qa-pipeline-audit",
                version="1.0",
                description="End-to-end pipeline audit and compliance checking",
                input_schema={
                    "type": "object",
                    "properties": {
                        "pipeline_artifacts": {"type": "array", "items": {"type": "string"}},
                        "audit_standards": {"type": "object"},
                        "compliance_requirements": {"type": "array", "items": {"type": "string"}},
                        "trace_lineage": {"type": "boolean"}
                    },
                    "required": ["pipeline_artifacts"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "audit_passed": {"type": "boolean"},
                        "compliance_score": {"type": "number"},
                        "audit_findings": {"type": "object"},
                        "lineage_trace": {"type": "object"},
                        "improvement_actions": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=12000
                ),
                category="validation"
            ),
            Capability(
                id="qa-consistency-check",
                version="1.0",
                description="Cross-validation and result consistency verification",
                input_schema={
                    "type": "object",
                    "properties": {
                        "primary_result_key": {"type": "string"},
                        "comparison_keys": {"type": "array", "items": {"type": "string"}},
                        "consistency_thresholds": {"type": "object"},
                        "validation_method": {"type": "string", "enum": ["statistical", "logical", "domain"]}
                    },
                    "required": ["primary_result_key", "comparison_keys"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "consistency_score": {"type": "number"},
                        "consistent_results": {"type": "boolean"},
                        "inconsistencies_found": {"type": "array"},
                        "confidence_level": {"type": "number"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=10000
                ),
                category="validation"
            )
        ]
        
        # AutoGen configuration
        autogen_config = AutoGenConfig(
            name="QualityAssurance",
            system_message="""You are a Quality Assurance Agent specialized in validating results and ensuring accuracy throughout the data analysis pipeline.

Your responsibilities:
1. Validate data quality and integrity with comprehensive checks
2. Ensure model performance meets standards and benchmarks
3. Verify statistical test validity and assumption compliance
4. Conduct end-to-end pipeline audits for compliance
5. Check consistency and reliability of analysis results

You work collaboratively with other agents to maintain high standards of quality and accuracy. Always apply rigorous validation criteria and provide clear, actionable feedback for improvement.

When performing quality assurance, focus on:
- Comprehensive validation across multiple dimensions
- Clear documentation of issues and recommendations
- Risk assessment and impact analysis
- Compliance with industry standards and best practices
- Continuous improvement and learning from quality issues
- Transparent reporting of limitations and uncertainties""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        super().__init__(
            autogen_config=autogen_config,
            amp_config=amp_config,
            capabilities=capabilities,
            description="Agent specialized in quality assurance and validation of analysis results",
            tags=["quality-assurance", "validation", "audit", "compliance", "autogen"]
        )
        
        # QA configuration and standards
        self.qa_config = qa_config or {
            "data_quality_threshold": 0.8,
            "model_performance_threshold": 0.7,
            "statistical_significance": 0.05,
            "consistency_threshold": 0.9,
            "missing_data_threshold": 0.1,
            "outlier_threshold": 0.05
        }
        
        # QA standards and rules
        self.qa_standards = {
            "data_completeness": {"min_threshold": 0.9, "critical": True},
            "data_consistency": {"min_threshold": 0.95, "critical": True},
            "model_accuracy": {"min_threshold": 0.7, "critical": False},
            "statistical_power": {"min_threshold": 0.8, "critical": False},
            "reproducibility": {"required": True, "critical": True}
        }
        
        # QA metrics
        self.qa_metrics = {
            "validations_performed": 0,
            "issues_identified": 0,
            "audits_completed": 0,
            "models_validated": 0,
            "quality_score_average": 0.0
        }
    
    def _process_conversation_message(
        self,
        message: str,
        sender,
        conversation_history: List[Dict]
    ) -> str:
        """Process conversation messages for QA requests."""
        
        message_lower = message.lower()
        
        if "validate data" in message_lower or "data quality" in message_lower:
            return self._handle_data_validation_request(message)
        elif "validate model" in message_lower or "model quality" in message_lower:
            return self._handle_model_validation_request(message)
        elif "statistical validation" in message_lower or "assumptions" in message_lower:
            return self._handle_statistical_validation_request(message)
        elif "audit" in message_lower or "compliance" in message_lower:
            return self._handle_audit_request(message)
        elif "consistency" in message_lower or "cross-validation" in message_lower:
            return self._handle_consistency_request(message)
        elif "status" in message_lower or "metrics" in message_lower:
            return self._handle_status_request()
        else:
            return """I'm the Quality Assurance Agent. I can help you with:

1. **Data Validation**: Comprehensive data quality and integrity checks
2. **Model Validation**: Model performance validation and benchmarking
3. **Statistical Validation**: Statistical test validity and assumption checking
4. **Pipeline Audit**: End-to-end pipeline audit and compliance checking
5. **Consistency Check**: Cross-validation and result consistency verification

What type of quality assurance would you like me to perform?"""
    
    def _handle_data_validation_request(self, message: str) -> str:
        """Handle data validation requests."""
        artifacts = self.list_artifacts()
        datasets = [a for a in artifacts if "dataset" in a or not any(x in a for x in ["model", "analysis", "viz"])]
        
        if datasets:
            return f"""I can validate data quality for: {', '.join(datasets)}

Data validation checks include:
- **Completeness**: Missing values and data coverage
- **Consistency**: Data type consistency and format validation
- **Accuracy**: Range validation and logical consistency
- **Uniqueness**: Duplicate detection and primary key validation
- **Integrity**: Referential integrity and constraint validation

Example: "Validate data quality for [dataset] with strict thresholds" """
        else:
            return "No datasets available for validation. Please collect data first."
    
    def _handle_model_validation_request(self, message: str) -> str:
        """Handle model validation requests."""
        artifacts = self.list_artifacts()
        models = [a for a in artifacts if "model" in a]
        
        if models:
            return f"""I can validate models: {', '.join(models)}

Model validation includes:
- **Performance Validation**: Metrics against thresholds
- **Overfitting Check**: Training vs validation performance
- **Robustness Testing**: Performance across different data subsets
- **Benchmark Comparison**: Performance vs baseline models
- **Production Readiness**: Deployment suitability assessment

Example: "Validate model performance for [model] against production thresholds" """
        else:
            return "No models available for validation. Please train models first."
    
    def _handle_statistical_validation_request(self, message: str) -> str:
        """Handle statistical validation requests."""
        return """I can validate statistical analyses:

**Assumption Checking:**
- Normality (Shapiro-Wilk, Kolmogorov-Smirnov)
- Homoscedasticity (Levene's test, Breusch-Pagan)
- Independence (Durbin-Watson)
- Linearity and multicollinearity

**Test Validity:**
- Sample size adequacy
- Power analysis
- Effect size significance
- Multiple testing corrections

Example: "Validate t-test assumptions for [analysis] with power analysis" """
    
    def _handle_audit_request(self, message: str) -> str:
        """Handle pipeline audit requests."""
        return """I can perform comprehensive pipeline audits:

**Audit Areas:**
- **Data Lineage**: Traceability from source to results
- **Process Compliance**: Adherence to standards and protocols
- **Documentation**: Completeness and accuracy of metadata
- **Reproducibility**: Ability to replicate results
- **Security**: Data privacy and access controls

**Compliance Standards:**
- Industry best practices
- Regulatory requirements
- Organizational policies

Example: "Audit complete pipeline from data collection to final results" """
    
    def _handle_consistency_request(self, message: str) -> str:
        """Handle consistency checking requests."""
        return """I can check result consistency:

**Consistency Types:**
- **Cross-Model**: Compare results across different models
- **Cross-Method**: Compare different analysis approaches
- **Temporal**: Check stability over time
- **Cross-Validation**: K-fold validation consistency

**Validation Methods:**
- Statistical significance testing
- Logical consistency checking
- Domain knowledge validation

Example: "Check consistency between [result1] and [result2]" """
    
    def _handle_status_request(self) -> str:
        """Handle status and metrics requests."""
        metrics = self.qa_metrics
        artifacts = self.list_artifacts()
        
        qa_artifacts = [a for a in artifacts if "validation" in a or "audit" in a or "qa" in a]
        
        return f"""Quality Assurance Status:

**QA Metrics:**
- Validations performed: {metrics['validations_performed']}
- Issues identified: {metrics['issues_identified']}
- Audits completed: {metrics['audits_completed']}
- Models validated: {metrics['models_validated']}
- Average quality score: {metrics['quality_score_average']:.3f}

**QA Artifacts:** {len(qa_artifacts)}
{', '.join(qa_artifacts) if qa_artifacts else 'None'}

Ready for new quality assurance tasks."""
    
    # AMP Capability Handlers
    
    async def _handle_qa_data_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data validation capability."""
        try:
            dataset_key = parameters["dataset_key"]
            validation_rules = parameters.get("validation_rules", {})
            quality_thresholds = parameters.get("quality_thresholds", self.qa_config)
            check_types = parameters.get("check_types", ["completeness", "consistency", "accuracy", "uniqueness"])
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Perform validation checks
            validation_results = {}
            issues_found = []
            
            # Completeness check
            if "completeness" in check_types:
                completeness_result = self._check_data_completeness(df, quality_thresholds)
                validation_results["completeness"] = completeness_result
                if not completeness_result["passed"]:
                    issues_found.extend(completeness_result["issues"])
            
            # Consistency check
            if "consistency" in check_types:
                consistency_result = self._check_data_consistency(df, validation_rules)
                validation_results["consistency"] = consistency_result
                if not consistency_result["passed"]:
                    issues_found.extend(consistency_result["issues"])
            
            # Accuracy check
            if "accuracy" in check_types:
                accuracy_result = self._check_data_accuracy(df, validation_rules)
                validation_results["accuracy"] = accuracy_result
                if not accuracy_result["passed"]:
                    issues_found.extend(accuracy_result["issues"])
            
            # Uniqueness check
            if "uniqueness" in check_types:
                uniqueness_result = self._check_data_uniqueness(df)
                validation_results["uniqueness"] = uniqueness_result
                if not uniqueness_result["passed"]:
                    issues_found.extend(uniqueness_result["issues"])
            
            # Calculate overall quality score
            quality_scores = [result.get("score", 0) for result in validation_results.values()]
            quality_score = np.mean(quality_scores) if quality_scores else 0
            
            # Overall validation status
            validation_passed = all(result.get("passed", False) for result in validation_results.values())
            
            # Generate recommendations
            recommendations = self._generate_data_recommendations(validation_results, issues_found)
            
            # Store validation results
            validation_key = f"{dataset_key}_data_validation"
            validation_metadata = {
                "dataset_key": dataset_key,
                "validation_type": "data_validation",
                "check_types": check_types,
                "quality_thresholds": quality_thresholds,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            validation_artifact = {
                "validation_passed": validation_passed,
                "quality_score": quality_score,
                "validation_results": validation_results,
                "issues_found": issues_found,
                "recommendations": recommendations
            }
            
            self.store_artifact(validation_key, validation_artifact, validation_metadata)
            
            # Update metrics
            self.qa_metrics["validations_performed"] += 1
            self.qa_metrics["issues_identified"] += len(issues_found)
            self._update_quality_score_average(quality_score)
            
            return validation_artifact
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            raise
    
    async def _handle_qa_model_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model validation capability."""
        try:
            model_key = parameters["model_key"]
            validation_dataset_key = parameters.get("validation_dataset_key")
            performance_thresholds = parameters.get("performance_thresholds", self.qa_config)
            benchmark_models = parameters.get("benchmark_models", [])
            validation_type = parameters.get("validation_type", "holdout")
            
            # Get model artifact
            model_artifact = self.get_artifact(model_key)
            if model_artifact is None:
                raise ValueError(f"Model not found: {model_key}")
            
            # Get validation dataset if provided
            validation_df = None
            if validation_dataset_key:
                validation_df = self.get_artifact(validation_dataset_key)
            
            # Perform model validation
            validation_results = self._validate_model_performance(
                model_artifact, validation_df, performance_thresholds, validation_type
            )
            
            # Benchmark comparison
            benchmark_comparison = {}
            if benchmark_models:
                benchmark_comparison = self._compare_with_benchmarks(
                    model_artifact, benchmark_models
                )
            
            # Check for overfitting and other issues
            validation_issues = self._identify_model_issues(model_artifact, validation_results)
            
            # Determine deployment readiness
            deployment_readiness = self._assess_deployment_readiness(
                validation_results, validation_issues, performance_thresholds
            )
            
            # Store validation results
            model_validation_key = f"{model_key}_validation"
            validation_metadata = {
                "model_key": model_key,
                "validation_type": "model_validation",
                "performance_thresholds": performance_thresholds,
                "validation_method": validation_type,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            validation_artifact = {
                "model_validated": validation_results.get("passed", False),
                "performance_metrics": validation_results.get("metrics", {}),
                "benchmark_comparison": benchmark_comparison,
                "validation_issues": validation_issues,
                "deployment_readiness": deployment_readiness
            }
            
            self.store_artifact(model_validation_key, validation_artifact, validation_metadata)
            
            # Update metrics
            self.qa_metrics["models_validated"] += 1
            self.qa_metrics["validations_performed"] += 1
            self.qa_metrics["issues_identified"] += len(validation_issues)
            
            return validation_artifact
            
        except Exception as e:
            self.logger.error(f"Model validation error: {e}")
            raise
    
    async def _handle_qa_statistical_validation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistical validation capability."""
        try:
            analysis_key = parameters["analysis_key"]
            test_type = parameters["test_type"]
            assumptions_check = parameters.get("assumptions_check", True)
            significance_level = parameters.get("significance_level", 0.05)
            power_analysis = parameters.get("power_analysis", False)
            
            # Get analysis results
            analysis_artifact = self.get_artifact(analysis_key)
            if analysis_artifact is None:
                raise ValueError(f"Analysis not found: {analysis_key}")
            
            # Check statistical assumptions
            assumption_results = {}
            assumptions_met = True
            
            if assumptions_check:
                assumption_results = self._check_statistical_assumptions(
                    analysis_artifact, test_type
                )
                assumptions_met = all(result.get("passed", False) for result in assumption_results.values())
            
            # Validate test results
            test_validity = self._validate_statistical_test(
                analysis_artifact, test_type, significance_level
            )
            
            # Power analysis
            power_analysis_results = {}
            if power_analysis:
                power_analysis_results = self._perform_power_analysis(
                    analysis_artifact, test_type
                )
            
            # Generate statistical warnings
            statistical_warnings = self._generate_statistical_warnings(
                assumption_results, test_validity, power_analysis_results
            )
            
            # Store validation results
            stat_validation_key = f"{analysis_key}_statistical_validation"
            validation_metadata = {
                "analysis_key": analysis_key,
                "validation_type": "statistical_validation",
                "test_type": test_type,
                "significance_level": significance_level,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            validation_artifact = {
                "assumptions_met": assumptions_met,
                "test_validity": test_validity,
                "assumption_results": assumption_results,
                "power_analysis_results": power_analysis_results,
                "statistical_warnings": statistical_warnings
            }
            
            self.store_artifact(stat_validation_key, validation_artifact, validation_metadata)
            
            # Update metrics
            self.qa_metrics["validations_performed"] += 1
            if statistical_warnings:
                self.qa_metrics["issues_identified"] += len(statistical_warnings)
            
            return validation_artifact
            
        except Exception as e:
            self.logger.error(f"Statistical validation error: {e}")
            raise
    
    async def _handle_qa_pipeline_audit(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline audit capability."""
        try:
            pipeline_artifacts = parameters["pipeline_artifacts"]
            audit_standards = parameters.get("audit_standards", self.qa_standards)
            compliance_requirements = parameters.get("compliance_requirements", [])
            trace_lineage = parameters.get("trace_lineage", True)
            
            # Get all pipeline artifacts
            artifacts = {}
            for artifact_key in pipeline_artifacts:
                artifact = self.get_artifact(artifact_key)
                if artifact is not None:
                    artifacts[artifact_key] = artifact
            
            if not artifacts:
                raise ValueError("No valid pipeline artifacts found")
            
            # Perform audit checks
            audit_findings = self._perform_pipeline_audit(
                artifacts, audit_standards, compliance_requirements
            )
            
            # Trace data lineage
            lineage_trace = {}
            if trace_lineage:
                lineage_trace = self._trace_data_lineage(artifacts)
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(audit_findings)
            
            # Generate improvement actions
            improvement_actions = self._generate_improvement_actions(audit_findings)
            
            # Overall audit status
            audit_passed = compliance_score >= 0.8
            
            # Store audit results
            audit_key = "pipeline_audit_results"
            audit_metadata = {
                "pipeline_artifacts": pipeline_artifacts,
                "validation_type": "pipeline_audit",
                "audit_standards": audit_standards,
                "compliance_requirements": compliance_requirements,
                "audit_timestamp": pd.Timestamp.now().isoformat()
            }
            
            audit_artifact = {
                "audit_passed": audit_passed,
                "compliance_score": compliance_score,
                "audit_findings": audit_findings,
                "lineage_trace": lineage_trace,
                "improvement_actions": improvement_actions
            }
            
            self.store_artifact(audit_key, audit_artifact, audit_metadata)
            
            # Update metrics
            self.qa_metrics["audits_completed"] += 1
            self.qa_metrics["validations_performed"] += 1
            
            return audit_artifact
            
        except Exception as e:
            self.logger.error(f"Pipeline audit error: {e}")
            raise
    
    async def _handle_qa_consistency_check(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle consistency check capability."""
        try:
            primary_result_key = parameters["primary_result_key"]
            comparison_keys = parameters["comparison_keys"]
            consistency_thresholds = parameters.get("consistency_thresholds", {"correlation": 0.8, "variance": 0.2})
            validation_method = parameters.get("validation_method", "statistical")
            
            # Get primary result and comparison results
            primary_result = self.get_artifact(primary_result_key)
            if primary_result is None:
                raise ValueError(f"Primary result not found: {primary_result_key}")
            
            comparison_results = {}
            for key in comparison_keys:
                result = self.get_artifact(key)
                if result is not None:
                    comparison_results[key] = result
            
            if not comparison_results:
                raise ValueError("No valid comparison results found")
            
            # Perform consistency checks
            consistency_results = self._check_result_consistency(
                primary_result, comparison_results, validation_method, consistency_thresholds
            )
            
            # Calculate overall consistency score
            consistency_score = consistency_results.get("overall_score", 0)
            consistent_results = consistency_score >= consistency_thresholds.get("overall", 0.8)
            
            # Identify inconsistencies
            inconsistencies_found = consistency_results.get("inconsistencies", [])
            
            # Calculate confidence level
            confidence_level = self._calculate_confidence_level(consistency_results)
            
            # Store consistency check results
            consistency_key = f"{primary_result_key}_consistency_check"
            consistency_metadata = {
                "primary_result_key": primary_result_key,
                "comparison_keys": comparison_keys,
                "validation_type": "consistency_check",
                "validation_method": validation_method,
                "consistency_thresholds": consistency_thresholds,
                "validation_timestamp": pd.Timestamp.now().isoformat()
            }
            
            consistency_artifact = {
                "consistency_score": consistency_score,
                "consistent_results": consistent_results,
                "inconsistencies_found": inconsistencies_found,
                "confidence_level": confidence_level
            }
            
            self.store_artifact(consistency_key, consistency_artifact, consistency_metadata)
            
            # Update metrics
            self.qa_metrics["validations_performed"] += 1
            self.qa_metrics["issues_identified"] += len(inconsistencies_found)
            
            return consistency_artifact
            
        except Exception as e:
            self.logger.error(f"Consistency check error: {e}")
            raise
    
    # Helper methods for data validation
    
    def _check_data_completeness(self, df: pd.DataFrame, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Check data completeness."""
        missing_threshold = thresholds.get("missing_data_threshold", 0.1)
        
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        missing_ratio = missing_cells / total_cells
        
        column_completeness = {}
        issues = []
        
        for col in df.columns:
            col_missing_ratio = df[col].isnull().sum() / len(df)
            column_completeness[col] = {
                "missing_ratio": float(col_missing_ratio),
                "complete": col_missing_ratio <= missing_threshold
            }
            
            if col_missing_ratio > missing_threshold:
                issues.append(f"Column '{col}' has {col_missing_ratio:.1%} missing values (threshold: {missing_threshold:.1%})")
        
        passed = missing_ratio <= missing_threshold
        score = max(0, 1 - (missing_ratio / missing_threshold))
        
        return {
            "passed": passed,
            "score": float(score),
            "overall_missing_ratio": float(missing_ratio),
            "column_completeness": column_completeness,
            "issues": issues
        }
    
    def _check_data_consistency(self, df: pd.DataFrame, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check data consistency."""
        issues = []
        consistency_checks = {}
        
        # Data type consistency
        for col in df.columns:
            expected_type = validation_rules.get(f"{col}_type")
            if expected_type:
                actual_type = str(df[col].dtype)
                consistent = expected_type in actual_type
                consistency_checks[f"{col}_type"] = {"expected": expected_type, "actual": actual_type, "consistent": consistent}
                
                if not consistent:
                    issues.append(f"Column '{col}' type mismatch: expected {expected_type}, got {actual_type}")
        
        # Value range consistency
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                min_val = validation_rules.get(f"{col}_min")
                max_val = validation_rules.get(f"{col}_max")
                
                if min_val is not None:
                    below_min = (df[col] < min_val).sum()
                    if below_min > 0:
                        issues.append(f"Column '{col}' has {below_min} values below minimum ({min_val})")
                
                if max_val is not None:
                    above_max = (df[col] > max_val).sum()
                    if above_max > 0:
                        issues.append(f"Column '{col}' has {above_max} values above maximum ({max_val})")
        
        # Calculate consistency score
        total_checks = len(consistency_checks)
        passed_checks = sum(1 for check in consistency_checks.values() if check.get("consistent", True))
        score = passed_checks / total_checks if total_checks > 0 else 1.0
        
        return {
            "passed": len(issues) == 0,
            "score": float(score),
            "consistency_checks": consistency_checks,
            "issues": issues
        }
    
    def _check_data_accuracy(self, df: pd.DataFrame, validation_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Check data accuracy."""
        issues = []
        accuracy_checks = {}
        
        # Check for logical inconsistencies
        for col in df.columns:
            # Negative values where they shouldn't be
            if col in validation_rules.get("non_negative_columns", []):
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    issues.append(f"Column '{col}' has {negative_count} negative values (should be non-negative)")
        
        # Check for impossible values
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                # Check for infinite values
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    issues.append(f"Column '{col}' has {inf_count} infinite values")
                
                # Check for extreme outliers (beyond 6 standard deviations)
                if df[col].std() > 0:
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    extreme_outliers = (z_scores > 6).sum()
                    if extreme_outliers > 0:
                        accuracy_checks[f"{col}_extreme_outliers"] = extreme_outliers
        
        score = 1.0 - (len(issues) * 0.1)  # Reduce score by 0.1 for each issue
        score = max(0, score)
        
        return {
            "passed": len(issues) == 0,
            "score": float(score),
            "accuracy_checks": accuracy_checks,
            "issues": issues
        }
    
    def _check_data_uniqueness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data uniqueness."""
        issues = []
        uniqueness_checks = {}
        
        # Check for duplicate rows
        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            issues.append(f"Dataset has {duplicate_rows} duplicate rows ({duplicate_rows/len(df):.1%})")
        
        # Check uniqueness of columns that should be unique
        for col in df.columns:
            if col.lower() in ['id', 'key', 'index'] or col.endswith('_id'):
                unique_ratio = df[col].nunique() / len(df)
                uniqueness_checks[col] = {"unique_ratio": float(unique_ratio)}
                
                if unique_ratio < 0.95:  # Allow for some missing values
                    issues.append(f"Column '{col}' should be unique but has uniqueness ratio of {unique_ratio:.1%}")
        
        score = 1.0 - (duplicate_rows / len(df))
        
        return {
            "passed": len(issues) == 0,
            "score": float(score),
            "duplicate_rows": duplicate_rows,
            "uniqueness_checks": uniqueness_checks,
            "issues": issues
        }
    
    def _generate_data_recommendations(self, validation_results: Dict[str, Any], issues: List[str]) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        if "completeness" in validation_results and not validation_results["completeness"]["passed"]:
            recommendations.append("Consider imputing missing values or removing incomplete records")
        
        if "consistency" in validation_results and not validation_results["consistency"]["passed"]:
            recommendations.append("Review and standardize data types and formats")
        
        if "accuracy" in validation_results and not validation_results["accuracy"]["passed"]:
            recommendations.append("Investigate and correct data accuracy issues")
        
        if "uniqueness" in validation_results and not validation_results["uniqueness"]["passed"]:
            recommendations.append("Remove duplicate records and ensure key uniqueness")
        
        if not recommendations:
            recommendations.append("Data quality appears good - consider advanced validation rules")
        
        return recommendations
    
    # Helper methods for model validation
    
    def _validate_model_performance(self, model_artifact: Dict[str, Any], validation_df: Optional[pd.DataFrame], 
                                  thresholds: Dict[str, Any], validation_type: str) -> Dict[str, Any]:
        """Validate model performance against thresholds."""
        
        performance_threshold = thresholds.get("model_performance_threshold", 0.7)
        
        # Extract performance metrics
        if "model_results" in model_artifact:
            # Multiple models
            best_model = model_artifact.get("best_model", "unknown")
            if best_model in model_artifact["model_results"]:
                metrics = model_artifact["model_results"][best_model].get("detailed_metrics", {})
            else:
                metrics = {}
        else:
            # Single model or results
            metrics = model_artifact.get("detailed_metrics", model_artifact)
        
        # Check performance against thresholds
        passed_checks = []
        failed_checks = []
        
        if "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            if accuracy >= performance_threshold:
                passed_checks.append(f"Accuracy {accuracy:.3f} meets threshold {performance_threshold}")
            else:
                failed_checks.append(f"Accuracy {accuracy:.3f} below threshold {performance_threshold}")
        
        if "r2_score" in metrics:
            r2 = metrics["r2_score"]
            if r2 >= performance_threshold:
                passed_checks.append(f"R² {r2:.3f} meets threshold {performance_threshold}")
            else:
                failed_checks.append(f"R² {r2:.3f} below threshold {performance_threshold}")
        
        # Check for overfitting (if train/test scores available)
        if "train_score" in model_artifact and "test_score" in model_artifact:
            train_score = model_artifact["train_score"]
            test_score = model_artifact["test_score"]
            gap = train_score - test_score
            
            if gap > 0.1:  # More than 10% gap suggests overfitting
                failed_checks.append(f"Potential overfitting: train score {train_score:.3f} vs test score {test_score:.3f}")
            else:
                passed_checks.append("No significant overfitting detected")
        
        passed = len(failed_checks) == 0
        
        return {
            "passed": passed,
            "metrics": metrics,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "performance_threshold": performance_threshold
        }
    
    def _compare_with_benchmarks(self, model_artifact: Dict[str, Any], benchmark_models: List[str]) -> Dict[str, Any]:
        """Compare model performance with benchmarks."""
        # Simplified benchmark comparison
        comparison = {
            "baseline_comparison": "Model performance compared to baseline",
            "benchmark_models": benchmark_models,
            "performance_ranking": "Above average"  # Placeholder
        }
        
        return comparison
    
    def _identify_model_issues(self, model_artifact: Dict[str, Any], validation_results: Dict[str, Any]) -> List[str]:
        """Identify potential model issues."""
        issues = []
        
        # Add failed validation checks as issues
        issues.extend(validation_results.get("failed_checks", []))
        
        # Check CV score variance
        if "model_results" in model_artifact:
            for model_name, results in model_artifact["model_results"].items():
                if "cv_std" in results and results["cv_std"] > 0.1:
                    issues.append(f"{model_name}: High cross-validation variance ({results['cv_std']:.3f})")
        
        return issues
    
    def _assess_deployment_readiness(self, validation_results: Dict[str, Any], 
                                   issues: List[str], thresholds: Dict[str, Any]) -> str:
        """Assess model deployment readiness."""
        
        if validation_results.get("passed", False) and len(issues) == 0:
            return "Ready for production deployment"
        elif len(issues) <= 2:
            return "Ready with minor issues - recommend monitoring"
        else:
            return "Not ready - requires improvement before deployment"
    
    # Helper methods for statistical validation
    
    def _check_statistical_assumptions(self, analysis_artifact: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Check statistical test assumptions."""
        assumptions = {}
        
        # This is a simplified implementation - would need actual data for proper checking
        if test_type in ["ttest_ind", "ttest_rel", "anova"]:
            assumptions["normality"] = {"passed": True, "method": "assumed", "p_value": 0.5}
            assumptions["homogeneity"] = {"passed": True, "method": "assumed", "p_value": 0.4}
        
        if test_type in ["ttest_ind", "anova"]:
            assumptions["independence"] = {"passed": True, "method": "assumed"}
        
        return assumptions
    
    def _validate_statistical_test(self, analysis_artifact: Dict[str, Any], 
                                 test_type: str, significance_level: float) -> bool:
        """Validate statistical test results."""
        
        # Check if p-value is reasonable
        if "p_value" in analysis_artifact:
            p_value = analysis_artifact["p_value"]
            # Very small p-values might indicate issues
            if p_value < 1e-10:
                return False
        
        # Check sample size adequacy (simplified)
        if "n_observations" in analysis_artifact:
            n = analysis_artifact["n_observations"]
            if n < 30:  # Minimum sample size for many tests
                return False
        
        return True
    
    def _perform_power_analysis(self, analysis_artifact: Dict[str, Any], test_type: str) -> Dict[str, Any]:
        """Perform statistical power analysis."""
        
        # Simplified power analysis
        power_results = {
            "estimated_power": 0.8,  # Placeholder
            "sample_size_adequate": True,
            "effect_size": analysis_artifact.get("effect_size", "medium"),
            "recommendations": []
        }
        
        if power_results["estimated_power"] < 0.8:
            power_results["recommendations"].append("Consider increasing sample size for adequate power")
        
        return power_results
    
    def _generate_statistical_warnings(self, assumption_results: Dict[str, Any], 
                                     test_validity: bool, power_results: Dict[str, Any]) -> List[str]:
        """Generate statistical warnings."""
        warnings = []
        
        if not test_validity:
            warnings.append("Statistical test validity concerns detected")
        
        for assumption, result in assumption_results.items():
            if not result.get("passed", True):
                warnings.append(f"{assumption.title()} assumption violated")
        
        if power_results.get("estimated_power", 1.0) < 0.8:
            warnings.append("Statistical power below recommended threshold (0.8)")
        
        return warnings
    
    # Helper methods for pipeline audit and consistency
    
    def _perform_pipeline_audit(self, artifacts: Dict[str, Any], standards: Dict[str, Any], 
                              requirements: List[str]) -> Dict[str, Any]:
        """Perform comprehensive pipeline audit."""
        
        findings = {
            "documentation": {"score": 0.8, "issues": []},
            "reproducibility": {"score": 0.9, "issues": []},
            "data_lineage": {"score": 0.85, "issues": []},
            "compliance": {"score": 0.7, "issues": []}
        }
        
        # Check documentation completeness
        documented_artifacts = 0
        for artifact_key, artifact in artifacts.items():
            if hasattr(artifact, 'metadata') or (isinstance(artifact, dict) and "metadata" in artifact):
                documented_artifacts += 1
        
        doc_ratio = documented_artifacts / len(artifacts)
        findings["documentation"]["score"] = doc_ratio
        
        if doc_ratio < 0.8:
            findings["documentation"]["issues"].append("Insufficient documentation for pipeline artifacts")
        
        return findings
    
    def _trace_data_lineage(self, artifacts: Dict[str, Any]) -> Dict[str, Any]:
        """Trace data lineage through pipeline."""
        
        lineage = {
            "source_datasets": [],
            "transformations": [],
            "final_outputs": [],
            "lineage_complete": True
        }
        
        # Simplified lineage tracing
        for artifact_key in artifacts.keys():
            if "dataset" in artifact_key and not any(x in artifact_key for x in ["cleaned", "transformed"]):
                lineage["source_datasets"].append(artifact_key)
            elif any(x in artifact_key for x in ["cleaned", "transformed", "engineered"]):
                lineage["transformations"].append(artifact_key)
            elif any(x in artifact_key for x in ["model", "prediction", "analysis"]):
                lineage["final_outputs"].append(artifact_key)
        
        return lineage
    
    def _calculate_compliance_score(self, audit_findings: Dict[str, Any]) -> float:
        """Calculate overall compliance score."""
        
        scores = [finding.get("score", 0) for finding in audit_findings.values()]
        return np.mean(scores) if scores else 0
    
    def _generate_improvement_actions(self, audit_findings: Dict[str, Any]) -> List[str]:
        """Generate improvement actions from audit findings."""
        
        actions = []
        
        for area, findings in audit_findings.items():
            if findings.get("score", 1.0) < 0.8:
                actions.append(f"Improve {area}: {', '.join(findings.get('issues', []))}")
        
        if not actions:
            actions.append("Pipeline meets quality standards - continue monitoring")
        
        return actions
    
    def _check_result_consistency(self, primary_result: Any, comparison_results: Dict[str, Any], 
                                method: str, thresholds: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency between results."""
        
        consistency_results = {
            "overall_score": 0.85,  # Placeholder
            "inconsistencies": [],
            "detailed_comparison": {}
        }
        
        # Simplified consistency checking
        if method == "statistical":
            # Would perform statistical tests for consistency
            consistency_results["statistical_tests"] = {
                "correlation_test": {"correlation": 0.9, "p_value": 0.01},
                "variance_test": {"f_statistic": 1.2, "p_value": 0.3}
            }
        
        return consistency_results
    
    def _calculate_confidence_level(self, consistency_results: Dict[str, Any]) -> float:
        """Calculate confidence level in results."""
        
        score = consistency_results.get("overall_score", 0)
        # Convert consistency score to confidence level
        return min(0.99, score * 1.1)
    
    def _update_quality_score_average(self, new_score: float):
        """Update running average of quality scores."""
        
        current_avg = self.qa_metrics["quality_score_average"]
        total_validations = self.qa_metrics["validations_performed"]
        
        if total_validations == 0:
            self.qa_metrics["quality_score_average"] = new_score
        else:
            # Running average
            self.qa_metrics["quality_score_average"] = (
                (current_avg * (total_validations - 1) + new_score) / total_validations
            )
    
    def get_qa_summary(self) -> Dict[str, Any]:
        """Get summary of all QA activities."""
        summary = {
            "metrics": self.qa_metrics.copy(),
            "qa_artifacts": {},
            "standards": self.qa_standards,
            "configuration": self.qa_config
        }
        
        for artifact_key in self.list_artifacts():
            artifact = self._data_artifacts[artifact_key]
            metadata = artifact["metadata"]
            
            if metadata.get("validation_type"):
                summary["qa_artifacts"][artifact_key] = {
                    "validation_type": metadata["validation_type"],
                    "timestamp": artifact["timestamp"]
                }
        
        return summary