"""
ML Analyst Agent.

Specialized AutoGen agent for machine learning analysis including predictive modeling,
feature engineering, model evaluation, and automated ML workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, RFE
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
)
import joblib
import warnings

from .base_agent import AutoGenAMPAgent, AutoGenConfig
from amp_types import Capability, CapabilityConstraints
from amp_client import AMPClientConfig

warnings.filterwarnings('ignore')


class MLAnalystAgent(AutoGenAMPAgent):
    """
    Agent specialized in machine learning analysis and predictive modeling.
    
    Capabilities:
    - Automated feature engineering and selection
    - Model training and hyperparameter tuning
    - Model evaluation and comparison
    - Predictive modeling (classification/regression)
    - Feature importance analysis
    - Cross-validation and model validation
    """
    
    def __init__(
        self,
        amp_config: AMPClientConfig,
        llm_config: Dict[str, Any],
        ml_config: Dict[str, Any] = None
    ):
        """
        Initialize ML Analyst Agent.
        
        Args:
            amp_config: AMP client configuration
            llm_config: LLM configuration for AutoGen
            ml_config: Configuration for ML operations
        """
        # Define capabilities
        capabilities = [
            Capability(
                id="ml-feature-engineering",
                version="1.0",
                description="Automated feature engineering and selection",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "target_variable": {"type": "string"},
                        "feature_operations": {"type": "array", "items": {"type": "string"}},
                        "selection_method": {"type": "string", "enum": ["univariate", "rfe", "importance", "pca"]},
                        "n_features": {"type": "integer", "minimum": 1}
                    },
                    "required": ["dataset_key", "target_variable"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "engineered_features": {"type": "array"},
                        "selected_features": {"type": "array"},
                        "feature_scores": {"type": "object"},
                        "transformed_dataset_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=15000
                ),
                category="data-processing"
            ),
            Capability(
                id="ml-model-training",
                version="1.0",
                description="Train and evaluate machine learning models",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "target_variable": {"type": "string"},
                        "model_type": {"type": "string", "enum": ["classification", "regression"]},
                        "algorithms": {"type": "array", "items": {"type": "string"}},
                        "test_size": {"type": "number", "minimum": 0.1, "maximum": 0.5},
                        "cross_validation": {"type": "boolean"},
                        "hyperparameter_tuning": {"type": "boolean"}
                    },
                    "required": ["dataset_key", "target_variable", "model_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "model_results": {"type": "object"},
                        "best_model": {"type": "string"},
                        "performance_metrics": {"type": "object"},
                        "trained_models_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=30000
                ),
                category="analysis"
            ),
            Capability(
                id="ml-model-evaluation",
                version="1.0",
                description="Comprehensive model evaluation and comparison",
                input_schema={
                    "type": "object",
                    "properties": {
                        "models_key": {"type": "string"},
                        "test_dataset_key": {"type": "string"},
                        "evaluation_metrics": {"type": "array", "items": {"type": "string"}},
                        "cross_validation_folds": {"type": "integer", "minimum": 2, "maximum": 10}
                    },
                    "required": ["models_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "evaluation_results": {"type": "object"},
                        "model_comparison": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=20000
                ),
                category="analysis"
            ),
            Capability(
                id="ml-prediction",
                version="1.0",
                description="Make predictions using trained models",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_key": {"type": "string"},
                        "dataset_key": {"type": "string"},
                        "prediction_type": {"type": "string", "enum": ["single", "batch"]},
                        "include_probabilities": {"type": "boolean"},
                        "confidence_intervals": {"type": "boolean"}
                    },
                    "required": ["model_key", "dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "predictions": {"type": "array"},
                        "probabilities": {"type": "array"},
                        "confidence_intervals": {"type": "array"},
                        "prediction_metadata": {"type": "object"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=10000
                ),
                category="generation"
            ),
            Capability(
                id="ml-feature-importance",
                version="1.0",
                description="Analyze feature importance and model interpretability",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_key": {"type": "string"},
                        "dataset_key": {"type": "string"},
                        "importance_method": {"type": "string", "enum": ["built-in", "permutation", "shap"]},
                        "top_n_features": {"type": "integer", "minimum": 1}
                    },
                    "required": ["model_key", "dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "feature_importance": {"type": "object"},
                        "top_features": {"type": "array"},
                        "importance_plot_data": {"type": "object"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=12000
                ),
                category="analysis"
            )
        ]
        
        # AutoGen configuration
        autogen_config = AutoGenConfig(
            name="MLAnalyst",
            system_message="""You are an ML Analyst Agent specialized in machine learning and predictive modeling.

Your responsibilities:
1. Perform automated feature engineering and selection
2. Train and evaluate machine learning models across different algorithms
3. Conduct comprehensive model evaluation and comparison
4. Make predictions using trained models with confidence estimates
5. Analyze feature importance and model interpretability

You work collaboratively with other agents in the data analysis pipeline. Always follow ML best practices including proper train/validation/test splits, cross-validation, and addressing overfitting.

When building ML models, focus on:
- Appropriate feature engineering based on data characteristics
- Selecting suitable algorithms for the problem type
- Proper model validation and evaluation metrics
- Interpretability and explainability of results
- Practical considerations for model deployment
- Ethical implications and bias detection""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        super().__init__(
            autogen_config=autogen_config,
            amp_config=amp_config,
            capabilities=capabilities,
            description="Agent specialized in machine learning analysis and predictive modeling",
            tags=["machine-learning", "modeling", "prediction", "feature-engineering", "autogen"]
        )
        
        # ML configuration
        self.ml_config = ml_config or {
            "default_test_size": 0.2,
            "default_cv_folds": 5,
            "default_random_state": 42,
            "max_features_auto": 50,
            "feature_selection_threshold": 0.01
        }
        
        # Available algorithms
        self.classification_algorithms = {
            "random_forest": RandomForestClassifier,
            "gradient_boosting": GradientBoostingClassifier,
            "logistic_regression": LogisticRegression,
            "svm": SVC,
            "knn": KNeighborsClassifier,
            "decision_tree": DecisionTreeClassifier,
            "naive_bayes": GaussianNB
        }
        
        self.regression_algorithms = {
            "random_forest": RandomForestRegressor,
            "gradient_boosting": GradientBoostingRegressor,
            "linear_regression": LinearRegression,
            "ridge": Ridge,
            "lasso": Lasso,
            "svr": SVR,
            "knn": KNeighborsRegressor,
            "decision_tree": DecisionTreeRegressor
        }
        
        # ML metrics
        self.ml_metrics = {
            "models_trained": 0,
            "features_engineered": 0,
            "predictions_made": 0,
            "evaluations_performed": 0,
            "feature_importance_analyses": 0
        }
    
    def _process_conversation_message(
        self,
        message: str,
        sender,
        conversation_history: List[Dict]
    ) -> str:
        """Process conversation messages for ML analysis requests."""
        
        message_lower = message.lower()
        
        if "feature engineering" in message_lower or "feature selection" in message_lower:
            return self._handle_feature_engineering_request(message)
        elif "train model" in message_lower or "build model" in message_lower:
            return self._handle_model_training_request(message)
        elif "evaluate model" in message_lower or "model performance" in message_lower:
            return self._handle_model_evaluation_request(message)
        elif "predict" in message_lower or "prediction" in message_lower:
            return self._handle_prediction_request(message)
        elif "feature importance" in message_lower or "interpretability" in message_lower:
            return self._handle_feature_importance_request(message)
        elif "status" in message_lower or "metrics" in message_lower:
            return self._handle_status_request()
        else:
            return """I'm the ML Analyst Agent. I can help you with:

1. **Feature Engineering**: Automated feature creation and selection
2. **Model Training**: Train ML models with multiple algorithms and hyperparameter tuning
3. **Model Evaluation**: Comprehensive evaluation and comparison of model performance
4. **Predictions**: Make predictions using trained models with confidence estimates
5. **Feature Importance**: Analyze feature importance and model interpretability

What type of machine learning analysis would you like me to perform?"""
    
    def _handle_feature_engineering_request(self, message: str) -> str:
        """Handle feature engineering requests."""
        artifacts = self.list_artifacts()
        if artifacts:
            return f"""I can perform feature engineering on: {', '.join(artifacts)}

Feature engineering operations:
- **Scaling**: Standardization, normalization
- **Encoding**: One-hot encoding for categorical variables
- **Polynomial**: Create polynomial features
- **Interaction**: Create interaction terms
- **Selection**: Univariate selection, RFE, importance-based, PCA

Example: "Engineer features for [dataset] with target [variable] using polynomial and selection" """
        else:
            return "No datasets available for feature engineering. Please collect and clean data first."
    
    def _handle_model_training_request(self, message: str) -> str:
        """Handle model training requests."""
        return """I can train various ML models:

**Classification Algorithms:**
- Random Forest, Gradient Boosting
- Logistic Regression, SVM
- K-Nearest Neighbors, Decision Tree
- Naive Bayes

**Regression Algorithms:**
- Random Forest, Gradient Boosting
- Linear Regression, Ridge, Lasso
- Support Vector Regression
- K-Nearest Neighbors, Decision Tree

Features:
- Automatic train/validation/test splits
- Cross-validation
- Hyperparameter tuning
- Model comparison

Example: "Train classification models for [target] using random_forest and svm with hyperparameter tuning" """
    
    def _handle_model_evaluation_request(self, message: str) -> str:
        """Handle model evaluation requests."""
        return """I can evaluate models using comprehensive metrics:

**Classification Metrics:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, Confusion Matrix
- Classification Report

**Regression Metrics:**
- R-squared, Mean Squared Error
- Mean Absolute Error, Root MSE

**Validation:**
- Cross-validation scores
- Learning curves
- Model comparison and ranking

Example: "Evaluate [model] using cross-validation with 5 folds" """
    
    def _handle_prediction_request(self, message: str) -> str:
        """Handle prediction requests."""
        return """I can make predictions using trained models:

**Prediction Types:**
- Single instance prediction
- Batch predictions on datasets
- Probability estimates (classification)
- Confidence intervals (regression)

**Output Options:**
- Raw predictions
- Class probabilities
- Prediction confidence
- Feature contributions

Example: "Make predictions on [dataset] using [model] with probabilities" """
    
    def _handle_feature_importance_request(self, message: str) -> str:
        """Handle feature importance requests."""
        return """I can analyze feature importance:

**Importance Methods:**
- Built-in feature importance (tree-based models)
- Permutation importance
- SHAP values (when available)

**Analysis:**
- Ranking of most important features
- Feature contribution plots
- Model interpretability insights

Example: "Analyze feature importance for [model] using permutation method" """
    
    def _handle_status_request(self) -> str:
        """Handle status and metrics requests."""
        metrics = self.ml_metrics
        artifacts = self.list_artifacts()
        
        return f"""ML Analysis Status:

**ML Metrics:**
- Models trained: {metrics['models_trained']}
- Features engineered: {metrics['features_engineered']}
- Predictions made: {metrics['predictions_made']}
- Evaluations performed: {metrics['evaluations_performed']}
- Feature importance analyses: {metrics['feature_importance_analyses']}

**ML Artifacts:** {len(artifacts)}
{', '.join(artifacts) if artifacts else 'None'}

Ready for new machine learning tasks."""
    
    # AMP Capability Handlers
    
    async def _handle_ml_feature_engineering(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature engineering capability."""
        try:
            dataset_key = parameters["dataset_key"]
            target_variable = parameters["target_variable"]
            feature_operations = parameters.get("feature_operations", ["scaling", "encoding"])
            selection_method = parameters.get("selection_method", "univariate")
            n_features = parameters.get("n_features", 10)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            if target_variable not in df.columns:
                raise ValueError(f"Target variable not found: {target_variable}")
            
            # Separate features and target
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            
            # Track engineered features
            engineered_features = []
            
            # Apply feature engineering operations
            if "scaling" in feature_operations:
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    scaler = StandardScaler()
                    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
                    engineered_features.extend([f"{col}_scaled" for col in numeric_cols])
            
            if "encoding" in feature_operations:
                categorical_cols = X.select_dtypes(include=['object', 'category']).columns
                for col in categorical_cols:
                    if X[col].nunique() <= 10:  # One-hot encode low cardinality
                        dummies = pd.get_dummies(X[col], prefix=col)
                        X = pd.concat([X.drop(columns=[col]), dummies], axis=1)
                        engineered_features.extend(dummies.columns.tolist())
                    else:  # Label encode high cardinality
                        le = LabelEncoder()
                        X[f"{col}_encoded"] = le.fit_transform(X[col].astype(str))
                        X = X.drop(columns=[col])
                        engineered_features.append(f"{col}_encoded")
            
            if "polynomial" in feature_operations:
                from sklearn.preprocessing import PolynomialFeatures
                numeric_cols = X.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0 and len(numeric_cols) <= 5:  # Limit to avoid explosion
                    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
                    poly_features = poly.fit_transform(X[numeric_cols])
                    poly_feature_names = poly.get_feature_names_out(numeric_cols)
                    
                    # Add polynomial features
                    for i, name in enumerate(poly_feature_names):
                        if name not in numeric_cols:  # Skip original features
                            X[name] = poly_features[:, i]
                            engineered_features.append(name)
            
            # Feature selection
            selected_features, feature_scores = self._select_features(
                X, y, selection_method, n_features, target_variable
            )
            
            # Create final dataset with selected features
            X_selected = X[selected_features]
            df_transformed = pd.concat([X_selected, y], axis=1)
            
            # Store transformed dataset
            transformed_key = f"{dataset_key}_feature_engineered"
            metadata = {
                "original_dataset": dataset_key,
                "target_variable": target_variable,
                "feature_operations": feature_operations,
                "selection_method": selection_method,
                "n_features_selected": len(selected_features),
                "original_features": X.columns.tolist(),
                "engineered_features": engineered_features,
                "selected_features": selected_features
            }
            
            self.store_artifact(transformed_key, df_transformed, metadata)
            
            # Update metrics
            self.ml_metrics["features_engineered"] += len(engineered_features)
            
            return {
                "engineered_features": engineered_features,
                "selected_features": selected_features,
                "feature_scores": feature_scores,
                "transformed_dataset_key": transformed_key
            }
            
        except Exception as e:
            self.logger.error(f"Feature engineering error: {e}")
            raise
    
    async def _handle_ml_model_training(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model training capability."""
        try:
            dataset_key = parameters["dataset_key"]
            target_variable = parameters["target_variable"]
            model_type = parameters["model_type"]
            algorithms = parameters.get("algorithms", ["random_forest", "logistic_regression"])
            test_size = parameters.get("test_size", self.ml_config["default_test_size"])
            cross_validation = parameters.get("cross_validation", True)
            hyperparameter_tuning = parameters.get("hyperparameter_tuning", False)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            if target_variable not in df.columns:
                raise ValueError(f"Target variable not found: {target_variable}")
            
            # Prepare data
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            
            # Handle missing values (basic)
            X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=self.ml_config["default_random_state"],
                stratify=y if model_type == "classification" and len(y.unique()) > 1 else None
            )
            
            # Select algorithm set
            if model_type == "classification":
                available_algorithms = self.classification_algorithms
            else:
                available_algorithms = self.regression_algorithms
            
            # Train models
            model_results = {}
            trained_models = {}
            
            for algorithm in algorithms:
                if algorithm not in available_algorithms:
                    continue
                
                try:
                    model_class = available_algorithms[algorithm]
                    
                    if hyperparameter_tuning:
                        # Hyperparameter tuning
                        param_grid = self._get_param_grid(algorithm, model_type)
                        model = GridSearchCV(
                            model_class(random_state=self.ml_config["default_random_state"]),
                            param_grid,
                            cv=3,
                            scoring='accuracy' if model_type == "classification" else 'r2',
                            n_jobs=-1
                        )
                        model.fit(X_train, y_train)
                        best_model = model.best_estimator_
                    else:
                        # Default parameters
                        model = model_class(random_state=self.ml_config["default_random_state"])
                        model.fit(X_train, y_train)
                        best_model = model
                    
                    # Evaluate model
                    train_score = best_model.score(X_train, y_train)
                    test_score = best_model.score(X_test, y_test)
                    
                    # Cross-validation
                    cv_scores = None
                    if cross_validation:
                        cv_scores = cross_val_score(
                            best_model, X_train, y_train,
                            cv=self.ml_config["default_cv_folds"],
                            scoring='accuracy' if model_type == "classification" else 'r2'
                        )
                    
                    # Detailed metrics
                    y_pred = best_model.predict(X_test)
                    detailed_metrics = self._calculate_detailed_metrics(
                        y_test, y_pred, model_type, best_model, X_test
                    )
                    
                    model_results[algorithm] = {
                        "train_score": float(train_score),
                        "test_score": float(test_score),
                        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
                        "cv_mean": float(cv_scores.mean()) if cv_scores is not None else None,
                        "cv_std": float(cv_scores.std()) if cv_scores is not None else None,
                        "detailed_metrics": detailed_metrics,
                        "hyperparameters": best_model.get_params() if hasattr(best_model, 'get_params') else {}
                    }
                    
                    trained_models[algorithm] = best_model
                    
                except Exception as e:
                    self.logger.warning(f"Failed to train {algorithm}: {e}")
                    continue
            
            if not model_results:
                raise ValueError("No models were successfully trained")
            
            # Find best model
            if model_type == "classification":
                best_model = max(model_results.keys(), key=lambda x: model_results[x]["test_score"])
            else:
                best_model = max(model_results.keys(), key=lambda x: model_results[x]["test_score"])
            
            # Store trained models
            models_key = f"{dataset_key}_trained_models"
            models_metadata = {
                "original_dataset": dataset_key,
                "target_variable": target_variable,
                "model_type": model_type,
                "algorithms": algorithms,
                "best_model": best_model,
                "test_size": test_size,
                "feature_names": X.columns.tolist()
            }
            
            models_artifact = {
                "models": trained_models,
                "results": model_results,
                "X_test": X_test,
                "y_test": y_test,
                "feature_names": X.columns.tolist()
            }
            
            self.store_artifact(models_key, models_artifact, models_metadata)
            
            # Update metrics
            self.ml_metrics["models_trained"] += len(trained_models)
            
            return {
                "model_results": model_results,
                "best_model": best_model,
                "performance_metrics": model_results[best_model]["detailed_metrics"],
                "trained_models_key": models_key
            }
            
        except Exception as e:
            self.logger.error(f"Model training error: {e}")
            raise
    
    async def _handle_ml_model_evaluation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model evaluation capability."""
        try:
            models_key = parameters["models_key"]
            test_dataset_key = parameters.get("test_dataset_key")
            evaluation_metrics = parameters.get("evaluation_metrics", ["accuracy", "precision", "recall"])
            cv_folds = parameters.get("cross_validation_folds", 5)
            
            # Get trained models
            models_artifact = self.get_artifact(models_key)
            if models_artifact is None:
                raise ValueError(f"Models not found: {models_key}")
            
            trained_models = models_artifact["models"]
            model_results = models_artifact["results"]
            
            # Use provided test dataset or existing test set
            if test_dataset_key:
                test_df = self.get_artifact(test_dataset_key)
                if test_df is None:
                    raise ValueError(f"Test dataset not found: {test_dataset_key}")
                # Extract test features and target (assuming same structure)
                target_var = list(set(test_df.columns) - set(models_artifact["feature_names"]))[0]
                X_test = test_df[models_artifact["feature_names"]]
                y_test = test_df[target_var]
            else:
                X_test = models_artifact["X_test"]
                y_test = models_artifact["y_test"]
            
            # Evaluate each model
            evaluation_results = {}
            model_comparison = []
            
            for model_name, model in trained_models.items():
                try:
                    # Make predictions
                    y_pred = model.predict(X_test)
                    
                    # Determine model type
                    model_type = "classification" if hasattr(model, "predict_proba") else "regression"
                    
                    # Calculate evaluation metrics
                    eval_metrics = self._calculate_evaluation_metrics(
                        y_test, y_pred, model_type, model, X_test, evaluation_metrics
                    )
                    
                    # Cross-validation on training data if available
                    if hasattr(models_artifact, "X_train") and hasattr(models_artifact, "y_train"):
                        X_train = models_artifact["X_train"]
                        y_train = models_artifact["y_train"]
                        cv_scores = cross_val_score(
                            model, X_train, y_train, cv=cv_folds,
                            scoring='accuracy' if model_type == "classification" else 'r2'
                        )
                        eval_metrics["cv_scores"] = cv_scores.tolist()
                        eval_metrics["cv_mean"] = float(cv_scores.mean())
                        eval_metrics["cv_std"] = float(cv_scores.std())
                    
                    evaluation_results[model_name] = eval_metrics
                    
                    # Add to comparison
                    comparison_entry = {
                        "model": model_name,
                        "primary_metric": eval_metrics.get("accuracy", eval_metrics.get("r2", 0)),
                        "detailed_metrics": eval_metrics
                    }
                    model_comparison.append(comparison_entry)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to evaluate {model_name}: {e}")
                    continue
            
            # Sort models by performance
            primary_metric_key = "accuracy" if "accuracy" in evaluation_results.get(list(evaluation_results.keys())[0], {}) else "r2"
            model_comparison.sort(key=lambda x: x["primary_metric"], reverse=True)
            
            # Generate recommendations
            recommendations = self._generate_model_recommendations(evaluation_results, model_comparison)
            
            # Store evaluation results
            eval_key = f"{models_key}_evaluation"
            eval_metadata = {
                "models_key": models_key,
                "test_dataset_key": test_dataset_key,
                "evaluation_metrics": evaluation_metrics,
                "cv_folds": cv_folds
            }
            
            self.store_artifact(eval_key, evaluation_results, eval_metadata)
            
            # Update metrics
            self.ml_metrics["evaluations_performed"] += 1
            
            return {
                "evaluation_results": evaluation_results,
                "model_comparison": model_comparison,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Model evaluation error: {e}")
            raise
    
    async def _handle_ml_prediction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction capability."""
        try:
            model_key = parameters["model_key"]
            dataset_key = parameters["dataset_key"]
            prediction_type = parameters.get("prediction_type", "batch")
            include_probabilities = parameters.get("include_probabilities", False)
            confidence_intervals = parameters.get("confidence_intervals", False)
            
            # Get model and dataset
            if model_key.endswith("_trained_models"):
                # Multiple models - use best one
                models_artifact = self.get_artifact(model_key)
                if models_artifact is None:
                    raise ValueError(f"Models not found: {model_key}")
                
                best_model_name = self._get_artifact(model_key)["metadata"]["best_model"]
                model = models_artifact["models"][best_model_name]
                feature_names = models_artifact["feature_names"]
            else:
                # Single model
                model = self.get_artifact(model_key)
                if model is None:
                    raise ValueError(f"Model not found: {model_key}")
                feature_names = None
            
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Prepare features
            if feature_names:
                X = df[feature_names]
            else:
                # Assume all columns are features
                X = df
            
            # Handle missing values
            X = X.fillna(X.median() if X.select_dtypes(include=[np.number]).shape[1] > 0 else X.mode().iloc[0])
            
            # Make predictions
            predictions = model.predict(X)
            
            # Get probabilities if requested and available
            probabilities = None
            if include_probabilities and hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(X).tolist()
            
            # Calculate confidence intervals (simplified approach)
            conf_intervals = None
            if confidence_intervals:
                if hasattr(model, "predict_proba"):
                    # Classification: use prediction probabilities as confidence
                    proba = model.predict_proba(X)
                    max_proba = np.max(proba, axis=1)
                    conf_intervals = [(p, p) for p in max_proba]  # Simplified
                else:
                    # Regression: use prediction intervals (simplified)
                    # This is a basic approximation
                    pred_std = np.std(predictions) * 0.1  # Rough estimate
                    conf_intervals = [(p - 1.96*pred_std, p + 1.96*pred_std) for p in predictions]
            
            # Store predictions
            pred_key = f"{dataset_key}_predictions"
            pred_metadata = {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "prediction_type": prediction_type,
                "include_probabilities": include_probabilities,
                "confidence_intervals": confidence_intervals,
                "n_predictions": len(predictions)
            }
            
            prediction_results = {
                "predictions": predictions.tolist(),
                "probabilities": probabilities,
                "confidence_intervals": conf_intervals,
                "prediction_metadata": {
                    "model_type": type(model).__name__,
                    "n_features": X.shape[1],
                    "n_predictions": len(predictions)
                }
            }
            
            self.store_artifact(pred_key, prediction_results, pred_metadata)
            
            # Update metrics
            self.ml_metrics["predictions_made"] += len(predictions)
            
            return prediction_results
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            raise
    
    async def _handle_ml_feature_importance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle feature importance analysis capability."""
        try:
            model_key = parameters["model_key"]
            dataset_key = parameters["dataset_key"]
            importance_method = parameters.get("importance_method", "built-in")
            top_n_features = parameters.get("top_n_features", 10)
            
            # Get model and dataset
            if model_key.endswith("_trained_models"):
                models_artifact = self.get_artifact(model_key)
                if models_artifact is None:
                    raise ValueError(f"Models not found: {model_key}")
                
                best_model_name = self._get_artifact(model_key)["metadata"]["best_model"]
                model = models_artifact["models"][best_model_name]
                feature_names = models_artifact["feature_names"]
            else:
                model = self.get_artifact(model_key)
                if model is None:
                    raise ValueError(f"Model not found: {model_key}")
                feature_names = None
            
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Prepare features
            if feature_names:
                X = df[feature_names]
            else:
                X = df.select_dtypes(include=[np.number])
                feature_names = X.columns.tolist()
            
            # Calculate feature importance
            if importance_method == "built-in" and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            elif importance_method == "permutation":
                from sklearn.inspection import permutation_importance
                # Need target variable for permutation importance
                target_cols = [col for col in df.columns if col not in feature_names]
                if target_cols:
                    y = df[target_cols[0]]
                    perm_importance = permutation_importance(model, X, y, n_repeats=10, random_state=42)
                    importances = perm_importance.importances_mean
                else:
                    raise ValueError("Target variable needed for permutation importance")
            else:
                # Default to built-in if available
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                elif hasattr(model, "coef_"):
                    importances = np.abs(model.coef_).flatten()
                else:
                    raise ValueError("Model does not support feature importance calculation")
            
            # Create feature importance dictionary
            feature_importance = dict(zip(feature_names, importances.astype(float)))
            
            # Get top features
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:top_n_features]
            
            # Prepare plot data
            importance_plot_data = {
                "features": [f[0] for f in top_features],
                "importances": [f[1] for f in top_features]
            }
            
            # Store importance analysis
            importance_key = f"{model_key}_feature_importance"
            importance_metadata = {
                "model_key": model_key,
                "dataset_key": dataset_key,
                "importance_method": importance_method,
                "top_n_features": top_n_features
            }
            
            importance_results = {
                "feature_importance": feature_importance,
                "top_features": [{"feature": f[0], "importance": f[1]} for f in top_features],
                "importance_plot_data": importance_plot_data
            }
            
            self.store_artifact(importance_key, importance_results, importance_metadata)
            
            # Update metrics
            self.ml_metrics["feature_importance_analyses"] += 1
            
            return importance_results
            
        except Exception as e:
            self.logger.error(f"Feature importance error: {e}")
            raise
    
    # Helper methods
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, method: str, n_features: int, target_var: str) -> Tuple[List[str], Dict[str, float]]:
        """Select features using specified method."""
        
        # Determine problem type
        is_classification = len(y.unique()) < 20 and y.dtype in ['object', 'int64', 'category']
        
        if method == "univariate":
            # Univariate feature selection
            if is_classification:
                selector = SelectKBest(score_func=f_classif, k=min(n_features, X.shape[1]))
            else:
                selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_scores = dict(zip(X.columns, selector.scores_))
            
        elif method == "rfe":
            # Recursive feature elimination
            if is_classification:
                estimator = RandomForestClassifier(n_estimators=10, random_state=42)
            else:
                estimator = RandomForestRegressor(n_estimators=10, random_state=42)
            
            selector = RFE(estimator, n_features_to_select=min(n_features, X.shape[1]))
            selector.fit(X, y)
            selected_features = X.columns[selector.support_].tolist()
            feature_scores = dict(zip(X.columns, selector.ranking_))
            
        elif method == "importance":
            # Feature importance based selection
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            importances = model.feature_importances_
            feature_scores = dict(zip(X.columns, importances))
            
            # Select top N features
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
            selected_features = [f[0] for f in top_features]
            
        elif method == "pca":
            # PCA dimensionality reduction
            pca = PCA(n_components=min(n_features, X.shape[1]))
            X_pca = pca.fit_transform(X)
            
            # Create new feature names
            selected_features = [f"PCA_{i}" for i in range(X_pca.shape[1])]
            feature_scores = dict(zip(selected_features, pca.explained_variance_ratio_))
            
            # Note: This changes the feature space, so we'd need to handle this differently
            # For now, fall back to importance-based selection
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            
            model.fit(X, y)
            importances = model.feature_importances_
            feature_scores = dict(zip(X.columns, importances))
            top_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)[:n_features]
            selected_features = [f[0] for f in top_features]
            
        else:
            # Default: use all features
            selected_features = X.columns.tolist()[:n_features]
            feature_scores = dict(zip(selected_features, [1.0] * len(selected_features)))
        
        return selected_features, feature_scores
    
    def _get_param_grid(self, algorithm: str, model_type: str) -> Dict[str, List]:
        """Get hyperparameter grid for algorithm."""
        
        param_grids = {
            "random_forest": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            },
            "gradient_boosting": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
                "max_depth": [3, 5, 7]
            },
            "logistic_regression": {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l1", "l2"],
                "solver": ["liblinear"]
            },
            "linear_regression": {},
            "ridge": {
                "alpha": [0.1, 1, 10, 100]
            },
            "lasso": {
                "alpha": [0.01, 0.1, 1, 10]
            },
            "svm": {
                "C": [0.1, 1, 10],
                "gamma": ["scale", "auto", 0.001, 0.01]
            },
            "knn": {
                "n_neighbors": [3, 5, 7, 9],
                "weights": ["uniform", "distance"]
            }
        }
        
        return param_grids.get(algorithm, {})
    
    def _calculate_detailed_metrics(self, y_true, y_pred, model_type: str, model, X_test) -> Dict[str, Any]:
        """Calculate detailed performance metrics."""
        
        if model_type == "classification":
            metrics = {
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "precision": float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
                "recall": float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
                "f1_score": float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            }
            
            # ROC-AUC for binary classification
            if len(np.unique(y_true)) == 2 and hasattr(model, "predict_proba"):
                try:
                    y_proba = model.predict_proba(X_test)[:, 1]
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                except:
                    pass
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics["confusion_matrix"] = cm.tolist()
            
        else:  # regression
            metrics = {
                "r2_score": float(r2_score(y_true, y_pred)),
                "mse": float(mean_squared_error(y_true, y_pred)),
                "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
                "mae": float(mean_absolute_error(y_true, y_pred))
            }
        
        return metrics
    
    def _calculate_evaluation_metrics(self, y_true, y_pred, model_type: str, model, X_test, requested_metrics: List[str]) -> Dict[str, Any]:
        """Calculate requested evaluation metrics."""
        
        metrics = {}
        
        if model_type == "classification":
            if "accuracy" in requested_metrics:
                metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            if "precision" in requested_metrics:
                metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted', zero_division=0))
            if "recall" in requested_metrics:
                metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted', zero_division=0))
            if "f1" in requested_metrics:
                metrics["f1_score"] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
            if "roc_auc" in requested_metrics and hasattr(model, "predict_proba"):
                try:
                    if len(np.unique(y_true)) == 2:
                        y_proba = model.predict_proba(X_test)[:, 1]
                        metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
                except:
                    pass
        else:  # regression
            if "r2" in requested_metrics:
                metrics["r2_score"] = float(r2_score(y_true, y_pred))
            if "mse" in requested_metrics:
                metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            if "rmse" in requested_metrics:
                metrics["rmse"] = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            if "mae" in requested_metrics:
                metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
        
        return metrics
    
    def _generate_model_recommendations(self, evaluation_results: Dict[str, Any], model_comparison: List[Dict[str, Any]]) -> List[str]:
        """Generate model recommendations based on evaluation."""
        
        recommendations = []
        
        if not model_comparison:
            return ["No models available for comparison"]
        
        # Best performing model
        best_model = model_comparison[0]
        recommendations.append(f"Best performing model: {best_model['model']} (score: {best_model['primary_metric']:.3f})")
        
        # Performance gaps
        if len(model_comparison) > 1:
            performance_gap = best_model['primary_metric'] - model_comparison[1]['primary_metric']
            if performance_gap < 0.05:
                recommendations.append("Top models have similar performance - consider simpler model for production")
            elif performance_gap > 0.2:
                recommendations.append("Clear performance leader identified - recommend for deployment")
        
        # Model complexity considerations
        complex_models = ["random_forest", "gradient_boosting", "svm"]
        simple_models = ["logistic_regression", "linear_regression", "naive_bayes"]
        
        best_model_name = best_model['model']
        if best_model_name in complex_models:
            recommendations.append("Best model is complex - ensure adequate training data and consider interpretability")
        elif best_model_name in simple_models:
            recommendations.append("Best model is interpretable - good for understanding feature relationships")
        
        # Cross-validation stability
        for model_name, results in evaluation_results.items():
            if "cv_std" in results and results["cv_std"] > 0.1:
                recommendations.append(f"{model_name} shows high variance - consider regularization or more data")
        
        return recommendations
    
    def get_ml_summary(self) -> Dict[str, Any]:
        """Get summary of all ML analyses."""
        summary = {
            "metrics": self.ml_metrics.copy(),
            "ml_artifacts": {}
        }
        
        for artifact_key in self.list_artifacts():
            artifact = self._data_artifacts[artifact_key]
            metadata = artifact["metadata"]
            
            if any(key in artifact_key for key in ["feature_engineered", "trained_models", "predictions", "evaluation", "feature_importance"]):
                summary["ml_artifacts"][artifact_key] = {
                    "type": metadata.get("analysis_type", "ml_artifact"),
                    "timestamp": artifact["timestamp"]
                }
        
        return summary