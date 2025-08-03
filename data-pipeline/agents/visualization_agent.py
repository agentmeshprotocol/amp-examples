"""
Visualization Agent.

Specialized AutoGen agent for creating charts, dashboards, and reports
from data analysis results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
from typing import Dict, Any, List, Optional, Tuple, Union
import warnings

from .base_agent import AutoGenAMPAgent, AutoGenConfig
from amp_types import Capability, CapabilityConstraints
from amp_client import AMPClientConfig

warnings.filterwarnings('ignore')
plt.style.use('default')
sns.set_palette("husl")


class VisualizationAgent(AutoGenAMPAgent):
    """
    Agent specialized in data visualization and reporting.
    
    Capabilities:
    - Statistical plots (histograms, boxplots, scatter plots)
    - Correlation and relationship visualizations
    - Model performance visualizations
    - Interactive dashboards with Plotly
    - Automated report generation
    - Time series plots
    """
    
    def __init__(
        self,
        amp_config: AMPClientConfig,
        llm_config: Dict[str, Any],
        viz_config: Dict[str, Any] = None
    ):
        """
        Initialize Visualization Agent.
        
        Args:
            amp_config: AMP client configuration
            llm_config: LLM configuration for AutoGen
            viz_config: Configuration for visualization settings
        """
        # Define capabilities
        capabilities = [
            Capability(
                id="visualization-statistical-plots",
                version="1.0",
                description="Create statistical plots and exploratory data visualizations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "plot_type": {"type": "string", "enum": ["histogram", "boxplot", "scatter", "pairplot", "heatmap", "violin"]},
                        "variables": {"type": "array", "items": {"type": "string"}},
                        "group_by": {"type": "string"},
                        "color_by": {"type": "string"},
                        "title": {"type": "string"},
                        "interactive": {"type": "boolean"}
                    },
                    "required": ["dataset_key", "plot_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "plot_data": {"type": "object"},
                        "plot_config": {"type": "object"},
                        "insights": {"type": "array"},
                        "visualization_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=10000
                ),
                category="generation"
            ),
            Capability(
                id="visualization-model-performance",
                version="1.0",
                description="Create model performance and evaluation visualizations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "model_results_key": {"type": "string"},
                        "plot_type": {"type": "string", "enum": ["confusion_matrix", "roc_curve", "feature_importance", "learning_curve", "residuals", "prediction_scatter"]},
                        "model_name": {"type": "string"},
                        "compare_models": {"type": "boolean"},
                        "interactive": {"type": "boolean"}
                    },
                    "required": ["model_results_key", "plot_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "plot_data": {"type": "object"},
                        "performance_insights": {"type": "array"},
                        "visualization_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=8000
                ),
                category="generation"
            ),
            Capability(
                id="visualization-correlation-analysis",
                version="1.0",
                description="Create correlation and relationship visualizations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "correlation_method": {"type": "string", "enum": ["pearson", "spearman", "kendall"]},
                        "variables": {"type": "array", "items": {"type": "string"}},
                        "plot_type": {"type": "string", "enum": ["heatmap", "network", "clustermap"]},
                        "annotation": {"type": "boolean"},
                        "cluster": {"type": "boolean"}
                    },
                    "required": ["dataset_key"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "correlation_matrix": {"type": "object"},
                        "visualization_data": {"type": "object"},
                        "strong_correlations": {"type": "array"},
                        "visualization_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=7000
                ),
                category="generation"
            ),
            Capability(
                id="visualization-dashboard",
                version="1.0",
                description="Create interactive dashboards combining multiple visualizations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "dataset_key": {"type": "string"},
                        "dashboard_type": {"type": "string", "enum": ["exploratory", "model_performance", "summary"]},
                        "components": {"type": "array", "items": {"type": "string"}},
                        "layout": {"type": "string", "enum": ["grid", "tabs", "sidebar"]},
                        "title": {"type": "string"}
                    },
                    "required": ["dataset_key", "dashboard_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "dashboard_data": {"type": "object"},
                        "component_summaries": {"type": "array"},
                        "dashboard_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=15000
                ),
                category="generation"
            ),
            Capability(
                id="visualization-report",
                version="1.0",
                description="Generate comprehensive analysis reports with visualizations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "analysis_keys": {"type": "array", "items": {"type": "string"}},
                        "report_type": {"type": "string", "enum": ["data_summary", "model_evaluation", "full_analysis"]},
                        "include_recommendations": {"type": "boolean"},
                        "format": {"type": "string", "enum": ["html", "pdf", "markdown"]}
                    },
                    "required": ["analysis_keys", "report_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "report_content": {"type": "string"},
                        "report_sections": {"type": "array"},
                        "visualizations_included": {"type": "array"},
                        "report_key": {"type": "string"}
                    }
                },
                constraints=CapabilityConstraints(
                    response_time_ms=20000
                ),
                category="generation"
            )
        ]
        
        # AutoGen configuration
        autogen_config = AutoGenConfig(
            name="VisualizationAgent",
            system_message="""You are a Visualization Agent specialized in creating charts, dashboards, and reports from data analysis results.

Your responsibilities:
1. Create statistical plots and exploratory data visualizations
2. Generate model performance and evaluation visualizations  
3. Build correlation and relationship visualizations
4. Create interactive dashboards combining multiple visualizations
5. Generate comprehensive analysis reports with integrated visualizations

You work collaboratively with other agents in the data analysis pipeline. Always create clear, informative visualizations that effectively communicate insights and support decision-making.

When creating visualizations, focus on:
- Choosing appropriate chart types for the data and message
- Using clear, descriptive titles and labels
- Applying consistent color schemes and styling
- Highlighting key insights and patterns
- Ensuring accessibility and readability
- Providing context and interpretation of visual patterns""",
            llm_config=llm_config,
            human_input_mode="NEVER",
            max_consecutive_auto_reply=3
        )
        
        super().__init__(
            autogen_config=autogen_config,
            amp_config=amp_config,
            capabilities=capabilities,
            description="Agent specialized in data visualization and report generation",
            tags=["visualization", "dashboards", "reports", "charts", "autogen"]
        )
        
        # Visualization configuration
        self.viz_config = viz_config or {
            "default_figure_size": (10, 6),
            "default_color_palette": "husl",
            "interactive_default": True,
            "high_dpi": True,
            "font_size": 12
        }
        
        # Set matplotlib and seaborn defaults
        plt.rcParams['figure.figsize'] = self.viz_config["default_figure_size"]
        plt.rcParams['font.size'] = self.viz_config["font_size"]
        plt.rcParams['figure.dpi'] = 150 if self.viz_config["high_dpi"] else 100
        
        # Visualization metrics
        self.viz_metrics = {
            "plots_created": 0,
            "dashboards_built": 0,
            "reports_generated": 0,
            "interactive_visualizations": 0,
            "static_visualizations": 0
        }
    
    def _process_conversation_message(
        self,
        message: str,
        sender,
        conversation_history: List[Dict]
    ) -> str:
        """Process conversation messages for visualization requests."""
        
        message_lower = message.lower()
        
        if "plot" in message_lower or "chart" in message_lower or "visualize" in message_lower:
            return self._handle_plot_request(message)
        elif "dashboard" in message_lower:
            return self._handle_dashboard_request(message)
        elif "report" in message_lower:
            return self._handle_report_request(message)
        elif "correlation" in message_lower and ("heatmap" in message_lower or "visualize" in message_lower):
            return self._handle_correlation_viz_request(message)
        elif "model performance" in message_lower or "confusion matrix" in message_lower:
            return self._handle_model_viz_request(message)
        elif "status" in message_lower or "metrics" in message_lower:
            return self._handle_status_request()
        else:
            return """I'm the Visualization Agent. I can help you with:

1. **Statistical Plots**: Histograms, boxplots, scatter plots, pairplots, heatmaps
2. **Model Performance**: Confusion matrices, ROC curves, feature importance, residual plots
3. **Correlation Analysis**: Correlation heatmaps, network plots, clustermaps
4. **Interactive Dashboards**: Multi-component dashboards with various layouts
5. **Analysis Reports**: Comprehensive reports combining visualizations and insights

What type of visualization would you like me to create?"""
    
    def _handle_plot_request(self, message: str) -> str:
        """Handle statistical plot requests."""
        artifacts = self.list_artifacts()
        if artifacts:
            return f"""I can create plots from: {', '.join(artifacts)}

Available plot types:
- **Histogram**: Distribution of numerical variables
- **Boxplot**: Distribution with quartiles and outliers
- **Scatter**: Relationship between two variables
- **Pairplot**: Multiple variable relationships
- **Heatmap**: Matrix visualization of data
- **Violin**: Distribution shape and density

Example: "Create histogram of [variable] from [dataset]"
Example: "Create scatter plot of [x] vs [y] colored by [group]" """
        else:
            return "No datasets available for plotting. Please collect and analyze data first."
    
    def _handle_dashboard_request(self, message: str) -> str:
        """Handle dashboard creation requests."""
        return """I can create interactive dashboards:

**Dashboard Types:**
- **Exploratory**: Data exploration with multiple statistical plots
- **Model Performance**: Model evaluation metrics and diagnostics
- **Summary**: Overview dashboard with key insights

**Layouts:**
- **Grid**: Organized grid of visualizations
- **Tabs**: Tabbed interface for different views
- **Sidebar**: Sidebar navigation with main content area

Example: "Create exploratory dashboard for [dataset] with grid layout" """
    
    def _handle_report_request(self, message: str) -> str:
        """Handle report generation requests."""
        return """I can generate comprehensive analysis reports:

**Report Types:**
- **Data Summary**: Overview of dataset characteristics and quality
- **Model Evaluation**: Detailed model performance analysis
- **Full Analysis**: Complete analysis workflow with all insights

**Formats:**
- **HTML**: Interactive web report
- **PDF**: Print-ready document
- **Markdown**: Text-based structured report

Example: "Generate full analysis report from [analysis_keys] in HTML format" """
    
    def _handle_correlation_viz_request(self, message: str) -> str:
        """Handle correlation visualization requests."""
        return """I can create correlation visualizations:

**Plot Types:**
- **Heatmap**: Traditional correlation matrix heatmap
- **Network**: Network graph showing relationships
- **Clustermap**: Hierarchically clustered correlation matrix

**Options:**
- Multiple correlation methods (Pearson, Spearman, Kendall)
- Annotation with correlation values
- Clustering by similarity

Example: "Create correlation heatmap for [dataset] using Spearman method" """
    
    def _handle_model_viz_request(self, message: str) -> str:
        """Handle model performance visualization requests."""
        return """I can create model performance visualizations:

**Performance Plots:**
- **Confusion Matrix**: Classification performance breakdown
- **ROC Curve**: Receiver operating characteristic
- **Feature Importance**: Variable importance ranking
- **Learning Curve**: Training vs validation performance
- **Residuals**: Regression model residual analysis
- **Prediction Scatter**: Predicted vs actual values

Example: "Create confusion matrix for [model] results"
Example: "Show feature importance plot for [model]" """
    
    def _handle_status_request(self) -> str:
        """Handle status and metrics requests."""
        metrics = self.viz_metrics
        artifacts = self.list_artifacts()
        
        viz_artifacts = [a for a in artifacts if "visualization" in a or "dashboard" in a or "report" in a]
        
        return f"""Visualization Status:

**Visualization Metrics:**
- Plots created: {metrics['plots_created']}
- Dashboards built: {metrics['dashboards_built']}
- Reports generated: {metrics['reports_generated']}
- Interactive visualizations: {metrics['interactive_visualizations']}
- Static visualizations: {metrics['static_visualizations']}

**Visualization Artifacts:** {len(viz_artifacts)}
{', '.join(viz_artifacts) if viz_artifacts else 'None'}

Ready for new visualization tasks."""
    
    # AMP Capability Handlers
    
    async def _handle_visualization_statistical_plots(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistical plot creation capability."""
        try:
            dataset_key = parameters["dataset_key"]
            plot_type = parameters["plot_type"]
            variables = parameters.get("variables", [])
            group_by = parameters.get("group_by")
            color_by = parameters.get("color_by")
            title = parameters.get("title", f"{plot_type.title()} Plot")
            interactive = parameters.get("interactive", self.viz_config["interactive_default"])
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Create the plot
            plot_data, plot_config, insights = self._create_statistical_plot(
                df, plot_type, variables, group_by, color_by, title, interactive
            )
            
            # Store visualization
            viz_key = f"{dataset_key}_{plot_type}_plot"
            viz_metadata = {
                "dataset_key": dataset_key,
                "plot_type": plot_type,
                "variables": variables,
                "group_by": group_by,
                "color_by": color_by,
                "interactive": interactive,
                "visualization_type": "statistical_plot"
            }
            
            viz_artifact = {
                "plot_data": plot_data,
                "plot_config": plot_config,
                "insights": insights
            }
            
            self.store_artifact(viz_key, viz_artifact, viz_metadata)
            
            # Update metrics
            self.viz_metrics["plots_created"] += 1
            if interactive:
                self.viz_metrics["interactive_visualizations"] += 1
            else:
                self.viz_metrics["static_visualizations"] += 1
            
            return {
                "plot_data": plot_data,
                "plot_config": plot_config,
                "insights": insights,
                "visualization_key": viz_key
            }
            
        except Exception as e:
            self.logger.error(f"Statistical plot creation error: {e}")
            raise
    
    async def _handle_visualization_model_performance(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model performance visualization capability."""
        try:
            model_results_key = parameters["model_results_key"]
            plot_type = parameters["plot_type"]
            model_name = parameters.get("model_name")
            compare_models = parameters.get("compare_models", False)
            interactive = parameters.get("interactive", True)
            
            # Get model results
            model_artifact = self.get_artifact(model_results_key)
            if model_artifact is None:
                raise ValueError(f"Model results not found: {model_results_key}")
            
            # Create performance visualization
            plot_data, performance_insights = self._create_model_performance_plot(
                model_artifact, plot_type, model_name, compare_models, interactive
            )
            
            # Store visualization
            viz_key = f"{model_results_key}_{plot_type}_viz"
            viz_metadata = {
                "model_results_key": model_results_key,
                "plot_type": plot_type,
                "model_name": model_name,
                "compare_models": compare_models,
                "interactive": interactive,
                "visualization_type": "model_performance"
            }
            
            viz_artifact = {
                "plot_data": plot_data,
                "performance_insights": performance_insights
            }
            
            self.store_artifact(viz_key, viz_artifact, viz_metadata)
            
            # Update metrics
            self.viz_metrics["plots_created"] += 1
            if interactive:
                self.viz_metrics["interactive_visualizations"] += 1
            else:
                self.viz_metrics["static_visualizations"] += 1
            
            return {
                "plot_data": plot_data,
                "performance_insights": performance_insights,
                "visualization_key": viz_key
            }
            
        except Exception as e:
            self.logger.error(f"Model performance visualization error: {e}")
            raise
    
    async def _handle_visualization_correlation_analysis(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle correlation visualization capability."""
        try:
            dataset_key = parameters["dataset_key"]
            correlation_method = parameters.get("correlation_method", "pearson")
            variables = parameters.get("variables")
            plot_type = parameters.get("plot_type", "heatmap")
            annotation = parameters.get("annotation", True)
            cluster = parameters.get("cluster", False)
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Create correlation visualization
            viz_data, strong_correlations = self._create_correlation_visualization(
                df, correlation_method, variables, plot_type, annotation, cluster
            )
            
            # Store visualization
            viz_key = f"{dataset_key}_correlation_{plot_type}"
            viz_metadata = {
                "dataset_key": dataset_key,
                "correlation_method": correlation_method,
                "plot_type": plot_type,
                "variables": variables,
                "annotation": annotation,
                "cluster": cluster,
                "visualization_type": "correlation_analysis"
            }
            
            viz_artifact = {
                "visualization_data": viz_data,
                "strong_correlations": strong_correlations,
                "correlation_matrix": viz_data.get("correlation_matrix", {})
            }
            
            self.store_artifact(viz_key, viz_artifact, viz_metadata)
            
            # Update metrics
            self.viz_metrics["plots_created"] += 1
            self.viz_metrics["interactive_visualizations"] += 1
            
            return {
                "correlation_matrix": viz_data.get("correlation_matrix", {}),
                "visualization_data": viz_data,
                "strong_correlations": strong_correlations,
                "visualization_key": viz_key
            }
            
        except Exception as e:
            self.logger.error(f"Correlation visualization error: {e}")
            raise
    
    async def _handle_visualization_dashboard(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dashboard creation capability."""
        try:
            dataset_key = parameters["dataset_key"]
            dashboard_type = parameters["dashboard_type"]
            components = parameters.get("components", [])
            layout = parameters.get("layout", "grid")
            title = parameters.get("title", f"{dashboard_type.title()} Dashboard")
            
            # Get the dataset
            df = self.get_artifact(dataset_key)
            if df is None:
                raise ValueError(f"Dataset not found: {dataset_key}")
            
            # Create dashboard
            dashboard_data, component_summaries = self._create_dashboard(
                df, dataset_key, dashboard_type, components, layout, title
            )
            
            # Store dashboard
            dashboard_key = f"{dataset_key}_{dashboard_type}_dashboard"
            dashboard_metadata = {
                "dataset_key": dataset_key,
                "dashboard_type": dashboard_type,
                "components": components,
                "layout": layout,
                "title": title,
                "visualization_type": "dashboard"
            }
            
            dashboard_artifact = {
                "dashboard_data": dashboard_data,
                "component_summaries": component_summaries
            }
            
            self.store_artifact(dashboard_key, dashboard_artifact, dashboard_metadata)
            
            # Update metrics
            self.viz_metrics["dashboards_built"] += 1
            self.viz_metrics["interactive_visualizations"] += 1
            
            return {
                "dashboard_data": dashboard_data,
                "component_summaries": component_summaries,
                "dashboard_key": dashboard_key
            }
            
        except Exception as e:
            self.logger.error(f"Dashboard creation error: {e}")
            raise
    
    async def _handle_visualization_report(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle report generation capability."""
        try:
            analysis_keys = parameters["analysis_keys"]
            report_type = parameters["report_type"]
            include_recommendations = parameters.get("include_recommendations", True)
            format_type = parameters.get("format", "html")
            
            # Get analysis artifacts
            analysis_artifacts = {}
            for key in analysis_keys:
                artifact = self.get_artifact(key)
                if artifact is not None:
                    analysis_artifacts[key] = artifact
            
            if not analysis_artifacts:
                raise ValueError("No valid analysis artifacts found")
            
            # Generate report
            report_content, report_sections, visualizations_included = self._generate_report(
                analysis_artifacts, report_type, include_recommendations, format_type
            )
            
            # Store report
            report_key = f"analysis_report_{report_type}"
            report_metadata = {
                "analysis_keys": analysis_keys,
                "report_type": report_type,
                "include_recommendations": include_recommendations,
                "format": format_type,
                "visualization_type": "report"
            }
            
            report_artifact = {
                "report_content": report_content,
                "report_sections": report_sections,
                "visualizations_included": visualizations_included
            }
            
            self.store_artifact(report_key, report_artifact, report_metadata)
            
            # Update metrics
            self.viz_metrics["reports_generated"] += 1
            
            return {
                "report_content": report_content,
                "report_sections": report_sections,
                "visualizations_included": visualizations_included,
                "report_key": report_key
            }
            
        except Exception as e:
            self.logger.error(f"Report generation error: {e}")
            raise
    
    # Helper methods
    
    def _create_statistical_plot(self, df: pd.DataFrame, plot_type: str, variables: List[str], 
                                group_by: str, color_by: str, title: str, interactive: bool) -> Tuple[Dict, Dict, List[str]]:
        """Create statistical plots."""
        
        plot_data = {}
        plot_config = {"title": title, "interactive": interactive}
        insights = []
        
        if plot_type == "histogram":
            if not variables:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()[:1]
            
            if variables and variables[0] in df.columns:
                var = variables[0]
                
                if interactive:
                    if group_by and group_by in df.columns:
                        fig = px.histogram(df, x=var, color=group_by, title=title, marginal="box")
                    else:
                        fig = px.histogram(df, x=var, title=title, marginal="box")
                    plot_data["plotly_json"] = fig.to_json()
                else:
                    plt.figure(figsize=self.viz_config["default_figure_size"])
                    if group_by and group_by in df.columns:
                        for group in df[group_by].unique():
                            subset = df[df[group_by] == group]
                            plt.hist(subset[var], alpha=0.7, label=str(group))
                        plt.legend()
                    else:
                        plt.hist(df[var], alpha=0.7)
                    plt.title(title)
                    plt.xlabel(var)
                    plt.ylabel("Frequency")
                    plot_data["matplotlib_b64"] = self._plt_to_base64()
                
                # Generate insights
                insights.append(f"{var} distribution: mean={df[var].mean():.2f}, std={df[var].std():.2f}")
                if df[var].skew() > 1:
                    insights.append(f"{var} is right-skewed")
                elif df[var].skew() < -1:
                    insights.append(f"{var} is left-skewed")
        
        elif plot_type == "boxplot":
            if not variables:
                variables = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if interactive:
                if len(variables) == 1 and group_by and group_by in df.columns:
                    fig = px.box(df, x=group_by, y=variables[0], title=title)
                else:
                    # Multiple variables
                    df_melted = df[variables].melt(var_name="Variable", value_name="Value")
                    fig = px.box(df_melted, x="Variable", y="Value", title=title)
                plot_data["plotly_json"] = fig.to_json()
            else:
                plt.figure(figsize=self.viz_config["default_figure_size"])
                if len(variables) == 1 and group_by and group_by in df.columns:
                    df.boxplot(column=variables[0], by=group_by)
                else:
                    df[variables].boxplot()
                plt.title(title)
                plot_data["matplotlib_b64"] = self._plt_to_base64()
            
            # Generate insights about outliers
            for var in variables:
                if var in df.columns and pd.api.types.is_numeric_dtype(df[var]):
                    Q1, Q3 = df[var].quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    outliers = df[(df[var] < Q1 - 1.5*IQR) | (df[var] > Q3 + 1.5*IQR)]
                    if len(outliers) > 0:
                        insights.append(f"{var} has {len(outliers)} outliers ({len(outliers)/len(df)*100:.1f}%)")
        
        elif plot_type == "scatter":
            if len(variables) >= 2:
                x_var, y_var = variables[0], variables[1]
                
                if interactive:
                    if color_by and color_by in df.columns:
                        fig = px.scatter(df, x=x_var, y=y_var, color=color_by, title=title)
                    else:
                        fig = px.scatter(df, x=x_var, y=y_var, title=title)
                    plot_data["plotly_json"] = fig.to_json()
                else:
                    plt.figure(figsize=self.viz_config["default_figure_size"])
                    if color_by and color_by in df.columns:
                        for group in df[color_by].unique():
                            subset = df[df[color_by] == group]
                            plt.scatter(subset[x_var], subset[y_var], label=str(group), alpha=0.7)
                        plt.legend()
                    else:
                        plt.scatter(df[x_var], df[y_var], alpha=0.7)
                    plt.xlabel(x_var)
                    plt.ylabel(y_var)
                    plt.title(title)
                    plot_data["matplotlib_b64"] = self._plt_to_base64()
                
                # Calculate correlation
                if pd.api.types.is_numeric_dtype(df[x_var]) and pd.api.types.is_numeric_dtype(df[y_var]):
                    corr = df[x_var].corr(df[y_var])
                    insights.append(f"Correlation between {x_var} and {y_var}: {corr:.3f}")
                    if abs(corr) > 0.7:
                        insights.append(f"Strong {'positive' if corr > 0 else 'negative'} correlation detected")
        
        elif plot_type == "heatmap":
            # Correlation heatmap
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                
                if interactive:
                    fig = px.imshow(corr_matrix, title=title, color_continuous_scale="RdBu")
                    plot_data["plotly_json"] = fig.to_json()
                else:
                    plt.figure(figsize=self.viz_config["default_figure_size"])
                    sns.heatmap(corr_matrix, annot=True, cmap="RdBu", center=0)
                    plt.title(title)
                    plot_data["matplotlib_b64"] = self._plt_to_base64()
                
                # Find strong correlations
                strong_corrs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = corr_matrix.iloc[i, j]
                        if abs(corr_val) > 0.7:
                            strong_corrs.append(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_val:.3f}")
                
                if strong_corrs:
                    insights.extend(strong_corrs[:3])  # Top 3
        
        return plot_data, plot_config, insights
    
    def _create_model_performance_plot(self, model_artifact: Dict[str, Any], plot_type: str, 
                                     model_name: str, compare_models: bool, interactive: bool) -> Tuple[Dict, List[str]]:
        """Create model performance visualizations."""
        
        plot_data = {}
        insights = []
        
        if "models" in model_artifact:
            # Multiple models artifact
            models = model_artifact["models"]
            results = model_artifact.get("results", {})
            
            if model_name and model_name in models:
                model = models[model_name]
                model_results = results.get(model_name, {})
            else:
                # Use first model
                model_name = list(models.keys())[0]
                model = models[model_name]
                model_results = results.get(model_name, {})
        else:
            # Single model result
            model_results = model_artifact
            model = None
        
        if plot_type == "confusion_matrix":
            if "detailed_metrics" in model_results and "confusion_matrix" in model_results["detailed_metrics"]:
                cm = np.array(model_results["detailed_metrics"]["confusion_matrix"])
                
                if interactive:
                    fig = px.imshow(cm, text_auto=True, title=f"Confusion Matrix - {model_name}")
                    plot_data["plotly_json"] = fig.to_json()
                else:
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f"Confusion Matrix - {model_name}")
                    plt.ylabel("True Label")
                    plt.xlabel("Predicted Label")
                    plot_data["matplotlib_b64"] = self._plt_to_base64()
                
                # Calculate accuracy from confusion matrix
                accuracy = np.trace(cm) / np.sum(cm)
                insights.append(f"Overall accuracy: {accuracy:.3f}")
        
        elif plot_type == "feature_importance":
            if model and hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
                feature_names = model_artifact.get("feature_names", [f"Feature_{i}" for i in range(len(importances))])
                
                # Sort by importance
                indices = np.argsort(importances)[::-1][:15]  # Top 15
                
                if interactive:
                    fig = px.bar(
                        x=[feature_names[i] for i in indices], 
                        y=[importances[i] for i in indices],
                        title=f"Feature Importance - {model_name}"
                    )
                    fig.update_layout(xaxis_tickangle=-45)
                    plot_data["plotly_json"] = fig.to_json()
                else:
                    plt.figure(figsize=(12, 6))
                    plt.bar(range(len(indices)), [importances[i] for i in indices])
                    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=45, ha='right')
                    plt.title(f"Feature Importance - {model_name}")
                    plt.tight_layout()
                    plot_data["matplotlib_b64"] = self._plt_to_base64()
                
                # Top features
                top_features = [feature_names[i] for i in indices[:3]]
                insights.append(f"Top 3 important features: {', '.join(top_features)}")
        
        elif plot_type == "learning_curve":
            # Simplified learning curve based on CV scores
            if "cv_scores" in model_results:
                cv_scores = model_results["cv_scores"]
                
                if interactive:
                    fig = px.line(
                        x=list(range(1, len(cv_scores)+1)),
                        y=cv_scores,
                        title=f"Cross-Validation Scores - {model_name}",
                        labels={"x": "CV Fold", "y": "Score"}
                    )
                    plot_data["plotly_json"] = fig.to_json()
                else:
                    plt.figure(figsize=(10, 6))
                    plt.plot(range(1, len(cv_scores)+1), cv_scores, 'o-')
                    plt.title(f"Cross-Validation Scores - {model_name}")
                    plt.xlabel("CV Fold")
                    plt.ylabel("Score")
                    plt.grid(True)
                    plot_data["matplotlib_b64"] = self._plt_to_base64()
                
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                insights.append(f"CV Score: {cv_mean:.3f} ± {cv_std:.3f}")
        
        return plot_data, insights
    
    def _create_correlation_visualization(self, df: pd.DataFrame, method: str, variables: List[str], 
                                        plot_type: str, annotation: bool, cluster: bool) -> Tuple[Dict, List[Dict]]:
        """Create correlation visualizations."""
        
        # Select numeric variables
        if variables:
            numeric_vars = [v for v in variables if v in df.columns and pd.api.types.is_numeric_dtype(df[v])]
        else:
            numeric_vars = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_vars) < 2:
            raise ValueError("Need at least 2 numeric variables for correlation analysis")
        
        # Calculate correlation matrix
        corr_data = df[numeric_vars]
        corr_matrix = corr_data.corr(method=method)
        
        viz_data = {"correlation_matrix": corr_matrix.to_dict()}
        
        if plot_type == "heatmap":
            if cluster:
                # Use seaborn clustermap
                g = sns.clustermap(corr_matrix, annot=annotation, cmap="RdBu", center=0, 
                                 figsize=self.viz_config["default_figure_size"])
                viz_data["matplotlib_b64"] = self._plt_to_base64(g.fig)
            else:
                # Regular heatmap with plotly
                fig = px.imshow(corr_matrix, 
                              title=f"Correlation Matrix ({method.title()})",
                              color_continuous_scale="RdBu",
                              aspect="auto")
                if annotation:
                    # Add text annotations
                    fig.update_traces(text=np.round(corr_matrix.values, 2), texttemplate="%{text}")
                viz_data["plotly_json"] = fig.to_json()
        
        # Find strong correlations
        strong_correlations = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # Moderate to strong correlation
                    strong_correlations.append({
                        "variable1": corr_matrix.columns[i],
                        "variable2": corr_matrix.columns[j],
                        "correlation": float(corr_val),
                        "strength": "strong" if abs(corr_val) > 0.7 else "moderate"
                    })
        
        # Sort by absolute correlation value
        strong_correlations.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        
        return viz_data, strong_correlations
    
    def _create_dashboard(self, df: pd.DataFrame, dataset_key: str, dashboard_type: str, 
                         components: List[str], layout: str, title: str) -> Tuple[Dict, List[Dict]]:
        """Create interactive dashboard."""
        
        dashboard_data = {
            "title": title,
            "layout": layout,
            "components": []
        }
        
        component_summaries = []
        
        if dashboard_type == "exploratory":
            # Data exploration dashboard
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            # Overview statistics
            overview_component = {
                "type": "overview",
                "title": "Dataset Overview",
                "data": {
                    "shape": list(df.shape),
                    "numeric_columns": len(numeric_cols),
                    "categorical_columns": len(categorical_cols),
                    "missing_values": int(df.isnull().sum().sum()),
                    "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                }
            }
            dashboard_data["components"].append(overview_component)
            component_summaries.append({"type": "overview", "title": "Dataset Overview"})
            
            # Distribution plots for numeric variables
            if numeric_cols:
                for col in numeric_cols[:4]:  # Limit to 4 plots
                    fig = px.histogram(df, x=col, title=f"Distribution of {col}", marginal="box")
                    hist_component = {
                        "type": "histogram",
                        "title": f"Distribution of {col}",
                        "plotly_json": fig.to_json()
                    }
                    dashboard_data["components"].append(hist_component)
                    component_summaries.append({"type": "histogram", "title": f"Distribution of {col}"})
            
            # Correlation heatmap
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, title="Correlation Matrix", color_continuous_scale="RdBu")
                corr_component = {
                    "type": "correlation",
                    "title": "Correlation Matrix",
                    "plotly_json": fig.to_json()
                }
                dashboard_data["components"].append(corr_component)
                component_summaries.append({"type": "correlation", "title": "Correlation Matrix"})
        
        elif dashboard_type == "model_performance":
            # Model performance dashboard (simplified)
            # This would typically use model results from other artifacts
            dashboard_data["components"].append({
                "type": "placeholder",
                "title": "Model Performance Dashboard",
                "message": "Model performance components would be populated from model results"
            })
            component_summaries.append({"type": "placeholder", "title": "Model Performance Dashboard"})
        
        return dashboard_data, component_summaries
    
    def _generate_report(self, analysis_artifacts: Dict[str, Any], report_type: str, 
                        include_recommendations: bool, format_type: str) -> Tuple[str, List[str], List[str]]:
        """Generate comprehensive analysis report."""
        
        report_sections = []
        visualizations_included = []
        
        # Report header
        report_content = f"""
# Data Analysis Report

**Report Type:** {report_type.title()}
**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

"""
        
        # Executive Summary
        report_content += """
## Executive Summary

This report presents a comprehensive analysis of the provided datasets and models.

"""
        report_sections.append("Executive Summary")
        
        # Data Overview Section
        dataset_artifacts = {k: v for k, v in analysis_artifacts.items() if "dataset" in k or isinstance(v, pd.DataFrame)}
        if dataset_artifacts:
            report_content += """
## Data Overview

### Dataset Characteristics

"""
            for key, artifact in dataset_artifacts.items():
                if isinstance(artifact, pd.DataFrame):
                    df = artifact
                    report_content += f"""
**Dataset: {key}**
- Shape: {df.shape[0]} rows × {df.shape[1]} columns
- Numeric columns: {len(df.select_dtypes(include=[np.number]).columns)}
- Categorical columns: {len(df.select_dtypes(include=['object', 'category']).columns)}
- Missing values: {df.isnull().sum().sum()} ({(df.isnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100):.1f}%)

"""
            report_sections.append("Data Overview")
        
        # Analysis Results Section
        analysis_results = {k: v for k, v in analysis_artifacts.items() if "analysis" in k or "results" in k}
        if analysis_results:
            report_content += """
## Analysis Results

"""
            for key, artifact in analysis_results.items():
                report_content += f"""
### {key.replace('_', ' ').title()}

"""
                # Extract key insights from analysis artifacts
                if isinstance(artifact, dict):
                    if "summary_insights" in artifact:
                        for insight in artifact["summary_insights"][:3]:  # Top 3 insights
                            report_content += f"- {insight}\n"
                    elif "correlation_insights" in artifact:
                        for insight in artifact["correlation_insights"][:3]:
                            report_content += f"- {insight}\n"
                
                report_content += "\n"
            
            report_sections.append("Analysis Results")
        
        # Model Performance Section
        model_artifacts = {k: v for k, v in analysis_artifacts.items() if "model" in k}
        if model_artifacts:
            report_content += """
## Model Performance

"""
            for key, artifact in model_artifacts.items():
                if isinstance(artifact, dict) and "model_results" in artifact:
                    results = artifact["model_results"]
                    best_model = artifact.get("best_model", "Unknown")
                    
                    report_content += f"""
**Best Model:** {best_model}

**Performance Metrics:**
"""
                    if best_model in results:
                        metrics = results[best_model].get("detailed_metrics", {})
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                report_content += f"- {metric.title()}: {value:.3f}\n"
                
                report_content += "\n"
            
            report_sections.append("Model Performance")
        
        # Recommendations Section
        if include_recommendations:
            report_content += """
## Recommendations

Based on the analysis results:

1. **Data Quality**: Review and address missing values and outliers identified in the analysis
2. **Feature Engineering**: Consider creating additional features based on correlation patterns
3. **Model Selection**: Evaluate model performance against business requirements
4. **Next Steps**: Consider collecting additional data or refining model parameters

"""
            report_sections.append("Recommendations")
        
        # Visualizations included
        viz_artifacts = {k: v for k, v in analysis_artifacts.items() if "visualization" in k or "plot" in k}
        visualizations_included = list(viz_artifacts.keys())
        
        # Format-specific formatting
        if format_type == "markdown":
            # Already in markdown format
            pass
        elif format_type == "html":
            # Convert markdown to HTML (simplified)
            report_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Data Analysis Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1, h2, h3 {{ color: #333; }}
        .metric {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
    </style>
</head>
<body>
{report_content.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>').replace('\n', '<br>')}
</body>
</html>
"""
        
        return report_content, report_sections, visualizations_included
    
    def _plt_to_base64(self, fig=None) -> str:
        """Convert matplotlib plot to base64 string."""
        if fig is None:
            fig = plt.gcf()
        
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        img_b64 = base64.b64encode(buffer.read()).decode()
        plt.close(fig)
        return img_b64
    
    def get_visualization_summary(self) -> Dict[str, Any]:
        """Get summary of all visualizations created."""
        summary = {
            "metrics": self.viz_metrics.copy(),
            "visualizations": {}
        }
        
        for artifact_key in self.list_artifacts():
            artifact = self._data_artifacts[artifact_key]
            metadata = artifact["metadata"]
            
            if metadata.get("visualization_type"):
                summary["visualizations"][artifact_key] = {
                    "type": metadata["visualization_type"],
                    "timestamp": artifact["timestamp"]
                }
        
        return summary