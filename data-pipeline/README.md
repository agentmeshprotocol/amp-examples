# AutoGen Data Analysis Pipeline

A comprehensive data analysis pipeline using Microsoft AutoGen agents with Agent Mesh Protocol (AMP) integration. This example demonstrates how multiple specialized AI agents can collaborate to perform end-to-end data analysis workflows.

## 🏗️ Architecture

The pipeline consists of 6 specialized AutoGen agents working together:

```mermaid
graph TD
    A[Data Collector Agent] --> B[Data Cleaner Agent]
    B --> C[Statistical Analyst Agent]
    B --> D[ML Analyst Agent]
    C --> E[Visualization Agent]
    D --> E
    E --> F[Quality Assurance Agent]
    F --> G[Final Report]
```

### 🤖 Agents

1. **Data Collector Agent** - Ingests data from files, databases, and APIs
2. **Data Cleaner Agent** - Handles missing values, outliers, and normalization
3. **Statistical Analyst Agent** - Performs descriptive statistics and hypothesis testing
4. **ML Analyst Agent** - Builds and evaluates machine learning models
5. **Visualization Agent** - Creates charts, dashboards, and reports
6. **Quality Assurance Agent** - Validates results and ensures accuracy

## 🌟 Features

- **Multi-Agent Collaboration**: Agents communicate using AutoGen's conversation framework
- **AMP Protocol Integration**: Standardized agent communication with capability discovery
- **End-to-End Automation**: Complete workflow from data ingestion to final reporting
- **Quality Assurance**: Built-in validation and quality checks at every step
- **Interactive Dashboards**: Rich visualizations and comprehensive reports
- **Extensible Architecture**: Easy to add new agents and capabilities
- **Multiple Data Sources**: Support for files, databases, and APIs
- **Advanced ML Pipeline**: Feature engineering, model selection, and evaluation

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key (for LLM capabilities)
- Git

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd amp-examples/data-pipeline
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
```

4. **Create sample datasets** (optional):
```bash
cd data
python create_sample_datasets.py
```

### Running the Pipeline

#### Option 1: Sample Analysis
```bash
python run_pipeline.py --sample
```

#### Option 2: Custom Data Analysis
```bash
python run_pipeline.py --data data/samples/employee_data.csv --request "Predict employee performance"
```

#### Option 3: Database Analysis
```bash
python run_pipeline.py --data "sqlite:///data/samples/sample_database.db" --request "Analyze customer behavior"
```

## 📊 Example Outputs

The pipeline generates comprehensive outputs:

### Data Collection Results
- Dataset metadata and validation report
- Data source connectivity status
- Sample data preview and statistics

### Data Cleaning Results
- Missing value imputation report
- Outlier detection and handling summary
- Data quality score (0-1) with recommendations

### Statistical Analysis Results
- Descriptive statistics for all variables
- Correlation matrix with significance testing
- Hypothesis test results and interpretations

### ML Analysis Results
- Feature importance rankings
- Model performance comparison (accuracy, precision, recall, etc.)
- Best model selection with cross-validation scores
- Predictions with confidence intervals

### Visualizations
- Distribution plots and boxplots
- Correlation heatmaps
- Model performance charts (ROC curves, confusion matrices)
- Interactive dashboards with multiple views

### Quality Assurance Report
- Data validation results against quality thresholds
- Model validation and performance benchmarking
- Pipeline audit with compliance scoring
- Final recommendations and next steps

## 📁 Project Structure

```
data-pipeline/
├── agents/                      # AutoGen agent implementations
│   ├── __init__.py
│   ├── base_agent.py           # Base AutoGen AMP agent class
│   ├── data_collector.py       # Data ingestion agent
│   ├── data_cleaner.py         # Data preprocessing agent
│   ├── statistical_analyst.py  # Statistical analysis agent
│   ├── ml_analyst.py           # Machine learning agent
│   ├── visualization_agent.py  # Visualization and reporting agent
│   └── quality_assurance.py    # Quality validation agent
│
├── pipelines/                   # Pipeline orchestration
│   ├── __init__.py
│   └── data_pipeline.py        # Main pipeline implementation
│
├── config/                      # Configuration files
│   ├── pipeline_config.yaml    # Pipeline settings
│   └── agent_config.yaml       # Agent configurations
│
├── data/                        # Sample datasets and utilities
│   ├── create_sample_datasets.py
│   └── samples/                 # Generated sample data
│
├── notebooks/                   # Jupyter examples
│   ├── 01_Pipeline_Overview.ipynb
│   ├── 02_Agent_Deep_Dive.ipynb
│   ├── 03_Custom_Analysis.ipynb
│   └── 04_Visualization_Examples.ipynb
│
├── examples/                    # Example scripts and workflows
├── tests/                       # Test scripts
├── requirements.txt             # Python dependencies
├── run_pipeline.py             # Main application runner
└── README.md                   # This file
```

## 🔧 Configuration

### Pipeline Configuration

Customize the pipeline behavior in `config/pipeline_config.yaml`:

```yaml
pipeline:
  name: "AutoGen Data Analysis Pipeline"
  components:
    data_collection: true
    data_cleaning: true
    statistical_analysis: true
    ml_analysis: true
    visualization: true
    quality_assurance: true

quality:
  data:
    completeness_threshold: 0.9
    consistency_threshold: 0.95
  models:
    min_accuracy: 0.7
    min_r2_score: 0.6
```

### Agent Configuration

Customize agent behaviors in `config/agent_config.yaml`:

```yaml
agents:
  data_cleaner:
    default_missing_strategy: "median"
    outlier_threshold: 3.0
    
  ml_analyst:
    default_test_size: 0.2
    cv_folds: 5
    hyperparameter_tuning: true
```

### LLM Configuration

Configure the language model in your environment or pipeline config:

```yaml
llm:
  openai:
    model: "gpt-4"
    api_key: "${OPENAI_API_KEY}"
    temperature: 0.1
    max_tokens: 2000
```

## 📝 Usage Examples

### Basic Usage

```python
from pipelines.data_pipeline import DataAnalysisPipeline, PipelineConfig

# Create configuration
config = PipelineConfig(
    pipeline_id="my_analysis",
    enable_all_components=True,
    llm_config={"config_list": [{"model": "gpt-4", "api_key": "your-key"}]}
)

# Initialize and run pipeline
pipeline = DataAnalysisPipeline(config)
results = await pipeline.run_pipeline(
    data_source="data/my_dataset.csv",
    analysis_request="Analyze customer satisfaction and predict churn"
)
```

### Advanced Configuration

```python
# Custom pipeline with specific components
config = PipelineConfig(
    pipeline_id="custom_analysis",
    enable_data_collection=True,
    enable_data_cleaning=True,
    enable_statistical_analysis=True,
    enable_ml_analysis=False,  # Skip ML for statistical analysis only
    enable_visualization=True,
    enable_quality_assurance=True,
    data_quality_threshold=0.85,
    generate_report=True
)
```

### Individual Agent Usage

```python
from agents.statistical_analyst import StatisticalAnalystAgent

# Use individual agent
analyst = StatisticalAnalystAgent(amp_config, llm_config)
await analyst.connect()

# Perform specific analysis
result = await analyst.invoke_capability(
    None, "statistical-descriptive-analysis",
    {"dataset_key": "my_data", "include_distribution": True}
)
```

## 🎯 Supported Data Sources

### Files
- **CSV**: Comma-separated values with automatic delimiter detection
- **JSON**: Structured JSON data with nested object support
- **Excel**: .xlsx and .xls files with multiple sheet support
- **Parquet**: Columnar format for large datasets

### Databases
- **SQLite**: Local database files
- **PostgreSQL**: Remote PostgreSQL databases
- **MySQL**: Remote MySQL databases

### APIs
- **REST APIs**: JSON responses with configurable authentication
- **Web Services**: SOAP and other web service protocols

### Example Data Source Configurations

```python
# File source
data_source = "data/sales_data.csv"

# Database source  
data_source = {
    "connection_string": "postgresql://user:pass@host:5432/db",
    "query": "SELECT * FROM customers WHERE active = true"
}

# API source
data_source = {
    "url": "https://api.example.com/data",
    "headers": {"Authorization": "Bearer token"},
    "data_path": "results.data"
}
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
python -m pytest tests/

# Run specific test categories
python -m pytest tests/test_agents.py
python -m pytest tests/test_integration.py

# Run with coverage
python -m pytest tests/ --cov=agents --cov=pipelines
```

## 📊 Performance and Scaling

### Performance Optimization

- **Parallel Processing**: Agents can run statistical and ML analysis in parallel
- **Chunked Processing**: Large datasets are processed in configurable chunks
- **Caching**: Results are cached to avoid recomputation
- **Memory Management**: Configurable memory limits per agent

### Scaling Considerations

- **Horizontal Scaling**: Deploy agents across multiple machines
- **Resource Allocation**: Configure memory and CPU limits per agent
- **Load Balancing**: Distribute work across agent instances
- **State Management**: Persistent storage for large analysis workflows

## 🔍 Monitoring and Debugging

### Logging

Configure logging levels in `config/pipeline_config.yaml`:

```yaml
logging:
  level: "INFO"
  components:
    agents: "INFO"
    amp_client: "DEBUG"
    pipeline: "INFO"
```

### Metrics and Monitoring

Each agent provides metrics:

```python
# Get pipeline metrics
metrics = await pipeline.get_pipeline_metrics()

# Get individual agent metrics
agent_metrics = await agent.get_metrics()
```

### Health Checks

Monitor agent health:

```python
# Pipeline health check
status = await pipeline.get_pipeline_status()

# Individual agent health
health = await agent.health_check()
```

## 🛠️ Extending the Pipeline

### Adding New Agents

1. **Create agent class**:
```python
from agents.base_agent import AutoGenAMPAgent

class MyCustomAgent(AutoGenAMPAgent):
    def __init__(self, amp_config, llm_config):
        # Define capabilities
        capabilities = [...]
        
        # Initialize base class
        super().__init__(autogen_config, amp_config, capabilities)
    
    def _process_conversation_message(self, message, sender, history):
        # Handle natural language interactions
        return "Custom response"
    
    async def _handle_my_capability(self, parameters):
        # Implement capability logic
        return {"result": "success"}
```

2. **Register in pipeline**:
```python
# Add to pipeline initialization
self.agents["my_custom"] = MyCustomAgent(config, llm_config)
```

### Adding New Capabilities

1. **Define capability schema**:
```python
capability = Capability(
    id="my-new-capability",
    version="1.0",
    description="My new analysis capability",
    input_schema={...},
    output_schema={...},
    category="analysis"
)
```

2. **Implement handler**:
```python
async def _handle_my_new_capability(self, parameters):
    # Implementation logic
    return results
```

3. **Register capability**:
```python
self.register_capability(capability, self._handle_my_new_capability)
```

## 🔐 Security Considerations

### API Key Management
- Store API keys in environment variables
- Use secure key management services in production
- Implement key rotation policies

### Data Privacy
- Configure data retention policies
- Implement data anonymization for sensitive datasets
- Use secure transport (HTTPS/TLS) for remote data sources

### Access Control
- Implement authentication for AMP network access
- Use role-based access control for different agent capabilities
- Audit and log all agent interactions

## 🐛 Troubleshooting

### Common Issues

1. **Connection Errors**:
   - Verify AMP network endpoint configuration
   - Check network connectivity and firewall settings
   - Ensure API keys are correctly configured

2. **Memory Issues**:
   - Reduce dataset size or enable chunked processing
   - Increase memory limits in configuration
   - Use data sampling for large datasets

3. **Model Performance Issues**:
   - Check data quality scores
   - Verify feature engineering results
   - Adjust model performance thresholds

### Debug Mode

Enable debug logging:

```bash
python run_pipeline.py --data mydata.csv --request "analysis" --log-level DEBUG
```

### Getting Help

- Check the [notebooks/](notebooks/) directory for detailed examples
- Review [configuration files](config/) for customization options
- Examine [test cases](tests/) for usage patterns

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

### Development Setup

1. **Clone and install in development mode**:
```bash
git clone <repository-url>
cd amp-examples/data-pipeline
pip install -e .
```

2. **Install development dependencies**:
```bash
pip install -r requirements-dev.txt
```

3. **Run pre-commit hooks**:
```bash
pre-commit install
```

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints for all public APIs
- Include docstrings for all classes and methods
- Write comprehensive tests for new features

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../../LICENSE) file for details.

## 🙏 Acknowledgments

- [Microsoft AutoGen](https://github.com/microsoft/autogen) for the multi-agent framework
- [Agent Mesh Protocol](../../docs/agent-interoperability-protocol.md) for standardized agent communication
- The open-source data science community for tools and libraries

## 📚 Additional Resources

- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [AMP Protocol Specification](../../docs/agent-interoperability-protocol.md)
- [Example Notebooks](notebooks/)
- [Configuration Reference](config/)
- [API Documentation](docs/)

---

**Note**: This is an example implementation demonstrating AutoGen integration with the Agent Mesh Protocol. For production use, additional security, monitoring, and scaling considerations should be implemented.