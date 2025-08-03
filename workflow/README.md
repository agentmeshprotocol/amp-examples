# AMP Workflow Orchestration

A comprehensive workflow orchestration system built on the Agent Mesh Protocol (AMP) that enables complex multi-agent coordination, state management, and error handling for automated business processes.

## Overview

This example demonstrates a production-ready workflow orchestration system using the AMP protocol. It provides a complete solution for defining, executing, and monitoring complex workflows with features including:

- **Multi-Agent Orchestration**: Coordinate multiple specialized agents
- **State Management**: Centralized workflow state with persistence
- **Conditional Logic**: Dynamic workflow branching and decision-making  
- **Error Handling**: Comprehensive failure recovery and retry strategies
- **Real-time Monitoring**: Performance metrics and alerting
- **Web Interface**: Modern dashboard for workflow management

## Architecture

### Core Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Workflow Engine │    │  Task Executor  │    │ State Manager   │
│                 │    │                 │    │                 │
│ - Orchestration │◄──►│ - Task Execution│◄──►│ - State Storage │
│ - Scheduling    │    │ - API Calls     │    │ - Context Mgmt  │
│ - Lifecycle     │    │ - Transformations│    │ - Persistence   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│Condition Eval.  │    │ Error Handler   │    │ Monitor Agent   │
│                 │    │                 │    │                 │
│ - Logic Eval.   │    │ - Failure Mgmt  │    │ - Metrics       │
│ - Branching     │    │ - Recovery      │    │ - Alerts        │
│ - Decision Trees│    │ - Circuit Break │    │ - Performance   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Agent Responsibilities

- **Workflow Engine**: Central orchestrator managing workflow definitions and execution
- **Task Executor**: Versatile executor handling various task types (API calls, data transformation, etc.)
- **State Manager**: Manages workflow state, context, and data persistence with caching
- **Condition Evaluator**: Evaluates conditional expressions for workflow branching
- **Error Handler**: Manages failures, implements recovery strategies, and handles escalation
- **Monitor Agent**: Tracks performance, generates alerts, and provides observability

## Features

### Workflow Capabilities

- **Complex Workflows**: Support for sequential, parallel, and conditional task execution
- **Dynamic Routing**: Conditional logic and decision trees for workflow branching
- **Error Recovery**: Automatic retry, fallback strategies, and manual intervention
- **State Persistence**: SQLite and Redis-backed state management with versioning
- **Real-time Updates**: WebSocket-based live monitoring and notifications

### Task Types

- **API Calls**: HTTP requests with retry logic and circuit breaker patterns
- **Data Transformation**: Built-in transformations (filter, map, reduce, aggregate, join)
- **Subprocess Execution**: System command execution with timeout and environment control
- **File Operations**: File I/O operations with atomic guarantees
- **Validation**: Schema validation and business rule enforcement
- **Custom Tasks**: Pluggable custom task handlers

### Monitoring & Observability

- **Performance Metrics**: Execution times, success rates, throughput analysis
- **Health Monitoring**: Agent health checks and system resource monitoring
- **Alerting**: Configurable alerts with multiple severity levels
- **Dashboard**: Real-time web interface with charts and status updates
- **Audit Trail**: Complete execution history and state change tracking

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd amp-examples/workflow

# Install dependencies
pip install -r requirements.txt
```

### 2. Start the Core Agents

```bash
# Terminal 1 - Workflow Engine
python -m agents.workflow_engine

# Terminal 2 - Task Executor
python -m agents.task_executor

# Terminal 3 - State Manager  
python -m agents.state_manager

# Terminal 4 - Condition Evaluator
python -m agents.condition_evaluator

# Terminal 5 - Error Handler
python -m agents.error_handler

# Terminal 6 - Monitor Agent
python -m agents.monitor_agent
```

### 3. Launch Web Dashboard

```bash
# Terminal 7 - Web Interface
python web/app.py --port 8090
```

Access the dashboard at: http://localhost:8090

### 4. Run Example Workflow

```python
import asyncio
from agents.workflow_engine import WorkflowEngine

async def run_example():
    engine = WorkflowEngine()
    await engine.start()
    
    # Load example workflow
    await engine.load_workflow_from_file("workflows/simple_example.yaml")
    
    # Start execution
    instance = await engine.start_workflow(
        workflow_id="hello-world-workflow",
        inputs={"name": "AMP User", "language": "en"}
    )
    
    print(f"Started workflow instance: {instance.id}")

asyncio.run(run_example())
```

## Workflow Definition

Workflows are defined in YAML or JSON format with a comprehensive schema:

### Basic Structure

```yaml
id: "my-workflow"
name: "My Workflow"
version: "1.0"
description: "Example workflow description"
tags: ["example", "tutorial"]

# Global settings
global_timeout_seconds: 3600
global_retry_config:
  strategy: "exponential_backoff"
  max_attempts: 3
  initial_delay_seconds: 2.0

# Input/Output schemas
input_schema:
  type: object
  properties:
    user_id: { type: string }
    amount: { type: number }
  required: ["user_id"]

output_schema:
  type: object
  properties:
    result: { type: string }
    processed_at: { type: string }

# Task definitions
tasks:
  - id: "validate_input"
    name: "Validate Input Data"
    type: "validation"
    agent_id: "task-executor-1"
    capability: "task-validation"
    parameters:
      data: "{user_input}"
      rules:
        - type: "required"
          field: "user_id"
        - type: "range"
          field: "amount"
          config: { min: 0, max: 10000 }
    timeout_seconds: 60
    outputs: ["validation_result"]
    depends_on: []

  - id: "process_data"
    name: "Process Data"
    type: "api_call"
    agent_id: "task-executor-1"
    capability: "task-api-call"
    parameters:
      method: "POST"
      url: "https://api.example.com/process"
      data: "{validated_data}"
    depends_on: ["validate_input"]
    condition:
      expression: "validation_result.valid == True"
      required_outputs: ["validation_result"]
```

### Task Types and Examples

#### API Call Task
```yaml
- id: "api_call_example"
  type: "api_call"
  parameters:
    method: "POST"
    url: "https://api.service.com/endpoint"
    headers:
      Authorization: "Bearer {api_token}"
    data:
      field1: "{input_value}"
    timeout: 30
    retry_count: 3
```

#### Data Transformation Task
```yaml
- id: "transform_data"
  type: "data_transform"
  parameters:
    input_data: "{raw_data}"
    transformations:
      - type: "filter"
        config:
          condition: "item['status'] == 'active'"
      - type: "map"
        config:
          expression: "{'id': item['id'], 'name': item['name'].upper()}"
      - type: "sort"
        config:
          key: "name"
```

#### Conditional Task
```yaml
- id: "conditional_logic"
  type: "conditional"
  agent_id: "condition-evaluator"
  capability: "condition-evaluate"
  parameters:
    expression: "amount > 1000 and user.verified == True"
    context: "{workflow_context}"
    condition_type: "python"
```

#### Parallel Execution
```yaml
- id: "parallel_processing"
  type: "parallel"
  parameters:
    task_type: "api_call"
    task_config:
      parallel_requests:
        - name: "service_a"
          url: "https://service-a.com/api"
        - name: "service_b"  
          url: "https://service-b.com/api"
```

## Configuration

### Agent Configuration

Create `config/agent_config.yaml`:

```yaml
workflow_engine:
  port: 8080
  max_concurrent_workflows: 50
  workflow_timeout_default: 3600

task_executor:
  port: 8081
  max_concurrent_tasks: 10
  task_timeout_default: 300

state_manager:
  port: 8082
  use_redis: true
  redis_url: "redis://localhost:6379"
  cache_ttl: 300

condition_evaluator:
  port: 8083
  cache_ttl: 60
  safe_evaluation: true

error_handler:
  port: 8084
  max_retry_attempts: 3
  escalation_timeout: 300

monitor_agent:
  port: 8085
  metrics_retention_hours: 168  # 7 days
  alert_check_interval: 60
```

### Database Configuration

The system uses SQLite for persistence and optionally Redis for caching:

```yaml
database:
  sqlite_path: "workflow_state.db"
  redis_enabled: true
  redis_url: "redis://localhost:6379"
  backup_interval_hours: 24
```

## Example Workflows

### 1. Business Process Automation

See `workflows/business_process.yaml` - Demonstrates approval workflows with:
- Request validation
- Authorization checks  
- Approval routing
- Notification handling
- Database updates

### 2. Data Processing Pipeline

See `workflows/data_pipeline.json` - Shows ETL workflow with:
- Data ingestion from APIs
- Schema validation
- Duplicate detection
- Data enrichment
- Quality assessment
- Multi-target output

### 3. Machine Learning Training

See `workflows/ml_training.yaml` - ML pipeline including:
- Data preparation
- Feature engineering
- Hyperparameter tuning
- Model training
- Validation
- Deployment

### 4. Simple Example

See `workflows/simple_example.yaml` - Basic workflow for learning:
- Input validation
- Data transformation
- Output generation

## API Reference

### Workflow Management

#### Start Workflow
```http
POST /api/workflows/{workflow_id}/start
Content-Type: application/json

{
  "inputs": {
    "user_id": "123",
    "amount": 1500
  }
}
```

#### Get Status
```http
GET /api/workflows/{instance_id}/status
```

#### Control Operations
```http
POST /api/workflows/{instance_id}/pause
POST /api/workflows/{instance_id}/resume  
POST /api/workflows/{instance_id}/stop
```

### Monitoring

#### Get Metrics
```http
GET /api/metrics
GET /api/metrics/workflow/{workflow_id}
```

#### Get Alerts
```http
GET /api/alerts
```

#### Performance Report
```http
GET /api/performance/report?report_type=summary
```

## Monitoring and Alerting

### Built-in Metrics

- **Workflow Metrics**: Success rate, execution time, failure count
- **Task Metrics**: Individual task performance and error rates  
- **Agent Metrics**: Health status, response times, resource usage
- **System Metrics**: Overall throughput and resource utilization

### Alert Types

- **Performance Alerts**: High failure rates, slow execution times
- **System Alerts**: Agent health, resource constraints
- **Business Alerts**: SLA violations, approval timeouts
- **Error Alerts**: Critical failures, escalated errors

### Dashboard Features

- **Real-time Status**: Live workflow and task status updates
- **Performance Charts**: Historical trends and performance analysis
- **Alert Management**: Alert acknowledgment and resolution tracking
- **Workflow Control**: Start, pause, resume, and stop operations

## Error Handling and Recovery

### Error Classification

Errors are automatically classified by severity:
- **Critical**: System failures, security breaches
- **High**: Network errors, timeouts, authentication failures
- **Medium**: Validation errors, resource constraints
- **Low**: Temporary issues, rate limits

### Recovery Strategies

- **Retry**: Automatic retry with exponential backoff
- **Skip**: Skip failed tasks and continue workflow
- **Fallback**: Execute alternative task paths
- **Pause**: Temporary pause for manual intervention
- **Rollback**: Revert to previous state
- **Escalate**: Human intervention required

### Circuit Breaker Pattern

Prevents cascade failures by temporarily disabling failing services:

```python
# Circuit breaker configuration
circuit_breaker:
  failure_threshold: 5
  timeout_seconds: 300
  half_open_max_calls: 3
```

## Testing

### Unit Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_agents.py
pytest tests/test_workflows.py
pytest tests/test_integration.py
```

### Integration Tests

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/
```

### Load Testing

```bash
# Install load testing dependencies
pip install locust

# Run load tests
locust -f tests/load/workflow_load_test.py
```

## Deployment

### Docker Deployment

```bash
# Build containers
docker-compose build

# Start all services
docker-compose up -d

# Scale task executors
docker-compose up -d --scale task-executor=3
```

### Kubernetes Deployment

```bash
# Apply manifests
kubectl apply -f k8s/

# Scale deployment
kubectl scale deployment task-executor --replicas=5
```

### Production Configuration

For production deployment:

1. **Security**: Enable authentication and encryption
2. **Monitoring**: Configure external monitoring (Prometheus, Grafana)
3. **Persistence**: Use external databases (PostgreSQL, Redis Cluster)
4. **Scaling**: Configure auto-scaling based on load
5. **Backup**: Implement automated backup strategies

## Performance Tuning

### Optimization Guidelines

1. **Concurrent Execution**: Tune max concurrent workflows/tasks
2. **Resource Allocation**: Optimize CPU and memory limits
3. **Database Tuning**: Configure connection pools and query optimization
4. **Caching Strategy**: Implement multi-level caching
5. **Network Optimization**: Use connection pooling and keep-alive

### Scaling Considerations

- **Horizontal Scaling**: Add more task executor instances
- **Vertical Scaling**: Increase resource allocation per agent
- **Database Scaling**: Use read replicas and sharding
- **Load Balancing**: Distribute requests across agent instances

## Security

### Authentication and Authorization

- **Agent Authentication**: Mutual TLS between agents
- **API Security**: JWT tokens for web interface
- **Role-Based Access**: Fine-grained permissions
- **Audit Logging**: Complete access and operation logs

### Data Protection

- **Encryption**: At-rest and in-transit encryption
- **Secrets Management**: Secure credential storage
- **Data Isolation**: Tenant-based data separation
- **Compliance**: GDPR, SOX compliance features

## Troubleshooting

### Common Issues

#### Agent Connection Problems
```bash
# Check agent health
curl http://localhost:8080/health

# Verify connectivity
telnet localhost 8080
```

#### Workflow Execution Issues
```bash
# Check workflow status
curl http://localhost:8090/api/workflows/{instance_id}/status

# View error logs
docker logs workflow-engine
```

#### Performance Issues
```bash
# Monitor resource usage
docker stats

# Check metrics
curl http://localhost:8090/api/metrics
```

### Debug Mode

Enable debug logging:

```yaml
logging:
  level: DEBUG
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Log Analysis

Key log patterns to monitor:
- Workflow state transitions
- Task execution times
- Error frequencies
- Agent health status

## Contributing

### Development Setup

1. **Fork Repository**: Create a fork of the main repository
2. **Install Dev Dependencies**: `pip install -r requirements-dev.txt`
3. **Pre-commit Hooks**: `pre-commit install`
4. **Run Tests**: `pytest tests/`

### Code Standards

- **Python Style**: Follow PEP 8 guidelines
- **Type Hints**: Use type annotations
- **Documentation**: Comprehensive docstrings
- **Testing**: Maintain >90% test coverage

### Submitting Changes

1. Create feature branch
2. Implement changes with tests
3. Update documentation
4. Submit pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:

- **Documentation**: See the `/docs` directory
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the development team

## Roadmap

### Planned Features

- **Visual Workflow Designer**: Drag-and-drop workflow creation
- **Advanced Analytics**: ML-powered performance insights
- **Multi-tenant Support**: Isolated tenant environments
- **Plugin System**: Custom agent and task plugins
- **Workflow Versioning**: Version control for workflow definitions

### Performance Improvements

- **Async Processing**: Enhanced async task execution
- **Message Queuing**: External message broker support
- **Caching Layers**: Multi-level caching strategies
- **Database Optimization**: Query optimization and indexing

---

This comprehensive workflow orchestration system demonstrates the power of the AMP protocol for building scalable, resilient, and observable distributed systems. The modular architecture allows for easy extension and customization while maintaining production-ready reliability and performance.