# AMP Customer Support System

A comprehensive, production-ready customer support system built with the Agent Mesh Protocol (AMP), demonstrating intelligent ticket routing, automated resolution, and seamless agent collaboration across multiple AI frameworks.

## 🚀 Overview

This example showcases a sophisticated customer support system that leverages multiple AI frameworks working together through the AMP protocol to provide intelligent, automated customer support. The system demonstrates framework-agnostic agent collaboration with real-world applications.

## 🎯 Key Features

### Intelligent Ticket Management
- **Automatic Ticket Classification**: NLP-powered categorization and priority assignment
- **Smart Routing**: Context-aware routing to appropriate specialist agents
- **SLA Monitoring**: Automated tracking and escalation based on service levels
- **Multi-channel Support**: Web interface, API, and potential for email/chat integration

### AI Agent Collaboration
- **Mixed Framework Architecture**: LangChain, CrewAI, AutoGen, and custom agents working together
- **Seamless Communication**: AMP protocol enables framework-agnostic agent messaging
- **Collaborative Problem-Solving**: Agents work together on complex cases
- **Escalation Management**: Intelligent escalation workflows with human oversight

### Production-Ready Features
- **Web Interface**: Complete customer and admin web portal
- **REST API**: Full API for integration with existing systems
- **Real-time Monitoring**: Health checks and system status monitoring
- **Comprehensive Logging**: Detailed logging for debugging and analytics

## 🏗️ Architecture

### Support Agents

1. **Ticket Classifier Agent** (LangChain)
   - Categorizes incoming tickets using NLP
   - Assigns priority levels based on urgency indicators
   - Performs sentiment analysis for customer satisfaction tracking
   - Routes tickets to appropriate specialist agents

2. **Technical Support Agent** (CrewAI) 
   - Handles technical troubleshooting using collaborative agents
   - Provides step-by-step solutions
   - Performs system diagnostics
   - Escalates complex issues when needed

3. **Billing Support Agent** (AutoGen)
   - Manages billing inquiries and payment issues
   - Processes refunds and payment method updates
   - Handles subscription management
   - Collaborates with multiple specialized agents

4. **Product Support Agent** (LangChain)
   - Provides product guidance and feature explanations
   - Processes feature requests and enhancement suggestions
   - Offers training recommendations
   - Maintains product knowledge base

5. **Escalation Manager Agent** (Custom)
   - Manages complex cases requiring elevated attention
   - Coordinates cross-team collaboration
   - Monitors SLA compliance and triggers escalations
   - Provides executive reporting and insights

6. **Knowledge Base Agent** (Custom)
   - Provides instant answers from documentation
   - Performs semantic search across knowledge articles
   - Suggests related content and resources
   - Tracks knowledge base effectiveness

### Web Interface
- **Customer Portal**: Ticket submission and tracking
- **Support Dashboard**: Agent workbench and metrics
- **Admin Interface**: System monitoring and configuration
- **API Documentation**: OpenAPI/Swagger integration

## 📋 Prerequisites

- **Python 3.9+**
- **OpenAI API Key** (for LangChain, CrewAI, and AutoGen agents)
- **AMP Protocol Implementation** (included in shared-lib)

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/agentmeshprotocol/amp-examples.git
   cd amp-examples/support-system
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

   Required environment variables:
   ```bash
   OPENAI_API_KEY=your_openai_api_key_here
   AMP_ENDPOINT=http://localhost:8000
   LOG_LEVEL=INFO
   ```

5. **Create necessary directories**:
   ```bash
   mkdir -p logs data/metrics data/knowledge_base
   ```

## 🚀 Quick Start

### Option 1: Run with Web Interface (Recommended)
```bash
python run_support_system.py --mode web
```

This starts:
- All 6 AI support agents
- Web interface at http://localhost:8080
- REST API endpoints
- Real-time system monitoring

### Option 2: Run Agents Only
```bash
python run_support_system.py --mode agents-only
```

This runs only the AI agents without the web interface.

### Option 3: Run Individual Agents
```bash
# Ticket Classifier
python -m agents.ticket_classifier

# Technical Support
python -m agents.technical_support

# Billing Support  
python -m agents.billing_support

# Product Support
python -m agents.product_support

# Escalation Manager
python -m agents.escalation_manager

# Knowledge Base
python -m agents.knowledge_base
```

## 🌐 Using the Web Interface

### Customer Portal
1. **Submit Tickets**: http://localhost:8080/submit
2. **Track Tickets**: http://localhost:8080/tickets
3. **View Status**: http://localhost:8080/

### Support Dashboard
1. **Agent Dashboard**: http://localhost:8080/dashboard
2. **System Health**: http://localhost:8080/api/health
3. **Metrics**: Real-time agent and ticket metrics

### API Endpoints
- `GET /api/health` - System health check
- `GET /api/tickets` - List all tickets
- `POST /api/tickets` - Create new ticket
- `GET /api/tickets/{id}` - Get specific ticket
- `POST /api/tickets/{id}/comment` - Add ticket comment

## 🔧 Configuration

### Agent Configuration
Each agent can be configured via environment variables or config files:

```yaml
# config/agent_config.yaml
agents:
  ticket_classifier:
    model: "gpt-4"
    temperature: 0.1
    max_tokens: 1000
    
  technical_support:
    collaboration_mode: "round_robin"
    max_iterations: 5
    
  billing_support:
    approval_threshold: 500.00
    refund_limit: 1000.00
```

### SLA Configuration
Customize SLA targets for different customer tiers:

```python
# support_types.py - SLA_TARGETS
SLA_TARGETS = {
    SLALevel.ENTERPRISE: {
        TicketPriority.CRITICAL: SLATarget(5, 0.5, 0.25),  # 5m, 30m, 15m
        # ... other priorities
    }
}
```

## 📊 Monitoring and Analytics

### System Health
- Real-time agent health monitoring
- Performance metrics tracking
- SLA compliance reporting
- Customer satisfaction analytics

### Logging
Comprehensive logging is available in the `logs/` directory:
- `support_system.log` - Main system log
- `agents/` - Individual agent logs
- `metrics/` - Performance metrics

### Metrics Collection
The system tracks:
- Ticket volume and resolution times
- Agent performance and utilization
- Customer satisfaction scores
- SLA compliance rates
- System health and uptime

## 🧪 Testing

### Unit Tests
```bash
python -m pytest tests/test_agents.py
```

### Integration Tests
```bash
python -m pytest tests/test_integration.py
```

### Load Testing
```bash
python tests/load_test.py --tickets 100 --concurrent 10
```

## 🔌 API Integration

### Creating Tickets via API
```python
import requests

ticket_data = {
    "subject": "Login issues",
    "description": "Cannot access my account",
    "category": "technical",
    "priority": "high",
    "customer_name": "John Doe",
    "customer_email": "john@example.com",
    "sla_level": "premium"
}

response = requests.post(
    "http://localhost:8080/api/tickets",
    json=ticket_data
)
```

### Webhook Integration
```python
# webhook_handler.py
@app.post("/webhook/ticket-created")
async def handle_ticket_created(ticket_data: dict):
    # Process incoming ticket from external system
    # Forward to AMP support system
    pass
```

## 🎨 Customization

### Adding Custom Agents
1. Create agent class inheriting from base AMP agent
2. Register capabilities with the AMP client
3. Add to orchestrator configuration
4. Update web interface if needed

```python
class CustomSupportAgent:
    def __init__(self, config: AMPClientConfig):
        self.client = AMPClient(config)
    
    async def start(self):
        await self._register_capabilities()
        await self.client.connect()
    
    async def _register_capabilities(self):
        # Register custom capabilities
        pass
```

### Custom Workflows
Extend the system with custom workflows:
- Multi-step approval processes
- Integration with external systems
- Custom escalation rules
- Automated resolution workflows

## 🚀 Deployment

### Docker Deployment
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_support_system.py", "--mode", "web"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  support-system:
    build: .
    ports:
      - "8080:8080"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - AMP_ENDPOINT=http://localhost:8000
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
```

### Production Considerations
- Use environment-specific configurations
- Implement proper logging and monitoring
- Set up database persistence
- Configure load balancing for high availability
- Implement security best practices

## 📚 Documentation

### API Documentation
Interactive API documentation is available at:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

### Agent Documentation
Each agent includes comprehensive docstrings and examples:
- Capability descriptions
- Input/output schemas
- Usage examples
- Integration patterns

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

Please see [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed guidelines.

## 🐛 Troubleshooting

### Common Issues

**Agents won't start**:
- Check OPENAI_API_KEY is set correctly
- Verify all dependencies are installed
- Check logs for specific error messages

**Web interface not accessible**:
- Ensure port 8080 is not in use
- Check firewall settings
- Verify FastAPI is running correctly

**Poor classification accuracy**:
- Review and expand training examples
- Adjust confidence thresholds
- Update classification prompts

### Debug Mode
Enable debug logging:
```bash
python run_support_system.py --log-level DEBUG
```

### Performance Tuning
- Adjust agent timeout settings
- Optimize LLM model selection
- Implement caching for knowledge base
- Use connection pooling for high load

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **Agent Mesh Protocol** community for the foundational framework
- **LangChain**, **CrewAI**, and **AutoGen** teams for excellent AI agent frameworks
- **FastAPI** and **Bootstrap** for web interface components
- **OpenAI** for powering the AI capabilities

## 📞 Support

For questions about this example:
- Open an issue on GitHub
- Join the AMP community discussions
- Check the documentation at [agentmeshprotocol.org](https://agentmeshprotocol.org)

---

*Built with ❤️ using the Agent Mesh Protocol*