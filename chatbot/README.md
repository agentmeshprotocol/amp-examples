# Multi-Agent Chatbot System (LangChain)

A sophisticated chatbot system that demonstrates multi-agent conversation handling using the Agent Mesh Protocol. The system routes user conversations to specialized agents based on intent and context, with seamless handoffs between agents.

## 🎯 Overview

This example showcases:
- **Domain-specific agent specialization** - Different agents handle FAQ, Sales, and Technical Support
- **Intelligent conversation routing** - Intent detection determines which agent handles each message
- **Context preservation** - Conversation history is maintained across agent handoffs
- **LangChain integration** - Uses LangChain for natural language processing and agent orchestration

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Input    │───▶│ Router Agent    │───▶│ Conversation    │
│                 │    │ (Intent         │    │ Orchestrator    │
└─────────────────┘    │  Detection)     │    └─────────────────┘
                       └─────────────────┘            │
                                                      ▼
                       ┌─────────────────────────────────────────────┐
                       │              Specialized Agents             │
                       ├─────────────┬─────────────┬─────────────────┤
                       │ FAQ Agent   │ Sales Agent │ Tech Support    │
                       │ - Common Q  │ - Product   │ - Troubleshoot  │
                       │ - Policies  │ - Pricing   │ - Complex Help  │
                       │ - Hours     │ - Demos     │ - Escalation    │
                       └─────────────┴─────────────┴─────────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install langchain openai faiss-cpu tiktoken
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Add your API keys
export OPENAI_API_KEY=your_openai_key
export AMP_REGISTRY_URL=http://localhost:8000
```

### Running the Example
```bash
# Start the chatbot system
python run_chatbot.py

# In another terminal, start the web interface
python web_interface.py

# Or use the CLI interface
python cli_interface.py
```

## 📁 Project Structure

```
chatbot/
├── agents/
│   ├── router_agent.py          # Intent detection and routing
│   ├── faq_agent.py            # FAQ handling with knowledge base
│   ├── sales_agent.py          # Sales inquiries and lead qualification
│   ├── tech_support_agent.py   # Technical support and troubleshooting
│   └── conversation_manager.py  # Conversation state management
├── config/
│   ├── agent_config.yaml       # Agent configurations
│   ├── intents.yaml           # Intent detection patterns
│   └── knowledge_base.yaml    # FAQ knowledge base
├── examples/
│   ├── cli_interface.py       # Command-line chat interface
│   ├── web_interface.py       # Web-based chat interface
│   └── api_client.py          # API client example
├── tests/
│   ├── test_routing.py        # Test intent detection
│   ├── test_agents.py         # Test individual agents
│   └── test_integration.py    # End-to-end tests
├── requirements.txt
├── run_chatbot.py             # Main application entry point
├── docker-compose.yml         # Docker deployment
└── README.md
```

## 🤖 Agent Descriptions

### Router Agent
- **Purpose**: Analyze user input and route to appropriate specialist
- **Capabilities**: `intent-detection`, `conversation-routing`
- **Technology**: LangChain + OpenAI for intent classification
- **Features**:
  - Multi-intent detection
  - Confidence scoring
  - Context-aware routing
  - Escalation handling

### FAQ Agent
- **Purpose**: Handle common questions and general inquiries
- **Capabilities**: `qa-factual`, `knowledge-retrieval`
- **Technology**: LangChain + FAISS vector store
- **Features**:
  - Semantic search over FAQ database
  - Dynamic response generation
  - Source attribution
  - Confidence scoring

### Sales Agent
- **Purpose**: Handle sales inquiries and lead qualification
- **Capabilities**: `lead-qualification`, `product-recommendation`
- **Technology**: LangChain + Custom sales logic
- **Features**:
  - Product matching
  - Price calculations
  - Demo scheduling
  - CRM integration ready

### Tech Support Agent
- **Purpose**: Provide technical assistance and troubleshooting
- **Capabilities**: `technical-support`, `problem-diagnosis`
- **Technology**: LangChain + Technical knowledge base
- **Features**:
  - Guided troubleshooting
  - Solution recommendations
  - Escalation to human support
  - Ticket creation

## 🔧 Configuration

### Agent Configuration (agent_config.yaml)
```yaml
router_agent:
  model: "gpt-3.5-turbo"
  temperature: 0.1
  max_tokens: 500
  confidence_threshold: 0.8

faq_agent:
  model: "gpt-3.5-turbo"
  temperature: 0.2
  vector_store: "faiss"
  similarity_threshold: 0.7

sales_agent:
  model: "gpt-3.5-turbo"
  temperature: 0.3
  crm_integration: true
  lead_scoring: true

tech_support_agent:
  model: "gpt-4"
  temperature: 0.1
  escalation_enabled: true
  ticket_system: "jira"
```

### Intent Patterns (intents.yaml)
```yaml
intents:
  faq:
    patterns:
      - "what are your hours"
      - "how to contact"
      - "company information"
      - "general question"
    examples:
      - "When are you open?"
      - "What's your address?"

  sales:
    patterns:
      - "pricing"
      - "buy"
      - "purchase"
      - "demo"
      - "trial"
    examples:
      - "How much does it cost?"
      - "I want to buy your product"

  technical:
    patterns:
      - "not working"
      - "error"
      - "bug"
      - "technical issue"
    examples:
      - "The app crashes when I try to login"
      - "I'm getting an error message"
```

## 💬 Conversation Flow Examples

### Simple FAQ Query
```
User: "What are your business hours?"
Router: [detects FAQ intent] → Routes to FAQ Agent
FAQ Agent: "Our business hours are Monday-Friday 9AM-6PM EST, 
           and Saturday 10AM-4PM EST. We're closed on Sundays."
```

### Complex Multi-Agent Interaction
```
User: "I'm having trouble with login and want to know about pricing"
Router: [detects multiple intents] → Routes to Tech Support first
Tech Support: "Let me help with the login issue first..."
[After resolving login]
Tech Support: [hands off to Sales Agent]
Sales Agent: "Now regarding pricing, here are our current plans..."
```

### Context-Aware Handoff
```
User: "Hi, I'm John from Acme Corp, we spoke yesterday about the enterprise plan"
Router: [recognizes return customer] → Routes to Sales Agent with context
Sales Agent: [accesses previous conversation] "Hi John! Yes, let me 
             continue where we left off with the enterprise features..."
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Individual Components
```bash
# Test intent detection
python -m pytest tests/test_routing.py

# Test agent responses
python -m pytest tests/test_agents.py

# Test full conversation flow
python -m pytest tests/test_integration.py
```

### Interactive Testing
```bash
# Test with CLI
python examples/cli_interface.py

# Test routing logic
python -c "
from agents.router_agent import RouterAgent
router = RouterAgent()
result = router.detect_intent('How much does your product cost?')
print(f'Intent: {result.intent}, Confidence: {result.confidence}')
"
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start all agents
python run_chatbot.py --mode development

# Start web interface
python examples/web_interface.py --port 8080
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Scale specific agents
docker-compose up -d --scale faq-agent=3
```

### Production Deployment
```bash
# Environment variables for production
export OPENAI_API_KEY=your_production_key
export AMP_REGISTRY_URL=https://your-registry.com
export LOG_LEVEL=WARNING
export ENABLE_METRICS=true

# Deploy with production config
python run_chatbot.py --config config/production.yaml
```

## 📊 Monitoring & Metrics

### Built-in Metrics
- Conversation volume by agent
- Intent detection accuracy
- Response times
- User satisfaction scores
- Handoff success rates

### Health Checks
```bash
# Check agent health
curl http://localhost:8000/health

# Check specific agent
curl http://localhost:8000/agents/faq-agent/health

# Get performance metrics
curl http://localhost:8000/metrics
```

## 🔧 Customization

### Adding New Agents
1. Create agent class inheriting from `AMPChatbotAgent`
2. Implement required capabilities
3. Update routing logic in `router_agent.py`
4. Add configuration to `agent_config.yaml`

### Custom Intent Detection
1. Update `intents.yaml` with new patterns
2. Train custom classifier if needed
3. Modify `router_agent.py` logic
4. Test with new conversation examples

### Integration Examples
- **CRM Integration**: Salesforce, HubSpot
- **Ticket Systems**: Jira, ServiceNow
- **Analytics**: Google Analytics, Mixpanel
- **Notifications**: Slack, Microsoft Teams

## 🤝 Contributing

See the main [Contributing Guide](../CONTRIBUTING.md) for general guidelines.

### Chatbot-Specific Guidelines
- Follow conversation design best practices
- Test with diverse user inputs
- Ensure graceful error handling
- Maintain conversation context
- Document new intent patterns

## 🐛 Troubleshooting

### Common Issues

**Issue**: Intent detection is inaccurate
**Solution**: 
- Check intent patterns in `intents.yaml`
- Increase training examples
- Adjust confidence thresholds
- Review conversation logs

**Issue**: Agents not responding
**Solution**:
- Check agent health endpoints
- Verify API keys are set
- Review agent logs
- Ensure proper AMP connectivity

**Issue**: Context not preserved
**Solution**:
- Check conversation manager state
- Verify session management
- Review handoff logic
- Check memory persistence

## 📚 Learning Resources

- [LangChain Documentation](https://langchain.readthedocs.io/)
- [Conversation Design Guide](https://developers.google.com/assistant/conversation-design)
- [Intent Recognition Best Practices](https://rasa.com/docs/rasa/training-data-format/)
- [Multi-Agent Systems](https://www.deeplearning.ai/courses/)

---

**Next Steps:**
- Explore the [Research Assistant Network](../research-assistant/) example
- Check out [Customer Support System](../support-system/) for advanced routing
- Learn about [Workflow Orchestration](../workflow/) for complex flows