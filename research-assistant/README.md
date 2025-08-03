# Research Assistant Network (CrewAI)

A sophisticated research assistant system that demonstrates multi-agent coordination using CrewAI. The system orchestrates specialized agents to perform comprehensive research, fact-checking, and content creation workflows.

## 🎯 Overview

This example showcases:
- **CrewAI multi-agent orchestration** - Coordinated workflow execution across specialized agents
- **Research pipeline automation** - Web search, content analysis, fact-checking, and synthesis
- **Agent specialization** - Dedicated agents for search, analysis, fact-checking, and writing
- **Quality assurance** - Multi-stage validation and review processes
- **AMP protocol integration** - Seamless communication between agents using Agent Mesh Protocol

## 🏗 Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│ Research Crew   │───▶│ Final Report    │
│                 │    │ Orchestrator    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                               │
                               ▼
       ┌─────────────────────────────────────────────────────────┐
       │              Research Agent Network                     │
       ├─────────────┬─────────────┬─────────────┬─────────────┤
       │ Web Search  │ Content     │ Fact Check  │ Synthesis   │
       │ Agent       │ Analyzer    │ Agent       │ Agent       │
       │ - Query web │ - Extract   │ - Verify    │ - Combine   │
       │ - Crawl     │ - Analyze   │ - Cross-ref │ - Structure │
       │ - Filter    │ - Summarize │ - Score     │ - Format    │
       └─────────────┴─────────────┴─────────────┴─────────────┘
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install crewai openai duckduckgo-search beautifulsoup4 newspaper3k
```

### Environment Setup
```bash
# Copy environment template
cp .env.example .env

# Add your API keys
export OPENAI_API_KEY=your_openai_key
export SERPER_API_KEY=your_serper_key  # Optional: for enhanced search
export AMP_REGISTRY_URL=http://localhost:8000
```

### Running the Example
```bash
# Start the research assistant system
python run_research_assistant.py

# Use the CLI interface for research queries
python examples/research_cli.py

# Start the web interface
python examples/web_interface.py

# Run a specific research workflow
python examples/research_workflow.py --topic "artificial intelligence trends 2024"
```

## 📁 Project Structure

```
research-assistant/
├── agents/
│   ├── web_search_agent.py      # Web search and content retrieval
│   ├── content_analyzer.py      # Content analysis and extraction
│   ├── fact_checker.py          # Fact verification and validation
│   ├── synthesis_agent.py       # Content synthesis and report generation
│   └── research_orchestrator.py # CrewAI crew coordination
├── crews/
│   ├── research_crew.py         # Main research crew definition
│   ├── fact_check_crew.py       # Specialized fact-checking crew
│   └── content_creation_crew.py # Content creation and formatting crew
├── tools/
│   ├── web_search_tools.py      # Web search utilities
│   ├── content_extraction.py    # Content extraction tools
│   ├── fact_checking_tools.py   # Fact verification tools
│   └── synthesis_tools.py       # Content synthesis utilities
├── config/
│   ├── crew_config.yaml         # CrewAI crew configurations
│   ├── agent_config.yaml        # Individual agent settings
│   └── search_config.yaml       # Search parameters and sources
├── examples/
│   ├── research_cli.py          # Command-line research interface
│   ├── web_interface.py         # Web-based research dashboard
│   ├── research_workflow.py     # Workflow automation examples
│   └── batch_research.py        # Batch processing capabilities
├── tests/
│   ├── test_agents.py           # Individual agent tests
│   ├── test_crews.py            # Crew coordination tests
│   └── test_workflows.py        # End-to-end workflow tests
├── requirements.txt
├── run_research_assistant.py   # Main application entry point
├── docker-compose.yml          # Docker deployment
└── README.md
```

## 🤖 Agent Descriptions

### Web Search Agent
- **Purpose**: Discover and retrieve relevant information from the web
- **Capabilities**: `web-search`, `content-retrieval`, `source-validation`
- **Technology**: CrewAI + DuckDuckGo/Serper + BeautifulSoup
- **Features**:
  - Multi-source web search
  - Content relevance scoring
  - Source credibility assessment
  - Duplicate detection and filtering

### Content Analyzer Agent
- **Purpose**: Analyze and extract key information from retrieved content
- **Capabilities**: `content-analysis`, `key-extraction`, `summarization`
- **Technology**: CrewAI + LangChain + NLP libraries
- **Features**:
  - Intelligent content parsing
  - Key concept extraction
  - Automatic summarization
  - Sentiment and bias detection

### Fact Checker Agent
- **Purpose**: Verify claims and validate information accuracy
- **Capabilities**: `fact-verification`, `source-validation`, `credibility-scoring`
- **Technology**: CrewAI + Multiple verification sources
- **Features**:
  - Cross-reference verification
  - Source authority scoring
  - Claim confidence assessment
  - Conflict detection and resolution

### Synthesis Agent
- **Purpose**: Combine verified information into coherent reports
- **Capabilities**: `content-synthesis`, `report-generation`, `formatting`
- **Technology**: CrewAI + LangChain for structured output
- **Features**:
  - Multi-source content integration
  - Structured report generation
  - Citation management
  - Quality assurance checks

## 🔧 Configuration

### Crew Configuration (crew_config.yaml)
```yaml
research_crew:
  name: "Comprehensive Research Crew"
  process: "sequential"  # or "hierarchical"
  verbose: true
  memory: true
  max_execution_time: 1800  # 30 minutes
  
  agents:
    - web_search_agent
    - content_analyzer
    - fact_checker
    - synthesis_agent
    
  tasks:
    - search_task
    - analysis_task
    - fact_check_task
    - synthesis_task

fact_check_crew:
  name: "Fact Verification Crew"
  process: "parallel"
  focus: "accuracy_validation"
  timeout: 600  # 10 minutes
```

### Search Configuration (search_config.yaml)
```yaml
search_sources:
  primary:
    - duckduckgo
    - serper  # If API key available
  
  specialized:
    - arxiv      # Academic papers
    - pubmed     # Medical research
    - news_api   # News articles
    - wikipedia  # General knowledge
    
search_parameters:
  max_results_per_source: 10
  search_depth: 3
  language: "en"
  time_range: "1y"  # Last year
  
content_filters:
  min_word_count: 100
  max_age_days: 365
  credibility_threshold: 0.7
```

## 💬 Research Flow Examples

### Simple Research Query
```
User: "What are the latest developments in quantum computing?"

1. Web Search Agent finds recent articles and papers
2. Content Analyzer extracts key developments and trends
3. Fact Checker verifies claims against multiple sources
4. Synthesis Agent creates comprehensive report with citations
```

### Complex Multi-Part Research
```
User: "Compare the environmental impact of electric vs hydrogen vehicles, 
       including manufacturing, operation, and disposal phases"

1. Search Agent gathers data on both vehicle types across all phases
2. Analyzer categorizes information by impact type and lifecycle phase
3. Fact Checker validates environmental claims and statistics
4. Synthesis Agent creates comparative analysis with recommendations
```

### Fact-Checking Workflow
```
User: "Verify: 'Solar panels lose 50% efficiency after 10 years'"

1. Search Agent finds studies on solar panel degradation
2. Analyzer extracts degradation rates from multiple sources
3. Fact Checker cross-references data and assesses claim accuracy
4. Synthesis Agent provides verdict with supporting evidence
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Test Individual Components
```bash
# Test agent functionality
python -m pytest tests/test_agents.py

# Test crew coordination
python -m pytest tests/test_crews.py

# Test complete workflows
python -m pytest tests/test_workflows.py
```

### Interactive Testing
```bash
# Test with CLI
python examples/research_cli.py

# Test specific research topics
python examples/research_workflow.py --topic "climate change solutions" --depth comprehensive

# Test fact-checking
python examples/fact_check_demo.py --claim "specific claim to verify"
```

## 🚀 Deployment

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Start all agents
python run_research_assistant.py --mode development

# Start web interface
python examples/web_interface.py --port 8080
```

### Docker Deployment
```bash
# Build and run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f research-assistant

# Scale search agents for higher throughput
docker-compose up -d --scale web-search-agent=3
```

### Production Deployment
```bash
# Environment variables for production
export OPENAI_API_KEY=your_production_key
export SERPER_API_KEY=your_serper_key
export AMP_REGISTRY_URL=https://your-registry.com
export LOG_LEVEL=WARNING
export ENABLE_METRICS=true

# Deploy with production config
python run_research_assistant.py --config config/production.yaml
```

## 📊 Monitoring & Metrics

### Built-in Metrics
- Research query volume and success rates
- Agent performance and response times
- Source reliability and coverage
- Fact-checking accuracy scores
- User satisfaction ratings

### Health Checks
```bash
# Check system health
curl http://localhost:8000/health

# Check specific crew status
curl http://localhost:8000/crews/research-crew/health

# Get performance metrics
curl http://localhost:8000/metrics
```

## 🔧 Customization

### Adding New Search Sources
1. Create source adapter in `tools/web_search_tools.py`
2. Update `search_config.yaml` with source parameters
3. Register source with Web Search Agent
4. Test with sample queries

### Custom Fact-Checking Rules
1. Define verification logic in `tools/fact_checking_tools.py`
2. Update Fact Checker Agent configuration
3. Add rule tests to validation suite
4. Deploy with updated crew configuration

### Integration Examples
- **CRM Integration**: Salesforce, HubSpot for lead research
- **Knowledge Management**: Confluence, Notion for internal sources
- **Analytics**: Google Analytics, Mixpanel for usage tracking
- **Notifications**: Slack, Microsoft Teams for alerts

## 🤝 Contributing

See the main [Contributing Guide](../CONTRIBUTING.md) for general guidelines.

### Research Assistant Specific Guidelines
- Follow CrewAI best practices for agent and crew design
- Ensure fact-checking accuracy and source attribution
- Test with diverse research topics and complexity levels
- Maintain response time performance standards
- Document new search sources and verification methods

## 🐛 Troubleshooting

### Common Issues

**Issue**: Search agents returning irrelevant results
**Solution**: 
- Adjust search parameters in `search_config.yaml`
- Improve query refinement logic
- Check source-specific configurations
- Review relevance scoring algorithms

**Issue**: Fact-checking taking too long
**Solution**:
- Implement parallel verification processes
- Cache frequently verified claims
- Optimize cross-reference lookup
- Set appropriate timeout limits

**Issue**: Report quality inconsistent
**Solution**:
- Review synthesis agent prompts
- Implement quality scoring metrics
- Add human-in-the-loop validation
- Enhance source diversity requirements

## 📚 Learning Resources

- [CrewAI Documentation](https://docs.crewai.com/)
- [Research Methodology Guide](https://www.scribbr.com/methodology/)
- [Fact-Checking Best Practices](https://www.poynter.org/fact-checking/)
- [Information Retrieval Techniques](https://nlp.stanford.edu/IR-book/)

---

**Next Steps:**
- Explore the [Data Analysis Pipeline](../data-pipeline/) example
- Check out [Customer Support System](../support-system/) for workflow automation
- Learn about [Workflow Orchestration](../workflow/) for complex processes