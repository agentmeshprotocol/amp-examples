# Knowledge Base Agent System

A comprehensive, production-ready knowledge management system built on the Agent Mesh Protocol (AMP), demonstrating advanced RAG (Retrieval-Augmented Generation), semantic search, knowledge graphs, and intelligent caching through distributed AI agent collaboration.

## Overview

This capstone example showcases the full power of the AMP protocol by implementing a sophisticated knowledge management system with the following components:

- **🧠 Knowledge Ingestion Agent**: Advanced document processing and indexing
- **🔍 Semantic Search Agent**: Vector-based similarity search with hybrid capabilities  
- **🕸️ Knowledge Graph Agent**: Entity extraction and relationship mapping
- **🎯 Query Router Agent**: Intelligent query routing and result aggregation
- **🚀 Cache Manager Agent**: Multi-level caching with similarity matching
- **📊 Knowledge Curator Agent**: Quality analysis and content optimization
- **🌐 Web Interface**: User-friendly knowledge management portal
- **⚡ Admin Interface**: Advanced curation and analytics dashboard

## Features

### Core Capabilities
- **Multi-format Document Processing**: PDF, DOCX, HTML, Markdown, plain text
- **Advanced Vector Search**: FAISS and ChromaDB integration with HNSW indexing
- **Knowledge Graph**: NetworkX and Neo4j support for entity relationships
- **Intelligent Caching**: Redis-backed multi-strategy caching system
- **Quality Management**: Automated content quality analysis and scoring
- **Hybrid Search**: Combination of semantic and keyword search
- **Real-time Analytics**: System performance and content quality metrics

### Agent Architecture
- **Distributed Processing**: Each agent specializes in specific knowledge tasks
- **Fault Tolerance**: Automatic failover and graceful error handling
- **Performance Optimization**: Parallel processing and intelligent routing
- **Scalability**: Horizontal scaling support through AMP protocol

## Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install system dependencies
sudo apt-get install -y python3-dev build-essential

# Install spaCy language model
python -m spacy download en_core_web_sm
```

### Installation

1. **Clone and Setup**:
```bash
cd amp-examples/knowledge-base
pip install -r requirements.txt
```

2. **Check Dependencies**:
```bash
python run_knowledge_base.py --check-deps
```

3. **Create Sample Data**:
```bash
python run_knowledge_base.py --create-samples
```

4. **Start the System**:
```bash
# Start all agents and interfaces
python run_knowledge_base.py

# Or with custom configuration
python run_knowledge_base.py --config config/custom_config.yaml
```

### Quick Test

Once the system is running:

1. **Web Interface**: http://localhost:8080
2. **Admin Interface**: http://localhost:8081 (admin/knowledge_admin_2024)
3. **Upload Documents**: Use the web interface to upload sample PDFs or text files
4. **Search**: Try semantic searches like "machine learning algorithms"
5. **Explore Graph**: View entity relationships in the knowledge graph

## Architecture

### Agent Communication Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Client    │    │  Admin Client   │    │  API Clients    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────▼───────────────┐
                    │      Query Router Agent     │
                    │   (Intelligent Routing)     │
                    └─────────────┬───────────────┘
                                  │
          ┌───────────────────────┼───────────────────────┐
          │                       │                       │
┌─────────▼─────────┐   ┌─────────▼─────────┐   ┌─────────▼─────────┐
│  Semantic Search  │   │ Knowledge Graph   │   │  Cache Manager    │
│      Agent        │   │      Agent        │   │      Agent        │
└─────────┬─────────┘   └─────────┬─────────┘   └─────────┬─────────┘
          │                       │                       │
          └───────────────────────┼───────────────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Knowledge Ingestion Agent  │
                    │   (Document Processing)     │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  Knowledge Curator Agent    │
                    │   (Quality Management)      │
                    └─────────────────────────────┘
```

### Data Flow

1. **Document Ingestion**:
   - Upload → Ingestion Agent → Text Extraction → Chunking → Embedding Generation
   - Chunks sent to Search Agent for indexing
   - Entities extracted and sent to Graph Agent

2. **Query Processing**:
   - Query → Router Agent → Intent Analysis → Agent Selection
   - Parallel/Sequential execution based on query complexity
   - Result aggregation and deduplication

3. **Caching Strategy**:
   - Query results cached with TTL
   - Similarity-based cache matching
   - Multi-level eviction policies

## Configuration

### Agent Configuration (`config/agent_config.yaml`)

```yaml
agents:
  knowledge-ingestion-agent:
    enabled: true
    config:
      chunk_size: 1000
      chunk_overlap: 200
      embedding_model: "all-MiniLM-L6-v2"
      
  semantic-search-agent:
    enabled: true
    config:
      vector_index:
        type: "HNSW"
        dimension: 384
      similarity_threshold: 0.0
      
  cache-manager-agent:
    enabled: true
    config:
      cache_backend: "redis"
      strategies:
        query_results:
          ttl: 3600
          max_size: 10000
```

### Docker Deployment

```bash
# Start with Docker Compose
cd config
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f knowledge-ingestion
```

## Usage Examples

### 1. Document Upload and Processing

```python
import asyncio
from agents.knowledge_ingestion import KnowledgeIngestionAgent

async def upload_document():
    agent = KnowledgeIngestionAgent()
    await agent.start()
    
    result = await agent._handle_document_ingestion({
        "file_path": "/path/to/document.pdf",
        "metadata": {
            "title": "Research Paper",
            "author": "John Doe",
            "tags": ["AI", "Machine Learning"]
        }
    })
    
    print(f"Processed {result['chunk_count']} chunks")
    await agent.stop()

asyncio.run(upload_document())
```

### 2. Semantic Search

```python
import asyncio
from agents.semantic_search import SemanticSearchAgent

async def search_knowledge():
    agent = SemanticSearchAgent()
    await agent.start()
    
    result = await agent._handle_semantic_search({
        "query": "machine learning algorithms for text analysis",
        "k": 10,
        "threshold": 0.7
    })
    
    for item in result['results']:
        print(f"Score: {item['similarity']:.3f} - {item['text'][:100]}...")
    
    await agent.stop()

asyncio.run(search_knowledge())
```

### 3. Knowledge Graph Exploration

```python
import asyncio
from agents.knowledge_graph import KnowledgeGraphAgent

async def explore_entities():
    agent = KnowledgeGraphAgent()
    await agent.start()
    
    # Find entities
    entities = await agent._handle_graph_search({
        "query": "artificial intelligence",
        "entity_type": "ORG",
        "max_results": 5
    })
    
    # Find relationships
    if entities['results']:
        entity_id = entities['results'][0]['id']
        related = await agent._handle_find_related({
            "entity_id": entity_id,
            "max_depth": 2
        })
        
        print(f"Found {len(related['related_entities'])} related entities")
    
    await agent.stop()

asyncio.run(explore_entities())
```

### 4. Quality Analysis

```python
import asyncio
from agents.knowledge_curator import KnowledgeCuratorAgent

async def analyze_quality():
    agent = KnowledgeCuratorAgent()
    await agent.start()
    
    content = {
        "id": "doc_1",
        "text": "Sample document content for quality analysis...",
        "metadata": {"title": "Sample Document"}
    }
    
    result = await agent._handle_quality_analysis({
        "content": content
    })
    
    quality = result['quality_analysis']
    print(f"Quality Score: {quality['quality_score']:.2f}")
    print(f"Issues: {quality['issues']}")
    print(f"Recommendations: {quality['recommendations']}")
    
    await agent.stop()

asyncio.run(analyze_quality())
```

## Web Interfaces

### User Interface (Port 8080)

- **Search**: Advanced search with filters and result ranking
- **Upload**: Drag-and-drop document upload with metadata
- **Analytics**: Content overview and quality metrics
- **Graph**: Interactive knowledge graph visualization

### Admin Interface (Port 8081)

- **Dashboard**: System health and performance metrics
- **Quality Management**: Bulk quality analysis and thresholds
- **Cache Management**: Cache optimization and statistics
- **Agent Management**: Agent status and restart capabilities
- **Data Management**: Backup, cleanup, and maintenance tools
- **System Logs**: Real-time log monitoring and filtering

## API Reference

### Knowledge Search API

```bash
# Semantic search
curl -X POST http://localhost:8080/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "machine learning algorithms",
    "search_type": "hybrid",
    "filters": {"content_type": "pdf"}
  }'

# Graph search
curl -X GET "http://localhost:8080/api/graph/search?query=AI&entity_type=ORG"

# Cache statistics
curl -X GET http://localhost:8080/api/cache/stats
```

### Admin API

```bash
# System health
curl -X GET http://localhost:8081/api/system/health \
  -u admin:knowledge_admin_2024

# Quality analysis
curl -X POST http://localhost:8081/api/quality/bulk-analysis \
  -u admin:knowledge_admin_2024

# Export metrics
curl -X GET "http://localhost:8081/api/metrics/export?format=csv&period=7d" \
  -u admin:knowledge_admin_2024
```

## Performance Optimization

### Vector Search Optimization

- **HNSW Index**: Hierarchical Navigable Small World for fast similarity search
- **Embedding Caching**: Cache embeddings to avoid recomputation
- **Batch Processing**: Process multiple queries in batches

### Caching Strategies

- **Query Result Caching**: Cache search results with TTL
- **Similarity Matching**: Find similar cached queries
- **Multi-level Eviction**: LRU and LFU policies for different data types

### Scaling Considerations

- **Horizontal Scaling**: Add more agent instances behind load balancer
- **Database Sharding**: Distribute vector indices across multiple stores
- **Async Processing**: Non-blocking operations for better throughput

## Monitoring and Maintenance

### Health Checks

```bash
# Check agent status
python run_knowledge_base.py --check-deps

# System health endpoint
curl http://localhost:8081/api/system/health
```

### Logs and Debugging

```bash
# View agent logs
tail -f logs/agents.log

# Debug mode
python run_knowledge_base.py --log-level DEBUG
```

### Backup and Recovery

```bash
# Create backup
curl -X POST http://localhost:8081/api/data/backup \
  -u admin:knowledge_admin_2024 \
  -o backup_$(date +%Y%m%d).json

# Cache cleanup
curl -X POST http://localhost:8081/api/cache/control \
  -u admin:knowledge_admin_2024 \
  -H "Content-Type: application/json" \
  -d '{"command": "cleanup"}'
```

## Troubleshooting

### Common Issues

1. **spaCy Model Missing**:
```bash
python -m spacy download en_core_web_sm
```

2. **Redis Connection Failed**:
```bash
# Start Redis
docker run -d -p 6379:6379 redis:alpine

# Or install locally
sudo apt-get install redis-server
sudo systemctl start redis-server
```

3. **Vector Index Corruption**:
```bash
# Remove corrupted index
rm -rf data/vector_index*

# Restart system to rebuild
python run_knowledge_base.py
```

4. **Memory Issues**:
```bash
# Reduce batch sizes in config
chunk_size: 500  # Reduce from 1000
processing_batch_size: 5  # Reduce from 10
```

### Performance Issues

- **Slow Search**: Check vector index type and parameters
- **High Memory Usage**: Reduce embedding cache size
- **Cache Misses**: Adjust similarity thresholds for cache matching

## Development

### Adding Custom Agents

1. Create agent class inheriting from base patterns
2. Register capabilities in AMP protocol
3. Add to orchestrator configuration
4. Implement message handlers

### Extending Search Capabilities

1. Add new embedding models in configuration
2. Implement custom similarity metrics
3. Create specialized retrieval strategies
4. Add domain-specific filtering

### Custom Quality Metrics

1. Extend QualityAnalyzer class
2. Add new scoring algorithms
3. Configure weights in agent config
4. Update validation standards

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-capability`
3. Implement changes with tests
4. Submit pull request with description

## License

This example is part of the Agent Mesh Protocol project and follows the same licensing terms.

## Support

For questions and support:
- Check the troubleshooting section above
- Review agent logs for error details
- Consult the AMP protocol documentation
- Open issues for bug reports or feature requests

---

**Note**: This is a demonstration system showcasing AMP capabilities. For production deployment, additional security, monitoring, and scaling considerations should be implemented.