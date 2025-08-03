"""
Knowledge Base Agents Module

This module contains specialized agents for knowledge management:
- Knowledge Ingestion Agent: Document processing and indexing
- Semantic Search Agent: Vector search and similarity matching
- Knowledge Graph Agent: Entity relationships and graph traversal
- Query Router Agent: Intelligent query routing
- Cache Manager Agent: Caching and retrieval optimization
- Knowledge Curator Agent: Quality management and updates
"""

from .knowledge_ingestion import KnowledgeIngestionAgent
from .semantic_search import SemanticSearchAgent
from .knowledge_graph import KnowledgeGraphAgent
from .query_router import QueryRouterAgent
from .cache_manager import CacheManagerAgent
from .knowledge_curator import KnowledgeCuratorAgent

__all__ = [
    'KnowledgeIngestionAgent',
    'SemanticSearchAgent',
    'KnowledgeGraphAgent',
    'QueryRouterAgent',
    'CacheManagerAgent',
    'KnowledgeCuratorAgent'
]