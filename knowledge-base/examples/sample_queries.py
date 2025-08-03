#!/usr/bin/env python3
"""
Sample Queries for Knowledge Base System

This script demonstrates various types of queries and interactions
with the knowledge base system through the AMP protocol.
"""

import asyncio
import json
import sys
import os
from typing import Dict, List

# Add shared-lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient


class KnowledgeBaseQueryExamples:
    """Demonstrates various query patterns for the knowledge base"""
    
    def __init__(self):
        self.client = AMPClient("query-examples")
        
    async def connect(self):
        """Connect to the AMP system"""
        await self.client.connect("ws://localhost:8000/ws")
        print("Connected to Knowledge Base System")
    
    async def disconnect(self):
        """Disconnect from the AMP system"""
        await self.client.disconnect()
        print("Disconnected from Knowledge Base System")
    
    async def semantic_search_examples(self):
        """Demonstrate semantic search capabilities"""
        print("\n=== Semantic Search Examples ===")
        
        # Example queries that showcase semantic understanding
        queries = [
            "machine learning algorithms for natural language processing",
            "artificial intelligence applications in healthcare",
            "deep learning techniques for computer vision",
            "data preprocessing methods for ML models",
            "neural network architectures for text classification",
            "unsupervised learning algorithms for clustering",
            "reinforcement learning in autonomous systems",
            "transfer learning in image recognition"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            try:
                response = await self.client.send_request(
                    "query-router-agent",
                    "route-query",
                    {
                        "query": query,
                        "routing_options": {
                            "search_type": "semantic"
                        }
                    }
                )
                
                results = response.get("results", [])
                routing_info = response.get("routing_info", {})
                
                print(f"   Found {len(results)} results")
                print(f"   Routed to: {routing_info.get('source_agents', [])}")
                print(f"   Confidence: {routing_info.get('confidence', 0):.2f}")
                
                # Show top result
                if results:
                    top_result = results[0]
                    score = top_result.get('score', 0) or top_result.get('similarity', 0)
                    text_preview = top_result.get('text', '')[:150] + '...'
                    print(f"   Top result ({score:.3f}): {text_preview}")
                    
            except Exception as e:
                print(f"   Error: {e}")
    
    async def hybrid_search_examples(self):
        """Demonstrate hybrid search (semantic + keyword)"""
        print("\n=== Hybrid Search Examples ===")
        
        queries = [
            "Python programming AND machine learning",
            "data science OR statistics",
            "artificial intelligence NOT robotics",
            '"neural networks" deep learning',
            "algorithm performance optimization"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n{i}. Hybrid Query: '{query}'")
            
            try:
                response = await self.client.send_request(
                    "query-router-agent", 
                    "route-query",
                    {
                        "query": query,
                        "routing_options": {
                            "search_type": "hybrid",
                            "vector_weight": 0.7,
                            "keyword_weight": 0.3
                        }
                    }
                )
                
                results = response.get("results", [])
                print(f"   Found {len(results)} results from hybrid search")
                
                if results:
                    for j, result in enumerate(results[:3], 1):
                        score = result.get('score', 0)
                        source = result.get('source_agent', 'unknown')
                        print(f"   {j}. Score: {score:.3f} | Source: {source}")
                        
            except Exception as e:
                print(f"   Error: {e}")
    
    async def knowledge_graph_examples(self):
        """Demonstrate knowledge graph queries"""
        print("\n=== Knowledge Graph Examples ===")
        
        # Entity search examples
        entity_queries = [
            {"query": "artificial intelligence", "entity_type": ""},
            {"query": "machine learning", "entity_type": "CONCEPT"},
            {"query": "Google", "entity_type": "ORG"},
            {"query": "Python", "entity_type": "LANGUAGE"}
        ]
        
        for i, query_data in enumerate(entity_queries, 1):
            query = query_data["query"]
            entity_type = query_data["entity_type"]
            
            print(f"\n{i}. Entity Search: '{query}' (Type: {entity_type or 'Any'})")
            
            try:
                response = await self.client.send_request(
                    "knowledge-graph-agent",
                    "graph-search",
                    {
                        "query": query,
                        "entity_type": entity_type if entity_type else None,
                        "max_results": 5
                    }
                )
                
                entities = response.get("results", [])
                print(f"   Found {len(entities)} entities")
                
                for entity in entities[:3]:
                    print(f"   - {entity.get('text', 'N/A')} ({entity.get('label', 'Unknown')})")
                    
                # If we found entities, try to find relationships
                if entities:
                    entity_id = entities[0].get('id')
                    if entity_id:
                        print(f"\n   Finding entities related to '{entities[0].get('text')}'...")
                        
                        related_response = await self.client.send_request(
                            "knowledge-graph-agent",
                            "find-related",
                            {
                                "entity_id": entity_id,
                                "max_depth": 2
                            }
                        )
                        
                        related_entities = related_response.get("related_entities", [])
                        print(f"   Found {len(related_entities)} related entities")
                        
                        for related in related_entities[:3]:
                            print(f"   → {related.get('text', 'N/A')}")
                            
            except Exception as e:
                print(f"   Error: {e}")
    
    async def quality_analysis_examples(self):
        """Demonstrate quality analysis capabilities"""
        print("\n=== Quality Analysis Examples ===")
        
        # Sample content for quality analysis
        sample_contents = [
            {
                "id": "high_quality_doc",
                "text": """
Artificial Intelligence: A Comprehensive Overview

Artificial Intelligence (AI) represents one of the most significant technological 
advancements of the 21st century. This field of computer science focuses on creating 
systems capable of performing tasks that typically require human intelligence, 
including learning, reasoning, problem-solving, and decision-making.

The foundation of modern AI rests on several key pillars: machine learning algorithms 
that enable systems to improve performance through experience, natural language 
processing for human-computer communication, computer vision for interpreting visual 
information, and robotics for physical world interaction.

Recent breakthroughs in deep learning, particularly with transformer architectures 
and large language models, have revolutionized AI capabilities across numerous domains. 
These developments have profound implications for industries ranging from healthcare 
and finance to transportation and education.

As we advance into an AI-driven future, it becomes increasingly important to consider 
ethical implications, ensure responsible development, and maintain human oversight 
in critical applications.
                """.strip(),
                "metadata": {
                    "title": "Artificial Intelligence: A Comprehensive Overview",
                    "author": "Dr. Jane Smith",
                    "date": "2024-01-15",
                    "source": "AI Research Journal"
                }
            },
            {
                "id": "low_quality_doc", 
                "text": """
ai is good. it help people. many company use ai now. ai make thing better maybe.
some people scared of ai but ai not bad if use good. future have more ai.
                """.strip(),
                "metadata": {
                    "title": "About AI"
                }
            },
            {
                "id": "medium_quality_doc",
                "text": """
Machine learning is a subset of artificial intelligence that focuses on algorithms 
and statistical models. These systems can automatically improve their performance 
on a specific task through experience without being explicitly programmed.

Common types include supervised learning, unsupervised learning, and reinforcement 
learning. Applications are found in recommendation systems, fraud detection, and 
predictive analytics. However, challenges remain in interpretability and bias.
                """.strip(),
                "metadata": {
                    "title": "Machine Learning Overview",
                    "author": "Student",
                    "date": "2024-02-01"
                }
            }
        ]
        
        for i, content in enumerate(sample_contents, 1):
            print(f"\n{i}. Analyzing: '{content['metadata'].get('title', 'Untitled')}'")
            
            try:
                response = await self.client.send_request(
                    "knowledge-curator-agent",
                    "quality-analysis",
                    {"content": content}
                )
                
                quality_analysis = response.get("quality_analysis", {})
                quality_score = quality_analysis.get("quality_score", 0)
                issues = quality_analysis.get("issues", [])
                recommendations = quality_analysis.get("recommendations", [])
                
                print(f"   Quality Score: {quality_score:.2f}")
                
                if issues:
                    print(f"   Issues Found: {len(issues)}")
                    for issue in issues[:3]:
                        print(f"   - {issue}")
                
                if recommendations:
                    print(f"   Recommendations: {len(recommendations)}")
                    for rec in recommendations[:2]:
                        print(f"   → {rec}")
                        
            except Exception as e:
                print(f"   Error: {e}")
    
    async def cache_performance_examples(self):
        """Demonstrate cache performance and optimization"""
        print("\n=== Cache Performance Examples ===")
        
        # Test similar queries to show cache hits
        similar_queries = [
            "machine learning algorithms",
            "machine learning techniques", 
            "ML algorithms and methods",
            "algorithms for machine learning"
        ]
        
        print("Testing cache performance with similar queries...")
        
        for i, query in enumerate(similar_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            
            try:
                # First check cache
                cache_response = await self.client.send_request(
                    "cache-manager-agent",
                    "cache-lookup",
                    {
                        "query": query,
                        "cache_type": "query_results",
                        "use_similarity": True
                    }
                )
                
                cached = cache_response.get("cached", False)
                cache_info = cache_response.get("cache_info", {})
                
                if cached:
                    similarity_match = cache_info.get("similarity_match", False)
                    lookup_time = cache_info.get("lookup_time", 0)
                    print(f"   ✓ Cache Hit! (Similarity: {similarity_match}, Time: {lookup_time:.3f}s)")
                    
                    if similarity_match:
                        original_query = cache_info.get("original_query", "")
                        similarity_score = cache_info.get("similarity_score", 0)
                        print(f"   Similar to: '{original_query}' (Score: {similarity_score:.3f})")
                else:
                    print(f"   ✗ Cache Miss - performing search...")
                    
                    # Perform actual search and cache result
                    search_response = await self.client.send_request(
                        "query-router-agent",
                        "route-query",
                        {"query": query}
                    )
                    
                    # Store in cache
                    await self.client.send_request(
                        "cache-manager-agent",
                        "cache-store",
                        {
                            "query": query,
                            "results": search_response.get("results", []),
                            "cache_type": "query_results"
                        }
                    )
                    
                    print(f"   Stored results in cache")
                    
            except Exception as e:
                print(f"   Error: {e}")
        
        # Show cache statistics
        try:
            print("\nCache Statistics:")
            cache_stats = await self.client.send_request(
                "cache-manager-agent",
                "cache-stats",
                {"detailed": True}
            )
            
            stats = cache_stats.get("stats", {}) if isinstance(cache_stats, dict) else cache_stats
            print(f"   Cache Hits: {stats.get('cache_hits', 0)}")
            print(f"   Cache Misses: {stats.get('cache_misses', 0)}")
            print(f"   Hit Rate: {stats.get('hit_rate_percent', 0):.1f}%")
            print(f"   Similarity Matches: {stats.get('similarity_matches', 0)}")
            
        except Exception as e:
            print(f"   Error getting cache stats: {e}")
    
    async def advanced_routing_examples(self):
        """Demonstrate advanced query routing capabilities"""
        print("\n=== Advanced Query Routing Examples ===")
        
        # Complex queries that require different routing strategies
        complex_queries = [
            {
                "query": "Find research papers about neural networks published after 2020 by Google researchers",
                "description": "Complex query requiring graph search + temporal filtering"
            },
            {
                "query": "What are the relationships between machine learning, deep learning, and artificial intelligence?",
                "description": "Conceptual relationship query for knowledge graph"
            },
            {
                "query": "Show me high-quality documents about Python programming with code examples",
                "description": "Quality-filtered search with content type filtering"
            },
            {
                "query": "artificial intelligence AND (ethics OR bias) NOT military",
                "description": "Boolean query requiring keyword processing"
            }
        ]
        
        for i, query_data in enumerate(complex_queries, 1):
            query = query_data["query"]
            description = query_data["description"]
            
            print(f"\n{i}. Complex Query: '{query}'")
            print(f"   Description: {description}")
            
            try:
                # First analyze the query
                analysis_response = await self.client.send_request(
                    "query-router-agent",
                    "analyze-query",
                    {"query": query}
                )
                
                analysis = analysis_response.get("analysis", {})
                intent = analysis.get("intent", {})
                complexity = analysis.get("complexity", {})
                
                print(f"   Intent Detection:")
                for intent_type, score in intent.items():
                    if score > 0.3:
                        print(f"   - {intent_type}: {score:.2f}")
                
                print(f"   Complexity: {complexity.get('complexity_score', 'unknown')}")
                print(f"   Entities Found: {len(analysis.get('entities', []))}")
                
                # Execute the routing
                routing_response = await self.client.send_request(
                    "query-router-agent",
                    "route-query",
                    {
                        "query": query,
                        "routing_options": {
                            "search_type": "hybrid"
                        }
                    }
                )
                
                routing_info = routing_response.get("routing_info", {})
                performance_stats = routing_response.get("performance_stats", {})
                results = routing_response.get("results", [])
                
                print(f"   Routing Strategy: {routing_info.get('routing_plan', {}).get('strategy', {})}")
                print(f"   Agents Used: {routing_info.get('source_agents', [])}")
                print(f"   Total Time: {performance_stats.get('total_time_seconds', 0):.2f}s")
                print(f"   Results Found: {len(results)}")
                
            except Exception as e:
                print(f"   Error: {e}")
    
    async def run_all_examples(self):
        """Run all example categories"""
        await self.connect()
        
        try:
            await self.semantic_search_examples()
            await self.hybrid_search_examples()
            await self.knowledge_graph_examples()
            await self.quality_analysis_examples()
            await self.cache_performance_examples()
            await self.advanced_routing_examples()
            
        finally:
            await self.disconnect()


async def main():
    """Main function to run all examples"""
    print("Knowledge Base System - Query Examples")
    print("=====================================")
    print("This script demonstrates various query patterns and capabilities")
    print("Make sure the Knowledge Base System is running before starting.")
    print()
    
    # Check if system is available
    test_client = AMPClient("connection-test")
    try:
        await test_client.connect("ws://localhost:8000/ws")
        await test_client.disconnect()
        print("✓ Knowledge Base System is accessible")
    except Exception as e:
        print(f"✗ Cannot connect to Knowledge Base System: {e}")
        print("Please start the system with: python run_knowledge_base.py")
        return
    
    # Run examples
    examples = KnowledgeBaseQueryExamples()
    await examples.run_all_examples()
    
    print("\n=== Examples Complete ===")
    print("For more advanced usage, check the README.md and API documentation.")


if __name__ == "__main__":
    asyncio.run(main())