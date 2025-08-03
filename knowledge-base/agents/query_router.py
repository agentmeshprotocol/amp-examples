"""
Query Router Agent

This agent intelligently routes queries to appropriate knowledge agents
based on query analysis, intent detection, and optimal routing strategies.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient
from amp_types import AMPMessage, MessageType

logger = logging.getLogger(__name__)


class QueryAnalyzer:
    """Analyzes queries to determine intent and optimal routing"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        
        # Query intent patterns
        self.intent_patterns = {
            'semantic_search': [
                r'find.*similar.*to',
                r'what.*about',
                r'tell me about',
                r'documents.*related.*to',
                r'search.*for',
                r'information.*on'
            ],
            'graph_search': [
                r'how.*connected',
                r'relationship.*between',
                r'related.*entities',
                r'entities.*related.*to',
                r'path.*between',
                r'connections.*of'
            ],
            'entity_search': [
                r'who.*is',
                r'what.*is',
                r'entities.*named',
                r'people.*involved',
                r'organizations.*mentioned',
                r'locations.*in'
            ],
            'analytical': [
                r'analyze',
                r'statistics.*about',
                r'trends.*in',
                r'summary.*of',
                r'insights.*from',
                r'patterns.*in'
            ]
        }
        
        # Agent capability mapping
        self.agent_capabilities = {
            'semantic-search-agent': {
                'primary': ['semantic_search', 'similarity_search'],
                'secondary': ['hybrid_search'],
                'confidence_threshold': 0.7
            },
            'knowledge-graph-agent': {
                'primary': ['graph_search', 'entity_search', 'relationship_analysis'],
                'secondary': ['entity_extraction'],
                'confidence_threshold': 0.6
            },
            'cache-manager-agent': {
                'primary': ['cached_results'],
                'secondary': ['performance_optimization'],
                'confidence_threshold': 0.9
            },
            'knowledge-curator-agent': {
                'primary': ['analytical', 'quality_assessment'],
                'secondary': ['content_validation'],
                'confidence_threshold': 0.5
            }
        }
    
    async def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine intent and routing strategy"""
        analysis = {
            'query': query,
            'intent': await self._detect_intent(query),
            'entities': await self._extract_query_entities(query),
            'complexity': await self._assess_complexity(query),
            'keywords': await self._extract_keywords(query),
            'embedding': self.model.encode(query).tolist(),
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return analysis
    
    async def _detect_intent(self, query: str) -> Dict[str, float]:
        """Detect query intent using pattern matching and similarity"""
        intent_scores = {}
        query_lower = query.lower()
        
        # Pattern-based detection
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score = max(score, 0.8)
            intent_scores[intent] = score
        
        # If no strong pattern match, use keyword-based scoring
        if max(intent_scores.values()) < 0.5:
            # Semantic search keywords
            semantic_keywords = ['find', 'search', 'similar', 'related', 'about', 'information']
            if any(keyword in query_lower for keyword in semantic_keywords):
                intent_scores['semantic_search'] = max(intent_scores.get('semantic_search', 0), 0.6)
            
            # Graph search keywords
            graph_keywords = ['relationship', 'connected', 'entities', 'path', 'connection']
            if any(keyword in query_lower for keyword in graph_keywords):
                intent_scores['graph_search'] = max(intent_scores.get('graph_search', 0), 0.6)
        
        return intent_scores
    
    async def _extract_query_entities(self, query: str) -> List[str]:
        """Extract potential entities from query"""
        # Simple entity extraction (could be enhanced with NLP)
        import re
        
        # Extract quoted entities
        quoted_entities = re.findall(r'"([^"]*)"', query)
        
        # Extract capitalized words (potential proper nouns)
        capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', query)
        
        entities = quoted_entities + capitalized_words
        return list(set(entities))
    
    async def _assess_complexity(self, query: str) -> Dict[str, Any]:
        """Assess query complexity"""
        words = query.split()
        
        complexity = {
            'word_count': len(words),
            'question_words': len([w for w in words if w.lower() in ['what', 'who', 'where', 'when', 'why', 'how']]),
            'conjunction_count': len([w for w in words if w.lower() in ['and', 'or', 'but', 'however']]),
            'complexity_score': 'simple'
        }
        
        # Determine complexity score
        if complexity['word_count'] > 20 or complexity['conjunction_count'] > 2:
            complexity['complexity_score'] = 'complex'
        elif complexity['word_count'] > 10 or complexity['question_words'] > 1:
            complexity['complexity_score'] = 'medium'
        
        return complexity
    
    async def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))


class RouteOptimizer:
    """Optimizes query routing based on agent capabilities and performance"""
    
    def __init__(self):
        self.agent_performance = {}
        self.query_history = []
        self.route_cache = {}
    
    async def determine_routing(self, query_analysis: Dict[str, Any], 
                              available_agents: List[str]) -> List[Dict[str, Any]]:
        """Determine optimal routing for a query"""
        intent_scores = query_analysis['intent']
        complexity = query_analysis['complexity']
        
        # Calculate agent scores
        agent_scores = []
        for agent_id in available_agents:
            score = await self._calculate_agent_score(agent_id, intent_scores, complexity)
            if score > 0:
                agent_scores.append({
                    'agent_id': agent_id,
                    'score': score,
                    'confidence': min(score, 1.0),
                    'routing_reason': await self._get_routing_reason(agent_id, intent_scores)
                })
        
        # Sort by score
        agent_scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Determine routing strategy
        routing_strategy = await self._determine_strategy(query_analysis, agent_scores)
        
        return {
            'primary_routes': agent_scores[:2],  # Top 2 agents
            'fallback_routes': agent_scores[2:4],  # Next 2 as fallbacks
            'strategy': routing_strategy,
            'should_parallel': routing_strategy['parallel_execution'],
            'should_aggregate': routing_strategy['aggregate_results']
        }
    
    async def _calculate_agent_score(self, agent_id: str, intent_scores: Dict[str, float], 
                                   complexity: Dict[str, Any]) -> float:
        """Calculate routing score for an agent"""
        from .query_router import QueryAnalyzer
        analyzer = QueryAnalyzer()
        
        agent_config = analyzer.agent_capabilities.get(agent_id, {})
        primary_capabilities = agent_config.get('primary', [])
        secondary_capabilities = agent_config.get('secondary', [])
        threshold = agent_config.get('confidence_threshold', 0.5)
        
        score = 0.0
        
        # Primary capability match
        for capability in primary_capabilities:
            if capability in intent_scores:
                score += intent_scores[capability] * 1.0
        
        # Secondary capability match
        for capability in secondary_capabilities:
            if capability in intent_scores:
                score += intent_scores[capability] * 0.5
        
        # Performance adjustment
        performance = self.agent_performance.get(agent_id, {})
        avg_response_time = performance.get('avg_response_time', 1.0)
        success_rate = performance.get('success_rate', 1.0)
        
        # Adjust score based on performance
        score = score * success_rate * (2.0 / (1.0 + avg_response_time))
        
        # Complexity adjustment
        complexity_score = complexity.get('complexity_score', 'simple')
        if complexity_score == 'complex' and agent_id in ['semantic-search-agent', 'knowledge-graph-agent']:
            score *= 1.2  # Boost for complex queries
        
        return score if score >= threshold else 0.0
    
    async def _get_routing_reason(self, agent_id: str, intent_scores: Dict[str, float]) -> str:
        """Get human-readable routing reason"""
        top_intent = max(intent_scores, key=intent_scores.get) if intent_scores else 'unknown'
        
        reasons = {
            'semantic-search-agent': f"Best for {top_intent} with vector similarity search",
            'knowledge-graph-agent': f"Optimal for {top_intent} using entity relationships",
            'cache-manager-agent': f"Provides cached results for {top_intent}",
            'knowledge-curator-agent': f"Offers analytical insights for {top_intent}"
        }
        
        return reasons.get(agent_id, f"Suitable for {top_intent}")
    
    async def _determine_strategy(self, query_analysis: Dict[str, Any], 
                                agent_scores: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine execution strategy"""
        complexity = query_analysis['complexity']['complexity_score']
        top_agents = agent_scores[:2]
        
        strategy = {
            'parallel_execution': False,
            'aggregate_results': False,
            'timeout': 30.0,
            'fallback_enabled': True
        }
        
        # Enable parallel execution for complex queries with multiple viable agents
        if complexity in ['medium', 'complex'] and len(top_agents) >= 2:
            score_diff = top_agents[0]['score'] - top_agents[1]['score']
            if score_diff < 0.3:  # Scores are close
                strategy['parallel_execution'] = True
                strategy['aggregate_results'] = True
        
        # Adjust timeout based on complexity
        if complexity == 'complex':
            strategy['timeout'] = 60.0
        elif complexity == 'medium':
            strategy['timeout'] = 45.0
        
        return strategy
    
    async def update_performance(self, agent_id: str, response_time: float, 
                               success: bool, result_quality: float = None):
        """Update agent performance metrics"""
        if agent_id not in self.agent_performance:
            self.agent_performance[agent_id] = {
                'total_requests': 0,
                'successful_requests': 0,
                'total_response_time': 0.0,
                'avg_response_time': 0.0,
                'success_rate': 0.0,
                'quality_scores': []
            }
        
        perf = self.agent_performance[agent_id]
        perf['total_requests'] += 1
        perf['total_response_time'] += response_time
        
        if success:
            perf['successful_requests'] += 1
        
        if result_quality is not None:
            perf['quality_scores'].append(result_quality)
            # Keep only last 100 quality scores
            if len(perf['quality_scores']) > 100:
                perf['quality_scores'] = perf['quality_scores'][-100:]
        
        # Update averages
        perf['avg_response_time'] = perf['total_response_time'] / perf['total_requests']
        perf['success_rate'] = perf['successful_requests'] / perf['total_requests']


class ResultAggregator:
    """Aggregates results from multiple agents"""
    
    async def aggregate_results(self, results: List[Dict[str, Any]], 
                              query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate results from multiple agents"""
        if not results:
            return {'aggregated_results': [], 'confidence': 0.0}
        
        if len(results) == 1:
            return {
                'aggregated_results': results[0].get('results', []),
                'confidence': results[0].get('confidence', 1.0),
                'source_agents': [results[0].get('agent_id', 'unknown')]
            }
        
        # Multi-agent aggregation
        aggregated = await self._merge_results(results, query_analysis)
        
        return {
            'aggregated_results': aggregated['results'],
            'confidence': aggregated['confidence'],
            'source_agents': aggregated['source_agents'],
            'aggregation_strategy': aggregated['strategy']
        }
    
    async def _merge_results(self, results: List[Dict[str, Any]], 
                           query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from multiple agents"""
        all_results = []
        source_agents = []
        
        for result in results:
            agent_results = result.get('results', [])
            agent_id = result.get('agent_id', 'unknown')
            
            # Add source information to each result
            for item in agent_results:
                item['source_agent'] = agent_id
                all_results.append(item)
            
            source_agents.append(agent_id)
        
        # Remove duplicates based on content similarity
        deduplicated = await self._deduplicate_results(all_results)
        
        # Score and rank results
        ranked_results = await self._rank_results(deduplicated, query_analysis)
        
        # Calculate confidence
        confidence = await self._calculate_aggregate_confidence(results)
        
        return {
            'results': ranked_results,
            'confidence': confidence,
            'source_agents': source_agents,
            'strategy': 'similarity_based_merge'
        }
    
    async def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity"""
        if len(results) <= 1:
            return results
        
        # Extract text content for similarity comparison
        texts = []
        for result in results:
            text = result.get('text', '') or result.get('content', '') or str(result.get('id', ''))
            texts.append(text)
        
        # Calculate similarity matrix
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        similarity_matrix = cosine_similarity(embeddings)
        
        # Group similar results
        threshold = 0.8
        groups = []
        assigned = set()
        
        for i in range(len(results)):
            if i in assigned:
                continue
            
            group = [i]
            assigned.add(i)
            
            for j in range(i + 1, len(results)):
                if j not in assigned and similarity_matrix[i][j] > threshold:
                    group.append(j)
                    assigned.add(j)
            
            groups.append(group)
        
        # Select best result from each group
        deduplicated = []
        for group in groups:
            # Select result with highest score or first one
            best_idx = group[0]
            best_score = results[best_idx].get('score', 0) or results[best_idx].get('similarity', 0)
            
            for idx in group[1:]:
                score = results[idx].get('score', 0) or results[idx].get('similarity', 0)
                if score > best_score:
                    best_idx = idx
                    best_score = score
            
            result = results[best_idx].copy()
            result['duplicate_count'] = len(group)
            result['source_agents'] = list(set([results[idx].get('source_agent', 'unknown') for idx in group]))
            deduplicated.append(result)
        
        return deduplicated
    
    async def _rank_results(self, results: List[Dict[str, Any]], 
                          query_analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Rank results based on relevance and quality"""
        if not results:
            return results
        
        # Calculate ranking scores
        for result in results:
            score = 0.0
            
            # Base score from similarity/relevance
            base_score = result.get('score', 0) or result.get('similarity', 0)
            score += base_score * 0.6
            
            # Boost for multiple source confirmation
            source_agents = result.get('source_agents', [result.get('source_agent', '')])
            if isinstance(source_agents, list) and len(source_agents) > 1:
                score += 0.2
            
            # Content quality indicators
            text_length = len(result.get('text', ''))
            if text_length > 100:  # Substantial content
                score += 0.1
            
            # Metadata richness
            metadata = result.get('metadata', {})
            if metadata and len(metadata) > 3:
                score += 0.1
            
            result['ranking_score'] = score
        
        # Sort by ranking score
        results.sort(key=lambda x: x.get('ranking_score', 0), reverse=True)
        
        return results
    
    async def _calculate_aggregate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence for aggregated results"""
        if not results:
            return 0.0
        
        confidences = []
        for result in results:
            confidence = result.get('confidence', 1.0)
            confidences.append(confidence)
        
        # Use weighted average with bias toward higher confidences
        if len(confidences) == 1:
            return confidences[0]
        
        # Boost confidence when multiple agents agree
        avg_confidence = sum(confidences) / len(confidences)
        agreement_boost = min(0.2, 0.1 * (len(confidences) - 1))
        
        return min(1.0, avg_confidence + agreement_boost)


class QueryRouterAgent:
    """Main query router agent that orchestrates query routing and result aggregation"""
    
    def __init__(self, agent_id: str = "query-router-agent"):
        self.agent_id = agent_id
        self.client = AMPClient(agent_id)
        
        # Initialize components
        self.query_analyzer = QueryAnalyzer()
        self.route_optimizer = RouteOptimizer()
        self.result_aggregator = ResultAggregator()
        
        # Available agents
        self.available_agents = [
            'semantic-search-agent',
            'knowledge-graph-agent',
            'cache-manager-agent',
            'knowledge-curator-agent'
        ]
        
        logger.info(f"Query Router Agent {agent_id} initialized")
    
    async def start(self, host: str = "localhost", port: int = 8000):
        """Start the agent and register capabilities"""
        await self.client.connect(f"ws://{host}:{port}/ws")
        
        # Register capabilities
        capabilities = [
            {
                "name": "route-query",
                "description": "Intelligently route queries to appropriate knowledge agents",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "routing_options": {"type": "object"},
                        "preferred_agents": {"type": "array"}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "routing_info": {"type": "object"},
                        "performance_stats": {"type": "object"}
                    }
                }
            },
            {
                "name": "analyze-query",
                "description": "Analyze query intent and complexity",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "get-routing-stats",
                "description": "Get routing statistics and agent performance",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "agent_id": {"type": "string"}
                    }
                }
            }
        ]
        
        for capability in capabilities:
            await self.client.register_capability(capability)
        
        # Start message handling
        await self.client.start_message_handler(self._handle_message)
        logger.info(f"Query Router Agent started on {host}:{port}")
    
    async def _handle_message(self, message: AMPMessage):
        """Handle incoming AMP messages"""
        try:
            capability = message.message.destination.capability
            payload = message.message.payload
            
            if capability == "route-query":
                result = await self._handle_route_query(payload)
            elif capability == "analyze-query":
                result = await self._handle_analyze_query(payload)
            elif capability == "get-routing-stats":
                result = await self._handle_get_routing_stats(payload)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            # Send response
            await self.client.send_response(
                message.message.id,
                message.message.source.agent_id,
                result
            )
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.client.send_error(
                message.message.id,
                message.message.source.agent_id,
                str(e),
                "ROUTING_ERROR"
            )
    
    async def _handle_route_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query routing request"""
        query = payload['query']
        routing_options = payload.get('routing_options', {})
        preferred_agents = payload.get('preferred_agents', self.available_agents)
        
        start_time = datetime.utcnow()
        
        # Analyze query
        query_analysis = await self.query_analyzer.analyze_query(query)
        
        # Determine routing
        routing_plan = await self.route_optimizer.determine_routing(
            query_analysis, 
            preferred_agents
        )
        
        # Execute routing
        results = await self._execute_routing(query, routing_plan, routing_options)
        
        # Aggregate results if needed
        if routing_plan['should_aggregate'] and len(results) > 1:
            aggregated = await self.result_aggregator.aggregate_results(results, query_analysis)
            final_results = aggregated['aggregated_results']
            confidence = aggregated['confidence']
            source_agents = aggregated['source_agents']
        else:
            final_results = results[0].get('results', []) if results else []
            confidence = results[0].get('confidence', 0.0) if results else 0.0
            source_agents = [results[0].get('agent_id')] if results else []
        
        end_time = datetime.utcnow()
        total_time = (end_time - start_time).total_seconds()
        
        return {
            "results": final_results,
            "routing_info": {
                "query_analysis": query_analysis,
                "routing_plan": routing_plan,
                "executed_routes": len(results),
                "source_agents": source_agents,
                "confidence": confidence
            },
            "performance_stats": {
                "total_time_seconds": total_time,
                "agent_response_times": [r.get('response_time', 0) for r in results],
                "success_rate": len([r for r in results if r.get('success', True)]) / max(len(results), 1)
            }
        }
    
    async def _handle_analyze_query(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle query analysis request"""
        query = payload['query']
        analysis = await self.query_analyzer.analyze_query(query)
        
        return {
            "query": query,
            "analysis": analysis
        }
    
    async def _handle_get_routing_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle routing statistics request"""
        agent_id = payload.get('agent_id')
        
        if agent_id:
            stats = self.route_optimizer.agent_performance.get(agent_id, {})
            return {
                "agent_id": agent_id,
                "performance_stats": stats
            }
        else:
            return {
                "all_agents_stats": self.route_optimizer.agent_performance
            }
    
    async def _execute_routing(self, query: str, routing_plan: Dict[str, Any], 
                             options: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute the routing plan"""
        primary_routes = routing_plan['primary_routes']
        parallel_execution = routing_plan['should_parallel']
        timeout = routing_plan['strategy']['timeout']
        
        results = []
        
        if parallel_execution and len(primary_routes) > 1:
            # Execute in parallel
            tasks = []
            for route in primary_routes:
                task = self._query_agent(
                    route['agent_id'], 
                    query, 
                    options, 
                    timeout
                )
                tasks.append(task)
            
            # Wait for all results
            parallel_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(parallel_results):
                if isinstance(result, Exception):
                    logger.error(f"Error from agent {primary_routes[i]['agent_id']}: {result}")
                    # Update performance with failure
                    await self.route_optimizer.update_performance(
                        primary_routes[i]['agent_id'], 
                        timeout, 
                        False
                    )
                else:
                    results.append(result)
        
        else:
            # Execute sequentially with fallback
            for route in primary_routes:
                try:
                    result = await self._query_agent(
                        route['agent_id'], 
                        query, 
                        options, 
                        timeout
                    )
                    results.append(result)
                    
                    # If we get a good result, we might stop here
                    if result.get('success', True) and len(result.get('results', [])) > 0:
                        break
                        
                except Exception as e:
                    logger.error(f"Error from agent {route['agent_id']}: {e}")
                    # Update performance with failure
                    await self.route_optimizer.update_performance(
                        route['agent_id'], 
                        timeout, 
                        False
                    )
                    continue
        
        return results
    
    async def _query_agent(self, agent_id: str, query: str, options: Dict[str, Any], 
                          timeout: float) -> Dict[str, Any]:
        """Query a specific agent"""
        start_time = datetime.utcnow()
        
        try:
            # Determine capability based on agent
            capability_map = {
                'semantic-search-agent': 'semantic-search',
                'knowledge-graph-agent': 'graph-search',
                'cache-manager-agent': 'cache-lookup',
                'knowledge-curator-agent': 'quality-analysis'
            }
            
            capability = capability_map.get(agent_id, 'search')
            
            # Prepare payload
            payload = {'query': query}
            payload.update(options.get(agent_id, {}))
            
            # Send request
            response = await self.client.send_request(
                agent_id,
                capability,
                payload,
                timeout=timeout
            )
            
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            # Update performance
            success = response.get('success', True)
            await self.route_optimizer.update_performance(
                agent_id, 
                response_time, 
                success
            )
            
            return {
                'agent_id': agent_id,
                'results': response.get('results', []),
                'confidence': response.get('confidence', 1.0),
                'response_time': response_time,
                'success': success
            }
            
        except Exception as e:
            end_time = datetime.utcnow()
            response_time = (end_time - start_time).total_seconds()
            
            # Update performance with failure
            await self.route_optimizer.update_performance(
                agent_id, 
                response_time, 
                False
            )
            
            raise e
    
    async def stop(self):
        """Stop the agent"""
        await self.client.disconnect()
        logger.info("Query Router Agent stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Query Router Agent")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=8000, help="Port to connect to")
    parser.add_argument("--agent-id", default="query-router-agent", help="Agent ID")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = QueryRouterAgent(args.agent_id)
    
    try:
        asyncio.run(agent.start(args.host, args.port))
    except KeyboardInterrupt:
        asyncio.run(agent.stop())