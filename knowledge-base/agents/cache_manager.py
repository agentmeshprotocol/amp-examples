"""
Cache Manager Agent

This agent provides intelligent caching and retrieval optimization
for the knowledge base system, including query result caching,
embedding caching, and performance optimization.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple

import redis.asyncio as redis
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient
from amp_types import AMPMessage, MessageType

logger = logging.getLogger(__name__)


class CacheStrategy:
    """Defines caching strategies and policies"""
    
    def __init__(self):
        self.strategies = {
            'query_results': {
                'ttl': 3600,  # 1 hour
                'max_size': 10000,
                'eviction_policy': 'lru'
            },
            'embeddings': {
                'ttl': 86400,  # 24 hours
                'max_size': 50000,
                'eviction_policy': 'lfu'
            },
            'graph_results': {
                'ttl': 7200,  # 2 hours
                'max_size': 5000,
                'eviction_policy': 'lru'
            },
            'aggregated_results': {
                'ttl': 1800,  # 30 minutes
                'max_size': 2000,
                'eviction_policy': 'lru'
            }
        }
    
    def get_strategy(self, cache_type: str) -> Dict[str, Any]:
        """Get caching strategy for a specific type"""
        return self.strategies.get(cache_type, self.strategies['query_results'])
    
    def should_cache(self, cache_type: str, data_size: int, computation_time: float) -> bool:
        """Determine if data should be cached"""
        strategy = self.get_strategy(cache_type)
        
        # Don't cache very small results that are quick to compute
        if data_size < 100 and computation_time < 0.1:
            return False
        
        # Always cache expensive computations
        if computation_time > 1.0:
            return True
        
        # Cache based on size and type
        if cache_type == 'embeddings' and data_size > 10:
            return True
        
        if cache_type in ['query_results', 'graph_results'] and data_size > 0:
            return True
        
        return False


class QuerySimilarityMatcher:
    """Matches similar queries for cache lookup"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.query_embeddings = {}
        self.similarity_threshold = 0.85
    
    async def find_similar_query(self, query: str, cached_queries: List[str]) -> Optional[Tuple[str, float]]:
        """Find most similar cached query"""
        if not cached_queries:
            return None
        
        # Get or compute query embedding
        query_embedding = await self._get_query_embedding(query)
        
        # Get embeddings for cached queries
        cached_embeddings = []
        for cached_query in cached_queries:
            cached_embedding = await self._get_query_embedding(cached_query)
            cached_embeddings.append(cached_embedding)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], cached_embeddings)[0]
        
        # Find best match above threshold
        best_idx = np.argmax(similarities)
        best_similarity = similarities[best_idx]
        
        if best_similarity >= self.similarity_threshold:
            return cached_queries[best_idx], best_similarity
        
        return None
    
    async def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get or compute embedding for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        
        if query_hash not in self.query_embeddings:
            embedding = self.model.encode(query)
            self.query_embeddings[query_hash] = embedding
        
        return self.query_embeddings[query_hash]
    
    def clear_embeddings_cache(self):
        """Clear embeddings cache to free memory"""
        if len(self.query_embeddings) > 1000:
            # Keep only recent embeddings
            self.query_embeddings = {}


class RedisCache:
    """Redis-based cache implementation"""
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.connected = False
    
    async def connect(self):
        """Connect to Redis"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=False)
            await self.redis_client.ping()
            self.connected = True
            logger.info("Connected to Redis cache")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}. Using in-memory cache.")
            self.connected = False
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.connected:
            return None
        
        try:
            value = await self.redis_client.get(key)
            if value:
                return pickle.loads(value)
        except Exception as e:
            logger.error(f"Error getting from cache: {e}")
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        if not self.connected:
            return False
        
        try:
            serialized = pickle.dumps(value)
            await self.redis_client.setex(key, ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Error setting cache: {e}")
            return False
    
    async def delete(self, key: str):
        """Delete key from cache"""
        if not self.connected:
            return False
        
        try:
            await self.redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Error deleting from cache: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self.connected:
            return False
        
        try:
            return await self.redis_client.exists(key) > 0
        except Exception as e:
            logger.error(f"Error checking cache existence: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        if not self.connected:
            return []
        
        try:
            keys = await self.redis_client.keys(pattern)
            return [key.decode() if isinstance(key, bytes) else key for key in keys]
        except Exception as e:
            logger.error(f"Error getting keys: {e}")
            return []
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """Get cache memory usage statistics"""
        if not self.connected:
            return {}
        
        try:
            info = await self.redis_client.info('memory')
            return {
                'used_memory': info.get('used_memory', 0),
                'used_memory_human': info.get('used_memory_human', '0B'),
                'max_memory': info.get('maxmemory', 0),
                'memory_fragmentation_ratio': info.get('mem_fragmentation_ratio', 0)
            }
        except Exception as e:
            logger.error(f"Error getting memory usage: {e}")
            return {}


class InMemoryCache:
    """Fallback in-memory cache implementation"""
    
    def __init__(self):
        self.cache = {}
        self.access_times = {}
        self.access_counts = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if key in self.cache:
            entry = self.cache[key]
            
            # Check TTL
            if entry['expires_at'] > datetime.utcnow():
                # Update access statistics
                self.access_times[key] = datetime.utcnow()
                self.access_counts[key] = self.access_counts.get(key, 0) + 1
                return entry['value']
            else:
                # Expired, remove
                await self.delete(key)
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 3600):
        """Set value in cache"""
        expires_at = datetime.utcnow() + timedelta(seconds=ttl)
        
        self.cache[key] = {
            'value': value,
            'expires_at': expires_at,
            'created_at': datetime.utcnow()
        }
        
        self.access_times[key] = datetime.utcnow()
        self.access_counts[key] = 1
        
        return True
    
    async def delete(self, key: str):
        """Delete key from cache"""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.access_counts.pop(key, None)
        return True
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if key in self.cache:
            entry = self.cache[key]
            if entry['expires_at'] > datetime.utcnow():
                return True
            else:
                await self.delete(key)
        return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        import fnmatch
        
        # Clean expired entries first
        expired_keys = []
        for key, entry in self.cache.items():
            if entry['expires_at'] <= datetime.utcnow():
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
        
        # Return matching keys
        if pattern == "*":
            return list(self.cache.keys())
        else:
            return [key for key in self.cache.keys() if fnmatch.fnmatch(key, pattern)]
    
    async def cleanup_expired(self):
        """Clean up expired entries"""
        current_time = datetime.utcnow()
        expired_keys = []
        
        for key, entry in self.cache.items():
            if entry['expires_at'] <= current_time:
                expired_keys.append(key)
        
        for key in expired_keys:
            await self.delete(key)
        
        return len(expired_keys)


class CacheManagerAgent:
    """Main cache manager agent that provides intelligent caching"""
    
    def __init__(self, agent_id: str = "cache-manager-agent"):
        self.agent_id = agent_id
        self.client = AMPClient(agent_id)
        
        # Initialize components
        self.cache_strategy = CacheStrategy()
        self.similarity_matcher = QuerySimilarityMatcher()
        
        # Cache backends
        self.redis_cache = RedisCache()
        self.memory_cache = InMemoryCache()
        self.primary_cache = None
        
        # Statistics
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'cache_sets': 0,
            'cache_evictions': 0,
            'similarity_matches': 0,
            'start_time': datetime.utcnow()
        }
        
        logger.info(f"Cache Manager Agent {agent_id} initialized")
    
    async def start(self, host: str = "localhost", port: int = 8000):
        """Start the agent and register capabilities"""
        await self.client.connect(f"ws://{host}:{port}/ws")
        
        # Try to connect to Redis, fallback to in-memory
        await self.redis_cache.connect()
        self.primary_cache = self.redis_cache if self.redis_cache.connected else self.memory_cache
        
        # Register capabilities
        capabilities = [
            {
                "name": "cache-lookup",
                "description": "Look up cached results for queries",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "cache_type": {"type": "string", "default": "query_results"},
                        "use_similarity": {"type": "boolean", "default": True}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "cached": {"type": "boolean"},
                        "results": {"type": "array"},
                        "cache_info": {"type": "object"}
                    }
                }
            },
            {
                "name": "cache-store",
                "description": "Store results in cache",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "results": {"type": "array"},
                        "cache_type": {"type": "string", "default": "query_results"},
                        "metadata": {"type": "object"}
                    },
                    "required": ["query", "results"]
                }
            },
            {
                "name": "cache-invalidate",
                "description": "Invalidate cached results",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "pattern": {"type": "string"},
                        "cache_type": {"type": "string"}
                    }
                }
            },
            {
                "name": "cache-stats",
                "description": "Get cache statistics and performance metrics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "detailed": {"type": "boolean", "default": False}
                    }
                }
            },
            {
                "name": "optimize-cache",
                "description": "Optimize cache performance and cleanup",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["cleanup", "rebalance", "analyze"]}
                    }
                }
            }
        ]
        
        for capability in capabilities:
            await self.client.register_capability(capability)
        
        # Start periodic cleanup task
        asyncio.create_task(self._periodic_cleanup())
        
        # Start message handling
        await self.client.start_message_handler(self._handle_message)
        logger.info(f"Cache Manager Agent started on {host}:{port}")
    
    async def _handle_message(self, message: AMPMessage):
        """Handle incoming AMP messages"""
        try:
            capability = message.message.destination.capability
            payload = message.message.payload
            
            if capability == "cache-lookup":
                result = await self._handle_cache_lookup(payload)
            elif capability == "cache-store":
                result = await self._handle_cache_store(payload)
            elif capability == "cache-invalidate":
                result = await self._handle_cache_invalidate(payload)
            elif capability == "cache-stats":
                result = await self._handle_cache_stats(payload)
            elif capability == "optimize-cache":
                result = await self._handle_optimize_cache(payload)
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
                "CACHE_ERROR"
            )
    
    async def _handle_cache_lookup(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cache lookup request"""
        query = payload['query']
        cache_type = payload.get('cache_type', 'query_results')
        use_similarity = payload.get('use_similarity', True)
        
        start_time = time.time()
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, cache_type)
        
        # Direct cache lookup
        cached_result = await self.primary_cache.get(cache_key)
        
        if cached_result:
            self.stats['cache_hits'] += 1
            return {
                "cached": True,
                "results": cached_result['results'],
                "cache_info": {
                    "cache_hit": True,
                    "similarity_match": False,
                    "lookup_time": time.time() - start_time,
                    "cache_age": (datetime.utcnow() - cached_result['timestamp']).total_seconds()
                }
            }
        
        # Try similarity matching if enabled
        if use_similarity:
            similar_result = await self._find_similar_cached_query(query, cache_type)
            
            if similar_result:
                self.stats['cache_hits'] += 1
                self.stats['similarity_matches'] += 1
                return {
                    "cached": True,
                    "results": similar_result['results'],
                    "cache_info": {
                        "cache_hit": True,
                        "similarity_match": True,
                        "similarity_score": similar_result['similarity'],
                        "original_query": similar_result['original_query'],
                        "lookup_time": time.time() - start_time
                    }
                }
        
        # Cache miss
        self.stats['cache_misses'] += 1
        return {
            "cached": False,
            "results": [],
            "cache_info": {
                "cache_hit": False,
                "similarity_match": False,
                "lookup_time": time.time() - start_time
            }
        }
    
    async def _handle_cache_store(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cache store request"""
        query = payload['query']
        results = payload['results']
        cache_type = payload.get('cache_type', 'query_results')
        metadata = payload.get('metadata', {})
        
        # Check if we should cache this
        data_size = len(json.dumps(results))
        computation_time = metadata.get('computation_time', 0)
        
        if not self.cache_strategy.should_cache(cache_type, data_size, computation_time):
            return {
                "stored": False,
                "reason": "Does not meet caching criteria"
            }
        
        # Generate cache key
        cache_key = self._generate_cache_key(query, cache_type)
        
        # Prepare cache entry
        cache_entry = {
            'query': query,
            'results': results,
            'metadata': metadata,
            'timestamp': datetime.utcnow(),
            'cache_type': cache_type,
            'size': data_size
        }
        
        # Get TTL from strategy
        strategy = self.cache_strategy.get_strategy(cache_type)
        ttl = strategy['ttl']
        
        # Store in cache
        success = await self.primary_cache.set(cache_key, cache_entry, ttl)
        
        if success:
            self.stats['cache_sets'] += 1
            return {
                "stored": True,
                "cache_key": cache_key,
                "ttl": ttl,
                "size": data_size
            }
        else:
            return {
                "stored": False,
                "reason": "Cache storage failed"
            }
    
    async def _handle_cache_invalidate(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cache invalidation request"""
        pattern = payload.get('pattern', '*')
        cache_type = payload.get('cache_type')
        
        # Build search pattern
        if cache_type:
            search_pattern = f"{cache_type}:*"
        else:
            search_pattern = pattern
        
        # Get matching keys
        keys = await self.primary_cache.keys(search_pattern)
        
        # Delete keys
        deleted_count = 0
        for key in keys:
            if await self.primary_cache.delete(key):
                deleted_count += 1
        
        self.stats['cache_evictions'] += deleted_count
        
        return {
            "invalidated": True,
            "keys_deleted": deleted_count,
            "pattern": search_pattern
        }
    
    async def _handle_cache_stats(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cache statistics request"""
        detailed = payload.get('detailed', False)
        
        # Basic statistics
        total_requests = self.stats['cache_hits'] + self.stats['cache_misses']
        hit_rate = self.stats['cache_hits'] / max(total_requests, 1) * 100
        
        uptime = (datetime.utcnow() - self.stats['start_time']).total_seconds()
        
        stats = {
            "cache_hits": self.stats['cache_hits'],
            "cache_misses": self.stats['cache_misses'],
            "hit_rate_percent": round(hit_rate, 2),
            "cache_sets": self.stats['cache_sets'],
            "cache_evictions": self.stats['cache_evictions'],
            "similarity_matches": self.stats['similarity_matches'],
            "uptime_seconds": uptime,
            "cache_backend": "redis" if self.redis_cache.connected else "memory"
        }
        
        if detailed:
            # Get cache size information
            if isinstance(self.primary_cache, RedisCache):
                memory_info = await self.redis_cache.get_memory_usage()
                stats['memory_usage'] = memory_info
            
            # Get cache distribution by type
            all_keys = await self.primary_cache.keys("*")
            type_distribution = {}
            
            for key in all_keys:
                cache_type = key.split(':')[0] if ':' in key else 'unknown'
                type_distribution[cache_type] = type_distribution.get(cache_type, 0) + 1
            
            stats['cache_distribution'] = type_distribution
            stats['total_cached_items'] = len(all_keys)
        
        return stats
    
    async def _handle_optimize_cache(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle cache optimization request"""
        operation = payload.get('operation', 'cleanup')
        
        if operation == 'cleanup':
            return await self._cleanup_cache()
        elif operation == 'analyze':
            return await self._analyze_cache()
        elif operation == 'rebalance':
            return await self._rebalance_cache()
        else:
            raise ValueError(f"Unknown optimization operation: {operation}")
    
    async def _cleanup_cache(self) -> Dict[str, Any]:
        """Clean up expired and old cache entries"""
        start_time = time.time()
        
        if isinstance(self.primary_cache, InMemoryCache):
            deleted_count = await self.memory_cache.cleanup_expired()
        else:
            # For Redis, get all keys and check expiration manually if needed
            deleted_count = 0
            # Redis handles TTL automatically, but we can clean up based on other criteria
        
        # Clean up similarity matcher cache
        self.similarity_matcher.clear_embeddings_cache()
        
        cleanup_time = time.time() - start_time
        
        return {
            "operation": "cleanup",
            "deleted_items": deleted_count,
            "cleanup_time_seconds": cleanup_time,
            "success": True
        }
    
    async def _analyze_cache(self) -> Dict[str, Any]:
        """Analyze cache usage patterns"""
        all_keys = await self.primary_cache.keys("*")
        
        analysis = {
            "total_items": len(all_keys),
            "cache_types": {},
            "recommendations": []
        }
        
        # Analyze by cache type
        for key in all_keys:
            cache_type = key.split(':')[0] if ':' in key else 'unknown'
            if cache_type not in analysis["cache_types"]:
                analysis["cache_types"][cache_type] = {"count": 0, "keys": []}
            
            analysis["cache_types"][cache_type]["count"] += 1
            analysis["cache_types"][cache_type]["keys"].append(key)
        
        # Generate recommendations
        if analysis["total_items"] > 10000:
            analysis["recommendations"].append("Consider implementing cache size limits")
        
        if self.stats['cache_hits'] / max(self.stats['cache_hits'] + self.stats['cache_misses'], 1) < 0.5:
            analysis["recommendations"].append("Low hit rate - consider adjusting TTL or similarity threshold")
        
        return analysis
    
    async def _rebalance_cache(self) -> Dict[str, Any]:
        """Rebalance cache based on usage patterns"""
        # This could implement more sophisticated rebalancing
        # For now, just cleanup and provide recommendations
        cleanup_result = await self._cleanup_cache()
        
        return {
            "operation": "rebalance",
            "cleanup_result": cleanup_result,
            "recommendations": [
                "Cache has been cleaned up",
                "Consider adjusting TTL values based on usage patterns"
            ]
        }
    
    def _generate_cache_key(self, query: str, cache_type: str) -> str:
        """Generate cache key for query"""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{cache_type}:{query_hash}"
    
    async def _find_similar_cached_query(self, query: str, cache_type: str) -> Optional[Dict[str, Any]]:
        """Find similar cached query"""
        # Get all keys for this cache type
        pattern = f"{cache_type}:*"
        keys = await self.primary_cache.keys(pattern)
        
        if not keys:
            return None
        
        # Get cached queries
        cached_queries = []
        cached_entries = {}
        
        for key in keys:
            entry = await self.primary_cache.get(key)
            if entry and 'query' in entry:
                cached_queries.append(entry['query'])
                cached_entries[entry['query']] = entry
        
        # Find similar query
        similar_match = await self.similarity_matcher.find_similar_query(query, cached_queries)
        
        if similar_match:
            similar_query, similarity = similar_match
            cached_entry = cached_entries[similar_query]
            
            return {
                'results': cached_entry['results'],
                'similarity': similarity,
                'original_query': similar_query,
                'metadata': cached_entry.get('metadata', {})
            }
        
        return None
    
    async def _periodic_cleanup(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self._cleanup_cache()
            except Exception as e:
                logger.error(f"Error in periodic cleanup: {e}")
    
    async def stop(self):
        """Stop the agent"""
        await self.client.disconnect()
        logger.info("Cache Manager Agent stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache Manager Agent")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=8000, help="Port to connect to")
    parser.add_argument("--agent-id", default="cache-manager-agent", help="Agent ID")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = CacheManagerAgent(args.agent_id)
    agent.redis_cache.redis_url = args.redis_url
    
    try:
        asyncio.run(agent.start(args.host, args.port))
    except KeyboardInterrupt:
        asyncio.run(agent.stop())