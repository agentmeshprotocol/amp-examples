"""
Capability registry for AMP agents.
"""

import asyncio
import aiohttp
import time
from typing import Dict, Any, List, Optional
from .amp_types import AgentIdentity, Capability, AgentHealth


class CapabilityRegistry:
    """Registry for discovering agents and capabilities."""
    
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint
        self.local_agents: Dict[str, Dict[str, Any]] = {}
        self.capabilities_cache: Dict[str, List[Dict[str, Any]]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def connect(self):
        """Connect to registry service."""
        if self.endpoint:
            self.session = aiohttp.ClientSession()
    
    async def disconnect(self):
        """Disconnect from registry service."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def register_agent(self, identity: AgentIdentity, 
                           capabilities: List[Dict[str, Any]]) -> bool:
        """Register an agent with the registry."""
        agent_data = {
            "identity": {
                "id": identity.id,
                "name": identity.name,
                "version": identity.version,
                "framework": identity.framework,
                "protocol_version": identity.protocol_version,
                "description": identity.description,
                "tags": identity.tags
            },
            "capabilities": [
                {
                    "id": cap["capability"].id,
                    "version": cap["capability"].version,
                    "description": cap["capability"].description,
                    "category": cap["capability"].category,
                    "input_schema": cap["capability"].input_schema,
                    "output_schema": cap["capability"].output_schema,
                    "constraints": {
                        "max_input_length": cap["capability"].constraints.max_input_length,
                        "supported_languages": cap["capability"].constraints.supported_languages,
                        "response_time_ms": cap["capability"].constraints.response_time_ms,
                        "max_tokens": cap["capability"].constraints.max_tokens
                    }
                }
                for cap in capabilities
            ],
            "endpoints": {
                "health": f"http://localhost:8000/agents/{identity.id}/health",
                "invoke": f"http://localhost:8000/agents/{identity.id}/invoke"
            },
            "registered_at": time.time()
        }
        
        # Store locally
        self.local_agents[identity.id] = agent_data
        
        # Register with remote registry if available
        if self.endpoint and self.session:
            try:
                async with self.session.post(
                    f"{self.endpoint}/agents",
                    json=agent_data
                ) as response:
                    return response.status == 201
            except Exception:
                pass  # Fall back to local only
        
        return True
    
    async def deregister_agent(self, agent_id: str) -> bool:
        """Deregister an agent."""
        # Remove from local registry
        self.local_agents.pop(agent_id, None)
        
        # Remove from remote registry if available
        if self.endpoint and self.session:
            try:
                async with self.session.delete(
                    f"{self.endpoint}/agents/{agent_id}"
                ) as response:
                    return response.status == 200
            except Exception:
                pass
        
        return True
    
    async def discover_agents(self, capability: Optional[str] = None,
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Discover agents by capability or filters."""
        cache_key = f"{capability}:{hash(str(filters)) if filters else 'none'}"
        
        # Check cache first
        if cache_key in self.capabilities_cache:
            cache_entry = self.capabilities_cache[cache_key]
            if time.time() - cache_entry["timestamp"] < self.cache_ttl:
                return cache_entry["data"]
        
        results = []
        
        # Search local agents
        for agent_id, agent_data in self.local_agents.items():
            if self._matches_criteria(agent_data, capability, filters):
                results.append(agent_data)
        
        # Search remote registry if available
        if self.endpoint and self.session:
            try:
                params = {}
                if capability:
                    params["capability"] = capability
                if filters:
                    params.update(filters)
                
                async with self.session.get(
                    f"{self.endpoint}/agents/search",
                    params=params
                ) as response:
                    if response.status == 200:
                        remote_results = await response.json()
                        # Merge with local results, avoiding duplicates
                        existing_ids = {agent["identity"]["id"] for agent in results}
                        for agent in remote_results.get("agents", []):
                            if agent["identity"]["id"] not in existing_ids:
                                results.append(agent)
            except Exception:
                pass  # Fall back to local only
        
        # Cache results
        self.capabilities_cache[cache_key] = {
            "data": results,
            "timestamp": time.time()
        }
        
        return results
    
    async def get_agent_info(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed agent information."""
        # Check local first
        if agent_id in self.local_agents:
            return self.local_agents[agent_id]
        
        # Check remote registry
        if self.endpoint and self.session:
            try:
                async with self.session.get(
                    f"{self.endpoint}/agents/{agent_id}"
                ) as response:
                    if response.status == 200:
                        return await response.json()
            except Exception:
                pass
        
        return None
    
    async def get_capability_providers(self, capability_id: str) -> List[Dict[str, Any]]:
        """Get all agents that provide a specific capability."""
        return await self.discover_agents(capability=capability_id)
    
    async def update_agent_health(self, agent_id: str, health: AgentHealth):
        """Update agent health status."""
        if agent_id in self.local_agents:
            self.local_agents[agent_id]["health"] = {
                "status": health.status,
                "last_seen": health.last_seen.isoformat(),
                "response_time_avg": health.response_time_avg,
                "success_rate": health.success_rate,
                "error_count": health.error_count,
                "capabilities_available": health.capabilities_available,
                "load_level": health.load_level,
                "metadata": health.metadata
            }
        
        # Update remote registry if available
        if self.endpoint and self.session:
            try:
                health_data = {
                    "status": health.status,
                    "last_seen": health.last_seen.isoformat(),
                    "response_time_avg": health.response_time_avg,
                    "success_rate": health.success_rate,
                    "error_count": health.error_count,
                    "capabilities_available": health.capabilities_available,
                    "load_level": health.load_level,
                    "metadata": health.metadata
                }
                
                async with self.session.put(
                    f"{self.endpoint}/agents/{agent_id}/health",
                    json=health_data
                ) as response:
                    pass  # Don't fail on health update errors
            except Exception:
                pass
    
    def _matches_criteria(self, agent_data: Dict[str, Any], 
                         capability: Optional[str],
                         filters: Optional[Dict[str, Any]]) -> bool:
        """Check if agent matches search criteria."""
        # Check capability
        if capability:
            agent_capabilities = [cap["id"] for cap in agent_data.get("capabilities", [])]
            if capability not in agent_capabilities:
                return False
        
        # Check filters
        if filters:
            for key, value in filters.items():
                if key == "framework":
                    if agent_data.get("identity", {}).get("framework") != value:
                        return False
                elif key == "tags":
                    agent_tags = agent_data.get("identity", {}).get("tags", [])
                    if not any(tag in agent_tags for tag in value):
                        return False
                elif key == "min_success_rate":
                    health = agent_data.get("health", {})
                    if health.get("success_rate", 0) < value:
                        return False
                elif key == "max_latency_ms":
                    health = agent_data.get("health", {})
                    if health.get("response_time_avg", 0) * 1000 > value:
                        return False
        
        return True
    
    def clear_cache(self):
        """Clear the capabilities cache."""
        self.capabilities_cache.clear()
    
    def get_local_agents(self) -> Dict[str, Dict[str, Any]]:
        """Get all locally registered agents."""
        return self.local_agents.copy()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "cache_entries": len(self.capabilities_cache),
            "local_agents": len(self.local_agents),
            "cache_ttl": self.cache_ttl
        }


class InMemoryRegistry(CapabilityRegistry):
    """In-memory implementation for testing and development."""
    
    def __init__(self):
        super().__init__(endpoint=None)
        self.global_agents: Dict[str, Dict[str, Any]] = {}
    
    async def register_agent(self, identity: AgentIdentity, 
                           capabilities: List[Dict[str, Any]]) -> bool:
        """Register agent in memory."""
        agent_data = {
            "identity": {
                "id": identity.id,
                "name": identity.name,
                "version": identity.version,
                "framework": identity.framework,
                "protocol_version": identity.protocol_version,
                "description": identity.description,
                "tags": identity.tags
            },
            "capabilities": [
                {
                    "id": cap["capability"].id,
                    "version": cap["capability"].version,
                    "description": cap["capability"].description,
                    "category": cap["capability"].category
                }
                for cap in capabilities
            ],
            "registered_at": time.time()
        }
        
        self.global_agents[identity.id] = agent_data
        self.local_agents[identity.id] = agent_data
        return True
    
    async def discover_agents(self, capability: Optional[str] = None,
                            filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Discover agents from memory."""
        results = []
        
        for agent_id, agent_data in self.global_agents.items():
            if self._matches_criteria(agent_data, capability, filters):
                results.append(agent_data)
        
        return results
