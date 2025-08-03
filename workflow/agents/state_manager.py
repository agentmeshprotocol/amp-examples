"""
State Manager Agent - Manages workflow state, context, and data persistence.
"""

import asyncio
import json
import sqlite3
import redis
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timezone
import logging
from pathlib import Path
import aiosqlite

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_types import AMPMessage, MessageType, CapabilityStatus, Capability, AgentIdentity
from amp_client import AMPClient

from ..workflow_types import WorkflowEventTypes, WorkflowErrorCodes


class StateManager:
    """
    Centralized state manager for workflow orchestration.
    Handles workflow state persistence, context management, and data sharing.
    """
    
    def __init__(self, agent_id: str = "state-manager", port: int = 8082, 
                 use_redis: bool = True, redis_url: str = "redis://localhost:6379"):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(f"StateManager.{agent_id}")
        
        # Storage configuration
        self.use_redis = use_redis
        self.redis_url = redis_url
        self.db_path = "workflow_state.db"
        
        # State storage
        self.amp_client = None
        self.redis_client = None
        self.sqlite_pool = None
        
        # In-memory cache for performance
        self.state_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_ttl = 300  # 5 minutes
        self.cache_timestamps: Dict[str, datetime] = {}
        
        # State versioning
        self.state_versions: Dict[str, int] = {}
        
    async def start(self):
        """Start the state manager agent."""
        self.logger.info(f"Starting State Manager {self.agent_id}")
        
        # Initialize storage backends
        await self._initialize_storage()
        
        # Initialize AMP client
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id=self.agent_id,
                name="State Manager",
                version="1.0.0",
                framework="AMP-Workflow",
                description="Centralized workflow state and context manager"
            ),
            port=self.port
        )
        
        # Register capabilities
        await self._register_capabilities()
        
        # Start AMP client
        await self.amp_client.start()
        
        # Register message handlers
        self.amp_client.register_capability_handler("state-get", self._handle_get_state)
        self.amp_client.register_capability_handler("state-set", self._handle_set_state)
        self.amp_client.register_capability_handler("state-update", self._handle_update_state)
        self.amp_client.register_capability_handler("state-delete", self._handle_delete_state)
        self.amp_client.register_capability_handler("state-list", self._handle_list_states)
        self.amp_client.register_capability_handler("state-query", self._handle_query_state)
        self.amp_client.register_capability_handler("state-backup", self._handle_backup_state)
        self.amp_client.register_capability_handler("state-restore", self._handle_restore_state)
        self.amp_client.register_capability_handler("context-merge", self._handle_merge_context)
        self.amp_client.register_capability_handler("context-isolate", self._handle_isolate_context)
        
        # Start cache cleanup task
        asyncio.create_task(self._cache_cleanup_task())
        
        self.logger.info("State Manager started successfully")
    
    async def stop(self):
        """Stop the state manager agent."""
        self.logger.info("Stopping State Manager")
        
        # Close storage connections
        if self.redis_client:
            await self.redis_client.close()
        
        if self.amp_client:
            await self.amp_client.stop()
    
    async def _initialize_storage(self):
        """Initialize storage backends."""
        # Initialize SQLite database
        await self._init_sqlite()
        
        # Initialize Redis if configured
        if self.use_redis:
            await self._init_redis()
    
    async def _init_sqlite(self):
        """Initialize SQLite database for persistent storage."""
        # Create database schema
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS workflow_states (
                    workflow_instance_id TEXT PRIMARY KEY,
                    state_data TEXT NOT NULL,
                    version INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS state_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_instance_id TEXT NOT NULL,
                    state_data TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    change_description TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (workflow_instance_id) REFERENCES workflow_states (workflow_instance_id)
                )
            """)
            
            await db.execute("""
                CREATE TABLE IF NOT EXISTS context_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_instance_id TEXT NOT NULL,
                    context_key TEXT NOT NULL,
                    context_value TEXT NOT NULL,
                    context_type TEXT DEFAULT 'json',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(workflow_instance_id, context_key)
                )
            """)
            
            await db.commit()
    
    async def _init_redis(self):
        """Initialize Redis connection for caching and real-time state."""
        try:
            import redis.asyncio as redis
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            
            # Test connection
            await self.redis_client.ping()
            self.logger.info("Redis connection established")
            
        except Exception as e:
            self.logger.warning(f"Redis connection failed: {e}. Using SQLite only.")
            self.use_redis = False
            self.redis_client = None
    
    async def _register_capabilities(self):
        """Register state management capabilities."""
        capabilities = [
            Capability(
                id="state-get",
                version="1.0",
                description="Retrieve workflow state",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workflow_instance_id": {"type": "string"},
                        "keys": {"type": "array", "items": {"type": "string"}, "default": []},
                        "include_history": {"type": "boolean", "default": False}
                    },
                    "required": ["workflow_instance_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "state": {"type": "object"},
                        "version": {"type": "integer"},
                        "history": {"type": "array", "default": []}
                    }
                }
            ),
            Capability(
                id="state-set",
                version="1.0",
                description="Set workflow state",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workflow_instance_id": {"type": "string"},
                        "state": {"type": "object"},
                        "overwrite": {"type": "boolean", "default": False},
                        "description": {"type": "string", "default": ""}
                    },
                    "required": ["workflow_instance_id", "state"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "version": {"type": "integer"}
                    }
                }
            ),
            Capability(
                id="state-update",
                version="1.0",
                description="Update workflow state",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workflow_instance_id": {"type": "string"},
                        "updates": {"type": "object"},
                        "merge_strategy": {"type": "string", "enum": ["merge", "replace"], "default": "merge"},
                        "description": {"type": "string", "default": ""}
                    },
                    "required": ["workflow_instance_id", "updates"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "version": {"type": "integer"},
                        "updated_keys": {"type": "array"}
                    }
                }
            ),
            Capability(
                id="state-query",
                version="1.0",
                description="Query workflow states with filters",
                input_schema={
                    "type": "object",
                    "properties": {
                        "filters": {"type": "object", "default": {}},
                        "projection": {"type": "array", "items": {"type": "string"}, "default": []},
                        "limit": {"type": "integer", "default": 100}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "total_count": {"type": "integer"}
                    }
                }
            ),
            Capability(
                id="context-merge",
                version="1.0",
                description="Merge contexts between workflow instances",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source_instance_id": {"type": "string"},
                        "target_instance_id": {"type": "string"},
                        "keys": {"type": "array", "items": {"type": "string"}, "default": []},
                        "merge_strategy": {"type": "string", "enum": ["merge", "replace"], "default": "merge"}
                    },
                    "required": ["source_instance_id", "target_instance_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "merged_keys": {"type": "array"}
                    }
                }
            )
        ]
        
        for capability in capabilities:
            self.amp_client.register_capability(capability)
    
    async def _handle_get_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state retrieval request."""
        try:
            workflow_instance_id = message.payload["parameters"]["workflow_instance_id"]
            keys = message.payload["parameters"].get("keys", [])
            include_history = message.payload["parameters"].get("include_history", False)
            
            # Get state from storage
            state_data, version = await self._get_workflow_state(workflow_instance_id)
            
            # Filter by keys if specified
            if keys and state_data:
                filtered_state = {k: state_data.get(k) for k in keys if k in state_data}
            else:
                filtered_state = state_data or {}
            
            result = {
                "state": filtered_state,
                "version": version
            }
            
            # Include history if requested
            if include_history:
                history = await self._get_state_history(workflow_instance_id)
                result["history"] = history
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get state: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.STATE_CORRUPTION,
                error_message=str(e)
            )
    
    async def _handle_set_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state setting request."""
        try:
            workflow_instance_id = message.payload["parameters"]["workflow_instance_id"]
            state = message.payload["parameters"]["state"]
            overwrite = message.payload["parameters"].get("overwrite", False)
            description = message.payload["parameters"].get("description", "State set")
            
            # Set state in storage
            version = await self._set_workflow_state(
                workflow_instance_id, state, overwrite, description
            )
            
            # Emit state update event
            await self._emit_state_event(
                workflow_instance_id,
                WorkflowEventTypes.STATE_UPDATED,
                {"action": "set", "version": version}
            )
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "success": True,
                    "version": version
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to set state: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.STATE_CORRUPTION,
                error_message=str(e)
            )
    
    async def _handle_update_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state update request."""
        try:
            workflow_instance_id = message.payload["parameters"]["workflow_instance_id"]
            updates = message.payload["parameters"]["updates"]
            merge_strategy = message.payload["parameters"].get("merge_strategy", "merge")
            description = message.payload["parameters"].get("description", "State updated")
            
            # Update state in storage
            version, updated_keys = await self._update_workflow_state(
                workflow_instance_id, updates, merge_strategy, description
            )
            
            # Emit state update event
            await self._emit_state_event(
                workflow_instance_id,
                WorkflowEventTypes.STATE_UPDATED,
                {"action": "update", "version": version, "updated_keys": updated_keys}
            )
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "success": True,
                    "version": version,
                    "updated_keys": updated_keys
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to update state: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.STATE_CORRUPTION,
                error_message=str(e)
            )
    
    async def _get_workflow_state(self, workflow_instance_id: str) -> tuple[Dict[str, Any], int]:
        """Retrieve workflow state from storage."""
        # Check cache first
        if workflow_instance_id in self.state_cache:
            cache_time = self.cache_timestamps.get(workflow_instance_id)
            if cache_time and (datetime.now(timezone.utc) - cache_time).total_seconds() < self.cache_ttl:
                version = self.state_versions.get(workflow_instance_id, 1)
                return self.state_cache[workflow_instance_id], version
        
        # Try Redis first if available
        if self.redis_client:
            try:
                state_json = await self.redis_client.get(f"workflow_state:{workflow_instance_id}")
                version_str = await self.redis_client.get(f"workflow_version:{workflow_instance_id}")
                
                if state_json:
                    state_data = json.loads(state_json)
                    version = int(version_str) if version_str else 1
                    
                    # Update cache
                    self.state_cache[workflow_instance_id] = state_data
                    self.cache_timestamps[workflow_instance_id] = datetime.now(timezone.utc)
                    self.state_versions[workflow_instance_id] = version
                    
                    return state_data, version
            except Exception as e:
                self.logger.warning(f"Redis get failed: {e}")
        
        # Fall back to SQLite
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT state_data, version FROM workflow_states WHERE workflow_instance_id = ?",
                (workflow_instance_id,)
            )
            row = await cursor.fetchone()
            
            if row:
                state_data = json.loads(row[0])
                version = row[1]
                
                # Update cache
                self.state_cache[workflow_instance_id] = state_data
                self.cache_timestamps[workflow_instance_id] = datetime.now(timezone.utc)
                self.state_versions[workflow_instance_id] = version
                
                # Update Redis if available
                if self.redis_client:
                    try:
                        await self.redis_client.setex(
                            f"workflow_state:{workflow_instance_id}",
                            self.cache_ttl,
                            json.dumps(state_data)
                        )
                        await self.redis_client.setex(
                            f"workflow_version:{workflow_instance_id}",
                            self.cache_ttl,
                            str(version)
                        )
                    except Exception as e:
                        self.logger.warning(f"Redis update failed: {e}")
                
                return state_data, version
            
            return {}, 1
    
    async def _set_workflow_state(self, workflow_instance_id: str, state: Dict[str, Any],
                                 overwrite: bool = False, description: str = "") -> int:
        """Set workflow state in storage."""
        # Get current state if not overwriting
        if not overwrite:
            current_state, current_version = await self._get_workflow_state(workflow_instance_id)
            if current_state:
                raise ValueError("State already exists. Use overwrite=True to replace.")
        else:
            current_version = self.state_versions.get(workflow_instance_id, 0)
        
        new_version = current_version + 1
        state_json = json.dumps(state)
        
        # Update SQLite
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO workflow_states 
                (workflow_instance_id, state_data, version, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (workflow_instance_id, state_json, new_version))
            
            # Add to history
            await db.execute("""
                INSERT INTO state_history 
                (workflow_instance_id, state_data, version, change_description)
                VALUES (?, ?, ?, ?)
            """, (workflow_instance_id, state_json, new_version, description))
            
            await db.commit()
        
        # Update Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"workflow_state:{workflow_instance_id}",
                    self.cache_ttl,
                    state_json
                )
                await self.redis_client.setex(
                    f"workflow_version:{workflow_instance_id}",
                    self.cache_ttl,
                    str(new_version)
                )
            except Exception as e:
                self.logger.warning(f"Redis set failed: {e}")
        
        # Update cache
        self.state_cache[workflow_instance_id] = state
        self.cache_timestamps[workflow_instance_id] = datetime.now(timezone.utc)
        self.state_versions[workflow_instance_id] = new_version
        
        return new_version
    
    async def _update_workflow_state(self, workflow_instance_id: str, updates: Dict[str, Any],
                                   merge_strategy: str = "merge", description: str = "") -> tuple[int, List[str]]:
        """Update workflow state in storage."""
        # Get current state
        current_state, current_version = await self._get_workflow_state(workflow_instance_id)
        
        # Apply updates based on strategy
        if merge_strategy == "merge":
            new_state = {**current_state, **updates}
        else:  # replace
            new_state = updates
        
        updated_keys = list(updates.keys())
        new_version = current_version + 1
        state_json = json.dumps(new_state)
        
        # Update SQLite
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                INSERT OR REPLACE INTO workflow_states 
                (workflow_instance_id, state_data, version, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            """, (workflow_instance_id, state_json, new_version))
            
            # Add to history
            await db.execute("""
                INSERT INTO state_history 
                (workflow_instance_id, state_data, version, change_description)
                VALUES (?, ?, ?, ?)
            """, (workflow_instance_id, state_json, new_version, description))
            
            await db.commit()
        
        # Update Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    f"workflow_state:{workflow_instance_id}",
                    self.cache_ttl,
                    state_json
                )
                await self.redis_client.setex(
                    f"workflow_version:{workflow_instance_id}",
                    self.cache_ttl,
                    str(new_version)
                )
            except Exception as e:
                self.logger.warning(f"Redis update failed: {e}")
        
        # Update cache
        self.state_cache[workflow_instance_id] = new_state
        self.cache_timestamps[workflow_instance_id] = datetime.now(timezone.utc)
        self.state_versions[workflow_instance_id] = new_version
        
        return new_version, updated_keys
    
    async def _get_state_history(self, workflow_instance_id: str) -> List[Dict[str, Any]]:
        """Get state change history for a workflow instance."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                SELECT state_data, version, change_description, timestamp
                FROM state_history 
                WHERE workflow_instance_id = ?
                ORDER BY timestamp DESC
                LIMIT 50
            """, (workflow_instance_id,))
            
            rows = await cursor.fetchall()
            
            history = []
            for row in rows:
                history.append({
                    "state_data": json.loads(row[0]),
                    "version": row[1],
                    "description": row[2],
                    "timestamp": row[3]
                })
            
            return history
    
    async def _emit_state_event(self, workflow_instance_id: str, event_type: str, data: Dict[str, Any]):
        """Emit state change event."""
        await self.amp_client.send_event(
            event_type=event_type,
            data={
                "workflow_instance_id": workflow_instance_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **data
            }
        )
    
    async def _cache_cleanup_task(self):
        """Periodically clean up expired cache entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                current_time = datetime.now(timezone.utc)
                expired_keys = []
                
                for key, timestamp in self.cache_timestamps.items():
                    if (current_time - timestamp).total_seconds() > self.cache_ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    self.state_cache.pop(key, None)
                    self.cache_timestamps.pop(key, None)
                    self.state_versions.pop(key, None)
                
                if expired_keys:
                    self.logger.debug(f"Cleaned up {len(expired_keys)} expired cache entries")
                    
            except Exception as e:
                self.logger.error(f"Cache cleanup failed: {e}")
    
    # Additional handler methods would continue here...
    async def _handle_delete_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state deletion request."""
        # Implementation for state deletion
        pass
    
    async def _handle_list_states(self, message: AMPMessage) -> AMPMessage:
        """Handle state listing request."""
        # Implementation for listing workflow states
        pass
    
    async def _handle_query_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state query request."""
        # Implementation for querying states with filters
        pass
    
    async def _handle_backup_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state backup request."""
        # Implementation for state backup
        pass
    
    async def _handle_restore_state(self, message: AMPMessage) -> AMPMessage:
        """Handle state restore request."""
        # Implementation for state restore
        pass
    
    async def _handle_merge_context(self, message: AMPMessage) -> AMPMessage:
        """Handle context merge request."""
        # Implementation for merging contexts between workflows
        pass
    
    async def _handle_isolate_context(self, message: AMPMessage) -> AMPMessage:
        """Handle context isolation request."""
        # Implementation for context isolation
        pass