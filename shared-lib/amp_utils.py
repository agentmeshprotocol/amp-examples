"""
Utility functions and builders for AMP.
"""

import asyncio
from typing import Dict, Any, List, Callable, Optional
from .amp_client import AMPClient, AMPClientConfig
from .amp_types import Capability, TransportType, CapabilityConstraints


class AMPBuilder:
    """Builder for creating AMP agents easily."""
    
    def __init__(self, agent_id: str, name: str):
        self.config = AMPClientConfig(
            agent_id=agent_id,
            agent_name=name
        )
        self.capabilities = []
        self.event_handlers = []
        self.middleware = []
    
    def with_transport(self, transport_type: TransportType, endpoint: str) -> 'AMPBuilder':
        """Configure transport."""
        self.config.transport_type = transport_type
        self.config.endpoint = endpoint
        return self
    
    def with_api_key(self, api_key: str) -> 'AMPBuilder':
        """Set API key."""
        self.config.api_key = api_key
        return self
    
    def with_framework(self, framework: str) -> 'AMPBuilder':
        """Set framework."""
        self.config.framework = framework
        return self
    
    def with_registry(self, registry_endpoint: str) -> 'AMPBuilder':
        """Set registry endpoint."""
        self.config.registry_endpoint = registry_endpoint
        return self
    
    def with_timeout(self, timeout_ms: int) -> 'AMPBuilder':
        """Set message timeout."""
        self.config.message_timeout = timeout_ms
        return self
    
    def with_auto_reconnect(self, enabled: bool = True) -> 'AMPBuilder':
        """Enable/disable auto-reconnect."""
        self.config.auto_reconnect = enabled
        return self
    
    def add_capability(self, capability_id: str, handler: Callable, 
                      description: str = "", category: str = "",
                      input_schema: Optional[Dict[str, Any]] = None,
                      output_schema: Optional[Dict[str, Any]] = None,
                      constraints: Optional[CapabilityConstraints] = None) -> 'AMPBuilder':
        """Add capability."""
        capability = Capability(
            id=capability_id,
            version="1.0",
            description=description or f"Handler for {capability_id}",
            category=category,
            input_schema=input_schema or {"type": "object"},
            output_schema=output_schema or {"type": "object"},
            constraints=constraints or CapabilityConstraints()
        )
        
        self.capabilities.append((capability, handler))
        return self
    
    def on_event(self, event_type: str, handler: Callable) -> 'AMPBuilder':
        """Add event handler."""
        self.event_handlers.append((event_type, handler))
        return self
    
    def add_middleware(self, middleware: Callable) -> 'AMPBuilder':
        """Add middleware."""
        self.middleware.append(middleware)
        return self
    
    async def build(self) -> AMPClient:
        """Build and connect the agent."""
        client = AMPClient(self.config)
        
        # Add middleware
        for middleware in self.middleware:
            client.router.add_middleware(middleware)
        
        # Register capabilities
        for capability, handler in self.capabilities:
            client.register_capability(capability, handler)
        
        # Register event handlers
        for event_type, handler in self.event_handlers:
            client.on_event(event_type, handler)
        
        # Connect
        connected = await client.connect()
        if not connected:
            raise RuntimeError("Failed to connect agent")
        
        return client


def create_simple_capability(capability_id: str, 
                           handler_func: Callable,
                           description: str = "",
                           category: str = "") -> Capability:
    """Create a simple capability definition."""
    return Capability(
        id=capability_id,
        version="1.0",
        description=description or f"Simple {capability_id} capability",
        category=category,
        input_schema={"type": "object"},
        output_schema={"type": "object"},
        constraints=CapabilityConstraints()
    )


def create_text_analysis_capability(handler: Callable) -> Capability:
    """Create a standard text analysis capability."""
    return Capability(
        id="text-analysis",
        version="1.0",
        description="Analyze and extract insights from text",
        category="text-processing",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "maxLength": 50000},
                "analysis_types": {
                    "type": "array",
                    "items": {
                        "enum": ["sentiment", "entities", "topics", "language", "statistics"]
                    }
                }
            },
            "required": ["text"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "sentiment": {"type": "object"},
                "entities": {"type": "array"},
                "topics": {"type": "array"},
                "language": {"type": "object"},
                "statistics": {"type": "object"}
            }
        },
        constraints=CapabilityConstraints(
            max_input_length=50000,
            supported_languages=["en", "es", "fr"],
            response_time_ms=5000
        )
    )


def create_qa_capability(handler: Callable) -> Capability:
    """Create a standard Q&A capability."""
    return Capability(
        id="qa-factual",
        version="1.0",
        description="Answer factual questions based on knowledge or context",
        category="question-answering",
        input_schema={
            "type": "object",
            "properties": {
                "question": {"type": "string"},
                "context": {"type": "string"},
                "answer_format": {
                    "enum": ["short", "detailed", "list", "yes_no"]
                }
            },
            "required": ["question"]
        },
        output_schema={
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "confidence": {"type": "number"},
                "sources": {"type": "array", "items": {"type": "string"}}
            }
        },
        constraints=CapabilityConstraints(
            response_time_ms=3000,
            min_confidence=0.7
        )
    )


class MessageLogger:
    """Middleware for logging messages."""
    
    def __init__(self, logger):
        self.logger = logger
    
    async def __call__(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Log incoming message."""
        msg_data = message.get("message", {})
        self.logger.info(
            f"Message {msg_data.get('type')}: {msg_data.get('id')} "
            f"from {msg_data.get('source', {}).get('agent_id')}"
        )
        return message


class MetricsCollector:
    """Middleware for collecting metrics."""
    
    def __init__(self):
        self.message_count = 0
        self.message_types = {}
        self.error_count = 0
    
    async def __call__(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Collect message metrics."""
        self.message_count += 1
        
        msg_type = message.get("message", {}).get("type", "unknown")
        self.message_types[msg_type] = self.message_types.get(msg_type, 0) + 1
        
        if msg_type == "error":
            self.error_count += 1
        
        return message
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        return {
            "total_messages": self.message_count,
            "message_types": self.message_types.copy(),
            "error_count": self.error_count,
            "error_rate": self.error_count / max(1, self.message_count)
        }


class RetryHandler:
    """Utility for handling retries."""
    
    def __init__(self, max_retries: int = 3, backoff_factor: float = 1.0):
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
    
    async def retry(self, func: Callable, *args, **kwargs):
        """Retry a function with exponential backoff."""
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self.max_retries:
                    wait_time = self.backoff_factor * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                else:
                    break
        
        raise last_exception


def validate_amp_message(message: Dict[str, Any]) -> bool:
    """Validate AMP message format."""
    try:
        # Check protocol version
        if message.get("protocol") != "AMP/1.0":
            return False
        
        msg = message.get("message", {})
        
        # Check required fields
        required_fields = ["id", "type", "timestamp", "source", "destination", "payload"]
        for field in required_fields:
            if field not in msg:
                return False
        
        # Check message type
        valid_types = ["request", "response", "event", "error"]
        if msg["type"] not in valid_types:
            return False
        
        # Check source
        source = msg.get("source", {})
        if "agent_id" not in source:
            return False
        
        return True
        
    except Exception:
        return False


async def wait_for_response(client: AMPClient, correlation_id: str, 
                          timeout: float = 30.0) -> Optional[Dict[str, Any]]:
    """Wait for a specific response message."""
    future = asyncio.Future()
    
    async def response_handler(message: Dict[str, Any]):
        msg_data = message.get("message", {})
        if msg_data.get("headers", {}).get("correlation_id") == correlation_id:
            if not future.done():
                future.set_result(message)
    
    # Add temporary handler
    client.router.add_route("response", response_handler)
    
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        return None
    finally:
        # Remove temporary handler
        client.router.remove_route("response", response_handler)
