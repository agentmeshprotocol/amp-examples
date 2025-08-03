"""
Core AMP data types and message structures.
"""

from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import uuid


class MessageType(Enum):
    """AMP message types."""
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    ERROR = "error"


class CapabilityStatus(Enum):
    """Capability execution status."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class TransportType(Enum):
    """Supported transport types."""
    HTTP = "http"
    WEBSOCKET = "websocket"
    GRPC = "grpc"
    MESSAGE_QUEUE = "message_queue"


@dataclass
class AgentIdentity:
    """Agent identity information."""
    id: str
    name: str
    version: str
    framework: str
    protocol_version: str = "1.0"
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CapabilityConstraints:
    """Capability execution constraints."""
    max_input_length: Optional[int] = None
    supported_languages: List[str] = field(default_factory=list)
    response_time_ms: int = 5000
    max_tokens: Optional[int] = None
    rate_limit_per_hour: Optional[int] = None
    min_confidence: Optional[float] = None


@dataclass
class Capability:
    """Capability definition."""
    id: str
    version: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    constraints: CapabilityConstraints = field(default_factory=CapabilityConstraints)
    category: Optional[str] = None
    subcategories: List[str] = field(default_factory=list)
    examples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MessageSource:
    """Message source information."""
    agent_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None


@dataclass
class MessageDestination:
    """Message destination information."""
    agent_id: Optional[str] = None
    capability: Optional[str] = None
    routing_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MessageHeaders:
    """Message headers."""
    correlation_id: Optional[str] = None
    priority: int = 5
    timeout_ms: int = 30000
    routing_hints: Dict[str, Any] = field(default_factory=dict)
    authentication: Optional[Dict[str, Any]] = None
    signature: Optional[Dict[str, Any]] = None
    tracing: Optional[Dict[str, Any]] = None


@dataclass
class AMPMessage:
    """Complete AMP message structure."""
    id: str
    type: MessageType
    timestamp: str
    source: MessageSource
    destination: MessageDestination
    headers: MessageHeaders
    payload: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format."""
        return {
            "protocol": "AMP/1.0",
            "message": {
                "id": self.id,
                "type": self.type.value,
                "timestamp": self.timestamp,
                "source": asdict(self.source),
                "destination": asdict(self.destination),
                "headers": asdict(self.headers),
                "payload": self.payload
            }
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AMPMessage':
        """Create message from dictionary."""
        msg = data["message"]
        return cls(
            id=msg["id"],
            type=MessageType(msg["type"]),
            timestamp=msg["timestamp"],
            source=MessageSource(**msg["source"]),
            destination=MessageDestination(**msg["destination"]),
            headers=MessageHeaders(**msg["headers"]),
            payload=msg["payload"]
        )
    
    @classmethod
    def create_request(cls, agent_id: str, target_agent: Optional[str], 
                      capability: str, parameters: Dict[str, Any],
                      timeout_ms: int = 30000) -> 'AMPMessage':
        """Create a capability request message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=MessageSource(agent_id=agent_id),
            destination=MessageDestination(agent_id=target_agent, capability=capability),
            headers=MessageHeaders(timeout_ms=timeout_ms),
            payload={
                "capability": capability,
                "parameters": parameters
            }
        )
    
    @classmethod
    def create_response(cls, agent_id: str, target_agent: str, 
                       correlation_id: str, result: Any,
                       status: CapabilityStatus = CapabilityStatus.SUCCESS) -> 'AMPMessage':
        """Create a response message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.RESPONSE,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=MessageSource(agent_id=agent_id),
            destination=MessageDestination(agent_id=target_agent),
            headers=MessageHeaders(correlation_id=correlation_id),
            payload={
                "status": status.value,
                "result": result,
                "metadata": {
                    "agent_id": agent_id,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )
    
    @classmethod
    def create_error(cls, agent_id: str, target_agent: str,
                    correlation_id: str, error_code: str, 
                    error_message: str, details: Optional[Dict[str, Any]] = None) -> 'AMPMessage':
        """Create an error message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.ERROR,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=MessageSource(agent_id=agent_id),
            destination=MessageDestination(agent_id=target_agent),
            headers=MessageHeaders(correlation_id=correlation_id),
            payload={
                "error": {
                    "code": error_code,
                    "message": error_message,
                    "details": details or {},
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            }
        )
    
    @classmethod
    def create_event(cls, agent_id: str, event_type: str, 
                    data: Dict[str, Any], broadcast: bool = True) -> 'AMPMessage':
        """Create an event message."""
        return cls(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            timestamp=datetime.now(timezone.utc).isoformat(),
            source=MessageSource(agent_id=agent_id),
            destination=MessageDestination() if broadcast else MessageDestination(),
            headers=MessageHeaders(),
            payload={
                "event_type": event_type,
                **data
            }
        )


@dataclass 
class AgentHealth:
    """Agent health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    last_seen: datetime
    response_time_avg: float
    success_rate: float
    error_count: int
    capabilities_available: List[str]
    load_level: float  # 0.0 to 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CapabilityMetrics:
    """Capability performance metrics."""
    capability_id: str
    invocation_count: int
    success_count: int
    error_count: int
    avg_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    last_invoked: datetime
    avg_confidence: Optional[float] = None


# Standard AMP error codes
class AMPErrorCodes:
    """Standard AMP error codes."""
    AGENT_NOT_FOUND = "AGENT_NOT_FOUND"
    CAPABILITY_NOT_FOUND = "CAPABILITY_NOT_FOUND"
    AUTHENTICATION_FAILED = "AUTHENTICATION_FAILED"
    AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
    RATE_LIMIT_EXCEEDED = "RATE_LIMIT_EXCEEDED"
    TIMEOUT = "TIMEOUT"
    INVALID_SCHEMA = "INVALID_SCHEMA"
    CONTEXT_NOT_FOUND = "CONTEXT_NOT_FOUND"
    INSUFFICIENT_RESOURCES = "INSUFFICIENT_RESOURCES"
    EXECUTION_ERROR = "EXECUTION_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    NETWORK_ERROR = "NETWORK_ERROR"
    INTERNAL_ERROR = "INTERNAL_ERROR"


# Capability categories from the taxonomy
class CapabilityCategories:
    """Standard capability categories."""
    TEXT_PROCESSING = "text-processing"
    GENERATION = "generation"
    QUESTION_ANSWERING = "question-answering"
    DATA_PROCESSING = "data-processing"
    ANALYSIS = "analysis"
    TOOL_USE = "tool-use"
    MEMORY = "memory"
    PLANNING = "planning"
    MULTIMODAL = "multimodal"