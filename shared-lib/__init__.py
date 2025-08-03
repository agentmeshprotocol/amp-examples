"""
Agent Mesh Protocol (AMP) Shared Library

This library provides production-ready implementations of the AMP protocol
for building interoperable AI agent systems.
"""

from .amp_client import AMPClient, AMPClientConfig
from .amp_types import (
    MessageType, 
    CapabilityStatus, 
    AgentIdentity, 
    Capability,
    AMPMessage
)
from .amp_transport import HTTPTransport, WebSocketTransport
from .amp_router import MessageRouter
from .amp_registry import CapabilityRegistry
from .amp_security import SecurityManager
from .amp_utils import AMPBuilder

__version__ = "1.0.0"
__all__ = [
    "AMPClient",
    "AMPClientConfig", 
    "MessageType",
    "CapabilityStatus",
    "AgentIdentity",
    "Capability",
    "AMPMessage",
    "HTTPTransport",
    "WebSocketTransport", 
    "MessageRouter",
    "CapabilityRegistry",
    "SecurityManager",
    "AMPBuilder"
]