"""
Production-ready AMP client implementation.
"""

import asyncio
import logging
import time
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

from .amp_types import (
    AMPMessage, MessageType, AgentIdentity, Capability, 
    TransportType, AMPErrorCodes, CapabilityStatus
)
from .amp_transport import HTTPTransport, WebSocketTransport, ITransport
from .amp_router import MessageRouter
from .amp_security import SecurityManager
from .amp_registry import CapabilityRegistry


@dataclass
class AMPClientConfig:
    """Configuration for AMP client."""
    agent_id: str
    agent_name: str
    framework: str = "custom"
    version: str = "1.0.0"
    transport_type: TransportType = TransportType.HTTP
    endpoint: str = "http://localhost:8000"
    api_key: Optional[str] = None
    auto_reconnect: bool = True
    message_timeout: int = 30000
    max_concurrent_requests: int = 100
    heartbeat_interval: int = 30
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    log_level: str = "INFO"
    enable_metrics: bool = True
    registry_endpoint: Optional[str] = None


class AMPClient:
    """Production-ready AMP client implementation."""
    
    def __init__(self, config: AMPClientConfig):
        self.config = config
        self.identity = AgentIdentity(
            id=config.agent_id,
            name=config.agent_name,
            version=config.version,
            framework=config.framework
        )
        
        # Core components
        self.transport: Optional[ITransport] = None
        self.router = MessageRouter()
        self.security = SecurityManager(config.api_key)
        self.registry = CapabilityRegistry(config.registry_endpoint)
        
        # State management
        self.capabilities: Dict[str, Dict[str, Any]] = {}
        self._pending_requests: Dict[str, asyncio.Future] = {}
        self._receive_task: Optional[asyncio.Task] = None
        self._connected = False
        self._message_queue = asyncio.Queue(maxsize=1000)
        self._send_semaphore = asyncio.Semaphore(config.max_concurrent_requests)
        
        # Metrics
        self._metrics = {
            "messages_sent": 0,
            "messages_received": 0,
            "errors": 0,
            "capabilities_invoked": 0,
            "avg_response_time": 0.0
        }
        
        # Set up logging
        self.logger = logging.getLogger(f"amp.client.{config.agent_id}")
        self.logger.setLevel(config.log_level)
        
    async def connect(self) -> bool:
        """Connect to AMP network."""
        try:
            # Create and connect transport
            self.transport = self._create_transport()
            connected = await self.transport.connect()
            
            if not connected:
                self.logger.error("Failed to connect transport")
                return False
            
            self._connected = True
            
            # Start message receiver
            self._receive_task = asyncio.create_task(self._receive_loop())
            
            # Register with network
            await self._register_agent()
            
            self.logger.info(f"Connected to AMP network as {self.config.agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from AMP network."""
        self._connected = False
        
        # Cancel receive task
        if self._receive_task:
            self._receive_task.cancel()
            try:
                await self._receive_task
            except asyncio.CancelledError:
                pass
        
        # Deregister agent
        try:
            await self._deregister_agent()
        except Exception as e:
            self.logger.warning(f"Failed to deregister: {e}")
        
        # Disconnect transport
        if self.transport:
            await self.transport.disconnect()
        
        # Cancel pending requests
        for future in self._pending_requests.values():
            if not future.done():
                future.cancel()
        self._pending_requests.clear()
        
        self.logger.info("Disconnected from AMP network")
    
    def register_capability(self, capability: Capability, handler: Callable):
        """Register a capability with handler."""
        self.capabilities[capability.id] = {
            "capability": capability,
            "handler": handler,
            "metrics": {
                "invocations": 0,
                "successes": 0,
                "errors": 0,
                "avg_response_time": 0.0
            }
        }
        
        # Add route for this capability
        self.router.add_route(
            f"request/{capability.id}",
            self._handle_capability_request
        )
        
        self.logger.info(f"Registered capability: {capability.id}")
    
    async def invoke_capability(self, target_agent: Optional[str], 
                               capability: str, parameters: Dict[str, Any],
                               timeout: Optional[int] = None) -> Dict[str, Any]:
        """Invoke a capability on another agent."""
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        # Create request message
        message = AMPMessage.create_request(
            self.identity.id,
            target_agent,
            capability,
            parameters,
            timeout or self.config.message_timeout
        )
        
        # Send and wait for response
        start_time = time.time()
        
        async with self._send_semaphore:
            try:
                response = await self._send_and_wait(message, (timeout or self.config.message_timeout) / 1000)
                self._update_metrics("capability_invoked", time.time() - start_time)
                return response
            except Exception as e:
                self._update_metrics("error")
                raise
    
    async def emit_event(self, event_type: str, data: Dict[str, Any], 
                        target_agent: Optional[str] = None):
        """Emit an event to the network."""
        if not self._connected:
            raise RuntimeError("Client not connected")
        
        message = AMPMessage.create_event(
            self.identity.id,
            event_type,
            data,
            broadcast=(target_agent is None)
        )
        
        if target_agent:
            message.destination.agent_id = target_agent
        
        await self._send_message(message)
    
    def on_event(self, event_type: str, handler: Callable):
        """Register an event handler."""
        self.router.add_route(
            "event/*",
            handler,
            event_type=event_type
        )
    
    def on_request(self, capability: str, handler: Callable):
        """Register a request handler."""
        # Create a basic capability definition
        capability_def = Capability(
            id=capability,
            version="1.0",
            description=f"Handler for {capability}",
            input_schema={"type": "object"},
            output_schema={"type": "object"}
        )
        self.register_capability(capability_def, handler)
    
    async def discover_agents(self, capability: Optional[str] = None,
                             filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Discover agents in the network."""
        if not self.registry.endpoint:
            raise RuntimeError("Registry endpoint not configured")
        
        return await self.registry.discover_agents(capability, filters)
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        return {
            "status": "healthy" if self._connected else "unhealthy",
            "agent_id": self.identity.id,
            "capabilities": list(self.capabilities.keys()),
            "connected": self._connected,
            "pending_requests": len(self._pending_requests),
            "metrics": self._metrics.copy()
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        return {
            **self._metrics,
            "capabilities": {
                cap_id: cap_data["metrics"] 
                for cap_id, cap_data in self.capabilities.items()
            }
        }
    
    def _create_transport(self) -> ITransport:
        """Create transport based on configuration."""
        if self.config.transport_type == TransportType.HTTP:
            return HTTPTransport(
                self.config.endpoint,
                self.config.api_key,
                timeout=self.config.message_timeout // 1000,
                max_retries=self.config.retry_attempts
            )
        elif self.config.transport_type == TransportType.WEBSOCKET:
            ws_url = self.config.endpoint.replace("http://", "ws://").replace("https://", "wss://")
            return WebSocketTransport(
                ws_url,
                self.config.api_key,
                self.config.heartbeat_interval
            )
        else:
            raise ValueError(f"Unsupported transport: {self.config.transport_type}")
    
    async def _send_message(self, message: AMPMessage) -> Optional[str]:
        """Send a message via transport."""
        # Add security headers
        message_dict = message.to_dict()
        self.security.sign_message(message_dict)
        
        # Send via transport
        result = await self.transport.send(message_dict)
        self._update_metrics("message_sent")
        return result
    
    async def _send_and_wait(self, message: AMPMessage, timeout: float) -> Dict[str, Any]:
        """Send message and wait for response."""
        msg_id = message.id
        
        # Create future for response
        future = asyncio.Future()
        self._pending_requests[msg_id] = future
        
        try:
            # Send message
            await self._send_message(message)
            
            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            
            # Check for error response
            if response["message"]["type"] == "error":
                error = response["message"]["payload"]["error"]
                raise Exception(f"{error['code']}: {error['message']}")
            
            return response["message"]["payload"]
            
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request {msg_id} timed out after {timeout}s")
        finally:
            self._pending_requests.pop(msg_id, None)
    
    async def _receive_loop(self):
        """Continuously receive messages from transport."""
        while self._connected:
            try:
                async for message in self.transport.receive():
                    self._update_metrics("message_received")
                    asyncio.create_task(self._handle_message(message))
                    
            except Exception as e:
                self.logger.error(f"Error in receive loop: {e}")
                self._update_metrics("error")
                
                if self.config.auto_reconnect and self._connected:
                    await asyncio.sleep(self.config.retry_backoff)
                    if await self._reconnect():
                        continue
                    else:
                        break
                else:
                    break
    
    async def _handle_message(self, message_dict: Dict[str, Any]):
        """Handle incoming message."""
        try:
            # Verify message signature
            if not self.security.verify_message(message_dict):
                self.logger.warning("Received message with invalid signature")
                return
            
            message = AMPMessage.from_dict(message_dict)
            
            # Handle responses to our requests
            if message.type in [MessageType.RESPONSE, MessageType.ERROR]:
                correlation_id = message.headers.correlation_id
                if correlation_id in self._pending_requests:
                    future = self._pending_requests.get(correlation_id)
                    if future and not future.done():
                        future.set_result(message_dict)
                    return
            
            # Route other messages
            await self.router.route_message(message_dict)
            
        except Exception as e:
            self.logger.error(f"Error handling message: {e}")
            self._update_metrics("error")
    
    async def _handle_capability_request(self, message_dict: Dict[str, Any]):
        """Handle incoming capability request."""
        message = AMPMessage.from_dict(message_dict)
        capability_id = message.payload["capability"]
        parameters = message.payload.get("parameters", {})
        
        if capability_id not in self.capabilities:
            # Send error response
            error_msg = AMPMessage.create_error(
                self.identity.id,
                message.source.agent_id,
                message.id,
                AMPErrorCodes.CAPABILITY_NOT_FOUND,
                f"Capability '{capability_id}' not found"
            )
            await self._send_message(error_msg)
            return
        
        cap_data = self.capabilities[capability_id]
        start_time = time.time()
        
        try:
            # Execute capability handler
            handler = cap_data["handler"]
            result = await self._call_handler(handler, parameters)
            
            # Send success response
            response = AMPMessage.create_response(
                self.identity.id,
                message.source.agent_id,
                message.id,
                result,
                CapabilityStatus.SUCCESS
            )
            await self._send_message(response)
            
            # Update metrics
            execution_time = time.time() - start_time
            self._update_capability_metrics(capability_id, True, execution_time)
            
        except Exception as e:
            # Send error response
            error_msg = AMPMessage.create_error(
                self.identity.id,
                message.source.agent_id,
                message.id,
                AMPErrorCodes.EXECUTION_ERROR,
                str(e)
            )
            await self._send_message(error_msg)
            
            # Update metrics
            self._update_capability_metrics(capability_id, False, time.time() - start_time)
    
    async def _call_handler(self, handler: Callable, parameters: Dict[str, Any]) -> Any:
        """Call capability handler (sync or async)."""
        if asyncio.iscoroutinefunction(handler):
            return await handler(parameters)
        else:
            return handler(parameters)
    
    async def _register_agent(self):
        """Register agent with the network."""
        if self.registry.endpoint:
            try:
                await self.registry.register_agent(
                    self.identity,
                    list(self.capabilities.values())
                )
            except Exception as e:
                self.logger.warning(f"Failed to register with registry: {e}")
        
        self.logger.info(f"Registered agent {self.identity.id}")
    
    async def _deregister_agent(self):
        """Deregister agent from the network."""
        if self.registry.endpoint:
            try:
                await self.registry.deregister_agent(self.identity.id)
            except Exception as e:
                self.logger.warning(f"Failed to deregister from registry: {e}")
        
        self.logger.info(f"Deregistered agent {self.identity.id}")
    
    async def _reconnect(self) -> bool:
        """Attempt to reconnect."""
        self.logger.info("Attempting to reconnect...")
        
        try:
            # Disconnect current transport
            if self.transport:
                await self.transport.disconnect()
            
            # Create new transport and connect
            self.transport = self._create_transport()
            connected = await self.transport.connect()
            
            if connected:
                self.logger.info("Reconnected successfully")
                await self._register_agent()
                return True
            else:
                self.logger.error("Reconnection failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Reconnection error: {e}")
            return False
    
    def _update_metrics(self, metric_type: str, value: float = 1.0):
        """Update metrics."""
        if metric_type == "message_sent":
            self._metrics["messages_sent"] += 1
        elif metric_type == "message_received":
            self._metrics["messages_received"] += 1
        elif metric_type == "error":
            self._metrics["errors"] += 1
        elif metric_type == "capability_invoked":
            self._metrics["capabilities_invoked"] += 1
            # Update average response time
            count = self._metrics["capabilities_invoked"]
            current_avg = self._metrics["avg_response_time"]
            self._metrics["avg_response_time"] = ((current_avg * (count - 1)) + value) / count
    
    def _update_capability_metrics(self, capability_id: str, success: bool, execution_time: float):
        """Update capability-specific metrics."""
        if capability_id in self.capabilities:
            metrics = self.capabilities[capability_id]["metrics"]
            metrics["invocations"] += 1
            
            if success:
                metrics["successes"] += 1
            else:
                metrics["errors"] += 1
            
            # Update average response time
            count = metrics["invocations"]
            current_avg = metrics["avg_response_time"]
            metrics["avg_response_time"] = ((current_avg * (count - 1)) + execution_time) / count


@asynccontextmanager
async def amp_client(config: AMPClientConfig):
    """Context manager for AMP client."""
    client = AMPClient(config)
    try:
        connected = await client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        yield client
    finally:
        await client.disconnect()