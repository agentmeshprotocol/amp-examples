"""
Transport layer implementations for AMP.
"""

import asyncio
import aiohttp
import websockets
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, AsyncIterator
import time


class ITransport(ABC):
    """Base interface for AMP message transport."""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Establish connection."""
        pass
    
    @abstractmethod
    async def disconnect(self):
        """Close connection."""
        pass
    
    @abstractmethod
    async def send(self, message: Dict[str, Any]) -> Optional[str]:
        """Send message and return message ID if successful."""
        pass
    
    @abstractmethod
    async def receive(self) -> AsyncIterator[Dict[str, Any]]:
        """Receive messages asynchronously."""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if transport is connected."""
        pass


class HTTPTransport(ITransport):
    """HTTP/REST transport implementation."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, 
                 timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self.logger = logging.getLogger(f"{__name__}.HTTPTransport")
        
    async def connect(self) -> bool:
        """Create HTTP session."""
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "AMP-SDK/1.0"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            self.session = aiohttp.ClientSession(
                headers=headers,
                timeout=self.timeout
            )
            
            # Test connection with health check
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status == 200:
                    self._connected = True
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to connect: {e}")
            if self.session:
                await self.session.close()
                self.session = None
        
        return False
    
    async def disconnect(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
        self._connected = False
    
    async def send(self, message: Dict[str, Any]) -> Optional[str]:
        """Send message via HTTP POST."""
        if not self._connected or not self.session:
            raise RuntimeError("Transport not connected")
        
        url = f"{self.base_url}/amp/v1/messages"
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(url, json=message) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("message_id")
                    elif response.status >= 500:
                        # Server error, retry
                        if attempt < self.max_retries - 1:
                            await asyncio.sleep(2 ** attempt)  # Exponential backoff
                            continue
                    else:
                        # Client error, don't retry
                        error_body = await response.text()
                        raise Exception(f"HTTP {response.status}: {error_body}")
                        
            except aiohttp.ClientError as e:
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise
        
        return None
    
    async def receive(self) -> AsyncIterator[Dict[str, Any]]:
        """HTTP doesn't support server push, use polling."""
        if not self._connected or not self.session:
            raise RuntimeError("Transport not connected")
        
        poll_url = f"{self.base_url}/amp/v1/messages/poll"
        
        while self._connected:
            try:
                async with self.session.get(poll_url) as response:
                    if response.status == 200:
                        data = await response.json()
                        messages = data.get("messages", [])
                        for msg in messages:
                            yield msg
                    elif response.status == 204:
                        # No messages available
                        pass
                    else:
                        self.logger.warning(f"Poll returned {response.status}")
                
                await asyncio.sleep(1)  # Poll interval
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error polling messages: {e}")
                await asyncio.sleep(5)  # Wait longer on error
    
    def is_connected(self) -> bool:
        return self._connected and self.session is not None


class WebSocketTransport(ITransport):
    """WebSocket transport implementation."""
    
    def __init__(self, ws_url: str, api_key: Optional[str] = None,
                 heartbeat_interval: int = 30):
        self.ws_url = ws_url
        self.api_key = api_key
        self.heartbeat_interval = heartbeat_interval
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self._connected = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(f"{__name__}.WebSocketTransport")
        
    async def connect(self) -> bool:
        """Establish WebSocket connection."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            self.websocket = await websockets.connect(
                self.ws_url,
                extra_headers=headers
            )
            self._connected = True
            
            # Start heartbeat
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            # Send initial handshake
            await self.send({
                "protocol": "AMP/1.0",
                "type": "handshake",
                "version": "1.0",
                "timestamp": time.time()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def disconnect(self):
        """Close WebSocket connection."""
        self._connected = False
        
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
    
    async def send(self, message: Dict[str, Any]) -> Optional[str]:
        """Send message via WebSocket."""
        if not self._connected or not self.websocket:
            raise RuntimeError("Transport not connected")
        
        try:
            await self.websocket.send(json.dumps(message))
            return message.get("message", {}).get("id")
        except Exception as e:
            self.logger.error(f"Failed to send WebSocket message: {e}")
            return None
    
    async def receive(self) -> AsyncIterator[Dict[str, Any]]:
        """Receive messages from WebSocket."""
        if not self._connected or not self.websocket:
            raise RuntimeError("Transport not connected")
        
        while self._connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                # Filter out heartbeat and handshake messages
                if data.get("type") not in ["heartbeat", "handshake"]:
                    yield data
                    
            except websockets.ConnectionClosed:
                self.logger.info("WebSocket connection closed")
                self._connected = False
                break
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error receiving WebSocket message: {e}")
    
    async def _heartbeat_loop(self):
        """Send periodic heartbeat messages."""
        while self._connected:
            try:
                await self.send({
                    "type": "heartbeat",
                    "timestamp": time.time()
                })
                await asyncio.sleep(self.heartbeat_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Heartbeat error: {e}")
                break
    
    def is_connected(self) -> bool:
        return self._connected and self.websocket is not None


class MockTransport(ITransport):
    """Mock transport for testing."""
    
    def __init__(self):
        self._connected = False
        self.sent_messages = []
        self.received_messages = []
        
    async def connect(self) -> bool:
        self._connected = True
        return True
    
    async def disconnect(self):
        self._connected = False
    
    async def send(self, message: Dict[str, Any]) -> Optional[str]:
        if not self._connected:
            raise RuntimeError("Transport not connected")
        
        self.sent_messages.append(message)
        return message.get("message", {}).get("id")
    
    async def receive(self) -> AsyncIterator[Dict[str, Any]]:
        while self._connected:
            if self.received_messages:
                yield self.received_messages.pop(0)
            else:
                await asyncio.sleep(0.1)
    
    def is_connected(self) -> bool:
        return self._connected
    
    def inject_message(self, message: Dict[str, Any]):
        """Inject a message for testing."""
        self.received_messages.append(message)
