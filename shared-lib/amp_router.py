"""
Message routing for AMP agents.
"""

import asyncio
import logging
from typing import Dict, Any, List, Callable, Optional
from dataclasses import dataclass, field
import fnmatch


@dataclass
class RouteHandler:
    """Handler for a specific message route."""
    pattern: str  # e.g., "request/text-analysis/*"
    handler: Callable[[Dict[str, Any]], Any]
    filters: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # Lower number = higher priority
    async_handler: bool = True


class MessageRouter:
    """Route incoming messages to appropriate handlers."""
    
    def __init__(self):
        self.routes: List[RouteHandler] = []
        self.default_handler: Optional[Callable] = None
        self.middleware: List[Callable] = []
        self.logger = logging.getLogger(f"{__name__}.MessageRouter")
        
    def add_route(self, pattern: str, handler: Callable, priority: int = 5, **filters):
        """Add a message route.
        
        Args:
            pattern: Route pattern with wildcards (e.g., "request/*/capability")
            handler: Function to handle matching messages
            priority: Route priority (lower = higher priority)
            **filters: Additional filters for message matching
        """
        route = RouteHandler(
            pattern=pattern,
            handler=handler,
            filters=filters,
            priority=priority,
            async_handler=asyncio.iscoroutinefunction(handler)
        )
        
        # Insert route in priority order
        inserted = False
        for i, existing_route in enumerate(self.routes):
            if route.priority < existing_route.priority:
                self.routes.insert(i, route)
                inserted = True
                break
        
        if not inserted:
            self.routes.append(route)
        
        self.logger.debug(f"Added route: {pattern} (priority: {priority})")
    
    def remove_route(self, pattern: str, handler: Callable = None):
        """Remove a route."""
        self.routes = [
            route for route in self.routes
            if not (route.pattern == pattern and 
                   (handler is None or route.handler == handler))
        ]
    
    def add_middleware(self, middleware: Callable):
        """Add middleware function that processes all messages."""
        self.middleware.append(middleware)
    
    def set_default_handler(self, handler: Callable):
        """Set default handler for unmatched messages."""
        self.default_handler = handler
    
    async def route_message(self, message: Dict[str, Any]) -> Any:
        """Route message to appropriate handler."""
        try:
            # Apply middleware
            for middleware in self.middleware:
                if asyncio.iscoroutinefunction(middleware):
                    message = await middleware(message)
                else:
                    message = middleware(message)
                
                if message is None:
                    return  # Middleware consumed the message
            
            # Extract routing key
            route_key = self._extract_route_key(message)
            
            # Find matching route
            for route in self.routes:
                if self._matches_pattern(route_key, route.pattern):
                    if self._matches_filters(message, route.filters):
                        return await self._call_handler(route, message)
            
            # Use default handler
            if self.default_handler:
                if asyncio.iscoroutinefunction(self.default_handler):
                    return await self.default_handler(message)
                else:
                    return self.default_handler(message)
            
            self.logger.warning(f"No handler for message: {route_key}")
            
        except Exception as e:
            self.logger.error(f"Error routing message: {e}")
            raise
    
    def _extract_route_key(self, message: Dict[str, Any]) -> str:
        """Extract routing key from message."""
        msg_data = message.get("message", {})
        msg_type = msg_data.get("type", "unknown")
        
        # Build route key based on message type
        if msg_type == "request":
            capability = msg_data.get("payload", {}).get("capability")
            if capability:
                return f"request/{capability}"
            else:
                return "request"
        elif msg_type == "response":
            return "response"
        elif msg_type == "event":
            event_type = msg_data.get("payload", {}).get("event_type")
            if event_type:
                return f"event/{event_type}"
            else:
                return "event"
        elif msg_type == "error":
            return "error"
        else:
            return msg_type
    
    def _matches_pattern(self, route_key: str, pattern: str) -> bool:
        """Check if route key matches pattern.
        
        Supports Unix shell-style wildcards:
        * matches everything
        ? matches any single character
        [seq] matches any character in seq
        [!seq] matches any char not in seq
        """
        return fnmatch.fnmatch(route_key, pattern)
    
    def _matches_filters(self, message: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if message matches all filters."""
        msg_data = message.get("message", {})
        
        for key, value in filters.items():
            # Support nested key access with dot notation
            if "." in key:
                current = msg_data
                for part in key.split("."):
                    if isinstance(current, dict) and part in current:
                        current = current[part]
                    else:
                        return False
                
                if current != value:
                    return False
            else:
                # Direct key access
                if key not in msg_data or msg_data[key] != value:
                    return False
        
        return True
    
    async def _call_handler(self, route: RouteHandler, message: Dict[str, Any]) -> Any:
        """Call route handler (async or sync)."""
        try:
            if route.async_handler:
                return await route.handler(message)
            else:
                return route.handler(message)
        except Exception as e:
            self.logger.error(f"Handler error for pattern {route.pattern}: {e}")
            raise
    
    def get_routes(self) -> List[Dict[str, Any]]:
        """Get all registered routes for debugging."""
        return [
            {
                "pattern": route.pattern,
                "handler": route.handler.__name__,
                "priority": route.priority,
                "filters": route.filters
            }
            for route in self.routes
        ]


class ConditionalRouter(MessageRouter):
    """Router that supports conditional routing logic."""
    
    def __init__(self):
        super().__init__()
        self.conditions: List[Callable[[Dict[str, Any]], bool]] = []
    
    def add_condition(self, condition: Callable[[Dict[str, Any]], bool]):
        """Add a global condition that must be met for any routing."""
        self.conditions.append(condition)
    
    async def route_message(self, message: Dict[str, Any]) -> Any:
        """Route message with conditional checks."""
        # Check global conditions
        for condition in self.conditions:
            if asyncio.iscoroutinefunction(condition):
                if not await condition(message):
                    self.logger.debug("Message failed global condition check")
                    return
            else:
                if not condition(message):
                    self.logger.debug("Message failed global condition check")
                    return
        
        # Proceed with normal routing
        return await super().route_message(message)


class LoadBalancingRouter(MessageRouter):
    """Router that supports load balancing across multiple handlers."""
    
    def __init__(self):
        super().__init__()
        self.handler_pools: Dict[str, List[Callable]] = {}
        self.round_robin_index: Dict[str, int] = {}
    
    def add_handler_pool(self, pattern: str, handlers: List[Callable], 
                        strategy: str = "round_robin", **filters):
        """Add a pool of handlers for load balancing.
        
        Args:
            pattern: Route pattern
            handlers: List of handler functions
            strategy: Load balancing strategy ("round_robin", "random")
            **filters: Additional filters
        """
        pool_key = f"{pattern}:{hash(tuple(filters.items()))}"
        self.handler_pools[pool_key] = handlers
        self.round_robin_index[pool_key] = 0
        
        # Create a wrapper handler that selects from the pool
        async def pool_handler(message: Dict[str, Any]):
            if strategy == "round_robin":
                index = self.round_robin_index[pool_key]
                handler = handlers[index]
                self.round_robin_index[pool_key] = (index + 1) % len(handlers)
            elif strategy == "random":
                import random
                handler = random.choice(handlers)
            else:
                handler = handlers[0]  # Default to first
            
            if asyncio.iscoroutinefunction(handler):
                return await handler(message)
            else:
                return handler(message)
        
        self.add_route(pattern, pool_handler, **filters)
    
    def get_pool_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for handler pools."""
        return {
            pool_key: {
                "handler_count": len(handlers),
                "current_index": self.round_robin_index.get(pool_key, 0)
            }
            for pool_key, handlers in self.handler_pools.items()
        }
