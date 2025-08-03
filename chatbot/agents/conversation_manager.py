"""
Conversation Manager for Multi-Agent Chatbot System

Manages conversation state, context, and agent coordination.
Handles session management and conversation history persistence.
"""

import asyncio
import logging
import json
from typing import Dict, Any, List, Optional, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

# AMP imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


class ConversationState(Enum):
    """Conversation state types."""
    ACTIVE = "active"
    IDLE = "idle"
    TRANSFERRED = "transferred"
    COMPLETED = "completed"
    ESCALATED = "escalated"


@dataclass
class ConversationContext:
    """Conversation context information."""
    session_id: str
    user_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    state: ConversationState = ConversationState.ACTIVE
    current_agent: Optional[str] = None
    previous_agents: List[str] = field(default_factory=list)
    intent_history: List[str] = field(default_factory=list)
    message_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    escalation_count: int = 0
    satisfaction_score: Optional[float] = None


@dataclass
class Message:
    """Conversation message."""
    id: str
    session_id: str
    timestamp: datetime
    role: str  # user, assistant, system
    content: str
    agent: Optional[str] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ConversationManager:
    """Manages conversation state and coordination between agents."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.ConversationManager")
        
        # Storage
        self.conversations: Dict[str, ConversationContext] = {}
        self.messages: Dict[str, List[Message]] = {}  # session_id -> messages
        
        # Configuration
        self.max_idle_time = timedelta(minutes=self.config.get("max_idle_minutes", 30))
        self.max_message_history = self.config.get("max_message_history", 100)
        self.auto_cleanup_enabled = self.config.get("auto_cleanup", True)
        
        # Statistics
        self.stats = {
            "total_conversations": 0,
            "active_conversations": 0,
            "completed_conversations": 0,
            "escalated_conversations": 0,
            "avg_session_length": 0.0,
            "agent_handoffs": 0
        }
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        
    async def start_conversation(self, session_id: str, user_id: Optional[str] = None,
                                initial_context: Optional[Dict[str, Any]] = None) -> ConversationContext:
        """Start a new conversation."""
        if session_id in self.conversations:
            # Resume existing conversation
            context = self.conversations[session_id]
            context.last_activity = datetime.now()
            context.state = ConversationState.ACTIVE
            self.logger.info(f"Resumed conversation {session_id}")
        else:
            # Create new conversation
            context = ConversationContext(
                session_id=session_id,
                user_id=user_id,
                metadata=initial_context or {}
            )
            self.conversations[session_id] = context
            self.messages[session_id] = []
            self.stats["total_conversations"] += 1
            self.logger.info(f"Started new conversation {session_id}")
        
        self._update_active_count()
        return context
    
    async def add_message(self, session_id: str, role: str, content: str,
                         agent: Optional[str] = None, intent: Optional[str] = None,
                         confidence: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Message:
        """Add a message to the conversation."""
        
        # Ensure conversation exists
        if session_id not in self.conversations:
            await self.start_conversation(session_id)
        
        # Create message
        message = Message(
            id=f"{session_id}-{len(self.messages[session_id])}",
            session_id=session_id,
            timestamp=datetime.now(),
            role=role,
            content=content,
            agent=agent,
            intent=intent,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # Add to conversation
        self.messages[session_id].append(message)
        
        # Update conversation context
        context = self.conversations[session_id]
        context.last_activity = datetime.now()
        context.message_count += 1
        
        if intent and intent not in context.intent_history:
            context.intent_history.append(intent)
        
        # Trim message history if too long
        if len(self.messages[session_id]) > self.max_message_history:
            self.messages[session_id] = self.messages[session_id][-self.max_message_history:]
        
        self.logger.debug(f"Added message to conversation {session_id}: {role} - {content[:50]}...")
        return message
    
    async def transfer_conversation(self, session_id: str, from_agent: str, to_agent: str,
                                   reason: str = "intent_change") -> bool:
        """Transfer conversation between agents."""
        if session_id not in self.conversations:
            return False
        
        context = self.conversations[session_id]
        
        # Record agent change
        if context.current_agent != from_agent:
            self.logger.warning(f"Agent mismatch in transfer: expected {context.current_agent}, got {from_agent}")
        
        if context.current_agent and context.current_agent not in context.previous_agents:
            context.previous_agents.append(context.current_agent)
        
        context.current_agent = to_agent
        context.state = ConversationState.TRANSFERRED
        self.stats["agent_handoffs"] += 1
        
        # Add system message about transfer
        await self.add_message(
            session_id,
            "system",
            f"Conversation transferred from {from_agent} to {to_agent}: {reason}",
            agent="conversation-manager",
            metadata={"transfer_reason": reason, "from_agent": from_agent, "to_agent": to_agent}
        )
        
        self.logger.info(f"Transferred conversation {session_id} from {from_agent} to {to_agent}")
        return True
    
    async def escalate_conversation(self, session_id: str, escalation_reason: str,
                                   escalation_level: int = 1) -> bool:
        """Escalate conversation to higher support level."""
        if session_id not in self.conversations:
            return False
        
        context = self.conversations[session_id]
        context.state = ConversationState.ESCALATED
        context.escalation_count += 1
        self.stats["escalated_conversations"] += 1
        
        # Add escalation message
        await self.add_message(
            session_id,
            "system",
            f"Conversation escalated (level {escalation_level}): {escalation_reason}",
            agent="conversation-manager",
            metadata={
                "escalation_level": escalation_level,
                "escalation_reason": escalation_reason,
                "escalation_count": context.escalation_count
            }
        )
        
        self.logger.info(f"Escalated conversation {session_id}: {escalation_reason}")
        return True
    
    async def complete_conversation(self, session_id: str, satisfaction_score: Optional[float] = None,
                                   completion_reason: str = "resolved") -> bool:
        """Mark conversation as completed."""
        if session_id not in self.conversations:
            return False
        
        context = self.conversations[session_id]
        context.state = ConversationState.COMPLETED
        context.satisfaction_score = satisfaction_score
        self.stats["completed_conversations"] += 1
        
        # Calculate session length
        session_length = (datetime.now() - context.created_at).total_seconds() / 60  # minutes
        self._update_avg_session_length(session_length)
        
        # Add completion message
        await self.add_message(
            session_id,
            "system",
            f"Conversation completed: {completion_reason}",
            agent="conversation-manager",
            metadata={
                "completion_reason": completion_reason,
                "satisfaction_score": satisfaction_score,
                "session_length_minutes": session_length
            }
        )
        
        self.logger.info(f"Completed conversation {session_id}: {completion_reason}")
        self._update_active_count()
        return True
    
    async def get_conversation_context(self, session_id: str) -> Optional[ConversationContext]:
        """Get conversation context."""
        return self.conversations.get(session_id)
    
    async def get_conversation_history(self, session_id: str, limit: Optional[int] = None) -> List[Message]:
        """Get conversation message history."""
        messages = self.messages.get(session_id, [])
        if limit:
            return messages[-limit:]
        return messages
    
    async def get_recent_context(self, session_id: str, message_count: int = 5) -> Dict[str, Any]:
        """Get recent conversation context for agents."""
        if session_id not in self.conversations:
            return {}
        
        context = self.conversations[session_id]
        recent_messages = await self.get_conversation_history(session_id, message_count)
        
        return {
            "session_id": session_id,
            "current_agent": context.current_agent,
            "intent_history": context.intent_history,
            "message_count": context.message_count,
            "escalation_count": context.escalation_count,
            "conversation_state": context.state.value,
            "recent_messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "agent": msg.agent,
                    "intent": msg.intent
                }
                for msg in recent_messages
            ],
            "metadata": context.metadata
        }
    
    async def search_conversations(self, filters: Dict[str, Any]) -> List[ConversationContext]:
        """Search conversations by filters."""
        results = []
        
        for context in self.conversations.values():
            match = True
            
            # Filter by state
            if "state" in filters and context.state.value != filters["state"]:
                match = False
            
            # Filter by agent
            if "current_agent" in filters and context.current_agent != filters["current_agent"]:
                match = False
            
            # Filter by date range
            if "date_from" in filters:
                date_from = datetime.fromisoformat(filters["date_from"])
                if context.created_at < date_from:
                    match = False
            
            if "date_to" in filters:
                date_to = datetime.fromisoformat(filters["date_to"])
                if context.created_at > date_to:
                    match = False
            
            # Filter by escalation
            if "escalated" in filters and (context.escalation_count > 0) != filters["escalated"]:
                match = False
            
            if match:
                results.append(context)
        
        return results
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get conversation statistics."""
        # Update active count
        self._update_active_count()
        
        # Add real-time stats
        current_stats = self.stats.copy()
        current_stats.update({
            "idle_conversations": sum(
                1 for c in self.conversations.values() 
                if c.state == ConversationState.IDLE
            ),
            "transferred_conversations": sum(
                1 for c in self.conversations.values()
                if len(c.previous_agents) > 0
            ),
            "avg_messages_per_conversation": (
                sum(c.message_count for c in self.conversations.values()) / 
                len(self.conversations) if self.conversations else 0
            )
        })
        
        return current_stats
    
    def _update_active_count(self):
        """Update active conversation count."""
        self.stats["active_conversations"] = sum(
            1 for c in self.conversations.values()
            if c.state == ConversationState.ACTIVE
        )
    
    def _update_avg_session_length(self, new_session_length: float):
        """Update average session length."""
        completed = self.stats["completed_conversations"]
        current_avg = self.stats["avg_session_length"]
        
        # Calculate new average
        self.stats["avg_session_length"] = (
            (current_avg * (completed - 1) + new_session_length) / completed
            if completed > 0 else new_session_length
        )
    
    async def cleanup_idle_conversations(self):
        """Clean up idle conversations."""
        if not self.auto_cleanup_enabled:
            return
        
        cutoff_time = datetime.now() - self.max_idle_time
        idle_sessions = []
        
        for session_id, context in self.conversations.items():
            if (context.state == ConversationState.ACTIVE and 
                context.last_activity < cutoff_time):
                
                context.state = ConversationState.IDLE
                idle_sessions.append(session_id)
        
        if idle_sessions:
            self.logger.info(f"Marked {len(idle_sessions)} conversations as idle")
        
        # Remove very old completed conversations
        old_cutoff = datetime.now() - timedelta(days=7)
        to_remove = [
            session_id for session_id, context in self.conversations.items()
            if (context.state in [ConversationState.COMPLETED, ConversationState.IDLE] and
                context.last_activity < old_cutoff)
        ]
        
        for session_id in to_remove:
            del self.conversations[session_id]
            if session_id in self.messages:
                del self.messages[session_id]
        
        if to_remove:
            self.logger.info(f"Cleaned up {len(to_remove)} old conversations")
    
    async def start_cleanup_task(self):
        """Start background cleanup task."""
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Started conversation cleanup task")
    
    async def stop_cleanup_task(self):
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            self.logger.info("Stopped conversation cleanup task")
    
    async def _cleanup_loop(self):
        """Background cleanup loop."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                await self.cleanup_idle_conversations()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    # AMP capability handlers
    async def handle_start_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Start conversation capability."""
        session_id = parameters.get("session_id", "")
        user_id = parameters.get("user_id")
        initial_context = parameters.get("context", {})
        
        context = await self.start_conversation(session_id, user_id, initial_context)
        
        return {
            "success": True,
            "session_id": session_id,
            "context": {
                "created_at": context.created_at.isoformat(),
                "state": context.state.value,
                "current_agent": context.current_agent
            }
        }
    
    async def handle_add_message(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Add message capability."""
        session_id = parameters.get("session_id", "")
        role = parameters.get("role", "user")
        content = parameters.get("content", "")
        agent = parameters.get("agent")
        intent = parameters.get("intent")
        confidence = parameters.get("confidence")
        metadata = parameters.get("metadata", {})
        
        message = await self.add_message(
            session_id, role, content, agent, intent, confidence, metadata
        )
        
        return {
            "success": True,
            "message_id": message.id,
            "timestamp": message.timestamp.isoformat()
        }
    
    async def handle_get_context(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get conversation context capability."""
        session_id = parameters.get("session_id", "")
        message_count = parameters.get("message_count", 5)
        
        context = await self.get_recent_context(session_id, message_count)
        
        return {
            "found": bool(context),
            "context": context
        }
    
    async def handle_transfer_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Transfer conversation capability."""
        session_id = parameters.get("session_id", "")
        from_agent = parameters.get("from_agent", "")
        to_agent = parameters.get("to_agent", "")
        reason = parameters.get("reason", "intent_change")
        
        success = await self.transfer_conversation(session_id, from_agent, to_agent, reason)
        
        return {
            "success": success,
            "session_id": session_id,
            "transferred_to": to_agent if success else None
        }
    
    async def handle_get_statistics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics capability."""
        stats = await self.get_statistics()
        return {"statistics": stats}
    
    async def start_amp_agent(self, agent_id: str = "conversation-manager",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Conversation Manager")
            .with_framework("custom")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "start-conversation",
                self.handle_start_conversation,
                "Start a new conversation session",
                "conversation"
            )
            .add_capability(
                "add-message",
                self.handle_add_message,
                "Add a message to conversation",
                "conversation"
            )
            .add_capability(
                "get-context",
                self.handle_get_context,
                "Get conversation context and history",
                "conversation"
            )
            .add_capability(
                "transfer-conversation",
                self.handle_transfer_conversation,
                "Transfer conversation between agents",
                "conversation"
            )
            .add_capability(
                "get-statistics",
                self.handle_get_statistics,
                "Get conversation statistics",
                "analytics"
            )
            .build()
        )
        
        # Start cleanup task
        await self.start_cleanup_task()
        
        return self.amp_client


async def main():
    """Main function for testing the conversation manager."""
    logging.basicConfig(level=logging.INFO)
    
    # Create conversation manager
    manager = ConversationManager()
    
    # Start AMP agent
    client = await manager.start_amp_agent()
    
    try:
        print("Conversation Manager started. Testing conversation management...")
        
        # Test conversation flow
        session_id = "test-session-001"
        
        # Start conversation
        await manager.start_conversation(session_id, user_id="user123")
        print(f"Started conversation: {session_id}")
        
        # Add messages
        await manager.add_message(session_id, "user", "Hello, I need help", intent="general")
        await manager.add_message(session_id, "assistant", "Hi! I'm here to help.", agent="router-agent")
        await manager.add_message(session_id, "user", "What are your prices?", intent="sales")
        
        # Transfer conversation
        await manager.transfer_conversation(session_id, "router-agent", "sales-agent", "sales inquiry")
        
        # Get context
        context = await manager.get_recent_context(session_id)
        print(f"Conversation context: {context['message_count']} messages, current agent: {context['current_agent']}")
        
        # Get statistics
        stats = await manager.get_statistics()
        print(f"Statistics: {stats}")
        
        print("Conversation Manager is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await manager.stop_cleanup_task()
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())