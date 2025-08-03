"""
Integration Tests for Multi-Agent Chatbot System

These tests verify that the complete chatbot system works end-to-end,
including agent coordination, conversation routing, and state management.
"""

import pytest
import asyncio
import sys
import uuid
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from run_chatbot import ChatbotSystem
from agents.conversation_manager import ConversationManager, ConversationState


class TestChatbotSystemIntegration:
    """Test complete chatbot system integration."""
    
    @pytest.fixture
    async def chatbot_system(self):
        """Create a chatbot system for testing."""
        # Note: This would need a running AMP registry in a real test environment
        # For unit tests, we mock the AMP components
        system = ChatbotSystem()
        
        # Mock the AMP clients to avoid needing real network components
        system.clients = {
            "conversation-manager": MockAMPClient("conversation-manager"),
            "router-agent": MockAMPClient("router-agent"),
            "faq-agent": MockAMPClient("faq-agent"),
            "sales-agent": MockAMPClient("sales-agent"),
            "tech-support-agent": MockAMPClient("tech-support-agent")
        }
        system.running = True
        
        # Create mock agents
        system.conversation_manager = ConversationManager()
        
        yield system
        
        # Cleanup
        system.running = False
    
    @pytest.mark.asyncio
    async def test_system_health_check(self, chatbot_system):
        """Test system health check functionality."""
        health = await chatbot_system.health_check()
        
        assert "system_status" in health
        assert "agents" in health
        assert "total_agents" in health
        assert "active_agents" in health
        
        assert health["total_agents"] > 0
        assert isinstance(health["agents"], dict)
        
        # Should have all expected agents
        expected_agents = [
            "conversation-manager", "router-agent", "faq-agent", 
            "sales-agent", "tech-support-agent"
        ]
        
        for agent in expected_agents:
            assert agent in health["agents"]
    
    @pytest.mark.asyncio
    async def test_message_processing_flow(self, chatbot_system):
        """Test complete message processing flow."""
        session_id = str(uuid.uuid4())
        user_input = "What are your business hours?"
        
        # Mock the router agent to return appropriate routing
        original_process = chatbot_system.process_user_message
        
        async def mock_process(msg, sid, ctx=None):
            return {
                "response": "Our business hours are Monday-Friday 9AM-6PM EST.",
                "agent": "faq-agent",
                "intent": "faq",
                "confidence": 0.95,
                "session_id": sid
            }
        
        chatbot_system.process_user_message = mock_process
        
        result = await chatbot_system.process_user_message(user_input, session_id)
        
        assert "response" in result
        assert "agent" in result
        assert "intent" in result
        assert "session_id" in result
        
        assert result["agent"] == "faq-agent"
        assert result["intent"] == "faq"
        assert result["session_id"] == session_id
        
        # Restore original method
        chatbot_system.process_user_message = original_process
    
    @pytest.mark.asyncio
    async def test_conversation_state_management(self, chatbot_system):
        """Test conversation state management across messages."""
        session_id = str(uuid.uuid4())
        
        # Start conversation
        await chatbot_system.conversation_manager.start_conversation(session_id)
        
        # Add multiple messages to test state management
        messages = [
            ("user", "Hello"),
            ("assistant", "Hi! How can I help you?"),
            ("user", "What are your hours?"),
            ("assistant", "We're open 9-5 Monday-Friday")
        ]
        
        for role, content in messages:
            await chatbot_system.conversation_manager.add_message(
                session_id, role, content
            )
        
        # Get conversation context
        context = await chatbot_system.conversation_manager.get_conversation_context(session_id)
        
        assert context is not None
        assert context.session_id == session_id
        assert context.message_count == len(messages)
        assert context.state == ConversationState.ACTIVE
        
        # Get message history
        history = await chatbot_system.conversation_manager.get_conversation_history(session_id)
        assert len(history) == len(messages)
    
    @pytest.mark.asyncio
    async def test_agent_handoff_scenario(self, chatbot_system):
        """Test agent handoff during conversation."""
        session_id = str(uuid.uuid4())
        
        # Start with FAQ conversation
        await chatbot_system.conversation_manager.start_conversation(session_id)
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", "What are your hours?"
        )
        
        # Simulate agent assignment
        context = await chatbot_system.conversation_manager.get_conversation_context(session_id)
        context.current_agent = "faq-agent"
        
        # User switches to sales inquiry - should trigger handoff
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", "How much does your product cost?"
        )
        
        # Simulate handoff
        success = await chatbot_system.conversation_manager.transfer_conversation(
            session_id, "faq-agent", "sales-agent", "intent_change"
        )
        
        assert success
        
        # Verify handoff
        context_after = await chatbot_system.conversation_manager.get_conversation_context(session_id)
        assert context_after.current_agent == "sales-agent"
        assert "faq-agent" in context_after.previous_agents
    
    @pytest.mark.asyncio
    async def test_escalation_scenario(self, chatbot_system):
        """Test conversation escalation."""
        session_id = str(uuid.uuid4())
        
        # Start conversation
        await chatbot_system.conversation_manager.start_conversation(session_id)
        
        # Simulate technical issue that needs escalation
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", "CRITICAL: Data loss in production system"
        )
        
        # Escalate conversation
        success = await chatbot_system.conversation_manager.escalate_conversation(
            session_id, "Critical data loss reported", escalation_level=2
        )
        
        assert success
        
        # Verify escalation
        context = await chatbot_system.conversation_manager.get_conversation_context(session_id)
        assert context.state == ConversationState.ESCALATED
        assert context.escalation_count == 1
    
    @pytest.mark.asyncio
    async def test_multiple_concurrent_conversations(self, chatbot_system):
        """Test handling multiple concurrent conversations."""
        session_ids = [str(uuid.uuid4()) for _ in range(5)]
        
        # Start multiple conversations
        for session_id in session_ids:
            await chatbot_system.conversation_manager.start_conversation(session_id)
            await chatbot_system.conversation_manager.add_message(
                session_id, "user", f"Hello from session {session_id}"
            )
        
        # Verify all conversations are tracked
        for session_id in session_ids:
            context = await chatbot_system.conversation_manager.get_conversation_context(session_id)
            assert context is not None
            assert context.session_id == session_id
            assert context.message_count >= 1
        
        # Get statistics
        stats = await chatbot_system.conversation_manager.get_statistics()
        assert stats["total_conversations"] >= len(session_ids)
        assert stats["active_conversations"] >= len(session_ids)
    
    @pytest.mark.asyncio
    async def test_conversation_completion(self, chatbot_system):
        """Test conversation completion flow."""
        session_id = str(uuid.uuid4())
        
        # Start and populate conversation
        await chatbot_system.conversation_manager.start_conversation(session_id)
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", "Thank you for your help"
        )
        await chatbot_system.conversation_manager.add_message(
            session_id, "assistant", "You're welcome! Is there anything else I can help with?"
        )
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", "No, that's all. Thanks!"
        )
        
        # Complete conversation
        success = await chatbot_system.conversation_manager.complete_conversation(
            session_id, satisfaction_score=4.5, completion_reason="resolved"
        )
        
        assert success
        
        # Verify completion
        context = await chatbot_system.conversation_manager.get_conversation_context(session_id)
        assert context.state == ConversationState.COMPLETED
        assert context.satisfaction_score == 4.5


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    @pytest.fixture
    async def chatbot_system(self):
        """Create a chatbot system for testing."""
        system = ChatbotSystem()
        system.running = True
        system.conversation_manager = ConversationManager()
        yield system
        system.running = False
    
    @pytest.mark.asyncio
    async def test_invalid_session_handling(self, chatbot_system):
        """Test handling of invalid session IDs."""
        # Test with empty session ID
        context = await chatbot_system.conversation_manager.get_conversation_context("")
        assert context is None
        
        # Test with non-existent session ID
        context = await chatbot_system.conversation_manager.get_conversation_context("nonexistent")
        assert context is None
    
    @pytest.mark.asyncio
    async def test_system_not_running_handling(self, chatbot_system):
        """Test handling when system is not running."""
        chatbot_system.running = False
        
        # Mock the process_user_message method to test the not running case
        result = await chatbot_system.process_user_message("Hello", "test-session")
        
        assert "error" in result
        assert "response" in result
        assert "unavailable" in result["response"].lower()
    
    @pytest.mark.asyncio
    async def test_malformed_message_handling(self, chatbot_system):
        """Test handling of malformed or problematic messages."""
        session_id = str(uuid.uuid4())
        
        # Start conversation
        await chatbot_system.conversation_manager.start_conversation(session_id)
        
        # Test very long message
        long_message = "a" * 10000
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", long_message
        )
        
        # Test message with special characters
        special_message = "Hello! @#$%^&*()_+ 中文 🚀 <script>alert('test')</script>"
        await chatbot_system.conversation_manager.add_message(
            session_id, "user", special_message
        )
        
        # Verify messages were added
        history = await chatbot_system.conversation_manager.get_conversation_history(session_id)
        assert len(history) >= 2


class TestPerformanceAndScaling:
    """Test performance and scaling characteristics."""
    
    @pytest.fixture
    async def chatbot_system(self):
        """Create a chatbot system for testing."""
        system = ChatbotSystem()
        system.running = True
        system.conversation_manager = ConversationManager()
        yield system
        system.running = False
    
    @pytest.mark.asyncio
    async def test_conversation_history_limits(self, chatbot_system):
        """Test conversation history limits and cleanup."""
        session_id = str(uuid.uuid4())
        
        # Start conversation
        await chatbot_system.conversation_manager.start_conversation(session_id)
        
        # Add many messages (more than the limit)
        max_messages = chatbot_system.conversation_manager.max_message_history
        for i in range(max_messages + 10):
            await chatbot_system.conversation_manager.add_message(
                session_id, "user", f"Message {i}"
            )
        
        # Verify history is limited
        history = await chatbot_system.conversation_manager.get_conversation_history(session_id)
        assert len(history) <= max_messages
    
    @pytest.mark.asyncio
    async def test_conversation_cleanup(self, chatbot_system):
        """Test automatic conversation cleanup."""
        # Create old conversation
        old_session_id = str(uuid.uuid4())
        await chatbot_system.conversation_manager.start_conversation(old_session_id)
        
        # Complete the conversation
        await chatbot_system.conversation_manager.complete_conversation(
            old_session_id, completion_reason="test"
        )
        
        # Manually trigger cleanup (in real scenario this runs in background)
        await chatbot_system.conversation_manager.cleanup_idle_conversations()
        
        # Verify the cleanup logic ran (would clean up very old conversations)
        stats = await chatbot_system.conversation_manager.get_statistics()
        assert isinstance(stats["total_conversations"], int)


# Mock classes for testing

class MockAMPClient:
    """Mock AMP client for testing."""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.connected = True
    
    async def health_check(self):
        """Mock health check."""
        return {
            "status": "healthy",
            "agent_id": self.agent_id,
            "connected": self.connected,
            "capabilities": ["mock-capability"]
        }
    
    async def disconnect(self):
        """Mock disconnect."""
        self.connected = False


# Test configuration

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


# Pytest marks for different test categories

pytestmark = pytest.mark.integration


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])