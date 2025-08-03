"""
Tests for Router Agent Intent Detection and Conversation Routing

These tests verify that the router agent correctly identifies user intents
and routes conversations to the appropriate specialized agents.
"""

import pytest
import asyncio
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from agents.router_agent import RouterAgent, IntentResult


class TestIntentDetection:
    """Test intent detection functionality."""
    
    @pytest.fixture
    async def router_agent(self):
        """Create a router agent for testing."""
        agent = RouterAgent()
        yield agent
        # Cleanup if needed
    
    @pytest.mark.asyncio
    async def test_faq_intent_detection(self, router_agent):
        """Test detection of FAQ-related intents."""
        test_cases = [
            "What are your business hours?",
            "How can I contact customer support?",
            "Where is your company located?",
            "What is your return policy?"
        ]
        
        for message in test_cases:
            result = await router_agent.detect_intent(message)
            
            assert isinstance(result, IntentResult)
            assert result.intent == "faq"
            assert result.confidence > 0.5
            assert result.routing_suggestion == "faq-agent"
    
    @pytest.mark.asyncio
    async def test_sales_intent_detection(self, router_agent):
        """Test detection of sales-related intents."""
        test_cases = [
            "How much does your product cost?",
            "I want to buy your software",
            "Can I get a demo?",
            "What are your pricing plans?",
            "Do you offer discounts?"
        ]
        
        for message in test_cases:
            result = await router_agent.detect_intent(message)
            
            assert isinstance(result, IntentResult)
            assert result.intent == "sales"
            assert result.confidence > 0.5
            assert result.routing_suggestion == "sales-agent"
    
    @pytest.mark.asyncio
    async def test_technical_intent_detection(self, router_agent):
        """Test detection of technical support intents."""
        test_cases = [
            "I'm getting an error when I try to login",
            "The application is not working",
            "There's a bug in the software",
            "I can't access my account",
            "The system is very slow"
        ]
        
        for message in test_cases:
            result = await router_agent.detect_intent(message)
            
            assert isinstance(result, IntentResult)
            assert result.intent == "technical"
            assert result.confidence > 0.5
            assert result.routing_suggestion == "tech-support-agent"
    
    @pytest.mark.asyncio
    async def test_general_intent_detection(self, router_agent):
        """Test detection of general conversation intents."""
        test_cases = [
            "Hello there",
            "Hi, can you help me?",
            "Good morning",
            "Thank you for your assistance"
        ]
        
        for message in test_cases:
            result = await router_agent.detect_intent(message)
            
            assert isinstance(result, IntentResult)
            assert result.intent == "general"
            assert result.confidence > 0.3  # Lower threshold for general intents
    
    @pytest.mark.asyncio
    async def test_fallback_intent_detection(self, router_agent):
        """Test fallback intent detection when LLM fails."""
        # Create a test case that might cause LLM to fail
        result = await router_agent._fallback_intent_detection("Some random text")
        
        assert isinstance(result, IntentResult)
        assert result.intent in ["sales", "technical", "faq", "general"]
        assert 0.0 <= result.confidence <= 1.0
        assert result.routing_suggestion in ["sales-agent", "tech-support-agent", "faq-agent"]
    
    @pytest.mark.asyncio
    async def test_context_aware_routing(self, router_agent):
        """Test that context affects routing decisions."""
        # Test with previous conversation context
        context = {
            "previous_intent": "sales",
            "previous_agent": "sales-agent"
        }
        
        history = [
            {"role": "user", "content": "I'm interested in your product"},
            {"role": "assistant", "content": "Great! Let me help you with that"}
        ]
        
        # This message could be ambiguous but context should help
        result = await router_agent.detect_intent(
            "Can you tell me more about the features?", 
            context=context, 
            history=history
        )
        
        assert isinstance(result, IntentResult)
        assert result.confidence > 0.0
        assert result.routing_suggestion is not None
    
    @pytest.mark.asyncio
    async def test_multiple_intent_keywords(self, router_agent):
        """Test messages with multiple intent keywords."""
        # Message that could match multiple intents
        message = "I want to buy your product but I'm having login issues"
        
        result = await router_agent.detect_intent(message)
        
        assert isinstance(result, IntentResult)
        # Should pick one primary intent
        assert result.intent in ["sales", "technical"]
        assert result.confidence > 0.0
        # Should have secondary intents or context requirements
        assert isinstance(result.context_requirements, list)


class TestConversationRouting:
    """Test conversation routing functionality."""
    
    @pytest.fixture
    async def router_agent(self):
        """Create a router agent for testing."""
        agent = RouterAgent()
        # Mock the AMP client since we're testing in isolation
        agent.amp_client = MockAMPClient()
        yield agent
    
    @pytest.mark.asyncio
    async def test_conversation_routing_flow(self, router_agent):
        """Test complete conversation routing flow."""
        session_id = "test-session-001"
        user_input = "What are your business hours?"
        
        # This would normally require a running AMP system, so we'll test the logic parts
        # In a real test environment, you'd have the full system running
        
        # Test intent detection part
        intent_result = await router_agent.detect_intent(user_input)
        
        assert intent_result.intent == "faq"
        assert intent_result.routing_suggestion == "faq-agent"
        
        # Test conversation context management
        assert session_id not in router_agent.conversation_contexts
        
        # After routing (simulated), context should be created
        router_agent.conversation_contexts[session_id] = {
            "history": [],
            "current_agent": "faq-agent",
            "context": {}
        }
        
        assert session_id in router_agent.conversation_contexts
        assert router_agent.conversation_contexts[session_id]["current_agent"] == "faq-agent"
    
    @pytest.mark.asyncio
    async def test_agent_handoff_detection(self, router_agent):
        """Test detection of when agent handoff is needed."""
        session_id = "test-session-002"
        
        # Setup existing conversation with FAQ agent
        router_agent.conversation_contexts[session_id] = {
            "history": [
                {"role": "user", "content": "What are your hours?"},
                {"role": "assistant", "content": "We're open 9-5 Monday-Friday"}
            ],
            "current_agent": "faq-agent",
            "context": {}
        }
        
        # User switches to sales inquiry
        sales_message = "How much does your product cost?"
        intent_result = await router_agent.detect_intent(sales_message)
        
        assert intent_result.intent == "sales"
        assert intent_result.routing_suggestion == "sales-agent"
        
        # Current agent is different from suggested agent - handoff needed
        current_agent = router_agent.conversation_contexts[session_id]["current_agent"]
        assert current_agent != intent_result.routing_suggestion
    
    @pytest.mark.asyncio
    async def test_conversation_history_management(self, router_agent):
        """Test conversation history management."""
        session_id = "test-session-003"
        
        # Start conversation
        router_agent.conversation_contexts[session_id] = {
            "history": [],
            "current_agent": None,
            "context": {}
        }
        
        # Add multiple messages
        messages = [
            "Hello",
            "What are your hours?",
            "How much does it cost?",
            "I need technical support"
        ]
        
        for i, message in enumerate(messages):
            # Simulate adding to history
            router_agent.conversation_contexts[session_id]["history"].append({
                "role": "user",
                "content": message,
                "timestamp": i,
                "intent": await router_agent.detect_intent(message)
            })
        
        history = router_agent.conversation_contexts[session_id]["history"]
        assert len(history) == len(messages)
        
        # Test history truncation (should keep last 10 messages)
        for i in range(15):  # Add more messages
            router_agent.conversation_contexts[session_id]["history"].append({
                "role": "user",
                "content": f"Message {i}",
                "timestamp": i + len(messages)
            })
        
        # Should only keep the last 10 messages
        history_after = router_agent.conversation_contexts[session_id]["history"]
        assert len(history_after) <= 10


class TestAMPCapabilities:
    """Test AMP capability handlers."""
    
    @pytest.fixture
    async def router_agent(self):
        """Create a router agent for testing."""
        agent = RouterAgent()
        yield agent
    
    @pytest.mark.asyncio
    async def test_handle_intent_detection_capability(self, router_agent):
        """Test the intent detection AMP capability handler."""
        parameters = {
            "user_input": "How much does your product cost?",
            "context": {},
            "history": []
        }
        
        result = await router_agent.handle_intent_detection(parameters)
        
        assert "intent" in result
        assert "confidence" in result
        assert "routing_suggestion" in result
        assert result["intent"] == "sales"
        assert result["routing_suggestion"] == "sales-agent"
    
    @pytest.mark.asyncio
    async def test_get_conversation_context_capability(self, router_agent):
        """Test the conversation context retrieval capability."""
        session_id = "test-session-004"
        
        # Test non-existent session
        result = await router_agent.get_conversation_context({"session_id": session_id})
        assert not result["exists"]
        
        # Add session
        router_agent.conversation_contexts[session_id] = {
            "history": [{"role": "user", "content": "Hello"}],
            "current_agent": "faq-agent",
            "context": {"test": "data"}
        }
        
        # Test existing session
        result = await router_agent.get_conversation_context({"session_id": session_id})
        assert result["exists"]
        assert "context" in result


# Mock classes for testing

class MockAMPClient:
    """Mock AMP client for testing."""
    
    async def invoke_capability(self, target_agent, capability, parameters):
        """Mock capability invocation."""
        return {
            "result": {
                "response": f"Mock response from {target_agent}",
                "agent": target_agent
            }
        }
    
    async def emit_event(self, event_type, data, target_agent=None):
        """Mock event emission."""
        pass


# Test configuration and fixtures

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])