"""
Tests for Individual Agent Functionality

Tests the functionality of individual agents (FAQ, Sales, Tech Support)
to ensure they work correctly in isolation.
"""

import pytest
import asyncio
import sys
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from agents.faq_agent import FAQAgent
from agents.sales_agent import SalesAgent, ProductRecommendation
from agents.tech_support_agent import TechSupportAgent, TechnicalDiagnosis


class TestFAQAgent:
    """Test FAQ Agent functionality."""
    
    @pytest.fixture
    async def faq_agent(self):
        """Create an FAQ agent for testing."""
        agent = FAQAgent()
        yield agent
    
    @pytest.mark.asyncio
    async def test_knowledge_base_search(self, faq_agent):
        """Test knowledge base search functionality."""
        # Test search for business hours
        results = await faq_agent.search_knowledge_base("business hours")
        
        assert len(results) > 0
        assert any("hours" in result["question"].lower() for result in results)
        
        # Test search for contact information
        results = await faq_agent.search_knowledge_base("contact support")
        
        assert len(results) > 0
        assert any("contact" in result["question"].lower() for result in results)
    
    @pytest.mark.asyncio
    async def test_response_generation(self, faq_agent):
        """Test FAQ response generation."""
        # Test with a question that should have a good match
        response = await faq_agent.generate_response("What are your business hours?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "hours" in response.lower() or "open" in response.lower()
    
    @pytest.mark.asyncio
    async def test_conversation_handler(self, faq_agent):
        """Test the conversation handler capability."""
        parameters = {
            "user_input": "How can I contact support?",
            "session_id": "test-session",
            "context": {}
        }
        
        result = await faq_agent.handle_conversation(parameters)
        
        assert "response" in result
        assert "agent" in result
        assert result["agent"] == "faq-agent"
        assert "related_questions" in result
        assert "confidence" in result
        assert isinstance(result["related_questions"], list)
    
    @pytest.mark.asyncio
    async def test_faq_search_capability(self, faq_agent):
        """Test the FAQ search capability."""
        parameters = {
            "query": "return policy",
            "max_results": 3
        }
        
        result = await faq_agent.search_faq(parameters)
        
        assert "results" in result
        assert "query" in result
        assert "total_results" in result
        assert result["query"] == "return policy"
        assert len(result["results"]) <= 3
        assert isinstance(result["results"], list)
    
    @pytest.mark.asyncio
    async def test_empty_query_handling(self, faq_agent):
        """Test handling of empty or invalid queries."""
        # Test empty query
        response = await faq_agent.generate_response("")
        assert isinstance(response, str)
        assert len(response) > 0
        
        # Test very short query
        response = await faq_agent.generate_response("a")
        assert isinstance(response, str)
        assert len(response) > 0


class TestSalesAgent:
    """Test Sales Agent functionality."""
    
    @pytest.fixture
    async def sales_agent(self):
        """Create a sales agent for testing."""
        agent = SalesAgent()
        yield agent
    
    @pytest.mark.asyncio
    async def test_product_recommendations(self, sales_agent):
        """Test product recommendation functionality."""
        # Test recommendation for small business
        recommendations = await sales_agent.recommend_products("small startup")
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        for rec in recommendations:
            assert isinstance(rec, ProductRecommendation)
            assert hasattr(rec, 'product_name')
            assert hasattr(rec, 'price')
            assert hasattr(rec, 'features')
            assert hasattr(rec, 'suitability_score')
    
    @pytest.mark.asyncio
    async def test_lead_qualification(self, sales_agent):
        """Test lead qualification functionality."""
        conversation_history = [
            {"role": "user", "content": "I'm interested in your enterprise plan"},
            {"role": "assistant", "content": "Great! Tell me about your needs"},
            {"role": "user", "content": "We need it for 100 users and have a budget of $10k/month"}
        ]
        
        qualification = await sales_agent.qualify_lead("test-session", conversation_history)
        
        assert hasattr(qualification, 'lead_score')
        assert hasattr(qualification, 'budget_range')
        assert hasattr(qualification, 'timeline')
        assert hasattr(qualification, 'decision_maker')
        assert 0 <= qualification.lead_score <= 100
    
    @pytest.mark.asyncio
    async def test_conversation_handler(self, sales_agent):
        """Test sales conversation handler."""
        parameters = {
            "user_input": "How much does your product cost?",
            "session_id": "test-sales-session",
            "context": {},
            "history": []
        }
        
        result = await sales_agent.handle_conversation(parameters)
        
        assert "response" in result
        assert "agent" in result
        assert result["agent"] == "sales-agent"
        assert "recommendations" in result
        assert "next_steps" in result
        assert isinstance(result["recommendations"], list)
        assert isinstance(result["next_steps"], list)
    
    @pytest.mark.asyncio
    async def test_demo_request_handling(self, sales_agent):
        """Test demo request handling."""
        response = await sales_agent._handle_demo_request(
            "Can I get a demo of your software?", {}
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert "demo" in response.lower()
    
    @pytest.mark.asyncio
    async def test_pricing_inquiry_handling(self, sales_agent):
        """Test pricing inquiry handling."""
        response = await sales_agent._handle_pricing_inquiry(
            "What are your prices?", {}
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
        assert any(word in response.lower() for word in ["price", "cost", "plan"])
    
    @pytest.mark.asyncio
    async def test_lead_management(self, sales_agent):
        """Test lead information management."""
        session_id = "test-lead-session"
        
        # Test getting non-existent lead
        result = await sales_agent.get_lead_info({"session_id": session_id})
        assert not result["lead_exists"]
        
        # Create a lead by handling a conversation
        await sales_agent.handle_conversation({
            "user_input": "I'm interested in your product",
            "session_id": session_id,
            "context": {},
            "history": []
        })
        
        # Test getting existing lead
        result = await sales_agent.get_lead_info({"session_id": session_id})
        assert result["lead_exists"]
        assert "lead_info" in result


class TestTechSupportAgent:
    """Test Technical Support Agent functionality."""
    
    @pytest.fixture
    async def tech_agent(self):
        """Create a tech support agent for testing."""
        agent = TechSupportAgent()
        yield agent
    
    @pytest.mark.asyncio
    async def test_issue_diagnosis(self, tech_agent):
        """Test technical issue diagnosis."""
        # Test login issue
        diagnosis = await tech_agent.diagnose_issue("I can't log into my account")
        
        assert isinstance(diagnosis, TechnicalDiagnosis)
        assert diagnosis.issue_category in ["login_issues", "general_issue"]
        assert diagnosis.severity in ["low", "medium", "high", "critical"]
        assert isinstance(diagnosis.troubleshooting_steps, list)
        assert len(diagnosis.troubleshooting_steps) > 0
    
    @pytest.mark.asyncio
    async def test_severity_assessment(self, tech_agent):
        """Test issue severity assessment."""
        # Test critical issue
        critical_diagnosis = await tech_agent.diagnose_issue("CRITICAL: Data loss in production system")
        assert critical_diagnosis.severity == "critical"
        
        # Test low severity issue
        low_diagnosis = await tech_agent.diagnose_issue("The button color looks wrong")
        assert low_diagnosis.severity in ["low", "medium"]
    
    @pytest.mark.asyncio
    async def test_escalation_logic(self, tech_agent):
        """Test escalation decision logic."""
        # Test issue that should be escalated
        escalation_diagnosis = await tech_agent.diagnose_issue("Security breach detected")
        assert escalation_diagnosis.escalation_needed
        
        # Test regular issue that shouldn't be escalated
        normal_diagnosis = await tech_agent.diagnose_issue("How do I change my password?")
        assert not normal_diagnosis.escalation_needed
    
    @pytest.mark.asyncio
    async def test_conversation_handler(self, tech_agent):
        """Test tech support conversation handler."""
        parameters = {
            "user_input": "I'm getting an error when I try to login",
            "session_id": "test-tech-session",
            "context": {},
            "intent": "technical"
        }
        
        result = await tech_agent.handle_conversation(parameters)
        
        assert "response" in result
        assert "agent" in result
        assert result["agent"] == "tech-support-agent"
        assert "ticket_id" in result
        assert "diagnosis" in result
        assert "next_steps" in result
        assert result["ticket_id"].startswith("TICKET-")
    
    @pytest.mark.asyncio
    async def test_ticket_creation(self, tech_agent):
        """Test support ticket creation."""
        diagnosis = TechnicalDiagnosis(
            issue_category="login_issues",
            severity="medium",
            root_cause="Password expired",
            symptoms=["Cannot access account"],
            affected_components=["authentication"],
            troubleshooting_steps=["Reset password"],
            escalation_needed=False,
            estimated_resolution_time="30 minutes"
        )
        
        ticket_id = await tech_agent.create_support_ticket(
            "test-session", diagnosis, "Can't login", {}
        )
        
        assert ticket_id.startswith("TICKET-")
        assert ticket_id in tech_agent.tickets
        
        ticket = tech_agent.tickets[ticket_id]
        assert ticket["category"] == "login_issues"
        assert ticket["severity"] == "medium"
        assert ticket["status"] == "open"
    
    @pytest.mark.asyncio
    async def test_ticket_status_management(self, tech_agent):
        """Test ticket status management."""
        # Create a ticket first
        diagnosis = TechnicalDiagnosis(
            issue_category="performance_issues",
            severity="low",
            root_cause="Browser cache",
            symptoms=["Slow loading"],
            affected_components=["user interface"],
            troubleshooting_steps=["Clear cache"],
            escalation_needed=False,
            estimated_resolution_time="15 minutes"
        )
        
        ticket_id = await tech_agent.create_support_ticket(
            "test-session", diagnosis, "App is slow", {}
        )
        
        # Test getting ticket status
        result = await tech_agent.get_ticket_status({"ticket_id": ticket_id})
        assert result["found"]
        assert result["ticket"]["id"] == ticket_id
        
        # Test updating ticket
        update_result = await tech_agent.update_ticket({
            "ticket_id": ticket_id,
            "update": {"status": "resolved"}
        })
        assert update_result["success"]
        
        # Test getting non-existent ticket
        result = await tech_agent.get_ticket_status({"ticket_id": "NONEXISTENT"})
        assert not result["found"]


class TestAgentInteraction:
    """Test interactions between agents."""
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self):
        """Test that all agents can be initialized properly."""
        faq_agent = FAQAgent()
        sales_agent = SalesAgent()
        tech_agent = TechSupportAgent()
        
        # Test that they have the required attributes
        assert hasattr(faq_agent, 'llm')
        assert hasattr(faq_agent, 'knowledge_base')
        assert hasattr(faq_agent, 'vector_store')
        
        assert hasattr(sales_agent, 'llm')
        assert hasattr(sales_agent, 'products')
        assert hasattr(sales_agent, 'leads')
        
        assert hasattr(tech_agent, 'llm')
        assert hasattr(tech_agent, 'troubleshooting_kb')
        assert hasattr(tech_agent, 'tickets')
    
    @pytest.mark.asyncio
    async def test_agent_capability_consistency(self):
        """Test that agents provide consistent capability interfaces."""
        agents = [FAQAgent(), SalesAgent(), TechSupportAgent()]
        
        for agent in agents:
            # All agents should have a conversation handler
            assert hasattr(agent, 'handle_conversation')
            
            # Test the conversation handler signature
            test_params = {
                "user_input": "test message",
                "session_id": "test-session",
                "context": {}
            }
            
            result = await agent.handle_conversation(test_params)
            
            # All should return a dictionary with these keys
            assert isinstance(result, dict)
            assert "response" in result
            assert "agent" in result
            assert isinstance(result["response"], str)
            assert isinstance(result["agent"], str)


# Test configuration

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])