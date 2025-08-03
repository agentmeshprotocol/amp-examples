"""
Sales Agent for Multi-Agent Chatbot System

Handles sales inquiries, product recommendations, and lead qualification.
Uses LangChain for sales conversations and product matching.
"""

import asyncio
import logging
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# AMP imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


class ProductRecommendation(BaseModel):
    """Product recommendation structure."""
    product_name: str = Field(description="Name of the recommended product")
    price: float = Field(description="Product price")
    features: List[str] = Field(description="Key features of the product")
    suitability_score: float = Field(description="How suitable this product is (0-1)")
    reason: str = Field(description="Why this product is recommended")


class LeadQualification(BaseModel):
    """Lead qualification structure."""
    lead_score: int = Field(description="Lead score from 1-100")
    budget_range: str = Field(description="Estimated budget range")
    timeline: str = Field(description="Purchase timeline")
    decision_maker: bool = Field(description="Is the contact a decision maker")
    pain_points: List[str] = Field(description="Identified pain points")
    next_steps: List[str] = Field(description="Recommended next steps")


class SalesAgent:
    """Sales agent that handles product inquiries and lead qualification."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.SalesAgent")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-3.5-turbo"),
            temperature=self.config.get("temperature", 0.3)
        )
        
        # Load product catalog
        self.products = self._load_product_catalog()
        
        # Lead tracking
        self.leads: Dict[str, Dict[str, Any]] = {}
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "agent_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("sales_agent", {})
        except FileNotFoundError:
            return {}
    
    def _load_product_catalog(self) -> List[Dict[str, Any]]:
        """Load product catalog."""
        catalog_path = Path(__file__).parent.parent / "config" / "product_catalog.yaml"
        
        try:
            with open(catalog_path, 'r') as f:
                catalog = yaml.safe_load(f)
                return catalog.get("products", [])
        except FileNotFoundError:
            return self._get_default_products()
    
    def _get_default_products(self) -> List[Dict[str, Any]]:
        """Get default product catalog."""
        return [
            {
                "name": "Basic Plan",
                "price": 29.99,
                "features": ["5 Users", "10GB Storage", "Email Support", "Basic Analytics"],
                "description": "Perfect for small teams getting started",
                "target_audience": "small business, startup, individual",
                "category": "basic"
            },
            {
                "name": "Professional Plan",
                "price": 79.99,
                "features": ["25 Users", "100GB Storage", "Priority Support", "Advanced Analytics", "API Access"],
                "description": "Ideal for growing businesses",
                "target_audience": "medium business, growing team",
                "category": "professional"
            },
            {
                "name": "Enterprise Plan",
                "price": 199.99,
                "features": ["Unlimited Users", "1TB Storage", "24/7 Support", "Custom Analytics", "Full API", "White Label"],
                "description": "Complete solution for large organizations",
                "target_audience": "enterprise, large business, corporation",
                "category": "enterprise"
            },
            {
                "name": "Custom Solution",
                "price": 0,  # Contact for pricing
                "features": ["Custom Features", "Dedicated Support", "On-premise Deployment", "Custom Integration"],
                "description": "Tailored solutions for unique requirements",
                "target_audience": "enterprise, specific needs, custom requirements",
                "category": "custom"
            }
        ]
    
    async def recommend_products(self, user_input: str, context: Dict[str, Any] = None) -> List[ProductRecommendation]:
        """Recommend products based on user input and context."""
        
        # Create recommendation prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a sales assistant helping to recommend products. 
            Analyze the user's input and recommend the most suitable products from our catalog.
            
            Available products:
            {self._format_products_for_prompt()}
            
            Consider:
            - User's stated needs and requirements
            - Budget hints in their message
            - Company size indicators
            - Technical requirements mentioned
            
            Recommend 1-3 most suitable products with reasoning."""),
            ("human", f"User input: {user_input}\nContext: {context or {}}\n\nRecommend suitable products:")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            
            # Parse recommendations (simplified - in production use structured output)
            recommendations = []
            for product in self.products:
                if any(keyword in user_input.lower() for keyword in product.get("target_audience", "").split(", ")):
                    recommendations.append(ProductRecommendation(
                        product_name=product["name"],
                        price=product["price"],
                        features=product["features"],
                        suitability_score=0.8,  # This would be calculated more sophisticatedly
                        reason=f"Matches your requirements for {product['target_audience']}"
                    ))
            
            # If no specific matches, recommend based on general categories
            if not recommendations:
                recommendations.append(ProductRecommendation(
                    product_name=self.products[1]["name"],  # Professional Plan as default
                    price=self.products[1]["price"],
                    features=self.products[1]["features"],
                    suitability_score=0.6,
                    reason="Good starting point for most businesses"
                ))
            
            return recommendations[:3]  # Return top 3
            
        except Exception as e:
            self.logger.error(f"Product recommendation failed: {e}")
            return []
    
    def _format_products_for_prompt(self) -> str:
        """Format products for LLM prompt."""
        formatted = []
        for product in self.products:
            price_str = f"${product['price']}/month" if product['price'] > 0 else "Contact for pricing"
            formatted.append(f"- {product['name']}: {price_str}, Features: {', '.join(product['features'])}")
        return "\n".join(formatted)
    
    async def qualify_lead(self, session_id: str, conversation_history: List[Dict[str, str]]) -> LeadQualification:
        """Qualify lead based on conversation history."""
        
        # Extract conversation text
        conversation_text = "\n".join([
            f"{msg.get('role', 'user')}: {msg.get('content', '')}"
            for msg in conversation_history
        ])
        
        # Create qualification prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a sales qualification specialist. Analyze the conversation to determine:
            1. Lead score (1-100): How likely they are to purchase
            2. Budget range: What they might be willing to spend
            3. Timeline: When they might make a purchase decision
            4. Decision maker: Whether they can make purchasing decisions
            5. Pain points: What problems they're trying to solve
            6. Next steps: What should happen next in the sales process
            
            Base your analysis on explicit and implicit cues in the conversation."""),
            ("human", f"Conversation:\n{conversation_text}\n\nProvide lead qualification:")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            
            # Simple qualification (in production, use structured output parsing)
            lead_score = 50  # Default medium score
            if any(word in conversation_text.lower() for word in ["budget", "price", "cost", "buy", "purchase"]):
                lead_score += 20
            if any(word in conversation_text.lower() for word in ["urgent", "soon", "asap", "immediately"]):
                lead_score += 15
            if any(word in conversation_text.lower() for word in ["demo", "trial", "test", "evaluate"]):
                lead_score += 10
            
            return LeadQualification(
                lead_score=min(lead_score, 100),
                budget_range="$50-200/month" if "small" in conversation_text.lower() else "$100-500/month",
                timeline="1-3 months" if "soon" in conversation_text.lower() else "3-6 months",
                decision_maker=any(word in conversation_text.lower() for word in ["ceo", "owner", "manager", "decide"]),
                pain_points=["efficiency", "scalability", "cost reduction"],  # Would extract from conversation
                next_steps=["Schedule demo", "Send proposal", "Follow up in 1 week"]
            )
            
        except Exception as e:
            self.logger.error(f"Lead qualification failed: {e}")
            return LeadQualification(
                lead_score=30,
                budget_range="Unknown",
                timeline="Unknown", 
                decision_maker=False,
                pain_points=[],
                next_steps=["Gather more information"]
            )
    
    async def handle_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle sales conversation."""
        user_input = parameters.get("user_input", "")
        session_id = parameters.get("session_id", "")
        context = parameters.get("context", {})
        history = parameters.get("history", [])
        
        # Get or create lead profile
        if session_id not in self.leads:
            self.leads[session_id] = {
                "created_at": datetime.now().isoformat(),
                "interactions": 0,
                "status": "new",
                "products_discussed": []
            }
        
        lead = self.leads[session_id]
        lead["interactions"] += 1
        lead["last_interaction"] = datetime.now().isoformat()
        
        # Determine conversation type
        is_pricing_inquiry = any(word in user_input.lower() for word in ["price", "cost", "pricing", "expensive"])
        is_demo_request = any(word in user_input.lower() for word in ["demo", "trial", "test", "show"])
        is_feature_inquiry = any(word in user_input.lower() for word in ["feature", "capability", "can it", "does it"])
        
        # Generate appropriate response
        if is_demo_request:
            response = await self._handle_demo_request(user_input, context)
        elif is_pricing_inquiry:
            response = await self._handle_pricing_inquiry(user_input, context)
        elif is_feature_inquiry:
            response = await self._handle_feature_inquiry(user_input, context)
        else:
            response = await self._handle_general_sales_inquiry(user_input, context, history)
        
        # Get product recommendations
        recommendations = await self.recommend_products(user_input, context)
        
        # Update lead with discussed products
        for rec in recommendations:
            if rec.product_name not in lead["products_discussed"]:
                lead["products_discussed"].append(rec.product_name)
        
        # Qualify lead if enough interaction
        qualification = None
        if lead["interactions"] >= 2:
            qualification = await self.qualify_lead(session_id, history)
        
        return {
            "response": response,
            "agent": "sales-agent",
            "recommendations": [rec.dict() for rec in recommendations],
            "lead_qualification": qualification.dict() if qualification else None,
            "next_steps": self._suggest_next_steps(user_input, lead),
            "confidence": 0.9
        }
    
    async def _handle_demo_request(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle demo requests."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a sales agent handling a demo request. Be enthusiastic and helpful. Explain the demo process and ask for their preferred time."),
            ("human", f"User wants a demo: {user_input}")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content
        except Exception:
            return "I'd be happy to schedule a demo for you! Our demos typically last 30 minutes and can be customized to your specific needs. What would be a good time for you this week?"
    
    async def _handle_pricing_inquiry(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle pricing inquiries."""
        # Include pricing information in response
        pricing_info = "\n".join([
            f"• {product['name']}: ${product['price']}/month" if product['price'] > 0 else f"• {product['name']}: Contact for pricing"
            for product in self.products
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a sales agent discussing pricing. Be transparent about costs and emphasize value.
            
            Our pricing tiers:
            {pricing_info}
            
            Always mention that we offer custom solutions and are happy to discuss their specific needs."""),
            ("human", f"Pricing question: {user_input}")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content
        except Exception:
            return f"Here's our pricing structure:\n\n{pricing_info}\n\nWe also offer custom solutions for specific needs. Would you like to discuss which plan might work best for your situation?"
    
    async def _handle_feature_inquiry(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle feature inquiries."""
        features_info = "\n".join([
            f"{product['name']}: {', '.join(product['features'])}"
            for product in self.products
        ])
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a sales agent discussing product features. Be detailed and helpful.
            
            Our product features by plan:
            {features_info}
            
            Match features to the user's specific needs."""),
            ("human", f"Feature question: {user_input}")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content
        except Exception:
            return "Our platform offers comprehensive features across all plans. Could you tell me more about your specific requirements so I can highlight the most relevant features for your needs?"
    
    async def _handle_general_sales_inquiry(self, user_input: str, context: Dict[str, Any], history: List[Dict[str, str]]) -> str:
        """Handle general sales inquiries."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a friendly sales agent. Help the customer understand how your product can solve their problems. Ask qualifying questions to better understand their needs."),
            ("human", f"Customer message: {user_input}")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content
        except Exception:
            return "I'd be happy to help you learn more about our solutions! Could you tell me a bit about your current challenges and what you're looking to improve?"
    
    def _suggest_next_steps(self, user_input: str, lead: Dict[str, Any]) -> List[str]:
        """Suggest next steps based on conversation."""
        steps = []
        
        if lead["interactions"] == 1:
            steps.append("Understand customer needs better")
        
        if any(word in user_input.lower() for word in ["demo", "show", "see"]):
            steps.append("Schedule product demonstration")
        
        if any(word in user_input.lower() for word in ["price", "cost", "budget"]):
            steps.append("Send detailed pricing proposal")
        
        if lead["interactions"] >= 3:
            steps.append("Connect with sales manager for next steps")
        
        if not steps:
            steps.append("Continue conversation to identify needs")
        
        return steps
    
    async def get_lead_info(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get lead information."""
        session_id = parameters.get("session_id", "")
        
        lead = self.leads.get(session_id, {})
        
        return {
            "lead_exists": session_id in self.leads,
            "lead_info": lead,
            "total_leads": len(self.leads)
        }
    
    async def update_lead(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update lead information."""
        session_id = parameters.get("session_id", "")
        updates = parameters.get("updates", {})
        
        if session_id in self.leads:
            self.leads[session_id].update(updates)
            return {"success": True, "updated": True}
        else:
            return {"success": False, "error": "Lead not found"}
    
    async def start_amp_agent(self, agent_id: str = "sales-agent",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Sales Agent")
            .with_framework("langchain")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "conversation-handler",
                self.handle_conversation,
                "Handle sales conversations and product inquiries",
                "conversation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string"},
                        "session_id": {"type": "string"},
                        "context": {"type": "object"},
                        "history": {"type": "array"}
                    },
                    "required": ["user_input"]
                },
                constraints=CapabilityConstraints(response_time_ms=4000)
            )
            .add_capability(
                "product-recommendation",
                self.recommend_products,
                "Recommend products based on customer needs",
                "sales"
            )
            .add_capability(
                "lead-qualification",
                self.qualify_lead,
                "Qualify sales leads",
                "sales"
            )
            .add_capability(
                "lead-info",
                self.get_lead_info,
                "Get lead information",
                "sales"
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the sales agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sales agent
    sales_agent = SalesAgent()
    
    # Start AMP agent
    client = await sales_agent.start_amp_agent()
    
    try:
        print("Sales Agent started. Testing product recommendations...")
        
        # Test queries
        test_queries = [
            "How much does your product cost?",
            "I need a solution for my small startup",
            "Can you show me a demo?",
            "What features do you have for large enterprises?"
        ]
        
        for query in test_queries:
            result = await sales_agent.handle_conversation({
                "user_input": query,
                "session_id": "test-session",
                "context": {},
                "history": []
            })
            print(f"Q: {query}")
            print(f"A: {result['response']}")
            if result.get('recommendations'):
                print(f"Recommended: {[r['product_name'] for r in result['recommendations']]}")
            print()
        
        print("Sales Agent is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())