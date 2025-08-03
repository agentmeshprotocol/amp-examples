"""
Ticket Classifier Agent using LangChain.

This agent automatically categorizes incoming support tickets by type, priority,
and sentiment using advanced NLP analysis.
"""

import logging
import re
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field

# AMP imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))

from amp_client import AMPClient, AMPClientConfig
from amp_types import Capability, CapabilityConstraints, TransportType

# Support system imports
from ..support_types import (
    Ticket, TicketCategory, TicketPriority, TicketStatus,
    CustomerInfo, SLALevel
)


class TicketClassification(BaseModel):
    """Structured output for ticket classification."""
    category: str = Field(description="Primary ticket category")
    subcategory: str = Field(description="More specific subcategory")
    priority: str = Field(description="Ticket priority level")
    urgency_indicators: List[str] = Field(description="List of urgency indicators found")
    sentiment_score: float = Field(description="Sentiment score from -1 (negative) to 1 (positive)")
    confidence: float = Field(description="Classification confidence score 0-1")
    keywords: List[str] = Field(description="Important keywords extracted")
    suggested_agent: Optional[str] = Field(description="Suggested agent type for routing")
    estimated_complexity: str = Field(description="Estimated complexity: simple, medium, complex")
    requires_escalation: bool = Field(description="Whether ticket likely needs escalation")


class TicketClassifierAgent:
    """
    Advanced ticket classification agent using LangChain for intelligent 
    categorization and routing of customer support tickets.
    """
    
    def __init__(self, config: AMPClientConfig, openai_api_key: str):
        self.config = config
        self.client = AMPClient(config)
        self.logger = logging.getLogger(f"support.classifier.{config.agent_id}")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4",
            temperature=0.1  # Low temperature for consistent classification
        )
        
        # Classification prompt template
        self.classification_prompt = self._create_classification_prompt()
        
        # Urgency detection patterns
        self.urgency_patterns = {
            "critical": [
                r"\b(critical|urgent|emergency|down|outage|broken|not working|can't access)\b",
                r"\b(production|live|customer facing|revenue impact)\b",
                r"\b(immediately|asap|right away|urgent|emergency)\b"
            ],
            "high": [
                r"\b(important|priority|deadline|soon|blocking|stuck)\b",
                r"\b(multiple users|many customers|widespread)\b",
                r"\b(payment|billing|invoice|charge|refund)\b"
            ],
            "escalation": [
                r"\b(escalate|manager|supervisor|complaint|unsatisfied|angry)\b",
                r"\b(legal|lawyer|attorney|sue|lawsuit)\b",
                r"\b(cancel|refund|terminate|leave|switch)\b"
            ]
        }
        
        # Category keywords
        self.category_keywords = {
            TicketCategory.TECHNICAL: [
                "error", "bug", "issue", "problem", "not working", "broken", "crash",
                "login", "password", "authentication", "performance", "slow", "api",
                "integration", "setup", "configuration", "install", "update"
            ],
            TicketCategory.BILLING: [
                "bill", "billing", "invoice", "payment", "charge", "refund", "credit",
                "subscription", "plan", "upgrade", "downgrade", "cancel", "pricing"
            ],
            TicketCategory.PRODUCT: [
                "feature", "functionality", "how to", "tutorial", "guide", "documentation",
                "usage", "training", "onboarding", "capability", "limitation"
            ],
            TicketCategory.ACCOUNT: [
                "account", "profile", "settings", "permissions", "access", "user",
                "role", "team", "organization", "member", "invite"
            ],
            TicketCategory.FEATURE_REQUEST: [
                "feature request", "enhancement", "improvement", "suggestion", "idea",
                "new feature", "add support", "would like", "wish", "could you"
            ],
            TicketCategory.BUG_REPORT: [
                "bug", "error", "incorrect", "wrong", "unexpected", "shouldn't",
                "malfunction", "glitch", "defect", "issue"
            ]
        }
        
    async def start(self):
        """Start the ticket classifier agent."""
        # Register capabilities
        await self._register_capabilities()
        
        # Connect to AMP network
        connected = await self.client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        
        self.logger.info("Ticket Classifier Agent started successfully")
    
    async def stop(self):
        """Stop the ticket classifier agent."""
        await self.client.disconnect()
        self.logger.info("Ticket Classifier Agent stopped")
    
    async def _register_capabilities(self):
        """Register agent capabilities with AMP."""
        
        # Ticket classification capability
        classify_capability = Capability(
            id="ticket-classification",
            version="1.0",
            description="Automatically classify support tickets by category, priority, and sentiment",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "object",
                        "properties": {
                            "subject": {"type": "string"},
                            "description": {"type": "string"},
                            "customer_info": {
                                "type": "object",
                                "properties": {
                                    "sla_level": {"type": "string"},
                                    "account_type": {"type": "string"},
                                    "language": {"type": "string"}
                                }
                            }
                        },
                        "required": ["subject", "description"]
                    }
                },
                "required": ["ticket"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "subcategory": {"type": "string"},
                    "priority": {"type": "string"},
                    "urgency_indicators": {"type": "array", "items": {"type": "string"}},
                    "sentiment_score": {"type": "number"},
                    "confidence": {"type": "number"},
                    "keywords": {"type": "array", "items": {"type": "string"}},
                    "suggested_agent": {"type": "string"},
                    "estimated_complexity": {"type": "string"},
                    "requires_escalation": {"type": "boolean"},
                    "routing_recommendations": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=5000,
                max_input_length=10000,
                supported_languages=["en", "es", "fr", "de"],
                min_confidence=0.7
            ),
            category="text-processing",
            subcategories=["text-classification", "sentiment-analysis"],
            examples=[
                {
                    "input": {
                        "ticket": {
                            "subject": "Login not working",
                            "description": "I can't log into my account. Getting error 500."
                        }
                    },
                    "output": {
                        "category": "technical",
                        "priority": "high",
                        "sentiment_score": -0.3,
                        "confidence": 0.9
                    }
                }
            ]
        )
        
        self.client.register_capability(classify_capability, self.classify_ticket)
        
        # Batch classification capability
        batch_classify_capability = Capability(
            id="ticket-batch-classification",
            version="1.0", 
            description="Classify multiple tickets in batch for efficiency",
            input_schema={
                "type": "object",
                "properties": {
                    "tickets": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "subject": {"type": "string"},
                                "description": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["tickets"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "classifications": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "batch_metrics": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=15000,
                max_input_length=50000
            )
        )
        
        self.client.register_capability(batch_classify_capability, self.classify_tickets_batch)
    
    async def classify_ticket(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify a single support ticket.
        
        Args:
            parameters: Contains ticket data with subject, description, and customer info
            
        Returns:
            Classification results with category, priority, sentiment, etc.
        """
        try:
            ticket_data = parameters["ticket"]
            subject = ticket_data["subject"]
            description = ticket_data["description"]
            customer_info = ticket_data.get("customer_info", {})
            
            # Combine subject and description for analysis
            full_text = f"Subject: {subject}\n\nDescription: {description}"
            
            # Get LLM classification
            classification = await self._llm_classify(full_text, customer_info)
            
            # Enhance with rule-based analysis
            enhanced_classification = await self._enhance_classification(
                classification, full_text, customer_info
            )
            
            # Generate routing recommendations
            routing_recommendations = self._generate_routing_recommendations(enhanced_classification)
            
            result = {
                **enhanced_classification.dict(),
                "routing_recommendations": routing_recommendations,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": self.config.agent_id
            }
            
            self.logger.info(f"Classified ticket: {classification.category}/{classification.priority} "
                           f"(confidence: {classification.confidence:.2f})")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error classifying ticket: {e}")
            raise
    
    async def classify_tickets_batch(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify multiple tickets in batch for efficiency.
        
        Args:
            parameters: Contains array of tickets to classify
            
        Returns:
            Batch classification results with metrics
        """
        try:
            tickets = parameters["tickets"]
            classifications = []
            
            start_time = datetime.now(timezone.utc)
            
            for ticket in tickets:
                try:
                    # Create parameters for single ticket classification
                    ticket_params = {"ticket": ticket}
                    classification = await self.classify_ticket(ticket_params)
                    classification["ticket_id"] = ticket.get("id", "unknown")
                    classifications.append(classification)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to classify ticket {ticket.get('id')}: {e}")
                    classifications.append({
                        "ticket_id": ticket.get("id", "unknown"),
                        "error": str(e),
                        "category": "general",
                        "priority": "medium",
                        "confidence": 0.0
                    })
            
            end_time = datetime.now(timezone.utc)
            processing_time = (end_time - start_time).total_seconds()
            
            # Calculate batch metrics
            batch_metrics = {
                "total_tickets": len(tickets),
                "successful_classifications": len([c for c in classifications if "error" not in c]),
                "failed_classifications": len([c for c in classifications if "error" in c]),
                "processing_time_seconds": processing_time,
                "avg_confidence": sum(c.get("confidence", 0) for c in classifications) / len(classifications),
                "category_distribution": self._calculate_category_distribution(classifications)
            }
            
            return {
                "classifications": classifications,
                "batch_metrics": batch_metrics,
                "processed_at": end_time.isoformat(),
                "agent_id": self.config.agent_id
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch classification: {e}")
            raise
    
    async def _llm_classify(self, text: str, customer_info: Dict[str, Any]) -> TicketClassification:
        """Use LLM to classify the ticket."""
        try:
            # Create parser for structured output
            parser = PydanticOutputParser(pydantic_object=TicketClassification)
            
            # Prepare the prompt
            prompt = self.classification_prompt.format_prompt(
                ticket_text=text,
                customer_sla=customer_info.get("sla_level", "basic"),
                customer_type=customer_info.get("account_type", "free"),
                format_instructions=parser.get_format_instructions()
            )
            
            # Get LLM response
            response = await self.llm.agenerate([prompt.to_messages()])
            result_text = response.generations[0][0].text
            
            # Parse structured output
            classification = parser.parse(result_text)
            
            return classification
            
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}, falling back to rule-based")
            return self._fallback_classification(text)
    
    def _fallback_classification(self, text: str) -> TicketClassification:
        """Fallback rule-based classification when LLM fails."""
        text_lower = text.lower()
        
        # Determine category by keywords
        category_scores = {}
        for cat, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                category_scores[cat] = score
        
        category = max(category_scores, key=category_scores.get) if category_scores else TicketCategory.GENERAL
        
        # Determine priority by urgency patterns
        priority = TicketPriority.MEDIUM
        urgency_indicators = []
        
        for urgency_level, patterns in self.urgency_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    urgency_indicators.append(pattern)
                    if urgency_level == "critical":
                        priority = TicketPriority.CRITICAL
                    elif urgency_level == "high" and priority != TicketPriority.CRITICAL:
                        priority = TicketPriority.HIGH
        
        # Basic sentiment analysis
        negative_words = ["bad", "terrible", "awful", "hate", "frustrated", "angry", "broken", "not working"]
        positive_words = ["good", "great", "love", "thanks", "appreciate", "excellent", "working"]
        
        neg_count = sum(1 for word in negative_words if word in text_lower)
        pos_count = sum(1 for word in positive_words if word in text_lower)
        
        sentiment_score = (pos_count - neg_count) / max(pos_count + neg_count, 1)
        
        return TicketClassification(
            category=category.value,
            subcategory="general",
            priority=priority.value,
            urgency_indicators=urgency_indicators,
            sentiment_score=sentiment_score,
            confidence=0.6,  # Lower confidence for rule-based
            keywords=list(category_scores.keys()),
            suggested_agent=self._map_category_to_agent(category),
            estimated_complexity="medium",
            requires_escalation="escalation" in [p for patterns in self.urgency_patterns.values() for p in patterns if p in text_lower]
        )
    
    async def _enhance_classification(self, classification: TicketClassification, 
                                    text: str, customer_info: Dict[str, Any]) -> TicketClassification:
        """Enhance LLM classification with rule-based adjustments."""
        
        # Adjust priority based on customer SLA level
        sla_level = customer_info.get("sla_level", "basic")
        if sla_level == "enterprise" and classification.priority in ["low", "medium"]:
            classification.priority = "high"
        elif sla_level == "premium" and classification.priority == "low":
            classification.priority = "medium"
        
        # Check for escalation indicators
        text_lower = text.lower()
        escalation_found = any(
            re.search(pattern, text_lower)
            for pattern in self.urgency_patterns["escalation"]
        )
        if escalation_found:
            classification.requires_escalation = True
            if classification.priority in ["low", "medium"]:
                classification.priority = "high"
        
        # Ensure suggested agent matches category
        if not classification.suggested_agent:
            classification.suggested_agent = self._map_category_to_agent(
                TicketCategory(classification.category)
            )
        
        return classification
    
    def _map_category_to_agent(self, category: TicketCategory) -> str:
        """Map ticket category to suggested agent type."""
        mapping = {
            TicketCategory.TECHNICAL: "technical-support",
            TicketCategory.BILLING: "billing-support", 
            TicketCategory.PRODUCT: "product-support",
            TicketCategory.ACCOUNT: "technical-support",
            TicketCategory.FEATURE_REQUEST: "product-support",
            TicketCategory.BUG_REPORT: "technical-support",
            TicketCategory.GENERAL: "general-support"
        }
        return mapping.get(category, "general-support")
    
    def _generate_routing_recommendations(self, classification: TicketClassification) -> Dict[str, Any]:
        """Generate routing recommendations based on classification."""
        recommendations = {
            "primary_agent_type": classification.suggested_agent,
            "alternative_agents": [],
            "queue_priority": self._calculate_queue_priority(classification),
            "auto_assign": classification.confidence > 0.8 and not classification.requires_escalation,
            "escalation_recommended": classification.requires_escalation,
            "sla_urgency": classification.priority in ["urgent", "critical"],
            "routing_metadata": {
                "classification_confidence": classification.confidence,
                "complexity": classification.estimated_complexity,
                "keywords": classification.keywords[:5]  # Top 5 keywords
            }
        }
        
        # Add alternative agents based on category
        if classification.category == "technical":
            recommendations["alternative_agents"] = ["escalation-manager", "senior-technical"]
        elif classification.category == "billing":
            recommendations["alternative_agents"] = ["account-manager", "billing-specialist"]
        
        return recommendations
    
    def _calculate_queue_priority(self, classification: TicketClassification) -> int:
        """Calculate queue priority score (1-10, higher = more priority)."""
        priority_scores = {
            "low": 2,
            "medium": 5,
            "high": 7,
            "urgent": 9,
            "critical": 10
        }
        
        base_score = priority_scores.get(classification.priority, 5)
        
        # Adjust based on other factors
        if classification.requires_escalation:
            base_score += 2
        
        if classification.sentiment_score < -0.5:
            base_score += 1  # Angry customers get higher priority
        
        return min(base_score, 10)
    
    def _calculate_category_distribution(self, classifications: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate distribution of categories in batch."""
        distribution = {}
        for classification in classifications:
            category = classification.get("category", "unknown")
            distribution[category] = distribution.get(category, 0) + 1
        return distribution
    
    def _create_classification_prompt(self) -> ChatPromptTemplate:
        """Create the LangChain prompt template for ticket classification."""
        
        system_template = """You are an expert customer support ticket classifier. Your job is to analyze support tickets and categorize them accurately.

Categories:
- technical: Technical issues, bugs, errors, login problems, API issues
- billing: Payment, invoicing, subscription, pricing questions  
- product: Feature questions, how-to, usage guidance, training
- account: Account settings, permissions, user management
- feature_request: Requests for new features or enhancements
- bug_report: Reports of software defects or unexpected behavior
- general: General inquiries that don't fit other categories

Priority Levels:
- low: General questions, minor issues, no urgency
- medium: Standard issues affecting single user
- high: Issues affecting multiple users or important functionality
- urgent: Critical business impact, many users affected
- critical: System down, security issues, revenue impact

Consider the customer's SLA level: {customer_sla}
Consider the customer's account type: {customer_type}

Analyze the sentiment carefully - frustrated customers may need higher priority.
Look for urgency indicators like "urgent", "critical", "emergency", "down", "not working".
Consider escalation needs for angry customers or complex technical issues.

{format_instructions}"""

        human_template = """Classify this support ticket:

{ticket_text}

Provide a thorough analysis including category, priority, sentiment, urgency indicators, and routing recommendations."""

        system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
        human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
        
        return ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        config = AMPClientConfig(
            agent_id="ticket-classifier-001",
            agent_name="Ticket Classifier",
            framework="langchain",
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:8000"
        )
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        agent = TicketClassifierAgent(config, openai_api_key)
        
        try:
            await agent.start()
            print("Ticket Classifier Agent is running...")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await agent.stop()
    
    asyncio.run(main())