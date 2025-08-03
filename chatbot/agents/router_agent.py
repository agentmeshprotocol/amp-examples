"""
Router Agent for Multi-Agent Chatbot System

This agent analyzes user input and routes conversations to appropriate specialist agents.
Uses LangChain for intent detection and context-aware routing.
"""

import asyncio
import logging
import yaml
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

# AMP imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient, AMPClientConfig
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder, create_simple_capability


@dataclass
class IntentResult:
    """Result of intent detection."""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    routing_suggestion: str
    context_requirements: List[str]


class IntentClassification(BaseModel):
    """Pydantic model for intent classification output."""
    primary_intent: str = Field(description="The primary intent detected (faq, sales, technical, general)")
    confidence: float = Field(description="Confidence score between 0 and 1")
    secondary_intents: List[str] = Field(description="Any secondary intents detected", default=[])
    entities: Dict[str, str] = Field(description="Named entities extracted from the text", default={})
    urgency: str = Field(description="Urgency level: low, medium, high", default="medium")
    routing_recommendation: str = Field(description="Which agent should handle this (faq-agent, sales-agent, tech-support-agent)")


class RouterAgent:
    """Router agent that detects intent and routes conversations."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.RouterAgent")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-3.5-turbo"),
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 500)
        )
        
        # Load intent patterns
        self.intent_patterns = self._load_intent_patterns()
        
        # Set up intent classification chain
        self.intent_classifier = self._setup_intent_classifier()
        
        # AMP client will be set up when agent starts
        self.amp_client: Optional[AMPClient] = None
        self.conversation_contexts: Dict[str, Dict[str, Any]] = {}
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "agent_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("router_agent", {})
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return {}
    
    def _load_intent_patterns(self) -> Dict[str, Any]:
        """Load intent detection patterns."""
        patterns_path = Path(__file__).parent.parent / "config" / "intents.yaml"
        
        try:
            with open(patterns_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Intent patterns not found: {patterns_path}, using defaults")
            return self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """Get default intent patterns."""
        return {
            "intents": {
                "faq": {
                    "patterns": ["what are your hours", "how to contact", "company information"],
                    "examples": ["When are you open?", "What's your address?"]
                },
                "sales": {
                    "patterns": ["pricing", "buy", "purchase", "demo", "trial"],
                    "examples": ["How much does it cost?", "I want to buy your product"]
                },
                "technical": {
                    "patterns": ["not working", "error", "bug", "technical issue"],
                    "examples": ["The app crashes", "I'm getting an error"]
                },
                "general": {
                    "patterns": ["hello", "hi", "help", "thanks"],
                    "examples": ["Hi there", "Can you help me?"]
                }
            }
        }
    
    def _setup_intent_classifier(self) -> Any:\n        \"\"\"Set up the intent classification chain.\"\"\"\n        parser = PydanticOutputParser(pydantic_object=IntentClassification)\n        \n        # Create the prompt template\n        system_template = \"\"\"You are an intelligent conversation router for a customer service chatbot system.\n        \nYour job is to analyze user input and determine:\n1. The primary intent (faq, sales, technical, general)\n2. Confidence level in your classification\n3. Any named entities in the text\n4. Urgency level of the request\n5. Which specialized agent should handle this conversation\n        \nAvailable agents:\n- faq-agent: Handles general questions, company info, policies, hours\n- sales-agent: Handles pricing, purchases, demos, product information\n- tech-support-agent: Handles technical issues, bugs, troubleshooting\n        \nIntent categories:\n- faq: General questions about the company, policies, hours, contact info\n- sales: Pricing questions, purchase inquiries, demo requests, product info\n- technical: Technical problems, error reports, troubleshooting, bug reports\n- general: Greetings, thanks, general help requests\n        \nConsider the conversation context if provided.\n        \n{format_instructions}\n        \"\"\"\n        \n        human_template = \"\"\"User input: {user_input}\n        \nConversation context: {context}\n        \nPrevious messages: {history}\n        \nAnalyze this input and provide the classification:\"\"\"\n        \n        system_message = SystemMessagePromptTemplate.from_template(system_template)\n        human_message = HumanMessagePromptTemplate.from_template(human_template)\n        \n        prompt = ChatPromptTemplate.from_messages([\n            system_message,\n            human_message\n        ])\n        \n        # Format the prompt with parser instructions\n        formatted_prompt = prompt.partial(format_instructions=parser.get_format_instructions())\n        \n        return formatted_prompt | self.llm | parser\n    \n    async def detect_intent(self, user_input: str, context: Optional[Dict[str, Any]] = None,\n                           history: Optional[List[Dict[str, str]]] = None) -> IntentResult:\n        \"\"\"Detect intent from user input.\"\"\"\n        try:\n            # Prepare context and history\n            context_str = str(context) if context else \"No previous context\"\n            history_str = \"\\n\".join([\n                f\"{msg.get('role', 'user')}: {msg.get('content', '')}\"\n                for msg in (history or [])\n            ]) or \"No previous messages\"\n            \n            # Run intent classification\n            result = await self.intent_classifier.ainvoke({\n                \"user_input\": user_input,\n                \"context\": context_str,\n                \"history\": history_str\n            })\n            \n            # Convert to IntentResult\n            return IntentResult(\n                intent=result.primary_intent,\n                confidence=result.confidence,\n                entities=result.entities,\n                routing_suggestion=result.routing_recommendation,\n                context_requirements=result.secondary_intents\n            )\n            \n        except Exception as e:\n            self.logger.error(f\"Intent detection failed: {e}\")\n            # Fallback to simple keyword matching\n            return await self._fallback_intent_detection(user_input)\n    \n    async def _fallback_intent_detection(self, user_input: str) -> IntentResult:\n        \"\"\"Fallback intent detection using keyword matching.\"\"\"\n        user_lower = user_input.lower()\n        \n        # Simple keyword-based classification\n        if any(word in user_lower for word in [\"price\", \"cost\", \"buy\", \"purchase\", \"demo\"]):\n            return IntentResult(\n                intent=\"sales\",\n                confidence=0.7,\n                entities={},\n                routing_suggestion=\"sales-agent\",\n                context_requirements=[]\n            )\n        elif any(word in user_lower for word in [\"error\", \"bug\", \"not working\", \"problem\"]):\n            return IntentResult(\n                intent=\"technical\",\n                confidence=0.7,\n                entities={},\n                routing_suggestion=\"tech-support-agent\",\n                context_requirements=[]\n            )\n        elif any(word in user_lower for word in [\"hours\", \"contact\", \"address\", \"info\"]):\n            return IntentResult(\n                intent=\"faq\",\n                confidence=0.7,\n                entities={},\n                routing_suggestion=\"faq-agent\",\n                context_requirements=[]\n            )\n        else:\n            return IntentResult(\n                intent=\"general\",\n                confidence=0.5,\n                entities={},\n                routing_suggestion=\"faq-agent\",\n                context_requirements=[]\n            )\n    \n    async def route_conversation(self, session_id: str, user_input: str,\n                               user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:\n        \"\"\"Route conversation to appropriate agent.\"\"\"\n        try:\n            # Get conversation history\n            conversation = self.conversation_contexts.get(session_id, {\n                \"history\": [],\n                \"current_agent\": None,\n                \"context\": {}\n            })\n            \n            # Detect intent\n            intent_result = await self.detect_intent(\n                user_input,\n                context=conversation.get(\"context\"),\n                history=conversation.get(\"history\", [])\n            )\n            \n            # Determine target agent\n            target_agent = intent_result.routing_suggestion\n            \n            # Check if we need to switch agents\n            current_agent = conversation.get(\"current_agent\")\n            if current_agent != target_agent:\n                await self._handle_agent_handoff(\n                    session_id, current_agent, target_agent, intent_result\n                )\n            \n            # Update conversation context\n            conversation[\"current_agent\"] = target_agent\n            conversation[\"history\"].append({\n                \"role\": \"user\",\n                \"content\": user_input,\n                \"timestamp\": asyncio.get_event_loop().time(),\n                \"intent\": intent_result.intent\n            })\n            \n            # Keep only last 10 messages to prevent context bloat\n            conversation[\"history\"] = conversation[\"history\"][-10:]\n            \n            self.conversation_contexts[session_id] = conversation\n            \n            # Route to target agent\n            response = await self.amp_client.invoke_capability(\n                target_agent,\n                \"conversation-handler\",\n                {\n                    \"user_input\": user_input,\n                    \"session_id\": session_id,\n                    \"intent\": intent_result.intent,\n                    \"entities\": intent_result.entities,\n                    \"context\": conversation[\"context\"],\n                    \"history\": conversation[\"history\"][-5:]  # Send last 5 messages\n                }\n            )\n            \n            # Update conversation with agent response\n            if \"result\" in response:\n                conversation[\"history\"].append({\n                    \"role\": \"assistant\",\n                    \"content\": response[\"result\"].get(\"response\", \"\"),\n                    \"timestamp\": asyncio.get_event_loop().time(),\n                    \"agent\": target_agent\n                })\n            \n            return {\n                \"response\": response.get(\"result\", {}).get(\"response\", \"I'm sorry, I couldn't process your request.\"),\n                \"agent\": target_agent,\n                \"intent\": intent_result.intent,\n                \"confidence\": intent_result.confidence,\n                \"session_id\": session_id\n            }\n            \n        except Exception as e:\n            self.logger.error(f\"Routing error: {e}\")\n            return {\n                \"response\": \"I'm sorry, I'm having trouble processing your request right now. Please try again.\",\n                \"agent\": \"router\",\n                \"intent\": \"error\",\n                \"confidence\": 0.0,\n                \"session_id\": session_id,\n                \"error\": str(e)\n            }\n    \n    async def _handle_agent_handoff(self, session_id: str, from_agent: Optional[str],\n                                   to_agent: str, intent_result: IntentResult):\n        \"\"\"Handle handoff between agents.\"\"\"\n        if from_agent and from_agent != to_agent:\n            self.logger.info(f\"Handing off conversation {session_id} from {from_agent} to {to_agent}\")\n            \n            # Notify the previous agent about handoff\n            try:\n                await self.amp_client.emit_event(\n                    \"conversation_handoff\",\n                    {\n                        \"session_id\": session_id,\n                        \"from_agent\": from_agent,\n                        \"to_agent\": to_agent,\n                        \"intent\": intent_result.intent,\n                        \"reason\": \"intent_change\"\n                    }\n                )\n            except Exception as e:\n                self.logger.warning(f\"Failed to notify agent handoff: {e}\")\n    \n    async def handle_conversation_routing(self, parameters: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"AMP capability handler for conversation routing.\"\"\"\n        user_input = parameters.get(\"user_input\", \"\")\n        session_id = parameters.get(\"session_id\", \"default\")\n        user_context = parameters.get(\"context\", {})\n        \n        result = await self.route_conversation(session_id, user_input, user_context)\n        \n        return {\n            \"routing_result\": result,\n            \"success\": True\n        }\n    \n    async def handle_intent_detection(self, parameters: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"AMP capability handler for intent detection only.\"\"\"\n        user_input = parameters.get(\"user_input\", \"\")\n        context = parameters.get(\"context\")\n        history = parameters.get(\"history\")\n        \n        intent_result = await self.detect_intent(user_input, context, history)\n        \n        return {\n            \"intent\": intent_result.intent,\n            \"confidence\": intent_result.confidence,\n            \"entities\": intent_result.entities,\n            \"routing_suggestion\": intent_result.routing_suggestion,\n            \"context_requirements\": intent_result.context_requirements\n        }\n    \n    async def get_conversation_context(self, parameters: Dict[str, Any]) -> Dict[str, Any]:\n        \"\"\"Get conversation context for a session.\"\"\"\n        session_id = parameters.get(\"session_id\", \"\")\n        \n        context = self.conversation_contexts.get(session_id, {})\n        \n        return {\n            \"context\": context,\n            \"exists\": session_id in self.conversation_contexts\n        }\n    \n    async def start_amp_agent(self, agent_id: str = \"router-agent\",\n                             endpoint: str = \"http://localhost:8000\") -> AMPClient:\n        \"\"\"Start the AMP agent.\"\"\"\n        \n        # Create capabilities\n        routing_capability = Capability(\n            id=\"conversation-routing\",\n            version=\"1.0\",\n            description=\"Route conversations to appropriate specialist agents\",\n            category=\"conversation\",\n            input_schema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"user_input\": {\"type\": \"string\"},\n                    \"session_id\": {\"type\": \"string\"},\n                    \"context\": {\"type\": \"object\"}\n                },\n                \"required\": [\"user_input\", \"session_id\"]\n            },\n            output_schema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"routing_result\": {\"type\": \"object\"},\n                    \"success\": {\"type\": \"boolean\"}\n                }\n            },\n            constraints=CapabilityConstraints(\n                response_time_ms=3000,\n                max_tokens=1000\n            )\n        )\n        \n        intent_capability = Capability(\n            id=\"intent-detection\",\n            version=\"1.0\",\n            description=\"Detect user intent from natural language input\",\n            category=\"nlp\",\n            input_schema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"user_input\": {\"type\": \"string\"},\n                    \"context\": {\"type\": \"object\"},\n                    \"history\": {\"type\": \"array\"}\n                },\n                \"required\": [\"user_input\"]\n            },\n            output_schema={\n                \"type\": \"object\",\n                \"properties\": {\n                    \"intent\": {\"type\": \"string\"},\n                    \"confidence\": {\"type\": \"number\"},\n                    \"entities\": {\"type\": \"object\"}\n                }\n            },\n            constraints=CapabilityConstraints(\n                response_time_ms=2000\n            )\n        )\n        \n        # Build and start AMP client\n        self.amp_client = await (\n            AMPBuilder(agent_id, \"Conversation Router Agent\")\n            .with_framework(\"langchain\")\n            .with_transport(TransportType.HTTP, endpoint)\n            .add_capability(\n                \"conversation-routing\",\n                self.handle_conversation_routing,\n                \"Route conversations to appropriate agents\",\n                \"conversation\"\n            )\n            .add_capability(\n                \"intent-detection\",\n                self.handle_intent_detection,\n                \"Detect user intent from input\",\n                \"nlp\"\n            )\n            .add_capability(\n                \"context-retrieval\",\n                self.get_conversation_context,\n                \"Get conversation context\",\n                \"conversation\"\n            )\n            .build()\n        )\n        \n        return self.amp_client\n\n\nasync def main():\n    \"\"\"Main function for testing the router agent.\"\"\"\n    logging.basicConfig(level=logging.INFO)\n    \n    # Create router agent\n    router = RouterAgent()\n    \n    # Start AMP agent\n    client = await router.start_amp_agent()\n    \n    try:\n        print(\"Router Agent started. Testing intent detection...\")\n        \n        # Test intent detection\n        test_inputs = [\n            \"What are your business hours?\",\n            \"How much does your product cost?\",\n            \"I'm getting an error when I try to login\",\n            \"Hi there, can you help me?\"\n        ]\n        \n        for test_input in test_inputs:\n            intent_result = await router.detect_intent(test_input)\n            print(f\"Input: {test_input}\")\n            print(f\"Intent: {intent_result.intent} (confidence: {intent_result.confidence:.2f})\")\n            print(f\"Routing: {intent_result.routing_suggestion}\\n\")\n        \n        print(\"Router Agent is running. Press Ctrl+C to stop.\")\n        await asyncio.Future()  # Run forever\n        \n    except KeyboardInterrupt:\n        print(\"Shutting down...\")\n    finally:\n        await client.disconnect()\n\n\nif __name__ == \"__main__\":\n    asyncio.run(main())"