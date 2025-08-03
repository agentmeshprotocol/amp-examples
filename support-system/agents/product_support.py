"""
Product Support Agent using LangChain.

This agent handles product questions, feature requests, usage guidance,
and provides comprehensive product knowledge and training support.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# LangChain imports
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.schema import BaseOutputParser
from pydantic import BaseModel, Field

# AMP imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))

from amp_client import AMPClient, AMPClientConfig
from amp_types import Capability, CapabilityConstraints, TransportType

# Support system imports
from ..support_types import Ticket, TicketCategory, TicketPriority


class ProductKnowledgeBase:
    """Simulated product knowledge base."""
    
    def __init__(self):
        self.features = {
            "user_management": {
                "description": "Manage users, roles, and permissions",
                "capabilities": ["user_creation", "role_assignment", "permission_management"],
                "documentation": "/docs/user-management",
                "tutorials": ["/tutorials/adding-users", "/tutorials/role-setup"],
                "common_issues": ["Permission denied errors", "Role inheritance problems"]
            },
            "api_integration": {
                "description": "REST API for third-party integrations", 
                "capabilities": ["webhook_endpoints", "api_authentication", "rate_limiting"],
                "documentation": "/docs/api-reference",
                "tutorials": ["/tutorials/api-quickstart", "/tutorials/webhook-setup"],
                "common_issues": ["API key authentication", "Rate limit exceeded"]
            },
            "reporting": {
                "description": "Analytics and reporting dashboard",
                "capabilities": ["custom_reports", "data_export", "scheduled_reports"],
                "documentation": "/docs/reporting",
                "tutorials": ["/tutorials/custom-reports", "/tutorials/data-export"],
                "common_issues": ["Report generation timeout", "Data export format issues"]
            }
        }
        
        self.product_roadmap = {
            "q1_2024": ["Enhanced API rate limiting", "Mobile app improvements"],
            "q2_2024": ["Advanced reporting features", "SSO integration"],
            "q3_2024": ["Machine learning insights", "Workflow automation"],
            "q4_2024": ["Enterprise scaling features", "Advanced security"]
        }
        
        self.best_practices = {
            "user_onboarding": [
                "Start with basic features",
                "Complete the setup wizard", 
                "Review security settings",
                "Integrate with existing tools"
            ],
            "performance_optimization": [
                "Regular data cleanup",
                "Monitor resource usage",
                "Optimize API calls",
                "Use caching effectively"
            ]
        }

    def search_features(self, query: str) -> List[Dict[str, Any]]:
        """Search for features matching the query."""
        results = []
        query_lower = query.lower()
        
        for feature_id, feature_data in self.features.items():
            if (query_lower in feature_data["description"].lower() or
                any(query_lower in cap.lower() for cap in feature_data["capabilities"])):
                results.append({
                    "feature_id": feature_id,
                    **feature_data,
                    "relevance": 0.9
                })
        
        return results

    def get_feature_info(self, feature_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific feature."""
        return self.features.get(feature_id)

    def get_roadmap_info(self, timeframe: str = None) -> Dict[str, Any]:
        """Get product roadmap information."""
        if timeframe:
            return {timeframe: self.product_roadmap.get(timeframe, [])}
        return self.product_roadmap

    def get_best_practices(self, category: str = None) -> Dict[str, Any]:
        """Get best practices for product usage."""
        if category:
            return {category: self.best_practices.get(category, [])}
        return self.best_practices


class ProductSearchTool(BaseTool):
    """Tool for searching product features and documentation."""
    
    name: str = "product_search"
    description: str = "Search for product features, documentation, and tutorials"
    
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, query: str) -> str:
        """Search product knowledge base."""
        results = self.knowledge_base.search_features(query)
        return json.dumps(results, indent=2)


class FeatureGuidanceTool(BaseTool):
    """Tool for providing feature usage guidance."""
    
    name: str = "feature_guidance"
    description: str = "Provide detailed guidance on how to use specific product features"
    
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, feature_id: str) -> str:
        """Get feature guidance."""
        feature_info = self.knowledge_base.get_feature_info(feature_id)
        if not feature_info:
            return json.dumps({"error": "Feature not found"})
        
        guidance = {
            "feature_info": feature_info,
            "getting_started": f"To get started with {feature_id}, follow these steps:",
            "best_practices": self.knowledge_base.get_best_practices(),
            "troubleshooting": f"Common issues with {feature_id}: {feature_info.get('common_issues', [])}"
        }
        
        return json.dumps(guidance, indent=2)


class RoadmapTool(BaseTool):
    """Tool for accessing product roadmap information."""
    
    name: str = "roadmap_info"
    description: str = "Get information about upcoming product features and roadmap"
    
    def __init__(self, knowledge_base: ProductKnowledgeBase):
        super().__init__()
        self.knowledge_base = knowledge_base
    
    def _run(self, timeframe: str = "") -> str:
        """Get roadmap information."""
        roadmap_data = self.knowledge_base.get_roadmap_info(timeframe if timeframe else None)
        return json.dumps(roadmap_data, indent=2)


class ProductSupportAgent:
    """
    Product Support Agent using LangChain for comprehensive product assistance.
    """
    
    def __init__(self, config: AMPClientConfig, openai_api_key: str):
        self.config = config
        self.client = AMPClient(config)
        self.logger = logging.getLogger(f"support.product.{config.agent_id}")
        
        # Initialize knowledge base and tools
        self.knowledge_base = ProductKnowledgeBase()
        self.product_search_tool = ProductSearchTool(self.knowledge_base)
        self.feature_guidance_tool = FeatureGuidanceTool(self.knowledge_base)
        self.roadmap_tool = RoadmapTool(self.knowledge_base)
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4",
            temperature=0.2
        )
        
        # Create LangChain agent with tools
        self.tools = [
            self.product_search_tool,
            self.feature_guidance_tool,
            self.roadmap_tool
        ]
        
        # Product support prompt
        self.support_prompt = self._create_support_prompt()
        
        # Create agent executor
        self.agent_executor = self._create_agent_executor()
        
    async def start(self):
        """Start the product support agent."""
        await self._register_capabilities()
        
        connected = await self.client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        
        self.logger.info("Product Support Agent started successfully")
    
    async def stop(self):
        """Stop the product support agent."""
        await self.client.disconnect()
        self.logger.info("Product Support Agent stopped")
    
    async def _register_capabilities(self):
        """Register agent capabilities with AMP."""
        
        # Product inquiry handling capability
        product_inquiry_capability = Capability(
            id="product-inquiry-handling",
            version="1.0",
            description="Handle product questions, feature guidance, and usage support",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "subject": {"type": "string"},
                            "description": {"type": "string"},
                            "category": {"type": "string"},
                            "customer_info": {"type": "object"}
                        },
                        "required": ["subject", "description"]
                    }
                },
                "required": ["ticket"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "product_analysis": {"type": "object"},
                    "solution_guidance": {"type": "object"},
                    "documentation_links": {"type": "array"},
                    "training_recommendations": {"type": "array"},
                    "feature_requests": {"type": "array"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=10000,
                max_input_length=8000,
                supported_languages=["en"],
                min_confidence=0.8
            ),
            category="product-support",
            subcategories=["feature-guidance", "documentation", "training"]
        )
        
        self.client.register_capability(product_inquiry_capability, self.handle_product_inquiry)
        
        # Feature request processing capability
        feature_request_capability = Capability(
            id="feature-request-processing",
            version="1.0",
            description="Process and analyze feature requests and enhancement suggestions",
            input_schema={
                "type": "object",
                "properties": {
                    "request": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "use_case": {"type": "string"},
                            "priority": {"type": "string"}
                        },
                        "required": ["title", "description"]
                    },
                    "customer_info": {"type": "object"}
                },
                "required": ["request"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "request_analysis": {"type": "object"},
                    "feasibility_assessment": {"type": "object"},
                    "roadmap_alignment": {"type": "object"},
                    "alternative_solutions": {"type": "array"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=8000,
                max_input_length=5000
            )
        )
        
        self.client.register_capability(feature_request_capability, self.process_feature_request)
    
    async def handle_product_inquiry(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a product support inquiry with comprehensive guidance.
        
        Args:
            parameters: Contains ticket information and context
            
        Returns:
            Comprehensive product support response with guidance
        """
        try:
            ticket_data = parameters["ticket"]
            
            # Analyze the inquiry type
            inquiry_type = self._analyze_inquiry_type(ticket_data)
            
            # Use LangChain agent to process the inquiry
            agent_input = {
                "ticket_subject": ticket_data["subject"],
                "ticket_description": ticket_data["description"],
                "inquiry_type": inquiry_type,
                "customer_account_type": ticket_data.get("customer_info", {}).get("account_type", "standard")
            }
            
            agent_response = await self._run_product_agent(agent_input)
            
            # Generate comprehensive response
            response = {
                "ticket_id": ticket_data.get("id"),
                "agent_id": self.config.agent_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "product_analysis": {
                    "inquiry_type": inquiry_type,
                    "complexity": self._assess_complexity(ticket_data),
                    "relevant_features": self._identify_relevant_features(ticket_data),
                    "customer_context": self._analyze_customer_context(ticket_data)
                },
                "solution_guidance": self._extract_solution_guidance(agent_response),
                "documentation_links": self._suggest_documentation(ticket_data),
                "training_recommendations": self._suggest_training(ticket_data, inquiry_type),
                "feature_requests": self._identify_feature_requests(ticket_data),
                "follow_up_actions": [
                    "Provide step-by-step guidance",
                    "Share relevant documentation",
                    "Schedule follow-up if needed"
                ],
                "estimated_resolution_time": self._estimate_resolution_time(inquiry_type),
                "customer_success_tips": self._generate_success_tips(ticket_data)
            }
            
            self.logger.info(f"Processed product inquiry {ticket_data.get('id')} - Type: {inquiry_type}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling product inquiry: {e}")
            raise
    
    async def process_feature_request(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and analyze a feature request.
        
        Args:
            parameters: Contains feature request details
            
        Returns:
            Feature request analysis and recommendations
        """
        try:
            request_data = parameters["request"]
            customer_info = parameters.get("customer_info", {})
            
            # Analyze the feature request
            analysis = {
                "request_category": self._categorize_feature_request(request_data),
                "complexity_estimate": self._estimate_feature_complexity(request_data),
                "business_value": self._assess_business_value(request_data, customer_info),
                "technical_feasibility": self._assess_technical_feasibility(request_data)
            }
            
            # Check roadmap alignment
            roadmap_alignment = self._check_roadmap_alignment(request_data)
            
            # Suggest alternatives
            alternatives = self._suggest_alternatives(request_data)
            
            response = {
                "request_id": f"FR-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                "agent_id": self.config.agent_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "request_analysis": analysis,
                "feasibility_assessment": {
                    "technical_feasibility": analysis["technical_feasibility"],
                    "estimated_effort": self._estimate_development_effort(request_data),
                    "dependencies": self._identify_dependencies(request_data),
                    "risks": self._identify_risks(request_data)
                },
                "roadmap_alignment": roadmap_alignment,
                "alternative_solutions": alternatives,
                "recommendation": self._generate_recommendation(analysis, roadmap_alignment),
                "next_steps": self._suggest_next_steps(analysis)
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing feature request: {e}")
            raise
    
    def _analyze_inquiry_type(self, ticket_data: Dict[str, Any]) -> str:
        """Analyze the type of product inquiry."""
        subject = ticket_data["subject"].lower()
        description = ticket_data["description"].lower()
        text = f"{subject} {description}"
        
        if any(word in text for word in ["how to", "tutorial", "guide", "usage"]):
            return "usage_guidance"
        elif any(word in text for word in ["feature", "request", "enhancement", "add", "improve"]):
            return "feature_request"
        elif any(word in text for word in ["integration", "api", "connect", "webhook"]):
            return "integration_support"
        elif any(word in text for word in ["best practice", "recommendation", "optimize"]):
            return "best_practices"
        elif any(word in text for word in ["training", "onboarding", "getting started"]):
            return "training_request"
        else:
            return "general_product"
    
    def _assess_complexity(self, ticket_data: Dict[str, Any]) -> str:
        """Assess the complexity of the inquiry."""
        description = ticket_data["description"].lower()
        
        if any(word in description for word in ["advanced", "complex", "custom", "enterprise"]):
            return "high"
        elif any(word in description for word in ["integration", "api", "workflow"]):
            return "medium"
        else:
            return "low"
    
    def _identify_relevant_features(self, ticket_data: Dict[str, Any]) -> List[str]:
        """Identify relevant product features."""
        text = f"{ticket_data['subject']} {ticket_data['description']}".lower()
        relevant_features = []
        
        for feature_id, feature_data in self.knowledge_base.features.items():
            if any(cap.lower() in text for cap in feature_data["capabilities"]):
                relevant_features.append(feature_id)
        
        return relevant_features
    
    def _analyze_customer_context(self, ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze customer context for personalized support."""
        customer_info = ticket_data.get("customer_info", {})
        
        return {
            "account_type": customer_info.get("account_type", "standard"),
            "experience_level": self._infer_experience_level(ticket_data),
            "use_case": self._infer_use_case(ticket_data),
            "urgency": self._assess_urgency(ticket_data)
        }
    
    def _infer_experience_level(self, ticket_data: Dict[str, Any]) -> str:
        """Infer customer experience level."""
        text = f"{ticket_data['subject']} {ticket_data['description']}".lower()
        
        if any(word in text for word in ["beginner", "new", "getting started", "first time"]):
            return "beginner"
        elif any(word in text for word in ["advanced", "expert", "complex", "custom"]):
            return "advanced"
        else:
            return "intermediate"
    
    def _infer_use_case(self, ticket_data: Dict[str, Any]) -> str:
        """Infer the customer's use case."""
        text = f"{ticket_data['subject']} {ticket_data['description']}".lower()
        
        if any(word in text for word in ["team", "collaboration", "organization"]):
            return "team_collaboration"
        elif any(word in text for word in ["integration", "api", "connect"]):
            return "system_integration"
        elif any(word in text for word in ["report", "analytics", "data"]):
            return "reporting_analytics"
        else:
            return "general_usage"
    
    def _assess_urgency(self, ticket_data: Dict[str, Any]) -> str:
        """Assess urgency of the inquiry."""
        text = f"{ticket_data['subject']} {ticket_data['description']}".lower()
        
        if any(word in text for word in ["urgent", "asap", "deadline", "critical"]):
            return "high"
        elif any(word in text for word in ["soon", "priority", "important"]):
            return "medium"
        else:
            return "low"
    
    async def _run_product_agent(self, agent_input: Dict[str, Any]) -> str:
        """Run the LangChain agent to process the inquiry."""
        try:
            query = f"""
            Product Support Inquiry:
            Subject: {agent_input['ticket_subject']}
            Description: {agent_input['ticket_description']}
            Type: {agent_input['inquiry_type']}
            Customer Type: {agent_input['customer_account_type']}
            
            Please provide comprehensive product support including:
            1. Specific guidance for this inquiry
            2. Relevant features and capabilities
            3. Documentation and tutorial references
            4. Best practices and recommendations
            """
            
            result = self.agent_executor.invoke({"input": query})
            return result.get("output", "No specific guidance available")
            
        except Exception as e:
            self.logger.warning(f"Agent execution failed: {e}")
            return "Standard product support guidance available through documentation"
    
    def _extract_solution_guidance(self, agent_response: str) -> Dict[str, Any]:
        """Extract solution guidance from agent response."""
        return {
            "step_by_step_guide": self._extract_steps(agent_response),
            "key_concepts": self._extract_concepts(agent_response),
            "common_pitfalls": self._extract_pitfalls(agent_response),
            "pro_tips": self._extract_tips(agent_response)
        }
    
    def _extract_steps(self, response: str) -> List[str]:
        """Extract step-by-step guidance from response."""
        # Simple extraction - in production, use more sophisticated NLP
        lines = response.split('\n')
        steps = []
        for line in lines:
            if line.strip().startswith(('1.', '2.', '3.', '-', '*')):
                steps.append(line.strip())
        return steps[:10]  # Limit to 10 steps
    
    def _extract_concepts(self, response: str) -> List[str]:
        """Extract key concepts from response."""
        # Simplified concept extraction
        concepts = []
        if "user management" in response.lower():
            concepts.append("User Management")
        if "api" in response.lower():
            concepts.append("API Integration")
        if "reporting" in response.lower():
            concepts.append("Reporting & Analytics")
        return concepts
    
    def _extract_pitfalls(self, response: str) -> List[str]:
        """Extract common pitfalls from response."""
        pitfalls = [
            "Not completing the initial setup wizard",
            "Skipping permission configuration",
            "Overlooking API rate limits"
        ]
        return pitfalls[:3]
    
    def _extract_tips(self, response: str) -> List[str]:
        """Extract pro tips from response."""
        tips = [
            "Start with basic features before advanced ones",
            "Regularly review security settings",
            "Keep API credentials secure"
        ]
        return tips[:3]
    
    def _suggest_documentation(self, ticket_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest relevant documentation."""
        relevant_features = self._identify_relevant_features(ticket_data)
        docs = []
        
        for feature in relevant_features:
            feature_info = self.knowledge_base.get_feature_info(feature)
            if feature_info:
                docs.append({
                    "title": f"{feature.replace('_', ' ').title()} Documentation",
                    "url": feature_info["documentation"],
                    "type": "documentation"
                })
                
                for tutorial in feature_info["tutorials"]:
                    docs.append({
                        "title": f"{feature.replace('_', ' ').title()} Tutorial",
                        "url": tutorial,
                        "type": "tutorial"
                    })
        
        return docs[:5]  # Limit to 5 suggestions
    
    def _suggest_training(self, ticket_data: Dict[str, Any], inquiry_type: str) -> List[Dict[str, str]]:
        """Suggest training recommendations."""
        customer_context = self._analyze_customer_context(ticket_data)
        
        if customer_context["experience_level"] == "beginner":
            return [
                {
                    "title": "Getting Started Guide",
                    "description": "Complete onboarding tutorial",
                    "duration": "30 minutes",
                    "type": "self_paced"
                },
                {
                    "title": "Basic Features Overview",
                    "description": "Learn core product capabilities",
                    "duration": "45 minutes",
                    "type": "video_tutorial"
                }
            ]
        elif inquiry_type == "integration_support":
            return [
                {
                    "title": "API Integration Workshop",
                    "description": "Hands-on API integration training",
                    "duration": "2 hours",
                    "type": "live_session"
                }
            ]
        
        return []
    
    def _identify_feature_requests(self, ticket_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential feature requests in the inquiry."""
        description = ticket_data["description"].lower()
        requests = []
        
        if any(word in description for word in ["wish", "would like", "need", "request"]):
            requests.append({
                "type": "enhancement",
                "description": "Potential feature enhancement identified",
                "priority": "medium"
            })
        
        return requests
    
    def _estimate_resolution_time(self, inquiry_type: str) -> str:
        """Estimate resolution time based on inquiry type."""
        time_estimates = {
            "usage_guidance": "1-2 hours",
            "feature_request": "1-2 weeks (review process)",
            "integration_support": "4-8 hours",
            "best_practices": "2-4 hours",
            "training_request": "24-48 hours",
            "general_product": "2-4 hours"
        }
        return time_estimates.get(inquiry_type, "2-4 hours")
    
    def _generate_success_tips(self, ticket_data: Dict[str, Any]) -> List[str]:
        """Generate customer success tips."""
        return [
            "Start with our quick start guide for best results",
            "Join our community forum for peer support",
            "Schedule regular check-ins with your account manager",
            "Keep your team trained on new features"
        ]
    
    # Feature request processing methods
    def _categorize_feature_request(self, request_data: Dict[str, Any]) -> str:
        """Categorize the feature request."""
        description = request_data["description"].lower()
        
        if any(word in description for word in ["ui", "interface", "design", "user experience"]):
            return "ui_ux_enhancement"
        elif any(word in description for word in ["api", "integration", "webhook"]):
            return "api_enhancement"
        elif any(word in description for word in ["report", "analytics", "dashboard"]):
            return "reporting_feature"
        elif any(word in description for word in ["security", "permission", "access"]):
            return "security_feature"
        else:
            return "general_enhancement"
    
    def _estimate_feature_complexity(self, request_data: Dict[str, Any]) -> str:
        """Estimate complexity of the feature request."""
        description = request_data["description"].lower()
        
        if any(word in description for word in ["simple", "basic", "quick"]):
            return "low"
        elif any(word in description for word in ["complex", "advanced", "enterprise"]):
            return "high"
        else:
            return "medium"
    
    def _assess_business_value(self, request_data: Dict[str, Any], customer_info: Dict[str, Any]) -> str:
        """Assess business value of the feature request."""
        # Consider customer type and request content
        if customer_info.get("account_type") == "enterprise":
            return "high"
        elif "productivity" in request_data["description"].lower():
            return "medium"
        else:
            return "low"
    
    def _assess_technical_feasibility(self, request_data: Dict[str, Any]) -> str:
        """Assess technical feasibility."""
        # Simplified assessment based on description
        description = request_data["description"].lower()
        
        if any(word in description for word in ["impossible", "complete rewrite", "fundamental change"]):
            return "low"
        elif any(word in description for word in ["easy", "simple", "minor change"]):
            return "high"
        else:
            return "medium"
    
    def _check_roadmap_alignment(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check alignment with product roadmap."""
        roadmap = self.knowledge_base.get_roadmap_info()
        
        # Simple alignment check
        description = request_data["description"].lower()
        aligned_quarters = []
        
        for quarter, features in roadmap.items():
            for feature in features:
                if any(word in description for word in feature.lower().split()):
                    aligned_quarters.append(quarter)
        
        return {
            "is_aligned": len(aligned_quarters) > 0,
            "aligned_quarters": aligned_quarters,
            "priority": "high" if aligned_quarters else "medium"
        }
    
    def _suggest_alternatives(self, request_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest alternative solutions."""
        return [
            {
                "title": "Existing Feature Usage",
                "description": "Check if current features can meet your needs with proper configuration"
            },
            {
                "title": "Third-party Integration",
                "description": "Consider integrating with specialized third-party tools"
            },
            {
                "title": "Custom Development",
                "description": "Implement custom solution using our API"
            }
        ]
    
    def _estimate_development_effort(self, request_data: Dict[str, Any]) -> str:
        """Estimate development effort."""
        complexity = self._estimate_feature_complexity(request_data)
        
        effort_map = {
            "low": "1-2 weeks",
            "medium": "4-8 weeks", 
            "high": "3-6 months"
        }
        
        return effort_map.get(complexity, "4-8 weeks")
    
    def _identify_dependencies(self, request_data: Dict[str, Any]) -> List[str]:
        """Identify feature dependencies."""
        return [
            "Core platform updates",
            "API stability requirements",
            "Security review process"
        ]
    
    def _identify_risks(self, request_data: Dict[str, Any]) -> List[str]:
        """Identify implementation risks."""
        return [
            "Potential impact on existing functionality",
            "Increased system complexity",
            "Additional maintenance overhead"
        ]
    
    def _generate_recommendation(self, analysis: Dict[str, Any], roadmap_alignment: Dict[str, Any]) -> str:
        """Generate overall recommendation."""
        if roadmap_alignment["is_aligned"] and analysis["business_value"] == "high":
            return "Recommend for inclusion in upcoming roadmap"
        elif analysis["technical_feasibility"] == "high" and analysis["complexity_estimate"] == "low":
            return "Consider for quick wins implementation"
        else:
            return "Recommend further analysis and customer validation"
    
    def _suggest_next_steps(self, analysis: Dict[str, Any]) -> List[str]:
        """Suggest next steps for the feature request."""
        return [
            "Document detailed requirements",
            "Conduct customer validation",
            "Technical feasibility review",
            "Prioritize against current roadmap"
        ]
    
    def _create_support_prompt(self) -> ChatPromptTemplate:
        """Create the product support prompt template."""
        system_template = """You are an expert product support specialist. Help customers with product questions, feature usage, and provide comprehensive guidance.

Your knowledge includes:
- All product features and capabilities
- Best practices for product usage  
- Integration patterns and API usage
- Troubleshooting common issues
- Training and onboarding guidance

Always provide:
1. Clear, actionable guidance
2. Relevant documentation references
3. Best practices and tips
4. Step-by-step instructions when needed
5. Alternative approaches when applicable

Focus on customer success and positive outcomes."""

        human_template = """{input}"""

        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template(human_template)
        ])
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create the LangChain agent executor."""
        try:
            agent = create_openai_functions_agent(
                llm=self.llm,
                tools=self.tools,
                prompt=self.support_prompt
            )
            
            return AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,
                handle_parsing_errors=True
            )
        except Exception as e:
            self.logger.warning(f"Failed to create agent executor: {e}")
            # Return a simple fallback
            return None


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        config = AMPClientConfig(
            agent_id="product-support-001",
            agent_name="Product Support Agent",
            framework="langchain",
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:8000"
        )
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        agent = ProductSupportAgent(config, openai_api_key)
        
        try:
            await agent.start()
            print("Product Support Agent is running...")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await agent.stop()
    
    asyncio.run(main())