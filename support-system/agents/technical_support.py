"""
Technical Support Agent using CrewAI.

This agent handles technical troubleshooting, diagnostics, and provides
step-by-step solutions for technical issues.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone

# CrewAI imports
from crewai import Agent, Task, Crew
from crewai.tools import BaseTool
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# AMP imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))

from amp_client import AMPClient, AMPClientConfig
from amp_types import Capability, CapabilityConstraints, TransportType

# Support system imports
from ..support_types import Ticket, TicketCategory, TicketPriority, TicketStatus


class TechnicalDiagnosticTool(BaseTool):
    """Tool for technical diagnostics and troubleshooting."""
    
    name: str = "technical_diagnostic"
    description: str = "Analyze technical issues and provide diagnostic information"
    
    def _run(self, issue_description: str, error_details: str = "") -> str:
        """Run technical diagnostic analysis."""
        # Simulate diagnostic process
        diagnostics = {
            "issue_type": self._classify_technical_issue(issue_description),
            "severity": self._assess_severity(issue_description + " " + error_details),
            "common_causes": self._identify_common_causes(issue_description),
            "diagnostic_questions": self._generate_diagnostic_questions(issue_description),
            "initial_checks": self._suggest_initial_checks(issue_description)
        }
        
        return json.dumps(diagnostics, indent=2)
    
    def _classify_technical_issue(self, description: str) -> str:
        """Classify the type of technical issue."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["login", "password", "authentication", "sign in"]):
            return "authentication"
        elif any(word in description_lower for word in ["api", "integration", "webhook", "endpoint"]):
            return "api_integration"
        elif any(word in description_lower for word in ["performance", "slow", "timeout", "lag"]):
            return "performance"
        elif any(word in description_lower for word in ["error", "exception", "crash", "bug"]):
            return "application_error"
        elif any(word in description_lower for word in ["network", "connection", "connectivity"]):
            return "network"
        else:
            return "general_technical"
    
    def _assess_severity(self, description: str) -> str:
        """Assess the severity of the technical issue."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["critical", "down", "outage", "emergency"]):
            return "critical"
        elif any(word in description_lower for word in ["urgent", "blocking", "production"]):
            return "high"
        elif any(word in description_lower for word in ["important", "affecting"]):
            return "medium"
        else:
            return "low"
    
    def _identify_common_causes(self, description: str) -> List[str]:
        """Identify common causes for the issue."""
        issue_type = self._classify_technical_issue(description)
        
        causes_map = {
            "authentication": [
                "Incorrect credentials",
                "Account locked or suspended",
                "Browser cache/cookies issues",
                "Two-factor authentication problems",
                "Password reset required"
            ],
            "api_integration": [
                "Invalid API key or credentials", 
                "Rate limiting exceeded",
                "Incorrect endpoint URL",
                "Request format issues",
                "Server-side API changes"
            ],
            "performance": [
                "High server load",
                "Network latency",
                "Large data processing",
                "Inefficient queries",
                "Resource constraints"
            ],
            "application_error": [
                "Software bug",
                "Compatibility issues",
                "Configuration problems",
                "Missing dependencies",
                "Data corruption"
            ],
            "network": [
                "Internet connectivity issues",
                "Firewall blocking",
                "DNS resolution problems",
                "Proxy configuration",
                "ISP-related issues"
            ]
        }
        
        return causes_map.get(issue_type, ["Unknown technical issue"])
    
    def _generate_diagnostic_questions(self, description: str) -> List[str]:
        """Generate diagnostic questions to gather more information."""
        issue_type = self._classify_technical_issue(description)
        
        questions_map = {
            "authentication": [
                "What error message do you see when trying to log in?",
                "Have you recently changed your password?",
                "Are you using two-factor authentication?",
                "Which browser are you using?",
                "Have you tried clearing your browser cache?"
            ],
            "api_integration": [
                "What is the exact error code/message you're receiving?",
                "Which API endpoint are you calling?",
                "Can you share the request headers and payload?",
                "When did this issue start occurring?",
                "Are you making requests within the rate limits?"
            ],
            "performance": [
                "How long has the system been running slowly?",
                "Is the slowness consistent or intermittent?",
                "Which specific features or pages are affected?",
                "How many users are currently on the system?",
                "Have you noticed any patterns to when it's slow?"
            ]
        }
        
        return questions_map.get(issue_type, [
            "Can you provide more details about the issue?",
            "When did you first notice this problem?",
            "Have you tried any troubleshooting steps already?"
        ])
    
    def _suggest_initial_checks(self, description: str) -> List[str]:
        """Suggest initial troubleshooting checks."""
        issue_type = self._classify_technical_issue(description)
        
        checks_map = {
            "authentication": [
                "Verify username and password are correct",
                "Try logging in from an incognito browser window",
                "Clear browser cache and cookies",
                "Check if account is locked",
                "Try password reset if needed"
            ],
            "api_integration": [
                "Verify API credentials are correct and active",
                "Check API endpoint URL is correct",
                "Validate request format against API documentation",
                "Test with a simple API call (like status check)",
                "Check API rate limits and usage"
            ],
            "performance": [
                "Check system status page for any known issues",
                "Try accessing from a different network/device",
                "Clear browser cache and try again",
                "Monitor system during off-peak hours",
                "Check for any recent system changes"
            ]
        }
        
        return checks_map.get(issue_type, [
            "Restart the application/browser",
            "Check internet connection",
            "Try from a different device or browser"
        ])


class SolutionSearchTool(BaseTool):
    """Tool for searching existing solutions and knowledge base."""
    
    name: str = "solution_search"
    description: str = "Search for existing solutions in knowledge base"
    
    def _run(self, query: str, issue_type: str = "") -> str:
        """Search for relevant solutions."""
        # Simulate knowledge base search
        solutions = {
            "query": query,
            "matching_articles": self._get_matching_articles(query, issue_type),
            "related_solutions": self._get_related_solutions(query),
            "troubleshooting_guides": self._get_troubleshooting_guides(issue_type)
        }
        
        return json.dumps(solutions, indent=2)
    
    def _get_matching_articles(self, query: str, issue_type: str) -> List[Dict[str, str]]:
        """Get matching knowledge base articles."""
        # Simulated knowledge base articles
        return [
            {
                "id": "kb-001",
                "title": f"Troubleshooting {issue_type.replace('_', ' ').title()} Issues",
                "summary": f"Step-by-step guide for resolving {issue_type} problems",
                "relevance": 0.9,
                "url": f"/kb/articles/{issue_type}-troubleshooting"
            },
            {
                "id": "kb-002", 
                "title": "Common Error Messages and Solutions",
                "summary": "Reference guide for frequently encountered error messages",
                "relevance": 0.7,
                "url": "/kb/articles/common-errors"
            }
        ]
    
    def _get_related_solutions(self, query: str) -> List[Dict[str, str]]:
        """Get related solutions."""
        return [
            {
                "solution_id": "sol-001",
                "title": "Quick Fix for Authentication Issues",
                "steps": ["Clear browser cache", "Try incognito mode", "Reset password"],
                "success_rate": 0.85
            }
        ]
    
    def _get_troubleshooting_guides(self, issue_type: str) -> List[Dict[str, str]]:
        """Get relevant troubleshooting guides."""
        return [
            {
                "guide_id": "guide-001",
                "title": f"{issue_type.replace('_', ' ').title()} Troubleshooting Guide",
                "description": f"Comprehensive troubleshooting steps for {issue_type} issues",
                "estimated_time": "10-15 minutes"
            }
        ]


class EscalationAnalysisTool(BaseTool):
    """Tool for analyzing if an issue needs escalation."""
    
    name: str = "escalation_analysis"
    description: str = "Analyze if a technical issue requires escalation"
    
    def _run(self, issue_description: str, customer_type: str = "standard", 
             attempts_made: int = 0) -> str:
        """Analyze escalation requirements."""
        
        analysis = {
            "should_escalate": self._should_escalate(issue_description, customer_type, attempts_made),
            "escalation_reason": self._get_escalation_reason(issue_description, attempts_made),
            "recommended_escalation_level": self._get_escalation_level(customer_type),
            "urgency_factors": self._get_urgency_factors(issue_description),
            "customer_impact": self._assess_customer_impact(issue_description, customer_type)
        }
        
        return json.dumps(analysis, indent=2)
    
    def _should_escalate(self, description: str, customer_type: str, attempts: int) -> bool:
        """Determine if escalation is needed."""
        description_lower = description.lower()
        
        # Critical issues always escalate
        if any(word in description_lower for word in ["critical", "emergency", "outage", "down"]):
            return True
        
        # Enterprise customers get faster escalation
        if customer_type == "enterprise" and attempts >= 1:
            return True
        
        # Standard escalation after multiple attempts
        if attempts >= 3:
            return True
        
        # Complex technical issues
        if any(word in description_lower for word in ["api", "integration", "custom", "development"]):
            return attempts >= 2
        
        return False
    
    def _get_escalation_reason(self, description: str, attempts: int) -> str:
        """Get reason for escalation."""
        if attempts >= 3:
            return "Multiple resolution attempts unsuccessful"
        elif "critical" in description.lower():
            return "Critical issue requiring immediate attention"
        elif "enterprise" in description.lower():
            return "Enterprise customer requiring specialized support"
        else:
            return "Complex technical issue requiring expert analysis"
    
    def _get_escalation_level(self, customer_type: str) -> str:
        """Get recommended escalation level."""
        if customer_type == "enterprise":
            return "senior_technical_specialist"
        else:
            return "technical_team_lead"
    
    def _get_urgency_factors(self, description: str) -> List[str]:
        """Identify urgency factors."""
        factors = []
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["production", "live", "customer-facing"]):
            factors.append("Production environment affected")
        
        if any(word in description_lower for word in ["revenue", "business", "sales"]):
            factors.append("Business impact")
        
        if any(word in description_lower for word in ["security", "breach", "vulnerability"]):
            factors.append("Security implications")
        
        return factors
    
    def _assess_customer_impact(self, description: str, customer_type: str) -> str:
        """Assess the impact on the customer."""
        if customer_type == "enterprise":
            return "high"
        elif any(word in description.lower() for word in ["blocking", "critical", "urgent"]):
            return "high" 
        elif any(word in description.lower() for word in ["important", "affecting"]):
            return "medium"
        else:
            return "low"


class TechnicalSupportAgent:
    """
    Technical Support Agent using CrewAI for advanced technical troubleshooting
    and collaborative problem-solving.
    """
    
    def __init__(self, config: AMPClientConfig, openai_api_key: str):
        self.config = config
        self.client = AMPClient(config)
        self.logger = logging.getLogger(f"support.technical.{config.agent_id}")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            model_name="gpt-4",
            temperature=0.2
        )
        
        # Initialize tools
        self.diagnostic_tool = TechnicalDiagnosticTool()
        self.solution_tool = SolutionSearchTool()
        self.escalation_tool = EscalationAnalysisTool()
        
        # Create CrewAI agents
        self.technical_analyst = Agent(
            role="Technical Analyst",
            goal="Analyze technical issues and provide accurate diagnostics",
            backstory="Expert technical analyst with deep knowledge of system architecture and common technical issues",
            llm=self.llm,
            tools=[self.diagnostic_tool, self.solution_tool],
            verbose=True
        )
        
        self.solution_engineer = Agent(
            role="Solution Engineer", 
            goal="Develop step-by-step solutions for technical problems",
            backstory="Experienced solution engineer who creates clear, actionable troubleshooting guides",
            llm=self.llm,
            tools=[self.solution_tool],
            verbose=True
        )
        
        self.escalation_specialist = Agent(
            role="Escalation Specialist",
            goal="Determine when issues need escalation and to whom",
            backstory="Specialist in triaging complex technical issues and managing escalation workflows",
            llm=self.llm,
            tools=[self.escalation_tool],
            verbose=True
        )
        
        # Create CrewAI crew
        self.support_crew = Crew(
            agents=[self.technical_analyst, self.solution_engineer, self.escalation_specialist],
            verbose=2
        )
        
    async def start(self):
        """Start the technical support agent."""
        await self._register_capabilities()
        
        connected = await self.client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        
        self.logger.info("Technical Support Agent started successfully")
    
    async def stop(self):
        """Stop the technical support agent."""
        await self.client.disconnect()
        self.logger.info("Technical Support Agent stopped")
    
    async def _register_capabilities(self):
        """Register agent capabilities with AMP."""
        
        # Technical troubleshooting capability
        troubleshoot_capability = Capability(
            id="technical-troubleshooting",
            version="1.0",
            description="Provide comprehensive technical troubleshooting and solutions",
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
                            "priority": {"type": "string"},
                            "customer_info": {"type": "object"},
                            "previous_attempts": {
                                "type": "array",
                                "items": {"type": "object"}
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
                    "diagnosis": {"type": "object"},
                    "solution_steps": {"type": "array"},
                    "escalation_recommendation": {"type": "object"},
                    "follow_up_actions": {"type": "array"},
                    "estimated_resolution_time": {"type": "string"},
                    "confidence_level": {"type": "number"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=15000,
                max_input_length=15000,
                supported_languages=["en"],
                min_confidence=0.7
            ),
            category="technical-support",
            subcategories=["troubleshooting", "diagnostics", "problem-solving"]
        )
        
        self.client.register_capability(troubleshoot_capability, self.handle_technical_issue)
        
        # Technical diagnostics capability
        diagnostics_capability = Capability(
            id="technical-diagnostics",
            version="1.0",
            description="Perform technical diagnostics and analysis",
            input_schema={
                "type": "object",
                "properties": {
                    "issue_description": {"type": "string"},
                    "error_details": {"type": "string"},
                    "system_info": {"type": "object"}
                },
                "required": ["issue_description"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "diagnostic_report": {"type": "object"},
                    "recommended_checks": {"type": "array"},
                    "risk_assessment": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=10000,
                max_input_length=10000
            )
        )
        
        self.client.register_capability(diagnostics_capability, self.perform_diagnostics)
    
    async def handle_technical_issue(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a technical support ticket with comprehensive analysis and solution.
        
        Args:
            parameters: Contains ticket information and context
            
        Returns:
            Comprehensive technical support response with solutions
        """
        try:
            ticket_data = parameters["ticket"]
            
            # Create tasks for the crew
            analysis_task = Task(
                description=f"""
                Analyze this technical issue and provide comprehensive diagnostics:
                
                Subject: {ticket_data['subject']}
                Description: {ticket_data['description']}
                Category: {ticket_data.get('category', 'technical')}
                Priority: {ticket_data.get('priority', 'medium')}
                
                Provide:
                1. Issue classification and severity assessment
                2. Potential root causes 
                3. Diagnostic questions to ask the customer
                4. Initial troubleshooting checks
                """,
                agent=self.technical_analyst,
                expected_output="Detailed technical analysis with diagnostics and initial assessment"
            )
            
            solution_task = Task(
                description=f"""
                Based on the technical analysis, create a comprehensive solution plan:
                
                Issue: {ticket_data['subject']}
                Details: {ticket_data['description']}
                
                Provide:
                1. Step-by-step troubleshooting guide
                2. Alternative solutions if initial steps fail
                3. Preventive measures for the future
                4. Knowledge base references
                5. Estimated resolution time
                """,
                agent=self.solution_engineer,
                expected_output="Detailed step-by-step solution guide with alternatives"
            )
            
            escalation_task = Task(
                description=f"""
                Evaluate if this technical issue requires escalation:
                
                Issue: {ticket_data['subject']}
                Details: {ticket_data['description']}
                Customer Type: {ticket_data.get('customer_info', {}).get('account_type', 'standard')}
                Previous Attempts: {len(ticket_data.get('previous_attempts', []))}
                
                Analyze:
                1. Complexity level of the issue
                2. Whether it requires specialized expertise
                3. Customer impact and urgency
                4. Escalation recommendation with justification
                """,
                agent=self.escalation_specialist,
                expected_output="Escalation analysis with clear recommendation and reasoning"
            )
            
            # Execute the crew
            crew_result = self.support_crew.kickoff(
                tasks=[analysis_task, solution_task, escalation_task]
            )
            
            # Process results
            analysis_result = analysis_task.output
            solution_result = solution_task.output
            escalation_result = escalation_task.output
            
            # Parse and structure the response
            response = {
                "ticket_id": ticket_data.get("id"),
                "agent_id": self.config.agent_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "diagnosis": {
                    "issue_type": self._extract_issue_type(analysis_result),
                    "severity": self._extract_severity(analysis_result),
                    "root_cause_analysis": analysis_result,
                    "confidence_level": 0.85
                },
                "solution_steps": self._parse_solution_steps(solution_result),
                "escalation_recommendation": self._parse_escalation_recommendation(escalation_result),
                "follow_up_actions": [
                    "Monitor customer feedback on solution effectiveness",
                    "Check if issue persists after 24 hours",
                    "Update knowledge base if new solution pattern emerges"
                ],
                "estimated_resolution_time": self._estimate_resolution_time(ticket_data),
                "additional_resources": {
                    "knowledge_base_articles": self._suggest_kb_articles(ticket_data),
                    "troubleshooting_guides": self._suggest_guides(ticket_data),
                    "escalation_contacts": self._get_escalation_contacts(ticket_data)
                },
                "customer_communication": {
                    "response_template": self._generate_customer_response(solution_result),
                    "follow_up_schedule": self._suggest_follow_up_schedule(ticket_data)
                }
            }
            
            self.logger.info(f"Processed technical ticket {ticket_data.get('id')} successfully")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling technical issue: {e}")
            raise
    
    async def perform_diagnostics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform focused technical diagnostics.
        
        Args:
            parameters: Contains issue description and error details
            
        Returns:
            Diagnostic report with recommendations
        """
        try:
            issue_description = parameters["issue_description"]
            error_details = parameters.get("error_details", "")
            system_info = parameters.get("system_info", {})
            
            # Use diagnostic tool
            diagnostic_result = self.diagnostic_tool._run(issue_description, error_details)
            diagnostic_data = json.loads(diagnostic_result)
            
            # Enhance with risk assessment
            risk_assessment = self._assess_risk(issue_description, error_details)
            
            response = {
                "diagnostic_report": diagnostic_data,
                "recommended_checks": diagnostic_data.get("initial_checks", []),
                "risk_assessment": risk_assessment,
                "next_steps": self._recommend_next_steps(diagnostic_data),
                "agent_id": self.config.agent_id,
                "diagnostic_timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error performing diagnostics: {e}")
            raise
    
    def _extract_issue_type(self, analysis: str) -> str:
        """Extract issue type from analysis."""
        if "authentication" in analysis.lower():
            return "authentication"
        elif "api" in analysis.lower():
            return "api_integration"
        elif "performance" in analysis.lower():
            return "performance"
        else:
            return "general_technical"
    
    def _extract_severity(self, analysis: str) -> str:
        """Extract severity from analysis."""
        if any(word in analysis.lower() for word in ["critical", "emergency"]):
            return "critical"
        elif any(word in analysis.lower() for word in ["high", "urgent"]):
            return "high"
        elif "medium" in analysis.lower():
            return "medium"
        else:
            return "low"
    
    def _parse_solution_steps(self, solution: str) -> List[Dict[str, Any]]:
        """Parse solution steps from crew output."""
        # Extract structured steps from the solution text
        steps = []
        lines = solution.split('\n')
        
        for i, line in enumerate(lines):
            if line.strip() and (line.startswith(str(i+1)) or line.startswith('-')):
                steps.append({
                    "step_number": len(steps) + 1,
                    "description": line.strip(),
                    "estimated_time": "5-10 minutes",
                    "difficulty": "medium",
                    "required_access": "user"
                })
        
        # If no structured steps found, create generic ones
        if not steps:
            steps = [
                {
                    "step_number": 1,
                    "description": "Follow the detailed solution provided by technical analysis",
                    "estimated_time": "10-15 minutes",
                    "difficulty": "medium",
                    "required_access": "user"
                }
            ]
        
        return steps
    
    def _parse_escalation_recommendation(self, escalation: str) -> Dict[str, Any]:
        """Parse escalation recommendation from crew output."""
        should_escalate = any(word in escalation.lower() for word in ["yes", "escalate", "recommend"])
        
        return {
            "should_escalate": should_escalate,
            "reason": escalation if should_escalate else "Standard technical support can handle this issue",
            "escalation_level": "senior_technical" if should_escalate else None,
            "urgency": "high" if "urgent" in escalation.lower() else "medium",
            "estimated_escalation_time": "1-2 hours" if should_escalate else None
        }
    
    def _estimate_resolution_time(self, ticket_data: Dict[str, Any]) -> str:
        """Estimate resolution time based on ticket data."""
        priority = ticket_data.get("priority", "medium")
        category = ticket_data.get("category", "general")
        
        if priority == "critical":
            return "1-4 hours"
        elif priority == "high":
            return "4-8 hours"
        elif category == "api_integration":
            return "8-24 hours"
        else:
            return "24-48 hours"
    
    def _suggest_kb_articles(self, ticket_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest relevant knowledge base articles."""
        return [
            {
                "title": "Common Technical Issues and Solutions",
                "url": "/kb/technical-issues",
                "relevance": 0.8
            },
            {
                "title": "API Integration Troubleshooting Guide",
                "url": "/kb/api-troubleshooting",
                "relevance": 0.7
            }
        ]
    
    def _suggest_guides(self, ticket_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Suggest troubleshooting guides."""
        return [
            {
                "title": "Step-by-Step Technical Troubleshooting",
                "url": "/guides/technical-troubleshooting",
                "estimated_time": "15-30 minutes"
            }
        ]
    
    def _get_escalation_contacts(self, ticket_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Get escalation contacts."""
        return [
            {
                "role": "Senior Technical Specialist",
                "contact": "escalation-technical@company.com",
                "availability": "24/7"
            }
        ]
    
    def _generate_customer_response(self, solution: str) -> str:
        """Generate customer-facing response."""
        return f"""
Thank you for contacting technical support. I've analyzed your issue and prepared a solution plan.

{solution}

I'll monitor this ticket closely and follow up to ensure the solution works for you. If you encounter any issues with these steps, please don't hesitate to reach out.

Best regards,
Technical Support Team
        """.strip()
    
    def _suggest_follow_up_schedule(self, ticket_data: Dict[str, Any]) -> List[str]:
        """Suggest follow-up schedule."""
        priority = ticket_data.get("priority", "medium")
        
        if priority in ["critical", "high"]:
            return ["1 hour", "4 hours", "24 hours"]
        else:
            return ["24 hours", "3 days"]
    
    def _assess_risk(self, issue_description: str, error_details: str) -> Dict[str, Any]:
        """Assess risk level of the issue."""
        text = (issue_description + " " + error_details).lower()
        
        risk_level = "low"
        risk_factors = []
        
        if any(word in text for word in ["security", "breach", "hack", "vulnerability"]):
            risk_level = "critical"
            risk_factors.append("Security implications")
        
        if any(word in text for word in ["production", "live", "customer-facing"]):
            risk_level = "high" if risk_level != "critical" else risk_level
            risk_factors.append("Production environment")
        
        if any(word in text for word in ["data", "database", "corruption"]):
            risk_level = "high" if risk_level not in ["critical"] else risk_level
            risk_factors.append("Data integrity concerns")
        
        return {
            "risk_level": risk_level,
            "risk_factors": risk_factors,
            "mitigation_priority": "immediate" if risk_level == "critical" else "standard"
        }
    
    def _recommend_next_steps(self, diagnostic_data: Dict[str, Any]) -> List[str]:
        """Recommend next steps based on diagnostics."""
        return [
            "Execute initial troubleshooting checks",
            "Gather additional information if needed",
            "Monitor system behavior",
            "Document findings for knowledge base"
        ]


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        config = AMPClientConfig(
            agent_id="technical-support-001",
            agent_name="Technical Support Agent",
            framework="crewai",
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:8000"
        )
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        agent = TechnicalSupportAgent(config, openai_api_key)
        
        try:
            await agent.start()
            print("Technical Support Agent is running...")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await agent.stop()
    
    asyncio.run(main())