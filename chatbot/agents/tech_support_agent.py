"""
Technical Support Agent for Multi-Agent Chatbot System

Handles technical issues, troubleshooting, and support escalations.
Uses LangChain for problem diagnosis and solution recommendation.
"""

import asyncio
import logging
import yaml
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum

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


class SeverityLevel(Enum):
    """Issue severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(Enum):
    """Support ticket status."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    CLOSED = "closed"


class TechnicalDiagnosis(BaseModel):
    """Technical issue diagnosis structure."""
    issue_category: str = Field(description="Category of the technical issue")
    severity: str = Field(description="Severity level: low, medium, high, critical")
    root_cause: str = Field(description="Likely root cause of the issue")
    symptoms: List[str] = Field(description="Identified symptoms")
    affected_components: List[str] = Field(description="System components affected")
    troubleshooting_steps: List[str] = Field(description="Recommended troubleshooting steps")
    escalation_needed: bool = Field(description="Whether escalation is needed")
    estimated_resolution_time: str = Field(description="Estimated time to resolve")


class SolutionRecommendation(BaseModel):
    """Solution recommendation structure."""
    solution_title: str = Field(description="Title of the recommended solution")
    difficulty: str = Field(description="Difficulty level: easy, medium, hard")
    steps: List[str] = Field(description="Step-by-step solution instructions")
    prerequisites: List[str] = Field(description="Prerequisites for this solution")
    success_probability: float = Field(description="Probability of success (0-1)")
    alternative_solutions: List[str] = Field(description="Alternative approaches")


class TechSupportAgent:
    """Technical support agent for handling technical issues."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.TechSupportAgent")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-4"),  # Use GPT-4 for better technical reasoning
            temperature=self.config.get("temperature", 0.1)  # Lower temperature for technical accuracy
        )
        
        # Load knowledge bases
        self.troubleshooting_kb = self._load_troubleshooting_kb()
        self.escalation_rules = self._load_escalation_rules()
        
        # Ticket tracking
        self.tickets: Dict[str, Dict[str, Any]] = {}
        self.ticket_counter = 1
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "agent_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("tech_support_agent", {})
        except FileNotFoundError:
            return {}
    
    def _load_troubleshooting_kb(self) -> List[Dict[str, Any]]:
        """Load troubleshooting knowledge base."""
        kb_path = Path(__file__).parent.parent / "config" / "troubleshooting_kb.yaml"
        
        try:
            with open(kb_path, 'r') as f:
                kb = yaml.safe_load(f)
                return kb.get("issues", [])
        except FileNotFoundError:
            return self._get_default_troubleshooting_kb()
    
    def _get_default_troubleshooting_kb(self) -> List[Dict[str, Any]]:
        """Get default troubleshooting knowledge base."""
        return [
            {
                "category": "login_issues",
                "keywords": ["login", "sign in", "password", "authentication", "access"],
                "common_causes": ["Wrong credentials", "Account locked", "Password expired", "Browser cache"],
                "solutions": [
                    "Verify username and password",
                    "Clear browser cache and cookies",
                    "Try incognito/private mode",
                    "Reset password if needed"
                ],
                "escalation_triggers": ["Account security concerns", "Multiple failed attempts"]
            },
            {
                "category": "performance_issues",
                "keywords": ["slow", "lag", "performance", "timeout", "loading"],
                "common_causes": ["Network issues", "Server overload", "Browser problems", "Large data sets"],
                "solutions": [
                    "Check internet connection",
                    "Try a different browser",
                    "Clear browser cache",
                    "Reduce data load if possible"
                ],
                "escalation_triggers": ["System-wide slowness", "Database performance"]
            },
            {
                "category": "functionality_errors",
                "keywords": ["error", "bug", "not working", "broken", "crash"],
                "common_causes": ["Software bug", "Browser compatibility", "Data corruption", "Configuration issue"],
                "solutions": [
                    "Refresh the page",
                    "Try a different browser",
                    "Clear browser data",
                    "Report the specific error message"
                ],
                "escalation_triggers": ["Data loss", "System crash", "Security issues"]
            },
            {
                "category": "integration_issues",
                "keywords": ["integration", "api", "sync", "connection", "third-party"],
                "common_causes": ["API changes", "Authentication issues", "Rate limiting", "Configuration"],
                "solutions": [
                    "Check API credentials",
                    "Verify integration settings",
                    "Test API connectivity",
                    "Review rate limits"
                ],
                "escalation_triggers": ["API security issues", "Data synchronization problems"]
            }
        ]
    
    def _load_escalation_rules(self) -> Dict[str, Any]:
        """Load escalation rules."""
        return {
            "auto_escalate_keywords": [
                "security", "breach", "hack", "data loss", "corruption",
                "outage", "down", "critical", "urgent", "emergency"
            ],
            "escalation_conditions": {
                "multiple_attempts": 3,
                "high_severity_immediate": True,
                "unresolved_time_hours": 24
            },
            "escalation_levels": [
                {"level": 1, "target": "senior-tech-support"},
                {"level": 2, "target": "engineering-team"},
                {"level": 3, "target": "emergency-response"}
            ]
        }
    
    async def diagnose_issue(self, user_input: str, context: Dict[str, Any] = None) -> TechnicalDiagnosis:
        """Diagnose technical issue from user input."""
        
        # Identify issue category
        issue_category = self._categorize_issue(user_input)
        kb_entry = next((item for item in self.troubleshooting_kb if item["category"] == issue_category), None)
        
        # Create diagnosis prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a technical support specialist with expertise in software troubleshooting.
            
            Analyze the user's technical issue and provide a comprehensive diagnosis.
            
            Known issue patterns:
            {self._format_kb_for_prompt()}
            
            Consider:
            - Issue severity and impact
            - Likely root causes
            - System components affected
            - Urgency indicators
            - Whether escalation is needed
            
            Provide structured diagnosis information."""),
            ("human", f"Technical issue reported: {user_input}\nContext: {context or {}}")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            
            # Determine severity
            severity = self._assess_severity(user_input)
            
            # Check for escalation triggers
            escalation_needed = self._should_escalate(user_input, severity)
            
            # Generate diagnosis
            diagnosis = TechnicalDiagnosis(
                issue_category=issue_category,
                severity=severity.value,
                root_cause=self._extract_likely_cause(user_input, kb_entry),
                symptoms=self._extract_symptoms(user_input),
                affected_components=self._identify_affected_components(user_input),
                troubleshooting_steps=self._generate_troubleshooting_steps(issue_category, kb_entry),
                escalation_needed=escalation_needed,
                estimated_resolution_time=self._estimate_resolution_time(severity, issue_category)
            )
            
            return diagnosis
            
        except Exception as e:
            self.logger.error(f"Issue diagnosis failed: {e}")
            return self._fallback_diagnosis(user_input)
    
    def _categorize_issue(self, user_input: str) -> str:
        """Categorize the technical issue."""
        user_lower = user_input.lower()
        
        for kb_entry in self.troubleshooting_kb:
            if any(keyword in user_lower for keyword in kb_entry["keywords"]):
                return kb_entry["category"]
        
        return "general_issue"
    
    def _assess_severity(self, user_input: str) -> SeverityLevel:
        """Assess issue severity."""
        user_lower = user_input.lower()
        
        critical_keywords = ["critical", "emergency", "outage", "down", "data loss", "security"]
        high_keywords = ["urgent", "important", "broken", "crash", "error"]
        medium_keywords = ["slow", "issue", "problem", "trouble"]
        
        if any(keyword in user_lower for keyword in critical_keywords):
            return SeverityLevel.CRITICAL
        elif any(keyword in user_lower for keyword in high_keywords):
            return SeverityLevel.HIGH
        elif any(keyword in user_lower for keyword in medium_keywords):
            return SeverityLevel.MEDIUM
        else:
            return SeverityLevel.LOW
    
    def _should_escalate(self, user_input: str, severity: SeverityLevel) -> bool:
        """Determine if issue should be escalated."""
        user_lower = user_input.lower()
        
        # Check for auto-escalation keywords
        auto_escalate = any(
            keyword in user_lower 
            for keyword in self.escalation_rules["auto_escalate_keywords"]
        )
        
        # High severity issues may need escalation
        severity_escalate = severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
        
        return auto_escalate or (severity_escalate and any(
            word in user_lower for word in ["can't fix", "tried everything", "still broken"]
        ))
    
    def _extract_likely_cause(self, user_input: str, kb_entry: Optional[Dict[str, Any]]) -> str:
        """Extract likely cause of the issue."""
        if kb_entry and kb_entry.get("common_causes"):
            # Simple keyword matching for cause identification
            for cause in kb_entry["common_causes"]:
                cause_keywords = cause.lower().split()
                if any(keyword in user_input.lower() for keyword in cause_keywords):
                    return cause
            return kb_entry["common_causes"][0]  # Default to first cause
        
        return "Unknown - requires further investigation"
    
    def _extract_symptoms(self, user_input: str) -> List[str]:
        """Extract symptoms from user input."""
        symptoms = []
        
        symptom_patterns = {
            "error messages": ["error", "message", "popup"],
            "slow performance": ["slow", "lag", "timeout"],
            "login failures": ["login", "sign in", "access denied"],
            "crashes": ["crash", "freeze", "stop working"],
            "data issues": ["missing", "corrupted", "lost"]
        }
        
        user_lower = user_input.lower()
        for symptom, keywords in symptom_patterns.items():
            if any(keyword in user_lower for keyword in keywords):
                symptoms.append(symptom)
        
        if not symptoms:
            symptoms.append("General malfunction")
        
        return symptoms
    
    def _identify_affected_components(self, user_input: str) -> List[str]:
        """Identify affected system components."""
        components = []
        
        component_keywords = {
            "authentication": ["login", "password", "auth", "sign in"],
            "database": ["data", "database", "save", "load"],
            "user interface": ["ui", "interface", "display", "screen"],
            "api": ["api", "integration", "sync", "connection"],
            "browser": ["browser", "chrome", "firefox", "safari"]
        }
        
        user_lower = user_input.lower()
        for component, keywords in component_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                components.append(component)
        
        if not components:
            components.append("application core")
        
        return components
    
    def _generate_troubleshooting_steps(self, category: str, kb_entry: Optional[Dict[str, Any]]) -> List[str]:
        """Generate troubleshooting steps."""
        if kb_entry and kb_entry.get("solutions"):
            return kb_entry["solutions"]
        
        # Default troubleshooting steps
        return [
            "Restart the application",
            "Clear browser cache and cookies",
            "Try using a different browser",
            "Check internet connection",
            "Contact support if issue persists"
        ]
    
    def _estimate_resolution_time(self, severity: SeverityLevel, category: str) -> str:
        """Estimate resolution time."""
        time_estimates = {
            SeverityLevel.CRITICAL: "Immediate - 1 hour",
            SeverityLevel.HIGH: "1-4 hours",
            SeverityLevel.MEDIUM: "4-24 hours",
            SeverityLevel.LOW: "1-3 business days"
        }
        
        return time_estimates.get(severity, "Unknown")
    
    def _fallback_diagnosis(self, user_input: str) -> TechnicalDiagnosis:
        """Fallback diagnosis when AI analysis fails."""
        return TechnicalDiagnosis(
            issue_category="general_issue",
            severity=SeverityLevel.MEDIUM.value,
            root_cause="Requires further investigation",
            symptoms=["User-reported issue"],
            affected_components=["application"],
            troubleshooting_steps=[
                "Document the exact issue",
                "Note any error messages",
                "Try basic troubleshooting",
                "Contact support for assistance"
            ],
            escalation_needed=False,
            estimated_resolution_time="4-24 hours"
        )
    
    def _format_kb_for_prompt(self) -> str:
        """Format knowledge base for LLM prompt."""
        formatted = []
        for entry in self.troubleshooting_kb:
            formatted.append(f"Category: {entry['category']}")
            formatted.append(f"Keywords: {', '.join(entry['keywords'])}")
            formatted.append(f"Common causes: {', '.join(entry['common_causes'])}")
            formatted.append("")
        return "\n".join(formatted)
    
    async def create_support_ticket(self, session_id: str, diagnosis: TechnicalDiagnosis, 
                                   user_input: str, context: Dict[str, Any]) -> str:
        """Create a support ticket."""
        ticket_id = f"TICKET-{self.ticket_counter:05d}"
        self.ticket_counter += 1
        
        ticket = {
            "id": ticket_id,
            "session_id": session_id,
            "created_at": datetime.now().isoformat(),
            "status": TicketStatus.OPEN.value,
            "severity": diagnosis.severity,
            "category": diagnosis.issue_category,
            "description": user_input,
            "diagnosis": diagnosis.dict(),
            "context": context,
            "updates": [],
            "escalation_level": 0
        }
        
        # Auto-escalate if needed
        if diagnosis.escalation_needed:
            ticket["status"] = TicketStatus.ESCALATED.value
            ticket["escalation_level"] = 1
        
        self.tickets[ticket_id] = ticket
        
        # Log ticket creation
        self.logger.info(f"Created ticket {ticket_id} for session {session_id}")
        
        return ticket_id
    
    async def handle_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle technical support conversation."""
        user_input = parameters.get("user_input", "")
        session_id = parameters.get("session_id", "")
        context = parameters.get("context", {})
        intent = parameters.get("intent", "technical")
        
        # Diagnose the issue
        diagnosis = await self.diagnose_issue(user_input, context)
        
        # Create support ticket
        ticket_id = await self.create_support_ticket(session_id, diagnosis, user_input, context)
        
        # Generate response based on diagnosis
        response = await self._generate_support_response(diagnosis, ticket_id)
        
        # Check if escalation is needed
        escalation_info = None
        if diagnosis.escalation_needed:
            escalation_info = await self._handle_escalation(ticket_id, diagnosis)
        
        return {
            "response": response,
            "agent": "tech-support-agent",
            "ticket_id": ticket_id,
            "diagnosis": diagnosis.dict(),
            "escalation": escalation_info,
            "next_steps": diagnosis.troubleshooting_steps,
            "confidence": 0.9
        }
    
    async def _generate_support_response(self, diagnosis: TechnicalDiagnosis, ticket_id: str) -> str:
        """Generate support response based on diagnosis."""
        if diagnosis.escalation_needed:
            return f"""I understand you're experiencing a {diagnosis.severity} priority technical issue. I've created ticket {ticket_id} and escalated it to our specialist team.
            
In the meantime, here are some immediate steps you can try:
{chr(10).join(f"• {step}" for step in diagnosis.troubleshooting_steps[:3])}

Our team will follow up with you within {diagnosis.estimated_resolution_time}. Is there anything else I can help clarify about this issue?"""
        
        else:
            return f"""I've analyzed your technical issue and created ticket {ticket_id} for tracking. Based on the symptoms, this appears to be related to {diagnosis.issue_category}.

Here's what I recommend trying first:
{chr(10).join(f"{i+1}. {step}" for i, step in enumerate(diagnosis.troubleshooting_steps))}

This type of issue typically resolves within {diagnosis.estimated_resolution_time}. Please try these steps and let me know if the issue persists!"""
    
    async def _handle_escalation(self, ticket_id: str, diagnosis: TechnicalDiagnosis) -> Dict[str, Any]:
        """Handle ticket escalation."""
        ticket = self.tickets[ticket_id]
        
        # Determine escalation target
        escalation_target = "senior-tech-support"
        if diagnosis.severity == SeverityLevel.CRITICAL.value:
            escalation_target = "engineering-team"
        
        # Update ticket
        ticket["escalation_level"] += 1
        ticket["status"] = TicketStatus.ESCALATED.value
        ticket["updates"].append({
            "timestamp": datetime.now().isoformat(),
            "action": "escalated",
            "target": escalation_target,
            "reason": f"Auto-escalated due to {diagnosis.severity} severity"
        })
        
        # Notify escalation target (would integrate with real systems)
        escalation_info = {
            "ticket_id": ticket_id,
            "escalated_to": escalation_target,
            "reason": f"{diagnosis.severity} severity issue requiring specialist attention",
            "estimated_response": "Within 1 hour" if diagnosis.severity == SeverityLevel.CRITICAL.value else "Within 4 hours"
        }
        
        self.logger.info(f"Escalated ticket {ticket_id} to {escalation_target}")
        
        return escalation_info
    
    async def get_ticket_status(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Get ticket status."""
        ticket_id = parameters.get("ticket_id", "")
        
        if ticket_id in self.tickets:
            ticket = self.tickets[ticket_id]
            return {
                "found": True,
                "ticket": ticket,
                "status": ticket["status"],
                "last_update": ticket.get("updates", [])[-1] if ticket.get("updates") else None
            }
        else:
            return {"found": False, "error": "Ticket not found"}
    
    async def update_ticket(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Update ticket information."""
        ticket_id = parameters.get("ticket_id", "")
        update_data = parameters.get("update", {})
        
        if ticket_id in self.tickets:
            ticket = self.tickets[ticket_id]
            
            # Add update record
            update_record = {
                "timestamp": datetime.now().isoformat(),
                "update": update_data
            }
            ticket["updates"].append(update_record)
            
            # Update status if provided
            if "status" in update_data:
                ticket["status"] = update_data["status"]
            
            return {"success": True, "ticket_id": ticket_id}
        else:
            return {"success": False, "error": "Ticket not found"}
    
    async def start_amp_agent(self, agent_id: str = "tech-support-agent",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "Technical Support Agent")
            .with_framework("langchain")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "conversation-handler",
                self.handle_conversation,
                "Handle technical support conversations",
                "support",
                input_schema={
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string"},
                        "session_id": {"type": "string"},
                        "context": {"type": "object"},
                        "intent": {"type": "string"}
                    },
                    "required": ["user_input"]
                },
                constraints=CapabilityConstraints(response_time_ms=5000)
            )
            .add_capability(
                "issue-diagnosis",
                self.diagnose_issue,
                "Diagnose technical issues",
                "support"
            )
            .add_capability(
                "ticket-status",
                self.get_ticket_status,
                "Get support ticket status",
                "support"
            )
            .add_capability(
                "ticket-update",
                self.update_ticket,
                "Update support ticket",
                "support"
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the tech support agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create tech support agent
    tech_agent = TechSupportAgent()
    
    # Start AMP agent
    client = await tech_agent.start_amp_agent()
    
    try:
        print("Tech Support Agent started. Testing issue diagnosis...")
        
        # Test technical issues
        test_issues = [
            "I can't log into my account, it keeps saying wrong password",
            "The application is very slow and sometimes crashes",
            "I'm getting a critical error that's causing data loss",
            "The API integration stopped working suddenly"
        ]
        
        for issue in test_issues:
            result = await tech_agent.handle_conversation({
                "user_input": issue,
                "session_id": f"test-session-{hash(issue) % 1000}",
                "context": {},
                "intent": "technical"
            })
            print(f"Issue: {issue}")
            print(f"Response: {result['response']}")
            print(f"Ticket ID: {result['ticket_id']}")
            print(f"Diagnosis: {result['diagnosis']['issue_category']} ({result['diagnosis']['severity']})")
            print()
        
        print("Tech Support Agent is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())