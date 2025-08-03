"""
Escalation Manager Agent.

This agent manages complex cases, escalations, and coordination between
support specialists for tickets that require elevated attention.
"""

import logging
import json
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone, timedelta
from enum import Enum

# AMP imports
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))

from amp_client import AMPClient, AMPClientConfig
from amp_types import Capability, CapabilityConstraints, TransportType

# Support system imports
from ..support_types import (
    Ticket, TicketCategory, TicketPriority, TicketStatus,
    CustomerInfo, SLALevel, AgentAvailability
)


class EscalationReason(Enum):
    """Escalation reasons."""
    SLA_BREACH = "sla_breach"
    COMPLEX_TECHNICAL = "complex_technical"
    CUSTOMER_DISSATISFACTION = "customer_dissatisfaction"
    MANAGER_REQUEST = "manager_request"
    SECURITY_INCIDENT = "security_incident"
    BILLING_DISPUTE = "billing_dispute"
    MULTIPLE_FAILURES = "multiple_failures"


class EscalationLevel(Enum):
    """Escalation levels."""
    TEAM_LEAD = "team_lead"
    SENIOR_SPECIALIST = "senior_specialist"
    MANAGER = "manager"
    DIRECTOR = "director"
    EXECUTIVE = "executive"


class EscalationManagerAgent:
    """
    Escalation Manager Agent for handling complex support cases and
    coordinating escalations across the support organization.
    """
    
    def __init__(self, config: AMPClientConfig):
        self.config = config
        self.client = AMPClient(config)
        self.logger = logging.getLogger(f"support.escalation.{config.agent_id}")
        
        # Escalation matrix
        self.escalation_matrix = {
            (TicketPriority.CRITICAL, SLALevel.ENTERPRISE): EscalationLevel.DIRECTOR,
            (TicketPriority.CRITICAL, SLALevel.PREMIUM): EscalationLevel.MANAGER,
            (TicketPriority.CRITICAL, SLALevel.STANDARD): EscalationLevel.SENIOR_SPECIALIST,
            (TicketPriority.URGENT, SLALevel.ENTERPRISE): EscalationLevel.MANAGER,
            (TicketPriority.URGENT, SLALevel.PREMIUM): EscalationLevel.SENIOR_SPECIALIST,
            (TicketPriority.HIGH, SLALevel.ENTERPRISE): EscalationLevel.SENIOR_SPECIALIST,
        }
        
        # Support team contacts
        self.escalation_contacts = {
            EscalationLevel.TEAM_LEAD: {
                "technical": "tech-lead@company.com",
                "billing": "billing-lead@company.com",
                "product": "product-lead@company.com"
            },
            EscalationLevel.SENIOR_SPECIALIST: {
                "technical": "senior-tech@company.com",
                "billing": "senior-billing@company.com",
                "product": "senior-product@company.com"
            },
            EscalationLevel.MANAGER: {
                "support": "support-manager@company.com",
                "customer_success": "cs-manager@company.com"
            },
            EscalationLevel.DIRECTOR: {
                "support": "support-director@company.com"
            }
        }
        
        # Agent availability tracking
        self.agent_availability = {}
        
    async def start(self):
        """Start the escalation manager agent."""
        await self._register_capabilities()
        
        connected = await self.client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        
        # Set up event listeners
        self.client.on_event("ticket.sla_breach", self.handle_sla_breach)
        self.client.on_event("ticket.customer_escalation", self.handle_customer_escalation)
        self.client.on_event("agent.availability_change", self.handle_agent_availability)
        
        self.logger.info("Escalation Manager Agent started successfully")
    
    async def stop(self):
        """Stop the escalation manager agent."""
        await self.client.disconnect()
        self.logger.info("Escalation Manager Agent stopped")
    
    async def _register_capabilities(self):
        """Register agent capabilities with AMP."""
        
        # Escalation analysis capability
        escalation_analysis_capability = Capability(
            id="escalation-analysis",
            version="1.0",
            description="Analyze tickets for escalation requirements and manage escalation workflow",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket": {"type": "object"},
                    "escalation_trigger": {"type": "string"},
                    "previous_attempts": {"type": "array"},
                    "agent_context": {"type": "object"}
                },
                "required": ["ticket", "escalation_trigger"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "should_escalate": {"type": "boolean"},
                    "escalation_level": {"type": "string"},
                    "escalation_reason": {"type": "string"},
                    "recommended_actions": {"type": "array"},
                    "escalation_timeline": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=5000,
                max_input_length=10000
            ),
            category="escalation-management"
        )
        
        self.client.register_capability(escalation_analysis_capability, self.analyze_escalation)
        
        # Case coordination capability
        case_coordination_capability = Capability(
            id="case-coordination",
            version="1.0",
            description="Coordinate complex cases across multiple support agents and teams",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket": {"type": "object"},
                    "involved_agents": {"type": "array"},
                    "coordination_type": {"type": "string"}
                },
                "required": ["ticket"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "coordination_plan": {"type": "object"},
                    "agent_assignments": {"type": "array"},
                    "communication_plan": {"type": "object"},
                    "timeline": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=8000,
                max_input_length=15000
            )
        )
        
        self.client.register_capability(case_coordination_capability, self.coordinate_case)
    
    async def analyze_escalation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze whether a ticket requires escalation and determine the appropriate level.
        
        Args:
            parameters: Contains ticket data and escalation context
            
        Returns:
            Escalation analysis with recommendations
        """
        try:
            ticket_data = parameters["ticket"]
            escalation_trigger = parameters["escalation_trigger"]
            previous_attempts = parameters.get("previous_attempts", [])
            agent_context = parameters.get("agent_context", {})
            
            # Create ticket object for analysis
            ticket = self._parse_ticket(ticket_data)
            
            # Analyze escalation requirements
            escalation_analysis = self._analyze_escalation_requirements(
                ticket, escalation_trigger, previous_attempts, agent_context
            )
            
            # Determine escalation level if needed
            if escalation_analysis["should_escalate"]:
                escalation_level = self._determine_escalation_level(ticket, escalation_trigger)
                escalation_contacts = self._get_escalation_contacts(escalation_level, ticket.category)
                escalation_timeline = self._create_escalation_timeline(ticket, escalation_level)
            else:
                escalation_level = None
                escalation_contacts = []
                escalation_timeline = {}
            
            # Generate recommended actions
            recommended_actions = self._generate_escalation_actions(
                ticket, escalation_analysis, escalation_level
            )
            
            response = {
                "ticket_id": ticket.id,
                "agent_id": self.config.agent_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "should_escalate": escalation_analysis["should_escalate"],
                "escalation_level": escalation_level.value if escalation_level else None,
                "escalation_reason": escalation_analysis["reason"],
                "escalation_urgency": escalation_analysis["urgency"],
                "recommended_actions": recommended_actions,
                "escalation_contacts": escalation_contacts,
                "escalation_timeline": escalation_timeline,
                "risk_assessment": escalation_analysis["risk_assessment"],
                "customer_impact": escalation_analysis["customer_impact"],
                "business_impact": escalation_analysis["business_impact"]
            }
            
            self.logger.info(f"Analyzed escalation for ticket {ticket.id}: "
                           f"Escalate={escalation_analysis['should_escalate']}")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error analyzing escalation: {e}")
            raise
    
    async def coordinate_case(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate a complex case across multiple agents and teams.
        
        Args:
            parameters: Contains ticket and coordination context
            
        Returns:
            Case coordination plan with assignments
        """
        try:
            ticket_data = parameters["ticket"]
            involved_agents = parameters.get("involved_agents", [])
            coordination_type = parameters.get("coordination_type", "standard")
            
            ticket = self._parse_ticket(ticket_data)
            
            # Create coordination plan
            coordination_plan = self._create_coordination_plan(ticket, coordination_type)
            
            # Assign agents based on availability and expertise
            agent_assignments = await self._assign_coordination_agents(ticket, involved_agents)
            
            # Create communication plan
            communication_plan = self._create_communication_plan(ticket, agent_assignments)
            
            # Create timeline
            timeline = self._create_coordination_timeline(ticket, coordination_plan)
            
            response = {
                "ticket_id": ticket.id,
                "agent_id": self.config.agent_id,
                "coordination_timestamp": datetime.now(timezone.utc).isoformat(),
                "coordination_plan": coordination_plan,
                "agent_assignments": agent_assignments,
                "communication_plan": communication_plan,
                "timeline": timeline,
                "success_criteria": self._define_success_criteria(ticket),
                "monitoring_plan": self._create_monitoring_plan(ticket)
            }
            
            self.logger.info(f"Created coordination plan for ticket {ticket.id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error coordinating case: {e}")
            raise
    
    async def handle_sla_breach(self, event_data: Dict[str, Any]):
        """Handle SLA breach events."""
        try:
            ticket_id = event_data.get("ticket_id")
            breach_type = event_data.get("breach_type")
            
            self.logger.warning(f"SLA breach detected for ticket {ticket_id}: {breach_type}")
            
            # Trigger escalation analysis
            escalation_params = {
                "ticket": event_data.get("ticket", {}),
                "escalation_trigger": f"sla_breach_{breach_type}"
            }
            
            escalation_result = await self.analyze_escalation(escalation_params)
            
            if escalation_result["should_escalate"]:
                await self._execute_escalation(escalation_result)
                
        except Exception as e:
            self.logger.error(f"Error handling SLA breach: {e}")
    
    async def handle_customer_escalation(self, event_data: Dict[str, Any]):
        """Handle customer-requested escalations."""
        try:
            ticket_id = event_data.get("ticket_id")
            escalation_reason = event_data.get("reason", "customer_request")
            
            self.logger.info(f"Customer escalation request for ticket {ticket_id}")
            
            # Trigger immediate escalation analysis
            escalation_params = {
                "ticket": event_data.get("ticket", {}),
                "escalation_trigger": "customer_escalation"
            }
            
            escalation_result = await self.analyze_escalation(escalation_params)
            
            # Customer escalations are typically honored
            escalation_result["should_escalate"] = True
            await self._execute_escalation(escalation_result)
            
        except Exception as e:
            self.logger.error(f"Error handling customer escalation: {e}")
    
    async def handle_agent_availability(self, event_data: Dict[str, Any]):
        """Handle agent availability changes."""
        try:
            agent_id = event_data.get("agent_id")
            availability_data = event_data.get("availability", {})
            
            self.agent_availability[agent_id] = AgentAvailability(**availability_data)
            
            self.logger.debug(f"Updated availability for agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Error handling agent availability: {e}")
    
    def _parse_ticket(self, ticket_data: Dict[str, Any]) -> Ticket:
        """Parse ticket data into Ticket object."""
        # Simplified parsing - in production, use proper validation
        return Ticket(
            id=ticket_data.get("id", "unknown"),
            subject=ticket_data.get("subject", ""),
            description=ticket_data.get("description", ""),
            customer=CustomerInfo(
                id=ticket_data.get("customer_info", {}).get("id", "unknown"),
                name=ticket_data.get("customer_info", {}).get("name", "Unknown"),
                email=ticket_data.get("customer_info", {}).get("email", ""),
                sla_level=SLALevel(ticket_data.get("customer_info", {}).get("sla_level", "basic"))
            ),
            status=TicketStatus(ticket_data.get("status", "open")),
            priority=TicketPriority(ticket_data.get("priority", "medium")),
            category=TicketCategory(ticket_data.get("category", "general"))
        )
    
    def _analyze_escalation_requirements(self, ticket: Ticket, trigger: str, 
                                       previous_attempts: List[Dict], 
                                       agent_context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze if escalation is required."""
        
        analysis = {
            "should_escalate": False,
            "reason": "",
            "urgency": "medium",
            "risk_assessment": "low",
            "customer_impact": "low",
            "business_impact": "low"
        }
        
        # SLA breach triggers
        if "sla_breach" in trigger:
            analysis["should_escalate"] = True
            analysis["reason"] = "SLA targets exceeded"
            analysis["urgency"] = "high"
            analysis["risk_assessment"] = "medium"
        
        # Customer escalation requests
        if "customer_escalation" in trigger:
            analysis["should_escalate"] = True
            analysis["reason"] = "Customer requested escalation"
            analysis["urgency"] = "high"
            analysis["customer_impact"] = "high"
        
        # Multiple failed attempts
        if len(previous_attempts) >= 3:
            analysis["should_escalate"] = True
            analysis["reason"] = "Multiple resolution attempts failed"
            analysis["urgency"] = "medium"
        
        # Critical priority tickets
        if ticket.priority == TicketPriority.CRITICAL:
            analysis["should_escalate"] = True
            analysis["reason"] = "Critical priority ticket"
            analysis["urgency"] = "critical"
            analysis["risk_assessment"] = "high"
            analysis["business_impact"] = "high"
        
        # Enterprise customer considerations
        if ticket.customer.sla_level == SLALevel.ENTERPRISE:
            if ticket.priority in [TicketPriority.HIGH, TicketPriority.URGENT]:
                analysis["should_escalate"] = True
                analysis["reason"] = "Enterprise customer high priority issue"
                analysis["customer_impact"] = "high"
                analysis["business_impact"] = "high"
        
        # Security-related issues
        if any(word in ticket.description.lower() for word in ["security", "breach", "hack", "vulnerability"]):
            analysis["should_escalate"] = True
            analysis["reason"] = "Security implications detected"
            analysis["urgency"] = "critical"
            analysis["risk_assessment"] = "critical"
        
        return analysis
    
    def _determine_escalation_level(self, ticket: Ticket, trigger: str) -> EscalationLevel:
        """Determine the appropriate escalation level."""
        
        # Security incidents go to highest level
        if "security" in trigger.lower():
            return EscalationLevel.DIRECTOR
        
        # Use escalation matrix
        matrix_key = (ticket.priority, ticket.customer.sla_level)
        if matrix_key in self.escalation_matrix:
            return self.escalation_matrix[matrix_key]
        
        # Default escalation levels
        if ticket.priority == TicketPriority.CRITICAL:
            return EscalationLevel.MANAGER
        elif ticket.priority == TicketPriority.URGENT:
            return EscalationLevel.SENIOR_SPECIALIST
        else:
            return EscalationLevel.TEAM_LEAD
    
    def _get_escalation_contacts(self, level: EscalationLevel, category: TicketCategory) -> List[Dict[str, str]]:
        """Get escalation contacts for the level and category."""
        contacts = []
        
        level_contacts = self.escalation_contacts.get(level, {})
        
        # Category-specific contact
        category_map = {
            TicketCategory.TECHNICAL: "technical",
            TicketCategory.BILLING: "billing", 
            TicketCategory.PRODUCT: "product"
        }
        
        category_key = category_map.get(category, "support")
        if category_key in level_contacts:
            contacts.append({
                "role": f"{level.value}_{category_key}",
                "contact": level_contacts[category_key],
                "type": "primary"
            })
        
        # General support contact as backup
        if "support" in level_contacts and category_key != "support":
            contacts.append({
                "role": f"{level.value}_support",
                "contact": level_contacts["support"],
                "type": "secondary"
            })
        
        return contacts
    
    def _create_escalation_timeline(self, ticket: Ticket, level: EscalationLevel) -> Dict[str, Any]:
        """Create escalation timeline."""
        now = datetime.now(timezone.utc)
        
        timeline_map = {
            EscalationLevel.TEAM_LEAD: timedelta(hours=1),
            EscalationLevel.SENIOR_SPECIALIST: timedelta(hours=2),
            EscalationLevel.MANAGER: timedelta(hours=4),
            EscalationLevel.DIRECTOR: timedelta(hours=8),
            EscalationLevel.EXECUTIVE: timedelta(hours=12)
        }
        
        response_time = timeline_map.get(level, timedelta(hours=2))
        
        return {
            "escalation_time": now.isoformat(),
            "expected_response": (now + response_time).isoformat(),
            "response_sla_hours": response_time.total_seconds() / 3600,
            "follow_up_schedule": [
                (now + response_time/2).isoformat(),
                (now + response_time).isoformat(),
                (now + response_time * 2).isoformat()
            ]
        }
    
    def _generate_escalation_actions(self, ticket: Ticket, analysis: Dict[str, Any], 
                                   level: Optional[EscalationLevel]) -> List[Dict[str, Any]]:
        """Generate recommended escalation actions."""
        actions = []
        
        if analysis["should_escalate"] and level:
            actions.extend([
                {
                    "action": "notify_escalation_contacts",
                    "priority": "immediate",
                    "description": f"Notify {level.value} contacts about escalation"
                },
                {
                    "action": "update_ticket_priority",
                    "priority": "high", 
                    "description": "Update ticket priority and status"
                },
                {
                    "action": "schedule_review_meeting",
                    "priority": "medium",
                    "description": "Schedule escalation review meeting"
                }
            ])
        
        if analysis["risk_assessment"] in ["high", "critical"]:
            actions.append({
                "action": "activate_incident_response",
                "priority": "immediate",
                "description": "Activate incident response procedures"
            })
        
        if analysis["customer_impact"] == "high":
            actions.append({
                "action": "customer_communication",
                "priority": "high",
                "description": "Proactive customer communication about escalation"
            })
        
        return actions
    
    def _create_coordination_plan(self, ticket: Ticket, coordination_type: str) -> Dict[str, Any]:
        """Create case coordination plan."""
        
        plan_templates = {
            "technical_incident": {
                "phases": ["assessment", "containment", "resolution", "post_incident"],
                "required_roles": ["technical_lead", "escalation_manager", "customer_liaison"],
                "communication_frequency": "every_30_minutes"
            },
            "customer_escalation": {
                "phases": ["acknowledgment", "investigation", "resolution", "follow_up"],
                "required_roles": ["account_manager", "subject_expert", "escalation_manager"],
                "communication_frequency": "every_2_hours"
            },
            "standard": {
                "phases": ["coordination", "execution", "monitoring", "closure"],
                "required_roles": ["case_coordinator", "subject_expert"],
                "communication_frequency": "daily"
            }
        }
        
        template = plan_templates.get(coordination_type, plan_templates["standard"])
        
        return {
            "coordination_type": coordination_type,
            "phases": template["phases"],
            "required_roles": template["required_roles"],
            "communication_frequency": template["communication_frequency"],
            "success_metrics": self._define_coordination_metrics(ticket),
            "escalation_triggers": self._define_coordination_escalation_triggers()
        }
    
    async def _assign_coordination_agents(self, ticket: Ticket, 
                                        involved_agents: List[str]) -> List[Dict[str, Any]]:
        """Assign agents for case coordination."""
        assignments = []
        
        # Primary coordinator (this agent)
        assignments.append({
            "agent_id": self.config.agent_id,
            "role": "case_coordinator",
            "responsibility": "Overall case coordination and communication",
            "availability": "primary"
        })
        
        # Subject matter experts based on ticket category
        if ticket.category == TicketCategory.TECHNICAL:
            assignments.append({
                "agent_id": "technical-support-001",
                "role": "technical_expert",
                "responsibility": "Technical analysis and resolution",
                "availability": "on_demand"
            })
        
        if ticket.category == TicketCategory.BILLING:
            assignments.append({
                "agent_id": "billing-support-001", 
                "role": "billing_expert",
                "responsibility": "Billing analysis and processing",
                "availability": "on_demand"
            })
        
        # Customer liaison for high-impact cases
        if ticket.customer.sla_level in [SLALevel.PREMIUM, SLALevel.ENTERPRISE]:
            assignments.append({
                "agent_id": "customer-success-001",
                "role": "customer_liaison",
                "responsibility": "Customer communication and relationship management",
                "availability": "dedicated"
            })
        
        return assignments
    
    def _create_communication_plan(self, ticket: Ticket, 
                                 assignments: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create communication plan for coordination."""
        return {
            "communication_channels": {
                "primary": "amp_messaging",
                "backup": "email",
                "urgent": "phone"
            },
            "update_frequency": {
                "internal": "every_2_hours",
                "customer": "daily_or_on_change",
                "management": "on_milestone"
            },
            "escalation_thresholds": {
                "no_progress_hours": 4,
                "customer_satisfaction_drop": 0.7,
                "sla_risk_percentage": 80
            },
            "reporting_requirements": [
                "Daily status update",
                "Issue resolution summary",
                "Customer satisfaction feedback"
            ]
        }
    
    def _create_coordination_timeline(self, ticket: Ticket, 
                                    coordination_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Create coordination timeline."""
        now = datetime.now(timezone.utc)
        
        phase_durations = {
            "assessment": timedelta(hours=2),
            "acknowledgment": timedelta(hours=1),
            "coordination": timedelta(hours=1),
            "containment": timedelta(hours=4),
            "investigation": timedelta(hours=8),
            "execution": timedelta(hours=12),
            "resolution": timedelta(hours=8),
            "monitoring": timedelta(hours=24),
            "post_incident": timedelta(hours=4),
            "follow_up": timedelta(hours=2),
            "closure": timedelta(hours=1)
        }
        
        timeline = {}
        current_time = now
        
        for phase in coordination_plan["phases"]:
            duration = phase_durations.get(phase, timedelta(hours=4))
            timeline[phase] = {
                "start_time": current_time.isoformat(),
                "end_time": (current_time + duration).isoformat(),
                "duration_hours": duration.total_seconds() / 3600
            }
            current_time += duration
        
        return timeline
    
    def _define_success_criteria(self, ticket: Ticket) -> List[Dict[str, Any]]:
        """Define success criteria for coordination."""
        return [
            {
                "metric": "resolution_time",
                "target": "within_sla",
                "measurement": "hours_to_resolution"
            },
            {
                "metric": "customer_satisfaction",
                "target": ">= 4.0",
                "measurement": "csat_score"
            },
            {
                "metric": "first_contact_resolution",
                "target": "true",
                "measurement": "boolean"
            },
            {
                "metric": "escalation_prevention", 
                "target": "no_further_escalation",
                "measurement": "boolean"
            }
        ]
    
    def _create_monitoring_plan(self, ticket: Ticket) -> Dict[str, Any]:
        """Create monitoring plan for coordinated case."""
        return {
            "monitoring_frequency": "every_30_minutes",
            "key_metrics": [
                "ticket_status",
                "agent_activity", 
                "customer_responses",
                "sla_compliance"
            ],
            "alerting_rules": [
                {
                    "condition": "no_agent_activity_2_hours",
                    "action": "alert_coordinator"
                },
                {
                    "condition": "sla_breach_risk_75_percent",
                    "action": "escalate_urgency"
                },
                {
                    "condition": "customer_satisfaction_decline",
                    "action": "initiate_recovery_process"
                }
            ],
            "reporting_schedule": {
                "coordinator_updates": "every_2_hours",
                "management_reports": "daily",
                "customer_updates": "as_needed"
            }
        }
    
    def _define_coordination_metrics(self, ticket: Ticket) -> List[str]:
        """Define coordination success metrics."""
        return [
            "time_to_coordination",
            "agent_response_time",
            "coordination_effectiveness",
            "customer_satisfaction_during_coordination"
        ]
    
    def _define_coordination_escalation_triggers(self) -> List[Dict[str, Any]]:
        """Define triggers for further escalation during coordination."""
        return [
            {
                "trigger": "coordination_timeout",
                "threshold": "6_hours_no_progress",
                "action": "escalate_to_next_level"
            },
            {
                "trigger": "agent_unavailability",
                "threshold": "key_agent_unavailable_2_hours",
                "action": "reassign_coordination_roles"
            },
            {
                "trigger": "customer_dissatisfaction",
                "threshold": "csat_score_below_3",
                "action": "activate_customer_recovery"
            }
        ]
    
    async def _execute_escalation(self, escalation_result: Dict[str, Any]):
        """Execute the escalation process."""
        try:
            # Send escalation notifications
            await self.client.emit_event(
                "escalation.initiated",
                {
                    "ticket_id": escalation_result["ticket_id"],
                    "escalation_level": escalation_result["escalation_level"],
                    "reason": escalation_result["escalation_reason"],
                    "contacts": escalation_result["escalation_contacts"],
                    "timeline": escalation_result["escalation_timeline"]
                }
            )
            
            # Execute recommended actions
            for action in escalation_result["recommended_actions"]:
                if action["priority"] == "immediate":
                    await self._execute_action(action)
            
            self.logger.info(f"Executed escalation for ticket {escalation_result['ticket_id']}")
            
        except Exception as e:
            self.logger.error(f"Error executing escalation: {e}")
    
    async def _execute_action(self, action: Dict[str, Any]):
        """Execute a specific escalation action."""
        try:
            action_type = action["action"]
            
            if action_type == "notify_escalation_contacts":
                await self.client.emit_event("escalation.notification", action)
            elif action_type == "update_ticket_priority":
                await self.client.emit_event("ticket.priority_update", action)
            elif action_type == "customer_communication":
                await self.client.emit_event("customer.escalation_notification", action)
            
            self.logger.debug(f"Executed action: {action_type}")
            
        except Exception as e:
            self.logger.error(f"Error executing action {action.get('action')}: {e}")


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        config = AMPClientConfig(
            agent_id="escalation-manager-001",
            agent_name="Escalation Manager",
            framework="custom",
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:8000"
        )
        
        agent = EscalationManagerAgent(config)
        
        try:
            await agent.start()
            print("Escalation Manager Agent is running...")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await agent.stop()
    
    asyncio.run(main())