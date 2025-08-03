"""
Billing Support Agent using AutoGen.

This agent handles billing inquiries, payment issues, subscription management,
and account-related financial questions using collaborative AutoGen agents.
"""

import logging
import json
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone, timedelta
import re

# AutoGen imports
import autogen
from autogen import ConversableAgent, GroupChat, GroupChatManager

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


class BillingDatabase:
    """Simulated billing database for billing operations."""
    
    def __init__(self):
        # Mock customer billing data
        self.customers = {
            "customer_123": {
                "account_id": "customer_123",
                "name": "John Doe",
                "email": "john.doe@example.com",
                "plan": "premium",
                "status": "active",
                "billing_cycle": "monthly",
                "next_billing_date": "2024-02-15",
                "payment_method": "credit_card",
                "outstanding_balance": 0.00,
                "billing_history": [
                    {
                        "invoice_id": "INV-001",
                        "date": "2024-01-15",
                        "amount": 99.99,
                        "status": "paid",
                        "description": "Premium Plan - January 2024"
                    }
                ]
            }
        }
        
        self.invoices = {
            "INV-001": {
                "id": "INV-001",
                "customer_id": "customer_123",
                "amount": 99.99,
                "status": "paid",
                "due_date": "2024-01-15",
                "items": [
                    {"description": "Premium Plan", "amount": 99.99}
                ]
            }
        }
        
        self.transactions = {
            "TXN-001": {
                "id": "TXN-001",
                "customer_id": "customer_123",
                "amount": 99.99,
                "type": "payment",
                "status": "completed",
                "date": "2024-01-15",
                "payment_method": "card_ending_1234"
            }
        }
    
    def get_customer_info(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get customer billing information."""
        return self.customers.get(customer_id)
    
    def get_invoice(self, invoice_id: str) -> Optional[Dict[str, Any]]:
        """Get invoice details."""
        return self.invoices.get(invoice_id)
    
    def get_customer_invoices(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get all invoices for a customer."""
        return [inv for inv in self.invoices.values() if inv["customer_id"] == customer_id]
    
    def get_transactions(self, customer_id: str) -> List[Dict[str, Any]]:
        """Get customer transactions."""
        return [txn for txn in self.transactions.values() if txn["customer_id"] == customer_id]
    
    def update_payment_method(self, customer_id: str, payment_method: str) -> bool:
        """Update customer payment method."""
        if customer_id in self.customers:
            self.customers[customer_id]["payment_method"] = payment_method
            return True
        return False
    
    def process_refund(self, customer_id: str, amount: float, reason: str) -> Dict[str, Any]:
        """Process a refund."""
        refund_id = f"REF-{len(self.transactions) + 1:03d}"
        refund = {
            "id": refund_id,
            "customer_id": customer_id,
            "amount": -amount,  # Negative for refund
            "type": "refund",
            "status": "processed",
            "date": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        }
        self.transactions[refund_id] = refund
        return refund


class BillingSupportAgent:
    """
    Billing Support Agent using AutoGen for collaborative billing issue resolution.
    """
    
    def __init__(self, config: AMPClientConfig, openai_api_key: str):
        self.config = config
        self.client = AMPClient(config)
        self.logger = logging.getLogger(f"support.billing.{config.agent_id}")
        
        # Initialize billing database
        self.billing_db = BillingDatabase()
        
        # Configure AutoGen LLM
        llm_config = {
            "config_list": [
                {
                    "model": "gpt-4",
                    "api_key": openai_api_key,
                    "temperature": 0.1
                }
            ]
        }
        
        # Create AutoGen agents
        self.billing_analyst = ConversableAgent(
            name="BillingAnalyst",
            system_message="""You are a billing analyst expert. Your role is to:
1. Analyze billing-related issues and customer account information
2. Identify billing discrepancies, payment failures, and account issues
3. Provide detailed analysis of billing history and patterns
4. Recommend appropriate billing actions and solutions

When analyzing billing issues:
- Review account status, billing history, and payment methods
- Identify root causes of billing problems
- Assess customer impact and urgency
- Provide clear recommendations for resolution""",
            llm_config=llm_config,
            code_execution_config=False,
            human_input_mode="NEVER"
        )
        
        self.payment_specialist = ConversableAgent(
            name="PaymentSpecialist", 
            system_message="""You are a payment processing specialist. Your role is to:
1. Handle payment-related issues, failed transactions, and refund requests
2. Manage payment method updates and subscription changes
3. Process refunds and billing adjustments
4. Resolve payment gateway and processing issues

Focus on:
- Payment method validation and updates
- Transaction failure analysis
- Refund processing and approval
- Subscription billing management""",
            llm_config=llm_config,
            code_execution_config=False,
            human_input_mode="NEVER"
        )
        
        self.account_manager = ConversableAgent(
            name="AccountManager",
            system_message="""You are an account management specialist. Your role is to:
1. Manage subscription plans, upgrades, and downgrades
2. Handle account status changes and billing preferences
3. Provide account optimization recommendations
4. Manage enterprise account billing and custom arrangements

Responsibilities:
- Plan change management
- Billing cycle adjustments
- Custom billing arrangements
- Account retention strategies""",
            llm_config=llm_config,
            code_execution_config=False,
            human_input_mode="NEVER"
        )
        
        self.billing_coordinator = ConversableAgent(
            name="BillingCoordinator",
            system_message="""You are the billing support coordinator. Your role is to:
1. Coordinate between different billing specialists
2. Synthesize recommendations from the team
3. Create comprehensive billing solutions
4. Ensure customer satisfaction and issue resolution

You should:
- Gather input from all specialists
- Create unified response plans
- Prioritize customer experience
- Ensure compliance with billing policies""",
            llm_config=llm_config,
            code_execution_config=False,
            human_input_mode="NEVER"
        )
        
        # Create group chat for collaborative billing support
        self.billing_group = GroupChat(
            agents=[
                self.billing_analyst,
                self.payment_specialist, 
                self.account_manager,
                self.billing_coordinator
            ],
            messages=[],
            max_round=10,
            speaker_selection_method="round_robin"
        )
        
        self.group_manager = GroupChatManager(
            groupchat=self.billing_group,
            llm_config=llm_config
        )
        
        # Billing operation functions
        self.billing_functions = {
            "get_customer_billing_info": self._get_customer_billing_info,
            "get_invoice_details": self._get_invoice_details,
            "process_refund": self._process_refund,
            "update_payment_method": self._update_payment_method,
            "get_payment_history": self._get_payment_history,
            "check_account_status": self._check_account_status
        }
        
    async def start(self):
        """Start the billing support agent."""
        await self._register_capabilities()
        
        connected = await self.client.connect()
        if not connected:
            raise RuntimeError("Failed to connect to AMP network")
        
        self.logger.info("Billing Support Agent started successfully")
    
    async def stop(self):
        """Stop the billing support agent."""
        await self.client.disconnect()
        self.logger.info("Billing Support Agent stopped")
    
    async def _register_capabilities(self):
        """Register agent capabilities with AMP."""
        
        # Billing inquiry handling capability
        billing_inquiry_capability = Capability(
            id="billing-inquiry-handling",
            version="1.0",
            description="Handle comprehensive billing inquiries and account issues",
            input_schema={
                "type": "object",
                "properties": {
                    "ticket": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "subject": {"type": "string"},
                            "description": {"type": "string"},
                            "customer_info": {"type": "object"},
                            "billing_context": {"type": "object"}
                        },
                        "required": ["subject", "description", "customer_info"]
                    }
                },
                "required": ["ticket"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "billing_analysis": {"type": "object"},
                    "resolution_plan": {"type": "object"},
                    "required_actions": {"type": "array"},
                    "customer_communication": {"type": "object"},
                    "follow_up_needed": {"type": "boolean"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=12000,
                max_input_length=8000,
                supported_languages=["en"],
                min_confidence=0.8
            ),
            category="billing-support",
            subcategories=["payment-processing", "account-management", "refunds"]
        )
        
        self.client.register_capability(billing_inquiry_capability, self.handle_billing_inquiry)
        
        # Payment processing capability
        payment_processing_capability = Capability(
            id="payment-processing",
            version="1.0",
            description="Process payments, refunds, and payment method updates",
            input_schema={
                "type": "object",
                "properties": {
                    "action": {"type": "string", "enum": ["refund", "update_payment", "retry_payment"]},
                    "customer_id": {"type": "string"},
                    "amount": {"type": "number"},
                    "reason": {"type": "string"},
                    "payment_details": {"type": "object"}
                },
                "required": ["action", "customer_id"]
            },
            output_schema={
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "transaction_id": {"type": "string"},
                    "details": {"type": "object"}
                }
            },
            constraints=CapabilityConstraints(
                response_time_ms=8000,
                max_input_length=5000
            )
        )
        
        self.client.register_capability(payment_processing_capability, self.process_payment_action)
    
    async def handle_billing_inquiry(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle a billing support inquiry using collaborative AutoGen agents.
        
        Args:
            parameters: Contains ticket information and billing context
            
        Returns:
            Comprehensive billing support response with resolution plan
        """
        try:
            ticket_data = parameters["ticket"]
            customer_info = ticket_data["customer_info"]
            customer_id = customer_info.get("id", "unknown")
            
            # Get customer billing information
            billing_info = self.billing_db.get_customer_info(customer_id)
            
            # Prepare context for AutoGen agents
            billing_context = {
                "ticket_subject": ticket_data["subject"],
                "ticket_description": ticket_data["description"],
                "customer_id": customer_id,
                "customer_info": customer_info,
                "billing_info": billing_info,
                "available_functions": list(self.billing_functions.keys())
            }
            
            # Create initial message for the group chat
            initial_message = f"""
New billing support ticket received:

Ticket ID: {ticket_data.get('id', 'N/A')}
Subject: {ticket_data['subject']}
Description: {ticket_data['description']}

Customer Information:
- Customer ID: {customer_id}
- Account Type: {customer_info.get('account_type', 'unknown')}
- SLA Level: {customer_info.get('sla_level', 'basic')}

Current Billing Status:
{json.dumps(billing_info, indent=2) if billing_info else 'Customer not found in billing system'}

Please analyze this billing issue and provide:
1. Detailed billing analysis
2. Root cause identification  
3. Resolution plan with specific steps
4. Required billing system actions
5. Customer communication plan
6. Follow-up requirements

BillingAnalyst should start the analysis.
            """
            
            # Reset group chat
            self.billing_group.reset()
            
            # Initiate group chat
            chat_result = self.billing_analyst.initiate_chat(
                self.group_manager,
                message=initial_message.strip(),
                max_turns=8
            )
            
            # Process the collaborative response
            conversation_history = chat_result.chat_history if hasattr(chat_result, 'chat_history') else []
            final_response = self._extract_billing_solution(conversation_history, billing_context)
            
            # Execute any required billing actions
            executed_actions = await self._execute_billing_actions(final_response.get("required_actions", []))
            
            # Prepare final response
            response = {
                "ticket_id": ticket_data.get("id"),
                "agent_id": self.config.agent_id,
                "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
                "billing_analysis": final_response.get("billing_analysis", {}),
                "resolution_plan": final_response.get("resolution_plan", {}),
                "required_actions": executed_actions,
                "customer_communication": self._generate_customer_communication(final_response, ticket_data),
                "follow_up_needed": final_response.get("follow_up_needed", False),
                "billing_adjustments": final_response.get("billing_adjustments", []),
                "estimated_resolution_time": self._estimate_billing_resolution_time(ticket_data),
                "collaboration_summary": {
                    "agents_involved": [agent.name for agent in self.billing_group.agents],
                    "discussion_rounds": len(conversation_history),
                    "consensus_reached": True
                }
            }
            
            self.logger.info(f"Processed billing inquiry {ticket_data.get('id')} with AutoGen collaboration")
            return response
            
        except Exception as e:
            self.logger.error(f"Error handling billing inquiry: {e}")
            raise
    
    async def process_payment_action(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process specific payment actions (refunds, payment updates, etc.).
        
        Args:
            parameters: Contains action type and payment details
            
        Returns:
            Payment processing results
        """
        try:
            action = parameters["action"]
            customer_id = parameters["customer_id"]
            
            if action == "refund":
                amount = parameters.get("amount", 0)
                reason = parameters.get("reason", "Customer request")
                result = self.billing_db.process_refund(customer_id, amount, reason)
                
                return {
                    "success": True,
                    "transaction_id": result["id"],
                    "details": {
                        "refund_amount": amount,
                        "refund_reason": reason,
                        "processing_time": "3-5 business days"
                    }
                }
            
            elif action == "update_payment":
                payment_method = parameters.get("payment_details", {}).get("method", "credit_card")
                success = self.billing_db.update_payment_method(customer_id, payment_method)
                
                return {
                    "success": success,
                    "transaction_id": f"UPM-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "details": {
                        "new_payment_method": payment_method,
                        "effective_date": datetime.now(timezone.utc).isoformat()
                    }
                }
            
            elif action == "retry_payment":
                # Simulate payment retry
                return {
                    "success": True,
                    "transaction_id": f"RTY-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "details": {
                        "retry_status": "scheduled",
                        "retry_date": (datetime.now() + timedelta(hours=24)).isoformat()
                    }
                }
            
            else:
                raise ValueError(f"Unknown payment action: {action}")
                
        except Exception as e:
            self.logger.error(f"Error processing payment action: {e}")
            return {
                "success": False,
                "error": str(e),
                "details": {}
            }
    
    def _extract_billing_solution(self, conversation_history: List, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract billing solution from AutoGen conversation."""
        # Parse the conversation to extract key insights
        solution = {
            "billing_analysis": {},
            "resolution_plan": {},
            "required_actions": [],
            "follow_up_needed": False,
            "billing_adjustments": []
        }
        
        # Analyze conversation content
        full_conversation = " ".join([msg.get("content", "") for msg in conversation_history])
        
        # Extract billing analysis
        if "analysis" in full_conversation.lower():
            solution["billing_analysis"] = {
                "issue_type": self._identify_billing_issue_type(context["ticket_description"]),
                "account_status": context.get("billing_info", {}).get("status", "unknown"),
                "payment_method": context.get("billing_info", {}).get("payment_method", "unknown"),
                "outstanding_balance": context.get("billing_info", {}).get("outstanding_balance", 0),
                "risk_level": self._assess_billing_risk(context)
            }
        
        # Extract resolution plan
        solution["resolution_plan"] = {
            "primary_solution": self._identify_primary_solution(full_conversation),
            "alternative_solutions": self._identify_alternative_solutions(full_conversation),
            "estimated_impact": "low",
            "approval_required": self._requires_approval(context)
        }
        
        # Extract required actions
        if "refund" in full_conversation.lower():
            solution["required_actions"].append({
                "action": "process_refund", 
                "priority": "high",
                "approval_needed": True
            })
        
        if "payment method" in full_conversation.lower():
            solution["required_actions"].append({
                "action": "update_payment_method",
                "priority": "medium",
                "approval_needed": False
            })
        
        # Determine follow-up needs
        solution["follow_up_needed"] = any(word in full_conversation.lower() 
                                         for word in ["follow up", "monitor", "check back"])
        
        return solution
    
    def _identify_billing_issue_type(self, description: str) -> str:
        """Identify the type of billing issue."""
        description_lower = description.lower()
        
        if any(word in description_lower for word in ["refund", "charge back", "dispute"]):
            return "refund_request"
        elif any(word in description_lower for word in ["payment", "failed", "declined"]):
            return "payment_failure"
        elif any(word in description_lower for word in ["invoice", "bill", "statement"]):
            return "billing_inquiry"
        elif any(word in description_lower for word in ["subscription", "plan", "upgrade"]):
            return "subscription_management"
        else:
            return "general_billing"
    
    def _assess_billing_risk(self, context: Dict[str, Any]) -> str:
        """Assess the risk level of the billing issue."""
        customer_info = context.get("customer_info", {})
        billing_info = context.get("billing_info", {})
        
        # High risk for enterprise customers
        if customer_info.get("account_type") == "enterprise":
            return "high"
        
        # High risk for outstanding balance
        if billing_info.get("outstanding_balance", 0) > 0:
            return "high"
        
        # Medium risk for premium customers  
        if customer_info.get("sla_level") == "premium":
            return "medium"
        
        return "low"
    
    def _identify_primary_solution(self, conversation: str) -> str:
        """Identify the primary solution from conversation."""
        if "refund" in conversation.lower():
            return "Process refund as requested"
        elif "payment method" in conversation.lower():
            return "Update payment method"
        elif "retry" in conversation.lower():
            return "Retry failed payment"
        else:
            return "Review account and provide clarification"
    
    def _identify_alternative_solutions(self, conversation: str) -> List[str]:
        """Identify alternative solutions."""
        alternatives = []
        
        if "credit" in conversation.lower():
            alternatives.append("Apply account credit")
        
        if "plan change" in conversation.lower():
            alternatives.append("Modify subscription plan")
        
        if "payment plan" in conversation.lower():
            alternatives.append("Setup payment plan")
        
        return alternatives
    
    def _requires_approval(self, context: Dict[str, Any]) -> bool:
        """Check if the solution requires approval."""
        # Refunds typically require approval
        if "refund" in context["ticket_description"].lower():
            return True
        
        # Enterprise customers may require approval
        if context.get("customer_info", {}).get("account_type") == "enterprise":
            return True
        
        return False
    
    async def _execute_billing_actions(self, actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute required billing actions."""
        executed_actions = []
        
        for action in actions:
            try:
                action_type = action.get("action")
                
                if action_type in self.billing_functions:
                    # Execute the billing function
                    result = await self.billing_functions[action_type]({})
                    executed_actions.append({
                        **action,
                        "executed": True,
                        "result": result,
                        "execution_time": datetime.now(timezone.utc).isoformat()
                    })
                else:
                    executed_actions.append({
                        **action,
                        "executed": False,
                        "error": f"Unknown action type: {action_type}"
                    })
                    
            except Exception as e:
                executed_actions.append({
                    **action,
                    "executed": False,
                    "error": str(e)
                })
        
        return executed_actions
    
    def _generate_customer_communication(self, solution: Dict[str, Any], ticket_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate customer communication plan."""
        return {
            "response_template": f"""
Dear {ticket_data.get('customer_info', {}).get('name', 'Valued Customer')},

Thank you for contacting our billing support team regarding: {ticket_data['subject']}

Our billing specialists have reviewed your account and issue. Here's what we found:

{solution.get('resolution_plan', {}).get('primary_solution', 'We are working to resolve your billing inquiry.')}

We will process any necessary adjustments to your account within 1-2 business days. You will receive email confirmation once the changes are complete.

If you have any questions or concerns, please don't hesitate to contact us.

Best regards,
Billing Support Team
            """.strip(),
            "communication_timeline": [
                "Immediate: Send initial response",
                "24 hours: Process billing adjustments", 
                "48 hours: Follow up confirmation"
            ],
            "escalation_path": "Billing Manager" if solution.get("resolution_plan", {}).get("approval_required") else None
        }
    
    def _estimate_billing_resolution_time(self, ticket_data: Dict[str, Any]) -> str:
        """Estimate billing resolution time."""
        description_lower = ticket_data["description"].lower()
        
        if "refund" in description_lower:
            return "3-5 business days"
        elif "payment" in description_lower:
            return "24-48 hours"
        elif "invoice" in description_lower:
            return "1-2 business days"
        else:
            return "2-3 business days"
    
    # Billing database operation methods
    async def _get_customer_billing_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get customer billing information."""
        customer_id = params.get("customer_id")
        return self.billing_db.get_customer_info(customer_id) or {}
    
    async def _get_invoice_details(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Get invoice details."""
        invoice_id = params.get("invoice_id")
        return self.billing_db.get_invoice(invoice_id) or {}
    
    async def _process_refund(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process a refund."""
        customer_id = params.get("customer_id")
        amount = params.get("amount", 0)
        reason = params.get("reason", "Customer request")
        return self.billing_db.process_refund(customer_id, amount, reason)
    
    async def _update_payment_method(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update payment method."""
        customer_id = params.get("customer_id")
        payment_method = params.get("payment_method", "credit_card")
        success = self.billing_db.update_payment_method(customer_id, payment_method)
        return {"success": success, "payment_method": payment_method}
    
    async def _get_payment_history(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get payment history."""
        customer_id = params.get("customer_id")
        return self.billing_db.get_transactions(customer_id)
    
    async def _check_account_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Check account status."""
        customer_id = params.get("customer_id")
        customer_info = self.billing_db.get_customer_info(customer_id)
        return {
            "status": customer_info.get("status", "unknown") if customer_info else "not_found",
            "account_in_good_standing": customer_info.get("outstanding_balance", 0) == 0 if customer_info else False
        }


# Example usage
if __name__ == "__main__":
    import asyncio
    import os
    
    async def main():
        config = AMPClientConfig(
            agent_id="billing-support-001",
            agent_name="Billing Support Agent",
            framework="autogen",
            transport_type=TransportType.HTTP,
            endpoint="http://localhost:8000"
        )
        
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            print("Please set OPENAI_API_KEY environment variable")
            return
        
        agent = BillingSupportAgent(config, openai_api_key)
        
        try:
            await agent.start()
            print("Billing Support Agent is running...")
            
            # Keep running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            await agent.stop()
    
    asyncio.run(main())