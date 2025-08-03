"""
Support System Agents Package.

This package contains all specialized agents for the Customer Support System,
demonstrating AMP's framework-agnostic capabilities.
"""

from .ticket_classifier import TicketClassifierAgent
from .technical_support import TechnicalSupportAgent
from .billing_support import BillingSupportAgent
from .product_support import ProductSupportAgent
from .escalation_manager import EscalationManagerAgent
from .knowledge_base import KnowledgeBaseAgent

__all__ = [
    'TicketClassifierAgent',
    'TechnicalSupportAgent', 
    'BillingSupportAgent',
    'ProductSupportAgent',
    'EscalationManagerAgent',
    'KnowledgeBaseAgent'
]