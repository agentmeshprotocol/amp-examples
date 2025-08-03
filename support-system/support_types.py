"""
Support System data types and structures.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Union
from enum import Enum


class TicketStatus(Enum):
    """Ticket status values."""
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    PENDING_CUSTOMER = "pending_customer"
    RESOLVED = "resolved"
    CLOSED = "closed"
    ESCALATED = "escalated"


class TicketPriority(Enum):
    """Ticket priority levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"
    CRITICAL = "critical"


class TicketCategory(Enum):
    """Ticket category types."""
    TECHNICAL = "technical"
    BILLING = "billing"
    PRODUCT = "product"
    ACCOUNT = "account"
    FEATURE_REQUEST = "feature_request"
    BUG_REPORT = "bug_report"
    GENERAL = "general"


class SLALevel(Enum):
    """Service Level Agreement levels."""
    BASIC = "basic"
    STANDARD = "standard"
    PREMIUM = "premium"
    ENTERPRISE = "enterprise"


@dataclass
class CustomerInfo:
    """Customer information."""
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    sla_level: SLALevel = SLALevel.BASIC
    account_type: str = "free"
    language: str = "en"
    timezone: str = "UTC"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SLATarget:
    """SLA targets for different priority levels."""
    first_response_minutes: int
    resolution_time_hours: int
    escalation_time_hours: int


SLA_TARGETS = {
    SLALevel.BASIC: {
        TicketPriority.LOW: SLATarget(480, 72, 48),      # 8h, 3d, 2d
        TicketPriority.MEDIUM: SLATarget(240, 48, 24),   # 4h, 2d, 1d
        TicketPriority.HIGH: SLATarget(120, 24, 12),     # 2h, 1d, 12h
        TicketPriority.URGENT: SLATarget(60, 8, 4),      # 1h, 8h, 4h
        TicketPriority.CRITICAL: SLATarget(30, 4, 2),    # 30m, 4h, 2h
    },
    SLALevel.STANDARD: {
        TicketPriority.LOW: SLATarget(240, 48, 24),      # 4h, 2d, 1d
        TicketPriority.MEDIUM: SLATarget(120, 24, 12),   # 2h, 1d, 12h
        TicketPriority.HIGH: SLATarget(60, 12, 6),       # 1h, 12h, 6h
        TicketPriority.URGENT: SLATarget(30, 4, 2),      # 30m, 4h, 2h
        TicketPriority.CRITICAL: SLATarget(15, 2, 1),    # 15m, 2h, 1h
    },
    SLALevel.PREMIUM: {
        TicketPriority.LOW: SLATarget(120, 24, 12),      # 2h, 1d, 12h
        TicketPriority.MEDIUM: SLATarget(60, 12, 6),     # 1h, 12h, 6h
        TicketPriority.HIGH: SLATarget(30, 6, 3),        # 30m, 6h, 3h
        TicketPriority.URGENT: SLATarget(15, 2, 1),      # 15m, 2h, 1h
        TicketPriority.CRITICAL: SLATarget(5, 1, 0.5),   # 5m, 1h, 30m
    },
    SLALevel.ENTERPRISE: {
        TicketPriority.LOW: SLATarget(60, 12, 6),        # 1h, 12h, 6h
        TicketPriority.MEDIUM: SLATarget(30, 6, 3),      # 30m, 6h, 3h
        TicketPriority.HIGH: SLATarget(15, 3, 1.5),      # 15m, 3h, 1.5h
        TicketPriority.URGENT: SLATarget(10, 1, 0.5),    # 10m, 1h, 30m
        TicketPriority.CRITICAL: SLATarget(5, 0.5, 0.25), # 5m, 30m, 15m
    }
}


@dataclass
class Attachment:
    """File attachment."""
    id: str
    filename: str
    content_type: str
    size_bytes: int
    url: str
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class TicketComment:
    """Ticket comment or interaction."""
    id: str
    author_id: str
    author_type: str  # "customer", "agent", "system"
    content: str
    is_internal: bool = False
    attachments: List[Attachment] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TicketAssignment:
    """Ticket assignment information."""
    agent_id: str
    agent_name: str
    assigned_at: datetime
    assigned_by: str
    reason: Optional[str] = None


@dataclass
class Ticket:
    """Support ticket."""
    id: str
    subject: str
    description: str
    customer: CustomerInfo
    status: TicketStatus = TicketStatus.OPEN
    priority: TicketPriority = TicketPriority.MEDIUM
    category: TicketCategory = TicketCategory.GENERAL
    tags: List[str] = field(default_factory=list)
    
    # Assignment and routing
    assigned_agent: Optional[TicketAssignment] = None
    escalated_to: Optional[str] = None
    routing_hints: Dict[str, Any] = field(default_factory=dict)
    
    # Timeline
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    first_response_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None
    
    # Content
    comments: List[TicketComment] = field(default_factory=list)
    attachments: List[Attachment] = field(default_factory=list)
    
    # Metadata
    source: str = "web"  # web, email, api, chat
    channel: str = "support"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # AI Analysis
    classification_confidence: Optional[float] = None
    sentiment_score: Optional[float] = None
    urgency_indicators: List[str] = field(default_factory=list)
    knowledge_base_suggestions: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_comment(self, author_id: str, author_type: str, content: str, 
                   is_internal: bool = False, attachments: Optional[List[Attachment]] = None) -> TicketComment:
        """Add a comment to the ticket."""
        comment = TicketComment(
            id=f"comment_{len(self.comments) + 1}",
            author_id=author_id,
            author_type=author_type,
            content=content,
            is_internal=is_internal,
            attachments=attachments or []
        )
        self.comments.append(comment)
        self.updated_at = datetime.now(timezone.utc)
        return comment
    
    def assign_to_agent(self, agent_id: str, agent_name: str, assigned_by: str, reason: Optional[str] = None):
        """Assign ticket to an agent."""
        self.assigned_agent = TicketAssignment(
            agent_id=agent_id,
            agent_name=agent_name,
            assigned_at=datetime.now(timezone.utc),
            assigned_by=assigned_by,
            reason=reason
        )
        self.status = TicketStatus.IN_PROGRESS
        self.updated_at = datetime.now(timezone.utc)
    
    def escalate(self, escalated_to: str, reason: str):
        """Escalate the ticket."""
        self.escalated_to = escalated_to
        self.status = TicketStatus.ESCALATED
        self.priority = TicketPriority.HIGH if self.priority in [TicketPriority.LOW, TicketPriority.MEDIUM] else self.priority
        self.updated_at = datetime.now(timezone.utc)
        
        # Add system comment about escalation
        self.add_comment(
            "system",
            "system",
            f"Ticket escalated to {escalated_to}. Reason: {reason}",
            is_internal=True
        )
    
    def mark_first_response(self):
        """Mark the first response time."""
        if not self.first_response_at:
            self.first_response_at = datetime.now(timezone.utc)
    
    def resolve(self, resolved_by: str, resolution_note: str):
        """Resolve the ticket."""
        self.status = TicketStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)
        self.updated_at = self.resolved_at
        
        # Add resolution comment
        self.add_comment(
            resolved_by,
            "agent",
            f"Ticket resolved: {resolution_note}"
        )
    
    def close(self, closed_by: str, closure_note: Optional[str] = None):
        """Close the ticket."""
        self.status = TicketStatus.CLOSED
        self.closed_at = datetime.now(timezone.utc)
        self.updated_at = self.closed_at
        
        if closure_note:
            self.add_comment(
                closed_by,
                "agent",
                f"Ticket closed: {closure_note}"
            )
    
    def get_sla_target(self) -> SLATarget:
        """Get SLA target for this ticket."""
        return SLA_TARGETS[self.customer.sla_level][self.priority]
    
    def is_sla_breached(self) -> Dict[str, bool]:
        """Check if SLA targets are breached."""
        sla = self.get_sla_target()
        now = datetime.now(timezone.utc)
        
        # First response SLA
        first_response_breached = (
            not self.first_response_at and
            (now - self.created_at).total_seconds() > sla.first_response_minutes * 60
        )
        
        # Resolution SLA  
        resolution_breached = (
            self.status not in [TicketStatus.RESOLVED, TicketStatus.CLOSED] and
            (now - self.created_at).total_seconds() > sla.resolution_time_hours * 3600
        )
        
        # Escalation SLA
        escalation_due = (
            not self.escalated_to and
            self.status not in [TicketStatus.RESOLVED, TicketStatus.CLOSED] and
            (now - self.created_at).total_seconds() > sla.escalation_time_hours * 3600
        )
        
        return {
            "first_response_breached": first_response_breached,
            "resolution_breached": resolution_breached, 
            "escalation_due": escalation_due
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert ticket to dictionary."""
        return {
            "id": self.id,
            "subject": self.subject,
            "description": self.description,
            "customer": {
                "id": self.customer.id,
                "name": self.customer.name,
                "email": self.customer.email,
                "phone": self.customer.phone,
                "sla_level": self.customer.sla_level.value,
                "account_type": self.customer.account_type,
                "language": self.customer.language,
                "timezone": self.customer.timezone,
                "metadata": self.customer.metadata
            },
            "status": self.status.value,
            "priority": self.priority.value,
            "category": self.category.value,
            "tags": self.tags,
            "assigned_agent": {
                "agent_id": self.assigned_agent.agent_id,
                "agent_name": self.assigned_agent.agent_name,
                "assigned_at": self.assigned_agent.assigned_at.isoformat(),
                "assigned_by": self.assigned_agent.assigned_by,
                "reason": self.assigned_agent.reason
            } if self.assigned_agent else None,
            "escalated_to": self.escalated_to,
            "routing_hints": self.routing_hints,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "first_response_at": self.first_response_at.isoformat() if self.first_response_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "comments": [
                {
                    "id": comment.id,
                    "author_id": comment.author_id,
                    "author_type": comment.author_type,
                    "content": comment.content,
                    "is_internal": comment.is_internal,
                    "created_at": comment.created_at.isoformat(),
                    "metadata": comment.metadata
                }
                for comment in self.comments
            ],
            "attachments": [
                {
                    "id": att.id,
                    "filename": att.filename,
                    "content_type": att.content_type,
                    "size_bytes": att.size_bytes,
                    "url": att.url,
                    "uploaded_at": att.uploaded_at.isoformat()
                }
                for att in self.attachments
            ],
            "source": self.source,
            "channel": self.channel,
            "metadata": self.metadata,
            "classification_confidence": self.classification_confidence,
            "sentiment_score": self.sentiment_score,
            "urgency_indicators": self.urgency_indicators,
            "knowledge_base_suggestions": self.knowledge_base_suggestions
        }


@dataclass
class SupportMetrics:
    """Support team metrics."""
    agent_id: str
    time_period: str  # "hour", "day", "week", "month"
    tickets_assigned: int = 0
    tickets_resolved: int = 0
    tickets_escalated: int = 0
    avg_response_time_minutes: float = 0.0
    avg_resolution_time_hours: float = 0.0
    customer_satisfaction_score: float = 0.0
    sla_compliance_rate: float = 0.0
    utilization_rate: float = 0.0  # % of time actively working on tickets
    
    
@dataclass
class AgentAvailability:
    """Agent availability and capacity."""
    agent_id: str
    agent_name: str
    is_online: bool = False
    status: str = "available"  # "available", "busy", "away", "offline"
    specialties: List[str] = field(default_factory=list)
    languages: List[str] = field(default_factory=list)
    current_tickets: int = 0
    max_tickets: int = 5
    priority_preference: Optional[TicketPriority] = None
    category_expertise: List[TicketCategory] = field(default_factory=list)
    shift_start: Optional[str] = None  # HH:MM format
    shift_end: Optional[str] = None    # HH:MM format
    timezone: str = "UTC"
    last_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def can_accept_ticket(self, ticket: Ticket) -> bool:
        """Check if agent can accept the ticket."""
        if not self.is_online or self.status != "available":
            return False
        
        if self.current_tickets >= self.max_tickets:
            return False
        
        # Check category expertise
        if self.category_expertise and ticket.category not in self.category_expertise:
            return False
        
        # Check language support
        if self.languages and ticket.customer.language not in self.languages:
            return False
        
        return True
    
    def get_load_factor(self) -> float:
        """Get current workload as percentage."""
        return self.current_tickets / self.max_tickets if self.max_tickets > 0 else 1.0