"""
Core workflow data types and structures for AMP workflow orchestration.
"""

from typing import Dict, Any, List, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import uuid


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status."""
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class TaskType(Enum):
    """Types of workflow tasks."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    LOOP = "loop"
    SUBPROCESS = "subprocess"
    MANUAL = "manual"
    API_CALL = "api_call"
    DATA_TRANSFORM = "data_transform"
    DECISION = "decision"


class RetryStrategy(Enum):
    """Task retry strategies."""
    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class RetryConfig:
    """Retry configuration for tasks."""
    strategy: RetryStrategy = RetryStrategy.FIXED_DELAY
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    retry_on_codes: List[str] = field(default_factory=lambda: ["TIMEOUT", "NETWORK_ERROR"])


@dataclass
class TaskCondition:
    """Condition for task execution."""
    expression: str  # Python expression to evaluate
    variables: Dict[str, Any] = field(default_factory=dict)
    required_outputs: List[str] = field(default_factory=list)


@dataclass
class TaskDefinition:
    """Individual task definition within a workflow."""
    id: str
    name: str
    type: TaskType
    agent_id: Optional[str] = None
    capability: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)
    condition: Optional[TaskCondition] = None
    timeout_seconds: int = 300
    retry_config: RetryConfig = field(default_factory=RetryConfig)
    outputs: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    version: str
    description: str
    tasks: List[TaskDefinition]
    global_timeout_seconds: int = 3600
    global_retry_config: RetryConfig = field(default_factory=RetryConfig)
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


@dataclass
class TaskExecution:
    """Runtime task execution state."""
    task_id: str
    workflow_instance_id: str
    status: TaskStatus
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    attempt_count: int = 0
    last_error: Optional[str] = None
    outputs: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None


@dataclass
class WorkflowInstance:
    """Runtime workflow instance."""
    id: str
    workflow_id: str
    status: WorkflowStatus
    inputs: Dict[str, Any]
    outputs: Dict[str, Any] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    task_executions: Dict[str, TaskExecution] = field(default_factory=dict)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowEvent:
    """Workflow execution event."""
    id: str
    workflow_instance_id: str
    task_id: Optional[str]
    event_type: str
    timestamp: datetime
    data: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def create(cls, workflow_instance_id: str, event_type: str, 
               task_id: Optional[str] = None, **data) -> 'WorkflowEvent':
        """Create a workflow event."""
        return cls(
            id=str(uuid.uuid4()),
            workflow_instance_id=workflow_instance_id,
            task_id=task_id,
            event_type=event_type,
            timestamp=datetime.now(timezone.utc),
            data=data
        )


@dataclass
class WorkflowMetrics:
    """Workflow performance metrics."""
    workflow_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_execution_time_seconds: float
    min_execution_time_seconds: float
    max_execution_time_seconds: float
    last_executed: datetime
    most_common_failures: List[str] = field(default_factory=list)


@dataclass
class TaskMetrics:
    """Task performance metrics."""
    task_id: str
    workflow_id: str
    total_executions: int
    successful_executions: int
    failed_executions: int
    avg_execution_time_seconds: float
    avg_retry_count: float
    most_common_failures: List[str] = field(default_factory=list)


# Workflow event types
class WorkflowEventTypes:
    """Standard workflow event types."""
    WORKFLOW_STARTED = "workflow.started"
    WORKFLOW_COMPLETED = "workflow.completed"
    WORKFLOW_FAILED = "workflow.failed"
    WORKFLOW_CANCELLED = "workflow.cancelled"
    WORKFLOW_PAUSED = "workflow.paused"
    WORKFLOW_RESUMED = "workflow.resumed"
    
    TASK_STARTED = "task.started"
    TASK_COMPLETED = "task.completed"
    TASK_FAILED = "task.failed"
    TASK_RETRYING = "task.retrying"
    TASK_SKIPPED = "task.skipped"
    
    STATE_UPDATED = "state.updated"
    CONDITION_EVALUATED = "condition.evaluated"
    ERROR_HANDLED = "error.handled"


# Standard workflow error codes
class WorkflowErrorCodes:
    """Standard workflow error codes."""
    WORKFLOW_NOT_FOUND = "WORKFLOW_NOT_FOUND"
    TASK_NOT_FOUND = "TASK_NOT_FOUND"
    INVALID_WORKFLOW_DEFINITION = "INVALID_WORKFLOW_DEFINITION"
    CIRCULAR_DEPENDENCY = "CIRCULAR_DEPENDENCY"
    CONDITION_EVALUATION_FAILED = "CONDITION_EVALUATION_FAILED"
    TASK_EXECUTION_FAILED = "TASK_EXECUTION_FAILED"
    TIMEOUT_EXCEEDED = "TIMEOUT_EXCEEDED"
    RESOURCE_NOT_AVAILABLE = "RESOURCE_NOT_AVAILABLE"
    INVALID_INPUT = "INVALID_INPUT"
    STATE_CORRUPTION = "STATE_CORRUPTION"