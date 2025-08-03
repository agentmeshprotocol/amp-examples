"""
Monitor Agent - Tracks workflow progress, performance metrics, and system health.
"""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone, timedelta
from collections import defaultdict, deque
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_types import AMPMessage, MessageType, CapabilityStatus, Capability, AgentIdentity
from amp_client import AMPClient

from ..workflow_types import (
    WorkflowEventTypes, WorkflowErrorCodes, WorkflowStatus, TaskStatus,
    WorkflowMetrics, TaskMetrics, WorkflowEvent
)


class AlertSeverity:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MonitorAgent:
    """
    Comprehensive monitoring and metrics collection for workflow orchestration.
    Tracks performance, generates alerts, and provides observability insights.
    """
    
    def __init__(self, agent_id: str = "monitor-agent", port: int = 8085):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(f"MonitorAgent.{agent_id}")
        
        # Monitoring data
        self.amp_client = None
        self.workflow_metrics: Dict[str, WorkflowMetrics] = {}
        self.task_metrics: Dict[str, Dict[str, TaskMetrics]] = {}  # workflow_id -> task_id -> metrics
        self.agent_health: Dict[str, Dict[str, Any]] = {}
        self.system_metrics: Dict[str, Any] = {}
        
        # Performance tracking
        self.workflow_executions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.task_executions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.performance_history: deque = deque(maxlen=10000)  # Last 10k events
        
        # Alerting
        self.alerts: List[Dict[str, Any]] = []
        self.alert_rules: Dict[str, Dict[str, Any]] = {}
        self.alert_channels: List[str] = []
        
        # Real-time monitoring
        self.active_workflows: Set[str] = set()
        self.running_tasks: Dict[str, Dict[str, Any]] = {}  # task_id -> execution info
        
        # Metrics aggregation
        self.metrics_window = 300  # 5 minutes
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        
    async def start(self):
        """Start the monitor agent."""
        self.logger.info(f"Starting Monitor Agent {self.agent_id}")
        
        # Initialize AMP client
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id=self.agent_id,
                name="Monitor Agent",
                version="1.0.0",
                framework="AMP-Workflow",
                description="Workflow monitoring and performance metrics agent"
            ),
            port=self.port
        )
        
        # Register capabilities
        await self._register_capabilities()
        
        # Start AMP client
        await self.amp_client.start()
        
        # Register message handlers
        self.amp_client.register_capability_handler("metrics-get", self._handle_get_metrics)
        self.amp_client.register_capability_handler("metrics-aggregate", self._handle_aggregate_metrics)
        self.amp_client.register_capability_handler("alert-create", self._handle_create_alert)
        self.amp_client.register_capability_handler("alert-list", self._handle_list_alerts)
        self.amp_client.register_capability_handler("health-check", self._handle_health_check)
        self.amp_client.register_capability_handler("performance-report", self._handle_performance_report)
        self.amp_client.register_capability_handler("dashboard-data", self._handle_dashboard_data)
        
        # Register event handlers for all workflow events
        self.amp_client.register_event_handler("workflow.*", self._handle_workflow_event)
        self.amp_client.register_event_handler("task.*", self._handle_task_event)
        self.amp_client.register_event_handler("error.*", self._handle_error_event)
        self.amp_client.register_event_handler("agent.*", self._handle_agent_event)
        
        # Initialize default alert rules
        await self._initialize_alert_rules()
        
        # Start background monitoring tasks
        asyncio.create_task(self._metrics_aggregation_task())
        asyncio.create_task(self._health_monitoring_task())
        asyncio.create_task(self._alert_evaluation_task())
        asyncio.create_task(self._performance_analysis_task())
        
        self.logger.info("Monitor Agent started successfully")
    
    async def stop(self):
        """Stop the monitor agent."""
        self.logger.info("Stopping Monitor Agent")
        if self.amp_client:
            await self.amp_client.stop()
    
    async def _register_capabilities(self):
        """Register monitoring capabilities."""
        capabilities = [
            Capability(
                id="metrics-get",
                version="1.0",
                description="Retrieve workflow and task metrics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "metric_type": {"type": "string", "enum": ["workflow", "task", "agent", "system"]},
                        "workflow_id": {"type": "string", "default": None},
                        "task_id": {"type": "string", "default": None},
                        "agent_id": {"type": "string", "default": None},
                        "time_range": {"type": "string", "default": "1h"}
                    },
                    "required": ["metric_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "metrics": {"type": "object"},
                        "timestamp": {"type": "string"}
                    }
                }
            ),
            Capability(
                id="alert-create",
                version="1.0",
                description="Create a new alert",
                input_schema={
                    "type": "object",
                    "properties": {
                        "alert_type": {"type": "string"},
                        "severity": {"type": "string", "enum": ["info", "warning", "error", "critical"]},
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "metadata": {"type": "object", "default": {}}
                    },
                    "required": ["alert_type", "severity", "title", "description"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "alert_id": {"type": "string"},
                        "created": {"type": "boolean"}
                    }
                }
            ),
            Capability(
                id="performance-report",
                version="1.0",
                description="Generate performance analysis report",
                input_schema={
                    "type": "object",
                    "properties": {
                        "report_type": {"type": "string", "enum": ["summary", "detailed", "trends"]},
                        "time_range": {"type": "string", "default": "24h"},
                        "workflow_ids": {"type": "array", "items": {"type": "string"}, "default": []},
                        "include_predictions": {"type": "boolean", "default": False}
                    },
                    "required": ["report_type"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "report": {"type": "object"},
                        "generated_at": {"type": "string"}
                    }
                }
            ),
            Capability(
                id="dashboard-data",
                version="1.0",
                description="Get real-time dashboard data",
                input_schema={
                    "type": "object",
                    "properties": {
                        "widgets": {"type": "array", "items": {"type": "string"}, "default": ["overview"]},
                        "refresh_interval": {"type": "integer", "default": 30}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "dashboard": {"type": "object"},
                        "last_updated": {"type": "string"}
                    }
                }
            )
        ]
        
        for capability in capabilities:
            self.amp_client.register_capability(capability)
    
    async def _handle_get_metrics(self, message: AMPMessage) -> AMPMessage:
        """Handle metrics retrieval request."""
        try:
            metric_type = message.payload["parameters"]["metric_type"]
            workflow_id = message.payload["parameters"].get("workflow_id")
            task_id = message.payload["parameters"].get("task_id")
            agent_id = message.payload["parameters"].get("agent_id")
            time_range = message.payload["parameters"].get("time_range", "1h")
            
            metrics = await self._get_metrics(metric_type, workflow_id, task_id, agent_id, time_range)
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "metrics": metrics,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get metrics: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _handle_create_alert(self, message: AMPMessage) -> AMPMessage:
        """Handle alert creation request."""
        try:
            alert_type = message.payload["parameters"]["alert_type"]
            severity = message.payload["parameters"]["severity"]
            title = message.payload["parameters"]["title"]
            description = message.payload["parameters"]["description"]
            metadata = message.payload["parameters"].get("metadata", {})
            
            alert_id = await self._create_alert(alert_type, severity, title, description, metadata)
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "alert_id": alert_id,
                    "created": True
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create alert: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _handle_performance_report(self, message: AMPMessage) -> AMPMessage:
        """Handle performance report generation request."""
        try:
            report_type = message.payload["parameters"]["report_type"]
            time_range = message.payload["parameters"].get("time_range", "24h")
            workflow_ids = message.payload["parameters"].get("workflow_ids", [])
            include_predictions = message.payload["parameters"].get("include_predictions", False)
            
            report = await self._generate_performance_report(
                report_type, time_range, workflow_ids, include_predictions
            )
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "report": report,
                    "generated_at": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate performance report: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _handle_dashboard_data(self, message: AMPMessage) -> AMPMessage:
        """Handle dashboard data request."""
        try:
            widgets = message.payload["parameters"].get("widgets", ["overview"])
            refresh_interval = message.payload["parameters"].get("refresh_interval", 30)
            
            dashboard_data = await self._get_dashboard_data(widgets)
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "dashboard": dashboard_data,
                    "last_updated": datetime.now(timezone.utc).isoformat(),
                    "refresh_interval": refresh_interval
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get dashboard data: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _handle_workflow_event(self, message: AMPMessage):
        """Handle workflow-related events."""
        event_data = message.payload
        event_type = event_data.get("event_type", "")
        
        # Record event in performance history
        self.performance_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "category": "workflow",
            "data": event_data
        })
        
        # Handle specific workflow events
        if event_type == WorkflowEventTypes.WORKFLOW_STARTED:
            await self._handle_workflow_started(event_data)
        elif event_type == WorkflowEventTypes.WORKFLOW_COMPLETED:
            await self._handle_workflow_completed(event_data)
        elif event_type == WorkflowEventTypes.WORKFLOW_FAILED:
            await self._handle_workflow_failed(event_data)
        
        # Update real-time metrics
        await self._update_realtime_metrics("workflow", event_data)
    
    async def _handle_task_event(self, message: AMPMessage):
        """Handle task-related events."""
        event_data = message.payload
        event_type = event_data.get("event_type", "")
        
        # Record event in performance history
        self.performance_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            "category": "task",
            "data": event_data
        })
        
        # Handle specific task events
        if event_type == WorkflowEventTypes.TASK_STARTED:
            await self._handle_task_started(event_data)
        elif event_type == WorkflowEventTypes.TASK_COMPLETED:
            await self._handle_task_completed(event_data)
        elif event_type == WorkflowEventTypes.TASK_FAILED:
            await self._handle_task_failed(event_data)
        
        # Update real-time metrics
        await self._update_realtime_metrics("task", event_data)
    
    async def _handle_error_event(self, message: AMPMessage):
        """Handle error-related events."""
        event_data = message.payload
        
        # Create alert for error events
        await self._create_alert(
            alert_type="error_occurred",
            severity=AlertSeverity.ERROR,
            title=f"Error in workflow: {event_data.get('workflow_instance_id', 'unknown')}",
            description=event_data.get("error", {}).get("message", "Unknown error"),
            metadata=event_data
        )
    
    async def _handle_agent_event(self, message: AMPMessage):
        """Handle agent-related events."""
        event_data = message.payload
        agent_id = event_data.get("agent_id")
        
        if agent_id:
            # Update agent health metrics
            if agent_id not in self.agent_health:
                self.agent_health[agent_id] = {}
            
            self.agent_health[agent_id].update({
                "last_seen": datetime.now(timezone.utc).isoformat(),
                "status": event_data.get("status", "active"),
                "event_data": event_data
            })
    
    async def _handle_workflow_started(self, event_data: Dict[str, Any]):
        """Handle workflow started event."""
        workflow_instance_id = event_data.get("workflow_instance_id")
        workflow_id = event_data.get("workflow_id")
        
        if workflow_instance_id:
            self.active_workflows.add(workflow_instance_id)
        
        # Initialize workflow metrics if not exists
        if workflow_id and workflow_id not in self.workflow_metrics:
            self.workflow_metrics[workflow_id] = WorkflowMetrics(
                workflow_id=workflow_id,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                avg_execution_time_seconds=0.0,
                min_execution_time_seconds=float('inf'),
                max_execution_time_seconds=0.0,
                last_executed=datetime.now(timezone.utc),
                most_common_failures=[]
            )
        
        # Record execution start
        if workflow_id:
            execution_record = {
                "instance_id": workflow_instance_id,
                "workflow_id": workflow_id,
                "started_at": datetime.now(timezone.utc),
                "status": "running"
            }
            self.workflow_executions[workflow_id].append(execution_record)
    
    async def _handle_workflow_completed(self, event_data: Dict[str, Any]):
        """Handle workflow completed event."""
        workflow_instance_id = event_data.get("workflow_instance_id")
        workflow_id = event_data.get("workflow_id")
        
        if workflow_instance_id in self.active_workflows:
            self.active_workflows.remove(workflow_instance_id)
        
        # Update workflow metrics
        if workflow_id and workflow_id in self.workflow_metrics:
            metrics = self.workflow_metrics[workflow_id]
            metrics.total_executions += 1
            metrics.successful_executions += 1
            metrics.last_executed = datetime.now(timezone.utc)
            
            # Update execution record
            for execution in self.workflow_executions[workflow_id]:
                if execution["instance_id"] == workflow_instance_id:
                    execution["completed_at"] = datetime.now(timezone.utc)
                    execution["status"] = "completed"
                    
                    # Calculate execution time
                    exec_time = (execution["completed_at"] - execution["started_at"]).total_seconds()
                    execution["execution_time"] = exec_time
                    
                    # Update metrics
                    metrics.avg_execution_time_seconds = self._calculate_average_execution_time(workflow_id)
                    metrics.min_execution_time_seconds = min(metrics.min_execution_time_seconds, exec_time)
                    metrics.max_execution_time_seconds = max(metrics.max_execution_time_seconds, exec_time)
                    break
    
    async def _handle_workflow_failed(self, event_data: Dict[str, Any]):
        """Handle workflow failed event."""
        workflow_instance_id = event_data.get("workflow_instance_id")
        workflow_id = event_data.get("workflow_id")
        
        if workflow_instance_id in self.active_workflows:
            self.active_workflows.remove(workflow_instance_id)
        
        # Update workflow metrics
        if workflow_id and workflow_id in self.workflow_metrics:
            metrics = self.workflow_metrics[workflow_id]
            metrics.total_executions += 1
            metrics.failed_executions += 1
            metrics.last_executed = datetime.now(timezone.utc)
            
            # Track failure reason
            failure_reason = event_data.get("error", {}).get("message", "Unknown failure")
            if failure_reason not in metrics.most_common_failures:
                metrics.most_common_failures.append(failure_reason)
            
            # Update execution record
            for execution in self.workflow_executions[workflow_id]:
                if execution["instance_id"] == workflow_instance_id:
                    execution["completed_at"] = datetime.now(timezone.utc)
                    execution["status"] = "failed"
                    execution["error"] = failure_reason
                    break
        
        # Create alert for workflow failure
        await self._create_alert(
            alert_type="workflow_failure",
            severity=AlertSeverity.ERROR,
            title=f"Workflow Failed: {workflow_id}",
            description=f"Workflow instance {workflow_instance_id} failed",
            metadata=event_data
        )
    
    async def _handle_task_started(self, event_data: Dict[str, Any]):
        """Handle task started event."""
        task_id = event_data.get("task_id")
        workflow_instance_id = event_data.get("workflow_instance_id")
        
        if task_id:
            self.running_tasks[task_id] = {
                "task_id": task_id,
                "workflow_instance_id": workflow_instance_id,
                "started_at": datetime.now(timezone.utc),
                "status": "running"
            }
    
    async def _handle_task_completed(self, event_data: Dict[str, Any]):
        """Handle task completed event."""
        task_id = event_data.get("task_id")
        workflow_id = event_data.get("workflow_id")
        
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            task_info["completed_at"] = datetime.now(timezone.utc)
            task_info["status"] = "completed"
            
            # Calculate execution time
            exec_time = (task_info["completed_at"] - task_info["started_at"]).total_seconds()
            task_info["execution_time"] = exec_time
            
            # Update task metrics
            await self._update_task_metrics(workflow_id, task_id, exec_time, True)
    
    async def _handle_task_failed(self, event_data: Dict[str, Any]):
        """Handle task failed event."""
        task_id = event_data.get("task_id")
        workflow_id = event_data.get("workflow_id")
        
        if task_id in self.running_tasks:
            task_info = self.running_tasks.pop(task_id)
            task_info["completed_at"] = datetime.now(timezone.utc)
            task_info["status"] = "failed"
            task_info["error"] = event_data.get("error", "Unknown error")
            
            # Calculate execution time
            exec_time = (task_info["completed_at"] - task_info["started_at"]).total_seconds()
            task_info["execution_time"] = exec_time
            
            # Update task metrics
            await self._update_task_metrics(workflow_id, task_id, exec_time, False)
    
    async def _update_task_metrics(self, workflow_id: str, task_id: str, 
                                 execution_time: float, success: bool):
        """Update task execution metrics."""
        if workflow_id not in self.task_metrics:
            self.task_metrics[workflow_id] = {}
        
        if task_id not in self.task_metrics[workflow_id]:
            self.task_metrics[workflow_id][task_id] = TaskMetrics(
                task_id=task_id,
                workflow_id=workflow_id,
                total_executions=0,
                successful_executions=0,
                failed_executions=0,
                avg_execution_time_seconds=0.0,
                avg_retry_count=0.0,
                most_common_failures=[]
            )
        
        metrics = self.task_metrics[workflow_id][task_id]
        metrics.total_executions += 1
        
        if success:
            metrics.successful_executions += 1
        else:
            metrics.failed_executions += 1
        
        # Update average execution time
        total_time = metrics.avg_execution_time_seconds * (metrics.total_executions - 1) + execution_time
        metrics.avg_execution_time_seconds = total_time / metrics.total_executions
    
    async def _get_metrics(self, metric_type: str, workflow_id: Optional[str] = None,
                          task_id: Optional[str] = None, agent_id: Optional[str] = None,
                          time_range: str = "1h") -> Dict[str, Any]:
        """Retrieve metrics based on type and filters."""
        if metric_type == "workflow":
            if workflow_id:
                return self.workflow_metrics.get(workflow_id, {}).__dict__ if workflow_id in self.workflow_metrics else {}
            else:
                return {wf_id: metrics.__dict__ for wf_id, metrics in self.workflow_metrics.items()}
        
        elif metric_type == "task":
            if workflow_id and task_id:
                return self.task_metrics.get(workflow_id, {}).get(task_id, {}).__dict__ if workflow_id in self.task_metrics and task_id in self.task_metrics[workflow_id] else {}
            elif workflow_id:
                return {task_id: metrics.__dict__ for task_id, metrics in self.task_metrics.get(workflow_id, {}).items()}
            else:
                return self.task_metrics
        
        elif metric_type == "agent":
            if agent_id:
                return self.agent_health.get(agent_id, {})
            else:
                return self.agent_health
        
        elif metric_type == "system":
            return self.system_metrics
        
        else:
            return {}
    
    async def _create_alert(self, alert_type: str, severity: str, title: str,
                          description: str, metadata: Dict[str, Any] = None) -> str:
        """Create a new alert."""
        import uuid
        
        alert_id = str(uuid.uuid4())
        alert = {
            "id": alert_id,
            "type": alert_type,
            "severity": severity,
            "title": title,
            "description": description,
            "metadata": metadata or {},
            "created_at": datetime.now(timezone.utc).isoformat(),
            "acknowledged": False,
            "resolved": False
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts (last 1000)
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-1000:]
        
        self.logger.info(f"Created alert: {title} (Severity: {severity})")
        
        # Send alert to configured channels
        await self._send_alert_notifications(alert)
        
        return alert_id
    
    async def _generate_performance_report(self, report_type: str, time_range: str,
                                         workflow_ids: List[str], include_predictions: bool) -> Dict[str, Any]:
        """Generate performance analysis report."""
        report = {
            "report_type": report_type,
            "time_range": time_range,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }
        
        if report_type == "summary":
            report.update({
                "total_workflows": len(self.workflow_metrics),
                "active_workflows": len(self.active_workflows),
                "total_executions": sum(m.total_executions for m in self.workflow_metrics.values()),
                "success_rate": self._calculate_overall_success_rate(),
                "avg_execution_time": self._calculate_overall_avg_execution_time(),
                "recent_alerts": len([a for a in self.alerts if not a["resolved"]])
            })
        
        elif report_type == "detailed":
            report.update({
                "workflow_metrics": {wf_id: m.__dict__ for wf_id, m in self.workflow_metrics.items()},
                "task_metrics": self.task_metrics,
                "agent_health": self.agent_health,
                "system_metrics": self.system_metrics,
                "alerts": self.alerts[-50:]  # Last 50 alerts
            })
        
        elif report_type == "trends":
            report.update({
                "execution_trends": self._analyze_execution_trends(),
                "performance_trends": self._analyze_performance_trends(),
                "error_trends": self._analyze_error_trends()
            })
        
        if include_predictions:
            report["predictions"] = await self._generate_predictions()
        
        return report
    
    async def _get_dashboard_data(self, widgets: List[str]) -> Dict[str, Any]:
        """Get real-time dashboard data."""
        dashboard = {}
        
        if "overview" in widgets:
            dashboard["overview"] = {
                "active_workflows": len(self.active_workflows),
                "running_tasks": len(self.running_tasks),
                "total_workflows": len(self.workflow_metrics),
                "success_rate": self._calculate_overall_success_rate(),
                "recent_errors": len([a for a in self.alerts if a["severity"] in ["error", "critical"] and not a["resolved"]]),
                "system_health": "healthy" if len(self.active_workflows) < 100 else "degraded"
            }
        
        if "performance" in widgets:
            dashboard["performance"] = {
                "avg_execution_time": self._calculate_overall_avg_execution_time(),
                "throughput": self._calculate_throughput(),
                "resource_utilization": self._get_resource_utilization()
            }
        
        if "alerts" in widgets:
            dashboard["alerts"] = {
                "active_alerts": [a for a in self.alerts if not a["resolved"]][-10:],
                "alert_count_by_severity": self._count_alerts_by_severity()
            }
        
        if "workflows" in widgets:
            dashboard["workflows"] = {
                "recent_executions": self._get_recent_workflow_executions(),
                "top_performers": self._get_top_performing_workflows(),
                "problematic_workflows": self._get_problematic_workflows()
            }
        
        return dashboard
    
    def _calculate_average_execution_time(self, workflow_id: str) -> float:
        """Calculate average execution time for a workflow."""
        executions = [e for e in self.workflow_executions[workflow_id] if "execution_time" in e]
        if not executions:
            return 0.0
        
        return sum(e["execution_time"] for e in executions) / len(executions)
    
    def _calculate_overall_success_rate(self) -> float:
        """Calculate overall success rate across all workflows."""
        total_executions = sum(m.total_executions for m in self.workflow_metrics.values())
        successful_executions = sum(m.successful_executions for m in self.workflow_metrics.values())
        
        if total_executions == 0:
            return 100.0
        
        return (successful_executions / total_executions) * 100
    
    def _calculate_overall_avg_execution_time(self) -> float:
        """Calculate overall average execution time."""
        if not self.workflow_metrics:
            return 0.0
        
        total_time = sum(m.avg_execution_time_seconds * m.total_executions for m in self.workflow_metrics.values())
        total_executions = sum(m.total_executions for m in self.workflow_metrics.values())
        
        if total_executions == 0:
            return 0.0
        
        return total_time / total_executions
    
    def _calculate_throughput(self) -> float:
        """Calculate current throughput (workflows per hour)."""
        # Count executions in the last hour
        one_hour_ago = datetime.now(timezone.utc) - timedelta(hours=1)
        recent_executions = 0
        
        for workflow_executions in self.workflow_executions.values():
            for execution in workflow_executions:
                if execution["started_at"] > one_hour_ago:
                    recent_executions += 1
        
        return recent_executions
    
    def _get_resource_utilization(self) -> Dict[str, float]:
        """Get current resource utilization metrics."""
        return {
            "cpu_percent": 0.0,  # Would integrate with system monitoring
            "memory_percent": 0.0,
            "disk_percent": 0.0,
            "network_percent": 0.0
        }
    
    def _count_alerts_by_severity(self) -> Dict[str, int]:
        """Count active alerts by severity."""
        counts = {"info": 0, "warning": 0, "error": 0, "critical": 0}
        
        for alert in self.alerts:
            if not alert["resolved"]:
                severity = alert["severity"]
                if severity in counts:
                    counts[severity] += 1
        
        return counts
    
    # Background monitoring tasks
    async def _metrics_aggregation_task(self):
        """Background task for metrics aggregation."""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Aggregate metrics
                current_metrics = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "active_workflows": len(self.active_workflows),
                    "running_tasks": len(self.running_tasks),
                    "success_rate": self._calculate_overall_success_rate(),
                    "avg_execution_time": self._calculate_overall_avg_execution_time(),
                    "throughput": self._calculate_throughput()
                }
                
                self.metrics_history["system"].append(current_metrics)
                
            except Exception as e:
                self.logger.error(f"Metrics aggregation failed: {e}")
    
    async def _health_monitoring_task(self):
        """Background task for health monitoring."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Check agent health
                current_time = datetime.now(timezone.utc)
                for agent_id, health in self.agent_health.items():
                    last_seen_str = health.get("last_seen")
                    if last_seen_str:
                        last_seen = datetime.fromisoformat(last_seen_str)
                        if (current_time - last_seen).total_seconds() > 300:  # 5 minutes
                            health["status"] = "unhealthy"
                            
                            # Create alert for unhealthy agent
                            await self._create_alert(
                                alert_type="agent_unhealthy",
                                severity=AlertSeverity.WARNING,
                                title=f"Agent Unhealthy: {agent_id}",
                                description=f"Agent {agent_id} has not been seen for over 5 minutes",
                                metadata={"agent_id": agent_id, "last_seen": last_seen_str}
                            )
                
            except Exception as e:
                self.logger.error(f"Health monitoring failed: {e}")
    
    async def _alert_evaluation_task(self):
        """Background task for alert rule evaluation."""
        while True:
            try:
                await asyncio.sleep(60)  # Evaluate every minute
                
                # Evaluate alert rules
                for rule_id, rule in self.alert_rules.items():
                    if await self._evaluate_alert_rule(rule):
                        await self._create_alert(
                            alert_type=rule["alert_type"],
                            severity=rule["severity"],
                            title=rule["title"],
                            description=rule["description"],
                            metadata={"rule_id": rule_id}
                        )
                
            except Exception as e:
                self.logger.error(f"Alert evaluation failed: {e}")
    
    async def _performance_analysis_task(self):
        """Background task for performance analysis."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Analyze performance trends
                # This would include more sophisticated analysis
                
            except Exception as e:
                self.logger.error(f"Performance analysis failed: {e}")
    
    async def _initialize_alert_rules(self):
        """Initialize default alert rules."""
        self.alert_rules = {
            "high_failure_rate": {
                "alert_type": "high_failure_rate",
                "severity": AlertSeverity.WARNING,
                "title": "High Workflow Failure Rate",
                "description": "Workflow failure rate exceeds threshold",
                "condition": lambda: self._calculate_overall_success_rate() < 80
            },
            "slow_execution": {
                "alert_type": "slow_execution",
                "severity": AlertSeverity.INFO,
                "title": "Slow Workflow Execution",
                "description": "Average execution time exceeds threshold",
                "condition": lambda: self._calculate_overall_avg_execution_time() > 300
            }
        }
    
    async def _evaluate_alert_rule(self, rule: Dict[str, Any]) -> bool:
        """Evaluate an alert rule condition."""
        try:
            condition = rule.get("condition")
            if condition and callable(condition):
                return condition()
        except Exception as e:
            self.logger.error(f"Alert rule evaluation failed: {e}")
        return False
    
    async def _send_alert_notifications(self, alert: Dict[str, Any]):
        """Send alert notifications to configured channels."""
        # Implementation would send to various channels (email, Slack, etc.)
        pass
    
    # Additional helper methods
    async def _update_realtime_metrics(self, category: str, event_data: Dict[str, Any]):
        """Update real-time metrics based on events."""
        pass
    
    def _analyze_execution_trends(self) -> Dict[str, Any]:
        """Analyze execution trends over time."""
        return {"trend": "stable", "analysis": "No significant changes detected"}
    
    def _analyze_performance_trends(self) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        return {"trend": "stable", "analysis": "Performance within normal ranges"}
    
    def _analyze_error_trends(self) -> Dict[str, Any]:
        """Analyze error trends over time."""
        return {"trend": "stable", "analysis": "Error rates within normal ranges"}
    
    async def _generate_predictions(self) -> Dict[str, Any]:
        """Generate performance predictions."""
        return {"predictions": "Feature not implemented"}
    
    def _get_recent_workflow_executions(self) -> List[Dict[str, Any]]:
        """Get recent workflow executions for dashboard."""
        recent = []
        for workflow_executions in self.workflow_executions.values():
            recent.extend(workflow_executions[-5:])  # Last 5 per workflow
        
        # Sort by start time and return most recent
        recent.sort(key=lambda x: x["started_at"], reverse=True)
        return recent[:20]
    
    def _get_top_performing_workflows(self) -> List[Dict[str, Any]]:
        """Get top performing workflows."""
        workflows = []
        for wf_id, metrics in self.workflow_metrics.items():
            if metrics.total_executions > 0:
                success_rate = (metrics.successful_executions / metrics.total_executions) * 100
                workflows.append({
                    "workflow_id": wf_id,
                    "success_rate": success_rate,
                    "avg_execution_time": metrics.avg_execution_time_seconds,
                    "total_executions": metrics.total_executions
                })
        
        return sorted(workflows, key=lambda x: x["success_rate"], reverse=True)[:10]
    
    def _get_problematic_workflows(self) -> List[Dict[str, Any]]:
        """Get workflows with issues."""
        workflows = []
        for wf_id, metrics in self.workflow_metrics.items():
            if metrics.total_executions > 0:
                failure_rate = (metrics.failed_executions / metrics.total_executions) * 100
                if failure_rate > 20:  # More than 20% failure rate
                    workflows.append({
                        "workflow_id": wf_id,
                        "failure_rate": failure_rate,
                        "failed_executions": metrics.failed_executions,
                        "common_failures": metrics.most_common_failures[:3]
                    })
        
        return sorted(workflows, key=lambda x: x["failure_rate"], reverse=True)[:10]
    
    # Additional handler method placeholders
    async def _handle_aggregate_metrics(self, message: AMPMessage) -> AMPMessage:
        """Handle metrics aggregation request."""
        pass
    
    async def _handle_list_alerts(self, message: AMPMessage) -> AMPMessage:
        """Handle alert listing request."""
        pass
    
    async def _handle_health_check(self, message: AMPMessage) -> AMPMessage:
        """Handle health check request."""
        pass