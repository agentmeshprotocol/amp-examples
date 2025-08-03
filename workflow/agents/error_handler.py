"""
Error Handler Agent - Manages workflow failures, recovery strategies, and error handling.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Set
from datetime import datetime, timezone, timedelta
from enum import Enum
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_types import AMPMessage, MessageType, CapabilityStatus, Capability, AgentIdentity
from amp_client import AMPClient

from ..workflow_types import (
    WorkflowEventTypes, WorkflowErrorCodes, RetryStrategy, 
    WorkflowStatus, TaskStatus
)


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryAction(Enum):
    """Available recovery actions."""
    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ESCALATE = "escalate"
    ROLLBACK = "rollback"
    PAUSE = "pause"
    ABORT = "abort"


class ErrorHandler:
    """
    Comprehensive error handling and recovery management for workflow orchestration.
    Handles failures, implements recovery strategies, and manages escalation policies.
    """
    
    def __init__(self, agent_id: str = "error-handler", port: int = 8084):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(f"ErrorHandler.{agent_id}")
        
        # Error handling
        self.amp_client = None
        self.error_history: Dict[str, List[Dict[str, Any]]] = {}
        self.recovery_handlers: Dict[str, Callable] = {}
        self.escalation_policies: Dict[str, Dict[str, Any]] = {}
        self.circuit_breakers: Dict[str, Dict[str, Any]] = {}
        
        # Error patterns and thresholds
        self.error_patterns: Dict[str, Dict[str, Any]] = {}
        self.alert_thresholds = {
            ErrorSeverity.LOW: {"count": 10, "window_minutes": 60},
            ErrorSeverity.MEDIUM: {"count": 5, "window_minutes": 30},
            ErrorSeverity.HIGH: {"count": 3, "window_minutes": 15},
            ErrorSeverity.CRITICAL: {"count": 1, "window_minutes": 5}
        }
        
        # Dependencies
        self.workflow_engine_id = "workflow-engine"
        self.state_manager_id = "state-manager"
        self.monitor_agent_id = "monitor-agent"
        
    async def start(self):
        """Start the error handler agent."""
        self.logger.info(f"Starting Error Handler {self.agent_id}")
        
        # Initialize AMP client
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id=self.agent_id,
                name="Error Handler",
                version="1.0.0",
                framework="AMP-Workflow",
                description="Workflow error handling and recovery manager"
            ),
            port=self.port
        )
        
        # Register capabilities
        await self._register_capabilities()
        
        # Start AMP client
        await self.amp_client.start()
        
        # Register message handlers
        self.amp_client.register_capability_handler("error-handle", self._handle_error)
        self.amp_client.register_capability_handler("error-recover", self._handle_recovery)
        self.amp_client.register_capability_handler("error-escalate", self._handle_escalation)
        self.amp_client.register_capability_handler("error-analyze", self._handle_error_analysis)
        self.amp_client.register_capability_handler("circuit-breaker", self._handle_circuit_breaker)
        self.amp_client.register_capability_handler("recovery-strategy", self._handle_recovery_strategy)
        self.amp_client.register_capability_handler("error-patterns", self._handle_error_patterns)
        
        # Register event handlers
        self.amp_client.register_event_handler("workflow.failed", self._handle_workflow_failed)
        self.amp_client.register_event_handler("task.failed", self._handle_task_failed)
        self.amp_client.register_event_handler("task.retrying", self._handle_task_retrying)
        
        # Initialize built-in error patterns and recovery strategies
        await self._initialize_builtin_patterns()
        
        # Start background tasks
        asyncio.create_task(self._error_monitoring_task())
        asyncio.create_task(self._circuit_breaker_task())
        
        self.logger.info("Error Handler started successfully")
    
    async def stop(self):
        """Stop the error handler agent."""
        self.logger.info("Stopping Error Handler")
        if self.amp_client:
            await self.amp_client.stop()
    
    async def _register_capabilities(self):
        """Register error handling capabilities."""
        capabilities = [
            Capability(
                id="error-handle",
                version="1.0",
                description="Handle and process workflow errors",
                input_schema={
                    "type": "object",
                    "properties": {
                        "error": {"type": "object"},
                        "context": {"type": "object", "default": {}},
                        "auto_recover": {"type": "boolean", "default": True},
                        "escalate_immediately": {"type": "boolean", "default": False}
                    },
                    "required": ["error"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "handled": {"type": "boolean"},
                        "recovery_action": {"type": "string"},
                        "escalated": {"type": "boolean"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            Capability(
                id="error-recover",
                version="1.0",
                description="Execute error recovery strategies",
                input_schema={
                    "type": "object",
                    "properties": {
                        "recovery_action": {"type": "string"},
                        "error_context": {"type": "object"},
                        "recovery_config": {"type": "object", "default": {}}
                    },
                    "required": ["recovery_action", "error_context"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "success": {"type": "boolean"},
                        "result": {"type": "object"},
                        "next_action": {"type": "string"}
                    }
                }
            ),
            Capability(
                id="error-analyze",
                version="1.0",
                description="Analyze error patterns and trends",
                input_schema={
                    "type": "object",
                    "properties": {
                        "time_window_hours": {"type": "number", "default": 24},
                        "error_types": {"type": "array", "default": []},
                        "workflow_ids": {"type": "array", "default": []}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "error_summary": {"type": "object"},
                        "patterns": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            Capability(
                id="circuit-breaker",
                version="1.0",
                description="Manage circuit breaker for system protection",
                input_schema={
                    "type": "object",
                    "properties": {
                        "operation": {"type": "string", "enum": ["check", "trip", "reset", "status"]},
                        "resource_id": {"type": "string"},
                        "config": {"type": "object", "default": {}}
                    },
                    "required": ["operation", "resource_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "state": {"type": "string"},
                        "failure_count": {"type": "integer"},
                        "last_failure": {"type": "string"},
                        "next_attempt": {"type": "string"}
                    }
                }
            )
        ]
        
        for capability in capabilities:
            self.amp_client.register_capability(capability)
    
    async def _handle_error(self, message: AMPMessage) -> AMPMessage:
        """Handle error processing request."""
        try:
            error = message.payload["parameters"]["error"]
            context = message.payload["parameters"].get("context", {})
            auto_recover = message.payload["parameters"].get("auto_recover", True)
            escalate_immediately = message.payload["parameters"].get("escalate_immediately", False)
            
            # Process the error
            error_info = await self._process_error(error, context)
            
            # Record the error
            await self._record_error(error_info)
            
            # Determine severity and recovery action
            severity = await self._classify_error_severity(error_info)
            recovery_action = await self._determine_recovery_action(error_info, severity)
            
            handled = False
            escalated = False
            recommendations = []
            
            # Handle escalation
            if escalate_immediately or severity == ErrorSeverity.CRITICAL:
                escalated = await self._escalate_error(error_info, severity)
            
            # Attempt recovery if enabled
            if auto_recover and recovery_action != RecoveryAction.ESCALATE:
                recovery_result = await self._execute_recovery(recovery_action, error_info)
                handled = recovery_result.get("success", False)
                
                if not handled:
                    escalated = await self._escalate_error(error_info, severity)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(error_info, severity)
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "handled": handled,
                    "recovery_action": recovery_action.value if recovery_action else None,
                    "escalated": escalated,
                    "recommendations": recommendations,
                    "severity": severity.value,
                    "error_id": error_info.get("id")
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle error: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.INTERNAL_ERROR,
                error_message=str(e)
            )
    
    async def _handle_recovery(self, message: AMPMessage) -> AMPMessage:
        """Handle error recovery execution request."""
        try:
            recovery_action = RecoveryAction(message.payload["parameters"]["recovery_action"])
            error_context = message.payload["parameters"]["error_context"]
            recovery_config = message.payload["parameters"].get("recovery_config", {})
            
            # Execute recovery strategy
            result = await self._execute_recovery(recovery_action, error_context, recovery_config)
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute recovery: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _handle_error_analysis(self, message: AMPMessage) -> AMPMessage:
        """Handle error analysis request."""
        try:
            time_window_hours = message.payload["parameters"].get("time_window_hours", 24)
            error_types = message.payload["parameters"].get("error_types", [])
            workflow_ids = message.payload["parameters"].get("workflow_ids", [])
            
            # Analyze error patterns
            analysis_result = await self._analyze_error_patterns(
                time_window_hours, error_types, workflow_ids
            )
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=analysis_result
            )
            
        except Exception as e:
            self.logger.error(f"Failed to analyze errors: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _handle_circuit_breaker(self, message: AMPMessage) -> AMPMessage:
        """Handle circuit breaker management request."""
        try:
            operation = message.payload["parameters"]["operation"]
            resource_id = message.payload["parameters"]["resource_id"]
            config = message.payload["parameters"].get("config", {})
            
            if operation == "check":
                result = await self._check_circuit_breaker(resource_id)
            elif operation == "trip":
                result = await self._trip_circuit_breaker(resource_id, config)
            elif operation == "reset":
                result = await self._reset_circuit_breaker(resource_id)
            elif operation == "status":
                result = await self._get_circuit_breaker_status(resource_id)
            else:
                raise ValueError(f"Unknown circuit breaker operation: {operation}")
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            self.logger.error(f"Failed to handle circuit breaker: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.EXECUTION_ERROR,
                error_message=str(e)
            )
    
    async def _process_error(self, error: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich error information."""
        import uuid
        
        error_info = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_code": error.get("code", "UNKNOWN_ERROR"),
            "error_message": error.get("message", "Unknown error"),
            "error_details": error.get("details", {}),
            "context": context,
            "workflow_instance_id": context.get("workflow_instance_id"),
            "task_id": context.get("task_id"),
            "agent_id": context.get("agent_id"),
            "original_error": error
        }
        
        # Enrich with additional context
        if error_info["workflow_instance_id"]:
            try:
                # Get workflow state
                state_response = await self.amp_client.send_request(
                    target_agent=self.state_manager_id,
                    capability="state-get",
                    parameters={"workflow_instance_id": error_info["workflow_instance_id"]}
                )
                
                if state_response.payload.get("status") == CapabilityStatus.SUCCESS.value:
                    error_info["workflow_state"] = state_response.payload.get("result", {}).get("state", {})
            except Exception as e:
                self.logger.warning(f"Failed to get workflow state: {e}")
        
        return error_info
    
    async def _record_error(self, error_info: Dict[str, Any]):
        """Record error in history for pattern analysis."""
        workflow_id = error_info.get("workflow_instance_id", "global")
        
        if workflow_id not in self.error_history:
            self.error_history[workflow_id] = []
        
        self.error_history[workflow_id].append(error_info)
        
        # Keep only recent errors (last 1000 per workflow)
        if len(self.error_history[workflow_id]) > 1000:
            self.error_history[workflow_id] = self.error_history[workflow_id][-1000:]
    
    async def _classify_error_severity(self, error_info: Dict[str, Any]) -> ErrorSeverity:
        """Classify error severity based on patterns and impact."""
        error_code = error_info.get("error_code", "")
        error_message = error_info.get("error_message", "")
        
        # Critical errors
        critical_patterns = [
            "SYSTEM_FAILURE",
            "MEMORY_EXHAUSTED",
            "DISK_FULL",
            "SECURITY_BREACH",
            "DATA_CORRUPTION"
        ]
        
        if any(pattern in error_code or pattern in error_message.upper() for pattern in critical_patterns):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        high_patterns = [
            "TIMEOUT",
            "NETWORK_ERROR",
            "AUTHENTICATION_FAILED",
            "PERMISSION_DENIED"
        ]
        
        if any(pattern in error_code or pattern in error_message.upper() for pattern in high_patterns):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        medium_patterns = [
            "VALIDATION_ERROR",
            "INVALID_INPUT",
            "RESOURCE_NOT_FOUND"
        ]
        
        if any(pattern in error_code or pattern in error_message.upper() for pattern in medium_patterns):
            return ErrorSeverity.MEDIUM
        
        # Check error frequency
        workflow_id = error_info.get("workflow_instance_id", "global")
        recent_errors = self._get_recent_errors(workflow_id, minutes=10)
        
        if len(recent_errors) >= 5:
            return ErrorSeverity.HIGH
        elif len(recent_errors) >= 3:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    async def _determine_recovery_action(self, error_info: Dict[str, Any], 
                                       severity: ErrorSeverity) -> RecoveryAction:
        """Determine the appropriate recovery action for an error."""
        error_code = error_info.get("error_code", "")
        
        # Critical errors - escalate immediately
        if severity == ErrorSeverity.CRITICAL:
            return RecoveryAction.ESCALATE
        
        # Check for specific error code patterns
        if error_code in ["TIMEOUT", "NETWORK_ERROR"]:
            return RecoveryAction.RETRY
        elif error_code in ["INVALID_INPUT", "VALIDATION_ERROR"]:
            return RecoveryAction.SKIP
        elif error_code in ["RESOURCE_NOT_AVAILABLE", "RATE_LIMIT_EXCEEDED"]:
            return RecoveryAction.PAUSE
        elif error_code in ["AUTHENTICATION_FAILED", "PERMISSION_DENIED"]:
            return RecoveryAction.ESCALATE
        
        # Check if we have a fallback task defined
        task_id = error_info.get("task_id")
        if task_id and await self._has_fallback_task(task_id):
            return RecoveryAction.FALLBACK
        
        # Check error frequency for this workflow
        workflow_id = error_info.get("workflow_instance_id", "global")
        recent_errors = self._get_recent_errors(workflow_id, minutes=30)
        
        if len(recent_errors) >= 3:
            return RecoveryAction.ESCALATE
        elif len(recent_errors) >= 2:
            return RecoveryAction.PAUSE
        else:
            return RecoveryAction.RETRY
    
    async def _execute_recovery(self, recovery_action: RecoveryAction, 
                              error_info: Dict[str, Any], 
                              config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute the specified recovery action."""
        if config is None:
            config = {}
        
        workflow_instance_id = error_info.get("workflow_instance_id")
        task_id = error_info.get("task_id")
        
        self.logger.info(f"Executing recovery action {recovery_action.value} for error {error_info.get('id')}")
        
        try:
            if recovery_action == RecoveryAction.RETRY:
                return await self._execute_retry_recovery(workflow_instance_id, task_id, config)
            
            elif recovery_action == RecoveryAction.SKIP:
                return await self._execute_skip_recovery(workflow_instance_id, task_id, config)
            
            elif recovery_action == RecoveryAction.FALLBACK:
                return await self._execute_fallback_recovery(workflow_instance_id, task_id, config)
            
            elif recovery_action == RecoveryAction.PAUSE:
                return await self._execute_pause_recovery(workflow_instance_id, config)
            
            elif recovery_action == RecoveryAction.ROLLBACK:
                return await self._execute_rollback_recovery(workflow_instance_id, config)
            
            elif recovery_action == RecoveryAction.ABORT:
                return await self._execute_abort_recovery(workflow_instance_id, config)
            
            else:
                return {"success": False, "message": f"Recovery action {recovery_action.value} not implemented"}
                
        except Exception as e:
            self.logger.error(f"Recovery execution failed: {e}")
            return {"success": False, "error": str(e), "next_action": "escalate"}
    
    async def _execute_retry_recovery(self, workflow_instance_id: str, task_id: str, 
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute retry recovery strategy."""
        retry_delay = config.get("retry_delay", 5)
        max_retries = config.get("max_retries", 3)
        
        # Schedule retry after delay
        await asyncio.sleep(retry_delay)
        
        try:
            # Request task retry from workflow engine
            response = await self.amp_client.send_request(
                target_agent=self.workflow_engine_id,
                capability="task-retry",
                parameters={
                    "workflow_instance_id": workflow_instance_id,
                    "task_id": task_id,
                    "retry_config": {
                        "max_retries": max_retries,
                        "delay": retry_delay
                    }
                }
            )
            
            if response.payload.get("status") == CapabilityStatus.SUCCESS.value:
                return {"success": True, "action": "retried", "next_action": None}
            else:
                return {"success": False, "error": "Retry request failed", "next_action": "escalate"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "next_action": "escalate"}
    
    async def _execute_skip_recovery(self, workflow_instance_id: str, task_id: str, 
                                   config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute skip recovery strategy."""
        try:
            # Mark task as skipped and continue workflow
            response = await self.amp_client.send_request(
                target_agent=self.workflow_engine_id,
                capability="task-skip",
                parameters={
                    "workflow_instance_id": workflow_instance_id,
                    "task_id": task_id,
                    "reason": "Error recovery: skip failed task"
                }
            )
            
            if response.payload.get("status") == CapabilityStatus.SUCCESS.value:
                return {"success": True, "action": "skipped", "next_action": None}
            else:
                return {"success": False, "error": "Skip request failed", "next_action": "escalate"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "next_action": "escalate"}
    
    async def _execute_pause_recovery(self, workflow_instance_id: str, 
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute pause recovery strategy."""
        pause_duration = config.get("pause_duration", 300)  # 5 minutes default
        
        try:
            # Pause workflow execution
            response = await self.amp_client.send_request(
                target_agent=self.workflow_engine_id,
                capability="workflow-pause",
                parameters={
                    "instance_id": workflow_instance_id,
                    "duration": pause_duration,
                    "reason": "Error recovery: temporary pause"
                }
            )
            
            if response.payload.get("status") == CapabilityStatus.SUCCESS.value:
                return {"success": True, "action": "paused", "resume_time": pause_duration}
            else:
                return {"success": False, "error": "Pause request failed", "next_action": "escalate"}
                
        except Exception as e:
            return {"success": False, "error": str(e), "next_action": "escalate"}
    
    async def _escalate_error(self, error_info: Dict[str, Any], severity: ErrorSeverity) -> bool:
        """Escalate error to appropriate handlers."""
        try:
            # Send escalation event
            await self.amp_client.send_event(
                event_type="error.escalated",
                data={
                    "error_id": error_info.get("id"),
                    "severity": severity.value,
                    "error_code": error_info.get("error_code"),
                    "error_message": error_info.get("error_message"),
                    "workflow_instance_id": error_info.get("workflow_instance_id"),
                    "task_id": error_info.get("task_id"),
                    "timestamp": error_info.get("timestamp"),
                    "escalation_reason": "Automatic escalation based on severity"
                }
            )
            
            # Send to monitor agent for alerting
            await self.amp_client.send_request(
                target_agent=self.monitor_agent_id,
                capability="alert-create",
                parameters={
                    "alert_type": "error_escalation",
                    "severity": severity.value,
                    "title": f"Error Escalation: {error_info.get('error_code')}",
                    "description": error_info.get("error_message"),
                    "metadata": error_info
                }
            )
            
            self.logger.warning(f"Escalated error {error_info.get('id')} with severity {severity.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to escalate error: {e}")
            return False
    
    async def _generate_recommendations(self, error_info: Dict[str, Any], 
                                      severity: ErrorSeverity) -> List[str]:
        """Generate recommendations for error resolution."""
        recommendations = []
        error_code = error_info.get("error_code", "")
        
        # General recommendations based on error patterns
        if "TIMEOUT" in error_code:
            recommendations.extend([
                "Consider increasing task timeout values",
                "Check network connectivity and latency",
                "Implement exponential backoff for retries"
            ])
        
        elif "MEMORY" in error_code:
            recommendations.extend([
                "Review memory usage patterns",
                "Consider processing data in smaller batches",
                "Implement memory cleanup between tasks"
            ])
        
        elif "NETWORK" in error_code:
            recommendations.extend([
                "Verify network connectivity",
                "Implement circuit breaker pattern",
                "Add network retry logic with jitter"
            ])
        
        elif "VALIDATION" in error_code:
            recommendations.extend([
                "Review input validation rules",
                "Implement input sanitization",
                "Add schema validation for task parameters"
            ])
        
        # Severity-based recommendations
        if severity == ErrorSeverity.CRITICAL:
            recommendations.extend([
                "Immediate manual intervention required",
                "Consider system rollback if possible",
                "Review system health and resources"
            ])
        
        elif severity == ErrorSeverity.HIGH:
            recommendations.extend([
                "Monitor error frequency closely",
                "Consider implementing fallback mechanisms",
                "Review system capacity"
            ])
        
        # Pattern-based recommendations
        workflow_id = error_info.get("workflow_instance_id", "global")
        recent_errors = self._get_recent_errors(workflow_id, minutes=60)
        
        if len(recent_errors) > 5:
            recommendations.append("High error frequency detected - consider workflow review")
        
        return recommendations
    
    def _get_recent_errors(self, workflow_id: str, minutes: int) -> List[Dict[str, Any]]:
        """Get recent errors for a workflow within the specified time window."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(minutes=minutes)
        
        errors = self.error_history.get(workflow_id, [])
        return [
            error for error in errors
            if datetime.fromisoformat(error["timestamp"]) > cutoff_time
        ]
    
    async def _initialize_builtin_patterns(self):
        """Initialize built-in error patterns and recovery strategies."""
        # Define common error patterns
        self.error_patterns = {
            "timeout_pattern": {
                "pattern": r".*timeout.*|.*timed out.*",
                "severity": ErrorSeverity.MEDIUM,
                "recovery_action": RecoveryAction.RETRY,
                "retry_config": {"max_retries": 3, "delay": 5}
            },
            "network_pattern": {
                "pattern": r".*network.*|.*connection.*|.*unreachable.*",
                "severity": ErrorSeverity.HIGH,
                "recovery_action": RecoveryAction.RETRY,
                "retry_config": {"max_retries": 2, "delay": 10}
            },
            "validation_pattern": {
                "pattern": r".*validation.*|.*invalid.*|.*schema.*",
                "severity": ErrorSeverity.MEDIUM,
                "recovery_action": RecoveryAction.SKIP,
                "skip_reason": "Invalid input data"
            }
        }
    
    async def _error_monitoring_task(self):
        """Background task for error monitoring and alerting."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check error thresholds for each severity level
                for severity, threshold in self.alert_thresholds.items():
                    for workflow_id in self.error_history:
                        recent_errors = self._get_recent_errors(workflow_id, threshold["window_minutes"])
                        
                        if len(recent_errors) >= threshold["count"]:
                            await self._send_threshold_alert(workflow_id, severity, len(recent_errors))
                
            except Exception as e:
                self.logger.error(f"Error monitoring task failed: {e}")
    
    async def _circuit_breaker_task(self):
        """Background task for circuit breaker management."""
        while True:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                current_time = datetime.now(timezone.utc)
                
                # Check circuit breakers for auto-reset
                for resource_id, breaker in list(self.circuit_breakers.items()):
                    if breaker["state"] == "open":
                        timeout = breaker.get("timeout", 300)  # 5 minutes default
                        last_failure = datetime.fromisoformat(breaker["last_failure"])
                        
                        if (current_time - last_failure).total_seconds() >= timeout:
                            breaker["state"] = "half_open"
                            self.logger.info(f"Circuit breaker {resource_id} moved to half-open state")
                
            except Exception as e:
                self.logger.error(f"Circuit breaker task failed: {e}")
    
    # Additional methods for circuit breaker management, fallback detection, etc.
    async def _check_circuit_breaker(self, resource_id: str) -> Dict[str, Any]:
        """Check circuit breaker state for a resource."""
        if resource_id not in self.circuit_breakers:
            self.circuit_breakers[resource_id] = {
                "state": "closed",
                "failure_count": 0,
                "last_failure": None,
                "next_attempt": None
            }
        
        return self.circuit_breakers[resource_id]
    
    async def _trip_circuit_breaker(self, resource_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """Trip circuit breaker for a resource."""
        breaker = await self._check_circuit_breaker(resource_id)
        breaker["state"] = "open"
        breaker["failure_count"] += 1
        breaker["last_failure"] = datetime.now(timezone.utc).isoformat()
        
        timeout = config.get("timeout", 300)
        breaker["next_attempt"] = (datetime.now(timezone.utc) + timedelta(seconds=timeout)).isoformat()
        
        return breaker
    
    async def _reset_circuit_breaker(self, resource_id: str) -> Dict[str, Any]:
        """Reset circuit breaker for a resource."""
        breaker = await self._check_circuit_breaker(resource_id)
        breaker["state"] = "closed"
        breaker["failure_count"] = 0
        breaker["last_failure"] = None
        breaker["next_attempt"] = None
        
        return breaker
    
    async def _get_circuit_breaker_status(self, resource_id: str) -> Dict[str, Any]:
        """Get circuit breaker status for a resource."""
        return await self._check_circuit_breaker(resource_id)
    
    # Event handlers
    async def _handle_workflow_failed(self, message: AMPMessage):
        """Handle workflow failure events."""
        event_data = message.payload
        
        error_info = {
            "code": "WORKFLOW_FAILED",
            "message": f"Workflow failed: {event_data.get('workflow_id')}",
            "details": event_data
        }
        
        context = {
            "workflow_instance_id": event_data.get("workflow_instance_id"),
            "event_type": "workflow_failure"
        }
        
        # Process the workflow failure
        await self._process_error(error_info, context)
    
    async def _handle_task_failed(self, message: AMPMessage):
        """Handle task failure events."""
        event_data = message.payload
        
        error_info = {
            "code": "TASK_FAILED",
            "message": f"Task failed: {event_data.get('task_id')}",
            "details": event_data
        }
        
        context = {
            "workflow_instance_id": event_data.get("workflow_instance_id"),
            "task_id": event_data.get("task_id"),
            "event_type": "task_failure"
        }
        
        # Process the task failure
        await self._process_error(error_info, context)
    
    async def _handle_task_retrying(self, message: AMPMessage):
        """Handle task retry events."""
        # Track retry patterns for analysis
        pass
    
    # Additional placeholder methods
    async def _has_fallback_task(self, task_id: str) -> bool:
        """Check if a fallback task is defined for the given task."""
        # Implementation would check workflow definition for fallback tasks
        return False
    
    async def _execute_fallback_recovery(self, workflow_instance_id: str, task_id: str, 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fallback recovery strategy."""
        # Implementation would execute fallback task
        return {"success": False, "message": "Fallback recovery not implemented"}
    
    async def _execute_rollback_recovery(self, workflow_instance_id: str, 
                                       config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute rollback recovery strategy."""
        # Implementation would rollback workflow state
        return {"success": False, "message": "Rollback recovery not implemented"}
    
    async def _execute_abort_recovery(self, workflow_instance_id: str, 
                                    config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute abort recovery strategy."""
        # Implementation would abort workflow
        return {"success": False, "message": "Abort recovery not implemented"}
    
    async def _analyze_error_patterns(self, time_window_hours: int, 
                                    error_types: List[str], workflow_ids: List[str]) -> Dict[str, Any]:
        """Analyze error patterns and trends."""
        # Implementation would analyze error history and return patterns
        return {"error_summary": {}, "patterns": [], "recommendations": []}
    
    async def _send_threshold_alert(self, workflow_id: str, severity: ErrorSeverity, error_count: int):
        """Send threshold-based error alert."""
        # Implementation would send alert to monitoring system
        pass
    
    # Additional handler method placeholders
    async def _handle_escalation(self, message: AMPMessage) -> AMPMessage:
        """Handle error escalation request."""
        pass
    
    async def _handle_recovery_strategy(self, message: AMPMessage) -> AMPMessage:
        """Handle recovery strategy management request."""
        pass
    
    async def _handle_error_patterns(self, message: AMPMessage) -> AMPMessage:
        """Handle error pattern management request."""
        pass