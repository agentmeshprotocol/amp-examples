"""
Workflow Engine Agent - Core orchestration engine for AMP workflows.
"""

import asyncio
import json
import yaml
from typing import Dict, Any, List, Optional, Set
from datetime import datetime, timezone
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_types import AMPMessage, MessageType, CapabilityStatus, Capability, AgentIdentity
from amp_client import AMPClient
from amp_utils import generate_correlation_id

from ..workflow_types import (
    WorkflowDefinition, WorkflowInstance, WorkflowStatus, TaskDefinition,
    TaskExecution, TaskStatus, WorkflowEvent, WorkflowEventTypes,
    WorkflowErrorCodes, TaskType
)


class WorkflowEngine:
    """
    Central workflow orchestration engine that manages workflow definitions,
    executes workflows, and coordinates with other agents.
    """
    
    def __init__(self, agent_id: str = "workflow-engine", port: int = 8080):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(f"WorkflowEngine.{agent_id}")
        
        # Workflow storage
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        self.workflow_instances: Dict[str, WorkflowInstance] = {}
        self.active_workflows: Set[str] = set()
        
        # Agent management
        self.amp_client = None
        self.task_executors: Set[str] = set()
        self.state_manager_id = "state-manager"
        self.condition_evaluator_id = "condition-evaluator"
        self.error_handler_id = "error-handler"
        self.monitor_agent_id = "monitor-agent"
        
        # Event handlers
        self.event_handlers = {}
        
    async def start(self):
        """Start the workflow engine."""
        self.logger.info(f"Starting Workflow Engine {self.agent_id}")
        
        # Initialize AMP client
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id=self.agent_id,
                name="Workflow Engine",
                version="1.0.0",
                framework="AMP-Workflow",
                description="Central workflow orchestration engine"
            ),
            port=self.port
        )
        
        # Register capabilities
        await self._register_capabilities()
        
        # Start AMP client
        await self.amp_client.start()
        
        # Register message handlers
        self.amp_client.register_capability_handler("workflow-create", self._handle_create_workflow)
        self.amp_client.register_capability_handler("workflow-start", self._handle_start_workflow)
        self.amp_client.register_capability_handler("workflow-stop", self._handle_stop_workflow)
        self.amp_client.register_capability_handler("workflow-pause", self._handle_pause_workflow)
        self.amp_client.register_capability_handler("workflow-resume", self._handle_resume_workflow)
        self.amp_client.register_capability_handler("workflow-status", self._handle_get_workflow_status)
        self.amp_client.register_capability_handler("workflow-list", self._handle_list_workflows)
        self.amp_client.register_capability_handler("workflow-load", self._handle_load_workflow_definition)
        
        # Register event handlers
        self.amp_client.register_event_handler("task.completed", self._handle_task_completed)
        self.amp_client.register_event_handler("task.failed", self._handle_task_failed)
        self.amp_client.register_event_handler("state.updated", self._handle_state_updated)
        
        self.logger.info("Workflow Engine started successfully")
    
    async def stop(self):
        """Stop the workflow engine."""
        self.logger.info("Stopping Workflow Engine")
        
        # Cancel all active workflows
        for workflow_id in list(self.active_workflows):
            await self._cancel_workflow(workflow_id)
        
        if self.amp_client:
            await self.amp_client.stop()
    
    async def _register_capabilities(self):
        """Register workflow engine capabilities."""
        capabilities = [
            Capability(
                id="workflow-create",
                version="1.0",
                description="Create a new workflow definition",
                input_schema={
                    "type": "object",
                    "properties": {
                        "definition": {"type": "object"},
                        "overwrite": {"type": "boolean", "default": False}
                    },
                    "required": ["definition"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            ),
            Capability(
                id="workflow-start",
                version="1.0",
                description="Start a workflow execution",
                input_schema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "inputs": {"type": "object", "default": {}},
                        "instance_id": {"type": "string", "default": None}
                    },
                    "required": ["workflow_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "instance_id": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            ),
            Capability(
                id="workflow-status",
                version="1.0",
                description="Get workflow instance status",
                input_schema={
                    "type": "object",
                    "properties": {
                        "instance_id": {"type": "string"}
                    },
                    "required": ["instance_id"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "instance": {"type": "object"},
                        "tasks": {"type": "object"}
                    }
                }
            ),
            Capability(
                id="workflow-load",
                version="1.0",
                description="Load workflow definition from file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string"},
                        "format": {"type": "string", "enum": ["yaml", "json"], "default": "yaml"}
                    },
                    "required": ["file_path"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "workflow_id": {"type": "string"},
                        "status": {"type": "string"}
                    }
                }
            )
        ]
        
        for capability in capabilities:
            self.amp_client.register_capability(capability)
    
    async def _handle_create_workflow(self, message: AMPMessage) -> AMPMessage:
        """Handle workflow creation request."""
        try:
            definition_data = message.payload["parameters"]["definition"]
            overwrite = message.payload["parameters"].get("overwrite", False)
            
            # Parse workflow definition
            workflow_def = self._parse_workflow_definition(definition_data)
            
            # Validate workflow
            validation_errors = await self._validate_workflow(workflow_def)
            if validation_errors:
                raise ValueError(f"Workflow validation failed: {validation_errors}")
            
            # Check if workflow exists
            if workflow_def.id in self.workflow_definitions and not overwrite:
                raise ValueError(f"Workflow {workflow_def.id} already exists")
            
            # Store workflow definition
            self.workflow_definitions[workflow_def.id] = workflow_def
            
            self.logger.info(f"Created workflow: {workflow_def.id}")
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "workflow_id": workflow_def.id,
                    "status": "created"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.INVALID_WORKFLOW_DEFINITION,
                error_message=str(e)
            )
    
    async def _handle_start_workflow(self, message: AMPMessage) -> AMPMessage:
        """Handle workflow start request."""
        try:
            workflow_id = message.payload["parameters"]["workflow_id"]
            inputs = message.payload["parameters"].get("inputs", {})
            instance_id = message.payload["parameters"].get("instance_id")
            
            if workflow_id not in self.workflow_definitions:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Create workflow instance
            instance = await self._create_workflow_instance(workflow_id, inputs, instance_id)
            
            # Start workflow execution
            await self._start_workflow_execution(instance)
            
            self.logger.info(f"Started workflow instance: {instance.id}")
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "instance_id": instance.id,
                    "status": instance.status.value
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to start workflow: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_get_workflow_status(self, message: AMPMessage) -> AMPMessage:
        """Handle workflow status request."""
        try:
            instance_id = message.payload["parameters"]["instance_id"]
            
            if instance_id not in self.workflow_instances:
                raise ValueError(f"Workflow instance {instance_id} not found")
            
            instance = self.workflow_instances[instance_id]
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "instance": {
                        "id": instance.id,
                        "workflow_id": instance.workflow_id,
                        "status": instance.status.value,
                        "created_at": instance.created_at.isoformat(),
                        "started_at": instance.started_at.isoformat() if instance.started_at else None,
                        "completed_at": instance.completed_at.isoformat() if instance.completed_at else None,
                        "inputs": instance.inputs,
                        "outputs": instance.outputs,
                        "error_message": instance.error_message
                    },
                    "tasks": {
                        task_id: {
                            "status": task_exec.status.value,
                            "started_at": task_exec.started_at.isoformat() if task_exec.started_at else None,
                            "completed_at": task_exec.completed_at.isoformat() if task_exec.completed_at else None,
                            "attempt_count": task_exec.attempt_count,
                            "last_error": task_exec.last_error,
                            "outputs": task_exec.outputs
                        }
                        for task_id, task_exec in instance.task_executions.items()
                    }
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to get workflow status: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.WORKFLOW_NOT_FOUND,
                error_message=str(e)
            )
    
    async def _handle_load_workflow_definition(self, message: AMPMessage) -> AMPMessage:
        """Handle workflow definition loading from file."""
        try:
            file_path = message.payload["parameters"]["file_path"]
            format_type = message.payload["parameters"].get("format", "yaml")
            
            # Load workflow definition from file
            workflow_def = await self._load_workflow_from_file(file_path, format_type)
            
            # Store workflow definition
            self.workflow_definitions[workflow_def.id] = workflow_def
            
            self.logger.info(f"Loaded workflow from file: {workflow_def.id}")
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "workflow_id": workflow_def.id,
                    "status": "loaded"
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to load workflow: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.INVALID_WORKFLOW_DEFINITION,
                error_message=str(e)
            )
    
    async def _create_workflow_instance(self, workflow_id: str, inputs: Dict[str, Any], 
                                      instance_id: Optional[str] = None) -> WorkflowInstance:
        """Create a new workflow instance."""
        import uuid
        
        if not instance_id:
            instance_id = str(uuid.uuid4())
        
        workflow_def = self.workflow_definitions[workflow_id]
        
        # Create task executions
        task_executions = {}
        for task in workflow_def.tasks:
            task_executions[task.id] = TaskExecution(
                task_id=task.id,
                workflow_instance_id=instance_id,
                status=TaskStatus.PENDING
            )
        
        instance = WorkflowInstance(
            id=instance_id,
            workflow_id=workflow_id,
            status=WorkflowStatus.PENDING,
            inputs=inputs,
            task_executions=task_executions
        )
        
        self.workflow_instances[instance_id] = instance
        return instance
    
    async def _start_workflow_execution(self, instance: WorkflowInstance):
        """Start executing a workflow instance."""
        self.logger.info(f"Starting workflow execution: {instance.id}")
        
        instance.status = WorkflowStatus.RUNNING
        instance.started_at = datetime.now(timezone.utc)
        self.active_workflows.add(instance.id)
        
        # Emit workflow started event
        await self._emit_workflow_event(
            instance.id,
            WorkflowEventTypes.WORKFLOW_STARTED,
            workflow_id=instance.workflow_id
        )
        
        # Update workflow state
        await self._update_workflow_state(instance.id, instance.inputs)
        
        # Start task execution loop
        asyncio.create_task(self._execute_workflow_loop(instance.id))
    
    async def _execute_workflow_loop(self, instance_id: str):
        """Main workflow execution loop."""
        try:
            while instance_id in self.active_workflows:
                instance = self.workflow_instances[instance_id]
                workflow_def = self.workflow_definitions[instance.workflow_id]
                
                # Find ready tasks
                ready_tasks = await self._find_ready_tasks(instance, workflow_def)
                
                if not ready_tasks:
                    # Check if workflow is complete
                    if await self._is_workflow_complete(instance):
                        await self._complete_workflow(instance_id)
                        break
                    else:
                        # Wait for task completions
                        await asyncio.sleep(1)
                        continue
                
                # Execute ready tasks
                for task_def in ready_tasks:
                    await self._execute_task(instance, task_def)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {e}")
            await self._fail_workflow(instance_id, str(e))
    
    async def _find_ready_tasks(self, instance: WorkflowInstance, 
                               workflow_def: WorkflowDefinition) -> List[TaskDefinition]:
        """Find tasks that are ready to execute."""
        ready_tasks = []
        
        for task_def in workflow_def.tasks:
            task_exec = instance.task_executions[task_def.id]
            
            # Skip if task is not pending
            if task_exec.status != TaskStatus.PENDING:
                continue
            
            # Check dependencies
            if await self._are_dependencies_satisfied(task_def, instance):
                # Check conditions
                if await self._evaluate_task_condition(task_def, instance):
                    ready_tasks.append(task_def)
        
        return ready_tasks
    
    async def _execute_task(self, instance: WorkflowInstance, task_def: TaskDefinition):
        """Execute a single task."""
        task_exec = instance.task_executions[task_def.id]
        task_exec.status = TaskStatus.RUNNING
        task_exec.started_at = datetime.now(timezone.utc)
        task_exec.attempt_count += 1
        
        self.logger.info(f"Executing task {task_def.id} (attempt {task_exec.attempt_count})")
        
        # Emit task started event
        await self._emit_workflow_event(
            instance.id,
            WorkflowEventTypes.TASK_STARTED,
            task_id=task_def.id,
            task_name=task_def.name
        )
        
        try:
            # Delegate to task executor
            if task_def.agent_id and task_def.capability:
                await self._delegate_to_agent(instance, task_def)
            else:
                await self._execute_built_in_task(instance, task_def)
                
        except Exception as e:
            self.logger.error(f"Task {task_def.id} failed: {e}")
            await self._handle_task_failure(instance, task_def, str(e))
    
    async def _delegate_to_agent(self, instance: WorkflowInstance, task_def: TaskDefinition):
        """Delegate task execution to a specific agent."""
        # Get current workflow state for task context
        state_response = await self.amp_client.send_request(
            target_agent=self.state_manager_id,
            capability="state-get",
            parameters={"workflow_instance_id": instance.id}
        )
        
        current_state = state_response.payload.get("result", {}).get("state", {})
        
        # Prepare task parameters with context
        task_parameters = {
            **task_def.parameters,
            "workflow_context": current_state,
            "workflow_instance_id": instance.id,
            "task_id": task_def.id
        }
        
        # Send task execution request
        response = await self.amp_client.send_request(
            target_agent=task_def.agent_id,
            capability=task_def.capability,
            parameters=task_parameters,
            timeout_ms=task_def.timeout_seconds * 1000
        )
        
        if response.payload.get("status") == CapabilityStatus.SUCCESS.value:
            await self._handle_task_success(instance, task_def, response.payload.get("result"))
        else:
            raise Exception(f"Task execution failed: {response.payload.get('error', 'Unknown error')}")
    
    async def _handle_task_success(self, instance: WorkflowInstance, 
                                  task_def: TaskDefinition, result: Any):
        """Handle successful task completion."""
        task_exec = instance.task_executions[task_def.id]
        task_exec.status = TaskStatus.COMPLETED
        task_exec.completed_at = datetime.now(timezone.utc)
        task_exec.outputs = result if isinstance(result, dict) else {"result": result}
        
        self.logger.info(f"Task {task_def.id} completed successfully")
        
        # Update workflow state with task outputs
        if task_def.outputs:
            state_updates = {}
            for output_key in task_def.outputs:
                if output_key in task_exec.outputs:
                    state_updates[output_key] = task_exec.outputs[output_key]
            
            if state_updates:
                await self._update_workflow_state(instance.id, state_updates)
        
        # Emit task completed event
        await self._emit_workflow_event(
            instance.id,
            WorkflowEventTypes.TASK_COMPLETED,
            task_id=task_def.id,
            outputs=task_exec.outputs
        )
    
    async def _handle_task_failure(self, instance: WorkflowInstance, 
                                  task_def: TaskDefinition, error_message: str):
        """Handle task failure and retry logic."""
        task_exec = instance.task_executions[task_def.id]
        task_exec.last_error = error_message
        
        # Check if we should retry
        if (task_exec.attempt_count < task_def.retry_config.max_attempts and
            self._should_retry_error(error_message, task_def.retry_config)):
            
            task_exec.status = TaskStatus.RETRYING
            self.logger.info(f"Retrying task {task_def.id} (attempt {task_exec.attempt_count + 1})")
            
            # Calculate retry delay
            delay = self._calculate_retry_delay(task_exec.attempt_count, task_def.retry_config)
            
            # Schedule retry
            asyncio.create_task(self._retry_task_after_delay(instance, task_def, delay))
        else:
            # Task failed permanently
            task_exec.status = TaskStatus.FAILED
            task_exec.completed_at = datetime.now(timezone.utc)
            
            self.logger.error(f"Task {task_def.id} failed permanently: {error_message}")
            
            # Emit task failed event
            await self._emit_workflow_event(
                instance.id,
                WorkflowEventTypes.TASK_FAILED,
                task_id=task_def.id,
                error=error_message
            )
            
            # Handle workflow failure if this is a critical task
            await self._handle_workflow_task_failure(instance, task_def)
    
    async def _update_workflow_state(self, instance_id: str, state_updates: Dict[str, Any]):
        """Update workflow state through state manager."""
        try:
            await self.amp_client.send_request(
                target_agent=self.state_manager_id,
                capability="state-update",
                parameters={
                    "workflow_instance_id": instance_id,
                    "updates": state_updates
                }
            )
        except Exception as e:
            self.logger.error(f"Failed to update workflow state: {e}")
    
    async def _emit_workflow_event(self, instance_id: str, event_type: str, 
                                  task_id: Optional[str] = None, **data):
        """Emit workflow event to monitor agent."""
        event = WorkflowEvent.create(
            workflow_instance_id=instance_id,
            event_type=event_type,
            task_id=task_id,
            **data
        )
        
        # Send to monitor agent
        await self.amp_client.send_event(
            event_type="workflow.event",
            data={
                "event": {
                    "id": event.id,
                    "workflow_instance_id": event.workflow_instance_id,
                    "task_id": event.task_id,
                    "event_type": event.event_type,
                    "timestamp": event.timestamp.isoformat(),
                    "data": event.data
                }
            }
        )
    
    def _parse_workflow_definition(self, definition_data: Dict[str, Any]) -> WorkflowDefinition:
        """Parse workflow definition from dictionary."""
        # Convert task definitions
        tasks = []
        for task_data in definition_data.get("tasks", []):
            task = TaskDefinition(
                id=task_data["id"],
                name=task_data["name"],
                type=TaskType(task_data.get("type", "sequential")),
                agent_id=task_data.get("agent_id"),
                capability=task_data.get("capability"),
                parameters=task_data.get("parameters", {}),
                depends_on=task_data.get("depends_on", []),
                timeout_seconds=task_data.get("timeout_seconds", 300),
                outputs=task_data.get("outputs", []),
                metadata=task_data.get("metadata", {})
            )
            tasks.append(task)
        
        return WorkflowDefinition(
            id=definition_data["id"],
            name=definition_data["name"],
            version=definition_data.get("version", "1.0"),
            description=definition_data.get("description", ""),
            tasks=tasks,
            global_timeout_seconds=definition_data.get("global_timeout_seconds", 3600),
            input_schema=definition_data.get("input_schema", {}),
            output_schema=definition_data.get("output_schema", {}),
            metadata=definition_data.get("metadata", {}),
            tags=definition_data.get("tags", [])
        )
    
    async def _validate_workflow(self, workflow_def: WorkflowDefinition) -> List[str]:
        """Validate workflow definition."""
        errors = []
        
        # Check for circular dependencies
        if self._has_circular_dependencies(workflow_def):
            errors.append("Circular dependencies detected")
        
        # Validate task references
        task_ids = {task.id for task in workflow_def.tasks}
        for task in workflow_def.tasks:
            for dep in task.depends_on:
                if dep not in task_ids:
                    errors.append(f"Task {task.id} depends on non-existent task {dep}")
        
        return errors
    
    def _has_circular_dependencies(self, workflow_def: WorkflowDefinition) -> bool:
        """Check for circular dependencies in workflow."""
        # Simple DFS-based cycle detection
        visited = set()
        rec_stack = set()
        
        def has_cycle(task_id: str) -> bool:
            if task_id in rec_stack:
                return True
            if task_id in visited:
                return False
            
            visited.add(task_id)
            rec_stack.add(task_id)
            
            # Find task definition
            task_def = None
            for task in workflow_def.tasks:
                if task.id == task_id:
                    task_def = task
                    break
            
            if task_def:
                for dep in task_def.depends_on:
                    if has_cycle(dep):
                        return True
            
            rec_stack.remove(task_id)
            return False
        
        for task in workflow_def.tasks:
            if task.id not in visited:
                if has_cycle(task.id):
                    return True
        
        return False
    
    # Additional methods would continue here...
    # (For brevity, I'm showing the core structure and key methods)
    
    async def _handle_task_completed(self, message: AMPMessage):
        """Handle task completion event."""
        # Implementation for handling external task completion events
        pass
    
    async def _handle_task_failed(self, message: AMPMessage):
        """Handle task failure event."""
        # Implementation for handling external task failure events
        pass
    
    async def _handle_state_updated(self, message: AMPMessage):
        """Handle state update event."""
        # Implementation for handling state update events
        pass