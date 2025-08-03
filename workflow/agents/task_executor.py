"""
Task Executor Agent - Executes individual workflow tasks with different capabilities.
"""

import asyncio
import json
import subprocess
import requests
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timezone
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_types import AMPMessage, MessageType, CapabilityStatus, Capability, AgentIdentity
from amp_client import AMPClient

from ..workflow_types import TaskType, WorkflowErrorCodes


class TaskExecutor:
    """
    Versatile task executor that can handle different types of workflow tasks
    including API calls, data transformations, subprocess execution, etc.
    """
    
    def __init__(self, agent_id: str = "task-executor-1", port: int = 8081):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(f"TaskExecutor.{agent_id}")
        
        # Task execution capabilities
        self.amp_client = None
        self.custom_handlers: Dict[str, Callable] = {}
        
        # Resource management
        self.max_concurrent_tasks = 10
        self.current_tasks = 0
        self.task_semaphore = asyncio.Semaphore(self.max_concurrent_tasks)
    
    async def start(self):
        """Start the task executor agent."""
        self.logger.info(f"Starting Task Executor {self.agent_id}")
        
        # Initialize AMP client
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id=self.agent_id,
                name="Task Executor",
                version="1.0.0",
                framework="AMP-Workflow",
                description="Versatile task executor for workflow operations"
            ),
            port=self.port
        )
        
        # Register capabilities
        await self._register_capabilities()
        
        # Start AMP client
        await self.amp_client.start()
        
        # Register message handlers
        self.amp_client.register_capability_handler("task-execute", self._handle_execute_task)
        self.amp_client.register_capability_handler("task-api-call", self._handle_api_call)
        self.amp_client.register_capability_handler("task-data-transform", self._handle_data_transform)
        self.amp_client.register_capability_handler("task-subprocess", self._handle_subprocess)
        self.amp_client.register_capability_handler("task-custom", self._handle_custom_task)
        self.amp_client.register_capability_handler("task-validation", self._handle_validation)
        self.amp_client.register_capability_handler("task-file-operation", self._handle_file_operation)
        
        self.logger.info("Task Executor started successfully")
    
    async def stop(self):
        """Stop the task executor agent."""
        self.logger.info("Stopping Task Executor")
        if self.amp_client:
            await self.amp_client.stop()
    
    async def _register_capabilities(self):
        """Register task execution capabilities."""
        capabilities = [
            Capability(
                id="task-execute",
                version="1.0",
                description="Execute a generic task based on task type",
                input_schema={
                    "type": "object",
                    "properties": {
                        "task_type": {"type": "string"},
                        "task_config": {"type": "object"},
                        "workflow_context": {"type": "object", "default": {}},
                        "workflow_instance_id": {"type": "string"},
                        "task_id": {"type": "string"}
                    },
                    "required": ["task_type", "task_config"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "object"},
                        "outputs": {"type": "object"},
                        "metadata": {"type": "object"}
                    }
                }
            ),
            Capability(
                id="task-api-call",
                version="1.0",
                description="Execute HTTP API calls",
                input_schema={
                    "type": "object",
                    "properties": {
                        "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]},
                        "url": {"type": "string"},
                        "headers": {"type": "object", "default": {}},
                        "data": {"type": "object", "default": {}},
                        "timeout": {"type": "number", "default": 30},
                        "retry_count": {"type": "integer", "default": 3}
                    },
                    "required": ["method", "url"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "status_code": {"type": "integer"},
                        "response_data": {"type": "object"},
                        "headers": {"type": "object"},
                        "success": {"type": "boolean"}
                    }
                }
            ),
            Capability(
                id="task-data-transform",
                version="1.0",
                description="Transform data using various operations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "input_data": {"type": "object"},
                        "transformations": {"type": "array"},
                        "output_format": {"type": "string", "default": "json"}
                    },
                    "required": ["input_data", "transformations"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "transformed_data": {"type": "object"},
                        "applied_transformations": {"type": "array"}
                    }
                }
            ),
            Capability(
                id="task-subprocess",
                version="1.0",
                description="Execute system commands and subprocesses",
                input_schema={
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "args": {"type": "array", "default": []},
                        "cwd": {"type": "string", "default": None},
                        "env": {"type": "object", "default": {}},
                        "timeout": {"type": "number", "default": 300}
                    },
                    "required": ["command"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "exit_code": {"type": "integer"},
                        "stdout": {"type": "string"},
                        "stderr": {"type": "string"},
                        "success": {"type": "boolean"}
                    }
                }
            )
        ]
        
        for capability in capabilities:
            self.amp_client.register_capability(capability)
    
    async def _handle_execute_task(self, message: AMPMessage) -> AMPMessage:
        """Handle generic task execution based on task type."""
        async with self.task_semaphore:
            try:
                self.current_tasks += 1
                
                task_type = message.payload["parameters"]["task_type"]
                task_config = message.payload["parameters"]["task_config"]
                workflow_context = message.payload["parameters"].get("workflow_context", {})
                workflow_instance_id = message.payload["parameters"].get("workflow_instance_id")
                task_id = message.payload["parameters"].get("task_id")
                
                self.logger.info(f"Executing task {task_id} of type {task_type}")
                
                # Route to appropriate handler based on task type
                if task_type == "api_call":
                    result = await self._execute_api_call(task_config, workflow_context)
                elif task_type == "data_transform":
                    result = await self._execute_data_transform(task_config, workflow_context)
                elif task_type == "subprocess":
                    result = await self._execute_subprocess(task_config, workflow_context)
                elif task_type == "validation":
                    result = await self._execute_validation(task_config, workflow_context)
                elif task_type == "file_operation":
                    result = await self._execute_file_operation(task_config, workflow_context)
                elif task_type == "custom":
                    result = await self._execute_custom_task(task_config, workflow_context)
                else:
                    raise ValueError(f"Unsupported task type: {task_type}")
                
                self.logger.info(f"Task {task_id} completed successfully")
                
                return AMPMessage.create_response(
                    agent_id=self.agent_id,
                    target_agent=message.source.agent_id,
                    correlation_id=message.headers.correlation_id,
                    result={
                        "result": result,
                        "outputs": result.get("outputs", {}),
                        "metadata": {
                            "task_id": task_id,
                            "task_type": task_type,
                            "executor_id": self.agent_id,
                            "execution_time": result.get("execution_time"),
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }
                    }
                )
                
            except Exception as e:
                self.logger.error(f"Task execution failed: {e}")
                return AMPMessage.create_error(
                    agent_id=self.agent_id,
                    target_agent=message.source.agent_id,
                    correlation_id=message.headers.correlation_id,
                    error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                    error_message=str(e),
                    details={"task_type": task_type, "task_id": task_id}
                )
            finally:
                self.current_tasks -= 1
    
    async def _handle_api_call(self, message: AMPMessage) -> AMPMessage:
        """Handle HTTP API call task."""
        try:
            params = message.payload["parameters"]
            result = await self._execute_api_call(params, {})
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )
    
    async def _execute_api_call(self, config: Dict[str, Any], 
                               context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute HTTP API call."""
        start_time = datetime.now(timezone.utc)
        
        method = config["method"].upper()
        url = self._interpolate_string(config["url"], context)
        headers = config.get("headers", {})
        data = config.get("data", {})
        timeout = config.get("timeout", 30)
        retry_count = config.get("retry_count", 3)
        
        # Interpolate headers and data with context
        headers = self._interpolate_dict(headers, context)
        data = self._interpolate_dict(data, context)
        
        last_exception = None
        for attempt in range(retry_count + 1):
            try:
                async with asyncio.timeout(timeout):
                    if method == "GET":
                        response = requests.get(url, headers=headers, params=data, timeout=timeout)
                    elif method == "POST":
                        response = requests.post(url, headers=headers, json=data, timeout=timeout)
                    elif method == "PUT":
                        response = requests.put(url, headers=headers, json=data, timeout=timeout)
                    elif method == "DELETE":
                        response = requests.delete(url, headers=headers, timeout=timeout)
                    elif method == "PATCH":
                        response = requests.patch(url, headers=headers, json=data, timeout=timeout)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    # Parse response
                    try:
                        response_data = response.json()
                    except json.JSONDecodeError:
                        response_data = {"text": response.text}
                    
                    execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
                    
                    return {
                        "status_code": response.status_code,
                        "response_data": response_data,
                        "headers": dict(response.headers),
                        "success": response.status_code < 400,
                        "execution_time": execution_time,
                        "outputs": {
                            "status_code": response.status_code,
                            "response": response_data,
                            "success": response.status_code < 400
                        }
                    }
                    
            except Exception as e:
                last_exception = e
                if attempt < retry_count:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    raise
        
        raise last_exception
    
    async def _execute_data_transform(self, config: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data transformation operations."""
        start_time = datetime.now(timezone.utc)
        
        input_data = config["input_data"]
        transformations = config["transformations"]
        output_format = config.get("output_format", "json")
        
        # Apply context interpolation to input data
        input_data = self._interpolate_dict(input_data, context)
        
        current_data = input_data
        applied_transformations = []
        
        for transform in transformations:
            transform_type = transform["type"]
            transform_config = transform.get("config", {})
            
            if transform_type == "filter":
                current_data = self._apply_filter(current_data, transform_config)
            elif transform_type == "map":
                current_data = self._apply_map(current_data, transform_config)
            elif transform_type == "reduce":
                current_data = self._apply_reduce(current_data, transform_config)
            elif transform_type == "group":
                current_data = self._apply_group(current_data, transform_config)
            elif transform_type == "sort":
                current_data = self._apply_sort(current_data, transform_config)
            elif transform_type == "aggregate":
                current_data = self._apply_aggregate(current_data, transform_config)
            elif transform_type == "join":
                join_data = transform_config.get("join_data", {})
                join_data = self._interpolate_dict(join_data, context)
                current_data = self._apply_join(current_data, join_data, transform_config)
            else:
                raise ValueError(f"Unsupported transformation type: {transform_type}")
            
            applied_transformations.append({
                "type": transform_type,
                "config": transform_config,
                "result_size": len(current_data) if isinstance(current_data, (list, dict)) else 1
            })
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            "transformed_data": current_data,
            "applied_transformations": applied_transformations,
            "execution_time": execution_time,
            "outputs": {
                "data": current_data,
                "transformation_count": len(applied_transformations)
            }
        }
    
    async def _execute_subprocess(self, config: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute subprocess command."""
        start_time = datetime.now(timezone.utc)
        
        command = self._interpolate_string(config["command"], context)
        args = [self._interpolate_string(arg, context) for arg in config.get("args", [])]
        cwd = config.get("cwd")
        env = config.get("env", {})
        timeout = config.get("timeout", 300)
        
        # Merge environment variables
        process_env = {**os.environ.copy(), **env} if env else None
        
        try:
            process = await asyncio.create_subprocess_exec(
                command, *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
                env=process_env
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), 
                timeout=timeout
            )
            
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            return {
                "exit_code": process.returncode,
                "stdout": stdout.decode('utf-8') if stdout else "",
                "stderr": stderr.decode('utf-8') if stderr else "",
                "success": process.returncode == 0,
                "execution_time": execution_time,
                "outputs": {
                    "exit_code": process.returncode,
                    "stdout": stdout.decode('utf-8') if stdout else "",
                    "success": process.returncode == 0
                }
            }
            
        except asyncio.TimeoutError:
            if process:
                process.terminate()
                await process.wait()
            raise Exception(f"Command timed out after {timeout} seconds")
    
    async def _execute_validation(self, config: Dict[str, Any], 
                                context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute data validation."""
        start_time = datetime.now(timezone.utc)
        
        data_to_validate = config["data"]
        validation_rules = config["rules"]
        
        # Apply context interpolation
        data_to_validate = self._interpolate_dict(data_to_validate, context)
        
        validation_results = []
        is_valid = True
        
        for rule in validation_rules:
            rule_type = rule["type"]
            rule_config = rule.get("config", {})
            field = rule.get("field")
            
            try:
                if rule_type == "required":
                    valid = field in data_to_validate and data_to_validate[field] is not None
                elif rule_type == "type":
                    expected_type = rule_config.get("type")
                    valid = isinstance(data_to_validate.get(field), eval(expected_type))
                elif rule_type == "range":
                    value = data_to_validate.get(field)
                    min_val = rule_config.get("min")
                    max_val = rule_config.get("max")
                    valid = (min_val is None or value >= min_val) and (max_val is None or value <= max_val)
                elif rule_type == "regex":
                    import re
                    pattern = rule_config.get("pattern")
                    value = str(data_to_validate.get(field, ""))
                    valid = re.match(pattern, value) is not None
                elif rule_type == "custom":
                    expression = rule_config.get("expression")
                    valid = eval(expression, {"data": data_to_validate, "field_value": data_to_validate.get(field)})
                else:
                    valid = False
                    
                validation_results.append({
                    "rule": rule,
                    "field": field,
                    "valid": valid,
                    "message": rule.get("message", f"Validation failed for {field}")
                })
                
                if not valid:
                    is_valid = False
                    
            except Exception as e:
                validation_results.append({
                    "rule": rule,
                    "field": field,
                    "valid": False,
                    "error": str(e)
                })
                is_valid = False
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            "is_valid": is_valid,
            "validation_results": validation_results,
            "execution_time": execution_time,
            "outputs": {
                "valid": is_valid,
                "failed_rules": [r for r in validation_results if not r["valid"]]
            }
        }
    
    async def _execute_file_operation(self, config: Dict[str, Any], 
                                    context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute file operations."""
        start_time = datetime.now(timezone.utc)
        
        operation = config["operation"]
        file_path = self._interpolate_string(config["file_path"], context)
        
        if operation == "read":
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            result = {"content": content, "size": len(content)}
            
        elif operation == "write":
            content = self._interpolate_string(config["content"], context)
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            result = {"bytes_written": len(content)}
            
        elif operation == "append":
            content = self._interpolate_string(config["content"], context)
            with open(file_path, 'a', encoding='utf-8') as f:
                f.write(content)
            result = {"bytes_appended": len(content)}
            
        elif operation == "delete":
            import os
            os.remove(file_path)
            result = {"deleted": True}
            
        elif operation == "copy":
            import shutil
            dest_path = self._interpolate_string(config["destination"], context)
            shutil.copy2(file_path, dest_path)
            result = {"copied": True, "destination": dest_path}
            
        else:
            raise ValueError(f"Unsupported file operation: {operation}")
        
        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        
        return {
            **result,
            "operation": operation,
            "file_path": file_path,
            "execution_time": execution_time,
            "outputs": result
        }
    
    async def _execute_custom_task(self, config: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute custom task using registered handlers."""
        task_name = config["task_name"]
        task_params = config.get("parameters", {})
        
        if task_name not in self.custom_handlers:
            raise ValueError(f"Custom task handler not found: {task_name}")
        
        handler = self.custom_handlers[task_name]
        
        # Apply context interpolation to parameters
        task_params = self._interpolate_dict(task_params, context)
        
        # Execute custom handler
        result = await handler(task_params, context)
        
        return {
            "result": result,
            "task_name": task_name,
            "outputs": result if isinstance(result, dict) else {"result": result}
        }
    
    def register_custom_handler(self, task_name: str, handler: Callable):
        """Register a custom task handler."""
        self.custom_handlers[task_name] = handler
        self.logger.info(f"Registered custom task handler: {task_name}")
    
    def _interpolate_string(self, template: str, context: Dict[str, Any]) -> str:
        """Interpolate template string with context variables."""
        if not isinstance(template, str):
            return template
        
        # Simple template interpolation using {variable} syntax
        import re
        
        def replace_var(match):
            var_name = match.group(1)
            return str(context.get(var_name, match.group(0)))
        
        return re.sub(r'\{([^}]+)\}', replace_var, template)
    
    def _interpolate_dict(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively interpolate dictionary with context variables."""
        if isinstance(data, dict):
            return {k: self._interpolate_dict(v, context) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._interpolate_dict(item, context) for item in data]
        elif isinstance(data, str):
            return self._interpolate_string(data, context)
        else:
            return data
    
    # Data transformation helpers
    def _apply_filter(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply filter transformation."""
        condition = config["condition"]
        return [item for item in data if eval(condition, {"item": item})]
    
    def _apply_map(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply map transformation."""
        expression = config["expression"]
        return [eval(expression, {"item": item}) for item in data]
    
    def _apply_reduce(self, data: List[Dict], config: Dict) -> Any:
        """Apply reduce transformation."""
        from functools import reduce
        expression = config["expression"]
        initial = config.get("initial", 0)
        return reduce(lambda acc, item: eval(expression, {"acc": acc, "item": item}), data, initial)
    
    def _apply_group(self, data: List[Dict], config: Dict) -> Dict[str, List]:
        """Apply group transformation."""
        group_by = config["group_by"]
        groups = {}
        for item in data:
            key = item.get(group_by)
            if key not in groups:
                groups[key] = []
            groups[key].append(item)
        return groups
    
    def _apply_sort(self, data: List[Dict], config: Dict) -> List[Dict]:
        """Apply sort transformation."""
        key_field = config["key"]
        reverse = config.get("reverse", False)
        return sorted(data, key=lambda x: x.get(key_field), reverse=reverse)
    
    def _apply_aggregate(self, data: List[Dict], config: Dict) -> Dict[str, Any]:
        """Apply aggregation transformation."""
        operations = config["operations"]
        result = {}
        
        for op_config in operations:
            op_type = op_config["type"]
            field = op_config["field"]
            result_key = op_config.get("result_key", f"{op_type}_{field}")
            
            values = [item.get(field) for item in data if item.get(field) is not None]
            
            if op_type == "sum":
                result[result_key] = sum(values)
            elif op_type == "avg":
                result[result_key] = sum(values) / len(values) if values else 0
            elif op_type == "min":
                result[result_key] = min(values) if values else None
            elif op_type == "max":
                result[result_key] = max(values) if values else None
            elif op_type == "count":
                result[result_key] = len(values)
        
        return result
    
    def _apply_join(self, data: List[Dict], join_data: List[Dict], config: Dict) -> List[Dict]:
        """Apply join transformation."""
        join_type = config.get("type", "inner")
        left_key = config["left_key"]
        right_key = config["right_key"]
        
        # Create lookup dictionary for join data
        join_lookup = {item[right_key]: item for item in join_data}
        
        result = []
        for item in data:
            join_key = item.get(left_key)
            join_item = join_lookup.get(join_key)
            
            if join_item:
                # Merge items
                merged = {**item, **join_item}
                result.append(merged)
            elif join_type == "left":
                result.append(item)
        
        return result
    
    # Additional handler methods would continue here...
    async def _handle_data_transform(self, message: AMPMessage) -> AMPMessage:
        """Handle data transform capability request."""
        try:
            params = message.payload["parameters"]
            result = await self._execute_data_transform(params, {})
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_subprocess(self, message: AMPMessage) -> AMPMessage:
        """Handle subprocess capability request."""
        try:
            params = message.payload["parameters"]
            result = await self._execute_subprocess(params, {})
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_custom_task(self, message: AMPMessage) -> AMPMessage:
        """Handle custom task capability request."""
        try:
            params = message.payload["parameters"]
            result = await self._execute_custom_task(params, {})
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_validation(self, message: AMPMessage) -> AMPMessage:
        """Handle validation capability request."""
        try:
            params = message.payload["parameters"]
            result = await self._execute_validation(params, {})
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_file_operation(self, message: AMPMessage) -> AMPMessage:
        """Handle file operation capability request."""
        try:
            params = message.payload["parameters"]
            result = await self._execute_file_operation(params, {})
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result=result
            )
            
        except Exception as e:
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.TASK_EXECUTION_FAILED,
                error_message=str(e)
            )