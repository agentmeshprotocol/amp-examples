"""
Condition Evaluator Agent - Handles conditional logic and workflow branching.
"""

import asyncio
import ast
import operator
import re
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime, timezone
import logging
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_types import AMPMessage, MessageType, CapabilityStatus, Capability, AgentIdentity
from amp_client import AMPClient

from ..workflow_types import WorkflowEventTypes, WorkflowErrorCodes


class ConditionEvaluator:
    """
    Evaluates conditional expressions for workflow branching and decision making.
    Supports various condition types including data comparisons, state checks, and custom expressions.
    """
    
    def __init__(self, agent_id: str = "condition-evaluator", port: int = 8083):
        self.agent_id = agent_id
        self.port = port
        self.logger = logging.getLogger(f"ConditionEvaluator.{agent_id}")
        
        # Condition evaluation
        self.amp_client = None
        self.custom_functions: Dict[str, Callable] = {}
        self.evaluation_cache: Dict[str, tuple] = {}  # (result, timestamp)
        self.cache_ttl = 60  # 1 minute cache for expensive evaluations
        
        # Safe evaluation environment
        self.safe_names = {
            'True': True,
            'False': False,
            'None': None,
            'abs': abs,
            'all': all,
            'any': any,
            'bool': bool,
            'dict': dict,
            'float': float,
            'int': int,
            'len': len,
            'list': list,
            'max': max,
            'min': min,
            'round': round,
            'str': str,
            'sum': sum,
            'tuple': tuple,
            'type': type,
            'isinstance': isinstance,
            'hasattr': hasattr,
            'getattr': getattr,
            're': re,
            'datetime': datetime,
        }
        
        # Supported operators
        self.operators = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.FloorDiv: operator.floordiv,
            ast.Mod: operator.mod,
            ast.Pow: operator.pow,
            ast.LShift: operator.lshift,
            ast.RShift: operator.rshift,
            ast.BitOr: operator.or_,
            ast.BitXor: operator.xor,
            ast.BitAnd: operator.and_,
            ast.Eq: operator.eq,
            ast.NotEq: operator.ne,
            ast.Lt: operator.lt,
            ast.LtE: operator.le,
            ast.Gt: operator.gt,
            ast.GtE: operator.ge,
            ast.Is: operator.is_,
            ast.IsNot: operator.is_not,
            ast.In: lambda x, y: x in y,
            ast.NotIn: lambda x, y: x not in y,
        }
    
    async def start(self):
        """Start the condition evaluator agent."""
        self.logger.info(f"Starting Condition Evaluator {self.agent_id}")
        
        # Initialize AMP client
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id=self.agent_id,
                name="Condition Evaluator",
                version="1.0.0",
                framework="AMP-Workflow",
                description="Conditional logic and workflow branching evaluator"
            ),
            port=self.port
        )
        
        # Register capabilities
        await self._register_capabilities()
        
        # Start AMP client
        await self.amp_client.start()
        
        # Register message handlers
        self.amp_client.register_capability_handler("condition-evaluate", self._handle_evaluate_condition)
        self.amp_client.register_capability_handler("condition-batch-evaluate", self._handle_batch_evaluate)
        self.amp_client.register_capability_handler("condition-validate", self._handle_validate_condition)
        self.amp_client.register_capability_handler("expression-parse", self._handle_parse_expression)
        self.amp_client.register_capability_handler("decision-tree", self._handle_decision_tree)
        self.amp_client.register_capability_handler("rule-engine", self._handle_rule_engine)
        
        # Register built-in custom functions
        self._register_builtin_functions()
        
        self.logger.info("Condition Evaluator started successfully")
    
    async def stop(self):
        """Stop the condition evaluator agent."""
        self.logger.info("Stopping Condition Evaluator")
        if self.amp_client:
            await self.amp_client.stop()
    
    async def _register_capabilities(self):
        """Register condition evaluation capabilities."""
        capabilities = [
            Capability(
                id="condition-evaluate",
                version="1.0",
                description="Evaluate a conditional expression",
                input_schema={
                    "type": "object",
                    "properties": {
                        "expression": {"type": "string"},
                        "context": {"type": "object", "default": {}},
                        "condition_type": {"type": "string", "enum": ["python", "simple", "regex", "custom"], "default": "python"},
                        "cache_key": {"type": "string", "default": None},
                        "use_cache": {"type": "boolean", "default": True}
                    },
                    "required": ["expression"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "boolean"},
                        "details": {"type": "object"},
                        "evaluation_time_ms": {"type": "number"},
                        "from_cache": {"type": "boolean"}
                    }
                }
            ),
            Capability(
                id="condition-batch-evaluate",
                version="1.0",
                description="Evaluate multiple conditions in batch",
                input_schema={
                    "type": "object",
                    "properties": {
                        "conditions": {"type": "array", "items": {"type": "object"}},
                        "context": {"type": "object", "default": {}},
                        "operator": {"type": "string", "enum": ["and", "or", "all", "any"], "default": "and"}
                    },
                    "required": ["conditions"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "overall_result": {"type": "boolean"},
                        "individual_results": {"type": "array"},
                        "evaluation_time_ms": {"type": "number"}
                    }
                }
            ),
            Capability(
                id="decision-tree",
                version="1.0",
                description="Evaluate a decision tree structure",
                input_schema={
                    "type": "object",
                    "properties": {
                        "tree": {"type": "object"},
                        "context": {"type": "object", "default": {}}
                    },
                    "required": ["tree"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "result": {"type": "any"},
                        "path": {"type": "array"},
                        "evaluations": {"type": "array"}
                    }
                }
            ),
            Capability(
                id="rule-engine",
                version="1.0",
                description="Execute a rule-based decision engine",
                input_schema={
                    "type": "object",
                    "properties": {
                        "rules": {"type": "array", "items": {"type": "object"}},
                        "context": {"type": "object", "default": {}},
                        "execution_mode": {"type": "string", "enum": ["first_match", "all_matches", "highest_priority"], "default": "first_match"}
                    },
                    "required": ["rules"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "matched_rules": {"type": "array"},
                        "actions": {"type": "array"},
                        "execution_time_ms": {"type": "number"}
                    }
                }
            )
        ]
        
        for capability in capabilities:
            self.amp_client.register_capability(capability)
    
    async def _handle_evaluate_condition(self, message: AMPMessage) -> AMPMessage:
        """Handle condition evaluation request."""
        try:
            start_time = datetime.now()
            
            expression = message.payload["parameters"]["expression"]
            context = message.payload["parameters"].get("context", {})
            condition_type = message.payload["parameters"].get("condition_type", "python")
            cache_key = message.payload["parameters"].get("cache_key")
            use_cache = message.payload["parameters"].get("use_cache", True)
            
            # Check cache if enabled
            from_cache = False
            if use_cache and cache_key:
                cached_result = self._get_from_cache(cache_key)
                if cached_result is not None:
                    result, details = cached_result
                    from_cache = True
                else:
                    result, details = await self._evaluate_condition(expression, context, condition_type)
                    self._store_in_cache(cache_key, (result, details))
            else:
                result, details = await self._evaluate_condition(expression, context, condition_type)
            
            evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "result": result,
                    "details": details,
                    "evaluation_time_ms": evaluation_time,
                    "from_cache": from_cache
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate condition: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.CONDITION_EVALUATION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_batch_evaluate(self, message: AMPMessage) -> AMPMessage:
        """Handle batch condition evaluation request."""
        try:
            start_time = datetime.now()
            
            conditions = message.payload["parameters"]["conditions"]
            context = message.payload["parameters"].get("context", {})
            operator_type = message.payload["parameters"].get("operator", "and")
            
            individual_results = []
            
            # Evaluate each condition
            for condition in conditions:
                expression = condition["expression"]
                condition_type = condition.get("type", "python")
                condition_context = {**context, **condition.get("context", {})}
                
                try:
                    result, details = await self._evaluate_condition(expression, condition_context, condition_type)
                    individual_results.append({
                        "expression": expression,
                        "result": result,
                        "details": details,
                        "success": True
                    })
                except Exception as e:
                    individual_results.append({
                        "expression": expression,
                        "result": False,
                        "error": str(e),
                        "success": False
                    })
            
            # Calculate overall result based on operator
            results = [r["result"] for r in individual_results if r["success"]]
            
            if operator_type == "and":
                overall_result = all(results)
            elif operator_type == "or":
                overall_result = any(results)
            elif operator_type == "all":
                overall_result = all(results) and len(results) == len(individual_results)
            elif operator_type == "any":
                overall_result = any(results)
            else:
                overall_result = all(results)  # Default to 'and'
            
            evaluation_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "overall_result": overall_result,
                    "individual_results": individual_results,
                    "evaluation_time_ms": evaluation_time
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to batch evaluate conditions: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.CONDITION_EVALUATION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_decision_tree(self, message: AMPMessage) -> AMPMessage:
        """Handle decision tree evaluation request."""
        try:
            tree = message.payload["parameters"]["tree"]
            context = message.payload["parameters"].get("context", {})
            
            result, path, evaluations = await self._evaluate_decision_tree(tree, context)
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "result": result,
                    "path": path,
                    "evaluations": evaluations
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to evaluate decision tree: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.CONDITION_EVALUATION_FAILED,
                error_message=str(e)
            )
    
    async def _handle_rule_engine(self, message: AMPMessage) -> AMPMessage:
        """Handle rule engine execution request."""
        try:
            start_time = datetime.now()
            
            rules = message.payload["parameters"]["rules"]
            context = message.payload["parameters"].get("context", {})
            execution_mode = message.payload["parameters"].get("execution_mode", "first_match")
            
            matched_rules, actions = await self._execute_rule_engine(rules, context, execution_mode)
            
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return AMPMessage.create_response(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                result={
                    "matched_rules": matched_rules,
                    "actions": actions,
                    "execution_time_ms": execution_time
                }
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute rule engine: {e}")
            return AMPMessage.create_error(
                agent_id=self.agent_id,
                target_agent=message.source.agent_id,
                correlation_id=message.headers.correlation_id,
                error_code=WorkflowErrorCodes.CONDITION_EVALUATION_FAILED,
                error_message=str(e)
            )
    
    async def _evaluate_condition(self, expression: str, context: Dict[str, Any], 
                                condition_type: str = "python") -> tuple[bool, Dict[str, Any]]:
        """Evaluate a condition expression."""
        details = {
            "expression": expression,
            "condition_type": condition_type,
            "context_keys": list(context.keys())
        }
        
        try:
            if condition_type == "python":
                result = await self._evaluate_python_expression(expression, context)
            elif condition_type == "simple":
                result = await self._evaluate_simple_condition(expression, context)
            elif condition_type == "regex":
                result = await self._evaluate_regex_condition(expression, context)
            elif condition_type == "custom":
                result = await self._evaluate_custom_condition(expression, context)
            else:
                raise ValueError(f"Unsupported condition type: {condition_type}")
            
            details["evaluation_successful"] = True
            return bool(result), details
            
        except Exception as e:
            details["evaluation_successful"] = False
            details["error"] = str(e)
            raise
    
    async def _evaluate_python_expression(self, expression: str, context: Dict[str, Any]) -> Any:
        """Evaluate a Python expression safely."""
        # Parse the expression to check for safety
        try:
            parsed = ast.parse(expression, mode='eval')
        except SyntaxError as e:
            raise ValueError(f"Invalid Python expression syntax: {e}")
        
        # Check for unsafe operations
        self._validate_ast_node(parsed.body)
        
        # Create safe evaluation environment
        safe_dict = {**self.safe_names, **context, **self.custom_functions}
        
        # Evaluate the expression
        try:
            result = eval(expression, {"__builtins__": {}}, safe_dict)
            return result
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {e}")
    
    async def _evaluate_simple_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a simple condition using basic operators."""
        # Parse simple conditions like: field == value, field > 10, field in [1,2,3]
        expression = expression.strip()
        
        # Handle various comparison operators
        operators_map = {
            '==': operator.eq,
            '!=': operator.ne,
            '>=': operator.ge,
            '<=': operator.le,
            '>': operator.gt,
            '<': operator.lt,
            'in': lambda x, y: x in y,
            'not in': lambda x, y: x not in y,
            'contains': lambda x, y: y in x,
            'startswith': lambda x, y: str(x).startswith(str(y)),
            'endswith': lambda x, y: str(x).endswith(str(y)),
            'matches': lambda x, y: bool(re.match(str(y), str(x))),
        }
        
        # Find the operator
        for op_str, op_func in operators_map.items():
            if op_str in expression:
                parts = expression.split(op_str, 1)
                if len(parts) == 2:
                    left_expr, right_expr = [p.strip() for p in parts]
                    
                    # Evaluate left side (usually a context variable)
                    left_value = self._resolve_value(left_expr, context)
                    
                    # Evaluate right side (value or expression)
                    right_value = self._resolve_value(right_expr, context)
                    
                    # Apply operator
                    return op_func(left_value, right_value)
        
        # If no operator found, treat as boolean context variable
        return bool(self._resolve_value(expression, context))
    
    async def _evaluate_regex_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a regex condition."""
        # Format: field ~ pattern or field !~ pattern
        if ' ~ ' in expression:
            field, pattern = expression.split(' ~ ', 1)
            field = field.strip()
            pattern = pattern.strip().strip('"\'')
            
            value = str(self._resolve_value(field, context))
            return bool(re.search(pattern, value))
        
        elif ' !~ ' in expression:
            field, pattern = expression.split(' !~ ', 1)
            field = field.strip()
            pattern = pattern.strip().strip('"\'')
            
            value = str(self._resolve_value(field, context))
            return not bool(re.search(pattern, value))
        
        else:
            raise ValueError("Invalid regex condition format. Use 'field ~ pattern' or 'field !~ pattern'")
    
    async def _evaluate_custom_condition(self, expression: str, context: Dict[str, Any]) -> bool:
        """Evaluate a custom condition using registered functions."""
        # Format: function_name(arg1, arg2, ...)
        import re
        
        # Parse function call
        match = re.match(r'(\w+)\((.*)\)', expression.strip())
        if not match:
            raise ValueError("Invalid custom condition format. Use 'function_name(arg1, arg2, ...)'")
        
        func_name, args_str = match.groups()
        
        if func_name not in self.custom_functions:
            raise ValueError(f"Custom function '{func_name}' not registered")
        
        # Parse arguments
        args = []
        if args_str.strip():
            for arg in args_str.split(','):
                arg = arg.strip()
                args.append(self._resolve_value(arg, context))
        
        # Call custom function
        func = self.custom_functions[func_name]
        return bool(func(*args))
    
    async def _evaluate_decision_tree(self, tree: Dict[str, Any], context: Dict[str, Any]) -> tuple[Any, List[str], List[Dict]]:
        """Evaluate a decision tree structure."""
        path = []
        evaluations = []
        current_node = tree
        
        while isinstance(current_node, dict) and 'condition' in current_node:
            condition = current_node['condition']
            condition_type = current_node.get('type', 'python')
            
            # Evaluate condition
            try:
                result, details = await self._evaluate_condition(condition, context, condition_type)
                evaluations.append({
                    "condition": condition,
                    "result": result,
                    "details": details
                })
                
                # Choose next node based on result
                if result:
                    path.append("true")
                    current_node = current_node.get('true_branch')
                else:
                    path.append("false")
                    current_node = current_node.get('false_branch')
                
                # If we reach a leaf node
                if not isinstance(current_node, dict) or 'condition' not in current_node:
                    break
                    
            except Exception as e:
                evaluations.append({
                    "condition": condition,
                    "result": False,
                    "error": str(e)
                })
                # Take default branch on error
                path.append("error")
                current_node = current_node.get('error_branch', current_node.get('false_branch'))
                break
        
        return current_node, path, evaluations
    
    async def _execute_rule_engine(self, rules: List[Dict[str, Any]], context: Dict[str, Any], 
                                 execution_mode: str) -> tuple[List[Dict], List[Any]]:
        """Execute rule engine with multiple rules."""
        matched_rules = []
        actions = []
        
        # Sort rules by priority if available
        sorted_rules = sorted(rules, key=lambda r: r.get('priority', 0), reverse=True)
        
        for rule in sorted_rules:
            rule_id = rule.get('id', f"rule_{len(matched_rules)}")
            condition = rule.get('condition')
            condition_type = rule.get('type', 'python')
            action = rule.get('action')
            
            if not condition:
                continue
            
            try:
                # Evaluate rule condition
                result, details = await self._evaluate_condition(condition, context, condition_type)
                
                if result:
                    matched_rule = {
                        "id": rule_id,
                        "condition": condition,
                        "priority": rule.get('priority', 0),
                        "details": details
                    }
                    matched_rules.append(matched_rule)
                    
                    if action:
                        actions.append(action)
                    
                    # Stop on first match if required
                    if execution_mode == "first_match":
                        break
                        
            except Exception as e:
                self.logger.warning(f"Rule {rule_id} evaluation failed: {e}")
                continue
        
        # For highest priority mode, keep only the highest priority matches
        if execution_mode == "highest_priority" and matched_rules:
            highest_priority = max(rule['priority'] for rule in matched_rules)
            matched_rules = [rule for rule in matched_rules if rule['priority'] == highest_priority]
            # Filter actions to match
            actions = actions[:len(matched_rules)]
        
        return matched_rules, actions
    
    def _resolve_value(self, expr: str, context: Dict[str, Any]) -> Any:
        """Resolve a value expression to its actual value."""
        expr = expr.strip()
        
        # Handle quoted strings
        if (expr.startswith('"') and expr.endswith('"')) or (expr.startswith("'") and expr.endswith("'")):
            return expr[1:-1]
        
        # Handle numbers
        try:
            if '.' in expr:
                return float(expr)
            else:
                return int(expr)
        except ValueError:
            pass
        
        # Handle booleans
        if expr.lower() == 'true':
            return True
        elif expr.lower() == 'false':
            return False
        elif expr.lower() == 'none' or expr.lower() == 'null':
            return None
        
        # Handle lists [1,2,3]
        if expr.startswith('[') and expr.endswith(']'):
            try:
                return ast.literal_eval(expr)
            except:
                pass
        
        # Handle context variables (supports dot notation)
        if '.' in expr:
            parts = expr.split('.')
            value = context
            for part in parts:
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    return None
            return value
        else:
            return context.get(expr, expr)
    
    def _validate_ast_node(self, node: ast.AST):
        """Validate AST node for safety."""
        allowed_nodes = (
            ast.Expression, ast.BoolOp, ast.BinOp, ast.UnaryOp, ast.Compare,
            ast.Call, ast.Constant, ast.Name, ast.Load, ast.List, ast.Tuple,
            ast.Dict, ast.Subscript, ast.Index, ast.Attribute
        )
        
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsafe AST node type: {type(node).__name__}")
        
        # Recursively validate child nodes
        for child in ast.iter_child_nodes(node):
            self._validate_ast_node(child)
        
        # Additional safety checks
        if isinstance(node, ast.Name):
            if node.id.startswith('_'):
                raise ValueError(f"Access to private attributes not allowed: {node.id}")
        
        if isinstance(node, ast.Attribute):
            if node.attr.startswith('_'):
                raise ValueError(f"Access to private attributes not allowed: {node.attr}")
    
    def _register_builtin_functions(self):
        """Register built-in custom functions."""
        self.custom_functions.update({
            'is_empty': lambda x: not bool(x),
            'is_not_empty': lambda x: bool(x),
            'between': lambda x, min_val, max_val: min_val <= x <= max_val,
            'not_between': lambda x, min_val, max_val: not (min_val <= x <= max_val),
            'is_numeric': lambda x: isinstance(x, (int, float)),
            'is_string': lambda x: isinstance(x, str),
            'is_list': lambda x: isinstance(x, list),
            'is_dict': lambda x: isinstance(x, dict),
            'contains_any': lambda x, items: any(item in x for item in items),
            'contains_all': lambda x, items: all(item in x for item in items),
            'date_after': lambda date_str, after_str: datetime.fromisoformat(date_str) > datetime.fromisoformat(after_str),
            'date_before': lambda date_str, before_str: datetime.fromisoformat(date_str) < datetime.fromisoformat(before_str),
            'days_ago': lambda days: (datetime.now() - datetime.timedelta(days=days)).isoformat(),
            'regex_match': lambda text, pattern: bool(re.match(pattern, str(text))),
            'regex_search': lambda text, pattern: bool(re.search(pattern, str(text))),
        })
    
    def register_custom_function(self, name: str, func: Callable):
        """Register a custom function for condition evaluation."""
        self.custom_functions[name] = func
        self.logger.info(f"Registered custom function: {name}")
    
    def _get_from_cache(self, cache_key: str) -> Optional[tuple]:
        """Get result from evaluation cache."""
        if cache_key in self.evaluation_cache:
            result, timestamp = self.evaluation_cache[cache_key]
            if (datetime.now() - timestamp).total_seconds() < self.cache_ttl:
                return result
            else:
                del self.evaluation_cache[cache_key]
        return None
    
    def _store_in_cache(self, cache_key: str, result: tuple):
        """Store result in evaluation cache."""
        self.evaluation_cache[cache_key] = (result, datetime.now())
    
    # Additional handler methods would continue here...
    async def _handle_validate_condition(self, message: AMPMessage) -> AMPMessage:
        """Handle condition validation request."""
        # Implementation for validating condition syntax
        pass
    
    async def _handle_parse_expression(self, message: AMPMessage) -> AMPMessage:
        """Handle expression parsing request."""
        # Implementation for parsing and analyzing expressions
        pass