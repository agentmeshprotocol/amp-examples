#!/usr/bin/env python3
"""
AMP Workflow Orchestration System Launcher

This script provides a convenient way to start all components of the
workflow orchestration system with proper configuration and coordination.
"""

import asyncio
import argparse
import logging
import signal
import sys
import time
from pathlib import Path
from typing import List, Optional

# Add shared-lib to path
sys.path.append(str(Path(__file__).parent.parent / "shared-lib"))

from agents.workflow_engine import WorkflowEngine
from agents.task_executor import TaskExecutor
from agents.state_manager import StateManager
from agents.condition_evaluator import ConditionEvaluator
from agents.error_handler import ErrorHandler
from agents.monitor_agent import MonitorAgent
from web.app import WebDashboard


class OrchestrationSystem:
    """
    Main orchestration system that manages all agents and the web interface.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/agent_config.yaml"
        self.logger = logging.getLogger("OrchestrationSystem")
        
        # System components
        self.components = []
        self.running = False
        
        # Component instances
        self.workflow_engine = None
        self.task_executors = []
        self.state_manager = None
        self.condition_evaluator = None
        self.error_handler = None
        self.monitor_agent = None
        self.web_dashboard = None
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.running = False
        
    async def start(self, components: List[str] = None, num_executors: int = 1):
        """
        Start the orchestration system.
        
        Args:
            components: List of components to start. If None, starts all.
            num_executors: Number of task executor instances to start.
        """
        if components is None:
            components = [
                'state-manager',
                'condition-evaluator', 
                'error-handler',
                'monitor-agent',
                'workflow-engine',
                'task-executor',
                'web-dashboard'
            ]
        
        self.logger.info("Starting AMP Workflow Orchestration System")
        self.running = True
        
        try:
            # Start components in dependency order
            await self._start_components(components, num_executors)
            
            # Wait for shutdown signal
            while self.running:
                await asyncio.sleep(1)
                
                # Health check - restart failed components
                await self._health_check()
                
        except Exception as e:
            self.logger.error(f"System startup failed: {e}")
            raise
        finally:
            await self.stop()
    
    async def _start_components(self, components: List[str], num_executors: int):
        """Start system components in proper order."""
        
        # Phase 1: Core infrastructure
        if 'state-manager' in components:
            self.logger.info("Starting State Manager...")
            self.state_manager = StateManager(port=8082)
            await self.state_manager.start()
            self.components.append(('state-manager', self.state_manager))
            await asyncio.sleep(2)  # Allow startup
        
        if 'condition-evaluator' in components:
            self.logger.info("Starting Condition Evaluator...")
            self.condition_evaluator = ConditionEvaluator(port=8083)
            await self.condition_evaluator.start()
            self.components.append(('condition-evaluator', self.condition_evaluator))
            await asyncio.sleep(1)
        
        if 'error-handler' in components:
            self.logger.info("Starting Error Handler...")
            self.error_handler = ErrorHandler(port=8084)
            await self.error_handler.start()
            self.components.append(('error-handler', self.error_handler))
            await asyncio.sleep(1)
        
        if 'monitor-agent' in components:
            self.logger.info("Starting Monitor Agent...")
            self.monitor_agent = MonitorAgent(port=8085)
            await self.monitor_agent.start()
            self.components.append(('monitor-agent', self.monitor_agent))
            await asyncio.sleep(1)
        
        # Phase 2: Workflow engine
        if 'workflow-engine' in components:
            self.logger.info("Starting Workflow Engine...")
            self.workflow_engine = WorkflowEngine(port=8080)
            await self.workflow_engine.start()
            self.components.append(('workflow-engine', self.workflow_engine))
            await asyncio.sleep(2)
        
        # Phase 3: Task executors
        if 'task-executor' in components:
            self.logger.info(f"Starting {num_executors} Task Executor(s)...")
            for i in range(num_executors):
                executor = TaskExecutor(
                    agent_id=f"task-executor-{i+1}",
                    port=8081 + i
                )
                await executor.start()
                self.task_executors.append(executor)
                self.components.append((f'task-executor-{i+1}', executor))
                await asyncio.sleep(0.5)
        
        # Phase 4: Web interface
        if 'web-dashboard' in components:
            self.logger.info("Starting Web Dashboard...")
            self.web_dashboard = WebDashboard(port=8090)
            # Start web dashboard in background task
            asyncio.create_task(self.web_dashboard.start())
            self.components.append(('web-dashboard', self.web_dashboard))
            await asyncio.sleep(1)
        
        self.logger.info("All components started successfully!")
        self._print_status()
    
    async def _health_check(self):
        """Perform health checks on all components."""
        # This is a simplified health check
        # In production, you'd implement proper health check endpoints
        pass
    
    async def stop(self):
        """Stop all system components gracefully."""
        self.logger.info("Stopping AMP Workflow Orchestration System...")
        
        # Stop components in reverse order
        for name, component in reversed(self.components):
            try:
                self.logger.info(f"Stopping {name}...")
                if hasattr(component, 'stop'):
                    await component.stop()
            except Exception as e:
                self.logger.error(f"Error stopping {name}: {e}")
        
        self.components.clear()
        self.logger.info("System stopped successfully")
    
    def _print_status(self):
        """Print system status information."""
        print("\n" + "="*60)
        print("AMP Workflow Orchestration System - RUNNING")
        print("="*60)
        print()
        print("Components:")
        for name, component in self.components:
            port = getattr(component, 'port', 'N/A')
            print(f"  ✓ {name:20} - Port {port}")
        print()
        print("Endpoints:")
        print(f"  Web Dashboard:     http://localhost:8090")
        print(f"  API Documentation: http://localhost:8090/docs")
        print(f"  Workflow Engine:   http://localhost:8080")
        print(f"  State Manager:     http://localhost:8082")
        print(f"  Monitor Agent:     http://localhost:8085")
        print()
        print("Press Ctrl+C to stop the system")
        print("="*60)
        print()
    
    async def load_example_workflows(self):
        """Load example workflows into the system."""
        if not self.workflow_engine:
            self.logger.warning("Workflow engine not running, skipping workflow loading")
            return
        
        workflow_dir = Path("workflows")
        if not workflow_dir.exists():
            self.logger.warning("Workflows directory not found")
            return
        
        workflow_files = list(workflow_dir.glob("*.yaml")) + list(workflow_dir.glob("*.yml")) + list(workflow_dir.glob("*.json"))
        
        self.logger.info(f"Loading {len(workflow_files)} example workflows...")
        
        for workflow_file in workflow_files:
            try:
                format_type = "yaml" if workflow_file.suffix in ['.yaml', '.yml'] else "json"
                await self.workflow_engine._load_workflow_from_file(str(workflow_file), format_type)
                self.logger.info(f"Loaded workflow: {workflow_file.name}")
            except Exception as e:
                self.logger.error(f"Failed to load workflow {workflow_file.name}: {e}")
    
    async def run_demo(self):
        """Run a demonstration workflow."""
        if not self.workflow_engine:
            self.logger.warning("Workflow engine not running, skipping demo")
            return
        
        self.logger.info("Running demonstration workflow...")
        
        try:
            # Start the simple example workflow
            instance = await self.workflow_engine._create_workflow_instance(
                workflow_id="hello-world-workflow",
                inputs={"name": "AMP Demo", "language": "en"}
            )
            
            await self.workflow_engine._start_workflow_execution(instance)
            
            self.logger.info(f"Demo workflow started with instance ID: {instance.id}")
            self.logger.info("Check the web dashboard to monitor progress")
            
        except Exception as e:
            self.logger.error(f"Failed to run demo workflow: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="AMP Workflow Orchestration System")
    
    parser.add_argument(
        '--components',
        nargs='+',
        choices=['state-manager', 'condition-evaluator', 'error-handler', 
                'monitor-agent', 'workflow-engine', 'task-executor', 'web-dashboard'],
        help='Components to start (default: all)'
    )
    
    parser.add_argument(
        '--executors',
        type=int,
        default=1,
        help='Number of task executor instances (default: 1)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config/agent_config.yaml',
        help='Configuration file path'
    )
    
    parser.add_argument(
        '--load-examples',
        action='store_true',
        help='Load example workflows on startup'
    )
    
    parser.add_argument(
        '--run-demo',
        action='store_true',
        help='Run demonstration workflow after startup'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start the orchestration system
    system = OrchestrationSystem(args.config)
    
    try:
        # Start the system
        await system.start(components=args.components, num_executors=args.executors)
        
        # Load example workflows if requested
        if args.load_examples:
            await system.load_example_workflows()
        
        # Run demo if requested
        if args.run_demo:
            await system.run_demo()
            
    except KeyboardInterrupt:
        logging.info("Shutdown requested by user")
    except Exception as e:
        logging.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())