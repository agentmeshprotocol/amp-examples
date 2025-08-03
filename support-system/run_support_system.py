#!/usr/bin/env python3
"""
Support System Main Runner

This script starts all support agents and the web interface for the
intelligent customer support system using the AMP protocol.
"""

import asyncio
import logging
import os
import signal
import sys
from typing import List, Dict, Any
from contextlib import asynccontextmanager

# Add shared-lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared-lib'))

from amp_client import AMPClientConfig
from amp_types import TransportType

# Import support agents
from agents.ticket_classifier import TicketClassifierAgent
from agents.technical_support import TechnicalSupportAgent
from agents.billing_support import BillingSupportAgent
from agents.product_support import ProductSupportAgent
from agents.escalation_manager import EscalationManagerAgent
from agents.knowledge_base import KnowledgeBaseAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/support_system.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("support_system")


class SupportSystemOrchestrator:
    """Orchestrates the entire support system with multiple agents."""
    
    def __init__(self):
        self.agents = []
        self.running = False
        self.amp_endpoint = os.getenv("AMP_ENDPOINT", "http://localhost:8000")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
    
    async def initialize_agents(self):
        """Initialize all support agents."""
        
        agent_configs = [
            {
                "class": TicketClassifierAgent,
                "config": AMPClientConfig(
                    agent_id="ticket-classifier-001",
                    agent_name="Ticket Classifier",
                    framework="langchain",
                    transport_type=TransportType.HTTP,
                    endpoint=self.amp_endpoint
                ),
                "requires_openai": True
            },
            {
                "class": TechnicalSupportAgent,
                "config": AMPClientConfig(
                    agent_id="technical-support-001", 
                    agent_name="Technical Support Agent",
                    framework="crewai",
                    transport_type=TransportType.HTTP,
                    endpoint=self.amp_endpoint
                ),
                "requires_openai": True
            },
            {
                "class": BillingSupportAgent,
                "config": AMPClientConfig(
                    agent_id="billing-support-001",
                    agent_name="Billing Support Agent", 
                    framework="autogen",
                    transport_type=TransportType.HTTP,
                    endpoint=self.amp_endpoint
                ),
                "requires_openai": True
            },
            {
                "class": ProductSupportAgent,
                "config": AMPClientConfig(
                    agent_id="product-support-001",
                    agent_name="Product Support Agent",
                    framework="langchain", 
                    transport_type=TransportType.HTTP,
                    endpoint=self.amp_endpoint
                ),
                "requires_openai": True
            },
            {
                "class": EscalationManagerAgent,
                "config": AMPClientConfig(
                    agent_id="escalation-manager-001",
                    agent_name="Escalation Manager",
                    framework="custom",
                    transport_type=TransportType.HTTP,
                    endpoint=self.amp_endpoint
                ),
                "requires_openai": False
            },
            {
                "class": KnowledgeBaseAgent,
                "config": AMPClientConfig(
                    agent_id="knowledge-base-001",
                    agent_name="Knowledge Base Agent",
                    framework="custom",
                    transport_type=TransportType.HTTP,
                    endpoint=self.amp_endpoint
                ),
                "requires_openai": False
            }
        ]
        
        logger.info("Initializing support agents...")
        
        for agent_spec in agent_configs:
            try:
                if agent_spec["requires_openai"]:
                    agent = agent_spec["class"](agent_spec["config"], self.openai_api_key)
                else:
                    agent = agent_spec["class"](agent_spec["config"])
                
                self.agents.append({
                    "agent": agent,
                    "config": agent_spec["config"],
                    "class_name": agent_spec["class"].__name__
                })
                
                logger.info(f"Initialized {agent_spec['config'].agent_name}")
                
            except Exception as e:
                logger.error(f"Failed to initialize {agent_spec['class'].__name__}: {e}")
                raise
    
    async def start_agents(self):
        """Start all support agents."""
        logger.info("Starting support agents...")
        
        start_tasks = []
        for agent_info in self.agents:
            try:
                start_tasks.append(agent_info["agent"].start())
                logger.info(f"Starting {agent_info['config'].agent_name}...")
            except Exception as e:
                logger.error(f"Failed to start {agent_info['class_name']}: {e}")
                raise
        
        # Start all agents concurrently
        await asyncio.gather(*start_tasks)
        logger.info("All support agents started successfully")
    
    async def stop_agents(self):
        """Stop all support agents."""
        logger.info("Stopping support agents...")
        
        stop_tasks = []
        for agent_info in self.agents:
            try:
                stop_tasks.append(agent_info["agent"].stop())
            except Exception as e:
                logger.error(f"Error stopping {agent_info['class_name']}: {e}")
        
        # Stop all agents concurrently
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        logger.info("All support agents stopped")
    
    async def health_check(self):
        """Perform health check on all agents."""
        healthy_agents = 0
        total_agents = len(self.agents)
        
        for agent_info in self.agents:
            try:
                # Basic health check - agent should be connected
                if hasattr(agent_info["agent"], "client") and hasattr(agent_info["agent"].client, "connected"):
                    if agent_info["agent"].client.connected:
                        healthy_agents += 1
                else:
                    # Assume healthy if no explicit health check available
                    healthy_agents += 1
            except Exception as e:
                logger.warning(f"Health check failed for {agent_info['class_name']}: {e}")
        
        health_status = {
            "healthy_agents": healthy_agents,
            "total_agents": total_agents,
            "health_percentage": (healthy_agents / total_agents * 100) if total_agents > 0 else 0,
            "status": "healthy" if healthy_agents == total_agents else "degraded" if healthy_agents > 0 else "unhealthy"
        }
        
        return health_status
    
    async def run(self):
        """Run the support system."""
        try:
            self.running = True
            
            # Initialize and start agents
            await self.initialize_agents()
            await self.start_agents()
            
            logger.info("🚀 Support System is now running!")
            logger.info(f"📊 Total agents: {len(self.agents)}")
            logger.info(f"🔗 AMP endpoint: {self.amp_endpoint}")
            
            # Print agent status
            for agent_info in self.agents:
                logger.info(f"✅ {agent_info['config'].agent_name} ({agent_info['config'].framework})")
            
            # Health check every 30 seconds
            while self.running:
                try:
                    health = await self.health_check()
                    if health["status"] != "healthy":
                        logger.warning(f"System health: {health['status']} "
                                     f"({health['healthy_agents']}/{health['total_agents']} agents healthy)")
                    
                    # Wait 30 seconds before next health check
                    await asyncio.sleep(30)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error: {e}")
                    await asyncio.sleep(10)
        
        except Exception as e:
            logger.error(f"Support system error: {e}")
            raise
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Gracefully shutdown the support system."""
        logger.info("🛑 Shutting down support system...")
        self.running = False
        
        if self.agents:
            await self.stop_agents()
        
        logger.info("✅ Support system shutdown complete")


def setup_signal_handlers(orchestrator):
    """Setup signal handlers for graceful shutdown."""
    
    def signal_handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating shutdown...")
        # Create new event loop for shutdown if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run shutdown
        loop.run_until_complete(orchestrator.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


async def run_with_web_interface():
    """Run support system with web interface."""
    try:
        # Import web interface
        from web.app import create_app, run_web_server
        
        # Create orchestrator
        orchestrator = SupportSystemOrchestrator()
        
        # Setup signal handlers
        setup_signal_handlers(orchestrator)
        
        # Create web app
        app = create_app(orchestrator)
        
        # Start both systems concurrently
        web_task = asyncio.create_task(run_web_server(app))
        support_task = asyncio.create_task(orchestrator.run())
        
        # Wait for either to complete (or fail)
        done, pending = await asyncio.wait(
            [web_task, support_task],
            return_when=asyncio.FIRST_COMPLETED
        )
        
        # Cancel remaining tasks
        for task in pending:
            task.cancel()
        
        # Check for exceptions
        for task in done:
            if task.exception():
                logger.error(f"Task failed: {task.exception()}")
                raise task.exception()
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


async def run_agents_only():
    """Run support system without web interface."""
    try:
        orchestrator = SupportSystemOrchestrator()
        setup_signal_handlers(orchestrator)
        await orchestrator.run()
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
        raise


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="AMP Support System")
    parser.add_argument(
        "--mode",
        choices=["web", "agents-only"],
        default="web",
        help="Run mode: 'web' (with web interface) or 'agents-only'"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    logger.info("🎯 Starting AMP Customer Support System")
    logger.info(f"📝 Mode: {args.mode}")
    logger.info(f"📋 Log level: {args.log_level}")
    
    try:
        if args.mode == "web":
            asyncio.run(run_with_web_interface())
        else:
            asyncio.run(run_agents_only())
    except KeyboardInterrupt:
        logger.info("👋 Goodbye!")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()