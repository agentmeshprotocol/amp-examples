#!/usr/bin/env python3
"""
Multi-Agent Chatbot System Runner

This script starts all the agents in the chatbot system and coordinates their operation.
Uses the Agent Mesh Protocol for inter-agent communication.
"""

import asyncio
import logging
import argparse
import signal
import sys
import yaml
from pathlib import Path
from typing import Dict, Any, List

# Import our agents
from agents.router_agent import RouterAgent
from agents.faq_agent import FAQAgent
from agents.sales_agent import SalesAgent
from agents.tech_support_agent import TechSupportAgent
from agents.conversation_manager import ConversationManager

# AMP imports
sys.path.append(str(Path(__file__).parent.parent / "shared-lib"))
from amp_client import AMPClient


class ChatbotSystem:
    """Main chatbot system that manages all agents."""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Agents
        self.router_agent = None
        self.faq_agent = None
        self.sales_agent = None
        self.tech_support_agent = None
        self.conversation_manager = None
        
        # AMP clients
        self.clients: Dict[str, AMPClient] = {}
        
        # System state
        self.running = False
        self.shutdown_event = asyncio.Event()
        
    def _load_config(self, config_path: str = None) -> Dict[str, Any]:
        """Load system configuration."""
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "agent_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.warning(f"Config file not found: {config_path}, using defaults")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            "global": {
                "log_level": "INFO",
                "max_concurrent_conversations": 100
            },
            "amp": {
                "registry_url": "http://localhost:8000",
                "transport_type": "http"
            }
        }
    
    async def start(self):
        """Start all agents and the chatbot system."""
        self.logger.info("Starting Multi-Agent Chatbot System...")
        
        try:
            # Start conversation manager first
            self.logger.info("Starting Conversation Manager...")
            self.conversation_manager = ConversationManager(
                self.config.get("conversation_manager", {})
            )
            self.clients["conversation-manager"] = await self.conversation_manager.start_amp_agent(
                "conversation-manager",
                self.config["amp"]["registry_url"]
            )
            
            # Start specialized agents
            await self._start_agents()
            
            # Register signal handlers
            self._setup_signal_handlers()
            
            self.running = True
            self.logger.info("🚀 Chatbot System is running!")
            self.logger.info("Available agents:")
            for agent_name in self.clients.keys():
                self.logger.info(f"  - {agent_name}")
            
            # Wait for shutdown signal
            await self.shutdown_event.wait()
            
        except Exception as e:
            self.logger.error(f"Failed to start chatbot system: {e}")
            raise
    
    async def _start_agents(self):
        """Start all specialized agents."""
        registry_url = self.config["amp"]["registry_url"]
        
        # Start Router Agent
        self.logger.info("Starting Router Agent...")
        self.router_agent = RouterAgent()
        self.clients["router-agent"] = await self.router_agent.start_amp_agent(
            "router-agent", registry_url
        )
        
        # Start FAQ Agent
        self.logger.info("Starting FAQ Agent...")
        self.faq_agent = FAQAgent()
        self.clients["faq-agent"] = await self.faq_agent.start_amp_agent(
            "faq-agent", registry_url
        )
        
        # Start Sales Agent
        self.logger.info("Starting Sales Agent...")
        self.sales_agent = SalesAgent()
        self.clients["sales-agent"] = await self.sales_agent.start_amp_agent(
            "sales-agent", registry_url
        )
        
        # Start Tech Support Agent
        self.logger.info("Starting Tech Support Agent...")
        self.tech_support_agent = TechSupportAgent()
        self.clients["tech-support-agent"] = await self.tech_support_agent.start_amp_agent(
            "tech-support-agent", registry_url
        )
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
            asyncio.create_task(self.shutdown())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    async def shutdown(self):
        """Shutdown all agents gracefully."""
        if not self.running:
            return
        
        self.logger.info("Shutting down Multi-Agent Chatbot System...")
        self.running = False
        
        # Disconnect all clients
        for agent_name, client in self.clients.items():
            try:
                self.logger.info(f"Disconnecting {agent_name}...")
                await client.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting {agent_name}: {e}")
        
        # Stop conversation manager cleanup
        if self.conversation_manager:
            try:
                await self.conversation_manager.stop_cleanup_task()
            except Exception as e:
                self.logger.error(f"Error stopping conversation manager: {e}")
        
        self.shutdown_event.set()
        self.logger.info("Chatbot system shutdown complete")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "system_status": "healthy" if self.running else "stopped",
            "agents": {},
            "total_agents": len(self.clients),
            "active_agents": 0
        }
        
        for agent_name, client in self.clients.items():
            try:
                agent_health = await client.health_check()
                health_status["agents"][agent_name] = agent_health
                if agent_health.get("status") == "healthy":
                    health_status["active_agents"] += 1
            except Exception as e:
                health_status["agents"][agent_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return health_status
    
    async def process_user_message(self, user_input: str, session_id: str = "default",
                                  user_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process a user message through the chatbot system."""
        if not self.running:
            return {
                "error": "Chatbot system is not running",
                "response": "I'm sorry, the system is currently unavailable."
            }
        
        try:
            # Route message through router agent
            result = await self.router_agent.route_conversation(
                session_id, user_input, user_context
            )
            
            # Add conversation tracking
            if self.conversation_manager:
                await self.conversation_manager.add_message(
                    session_id, "user", user_input
                )
                if "response" in result:
                    await self.conversation_manager.add_message(
                        session_id, "assistant", result["response"],
                        agent=result.get("agent")
                    )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return {
                "error": str(e),
                "response": "I'm sorry, I encountered an error processing your message. Please try again."
            }


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Chatbot System")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--interactive", action="store_true", help="Start interactive mode")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start chatbot system
    chatbot = ChatbotSystem(args.config)
    
    try:
        if args.interactive:
            # Start in interactive mode
            start_task = asyncio.create_task(chatbot.start())
            await asyncio.sleep(2)  # Wait for system to start
            
            print("\n🤖 Multi-Agent Chatbot Interactive Mode")
            print("Type your messages below. Use 'quit' or 'exit' to stop.\n")
            
            session_id = "interactive-session"
            
            while chatbot.running:
                try:
                    user_input = input("You: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', 'bye']:
                        break
                    
                    if not user_input:
                        continue
                    
                    # Process message
                    result = await chatbot.process_user_message(user_input, session_id)
                    
                    # Display response
                    agent = result.get("agent", "system")
                    response = result.get("response", "No response generated")
                    
                    print(f"Bot ({agent}): {response}")
                    
                    # Show additional info if available
                    if result.get("intent"):
                        print(f"  └─ Intent: {result['intent']} (confidence: {result.get('confidence', 0):.2f})")
                    
                    print()
                    
                except KeyboardInterrupt:
                    break
                except EOFError:
                    break
                except Exception as e:
                    print(f"Error: {e}")
            
            await chatbot.shutdown()
            
        else:
            # Start in service mode
            await chatbot.start()
            
    except KeyboardInterrupt:
        print("\nShutdown requested...")
    except Exception as e:
        logging.error(f"System error: {e}")
        sys.exit(1)
    finally:
        if chatbot.running:
            await chatbot.shutdown()


if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())