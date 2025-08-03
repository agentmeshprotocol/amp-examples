#!/usr/bin/env python3
"""
Command Line Interface for Multi-Agent Chatbot

A simple CLI interface for testing and interacting with the chatbot system.
"""

import asyncio
import argparse
import sys
import json
import uuid
from pathlib import Path
from typing import Dict, Any

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from run_chatbot import ChatbotSystem
from amp_client import AMPClient, AMPClientConfig
from amp_types import TransportType


class ChatbotCLI:
    """Command-line interface for the chatbot system."""
    
    def __init__(self, endpoint: str = "http://localhost:8000"):
        self.endpoint = endpoint
        self.session_id = str(uuid.uuid4())
        self.chatbot_system = None
        self.conversation_history = []
        
    async def start_embedded_system(self):
        """Start the chatbot system in embedded mode."""
        print("🚀 Starting embedded chatbot system...")
        self.chatbot_system = ChatbotSystem()
        
        # Start in background
        start_task = asyncio.create_task(self.chatbot_system.start())
        await asyncio.sleep(3)  # Wait for system to initialize
        
        if not self.chatbot_system.running:
            raise RuntimeError("Failed to start chatbot system")
        
        print("✅ Chatbot system is ready!")
        return start_task
    
    async def connect_to_remote_system(self):
        """Connect to a remote chatbot system."""
        print(f"🔗 Connecting to remote chatbot at {self.endpoint}...")
        
        # Create AMP client to connect to router agent
        config = AMPClientConfig(
            agent_id="cli-client",
            agent_name="CLI Client",
            framework="cli",
            transport_type=TransportType.HTTP,
            endpoint=self.endpoint
        )
        
        self.amp_client = AMPClient(config)
        connected = await self.amp_client.connect()
        
        if not connected:
            raise RuntimeError(f"Failed to connect to {self.endpoint}")
        
        print("✅ Connected to remote chatbot system!")
    
    async def send_message_embedded(self, message: str) -> Dict[str, Any]:
        """Send message to embedded chatbot system."""
        if not self.chatbot_system or not self.chatbot_system.running:
            return {"error": "Chatbot system not running"}
        
        return await self.chatbot_system.process_user_message(
            message, self.session_id
        )
    
    async def send_message_remote(self, message: str) -> Dict[str, Any]:
        """Send message to remote chatbot system."""
        try:
            # Send to router agent for conversation routing
            response = await self.amp_client.invoke_capability(
                "router-agent",
                "conversation-routing",
                {
                    "user_input": message,
                    "session_id": self.session_id,
                    "context": {}
                }
            )
            
            if "result" in response:
                return response["result"].get("routing_result", {})
            else:
                return {"error": "No result in response", "response": "Sorry, I couldn't process your message."}
                
        except Exception as e:
            return {"error": str(e), "response": "Sorry, there was an error processing your message."}
    
    async def interactive_mode(self, mode: str = "embedded"):
        """Run interactive chat mode."""
        print(f"\n🤖 Multi-Agent Chatbot CLI ({mode} mode)")
        print("=" * 50)
        print("Commands:")
        print("  /help     - Show this help")
        print("  /stats    - Show conversation statistics")
        print("  /history  - Show conversation history")
        print("  /clear    - Clear conversation history")
        print("  /session  - Show current session ID")
        print("  /quit     - Exit the chatbot")
        print("=" * 50)
        print("Type your message and press Enter. Type /quit to exit.\n")
        
        send_message = (self.send_message_embedded if mode == "embedded" 
                       else self.send_message_remote)
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/'):
                    await self._handle_command(user_input, mode)
                    continue
                
                # Send message to chatbot
                print("🤔 Processing...", end="", flush=True)
                result = await send_message(user_input)
                print("\\r" + " " * 15 + "\\r", end="")  # Clear processing message
                
                # Store in history
                self.conversation_history.append({
                    "role": "user",
                    "content": user_input,
                    "timestamp": asyncio.get_event_loop().time()
                })
                
                # Display response
                if "error" in result:
                    print(f"❌ Error: {result['error']}")
                else:
                    agent = result.get("agent", "system")
                    response = result.get("response", "No response generated")
                    
                    print(f"🤖 Bot ({agent}): {response}")
                    
                    # Show additional info
                    self._show_response_details(result)
                    
                    # Store response in history
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response,
                        "agent": agent,
                        "timestamp": asyncio.get_event_loop().time(),
                        "metadata": result
                    })
                
                print()
                
            except KeyboardInterrupt:
                print("\\n👋 Goodbye!")
                break
            except EOFError:
                print("\\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
    
    def _show_response_details(self, result: Dict[str, Any]):
        """Show additional response details."""
        details = []
        
        if result.get("intent"):
            confidence = result.get("confidence", 0)
            details.append(f"Intent: {result['intent']} ({confidence:.2f})")
        
        if result.get("ticket_id"):
            details.append(f"Ticket: {result['ticket_id']}")
        
        if result.get("recommendations"):
            rec_count = len(result["recommendations"])
            details.append(f"Recommendations: {rec_count}")
        
        if result.get("escalation"):
            details.append("Escalated to specialist")
        
        if details:
            print(f"   └─ {' | '.join(details)}")
    
    async def _handle_command(self, command: str, mode: str):
        """Handle CLI commands."""
        cmd = command.lower().strip()
        
        if cmd == "/help":
            print("📖 Available commands:")
            print("  /help     - Show this help message")
            print("  /stats    - Show conversation statistics")
            print("  /history  - Show conversation history")
            print("  /clear    - Clear conversation history")
            print("  /session  - Show current session ID")
            print("  /health   - Check system health (embedded mode only)")
            print("  /quit     - Exit the chatbot")
            
        elif cmd == "/stats":
            await self._show_stats(mode)
            
        elif cmd == "/history":
            self._show_history()
            
        elif cmd == "/clear":
            self.conversation_history.clear()
            print("🗑️  Conversation history cleared")
            
        elif cmd == "/session":
            print(f"📋 Current session ID: {self.session_id}")
            print(f"💬 Messages in session: {len(self.conversation_history)}")
            
        elif cmd == "/health" and mode == "embedded":
            await self._show_health()
            
        elif cmd == "/quit":
            print("👋 Goodbye!")
            sys.exit(0)
            
        else:
            print(f"❓ Unknown command: {command}")
            print("Type /help for available commands")
    
    async def _show_stats(self, mode: str):
        """Show conversation statistics."""
        print("📊 Conversation Statistics:")
        print(f"  Session ID: {self.session_id}")
        print(f"  Total messages: {len(self.conversation_history)}")
        
        # Count messages by role
        user_count = sum(1 for msg in self.conversation_history if msg["role"] == "user")
        bot_count = sum(1 for msg in self.conversation_history if msg["role"] == "assistant")
        
        print(f"  User messages: {user_count}")
        print(f"  Bot responses: {bot_count}")
        
        # Count by agent (for bot responses)
        if bot_count > 0:
            agents = {}
            for msg in self.conversation_history:
                if msg["role"] == "assistant":
                    agent = msg.get("agent", "unknown")
                    agents[agent] = agents.get(agent, 0) + 1
            
            print("  Responses by agent:")
            for agent, count in agents.items():
                print(f"    {agent}: {count}")
        
        # Show system stats for embedded mode
        if mode == "embedded" and self.chatbot_system:
            try:
                health = await self.chatbot_system.health_check()
                print(f"  System status: {health.get('system_status', 'unknown')}")
                print(f"  Active agents: {health.get('active_agents', 0)}/{health.get('total_agents', 0)}")
            except Exception as e:
                print(f"  System stats unavailable: {e}")
    
    def _show_history(self):
        """Show conversation history."""
        if not self.conversation_history:
            print("📝 No conversation history")
            return
        
        print(f"📝 Conversation History ({len(self.conversation_history)} messages):")
        print("-" * 60)
        
        for i, msg in enumerate(self.conversation_history[-10:], 1):  # Show last 10
            role = msg["role"]
            content = msg["content"]
            
            if role == "user":
                print(f"{i:2d}. You: {content}")
            else:
                agent = msg.get("agent", "bot")
                print(f"{i:2d}. Bot ({agent}): {content}")
        
        if len(self.conversation_history) > 10:
            print(f"... and {len(self.conversation_history) - 10} earlier messages")
    
    async def _show_health(self):
        """Show system health."""
        if not self.chatbot_system:
            print("❌ System not available")
            return
        
        try:
            health = await self.chatbot_system.health_check()
            print("🏥 System Health:")
            print(f"  Overall status: {health.get('system_status', 'unknown')}")
            print(f"  Active agents: {health.get('active_agents', 0)}/{health.get('total_agents', 0)}")
            
            print("  Agent details:")
            for agent_name, agent_health in health.get("agents", {}).items():
                status = agent_health.get("status", "unknown")
                print(f"    {agent_name}: {status}")
                
        except Exception as e:
            print(f"❌ Failed to get health status: {e}")
    
    async def cleanup(self):
        """Clean up resources."""
        if hasattr(self, "amp_client"):
            try:
                await self.amp_client.disconnect()
            except Exception:
                pass
        
        if self.chatbot_system and self.chatbot_system.running:
            try:
                await self.chatbot_system.shutdown()
            except Exception:
                pass


async def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(description="Multi-Agent Chatbot CLI")
    parser.add_argument("--mode", choices=["embedded", "remote"], default="embedded",
                       help="Run mode: embedded (start local system) or remote (connect to existing)")
    parser.add_argument("--endpoint", default="http://localhost:8000",
                       help="Endpoint for remote mode")
    parser.add_argument("--test", action="store_true",
                       help="Run in test mode with predefined messages")
    args = parser.parse_args()
    
    cli = ChatbotCLI(args.endpoint)
    
    try:
        # Initialize based on mode
        if args.mode == "embedded":
            start_task = await cli.start_embedded_system()
        else:
            await cli.connect_to_remote_system()
        
        if args.test:
            await run_test_mode(cli, args.mode)
        else:
            await cli.interactive_mode(args.mode)
            
    except KeyboardInterrupt:
        print("\\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    finally:
        await cli.cleanup()


async def run_test_mode(cli: ChatbotCLI, mode: str):
    """Run predefined test messages."""
    print("🧪 Running test mode...")
    
    test_messages = [
        "Hello there!",
        "What are your business hours?",
        "How much do your products cost?",
        "I'm having trouble logging in",
        "Can I get a demo of your software?",
        "Thank you for your help!"
    ]
    
    send_message = (cli.send_message_embedded if mode == "embedded" 
                   else cli.send_message_remote)
    
    for i, message in enumerate(test_messages, 1):
        print(f"\\n[Test {i}/{len(test_messages)}]")
        print(f"You: {message}")
        
        result = await send_message(message)
        
        if "error" in result:
            print(f"❌ Error: {result['error']}")
        else:
            agent = result.get("agent", "system")
            response = result.get("response", "No response")
            print(f"🤖 Bot ({agent}): {response}")
            
            if result.get("intent"):
                print(f"   └─ Intent: {result['intent']} ({result.get('confidence', 0):.2f})")
        
        await asyncio.sleep(1)  # Brief pause between messages
    
    print("\\n✅ Test completed!")


if __name__ == "__main__":
    # Handle Windows event loop policy
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    asyncio.run(main())