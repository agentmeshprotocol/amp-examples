#!/usr/bin/env python3
"""
API Client Example for Multi-Agent Chatbot

Demonstrates how to integrate with the chatbot system using HTTP API calls.
Useful for integrating the chatbot into external applications.
"""

import asyncio
import aiohttp
import json
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class ChatMessage:
    """Represents a chat message."""
    content: str
    role: str  # 'user' or 'assistant'
    timestamp: datetime
    agent: Optional[str] = None
    intent: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ChatbotAPIClient:
    """Client for interacting with the chatbot API."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip('/')
        self.session_id = str(uuid.uuid4())
        self.session: Optional[aiohttp.ClientSession] = None
        self.conversation_history: List[ChatMessage] = []
        
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def send_message(self, message: str, user_context: Optional[Dict[str, Any]] = None) -> ChatMessage:
        """Send a message to the chatbot and get response."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        # Prepare request
        payload = {
            "message": message,
            "session_id": self.session_id,
            "user_context": user_context or {}
        }
        
        # Add user message to history
        user_msg = ChatMessage(
            content=message,
            role="user",
            timestamp=datetime.now()
        )
        self.conversation_history.append(user_msg)
        
        try:
            # Send to API
            async with self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error_text}")
                
                result = await response.json()
                
                # Create response message
                bot_msg = ChatMessage(
                    content=result["response"],
                    role="assistant",
                    timestamp=datetime.fromisoformat(result["timestamp"]),
                    agent=result["agent"],
                    intent=result.get("intent"),
                    confidence=result.get("confidence"),
                    metadata=result.get("metadata")
                )
                
                # Add to history
                self.conversation_history.append(bot_msg)
                
                return bot_msg
                
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error: {e}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get the health status of the chatbot system."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/api/health") as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error_text}")
                
                return await response.json()
                
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error: {e}")
    
    async def get_conversation_history(self, session_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get conversation history from the server."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        target_session = session_id or self.session_id
        
        try:
            async with self.session.get(
                f"{self.base_url}/api/sessions/{target_session}/history"
            ) as response:
                if response.status == 404:
                    return []
                elif response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error_text}")
                
                result = await response.json()
                return result.get("messages", [])
                
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error: {e}")
    
    async def get_available_agents(self) -> List[Dict[str, Any]]:
        """Get list of available agents."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            async with self.session.get(f"{self.base_url}/api/agents") as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise RuntimeError(f"API error {response.status}: {error_text}")
                
                result = await response.json()
                return result.get("agents", [])
                
        except aiohttp.ClientError as e:
            raise RuntimeError(f"Network error: {e}")
    
    def get_local_history(self) -> List[ChatMessage]:
        """Get the local conversation history."""
        return self.conversation_history.copy()
    
    def clear_local_history(self):
        """Clear the local conversation history."""
        self.conversation_history.clear()
    
    def print_conversation(self, max_messages: Optional[int] = None):
        """Print the conversation history."""
        messages = self.conversation_history
        if max_messages:
            messages = messages[-max_messages:]
        
        print(f"\\nConversation History (Session: {self.session_id})")
        print("=" * 60)
        
        for msg in messages:
            role_display = "You" if msg.role == "user" else f"Bot ({msg.agent})"
            time_str = msg.timestamp.strftime("%H:%M:%S")
            
            print(f"[{time_str}] {role_display}: {msg.content}")
            
            if msg.intent and msg.confidence:
                print(f"  └─ Intent: {msg.intent} (confidence: {msg.confidence:.2f})")
        
        print("=" * 60)


# Example usage functions

async def interactive_demo():
    """Interactive demo of the API client."""
    print("🤖 Chatbot API Client Interactive Demo")
    print("Type 'quit' to exit, 'history' to show conversation history")
    print("=" * 50)
    
    async with ChatbotAPIClient() as client:
        # Check if system is healthy
        try:
            health = await client.get_health_status()
            print(f"✅ Connected to chatbot system (Status: {health['status']})")
        except Exception as e:
            print(f"❌ Failed to connect: {e}")
            return
        
        print(f"Session ID: {client.session_id}\\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit']:
                    break
                elif user_input.lower() == 'history':
                    client.print_conversation()
                    continue
                elif user_input.lower() == 'health':
                    health = await client.get_health_status()
                    print(f"System Status: {health}")
                    continue
                elif not user_input:
                    continue
                
                # Send message
                print("🤔 Processing...", end="", flush=True)
                response = await client.send_message(user_input)
                print("\\r" + " " * 15 + "\\r", end="")  # Clear processing message
                
                # Display response
                print(f"Bot ({response.agent}): {response.content}")
                
                if response.intent:
                    conf_pct = int(response.confidence * 100) if response.confidence else 0
                    print(f"  └─ Intent: {response.intent} ({conf_pct}%)")
                
                print()
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\\n👋 Goodbye!")


async def automated_test():
    """Automated test of the chatbot API."""
    print("🧪 Running Automated API Test")
    print("=" * 30)
    
    test_messages = [
        "Hello there!",
        "What are your business hours?",
        "How much do your products cost?",
        "I'm having trouble logging in, can you help?",
        "Can I schedule a demo?",
        "Thank you for your help!"
    ]
    
    async with ChatbotAPIClient() as client:
        # Check system health
        try:
            health = await client.get_health_status()
            print(f"System Status: {health['status']}")
            
            agents = await client.get_available_agents()
            print(f"Available Agents: {len(agents)}")
            for agent in agents:
                print(f"  - {agent['name']}: {agent['status']}")
            
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return
        
        print(f"\\nTesting with {len(test_messages)} messages...")
        print("=" * 50)
        
        # Send test messages
        for i, message in enumerate(test_messages, 1):
            try:
                print(f"[{i}/{len(test_messages)}] Sending: {message}")
                
                response = await client.send_message(message)
                
                print(f"Response from {response.agent}: {response.content[:100]}...")
                if response.intent:
                    print(f"Intent: {response.intent} ({response.confidence:.2f})")
                
                print()
                
                # Brief pause between messages
                await asyncio.sleep(0.5)
                
            except Exception as e:
                print(f"❌ Error sending message: {e}")
        
        # Show final statistics
        history = client.get_local_history()
        user_messages = [m for m in history if m.role == "user"]
        bot_messages = [m for m in history if m.role == "assistant"]
        
        print("📊 Test Summary:")
        print(f"  Messages sent: {len(user_messages)}")
        print(f"  Responses received: {len(bot_messages)}")
        
        # Count responses by agent
        agent_counts = {}
        for msg in bot_messages:
            agent = msg.agent or "unknown"
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        print("  Responses by agent:")
        for agent, count in agent_counts.items():
            print(f"    {agent}: {count}")
        
        print("\\n✅ Test completed successfully!")


async def webhook_simulation():
    """Simulate integrating the chatbot into an external system via webhooks."""
    print("🔗 Webhook Integration Simulation")
    print("Simulating external system sending messages to chatbot API")
    print("=" * 60)
    
    # Simulate different types of external integrations
    integrations = [
        {
            "name": "Customer Support Ticket",
            "user_context": {"source": "support_ticket", "priority": "high", "customer_id": "12345"},
            "message": "I'm getting a critical error in the application that's preventing me from working"
        },
        {
            "name": "Sales Lead Form",
            "user_context": {"source": "lead_form", "company": "Acme Corp", "interest": "enterprise"},
            "message": "I'm interested in your enterprise plan and would like to schedule a demo"
        },
        {
            "name": "FAQ Chatbot Widget",
            "user_context": {"source": "website_widget", "page": "/pricing"},
            "message": "What are your business hours and how can I contact support?"
        }
    ]
    
    async with ChatbotAPIClient() as client:
        for integration in integrations:
            print(f"\\n📥 {integration['name']}")
            print(f"Message: {integration['message']}")
            print(f"Context: {integration['user_context']}")
            
            try:
                # Use a unique session for each integration
                client.session_id = f"webhook_{integration['name'].lower().replace(' ', '_')}_{uuid.uuid4().hex[:8]}"
                
                response = await client.send_message(
                    integration['message'], 
                    integration['user_context']
                )
                
                print(f"\\n📤 Routed to: {response.agent}")
                print(f"Response: {response.content}")
                
                if response.metadata:
                    if response.metadata.get('ticket_id'):
                        print(f"Support Ticket: {response.metadata['ticket_id']}")
                    if response.metadata.get('recommendations'):
                        print(f"Product Recommendations: {len(response.metadata['recommendations'])}")
                
                print("-" * 40)
                
            except Exception as e:
                print(f"❌ Error processing integration: {e}")
        
        print("\\n✅ Webhook simulation completed!")


def main():
    """Main function with different demo modes."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Chatbot API Client Examples")
    parser.add_argument("--mode", choices=["interactive", "test", "webhook"], default="interactive",
                       help="Demo mode to run")
    parser.add_argument("--base-url", default="http://localhost:8080",
                       help="Base URL of the chatbot API")
    args = parser.parse_args()
    
    # Update the global base URL if specified
    global_base_url = args.base_url
    
    # Override the default in the class
    original_init = ChatbotAPIClient.__init__
    def new_init(self, base_url=global_base_url):
        original_init(self, base_url)
    ChatbotAPIClient.__init__ = new_init
    
    # Run the selected demo
    if args.mode == "interactive":
        asyncio.run(interactive_demo())
    elif args.mode == "test":
        asyncio.run(automated_test())
    elif args.mode == "webhook":
        asyncio.run(webhook_simulation())


if __name__ == "__main__":
    main()