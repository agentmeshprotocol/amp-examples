#!/usr/bin/env python3
"""
Web Interface for Multi-Agent Chatbot

A simple web interface using FastAPI for interacting with the chatbot system.
Provides REST API endpoints and a basic HTML chat interface.
"""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directories to path
import sys
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from run_chatbot import ChatbotSystem


# Pydantic models for API
class ChatMessage(BaseModel):
    message: str
    session_id: Optional[str] = None
    user_context: Optional[Dict[str, Any]] = None


class ChatResponse(BaseModel):
    response: str
    agent: str
    session_id: str
    intent: Optional[str] = None
    confidence: Optional[float] = None
    timestamp: str
    metadata: Optional[Dict[str, Any]] = None


class HealthStatus(BaseModel):
    status: str
    agents: Dict[str, Any]
    timestamp: str


# Global chatbot system instance
chatbot_system: Optional[ChatbotSystem] = None
active_connections: List[WebSocket] = []

# FastAPI app
app = FastAPI(
    title="Multi-Agent Chatbot API",
    description="REST API and WebSocket interface for the multi-agent chatbot system",
    version="1.0.0"
)

# Mount static files if they exist
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.on_event("startup")
async def startup_event():
    """Initialize the chatbot system on startup."""
    global chatbot_system
    
    print("🚀 Starting Multi-Agent Chatbot Web Interface...")
    
    try:
        chatbot_system = ChatbotSystem()
        
        # Start the chatbot system in background
        asyncio.create_task(chatbot_system.start())
        
        # Wait a bit for system to initialize
        await asyncio.sleep(3)
        
        if not chatbot_system.running:
            raise RuntimeError("Failed to start chatbot system")
            
        print("✅ Chatbot system is ready!")
        
    except Exception as e:
        print(f"❌ Failed to start chatbot system: {e}")
        chatbot_system = None


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global chatbot_system
    
    if chatbot_system and chatbot_system.running:
        print("🛑 Shutting down chatbot system...")
        await chatbot_system.shutdown()
        print("✅ Chatbot system shutdown complete")


# REST API Endpoints

@app.get("/", response_class=HTMLResponse)
async def get_chat_interface():
    """Serve the main chat interface."""
    return HTMLResponse(content=get_chat_html(), status_code=200)


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(message: ChatMessage):
    """Send a message to the chatbot."""
    global chatbot_system
    
    if not chatbot_system or not chatbot_system.running:
        raise HTTPException(status_code=503, detail="Chatbot system not available")
    
    # Generate session ID if not provided
    session_id = message.session_id or str(uuid.uuid4())
    
    try:
        # Process message through chatbot system
        result = await chatbot_system.process_user_message(
            message.message, session_id, message.user_context
        )
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        # Format response
        response = ChatResponse(
            response=result.get("response", "No response generated"),
            agent=result.get("agent", "system"),
            session_id=session_id,
            intent=result.get("intent"),
            confidence=result.get("confidence"),
            timestamp=datetime.now().isoformat(),
            metadata={
                "ticket_id": result.get("ticket_id"),
                "recommendations": result.get("recommendations"),
                "escalation": result.get("escalation"),
                "next_steps": result.get("next_steps")
            }
        )
        
        # Broadcast to WebSocket connections
        await broadcast_message({
            "type": "chat_response",
            "session_id": session_id,
            "user_message": message.message,
            "bot_response": response.dict()
        })
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthStatus)
async def health_check():
    """Get system health status."""
    global chatbot_system
    
    if not chatbot_system:
        return HealthStatus(
            status="unavailable",
            agents={},
            timestamp=datetime.now().isoformat()
        )
    
    try:
        health = await chatbot_system.health_check()
        return HealthStatus(
            status=health.get("system_status", "unknown"),
            agents=health.get("agents", {}),
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        return HealthStatus(
            status="error",
            agents={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )


@app.get("/api/sessions/{session_id}/history")
async def get_session_history(session_id: str):
    """Get conversation history for a session."""
    global chatbot_system
    
    if not chatbot_system or not chatbot_system.conversation_manager:
        raise HTTPException(status_code=503, detail="Conversation manager not available")
    
    try:
        history = await chatbot_system.conversation_manager.get_conversation_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(history),
            "messages": [
                {
                    "id": msg.id,
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "agent": msg.agent,
                    "intent": msg.intent
                }
                for msg in history
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/agents")
async def get_agents():
    """Get list of available agents."""
    global chatbot_system
    
    if not chatbot_system:
        raise HTTPException(status_code=503, detail="Chatbot system not available")
    
    try:
        health = await chatbot_system.health_check()
        agents = []
        
        for agent_name, agent_health in health.get("agents", {}).items():
            agents.append({
                "name": agent_name,
                "status": agent_health.get("status", "unknown"),
                "capabilities": agent_health.get("capabilities", []),
                "connected": agent_health.get("connected", False)
            })
        
        return {"agents": agents, "total": len(agents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# WebSocket endpoint for real-time chat

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat."""
    await websocket.accept()
    active_connections.append(websocket)
    
    try:
        await websocket.send_json({
            "type": "connection_established",
            "session_id": session_id,
            "message": "Connected to chatbot"
        })
        
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            if data.get("type") == "chat_message":
                user_message = data.get("message", "")
                user_context = data.get("context", {})
                
                # Process through chatbot
                if chatbot_system and chatbot_system.running:
                    try:
                        result = await chatbot_system.process_user_message(
                            user_message, session_id, user_context
                        )
                        
                        # Send response back
                        await websocket.send_json({
                            "type": "chat_response",
                            "session_id": session_id,
                            "user_message": user_message,
                            "bot_response": {
                                "response": result.get("response", "No response"),
                                "agent": result.get("agent", "system"),
                                "intent": result.get("intent"),
                                "confidence": result.get("confidence"),
                                "timestamp": datetime.now().isoformat(),
                                "metadata": result
                            }
                        })
                        
                    except Exception as e:
                        await websocket.send_json({
                            "type": "error",
                            "error": str(e),
                            "message": "Sorry, I encountered an error processing your message."
                        })
                else:
                    await websocket.send_json({
                        "type": "error",
                        "error": "System unavailable",
                        "message": "The chatbot system is currently unavailable."
                    })
            
            elif data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
                
    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


async def broadcast_message(message: Dict[str, Any]):
    """Broadcast message to all active WebSocket connections."""
    if not active_connections:
        return
    
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            disconnected.append(connection)
    
    # Remove disconnected connections
    for connection in disconnected:
        active_connections.remove(connection)


def get_chat_html() -> str:
    """Generate the HTML chat interface."""
    return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multi-Agent Chatbot</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            height: 600px;
            display: flex;
            flex-direction: column;
        }
        .header {
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px 10px 0 0;
        }
        .header h1 {
            margin: 0;
            font-size: 24px;
        }
        .header p {
            margin: 5px 0 0 0;
            opacity: 0.9;
        }
        .chat-container {
            flex: 1;
            display: flex;
            flex-direction: column;
            padding: 20px;
        }
        .messages {
            flex: 1;
            overflow-y: auto;
            border: 1px solid #e0e0e0;
            border-radius: 5px;
            padding: 15px;
            margin-bottom: 15px;
            background: #fafafa;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
            max-width: 80%;
        }
        .user-message {
            background: #e3f2fd;
            margin-left: auto;
            text-align: right;
        }
        .bot-message {
            background: #f1f8e9;
            margin-right: auto;
        }
        .message-meta {
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }
        .input-container {
            display: flex;
            gap: 10px;
        }
        .message-input {
            flex: 1;
            padding: 12px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 16px;
        }
        .send-button {
            padding: 12px 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        .send-button:hover {
            background: #5a6fd8;
        }
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        .status {
            text-align: center;
            color: #666;
            font-style: italic;
            padding: 10px;
        }
        .connection-status {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        .connected { background: #4caf50; }
        .disconnected { background: #f44336; }
        .typing-indicator {
            background: #e0e0e0;
            padding: 10px;
            border-radius: 8px;
            margin: 10px 0;
            font-style: italic;
            color: #666;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Multi-Agent Chatbot</h1>
            <p>
                <span id="connection-status" class="connection-status disconnected"></span>
                <span id="status-text">Connecting...</span>
            </p>
        </div>
        
        <div class="chat-container">
            <div id="messages" class="messages">
                <div class="status">Welcome! Ask me anything about our products, services, or if you need technical support.</div>
            </div>
            
            <div class="input-container">
                <input 
                    type="text" 
                    id="messageInput" 
                    class="message-input" 
                    placeholder="Type your message here..." 
                    disabled
                >
                <button id="sendButton" class="send-button" disabled>Send</button>
            </div>
        </div>
    </div>

    <script>
        class ChatInterface {
            constructor() {
                this.sessionId = this.generateSessionId();
                this.websocket = null;
                this.messageInput = document.getElementById('messageInput');
                this.sendButton = document.getElementById('sendButton');
                this.messagesContainer = document.getElementById('messages');
                this.connectionStatus = document.getElementById('connection-status');
                this.statusText = document.getElementById('status-text');
                
                this.setupEventListeners();
                this.connect();
            }
            
            generateSessionId() {
                return 'web_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
            }
            
            setupEventListeners() {
                this.sendButton.addEventListener('click', () => this.sendMessage());
                
                this.messageInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        this.sendMessage();
                    }
                });
            }
            
            connect() {
                const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                const wsUrl = `${protocol}//${window.location.host}/ws/${this.sessionId}`;
                
                this.websocket = new WebSocket(wsUrl);
                
                this.websocket.onopen = () => {
                    this.updateConnectionStatus(true);
                    this.messageInput.disabled = false;
                    this.sendButton.disabled = false;
                    this.messageInput.focus();
                };
                
                this.websocket.onmessage = (event) => {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                };
                
                this.websocket.onclose = () => {
                    this.updateConnectionStatus(false);
                    this.messageInput.disabled = true;
                    this.sendButton.disabled = true;
                    
                    // Try to reconnect after 3 seconds
                    setTimeout(() => this.connect(), 3000);
                };
                
                this.websocket.onerror = (error) => {
                    console.error('WebSocket error:', error);
                    this.updateConnectionStatus(false);
                };
            }
            
            updateConnectionStatus(connected) {
                if (connected) {
                    this.connectionStatus.className = 'connection-status connected';
                    this.statusText.textContent = 'Connected';
                } else {
                    this.connectionStatus.className = 'connection-status disconnected';
                    this.statusText.textContent = 'Disconnected';
                }
            }
            
            handleMessage(data) {
                switch (data.type) {
                    case 'connection_established':
                        this.addSystemMessage('Connected to chatbot system');
                        break;
                        
                    case 'chat_response':
                        this.removeTypingIndicator();
                        this.addBotMessage(data.bot_response);
                        break;
                        
                    case 'error':
                        this.removeTypingIndicator();
                        this.addSystemMessage(`Error: ${data.message}`);
                        break;
                }
            }
            
            sendMessage() {
                const message = this.messageInput.value.trim();
                if (!message || !this.websocket) return;
                
                // Add user message to chat
                this.addUserMessage(message);
                
                // Show typing indicator
                this.addTypingIndicator();
                
                // Send to websocket
                this.websocket.send(JSON.stringify({
                    type: 'chat_message',
                    message: message,
                    context: {}
                }));
                
                // Clear input
                this.messageInput.value = '';
            }
            
            addUserMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message user-message';
                messageDiv.innerHTML = `
                    <div>${this.escapeHtml(message)}</div>
                    <div class="message-meta">You • ${this.formatTime(new Date())}</div>
                `;
                this.messagesContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            addBotMessage(response) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot-message';
                
                let metaInfo = `${response.agent} • ${this.formatTime(new Date(response.timestamp))}`;
                if (response.intent) {
                    metaInfo += ` • Intent: ${response.intent}`;
                    if (response.confidence) {
                        metaInfo += ` (${(response.confidence * 100).toFixed(0)}%)`;
                    }
                }
                
                messageDiv.innerHTML = `
                    <div>${this.escapeHtml(response.response)}</div>
                    <div class="message-meta">${metaInfo}</div>
                `;
                
                this.messagesContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            addSystemMessage(message) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'status';
                messageDiv.textContent = message;
                this.messagesContainer.appendChild(messageDiv);
                this.scrollToBottom();
            }
            
            addTypingIndicator() {
                const indicator = document.createElement('div');
                indicator.className = 'typing-indicator';
                indicator.id = 'typing-indicator';
                indicator.textContent = 'Bot is typing...';
                this.messagesContainer.appendChild(indicator);
                this.scrollToBottom();
            }
            
            removeTypingIndicator() {
                const indicator = document.getElementById('typing-indicator');
                if (indicator) {
                    indicator.remove();
                }
            }
            
            scrollToBottom() {
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }
            
            formatTime(date) {
                return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
            }
            
            escapeHtml(text) {
                const div = document.createElement('div');
                div.textContent = text;
                return div.innerHTML;
            }
        }
        
        // Initialize chat interface when page loads
        document.addEventListener('DOMContentLoaded', () => {
            new ChatInterface();
        });
    </script>
</body>
</html>
'''


def main():
    """Run the web interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Multi-Agent Chatbot Web Interface")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()
    
    print(f"🌐 Starting web interface at http://{args.host}:{args.port}")
    print("📖 API documentation available at http://localhost:8080/docs")
    
    uvicorn.run(
        "web_interface:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()