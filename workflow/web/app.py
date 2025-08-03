"""
Web interface for AMP Workflow Orchestration.
FastAPI-based dashboard for workflow management and monitoring.
"""

import asyncio
import json
import yaml
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI, WebSocket, HTTPException, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))

from amp_client import AMPClient
from amp_types import AgentIdentity


class WorkflowRequest(BaseModel):
    workflow_id: str
    inputs: Dict[str, Any] = {}


class WorkflowDefinition(BaseModel):
    definition: Dict[str, Any]
    overwrite: bool = False


class WebDashboard:
    """Web dashboard for workflow orchestration management."""
    
    def __init__(self, port: int = 8090):
        self.port = port
        self.app = FastAPI(title="AMP Workflow Dashboard", version="1.0.0")
        self.templates = Jinja2Templates(directory="templates")
        
        # AMP client for backend communication
        self.amp_client = None
        self.workflow_engine_id = "workflow-engine"
        self.monitor_agent_id = "monitor-agent"
        self.state_manager_id = "state-manager"
        
        # WebSocket connections for real-time updates
        self.websocket_connections: List[WebSocket] = []
        
        # Setup routes
        self._setup_routes()
        
        # Mount static files
        self.app.mount("/static", StaticFiles(directory="static"), name="static")
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Main dashboard page."""
            return self.templates.TemplateResponse("dashboard.html", {"request": {}})
        
        @self.app.get("/workflows", response_class=HTMLResponse)
        async def workflows_page():
            """Workflow management page."""
            return self.templates.TemplateResponse("workflows.html", {"request": {}})
        
        @self.app.get("/monitoring", response_class=HTMLResponse)
        async def monitoring_page():
            """Monitoring and metrics page."""
            return self.templates.TemplateResponse("monitoring.html", {"request": {}})
        
        @self.app.get("/designer", response_class=HTMLResponse)
        async def designer_page():
            """Workflow designer page."""
            return self.templates.TemplateResponse("designer.html", {"request": {}})
        
        # API Endpoints
        @self.app.get("/api/dashboard")
        async def get_dashboard_data():
            """Get dashboard overview data."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.monitor_agent_id,
                    capability="dashboard-data",
                    parameters={"widgets": ["overview", "performance", "alerts", "workflows"]}
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]["dashboard"]
                else:
                    raise HTTPException(status_code=500, detail="Failed to get dashboard data")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/workflows")
        async def list_workflows():
            """List all workflow definitions."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-list",
                    parameters={}
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=500, detail="Failed to list workflows")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows")
        async def create_workflow(workflow_def: WorkflowDefinition):
            """Create a new workflow definition."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-create",
                    parameters={
                        "definition": workflow_def.definition,
                        "overwrite": workflow_def.overwrite
                    }
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=400, detail=response.payload.get("error", "Failed to create workflow"))
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows/{workflow_id}/start")
        async def start_workflow(workflow_id: str, request: WorkflowRequest):
            """Start a workflow execution."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-start",
                    parameters={
                        "workflow_id": workflow_id,
                        "inputs": request.inputs
                    }
                )
                
                if response.payload.get("status") == "success":
                    # Notify WebSocket clients
                    await self._broadcast_websocket({
                        "type": "workflow_started",
                        "data": response.payload["result"]
                    })
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=400, detail=response.payload.get("error", "Failed to start workflow"))
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/workflows/{instance_id}/status")
        async def get_workflow_status(instance_id: str):
            """Get workflow execution status."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-status",
                    parameters={"instance_id": instance_id}
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=404, detail="Workflow not found")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows/{instance_id}/pause")
        async def pause_workflow(instance_id: str):
            """Pause a workflow execution."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-pause",
                    parameters={"instance_id": instance_id}
                )
                
                if response.payload.get("status") == "success":
                    await self._broadcast_websocket({
                        "type": "workflow_paused",
                        "data": {"instance_id": instance_id}
                    })
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=400, detail="Failed to pause workflow")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows/{instance_id}/resume")
        async def resume_workflow(instance_id: str):
            """Resume a paused workflow execution."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-resume",
                    parameters={"instance_id": instance_id}
                )
                
                if response.payload.get("status") == "success":
                    await self._broadcast_websocket({
                        "type": "workflow_resumed",
                        "data": {"instance_id": instance_id}
                    })
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=400, detail="Failed to resume workflow")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows/{instance_id}/stop")
        async def stop_workflow(instance_id: str):
            """Stop a workflow execution."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-stop",
                    parameters={"instance_id": instance_id}
                )
                
                if response.payload.get("status") == "success":
                    await self._broadcast_websocket({
                        "type": "workflow_stopped",
                        "data": {"instance_id": instance_id}
                    })
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=400, detail="Failed to stop workflow")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/api/workflows/upload")
        async def upload_workflow(file: UploadFile = File(...)):
            """Upload a workflow definition file."""
            try:
                content = await file.read()
                
                if file.filename.endswith('.yaml') or file.filename.endswith('.yml'):
                    workflow_def = yaml.safe_load(content.decode('utf-8'))
                elif file.filename.endswith('.json'):
                    workflow_def = json.loads(content.decode('utf-8'))
                else:
                    raise HTTPException(status_code=400, detail="Unsupported file format. Use YAML or JSON.")
                
                response = await self.amp_client.send_request(
                    target_agent=self.workflow_engine_id,
                    capability="workflow-create",
                    parameters={
                        "definition": workflow_def,
                        "overwrite": False
                    }
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=400, detail=response.payload.get("error", "Failed to create workflow"))
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics")
        async def get_metrics():
            """Get system metrics."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.monitor_agent_id,
                    capability="metrics-get",
                    parameters={"metric_type": "system"}
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]["metrics"]
                else:
                    raise HTTPException(status_code=500, detail="Failed to get metrics")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/metrics/workflow/{workflow_id}")
        async def get_workflow_metrics(workflow_id: str):
            """Get metrics for a specific workflow."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.monitor_agent_id,
                    capability="metrics-get",
                    parameters={
                        "metric_type": "workflow",
                        "workflow_id": workflow_id
                    }
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]["metrics"]
                else:
                    raise HTTPException(status_code=500, detail="Failed to get workflow metrics")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/alerts")
        async def get_alerts():
            """Get system alerts."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.monitor_agent_id,
                    capability="alert-list",
                    parameters={}
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]
                else:
                    raise HTTPException(status_code=500, detail="Failed to get alerts")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/api/performance/report")
        async def get_performance_report(report_type: str = "summary"):
            """Get performance analysis report."""
            try:
                response = await self.amp_client.send_request(
                    target_agent=self.monitor_agent_id,
                    capability="performance-report",
                    parameters={"report_type": report_type}
                )
                
                if response.payload.get("status") == "success":
                    return response.payload["result"]["report"]
                else:
                    raise HTTPException(status_code=500, detail="Failed to get performance report")
                    
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            """WebSocket endpoint for real-time updates."""
            await websocket.accept()
            self.websocket_connections.append(websocket)
            
            try:
                while True:
                    # Keep connection alive and handle incoming messages
                    data = await websocket.receive_text()
                    # Echo back for now (could handle commands)
                    await websocket.send_text(f"Echo: {data}")
            except Exception as e:
                print(f"WebSocket error: {e}")
            finally:
                if websocket in self.websocket_connections:
                    self.websocket_connections.remove(websocket)
        
        @self.app.on_event("startup")
        async def startup_event():
            """Initialize AMP client on startup."""
            await self._initialize_amp_client()
        
        @self.app.on_event("shutdown")
        async def shutdown_event():
            """Cleanup on shutdown."""
            if self.amp_client:
                await self.amp_client.stop()
    
    async def _initialize_amp_client(self):
        """Initialize AMP client for backend communication."""
        self.amp_client = AMPClient(
            agent_identity=AgentIdentity(
                id="web-dashboard",
                name="Web Dashboard",
                version="1.0.0",
                framework="FastAPI",
                description="Web interface for workflow orchestration"
            ),
            port=8091  # Different port from web server
        )
        
        # Register event handlers for real-time updates
        self.amp_client.register_event_handler("workflow.*", self._handle_workflow_event)
        self.amp_client.register_event_handler("task.*", self._handle_task_event)
        self.amp_client.register_event_handler("error.*", self._handle_error_event)
        
        await self.amp_client.start()
    
    async def _handle_workflow_event(self, message):
        """Handle workflow events for real-time updates."""
        event_data = message.payload
        await self._broadcast_websocket({
            "type": "workflow_event",
            "data": event_data
        })
    
    async def _handle_task_event(self, message):
        """Handle task events for real-time updates."""
        event_data = message.payload
        await self._broadcast_websocket({
            "type": "task_event",
            "data": event_data
        })
    
    async def _handle_error_event(self, message):
        """Handle error events for real-time updates."""
        event_data = message.payload
        await self._broadcast_websocket({
            "type": "error_event",
            "data": event_data
        })
    
    async def _broadcast_websocket(self, message: Dict[str, Any]):
        """Broadcast message to all WebSocket connections."""
        if not self.websocket_connections:
            return
        
        message_text = json.dumps(message)
        disconnected = []
        
        for websocket in self.websocket_connections:
            try:
                await websocket.send_text(message_text)
            except Exception:
                disconnected.append(websocket)
        
        # Remove disconnected websockets
        for websocket in disconnected:
            self.websocket_connections.remove(websocket)
    
    async def start(self):
        """Start the web dashboard server."""
        config = uvicorn.Config(
            app=self.app,
            host="0.0.0.0",
            port=self.port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()


# Standalone script runner
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AMP Workflow Dashboard")
    parser.add_argument("--port", type=int, default=8090, help="Port to run the dashboard on")
    args = parser.parse_args()
    
    dashboard = WebDashboard(port=args.port)
    
    try:
        asyncio.run(dashboard.start())
    except KeyboardInterrupt:
        print("Dashboard stopped by user")