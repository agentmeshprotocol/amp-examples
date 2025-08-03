"""
Knowledge Base Admin Interface

Advanced admin interface for knowledge base curation, analytics, 
and system management.
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Depends, Form, Query
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel
import secrets

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Knowledge Base Admin", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# Basic authentication
security = HTTPBasic()

# Admin credentials (in production, use proper authentication)
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "knowledge_admin_2024"

def authenticate_admin(credentials: HTTPBasicCredentials = Depends(security)):
    """Authenticate admin user"""
    is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# AMP client for admin operations
class AdminClient:
    def __init__(self):
        self.client = AMPClient("admin-interface")
        self.connected = False
    
    async def connect(self):
        """Connect to the AMP system"""
        try:
            await self.client.connect("ws://localhost:8000/ws")
            self.connected = True
            logger.info("Admin connected to AMP system")
        except Exception as e:
            logger.error(f"Failed to connect to AMP system: {e}")
            self.connected = False
    
    async def get_system_health(self) -> Dict:
        """Get comprehensive system health"""
        if not self.connected:
            await self.connect()
        
        health_data = {
            "agents": {},
            "performance": {},
            "errors": [],
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Check each agent
        agents = [
            "query-router-agent",
            "knowledge-ingestion-agent", 
            "semantic-search-agent",
            "knowledge-graph-agent",
            "cache-manager-agent",
            "knowledge-curator-agent"
        ]
        
        for agent_id in agents:
            try:
                response = await asyncio.wait_for(
                    self.client.send_request(agent_id, "health-check", {}),
                    timeout=5.0
                )
                health_data["agents"][agent_id] = {
                    "status": "healthy",
                    "response_time": response.get("response_time", 0),
                    "last_seen": datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_data["agents"][agent_id] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "last_seen": None
                }
                health_data["errors"].append(f"Agent {agent_id}: {str(e)}")
        
        return health_data
    
    async def get_detailed_analytics(self) -> Dict:
        """Get detailed analytics for admin dashboard"""
        try:
            # Get comprehensive analytics from curator
            analytics = await self.client.send_request(
                "knowledge-curator-agent",
                "knowledge-analytics",
                {
                    "content_list": [],  # Would be populated with actual content
                    "report_type": "detailed"
                }
            )
            
            # Get cache performance
            cache_stats = await self.client.send_request(
                "cache-manager-agent",
                "cache-stats",
                {"detailed": True}
            )
            
            # Get routing statistics
            routing_stats = await self.client.send_request(
                "query-router-agent",
                "get-routing-stats",
                {}
            )
            
            return {
                "knowledge_analytics": analytics,
                "cache_performance": cache_stats,
                "routing_performance": routing_stats
            }
        except Exception as e:
            logger.error(f"Error getting analytics: {e}")
            return {"error": str(e)}

# Global admin client
admin_client = AdminClient()

# Pydantic models
class SystemCommand(BaseModel):
    command: str
    parameters: Optional[Dict] = {}

class QualityThreshold(BaseModel):
    minimum_score: float
    action: str  # "warn", "hide", "delete"

# Routes
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    await admin_client.connect()

@app.get("/", response_class=HTMLResponse)
async def admin_dashboard(request: Request, username: str = Depends(authenticate_admin)):
    """Admin dashboard"""
    return templates.TemplateResponse("admin_dashboard.html", {
        "request": request,
        "username": username
    })

@app.get("/api/system/health")
async def api_system_health(username: str = Depends(authenticate_admin)):
    """Get system health status"""
    try:
        health = await admin_client.get_system_health()
        return {"success": True, "health": health}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/detailed")
async def api_detailed_analytics(username: str = Depends(authenticate_admin)):
    """Get detailed analytics"""
    try:
        analytics = await admin_client.get_detailed_analytics()
        return {"success": True, "analytics": analytics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality-management", response_class=HTMLResponse)
async def quality_management(request: Request, username: str = Depends(authenticate_admin)):
    """Quality management page"""
    return templates.TemplateResponse("quality_management.html", {
        "request": request,
        "username": username
    })

@app.post("/api/quality/bulk-analysis")
async def api_bulk_quality_analysis(username: str = Depends(authenticate_admin)):
    """Perform bulk quality analysis"""
    try:
        if not admin_client.connected:
            await admin_client.connect()
        
        # In a real system, this would fetch all content from the database
        sample_content = []  # Would be populated with actual content
        
        result = await admin_client.client.send_request(
            "knowledge-curator-agent",
            "quality-analysis",
            {"batch_content": sample_content}
        )
        
        return {"success": True, "analysis": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/quality/set-thresholds")
async def api_set_quality_thresholds(
    threshold: QualityThreshold,
    username: str = Depends(authenticate_admin)
):
    """Set quality thresholds"""
    try:
        # Store quality thresholds (in real system, would save to database)
        return {
            "success": True,
            "message": f"Quality threshold set to {threshold.minimum_score} with action '{threshold.action}'"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/cache-management", response_class=HTMLResponse)
async def cache_management(request: Request, username: str = Depends(authenticate_admin)):
    """Cache management page"""
    return templates.TemplateResponse("cache_management.html", {
        "request": request,
        "username": username
    })

@app.post("/api/cache/control")
async def api_cache_control(
    command: SystemCommand,
    username: str = Depends(authenticate_admin)
):
    """Control cache operations"""
    try:
        if not admin_client.connected:
            await admin_client.connect()
        
        result = await admin_client.client.send_request(
            "cache-manager-agent",
            "optimize-cache",
            {"operation": command.command, **command.parameters}
        )
        
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agent-management", response_class=HTMLResponse)
async def agent_management(request: Request, username: str = Depends(authenticate_admin)):
    """Agent management page"""
    return templates.TemplateResponse("agent_management.html", {
        "request": request,
        "username": username
    })

@app.post("/api/agents/restart")
async def api_restart_agent(
    agent_id: str = Form(...),
    username: str = Depends(authenticate_admin)
):
    """Restart a specific agent"""
    try:
        # In a real system, this would send restart command to agent manager
        return {
            "success": True,
            "message": f"Restart signal sent to {agent_id}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/data-management", response_class=HTMLResponse)
async def data_management(request: Request, username: str = Depends(authenticate_admin)):
    """Data management page"""
    return templates.TemplateResponse("data_management.html", {
        "request": request,
        "username": username
    })

@app.post("/api/data/backup")
async def api_backup_data(username: str = Depends(authenticate_admin)):
    """Create data backup"""
    try:
        # Create backup (simplified for demo)
        backup_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0",
            "agents": {},
            "indices": {},
            "cache": {}
        }
        
        backup_json = json.dumps(backup_data, indent=2)
        
        return StreamingResponse(
            iter([backup_json.encode()]),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=kb_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/data/cleanup")
async def api_cleanup_data(
    cleanup_type: str = Form(...),
    username: str = Depends(authenticate_admin)
):
    """Cleanup data based on type"""
    try:
        if not admin_client.connected:
            await admin_client.connect()
        
        if cleanup_type == "duplicates":
            result = await admin_client.client.send_request(
                "knowledge-curator-agent",
                "duplicate-detection",
                {"content_list": [], "similarity_threshold": 0.9}
            )
        elif cleanup_type == "low_quality":
            result = await admin_client.client.send_request(
                "knowledge-curator-agent",
                "quality-analysis",
                {"batch_content": []}
            )
        elif cleanup_type == "old_cache":
            result = await admin_client.client.send_request(
                "cache-manager-agent",
                "optimize-cache",
                {"operation": "cleanup"}
            )
        else:
            raise ValueError(f"Unknown cleanup type: {cleanup_type}")
        
        return {"success": True, "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/logs", response_class=HTMLResponse)
async def logs_viewer(request: Request, username: str = Depends(authenticate_admin)):
    """Logs viewer page"""
    return templates.TemplateResponse("logs_viewer.html", {
        "request": request,
        "username": username
    })

@app.get("/api/logs")
async def api_get_logs(
    agent_id: Optional[str] = Query(None),
    level: Optional[str] = Query("INFO"),
    lines: Optional[int] = Query(100),
    username: str = Depends(authenticate_admin)
):
    """Get system logs"""
    try:
        # In a real system, this would read from log files or log aggregation service
        sample_logs = [
            {
                "timestamp": (datetime.utcnow() - timedelta(minutes=i)).isoformat(),
                "level": "INFO" if i % 3 != 0 else "WARNING",
                "agent_id": agent_id or f"agent-{i % 3}",
                "message": f"Sample log message {i}"
            }
            for i in range(lines)
        ]
        
        return {"success": True, "logs": sample_logs}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/settings", response_class=HTMLResponse)
async def system_settings(request: Request, username: str = Depends(authenticate_admin)):
    """System settings page"""
    return templates.TemplateResponse("system_settings.html", {
        "request": request,
        "username": username
    })

@app.post("/api/settings/update")
async def api_update_settings(
    settings: Dict,
    username: str = Depends(authenticate_admin)
):
    """Update system settings"""
    try:
        # In a real system, this would update configuration files or database
        return {
            "success": True,
            "message": "Settings updated successfully",
            "updated_settings": settings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics/export")
async def api_export_metrics(
    format: str = Query("json"),
    period: str = Query("7d"),
    username: str = Depends(authenticate_admin)
):
    """Export metrics data"""
    try:
        # Generate sample metrics data
        metrics = {
            "period": period,
            "generated_at": datetime.utcnow().isoformat(),
            "system_metrics": {
                "query_count": 1250,
                "average_response_time": 0.85,
                "cache_hit_rate": 0.73,
                "error_rate": 0.02
            },
            "agent_metrics": {
                "query-router-agent": {"uptime": 0.99, "requests": 1250},
                "semantic-search-agent": {"uptime": 0.98, "requests": 890},
                "knowledge-graph-agent": {"uptime": 0.97, "requests": 450}
            },
            "quality_metrics": {
                "average_quality_score": 0.78,
                "content_count": 5420,
                "duplicate_ratio": 0.05
            }
        }
        
        if format == "json":
            return {"success": True, "metrics": metrics}
        elif format == "csv":
            # Convert to CSV format
            csv_data = "metric,value\n"
            csv_data += f"query_count,{metrics['system_metrics']['query_count']}\n"
            csv_data += f"avg_response_time,{metrics['system_metrics']['average_response_time']}\n"
            csv_data += f"cache_hit_rate,{metrics['system_metrics']['cache_hit_rate']}\n"
            
            return StreamingResponse(
                iter([csv_data.encode()]),
                media_type="text/csv",
                headers={"Content-Disposition": f"attachment; filename=metrics_{period}.csv"}
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoint for real-time monitoring
@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket, username: str = Depends(authenticate_admin)):
    """WebSocket for real-time monitoring"""
    await websocket.accept()
    
    try:
        while True:
            # Send real-time metrics
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_usage": 45.2,  # Would be real system metrics
                "memory_usage": 62.8,
                "active_queries": 12,
                "cache_size": "2.4GB"
            }
            
            await websocket.send_json(metrics)
            await asyncio.sleep(5)  # Send every 5 seconds
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8081, log_level="info")