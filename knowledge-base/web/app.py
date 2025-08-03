"""
Knowledge Base Web Interface

FastAPI web application for interacting with the knowledge base system
through a user-friendly web interface.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(title="Knowledge Base System", version="1.0.0")

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates
templates = Jinja2Templates(directory="templates")

# AMP client for communicating with agents
class KnowledgeBaseClient:
    def __init__(self):
        self.router_client = AMPClient("web-interface")
        self.connected = False
    
    async def connect(self):
        """Connect to the AMP system"""
        try:
            await self.router_client.connect("ws://localhost:8000/ws")
            self.connected = True
            logger.info("Connected to AMP system")
        except Exception as e:
            logger.error(f"Failed to connect to AMP system: {e}")
            self.connected = False
    
    async def search(self, query: str, search_type: str = "hybrid") -> Dict:
        """Perform search through query router"""
        if not self.connected:
            await self.connect()
        
        try:
            response = await self.router_client.send_request(
                "query-router-agent",
                "route-query",
                {
                    "query": query,
                    "routing_options": {
                        "search_type": search_type
                    }
                }
            )
            return response
        except Exception as e:
            logger.error(f"Search error: {e}")
            return {"error": str(e)}
    
    async def ingest_document(self, file_path: str, metadata: Dict = None) -> Dict:
        """Ingest a document"""
        if not self.connected:
            await self.connect()
        
        try:
            response = await self.router_client.send_request(
                "knowledge-ingestion-agent",
                "document-ingestion",
                {
                    "file_path": file_path,
                    "metadata": metadata or {}
                }
            )
            return response
        except Exception as e:
            logger.error(f"Ingestion error: {e}")
            return {"error": str(e)}
    
    async def get_analytics(self) -> Dict:
        """Get knowledge base analytics"""
        if not self.connected:
            await self.connect()
        
        try:
            # Get sample content for analytics (in real system, this would be from a database)
            response = await self.router_client.send_request(
                "knowledge-curator-agent",
                "knowledge-analytics",
                {
                    "content_list": [],  # Would be populated with actual content
                    "report_type": "overview"
                }
            )
            return response
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {"error": str(e)}

# Global client instance
kb_client = KnowledgeBaseClient()

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    search_type: Optional[str] = "hybrid"
    filters: Optional[Dict] = {}

class DocumentUpload(BaseModel):
    title: str
    content: Optional[str] = None
    metadata: Optional[Dict] = {}

# Routes
@app.on_event("startup")
async def startup_event():
    """Initialize connections on startup"""
    await kb_client.connect()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Home page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """Search page"""
    return templates.TemplateResponse("search.html", {"request": request})

@app.post("/api/search")
async def api_search(search_request: SearchRequest):
    """API endpoint for search"""
    try:
        results = await kb_client.search(
            search_request.query,
            search_request.search_type
        )
        
        return {
            "success": True,
            "results": results.get("results", []),
            "routing_info": results.get("routing_info", {}),
            "performance_stats": results.get("performance_stats", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Document upload page"""
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/api/upload")
async def api_upload(
    title: str = Form(...),
    file: UploadFile = File(None),
    content: str = Form(None),
    author: str = Form(""),
    tags: str = Form("")
):
    """API endpoint for document upload"""
    try:
        if file:
            # Save uploaded file
            upload_dir = "./data/uploads"
            os.makedirs(upload_dir, exist_ok=True)
            
            file_path = os.path.join(upload_dir, file.filename)
            with open(file_path, "wb") as f:
                content_bytes = await file.read()
                f.write(content_bytes)
            
            # Prepare metadata
            metadata = {
                "title": title,
                "author": author,
                "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                "uploaded_at": datetime.utcnow().isoformat(),
                "original_filename": file.filename,
                "mime_type": file.content_type
            }
            
            # Ingest document
            result = await kb_client.ingest_document(file_path, metadata)
            
        elif content:
            # Process text content
            result = await kb_client.router_client.send_request(
                "knowledge-ingestion-agent",
                "text-processing",
                {
                    "text": content,
                    "metadata": {
                        "title": title,
                        "author": author,
                        "tags": [tag.strip() for tag in tags.split(",") if tag.strip()],
                        "uploaded_at": datetime.utcnow().isoformat()
                    }
                }
            )
        else:
            raise HTTPException(status_code=400, detail="Either file or content must be provided")
        
        return {
            "success": True,
            "document_id": result.get("document_id"),
            "message": "Document uploaded and processed successfully"
        }
        
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analytics", response_class=HTMLResponse)
async def analytics_page(request: Request):
    """Analytics page"""
    return templates.TemplateResponse("analytics.html", {"request": request})

@app.get("/api/analytics")
async def api_analytics():
    """API endpoint for analytics"""
    try:
        analytics = await kb_client.get_analytics()
        return {
            "success": True,
            "analytics": analytics
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/graph", response_class=HTMLResponse)
async def graph_page(request: Request):
    """Knowledge graph visualization page"""
    return templates.TemplateResponse("graph.html", {"request": request})

@app.get("/api/graph/search")
async def api_graph_search(query: str = "", entity_type: str = ""):
    """API endpoint for graph search"""
    try:
        if not kb_client.connected:
            await kb_client.connect()
        
        response = await kb_client.router_client.send_request(
            "knowledge-graph-agent",
            "graph-search",
            {
                "query": query,
                "entity_type": entity_type if entity_type else None,
                "max_results": 20
            }
        )
        
        return {
            "success": True,
            "entities": response.get("results", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/related/{entity_id}")
async def api_graph_related(entity_id: str, max_depth: int = 2):
    """API endpoint for finding related entities"""
    try:
        if not kb_client.connected:
            await kb_client.connect()
        
        response = await kb_client.router_client.send_request(
            "knowledge-graph-agent",
            "find-related",
            {
                "entity_id": entity_id,
                "max_depth": max_depth
            }
        )
        
        return {
            "success": True,
            "related_entities": response.get("related_entities", [])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cache/stats")
async def api_cache_stats():
    """API endpoint for cache statistics"""
    try:
        if not kb_client.connected:
            await kb_client.connect()
        
        response = await kb_client.router_client.send_request(
            "cache-manager-agent",
            "cache-stats",
            {"detailed": True}
        )
        
        return {
            "success": True,
            "stats": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/optimize")
async def api_cache_optimize(operation: str = "cleanup"):
    """API endpoint for cache optimization"""
    try:
        if not kb_client.connected:
            await kb_client.connect()
        
        response = await kb_client.router_client.send_request(
            "cache-manager-agent",
            "optimize-cache",
            {"operation": operation}
        )
        
        return {
            "success": True,
            "result": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/quality", response_class=HTMLResponse)
async def quality_page(request: Request):
    """Quality management page"""
    return templates.TemplateResponse("quality.html", {"request": request})

@app.get("/api/quality/duplicates")
async def api_quality_duplicates(threshold: float = 0.85):
    """API endpoint for duplicate detection"""
    try:
        if not kb_client.connected:
            await kb_client.connect()
        
        # In a real system, this would fetch actual content from the database
        sample_content = []  # Would be populated with actual content
        
        response = await kb_client.router_client.send_request(
            "knowledge-curator-agent",
            "duplicate-detection",
            {
                "content_list": sample_content,
                "similarity_threshold": threshold
            }
        )
        
        return {
            "success": True,
            "duplicates": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/system/status")
async def api_system_status():
    """API endpoint for system status"""
    try:
        # Check connectivity to each agent
        agents_status = {}
        
        if not kb_client.connected:
            await kb_client.connect()
        
        # Test each agent
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
                # Send a simple ping or status request
                response = await asyncio.wait_for(
                    kb_client.router_client.send_request(agent_id, "ping", {}),
                    timeout=5.0
                )
                agents_status[agent_id] = "online"
            except:
                agents_status[agent_id] = "offline"
        
        return {
            "success": True,
            "system_status": "operational" if all(status == "online" for status in agents_status.values()) else "degraded",
            "agents": agents_status,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "system_status": "error"
        }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("404.html", {"request": request}, status_code=404)

@app.exception_handler(500)
async def server_error_handler(request: Request, exc: HTTPException):
    return templates.TemplateResponse("500.html", {"request": request}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")