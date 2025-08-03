"""
FastAPI Web Application for the Support System

Provides a web interface for ticket submission, tracking, and support
team management dashboard.
"""

import os
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, Depends, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
import uvicorn

# Add support system to path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from support_types import (
    Ticket, TicketStatus, TicketPriority, TicketCategory,
    CustomerInfo, SLALevel
)


# Pydantic models for API
class TicketCreateRequest(BaseModel):
    subject: str
    description: str
    category: str = "general"
    priority: str = "medium"
    customer_name: str
    customer_email: str
    customer_phone: Optional[str] = None
    sla_level: str = "basic"


class TicketResponse(BaseModel):
    id: str
    subject: str
    description: str
    status: str
    priority: str
    category: str
    customer_name: str
    customer_email: str
    created_at: str
    updated_at: str


class SystemHealthResponse(BaseModel):
    status: str
    healthy_agents: int
    total_agents: int
    health_percentage: float
    agents: List[Dict[str, Any]]


def create_app(orchestrator=None):
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title="AMP Customer Support System",
        description="Intelligent customer support system powered by AMP protocol",
        version="1.0.0"
    )
    
    # Configure templates and static files
    templates = Jinja2Templates(directory="web/templates")
    app.mount("/static", StaticFiles(directory="web/static"), name="static")
    
    # In-memory ticket storage (in production, use a real database)
    tickets_db = {}
    
    # Store orchestrator reference
    app.state.orchestrator = orchestrator
    
    @app.get("/", response_class=HTMLResponse)
    async def home(request: Request):
        """Home page with system overview."""
        # Get system health if orchestrator is available
        system_health = None
        if app.state.orchestrator:
            try:
                health = await app.state.orchestrator.health_check()
                system_health = {
                    "status": health["status"],
                    "healthy_agents": health["healthy_agents"],
                    "total_agents": health["total_agents"],
                    "health_percentage": health["health_percentage"]
                }
            except Exception as e:
                system_health = {"status": "error", "message": str(e)}
        
        # Get recent tickets
        recent_tickets = list(tickets_db.values())[-5:] if tickets_db else []
        
        return templates.TemplateResponse("index.html", {
            "request": request,
            "system_health": system_health,
            "recent_tickets": recent_tickets,
            "total_tickets": len(tickets_db),
            "open_tickets": len([t for t in tickets_db.values() if t.status != TicketStatus.CLOSED])
        })
    
    @app.get("/submit", response_class=HTMLResponse)
    async def submit_ticket_form(request: Request):
        """Show ticket submission form."""
        return templates.TemplateResponse("submit_ticket.html", {
            "request": request,
            "categories": [cat.value for cat in TicketCategory],
            "priorities": [pri.value for pri in TicketPriority],
            "sla_levels": [sla.value for sla in SLALevel]
        })
    
    @app.post("/submit", response_class=HTMLResponse)
    async def submit_ticket(
        request: Request,
        subject: str = Form(...),
        description: str = Form(...),
        category: str = Form(...),
        priority: str = Form(...),
        customer_name: str = Form(...),
        customer_email: str = Form(...),
        customer_phone: Optional[str] = Form(None),
        sla_level: str = Form("basic")
    ):
        """Submit a new support ticket."""
        try:
            # Create customer info
            customer = CustomerInfo(
                id=str(uuid.uuid4()),
                name=customer_name,
                email=customer_email,
                phone=customer_phone,
                sla_level=SLALevel(sla_level)
            )
            
            # Create ticket
            ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}"
            ticket = Ticket(
                id=ticket_id,
                subject=subject,
                description=description,
                customer=customer,
                status=TicketStatus.OPEN,
                priority=TicketPriority(priority),
                category=TicketCategory(category)
            )
            
            # Store ticket
            tickets_db[ticket_id] = ticket
            
            # Add initial system comment
            ticket.add_comment(
                "system",
                "system",
                f"Ticket created via web interface. Priority: {priority}, Category: {category}"
            )
            
            # If orchestrator is available, trigger classification
            if app.state.orchestrator:
                try:
                    # Find classifier agent
                    classifier_agent = None
                    for agent_info in app.state.orchestrator.agents:
                        if "classifier" in agent_info["config"].agent_id:
                            classifier_agent = agent_info["agent"]
                            break
                    
                    if classifier_agent:
                        # Trigger classification (fire and forget)
                        classification_params = {
                            "ticket": {
                                "id": ticket.id,
                                "subject": ticket.subject,
                                "description": ticket.description,
                                "customer_info": {
                                    "id": customer.id,
                                    "sla_level": customer.sla_level.value,
                                    "account_type": "standard"
                                }
                            }
                        }
                        
                        # This would normally be done asynchronously
                        # For demo purposes, we'll just log it
                        pass
                
                except Exception as e:
                    # Don't fail ticket creation if classification fails
                    print(f"Classification error: {e}")
            
            return templates.TemplateResponse("ticket_submitted.html", {
                "request": request,
                "ticket": ticket,
                "success": True
            })
            
        except Exception as e:
            return templates.TemplateResponse("submit_ticket.html", {
                "request": request,
                "error": f"Failed to submit ticket: {str(e)}",
                "categories": [cat.value for cat in TicketCategory],
                "priorities": [pri.value for pri in TicketPriority],
                "sla_levels": [sla.value for sla in SLALevel]
            })
    
    @app.get("/tickets", response_class=HTMLResponse)
    async def list_tickets(request: Request, status: Optional[str] = None):
        """List all tickets with optional status filter."""
        tickets = list(tickets_db.values())
        
        if status:
            try:
                status_filter = TicketStatus(status)
                tickets = [t for t in tickets if t.status == status_filter]
            except ValueError:
                pass
        
        # Sort by creation date (newest first)
        tickets.sort(key=lambda x: x.created_at, reverse=True)
        
        return templates.TemplateResponse("tickets_list.html", {
            "request": request,
            "tickets": tickets,
            "status_filter": status,
            "statuses": [s.value for s in TicketStatus]
        })
    
    @app.get("/ticket/{ticket_id}", response_class=HTMLResponse)
    async def view_ticket(request: Request, ticket_id: str):
        """View a specific ticket."""
        ticket = tickets_db.get(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Get SLA information
        sla_target = ticket.get_sla_target()
        sla_breaches = ticket.is_sla_breached()
        
        return templates.TemplateResponse("ticket_detail.html", {
            "request": request,
            "ticket": ticket,
            "sla_target": sla_target,
            "sla_breaches": sla_breaches
        })
    
    @app.post("/ticket/{ticket_id}/comment")
    async def add_comment(
        ticket_id: str,
        content: str = Form(...),
        author_name: str = Form(...),
        is_internal: bool = Form(False)
    ):
        """Add a comment to a ticket."""
        ticket = tickets_db.get(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        # Add comment
        ticket.add_comment(
            author_id=author_name,
            author_type="agent" if is_internal else "customer",
            content=content,
            is_internal=is_internal
        )
        
        return RedirectResponse(url=f"/ticket/{ticket_id}", status_code=303)
    
    @app.get("/dashboard", response_class=HTMLResponse)
    async def dashboard(request: Request):
        """Support team dashboard."""
        
        # Calculate metrics
        total_tickets = len(tickets_db)
        open_tickets = len([t for t in tickets_db.values() if t.status != TicketStatus.CLOSED])
        critical_tickets = len([t for t in tickets_db.values() if t.priority == TicketPriority.CRITICAL])
        
        # SLA compliance
        sla_breached = 0
        for ticket in tickets_db.values():
            breaches = ticket.is_sla_breached()
            if any(breaches.values()):
                sla_breached += 1
        
        sla_compliance = ((total_tickets - sla_breached) / total_tickets * 100) if total_tickets > 0 else 100
        
        # Recent activity
        recent_tickets = sorted(tickets_db.values(), key=lambda x: x.updated_at, reverse=True)[:10]
        
        # System health
        system_health = None
        agent_status = []
        if app.state.orchestrator:
            try:
                health = await app.state.orchestrator.health_check()
                system_health = health
                
                # Get individual agent status
                for agent_info in app.state.orchestrator.agents:
                    agent_status.append({
                        "name": agent_info["config"].agent_name,
                        "id": agent_info["config"].agent_id,
                        "framework": agent_info["config"].framework,
                        "status": "healthy"  # Simplified for demo
                    })
            except Exception as e:
                system_health = {"status": "error", "message": str(e)}
        
        return templates.TemplateResponse("dashboard.html", {
            "request": request,
            "metrics": {
                "total_tickets": total_tickets,
                "open_tickets": open_tickets,
                "critical_tickets": critical_tickets,
                "sla_compliance": round(sla_compliance, 1)
            },
            "recent_tickets": recent_tickets,
            "system_health": system_health,
            "agent_status": agent_status
        })
    
    # API endpoints
    @app.get("/api/health")
    async def api_health():
        """API endpoint for system health check."""
        if not app.state.orchestrator:
            return JSONResponse({
                "status": "degraded",
                "message": "Orchestrator not available",
                "agents": []
            })
        
        try:
            health = await app.state.orchestrator.health_check()
            
            agents = []
            for agent_info in app.state.orchestrator.agents:
                agents.append({
                    "id": agent_info["config"].agent_id,
                    "name": agent_info["config"].agent_name,
                    "framework": agent_info["config"].framework,
                    "status": "healthy"  # Simplified
                })
            
            return SystemHealthResponse(
                status=health["status"],
                healthy_agents=health["healthy_agents"],
                total_agents=health["total_agents"],
                health_percentage=health["health_percentage"],
                agents=agents
            )
            
        except Exception as e:
            return JSONResponse({
                "status": "error",
                "message": str(e),
                "agents": []
            }, status_code=500)
    
    @app.get("/api/tickets", response_model=List[TicketResponse])
    async def api_list_tickets(status: Optional[str] = None):
        """API endpoint to list tickets."""
        tickets = list(tickets_db.values())
        
        if status:
            try:
                status_filter = TicketStatus(status)
                tickets = [t for t in tickets if t.status == status_filter]
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status")
        
        return [
            TicketResponse(
                id=ticket.id,
                subject=ticket.subject,
                description=ticket.description,
                status=ticket.status.value,
                priority=ticket.priority.value,
                category=ticket.category.value,
                customer_name=ticket.customer.name,
                customer_email=ticket.customer.email,
                created_at=ticket.created_at.isoformat(),
                updated_at=ticket.updated_at.isoformat()
            )
            for ticket in tickets
        ]
    
    @app.post("/api/tickets", response_model=TicketResponse)
    async def api_create_ticket(ticket_request: TicketCreateRequest):
        """API endpoint to create a ticket."""
        try:
            # Create customer info
            customer = CustomerInfo(
                id=str(uuid.uuid4()),
                name=ticket_request.customer_name,
                email=ticket_request.customer_email,
                phone=ticket_request.customer_phone,
                sla_level=SLALevel(ticket_request.sla_level)
            )
            
            # Create ticket
            ticket_id = f"TKT-{datetime.now().strftime('%Y%m%d%H%M%S')}-API"
            ticket = Ticket(
                id=ticket_id,
                subject=ticket_request.subject,
                description=ticket_request.description,
                customer=customer,
                status=TicketStatus.OPEN,
                priority=TicketPriority(ticket_request.priority),
                category=TicketCategory(ticket_request.category)
            )
            
            # Store ticket
            tickets_db[ticket_id] = ticket
            
            return TicketResponse(
                id=ticket.id,
                subject=ticket.subject,
                description=ticket.description,
                status=ticket.status.value,
                priority=ticket.priority.value,
                category=ticket.category.value,
                customer_name=ticket.customer.name,
                customer_email=ticket.customer.email,
                created_at=ticket.created_at.isoformat(),
                updated_at=ticket.updated_at.isoformat()
            )
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/api/tickets/{ticket_id}")
    async def api_get_ticket(ticket_id: str):
        """API endpoint to get a specific ticket."""
        ticket = tickets_db.get(ticket_id)
        if not ticket:
            raise HTTPException(status_code=404, detail="Ticket not found")
        
        return ticket.to_dict()
    
    return app


async def run_web_server(app: FastAPI, host: str = "0.0.0.0", port: int = 8080):
    """Run the web server."""
    config = uvicorn.Config(
        app=app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    # For testing the web app independently
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8080)