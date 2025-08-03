#!/usr/bin/env python3
"""
Research Assistant Network - Main Application Runner

Starts and coordinates all research assistant agents and crews.
Provides AMP protocol integration and workflow management.
"""

import asyncio
import logging
import signal
import sys
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import yaml
import json
from datetime import datetime

# Add shared-lib to path
sys.path.append(str(Path(__file__).parent.parent / "shared-lib"))

# Import agents
from agents.web_search_agent import WebSearchAgent
from agents.content_analyzer import ContentAnalyzer
from agents.fact_checker import FactChecker
from agents.synthesis_agent import SynthesisAgent
from agents.research_orchestrator import ResearchOrchestrator, ResearchQuery

# Import crews
from crews.research_crew import ResearchCrew
from crews.fact_check_crew import FactCheckCrew
from crews.content_creation_crew import ContentCreationCrew

# AMP imports
from amp_client import AMPClient
from amp_types import TransportType
from amp_registry import AMPRegistry


class ResearchAssistantSystem:
    """Main research assistant system coordinator."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config"
        self.config = self._load_configuration()
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(f"{__name__}.ResearchAssistantSystem")
        
        # Initialize components
        self.agents: Dict[str, Any] = {}
        self.crews: Dict[str, Any] = {}
        self.amp_clients: List[AMPClient] = []
        self.orchestrator: Optional[ResearchOrchestrator] = None
        
        # System state
        self.running = False
        self.startup_time: Optional[datetime] = None
        
        # Performance metrics
        self.metrics = {
            "requests_processed": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "average_response_time": 0.0,
            "uptime_seconds": 0
        }
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from YAML files."""
        config = {}
        config_dir = Path(self.config_path)
        
        # Load configuration files
        config_files = {
            "agent": "agent_config.yaml",
            "crew": "crew_config.yaml",
            "search": "search_config.yaml"
        }
        
        for config_type, filename in config_files.items():
            config_file = config_dir / filename
            if config_file.exists():
                try:
                    with open(config_file, 'r') as f:
                        config[config_type] = yaml.safe_load(f)
                    print(f"Loaded {config_type} configuration from {config_file}")
                except Exception as e:
                    print(f"Failed to load {config_file}: {e}")
                    config[config_type] = {}
            else:
                print(f"Configuration file not found: {config_file}")
                config[config_type] = {}
        
        return config
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get("agent", {}).get("global_agent_settings", {}).get("logging", {})
        
        log_level = getattr(logging, log_config.get("level", "INFO"))
        log_format = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        
        # Create logs directory if it doesn't exist
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_dir / "research_assistant.log")
            ]
        )
    
    async def initialize_agents(self):
        """Initialize all research agents."""
        self.logger.info("Initializing research agents...")
        
        agent_configs = self.config.get("agent", {})
        
        try:
            # Initialize Web Search Agent
            search_config = agent_configs.get("web_search_agent", {})
            self.agents["web_search"] = WebSearchAgent(search_config)
            
            # Initialize Content Analyzer
            analyzer_config = agent_configs.get("content_analyzer", {})
            self.agents["content_analyzer"] = ContentAnalyzer(analyzer_config)
            
            # Initialize Fact Checker
            fact_check_config = agent_configs.get("fact_checker", {})
            self.agents["fact_checker"] = FactChecker(fact_check_config)
            
            # Initialize Synthesis Agent
            synthesis_config = agent_configs.get("synthesis_agent", {})
            self.agents["synthesis_agent"] = SynthesisAgent(synthesis_config)
            
            # Initialize Research Orchestrator
            orchestrator_config = agent_configs.get("research_orchestrator", {})
            self.orchestrator = ResearchOrchestrator(orchestrator_config)
            
            self.logger.info(f"Initialized {len(self.agents)} agents successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def initialize_crews(self):
        """Initialize CrewAI crews."""
        self.logger.info("Initializing CrewAI crews...")
        
        crew_configs = self.config.get("crew", {})
        
        try:
            # Initialize Research Crew
            research_config = crew_configs.get("research_crew", {})
            self.crews["research"] = ResearchCrew(research_config)
            
            # Initialize Fact Check Crew
            fact_check_config = crew_configs.get("fact_check_crew", {})
            self.crews["fact_check"] = FactCheckCrew(fact_check_config)
            
            # Initialize Content Creation Crew
            content_config = crew_configs.get("content_creation_crew", {})
            self.crews["content_creation"] = ContentCreationCrew(content_config)
            
            self.logger.info(f"Initialized {len(self.crews)} crews successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize crews: {e}")
            raise
    
    async def start_amp_agents(self, registry_endpoint: str = "http://localhost:8000"):
        """Start AMP agents and register with registry."""
        self.logger.info("Starting AMP agents...")
        
        try:
            # Start individual agents
            agent_endpoints = {
                "web_search": f"{registry_endpoint}/web-search",
                "content_analyzer": f"{registry_endpoint}/content-analyzer", 
                "fact_checker": f"{registry_endpoint}/fact-checker",
                "synthesis_agent": f"{registry_endpoint}/synthesis-agent"
            }
            
            for agent_name, agent in self.agents.items():
                endpoint = agent_endpoints.get(agent_name, registry_endpoint)
                client = await agent.start_amp_agent(agent_name.replace("_", "-"), endpoint)
                self.amp_clients.append(client)
                self.logger.info(f"Started AMP agent: {agent_name}")
            
            # Start orchestrator
            if self.orchestrator:
                orchestrator_client = await self.orchestrator.start_amp_agent(
                    "research-orchestrator", f"{registry_endpoint}/research-orchestrator"
                )
                self.amp_clients.append(orchestrator_client)
                self.logger.info("Started research orchestrator")
            
            self.logger.info(f"Started {len(self.amp_clients)} AMP agents")
            
        except Exception as e:
            self.logger.error(f"Failed to start AMP agents: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": (datetime.now() - self.startup_time).total_seconds() if self.startup_time else 0,
            "components": {},
            "metrics": self.metrics
        }
        
        # Check agents
        for agent_name, agent in self.agents.items():
            try:
                # Simple health check - could be enhanced
                health_status["components"][agent_name] = {
                    "status": "healthy",
                    "type": "agent"
                }
            except Exception as e:
                health_status["components"][agent_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "type": "agent"
                }
                health_status["status"] = "degraded"
        
        # Check crews
        for crew_name, crew in self.crews.items():
            try:
                health_status["components"][crew_name] = {
                    "status": "healthy",
                    "type": "crew"
                }
            except Exception as e:
                health_status["components"][crew_name] = {
                    "status": "unhealthy",
                    "error": str(e),
                    "type": "crew"
                }
                health_status["status"] = "degraded"
        
        # Check AMP clients
        healthy_clients = sum(1 for client in self.amp_clients if client.connected)
        health_status["components"]["amp_clients"] = {
            "status": "healthy" if healthy_clients == len(self.amp_clients) else "degraded",
            "connected": healthy_clients,
            "total": len(self.amp_clients),
            "type": "connectivity"
        }
        
        return health_status
    
    async def conduct_research(self, query: str, parameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Conduct research using the orchestrator."""
        if not self.orchestrator:
            raise RuntimeError("Research orchestrator not initialized")
        
        self.logger.info(f"Starting research for query: {query}")
        start_time = datetime.now()
        
        try:
            # Create research query
            research_query = ResearchQuery(
                query=query,
                depth=parameters.get("depth", "standard"),
                max_sources=parameters.get("max_sources", 10),
                focus_areas=parameters.get("focus_areas", []),
                report_format=parameters.get("report_format", "academic"),
                target_length=parameters.get("target_length", 1500),
                include_fact_checking=parameters.get("include_fact_checking", True),
                quality_threshold=parameters.get("quality_threshold", 0.6)
            )
            
            # Conduct research
            result = await self.orchestrator.conduct_research(research_query)
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics["requests_processed"] += 1
            self.metrics["successful_requests"] += 1
            self._update_average_response_time(response_time)
            
            self.logger.info(f"Research completed in {response_time:.2f} seconds")
            return {
                "success": True,
                "result": result,
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self.metrics["requests_processed"] += 1
            self.metrics["failed_requests"] += 1
            self._update_average_response_time(response_time)
            
            self.logger.error(f"Research failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response_time": response_time,
                "timestamp": datetime.now().isoformat()
            }
    
    def _update_average_response_time(self, response_time: float):
        """Update average response time metric."""
        current_avg = self.metrics["average_response_time"]
        total_requests = self.metrics["requests_processed"]
        
        if total_requests == 1:
            self.metrics["average_response_time"] = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics["average_response_time"] = (1 - alpha) * current_avg + alpha * response_time
    
    async def start(self, registry_endpoint: str = "http://localhost:8000"):
        """Start the research assistant system."""
        self.logger.info("Starting Research Assistant Network...")
        self.startup_time = datetime.now()
        
        try:
            # Initialize components
            await self.initialize_agents()
            await self.initialize_crews()
            
            # Start AMP integration
            await self.start_amp_agents(registry_endpoint)
            
            self.running = True
            self.logger.info("Research Assistant Network started successfully")
            
            # Print startup summary
            self._print_startup_summary()
            
        except Exception as e:
            self.logger.error(f"Failed to start Research Assistant Network: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Shutdown the research assistant system."""
        self.logger.info("Shutting down Research Assistant Network...")
        self.running = False
        
        # Disconnect AMP clients
        for client in self.amp_clients:
            try:
                await client.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting AMP client: {e}")
        
        self.logger.info("Research Assistant Network shutdown complete")
    
    def _print_startup_summary(self):
        """Print system startup summary."""
        print("\\n" + "=" * 60)
        print("🤖 Research Assistant Network - READY")
        print("=" * 60)
        print(f"📅 Started: {self.startup_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔧 Agents: {len(self.agents)}")
        print(f"👥 Crews: {len(self.crews)}")
        print(f"🌐 AMP Clients: {len(self.amp_clients)}")
        print("\\n📋 Available Components:")
        
        for agent_name in self.agents.keys():
            print(f"  • {agent_name.replace('_', ' ').title()}")
        
        for crew_name in self.crews.keys():
            print(f"  • {crew_name.replace('_', ' ').title()} Crew")
        
        print("\\n🚀 System Ready for Research Requests!")
        print("=" * 60)


async def main():
    """Main application entry point."""
    parser = argparse.ArgumentParser(description="Research Assistant Network")
    parser.add_argument("--config", help="Configuration directory path", default="config")
    parser.add_argument("--registry", help="AMP registry endpoint", default="http://localhost:8000")
    parser.add_argument("--demo", help="Run demonstration research", action="store_true")
    parser.add_argument("--query", help="Research query for demo mode", 
                       default="latest developments in renewable energy storage technologies")
    
    args = parser.parse_args()
    
    # Create research assistant system
    system = ResearchAssistantSystem(config_path=args.config)
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(signum, frame):
        print("\\nReceived shutdown signal...")
        asyncio.create_task(system.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Start the system
        await system.start(registry_endpoint=args.registry)
        
        if args.demo:
            # Run demonstration research
            print(f"\\n🔍 Running demonstration research...")
            print(f"Query: {args.query}")
            print("-" * 60)
            
            result = await system.conduct_research(
                query=args.query,
                parameters={
                    "depth": "standard",
                    "max_sources": 5,
                    "focus_areas": ["technology", "efficiency"],
                    "report_format": "academic",
                    "target_length": 1200
                }
            )
            
            if result["success"]:
                research_result = result["result"]
                print(f"\\n✅ Research completed successfully!")
                print(f"📊 Title: {research_result.title}")
                print(f"📝 Word Count: {research_result.word_count}")
                print(f"⭐ Quality Score: {research_result.quality_score:.2f}")
                print(f"📚 Sources: {len(research_result.sources)}")
                print(f"🔍 Fact Checks: {len(research_result.fact_check_results)}")
                print(f"⏱️  Response Time: {result['response_time']:.2f} seconds")
                
                print(f"\\n📄 Executive Summary:")
                print(research_result.executive_summary)
                
                if research_result.conclusions:
                    print(f"\\n🎯 Key Conclusions:")
                    for i, conclusion in enumerate(research_result.conclusions[:3], 1):
                        print(f"{i}. {conclusion}")
                
                if research_result.recommendations:
                    print(f"\\n💡 Recommendations:")
                    for i, recommendation in enumerate(research_result.recommendations[:3], 1):
                        print(f"{i}. {recommendation}")
                        
            else:
                print(f"\\n❌ Research failed: {result['error']}")
        else:
            # Keep running
            print("\\n💤 System running... Press Ctrl+C to stop")
            
            while system.running:
                # Periodic health check
                health = await system.health_check()
                if health["status"] != "healthy":
                    system.logger.warning(f"System health degraded: {health}")
                
                await asyncio.sleep(60)  # Check every minute
    
    except KeyboardInterrupt:
        print("\\nShutdown requested by user")
    except Exception as e:
        print(f"\\nFatal error: {e}")
        logging.error(f"Fatal error: {e}")
    finally:
        await system.shutdown()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\\nGoodbye!")
    except Exception as e:
        print(f"Application error: {e}")
        sys.exit(1)