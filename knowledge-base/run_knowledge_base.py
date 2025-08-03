#!/usr/bin/env python3
"""
Knowledge Base System Runner

Main orchestration script for running the complete knowledge base system
with all agents, web interfaces, and supporting services.
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import sys
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional

# Add shared-lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'shared-lib'))

# Import agents
from agents.knowledge_ingestion import KnowledgeIngestionAgent
from agents.semantic_search import SemanticSearchAgent
from agents.knowledge_graph import KnowledgeGraphAgent
from agents.query_router import QueryRouterAgent
from agents.cache_manager import CacheManagerAgent
from agents.knowledge_curator import KnowledgeCuratorAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeBaseOrchestrator:
    """Orchestrates the complete knowledge base system"""
    
    def __init__(self, config_path: str = "config/agent_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.agents = {}
        self.running = False
        self.setup_signal_handlers()
        
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if file loading fails"""
        return {
            'network': {
                'host': 'localhost',
                'port': 8000
            },
            'agents': {
                'knowledge-ingestion-agent': {'enabled': True},
                'semantic-search-agent': {'enabled': True},
                'knowledge-graph-agent': {'enabled': True},
                'query-router-agent': {'enabled': True},
                'cache-manager-agent': {'enabled': True},
                'knowledge-curator-agent': {'enabled': True}
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
    
    async def setup_directories(self):
        """Setup required directories"""
        directories = [
            "data/documents",
            "data/embeddings", 
            "data/graphs",
            "data/cache",
            "data/uploads",
            "data/samples",
            "logs",
            "temp"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Ensured directory exists: {directory}")
    
    async def initialize_agents(self):
        """Initialize and configure all agents"""
        host = self.config['network']['host']
        port = self.config['network']['port']
        
        agent_configs = self.config.get('agents', {})
        
        # Initialize agents if enabled
        if agent_configs.get('knowledge-ingestion-agent', {}).get('enabled', True):
            self.agents['knowledge-ingestion'] = KnowledgeIngestionAgent()
            
        if agent_configs.get('semantic-search-agent', {}).get('enabled', True):
            self.agents['semantic-search'] = SemanticSearchAgent()
            
        if agent_configs.get('knowledge-graph-agent', {}).get('enabled', True):
            self.agents['knowledge-graph'] = KnowledgeGraphAgent()
            
        if agent_configs.get('query-router-agent', {}).get('enabled', True):
            self.agents['query-router'] = QueryRouterAgent()
            
        if agent_configs.get('cache-manager-agent', {}).get('enabled', True):
            self.agents['cache-manager'] = CacheManagerAgent()
            
        if agent_configs.get('knowledge-curator-agent', {}).get('enabled', True):
            self.agents['knowledge-curator'] = KnowledgeCuratorAgent()
        
        logger.info(f"Initialized {len(self.agents)} agents")
    
    async def start_agents(self):
        """Start all agents"""
        host = self.config['network']['host']
        port = self.config['network']['port']
        
        start_tasks = []
        for agent_name, agent in self.agents.items():
            task = asyncio.create_task(agent.start(host, port))
            start_tasks.append(task)
            logger.info(f"Starting {agent_name} agent...")
        
        # Wait for all agents to start
        try:
            await asyncio.gather(*start_tasks)
            logger.info("All agents started successfully")
        except Exception as e:
            logger.error(f"Error starting agents: {e}")
            raise
    
    async def start_web_interfaces(self):
        """Start web and admin interfaces"""
        web_config = self.config.get('web_interface', {})
        admin_config = self.config.get('admin_interface', {})
        
        if web_config.get('enabled', True):
            logger.info("Starting web interface...")
            # In production, this would start the web server
            # For this example, we'll just log that it would start
            
        if admin_config.get('enabled', True):
            logger.info("Starting admin interface...")
            # In production, this would start the admin server
    
    async def health_check(self):
        """Perform periodic health checks"""
        while self.running:
            try:
                healthy_agents = 0
                total_agents = len(self.agents)
                
                for agent_name, agent in self.agents.items():
                    # Check if agent is still running
                    if hasattr(agent, 'client') and hasattr(agent.client, 'connected'):
                        if agent.client.connected:
                            healthy_agents += 1
                    else:
                        healthy_agents += 1  # Assume healthy if can't check
                
                health_ratio = healthy_agents / max(total_agents, 1)
                
                if health_ratio < 0.8:
                    logger.warning(f"System health degraded: {healthy_agents}/{total_agents} agents healthy")
                else:
                    logger.debug(f"System health good: {healthy_agents}/{total_agents} agents healthy")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def run(self):
        """Run the complete knowledge base system"""
        logger.info("Starting Knowledge Base System...")
        
        try:
            # Setup
            await self.setup_directories()
            await self.initialize_agents()
            
            # Start components
            self.running = True
            await self.start_agents()
            await self.start_web_interfaces()
            
            logger.info("Knowledge Base System started successfully!")
            logger.info("=== System Information ===")
            logger.info(f"Agents running: {list(self.agents.keys())}")
            logger.info(f"Configuration: {self.config_path}")
            logger.info(f"Web interface: http://localhost:8080")
            logger.info(f"Admin interface: http://localhost:8081")
            logger.info("=========================")
            
            # Start health monitoring
            health_task = asyncio.create_task(self.health_check())
            
            # Keep running until shutdown signal
            while self.running:
                await asyncio.sleep(1)
            
            # Cleanup
            health_task.cancel()
            await self.shutdown()
            
        except KeyboardInterrupt:
            logger.info("Shutdown requested by user")
            await self.shutdown()
        except Exception as e:
            logger.error(f"Fatal error: {e}")
            await self.shutdown()
            raise
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        logger.info("Shutting down Knowledge Base System...")
        
        # Stop agents
        stop_tasks = []
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'stop'):
                task = asyncio.create_task(agent.stop())
                stop_tasks.append(task)
                logger.info(f"Stopping {agent_name} agent...")
        
        if stop_tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*stop_tasks), timeout=10)
                logger.info("All agents stopped successfully")
            except asyncio.TimeoutError:
                logger.warning("Some agents did not stop gracefully within timeout")
            except Exception as e:
                logger.error(f"Error stopping agents: {e}")
        
        logger.info("Knowledge Base System shutdown complete")


class SystemManager:
    """High-level system management utilities"""
    
    @staticmethod
    def check_dependencies():
        """Check if required dependencies are available"""
        required_packages = [
            'faiss-cpu',
            'sentence-transformers', 
            'spacy',
            'networkx',
            'redis',
            'fastapi',
            'uvicorn'
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            logger.error(f"Missing required packages: {missing_packages}")
            logger.error("Please install missing packages with: pip install -r requirements.txt")
            return False
        
        return True
    
    @staticmethod
    def check_spacy_model():
        """Check if required spaCy model is available"""
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            return True
        except OSError:
            logger.error("spaCy model 'en_core_web_sm' not found")
            logger.error("Please install with: python -m spacy download en_core_web_sm")
            return False
    
    @staticmethod
    def create_sample_data():
        """Create sample documents for testing"""
        samples_dir = Path("data/samples")
        samples_dir.mkdir(parents=True, exist_ok=True)
        
        sample_documents = [
            {
                "filename": "ai_overview.txt",
                "content": """
Artificial Intelligence: An Overview

Artificial Intelligence (AI) is a branch of computer science that aims to create 
intelligent machines that can perform tasks that typically require human intelligence. 
These tasks include learning, reasoning, problem-solving, perception, and language understanding.

Machine learning is a subset of AI that enables computers to learn and improve from 
experience without being explicitly programmed. Deep learning, a subset of machine learning, 
uses neural networks with multiple layers to model and understand complex patterns in data.

Applications of AI include natural language processing, computer vision, robotics, 
expert systems, and autonomous vehicles. As AI technology continues to advance, 
it is transforming industries and creating new possibilities for innovation.
                """.strip()
            },
            {
                "filename": "machine_learning_basics.txt", 
                "content": """
Machine Learning Fundamentals

Machine learning is a method of data analysis that automates analytical model building. 
It is based on the idea that systems can learn from data, identify patterns, and make 
decisions with minimal human intervention.

Types of machine learning include:
1. Supervised Learning - Learning with labeled training data
2. Unsupervised Learning - Finding patterns in data without labels  
3. Reinforcement Learning - Learning through interaction with environment

Common algorithms include linear regression, decision trees, random forests, 
support vector machines, and neural networks. The choice of algorithm depends 
on the problem type, data size, and desired accuracy.
                """.strip()
            },
            {
                "filename": "data_science_workflow.txt",
                "content": """
Data Science Workflow

The data science process typically follows these steps:

1. Problem Definition - Clearly define the business problem or research question
2. Data Collection - Gather relevant data from various sources
3. Data Cleaning - Remove inconsistencies and handle missing values
4. Exploratory Data Analysis - Understand data patterns and relationships
5. Feature Engineering - Create new variables and select relevant features
6. Model Building - Apply machine learning algorithms
7. Model Evaluation - Assess model performance using appropriate metrics
8. Deployment - Implement the model in production environment
9. Monitoring - Track model performance and update as needed

This iterative process requires domain expertise, statistical knowledge, 
and programming skills to extract actionable insights from data.
                """.strip()
            }
        ]
        
        for doc in sample_documents:
            file_path = samples_dir / doc["filename"]
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(doc["content"])
        
        logger.info(f"Created {len(sample_documents)} sample documents in {samples_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Knowledge Base System")
    parser.add_argument(
        "--config", 
        default="config/agent_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--check-deps", 
        action="store_true",
        help="Check dependencies and exit"
    )
    parser.add_argument(
        "--create-samples",
        action="store_true", 
        help="Create sample data and exit"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Set logging level"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Check dependencies
    if args.check_deps:
        if SystemManager.check_dependencies() and SystemManager.check_spacy_model():
            logger.info("All dependencies are satisfied")
            sys.exit(0)
        else:
            sys.exit(1)
    
    # Create sample data
    if args.create_samples:
        SystemManager.create_sample_data()
        sys.exit(0)
    
    # Verify dependencies before starting
    if not SystemManager.check_dependencies():
        sys.exit(1)
        
    if not SystemManager.check_spacy_model():
        sys.exit(1)
    
    # Start the system
    orchestrator = KnowledgeBaseOrchestrator(args.config)
    
    try:
        asyncio.run(orchestrator.run())
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    except Exception as e:
        logger.error(f"System error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()