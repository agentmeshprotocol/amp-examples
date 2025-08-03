"""
AutoGen AMP Agent Base Class.

Integrates Microsoft AutoGen framework with AMP protocol for
collaborative data analysis workflows.
"""

import asyncio
import logging
import sys
import os
from typing import Dict, Any, List, Optional, Callable, Union
from abc import ABC, abstractmethod
from dataclasses import dataclass

# Add shared-lib to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../shared-lib'))

import autogen
from autogen import ConversableAgent, UserProxyAgent, GroupChat, GroupChatManager

from amp_client import AMPClient, AMPClientConfig
from amp_types import (
    Capability, CapabilityConstraints, AgentIdentity, 
    AMPMessage, MessageType, TransportType
)


@dataclass
class AutoGenConfig:
    """Configuration for AutoGen agent."""
    name: str
    system_message: str
    llm_config: Dict[str, Any]
    human_input_mode: str = "NEVER"
    max_consecutive_auto_reply: int = 10
    is_termination_msg: Optional[Callable] = None
    code_execution_config: Optional[Dict[str, Any]] = None


class AutoGenAMPAgent(ConversableAgent, ABC):
    """
    Base class for AutoGen agents with AMP protocol integration.
    
    Combines AutoGen's conversational AI capabilities with AMP's
    standardized agent communication protocol.
    """
    
    def __init__(
        self,
        autogen_config: AutoGenConfig,
        amp_config: AMPClientConfig,
        capabilities: List[Capability],
        description: str = "",
        tags: List[str] = None
    ):
        """
        Initialize AutoGen AMP Agent.
        
        Args:
            autogen_config: AutoGen agent configuration
            amp_config: AMP client configuration  
            capabilities: List of capabilities this agent provides
            description: Agent description
            tags: Agent tags for discovery
        """
        # Initialize AutoGen agent
        super().__init__(
            name=autogen_config.name,
            system_message=autogen_config.system_message,
            llm_config=autogen_config.llm_config,
            human_input_mode=autogen_config.human_input_mode,
            max_consecutive_auto_reply=autogen_config.max_consecutive_auto_reply,
            is_termination_msg=autogen_config.is_termination_msg,
            code_execution_config=autogen_config.code_execution_config
        )
        
        # Initialize AMP client
        self.amp_client = AMPClient(amp_config)
        self.amp_config = amp_config
        
        # Agent metadata
        self.description = description
        self.tags = tags or []
        self.capabilities = {cap.id: cap for cap in capabilities}
        
        # State management
        self._connected = False
        self._conversation_context: Dict[str, Any] = {}
        self._data_artifacts: Dict[str, Any] = {}
        
        # Set up logging
        self.logger = logging.getLogger(f"amp.autogen.{amp_config.agent_id}")
        self.logger.setLevel(amp_config.log_level)
        
        # Register AMP capabilities
        self._register_amp_capabilities()
        
        # Set up AutoGen message handling
        self.register_reply(
            [autogen.Agent, type(None)],
            reply_func=self._autogen_reply_handler
        )
    
    async def connect(self) -> bool:
        """Connect to AMP network."""
        try:
            connected = await self.amp_client.connect()
            if connected:
                self._connected = True
                self.logger.info(f"Agent {self.name} connected to AMP network")
                return True
            else:
                self.logger.error(f"Failed to connect agent {self.name}")
                return False
        except Exception as e:
            self.logger.error(f"Connection error for {self.name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from AMP network."""
        if self._connected:
            await self.amp_client.disconnect()
            self._connected = False
            self.logger.info(f"Agent {self.name} disconnected")
    
    def _register_amp_capabilities(self):
        """Register capabilities with AMP client."""
        for capability in self.capabilities.values():
            handler = getattr(self, f"_handle_{capability.id.replace('-', '_')}", None)
            if handler:
                self.amp_client.register_capability(capability, handler)
            else:
                self.logger.warning(f"No handler found for capability: {capability.id}")
    
    async def invoke_capability(
        self, 
        target_agent: Optional[str],
        capability: str, 
        parameters: Dict[str, Any],
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Invoke capability on another agent via AMP."""
        if not self._connected:
            raise RuntimeError("Agent not connected to AMP network")
        
        return await self.amp_client.invoke_capability(
            target_agent, capability, parameters, timeout
        )
    
    async def emit_event(
        self, 
        event_type: str, 
        data: Dict[str, Any],
        target_agent: Optional[str] = None
    ):
        """Emit event to AMP network."""
        if not self._connected:
            raise RuntimeError("Agent not connected to AMP network")
        
        await self.amp_client.emit_event(event_type, data, target_agent)
    
    def store_artifact(self, key: str, data: Any, metadata: Dict[str, Any] = None):
        """Store data artifact with metadata."""
        self._data_artifacts[key] = {
            "data": data,
            "metadata": metadata or {},
            "timestamp": asyncio.get_event_loop().time()
        }
        self.logger.debug(f"Stored artifact: {key}")
    
    def get_artifact(self, key: str) -> Optional[Any]:
        """Retrieve stored data artifact."""
        artifact = self._data_artifacts.get(key)
        if artifact:
            return artifact["data"]
        return None
    
    def list_artifacts(self) -> List[str]:
        """List all stored artifact keys."""
        return list(self._data_artifacts.keys())
    
    def set_context(self, key: str, value: Any):
        """Set conversation context."""
        self._conversation_context[key] = value
    
    def get_context(self, key: str, default: Any = None) -> Any:
        """Get conversation context."""
        return self._conversation_context.get(key, default)
    
    def _autogen_reply_handler(
        self, 
        messages: Optional[List[Dict]] = None,
        sender: Optional[autogen.Agent] = None,
        config: Optional[Any] = None
    ) -> Union[str, Dict, None]:
        """
        Handle AutoGen conversation messages.
        
        This integrates AutoGen's conversation flow with AMP capabilities.
        """
        if not messages:
            return None
        
        # Get the last message
        last_message = messages[-1]
        message_content = last_message.get("content", "")
        
        # Process message through agent-specific logic
        try:
            response = self._process_conversation_message(
                message_content, sender, messages
            )
            return response
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return f"Error processing request: {str(e)}"
    
    @abstractmethod
    def _process_conversation_message(
        self, 
        message: str, 
        sender: Optional[autogen.Agent],
        conversation_history: List[Dict]
    ) -> str:
        """
        Process conversation message - must be implemented by subclasses.
        
        Args:
            message: The message content
            sender: The sending agent
            conversation_history: Full conversation history
            
        Returns:
            Response message
        """
        pass
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
        autogen_health = {
            "autogen_name": self.name,
            "autogen_system_message": self.system_message[:100] + "..." if len(self.system_message) > 100 else self.system_message,
            "conversation_context_size": len(self._conversation_context),
            "stored_artifacts": len(self._data_artifacts)
        }
        
        if self._connected:
            amp_health = await self.amp_client.health_check()
            return {**amp_health, **autogen_health}
        else:
            return {
                "status": "disconnected",
                "agent_id": self.amp_config.agent_id,
                **autogen_health
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get agent metrics."""
        base_metrics = {
            "agent_name": self.name,
            "capabilities_count": len(self.capabilities),
            "artifacts_stored": len(self._data_artifacts),
            "context_size": len(self._conversation_context)
        }
        
        if self._connected:
            amp_metrics = self.amp_client.get_metrics()
            return {**base_metrics, **amp_metrics}
        else:
            return base_metrics
    
    def __str__(self) -> str:
        """String representation."""
        return f"AutoGenAMPAgent(name={self.name}, agent_id={self.amp_config.agent_id}, capabilities={list(self.capabilities.keys())})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()


class DataAnalysisOrchestrator:
    """
    Orchestrates AutoGen agents in data analysis workflows.
    
    Manages group chats and coordinates agent interactions for
    complex data analysis tasks.
    """
    
    def __init__(self, agents: List[AutoGenAMPAgent], max_round: int = 50):
        """
        Initialize orchestrator.
        
        Args:
            agents: List of AutoGen AMP agents
            max_round: Maximum conversation rounds
        """
        self.agents = agents
        self.max_round = max_round
        
        # Create user proxy for human interaction
        self.user_proxy = UserProxyAgent(
            name="DataAnalyst",
            system_message="A data analyst coordinating the analysis pipeline.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=0,
            code_execution_config={"work_dir": "data_analysis_workspace"}
        )
        
        # Create group chat
        all_agents = [self.user_proxy] + agents
        self.group_chat = GroupChat(
            agents=all_agents,
            messages=[],
            max_round=max_round
        )
        
        # Create group chat manager
        self.manager = GroupChatManager(
            groupchat=self.group_chat,
            llm_config={"config_list": []}  # Will be overridden by first agent's config
        )
        
        # Set up logging
        self.logger = logging.getLogger("amp.autogen.orchestrator")
        
    async def connect_all_agents(self) -> bool:
        """Connect all agents to AMP network."""
        results = []
        for agent in self.agents:
            try:
                connected = await agent.connect()
                results.append(connected)
                if connected:
                    self.logger.info(f"Connected agent: {agent.name}")
                else:
                    self.logger.error(f"Failed to connect agent: {agent.name}")
            except Exception as e:
                self.logger.error(f"Error connecting agent {agent.name}: {e}")
                results.append(False)
        
        return all(results)
    
    async def disconnect_all_agents(self):
        """Disconnect all agents."""
        for agent in self.agents:
            try:
                await agent.disconnect()
            except Exception as e:
                self.logger.error(f"Error disconnecting agent {agent.name}: {e}")
    
    async def run_analysis_workflow(
        self, 
        initial_message: str,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Run a complete data analysis workflow.
        
        Args:
            initial_message: Initial analysis request
            context: Additional context for the analysis
            
        Returns:
            Analysis results and conversation summary
        """
        try:
            # Set context for all agents
            if context:
                for agent in self.agents:
                    for key, value in context.items():
                        agent.set_context(key, value)
            
            # Start group conversation
            self.logger.info("Starting data analysis workflow")
            
            # Initiate chat through user proxy
            result = self.user_proxy.initiate_chat(
                self.manager,
                message=initial_message,
                clear_history=True
            )
            
            # Collect results from all agents
            workflow_results = {
                "conversation_summary": result,
                "agent_artifacts": {},
                "agent_metrics": {}
            }
            
            for agent in self.agents:
                try:
                    # Collect artifacts
                    workflow_results["agent_artifacts"][agent.name] = {
                        artifact_key: agent.get_artifact(artifact_key)
                        for artifact_key in agent.list_artifacts()
                    }
                    
                    # Collect metrics
                    workflow_results["agent_metrics"][agent.name] = await agent.get_metrics()
                    
                except Exception as e:
                    self.logger.error(f"Error collecting results from {agent.name}: {e}")
            
            self.logger.info("Data analysis workflow completed")
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Error in analysis workflow: {e}")
            raise
    
    def get_agent_by_name(self, name: str) -> Optional[AutoGenAMPAgent]:
        """Get agent by name."""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None
    
    def list_agents(self) -> List[str]:
        """List all agent names."""
        return [agent.name for agent in self.agents]
    
    async def health_check_all(self) -> Dict[str, Dict[str, Any]]:
        """Health check for all agents."""
        health_status = {}
        for agent in self.agents:
            try:
                health_status[agent.name] = await agent.health_check()
            except Exception as e:
                health_status[agent.name] = {
                    "status": "error",
                    "error": str(e)
                }
        return health_status