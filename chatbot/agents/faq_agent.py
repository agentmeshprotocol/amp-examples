"""
FAQ Agent for Multi-Agent Chatbot System

Handles frequently asked questions using a vector store knowledge base.
Uses LangChain for semantic search and response generation.
"""

import asyncio
import logging
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path

# LangChain imports
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

# AMP imports
import sys
sys.path.append(str(Path(__file__).parent.parent.parent / "shared-lib"))
from amp_client import AMPClient
from amp_types import Capability, CapabilityConstraints, TransportType
from amp_utils import AMPBuilder


class FAQAgent:
    """FAQ agent that handles common questions using vector search."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.logger = logging.getLogger(f"{__name__}.FAQAgent")
        
        # Initialize LangChain components
        self.llm = ChatOpenAI(
            model=self.config.get("model", "gpt-3.5-turbo"),
            temperature=self.config.get("temperature", 0.2)
        )
        
        self.embeddings = OpenAIEmbeddings()
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        self.vector_store = self._setup_vector_store()
        
        # AMP client
        self.amp_client: Optional[AMPClient] = None
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load agent configuration."""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "agent_config.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                return config.get("faq_agent", {})
        except FileNotFoundError:
            return {}
    
    def _load_knowledge_base(self) -> List[Dict[str, str]]:
        """Load FAQ knowledge base."""
        kb_path = Path(__file__).parent.parent / "config" / "knowledge_base.yaml"
        
        try:
            with open(kb_path, 'r') as f:
                kb = yaml.safe_load(f)
                return kb.get("faqs", [])
        except FileNotFoundError:
            return self._get_default_knowledge_base()
    
    def _get_default_knowledge_base(self) -> List[Dict[str, str]]:
        """Get default knowledge base."""
        return [
            {
                "question": "What are your business hours?",
                "answer": "Our business hours are Monday-Friday 9AM-6PM EST, and Saturday 10AM-4PM EST. We're closed on Sundays.",
                "category": "hours"
            },
            {
                "question": "How can I contact customer support?",
                "answer": "You can contact customer support by email at support@company.com, phone at 1-800-SUPPORT, or through this chat.",
                "category": "contact"
            },
            {
                "question": "What is your return policy?",
                "answer": "We offer a 30-day return policy for all products. Items must be in original condition with receipt.",
                "category": "policy"
            },
            {
                "question": "Where is your company located?",
                "answer": "Our headquarters is located at 123 Business St, Suite 100, Tech City, TC 12345.",
                "category": "company"
            },
            {
                "question": "Do you offer international shipping?",
                "answer": "Yes, we ship internationally to most countries. Shipping costs and delivery times vary by location.",
                "category": "shipping"
            }
        ]
    
    def _setup_vector_store(self) -> FAISS:
        """Set up vector store with knowledge base."""
        documents = []
        
        for faq in self.knowledge_base:
            # Create documents from Q&A pairs
            doc = Document(
                page_content=f"Question: {faq['question']}\\nAnswer: {faq['answer']}",
                metadata={
                    "question": faq["question"],
                    "answer": faq["answer"],
                    "category": faq.get("category", "general")
                }
            )
            documents.append(doc)
        
        # Create vector store
        if documents:
            return FAISS.from_documents(documents, self.embeddings)
        else:
            # Create empty vector store
            return FAISS.from_texts(["dummy"], self.embeddings)
    
    async def search_knowledge_base(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search knowledge base for relevant information."""
        try:
            # Perform semantic search
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "question": doc.metadata.get("question", ""),
                    "answer": doc.metadata.get("answer", ""),
                    "category": doc.metadata.get("category", "general"),
                    "relevance_score": float(score),
                    "content": doc.page_content
                })
            
            return formatted_results
            
        except Exception as e:
            self.logger.error(f"Knowledge base search failed: {e}")
            return []
    
    async def generate_response(self, user_input: str, context: Dict[str, Any] = None) -> str:
        """Generate response to user query."""
        # Search knowledge base
        kb_results = await self.search_knowledge_base(user_input)
        
        if not kb_results:
            return "I'm sorry, I don't have information about that. Let me connect you with someone who can help."
        
        # Get the best match
        best_match = kb_results[0]
        similarity_threshold = self.config.get("similarity_threshold", 0.7)
        
        # If similarity is too low, provide general response
        if best_match["relevance_score"] > similarity_threshold:
            return "I'm not sure about that specific question. Let me connect you with a specialist who can help."
        
        # Use LLM to generate contextual response
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful customer service agent. Use the provided FAQ information to answer the user's question. Be friendly and concise."),
            ("human", f"User question: {user_input}\\n\\nRelevant FAQ:\\nQ: {best_match['question']}\\nA: {best_match['answer']}\\n\\nProvide a helpful response:")
        ])
        
        try:
            response = await self.llm.ainvoke(prompt.format_messages())
            return response.content
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return best_match["answer"]
    
    async def handle_conversation(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle conversation for FAQ topics."""
        user_input = parameters.get("user_input", "")
        session_id = parameters.get("session_id", "")
        context = parameters.get("context", {})
        
        # Generate response
        response = await self.generate_response(user_input, context)
        
        # Search for related questions
        related = await self.search_knowledge_base(user_input, k=3)
        
        return {
            "response": response,
            "agent": "faq-agent",
            "related_questions": [r["question"] for r in related[1:4]],  # Exclude the best match
            "confidence": 0.9 if related and related[0]["relevance_score"] < 0.3 else 0.7,
            "source": "knowledge_base"
        }
    
    async def search_faq(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Search FAQ capability."""
        query = parameters.get("query", "")
        max_results = parameters.get("max_results", 5)
        
        results = await self.search_knowledge_base(query, k=max_results)
        
        return {
            "results": results,
            "query": query,
            "total_results": len(results)
        }
    
    async def start_amp_agent(self, agent_id: str = "faq-agent",
                             endpoint: str = "http://localhost:8000") -> AMPClient:
        """Start the AMP agent."""
        
        self.amp_client = await (
            AMPBuilder(agent_id, "FAQ Agent")
            .with_framework("langchain")
            .with_transport(TransportType.HTTP, endpoint)
            .add_capability(
                "conversation-handler",
                self.handle_conversation,
                "Handle FAQ conversations and questions",
                "conversation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "user_input": {"type": "string"},
                        "session_id": {"type": "string"},
                        "context": {"type": "object"}
                    },
                    "required": ["user_input"]
                },
                constraints=CapabilityConstraints(response_time_ms=3000)
            )
            .add_capability(
                "faq-search",
                self.search_faq,
                "Search FAQ knowledge base",
                "search",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "max_results": {"type": "integer", "default": 5}
                    },
                    "required": ["query"]
                }
            )
            .build()
        )
        
        return self.amp_client


async def main():
    """Main function for testing the FAQ agent."""
    logging.basicConfig(level=logging.INFO)
    
    # Create FAQ agent
    faq_agent = FAQAgent()
    
    # Start AMP agent
    client = await faq_agent.start_amp_agent()
    
    try:
        print("FAQ Agent started. Testing knowledge base search...")
        
        # Test queries
        test_queries = [
            "What are your hours?",
            "How do I return something?",
            "Where are you located?",
            "Do you ship worldwide?"
        ]
        
        for query in test_queries:
            response = await faq_agent.generate_response(query)
            print(f"Q: {query}")
            print(f"A: {response}\\n")
        
        print("FAQ Agent is running. Press Ctrl+C to stop.")
        await asyncio.Future()  # Run forever
        
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())