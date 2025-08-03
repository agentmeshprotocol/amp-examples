"""
Knowledge Graph Agent

This agent manages entity extraction, relationship mapping, and graph traversal
for building and querying knowledge graphs from processed documents.
"""

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import networkx as nx
import spacy
from neo4j import AsyncGraphDatabase
from rdflib import Graph, Literal, RDF, RDFS, URIRef, Namespace

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient
from amp_types import AMPMessage, MessageType

logger = logging.getLogger(__name__)


class EntityExtractor:
    """Extracts entities and relationships from text using NLP"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        
        # Define relationship patterns
        self.relationship_patterns = [
            # Subject-Verb-Object patterns
            {"PATTERN": [{"POS": "NOUN"}, {"POS": "VERB"}, {"POS": "NOUN"}]},
            {"PATTERN": [{"ENT_TYPE": "PERSON"}, {"POS": "VERB"}, {"ENT_TYPE": "ORG"}]},
            {"PATTERN": [{"ENT_TYPE": "ORG"}, {"LEMMA": "acquire"}, {"ENT_TYPE": "ORG"}]},
        ]
    
    async def extract_entities(self, text: str, chunk_id: str) -> Dict[str, Any]:
        """Extract entities from text"""
        doc = self.nlp(text)
        
        entities = []
        for ent in doc.ents:
            entity = {
                'id': f"{chunk_id}_{ent.start}_{ent.end}",
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start,
                'end': ent.end,
                'confidence': 1.0,  # SpaCy doesn't provide confidence scores by default
                'chunk_id': chunk_id
            }
            entities.append(entity)
        
        return {
            'chunk_id': chunk_id,
            'entities': entities,
            'entity_count': len(entities)
        }
    
    async def extract_relationships(self, text: str, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities"""
        doc = self.nlp(text)
        relationships = []
        
        # Simple relationship extraction based on dependency parsing
        for token in doc:
            if token.dep_ in ['nsubj', 'dobj', 'pobj']:
                head = token.head
                
                # Find entities that correspond to these tokens
                subject_entity = self._find_entity_by_token(token, entities)
                object_entity = self._find_entity_by_token(head, entities)
                
                if subject_entity and object_entity and subject_entity != object_entity:
                    relationship = {
                        'id': str(uuid.uuid4()),
                        'subject': subject_entity,
                        'predicate': head.lemma_,
                        'object': object_entity,
                        'confidence': 0.8,
                        'source_text': text[min(token.i, head.i):max(token.i, head.i) + 1]
                    }
                    relationships.append(relationship)
        
        return relationships
    
    def _find_entity_by_token(self, token, entities: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Find entity that contains the given token"""
        for entity in entities:
            if entity['start'] <= token.i < entity['end']:
                return entity
        return None


class GraphDatabase:
    """Manages knowledge graph storage and operations"""
    
    def __init__(self, db_type: str = "networkx"):
        self.db_type = db_type
        
        if db_type == "networkx":
            self.graph = nx.DiGraph()
        elif db_type == "neo4j":
            self.driver = None
        elif db_type == "rdf":
            self.graph = Graph()
            self.namespace = Namespace("http://example.org/knowledge/")
        
        self.entity_index = {}
        self.relationship_index = {}
    
    async def connect(self, connection_params: Dict[str, Any] = None):
        """Connect to the database"""
        if self.db_type == "neo4j" and connection_params:
            uri = connection_params.get('uri', 'bolt://localhost:7687')
            username = connection_params.get('username', 'neo4j')
            password = connection_params.get('password', 'password')
            
            self.driver = AsyncGraphDatabase.driver(uri, auth=(username, password))
    
    async def add_entity(self, entity: Dict[str, Any]) -> str:
        """Add an entity to the graph"""
        entity_id = entity['id']
        
        if self.db_type == "networkx":
            self.graph.add_node(entity_id, **entity)
        elif self.db_type == "neo4j" and self.driver:
            async with self.driver.session() as session:
                await session.run(
                    "MERGE (e:Entity {id: $id, text: $text, label: $label})",
                    id=entity_id,
                    text=entity['text'],
                    label=entity['label']
                )
        elif self.db_type == "rdf":
            entity_uri = URIRef(self.namespace + entity_id)
            self.graph.add((entity_uri, RDF.type, URIRef(self.namespace + entity['label'])))
            self.graph.add((entity_uri, RDFS.label, Literal(entity['text'])))
        
        self.entity_index[entity_id] = entity
        return entity_id
    
    async def add_relationship(self, relationship: Dict[str, Any]) -> str:
        """Add a relationship to the graph"""
        rel_id = relationship['id']
        subject_id = relationship['subject']['id']
        object_id = relationship['object']['id']
        predicate = relationship['predicate']
        
        if self.db_type == "networkx":
            self.graph.add_edge(
                subject_id, 
                object_id, 
                id=rel_id,
                predicate=predicate,
                **relationship
            )
        elif self.db_type == "neo4j" and self.driver:
            async with self.driver.session() as session:
                await session.run(
                    """
                    MATCH (s:Entity {id: $subject_id}), (o:Entity {id: $object_id})
                    MERGE (s)-[r:RELATIONSHIP {id: $rel_id, predicate: $predicate}]->(o)
                    """,
                    subject_id=subject_id,
                    object_id=object_id,
                    rel_id=rel_id,
                    predicate=predicate
                )
        elif self.db_type == "rdf":
            subject_uri = URIRef(self.namespace + subject_id)
            object_uri = URIRef(self.namespace + object_id)
            predicate_uri = URIRef(self.namespace + predicate)
            self.graph.add((subject_uri, predicate_uri, object_uri))
        
        self.relationship_index[rel_id] = relationship
        return rel_id
    
    async def find_entities(self, entity_type: str = None, text_contains: str = None) -> List[Dict[str, Any]]:
        """Find entities matching criteria"""
        results = []
        
        for entity_id, entity in self.entity_index.items():
            include = True
            
            if entity_type and entity.get('label') != entity_type:
                include = False
            
            if text_contains and text_contains.lower() not in entity.get('text', '').lower():
                include = False
            
            if include:
                results.append(entity)
        
        return results
    
    async def find_related_entities(self, entity_id: str, max_depth: int = 2) -> List[Dict[str, Any]]:
        """Find entities related to the given entity"""
        if self.db_type == "networkx":
            # Use NetworkX to find connected entities
            if entity_id in self.graph:
                related_ids = set()
                
                # BFS to find related entities within max_depth
                queue = [(entity_id, 0)]
                visited = set([entity_id])
                
                while queue:
                    current_id, depth = queue.pop(0)
                    
                    if depth < max_depth:
                        # Get neighbors (both incoming and outgoing)
                        neighbors = list(self.graph.successors(current_id)) + list(self.graph.predecessors(current_id))
                        
                        for neighbor_id in neighbors:
                            if neighbor_id not in visited:
                                visited.add(neighbor_id)
                                related_ids.add(neighbor_id)
                                queue.append((neighbor_id, depth + 1))
                
                return [self.entity_index[eid] for eid in related_ids if eid in self.entity_index]
        
        return []
    
    async def find_shortest_path(self, entity1_id: str, entity2_id: str) -> List[Dict[str, Any]]:
        """Find shortest path between two entities"""
        if self.db_type == "networkx":
            try:
                path = nx.shortest_path(self.graph, entity1_id, entity2_id)
                path_info = []
                
                for i in range(len(path) - 1):
                    current_id = path[i]
                    next_id = path[i + 1]
                    
                    # Get edge data
                    edge_data = self.graph.get_edge_data(current_id, next_id)
                    
                    path_info.append({
                        'from_entity': self.entity_index.get(current_id),
                        'to_entity': self.entity_index.get(next_id),
                        'relationship': edge_data
                    })
                
                return path_info
            except nx.NetworkXNoPath:
                return []
        
        return []
    
    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph"""
        if self.db_type == "networkx":
            return {
                'entity_count': self.graph.number_of_nodes(),
                'relationship_count': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_connected': nx.is_weakly_connected(self.graph),
                'clustering_coefficient': nx.average_clustering(self.graph.to_undirected())
            }
        
        return {
            'entity_count': len(self.entity_index),
            'relationship_count': len(self.relationship_index)
        }
    
    async def save_graph(self, filepath: str):
        """Save graph to file"""
        if self.db_type == "networkx":
            nx.write_gexf(self.graph, f"{filepath}.gexf")
            
            # Save indices
            with open(f"{filepath}_entities.json", 'w') as f:
                json.dump(self.entity_index, f, indent=2)
            
            with open(f"{filepath}_relationships.json", 'w') as f:
                json.dump(self.relationship_index, f, indent=2)
        
        elif self.db_type == "rdf":
            self.graph.serialize(destination=f"{filepath}.ttl", format="turtle")
    
    async def load_graph(self, filepath: str):
        """Load graph from file"""
        if self.db_type == "networkx":
            if os.path.exists(f"{filepath}.gexf"):
                self.graph = nx.read_gexf(f"{filepath}.gexf")
            
            if os.path.exists(f"{filepath}_entities.json"):
                with open(f"{filepath}_entities.json", 'r') as f:
                    self.entity_index = json.load(f)
            
            if os.path.exists(f"{filepath}_relationships.json"):
                with open(f"{filepath}_relationships.json", 'r') as f:
                    self.relationship_index = json.load(f)
        
        elif self.db_type == "rdf":
            if os.path.exists(f"{filepath}.ttl"):
                self.graph.parse(f"{filepath}.ttl", format="turtle")


class KnowledgeGraphAgent:
    """Main knowledge graph agent that manages entities and relationships"""
    
    def __init__(self, agent_id: str = "knowledge-graph-agent"):
        self.agent_id = agent_id
        self.client = AMPClient(agent_id)
        
        # Initialize components
        self.entity_extractor = EntityExtractor()
        self.graph_db = GraphDatabase()
        
        # Configuration
        self.graph_filepath = "./data/knowledge_graph"
        
        logger.info(f"Knowledge Graph Agent {agent_id} initialized")
    
    async def start(self, host: str = "localhost", port: int = 8000):
        """Start the agent and register capabilities"""
        await self.client.connect(f"ws://{host}:{port}/ws")
        
        # Load existing graph if available
        await self._load_graph()
        
        # Register capabilities
        capabilities = [
            {
                "name": "extract-entities",
                "description": "Extract entities from text chunks",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunks": {"type": "array"},
                        "extract_relationships": {"type": "boolean", "default": True}
                    },
                    "required": ["chunks"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "entities": {"type": "array"},
                        "relationships": {"type": "array"},
                        "statistics": {"type": "object"}
                    }
                }
            },
            {
                "name": "graph-search",
                "description": "Search entities and relationships in the knowledge graph",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "entity_type": {"type": "string"},
                        "max_results": {"type": "integer", "default": 10}
                    }
                }
            },
            {
                "name": "find-related",
                "description": "Find entities related to a given entity",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "max_depth": {"type": "integer", "default": 2},
                        "relationship_types": {"type": "array"}
                    },
                    "required": ["entity_id"]
                }
            },
            {
                "name": "find-path",
                "description": "Find shortest path between two entities",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "entity1_id": {"type": "string"},
                        "entity2_id": {"type": "string"}
                    },
                    "required": ["entity1_id", "entity2_id"]
                }
            },
            {
                "name": "graph-analytics",
                "description": "Get graph analytics and statistics",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "analysis_type": {"type": "string", "enum": ["statistics", "centrality", "communities"]}
                    }
                }
            }
        ]
        
        for capability in capabilities:
            await self.client.register_capability(capability)
        
        # Start message handling
        await self.client.start_message_handler(self._handle_message)
        logger.info(f"Knowledge Graph Agent started on {host}:{port}")
    
    async def _handle_message(self, message: AMPMessage):
        """Handle incoming AMP messages"""
        try:
            capability = message.message.destination.capability
            payload = message.message.payload
            
            if capability == "extract-entities":
                result = await self._handle_extract_entities(payload)
            elif capability == "graph-search":
                result = await self._handle_graph_search(payload)
            elif capability == "find-related":
                result = await self._handle_find_related(payload)
            elif capability == "find-path":
                result = await self._handle_find_path(payload)
            elif capability == "graph-analytics":
                result = await self._handle_graph_analytics(payload)
            else:
                raise ValueError(f"Unknown capability: {capability}")
            
            # Send response
            await self.client.send_response(
                message.message.id,
                message.message.source.agent_id,
                result
            )
            
        except Exception as e:
            logger.error(f"Error handling message: {e}")
            await self.client.send_error(
                message.message.id,
                message.message.source.agent_id,
                str(e),
                "GRAPH_ERROR"
            )
    
    async def _handle_extract_entities(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle entity extraction from chunks"""
        chunks = payload['chunks']
        extract_relationships = payload.get('extract_relationships', True)
        
        all_entities = []
        all_relationships = []
        
        for chunk in chunks:
            chunk_id = chunk['id']
            text = chunk['text']
            
            # Extract entities
            entity_result = await self.entity_extractor.extract_entities(text, chunk_id)
            entities = entity_result['entities']
            
            # Add entities to graph
            for entity in entities:
                entity_id = await self.graph_db.add_entity(entity)
                entity['graph_id'] = entity_id
                all_entities.append(entity)
            
            # Extract relationships if requested
            if extract_relationships and len(entities) > 1:
                relationships = await self.entity_extractor.extract_relationships(text, entities)
                
                for relationship in relationships:
                    rel_id = await self.graph_db.add_relationship(relationship)
                    relationship['graph_id'] = rel_id
                    all_relationships.append(relationship)
        
        # Save graph
        await self._save_graph()
        
        # Get updated statistics
        stats = await self.graph_db.get_graph_statistics()
        
        return {
            "entities": all_entities,
            "relationships": all_relationships,
            "statistics": {
                "processed_chunks": len(chunks),
                "extracted_entities": len(all_entities),
                "extracted_relationships": len(all_relationships),
                "graph_stats": stats
            }
        }
    
    async def _handle_graph_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph search request"""
        query = payload.get('query', '')
        entity_type = payload.get('entity_type')
        max_results = payload.get('max_results', 10)
        
        # Search entities
        entities = await self.graph_db.find_entities(
            entity_type=entity_type,
            text_contains=query if query else None
        )
        
        # Limit results
        entities = entities[:max_results]
        
        return {
            "query": query,
            "entity_type": entity_type,
            "results": entities,
            "total_found": len(entities)
        }
    
    async def _handle_find_related(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find related entities request"""
        entity_id = payload['entity_id']
        max_depth = payload.get('max_depth', 2)
        
        # Find related entities
        related_entities = await self.graph_db.find_related_entities(entity_id, max_depth)
        
        return {
            "entity_id": entity_id,
            "max_depth": max_depth,
            "related_entities": related_entities,
            "total_found": len(related_entities)
        }
    
    async def _handle_find_path(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle find path request"""
        entity1_id = payload['entity1_id']
        entity2_id = payload['entity2_id']
        
        # Find shortest path
        path = await self.graph_db.find_shortest_path(entity1_id, entity2_id)
        
        return {
            "entity1_id": entity1_id,
            "entity2_id": entity2_id,
            "path": path,
            "path_length": len(path)
        }
    
    async def _handle_graph_analytics(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle graph analytics request"""
        analysis_type = payload.get('analysis_type', 'statistics')
        
        if analysis_type == 'statistics':
            stats = await self.graph_db.get_graph_statistics()
            return {
                "analysis_type": analysis_type,
                "statistics": stats
            }
        
        elif analysis_type == 'centrality':
            # Calculate centrality measures for NetworkX
            if self.graph_db.db_type == "networkx":
                centrality_measures = {
                    'degree_centrality': nx.degree_centrality(self.graph_db.graph),
                    'betweenness_centrality': nx.betweenness_centrality(self.graph_db.graph),
                    'closeness_centrality': nx.closeness_centrality(self.graph_db.graph)
                }
                
                # Get top entities for each measure
                top_entities = {}
                for measure, scores in centrality_measures.items():
                    top_entities[measure] = sorted(
                        scores.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10]
                
                return {
                    "analysis_type": analysis_type,
                    "centrality_measures": top_entities
                }
        
        elif analysis_type == 'communities':
            # Detect communities for NetworkX
            if self.graph_db.db_type == "networkx":
                try:
                    import networkx.algorithms.community as nx_comm
                    communities = list(nx_comm.greedy_modularity_communities(
                        self.graph_db.graph.to_undirected()
                    ))
                    
                    community_info = []
                    for i, community in enumerate(communities):
                        community_info.append({
                            'id': i,
                            'size': len(community),
                            'entities': list(community)
                        })
                    
                    return {
                        "analysis_type": analysis_type,
                        "communities": community_info,
                        "total_communities": len(communities)
                    }
                except ImportError:
                    return {
                        "analysis_type": analysis_type,
                        "error": "Community detection not available"
                    }
        
        return {
            "analysis_type": analysis_type,
            "error": "Unknown analysis type"
        }
    
    async def _save_graph(self):
        """Save knowledge graph to disk"""
        try:
            os.makedirs(os.path.dirname(self.graph_filepath), exist_ok=True)
            await self.graph_db.save_graph(self.graph_filepath)
            logger.info("Knowledge graph saved successfully")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {e}")
    
    async def _load_graph(self):
        """Load knowledge graph from disk"""
        try:
            await self.graph_db.load_graph(self.graph_filepath)
            logger.info("Knowledge graph loaded successfully")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")
    
    async def stop(self):
        """Stop the agent"""
        await self._save_graph()
        await self.client.disconnect()
        logger.info("Knowledge Graph Agent stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Knowledge Graph Agent")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=8000, help="Port to connect to")
    parser.add_argument("--agent-id", default="knowledge-graph-agent", help="Agent ID")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = KnowledgeGraphAgent(args.agent_id)
    
    try:
        asyncio.run(agent.start(args.host, args.port))
    except KeyboardInterrupt:
        asyncio.run(agent.stop())