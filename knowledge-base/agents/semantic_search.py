"""
Semantic Search Agent

This agent handles vector search, similarity matching, and retrieval
from the knowledge base using advanced embedding techniques.
"""

import asyncio
import json
import logging
import os
import pickle
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'shared-lib'))
from amp_client import AMPClient
from amp_types import AMPMessage, MessageType

logger = logging.getLogger(__name__)


class VectorIndex:
    """Manages vector indexing using FAISS for fast similarity search"""
    
    def __init__(self, dimension: int = 384, index_type: str = "HNSW"):
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.metadata_store = {}
        self.id_to_idx = {}
        self.idx_to_id = {}
        self.next_idx = 0
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_type == "HNSW":
            # Hierarchical Navigable Small World for high-quality search
            self.index = faiss.IndexHNSWFlat(self.dimension, 32)
            self.index.hnsw.efSearch = 64
        elif self.index_type == "IVF":
            # Inverted File for large-scale search
            quantizer = faiss.IndexFlatL2(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, 100)
        else:
            # Default to flat L2 index
            self.index = faiss.IndexFlatL2(self.dimension)
    
    async def add_vectors(self, vectors: np.ndarray, ids: List[str], metadata: List[Dict[str, Any]]):
        """Add vectors to the index"""
        if len(vectors) != len(ids) or len(ids) != len(metadata):
            raise ValueError("Vectors, IDs, and metadata must have the same length")
        
        # Normalize vectors for cosine similarity
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        
        # Add to FAISS index
        start_idx = self.next_idx
        self.index.add(vectors.astype(np.float32))
        
        # Update mappings and metadata
        for i, (vector_id, meta) in enumerate(zip(ids, metadata)):
            idx = start_idx + i
            self.id_to_idx[vector_id] = idx
            self.idx_to_id[idx] = vector_id
            self.metadata_store[vector_id] = meta
        
        self.next_idx += len(vectors)
        
        # Train index if needed
        if hasattr(self.index, 'is_trained') and not self.index.is_trained:
            if self.index.ntotal >= 100:  # Need at least 100 vectors for IVF training
                self.index.train(self._get_all_vectors())
    
    async def search(self, query_vector: np.ndarray, k: int = 10, threshold: float = 0.0) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Search for similar vectors"""
        if self.index.ntotal == 0:
            return []
        
        # Normalize query vector
        query_vector = query_vector / np.linalg.norm(query_vector)
        query_vector = query_vector.reshape(1, -1).astype(np.float32)
        
        # Search
        distances, indices = self.index.search(query_vector, min(k, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx == -1:  # Invalid result
                continue
                
            vector_id = self.idx_to_id[idx]
            # Convert L2 distance to cosine similarity
            similarity = 1 - (distance ** 2) / 2
            
            if similarity >= threshold:
                metadata = self.metadata_store[vector_id]
                results.append((vector_id, similarity, metadata))
        
        return results
    
    def _get_all_vectors(self) -> np.ndarray:
        """Get all vectors from the index (for training)"""
        vectors = []
        for i in range(self.index.ntotal):
            vector = self.index.reconstruct(i)
            vectors.append(vector)
        return np.array(vectors)
    
    async def save(self, filepath: str):
        """Save index to disk"""
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata
        metadata = {
            'dimension': self.dimension,
            'index_type': self.index_type,
            'metadata_store': self.metadata_store,
            'id_to_idx': self.id_to_idx,
            'idx_to_id': self.idx_to_id,
            'next_idx': self.next_idx
        }
        
        with open(f"{filepath}.metadata", 'wb') as f:
            pickle.dump(metadata, f)
    
    async def load(self, filepath: str):
        """Load index from disk"""
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.metadata", 'rb') as f:
            metadata = pickle.load(f)
        
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        self.metadata_store = metadata['metadata_store']
        self.id_to_idx = metadata['id_to_idx']
        self.idx_to_id = metadata['idx_to_id']
        self.next_idx = metadata['next_idx']


class ChromaDBClient:
    """Client for ChromaDB vector database"""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = persist_directory
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collections = {}
    
    async def get_or_create_collection(self, name: str, embedding_function=None) -> chromadb.Collection:
        """Get or create a collection"""
        if name not in self.collections:
            try:
                collection = self.client.get_collection(name)
            except Exception:
                collection = self.client.create_collection(
                    name=name,
                    embedding_function=embedding_function
                )
            self.collections[name] = collection
        
        return self.collections[name]
    
    async def add_documents(self, collection_name: str, documents: List[Dict[str, Any]]):
        """Add documents to a collection"""
        collection = await self.get_or_create_collection(collection_name)
        
        ids = [doc['id'] for doc in documents]
        texts = [doc['text'] for doc in documents]
        embeddings = [doc['embedding'] for doc in documents] if 'embedding' in documents[0] else None
        metadatas = [doc.get('metadata', {}) for doc in documents]
        
        if embeddings:
            collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas
            )
        else:
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
    
    async def search(self, collection_name: str, query_texts: List[str] = None, 
                    query_embeddings: List[List[float]] = None, n_results: int = 10,
                    where: Dict[str, Any] = None) -> Dict[str, Any]:
        """Search in a collection"""
        collection = await self.get_or_create_collection(collection_name)
        
        return collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )


class HybridSearch:
    """Combines vector search with keyword search for better results"""
    
    def __init__(self, vector_weight: float = 0.7, keyword_weight: float = 0.3):
        self.vector_weight = vector_weight
        self.keyword_weight = keyword_weight
    
    async def search(self, query: str, vector_results: List[Tuple[str, float, Dict[str, Any]]], 
                    keyword_results: List[Tuple[str, float, Dict[str, Any]]]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Combine vector and keyword search results"""
        # Create score maps
        vector_scores = {result[0]: result[1] for result in vector_results}
        keyword_scores = {result[0]: result[1] for result in keyword_results}
        
        # Get all unique document IDs
        all_ids = set(vector_scores.keys()) | set(keyword_scores.keys())
        
        # Calculate hybrid scores
        hybrid_results = []
        for doc_id in all_ids:
            vector_score = vector_scores.get(doc_id, 0)
            keyword_score = keyword_scores.get(doc_id, 0)
            
            hybrid_score = (self.vector_weight * vector_score + 
                          self.keyword_weight * keyword_score)
            
            # Get metadata from either source
            metadata = None
            for result in vector_results + keyword_results:
                if result[0] == doc_id:
                    metadata = result[2]
                    break
            
            if metadata:
                hybrid_results.append((doc_id, hybrid_score, metadata))
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x[1], reverse=True)
        return hybrid_results


class QueryProcessor:
    """Processes and enhances queries for better search results"""
    
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    async def process_query(self, query: str, query_type: str = "semantic") -> Dict[str, Any]:
        """Process and enhance query"""
        processed_query = {
            'original': query,
            'type': query_type,
            'processed_at': datetime.utcnow().isoformat()
        }
        
        if query_type == "semantic":
            # Generate embedding for semantic search
            embedding = self.model.encode(query)
            processed_query['embedding'] = embedding.tolist()
            processed_query['embedding_dimension'] = len(embedding)
        
        elif query_type == "keyword":
            # Extract keywords for keyword search
            keywords = self._extract_keywords(query)
            processed_query['keywords'] = keywords
        
        elif query_type == "hybrid":
            # Both semantic and keyword processing
            embedding = self.model.encode(query)
            keywords = self._extract_keywords(query)
            
            processed_query['embedding'] = embedding.tolist()
            processed_query['embedding_dimension'] = len(embedding)
            processed_query['keywords'] = keywords
        
        return processed_query
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text"""
        # Simple keyword extraction (could be enhanced with NLP)
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        # Filter out common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return list(set(keywords))


class SemanticSearchAgent:
    """Main semantic search agent that handles vector search and retrieval"""
    
    def __init__(self, agent_id: str = "semantic-search-agent"):
        self.agent_id = agent_id
        self.client = AMPClient(agent_id)
        
        # Initialize components
        self.vector_index = VectorIndex()
        self.chroma_client = ChromaDBClient()
        self.hybrid_search = HybridSearch()
        self.query_processor = QueryProcessor()
        
        # Configuration
        self.index_filepath = "./data/vector_index"
        self.default_collection = "knowledge_base"
        
        logger.info(f"Semantic Search Agent {agent_id} initialized")
    
    async def start(self, host: str = "localhost", port: int = 8000):
        """Start the agent and register capabilities"""
        await self.client.connect(f"ws://{host}:{port}/ws")
        
        # Load existing index if available
        await self._load_index()
        
        # Register capabilities
        capabilities = [
            {
                "name": "semantic-search",
                "description": "Perform semantic search using vector embeddings",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                        "threshold": {"type": "number", "default": 0.0},
                        "filters": {"type": "object"}
                    },
                    "required": ["query"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "results": {"type": "array"},
                        "query_info": {"type": "object"},
                        "search_stats": {"type": "object"}
                    }
                }
            },
            {
                "name": "hybrid-search",
                "description": "Perform hybrid search combining vector and keyword search",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "k": {"type": "integer", "default": 10},
                        "vector_weight": {"type": "number", "default": 0.7},
                        "keyword_weight": {"type": "number", "default": 0.3}
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "index-chunks",
                "description": "Index text chunks for semantic search",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "chunks": {"type": "array"},
                        "collection": {"type": "string"}
                    },
                    "required": ["chunks"]
                }
            },
            {
                "name": "similarity-search",
                "description": "Find similar documents to a given document",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "document_id": {"type": "string"},
                        "k": {"type": "integer", "default": 5}
                    },
                    "required": ["document_id"]
                }
            }
        ]
        
        for capability in capabilities:
            await self.client.register_capability(capability)
        
        # Start message handling
        await self.client.start_message_handler(self._handle_message)
        logger.info(f"Semantic Search Agent started on {host}:{port}")
    
    async def _handle_message(self, message: AMPMessage):
        """Handle incoming AMP messages"""
        try:
            capability = message.message.destination.capability
            payload = message.message.payload
            
            if capability == "semantic-search":
                result = await self._handle_semantic_search(payload)
            elif capability == "hybrid-search":
                result = await self._handle_hybrid_search(payload)
            elif capability == "index-chunks":
                result = await self._handle_index_chunks(payload)
            elif capability == "similarity-search":
                result = await self._handle_similarity_search(payload)
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
                "SEARCH_ERROR"
            )
    
    async def _handle_semantic_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle semantic search request"""
        query = payload['query']
        k = payload.get('k', 10)
        threshold = payload.get('threshold', 0.0)
        filters = payload.get('filters', {})
        
        start_time = datetime.utcnow()
        
        # Process query
        query_info = await self.query_processor.process_query(query, "semantic")
        query_embedding = np.array(query_info['embedding'])
        
        # Search vector index
        vector_results = await self.vector_index.search(query_embedding, k, threshold)
        
        # Apply filters if specified
        if filters:
            vector_results = self._apply_filters(vector_results, filters)
        
        # Format results
        formatted_results = []
        for result_id, similarity, metadata in vector_results:
            formatted_results.append({
                'id': result_id,
                'similarity': similarity,
                'text': metadata.get('text', ''),
                'metadata': metadata,
                'score': similarity
            })
        
        end_time = datetime.utcnow()
        search_duration = (end_time - start_time).total_seconds()
        
        return {
            "results": formatted_results,
            "query_info": query_info,
            "search_stats": {
                "total_results": len(formatted_results),
                "search_duration_seconds": search_duration,
                "index_size": self.vector_index.index.ntotal
            }
        }
    
    async def _handle_hybrid_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle hybrid search request"""
        query = payload['query']
        k = payload.get('k', 10)
        vector_weight = payload.get('vector_weight', 0.7)
        keyword_weight = payload.get('keyword_weight', 0.3)
        
        start_time = datetime.utcnow()
        
        # Update hybrid search weights
        self.hybrid_search.vector_weight = vector_weight
        self.hybrid_search.keyword_weight = keyword_weight
        
        # Process query for hybrid search
        query_info = await self.query_processor.process_query(query, "hybrid")
        
        # Vector search
        query_embedding = np.array(query_info['embedding'])
        vector_results = await self.vector_index.search(query_embedding, k * 2)  # Get more results for hybrid
        
        # Keyword search using ChromaDB
        chroma_results = await self.chroma_client.search(
            self.default_collection,
            query_texts=[query],
            n_results=k * 2
        )
        
        # Convert ChromaDB results to standard format
        keyword_results = []
        if chroma_results['ids']:
            for i, doc_id in enumerate(chroma_results['ids'][0]):
                score = 1.0 - chroma_results['distances'][0][i]  # Convert distance to similarity
                metadata = chroma_results['metadatas'][0][i] if chroma_results['metadatas'] else {}
                metadata['text'] = chroma_results['documents'][0][i]
                keyword_results.append((doc_id, score, metadata))
        
        # Combine results
        hybrid_results = await self.hybrid_search.search(query, vector_results, keyword_results)
        
        # Take top k results
        hybrid_results = hybrid_results[:k]
        
        # Format results
        formatted_results = []
        for result_id, score, metadata in hybrid_results:
            formatted_results.append({
                'id': result_id,
                'score': score,
                'text': metadata.get('text', ''),
                'metadata': metadata
            })
        
        end_time = datetime.utcnow()
        search_duration = (end_time - start_time).total_seconds()
        
        return {
            "results": formatted_results,
            "query_info": query_info,
            "search_stats": {
                "total_results": len(formatted_results),
                "search_duration_seconds": search_duration,
                "search_type": "hybrid",
                "vector_weight": vector_weight,
                "keyword_weight": keyword_weight
            }
        }
    
    async def _handle_index_chunks(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle chunk indexing request"""
        chunks = payload['chunks']
        collection = payload.get('collection', self.default_collection)
        
        if not chunks:
            return {"status": "no_chunks", "indexed_count": 0}
        
        start_time = datetime.utcnow()
        
        # Extract vectors and metadata for FAISS index
        vectors = []
        ids = []
        metadata_list = []
        
        for chunk in chunks:
            if 'embedding' in chunk and chunk['embedding']:
                vectors.append(chunk['embedding'])
                ids.append(chunk['id'])
                metadata_list.append(chunk)
        
        if vectors:
            # Add to FAISS index
            vectors_array = np.array(vectors)
            await self.vector_index.add_vectors(vectors_array, ids, metadata_list)
            
            # Save index
            await self._save_index()
        
        # Add to ChromaDB
        chroma_documents = []
        for chunk in chunks:
            chroma_doc = {
                'id': chunk['id'],
                'text': chunk['text'],
                'metadata': {k: v for k, v in chunk.items() if k not in ['id', 'text', 'embedding']}
            }
            if 'embedding' in chunk:
                chroma_doc['embedding'] = chunk['embedding']
            chroma_documents.append(chroma_doc)
        
        await self.chroma_client.add_documents(collection, chroma_documents)
        
        end_time = datetime.utcnow()
        indexing_duration = (end_time - start_time).total_seconds()
        
        return {
            "status": "indexed",
            "indexed_count": len(chunks),
            "collection": collection,
            "indexing_duration_seconds": indexing_duration
        }
    
    async def _handle_similarity_search(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Handle similarity search request"""
        document_id = payload['document_id']
        k = payload.get('k', 5)
        
        # Get document embedding from index
        if document_id not in self.vector_index.metadata_store:
            raise ValueError(f"Document {document_id} not found in index")
        
        document_metadata = self.vector_index.metadata_store[document_id]
        if 'embedding' not in document_metadata:
            raise ValueError(f"Document {document_id} has no embedding")
        
        document_embedding = np.array(document_metadata['embedding'])
        
        # Search for similar documents
        results = await self.vector_index.search(document_embedding, k + 1)  # +1 to exclude self
        
        # Remove the document itself from results
        filtered_results = [(r_id, score, meta) for r_id, score, meta in results if r_id != document_id]
        
        # Format results
        formatted_results = []
        for result_id, similarity, metadata in filtered_results[:k]:
            formatted_results.append({
                'id': result_id,
                'similarity': similarity,
                'text': metadata.get('text', ''),
                'metadata': metadata
            })
        
        return {
            "query_document_id": document_id,
            "similar_documents": formatted_results,
            "total_found": len(formatted_results)
        }
    
    def _apply_filters(self, results: List[Tuple[str, float, Dict[str, Any]]], 
                      filters: Dict[str, Any]) -> List[Tuple[str, float, Dict[str, Any]]]:
        """Apply filters to search results"""
        filtered_results = []
        
        for result_id, similarity, metadata in results:
            include = True
            
            for filter_key, filter_value in filters.items():
                if filter_key in metadata:
                    if isinstance(filter_value, dict):
                        # Range or operator filters
                        if '$gte' in filter_value and metadata[filter_key] < filter_value['$gte']:
                            include = False
                            break
                        if '$lte' in filter_value and metadata[filter_key] > filter_value['$lte']:
                            include = False
                            break
                        if '$in' in filter_value and metadata[filter_key] not in filter_value['$in']:
                            include = False
                            break
                    else:
                        # Exact match
                        if metadata[filter_key] != filter_value:
                            include = False
                            break
                else:
                    include = False
                    break
            
            if include:
                filtered_results.append((result_id, similarity, metadata))
        
        return filtered_results
    
    async def _save_index(self):
        """Save vector index to disk"""
        try:
            os.makedirs(os.path.dirname(self.index_filepath), exist_ok=True)
            await self.vector_index.save(self.index_filepath)
            logger.info("Vector index saved successfully")
        except Exception as e:
            logger.error(f"Error saving vector index: {e}")
    
    async def _load_index(self):
        """Load vector index from disk"""
        try:
            if os.path.exists(f"{self.index_filepath}.faiss"):
                await self.vector_index.load(self.index_filepath)
                logger.info("Vector index loaded successfully")
            else:
                logger.info("No existing vector index found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading vector index: {e}")
    
    async def stop(self):
        """Stop the agent"""
        await self._save_index()
        await self.client.disconnect()
        logger.info("Semantic Search Agent stopped")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Semantic Search Agent")
    parser.add_argument("--host", default="localhost", help="Host to connect to")
    parser.add_argument("--port", type=int, default=8000, help="Port to connect to")
    parser.add_argument("--agent-id", default="semantic-search-agent", help="Agent ID")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    agent = SemanticSearchAgent(args.agent_id)
    
    try:
        asyncio.run(agent.start(args.host, args.port))
    except KeyboardInterrupt:
        asyncio.run(agent.stop())