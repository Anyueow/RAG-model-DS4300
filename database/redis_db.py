import redis
import numpy as np
import json
from typing import List, Dict, Any, Optional
from .base_db import BaseDB, SearchResult

class RedisDB(BaseDB):
    """Redis implementation for vector storage."""
    
    def __init__(self, collection_name: str = "default", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        """Initialize Redis.
        
        Args:
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        super().__init__()
        self.client = redis.Redis(host='localhost', port=6379, db=0)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Get vector size based on model
        if "minilm" in embedding_model.lower():
            self.vector_size = 384
        elif "mpnet" in embedding_model.lower():
            self.vector_size = 768
        else:  # default to nomic-embed-text-v1.5
            self.vector_size = 768
    
    def index(self, embeddings: List[List[float]], chunks: List[str], 
              doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index embeddings with their associated chunks and metadata.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        # Debug: Print information about embeddings
        print(f"\n[DEBUG] Indexing {len(embeddings)} embeddings")
        print(f"[DEBUG] First embedding type: {type(embeddings[0])}")
        print(f"[DEBUG] First embedding shape: {np.array(embeddings[0]).shape}")
        
        # Verify embedding dimensions
        for emb in embeddings:
            if len(emb) != self.vector_size:
                raise ValueError(f"Embedding dimension mismatch. Expected {self.vector_size}, got {len(emb)}")
        
        super().index(embeddings, chunks, doc_ids, metadata)
    
    def _index_impl(self, embeddings: List[List[float]], chunks: List[str], 
                   doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index embeddings using Redis.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        # Convert embeddings to numpy arrays if needed
        embeddings = [np.array(emb) for emb in embeddings]
        
        # Debug: Print numpy array information
        print(f"[DEBUG] First numpy array type: {type(embeddings[0])}")
        print(f"[DEBUG] First numpy array shape: {embeddings[0].shape}")
        
        # Store each embedding and its metadata
        for i, (emb, chunk, doc_id) in enumerate(zip(embeddings, chunks, doc_ids)):
            # Prepare metadata
            point_metadata = metadata[i] if metadata else {}
            point_metadata['chunk'] = chunk
            point_metadata['embedding_model'] = self.embedding_model
            point_metadata['vector_size'] = self.vector_size
            
            # Debug: Print point information
            print(f"\n[DEBUG] Creating point {i}")
            print(f"[DEBUG] Embedding type: {type(emb)}")
            print(f"[DEBUG] Embedding shape: {emb.shape}")
            print(f"[DEBUG] Metadata: {point_metadata}")
            
            # Store vector and metadata
            key = f"{self.collection_name}:{doc_id}:{i}"
            self.client.set(
                f"{key}:vector",
                emb.tobytes()
            )
            self.client.set(
                f"{key}:metadata",
                json.dumps(point_metadata)
            )
    
    def _search_impl(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        """Search for similar vectors using Redis.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of SearchResult objects containing matches
        """
        # Debug: Print query information
        print(f"\n[DEBUG] Search query type: {type(query_embedding)}")
        print(f"[DEBUG] Search query shape: {np.array(query_embedding).shape}")
        
        # Verify query dimension
        if len(query_embedding) != self.vector_size:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.vector_size}, got {len(query_embedding)}")
        
        # Convert query to numpy array
        query_embedding = np.array(query_embedding)
        
        # Debug: Print numpy query information
        print(f"[DEBUG] Numpy query type: {type(query_embedding)}")
        print(f"[DEBUG] Numpy query shape: {query_embedding.shape}")
        
        # Get all keys in collection
        keys = self.client.keys(f"{self.collection_name}:*:vector")
        
        # Calculate similarities
        similarities = []
        for key in keys:
            # Get vector
            vector_bytes = self.client.get(key)
            if vector_bytes is None:
                continue
                
            vector = np.frombuffer(vector_bytes)
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, vector) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(vector)
            )
            
            # Get metadata
            metadata_key = key.replace(":vector", ":metadata")
            metadata_bytes = self.client.get(metadata_key)
            if metadata_bytes is None:
                continue
                
            metadata = json.loads(metadata_bytes)
            
            # Extract doc_id from key
            doc_id = key.decode().split(":")[1]
            
            similarities.append((
                doc_id,
                metadata['chunk'],
                similarity,
                metadata
            ))
        
        # Sort by similarity and get top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        top_k = similarities[:k]
        
        # Debug: Print search results
        print(f"\n[DEBUG] Search results count: {len(top_k)}")
        if top_k:
            print(f"[DEBUG] First result score: {top_k[0][2]}")
            print(f"[DEBUG] First result metadata: {top_k[0][3]}")
        
        # Convert to SearchResult objects
        return [
            SearchResult(
                doc_id=doc_id,
                chunk=chunk,
                score=float(score),
                metadata=metadata
            )
            for doc_id, chunk, score, metadata in top_k
        ]
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        keys = self.client.keys(f"{self.collection_name}:*")
        if keys:
            self.client.delete(*keys) 