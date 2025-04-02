import redis
import numpy as np
import json
from typing import List, Dict, Any, Optional
from .base_db import BaseDB, SearchResult

class RedisDB(BaseDB):
    """Redis implementation for vector storage."""
    
    def __init__(self, collection_name: str = "default"):
        """Initialize Redis.
        
        Args:
            collection_name: Name of the collection to use
        """
        super().__init__()
        self.client = redis.Redis(host='localhost', port=6379, db=0)
        self.collection_name = collection_name
        self.embedding_dim = None  # Will be set on first insert
    
    def index(self, embeddings: List[List[float]], chunks: List[str], 
              doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index embeddings with their associated chunks and metadata.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        # Set embedding dimension on first insert if not set
        if self.embedding_dim is None:
            self.embedding_dim = len(embeddings[0])
        else:
            # Verify all embeddings have the same dimension
            for emb in embeddings:
                if len(emb) != self.embedding_dim:
                    raise ValueError(f"Embedding dimension mismatch. Expected {self.embedding_dim}, got {len(emb)}")
        
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
        
        # Store each embedding and its metadata
        for i, (emb, chunk, doc_id) in enumerate(zip(embeddings, chunks, doc_ids)):
            # Prepare metadata
            point_metadata = metadata[i] if metadata else {}
            point_metadata['chunk'] = chunk
            point_metadata['embedding_dim'] = self.embedding_dim  # Store dimension in metadata
            
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
        # Convert query to numpy array
        query_embedding = np.array(query_embedding)
        
        # Verify query dimension matches stored dimension
        if self.embedding_dim is not None and len(query_embedding) != self.embedding_dim:
            raise ValueError(f"Query embedding dimension mismatch. Expected {self.embedding_dim}, got {len(query_embedding)}")
        
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
        self.embedding_dim = None  # Reset embedding dimension 