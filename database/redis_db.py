from typing import List, Dict, Any, Optional
import numpy as np
import redis
import json
from .base_db import BaseVectorDB

class RedisVectorDB(BaseVectorDB):
    """Redis Vector DB adapter implementation."""
    
    def __init__(self, 
                 host: str = "localhost",
                 port: int = 6379,
                 db: int = 0,
                 key_prefix: str = "vec:"):
        """Initialize the Redis Vector DB adapter.
        
        Args:
            host: Redis host
            port: Redis port
            db: Redis database number
            key_prefix: Prefix for vector keys
        """
        self.redis_client = redis.Redis(
            host=host,
            port=port,
            db=db,
            decode_responses=True
        )
        self.key_prefix = key_prefix
    
    def add_vectors(self, 
                   vectors: np.ndarray, 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add vectors to Redis Vector DB.
        
        Args:
            vectors: Array of vectors to add
            metadata: Optional list of metadata dictionaries for each vector
        """
        for i, vector in enumerate(vectors):
            # Generate key for the vector
            key = f"{self.key_prefix}{i}"
            
            # Store vector data
            vector_data = {
                'vector': vector.tolist(),
                'metadata': metadata[i] if metadata and i < len(metadata) else {}
            }
            
            # Store in Redis
            self.redis_client.set(key, json.dumps(vector_data))
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in Redis Vector DB.
        
        Args:
            query_vector: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results and their metadata
        """
        results = []
        
        # Get all vector keys
        keys = self.redis_client.keys(f"{self.key_prefix}*")
        
        # Calculate cosine similarity with all vectors
        similarities = []
        for key in keys:
            vector_data = json.loads(self.redis_client.get(key))
            stored_vector = np.array(vector_data['vector'])
            
            # Calculate cosine similarity
            similarity = np.dot(query_vector, stored_vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(stored_vector)
            )
            
            similarities.append((similarity, vector_data['metadata']))
        
        # Sort by similarity and get top k results
        similarities.sort(reverse=True)
        top_k = similarities[:k]
        
        # Format results
        for similarity, metadata in top_k:
            results.append({
                'text': metadata.get('text', ''),
                'metadata': metadata,
                'distance': 1 - similarity  # Convert similarity to distance
            })
        
        return results
    
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
        """
        for vector_id in ids:
            key = f"{self.key_prefix}{vector_id}"
            self.redis_client.delete(key)
    
    def get_vector_count(self) -> int:
        """Get the total number of vectors in the database.
        
        Returns:
            int: Number of vectors
        """
        return len(self.redis_client.keys(f"{self.key_prefix}*"))
    
    def clear(self) -> None:
        """Clear all vectors from the database."""
        keys = self.redis_client.keys(f"{self.key_prefix}*")
        if keys:
            self.redis_client.delete(*keys) 