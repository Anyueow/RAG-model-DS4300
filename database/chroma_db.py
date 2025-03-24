from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from .base_db import BaseVectorDB

class ChromaDB(BaseVectorDB):
    """ChromaDB adapter implementation."""
    
    def __init__(self, 
                 collection_name: str = "course_notes",
                 persist_directory: str = "chroma_db"):
        """Initialize the ChromaDB adapter.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Directory to persist the database
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )
    
    def add_vectors(self, 
                   vectors: np.ndarray, 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add vectors to ChromaDB.
        
        Args:
            vectors: Array of vectors to add
            metadata: Optional list of metadata dictionaries for each vector
        """
        # Generate IDs for the vectors
        ids = [f"vec_{i}" for i in range(len(vectors))]
        
        # Convert vectors to list format for ChromaDB
        vectors_list = vectors.tolist()
        
        # Add vectors to collection
        self.collection.add(
            embeddings=vectors_list,
            documents=[meta.get('text', '') for meta in (metadata or [])],
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors in ChromaDB.
        
        Args:
            query_vector: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results and their metadata
        """
        # Convert query vector to list format
        query_vector_list = query_vector.tolist()
        
        # Search for similar vectors
        results = self.collection.query(
            query_embeddings=[query_vector_list],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else None
            })
        
        return formatted_results
    
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
        """
        self.collection.delete(ids=ids)
    
    def get_vector_count(self) -> int:
        """Get the total number of vectors in the database.
        
        Returns:
            int: Number of vectors
        """
        return self.collection.count()
    
    def clear(self) -> None:
        """Clear all vectors from the database."""
        self.collection.delete(where={}) 