from typing import List, Dict, Any, Optional
import numpy as np
import chromadb
from chromadb.config import Settings
from .base_db import BaseVectorDB
import time
import random

class ChromaDB(BaseVectorDB):
    """ChromaDB adapter implementation with multimodal support."""
    
    def __init__(self, 
                 collection_name: str = "course_notes",
                 persist_directory: str = "chroma_db"):
        """Initialize the ChromaDB adapter.
        
        Args:
            collection_name: Base name for collections
            persist_directory: Directory to persist the database
        """
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Create separate collections for text and images
        self.text_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_text"
        )
        self.image_collection = self.client.get_or_create_collection(
            name=f"{collection_name}_images"
        )
    
    def add_vectors(self, 
                   vectors: List[np.ndarray], 
                   metadata: Optional[List[Dict[str, Any]]] = None,
                   modality: str = "text") -> None:
        """Add vectors to ChromaDB.
        
        Args:
            vectors: List of vectors to add
            metadata: Optional list of metadata dictionaries for each vector
            modality: Type of vectors ('text' or 'image')
        """
        # Generate unique IDs using timestamp and random numbers
        ids = [f"{modality}_{int(time.time())}_{random.randint(1000, 9999)}_{i}" 
               for i in range(len(vectors))]
        
        # Convert vectors to list format for ChromaDB
        vectors_list = [v.tolist() for v in vectors]
        
        # Select appropriate collection
        collection = self.text_collection if modality == "text" else self.image_collection
        
        # Add vectors to collection
        collection.add(
            embeddings=vectors_list,
            documents=[meta.get('text', '') for meta in (metadata or [])],
            metadatas=metadata,
            ids=ids
        )
    
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5,
               modality: str = "text") -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            modality: Type of query ('text' or 'image')
            
        Returns:
            List of results with metadata and distances
        """
        # Select appropriate collection
        collection = self.text_collection if modality == "text" else self.image_collection
        
        # Convert query vector to list
        query_list = query_vector.tolist()
        
        # Search in collection
        results = collection.query(
            query_embeddings=[query_list],
            n_results=k
        )
        
        # Format results
        formatted_results = []
        if results['distances'] and results['metadatas']:
            for i in range(len(results['ids'][0])):
                formatted_results.append({
                    'id': results['ids'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
                
        return formatted_results

    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
        """
        # Separate IDs by modality
        text_ids = [id_ for id_ in ids if id_.startswith('text_')]
        image_ids = [id_ for id_ in ids if id_.startswith('image_')]
        
        # Delete from respective collections
        if text_ids:
            self.text_collection.delete(ids=text_ids)
        if image_ids:
            self.image_collection.delete(ids=image_ids)

    def get_vector_count(self) -> int:
        """Get the total number of vectors in the database."""
        return len(self.text_collection.get()['ids']) + len(self.image_collection.get()['ids'])

    def clear(self) -> None:
        """Clear all vectors from the database."""
        self.text_collection.delete(ids=self.text_collection.get()['ids'])
        self.image_collection.delete(ids=self.image_collection.get()['ids']) 