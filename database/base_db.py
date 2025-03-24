from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import numpy as np

class BaseVectorDB(ABC):
    """Abstract base class for vector databases."""
    
    @abstractmethod
    def add_vectors(self, 
                   vectors: np.ndarray, 
                   metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Add vectors to the database.
        
        Args:
            vectors: Array of vectors to add
            metadata: Optional list of metadata dictionaries for each vector
        """
        pass
    
    @abstractmethod
    def search(self, 
               query_vector: np.ndarray, 
               k: int = 5) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of dictionaries containing results and their metadata
        """
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
        """
        pass
    
    @abstractmethod
    def get_vector_count(self) -> int:
        """Get the total number of vectors in the database.
        
        Returns:
            int: Number of vectors
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all vectors from the database."""
        pass 