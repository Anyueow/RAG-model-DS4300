from typing import List, Dict, Any
from abc import ABC, abstractmethod
from database.base_db import BaseVectorDB
from embeddings.base_embedder import BaseEmbedder

class BaseQueryHandler(ABC):
    """Abstract base class for query handlers."""
    
    @abstractmethod
    def process_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query and return relevant chunks.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        pass

class VectorQueryHandler(BaseQueryHandler):
    """Query handler that uses vector similarity search."""
    
    def __init__(self, 
                 vector_db: BaseVectorDB,
                 embedder: BaseEmbedder):
        """Initialize the query handler.
        
        Args:
            vector_db: Vector database instance
            embedder: Embedding model instance
        """
        self.vector_db = vector_db
        self.embedder = embedder
    
    def process_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query using vector similarity search.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Convert query to embedding
        query_embedding = self.embedder.embed_texts(query)
        
        # Search for similar vectors
        results = self.vector_db.search(query_embedding, k=k)
        
        return results

class QueryPipeline:
    """Pipeline for processing queries and generating responses."""
    
    def __init__(self, query_handler: BaseQueryHandler):
        """Initialize the pipeline.
        
        Args:
            query_handler: Query handler instance
        """
        self.query_handler = query_handler
    
    def process_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query through the pipeline.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        return self.query_handler.process_query(query, k)
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results into a readable string.
        
        Args:
            results: List of results from query processing
            
        Returns:
            Formatted string of results
        """
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"Result {i}:\n"
                f"Text: {result['text']}\n"
                f"Source: {result.get('document_id', 'Unknown')}\n"
                f"Score: {result.get('score', 'N/A')}\n"
            )
        return "\n".join(formatted_results) 