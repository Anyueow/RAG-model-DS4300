"""Base database interface for vector storage."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import time
import psutil
from dataclasses import dataclass

@dataclass
class SearchResult:
    """Container for search results with metadata."""
    doc_id: str
    chunk: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class PerformanceMetrics:
    """Container for database performance metrics."""
    indexing_time: float
    query_time: float
    memory_used: int
    num_documents: int
    num_chunks: int

class BaseDB(ABC):
    """Abstract base class for vector databases."""
    
    def __init__(self):
        """Initialize the database."""
        self.metrics = PerformanceMetrics(
            indexing_time=0.0,
            query_time=0.0,
            memory_used=0,
            num_documents=0,
            num_chunks=0
        )
    
    @abstractmethod
    def index(self, embeddings: List[List[float]], chunks: List[str], 
              doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index embeddings with their associated chunks and metadata.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        # Track start time and memory
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Call implementation-specific indexing
        self._index_impl(embeddings, chunks, doc_ids, metadata)
        
        # Update metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        self.metrics.indexing_time = end_time - start_time
        self.metrics.memory_used = end_memory - start_memory
        self.metrics.num_documents = len(set(doc_ids))
        self.metrics.num_chunks = len(chunks)
    
    @abstractmethod
    def _index_impl(self, embeddings: List[List[float]], chunks: List[str], 
                   doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Implementation-specific indexing logic."""
        pass
    
    def search(self, query_embedding: List[float], k: int = 5) -> List[SearchResult]:
        """Search for similar vectors and return results with metadata.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of SearchResult objects containing matches
        """
        # Track start time
        start_time = time.time()
        
        # Call implementation-specific search
        results = self._search_impl(query_embedding, k)
        
        # Update metrics
        self.metrics.query_time = time.time() - start_time
        
        return results
    
    @abstractmethod
    def _search_impl(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        """Implementation-specific search logic."""
        pass
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics.
        
        Returns:
            PerformanceMetrics object containing current metrics
        """
        return self.metrics
    
    def reset_metrics(self) -> None:
        """Reset performance metrics."""
        self.metrics = PerformanceMetrics(
            indexing_time=0.0,
            query_time=0.0,
            memory_used=0,
            num_documents=0,
            num_chunks=0
        )
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all data from the database."""
        pass 