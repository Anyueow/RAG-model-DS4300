from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional
import time
import psutil
import numpy as np
from dataclasses import dataclass

@dataclass
class EmbeddingMetrics:
    """Metrics for embedding performance."""
    time_taken: float  # Time taken in seconds
    memory_used: float  # Memory used in MB
    vector_dimension: int  # Dimension of the embedding vectors

class BaseEmbedder(ABC):
    """Abstract base class for embedders."""
    
    def __init__(self):
        """Initialize the embedder."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model."""
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        pass
    
    @abstractmethod
    def _encode_text(self, text: str) -> np.ndarray:
        """Encode a single text into an embedding vector.
        
        Args:
            text: Input text to encode
            
        Returns:
            Embedding vector as numpy array
        """
        pass
    
    def embed_text(self, text: str) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed a single text.
        
        Args:
            text: Input text to embed
            
        Returns:
            Tuple of (embedding vector, metrics dict)
        """
        # Track start time and memory
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Convert text to embedding
        embedding = self._encode_text(text)
        
        # Track metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = {
            "time_taken": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "vector_dimension": embedding.shape[0]
        }
        
        return embedding, metrics
    
    def embed_batch(self, texts: List[str]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Tuple of (embeddings array, metrics dict)
        """
        # Track start time and memory
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss
        
        # Convert texts to embeddings
        embeddings = np.stack([self._encode_text(text) for text in texts])
        
        # Track metrics
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss
        
        metrics = {
            "time_taken": end_time - start_time,
            "memory_used": end_memory - start_memory,
            "num_texts": len(texts),
            "vector_dimension": embeddings.shape[1]
        }
        
        return embeddings, metrics
    
    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Embed a list of text chunks with metadata.
        
        Args:
            chunks: List of dictionaries containing text and metadata
            
        Returns:
            List of dictionaries with embeddings and metrics added
        """
        # Extract texts from chunks
        texts = [chunk["text"] for chunk in chunks]
        
        # Get embeddings for all texts
        embeddings, batch_metrics = self.embed_batch(texts)
        
        # Add embeddings and metrics to chunks
        for i, chunk in enumerate(chunks):
            chunk["embedding"] = embeddings[i].tolist()
            chunk["embedding_metrics"] = {
                "time_taken": batch_metrics["time_taken"] / len(chunks),
                "memory_used": batch_metrics["memory_used"] / len(chunks),
                "vector_dimension": batch_metrics["vector_dimension"]
            }
        
        return chunks 