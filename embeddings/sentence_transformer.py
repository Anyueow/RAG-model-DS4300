"""Sentence Transformer embedding implementation."""

from typing import List, Dict, Any, Tuple
import os
import numpy as np
import time
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder
from .test_config import EmbeddingModelConfig

# Load environment variables
load_dotenv()

class BatchEmbeddings(tuple):
    """
    A custom tuple subclass that holds a NumPy array of embeddings and
    an associated metrics dictionary. It supports tuple unpacking so that:
    
        embeddings, metrics = embed_batch(texts)
    
    And if used directly (e.g., via len() or indexing), it behaves like the
    underlying embeddings array.
    """
    def __new__(cls, embeddings: np.ndarray, metrics: Dict[str, Any]):
        return super().__new__(cls, (embeddings, metrics))
    
    def __len__(self):
        # Return the number of embeddings (rows) from the underlying array.
        return self[0].shape[0]
    
    def __getitem__(self, idx):
        # Allow indexing into the underlying embeddings array.
        return self[0][idx]

class SentenceTransformerEmbedder(BaseEmbedder):
    """Embedder using Sentence Transformers."""
    
    def __init__(self, model_config: EmbeddingModelConfig):
        """Initialize the embedder.
        
        Args:
            model_config: Configuration for the embedding model
        """
        super().__init__()
        self.model_config = model_config
        
        # Initialize the model with authentication if required
        if model_config.requires_auth:
            try:
                hf_token = os.getenv("HF_TOKEN")
                if not hf_token:
                    raise ValueError("HF_TOKEN environment variable not set")
                self.model = SentenceTransformer(
                    model_config.model_name,
                    use_auth_token=hf_token,
                    trust_remote_code=True  # Required for Nomic model
                )
            except Exception as e:
                raise RuntimeError(
                    f"Failed to initialize {model_config.name}. Please ensure you have set the HF_TOKEN environment variable with your Hugging Face token. Error: {str(e)}"
                )
        else:
            self.model = SentenceTransformer(
                model_config.model_name,
                trust_remote_code=True  # Required for Nomic model
            )
        
        # Set model parameters
        self.model.max_seq_length = model_config.max_length
        
        # Check if this is the Nomic model
        self.is_nomic = "nomic-embed-text-v2-moe" in model_config.model_name
    
    def get_model_name(self) -> str:
        """Get the name of the model."""
        return self.model_config.name
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        return self.model_config.embedding_dim
    
    def _format_text(self, text: str, is_query: bool = False) -> str:
        """Format text with appropriate prefix for Nomic model.
        
        Args:
            text: Input text to format
            is_query: Whether this is a query or document
            
        Returns:
            Formatted text with appropriate prefix
        """
        if self.is_nomic:
            prefix = "search_query: " if is_query else "search_document: "
            return f"{prefix}{text}"
        return text
    
    def _encode_text(self, text: str, is_query: bool = False) -> np.ndarray:
        """Encode a single text into an embedding vector.
        
        Args:
            text: Input text to encode
            is_query: Whether this is a query or document
            
        Returns:
            Normalized embedding vector as a numpy array
        """
        formatted_text = self._format_text(text, is_query)
        embedding = self.model.encode(formatted_text, convert_to_numpy=True)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
        return embedding
    
    def embed_text(self, text: str, is_query: bool = False) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Embed a single text with metrics.
        
        Args:
            text: The text to embed.
            is_query: Whether the text is a query.
            
        Returns:
            A tuple of (embedding, metrics)
        """
        start_time = time.time()
        embedding = self._encode_text(text, is_query)
        end_time = time.time()
        metrics = {
            "time_taken": end_time - start_time,
            "memory_used": 0,
            "vector_dimension": embedding.shape[0]
        }
        return embedding, metrics
    
    def embed_batch(self, texts: List[str], is_query: bool = False) -> BatchEmbeddings:
        """Embed a batch of texts with metrics.
        
        Args:
            texts: List of texts to embed.
            is_query: Whether these are queries or documents.
            
        Returns:
            A BatchEmbeddings object that supports both tuple unpacking 
            (returning (embeddings, metrics)) and behaves like the embeddings array.
        """
        start_time = time.time()
        formatted_texts = [self._format_text(text, is_query) for text in texts]
        embeddings = self.model.encode(formatted_texts, convert_to_numpy=True)
        # Normalize embeddings row-wise.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero.
        embeddings = embeddings / norms
        end_time = time.time()
        metrics = {
            "time_taken": end_time - start_time,
            "memory_used": 0,
            "num_texts": len(texts),
            "vector_dimension": embeddings.shape[1]
        }
        return BatchEmbeddings(embeddings, metrics)
    
    def embed_chunks(self, chunks: List[Dict[str, Any]], is_query: bool = False) -> List[Dict[str, Any]]:
        """Embed a list of text chunks.
        
        Args:
            chunks: List of chunks with text and metadata.
            is_query: Whether these are queries or documents.
            
        Returns:
            List of chunks with added embeddings and embedding_metrics.
        """
        # Extract texts from chunks.
        texts = [chunk['text'] for chunk in chunks]
        batch_result = self.embed_batch(texts, is_query)
        embeddings = batch_result[0]  # Underlying NumPy array.
        batch_metrics = batch_result[1]
        for i, chunk in enumerate(chunks):
            chunk['embedding'] = embeddings[i].tolist()
            chunk['embedding_metrics'] = {
                'time_taken': batch_metrics.get('time_taken', 0) / len(chunks),
                'memory_used': batch_metrics.get('memory_used', 0) / len(chunks),
                'vector_dimension': batch_metrics['vector_dimension']
            }
        return chunks 