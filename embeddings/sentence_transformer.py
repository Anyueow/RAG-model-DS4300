from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder

class SentenceTransformerEmbedder(BaseEmbedder):
    """SentenceTransformer-based embedder implementation."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize the embedder.
        
        Args:
            model_name: Name of the SentenceTransformer model to use
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
    
    def embed_text(self, text: str) -> np.ndarray:
        """Convert a single text to embedding.
        
        Args:
            text: Text string to embed
            
        Returns:
            numpy.ndarray: Text embedding
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Convert text(s) to embeddings using SentenceTransformer.
        
        Args:
            texts: Single text string or list of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings.
        
        Returns:
            int: Dimension of the embeddings
        """
        return self.model.get_sentence_embedding_dimension()
    
    def get_model_name(self) -> str:
        """Get the name of the embedding model.
        
        Returns:
            str: Name of the model
        """
        return self.model_name 