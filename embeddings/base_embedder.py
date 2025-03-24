from abc import ABC, abstractmethod
from typing import List, Union
import numpy as np

class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Convert text(s) to embeddings.
        
        Args:
            texts: Single text string or list of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        pass
    
    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings produced by this model.
        
        Returns:
            int: Dimension of the embeddings
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the embedding model.
        
        Returns:
            str: Name of the model
        """
        pass 