"""Factory for creating embedder instances."""

from typing import Dict, Type
from .base_embedder import BaseEmbedder
from .sentence_transformer import SentenceTransformerEmbedder
from .test_config import EmbeddingModelConfig

class EmbedderFactory:
    """Factory class for creating embedder instances."""
    
    _embedders: Dict[str, Type[BaseEmbedder]] = {
        "sentence_transformer": SentenceTransformerEmbedder
    }
    
    @classmethod
    def create(cls, model_config: EmbeddingModelConfig) -> BaseEmbedder:
        """Create an embedder instance based on the model configuration.
        
        Args:
            model_config: Configuration for the embedding model
            
        Returns:
            An instance of the appropriate embedder class
            
        Raises:
            ValueError: If the model type is not supported
        """
        model_type = model_config.model_type
        if model_type not in cls._embedders:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        embedder_class = cls._embedders[model_type]
        return embedder_class(model_config) 