"""Test configuration for embedding models."""

from dataclasses import dataclass
from typing import Dict, Any, List

@dataclass
class EmbeddingModelConfig:
    """Configuration for an embedding model."""
    name: str
    model_name: str
    embedding_dim: int
    max_length: int
    description: str
    model_type: str = "sentence_transformer"
    requires_auth: bool = False
    device: str = "mps"  # or "mps" for Apple Silicon
    use_flash_attention: bool = False  # Disable flash attention for compatibility

# Available embedding models
EMBEDDING_MODELS = {
    "all-mpnet-base-v2": EmbeddingModelConfig(
        name="all-mpnet-base-v2",
        model_name="sentence-transformers/all-mpnet-base-v2",
        embedding_dim=768,
        max_length=128,
        description="High-quality general-purpose model optimized for semantic search",
        model_type="sentence_transformer",
        use_flash_attention=False
    ),
    "multi-qa-MiniLM-L6-cos-v1": EmbeddingModelConfig(
        name="MultiQA-MiniLM-L6-Cos",
        model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        embedding_dim=384,
        max_length=256,
        description="Fast and efficient model for question-answering",
        use_flash_attention=False
    ),
    "nomic-ai/nomic-embed-text-v1.5": EmbeddingModelConfig(
        name="nomic-ai/nomic-embed-text-v1.5",
        model_name="nomic-ai/nomic-embed-text-v1.5",
        embedding_dim=768,
        max_length=512,
        description="State-of-the-art multilingual text embedding model via Ollama",
        model_type="ollama",
        use_flash_attention=False
    )
}

# Test data configurations - using smaller test sets
TEST_DATA = {
    "short": [
        "This is a sample text for testing embeddings.",
        "Another example text with different content."
    ],
    "long": [
        "This is a longer text that tests the model's ability to handle more complex content. " * 3,
        "Another long example that includes various types of content and formatting. " * 3
    ],
    "multilingual": [
        "This is an English text for testing.",
        "Ceci est un texte en français pour les tests."
    ],
    "special": [
        "Text with special characters! @#$%^&*()",
        "Text with numbers: 1234567890"
    ],
    "qa": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?"
    ],
    "retrieval": [
        "Find documents about machine learning.",
        "Search for information about climate change."
    ]
} 