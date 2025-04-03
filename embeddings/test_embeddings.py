"""Tests for embedding models."""

import pytest
import numpy as np
import time
from typing import Dict, Any, List
from .test_config import EMBEDDING_MODELS, TEST_DATA
from .sentence_transformer import SentenceTransformerEmbedder
from .embedder_factory import EmbedderFactory


# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@pytest.fixture
def model_config():
    """Create a model configuration."""
    return EMBEDDING_MODELS["nomic-embed-text-v2-moe"]

@pytest.fixture
def embedder():
    """Create an embedder with default configuration."""
    model_config = EMBEDDING_MODELS["nomic-embed-text-v2-moe"]
    return SentenceTransformerEmbedder(model_config)

class TestEmbeddingModels:
    """Test suite for embedding models."""
    
    def test_model_initialization(self, embedder):
        """Test model initialization."""
        logger.info(f"Testing model initialization for {embedder.get_model_name()}")
        assert embedder is not None
        assert embedder.get_embedding_dim() > 0
        logger.info(f"Model initialization test passed for {embedder.get_model_name()}")
    
    def test_single_text_embedding(self, embedder):
        """Test single text embedding."""
        logger.info(f"Testing single text embedding for {embedder.get_model_name()}")
        text = "This is a test sentence."
        start_time = time.time()
        embedding = embedder._encode_text(text)
        duration = time.time() - start_time
        logger.info(f"Single text embedding completed in {duration:.2f} seconds")
        assert embedding is not None
        assert len(embedding) == embedder.get_embedding_dim()
        logger.info(f"Single text embedding test passed for {embedder.get_model_name()}")
    
    def test_batch_embedding(self, embedder):
        """Test batch text embedding."""
        logger.info(f"Testing batch embedding for {embedder.get_model_name()}")
        texts = TEST_DATA["short"]
        start_time = time.time()
        embeddings = embedder.embed_batch(texts)
        duration = time.time() - start_time
        logger.info(f"Batch embedding completed in {duration:.2f} seconds")
        assert embeddings is not None
        assert len(embeddings) == len(texts)
        assert len(embeddings[0]) == embedder.get_embedding_dim()
        logger.info(f"Batch embedding test passed for {embedder.get_model_name()}")
    
    def test_chunk_embedding(self, embedder):
        """Test chunk embedding with metadata."""
        logger.info(f"Testing chunk embedding for {embedder.get_model_name()}")
        chunks = [
            {"text": text, "metadata": {"source": "test", "id": i}}
            for i, text in enumerate(TEST_DATA["short"])
        ]
        start_time = time.time()
        embedded_chunks = embedder.embed_chunks(chunks)
        duration = time.time() - start_time
        logger.info(f"Chunk embedding completed in {duration:.2f} seconds")
        assert len(embedded_chunks) == len(chunks)
        for chunk in embedded_chunks:
            assert "embedding" in chunk
            assert "metrics" in chunk
            assert len(chunk["embedding"]) == embedder.get_embedding_dim()
        logger.info(f"Chunk embedding test passed for {embedder.get_model_name()}")
    
    def test_vector_normalization(self, embedder):
        """Test vector normalization."""
        logger.info(f"Testing vector normalization for {embedder.get_model_name()}")
        text = "Test normalization"
        embedding = embedder._encode_text(text)
        norm = np.linalg.norm(embedding)
        logger.info(f"Vector norm: {norm:.6f}")
        assert np.isclose(norm, 1.0, rtol=1e-5)
        logger.info(f"Vector normalization test passed for {embedder.get_model_name()}")
    
    def test_long_text_handling(self, embedder):
        """Test handling of long texts."""
        logger.info(f"Testing long text handling for {embedder.get_model_name()}")
        text = TEST_DATA["long"][0]
        start_time = time.time()
        embedding = embedder._encode_text(text)
        duration = time.time() - start_time
        logger.info(f"Long text embedding completed in {duration:.2f} seconds")
        assert embedding is not None
        assert len(embedding) == embedder.get_embedding_dim()
        logger.info(f"Long text handling test passed for {embedder.get_model_name()}")
    
    def test_multilingual_text(self, embedder):
        """Test multilingual text handling."""
        logger.info(f"Testing multilingual text handling for {embedder.get_model_name()}")
        texts = TEST_DATA["multilingual"]
        start_time = time.time()
        embeddings = embedder.embed_batch(texts)
        duration = time.time() - start_time
        logger.info(f"Multilingual text embedding completed in {duration:.2f} seconds")
        assert embeddings is not None
        assert len(embeddings) == len(texts)
        logger.info(f"Multilingual text test passed for {embedder.get_model_name()}")
    
    def test_special_characters(self, embedder):
        """Test handling of special characters."""
        logger.info(f"Testing special character handling for {embedder.get_model_name()}")
        texts = TEST_DATA["special"]
        start_time = time.time()
        embeddings = embedder.embed_batch(texts)
        duration = time.time() - start_time
        logger.info(f"Special character embedding completed in {duration:.2f} seconds")
        assert embeddings is not None
        assert len(embeddings) == len(texts)
        logger.info(f"Special character test passed for {embedder.get_model_name()}")
    
    def test_query_formatting(self, embedder):
        """Test query-specific formatting."""
        logger.info(f"Testing query formatting for {embedder.get_model_name()}")
        text = "Find information about machine learning"
        start_time = time.time()
        embedding = embedder._encode_text(text, is_query=True)
        duration = time.time() - start_time
        logger.info(f"Query formatting completed in {duration:.2f} seconds")
        assert embedding is not None
        assert len(embedding) == embedder.get_embedding_dim()
        logger.info(f"Query formatting test passed for {embedder.get_model_name()}")
    
    def test_semantic_similarity(self, embedder):
        """Test semantic similarity between related texts."""
        logger.info(f"Testing semantic similarity for {embedder.get_model_name()}")
        text1 = "The cat is on the mat."
        text2 = "A feline is sitting on the rug."
        start_time = time.time()
        embedding1 = embedder._encode_text(text1)
        embedding2 = embedder._encode_text(text2)
        duration = time.time() - start_time
        logger.info(f"Semantic similarity test completed in {duration:.2f} seconds")
        similarity = np.dot(embedding1, embedding2)
        logger.info(f"Semantic similarity score: {similarity:.4f}")
        assert similarity > 0.7  # High similarity for semantically related texts
        logger.info(f"Semantic similarity test passed for {embedder.get_model_name()}")
    
    def test_embedder_factory(self):
        """Test embedder factory creation."""
        logger.info("Testing embedder factory")
        for model_name, config in EMBEDDING_MODELS.items():
            embedder = EmbedderFactory.create(config)
            assert embedder is not None
            assert embedder.get_model_name() == config.name
            assert embedder.get_embedding_dim() == config.embedding_dim
        logger.info("Embedder factory test passed")

def test_embed_text(embedder):
    """Test embedding a single text."""
    text = "This is a test sentence."
    embedding, metrics = embedder.embed_text(text)
    
    assert isinstance(embedding, np.ndarray)
    assert embedding.shape[0] == embedder.get_embedding_dim()
    assert "time_taken" in metrics
    assert "memory_used" in metrics
    assert "vector_dimension" in metrics

def test_embed_batch(embedder):
    """Test embedding a batch of texts."""
    texts = ["First sentence.", "Second sentence.", "Third sentence."]
    embeddings, metrics = embedder.embed_batch(texts)
    
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape[0] == len(texts)
    assert embeddings.shape[1] == embedder.get_embedding_dim()
    assert "time_taken" in metrics
    assert "memory_used" in metrics
    assert "num_texts" in metrics
    assert "vector_dimension" in metrics

def test_embed_chunks(embedder):
    """Test embedding chunks with metadata."""
    chunks = [
        {"text": "First chunk.", "metadata": {"source": "doc1"}},
        {"text": "Second chunk.", "metadata": {"source": "doc1"}},
        {"text": "Third chunk.", "metadata": {"source": "doc2"}}
    ]
    
    result = embedder.embed_chunks(chunks)
    
    assert len(result) == len(chunks)
    for chunk in result:
        assert "embedding" in chunk
        assert "embedding_metrics" in chunk
        assert isinstance(chunk["embedding"], list)
        assert len(chunk["embedding"]) == embedder.get_embedding_dim()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 