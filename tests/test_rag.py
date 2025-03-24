import pytest
import numpy as np
from pathlib import Path
import os

from embeddings.sentence_transformer import SentenceTransformerEmbedder
from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from preprocessing.chunker import TokenChunker
from query.query_handler import VectorQueryHandler
from llm.llm_interface import OllamaLLM

@pytest.fixture
def sample_text():
    return """
    This is a sample course note about machine learning.
    Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.
    Deep learning is a subset of machine learning that uses neural networks with multiple layers.
    Neural networks are computing systems inspired by the biological neural networks in human brains.
    """

@pytest.fixture
def embedder():
    return SentenceTransformerEmbedder()

@pytest.fixture
def chroma_db():
    db = ChromaDB(collection_name="test_collection")
    yield db
    db.clear()

@pytest.fixture
def redis_db():
    db = RedisVectorDB()
    yield db
    db.clear()

@pytest.fixture
def chunker():
    return TokenChunker(chunk_size=100, overlap=20)

def test_chunking(chunker, sample_text):
    chunks = chunker.chunk_text(sample_text)
    assert len(chunks) > 0
    assert all(len(chunk.split()) <= 100 for chunk in chunks)

def test_embedding(embedder):
    text = "This is a test sentence."
    embedding = embedder.embed_texts(text)
    assert embedding.shape[0] == 1
    assert embedding.shape[1] == embedder.get_embedding_dim()

def test_chroma_db(chroma_db, embedder, sample_text):
    # Create test chunks
    chunker = TokenChunker(chunk_size=100)
    chunks = chunker.chunk_text(sample_text)
    
    # Create embeddings
    embeddings = embedder.embed_texts(chunks)
    
    # Add to database
    metadata = [{'text': chunk} for chunk in chunks]
    chroma_db.add_vectors(embeddings, metadata)
    
    # Test search
    query = "What is machine learning?"
    query_embedding = embedder.embed_texts(query)
    results = chroma_db.search(query_embedding, k=2)
    
    assert len(results) == 2
    assert 'text' in results[0]
    assert 'metadata' in results[0]

def test_redis_db(redis_db, embedder, sample_text):
    # Create test chunks
    chunker = TokenChunker(chunk_size=100)
    chunks = chunker.chunk_text(sample_text)
    
    # Create embeddings
    embeddings = embedder.embed_texts(chunks)
    
    # Add to database
    metadata = [{'text': chunk} for chunk in chunks]
    redis_db.add_vectors(embeddings, metadata)
    
    # Test search
    query = "What is machine learning?"
    query_embedding = embedder.embed_texts(query)
    results = redis_db.search(query_embedding, k=2)
    
    assert len(results) == 2
    assert 'text' in results[0]
    assert 'metadata' in results[0]

def test_query_handler(chroma_db, embedder):
    # Create test data
    texts = [
        "Machine learning is a field of study.",
        "Deep learning uses neural networks.",
        "Neural networks are computing systems."
    ]
    embeddings = embedder.embed_texts(texts)
    metadata = [{'text': text} for text in texts]
    
    # Add to database
    chroma_db.add_vectors(embeddings, metadata)
    
    # Create query handler
    handler = VectorQueryHandler(chroma_db, embedder)
    
    # Test query
    results = handler.process_query("What is machine learning?", k=2)
    assert len(results) == 2
    assert 'text' in results[0]

def test_llm_interface():
    llm = OllamaLLM(model_name="llama2")
    response = llm.generate_response("What is 2+2?")
    assert isinstance(response, str)
    assert len(response) > 0 