import os
import pytest
import fitz  # PyMuPDF
import numpy as np
from main import RAGSystem
from database.chroma_db import ChromaDB
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from embeddings.test_config import EMBEDDING_MODELS

def test_rag_system():
    """Test basic RAG system functionality."""
    # Initialize components with proper configuration
    model_config = EMBEDDING_MODELS["all-mpnet-base-v2"]
    vector_db = ChromaDB(collection_name="test_collection")
    embedder = SentenceTransformerEmbedder(model_config)
    
    # Initialize RAG system with explicit parameters
    rag = RAGSystem(
        embedder=embedder,
        vector_db=vector_db,
        semantic_weight=0.7,
        keyword_weight=0.3,
        chunk_size=256,  # Smaller chunks for testing
        chunk_overlap=20,
        top_k=3  # Add top_k parameter for context retrieval
    )
    
    # Create test data directory
    test_dir = "test_data"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a test PDF file with proper content
    test_pdf = os.path.join(test_dir, "test.pdf")
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "This is a test document with some content.")
    page.insert_text((50, 100), "It contains multiple lines of text.")
    doc.save(test_pdf)
    doc.close()
    
    try:
        # Test ingestion
        rag.ingest_documents(test_dir)
        
        # Verify documents were loaded
        assert len(rag.documents) > 0
        
        # Test query
        result = rag.query("What is in the test document?")
        assert result is not None
        assert 'response' in result
        assert result['response'] is not None
        
        # Test hybrid search weights
        assert rag.search.semantic_weight == 0.7
        assert rag.search.keyword_weight == 0.3
        
        # Test embedder functionality
        test_text = "This is a test sentence."
        embedding, metrics = rag.embedder.embed_text(test_text)
        assert isinstance(embedding, np.ndarray)
        assert embedding.shape[0] == model_config.embedding_dim
        assert 'time_taken' in metrics
        assert 'vector_dimension' in metrics
        
        # Test batch embedding
        test_texts = ["First sentence.", "Second sentence."]
        embeddings, batch_metrics = rag.embedder.embed_batch(test_texts)
        assert isinstance(embeddings, np.ndarray)
        assert embeddings.shape == (2, model_config.embedding_dim)
        assert 'time_taken' in batch_metrics
        assert 'num_texts' in batch_metrics
        
    finally:
        # Clean up
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        if os.path.exists(test_dir):
            os.rmdir(test_dir)
        # Clear the test collection
        vector_db.clear() 