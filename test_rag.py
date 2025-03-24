import os
import pytest
import fitz  # PyMuPDF
from main import RAGSystem
from database.chroma_db import ChromaDB
from embeddings.sentence_transformer import SentenceTransformerEmbedder

def test_rag_system():
    """Test basic RAG system functionality."""
    # Initialize components
    vector_db = ChromaDB()
    embedder = SentenceTransformerEmbedder()
    
    # Initialize RAG system
    rag = RAGSystem(
        embedder=embedder,
        vector_db=vector_db,
        semantic_weight=0.7,
        keyword_weight=0.3
    )
    
    # Create test data directory
    test_dir = "test"
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
        
        # Test query
        result = rag.query("What is in the test document?")
        assert result['response'] is not None
        assert len(result['contexts']) > 0
        
        # Test hybrid search weights
        assert rag.search.semantic_weight == 0.7
        assert rag.search.keyword_weight == 0.3
        
    finally:
        # Clean up
        if os.path.exists(test_pdf):
            os.remove(test_pdf)
        if os.path.exists(test_dir):
            os.rmdir(test_dir) 