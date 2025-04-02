import pytest
import numpy as np
import time
from typing import Dict, Any, List
from .base_db import SearchResult
from .chroma_db import ChromaDB
from .redis_db import RedisDB
from .qdrant_db import QdrantDB

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@pytest.fixture
def test_data():
    """Create test data."""
    # Create sample embeddings and metadata
    embeddings = [
        np.random.rand(384).tolist(),  # Using 384 dimensions as a default
        np.random.rand(384).tolist(),
        np.random.rand(384).tolist()
    ]
    chunks = [
        "This is the first test chunk.",
        "This is the second test chunk.",
        "This is the third test chunk."
    ]
    doc_ids = ["doc1", "doc1", "doc2"]
    metadata = [
        {"source": "test1.txt", "page": 1},
        {"source": "test1.txt", "page": 2},
        {"source": "test2.txt", "page": 1}
    ]
    return {
        "embeddings": embeddings,
        "chunks": chunks,
        "doc_ids": doc_ids,
        "metadata": metadata
    }

@pytest.fixture(params=["chroma", "redis", "qdrant"])
def db(request):
    """Create a database instance."""
    db_type = request.param
    logger.info(f"Initializing database: {db_type}")
    
    if db_type == "chroma":
        db = ChromaDB(collection_name="test_collection")
    elif db_type == "redis":
        db = RedisDB(collection_name="test_collection")
    else:  # qdrant
        db = QdrantDB(collection_name="test_collection")
        
    yield db
    
    # Cleanup after tests
    db.clear()

class TestDatabases:
    """Test suite for database implementations."""
    
    def test_initialization(self, db):
        """Test database initialization."""
        logger.info(f"Testing initialization for {db.__class__.__name__}")
        assert db is not None
        
    def test_indexing(self, db, test_data):
        """Test indexing functionality."""
        logger.info(f"Testing indexing for {db.__class__.__name__}")
        start_time = time.time()
        db.index(
            embeddings=test_data["embeddings"],
            chunks=test_data["chunks"],
            doc_ids=test_data["doc_ids"],
            metadata=test_data["metadata"]
        )
        duration = time.time() - start_time
        logger.info(f"Indexing completed in {duration:.2f} seconds")
        
        # Check metrics
        metrics = db.get_metrics()
        assert metrics.indexing_time > 0
        assert metrics.num_documents > 0
        assert metrics.num_chunks == len(test_data["chunks"])
        
    def test_search(self, db, test_data):
        """Test search functionality."""
        logger.info(f"Testing search for {db.__class__.__name__}")
        
        # Index test data
        db.index(
            embeddings=test_data["embeddings"],
            chunks=test_data["chunks"],
            doc_ids=test_data["doc_ids"],
            metadata=test_data["metadata"]
        )
        
        # Create a query vector
        query_vector = np.random.rand(384).tolist()
        
        # Test search
        start_time = time.time()
        results = db.search(query_vector, k=2)
        duration = time.time() - start_time
        logger.info(f"Search completed in {duration:.2f} seconds")
        
        # Check results
        assert len(results) <= 2  # Should not return more than k results
        assert all(isinstance(r, SearchResult) for r in results)
        for result in results:
            assert result.doc_id in test_data["doc_ids"]
            assert result.chunk in test_data["chunks"]
            assert isinstance(result.score, float)
            assert result.metadata is not None
            
    def test_clear(self, db, test_data):
        """Test clearing the database."""
        logger.info(f"Testing clear for {db.__class__.__name__}")
        
        # Index test data
        db.index(
            embeddings=test_data["embeddings"],
            chunks=test_data["chunks"],
            doc_ids=test_data["doc_ids"],
            metadata=test_data["metadata"]
        )
        
        # Clear database
        start_time = time.time()
        db.clear()
        duration = time.time() - start_time
        logger.info(f"Clear completed in {duration:.2f} seconds")
        
        # Verify metrics are reset
        metrics = db.get_metrics()
        assert metrics.num_documents == 0
        assert metrics.num_chunks == 0
        
    def test_performance_metrics(self, db, test_data):
        """Test performance metrics tracking."""
        logger.info(f"Testing performance metrics for {db.__class__.__name__}")
        
        # Reset metrics
        db.reset_metrics()
        
        # Index test data
        db.index(
            embeddings=test_data["embeddings"],
            chunks=test_data["chunks"],
            doc_ids=test_data["doc_ids"],
            metadata=test_data["metadata"]
        )
        
        # Perform search
        query_vector = np.random.rand(384).tolist()
        db.search(query_vector, k=2)
        
        # Check metrics
        metrics = db.get_metrics()
        assert metrics.indexing_time > 0
        assert metrics.query_time > 0
        assert metrics.memory_used >= 0
        assert metrics.num_documents > 0
        assert metrics.num_chunks > 0
        
        logger.info(f"Metrics for {db.__class__.__name__}:")
        logger.info(f"Indexing time: {metrics.indexing_time:.2f} seconds")
        logger.info(f"Query time: {metrics.query_time:.2f} seconds")
        logger.info(f"Memory used: {metrics.memory_used} bytes")
        logger.info(f"Number of documents: {metrics.num_documents}")
        logger.info(f"Number of chunks: {metrics.num_chunks}") 