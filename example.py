import os
from pathlib import Path
import time
from typing import Dict, Any

from embeddings.sentence_transformer import SentenceTransformerEmbedder
from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from main import RAGSystem

def benchmark_vector_db(db_name: str, 
                       vector_db: Any, 
                       rag_system: RAGSystem, 
                       test_queries: list) -> Dict[str, float]:
    """Benchmark vector database performance.
    
    Args:
        db_name: Name of the vector database
        vector_db: Vector database instance
        rag_system: RAG system instance
        test_queries: List of test queries
        
    Returns:
        Dictionary containing benchmark results
    """
    print(f"\nBenchmarking {db_name}...")
    
    # Test ingestion time
    start_time = time.time()
    rag_system.ingest_documents("data/raw_notes")
    ingestion_time = time.time() - start_time
    
    # Test query times
    query_times = []
    for query in test_queries:
        start_time = time.time()
        result = rag_system.query(query)
        query_time = time.time() - start_time
        query_times.append(query_time)
        
        print(f"\nQuery: {query}")
        print(f"Response: {result['response'][:200]}...")
        print(f"Query time: {query_time:.2f} seconds")
    
    avg_query_time = sum(query_times) / len(query_times)
    
    return {
        'ingestion_time': ingestion_time,
        'avg_query_time': avg_query_time,
        'total_vectors': vector_db.get_vector_count()
    }

def main():
    # Create data directory if it doesn't exist
    Path("data/raw_notes").mkdir(parents=True, exist_ok=True)
    
    # Test queries
    test_queries = [
        "What is the main topic of the course?",
        "Can you explain the key concepts?",
        "What are the prerequisites for this course?"
    ]
    
    # Initialize components
    embedder = SentenceTransformerEmbedder()
    chroma_db = ChromaDB()
    redis_db = RedisVectorDB()
    
    # Benchmark ChromaDB
    chroma_rag = RAGSystem(embedder, chroma_db)
    chroma_results = benchmark_vector_db("ChromaDB", chroma_db, chroma_rag, test_queries)
    
    # Benchmark Redis Vector DB
    redis_rag = RAGSystem(embedder, redis_db)
    redis_results = benchmark_vector_db("Redis Vector DB", redis_db, redis_rag, test_queries)
    
    # Print comparison
    print("\nBenchmark Results:")
    print("\nChromaDB:")
    print(f"Ingestion time: {chroma_results['ingestion_time']:.2f} seconds")
    print(f"Average query time: {chroma_results['avg_query_time']:.2f} seconds")
    print(f"Total vectors: {chroma_results['total_vectors']}")
    
    print("\nRedis Vector DB:")
    print(f"Ingestion time: {redis_results['ingestion_time']:.2f} seconds")
    print(f"Average query time: {redis_results['avg_query_time']:.2f} seconds")
    print(f"Total vectors: {redis_results['total_vectors']}")

if __name__ == "__main__":
    main() 