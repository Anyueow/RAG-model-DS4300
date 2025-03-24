import argparse
import sys
from pathlib import Path
from typing import Optional
from PIL import Image

from main import RAGSystem
from embeddings.multimodal_embedder import MultiModalEmbedder
from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from database.qdrant_db import QdrantDB

def get_vector_db(db_type: str):
    """Get vector database instance based on type."""
    if db_type == "chroma":
        return ChromaDB()
    elif db_type == "redis":
        return RedisVectorDB()
    elif db_type == "qdrant":
        return QdrantDB()
    else:
        raise ValueError(f"Unknown database type: {db_type}")

def main():
    parser = argparse.ArgumentParser(description='RAG System CLI')
    parser.add_argument('--db', type=str, choices=['chroma', 'redis', 'qdrant'],
                      default='qdrant', help='Vector database to use')
    parser.add_argument('--data-dir', type=str, default='data/raw_notes',
                      help='Directory containing course notes')
    parser.add_argument('--semantic-weight', type=float, default=0.7,
                      help='Weight for semantic search (0-1)')
    parser.add_argument('--keyword-weight', type=float, default=0.3,
                      help='Weight for keyword search (0-1)')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', type=str, help='Query text')
    query_parser.add_argument('--image', type=str, help='Path to query image')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--num-queries', type=int, default=100,
                               help='Number of queries for benchmarking')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize components
    vector_db = get_vector_db(args.db)
    embedder = MultiModalEmbedder()
    
    # Initialize RAG system
    rag = RAGSystem(
        embedder=embedder,
        vector_db=vector_db,
        semantic_weight=args.semantic_weight,
        keyword_weight=args.keyword_weight
    )
    
    if args.command == 'ingest':
        print(f"Ingesting documents from {args.data_dir}...")
        rag.ingest_documents(args.data_dir)
        print("Ingestion complete!")
        
    elif args.command == 'query':
        # Handle image query if provided
        image_query = None
        modality = "text"
        
        if args.image:
            try:
                with open(args.image, 'rb') as f:
                    image_query = f.read()
                modality = "image"
            except Exception as e:
                print(f"Error loading image: {str(e)}")
                sys.exit(1)
        
        # Process query
        print("Processing query...")
        result = rag.query(
            query=args.query,
            image_query=image_query,
            modality=modality
        )
        
        # Print response
        print("\nResponse:")
        print(result['response'])
        
        print("\nSources:")
        for source in result['sources']:
            print(f"- {source['citation']} ({source['source']})")
            if source['page'] is not None:
                print(f"  Page: {source['page']}")
        
    elif args.command == 'benchmark':
        from benchmark_dbs import VectorDBBenchmark
        print("Running benchmarks...")
        benchmark = VectorDBBenchmark(args.data_dir)
        
        print("\nRunning ingestion benchmarks...")
        ingestion_results = benchmark.benchmark_ingestion()
        
        print("\nRunning query benchmarks...")
        query_results = benchmark.benchmark_queries(args.num_queries)
        
        print("\nGenerating plots and saving results...")
        benchmark.plot_results(ingestion_results, query_results)
        
        print("\nBenchmark complete! Results saved to benchmark_results.json and benchmark_results.png")

if __name__ == "__main__":
    main() 