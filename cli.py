import argparse
import sys
from pathlib import Path
from main import RAGSystem
from database.chroma_db import ChromaDB
from embeddings.sentence_transformer import SentenceTransformerEmbedder

def main():
    parser = argparse.ArgumentParser(description='RAG System CLI')
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
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--num-queries', type=int, default=100,
                               help='Number of queries for benchmarking')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize components
    vector_db = ChromaDB()
    embedder = SentenceTransformerEmbedder()
    
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
        print(f"Processing query: {args.query}")
        result = rag.query(args.query)
        print("\nResponse:")
        print(result['response'])
        print("\nSources:")
        for ctx in result['contexts']:
            print(f"- {ctx['metadata']['source']} (Page {ctx['metadata']['page']})")
            
    elif args.command == 'benchmark':
        print(f"Running benchmarks with {args.num_queries} queries...")
        # Implement benchmark logic here
        print("Benchmarking complete!")

if __name__ == '__main__':
    main() 