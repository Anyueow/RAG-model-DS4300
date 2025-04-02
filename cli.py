import argparse
import sys
from pathlib import Path
from main import RAGSystem
from database.chroma_db import ChromaDB
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from embeddings.test_config import EMBEDDING_MODELS
from llm.llm_interface import OllamaLLM

def main():
    parser = argparse.ArgumentParser(description='RAG System CLI for Database, ML/AI, and AWS Questions')
    parser.add_argument('--data-dir', type=str, default='data',
                      help='Directory containing course notes')
    parser.add_argument('--semantic-weight', type=float, default=0.8,
                      help='Weight for semantic search (0-1)')
    parser.add_argument('--keyword-weight', type=float, default=0.2,
                      help='Weight for keyword search (0-1)')
    parser.add_argument('--model', type=str, default='nomic-embed-text-v2-moe',
                      help='Embedding model to use')
    parser.add_argument('--collection', type=str, default='cli_collection',
                      help='ChromaDB collection name')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for LLM response generation')
    parser.add_argument('--llm-model', type=str, default='qwen:7b',
                      help='Ollama model to use')
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser('ingest', help='Ingest documents')
    ingest_parser.add_argument('--chunk-size', type=int, default=512,
                            help='Size of text chunks')
    ingest_parser.add_argument('--chunk-overlap', type=int, default=50,
                            help='Overlap between chunks')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the system')
    query_parser.add_argument('query', type=str, help='Query text')
    query_parser.add_argument('--no-general-knowledge', action='store_true',
                           help='Disable use of general knowledge')
    query_parser.add_argument('--top-k', type=int, default=3,
                           help='Number of context chunks to retrieve')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run benchmarks')
    benchmark_parser.add_argument('--num-queries', type=int, default=100,
                               help='Number of queries for benchmarking')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Initialize components
        model_config = EMBEDDING_MODELS[args.model]
        embedder = SentenceTransformerEmbedder(model_config)
        vector_db = ChromaDB(collection_name=args.collection)
        llm = OllamaLLM(model_name=args.llm_model, temperature=args.temperature)
        
        # Initialize RAG system
        rag = RAGSystem(
            embedder=embedder,
            vector_db=vector_db,
            llm=llm,
            semantic_weight=args.semantic_weight,
            keyword_weight=args.keyword_weight,
            model_config=model_config,
            chunk_size=getattr(args, 'chunk_size', 512),
            chunk_overlap=getattr(args, 'chunk_overlap', 50),
            collection_name=args.collection,
            temperature=args.temperature
        )
        
        if args.command == 'ingest':
            print(f"Ingesting documents from {args.data_dir}...")
            data_path = Path(args.data_dir)
            if not data_path.exists():
                print(f"Error: Directory {args.data_dir} does not exist!")
                sys.exit(1)
            rag.ingest_documents(str(data_path))
            print("Ingestion complete!")
            
        elif args.command == 'query':
            print(f"\nProcessing query: {args.query}")
            result = rag.query(
                query_text=args.query,
                use_general_knowledge=not args.no_general_knowledge
            )
            
            print("\nResponse:")
            print("-" * 80)
            print(result['response'])
            print("-" * 80)
            
            print("\nSources:")
            for ctx in result['contexts']:
                source = ctx['metadata'].get('source', 'Unknown')
                page = ctx['metadata'].get('page', 'N/A')
                print(f"- {source} (Page {page})")
                print(f"  Score: {ctx.get('score', 'N/A')}")
            
        elif args.command == 'benchmark':
            print(f"Running benchmarks with {args.num_queries} queries...")
            # Implement benchmark logic here
            print("Benchmarking complete!")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main() 