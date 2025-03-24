#!/usr/bin/env python3
import argparse
import sys
from PIL import Image
from main import RAGSystem
from embeddings.multimodal_embedder import MultiModalEmbedder
from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
import io

def initialize_rag_system():
    """Initialize the RAG system with default settings."""
    try:
        embedder = MultiModalEmbedder()
        rag = RAGSystem(
            embedder=embedder,
            semantic_weight=0.8,
            keyword_weight=0.2,
            top_k=3,
            temperature=0.3
        )
        return rag
    except Exception as e:
        print(f"Error initializing RAG system: {str(e)}")
        return None

def load_image(image_path: str) -> Image.Image:
    """Load and validate an image file."""
    try:
        image = Image.open(image_path)
        return image
    except Exception as e:
        print(f"Error loading image: {str(e)}")
        sys.exit(1)

def image_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to bytes."""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format=image.format or 'PNG')
        img_byte_arr = img_byte_arr.getvalue()
        return img_byte_arr
    except Exception as e:
        print(f"Error converting image to bytes: {str(e)}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Query the RAG system from terminal')
    parser.add_argument('query', type=str, help='The query to process')
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing documents')
    parser.add_argument('--process-docs', action='store_true', help='Process documents before querying')
    parser.add_argument('--no-general-knowledge', action='store_true', 
                       help='Disable general knowledge responses, use only context from documents')
    parser.add_argument('--image', type=str, help='Path to image file for image-based queries')
    args = parser.parse_args()

    # Initialize RAG system
    rag_system = initialize_rag_system()
    if not rag_system:
        print("Failed to initialize RAG system")
        sys.exit(1)

    # Process documents if requested
    if args.process_docs:
        print(f"Processing documents from {args.data_dir}...")
        try:
            rag_system.ingest_documents(args.data_dir)
            print("Documents processed successfully")
        except Exception as e:
            print(f"Error processing documents: {str(e)}")
            sys.exit(1)

    # Load image if provided
    query_image = None
    if args.image:
        print(f"Loading image from {args.image}...")
        pil_image = load_image(args.image)
        query_image = image_to_bytes(pil_image)
        print("Image loaded and converted successfully")

    # Process query
    print(f"\nProcessing query: {args.query}")
    try:
        result = rag_system.query(
            query=args.query,
            query_image=query_image,
            use_general_knowledge=not args.no_general_knowledge
        )
        
        if result:
            print("\nResponse:")
            print(result.get('response', 'No response generated'))
            
            if 'contexts' in result and result['contexts']:
                print("\nRelevant Contexts:")
                for idx, context in enumerate(result['contexts'], 1):
                    score = context.get('combined_score', 'N/A')
                    score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
                    print(f"\nContext {idx} (Score: {score_str}):")
                    if 'text' in context:
                        print(context['text'])
                    if 'metadata' in context:
                        metadata = context['metadata']
                        if 'source' in metadata:
                            print(f"Source: {metadata['source']}")
                        if 'page' in metadata:
                            print(f"Page: {metadata['page']}")
        else:
            print("No results found for your query.")
            
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 