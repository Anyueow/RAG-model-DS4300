import os
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from ingestion.data_loader import DataLoader
from preprocessing.chunker import TokenChunker, ChunkingPipeline
from embeddings.base_embedder import BaseEmbedder
from database.base_db import BaseVectorDB
from query.query_handler import VectorQueryHandler, QueryPipeline
from llm.llm_interface import OllamaLLM, LLMPipeline
from embeddings.multimodal_embedder import MultiModalEmbedder
from database.qdrant_db import QdrantDB
from ingestion.multimodal_loader import MultiModalPDFLoader, MultiModalDocument
from query.hybrid_search import HybridSearch
from llm.prompt_generator import PromptGenerator
import base64
from PIL import Image
import io

class RAGSystem:
    """Multimodal RAG system that handles both text and image queries."""
    
    def __init__(
        self,
        embedder: MultiModalEmbedder = None,
        vector_db: QdrantDB = None,
        document_loader: MultiModalPDFLoader = None,
        llm: OllamaLLM = None,
        prompt_generator: PromptGenerator = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 5
    ):
        """Initialize the RAG system.
        
        Args:
            embedder: Multimodal embedder instance
            vector_db: Vector database instance
            document_loader: Document loader instance
            llm: LLM interface instance
            prompt_generator: Prompt generator instance
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            top_k: Number of contexts to retrieve
        """
        # Initialize components with defaults if not provided
        self.embedder = embedder or MultiModalEmbedder()
        self.vector_db = vector_db or QdrantDB()
        self.document_loader = document_loader or MultiModalPDFLoader()
        self.llm = llm or OllamaLLM("qwen:7b")
        self.prompt_generator = prompt_generator or PromptGenerator()
        
        # Initialize hybrid search
        self.search = HybridSearch(
            vector_db=self.vector_db,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        self.top_k = top_k
        self.documents = []

    def ingest_documents(self, directory_path: str) -> None:
        """Ingest documents from a directory.
        
        Args:
            directory_path: Path to directory containing documents
        """
        # Load documents
        documents = self.document_loader.load_directory(directory_path)
        self.documents = documents
        
        # Process text and images
        for doc_idx, doc in enumerate(documents):
            # Process text
            text_embedding = self.embedder.embed_text(doc.text)
            self.vector_db.add_vectors(
                vectors=[text_embedding],
                metadata=[{
                    'text': doc.text,
                    'source': doc.metadata.get('source'),
                    'page': doc.metadata.get('page'),
                    'doc_idx': doc_idx,
                    'type': 'text'
                }],
                modality="text"
            )
            
            # Process images
            for img in doc.images:
                image_data = img['data']
                image_embedding = self.embedder.embed_image(image_data)
                
                # Add image vector
                self.vector_db.add_vectors(
                    vectors=[image_embedding],
                    metadata=[{
                        'image': image_data,
                        'bbox': img.get('bbox'),
                        'page': img.get('page'),
                        'source': doc.metadata.get('source'),
                        'doc_idx': doc_idx,
                        'type': 'image'
                    }],
                    modality="image"
                )
        
        # Index documents for keyword search
        self.search.index_documents([
            {'text': doc.text, 'doc_idx': idx}
            for idx, doc in enumerate(documents)
        ])

    def query(
        self,
        query: str,
        image_query: Optional[Union[str, bytes, Image.Image]] = None,
        modality: str = "text"
    ) -> Dict[str, Any]:
        """Process a query and return relevant results.
        
        Args:
            query: Text query
            image_query: Optional image query
            modality: Type of query ('text' or 'image')
            
        Returns:
            Dictionary containing:
            - response: Generated response
            - sources: List of sources used
            - images: List of relevant images
        """
        # Get query embeddings
        if modality == "text":
            query_embedding = self.embedder.embed_text(query)
        else:
            query_embedding = self.embedder.embed_image(image_query)
        
        # Search for relevant contexts
        results = self.search.search(
            query=query,
            query_embedding=query_embedding,
            k=self.top_k,
            modality=modality
        )
        
        # Separate text and image contexts
        text_contexts = []
        image_contexts = []
        
        for result in results:
            if result['metadata'].get('type') == 'text':
                text_contexts.append(result)
            else:
                image_contexts.append(result)
        
        # Generate augmented prompt
        prompt_data = self.prompt_generator.generate_prompt(
            query=query,
            text_contexts=text_contexts,
            image_contexts=image_contexts
        )
        
        # Generate response using LLM
        response = self.llm.generate_response(
            prompt=prompt_data['prompt'],
            images=prompt_data['images']
        )
        
        # Format response with citations
        formatted_response = self.prompt_generator.format_response(
            response=response,
            sources=results
        )
        
        return formatted_response

    def clear(self) -> None:
        """Clear all data from the system."""
        self.vector_db.clear()
        self.documents = []

def main():
    """Main function to demonstrate RAG system usage."""
    # Initialize components (implementations will be added later)
    # embedder = SentenceTransformerEmbedder()
    # vector_db = ChromaDB()
    
    # Initialize RAG system
    # rag_system = RAGSystem(embedder, vector_db)
    
    # Example usage
    # rag_system.ingest_documents("data/raw_notes")
    # result = rag_system.query("What is the main topic of the course?")
    # print(result['response'])
    # print("\nRelevant context:")
    # print(result['formatted_results'])

if __name__ == "__main__":
    main() 