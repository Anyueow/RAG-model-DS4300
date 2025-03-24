import os
from typing import List, Dict, Any
from pathlib import Path

from ingestion.data_loader import DataLoader
from preprocessing.chunker import TokenChunker, ChunkingPipeline
from embeddings.base_embedder import BaseEmbedder
from database.base_db import BaseVectorDB
from query.query_handler import VectorQueryHandler, QueryPipeline
from llm.llm_interface import OllamaLLM, LLMPipeline

class RAGSystem:
    """Main RAG system class that coordinates all components."""
    
    def __init__(self,
                 embedder: BaseEmbedder,
                 vector_db: BaseVectorDB,
                 llm_model: str = "llama2"):
        """Initialize the RAG system.
        
        Args:
            embedder: Embedding model instance
            vector_db: Vector database instance
            llm_model: Name of the Ollama model to use
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.data_loader = DataLoader()
        self.chunker = TokenChunker()
        self.chunking_pipeline = ChunkingPipeline(self.chunker)
        self.query_handler = VectorQueryHandler(self.vector_db, self.embedder)
        self.query_pipeline = QueryPipeline(self.query_handler)
        self.llm = OllamaLLM(model_name=llm_model)
        self.llm_pipeline = LLMPipeline(self.llm)
    
    def ingest_documents(self, directory: str) -> None:
        """Ingest documents from a directory.
        
        Args:
            directory: Path to directory containing documents
        """
        # Load documents
        documents = self.data_loader.load_documents(directory)
        
        # Process documents into chunks
        chunks = self.chunking_pipeline.process_documents(documents)
        
        # Convert chunks to embeddings
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.embed_texts(texts)
        
        # Add vectors to database
        self.vector_db.add_vectors(embeddings, chunks)
    
    def query(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Process a query and generate a response.
        
        Args:
            query: User query string
            k: Number of relevant chunks to retrieve
            
        Returns:
            Dictionary containing response and metadata
        """
        # Get relevant chunks
        results = self.query_pipeline.process_query(query, k)
        
        # Generate response using LLM
        response = self.llm_pipeline.generate_response(query, results)
        
        return {
            'response': response,
            'context': results,
            'formatted_results': self.query_pipeline.format_results(results)
        }

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