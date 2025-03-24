import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import asyncio
import streamlit as st
from PIL import Image
import io

from ingestion.data_loader import DataLoader
from preprocessing.chunker import TokenChunker, ChunkingPipeline
from embeddings.base_embedder import BaseEmbedder
from database.chroma_db import ChromaDB
from query.query_handler import VectorQueryHandler, QueryPipeline
from llm.llm_interface import OllamaLLM, LLMPipeline
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from embeddings.multimodal_embedder import MultiModalEmbedder
from ingestion.multimodal_loader import PDFLoader, Document
from query.hybrid_search import HybridSearch
from llm.prompt_generator import PromptGenerator

class RAGSystem:
    """RAG system that handles document ingestion and querying."""
    
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_db: Any = None,
        document_loader: Any = None,
        llm: OllamaLLM = None,
        prompt_generator: PromptGenerator = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 3,
        temperature: float = 0.7
    ):
        """Initialize the RAG system.
        
        Args:
            embedder: Embedder for text and images
            vector_db: Vector database instance
            document_loader: Document loader instance
            llm: LLM interface instance
            prompt_generator: Prompt generator instance
            semantic_weight: Weight for semantic search
            keyword_weight: Weight for keyword search
            top_k: Number of results to return
            temperature: Temperature for response generation
        """
        # Initialize components with defaults if not provided
        self.embedder = embedder or MultiModalEmbedder()
        self.vector_db = vector_db or ChromaDB()
        self.document_loader = document_loader or PDFLoader()
        self.llm = llm or OllamaLLM(model_name="qwen:7b", temperature=temperature)
        self.prompt_generator = prompt_generator or PromptGenerator()
        
        # Initialize hybrid search with adjusted weights
        self.search = HybridSearch(
            vector_db=self.vector_db,
            embedder=self.embedder,  # Pass embedder directly
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight,
            top_k=top_k
        )
        
        self.documents = []
        self.query_cache = {}  # Cache for query results

    def ingest_documents(self, data_dir: str) -> None:
        """Ingest documents from a directory.
        
        Args:
            data_dir: Directory containing documents to ingest
        """
        # Load documents
        self.documents = self.document_loader.load_directory(data_dir)
        
        # Process documents in chunks for better memory management
        chunk_size = 10  # Process 10 documents at a time
        for i in range(0, len(self.documents), chunk_size):
            chunk = self.documents[i:i + chunk_size]
            texts = [doc.text for doc in chunk]
            
            # Generate embeddings for the chunk
            embeddings = self.embedder.embed_texts(texts)
            
            # Prepare metadata for the chunk
            metadata = [
                {
                    'text': doc.text,
                    'source': doc.metadata.get('source'),
                    'page': doc.metadata.get('page'),
                    'doc_idx': idx + i,  # Maintain correct document index
                    'type': 'text'
                }
                for idx, doc in enumerate(chunk)
            ]
            
            # Add chunk to vector database
            self.vector_db.add_vectors(embeddings, metadata)
        
        # Index documents for hybrid search
        self.search.index_documents([
            {
                'text': doc.text,
                'source': doc.metadata.get('source'),
                'page': doc.metadata.get('page'),
                'doc_idx': idx,
                'type': 'text'
            }
            for idx, doc in enumerate(self.documents)
        ])

    def query(self, 
              query_text: str, 
              query_image: Optional[bytes] = None,
              use_general_knowledge: bool = True) -> Dict[str, Any]:
        """Process a query and return results.
        
        Args:
            query_text: Text query
            query_image: Optional image query
            use_general_knowledge: Whether to use general knowledge
            
        Returns:
            Dictionary containing response and contexts
        """
        # Check cache first
        cache_key = f"{query_text}_{query_image is not None}_{use_general_knowledge}"
        if cache_key in self.query_cache:
            return self.query_cache[cache_key]
        
        # Process query
        contexts = self.search.search(query_text, query_image)
        
        # Generate response
        response = self.llm.generate_response(
            query_text,
            contexts,
            [{'data': query_image, 'index': 0}] if query_image else None,
            use_general_knowledge=use_general_knowledge
        )
        
        result = {
            'response': response,
            'contexts': contexts
        }
        
        # Cache the result
        self.query_cache[cache_key] = result
        return result

def setup_streamlit():
    """Set up Streamlit interface."""
    st.set_page_config(
        page_title="RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("RAG System Interface")
    
    # Initialize RAG system
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        # Create temporary directory for uploaded files
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save uploaded files
            for file in uploaded_files:
                file_path = os.path.join(temp_dir, file.name)
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())
            
            # Ingest documents
            st.session_state.rag_system.ingest_documents(temp_dir)
            st.success("Documents ingested successfully!")
            
        finally:
            # Clean up temporary files
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
    
    # Query interface with multi-line support
    st.subheader("Enter your query:")
    query = st.text_area(
        "Query",
        height=150,  # Increased height for better visibility
        placeholder="Enter your query here...\nYou can use multiple lines to format your query.\nThe formatting will be preserved."
    )
    
    if query:
        with st.spinner("Processing query..."):
            result = st.session_state.rag_system.query(query)
            
            # Display response
            st.subheader("Response")
            st.write(result['response'])
            
            # Display contexts
            st.subheader("Relevant Contexts")
            for context in result['contexts']:
                st.text_area(
                    f"Context (Score: {context.get('combined_score', 'N/A'):.3f})",
                    context['metadata']['text'],
                    height=100
                )

def main():
    """Main function to run the RAG system."""
    # Set up Streamlit interface
    setup_streamlit()

if __name__ == "__main__":
    # Run the Streamlit app
    main() 