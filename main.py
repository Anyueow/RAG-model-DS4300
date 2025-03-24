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
        embedder: Optional[BaseEmbedder] = None,
        vector_db: ChromaDB = None,
        document_loader: PDFLoader = None,
        llm: OllamaLLM = None,
        prompt_generator: PromptGenerator = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 5,
        temperature: float = 0.7
    ):
        """Initialize the RAG system.
        
        Args:
            embedder: Text/image embedder instance (defaults to MultiModalEmbedder)
            vector_db: Vector database instance
            document_loader: Document loader instance
            llm: LLM interface instance
            prompt_generator: Prompt generator instance
            semantic_weight: Weight for semantic search (default: 0.7)
            keyword_weight: Weight for keyword search (default: 0.3)
            top_k: Number of contexts to retrieve
            temperature: Temperature for LLM response generation (default: 0.7)
        """
        # Initialize components with defaults if not provided
        self.embedder = embedder or MultiModalEmbedder()
        self.vector_db = vector_db or ChromaDB()
        self.document_loader = document_loader or PDFLoader()
        self.llm = llm or OllamaLLM("qwen:7b", temperature=temperature)
        self.prompt_generator = prompt_generator or PromptGenerator()
        
        # Initialize hybrid search
        self.search = HybridSearch(
            vector_db=self.vector_db,
            semantic_weight=semantic_weight,
            keyword_weight=keyword_weight
        )
        
        self.top_k = top_k
        self.documents = []

    def ingest_documents(self, data_dir: str) -> None:
        """Ingest documents from a directory.
        
        Args:
            data_dir: Directory containing documents to ingest
        """
        # Load documents
        self.documents = self.document_loader.load_directory(data_dir)
        
        # Process documents
        texts = [doc.text for doc in self.documents]
        embeddings = self.embedder.embed_texts(texts)
        
        # Prepare metadata
        metadata = [
            {
                'text': doc.text,
                'source': doc.metadata.get('source'),
                'page': doc.metadata.get('page'),
                'doc_idx': idx,
                'type': 'text'
            }
            for idx, doc in enumerate(self.documents)
        ]
        
        # Add to vector database
        self.vector_db.add_vectors(embeddings, metadata)
        
        # Index documents for hybrid search
        self.search.index_documents(metadata)

    def query(self, query: str, query_image: Optional[Image.Image] = None) -> Dict[str, Any]:
        """Process a query and generate a response.
        
        Args:
            query: User query string
            query_image: Optional image for image-based search
            
        Returns:
            Dictionary containing response and metadata
        """
        # Get query embedding based on modality
        if query_image is not None:
            query_embedding = self.embedder.embed_image(query_image)
            modality = "image"
        else:
            query_embedding = self.embedder.embed_text(query)
            modality = "text"
        
        # Perform hybrid search
        results = self.search.search(
            query=query,
            query_embedding=query_embedding,
            k=self.top_k,
            modality=modality
        )
        
        # Generate response using LLM
        response = self.llm.generate_response(query, results)
        
        return {
            'response': response,
            'contexts': results,
            'modality': modality
        }

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
    
    # Query interface
    query = st.text_input("Enter your query:")
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