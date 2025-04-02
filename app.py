"""Streamlit UI for the RAG system."""

# Standard library imports
import os
import sys
import logging
import socket
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import streamlit as st
from PIL import Image
import sentence_transformers

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Local imports
from database.chroma_db import ChromaDB
from database.redis_db import RedisDB
from database.qdrant_db import QdrantDB
from llm.llm_interface import OllamaLLM
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from embeddings.test_config import EMBEDDING_MODELS
from main import RAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check sentence-transformers version
if sentence_transformers.__version__ < "3.3.0":
    logger.warning(
        "You are using an older version of sentence-transformers. "
        "Please upgrade to version 3.3.0 or later for better compatibility: "
        "pip install --upgrade sentence-transformers"
    )

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="RAG Ds4300 Midterm Cheat Sheet",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def check_redis_status() -> bool:
    """Check if Redis is running."""
    try:
        s = socket.socket()
        s.connect(('localhost', 6379))
        s.close()
        return True
    except:
        return False

def check_ollama_status() -> tuple[bool, List[str]]:
    """Check if Ollama is running and get available models."""
    try:
        result = subprocess.run(['ollama', 'list'], 
                              capture_output=True, 
                              text=True)
        
        if result.returncode == 0:
            # Parse the output to get models
            lines = result.stdout.strip().split('\n')
            models = []
            
            # Skip header line and parse model names
            for line in lines[1:]:
                if line.strip():  # Skip empty lines
                    models.append(line.split()[0])
            
            return True, models
        return False, []
    except:
        return False, []

def initialize_rag_system() -> Optional[RAGSystem]:
    """Initialize the RAG system with default settings."""
    try:
        # Use Nomic embedder as default
        model_config = EMBEDDING_MODELS["nomic-embed-text-v2-moe"]
        embedder = SentenceTransformerEmbedder(model_config)
        
        # Initialize RAG system with optimized settings
        rag = RAGSystem(
            embedder=embedder,
            vector_db=ChromaDB(),  # Use persistent storage
            semantic_weight=0.8,  # Increased semantic weight for better semantic understanding
            keyword_weight=0.2,   # Reduced keyword weight to focus more on semantic meaning
            top_k=3,              # Reduced number of contexts to focus on most relevant ones
            temperature=0.7,      # Added temperature for more focused responses
            model_config=model_config,  # Pass the model configuration
            chunk_size=512,       # Set chunk size for text chunking
            chunk_overlap=50      # Set chunk overlap for text chunking
        )
        return rag
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        # Add more detailed error information
        import traceback
        logger.error(f"Full error traceback: {traceback.format_exc()}")
        return None

def process_uploaded_files(uploaded_files: List[Any], rag_system: RAGSystem) -> List[str]:
    """Process uploaded files and return list of processed file names."""
    processed_files = []
    temp_dir = "data"
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save uploaded files
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
        
        # Ingest documents
        rag_system.ingest_documents(temp_dir)
        processed_files = [f.name for f in uploaded_files]
        return processed_files
    except Exception as e:
        logger.error(f"Error processing documents: {str(e)}")
        raise
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

def process_data_directory(data_dir: str, rag_system: RAGSystem) -> List[str]:
    """Process files from a directory and return list of processed file paths."""
    processed_files = []
    try:
        rag_system.ingest_documents(data_dir)
        # Add all files to processed list
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    processed_files.append(file_path)
        return processed_files
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        raise

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
    
if 'vector_dbs' not in st.session_state:
    st.session_state.vector_dbs = {
        'chroma': ChromaDB(),
        'redis': RedisDB()
    }

# Main title
st.title("RAG Ds4300 Midterm Cheat Sheet")

# Sidebar with settings
with st.sidebar:
    st.header("System Settings")
    
    # System status
    st.subheader("System Status")
    
    # Check Redis status
    redis_status = check_redis_status()
    st.write("Redis: ", "✅ Running" if redis_status else "❌ Not Running")
    
    # Check Ollama status and models
    ollama_status, available_models = check_ollama_status()
    st.write("Ollama: ", "✅ Running" if ollama_status else "❌ Not Running")
    
    if not ollama_status:
        st.error("Please ensure Ollama is running")
    
    # Initialize/Update RAG system
    if st.button("Initialize/Update RAG System"):
        st.session_state.rag_system = initialize_rag_system()
        if st.session_state.rag_system:
            st.session_state.initialized = True
            st.success("RAG system initialized successfully!")
    
    # Document processing section
    st.header("Document Processing")
    
    # Process data directory
    if st.button("Process Data Directory"):
        if not st.session_state.initialized:
            st.error("Please initialize the RAG system first!")
        else:
            data_dir = "data"
            if not os.path.exists(data_dir):
                st.error(f"Directory {data_dir} does not exist!")
            else:
                with st.spinner(f"Processing documents from {data_dir}..."):
                    try:
                        new_processed_files = process_data_directory(data_dir, st.session_state.rag_system)
                        st.session_state.processed_files.extend(new_processed_files)
                        st.success(f"Successfully processed documents from {data_dir}")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
    
    # Show processed files
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # Document upload
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files and st.session_state.rag_system:
        try:
            new_processed_files = process_uploaded_files(uploaded_files, st.session_state.rag_system)
            st.session_state.processed_files.extend(new_processed_files)
            st.success(f"Successfully ingested {len(uploaded_files)} documents!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")

# Search type selection
search_type = st.radio(
    "Search Type",
    ["Text Search", "Image Search"],
    horizontal=True
)

if search_type == "Text Search":
    # Text search interface
    st.subheader("Enter your query:")
    query = st.text_area(
        "Query",
        height=150,
        placeholder="multiple line query ftw"
    )
    if st.button("Search"):
        if not st.session_state.initialized:
            st.error("Please initialize the RAG system first!")
        elif not st.session_state.processed_files:
            st.warning("No documents have been processed yet!")
        else:
            with st.spinner("Searching..."):
                try:
                    if st.session_state.rag_system is not None:
                        result = st.session_state.rag_system.query(query)
                        
                        if result:
                            # Display response
                            st.subheader("Response")
                            if 'response' in result:
                                st.write(result['response'])
                            else:
                                st.warning("No response generated.")
                            
                            # Display contexts
                            st.subheader("Relevant Contexts")
                            if 'contexts' in result and result['contexts']:
                                for idx, context in enumerate(result['contexts'], 1):
                                    score = context.get('combined_score', 'N/A')
                                    score_str = f"{score:.3f}" if isinstance(score, float) else str(score)
                                    with st.expander(f"Context {idx} (Score: {score_str})"):
                                        if 'text' in context:
                                            st.write(context['text'])
                                        if 'metadata' in context:
                                            if 'source' in context['metadata']:
                                                st.caption(f"Source: {context['metadata']['source']}")
                                            if 'page' in context['metadata']:
                                                st.caption(f"Page: {context['metadata']['page']}")
                            else:
                                st.warning("No relevant contexts found.")
                        else:
                            st.warning("No results found for your query.")
                    else:
                        st.error("RAG system is not initialized. Please reinitialize the system.")
                        
                except Exception as e:
                    logger.error(f"Error searching documents: {str(e)}")
                    st.error(f"Error searching documents: {str(e)}")

else:
    # Image search interface
    uploaded_image = st.file_uploader(
        "Upload an image to search",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_image and st.session_state.rag_system:
        try:
            # Convert uploaded image to PIL Image
            image = Image.open(uploaded_image)
            
            # Display the uploaded image
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Optional text query
            query = st.text_input("Enter additional text query (optional):")
            
            if st.button("Search by Image"):
                if not st.session_state.initialized:
                    st.error("Please initialize the RAG system first!")
                else:
                    with st.spinner("Searching..."):
                        try:
                            result = st.session_state.rag_system.query(
                                query_text=query or "Find similar images",
                                query_image=image
                            )
                            
                            # Display response
                            st.subheader("Response")
                            st.write(result['response'])
                            
                            # Display contexts
                            st.subheader("Relevant Contexts")
                            for idx, context in enumerate(result['contexts'], 1):
                                with st.expander(f"Context {idx} (Score: {context.get('combined_score', 'N/A'):.3f})"):
                                    if 'image' in context['metadata']:
                                        # Convert base64 image to PIL Image and display
                                        image_data = context['metadata']['image']
                                        st.image(image_data, caption=f"Similar Image {idx}")
                                    if 'text' in context:
                                        st.write(context['text'])
                                    if 'source' in context['metadata']:
                                        st.caption(f"Source: {context['metadata']['source']}")
                                    if 'page' in context['metadata']:
                                        st.caption(f"Page: {context['metadata']['page']}")
                        except Exception as e:
                            logger.error(f"Error searching documents: {str(e)}")
                            st.error(f"Error searching documents: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            st.error(f"Error processing image: {str(e)}")

