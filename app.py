import streamlit as st
import os
import sys
import pandas as pd
from PIL import Image
import io
import time
import socket
import subprocess
import logging
from typing import List, Dict, Any, Optional
import tempfile
from pathlib import Path

# Add the current directory to the path so we can import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from main import RAGSystem
from llm.llm_interface import OllamaLLM
from embeddings.multimodal_embedder import MultiModalEmbedder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config - this must be the first Streamlit command
st.set_page_config(
    page_title="RAG Document Search System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper functions
def check_redis_status():
    """Check if Redis is running."""
    try:
        s = socket.socket()
        s.connect(('localhost', 6379))
        s.close()
        return True
    except:
        return False

def check_ollama_status():
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

def get_preferred_models(available_models):
    """Get a list of preferred models for RAG from available models."""
    # Preferred models in order of preference - place qwen:7b at the top
    preferred = ["qwen:7b"]
    
    # Filter available models to prioritize preferred ones
    preferred_available = [model for model in preferred if model in available_models]
    other_available = [model for model in available_models if model not in preferred]
    
    # Combine lists with preferred models first
    return preferred_available + other_available

def initialize_rag_system():
    """Initialize the RAG system with default settings."""
    try:
        embedder = MultiModalEmbedder()
        rag = RAGSystem(
            embedder=embedder,
            semantic_weight=0.8,  # Increased semantic weight for better semantic understanding
            keyword_weight=0.2,   # Reduced keyword weight to focus more on semantic meaning
            top_k=3,              # Reduced number of contexts to focus on most relevant ones
            temperature=0.7       # Added temperature for more focused responses
        )
        return rag
    except Exception as e:
        st.error(f"Error initializing RAG system: {str(e)}")
        return None

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False

if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
    
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
    
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = []

if 'vector_dbs' not in st.session_state:
    st.session_state.vector_dbs = {
        'chroma': ChromaDB(persist_directory="chroma_db"),
        'redis': RedisVectorDB()
    }

# Main title
st.title("RAG Document Search System")

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
                        if st.session_state.rag_system:
                            st.session_state.rag_system.ingest_documents(data_dir)
                            # Add all files to processed list
                            for root, _, files in os.walk(data_dir):
                                for file in files:
                                    if file.endswith('.pdf'):
                                        file_path = os.path.join(root, file)
                                        if file_path not in st.session_state.processed_files:
                                            st.session_state.processed_files.append(file_path)
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
    
    if uploaded_files:
        # Create temporary directory for uploaded files
        temp_dir = "temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        
        try:
            # Save uploaded files
            for uploaded_file in uploaded_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
            
            # Ingest documents
            if st.session_state.rag_system:
                st.session_state.rag_system.ingest_documents(temp_dir)
                # Add uploaded files to processed list
                for uploaded_file in uploaded_files:
                    if uploaded_file.name not in st.session_state.processed_files:
                        st.session_state.processed_files.append(uploaded_file.name)
                st.success(f"Successfully ingested {len(uploaded_files)} documents!")
        except Exception as e:
            st.error(f"Error processing documents: {str(e)}")
        finally:
            # Clean up temporary files only if directory exists
            if os.path.exists(temp_dir):
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)

# Main content area
st.header("Search")

# Search type selection
search_type = st.radio(
    "Search Type",
    ["Text Search", "Image Search"],
    horizontal=True
)

if search_type == "Text Search":
    # Text search interface
    query = st.text_input("Enter your query:")
    if st.button("Search"):
        if not st.session_state.initialized:
            st.error("Please initialize the RAG system first!")
        elif not st.session_state.processed_files:
            st.warning("No documents have been processed yet!")
        else:
            with st.spinner("Searching..."):
                try:
                    # Add debug information
                    st.write("Debug: Starting search with query:", query)
                    st.write("Debug: Number of processed files:", len(st.session_state.processed_files))
                    
                    # Ensure RAG system is properly initialized
                    if st.session_state.rag_system is not None:
                        # Perform the search
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
                                    with st.expander(f"Context {idx} (Score: {context.get('combined_score', 'N/A'):.3f})"):
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
                    st.error(f"Error searching documents: {str(e)}")
                    # Add more detailed error information
                    st.write("Debug: Full error details:", e.__class__.__name__)
                    import traceback
                    st.code(traceback.format_exc())

else:
    # Image search interface
    uploaded_image = st.file_uploader(
        "Upload an image to search",
        type=["jpg", "jpeg", "png"],
        key="image_uploader"
    )
    
    if uploaded_image:
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
                                query=query or "Find similar images",
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
                            st.error(f"Error searching documents: {str(e)}")
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

# Help section
with st.expander("Help"):
    st.markdown("""
    ### How to use this RAG system:
    
    1. **Initialize the System**:
       - Click the "Initialize/Update RAG System" button in the sidebar
       - Wait for the initialization to complete
    
    2. **Upload Documents**:
       - Use the file uploader in the sidebar to upload PDF documents
       - The system will automatically process and index the documents
    
    3. **Search**:
       - Choose between Text Search or Image Search
       - For Text Search:
         - Enter your query in the text box
         - Click "Search" to get results
       - For Image Search:
         - Upload an image using the image uploader
         - Optionally add text to refine the search
         - Click "Search by Image" to find similar images
    
    4. **View Results**:
       - The system will show a response and relevant contexts
       - Expand each context to see more details
    """)