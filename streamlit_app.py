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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page config
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
    preferred = ["qwen:7b", "qwen:14b", "deepseek-coder", "mistral", "llama2", "codellama"]
    
    # Filter available models to prioritize preferred ones
    preferred_available = [model for model in preferred if model in available_models]
    other_available = [model for model in available_models if model not in preferred]
    
    # Combine lists with preferred models first
    return preferred_available + other_available

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

# Sidebar with settings
with st.sidebar:
    st.title("RAG System Settings")
    
    # System status
    st.header("System Status")
    
    # Check Redis status
    redis_status = check_redis_status()
    st.write("Redis: ", "✅ Running" if redis_status else "❌ Not Running")
    
    # Check Ollama status and models
    ollama_status, available_models = check_ollama_status()
    st.write("Ollama: ", "✅ Running" if ollama_status else "❌ Not Running")
    
    if not ollama_status:
        st.error("Please ensure Ollama is running")
    
    # System settings
    st.header("Settings")
    
    # Select vector database - default to ChromaDB now
    db_options = ['chroma', 'redis']
    if 'selected_db' not in st.session_state:
        st.session_state.selected_db = 'chroma'  # Default to ChromaDB
        
    selected_db = st.selectbox(
        "Vector Database", 
        options=db_options,
        index=db_options.index(st.session_state.selected_db),
        help="ChromaDB is recommended for multimodal RAG compatibility"
    )
    st.session_state.selected_db = selected_db
    
    # Add info message about ChromaDB
    if selected_db == 'chroma':
        st.info("ChromaDB is the recommended database for best multimodal compatibility.")
    else:
        st.warning("Redis may have issues with multimodal searches. Consider using ChromaDB for full feature support.")
    
    # Select Ollama model - emphasize Qwen:7b
    if ollama_status and available_models:
        preferred_models = get_preferred_models(available_models)
        
        # Default to qwen:7b if available
        default_model = "qwen:7b" if "qwen:7b" in preferred_models else (
            preferred_models[0] if preferred_models else available_models[0]
        )
        
        # Set default in session state if not already set
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = default_model
            
        selected_model = st.selectbox(
            "Ollama Model",
            options=preferred_models,
            index=preferred_models.index(st.session_state.selected_model) 
                if st.session_state.selected_model in preferred_models else 0,
            help="Qwen:7b is recommended for fastest response times (4.6-8.3s)"
        )
        st.session_state.selected_model = selected_model
        
        # Add informational message about model choice
        if selected_model == "qwen:7b":
            st.success("Qwen:7b provides the fastest responses (4.6-8.3s) with excellent RAG performance.")
        elif "qwen" in selected_model.lower():
            st.info("Qwen models are well-suited for RAG with excellent context understanding.")
        elif "mistral" in selected_model.lower():
            st.warning("Mistral has slower response times (8.3-13.3s) than Qwen:7b.")
        elif "llama" in selected_model.lower():
            st.warning("Llama2 has slower response times (8.9-13.6s) than Qwen:7b.")
    else:
        st.error("Ollama is not running or no models are available")
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = "qwen:7b"  # Default to qwen:7b
    
    # Semantic search weight - change default to 0.3
    semantic_weight = st.slider(
        "Semantic Search Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3,  # Changed from 0.7 to 0.3 based on evaluation results
        step=0.1,
        help="Recommended setting: 0.3 (30% semantic, 70% keyword) for best performance"
    )
    
    # Add info about semantic weight
    st.info("Setting 0.3 means 30% semantic search and 70% keyword search - optimized for speed based on evaluations.")
    
    # Initialize/update button
    if st.button("Initialize/Update System"):
        with st.spinner("Setting up RAG system..."):
            try:
                # Get selected vector DB
                vector_db = st.session_state.vector_dbs[selected_db]
                
                # Create LLM with selected model
                llm = OllamaLLM(model_name=st.session_state.selected_model)
                
                # Create RAG system
                st.session_state.rag_system = RAGSystem(
                    vector_db=vector_db,
                    llm=llm,
                    semantic_weight=semantic_weight,
                    keyword_weight=1.0 - semantic_weight
                )
                
                st.session_state.initialized = True
                st.success("RAG system initialized successfully!")
            except Exception as e:
                st.error(f"Error initializing RAG system: {str(e)}")
    
    # Show processed files
    if st.session_state.processed_files:
        st.header("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")
    
    # Clear system button
    if st.button("Clear System"):
        if st.session_state.rag_system:
            st.session_state.rag_system.clear()
            st.session_state.processed_files = []
            st.session_state.uploaded_files = []
            st.success("System cleared successfully!")

# Main panel
st.title("RAG Document Search System")
st.info("""
### Recommended Settings (based on evaluation):
- **Database**: ChromaDB (most compatible with multimodal search)
- **Model**: Qwen:7b (fastest responses: 4.6-8.3 seconds)
- **Semantic Weight**: 0.3 (optimal balance for performance)
""")

# Document processing section
st.header("Document Processing")

# Option to process a data directory
data_dir = st.text_input("Data Directory Path", "data/raw_notes")
if st.button("Process Data Directory"):
    if not st.session_state.initialized:
        st.error("Please initialize the RAG system first!")
    elif not os.path.exists(data_dir):
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

# Option to upload files
uploaded_files = st.file_uploader("Upload Documents", accept_multiple_files=True, type=['pdf'])

if uploaded_files:
    # Check if any new files were uploaded
    new_files = [f for f in uploaded_files if f not in st.session_state.uploaded_files]
    
    if new_files and st.button("Process Uploaded Files"):
        if not st.session_state.initialized:
            st.error("Please initialize the RAG system first!")
        else:
            # Create a temporary directory to store uploaded files
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save uploaded files to temp directory
                for uploaded_file in new_files:
                    file_path = os.path.join(temp_dir, uploaded_file.name)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                
                # Process documents
                with st.spinner("Processing uploaded documents..."):
                    try:
                        if st.session_state.rag_system:
                            st.session_state.rag_system.ingest_documents(temp_dir)
                            
                            # Add to processed files list
                            for uploaded_file in new_files:
                                if uploaded_file.name not in st.session_state.processed_files:
                                    st.session_state.processed_files.append(uploaded_file.name)
                                    st.session_state.uploaded_files.append(uploaded_file)
                                    
                            st.success("Successfully processed uploaded documents")
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")

# Search section
st.header("Search Documents")

# Search input
query = st.text_input("Enter your query:")

if query and st.button("Search"):
    if not st.session_state.initialized:
        st.error("Please initialize the RAG system first!")
    elif not st.session_state.processed_files:
        st.warning("No documents have been processed yet!")
    else:
        with st.spinner("Searching..."):
            try:
                result = st.session_state.rag_system.query(query)
                
                # Display response
                st.subheader("Response:")
                st.write(result.get("response", "No response generated"))
                
                # Display sources
                st.subheader("Sources:")
                sources = result.get("sources", [])
                
                if sources:
                    for i, source in enumerate(sources):
                        with st.expander(f"Source {i+1}: {source.get('metadata', {}).get('source', 'Unknown')} - Page {source.get('metadata', {}).get('page', 'N/A')}"):
                            st.write(source.get('metadata', {}).get('text', 'No text available'))
                else:
                    st.write("No sources found")
                
                # Display images if available
                images = result.get("images", [])
                if images:
                    st.subheader("Related Images:")
                    cols = st.columns(min(3, len(images)))
                    
                    for i, img_data in enumerate(images):
                        col_idx = i % len(cols)
                        img = Image.open(io.BytesIO(img_data["data"]))
                        cols[col_idx].image(img, caption=f"Image {i+1} - Page {img_data.get('page', 'N/A')}")
            except Exception as e:
                st.error(f"Error searching documents: {str(e)}") 