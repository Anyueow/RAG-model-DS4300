import streamlit as st
import os
from typing import Dict, Any
import tempfile
import requests
from main import RAGSystem
from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from database.qdrant_db import QdrantDB
from embeddings.multimodal_embedder import MultiModalEmbedder
import ollama

# Create necessary directories
if not os.path.exists("chroma_db"):
    os.makedirs("chroma_db")

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'vector_dbs' not in st.session_state:
    try:
        st.session_state.vector_dbs = {
            "chroma": ChromaDB(persist_directory="chroma_db"),
            "redis": RedisVectorDB(),
            "qdrant": QdrantDB(path="./qdrant_db")
        }
    except Exception as e:
        st.error(f"Error initializing vector databases: {str(e)}")
        st.session_state.vector_dbs = {
            "chroma": ChromaDB(persist_directory="chroma_db"),
            "redis": RedisVectorDB()
        }

def check_redis_status():
    """Check if Redis is running."""
    try:
        from redis import Redis
        client = Redis(host="localhost", port=6379, db=0)
        client.ping()
        return True
    except:
        return False

def check_ollama_status():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/version")
        return response.status_code == 200
    except:
        return False

def get_pdf_files(directory: str) -> list:
    """Get list of PDF files in the specified directory."""
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.pdf')]

def initialize_rag_system(db_type: str, model_name: str, semantic_weight: float):
    """Initialize or update the RAG system with selected settings."""
    embedder = MultiModalEmbedder()
    vector_db = st.session_state.vector_dbs[db_type]
    
    st.session_state.rag_system = RAGSystem(
        embedder=embedder,
        vector_db=vector_db,
        semantic_weight=semantic_weight,
        keyword_weight=1 - semantic_weight
    )

def process_documents(directory: str):
    """Process documents using the current RAG system."""
    if not st.session_state.rag_system:
        st.error("Please initialize the RAG system first")
        return False
    
    try:
        st.session_state.rag_system.ingest_documents(directory)
        return True
    except Exception as e:
        st.error(f"Error processing documents: {str(e)}")
        return False

# Page configuration
st.set_page_config(page_title="RAG Document Search", layout="wide")
st.title("📚 RAG Document Search System")

# System status indicators
col1, col2 = st.columns(2)
with col1:
    redis_status = check_redis_status()
    st.metric("Redis Status", "Connected" if redis_status else "Disconnected")
    if not redis_status:
        st.error("Redis not running. Start with: docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest")

with col2:
    ollama_status = check_ollama_status()
    st.metric("Ollama Status", "Running" if ollama_status else "Not Running")
    if not ollama_status:
        st.error("Ollama not running. Start with: ollama serve")

# Sidebar for settings and document management
with st.sidebar:
    st.header("System Settings")
    
    # Vector Database Selection
    db_type = st.selectbox(
        "Select Vector Database",
        options=["chroma", "redis", "qdrant"],
        help="Choose the vector database for document storage"
    )
    
    # Model Selection
    try:
        models = [model["name"] for model in ollama.list()["models"]]
    except:
        models = ["llama2"]
    
    model_name = st.selectbox(
        "Select LLM Model",
        options=models,
        help="Choose the Ollama model for text generation"
    )
    
    # Search Weight
    semantic_weight = st.slider(
        "Semantic Search Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        help="Balance between semantic (1.0) and keyword (0.0) search"
    )
    
    # Initialize/Update System
    if st.button("Initialize/Update System"):
        with st.spinner("Initializing RAG system..."):
            initialize_rag_system(db_type, model_name, semantic_weight)
            st.success("System initialized!")
    
    st.divider()
    
    # Document Management
    st.header("Document Management")
    
    # Data Directory Processing
    data_dir = "data/raw_notes"
    pdf_files = get_pdf_files(data_dir)
    if pdf_files:
        st.write(f"Found {len(pdf_files)} PDF files in data directory")
        if st.button("Process Data Directory"):
            if not st.session_state.rag_system:
                st.error("Please initialize the system first")
            else:
                with st.spinner("Processing documents..."):
                    success = process_documents(data_dir)
                    if success:
                        st.session_state.documents_processed = True
                        st.success("Documents processed successfully!")
    else:
        st.warning("No PDF files found in data directory")
    
    # File Upload
    st.subheader("Manual Upload")
    uploaded_files = st.file_uploader(
        "Upload PDF Documents",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        if st.button("Process Uploaded Documents"):
            if not st.session_state.rag_system:
                st.error("Please initialize the system first")
            else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Save uploaded files
                    for uploaded_file in uploaded_files:
                        file_path = os.path.join(temp_dir, uploaded_file.name)
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getvalue())
                    
                    # Process documents
                    with st.spinner("Processing uploaded documents..."):
                        success = process_documents(temp_dir)
                        if success:
                            st.session_state.documents_processed = True
                            st.success("Documents processed successfully!")

# Main content area for search
st.header("Search Documents")

# Create tabs for text and image search
tab1, tab2 = st.tabs(["Text Search", "Image Search"])

with tab1:
    # Text search interface
    query = st.text_input("Enter your question:", placeholder="Ask a question about your documents...")
    
    if query:
        if not st.session_state.rag_system:
            st.error("Please initialize the system first")
        elif not st.session_state.documents_processed:
            st.error("Please process documents before searching")
        else:
            with st.spinner("Searching..."):
                try:
                    # Get response from RAG system
                    result = st.session_state.rag_system.query(query)
                    
                    # Display response
                    st.subheader("Answer:")
                    st.write(result['response'])
                    
                    # Display sources
                    if result.get('sources'):
                        st.subheader("📚 Sources:")
                        
                        # Display sources by type
                        for source in result['sources']:
                            if isinstance(source, dict):
                                if source.get('header'):
                                    # Section header
                                    st.markdown(f"**{source['header']}**")
                                else:
                                    # Source entry
                                    citation = source.get('citation', '')
                                    source_name = source.get('source', 'Unknown')
                                    page = source.get('page')
                                    
                                    source_text = f"• {citation} - {source_name}"
                                    if page is not None:
                                        source_text += f" (Page {page})"
                                    st.write(source_text)
                    
                    # Display images
                    if result.get('images'):
                        st.subheader("📷 Related Images:")
                        for img in result['images']:
                            if img.get('data'):
                                st.image(img['data'], caption=f"Image {img.get('index')}")
                                
                except Exception as e:
                    st.error(f"Error during search: {str(e)}")

with tab2:
    # Image search interface
    st.subheader("Upload a Tree Diagram or Related Image")
    uploaded_image = st.file_uploader(
        "Upload an image",
        type=['png', 'jpg', 'jpeg'],
        key="image_uploader"
    )
    
    if uploaded_image:
        # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        # Add a question input for the image
        image_query = st.text_input(
            "Ask about this image:",
            placeholder="e.g., What type of tree traversal is shown in this diagram?",
            key="image_query"
        )
        
        if image_query:
            if not st.session_state.rag_system:
                st.error("Please initialize the system first")
            elif not st.session_state.documents_processed:
                st.error("Please process documents before searching")
            else:
                with st.spinner("Analyzing image and searching..."):
                    try:
                        # Convert uploaded image to bytes
                        image_bytes = uploaded_image.getvalue()
                        
                        # Get response from RAG system with image
                        result = st.session_state.rag_system.query(
                            image_query,
                            image_data=image_bytes,
                            modality="image"
                        )
                        
                        # Display response
                        st.subheader("Answer:")
                        st.write(result['response'])
                        
                        # Display sources
                        if result.get('sources'):
                            st.subheader("📚 Sources:")
                            
                            # Display sources by type
                            for source in result['sources']:
                                if isinstance(source, dict):
                                    if source.get('header'):
                                        # Section header
                                        st.markdown(f"**{source['header']}**")
                                    else:
                                        # Source entry
                                        citation = source.get('citation', '')
                                        source_name = source.get('source', 'Unknown')
                                        page = source.get('page')
                                        
                                        source_text = f"• {citation} - {source_name}"
                                        if page is not None:
                                            source_text += f" (Page {page})"
                                        st.write(source_text)
                        
                        # Display related images
                        if result.get('images'):
                            st.subheader("📷 Related Images:")
                            for img in result['images']:
                                if img.get('data'):
                                    st.image(img['data'], caption=f"Image {img.get('index')}")
                                    
                    except Exception as e:
                        st.error(f"Error during image search: {str(e)}")

# Help section
with st.expander("ℹ️ How to use this system"):
    st.markdown("""
    1. **System Setup**: 
       - Choose your preferred vector database (ChromaDB, Redis, or Qdrant)
       - Select an Ollama model for text generation
       - Adjust the semantic search weight
       - Click "Initialize/Update System"
       
    2. **Document Loading**: 
       - Automatic: Place PDFs in the `data` directory and click "Process Data Directory"
       - Manual: Upload PDFs using the file uploader and click "Process Uploaded Documents"
       
    3. **Search**: 
       - Text Search: Enter your question in the search box
       - Image Search: Upload a tree diagram or related image and ask questions about it
       - View the AI-generated answer with sources
       - Check related images when available
       
    4. **Tips**:
       - Higher semantic weight (→1.0) favors meaning-based search
       - Lower semantic weight (→0.0) favors keyword-based search
       - Different vector databases may perform differently for your use case
       - For tree diagrams, try asking specific questions about traversal types or node relationships
    """) 