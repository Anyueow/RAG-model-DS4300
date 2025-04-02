"""Streamlit UI for the RAG system."""

# Standard library imports
import os
import sys
import logging
import socket
import subprocess
import traceback
from pathlib import Path
from typing import List, Dict, Any, Optional

# Third-party imports
import streamlit as st
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
logging.basicConfig(level=logging.DEBUG)
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
        logger.info("Initializing RAG system...")
        
        # Use Nomic embedder as default
        model_config = EMBEDDING_MODELS["nomic-ai/nomic-embed-text-v1.5"]
        logger.debug(f"Using model config: {model_config}")
        
        embedder = SentenceTransformerEmbedder(model_config)
        logger.debug("Created embedder")
        
        # Initialize LLM with Mistral
        llm = OllamaLLM(model_name="qwen:7b", temperature=0.4)
        logger.debug("Created LLM with Qwen")
        
        # Initialize RAG system with optimized settings
        rag = RAGSystem(
            embedder=embedder,
            vector_db=ChromaDB(collection_name="app_collection"),
            llm=llm,  # Explicitly pass the LLM instance
            semantic_weight=0.8,  # Increased semantic weight for better semantic understanding
            keyword_weight=0.2,   # Reduced keyword weight to focus more on semantic meaning
            top_k=3,              # Reduced number of contexts to focus on most relevant ones
            temperature=0.7,      # Added temperature for more focused responses
            model_config=model_config,  # Pass the model configuration
            chunk_size=512,       # Set chunk size for text chunking
            chunk_overlap=50,     # Set chunk overlap for text chunking
            collection_name="app_collection"  # Set collection name for vector DB
        )
        logger.info("RAG system initialized successfully")
        return rag
    except Exception as e:
        logger.error(f"Error initializing RAG system: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None

def process_data_directory(data_dir: str, rag_system: RAGSystem) -> List[str]:
    """Process files from a directory and return list of processed file paths."""
    processed_files = []
    try:
        logger.info(f"Processing data directory: {data_dir}")
        rag_system.ingest_documents(data_dir)
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.pdf'):
                    file_path = os.path.join(root, file)
                    processed_files.append(file_path)
        logger.info(f"Processed {len(processed_files)} files")
        return processed_files
    except Exception as e:
        logger.error(f"Error processing directory: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
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
    
    st.subheader("System Status")
    
    redis_status = check_redis_status()
    st.write("Redis: ", "✅ Running" if redis_status else "❌ Not Running")
    
    ollama_status, available_models = check_ollama_status()
    st.write("Ollama: ", "✅ Running" if ollama_status else "❌ Not Running")
    
    if not ollama_status:
        st.error("Please ensure Ollama is running")
    
    if st.button("Initialize/Update RAG System"):
        st.session_state.rag_system = initialize_rag_system()
        if st.session_state.rag_system:
            st.session_state.initialized = True
            st.success("RAG system initialized successfully!")
        else:
            st.error("Failed to initialize RAG system. Check logs for details.")
    
    st.header("Document Processing")
    
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
                        logger.error(f"Full traceback: {traceback.format_exc()}")
    
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")


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
                    logger.info(f"Processing query: {query[:100]}...")
                    result = st.session_state.rag_system.query(query)
                    
                    if result:
                        st.subheader("Response")
                        if 'response' in result:
                            st.write(result['response'])
                        else:
                            st.warning("No response generated.")
                            logger.warning("No response in result")
                        
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
                            logger.warning("No contexts in result")
                    else:
                        st.warning("No results found for your query.")
                        logger.warning("No results returned from query")
                else:
                    st.error("RAG system is not initialized. Please reinitialize the system.")
                    logger.error("RAG system is None")
                    
            except Exception as e:
                logger.error(f"Error searching documents: {str(e)}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                st.error(f"Error searching documents: {str(e)}")
