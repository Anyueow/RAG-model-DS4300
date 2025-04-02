"""Main RAG system implementation."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import os

from ingestion.data_loader import PDFLoader
from preprocessing.chunker import TextChunker, ChunkingPipeline, ChunkingConfig
from embeddings.base_embedder import BaseEmbedder
from database.chroma_db import ChromaDB
from query.query_handler import QueryPipeline
from llm.llm_interface import OllamaLLM, LLMPipeline
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from query.hybrid_search import HybridSearch
from llm.prompt_generator import PromptGenerator
from embeddings.test_config import EMBEDDING_MODELS, EmbeddingModelConfig

class RAGSystem:
    """RAG system that handles document ingestion and querying."""
    
    def __init__(
        self,
        embedder: BaseEmbedder = None,
        vector_db: Any = None,
        document_loader: Any = None,
        llm: OllamaLLM = None,
        prompt_generator: PromptGenerator = None,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 3,
        temperature: float = 0.7,
        model_config: EmbeddingModelConfig = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
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
            model_config: Configuration for the embedding model
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
        """
        # Initialize components with defaults if not provided
        self.model_config = model_config or EMBEDDING_MODELS["nomic-embed-text-v2-moe"]
        self.embedder = embedder or SentenceTransformerEmbedder(self.model_config)
        self.vector_db = vector_db or ChromaDB(persist_directory="chroma_db")
        self.document_loader = document_loader or PDFLoader()
        self.llm = llm or OllamaLLM(model_name="qwen:7b", temperature=temperature)
        self.prompt_generator = prompt_generator or PromptGenerator()
        
        # Initialize chunking pipeline with provided parameters
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size,
            overlap=chunk_overlap,
            use_tiktoken=True
        )
        self.chunker = TextChunker(config=chunking_config)
        self.chunking_pipeline = ChunkingPipeline(self.chunker)
        
        # Initialize query pipeline
        self.query_pipeline = QueryPipeline(
            embedder=self.embedder,
            vector_db=self.vector_db,
            llm=self.llm,
            prompt_generator=self.prompt_generator
        )
        
        # Initialize hybrid search with adjusted weights
        self.search = HybridSearch(
            vector_db=self.vector_db,
            embedder=self.embedder,
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
        documents = []
        
        # Load from main directory
        if os.path.exists(data_dir):
            documents.extend(self.document_loader.load_directory(data_dir))
            
            
        # Validate documents
        if not documents:
            raise ValueError(f"No documents found in {data_dir}")
            
        self.documents = documents
        
        # Process documents in chunks for better memory management
        chunk_size = 10  # Process 10 documents at a time
        for i in range(0, len(self.documents), chunk_size):
            chunk = self.documents[i:i + chunk_size]
            
            # Process each document through the chunking pipeline
            chunked_texts = []
            for doc in chunk:
                if doc is None or not doc.text:
                    continue
                    
                try:
                    chunks = self.chunking_pipeline.process(doc.text)
                    if chunks:  # Only add if we got valid chunks
                        chunked_texts.extend(chunks)
                except Exception as e:
                    print(f"Error processing document: {str(e)}")
                    continue
            
            if chunked_texts:  # Only proceed if we have chunks
                # Generate embeddings for the chunks
                embeddings = self.embedder.embed_batch(chunked_texts)
                
                # Prepare metadata for the chunks
                metadata = []
                for idx, doc in enumerate(chunk):
                    if doc is None or not doc.text:
                        continue
                        
                    try:
                        for chunk_idx, chunk_text in enumerate(self.chunking_pipeline.process(doc.text)):
                            metadata.append({
                                'text': chunk_text,
                                'source': doc.metadata.get('source'),
                                'page': doc.metadata.get('page'),
                                'doc_idx': idx + i,
                                'chunk_idx': chunk_idx,
                                'type': 'text'
                            })
                    except Exception as e:
                        print(f"Error processing document metadata: {str(e)}")
                        continue
                
                if metadata:  # Only proceed if we have valid metadata
                    # Generate unique IDs for chunks
                    doc_ids = [f"doc_{i}_{j}" for i in range(i, i + len(chunk)) 
                              for j in range(len(self.chunking_pipeline.process(chunk[i-i].text)))]
                    
                    # Add chunks to vector database
                    self.vector_db.index(embeddings, chunked_texts, doc_ids, metadata)
        
        # Index documents for hybrid search
        valid_documents = []
        for idx, doc in enumerate(self.documents):
            if doc is None or not doc.text:
                continue
                
            try:
                valid_documents.append({
                    'text': doc.text,
                    'source': doc.metadata.get('source'),
                    'page': doc.metadata.get('page'),
                    'doc_idx': idx,
                    'type': 'text'
                })
            except Exception as e:
                print(f"Error processing document for hybrid search: {str(e)}")
                continue
        
        if valid_documents:
            self.search.index_documents(valid_documents)

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
        
        # Process query through the query pipeline
        result = self.query_pipeline.process(
            query_text=query_text,
            query_image=query_image,
            use_general_knowledge=use_general_knowledge
        )
        
        # Cache the result
        self.query_cache[cache_key] = result
        return result 