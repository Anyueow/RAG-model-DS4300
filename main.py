"""Main RAG system implementation."""

from typing import List, Dict, Any, Optional
from pathlib import Path
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        temperature: float = 0.7,
        model_config: EmbeddingModelConfig = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        collection_name: str = "default",
        top_k: int = 3
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
            temperature: Temperature for response generation
            model_config: Configuration for the embedding model
            chunk_size: Size of text chunks for processing
            chunk_overlap: Overlap between chunks
            collection_name: Name of the collection in vector DB
            top_k: Number of results to return
        """
        # Initialize components with defaults if not provided
        self.model_config = model_config or EMBEDDING_MODELS["nomic-ai/nomic-embed-text-v1.5"]
        self.embedder = embedder or SentenceTransformerEmbedder(self.model_config)
        self.vector_db = vector_db or ChromaDB(collection_name=collection_name)
        self.document_loader = document_loader or PDFLoader()
        self.llm = llm or OllamaLLM(model_name="mistral:7b", temperature=temperature)
        self.prompt_generator = prompt_generator or PromptGenerator()
        self.top_k = top_k
        
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
        self.embedding_cache = {}  # Cache for embeddings
        self.context_cache = {}  # Cache for contexts

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
        
        # Process documents in parallel chunks for better memory management
        chunk_size = 10  # Process 10 documents at a time
        
        def process_document_chunk(chunk):
            chunked_texts = []
            doc_ids = []
            metadata = []
            
            for doc_idx, doc in enumerate(chunk):
                if doc is None or not doc.text:
                    continue
                    
                try:
                    # Get chunks for this document
                    doc_chunks = self.chunking_pipeline.process(doc.text)
                    if doc_chunks:  # Only add if we got valid chunks
                        # Add chunks and their metadata
                        for chunk_idx, chunk_data in enumerate(doc_chunks):
                            chunked_texts.append(chunk_data['text'])
                            doc_ids.append(f"doc_{doc_idx}_{chunk_idx}")
                            metadata.append({
                                'text': chunk_data['text'],
                                'source': doc.metadata.get('source'),
                                'page': doc.metadata.get('page'),
                                'doc_idx': doc_idx,
                                'chunk_idx': chunk_idx,
                                'type': 'text'
                            })
                except Exception as e:
                    print(f"Error processing document: {str(e)}")
                    continue
            
            return chunked_texts, doc_ids, metadata
            
        # Process documents in parallel
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for i in range(0, len(self.documents), chunk_size):
                chunk = self.documents[i:i + chunk_size]
                futures.append(executor.submit(process_document_chunk, chunk))
            
            # Collect results
            for future in as_completed(futures):
                try:
                    chunked_texts, doc_ids, metadata = future.result()
                    if chunked_texts:  # Only proceed if we have chunks
                        # Generate embeddings for the chunks
                        embeddings, _ = self.embedder.embed_batch(chunked_texts)
                        
                        # Add chunks to vector database
                        self.vector_db.index(embeddings, chunked_texts, doc_ids, metadata)
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue
        
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
        
        # Check embedding cache
        if query_text in self.embedding_cache:
            query_embedding = self.embedding_cache[query_text]
        else:
            query_embedding, _ = self.embedder.embed_text(query_text)
            self.embedding_cache[query_text] = query_embedding
        
        # Check context cache
        context_cache_key = f"{query_text}_{self.top_k}"
        if context_cache_key in self.context_cache:
            contexts = self.context_cache[context_cache_key]
        else:
            # Get contexts from vector DB
            contexts = self.vector_db.search(query_embedding, k=self.top_k)
            self.context_cache[context_cache_key] = contexts
        
        # Process query through the query pipeline
        result = self.query_pipeline.process(
            query_text=query_text,
            query_image=query_image,
            use_general_knowledge=use_general_knowledge
        )
        
        # Cache the result
        self.query_cache[cache_key] = result
        return result 