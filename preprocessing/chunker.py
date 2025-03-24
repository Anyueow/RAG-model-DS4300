from typing import List, Dict, Any
import re
from abc import ABC, abstractmethod

class BaseChunker(ABC):
    """Abstract base class for text chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        pass

class TokenChunker(BaseChunker):
    """Chunker that splits text based on token count."""
    
    def __init__(self, chunk_size: int = 400, overlap: int = 30):
        """Initialize the chunker.
        
        Args:
            chunk_size: Target size for each chunk in tokens
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on token count.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Simple tokenization by splitting on whitespace
        tokens = text.split()
        chunks = []
        start = 0
        
        while start < len(tokens):
            end = start + self.chunk_size
            chunk = ' '.join(tokens[start:end])
            chunks.append(chunk)
            start = end - self.overlap
            
        return chunks

class SentenceChunker(BaseChunker):
    """Chunker that splits text based on sentence boundaries."""
    
    def __init__(self, chunk_size: int = 5, overlap: int = 1):
        """Initialize the chunker.
        
        Args:
            chunk_size: Number of sentences per chunk
            overlap: Number of sentences to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks based on sentence boundaries.
        
        Args:
            text: Text to split into chunks
            
        Returns:
            List of text chunks
        """
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        start = 0
        
        while start < len(sentences):
            end = start + self.chunk_size
            chunk = '. '.join(sentences[start:end])
            chunks.append(chunk)
            start = end - self.overlap
            
        return chunks

class ChunkingPipeline:
    """Pipeline for processing documents into chunks."""
    
    def __init__(self, chunker: BaseChunker):
        """Initialize the pipeline.
        
        Args:
            chunker: Chunking strategy to use
        """
        self.chunker = chunker
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process documents into chunks.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed chunks with metadata
        """
        processed_chunks = []
        
        for doc in documents:
            chunks = self.chunker.chunk_text(doc['text'])
            for i, chunk in enumerate(chunks):
                processed_chunks.append({
                    'text': chunk,
                    'document_id': doc['file_path'],
                    'chunk_id': f"{doc['file_path']}_{i}",
                    'file_type': doc['file_type']
                })
        
        return processed_chunks 