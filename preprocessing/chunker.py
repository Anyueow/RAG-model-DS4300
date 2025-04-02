from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import tiktoken  
import re

@dataclass
class ChunkingConfig:
    """Configuration for text chunking."""
    chunk_size: int = 400  # In tokens
    overlap: int = 30      # Token overlap between chunks
    use_tiktoken: bool = True  # Whether to use tiktoken for tokenization

class TextChunker:
    """Handles document chunking with token-based strategy."""
    
    def __init__(self, config: ChunkingConfig = None):
        self.config = config or ChunkingConfig()
        self.tokenizer = tiktoken.get_encoding("cl100k_base") if self.config.use_tiktoken else None

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using either tiktoken or simple word count."""
        if self.config.use_tiktoken:
            return len(self.tokenizer.encode(text))
        return len(text.split())

    def _encode_text(self, text: str) -> List[str]:
        """Encode text into tokens using either tiktoken or simple word split."""
        if self.config.use_tiktoken:
            return self.tokenizer.encode(text)
        return text.split()

    def _decode_tokens(self, tokens: List[str]) -> str:
        """Decode tokens back to text using either tiktoken or simple join."""
        if self.config.use_tiktoken:
            return self.tokenizer.decode(tokens)
        return " ".join(tokens)

    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a document based on token count."""
        text = document.get("text", "")
        metadata = document.get("metadata", {})
        
        if not text.strip():
            return []
            
        return self._token_based_chunking(text, metadata)

    def _token_based_chunking(self, text: str, metadata: dict) -> List[Dict[str, Any]]:
        """Split text into chunks based on token count."""
        tokens = self._encode_text(text)
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Calculate end position for this chunk
            end = min(start + self.config.chunk_size, len(tokens))
            
            # Get the chunk tokens and decode back to text
            chunk_tokens = tokens[start:end]
            chunk_text = self._decode_tokens(chunk_tokens)
            
            # Create chunk with metadata
            chunks.append({
                "text": chunk_text,
                "metadata": metadata,
                "chunk_id": f"{metadata.get('source', 'doc')}_{len(chunks)}",
                "token_count": len(chunk_tokens)
            })
            
            # Break if we've reached the end
            if end >= len(tokens):
                break
                
            # Move start position with overlap, ensuring we don't go backwards
            start = max(end - self.config.overlap, start + 1)
        
        return chunks

class ChunkingPipeline:
    """Orchestrates the chunking process."""
    
    def __init__(self, chunker: TextChunker):
        self.chunker = chunker
    
    def process(self, text: str) -> List[str]:
        """Process text through the chunker.
        
        Args:
            text: Text to be chunked
            
        Returns:
            List of text chunks
        """
        if not text or not text.strip():
            return []
            
        try:
            # Create a document dict for the chunker
            document = {
                "text": text,
                "metadata": {}
            }
            
            # Get chunks from the chunker
            chunks = self.chunker.chunk_document(document)
            
            # Extract just the text from the chunks
            return [chunk["text"] for chunk in chunks]
        except Exception as e:
            print(f"Error processing text: {e}")
            return []
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple documents through the chunker.
        
        Args:
            documents: List of documents to process
            
        Returns:
            List of processed chunks with metadata
        """
        all_chunks = []
        
        for doc in documents:
            try:
                chunks = self.chunker.chunk_document(doc)
                all_chunks.extend(chunks)
            except Exception as e:
                print(f"Failed to chunk document: {e}")
                continue
                
        return all_chunks