"""ChromaDB implementation for vector storage."""

import chromadb
from typing import List, Dict, Any, Optional
import numpy as np
from .base_db import BaseDB, SearchResult

class ChromaDB(BaseDB):
    """ChromaDB implementation for vector storage."""
    
    def __init__(self, collection_name: str = "default"):
        """Initialize ChromaDB.
        
        Args:
            collection_name: Name of the collection to use
        """
        super().__init__()
        self.client = chromadb.Client()
        
        # Try to get existing collection or create new one
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except ValueError:
            self.collection = self.client.create_collection(name=collection_name)
    
    def index(self, embeddings: List[List[float]], chunks: List[str], 
              doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index embeddings with their associated chunks and metadata.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        super().index(embeddings, chunks, doc_ids, metadata)
    
    def _index_impl(self, embeddings: List[List[float]], chunks: List[str], 
                   doc_ids: List[str], metadata: Optional[List[Dict[str, Any]]] = None) -> None:
        """Index embeddings using ChromaDB.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        # Convert embeddings to numpy arrays if needed
        embeddings = [np.array(emb) for emb in embeddings]
        
        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in chunks]
        
        # Add documents to collection
        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata,
            ids=doc_ids
        )
    
    def search(self, query_embedding: List[float], k: int = 3) -> List[SearchResult]:
        """Search for similar vectors using ChromaDB.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of SearchResult objects containing matches
        """
        return self._search_impl(query_embedding, k)
    
    def _search_impl(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        """Search for similar vectors using ChromaDB.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of SearchResult objects containing matches
        """
        # Convert query to numpy array if needed
        query_embedding = np.array(query_embedding)
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        
        # Convert results to SearchResult objects
        search_results = []
        for i in range(len(results['documents'][0])):
            search_results.append(SearchResult(
                doc_id=results['ids'][0][i],
                chunk=results['documents'][0][i],
                score=float(results['distances'][0][i]),
                metadata=results['metadatas'][0][i]
            ))
        
        return search_results
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.create_collection(name=self.collection.name) 