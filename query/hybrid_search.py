from typing import List, Dict, Any, Union, Optional
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from database.base_db import BaseVectorDB

class HybridSearch:
    """Hybrid search combining semantic and keyword-based search."""
    
    def __init__(
        self,
        vector_db: BaseVectorDB,
        embedder: Any,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
        top_k: int = 3
    ):
        """Initialize hybrid search.
        
        Args:
            vector_db: Vector database instance
            embedder: Embedder instance for generating embeddings
            semantic_weight: Weight for semantic search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
            top_k: Number of results to return
        """
        if not 0 <= semantic_weight <= 1 or not 0 <= keyword_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        if abs(semantic_weight + keyword_weight - 1) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        self.vector_db = vector_db
        self.embedder = embedder
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        self.top_k = top_k
        
        # Initialize BM25 for keyword search
        self.bm25 = None
        self.documents = []
        self.scaler = MinMaxScaler()

    def index_documents(self, documents: List[Dict[str, Any]]):
        """Index documents for keyword search.
        
        Args:
            documents: List of documents with 'text' field
        """
        self.documents = documents
        
        # Tokenize documents for BM25
        tokenized_docs = [doc['text'].lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)

    def search(self, query_text: str, query_image: Optional[bytes] = None) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search.
        
        Args:
            query_text: Text query
            query_image: Optional image query
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding based on modality
            if query_image is not None:
                query_embedding = self.embedder.embed_image(query_image)
                modality = "image"
            else:
                query_embedding = self.embedder.embed_text(query_text)
                modality = "text"
            
            # Perform semantic search
            semantic_results = self.vector_db.search(
                query_embedding,
                k=self.top_k,
                modality=modality
            )
            
            # Perform keyword search
            keyword_results = self.keyword_search(query_text)
            
            # Combine results using weighted scoring
            combined_results = self._combine_results(semantic_results, keyword_results)
            
            return combined_results[:self.top_k]
            
        except Exception as e:
            print(f"Error in hybrid search: {str(e)}")
            return []

    def adjust_weights(self, semantic_weight: float, keyword_weight: float):
        """Adjust the weights for semantic and keyword search.
        
        Args:
            semantic_weight: New weight for semantic search (0-1)
            keyword_weight: New weight for keyword search (0-1)
        """
        if not 0 <= semantic_weight <= 1 or not 0 <= keyword_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        if abs(semantic_weight + keyword_weight - 1) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight 

    def keyword_search(self, query_text: str) -> List[Dict[str, Any]]:
        """Perform keyword-based search using BM25.
        
        Args:
            query_text: Text query
            
        Returns:
            List of results with keyword scores
        """
        if not self.bm25:
            return []
        
        # Get keyword search scores
        tokenized_query = query_text.lower().split()
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize keyword scores
        keyword_scores = self.scaler.fit_transform(
            keyword_scores.reshape(-1, 1)
        ).flatten()
        
        # Create results list
        results = []
        for doc_idx, score in enumerate(keyword_scores):
            results.append({
                'keyword_score': score,
                'doc_idx': doc_idx
            })
        
        return results

    def _combine_results(self, semantic_results: List[Dict[str, Any]], 
                        keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine semantic and keyword search results.
        
        Args:
            semantic_results: Results from semantic search
            keyword_results: Results from keyword search
            
        Returns:
            Combined results with weighted scores
        """
        combined_results = []
        
        # Create a mapping of doc_idx to keyword scores
        keyword_scores = {r['doc_idx']: r['keyword_score'] for r in keyword_results}
        
        for sem_result in semantic_results:
            doc_idx = sem_result['metadata'].get('doc_idx', -1)
            if doc_idx >= 0:
                semantic_score = 1 - sem_result['distance']  # Convert distance to similarity
                keyword_score = keyword_scores.get(doc_idx, 0)
                
                combined_score = (
                    self.semantic_weight * semantic_score +
                    self.keyword_weight * keyword_score
                )
                
                combined_results.append({
                    **sem_result,
                    'semantic_score': semantic_score,
                    'keyword_score': keyword_score,
                    'combined_score': combined_score
                })
        
        # Sort by combined score
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results 