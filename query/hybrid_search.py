from typing import List, Dict, Any, Union
import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.preprocessing import MinMaxScaler
from database.base_db import BaseVectorDB

class HybridSearch:
    """Hybrid search combining semantic and keyword-based search."""
    
    def __init__(
        self,
        vector_db: BaseVectorDB,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3
    ):
        """Initialize hybrid search.
        
        Args:
            vector_db: Vector database instance
            semantic_weight: Weight for semantic search scores (0-1)
            keyword_weight: Weight for keyword search scores (0-1)
        """
        if not 0 <= semantic_weight <= 1 or not 0 <= keyword_weight <= 1:
            raise ValueError("Weights must be between 0 and 1")
        if abs(semantic_weight + keyword_weight - 1) > 1e-6:
            raise ValueError("Weights must sum to 1")
            
        self.vector_db = vector_db
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
        
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

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 5,
        modality: str = "text"
    ) -> List[Dict[str, Any]]:
        """Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Query string
            query_embedding: Query embedding vector
            k: Number of results to return
            modality: Type of search ('text' or 'image')
            
        Returns:
            List of results with combined scores
        """
        # Get semantic search results
        semantic_results = self.vector_db.search(
            query_vector=query_embedding,
            k=k,
            modality=modality
        )
        
        if not self.bm25 or modality == "image":
            # If no keyword index or image search, return semantic results only
            return semantic_results
        
        # Get keyword search scores
        tokenized_query = query.lower().split()
        keyword_scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize keyword scores
        keyword_scores = self.scaler.fit_transform(
            keyword_scores.reshape(-1, 1)
        ).flatten()
        
        # Combine scores
        combined_results = []
        for sem_result in semantic_results:
            doc_idx = sem_result['metadata'].get('doc_idx', -1)
            if doc_idx >= 0:
                semantic_score = 1 - sem_result['distance']  # Convert distance to similarity
                keyword_score = keyword_scores[doc_idx]
                
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
        return combined_results[:k]

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