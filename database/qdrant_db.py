from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from .base_db import BaseDB, SearchResult
import uuid
import os
import shutil

class QdrantDB(BaseDB):
    """Qdrant implementation for vector storage."""
    
    def __init__(self, collection_name: str = "default"):
        """Initialize Qdrant.
        
        Args:
            collection_name: Name of the collection to use
        """
        super().__init__()
        self.client = QdrantClient(":memory:")  # Use in-memory storage for testing
        self.collection_name = collection_name
        
        # Create collection if it doesn't exist
        self.client.recreate_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=768,  # Default to 768 dimensions for nomic-embed-text-v1.5
                distance=models.Distance.COSINE
            )
        )
    
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
        """Index embeddings using Qdrant.
        
        Args:
            embeddings: List of embedding vectors
            chunks: List of text chunks
            doc_ids: List of document IDs
            metadata: Optional list of metadata dictionaries
        """
        # Convert embeddings to numpy arrays if needed
        embeddings = [np.array(emb) for emb in embeddings]
        
        # Prepare points for insertion
        points = []
        for i, (emb, chunk, doc_id) in enumerate(zip(embeddings, chunks, doc_ids)):
            point_metadata = metadata[i] if metadata else {}
            point_metadata['chunk'] = chunk
            
            points.append(models.PointStruct(
                id=i,
                vector=emb.tolist(),
                payload={
                    'doc_id': doc_id,
                    'metadata': point_metadata
                }
            ))
        
        # Insert points into collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def _search_impl(self, query_embedding: List[float], k: int) -> List[SearchResult]:
        """Search for similar vectors using Qdrant.
        
        Args:
            query_embedding: Query vector to search for
            k: Number of results to return
            
        Returns:
            List of SearchResult objects containing matches
        """
        # Convert query to numpy array if needed
        query_embedding = np.array(query_embedding)
        
        # Search collection
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=k
        )
        
        # Convert results to SearchResult objects
        search_results = []
        for hit in results:
            payload = hit.payload
            search_results.append(SearchResult(
                doc_id=payload['doc_id'],
                chunk=payload['metadata']['chunk'],
                score=float(hit.score),
                metadata=payload['metadata']
            ))
        
        return search_results
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=768,  # Default size
                distance=models.Distance.COSINE
            )
        )

    def add_vectors(
        self,
        vectors: List[np.ndarray],
        metadata: List[Dict[str, Any]],
        modality: str = "text"
    ) -> List[str]:
        """Add vectors to the database.
        
        Args:
            vectors: List of vectors to add
            metadata: List of metadata dictionaries
            modality: Type of vectors ('text' or 'image')
            
        Returns:
            List of IDs for added vectors
        """
        if len(vectors) != len(metadata):
            raise ValueError("Number of vectors and metadata entries must match")
            
        # Generate IDs for new vectors
        ids = [str(uuid.uuid4()) for _ in range(len(vectors))]
        
        # Create points
        points = [
            models.PointStruct(
                id=id_,
                vector={modality: vector.tolist()},
                payload={
                    "metadata": meta,
                    "modality": modality,
                    "doc_idx": idx
                }
            )
            for idx, (id_, vector, meta) in enumerate(zip(ids, vectors, metadata))
        ]
        
        # Add points to collection
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        return ids

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        modality: str = "text",
        filter_conditions: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            modality: Type of query ('text' or 'image')
            filter_conditions: Optional filtering conditions
            
        Returns:
            List of results with metadata and distances
        """
        # Create search query
        search_query = {modality: query_vector.tolist()}
        
        # Add filter if provided
        search_params = {}
        if filter_conditions:
            search_params["filter"] = models.Filter(**filter_conditions)
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=search_query,
            limit=k,
            **search_params
        )
        
        # Format results
        formatted_results = []
        for res in results:
            formatted_results.append({
                'id': res.id,
                'metadata': res.payload.get('metadata', {}),
                'modality': res.payload.get('modality', modality),
                'doc_idx': res.payload.get('doc_idx', -1),
                'distance': res.score
            })
            
        return formatted_results

    def delete_vectors(self, ids: List[str]) -> None:
        """Delete vectors by their IDs.
        
        Args:
            ids: List of vector IDs to delete
        """
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.PointIdsList(
                points=ids
            )
        )

    def get_vector_count(self) -> int:
        """Get the total number of vectors in the database."""
        collection_info = self.client.get_collection(self.collection_name)
        return collection_info.vectors_count
 