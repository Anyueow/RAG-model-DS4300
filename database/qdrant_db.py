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
    
    def __init__(self, collection_name: str = "default", embedding_model: str = "nomic-ai/nomic-embed-text-v1.5"):
        """Initialize Qdrant.
        
        Args:
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        super().__init__()
        self.client = QdrantClient(":memory:")  # Use in-memory storage for testing
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        
        # Get vector size based on model
        if "minilm" in embedding_model.lower():
            vector_size = 384
        elif "mpnet" in embedding_model.lower():
            vector_size = 768
        else:  # default to nomic-embed-text-v1.5
            vector_size = 768
            
        # Create collection if it doesn't exist
        try:
            self.client.get_collection(self.collection_name)
        except Exception:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
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
        # Debug: Print information about embeddings
        print(f"\n[DEBUG] Indexing {len(embeddings)} embeddings")
        print(f"[DEBUG] First embedding type: {type(embeddings[0])}")
        print(f"[DEBUG] First embedding shape: {np.array(embeddings[0]).shape}")
        
        # Convert embeddings to numpy arrays if needed
        embeddings = [np.array(emb) for emb in embeddings]
        
        # Debug: Print numpy array information
        print(f"[DEBUG] First numpy array type: {type(embeddings[0])}")
        print(f"[DEBUG] First numpy array shape: {embeddings[0].shape}")
        
        # Create points
        points = []
        for i, (emb, chunk, doc_id) in enumerate(zip(embeddings, chunks, doc_ids)):
            # Prepare metadata
            point_metadata = metadata[i] if metadata else {}
            point_metadata['chunk'] = chunk
            
            # Debug: Print point information
            print(f"\n[DEBUG] Creating point {i}")
            print(f"[DEBUG] Embedding type: {type(emb)}")
            print(f"[DEBUG] Embedding shape: {emb.shape}")
            print(f"[DEBUG] Metadata: {point_metadata}")
            
            # Create point
            point = models.PointStruct(
                id=i,
                vector=emb.tolist(),  # Convert numpy array to list
                payload=point_metadata
            )
            points.append(point)
        
        # Debug: Print batch information
        print(f"\n[DEBUG] Batch size: {len(points)}")
        print(f"[DEBUG] First point vector type: {type(points[0].vector)}")
        print(f"[DEBUG] First point vector length: {len(points[0].vector)}")
        
        # Upsert points
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
        # Debug: Print query information
        print(f"\n[DEBUG] Search query type: {type(query_embedding)}")
        print(f"[DEBUG] Search query shape: {np.array(query_embedding).shape}")
        
        # Convert query to numpy array if needed
        query_embedding = np.array(query_embedding)
        
        # Debug: Print numpy query information
        print(f"[DEBUG] Numpy query type: {type(query_embedding)}")
        print(f"[DEBUG] Numpy query shape: {query_embedding.shape}")
        
        # Search
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),  # Convert numpy array to list
            limit=k
        )
        
        # Debug: Print search results
        print(f"\n[DEBUG] Search results count: {len(search_result)}")
        if search_result:
            print(f"[DEBUG] First result score: {search_result[0].score}")
            print(f"[DEBUG] First result payload: {search_result[0].payload}")
        
        # Convert to SearchResult objects
        return [
            SearchResult(
                doc_id=str(result.id),
                chunk=result.payload.get('chunk', ''),
                score=float(result.score),
                metadata=result.payload
            )
            for result in search_result
        ]
    
    def clear(self) -> None:
        """Clear all data from the collection."""
        # Delete the collection and recreate it
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        
        # Get vector size based on model
        if "minilm" in self.embedding_model.lower():
            vector_size = 384
        elif "mpnet" in self.embedding_model.lower():
            vector_size = 768
        else:  # default to nomic-embed-text-v1.5
            vector_size = 768
            
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
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
        # Convert query vector to list if it's a numpy array
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()
        
        # Add filter if provided
        search_params = {}
        if filter_conditions:
            search_params["filter"] = models.Filter(**filter_conditions)
        
        # Perform search
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,  # Pass the vector directly
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
 