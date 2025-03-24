from typing import List, Dict, Any, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from .base_db import BaseVectorDB
import uuid
import os
import shutil

class QdrantDB(BaseVectorDB):
    """Qdrant adapter implementation with multimodal support."""
    
    def __init__(self, 
                 collection_name: str = "course_notes",
                 path: str = "./qdrant_db"):
        """Initialize the Qdrant adapter.
        
        Args:
            collection_name: Base name for collections
            path: Path to store the database
        """
        # Clean up any existing lock files
        lock_file = os.path.join(path, "lock")
        if os.path.exists(lock_file):
            try:
                os.remove(lock_file)
            except:
                pass
        
        # If path exists and is locked, try to use a temporary directory
        if os.path.exists(path):
            try:
                self.client = QdrantClient(path=path)
            except RuntimeError:
                # If locked, use a temporary directory
                temp_path = os.path.join(os.path.dirname(path), "qdrant_db_temp")
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
                self.client = QdrantClient(path=temp_path)
        else:
            self.client = QdrantClient(path=path)
        
        # Create separate collections for text and images
        self.text_collection = f"{collection_name}_text"
        self.image_collection = f"{collection_name}_images"
        
        # Create collections if they don't exist
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.text_collection not in collection_names:
            self.client.create_collection(
                collection_name=self.text_collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
            )
            
        if self.image_collection not in collection_names:
            self.client.create_collection(
                collection_name=self.image_collection,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE)
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
            collection_name=self.text_collection if modality == "text" else self.image_collection,
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
            collection_name=self.text_collection if modality == "text" else self.image_collection,
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
            collection_name=self.text_collection if ids[0].startswith("text_") else self.image_collection,
            points_selector=models.PointIdsList(
                points=ids
            )
        )

    def get_vector_count(self) -> int:
        """Get the total number of vectors in the database."""
        collection_info = self.client.get_collection(self.text_collection)
        return collection_info.vectors_count

    def clear(self) -> None:
        """Clear all vectors from the database."""
        self.client.delete(
            collection_name=self.text_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_idx",
                            match=models.MatchValue(
                                value=-1,
                                is_negative=True
                            )
                        )
                    ]
                )
            )
        )
        self.client.delete(
            collection_name=self.image_collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_idx",
                            match=models.MatchValue(
                                value=-1,
                                is_negative=True
                            )
                        )
                    ]
                )
            )
        ) 