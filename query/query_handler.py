from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from database.base_db import BaseDB
from embeddings.base_embedder import BaseEmbedder

class BaseQueryHandler(ABC):
    """Abstract base class for query handlers."""
    
    @abstractmethod
    def process_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query and return relevant chunks.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        pass

class VectorQueryHandler(BaseQueryHandler):
    """Query handler that uses vector similarity search."""
    
    def __init__(self, 
                 vector_db: BaseDB,
                 embedder: BaseEmbedder):
        """Initialize the query handler.
        
        Args:
            vector_db: Vector database instance
            embedder: Embedding model instance
        """
        self.vector_db = vector_db
        self.embedder = embedder
    
    def process_query(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query using vector similarity search.
        
        Args:
            query: User query string
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        # Convert query to embedding
        query_embedding = self.embedder.embed_texts(query)
        
        # Search for similar vectors
        results = self.vector_db.search(query_embedding, k=k)
        
        return results

class QueryPipeline:
    """Pipeline for processing queries and generating responses."""
    
    def __init__(self, 
                 embedder: BaseEmbedder,
                 vector_db: BaseDB,
                 llm: Any,
                 prompt_generator: Any):
        """Initialize the pipeline.
        
        Args:
            embedder: Embedding model instance
            vector_db: Vector database instance
            llm: Language model instance
            prompt_generator: Prompt generator instance
        """
        self.embedder = embedder
        self.vector_db = vector_db
        self.llm = llm
        self.prompt_generator = prompt_generator
        self.query_handler = VectorQueryHandler(vector_db, embedder)
    
    def process(self, 
                query_text: str, 
                query_image: Optional[bytes] = None,
                use_general_knowledge: bool = True) -> Dict[str, Any]:
        """Process a query through the pipeline.
        
        Args:
            query_text: User query string
            query_image: Optional image query
            use_general_knowledge: Whether to use general knowledge
            
        Returns:
            Dictionary containing response and contexts
        """
        # Get relevant contexts
        contexts = self.query_handler.process_query(query_text)
        
        # Generate response using LLM
        prompt = self.prompt_generator.generate_prompt(
            query=query_text,
            contexts=contexts,
            use_general_knowledge=use_general_knowledge
        )
        
        response = self.llm.generate(prompt)
        
        return {
            'response': response,
            'contexts': contexts
        }
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results into a readable string.
        
        Args:
            results: List of results from query processing
            
        Returns:
            Formatted string of results
        """
        formatted_results = []
        for i, result in enumerate(results, 1):
            formatted_results.append(
                f"Result {i}:\n"
                f"Text: {result['text']}\n"
                f"Source: {result.get('document_id', 'Unknown')}\n"
                f"Score: {result.get('score', 'N/A')}\n"
            )
        return "\n".join(formatted_results) 