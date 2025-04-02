from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import logging
import traceback
from database.base_db import BaseDB
from embeddings.base_embedder import BaseEmbedder

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

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
        logger.info("Initialized VectorQueryHandler")
    
    def process_query(self, query_text: str, k: int = 5) -> List[Dict[str, Any]]:
        """Process a query and return relevant chunks.
        
        Args:
            query_text: User query string
            k: Number of results to return
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            logger.debug(f"Processing query: {query_text[:100]}...")
            
            # Generate query embedding
            logger.debug("Generating query embedding...")
            query_embedding, _ = self.embedder.embed_text(query_text, is_query=True)
            logger.debug(f"Generated embedding of length: {len(query_embedding)}")
            
            # Search for similar chunks
            logger.debug(f"Searching for {k} similar chunks...")
            results = self.vector_db.search(query_embedding, k=k)
            logger.debug(f"Found {len(results)} results")
            
            # Convert results to expected format
            contexts = []
            for result in results:
                contexts.append({
                    'text': result.chunk,
                    'metadata': result.metadata,
                    'score': result.score
                })
            
            logger.debug(f"Returning {len(contexts)} contexts")
            return contexts
            
        except Exception as e:
            logger.error(f"Error in process_query: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

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
        logger.info("Initialized QueryPipeline")
    
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
        try:
            logger.debug(f"Processing query: {query_text[:100]}...")
            
            # Get relevant contexts
            logger.debug("Getting relevant contexts...")
            contexts = self.query_handler.process_query(query_text)
            logger.debug(f"Found {len(contexts)} contexts")
            
            # Generate response using LLM
            logger.debug("Generating prompt...")
            prompt = self.prompt_generator.generate_prompt(
                query=query_text,
                contexts=contexts,
                use_general_knowledge=use_general_knowledge
            )
            logger.debug(f"Generated prompt with length: {prompt.get('context_length', 0)}")
            
            logger.debug("Generating LLM response...")
            response = self.llm.generate_response(
                prompt=prompt['prompt'],
                context=contexts,
                use_general_knowledge=use_general_knowledge
            )
            logger.debug("Generated LLM response")
            
            return {
                'response': response,
                'contexts': contexts,
                'prompt': prompt
            }
            
        except Exception as e:
            logger.error(f"Error in process: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """Format results into a readable string.
        
        Args:
            results: List of results from query processing
            
        Returns:
            Formatted string of results
        """
        try:
            formatted_results = []
            for i, result in enumerate(results, 1):
                formatted_results.append(
                    f"Result {i}:\n"
                    f"Text: {result['text']}\n"
                    f"Source: {result.get('document_id', 'Unknown')}\n"
                    f"Score: {result.get('score', 'N/A')}\n"
                )
            return "\n".join(formatted_results)
        except Exception as e:
            logger.error(f"Error in format_results: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise 