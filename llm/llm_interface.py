"""Interface for LLM interactions."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import ollama
import base64
import logging
import traceback
from database.base_db import SearchResult

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[SearchResult]] = None,
                         images: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant search results
            images: Optional list of image data
            
        Returns:
            Generated response text
        """
        pass

class OllamaLLM(BaseLLM):
    """Interface for Ollama-based local LLMs."""
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for response generation (default: 0.7)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_context_chunks = 3   # Maximum number of context chunks to include
        logger.info(f"Initialized OllamaLLM with model: {model_name}, temperature: {temperature}")
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[SearchResult]] = None,
                         images: Optional[List[Dict[str, Any]]] = None,
                         use_general_knowledge: bool = True) -> str:
        """Generate a response using the Ollama model.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant search results
            images: Optional list of image data
            use_general_knowledge: Whether to allow using general knowledge when context is insufficient
            
        Returns:
            Generated response text
        """
        try:
            logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            logger.debug(f"Context provided: {bool(context)}")
            logger.debug(f"Images provided: {bool(images)}")
            
            # Limit context chunks if provided
            if context:
                context = context[:self.max_context_chunks]
                logger.debug(f"Using {len(context)} context chunks")
            
            messages = []
            
            # Enhanced system message for Mistral
            system_message = """You are a helpful tutor specializing in relational databases, algorithms, and machine learning.
            You provide clear, technical explanations with examples and best practices.
            When answering questions:
            1. Be precise and technical
            2. Use proper terminology
            3. Provide step-by-step explanations
            4. Include relevant examples
            5. Compare with related concepts
            6. Cite sources when available"""
            
            messages.append({
                'role': 'system',
                'content': system_message
            })
            
            # Construct the full prompt with context
            if context:
                context_text = "\n\nRelevant context:\n"
                for i, result in enumerate(context, 1):
                    if isinstance(result, dict):
                        chunk_text = result.get('text', '')
                        source = result.get('metadata', {}).get('source', '')
                    else:
                        chunk_text = result.chunk
                        source = result.metadata.get('source', '') if result.metadata else ''
                    
                    context_text += f"\n{i}. {chunk_text}"
                    if source:
                        context_text += f" (Source: {source})"
                
                full_prompt = f"{prompt}{context_text}"
            else:
                full_prompt = prompt
            
            messages.append({
                'role': 'user',
                'content': full_prompt
            })
            
            logger.debug("Sending request to Ollama...")
            # Generate response with adjusted temperature for technical accuracy
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': 0.3,  # Lower temperature for more focused technical responses
                    'num_predict': 2048,  # Increase max tokens for longer responses
                    'top_k': 40,  # Adjust sampling parameters
                    'top_p': 0.9,
                    'repeat_penalty': 1.1
                }
            )
            logger.debug("Received response from Ollama")
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return f"I apologize, but I encountered an error while generating the response. Error details: {str(e)}"
    
    def _format_context(self, context: List[SearchResult]) -> str:
        """Format search results into a readable string."""
        formatted_context = []
        for idx, result in enumerate(context, 1):
            # Handle both SearchResult objects and dictionaries
            if isinstance(result, dict):
                chunk_text = f"Context {idx}:\n{result.get('text', '')}"
                if 'metadata' in result and 'source' in result['metadata']:
                    chunk_text += f"\nSource: {result['metadata']['source']}"
            else:
                chunk_text = f"Context {idx}:\n{result.chunk}"
                if result.metadata and 'source' in result.metadata:
                    chunk_text += f"\nSource: {result.metadata['source']}"
            formatted_context.append(chunk_text)
        return "\n\n".join(formatted_context)
    
    def _construct_prompt(self, 
                         prompt: str, 
                         context: Optional[List[SearchResult]] = None) -> str:
        """Construct the full prompt with context.
        
        Args:
            prompt: Original prompt
            context: Optional list of relevant search results
            
        Returns:
            Constructed prompt with context
        """
        if not context:
            return prompt
        
        # Add context to the prompt
        context_text = "\n\nRelevant context:\n"
        for i, result in enumerate(context, 1):
            # Handle both SearchResult objects and dictionaries
            if isinstance(result, dict):
                chunk_text = result.get('text', '')
                source = result.get('metadata', {}).get('source', '')
            else:
                chunk_text = result.chunk
                source = result.metadata.get('source', '') if result.metadata else ''
                
            context_text += f"\n{i}. {chunk_text}"
            if source:
                context_text += f" (Source: {source})"
        
        return f"{prompt}{context_text}"

class LLMPipeline:
    """Pipeline for generating responses using LLMs."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize the pipeline.
        
        Args:
            llm: LLM instance to use
        """
        self.llm = llm
        logger.info("Initialized LLMPipeline")
    
    def generate_response(self, 
                         query: str, 
                         context: Optional[List[SearchResult]] = None) -> str:
        """Generate a response using the LLM pipeline.
        
        Args:
            query: User query
            context: Optional list of relevant search results
            
        Returns:
            Generated response
        """
        logger.debug(f"Generating response for query: {query[:100]}...")
        return self.llm.generate_response(query, context) 