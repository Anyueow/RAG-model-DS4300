from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import ollama

class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant context chunks
            
        Returns:
            Generated response text
        """
        pass

class OllamaLLM(BaseLLM):
    """Interface for Ollama-based local LLMs."""
    
    def __init__(self, model_name: str = "llama2"):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the Ollama model.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant context chunks
            
        Returns:
            Generated response text
        """
        # Construct the full prompt with context if provided
        full_prompt = self._construct_prompt(prompt, context)
        
        try:
            response = ollama.chat(model=self.model_name, messages=[
                {
                    'role': 'user',
                    'content': full_prompt
                }
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "Error: Failed to generate response"
    
    def _construct_prompt(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Construct the full prompt with context.
        
        Args:
            prompt: Original prompt
            context: Optional list of relevant context chunks
            
        Returns:
            Constructed prompt with context
        """
        if not context:
            return prompt
        
        # Add context to the prompt
        context_text = "\n\nRelevant context:\n"
        for i, chunk in enumerate(context, 1):
            context_text += f"\n{i}. {chunk['text']}"
        
        return f"{prompt}{context_text}"

class LLMPipeline:
    """Pipeline for generating responses using LLMs."""
    
    def __init__(self, llm: BaseLLM):
        """Initialize the pipeline.
        
        Args:
            llm: LLM instance to use
        """
        self.llm = llm
    
    def generate_response(self, 
                         query: str, 
                         context: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the LLM pipeline.
        
        Args:
            query: User query
            context: Optional list of relevant context chunks
            
        Returns:
            Generated response
        """
        return self.llm.generate_response(query, context) 