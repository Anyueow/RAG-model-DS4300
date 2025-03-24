from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
import ollama
import base64

class BaseLLM(ABC):
    """Abstract base class for LLM interfaces."""
    
    @abstractmethod
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, Any]]] = None,
                         images: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant context chunks
            images: Optional list of image data
            
        Returns:
            Generated response text
        """
        pass

class OllamaLLM(BaseLLM):
    """Interface for Ollama-based local LLMs."""
    
    def __init__(self, model_name:str):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, Any]]] = None,
                         images: Optional[List[Dict[str, Any]]] = None) -> str:
        """Generate a response using the Ollama model.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant context chunks
            images: Optional list of image data
            
        Returns:
            Generated response text
        """
        # Construct the full prompt with context if provided
        full_prompt = self._construct_prompt(prompt, context)
        
        try:
            # Prepare messages
            messages = []
            
            # Add system message if needed
            if images:
                messages.append({
                    'role': 'system',
                    'content': 'You are a helpful assistant that can understand both text and images. When referring to images, use their reference numbers [Image #].'
                })
            
            # Add images if provided
            if images:
                for img_data in images:
                    if 'data' in img_data:
                        # If image is already base64 encoded
                        if isinstance(img_data['data'], str):
                            img_base64 = img_data['data']
                        # If image is bytes
                        else:
                            img_base64 = base64.b64encode(img_data['data']).decode()
                            
                        messages.append({
                            'role': 'user',
                            'content': f"[Image {img_data.get('index', 0)}]",
                            'images': [img_base64]
                        })
            
            # Add main prompt
            messages.append({
                'role': 'user',
                'content': full_prompt
            })
            
            # Generate response
            response = ollama.chat(
                model=self.model_name,
                messages=messages
            )
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