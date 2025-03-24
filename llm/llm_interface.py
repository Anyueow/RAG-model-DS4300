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
    
    def __init__(self, model_name: str, temperature: float = 0.7):
        """Initialize the LLM interface.
        
        Args:
            model_name: Name of the Ollama model to use
            temperature: Temperature for response generation (default: 0.7)
        """
        self.model_name = model_name
        self.temperature = temperature
    
    def generate_response(self, 
                         prompt: str, 
                         context: Optional[List[Dict[str, Any]]] = None,
                         images: Optional[List[Dict[str, Any]]] = None,
                         use_general_knowledge: bool = True) -> str:
        """Generate a response using the Ollama model.
        
        Args:
            prompt: The prompt to send to the LLM
            context: Optional list of relevant context chunks
            images: Optional list of image data
            use_general_knowledge: Whether to allow using general knowledge when context is insufficient
            
        Returns:
            Generated response text
        """
        try:
            # Prepare messages
            messages = []
            
            # Add system message
            system_message = """You are a helpful assistant that can provide both specific information from the provided context and general knowledge when needed. 
            When using information from the context, cite the source. When using general knowledge, be transparent about it.
            Always respond in English only."""
            
            if images:
                system_message += " You can also understand and analyze images. When referring to images, use their reference numbers [Image #]."
            
            messages.append({
                'role': 'system',
                'content': system_message
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
            
            # Construct the prompt based on context and general knowledge setting
            if context and not use_general_knowledge:
                # Use only context-based prompt
                full_prompt = self._construct_prompt(prompt, context)
            elif context and use_general_knowledge:
                # Use context but allow general knowledge
                full_prompt = f"""Based on the following context and your general knowledge, please answer the question.
                If the context is insufficient, feel free to use your general knowledge, but please indicate when you're doing so.

                Question: {prompt}

                Context:
                {self._format_context(context)}

                Please provide a comprehensive answer, using both the context and your general knowledge as needed."""
            else:
                # Use only general knowledge
                full_prompt = f"""Please answer the following question using your general knowledge:

                Question: {prompt}

                Please provide a comprehensive answer based on your knowledge."""
            
            # Add main prompt
            messages.append({
                'role': 'user',
                'content': full_prompt
            })
            
            # Generate response with temperature
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': self.temperature
                }
            )
            return response['message']['content']
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try again."
    
    def _format_context(self, context: List[Dict[str, Any]]) -> str:
        """Format context chunks into a readable string."""
        formatted_context = []
        for idx, chunk in enumerate(context, 1):
            chunk_text = f"Context {idx}:\n{chunk.get('text', '')}"
            if 'metadata' in chunk and 'source' in chunk['metadata']:
                chunk_text += f"\nSource: {chunk['metadata']['source']}"
            formatted_context.append(chunk_text)
        return "\n\n".join(formatted_context)
    
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
            # Get text from metadata if available, otherwise use empty string
            text = chunk.get('metadata', {}).get('text', '')
            context_text += f"\n{i}. {text}"
        
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