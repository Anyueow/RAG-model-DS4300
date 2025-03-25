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
        self.max_query_length = 1500  # Maximum length for user queries
        self.max_context_chunks = 3   # Maximum number of context chunks to include
    
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
            # Validate input length
            if len(prompt) > self.max_query_length:
                return f"Query is too long. Please keep queries under {self.max_query_length} characters."
            
            # Limit context chunks if provided
            if context:
                context = context[:self.max_context_chunks]
            
            messages = []
            
            # Enhanced system message with better technical capabilities
            system_message = """you're a whiz at balancing AVL trees. 
            You always answer the question provided and then provide an explanation. 
            

            - Start with the answer to the question
            - Provide step-by-step explanations
            - Use proper technical terminology"""
            
            if images:
                system_message += """
                For image analysis:
                - Identify B tree and B+ tree
                - Identify AVL trees 
                - Be able to traverse a tree 
                - Be able to a balance a AVL tree
                - Be able to insert and delete a node in a B tree and B+ tree
                - Be able to insert and delete a node in a AVL tree
                - Analyze code snippets and algorithms
                - Describe visual patterns and structures
                - Reference specific parts of the image using [Image #]"""
            
            messages.append({
                'role': 'system',
                'content': system_message
            })
            
            # Add images if provided
            if images:
                for img_data in images:
                    if 'data' in img_data:
                        if isinstance(img_data['data'], str):
                            img_base64 = img_data['data']
                        else:
                            img_base64 = base64.b64encode(img_data['data']).decode()
                            
                        messages.append({
                            'role': 'user',
                            'content': f"[Image {img_data.get('index', 0)}]",
                            'images': [img_base64]
                        })
            
            # Enhanced prompt construction for technical topics
            if context and not use_general_knowledge:
                full_prompt = self._construct_prompt(prompt, context)
            elif context and use_general_knowledge:
                full_prompt = f"""Based on the following context and your technical expertise, please provide a comprehensive answer.
                If the context is insufficient, use your general knowledge to provide a detailed technical explanation.

                Question: {prompt}

                Context:
                {self._format_context(context)}

                Please provide:
                1. A clear technical explanation
                2. Key concepts and principles
                3. Relevant examples or comparisons
                4. Any additional technical details that would be helpful"""
            else:
                full_prompt = f"""Please provide a comprehensive technical explanation for the following question:

                Question: {prompt}

                Please include:
                1. the answer to the question
                1. Clear definitions and key principles
                2. Step-by-step explanation
                3. Relevant examples
                4. Comparisons with related concepts
                5. Technical terminology and best practices"""
            
            messages.append({
                'role': 'user',
                'content': full_prompt
            })
            
            # Generate response with adjusted temperature for technical accuracy
            response = ollama.chat(
                model=self.model_name,
                messages=messages,
                options={
                    'temperature': 0.3  # Lower temperature for more focused technical responses
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