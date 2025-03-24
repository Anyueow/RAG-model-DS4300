from typing import List, Dict, Any, Optional
import base64
from PIL import Image
import io

class PromptGenerator:
    """Generator for creating augmented prompts that combine text and image contexts."""
    
    def __init__(
        self,
        system_prompt: str = None,
        max_context_length: int = 4000,
        image_format: str = "PNG"
    ):
        """Initialize the prompt generator.
        
        Args:
            system_prompt: Base system prompt to use
            max_context_length: Maximum context length in tokens
            image_format: Format for image conversion
        """
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        self.max_context_length = max_context_length
        self.image_format = image_format

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt."""
        return """You are a helpful AI assistant that can understand both text and images.
When providing information, clearly distinguish between two types of sources:
1. [KB] Knowledge Base - Information from the course materials and documents
2. [GK] General Knowledge - Information from your training data

For text from the knowledge base, cite using [KB-Text 'Name of the text'].
For images from the knowledge base, cite using [KB-Image 'Name of the image'].
When using general knowledge, mark it with [GK] and provide a brief explanation of the source.

Always provide citations for your information. If you're not sure about something, say so rather than making assumptions."""

    def _format_text_context(self, contexts: List[Dict[str, Any]]) -> str:
        """Format text contexts with citations.
        
        Args:
            contexts: List of text contexts with metadata
            
        Returns:
            Formatted text context string
        """
        formatted_contexts = []
        for idx, ctx in enumerate(contexts, 1):
            text = ctx.get('text', '').strip()
            source = ctx.get('metadata', {}).get('source', 'Unknown')
            page = ctx.get('metadata', {}).get('page', None)
            
            
            citation = f"[KB-Text {idx}] (Source: {source}"
            if page is not None:
                citation += f", Page: {page}"
            citation += ")"
            
            formatted_contexts.append(f"{citation}\n{text}\n")
            
        return "\n".join(formatted_contexts)

    def _format_image_context(self, contexts: List[Dict[str, Any]]) -> tuple[str, List[Dict]]:
        """Format image contexts and prepare images.
        
        Args:
            contexts: List of image contexts with metadata
            
        Returns:
            Tuple of (formatted context string, list of image data)
        """
        formatted_contexts = []
        image_data = []
        
        for idx, ctx in enumerate(contexts, 1):
            source = ctx.get('metadata', {}).get('source', 'Unknown')
            page = ctx.get('metadata', {}).get('page', None)
            caption = ctx.get('text', '')
            
            citation = f"[KB-Image {idx}] (Source: {source}"
            if page is not None:
                citation += f", Page: {page}"
            citation += ")"
            
            if caption:
                formatted_contexts.append(f"{citation}\nCaption: {caption}\n")
            else:
                formatted_contexts.append(f"{citation}\n")
                
            # Prepare image data
            image_data.append({
                'index': idx,
                'data': ctx.get('image', None),
                'metadata': ctx.get('metadata', {})
            })
            
        return "\n".join(formatted_contexts), image_data

    def generate_prompt(
        self,
        query: str,
        text_contexts: Optional[List[Dict[str, Any]]] = None,
        image_contexts: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Generate an augmented prompt combining query, text, and image contexts.
        
        Args:
            query: User query
            text_contexts: List of relevant text contexts
            image_contexts: List of relevant image contexts
            
        Returns:
            Dictionary containing:
            - prompt: Combined prompt string
            - images: List of image data
            - context_length: Estimated context length
        """
        prompt_parts = [self.system_prompt, "\n\n"]
        images = []
        
        # Add text contexts if available
        if text_contexts:
            prompt_parts.extend([
                "Knowledge Base Text Contexts:",
                self._format_text_context(text_contexts),
                "\n"
            ])
            
        # Add image contexts if available
        if image_contexts:
            image_context, image_data = self._format_image_context(image_contexts)
            prompt_parts.extend([
                "Knowledge Base Images:",
                image_context,
                "\n"
            ])
            images = image_data
            
        # Add query
        prompt_parts.extend([
            "User Query:",
            query,
            "\n",
            "Assistant: "
        ])
        
        # Combine prompt parts
        prompt = "\n".join(prompt_parts)
        
        # Estimate context length (rough approximation)
        context_length = len(prompt.split())
        
        return {
            'prompt': prompt,
            'images': images,
            'context_length': context_length
        }

    def format_response(
        self,
        response: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Format the LLM response with proper citations and source information.
        
        Args:
            response: Raw LLM response
            sources: List of source documents used
            
        Returns:
            Dictionary containing:
            - response: Formatted response text
            - sources: List of cited sources
            - images: List of referenced images
        """
        # Extract citations from response
        kb_text_citations = set()
        kb_image_citations = set()
        gk_citations = set()
        
        # First, process knowledge base citations
        for source in sources:
            citation = source.get('citation', '')
            if citation.startswith('[KB-Text'):
                kb_text_citations.add(citation)
            elif citation.startswith('[KB-Image'):
                kb_image_citations.add(citation)
                
        # Format source information
        formatted_sources = []
        
        # Add knowledge base text sources
        if kb_text_citations:
            formatted_sources.append({'header': 'Knowledge Base Text Sources:'})
            for citation in sorted(kb_text_citations):
                source = next((s for s in sources if s.get('citation') == citation), None)
                if source:
                    formatted_sources.append({
                        'citation': citation,
                        'source': source.get('metadata', {}).get('source', 'Unknown'),
                        'page': source.get('metadata', {}).get('page'),
                        'type': 'kb_text'
                    })
        
        # Add knowledge base image sources
        if kb_image_citations:
            formatted_sources.append({'header': 'Knowledge Base Image Sources:'})
            for citation in sorted(kb_image_citations):
                source = next((s for s in sources if s.get('citation') == citation), None)
                if source:
                    formatted_sources.append({
                        'citation': citation,
                        'source': source.get('metadata', {}).get('source', 'Unknown'),
                        'page': source.get('metadata', {}).get('page'),
                        'type': 'kb_image'
                    })
        
        # Extract and add general knowledge citations
        import re
        gk_matches = re.findall(r'\[GK:([^\]]+)\]', response)
        if gk_matches:
            formatted_sources.append({'header': 'General Knowledge Sources:'})
            for idx, match in enumerate(gk_matches, 1):
                formatted_sources.append({
                    'citation': f'[GK-{idx}]',
                    'source': match.strip(),
                    'type': 'general_knowledge'
                })
        
        return {
            'response': response,
            'sources': formatted_sources,
            'images': [s for s in formatted_sources if s.get('type') == 'kb_image']
        } 