from typing import List, Union, Dict, Any
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sentence_transformers import SentenceTransformer
from .base_embedder import BaseEmbedder
import numpy as np
import io
import base64
import logging
from typing import Optional

class MultiModalEmbedder(BaseEmbedder):
    """Multimodal embedder that handles both text and image embeddings using CLIP and SentenceTransformer."""
    
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        max_retries: int = 3
    ):
        """Initialize the multimodal embedder.
        
        Args:
            clip_model_name: Name of the CLIP model to use
            text_model_name: Name of the text embedding model to use
            device: Device to run the models on (auto-detected if None)
            max_retries: Maximum number of retries for model loading
        """
        self.device = self._get_device(device)
        self.max_retries = max_retries
        
        # Initialize models with retry logic
        self.clip_model, self.clip_processor = self._load_clip_model(clip_model_name)
        self.text_model = self._load_text_model(text_model_name)
        
        # Store model names
        self._clip_model_name = clip_model_name
        self._text_model_name = text_model_name
        
        logging.info(f"Initialized MultiModalEmbedder on device: {self.device}")
        
    def _get_device(self, device: Optional[str]) -> str:
        """Get the appropriate device for model execution.
        
        Args:
            device: User-specified device or None
            
        Returns:
            str: Device to use
        """
        if device is not None:
            return device
            
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
            
    def _load_clip_model(self, model_name: str) -> tuple[CLIPModel, CLIPProcessor]:
        """Load CLIP model with retry logic.
        
        Args:
            model_name: Name of the CLIP model to load
            
        Returns:
            tuple: (CLIPModel, CLIPProcessor)
        """
        for attempt in range(self.max_retries):
            try:
                # First load the processor
                processor = CLIPProcessor.from_pretrained(model_name)
                
                # Then load the model with basic settings first
                model = CLIPModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32  # Force float32 to avoid precision issues
                )
                
                # Move model to device after loading
                model = model.to(self.device)
                
                # Force model to eval mode
                model.eval()
                
                return model, processor
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to load CLIP model after {self.max_retries} attempts: {str(e)}")
                logging.warning(f"Attempt {attempt + 1} failed to load CLIP model: {str(e)}")
                # Clear CUDA cache if available
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
                
    def _load_text_model(self, model_name: str) -> SentenceTransformer:
        """Load SentenceTransformer model with retry logic.
        
        Args:
            model_name: Name of the text model to load
            
        Returns:
            SentenceTransformer: Loaded model
        """
        for attempt in range(self.max_retries):
            try:
                model = SentenceTransformer(model_name)
                model = model.to(self.device)
                return model
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Failed to load text model after {self.max_retries} attempts: {str(e)}")
                logging.warning(f"Attempt {attempt + 1} failed to load text model: {str(e)}")
                continue

    def embed_text(self, text: str) -> np.ndarray:
        """Get text embeddings using both CLIP and SentenceTransformer.
        
        Args:
            text: Text to embed
            
        Returns:
            Combined text embedding
        """
        try:
            # Get CLIP text embeddings
            with torch.no_grad():  # Disable gradient computation
                clip_inputs = self.clip_processor(
                    text=text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.device)
                
                clip_embedding = self.clip_model.get_text_features(**clip_inputs)
                clip_embedding = clip_embedding.detach().cpu().numpy()
            
            # Get SentenceTransformer embeddings
            st_embedding = self.text_model.encode(text, convert_to_numpy=True)
            
            # Combine embeddings (concatenate and normalize)
            combined = np.concatenate([clip_embedding[0], st_embedding])
            return combined / np.linalg.norm(combined)
        except Exception as e:
            logging.error(f"Error in embed_text: {str(e)}")
            # Fallback to just SentenceTransformer if CLIP fails
            return self.text_model.encode(text, convert_to_numpy=True)

    def embed_image(self, image_data: Union[str, bytes, Image.Image]) -> np.ndarray:
        """Get image embeddings using CLIP.
        
        Args:
            image_data: Image to embed (can be base64 string, bytes, or PIL Image)
            
        Returns:
            Image embedding
        """
        try:
            # Convert input to PIL Image
            if isinstance(image_data, str):
                # Assume base64 string
                image_bytes = base64.b64decode(image_data)
                image = Image.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                image = Image.open(io.BytesIO(image_data))
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError("Unsupported image format")
                
            # Get CLIP image embeddings
            with torch.no_grad():  # Disable gradient computation
                inputs = self.clip_processor(
                    images=image,
                    return_tensors="pt"
                ).to(self.device)
                
                embedding = self.clip_model.get_image_features(**inputs)
                embedding = embedding.detach().cpu().numpy()
            
            # Normalize embedding
            return embedding[0] / np.linalg.norm(embedding[0])
        except Exception as e:
            logging.error(f"Error in embed_image: {str(e)}")
            raise  # Re-raise the exception as we don't have a fallback for image embedding

    def embed_batch(self, texts: List[str] = None, images: List[Union[str, bytes, Image.Image]] = None) -> Dict[str, np.ndarray]:
        """Batch embed texts and/or images.
        
        Args:
            texts: List of texts to embed
            images: List of images to embed
            
        Returns:
            Dictionary with 'text_embeddings' and 'image_embeddings' keys
        """
        results = {}
        
        if texts:
            text_embeddings = []
            for text in texts:
                text_embeddings.append(self.embed_text(text))
            results['text_embeddings'] = np.array(text_embeddings)
            
        if images:
            image_embeddings = []
            for image in images:
                image_embeddings.append(self.embed_image(image))
            results['image_embeddings'] = np.array(image_embeddings)
            
        return results

    def get_embedding_dim(self) -> int:
        """Get the dimension of the embeddings."""
        # CLIP dimension + SentenceTransformer dimension
        return self.clip_model.config.projection_dim + self.text_model.get_sentence_embedding_dimension()

    def get_model_name(self) -> str:
        """Get the names of the underlying models."""
        return f"CLIP: {self._clip_model_name}, Text: {self._text_model_name}"

    def embed_texts(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Convert text(s) to embeddings.
        
        Args:
            texts: Single text string or list of text strings to embed
            
        Returns:
            numpy.ndarray: Array of embeddings
        """
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            
        # Get embeddings for each text
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_text(text))
            
        return np.array(embeddings) 