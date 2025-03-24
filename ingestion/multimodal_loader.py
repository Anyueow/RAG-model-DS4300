from typing import List, Dict, Any, Tuple
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import numpy as np
from .data_loader import BaseDocumentLoader
import os

class MultiModalDocument:
    """Class to represent a multimodal document containing both text and images."""
    
    def __init__(
        self,
        text: str = "",
        images: List[Dict[str, Any]] = None,
        metadata: Dict[str, Any] = None
    ):
        """Initialize a multimodal document.
        
        Args:
            text: Text content
            images: List of image dictionaries with keys:
                   - 'data': Image data (bytes or base64)
                   - 'bbox': Bounding box coordinates
                   - 'page': Page number
            metadata: Additional metadata
        """
        self.text = text
        self.images = images or []
        self.metadata = metadata or {}

class MultiModalPDFLoader(BaseDocumentLoader):
    """Loader for PDF documents that extracts both text and images."""
    
    def __init__(
        self,
        min_image_size: Tuple[int, int] = (100, 100),
        image_format: str = "PNG"
    ):
        """Initialize the PDF loader.
        
        Args:
            min_image_size: Minimum size (width, height) for extracted images
            image_format: Format to save extracted images
        """
        self.min_image_size = min_image_size
        self.image_format = image_format

    def load_document(self, file_path: str) -> List[MultiModalDocument]:
        """Load a PDF document and extract text and images.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of MultiModalDocument objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        documents = []
        pdf_document = fitz.open(file_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text
            text = page.get_text()
            
            # Extract images
            images = []
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image for size check
                    pil_image = Image.open(io.BytesIO(image_bytes))
                    width, height = pil_image.size
                    
                    # Skip if image is too small
                    if width < self.min_image_size[0] or height < self.min_image_size[1]:
                        continue
                    
                    # Convert image to desired format
                    output = io.BytesIO()
                    pil_image.save(output, format=self.image_format)
                    image_data = base64.b64encode(output.getvalue()).decode()
                    
                    # Get image location on page
                    bbox = page.get_image_bbox(img)
                    
                    images.append({
                        'data': image_data,
                        'bbox': bbox,
                        'page': page_num,
                        'format': self.image_format.lower(),
                        'size': (width, height)
                    })
                    
                except Exception as e:
                    print(f"Error processing image {img_idx} on page {page_num}: {str(e)}")
                    continue
            
            # Create document for this page
            doc = MultiModalDocument(
                text=text,
                images=images,
                metadata={
                    'source': file_path,
                    'page': page_num,
                    'total_pages': len(pdf_document)
                }
            )
            documents.append(doc)
        
        pdf_document.close()
        return documents

    def load_directory(self, directory_path: str) -> List[MultiModalDocument]:
        """Load all PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of MultiModalDocument objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        documents = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                documents.extend(self.load_document(file_path))
                
        return documents 