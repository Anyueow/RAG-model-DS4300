from typing import List, Dict, Any
import os
from pathlib import Path
import fitz
import re
from abc import ABC, abstractmethod

class Document:
    """Class to represent a document containing text."""
    
    def __init__(
        self,
        text: str = "",
        metadata: Dict[str, Any] = None
    ):
        """Initialize a document.
        
        Args:
            text: Text content
            metadata: Additional metadata
        """
        self.text = text
        self.metadata = metadata or {}


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load_document(self, file_path: str) -> str:
        """Load and extract text from a document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            str: Extracted text from the document
        """
        pass


class PDFLoader(BaseDocumentLoader):
    """Loader for PDF documents that focuses on robust text extraction."""
    
    def __init__(self):
        """Initialize the PDF loader."""
        self.cache = {}  # Cache for loaded documents
    
    def _validate_text(self, text: str) -> bool:
        """Validate extracted text for quality.
        
        Args:
            text: Extracted text to validate
            
        Returns:
            bool: True if text is valid, False otherwise
        """
        # Check if text is empty or only whitespace
        if not text or text.isspace():
            return False
            
        # Check for gibberish (high ratio of special characters)
        special_chars = re.findall(r'[^\w\s]', text)
        if len(special_chars) / (len(text) + 1) > 0.4:  # Allow up to 40% special chars
            return False
            
        # Check for minimum text length (to avoid fragments)
        if len(text.strip()) < 5:
            return False
            
        # Check for encoding issues (common in PDF extraction)
        if '\uFFFD' in text or '�' in text:
            return False
            
        return True
    
    def _extract_text_with_fallback(self, page) -> str:
        """Extract text with fallback methods if primary extraction fails.
        
        Args:
            page: PDF page object
            
        Returns:
            str: Extracted text
        """
        # Primary extraction method
        text = page.get_text("text", sort=True)
        
        # If primary extraction fails validation, try alternative methods
        if not self._validate_text(text):
            # Try with different parameters
            text = page.get_text("blocks", sort=True)
            if isinstance(text, list):
                text = " ".join([block[4] for block in text if len(block) > 4])
                
            # If still not valid, try raw extraction
            if not self._validate_text(text):
                text = page.get_text("rawdict")
                if isinstance(text, dict) and 'blocks' in text:
                    blocks = []
                    for block in text['blocks']:
                        if 'lines' in block:
                            for line in block['lines']:
                                if 'spans' in line:
                                    for span in line['spans']:
                                        if 'text' in span:
                                            blocks.append(span['text'])
                    text = " ".join(blocks)
        
        return text
    
    def load_document(self, file_path: str) -> List[Document]:
        """Load a PDF document and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        # Check cache first - for faster processing
        if file_path in self.cache:
            return self.cache[file_path]
            
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        documents = []
        pdf_document = fitz.open(file_path)
        
        # Process pages
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text with enhanced validation
            text = self._extract_text_with_fallback(page)
            
            # Create document for this page
            doc = Document(
                text=text,
                metadata={
                    'source': file_path,
                    'page': page_num,
                    'total_pages': len(pdf_document),
                    'extraction_quality': 'high' if self._validate_text(text) else 'low'
                }
            )
            documents.append(doc)
        
        pdf_document.close()
        
        # Cache the result
        self.cache[file_path] = documents
        return documents

    def load_directory(self, directory_path: str) -> List[Document]:
        """Load all PDF documents from a directory.
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
            
        documents = []
        for filename in os.listdir(directory_path):
            if filename.lower().endswith('.pdf'):
                file_path = os.path.join(directory_path, filename)
                documents.extend(self.load_document(file_path))
                
        return documents