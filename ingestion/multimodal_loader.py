from typing import List, Dict, Any
import fitz  # PyMuPDF
from .data_loader import BaseDocumentLoader
import os

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

class PDFLoader(BaseDocumentLoader):
    """Loader for PDF documents that focuses on robust text extraction."""
    
    def __init__(self):
        """Initialize the PDF loader."""
        pass

    def load_document(self, file_path: str) -> List[Document]:
        """Load a PDF document and extract text.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            List of Document objects
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
            
        documents = []
        pdf_document = fitz.open(file_path)
        
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Extract text with enhanced settings for handwritten text
            text = page.get_text("text", sort=True)  # sort=True helps with handwritten text
            
            # Create document for this page
            doc = Document(
                text=text,
                metadata={
                    'source': file_path,
                    'page': page_num,
                    'total_pages': len(pdf_document)
                }
            )
            documents.append(doc)
        
        pdf_document.close()
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