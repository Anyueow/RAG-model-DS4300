from typing import List, Dict, Any
import os
from pathlib import Path
import PyPDF2
import pdfplumber
from abc import ABC, abstractmethod

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
    """Loader for PDF documents."""
    
    def load_document(self, file_path: str) -> str:
        """Load and extract text from a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        text = ""
        try:
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"Error loading PDF {file_path}: {str(e)}")
            return ""
        return text

class DataLoader:
    """Main data loader class that handles different document types."""
    
    def __init__(self):
        self.loaders: Dict[str, BaseDocumentLoader] = {
            '.pdf': PDFLoader(),
            # Add more loaders for different file types here
        }
    
    def load_documents(self, directory: str) -> List[Dict[str, Any]]:
        """Load all documents from a directory.
        
        Args:
            directory: Path to the directory containing documents
            
        Returns:
            List of dictionaries containing document text and metadata
        """
        documents = []
        for file_path in Path(directory).rglob('*'):
            if file_path.suffix.lower() in self.loaders:
                loader = self.loaders[file_path.suffix.lower()]
                text = loader.load_document(str(file_path))
                if text:
                    documents.append({
                        'text': text,
                        'file_path': str(file_path),
                        'file_type': file_path.suffix.lower()
                    })
        return documents
    
    def add_loader(self, extension: str, loader: BaseDocumentLoader) -> None:
        """Add a new document loader.
        
        Args:
            extension: File extension (e.g., '.txt')
            loader: Document loader instance
        """
        self.loaders[extension.lower()] = loader 