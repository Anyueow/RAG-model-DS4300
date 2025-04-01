from typing import List, Optional
import re
from nltk.corpus import stopwords
import nltk

class TextPreprocessor:
    """Handles text preprocessing operations."""
    
    def __init__(self, remove_stopwords: bool = False):
        """Initialize the preprocessor.
        
        Args:
            remove_stopwords: Whether to remove stopwords
        """
        self.remove_stopwords = remove_stopwords
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def preprocess_text(self, text: str) -> str:
        """Apply all preprocessing steps to the text.
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and punctuation
        # Keep basic punctuation that might be important for sentence structure
        text = re.sub(r'[^\w\s.,!?]', '', text)
        
        # Remove stopwords if enabled
        if self.remove_stopwords:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            text = ' '.join([word for word in words if word not in stop_words])
        
        # Final whitespace cleanup
        text = text.strip()
        
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        """Preprocess a batch of texts.
        
        Args:
            texts: List of texts to preprocess
            
        Returns:
            List of preprocessed texts
        """
        return [self.preprocess_text(text) for text in texts] 