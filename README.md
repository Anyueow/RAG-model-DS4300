# RAG Model for Course Notes

This project implements a Retrieval-Augmented Generation (RAG) system for processing and querying course notes. The system uses local LLMs and vector databases to provide accurate, context-aware responses to questions about course content.

## Features

- Document ingestion from various formats (PDF, text)
- Flexible text chunking strategies
- Multiple embedding model support
- Vector database integration (Redis, Chroma)
- Local LLM integration via Ollama
- Modular architecture following SOLID principles

## Project Structure

```
project/
├── data/
│   └── raw_notes/          # Place course notes here
├── ingestion/
│   └── data_loader.py      # Document loading and preprocessing
├── preprocessing/
│   └── chunker.py          # Text chunking strategies
├── embeddings/
│   ├── base_embedder.py    # Base embedding interface
│   └── model_*.py          # Specific embedding model implementations
├── database/
│   ├── base_db.py          # Base vector DB interface
│   └── db_*.py             # Specific DB implementations
├── query/
│   └── query_handler.py    # Query processing and retrieval
├── llm/
│   └── llm_interface.py    # Local LLM integration
└── main.py                 # Main RAG system implementation
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd rag-model-ds4300
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama and download required models:
```bash
# Follow Ollama installation instructions for your OS
ollama pull llama2
```

## Usage

1. Place your course notes in the `data/raw_notes` directory.

2. Initialize and use the RAG system:
```python
from main import RAGSystem
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from database.chroma_db import ChromaDB

# Initialize components
embedder = SentenceTransformerEmbedder()
vector_db = ChromaDB()

# Create RAG system
rag_system = RAGSystem(embedder, vector_db)

# Ingest documents
rag_system.ingest_documents("data/raw_notes")

# Query the system
result = rag_system.query("What is the main topic of the course?")
print(result['response'])
```

## Development

The project follows SOLID principles and uses abstract base classes to ensure modularity and extensibility. To add new features:

1. Create new implementations of base classes (e.g., new embedding models)
2. Add new components following the existing patterns
3. Update the main RAG system to support new features

## Testing

Run tests using pytest:
```bash
pytest tests/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 