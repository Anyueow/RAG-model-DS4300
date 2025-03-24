# RAG Model for Course Notes

A Retrieval-Augmented Generation (RAG) system for processing and querying course notes using local LLMs.

## Features

- Document ingestion from PDF files
- Text chunking with configurable strategies
- Vector embeddings using SentenceTransformer
- Vector database support (ChromaDB and Redis)
- Local LLM integration using Ollama
- SOLID principles implementation
- Comprehensive testing suite

## Project Structure

```
rag-model/
├── data/
│   └── raw_notes/          # Place your course notes here
├── database/
│   ├── base_db.py          # Base vector database interface
│   ├── chroma_db.py        # ChromaDB implementation
│   └── redis_db.py         # Redis vector database implementation
├── embeddings/
│   ├── base_embedder.py    # Base embedding interface
│   └── sentence_transformer.py  # SentenceTransformer implementation
├── ingestion/
│   └── data_loader.py      # Document loading and processing
├── llm/
│   └── llm_interface.py    # Local LLM integration
├── preprocessing/
│   └── chunker.py          # Text chunking strategies
├── query/
│   └── query_handler.py    # Query processing
├── tests/
│   └── test_rag.py         # Test suite
├── main.py                 # Main RAG system implementation
├── example.py              # Example usage and benchmarking
├── setup.py               # Setup script
└── requirements.txt       # Project dependencies
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/rag-model.git
cd rag-model
```

2. Run the setup script:
```bash
python setup.py
```

3. Install Ollama:
   - Visit [Ollama.ai](https://ollama.ai/) to download and install
   - The setup script will automatically pull the required model

4. Set up Vector Database (Choose one or both):

   ### Option 1: ChromaDB (Default)
   - No additional setup required
   - Data is stored locally in the `chroma_db` directory

   ### Option 2: Redis Vector DB
   - Install Docker if not already installed
   - Start Redis container:
     ```bash
     # If container doesn't exist:
     docker run -d --name redis-stack -p 6379:6379 -p 8001:8001 redis/redis-stack:latest
     
     # If container exists but is stopped:
     docker start redis-stack
     ```
   - Redis will be available at `localhost:6379`

## Usage

1. Activate the conda environment:
```bash
conda activate rag-model
```

2. Place your course notes in the `data/raw_notes` directory

3. Run the example script to test the system:
```bash
python example.py
```

4. Run the tests to verify functionality:
```bash
pytest tests/
```

## Development

### Adding New Components

1. **New Embedding Model**:
   - Implement `BaseEmbedder` interface
   - Add to `embeddings/` directory

2. **New Vector Database**:
   - Implement `BaseVectorDB` interface
   - Add to `database/` directory

3. **New Chunking Strategy**:
   - Implement `BaseChunker` interface
   - Add to `preprocessing/` directory

4. **New LLM Interface**:
   - Implement `BaseLLM` interface
   - Add to `llm/` directory

### Testing

- Run all tests: `pytest tests/`
- Run specific test: `pytest tests/test_rag.py -k "test_name"`

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 