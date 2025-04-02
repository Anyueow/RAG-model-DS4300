# RAG System for DS4300 Midterm Cheat Sheet

A Retrieval-Augmented Generation (RAG) system built for DS4300 Midterm preparation, featuring a Streamlit-based user interface and multiple embedding model support.

## Prerequisites

- Python 3.8+
- Ollama installed and running locally
- Redis server running locally (for vector database)
- Required Python packages (install via `pip install -r requirements.txt`)

## Ollama Models Required

Before running the system, ensure you have the following Ollama models pulled:

```bash
ollama pull nomic-ai/nomic-embed-text-v1.5  # For embeddings
ollama pull qwen:7b                          # For LLM
```

## Running the Streamlit UI

1. First, ensure Redis is running:
```bash
redis-server
```

2. In a new terminal, start the Streamlit app:
```bash
streamlit run app.py
```

3. The UI will open in your default web browser at `http://localhost:8501`

## Final RAG Configuration (Streamlit UI)

The Streamlit UI uses the following optimized configuration:

### Embedding Model
- Model: `nomic-ai/nomic-embed-text-v1.5`
- Type: Ollama-based embedding model
- Embedding Dimension: 768
- Max Length: 512 tokens

### LLM Configuration
- Model: `qwen:7b`
- Temperature: 0.4 (for balanced creativity and consistency)
- Context Window: 4096 tokens

### Vector Database
- Type: ChromaDB
- Collection Name: "app_collection"
- Distance Metric: Cosine Similarity

### Search Configuration
- Semantic Weight: 0.8 (emphasizes semantic understanding)
- Keyword Weight: 0.2 (supplementary keyword matching)
- Top-k Results: 3 (number of contexts retrieved)

### Text Processing
- Chunk Size: 512 tokens
- Chunk Overlap: 50 tokens
- Tokenizer: tiktoken (for accurate token counting)

### Caching
- Query Results Cache: Enabled
- Embedding Cache: Enabled
- Context Cache: Enabled

## Features

- Interactive query interface
- Real-time document processing
- Support for PDF documents
- Hybrid search (semantic + keyword)
- Context-aware responses
- Memory-efficient document processing
- Parallel document ingestion
- System status monitoring

## Usage

1. **Initialize the System**
   - Click "Initialize/Update RAG System" in the sidebar
   - Wait for the initialization to complete

2. **Process Documents**
   - Place your PDF documents in the `data` directory
   - Click "Process Data Directory" in the sidebar
   - Monitor the processing status

3. **Query the System**
   - Enter your question in the text area
   - Click "Search"
   - View the response and relevant contexts

## Performance Considerations

- The system uses parallel processing for document ingestion
- Documents are processed in chunks of 10 for optimal memory usage
- Embeddings and responses are cached for faster subsequent queries
- The UI provides real-time feedback on system status and processing

## Troubleshooting

1. **Redis Connection Issues**
   - Ensure Redis server is running: `redis-server`
   - Check Redis port (default: 6379)

2. **Ollama Issues**
   - Verify Ollama is running: `ollama list`
   - Ensure required models are pulled
   - Check model availability in Ollama

3. **Memory Issues**
   - Monitor system memory usage
   - Adjust chunk size if needed
   - Clear vector database if necessary

## Development

For development and evaluation purposes, separate evaluation scripts are available:
- `evaluate_mpnet.py`: For all-mpnet-base-v2 model
- `evaluate_minilm.py`: For multi-qa-MiniLM-L6-cos-v1 model
- `evaluate_nomic.py`: For nomic-ai/nomic-embed-text-v1.5 model

Each script can be run independently to evaluate different embedding models and configurations. 