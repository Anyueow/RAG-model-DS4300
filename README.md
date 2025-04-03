# Local Retrieval-Augmented Generation System for DS4300 Notes
A local RAG system that leverages course notes to provide accurate, context-aware responses to questions about data science topics.

## Overview

Retrieval-Augmented Generation (RAG) is an AI architecture that combines the power of large language models with external knowledge retrieval to provide more accurate and contextual responses. This project implements a local RAG system specifically designed to answer questions about data science topics using course notes as the knowledge base.

The system's primary goal is to demonstrate the effectiveness of different RAG configurations by comparing various embedding models, vector databases, chunking strategies, and LLM models in a controlled environment.

## Tech Stack

- **Python 3.8+**: Core programming language
- **Ollama**: Local LLM deployment
  - Mistral: Latest version for primary testing
  - Qwen: 7B model for comparative analysis
- **Vector Databases**:
  - Redis: High-performance in-memory vector store (fastest for real-time queries)
  - Qdrant: Vector similarity search engine (best for complex similarity metrics)
  - Chroma: Open-source vector database (easiest to set up and maintain)
- **Embedding Models**:
  - Nomic AI's nomic-embed-text-v1.5
  - MiniLM (multi-qa-MiniLM-L6-cos-v1)
  - MPNet (all-mpnet-base-v2)
- **Additional Libraries**:
  - Sentence Transformers for embeddings
  - LangChain for text processing
  - Plotly for visualization
  - Pandas for data analysis

## Features

- **Document Processing**:
  - Automatic document ingestion from various formats
  - Configurable text chunking with overlap control
  - Metadata extraction and storage

- **Embedding Pipeline**:
  - Support for multiple embedding models
  - Vector dimension validation
  - Efficient batch processing

- **Vector Database Integration**:
  - Unified interface for multiple vector DBs
  - Automatic collection management
  - Configurable similarity search parameters

- **LLM Integration**:
  - Local LLM deployment via Ollama
  - Configurable prompt templates
  - Temperature and sampling controls

- **Evaluation Framework**:
  - Standardized question sets
  - Performance metrics tracking
  - Interactive visualization tools
  - CSV export capabilities

## Project Structure

```
.
├── data/                   # Course notes and evaluation data
├── database/              # Vector database implementations
│   ├── chroma_db.py      # Chroma DB client
│   ├── qdrant_db.py      # Qdrant DB client
│   └── redis_db.py       # Redis DB client
├── embeddings/           # Embedding model implementations
│   ├── sentence_transformer.py
│   └── test_config.py    # Model configurations
├── llm/                  # LLM interface
│   └── llm_interface.py  # Ollama integration
├── evaluation/          # Evaluation scripts and tools
│   ├── evaluate_rag.py  # Main evaluation script
│   └── generate_evaluation_responses_*.py  # Response generators
├── main.py             # Main RAG system implementation
└── requirements.txt    # Project dependencies
```

## Installation

1. **Set up Python Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up Ollama**:
   ```bash
   # Install Ollama (follow instructions at https://ollama.ai)
   # Pull the required models
   ollama pull mistral:latest
   ollama pull qwen:7b
   ```

4. **Vector Database Setup**:
   ```bash
   # Redis (if using)
   docker run -d -p 6379:6379 redis

   # Qdrant (if using)
   docker run -d -p 6333:6333 qdrant/qdrant
   ```

## Usage

1. **Run Evaluation Pipeline**:
   ```bash
   # Generate responses with Redis
   python generate_evaluation_responses_redis.py

   # Generate responses with Qdrant
   python generate_evaluation_responses_qdrant.py

   # Generate responses with Chroma
   python generate_evaluation_responses_chroma.py
   ```

2. **Evaluate Results**:
   ```bash
   python evaluate_rag.py
   ```

3. **View Results**:
   - Check `evaluation_results/` for JSON outputs
   - View interactive visualizations in `evaluation_results/visualizations/`
   - Access raw data in `evaluation_results/raw_responses.csv`

## Configuration

The system supports various configuration options:

1. **Chunking Strategies**:
   ```json
   {
     "chunk_size": 256,  // or 512, 1024
     "chunk_overlap": 25 // or 50, 100
   }
   ```

2. **Embedding Models**:
   ```python
   EMBEDDING_MODELS = {
       "nomic-ai/nomic-embed-text-v1.5": {...},
       "multi-qa-MiniLM-L6-cos-v1": {...},
       "all-mpnet-base-v2": {...}
   }
   ```

3. **Vector Database Settings**:
   ```python
   vector_db = RedisDB(
       collection_name="eval_config_name",
       embedding_model="model_name"
   )
   ```

4. **LLM Parameters**:
   ```python
   # Mistral configuration
   llm = OllamaLLM(
       model_name="mistral:latest",
       temperature=0.4
   )
   
   # Qwen configuration
   llm = OllamaLLM(
       model_name="qwen:7b",
       temperature=0.4
   )
   ```

## Experiments

The system evaluates different configurations:

1. **Chunking Strategies**:
   - Small (256/25): Fine-grained retrieval
   - Medium (512/50): Balanced approach
   - Large (1024/100): Context preservation

2. **Embedding Models**:
   - Nomic AI: Latest generation embeddings
   - MiniLM: Fast and efficient
   - MPNet: Strong semantic understanding

3. **Vector Databases**:
   - Redis: In-memory performance
   - Qdrant: Advanced similarity search
   - Chroma: Local persistence

4. **LLM Models**:
   - Mistral: Latest version for primary testing
   - Qwen: 7B model for comparative analysis

## Results

Key findings from the evaluation:

1. **Performance Metrics**:
   - Memory usage across configurations
   - Execution time for different components
   - Response quality and relevance
   - LLM model comparison (Mistral vs. Qwen)

2. **Best Performing Configuration**:
   - Optimal chunking strategy
   - Most effective embedding model
   - Preferred vector database
   - Preferred LLM model

3. **Trade-offs**:
   - Speed vs. accuracy
   - Memory usage vs. context size
   - Local vs. distributed storage
   - LLM performance vs. resource usage

Detailed results and visualizations are available in the `evaluation_results/` directory. 