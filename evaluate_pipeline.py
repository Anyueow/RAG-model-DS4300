"""Comprehensive evaluation script for RAG pipeline configurations."""

import time
import psutil
import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
import logging
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

from main import RAGSystem
from database.chroma_db import ChromaDB
from database.redis_db import RedisDB
from database.qdrant_db import QdrantDB
from embeddings.test_config import EMBEDDING_MODELS
from llm.llm_interface import OllamaLLM
from preprocessing.chunker import ChunkingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline_evaluation.log'),
        logging.StreamHandler()
    ]
)

def measure_memory() -> float:
    """Measure current memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

# Global mapping for vector DB constructors used in evaluation
VECTOR_DBS = {
    "chroma": lambda: ChromaDB(collection_name="eval_collection"),
    "redis": lambda: RedisDB(index_name="eval_index"),
    "qdrant": lambda: QdrantDB(collection_name="eval_collection")
}

def evaluate_config_task(vector_db: str, embedding_model: str, llm_model: str,
                         chunking_config: ChunkingConfig, data_dir: str,
                         test_queries: List[str]) -> List[Dict[str, Any]]:
    """Evaluate a single pipeline configuration."""
    config_results = []
    try:
        # Initialize components
        db_instance = VECTOR_DBS[vector_db]()
        model_config = EMBEDDING_MODELS[embedding_model]
        llm = OllamaLLM(model_name=llm_model)
        
        start_time = time.time()
        start_memory = measure_memory()
        
        # Initialize RAG system
        rag = RAGSystem(
            embedder=None,  # Will be initialized with model_config internally
            vector_db=db_instance,
            llm=llm,
            model_config=model_config,
            chunk_size=chunking_config.chunk_size,
            chunk_overlap=chunking_config.overlap
        )
        
        # Ingest documents
        rag.ingest_documents(str(data_dir))
        
        ingestion_time = time.time() - start_time
        memory_usage = measure_memory() - start_memory
        
        # Process each query
        for query in test_queries:
            query_start_time = time.time()
            query_start_memory = measure_memory()
            try:
                result = rag.query(query)
                config_results.append({
                    'vector_db': vector_db,
                    'embedding_model': embedding_model,
                    'llm_model': llm_model,
                    'chunk_size': chunking_config.chunk_size,
                    'chunk_overlap': chunking_config.overlap,
                    'query': query,
                    'response': result.get('response'),
                    'num_contexts': len(result.get('contexts', [])),
                    'query_time': time.time() - query_start_time,
                    'query_memory': measure_memory() - query_start_memory,
                    'ingestion_time': ingestion_time,
                    'ingestion_memory': memory_usage,
                    'status': 'success',
                    'error': None
                })
            except Exception as e:
                config_results.append({
                    'vector_db': vector_db,
                    'embedding_model': embedding_model,
                    'llm_model': llm_model,
                    'chunk_size': chunking_config.chunk_size,
                    'chunk_overlap': chunking_config.overlap,
                    'query': query,
                    'response': None,
                    'num_contexts': 0,
                    'query_time': time.time() - query_start_time,
                    'query_memory': measure_memory() - query_start_memory,
                    'ingestion_time': ingestion_time,
                    'ingestion_memory': memory_usage,
                    'status': 'error',
                    'error': str(e)
                })
        
        # Clear the database after evaluation
        db_instance.clear()
        
    except Exception as e:
        logging.error(f"Error evaluating configuration: {str(e)}")
    return config_results

class PipelineEvaluator:
    """Evaluator for different RAG pipeline configurations."""
    
    def __init__(self, data_dir: str):
        """Initialize the pipeline evaluator.
        
        Args:
            data_dir: Directory containing test documents
        """
        self.data_dir = Path(data_dir)
        
        # Standard test queries
        self.test_queries = [
            "What is the difference between B-trees and B+ trees?",
            "Explain how AVL tree rotation works",
            "What are the main advantages of MongoDB?",
            "How does AWS S3 handle data replication?",
            "Explain the concept of database indexing",
        ]
        
        # Pipeline configurations
        self.vector_dbs = list(VECTOR_DBS.keys())
        self.embedding_models = list(EMBEDDING_MODELS.keys())
        self.llm_models = [
            "qwen:7b",
            "llama2:7b",
            "mistral:7b"
        ]
        self.chunking_configs = [
            ChunkingConfig(chunk_size=256, overlap=20),
            ChunkingConfig(chunk_size=512, overlap=50),
            ChunkingConfig(chunk_size=1024, overlap=100)
        ]
        
        self.results = []
        
    def run_evaluation(self):
        """Run the complete pipeline evaluation in parallel."""
        tasks = []
        # Prepare a list of configuration tuples
        for vector_db in self.vector_dbs:
            for embedding_model in self.embedding_models:
                for llm_model in self.llm_models:
                    for chunking_config in self.chunking_configs:
                        tasks.append((
                            vector_db,
                            embedding_model,
                            llm_model,
                            chunking_config,
                            str(self.data_dir),
                            self.test_queries
                        ))
                        
        logging.info(f"Total configurations to evaluate: {len(tasks)}")
        
        # Run tasks in parallel
        all_results = []
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(evaluate_config_task, *task): task for task in tasks}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating configurations"):
                try:
                    res = future.result()
                    all_results.extend(res)
                except Exception as e:
                    logging.error(f"Error in parallel task: {str(e)}")
                    
        self.results = all_results
        # Save the results after evaluation
        self.save_results()
    
    def save_results(self):
        """Save evaluation results to CSV and JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save to CSV
        csv_file = f"pipeline_evaluation_{timestamp}.csv"
        df = pd.DataFrame(self.results)
        df.to_csv(csv_file, index=False)
        
        # Save to JSON
        json_file = f"pipeline_evaluation_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Generate summary
        self.generate_summary(df, timestamp)
        
        logging.info(f"Results saved to {csv_file} and {json_file}")
    
    def generate_summary(self, df: pd.DataFrame, timestamp: str):
        """Generate a summary of the evaluation results.
        
        Args:
            df: DataFrame containing results
            timestamp: Timestamp string
        """
        summary_file = f"pipeline_evaluation_summary_{timestamp}.txt"
        
        with open(summary_file, 'w') as f:
            f.write("=== RAG Pipeline Evaluation Summary ===\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write(f"Total configurations tested: {len(df['vector_db'].unique()) * len(df['embedding_model'].unique()) * len(df['llm_model'].unique())}\n")
            f.write(f"Total queries executed: {len(df)}\n")
            f.write(f"Success rate: {(df['status'] == 'success').mean():.2%}\n\n")
            
            # Performance by vector DB
            f.write("Performance by Vector DB:\n")
            db_stats = df.groupby('vector_db').agg({
                'query_time': ['mean', 'std'],
                'ingestion_time': 'mean',
                'ingestion_memory': 'mean'
            }).round(3)
            f.write(db_stats.to_string())
            f.write("\n\n")
            
            # Performance by embedding model
            f.write("Performance by Embedding Model:\n")
            embed_stats = df.groupby('embedding_model').agg({
                'query_time': ['mean', 'std'],
                'query_memory': 'mean'
            }).round(3)
            f.write(embed_stats.to_string())
            f.write("\n\n")
            
            # Performance by LLM
            f.write("Performance by LLM:\n")
            llm_stats = df.groupby('llm_model').agg({
                'query_time': ['mean', 'std']
            }).round(3)
            f.write(llm_stats.to_string())
            f.write("\n\n")
            
            # Performance by chunking configuration
            f.write("Performance by Chunking Configuration:\n")
            chunk_stats = df.groupby(['chunk_size', 'chunk_overlap']).agg({
                'query_time': 'mean',
                'num_contexts': 'mean'
            }).round(3)
            f.write(chunk_stats.to_string())
            
        logging.info(f"Summary saved to {summary_file}")

def main():
    """Run the pipeline evaluation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline configurations")
    parser.add_argument("--data-dir", type=str, default="data",
                      help="Directory containing test documents")
    args = parser.parse_args()
    
    evaluator = PipelineEvaluator(args.data_dir)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main()
