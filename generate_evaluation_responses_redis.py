import os
import json
import time
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import argparse

from main import RAGSystem
from database.redis_db import RedisDB
from embeddings.sentence_transformer import SentenceTransformerEmbedder
from embeddings.test_config import EMBEDDING_MODELS
from llm.llm_interface import OllamaLLM

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_generation_redis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ResponseGenerator:
    """Generate evaluation responses using different RAG configurations with Redis DB."""
    
    def __init__(self, data_dir: str = "data", results_dir: str = "evaluation_results"):
        """Initialize the response generator."""
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.questions = [
            "What is the difference between B-trees and B+ trees?",
            "Explain how AVL tree rotation works",
            "What are the main advantages of MongoDB?",
            "How does AWS S3 handle data replication?",
        ]
        
        # Define configurations to test
        self.configurations = [
            # Small chunks (256/25)
            {
                "name": "small_nomic_redis_mistral",
                "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "small",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 256,
                "chunk_overlap": 25,
                "top_k": 3
            },
            {
                "name": "small_minilm_redis_mistral",
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "small",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 256,
                "chunk_overlap": 25,
                "top_k": 3
            },
            {
                "name": "small_mpnet_redis_mistral",
                "embedding_model": "all-mpnet-base-v2",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "small",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 256,
                "chunk_overlap": 25,
                "top_k": 3
            },
            # Medium chunks (512/50)
            {
                "name": "medium_nomic_redis_mistral",
                "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "medium",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 3
            },
            {
                "name": "medium_minilm_redis_mistral",
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "medium",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 3
            },
            {
                "name": "medium_mpnet_redis_mistral",
                "embedding_model": "all-mpnet-base-v2",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "medium",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 512,
                "chunk_overlap": 50,
                "top_k": 3
            },
            # Large chunks (1024/100)
            {
                "name": "large_nomic_redis_mistral",
                "embedding_model": "nomic-ai/nomic-embed-text-v1.5",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "large",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "top_k": 3
            },
            {
                "name": "large_minilm_redis_mistral",
                "embedding_model": "multi-qa-MiniLM-L6-cos-v1",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "large",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "top_k": 3
            },
            {
                "name": "large_mpnet_redis_mistral",
                "embedding_model": "all-mpnet-base-v2",
                "llm_model": "mistralai/Mistral-7B-Instruct-v0.2",
                "vector_db": "redis",
                "chunking_strategy": "large",
                "semantic_weight": 0.7,
                "keyword_weight": 0.3,
                "chunk_size": 1024,
                "chunk_overlap": 100,
                "top_k": 3
            }
        ]
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def create_rag_system(self, config: Dict) -> RAGSystem:
        """Create a RAG system with the specified configuration."""
        # Initialize embedder
        model_config = EMBEDDING_MODELS[config["embedding_model"]]
        embedder = SentenceTransformerEmbedder(model_config)
        
        # Initialize Redis vector database with the correct embedding model
        vector_db = RedisDB(
            collection_name=f"eval_{config['name']}",
            embedding_model=config["embedding_model"]
        )
        
        # Initialize LLM
        llm = OllamaLLM(model_name=config["llm_model"], temperature=0.4)
        
        # Create and return RAG system
        return RAGSystem(
            embedder=embedder,
            vector_db=vector_db,
            llm=llm,
            semantic_weight=config["semantic_weight"],
            keyword_weight=config["keyword_weight"],
            chunk_size=config["chunk_size"],
            chunk_overlap=config["chunk_overlap"],
            top_k=config["top_k"]
        )
    
    def process_question(self, rag: RAGSystem, question: str, config: Dict) -> Dict:
        """Process a single question and return results with performance metrics."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        try:
            # Get response
            result = rag.query(question)
            
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            return {
                "question": question,
                "response": result.get("response", ""),
                "execution_time": end_time - start_time,
                "memory_usage": end_memory - start_memory,
                "contexts": result.get("contexts", [])
            }
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return {
                "question": question,
                "response": f"Error: {str(e)}",
                "execution_time": time.time() - start_time,
                "memory_usage": self.get_memory_usage() - start_memory,
                "contexts": []
            }
    
    def generate_responses(self) -> None:
        """Generate responses for all configurations and questions."""
        logger.info("Starting response generation with Redis DB...")
        
        for config in self.configurations:
            logger.info(f"\nProcessing configuration: {config['name']}")
            
            try:
                # Create RAG system
                rag = self.create_rag_system(config)
                
                # Process documents if needed
                if not os.path.exists(self.data_dir / "processed_redis"):
                    logger.info("Processing documents...")
                    rag.ingest_documents(str(self.data_dir))
                    (self.data_dir / "processed_redis").touch()
                
                # Process all questions
                responses = {}
                for i, question in enumerate(self.questions):
                    logger.info(f"Processing question {i+1}/{len(self.questions)}")
                    responses[f"question_{i+1}"] = self.process_question(rag, question, config)
                
                # Save results
                result = {
                    "configuration": config,
                    "responses": responses,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                output_file = self.results_dir / f"{config['name']}_results.json"
                with open(output_file, 'w') as f:
                    json.dump(result, f, indent=2)
                
                logger.info(f"Results saved to {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing configuration {config['name']}: {str(e)}")
                continue
    
    def generate_summary(self) -> None:
        """Generate a summary of all results."""
        summary = {
            "total_configurations": len(self.configurations),
            "total_questions": len(self.questions),
            "configurations": []
        }
        
        for config in self.configurations:
            result_file = self.results_dir / f"{config['name']}_results.json"
            if result_file.exists():
                with open(result_file, 'r') as f:
                    result = json.load(f)
                    
                    # Calculate average metrics
                    times = [resp['execution_time'] for resp in result['responses'].values()]
                    memories = [resp['memory_usage'] for resp in result['responses'].values()]
                    
                    config_summary = {
                        "name": config['name'],
                        "embedding_model": config['embedding_model'],
                        "llm_model": config['llm_model'],
                        "vector_db": config['vector_db'],
                        "avg_time": np.mean(times),
                        "avg_memory": np.mean(memories),
                        "total_questions": len(result['responses'])
                    }
                    summary["configurations"].append(config_summary)
        
        # Save summary
        with open(self.results_dir / "summary_redis.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to {self.results_dir}/summary_redis.json")

def main():
    parser = argparse.ArgumentParser(description='Generate evaluation responses for RAG system with Redis DB')
    parser.add_argument('--data-dir', default='data',
                      help='Directory containing documents to process')
    parser.add_argument('--results-dir', default='evaluation_results',
                      help='Directory to save evaluation results')
    args = parser.parse_args()
    
    generator = ResponseGenerator(args.data_dir, args.results_dir)
    generator.generate_responses()
    generator.generate_summary()

if __name__ == "__main__":
    main() 