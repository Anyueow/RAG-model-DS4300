import time
import random
import argparse
import json
import os
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import logging
from datetime import datetime
from main import RAGSystem
from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from database.qdrant_db import QdrantDB

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_evaluation.log'),
        logging.StreamHandler()
    ]
)

class RAGEvaluator:
    def __init__(self, data_dir: str, test_queries: List[str]):
        """Initialize the RAG evaluator.
        
        Args:
            data_dir: Directory containing test documents
            test_queries: List of test queries
        """
        self.data_dir = data_dir
        self.test_queries = test_queries
        self.results = {}
        
        # Vector database configurations
        self.vector_dbs = {
            "chroma": ChromaDB(persist_directory="chroma_db_eval"),
            "redis": RedisVectorDB(),
            "qdrant": QdrantDB(path="./qdrant_db_eval")
        }
        
        # Model configurations
        self.models = [
            "qwen:7b",
            "mistral",
            "llama2"
        ]
        
        # Semantic weights to test
        self.semantic_weights = [0.0, 0.3, 0.5]
        
    def evaluate_vector_db(self, db_name: str, db_instance: Any) -> Dict[str, float]:
        """Evaluate a vector database's performance.
        
        Args:
            db_name: Name of the vector database
            db_instance: Instance of the vector database
            
        Returns:
            Dictionary of performance metrics
        """
        logging.info(f"Evaluating vector database: {db_name}")
        metrics = {
            "indexing_time": 0.0,
            "query_times": [],
            "memory_usage": 0.0
        }
        
        # Test indexing
        start_time = time.time()
        try:
            # Process test documents
            pdf_files = [f for f in os.listdir(self.data_dir) if f.endswith('.pdf')]
            for pdf_file in tqdm(pdf_files, desc=f"Indexing {db_name}"):
                file_path = os.path.join(self.data_dir, pdf_file)
                # Add vectors to database
                # This is a placeholder - implement actual indexing
                time.sleep(0.1)  # Simulate processing
        except Exception as e:
            logging.error(f"Error indexing {db_name}: {str(e)}")
        metrics["indexing_time"] = time.time() - start_time
        
        # Test querying
        for query in tqdm(self.test_queries, desc=f"Querying {db_name}"):
            start_time = time.time()
            try:
                # Perform query
                # This is a placeholder - implement actual querying
                time.sleep(0.05)  # Simulate processing
            except Exception as e:
                logging.error(f"Error querying {db_name}: {str(e)}")
            metrics["query_times"].append(time.time() - start_time)
        
        # Calculate average query time
        metrics["avg_query_time"] = np.mean(metrics["query_times"])
        metrics["query_time_std"] = np.std(metrics["query_times"])
        
        return metrics
    
    def evaluate_model(self, model_name: str, vector_db: Any, semantic_weight: float) -> Dict[str, Any]:
        """Evaluate an Ollama model's performance.
        
        Args:
            model_name: Name of the Ollama model
            vector_db: Vector database instance
            semantic_weight: Semantic search weight
            
        Returns:
            Dictionary of performance metrics
        """
        logging.info(f"Evaluating model: {model_name} with semantic_weight: {semantic_weight}")
        
        # Initialize metrics with default values
        metrics = {
            "response_times": [],
            "context_lengths": [],
            "source_counts": [],
            "avg_response_time": 0.0,
            "response_time_std": 0.0,
            "avg_context_length": 0.0,
            "avg_source_count": 0.0,
            "success_count": 0,
            "error_count": 0,
            "error_messages": []  # Track specific error messages
        }
        
        # Initialize RAG system
        try:
            logging.info(f"Initializing RAG system with model {model_name} and vector_db {type(vector_db).__name__}")
            # Create an OllamaLLM instance with the model name
            from llm.llm_interface import OllamaLLM
            llm = OllamaLLM(model_name)
            
            # Calculate keyword weight to ensure weights sum to 1
            keyword_weight = 1.0 - semantic_weight
            
            rag_system = RAGSystem(
                vector_db=vector_db,
                llm=llm,  # Pass the OllamaLLM instance
                semantic_weight=semantic_weight,
                keyword_weight=keyword_weight  # Explicitly set keyword_weight
            )
            logging.info("RAG system initialized successfully")
        except Exception as e:
            error_msg = f"Error initializing RAG system: {str(e)}"
            logging.error(error_msg)
            metrics["error_messages"].append(error_msg)
            metrics["error_count"] += 1
            return metrics
        
        # Test queries
        for query in tqdm(self.test_queries, desc=f"Testing {model_name}"):
            start_time = time.time()
            try:
                logging.info(f"Processing query: {query}")
                # Get response
                result = rag_system.query(query)
                
                # Record metrics
                metrics["response_times"].append(time.time() - start_time)
                metrics["context_lengths"].append(len(result.get("context", "")))
                metrics["source_counts"].append(len(result.get("sources", [])))
                metrics["success_count"] += 1
                logging.info(f"Successfully processed query with {model_name}")
                
            except Exception as e:
                error_msg = f"Error processing query with {model_name}: {str(e)}"
                logging.error(error_msg)
                metrics["error_messages"].append(error_msg)
                metrics["error_count"] += 1
        
        # Calculate statistics only if we have successful queries
        if metrics["success_count"] > 0:
            metrics["avg_response_time"] = np.mean(metrics["response_times"])
            metrics["response_time_std"] = np.std(metrics["response_times"])
            metrics["avg_context_length"] = np.mean(metrics["context_lengths"])
            metrics["avg_source_count"] = np.mean(metrics["source_counts"])
        
        return metrics
    
    def run_evaluation(self):
        """Run the complete evaluation."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results = {
            "timestamp": timestamp,
            "vector_dbs": {},
            "models": {}
        }
        
        # Evaluate vector databases
        for db_name, db_instance in self.vector_dbs.items():
            self.results["vector_dbs"][db_name] = self.evaluate_vector_db(db_name, db_instance)
        
        # Evaluate models with different configurations
        for model_name in self.models:
            self.results["models"][model_name] = {}
            for db_name, db_instance in self.vector_dbs.items():
                self.results["models"][model_name][db_name] = {}
                for weight in self.semantic_weights:
                    self.results["models"][model_name][db_name][f"weight_{weight}"] = \
                        self.evaluate_model(model_name, db_instance, weight)
        
        # Save results
        self.save_results()
    
    def save_results(self):
        """Save evaluation results to a JSON file."""
        timestamp = self.results["timestamp"]
        filename = f"rag_evaluation_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logging.info(f"Results saved to {filename}")
        
        # Print summary
        self.print_summary()
    
    def print_summary(self):
        """Print a summary of the evaluation results."""
        print("\n=== RAG System Evaluation Summary ===")
        print(f"Timestamp: {self.results['timestamp']}")
        
        print("\nVector Database Performance:")
        for db_name, metrics in self.results["vector_dbs"].items():
            print(f"\n{db_name}:")
            print(f"  Indexing Time: {metrics['indexing_time']:.2f}s")
            print(f"  Average Query Time: {metrics['avg_query_time']:.2f}s (±{metrics['query_time_std']:.2f}s)")
        
        print("\nModel Performance:")
        for model_name, db_results in self.results["models"].items():
            print(f"\n{model_name}:")
            for db_name, weight_results in db_results.items():
                print(f"  {db_name}:")
                for weight, metrics in weight_results.items():
                    print(f"    {weight}:")
                    if metrics["success_count"] > 0:
                        print(f"      Success Rate: {metrics['success_count']}/{len(self.test_queries)}")
                        print(f"      Average Response Time: {metrics['avg_response_time']:.2f}s (±{metrics['response_time_std']:.2f}s)")
                        print(f"      Average Context Length: {metrics['avg_context_length']:.0f} chars")
                        print(f"      Average Source Count: {metrics['avg_source_count']:.1f}")
                    else:
                        print("      No successful queries")
                        if metrics.get("error_messages"):
                            print("      Error Messages:")
                            for msg in metrics["error_messages"]:
                                print(f"        - {msg}")
                    if metrics["error_count"] > 0:
                        print(f"      Total Errors: {metrics['error_count']}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG system configurations")
    parser.add_argument("--data_dir", type=str, default="data", help="Directory containing test documents")
    parser.add_argument("--num_queries", type=int, default=10, help="Number of test queries to generate")
    args = parser.parse_args()
    
    # Generate test queries
    test_queries = [
        "What are the main sorting algorithms covered in the course materials?",
        "Explain the concept of binary search trees.",
        "How does dynamic programming work?",
       
    ]
    
    # Initialize and run evaluation
    evaluator = RAGEvaluator(args.data_dir, test_queries)
    evaluator.run_evaluation()

if __name__ == "__main__":
    main() 