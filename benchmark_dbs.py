import time
import psutil
import numpy as np
from typing import Dict, List
import matplotlib.pyplot as plt
from pathlib import Path
import json

from database.chroma_db import ChromaDB
from database.redis_db import RedisVectorDB
from database.qdrant_db import QdrantDB
from embeddings.multimodal_embedder import MultiModalEmbedder
from ingestion.multimodal_loader import MultiModalPDFLoader

class VectorDBBenchmark:
    """Benchmark different vector databases for RAG system."""
    
    def __init__(self, data_dir: str = "data/raw_notes"):
        """Initialize benchmark with data directory.
        
        Args:
            data_dir: Directory containing PDF documents
        """
        self.data_dir = data_dir
        self.embedder = MultiModalEmbedder()
        self.loader = MultiModalPDFLoader()
        
        # Initialize databases
        self.dbs = {
            "ChromaDB": ChromaDB(),
            "RedisDB": RedisVectorDB(),
            "QdrantDB": QdrantDB()
        }
        
        # Load documents
        self.documents = self.loader.load_directory(data_dir)
        
        # Prepare vectors
        self.text_vectors = []
        self.text_metadata = []
        self.image_vectors = []
        self.image_metadata = []
        
        for doc_idx, doc in enumerate(self.documents):
            # Process text
            text_embedding = self.embedder.embed_text(doc.text)
            self.text_vectors.append(text_embedding)
            self.text_metadata.append({
                'text': doc.text,
                'source': doc.metadata.get('source'),
                'page': doc.metadata.get('page'),
                'doc_idx': doc_idx,
                'type': 'text'
            })
            
            # Process images
            for img in doc.images:
                image_data = img['data']
                image_embedding = self.embedder.embed_image(image_data)
                self.image_vectors.append(image_embedding)
                self.image_metadata.append({
                    'image': image_data,
                    'bbox': img.get('bbox'),
                    'page': img.get('page'),
                    'source': doc.metadata.get('source'),
                    'doc_idx': doc_idx,
                    'type': 'image'
                })

    def measure_memory(self) -> float:
        """Measure current memory usage."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024  # Convert to MB

    def benchmark_ingestion(self) -> Dict[str, Dict[str, float]]:
        """Benchmark document ingestion for each database."""
        results = {}
        
        for db_name, db in self.dbs.items():
            print(f"\nBenchmarking {db_name} ingestion...")
            
            # Clear database
            db.clear()
            
            # Measure ingestion time and memory
            start_time = time.time()
            start_memory = self.measure_memory()
            
            # Add vectors
            db.add_vectors(self.text_vectors, self.text_metadata, modality="text")
            db.add_vectors(self.image_vectors, self.image_metadata, modality="image")
            
            end_time = time.time()
            end_memory = self.measure_memory()
            
            results[db_name] = {
                'ingestion_time': end_time - start_time,
                'memory_usage': end_memory - start_memory
            }
            
        return results

    def benchmark_queries(self, num_queries: int = 100) -> Dict[str, Dict[str, float]]:
        """Benchmark query performance for each database."""
        results = {}
        
        # Generate random query vectors
        text_queries = [
            self.text_vectors[np.random.randint(0, len(self.text_vectors))]
            for _ in range(num_queries)
        ]
        
        image_queries = [
            self.image_vectors[np.random.randint(0, len(self.image_vectors))]
            for _ in range(num_queries)
        ]
        
        for db_name, db in self.dbs.items():
            print(f"\nBenchmarking {db_name} queries...")
            query_times = []
            
            # Measure text query time
            for query in text_queries:
                start_time = time.time()
                db.search(query, k=5, modality="text")
                query_times.append(time.time() - start_time)
            
            # Measure image query time
            for query in image_queries:
                start_time = time.time()
                db.search(query, k=5, modality="image")
                query_times.append(time.time() - start_time)
            
            results[db_name] = {
                'avg_query_time': np.mean(query_times),
                'p95_query_time': np.percentile(query_times, 95),
                'p99_query_time': np.percentile(query_times, 99)
            }
            
        return results

    def plot_results(self, ingestion_results: Dict, query_results: Dict):
        """Plot benchmark results."""
        plt.figure(figsize=(15, 5))
        
        # Plot ingestion results
        plt.subplot(131)
        names = list(ingestion_results.keys())
        times = [r['ingestion_time'] for r in ingestion_results.values()]
        plt.bar(names, times)
        plt.title('Ingestion Time (s)')
        plt.xticks(rotation=45)
        
        # Plot memory usage
        plt.subplot(132)
        memory = [r['memory_usage'] for r in ingestion_results.values()]
        plt.bar(names, memory)
        plt.title('Memory Usage (MB)')
        plt.xticks(rotation=45)
        
        # Plot query times
        plt.subplot(133)
        query_times = [r['avg_query_time'] * 1000 for r in query_results.values()]  # Convert to ms
        plt.bar(names, query_times)
        plt.title('Average Query Time (ms)')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('benchmark_results.png')
        
        # Save results to JSON
        all_results = {
            'ingestion': ingestion_results,
            'queries': query_results
        }
        with open('benchmark_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)

def main():
    """Run benchmarks and display results."""
    import argparse
    parser = argparse.ArgumentParser(description='Benchmark vector databases for RAG system')
    parser.add_argument('--data-dir', type=str, default='data/raw_notes',
                      help='Directory containing PDF documents')
    parser.add_argument('--num-queries', type=int, default=100,
                      help='Number of queries to run for benchmarking')
    
    args = parser.parse_args()
    
    print("Starting vector database benchmarks...")
    benchmark = VectorDBBenchmark(args.data_dir)
    
    print("\nRunning ingestion benchmarks...")
    ingestion_results = benchmark.benchmark_ingestion()
    
    print("\nRunning query benchmarks...")
    query_results = benchmark.benchmark_queries(args.num_queries)
    
    print("\nGenerating plots and saving results...")
    benchmark.plot_results(ingestion_results, query_results)
    
    print("\nResults summary:")
    for db_name in ingestion_results:
        print(f"\n{db_name}:")
        print(f"  Ingestion time: {ingestion_results[db_name]['ingestion_time']:.2f}s")
        print(f"  Memory usage: {ingestion_results[db_name]['memory_usage']:.2f}MB")
        print(f"  Avg query time: {query_results[db_name]['avg_query_time']*1000:.2f}ms")
        print(f"  P95 query time: {query_results[db_name]['p95_query_time']*1000:.2f}ms")
        print(f"  P99 query time: {query_results[db_name]['p99_query_time']*1000:.2f}ms")

if __name__ == "__main__":
    main() 