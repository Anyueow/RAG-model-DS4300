import os
import json
import pandas as pd
from tabulate import tabulate
import argparse
import matplotlib.pyplot as plt
import numpy as np
import time
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Any, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('rag_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RAGEvaluator:
    
    def __init__(self, results_dir: str = "evaluation_results"):
        """Initialize the evaluator with path to results directory."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.results = []
        self.load_results()
        
    def load_results(self) -> None:
        """Load all test results from the results directory using parallel processing."""
        logger.info("Loading evaluation results...")
        json_files = list(self.results_dir.glob("*.json"))
        
        def load_single_file(file_path: Path) -> Dict:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    if 'error' not in result:
                        return result
            except json.JSONDecodeError:
                logger.error(f"Could not parse {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
            return None
        
        # Use ThreadPoolExecutor for parallel file loading
        with ThreadPoolExecutor(max_workers=min(32, len(json_files))) as executor:
            results = list(filter(None, executor.map(load_single_file, json_files)))
        
        self.results = results
        logger.info(f"Loaded {len(self.results)} valid results")
    
    def compare_responses(self, question_idx: int = 0) -> None:
        """Compare responses from different configurations for a specific question."""
        question_key = f"question_{question_idx+1}"
        
        # Extract responses for the specific question using list comprehension
        responses = [
            {
                'config': (
                    result['chunking_strategy'],
                    result['embedding_model'],
                    result['vector_db'],
                    result['llm_model']
                ),
                'question': result['responses'][question_key]['question'],
                'response': result['responses'][question_key]['response'],
                'time': result['responses'][question_key]['execution_time'],
                'memory': result['responses'][question_key]['memory_usage']
            }
            for result in self.results
            if 'responses' in result and question_key in result['responses']
        ]
        
        if responses:
            logger.info(f"\n===== Responses for Question: {responses[0]['question']} =====\n")
            
            for i, resp in enumerate(responses):
                config_str = ' + '.join(resp['config'])
                logger.info(f"Configuration {i+1}: {config_str}")
                logger.info(f"Time: {resp['time']:.2f}s | Memory: {resp['memory']:.2f}MB")
                logger.info("Response:")
                logger.info("-" * 80)
                logger.info(resp['response'])
                logger.info("=" * 80)
                logger.info("")
        else:
            logger.warning(f"No responses found for question {question_idx+1}")
    
    def evaluate_responses(self) -> None:
        """Interactive tool to evaluate responses with performance monitoring."""
        logger.info("\n===== RAG Response Evaluator =====")
        
        # Calculate total questions efficiently
        max_questions = max(
            (len(result['responses']) for result in self.results if 'responses' in result),
            default=0
        )
        
        if max_questions == 0:
            logger.warning("No responses found in the results.")
            return
        
        ratings = {}
        
        while True:
            print("\nOptions:")
            print("1. Compare responses for a question")
            print("2. Rate responses")
            print("3. Show current ratings")
            print("4. Export ratings")
            print("5. Exit")
            
            choice = input("\nEnter choice (1-5): ")
            
            if choice == '1':
                question_idx = int(input(f"Enter question number (1-{max_questions}): ")) - 1
                self.compare_responses(question_idx)
            
            elif choice == '2':
                question_idx = int(input(f"Enter question number (1-{max_questions}): ")) - 1
                self.compare_responses(question_idx)
                
                # Display configurations with performance metrics
                for i, result in enumerate(self.results):
                    config = (
                        result['chunking_strategy'],
                        result['embedding_model'],
                        result['vector_db'],
                        result['llm_model']
                    )
                    config_str = ' + '.join(config)
                    avg_time = np.mean([
                        resp['execution_time'] 
                        for resp in result['responses'].values()
                    ])
                    print(f"{i+1}. {config_str} (Avg Time: {avg_time:.2f}s)")
                
                config_idx = int(input("\nEnter configuration number to rate: ")) - 1
                if 0 <= config_idx < len(self.results):
                    config = (
                        self.results[config_idx]['chunking_strategy'],
                        self.results[config_idx]['embedding_model'],
                        self.results[config_idx]['vector_db'],
                        self.results[config_idx]['llm_model']
                    )
                    config_str = ' + '.join(config)
                    
                    # Get ratings with validation
                    while True:
                        try:
                            relevance = int(input("Rate relevance (1-10): "))
                            completeness = int(input("Rate completeness (1-10): "))
                            coherence = int(input("Rate coherence/fluency (1-10): "))
                            accuracy = int(input("Rate accuracy (1-10): "))
                            
                            if all(1 <= x <= 10 for x in [relevance, completeness, coherence, accuracy]):
                                break
                            print("All ratings must be between 1 and 10")
                        except ValueError:
                            print("Please enter valid numbers")
                    
                    if config_str not in ratings:
                        ratings[config_str] = {'questions': {}}
                    
                    question_key = f"question_{question_idx+1}"
                    ratings[config_str]['questions'][question_key] = {
                        'relevance': relevance,
                        'completeness': completeness,
                        'coherence': coherence,
                        'accuracy': accuracy,
                        'overall': (relevance + completeness + coherence + accuracy) / 4
                    }
                    
                    # Calculate average across all questions efficiently
                    overall_scores = [v['overall'] for v in ratings[config_str]['questions'].values()]
                    ratings[config_str]['overall_score'] = sum(overall_scores) / len(overall_scores)
                    
                    logger.info(f"\nRating saved for {config_str}")
            
            elif choice == '3':
                if not ratings:
                    logger.warning("No ratings yet.")
                    continue
                
                logger.info("\n===== Current Ratings =====")
                ratings_df = pd.DataFrame([
                    {'Configuration': config, 'Overall Score': data['overall_score']}
                    for config, data in ratings.items()
                ])
                print(tabulate(
                    ratings_df.sort_values('Overall Score', ascending=False),
                    headers='keys',
                    tablefmt='pipe',
                    showindex=False
                ))
            
            elif choice == '4':
                if not ratings:
                    logger.warning("No ratings to export.")
                    continue
                
                filename = input("Enter filename to save ratings (default: ratings.json): ") or "ratings.json"
                base_filename = os.path.splitext(filename)[0]
                
                # Save ratings
                with open(filename, 'w') as f:
                    json.dump(ratings, f, indent=2)
                logger.info(f"Ratings saved to {filename}")
                
                # Generate visualizations
                self._generate_rating_visualizations(ratings, base_filename)
            
            elif choice == '5':
                break
            
            else:
                logger.warning("Invalid choice. Please try again.")
    
    def _generate_rating_visualizations(self, ratings: Dict, base_filename: str) -> None:
        """Generate optimized visualizations for the ratings."""
        # Overall scores
        configs = list(ratings.keys())
        scores = [ratings[config]['overall_score'] for config in configs]
        
        plt.figure(figsize=(12, 6))
        plt.barh(configs, scores, color='skyblue')
        plt.xlabel('Average Score')
        plt.ylabel('Configuration')
        plt.title('Overall RAG Configuration Scores')
        plt.xlim(0, 10)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(f"{base_filename}_overall_scores.png")
        plt.close()
        
        # Radar chart for top configurations
        top_configs = sorted(ratings.items(), key=lambda x: x[1]['overall_score'], reverse=True)[:3]
        
        if top_configs:
            # Sample question for radar chart
            sample_question = list(top_configs[0][1]['questions'].keys())[0]
            
            categories = ['Relevance', 'Completeness', 'Coherence', 'Accuracy']
            N = len(categories)
            
            angles = [n / float(N) * 2 * np.pi for n in range(N)]
            angles += angles[:1]  # Close the loop
            
            plt.figure(figsize=(8, 8))
            ax = plt.subplot(111, polar=True)
            
            for config, data in top_configs:
                values = [
                    data['questions'][sample_question]['relevance'],
                    data['questions'][sample_question]['completeness'],
                    data['questions'][sample_question]['coherence'],
                    data['questions'][sample_question]['accuracy']
                ]
                values += values[:1]  # Close the loop
                
                ax.plot(angles, values, linewidth=2, label=config)
                ax.fill(angles, values, alpha=0.25)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(categories)
            plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
            plt.title('Top 3 Configurations - Performance Metrics')
            plt.tight_layout()
            plt.savefig(f"{base_filename}_radar_chart.png")
            plt.close()

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system performance')
    parser.add_argument('--results-dir', default='evaluation_results',
                      help='Directory containing evaluation results')
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(args.results_dir)
    evaluator.evaluate_responses()

if __name__ == "__main__":
    main() 