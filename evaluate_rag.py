import os
import json
import pandas as pd
from tabulate import tabulate
import argparse
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
        self.export_dir = Path("evaluation_analysis")
        self.export_dir.mkdir(exist_ok=True)
        self.results = []
        self.load_results()
        
    def load_results(self) -> None:
        """Load all test results from the results directory using parallel processing."""
        logger.info("Loading evaluation results...")
        json_files = list(self.results_dir.glob("**/*.json"))
        
        def load_single_file(file_path: Path) -> Dict:
            try:
                with open(file_path, 'r') as f:
                    result = json.load(f)
                    if 'error' not in result:
                        config_name = file_path.stem.replace('_results', '')
                        parts = config_name.split('_')
                        if len(parts) >= 4:
                            result['embedding_model'] = parts[0]
                            result['llm_model'] = parts[1]
                            result['vector_db'] = parts[2]
                            result['chunking_strategy'] = parts[3]
                        return result
            except json.JSONDecodeError:
                logger.error(f"Could not parse {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {str(e)}")
            return None
        
        with ThreadPoolExecutor(max_workers=min(32, len(json_files))) as executor:
            results = list(filter(None, executor.map(load_single_file, json_files)))
        
        self.results = results
        logger.info(f"Loaded {len(self.results)} valid results")
        
        # Export raw responses to CSV
        self._export_raw_responses()
    
    def _export_raw_responses(self) -> None:
        """Export all responses to a CSV file."""
        rows = []
        for result in self.results:
            if 'responses' not in result:
                continue
                
            for q_idx, (q_key, q_data) in enumerate(result['responses'].items()):
                rows.append({
                    'question_number': q_idx + 1,
                    'question': q_data['question'],
                    'response': q_data['response'],
                    'embedding_model': result['embedding_model'],
                    'llm_model': result['llm_model'],
                    'vector_db': result['vector_db'],
                    'chunking_strategy': result['chunking_strategy'],
                    'execution_time': q_data['execution_time'],
                    'memory_usage': q_data['memory_usage']
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(self.export_dir / 'raw_responses.csv', index=False)
        logger.info(f"Exported raw responses to {self.export_dir / 'raw_responses.csv'}")
    
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
    
    def analyze_performance_metrics(self) -> None:
        """Analyze and display performance metrics across different configurations."""
        # Prepare data for analysis
        rows = []
        for result in self.results:
            if 'responses' not in result:
                continue
                
            for q_data in result['responses'].values():
                rows.append({
                    'embedding_model': result['embedding_model'],
                    'llm_model': result['llm_model'],
                    'vector_db': result['vector_db'],
                    'chunking_strategy': result['chunking_strategy'],
                    'execution_time': q_data['execution_time'],
                    'memory_usage': q_data['memory_usage']
                })
        
        df = pd.DataFrame(rows)
        
        # Calculate average metrics for each configuration
        metrics_df = df.groupby(['embedding_model', 'vector_db', 'chunking_strategy']).agg({
            'execution_time': 'mean',
            'memory_usage': 'mean'
        }).reset_index()
        
        # Export metrics to CSV
        metrics_df.to_csv(self.export_dir / 'performance_metrics.csv', index=False)
        logger.info(f"Exported performance metrics to {self.export_dir / 'performance_metrics.csv'}")
        
        # Generate visualizations
        self._generate_performance_visualizations(metrics_df)
    
    def _generate_performance_visualizations(self, df: pd.DataFrame) -> None:
        """Generate Plotly visualizations for performance metrics."""
        # 1. By Chunking Strategy
        for chunking in df['chunking_strategy'].unique():
            chunk_df = df[df['chunking_strategy'] == chunking]
            
            # Memory usage
            fig = go.Figure()
            for db in chunk_df['vector_db'].unique():
                db_data = chunk_df[chunk_df['vector_db'] == db]
                fig.add_trace(go.Bar(
                    name=db,
                    x=db_data['embedding_model'],
                    y=db_data['memory_usage']
                ))
            
            fig.update_layout(
                title=f'Memory Usage by Embedding Model and Vector DB (Chunking: {chunking})',
                xaxis_title='Embedding Model',
                yaxis_title='Memory Usage (MB)',
                barmode='group',
                template='plotly_white'
            )
            fig.write_html(self.export_dir / f'memory_chunking_{chunking}.html')
            
            # Execution time
            fig = go.Figure()
            for db in chunk_df['vector_db'].unique():
                db_data = chunk_df[chunk_df['vector_db'] == db]
                fig.add_trace(go.Bar(
                    name=db,
                    x=db_data['embedding_model'],
                    y=db_data['execution_time']
                ))
            
            fig.update_layout(
                title=f'Execution Time by Embedding Model and Vector DB (Chunking: {chunking})',
                xaxis_title='Embedding Model',
                yaxis_title='Execution Time (s)',
                barmode='group',
                template='plotly_white'
            )
            fig.write_html(self.export_dir / f'time_chunking_{chunking}.html')
        
        # 2. By Embedding Model
        for model in df['embedding_model'].unique():
            model_df = df[df['embedding_model'] == model]
            
            # Memory usage
            fig = go.Figure()
            for db in model_df['vector_db'].unique():
                db_data = model_df[model_df['vector_db'] == db]
                fig.add_trace(go.Bar(
                    name=db,
                    x=db_data['chunking_strategy'],
                    y=db_data['memory_usage']
                ))
            
            fig.update_layout(
                title=f'Memory Usage by Chunking Strategy and Vector DB (Model: {model})',
                xaxis_title='Chunking Strategy',
                yaxis_title='Memory Usage (MB)',
                barmode='group',
                template='plotly_white'
            )
            fig.write_html(self.export_dir / f'memory_model_{model}.html')
            
            # Execution time
            fig = go.Figure()
            for db in model_df['vector_db'].unique():
                db_data = model_df[model_df['vector_db'] == db]
                fig.add_trace(go.Bar(
                    name=db,
                    x=db_data['chunking_strategy'],
                    y=db_data['execution_time']
                ))
            
            fig.update_layout(
                title=f'Execution Time by Chunking Strategy and Vector DB (Model: {model})',
                xaxis_title='Chunking Strategy',
                yaxis_title='Execution Time (s)',
                barmode='group',
                template='plotly_white'
            )
            fig.write_html(self.export_dir / f'time_model_{model}.html')
        
        # 3. By Vector DB
        for db in df['vector_db'].unique():
            db_df = df[df['vector_db'] == db]
            
            # Memory usage
            fig = go.Figure()
            for model in db_df['embedding_model'].unique():
                model_data = db_df[db_df['embedding_model'] == model]
                fig.add_trace(go.Bar(
                    name=model,
                    x=model_data['chunking_strategy'],
                    y=model_data['memory_usage']
                ))
            
            fig.update_layout(
                title=f'Memory Usage by Chunking Strategy and Embedding Model (DB: {db})',
                xaxis_title='Chunking Strategy',
                yaxis_title='Memory Usage (MB)',
                barmode='group',
                template='plotly_white'
            )
            fig.write_html(self.export_dir / f'memory_db_{db}.html')
            
            # Execution time
            fig = go.Figure()
            for model in db_df['embedding_model'].unique():
                model_data = db_df[db_df['embedding_model'] == model]
                fig.add_trace(go.Bar(
                    name=model,
                    x=model_data['chunking_strategy'],
                    y=model_data['execution_time']
                ))
            
            fig.update_layout(
                title=f'Execution Time by Chunking Strategy and Embedding Model (DB: {db})',
                xaxis_title='Chunking Strategy',
                yaxis_title='Execution Time (s)',
                barmode='group',
                template='plotly_white'
            )
            fig.write_html(self.export_dir / f'time_db_{db}.html')
        
        # 4. Create a summary dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Memory Usage by Vector DB',
                'Execution Time by Vector DB',
                'Memory Usage by Embedding Model',
                'Execution Time by Embedding Model'
            )
        )
        
        # Memory by Vector DB
        for db in df['vector_db'].unique():
            db_data = df[df['vector_db'] == db]
            fig.add_trace(
                go.Bar(name=db, x=db_data['chunking_strategy'], y=db_data['memory_usage']),
                row=1, col=1
            )
        
        # Time by Vector DB
        for db in df['vector_db'].unique():
            db_data = df[df['vector_db'] == db]
            fig.add_trace(
                go.Bar(name=db, x=db_data['chunking_strategy'], y=db_data['execution_time']),
                row=1, col=2
            )
        
        # Memory by Embedding Model
        for model in df['embedding_model'].unique():
            model_data = df[df['embedding_model'] == model]
            fig.add_trace(
                go.Bar(name=model, x=model_data['chunking_strategy'], y=model_data['memory_usage']),
                row=2, col=1
            )
        
        # Time by Embedding Model
        for model in df['embedding_model'].unique():
            model_data = df[df['embedding_model'] == model]
            fig.add_trace(
                go.Bar(name=model, x=model_data['chunking_strategy'], y=model_data['execution_time']),
                row=2, col=2
            )
        
        fig.update_layout(
            height=800,
            title_text="RAG System Performance Summary",
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Memory Usage (MB)", row=1, col=1)
        fig.update_yaxes(title_text="Execution Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Memory Usage (MB)", row=2, col=1)
        fig.update_yaxes(title_text="Execution Time (s)", row=2, col=2)
        
        fig.write_html(self.export_dir / 'performance_summary.html')
    
    def evaluate_responses(self) -> None:
        """Interactive tool to evaluate responses with performance monitoring."""
        logger.info("\n===== RAG Response Evaluator =====")
        
        while True:
            print("\nOptions:")
            print("1. Compare responses for a question")
            print("2. Analyze performance metrics")
            print("3. Exit")
            
            choice = input("\nEnter choice (1-3): ")
            
            if choice == '1':
                question_idx = int(input(f"Enter question number (1-4): ")) - 1
                self.compare_responses(question_idx)
            
            elif choice == '2':
                self.analyze_performance_metrics()
            
            elif choice == '3':
                break
            
            else:
                logger.warning("Invalid choice. Please try again.")

def main():
    parser = argparse.ArgumentParser(description='Evaluate RAG system performance')
    parser.add_argument('--results-dir', default='evaluation_results',
                      help='Directory containing evaluation results')
    args = parser.parse_args()
    
    evaluator = RAGEvaluator(args.results_dir)
    evaluator.evaluate_responses()

if __name__ == "__main__":
    main() 