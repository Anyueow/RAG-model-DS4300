import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import os

# Create output directory if it doesn't exist
os.makedirs('visualizations', exist_ok=True)

# Read the performance metrics
mistral_df = pd.read_csv('evaluation_analysis/mistral/performance_metrics.csv')
qwen_df = pd.read_csv('evaluation_analysis/qwen/performance_metrics.csv')

# Color scheme for consistent visualization
COLOR_SCHEME = {
    # Embedding Models
    'minilm': '#5551FF',  # Primary blue
    'mpnet': '#FF7262',   # Primary red
    'nomic': '#E4E3FF',   # Light blue
    
    # Vector Databases
    'chroma': '#5551FF',  # Primary blue
    'qdrant': '#FF7262',  # Primary red
    'redis': '#E4E3FF',   # Light blue
    
    # Chunking Strategies
    'small': '#5551FF',   # Primary blue
    'medium': '#FF7262',  # Primary red
    'large': '#E4E3FF',   # Light blue
}

# Additional color schemes for different visualizations
HEATMAP_COLORS = ['#FFFFFF', '#E4E3FF', '#5551FF']  # White to light blue to primary blue
DUAL_AXIS_COLORS = {
    'time': '#5551FF',    # Primary blue
    'memory': '#FF7262'   # Primary red
}

def create_document_processing_flow():
    # Define nodes and their positions
    nodes = [
        {'id': 'raw_docs', 'label': 'Raw Documents', 'x': 0, 'y': 0},
        {'id': 'preprocessing', 'label': 'Text Preprocessing', 'x': 1, 'y': 0},
        {'id': 'chunking', 'label': 'Text Chunking\n(256/512/1024 tokens)', 'x': 2, 'y': 0},
        {'id': 'embedding', 'label': 'Embedding Generation\n(MiniLM/MPNet/Nomic)', 'x': 3, 'y': 0},
        {'id': 'vector_db', 'label': 'Vector Storage\n(Redis/Qdrant/Chroma)', 'x': 4, 'y': 0},
        {'id': 'query', 'label': 'User Query', 'x': 0, 'y': 1},
        {'id': 'query_embedding', 'label': 'Query Embedding', 'x': 1, 'y': 1},
        {'id': 'retrieval', 'label': 'Similarity Search', 'x': 2, 'y': 1},
        {'id': 'context', 'label': 'Context Assembly', 'x': 3, 'y': 1},
        {'id': 'llm', 'label': 'LLM Generation\n(Mistral/Qwen)', 'x': 4, 'y': 1},
        {'id': 'response', 'label': 'Final Response', 'x': 5, 'y': 1}
    ]
    
    # Define edges (connections between nodes)
    edges = [
        {'from': 'raw_docs', 'to': 'preprocessing'},
        {'from': 'preprocessing', 'to': 'chunking'},
        {'from': 'chunking', 'to': 'embedding'},
        {'from': 'embedding', 'to': 'vector_db'},
        {'from': 'query', 'to': 'query_embedding'},
        {'from': 'query_embedding', 'to': 'retrieval'},
        {'from': 'vector_db', 'to': 'retrieval'},
        {'from': 'retrieval', 'to': 'context'},
        {'from': 'context', 'to': 'llm'},
        {'from': 'llm', 'to': 'response'}
    ]
    
    # Create the figure
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        fig.add_trace(go.Scatter(
            x=[nodes[edge['from']]['x'], nodes[edge['to']]['x']],
            y=[nodes[edge['from']]['y'], nodes[edge['to']]['y']],
            mode='lines',
            line=dict(color='#5551FF', width=2),
            hoverinfo='none'
        ))
    
    # Add nodes
    for node in nodes:
        fig.add_trace(go.Scatter(
            x=[node['x']],
            y=[node['y']],
            mode='markers+text',
            marker=dict(
                size=40,
                color='#E4E3FF',
                line=dict(color='#5551FF', width=2)
            ),
            text=node['label'],
            textposition='middle center',
            hoverinfo='none'
        ))
    
    # Update layout
    fig.update_layout(
        title='Document Processing Pipeline',
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        margin=dict(l=20, r=20, t=40, b=20),
        width=1200,
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    save_figure(fig, 'document_processing_flow')

def save_figure(fig, filename):
    """Helper function to save figure as PNG with consistent settings"""
    fig.update_layout(
        width=1200,
        height=800,
        template='plotly_white',
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(size=14)
    )
    fig.write_image(f'visualizations/{filename}.png')

def create_embedding_model_charts(df, title_prefix):
    # Calculate averages for each embedding model
    model_metrics = df.groupby('embedding_model').agg({
        'execution_time': 'mean',
        'memory_usage': 'mean'
    }).reset_index()
    
    # Execution Time Chart
    fig_time = go.Figure()
    for model in model_metrics['embedding_model'].unique():
        model_data = model_metrics[model_metrics['embedding_model'] == model]
        fig_time.add_trace(go.Bar(
            name=model,
            x=[model],
            y=model_data['execution_time'],
            marker_color=COLOR_SCHEME[model]
        ))
    
    fig_time.update_layout(
        title=f'{title_prefix} - Average Execution Time by Embedding Model',
        xaxis_title='Embedding Model',
        yaxis_title='Execution Time (s)',
        showlegend=False
    )
    save_figure(fig_time, f'embedding_model_time_{title_prefix.lower()}')
    
    # Memory Usage Chart
    fig_mem = go.Figure()
    for model in model_metrics['embedding_model'].unique():
        model_data = model_metrics[model_metrics['embedding_model'] == model]
        fig_mem.add_trace(go.Bar(
            name=model,
            x=[model],
            y=model_data['memory_usage'],
            marker_color=COLOR_SCHEME[model]
        ))
    
    fig_mem.update_layout(
        title=f'{title_prefix} - Average Memory Usage by Embedding Model',
        xaxis_title='Embedding Model',
        yaxis_title='Memory Usage (MB)',
        showlegend=False
    )
    save_figure(fig_mem, f'embedding_model_memory_{title_prefix.lower()}')

def create_scatter_plot(df, title_prefix):
    fig = px.scatter(df,
                    x='execution_time',
                    y='memory_usage',
                    color='embedding_model',
                    hover_data=['vector_db', 'chunking_strategy'],
                    color_discrete_map=COLOR_SCHEME)
    
    fig.update_layout(
        title=f'{title_prefix} - Execution Time vs Memory Usage'
    )
    save_figure(fig, f'scatter_plot_{title_prefix.lower()}')

def create_vector_db_comparison(df, title_prefix):
    # Calculate averages for each vector DB
    db_metrics = df.groupby('vector_db').agg({
        'execution_time': 'mean',
        'memory_usage': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            name='Execution Time',
            x=db_metrics['vector_db'],
            y=db_metrics['execution_time'],
            marker_color=DUAL_AXIS_COLORS['time']
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            name='Memory Usage',
            x=db_metrics['vector_db'],
            y=db_metrics['memory_usage'],
            marker_color=DUAL_AXIS_COLORS['memory']
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'{title_prefix} - Vector Database Performance Comparison'
    )
    
    fig.update_xaxes(title_text="Vector Database")
    fig.update_yaxes(title_text="Execution Time (s)", secondary_y=False)
    fig.update_yaxes(title_text="Memory Usage (MB)", secondary_y=True)
    
    save_figure(fig, f'vector_db_comparison_{title_prefix.lower()}')

def create_chunking_strategy_chart(df, title_prefix):
    # Calculate averages for each chunking strategy
    chunk_metrics = df.groupby('chunking_strategy').agg({
        'execution_time': 'mean',
        'memory_usage': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            name='Execution Time',
            x=chunk_metrics['chunking_strategy'],
            y=chunk_metrics['execution_time'],
            marker_color=DUAL_AXIS_COLORS['time']
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            name='Memory Usage',
            x=chunk_metrics['chunking_strategy'],
            y=chunk_metrics['memory_usage'],
            marker_color=DUAL_AXIS_COLORS['memory']
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title=f'{title_prefix} - Chunking Strategy Performance'
    )
    
    fig.update_xaxes(title_text="Chunking Strategy")
    fig.update_yaxes(title_text="Execution Time (s)", secondary_y=False)
    fig.update_yaxes(title_text="Memory Usage (MB)", secondary_y=True)
    
    save_figure(fig, f'chunking_strategy_{title_prefix.lower()}')

def create_heatmaps(df, title_prefix):
    # Prepare data for heatmaps
    pivot_time = df.pivot_table(
        values='execution_time',
        index='embedding_model',
        columns=['vector_db', 'chunking_strategy'],
        aggfunc='mean'
    )
    
    pivot_memory = df.pivot_table(
        values='memory_usage',
        index='embedding_model',
        columns=['vector_db', 'chunking_strategy'],
        aggfunc='mean'
    )
    
    # Execution Time Heatmap
    fig_time = go.Figure(data=go.Heatmap(
        z=pivot_time.values,
        x=pivot_time.columns,
        y=pivot_time.index,
        colorscale=HEATMAP_COLORS
    ))
    
    fig_time.update_layout(
        title=f'{title_prefix} - Execution Time Heatmap',
        xaxis_title='Vector DB + Chunking Strategy',
        yaxis_title='Embedding Model'
    )
    save_figure(fig_time, f'heatmap_time_{title_prefix.lower()}')
    
    # Memory Usage Heatmap
    fig_memory = go.Figure(data=go.Heatmap(
        z=pivot_memory.values,
        x=pivot_memory.columns,
        y=pivot_memory.index,
        colorscale=HEATMAP_COLORS
    ))
    
    fig_memory.update_layout(
        title=f'{title_prefix} - Memory Usage Heatmap',
        xaxis_title='Vector DB + Chunking Strategy',
        yaxis_title='Embedding Model'
    )
    save_figure(fig_memory, f'heatmap_memory_{title_prefix.lower()}')

def create_llm_comparison():
    # Combine data from both LLMs
    mistral_df['llm'] = 'Mistral'
    qwen_df['llm'] = 'Qwen'
    combined_df = pd.concat([mistral_df, qwen_df])
    
    # Overall performance comparison
    llm_metrics = combined_df.groupby('llm').agg({
        'execution_time': 'mean',
        'memory_usage': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Bar(
            name='Execution Time',
            x=llm_metrics['llm'],
            y=llm_metrics['execution_time'],
            marker_color=DUAL_AXIS_COLORS['time']
        ),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Bar(
            name='Memory Usage',
            x=llm_metrics['llm'],
            y=llm_metrics['memory_usage'],
            marker_color=DUAL_AXIS_COLORS['memory']
        ),
        secondary_y=True
    )
    
    fig.update_layout(
        title='LLM Performance Comparison'
    )
    
    fig.update_xaxes(title_text="LLM Model")
    fig.update_yaxes(title_text="Execution Time (s)", secondary_y=False)
    fig.update_yaxes(title_text="Memory Usage (MB)", secondary_y=True)
    
    save_figure(fig, 'llm_comparison')

def main():
    # Generate document processing flow chart
    create_document_processing_flow()
    
    # Generate charts for Mistral
    create_embedding_model_charts(mistral_df, 'Mistral')
    create_scatter_plot(mistral_df, 'Mistral')
    create_vector_db_comparison(mistral_df, 'Mistral')
    create_chunking_strategy_chart(mistral_df, 'Mistral')
    create_heatmaps(mistral_df, 'Mistral')
    
    # Generate charts for Qwen
    create_embedding_model_charts(qwen_df, 'Qwen')
    create_scatter_plot(qwen_df, 'Qwen')
    create_vector_db_comparison(qwen_df, 'Qwen')
    create_chunking_strategy_chart(qwen_df, 'Qwen')
    create_heatmaps(qwen_df, 'Qwen')
    
    # Generate LLM comparison
    create_llm_comparison()

if __name__ == "__main__":
    main() 