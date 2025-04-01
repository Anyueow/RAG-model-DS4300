from preprocessing.chunker import ChunkingConfig, TokenChunker, ChunkingPipeline
import yaml
from pathlib import Path
import pytest
from preprocessing.preprocessor import TextPreprocessor

def load_chunking_config(config_path: str, config_name: str = 'small_chunks') -> ChunkingConfig:
    """Load chunking configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        config_name: Name of the configuration to load
        
    Returns:
        ChunkingConfig instance
    """
    with open(config_path, 'r') as f:
        configs = yaml.safe_load(f)
    
    if config_name not in configs:
        raise ValueError(f"Configuration '{config_name}' not found in {config_path}")
    
    return ChunkingConfig(**configs[config_name])



def main():
    config_path = Path(__file__).parent / 'config' / 'chunking_config.yaml'


    
    # Example document
    document = {
        
        "text": "Amazon Web Services (AWS) is a comprehensive cloud computing platform offered by Amazon."
                "It provides a mix of infrastructure as a service (IaaS), platform as a service (PaaS), and packaged "
                "software as a service (SaaS) offerings. AWS services can provide organizations with compute power, database storage, content delivery,"
                 " and other functionality to help businesses scale and grow. One of the core AWS services is Amazon Elastic Compute Cloud (EC2), which allows users to rent virtual computers to run their own computer applications. EC2 offers scalable computing capacity in the cloud, eliminating the need to invest in hardware up front. Another fundamental service is Amazon Simple Storage Service (S3), which provides object storage through a web service interface. S3 is designed to deliver 99.999999999% durability and stores data for millions of applications used by market leaders in every industry. AWS also offers Amazon Relational Database Service (RDS), which simplifies the setup, operation, and scaling of relational databases in the cloud. RDS provides cost-efficient and resizable capacity while automating time-consuming administration tasks such as hardware provisioning, database setup, patching, and backups. For serverless computing, AWS Lambda lets you run code without provisioning or managing servers. You pay only for the compute time you consume, making it highly efficient for event-driven applications. AWS has a global infrastructure with data centers in multiple geographic regions worldwide. This allows customers to deploy applications closer to their end-users for lower latency and better performance. Security in AWS is managed through shared responsibility, where AWS manages security of the cloud while customers are responsible for security in the cloud. The platform provides numerous security features including encryption, identity and access management (IAM), and monitoring tools like Amazon GuardDuty. AWS continues to innovate rapidly, releasing new services and features regularly to help businesses leverage cutting-edge technologies like machine learning, artificial intelligence, and the Internet of Things (IoT). Many enterprises, from startups to large corporations, use AWS to reduce costs, become more agile, and innovate faster. The AWS free tier allows new users to gain hands-on experience with many AWS services at no charge for the first year. As cloud computing adoption grows, AWS remains the market leader, offering the most comprehensive and widely adopted cloud platform available today.",
        'file_path': 'example.txt',
        'file_type': 'txt'
    }
    
    configs = {
        'balanced': load_chunking_config(config_path, 'balanced'),
        'small_chunks': load_chunking_config(config_path, 'small_chunks'),
        'large_chunks': load_chunking_config(config_path, 'large_chunks')
    }
    
    # Process document with each configuration
    for config_name, config in configs.items():
        print(f"\nProcessing with {config_name} configuration:")
        print(f"Chunk size: {config.chunk_size}, Overlap: {config.overlap}")
        
        chunker = TokenChunker(config)
        pipeline = ChunkingPipeline(chunker)
        
        chunks = pipeline.process_documents([document])
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i + 1}:")
            print(f"Text: {chunk['text']}")
            print(f"ID: {chunk['chunk_id']}")

def test_token_chunker():
    """Test token-based chunking."""
    config = ChunkingConfig(chunk_size=3, overlap=1)
    chunker = TokenChunker(config)
    
    text = "This is a test sentence that will be chunked."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 4  # Should create 4 chunks with size 3 and overlap 1
    assert chunks[0] == "This is a"
    assert chunks[1] == "a test sentence"
    assert chunks[2] == "sentence that will"
    assert chunks[3] == "will be chunked."

def test_sentence_chunker():
    """Test sentence-based chunking."""
    config = ChunkingConfig(chunk_size=2, overlap=1)
    chunker = SentenceChunker(config)
    
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = chunker.chunk_text(text)
    
    assert len(chunks) == 3  # Should create 3 chunks with size 2 and overlap 1
    assert chunks[0] == "First sentence. Second sentence"
    assert chunks[1] == "Second sentence. Third sentence"
    assert chunks[2] == "Third sentence. Fourth sentence"

def test_chunking_pipeline_with_preprocessing():
    """Test the full pipeline with preprocessing."""
    # Create preprocessor
    preprocessor = TextPreprocessor(remove_stopwords=True)
    
    # Create chunker
    config = ChunkingConfig(chunk_size=3, overlap=1)
    chunker = TokenChunker(config)
    
    # Create pipeline
    pipeline = ChunkingPipeline(chunker, preprocessor)
    
    # Test document
    document = {
        'text': "This IS a TEST sentence that WILL be CHUNKED!",
        'file_path': 'test.txt',
        'file_type': 'txt'
    }
    
    # Process document
    chunks = pipeline.process_documents([document])
    
    # Verify preprocessing and chunking
    assert len(chunks) > 0
    assert all(isinstance(chunk['text'], str) for chunk in chunks)
    assert all(chunk['document_id'] == 'test.txt' for chunk in chunks)
    assert all(chunk['file_type'] == 'txt' for chunk in chunks)
    
    # Verify preprocessing worked (lowercase)
    assert all(chunk['text'].islower() for chunk in chunks)

def test_chunking_pipeline_with_yaml_config():
    """Test pipeline with YAML configuration."""
    # Load config from YAML
    config_path = 'preprocessing/config/chunking_config.yaml'
    config = ChunkingConfig.from_yaml(config_path)
    
    # Create components
    preprocessor = TextPreprocessor()
    chunker = TokenChunker(config)
    pipeline = ChunkingPipeline(chunker, preprocessor)
    
    # Test document
    document = {
        'text': "This is a test document that will be processed.",
        'file_path': 'test.txt',
        'file_type': 'txt'
    }
    
    # Process document
    chunks = pipeline.process_documents([document])
    
    # Verify processing
    assert len(chunks) > 0
    assert all(isinstance(chunk['text'], str) for chunk in chunks)
    assert all(chunk['document_id'] == 'test.txt' for chunk in chunks)
    assert all(chunk['file_type'] == 'txt' for chunk in chunks)

def test_chunking_pipeline_with_large_text():
    """Test pipeline with a larger text document."""
    # Create components
    preprocessor = TextPreprocessor()
    config = ChunkingConfig(chunk_size=100, overlap=20)
    chunker = TokenChunker(config)
    pipeline = ChunkingPipeline(chunker, preprocessor)
    
    # Create a larger test document
    text = " ".join(["This is a test sentence." for _ in range(50)])
    document = {
        'text': text,
        'file_path': 'large_test.txt',
        'file_type': 'txt'
    }
    
    # Process document
    chunks = pipeline.process_documents([document])
    
    # Verify processing
    assert len(chunks) > 1  # Should create multiple chunks
    assert all(len(chunk['text'].split()) <= 100 for chunk in chunks)  # Check chunk size
    assert all(chunk['document_id'] == 'large_test.txt' for chunk in chunks)

if __name__ == '__main__':
    main() 