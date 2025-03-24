import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    import sys
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def create_conda_environment():
    """ Activate a conda environment."""
    print("conda environment...")
    subprocess.run(["conda", "activate", "ds-4300_project"])
    
    print("\nConda environment created. Activate it with:")
    print("conda activate rag-model")

def install_dependencies():
    """Install required dependencies using conda and pip."""
    print("\nInstalling dependencies...")
    
    # Install packages available in conda
    conda_packages = [
        "numpy",
        "pandas",
        "torch",
        "transformers",
        "sentence-transformers",
        "chromadb",
        "redis",
        "pytest"
    ]
    
    subprocess.run(["conda", "install", "-n", "rag-model", "-c", "conda-forge"] + conda_packages + ["-y"])
    
    # Install remaining packages via pip
    pip_packages = [
        "pypdf",
        "faiss-cpu",
        "nltk",
        "ollama"
    ]
    
    subprocess.run(["conda", "run", "-n", "rag-model", "pip", "install"] + pip_packages)

def setup_directories():
    """Create necessary directories."""
    print("\nSetting up directories...")
    directories = [
        "data/raw_notes",
        "chroma_db",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def check_ollama():
    """Check if Ollama is installed and running."""
    try:
        import ollama
        print("\nChecking Ollama installation...")
        ollama.list()
        print("Ollama is installed and running")
    except ImportError:
        print("\nError: Ollama is not installed. Please install it from https://ollama.ai/")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: Ollama is not running. Please start the Ollama service: {str(e)}")
        sys.exit(1)

def pull_llama_model():
    """Pull the Llama 2 model for Ollama."""
    print("\nPulling Llama 2 model...")
    subprocess.run(["ollama", "pull", "llama2"])

def main():
    """Main setup function."""
    print("Setting up RAG Model for Course Notes...")
    
    # Check Python version
    check_python_version()
    
    # Create conda environment
    create_conda_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Setup directories
    setup_directories()
    
    # Check Ollama installation
    check_ollama()
    
    # Pull Llama 2 model
    pull_llama_model()
    
    print("\nSetup completed successfully!")
    print("\nTo get started:")
    print("1. Activate the conda environment: conda activate rag-model")
    print("2. Place your course notes in the data/raw_notes directory")
    print("3. Run example.py to test the system")
    print("4. Run tests/test_rag.py to verify functionality")

if __name__ == "__main__":
    main() 