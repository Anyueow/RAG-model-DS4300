import os
import subprocess
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    import sys
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)

def create_virtual_environment():
    """Create and activate a virtual environment."""
    print("Creating virtual environment...")
    subprocess.run(["python", "-m", "venv", "venv"])
    
    # Determine the activation script based on OS
    if os.name == 'nt':  # Windows
        activate_script = "venv\\Scripts\\activate"
    else:  # Unix/Linux/MacOS
        activate_script = "source venv/bin/activate"
    
    print(f"\nVirtual environment created. Activate it with:\n{activate_script}")

def install_dependencies():
    """Install required dependencies."""
    print("\nInstalling dependencies...")
    subprocess.run(["pip", "install", "-r", "requirements.txt"])

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
    
    # Create virtual environment
    create_virtual_environment()
    
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
    print("1. Activate the virtual environment")
    print("2. Place your course notes in the data/raw_notes directory")
    print("3. Run example.py to test the system")
    print("4. Run tests/test_rag.py to verify functionality")

if __name__ == "__main__":
    main() 