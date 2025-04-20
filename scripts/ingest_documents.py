"""Script to ingest documents into the vector store."""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.document_loader import DocumentLoader
from src.ingestion.vector_store import VectorStore

def main():
    """Main function to ingest documents."""
    # Load configuration
    with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize document loader
    loader = DocumentLoader(config)
    
    # Load documents from the raw data directory
    data_dir = os.path.join(project_root, "data", "raw")
    documents = loader.load_from_directory(data_dir)
    
    # Split documents into chunks
    chunks = loader.split_documents(documents)
    
    # Initialize and create vector store
    vector_store = VectorStore(config)
    vector_store.create_or_load(chunks)
    
    print("Document ingestion completed successfully!")

if __name__ == "__main__":
    main()
