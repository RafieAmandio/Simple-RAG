"""Script to ask questions to the RAG system."""

import os
import sys
import yaml
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.rag_generator import RAGGenerator

def main():
    """Main function to ask questions to the RAG system."""
    # Load configuration
    with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Load vector store
    vector_store = VectorStore(config).create_or_load()
    
    # Initialize retriever
    retriever = Retriever(vector_store, config)
    
    # Initialize generator
    generator = RAGGenerator(config)
    
    # Interactive question answering loop
    print("\n====== RAG Question-Answering System ======")
    print("Type 'exit' or 'quit' to end the session.\n")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ["exit", "quit"]:
            break
        
        # Retrieve relevant documents
        retrieved_documents = retriever.retrieve(query)
        
        # Generate answer
        answer = generator.generate(query, retrieved_documents)
        
        print("\nAnswer:")
        print(answer)
        
        print("\nSources:")
        for i, doc in enumerate(retrieved_documents, 1):
            source = doc.metadata.get("source", "Unknown")
            print(f"{i}. {source}")

if __name__ == "__main__":
    main()
