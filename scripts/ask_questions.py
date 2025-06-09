"""Script to ask questions to the RAG system with chat memory integration."""

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
from src.memory.chat_memory import ChatMemory

def main():
    """Main function to ask questions to the RAG system."""
    # Load configuration
    with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize chat memory
    chat_memory = ChatMemory(config)
    
    # Load vector store
    vector_store = VectorStore(config).create_or_load()
    
    # Initialize retriever
    retriever = Retriever(vector_store, config)
    
    # Initialize generator with chat memory
    generator = RAGGenerator(config, chat_memory)
    
    # Interactive question answering loop
    print("\n====== RAG Question-Answering System with Chat Memory ======")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'clear' to clear chat history.")
    print("Type 'history' to view chat history.\n")
    
    while True:
        query = input("\nEnter your question: ")
        
        if query.lower() in ["exit", "quit"]:
            break
        
        if query.lower() == "clear":
            chat_memory.clear_history()
            continue
        
        if query.lower() == "history":
            history = chat_memory.get_history()
            print("\n===== Chat History =====")
            for i, msg in enumerate(history, 1):
                role = msg["role"].capitalize()
                content = msg["content"]
                print(f"{i}. {role}: {content[:100]}..." if len(content) > 100 else f"{i}. {role}: {content}")
            print("=======================")
            continue
        
        # Retrieve relevant documents
        retrieved_documents = retriever.retrieve(query)
        
        # Generate answer using retrieved documents and chat history
        answer = generator.generate(query, retrieved_documents)
        
        print("\nAnswer:")
        print(answer)
        
        print("\nSources:")
        for i, doc in enumerate(retrieved_documents, 1):
            source = doc.metadata.get("source", "Unknown")
            print(f"{i}. {source}")

if __name__ == "__main__":
    main()