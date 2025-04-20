"""Vector store module for document embeddings."""

import os
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

class VectorStore:
    """Vector store for document embeddings using ChromaDB."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config settings."""
        self.config = config
        self.persist_directory = config["vector_db"]["persist_directory"]
        self.collection_name = config["vector_db"]["collection_name"]
        self.embedding_model = OpenAIEmbeddings(
            model=config["vector_db"]["embedding_model"],
        )
    
    def create_or_load(self, documents: Optional[List[Document]] = None):
        """Create a new vector store or load existing one."""
        if documents:
            print(f"Creating vector store at {self.persist_directory}...")
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.persist_directory), exist_ok=True)
            
            # Create and persist the vector store
            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
            )
            vector_store.persist()
            print(f"Created vector store with {len(documents)} documents.")
            return vector_store
        else:
            print(f"Loading existing vector store from {self.persist_directory}...")
            # Load existing vector store
            vector_store = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_model,
                collection_name=self.collection_name,
            )
            print(f"Loaded vector store with {vector_store._collection.count()} documents.")
            return vector_store
