"""Retriever module for the RAG system."""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import Chroma

class Retriever:
    """Retrieves relevant documents from vector store."""
    
    def __init__(self, vector_store: Chroma, config: Dict[str, Any]):
        """Initialize with vector store and config."""
        self.vector_store = vector_store
        self.config = config
        self.top_k = config["retrieval"]["top_k"]
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        print(f"Retrieving documents for query: {query}")
        documents = self.vector_store.similarity_search(query, k=self.top_k)
        print(f"Retrieved {len(documents)} documents.")
        return documents
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores."""
        print(f"Retrieving documents with scores for query: {query}")
        documents = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        print(f"Retrieved {len(documents)} documents with scores.")
        return documents
