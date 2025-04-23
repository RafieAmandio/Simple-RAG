"""Retriever module for the RAG system."""

from typing import List, Dict, Any
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
from collections import defaultdict
import re

class Retriever:
    """Retrieves relevant documents from vector store."""
    
    def __init__(self, vector_store: Chroma, config: Dict[str, Any]):
        """Initialize with vector store and config."""
        self.vector_store = vector_store
        self.config = config
        self.top_k = config["retrieval"]["top_k"]
        self.embeddings = OpenAIEmbeddings()
        # Use larger chunk size for better context retention
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["document_processing"]["chunk_size"] * 2,  # Double the chunk size
            chunk_overlap=config["document_processing"]["chunk_overlap"] * 2,  # Double the overlap
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def _clean_document(self, content: str) -> str:
        """Clean document content."""
        # Remove formatting characters
        content = content.replace(".............................................................................................", "")
        content = content.replace("............................................................................................", "")
        # Remove question numbers and formatting
        content = " ".join(content.split())
        # Remove page numbers and headers
        content = " ".join([line for line in content.split("\n") if not line.strip().startswith("Bab")])
        # Remove citations and references
        content = re.sub(r'\[\d+\]', '', content)
        return content.strip()
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from the query."""
        # Remove common words and keep important terms
        common_words = {'what', 'is', 'the', 'of', 'in', 'to', 'and', 'a', 'an', 'for', 'with', 'on', 'at', 'by'}
        words = query.lower().split()
        key_terms = [word for word in words if word not in common_words and len(word) > 2]
        return key_terms
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with related terms and concepts."""
        expanded_queries = [query]
        key_terms = self._extract_key_terms(query)
        
        # Add variations with key terms
        for term in key_terms:
            expanded_queries.extend([
                f"explain {term}",
                f"what is {term}",
                f"how does {term} work",
                f"importance of {term}",
                f"role of {term}"
            ])
        
        return expanded_queries
    
    def _merge_similar_documents(self, documents: List[Document]) -> List[Document]:
        """Merge documents that are part of the same context."""
        merged_docs = []
        current_doc = None
        
        for doc in sorted(documents, key=lambda x: x.metadata.get("source", "")):
            if current_doc is None:
                current_doc = doc
            elif (doc.metadata.get("source") == current_doc.metadata.get("source") and 
                  abs(doc.metadata.get("page", 0) - current_doc.metadata.get("page", 0)) <= 2):  # Increased page range
                # Merge if from same source and consecutive pages
                current_doc.page_content += "\n" + doc.page_content
            else:
                merged_docs.append(current_doc)
                current_doc = doc
        
        if current_doc:
            merged_docs.append(current_doc)
        
        return merged_docs
    
    def _remove_duplicates(self, documents: List[Document]) -> List[Document]:
        """Remove duplicate documents based on content similarity."""
        unique_docs = []
        seen_contents = set()
        
        for doc in documents:
            cleaned_content = self._clean_document(doc.page_content)
            if cleaned_content not in seen_contents:
                seen_contents.add(cleaned_content)
                unique_docs.append(doc)
        
        return unique_docs
    
    def _calculate_relevance_score(self, query: str, document: Document) -> float:
        """Calculate relevance score between query and document."""
        # Use embed_query for the query
        query_embedding = self.embeddings.embed_query(query)
        # Use embed_documents for the document
        doc_embedding = self.embeddings.embed_documents([document.page_content])[0]
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        # Boost score if key terms are present
        key_terms = self._extract_key_terms(query)
        term_boost = 0.1 * sum(1 for term in key_terms if term.lower() in document.page_content.lower())
        
        return float(similarity + term_boost)
    
    def retrieve(self, query: str) -> List[Document]:
        """Retrieve relevant documents for a query."""
        print(f"Retrieving documents for query: {query}")
        
        # Expand query
        expanded_queries = self._expand_query(query)
        
        # Get documents for each expanded query
        all_docs = []
        for expanded_query in expanded_queries:
            docs = self.vector_store.similarity_search(expanded_query, k=self.top_k * 3)  # Increased initial retrieval
            all_docs.extend(docs)
        
        # Remove duplicates
        unique_docs = self._remove_duplicates(all_docs)
        
        # Clean documents
        cleaned_docs = []
        for doc in unique_docs:
            cleaned_content = self._clean_document(doc.page_content)
            cleaned_docs.append(Document(page_content=cleaned_content, metadata=doc.metadata))
        
        # Merge similar documents
        merged_docs = self._merge_similar_documents(cleaned_docs)
        
        # Split merged documents into semantic chunks
        final_docs = []
        for doc in merged_docs:
            chunks = self.text_splitter.split_documents([doc])
            final_docs.extend(chunks)
        
        # Calculate relevance scores and filter
        scored_docs = []
        for doc in final_docs:
            score = self._calculate_relevance_score(query, doc)
            if score > 0.6:  # Lowered threshold for better recall
                scored_docs.append((doc, score))
        
        # Sort by relevance score and take top k
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        final_docs = [doc for doc, _ in scored_docs[:self.top_k]]
        
        print(f"Retrieved {len(final_docs)} documents after filtering.")
        return final_docs
    
    def retrieve_with_scores(self, query: str) -> List[tuple[Document, float]]:
        """Retrieve relevant documents with similarity scores."""
        print(f"Retrieving documents with scores for query: {query}")
        documents = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        print(f"Retrieved {len(documents)} documents with scores.")
        return documents
