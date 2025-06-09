"""Document loader module to load and process documents for the RAG system."""

import os
from typing import List, Dict, Any, Optional
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyPDFLoader,
    CSVLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentLoader:
    """Loads and processes documents from various sources."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize with config settings."""
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config["document_processing"]["chunk_size"],
            chunk_overlap=config["document_processing"]["chunk_overlap"],
        )
    
    def load_from_directory(self, directory_path: str, glob_pattern: str = "**/*.*") -> List[Document]:
        """Load documents from a directory with specified pattern."""
        # Define the loader based on file extension
        def get_loader(file_path: str):
            ext = os.path.splitext(file_path)[1].lower()
            if ext == ".txt":
                return TextLoader(file_path)
            elif ext == ".pdf":
                return PyPDFLoader(file_path)
            # Default to text loader for unknown types
            return TextLoader(file_path)
        
        # Load PDFs from raw directory
        pdf_dir = os.path.join(os.path.dirname(directory_path), "raw")
        pdf_loader = DirectoryLoader(
            pdf_dir,
            glob="**/*.pdf",
            loader_cls=lambda path: get_loader(path),
            show_progress=True,
        )
        
        print(f"Loading PDF documents from {pdf_dir}...")
        pdf_documents = pdf_loader.load()
        print(f"Loaded {len(pdf_documents)} PDF documents.")
        
        # Combine all documents
        all_documents = pdf_documents
        print(f"Total documents loaded: {len(all_documents)}")
        
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks."""
        print("Splitting documents into chunks...")
        chunks = self.text_splitter.split_documents(documents)
        print(f"Created {len(chunks)} chunks.")
        return chunks
