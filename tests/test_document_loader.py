"""Tests for the document loader module."""

import os
import sys
import unittest
from pathlib import Path
import tempfile
import yaml

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.document_loader import DocumentLoader

class TestDocumentLoader(unittest.TestCase):
    """Test cases for the DocumentLoader class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a sample config
        self.config = {
            "document_processing": {
                "chunk_size": 1000,
                "chunk_overlap": 200,
            }
        }
        
        # Create a temporary directory for test documents
        self.temp_dir = tempfile.TemporaryDirectory()
        
        # Create a sample text file
        with open(os.path.join(self.temp_dir.name, "sample.txt"), "w") as f:
            f.write("This is a sample document for testing.\n" * 10)
    
    def tearDown(self):
        """Clean up test environment."""
        self.temp_dir.cleanup()
    
    def test_load_from_directory(self):
        """Test loading documents from a directory."""
        loader = DocumentLoader(self.config)
        docs = loader.load_from_directory(self.temp_dir.name)
        
        self.assertGreater(len(docs), 0, "Should load at least one document")
    
    def test_split_documents(self):
        """Test splitting documents into chunks."""
        loader = DocumentLoader(self.config)
        docs = loader.load_from_directory(self.temp_dir.name)
        chunks = loader.split_documents(docs)
        
        self.assertGreaterEqual(len(chunks), len(docs), 
                          "Number of chunks should be at least the number of documents")

if __name__ == "__main__":
    unittest.main()
