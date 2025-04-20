"""Main module for the RAG system."""

import os
import argparse
import sys
import yaml
from pathlib import Path
from dotenv import load_dotenv

def main():
    """Main entry point for the RAG system."""
    # Load environment variables from .env file
    load_dotenv()
    
    # Check if OPENAI_API_KEY is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please create a .env file with your OpenAI API key or set it in your environment.")
        print("Example: OPENAI_API_KEY=your-api-key-here")
        sys.exit(1)
    
    parser = argparse.ArgumentParser(description="RAG System")
    parser.add_argument("--ingest", action="store_true", help="Ingest documents into the vector store")
    parser.add_argument("--ask", action="store_true", help="Ask questions to the RAG system")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the RAG system")
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        parser.print_help()
        return
    
    # Execute based on arguments
    if args.ingest:
        os.system("python scripts/ingest_documents.py")
    
    if args.ask:
        os.system("python scripts/ask_questions.py")
    
    if args.evaluate:
        os.system("python scripts/run_evaluation.py")

if __name__ == "__main__":
    main()
