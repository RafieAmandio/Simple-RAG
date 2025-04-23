"""Script to run evaluation on the RAG system."""

import os
import sys
import yaml
import argparse
from pathlib import Path
from typing import List, Dict

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.rag_generator import RAGGenerator

from ragas import evaluate, EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

def run_evaluation(config: Dict, api_token: str):
    """Run evaluation on the RAG system."""
    # Define your questions and expected responses
    sample_queries = [
    "Siapa ketua BPUPKI?",
    "Tanggal berapa ditetapkan sebagai hari kelahiran Pancasila?",
    "Sebutkan tahun dilaksanakannya sayembara rancangan gambar Garuda Pancasila yang pertama.",
    "Ada berapa simbol di dalam perisai Garuda Pancasila?",
    "Sebutkan kepanjangan dari PPKI!",
    "Apa itu piagam Jakarta",
    "Aoa yang dimaksud dengan hak?"
    ]

    expected_responses = [
        "Rajiman Wedyodiningrat.",
        "1 Juni.",
        "Tahun 1950.",
        "Lima.",
        "Panitia Persiapan Kemerdekaan Indonesia.",
        "Piagam Jakarta adalah dokumen yang berisi lima nilai dasar Pancasila.",
        "Hak adalah sesuatu yang harus kita terima"
    ]

    # Initialize components
    vector_store = VectorStore(config).create_or_load()
    retriever = Retriever(vector_store, config)
    generator = RAGGenerator(config)

    # Initialize evaluator LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"]
        )
    )

    # Prepare evaluation dataset
    dataset = []
    
    print("Preparing evaluation dataset...")
    for query, reference in zip(sample_queries, expected_responses):
        print(f"\nProcessing question: {query}")
        
        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(query)
        
        # Generate answer
        response = generator.generate(query, retrieved_docs)
        
        # Append to dataset
        dataset.append({
            "user_input": query,
            "retrieved_contexts": [doc.page_content for doc in retrieved_docs],
            "response": response,
            "reference": reference,
        })
        print(f"Generated answer: {response}")

    # Create evaluation dataset
    evaluation_dataset = EvaluationDataset.from_list(dataset)
    
    print("\nRunning evaluation...")
    # Run evaluation with specified metrics
    results = evaluate(
        dataset=evaluation_dataset,
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ],
    )
    
    # Print results
    print("\n===== RAGAS Evaluation Results =====")
    print(results)

    # Upload results to Ragas dashboard if API token is provided
    if api_token:
        try:
            print("\nUploading results to Ragas dashboard...")
            results.upload()
            print("Results successfully uploaded to Ragas dashboard!")
        except Exception as e:
            print(f"Error uploading results to Ragas dashboard: {e}")
    else:
        print("\nNo API token provided. Skipping upload to Ragas dashboard.")

def main():
    """Main function to run evaluation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run evaluation on the RAG system")
    parser.add_argument("--token", type=str, help="RAGAS API token for dashboard upload")
    args = parser.parse_args()
    
    # Get token from environment variable if not provided via command line
    api_token = args.token or os.environ.get("RAGAS_API_TOKEN")
    
    if not api_token:
        print("Warning: No RAGAS API token provided. Results won't be uploaded to dashboard.")
        print("Set RAGAS_API_TOKEN environment variable or use --token argument to enable dashboard upload.")
    
    # Load configuration
    with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Run evaluation
    run_evaluation(config, api_token)

if __name__ == "__main__":
    main()
