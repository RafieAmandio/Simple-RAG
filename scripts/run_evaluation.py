"""Script to run evaluation on the RAG system with chat memory integration."""

import os
import sys
import yaml
import argparse
import pandas as pd
from pathlib import Path
from typing import List, Dict
from datetime import datetime

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.rag_generator import RAGGenerator
from src.memory.chat_memory import ChatMemory

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
    # Read questions and expected responses from CSV
    eval_file = os.path.join(project_root, "data", "csv", "Probstok.csv")
    try:
        # Read CSV with explicit encoding and quoting
        df = pd.read_csv(eval_file, encoding='utf-8', quoting=1)  # quoting=1 for QUOTE_ALL
        
        # Verify required columns exist
        required_columns = ['question', 'ground_truth']
        if not all(col in df.columns for col in required_columns):
            print(f"Error: CSV must contain columns: {required_columns}")
            return
            
        # Convert to lists
        sample_queries = df['question'].tolist()
        expected_responses = df['ground_truth'].tolist()
        
        print(f"Successfully loaded {len(sample_queries)} questions from CSV")
        
    except Exception as e:
        print(f"Error reading evaluation file: {e}")
        print("Please ensure the CSV file is properly formatted with 'question' and 'ground_truth' columns")
        return

    # Initialize chat memory
    chat_memory = ChatMemory(config)
    print(f"Initialized chat memory system")

    # Initialize components
    vector_store = VectorStore(config).create_or_load()
    retriever = Retriever(vector_store, config)
    generator = RAGGenerator(config, chat_memory)  # Pass chat memory to generator

    # Initialize evaluator LLM
    evaluator_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model_name=config["llm"]["model_name"],
            temperature=config["llm"]["temperature"]
        )
    )

    # First, generate answers using RAG
    print("\nGenerating answers using RAG...")
    generated_answers = []
    retrieved_contexts = []
    
    for query in sample_queries:
        print(f"\nProcessing question: {query}")
        
        # Retrieve relevant documents
        docs = retriever.retrieve(query)
        retrieved_contexts.append([doc.page_content for doc in docs])
        
        # Generate answer
        # Chat memory will be automatically updated by the generator
        response = generator.generate(query, docs)
        generated_answers.append(response)
        print(f"Generated answer: {response}")

    # Then, create evaluation dataset without ground truth in input
    print("\nPreparing evaluation dataset...")
    eval_dataset = []
    for i in range(len(sample_queries)):
        eval_dataset.append({
            "user_input": sample_queries[i],
            "retrieved_contexts": retrieved_contexts[i],
            "response": generated_answers[i],
            "reference": expected_responses[i]  # Ground truth only used for evaluation
        })
    
    # Create evaluation dataset
    evaluation_dataset = EvaluationDataset.from_list(eval_dataset)
    
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

    # Convert results to dictionary
    metrics_dict = {}
    for metric in [context_precision, context_recall, faithfulness, answer_relevancy]:
        metric_name = metric.__class__.__name__
        try:
            if hasattr(results, metric_name.lower()):
                metrics_dict[metric_name] = getattr(results, metric_name.lower())
        except Exception as e:
            print(f"Error extracting {metric_name}: {e}")


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
    
    # Print message about chat history
    history_count = len(chat_memory.get_history())
    print(f"\nChat history has been updated with {history_count} messages from the evaluation.")
    print(f"You can view the history in the chat memory file.")

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