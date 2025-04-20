"""Script to run evaluation on the RAG system."""

import os
import sys
import yaml
import pandas as pd
import argparse
from pathlib import Path
from typing import List, Dict

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.ingestion.vector_store import VectorStore
from src.retrieval.retriever import Retriever
from src.generation.rag_generator import RAGGenerator
from src.evaluation.ragas_evaluator import RAGASEvaluator

def load_eval_questions(csv_path: str) -> pd.DataFrame:
    """Load evaluation questions from a CSV file."""
    return pd.read_csv(csv_path)

def run_evaluation(csv_path: str, config: Dict, use_online: bool = False, api_token: str = None):
    """Run evaluation on the RAG system."""
    # Load evaluation questions
    try:
        eval_df = load_eval_questions(csv_path)
        print(f"Loaded {len(eval_df)} evaluation questions.")
    except Exception as e:
        print(f"Error loading evaluation questions: {e}")
        print("Creating sample evaluation questions...")
        
        # Create sample evaluation questions if file doesn't exist
        eval_df = pd.DataFrame({
            "question": [
                "What is RAG?",
                "How does vector search work?",
                "What are the benefits of ChromaDB?",
            ],
            "ground_truth": [
                "RAG (Retrieval-Augmented Generation) is an AI framework that enhances LLM outputs by retrieving relevant information from external sources.",
                "Vector search works by converting text into numerical vectors and finding similar vectors in a high-dimensional space.",
                "ChromaDB benefits include easy integration, efficient vector storage, and good performance for small to medium-sized datasets.",
            ]
        })
        
        # Save sample questions for future use
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        eval_df.to_csv(csv_path, index=False)
    
    # Load vector store
    vector_store = VectorStore(config).create_or_load()
    
    # Initialize retriever
    retriever = Retriever(vector_store, config)
    
    # Initialize generator
    generator = RAGGenerator(config)
    
    # Initialize evaluator
    evaluator = RAGASEvaluator(use_online=use_online, api_token=api_token)
    
    # Lists to store evaluation data
    questions = []
    contexts = []
    answers = []
    ground_truths = []
    
    # Process each question
    for _, row in eval_df.iterrows():
        question = row["question"]
        print(f"Processing question: {question}")
        
        # Retrieve relevant documents
        retrieved_docs = retriever.retrieve(question)
        
        # Extract context texts
        context = [doc.page_content for doc in retrieved_docs]
        
        # Generate answer
        answer = generator.generate(question, retrieved_docs)
        print(f"Answer: {answer}")
        
        # Get ground truth if available
        ground_truth = row.get("ground_truth", None)
        
        # Store data
        questions.append(question)
        contexts.append(context)
        answers.append(answer)
        if ground_truth:
            ground_truths.append(ground_truth)
    
    # Prepare evaluation dataset
    eval_dataset = evaluator.prepare_eval_dataset(
        questions=questions,
        contexts=contexts,
        answers=answers,
        ground_truths=ground_truths if ground_truths else None
    )
    
    # Run evaluation
    results = evaluator.evaluate(eval_dataset)
    
    # Print results
    print("\n===== RAGAS Evaluation Results =====")
    
    # Print the results
    for metric, score in results.items():
        if isinstance(score, (int, float)):
            print(f"{metric}: {score:.4f}")
        else:
            print(f"{metric}: {score}")
    
    # Save results
    try:
        results_df = pd.DataFrame([results])
        results_path = os.path.join(project_root, "data", "processed", "evaluation_results.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\nEvaluation results saved to {results_path}")
    except Exception as e:
        print(f"Warning: Could not save evaluation results: {e}")

def main():
    """Main function to run evaluation."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run evaluation on the RAG system")
    parser.add_argument("--online", action="store_true", help="Use online RAGAS evaluation")
    parser.add_argument("--token", type=str, help="RAGAS API token for online evaluation")
    args = parser.parse_args()
    
    # Get token from environment variable if not provided
    api_token = args.token or os.environ.get("RAGAS_API_TOKEN")
    
    # Load configuration
    with open(os.path.join(project_root, "config", "config.yaml"), "r") as f:
        config = yaml.safe_load(f)
    
    # Define path for evaluation questions
    eval_questions_path = os.path.join(project_root, "data", "raw", "eval_questions.csv")
    
    # Run evaluation
    run_evaluation(eval_questions_path, config, use_online=args.online, api_token=api_token)

if __name__ == "__main__":
    main()
