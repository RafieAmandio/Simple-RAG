"""RAGAS evaluator module for the RAG system."""

import os
import json
import pandas as pd
import requests
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas import EvaluationDataset
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

class RAGASEvaluator:
    """Evaluator using RAGAS metrics for RAG evaluation."""
    
    def __init__(self, use_online: bool = False, api_token: Optional[str] = None):
        """
        Initialize RAGAS evaluator.
        
        Args:
            use_online: Whether to use the online RAGAS evaluation API
            api_token: RAGAS app API token for online evaluation
        """
        self.use_online = use_online
        self.api_token = api_token or os.environ.get("RAGAS_API_TOKEN")
        
        if self.use_online and not self.api_token:
            print("Warning: Online RAGAS evaluation requested but no API token provided.")
            print("Set the RAGAS_API_TOKEN environment variable or pass the token to the constructor.")
            print("Falling back to local evaluation.")
            self.use_online = False
    
    def prepare_eval_dataset(
        self, 
        questions: List[str], 
        contexts: List[List[str]], 
        answers: List[str],
        ground_truths: List[str] = None
    ) -> EvaluationDataset:
        """
        Prepare evaluation dataset in RAGAS format.
        
        Args:
            questions: List of questions
            contexts: List of contexts (each context is a list of document contents)
            answers: List of generated answers
            ground_truths: Optional list of ground truth answers
            
        Returns:
            Dataset in RAGAS format
        """
        dataset = []
        
        for i in range(len(questions)):
            item = {
                "user_input": questions[i],
                "retrieved_contexts": contexts[i],
                "response": answers[i],
            }
            
            # Add reference if ground truth is available
            if ground_truths and i < len(ground_truths):
                item["reference"] = ground_truths[i]
                
            dataset.append(item)
        
        return EvaluationDataset.from_list(dataset)
    
    def evaluate_online(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        Evaluate using the RAGAS online API.
        
        Args:
            dataset: Dataset in RAGAS format
            
        Returns:
            Dictionary of evaluation results
        """
        print("Running RAGAS online evaluation...")
        
        # In the latest RAGAS, we can just evaluate and upload
        metrics=[
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]
        
        try:
            # Evaluate the dataset
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
            )
            
            print("Uploading results to RAGAS dashboard...")
            # Upload the results to RAGAS dashboard
            result.upload()
            
            # Convert the result to a dictionary safely
            result_dict = {}
            
            # The result object has scores as attributes
            for metric in metrics:
                metric_name = metric.__class__.__name__
                try:
                    # Try to access as attribute first
                    if hasattr(result, metric_name.lower()):
                        metric_value = getattr(result, metric_name.lower())
                        result_dict[metric_name] = float(metric_value) if isinstance(metric_value, (int, float)) else metric_value
                    # Try various other methods to get the scores
                    elif hasattr(result, 'get_scores') and callable(getattr(result, 'get_scores')):
                        scores = result.get_scores()
                        if metric_name in scores:
                            result_dict[metric_name] = scores[metric_name]
                    # Print debugging info
                    else:
                        print(f"Could not extract {metric_name} from results")
                except Exception as e:
                    print(f"Error extracting {metric_name}: {e}")
            
            # If we couldn't get any results, try to extract all attributes
            if not result_dict:
                print("Attempting to extract all attributes from result...")
                for attr_name in dir(result):
                    if not attr_name.startswith('_') and not callable(getattr(result, attr_name)):
                        try:
                            result_dict[attr_name] = getattr(result, attr_name)
                        except Exception:
                            pass
                            
            return result_dict
            
        except Exception as e:
            print(f"Error during online RAGAS evaluation: {e}")
            print("Falling back to local evaluation...")
            return {"error": str(e)}
    
    def evaluate_local(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        Evaluate using local RAGAS metrics.
        
        Args:
            dataset: Dataset in RAGAS format
            
        Returns:
            Dictionary of evaluation results
        """
        # Default metrics
        metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]
        
        print("Running local RAGAS evaluation...")
        try:
            result = evaluate(dataset=dataset, metrics=metrics)
            
            # Even when using local evaluation, upload the results to RAGAS dashboard
            print("Uploading results to RAGAS dashboard...")
            try:
                result.upload()
                print("Results successfully uploaded to RAGAS dashboard")
            except Exception as upload_error:
                print(f"Failed to upload results to RAGAS dashboard: {upload_error}")
            
            # Extract the results into a dictionary safely
            result_dict = {}
            
            # The result object has scores as attributes
            for metric in metrics:
                metric_name = metric.__class__.__name__
                try:
                    # Try to access as attribute first
                    if hasattr(result, metric_name.lower()):
                        metric_value = getattr(result, metric_name.lower())
                        result_dict[metric_name] = float(metric_value) if isinstance(metric_value, (int, float)) else metric_value
                    # Try various other methods to get the scores
                    elif hasattr(result, 'get_scores') and callable(getattr(result, 'get_scores')):
                        scores = result.get_scores()
                        if metric_name in scores:
                            result_dict[metric_name] = scores[metric_name]
                    # Print debugging info
                    else:
                        print(f"Could not extract {metric_name} from results")
                except Exception as e:
                    print(f"Error extracting {metric_name}: {e}")
            
            # If we couldn't get any results, try to print the result object for inspection
            if not result_dict:
                print("Attempting to extract all attributes from result...")
                print(f"Result type: {type(result)}")
                
                for attr_name in dir(result):
                    if not attr_name.startswith('_') and not callable(getattr(result, attr_name)):
                        try:
                            attr_value = getattr(result, attr_name)
                            result_dict[attr_name] = attr_value
                            print(f"Found attribute '{attr_name}': {attr_value}")
                        except Exception:
                            pass
            
            return result_dict
            
        except Exception as e:
            print(f"Error during local RAGAS evaluation: {e}")
            return {"error": str(e)}
    
    def evaluate(self, dataset: EvaluationDataset) -> Dict[str, Any]:
        """
        Evaluate the RAG system using RAGAS metrics.
        
        Args:
            dataset: Dataset in RAGAS format
            
        Returns:
            Dictionary of evaluation results
        """
        if self.use_online:
            return self.evaluate_online(dataset)
        else:
            return self.evaluate_local(dataset)
