#!/usr/bin/env python
"""
Runner script for Mixture of Experts (MoE) in the ML monitoring system.

This script initializes and runs the Mixture of Experts for ML model monitoring.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any, Optional

from moe import MixtureOfExperts, Expert, create_default_experts

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_moe(router_model: str = "gpt-3.5-turbo", custom_experts: List[Expert] = None) -> MixtureOfExperts:
    """
    Set up the Mixture of Experts.
    
    Args:
        router_model: Model to use for routing
        custom_experts: Optional list of custom experts to add
        
    Returns:
        Configured Mixture of Experts
    """
    # Initialize the MoE
    moe = MixtureOfExperts(router_model_name=router_model)
    
    # Add default experts
    for expert in create_default_experts():
        moe.add_expert(expert)
    
    # Add custom experts if provided
    if custom_experts:
        for expert in custom_experts:
            moe.add_expert(expert)
    
    # Initialize the MoE
    moe.initialize()
    
    logger.info(f"Mixture of Experts set up with router model {router_model} and {len(moe.experts)} experts")
    return moe


def process_query(moe: MixtureOfExperts, query: str, context: Optional[str] = None) -> Dict[str, Any]:
    """
    Process a query using the Mixture of Experts.
    
    Args:
        moe: Mixture of Experts
        query: Query to process
        context: Optional context to provide
        
    Returns:
        Dictionary with the response and metadata
    """
    logger.info(f"Processing query: {query}")
    result = moe.query(query, context)
    logger.info(f"Query processed by expert: {result['expert']}")
    return result


def process_batch_queries(moe: MixtureOfExperts, queries: List[str], contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Process multiple queries in batch.
    
    Args:
        moe: Mixture of Experts
        queries: List of queries to process
        contexts: Optional list of contexts to provide
        
    Returns:
        List of dictionaries with responses and metadata
    """
    logger.info(f"Processing batch of {len(queries)} queries")
    results = moe.batch_query(queries, contexts)
    
    # Log which expert handled each query
    for i, result in enumerate(results):
        logger.info(f"Query {i+1} processed by expert: {result['expert']}")
    
    return results


def load_queries_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load queries from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of dictionaries with queries and optional contexts
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.error(f"Invalid format in {file_path}. Expected a list of queries.")
            return []
        
        return data
    except Exception as e:
        logger.error(f"Error loading queries from {file_path}: {str(e)}")
        return []


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run Mixture of Experts for ML model monitoring")
    parser.add_argument("--query", type=str, help="Query to process")
    parser.add_argument("--context", type=str, help="Optional context for the query")
    parser.add_argument("--batch-file", type=str, help="Path to a JSON file containing batch queries")
    parser.add_argument("--router-model", type=str, default="gpt-3.5-turbo", help="Model to use for routing")
    parser.add_argument("--output-file", type=str, help="Path to save the output to")
    args = parser.parse_args()
    
    # Check if API key is available
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.error("OPENAI_API_KEY not found in environment variables. Please set it before running this script.")
        sys.exit(1)
    
    # Set up the MoE
    moe = setup_moe(router_model=args.router_model)
    
    try:
        if args.batch_file:
            # Process batch queries from file
            queries_data = load_queries_from_file(args.batch_file)
            if not queries_data:
                logger.error("No valid queries found in the batch file.")
                sys.exit(1)
            
            queries = [item["query"] for item in queries_data]
            contexts = [item.get("context") for item in queries_data]
            
            results = process_batch_queries(moe, queries, contexts)
        elif args.query:
            # Process a single query
            result = process_query(moe, args.query, args.context)
            results = [result]
        else:
            logger.error("Either --query or --batch-file must be provided.")
            parser.print_help()
            sys.exit(1)
        
        # Output the results
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {args.output_file}")
        else:
            print(json.dumps(results, indent=2))
    
    finally:
        # Shutdown the MoE
        moe.shutdown()


if __name__ == "__main__":
    main() 