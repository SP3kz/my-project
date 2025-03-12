#!/usr/bin/env python
"""
Runner script for vector store operations in the ML monitoring system.

This script provides a command-line interface for managing the vector store
and knowledge base for ML monitoring.
"""

import os
import sys
import argparse
import logging
import json
from typing import List, Dict, Any, Optional
from pathlib import Path

from vector_store import VectorStoreManager, HaystackKnowledgeBase
from langchain.schema import Document

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def load_documents_from_directory(directory: str, extensions: List[str] = None) -> List[Document]:
    """
    Load documents from a directory.
    
    Args:
        directory: Directory to load documents from
        extensions: List of file extensions to include (default: ['.md', '.txt', '.pdf'])
        
    Returns:
        List of Document objects
    """
    if extensions is None:
        extensions = ['.md', '.txt', '.pdf']
    
    documents = []
    dir_path = Path(directory)
    
    if not dir_path.exists():
        logger.error(f"Directory {directory} does not exist")
        return []
    
    for file_path in dir_path.glob('**/*'):
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                metadata = {
                    'source': str(file_path),
                    'filename': file_path.name,
                    'filetype': file_path.suffix.lower()[1:],
                    'created': file_path.stat().st_ctime,
                    'modified': file_path.stat().st_mtime
                }
                
                documents.append(Document(page_content=content, metadata=metadata))
                logger.debug(f"Loaded document: {file_path}")
            except Exception as e:
                logger.warning(f"Error loading document {file_path}: {str(e)}")
    
    logger.info(f"Loaded {len(documents)} documents from {directory}")
    return documents


def add_documents_to_knowledge_base(knowledge_base: HaystackKnowledgeBase, documents: List[Document]) -> List[str]:
    """
    Add documents to the knowledge base.
    
    Args:
        knowledge_base: Knowledge base to add documents to
        documents: List of documents to add
        
    Returns:
        List of document IDs
    """
    logger.info(f"Adding {len(documents)} documents to knowledge base")
    return knowledge_base.add_documents(documents)


def query_knowledge_base(knowledge_base: HaystackKnowledgeBase, query: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query the knowledge base.
    
    Args:
        knowledge_base: Knowledge base to query
        query: Query string
        top_k: Number of results to return
        
    Returns:
        Dictionary with query results
    """
    logger.info(f"Querying knowledge base with: {query}")
    return knowledge_base.query(query, top_k=top_k)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Manage vector store for ML monitoring knowledge base")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Add documents command
    add_parser = subparsers.add_parser("add", help="Add documents to the knowledge base")
    add_parser.add_argument("--directory", type=str, required=True, help="Directory containing documents to add")
    add_parser.add_argument("--extensions", type=str, nargs="+", help="File extensions to include")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the knowledge base")
    query_parser.add_argument("--query", type=str, required=True, help="Query string")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    
    # Clear command
    clear_parser = subparsers.add_parser("clear", help="Clear the knowledge base")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize vector store and knowledge base
    vector_store = VectorStoreManager(
        host=os.environ.get("MILVUS_HOST", "localhost"),
        port=os.environ.get("MILVUS_PORT", "19530"),
        collection_name=os.environ.get("MILVUS_COLLECTION", "ml_monitoring_knowledge")
    )
    
    knowledge_base = HaystackKnowledgeBase(vector_store)
    
    try:
        if args.command == "add":
            # Load and add documents
            extensions = args.extensions if args.extensions else ['.md', '.txt', '.pdf']
            documents = load_documents_from_directory(args.directory, extensions)
            if documents:
                doc_ids = add_documents_to_knowledge_base(knowledge_base, documents)
                logger.info(f"Added {len(doc_ids)} documents to knowledge base")
            else:
                logger.warning(f"No documents found in {args.directory} with extensions {extensions}")
        
        elif args.command == "query":
            # Query the knowledge base
            results = query_knowledge_base(knowledge_base, args.query, args.top_k)
            print(json.dumps(results, indent=2))
        
        elif args.command == "clear":
            # Clear the knowledge base
            vector_store.clear_collection()
            logger.info("Knowledge base cleared")
        
        else:
            parser.print_help()
    
    finally:
        # Close the vector store connection
        vector_store.close()


if __name__ == "__main__":
    main() 