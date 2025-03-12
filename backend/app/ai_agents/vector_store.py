"""
Vector Database Integration for ML Monitoring System

This module provides integration with Milvus vector database for storing and retrieving
embeddings related to model performance, monitoring, and knowledge.
"""

import os
import numpy as np
from typing import List, Dict, Any, Optional, Union
from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Milvus
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


class VectorStoreManager:
    """
    Manager for vector database operations using Milvus.
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: str = "19530",
        collection_name: str = "ml_monitoring_knowledge",
        embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    ):
        """
        Initialize the vector store manager.
        
        Args:
            host: Milvus host
            port: Milvus port
            collection_name: Name of the collection to use
            embedding_model: Name of the embedding model to use
        """
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model
        self.embedding_model = HuggingFaceEmbeddings(model_name=embedding_model)
        self._connect()
        self._ensure_collection_exists()
    
    def _connect(self):
        """Connect to Milvus server."""
        connections.connect(
            alias="default",
            host=self.host,
            port=self.port
        )
        print(f"Connected to Milvus server at {self.host}:{self.port}")
    
    def _ensure_collection_exists(self):
        """Ensure that the collection exists, create it if it doesn't."""
        if not utility.has_collection(self.collection_name):
            print(f"Creating collection {self.collection_name}")
            
            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=768)  # Dimension for the embedding model
            ]
            
            schema = CollectionSchema(fields=fields, description=f"ML Monitoring Knowledge Collection")
            self.collection = Collection(name=self.collection_name, schema=schema)
            
            # Create an IVF_FLAT index for the embeddings
            index_params = {
                "metric_type": "L2",
                "index_type": "IVF_FLAT",
                "params": {"nlist": 1024}
            }
            self.collection.create_index(field_name="embedding", index_params=index_params)
            self.collection.load()
        else:
            self.collection = Collection(name=self.collection_name)
            self.collection.load()
    
    def add_documents(self, documents: List[Union[str, Document]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the vector store.
        
        Args:
            documents: List of documents or strings to add
            metadatas: Optional list of metadata dictionaries
        
        Returns:
            List of IDs of the added documents
        """
        # Convert strings to Document objects if needed
        if documents and isinstance(documents[0], str):
            if metadatas:
                docs = [Document(page_content=doc, metadata=meta) for doc, meta in zip(documents, metadatas)]
            else:
                docs = [Document(page_content=doc) for doc in documents]
        else:
            docs = documents
        
        # Split documents into chunks if they're too large
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(docs)
        
        # Create Milvus vector store
        vector_store = Milvus.from_documents(
            documents=split_docs,
            embedding=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": self.host, "port": self.port}
        )
        
        return [str(i) for i in range(len(split_docs))]  # Placeholder IDs
    
    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """
        Perform similarity search for a query.
        
        Args:
            query: Query string
            k: Number of results to return
        
        Returns:
            List of Document objects
        """
        vector_store = Milvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={"host": self.host, "port": self.port}
        )
        
        return vector_store.similarity_search(query, k=k)
    
    def delete_documents(self, ids: List[str]):
        """
        Delete documents from the vector store.
        
        Args:
            ids: List of document IDs to delete
        """
        # Convert string IDs to integers
        int_ids = [int(id_str) for id_str in ids]
        self.collection.delete(f"id in {int_ids}")
    
    def clear_collection(self):
        """Clear all documents from the collection."""
        if utility.has_collection(self.collection_name):
            utility.drop_collection(self.collection_name)
            self._ensure_collection_exists()
    
    def close(self):
        """Close the connection to Milvus."""
        connections.disconnect("default")
        print("Disconnected from Milvus server")


class HaystackKnowledgeBase:
    """
    Knowledge base using Haystack for document retrieval and question answering.
    """
    
    def __init__(
        self,
        vector_store_manager: VectorStoreManager,
        model_name: str = "deepset/roberta-base-squad2"
    ):
        """
        Initialize the knowledge base.
        
        Args:
            vector_store_manager: Vector store manager
            model_name: Name of the question answering model
        """
        self.vector_store_manager = vector_store_manager
        self.model_name = model_name
        # In a real implementation, we would initialize Haystack components here
    
    def add_documents(self, documents: List[Union[str, Document]], metadatas: Optional[List[Dict[str, Any]]] = None):
        """
        Add documents to the knowledge base.
        
        Args:
            documents: List of documents or strings to add
            metadatas: Optional list of metadata dictionaries
        
        Returns:
            List of IDs of the added documents
        """
        return self.vector_store_manager.add_documents(documents, metadatas)
    
    def query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base.
        
        Args:
            query: Query string
            top_k: Number of results to return
        
        Returns:
            Dictionary with query results
        """
        # In a real implementation, this would use Haystack's retriever and reader
        # For now, we'll just use the vector store's similarity search
        documents = self.vector_store_manager.similarity_search(query, k=top_k)
        
        return {
            "query": query,
            "results": [
                {
                    "content": doc.page_content,
                    "score": 0.9 - i * 0.05,  # Mock scores
                    "source": doc.metadata.get("source", "unknown")
                }
                for i, doc in enumerate(documents)
            ],
            "status": "success"
        } 