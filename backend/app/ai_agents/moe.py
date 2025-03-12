"""
Mixture of Experts (MoE) Implementation for ML Monitoring

This module implements a Mixture of Experts approach using vLLM for efficient
routing of requests to specialized AI models.
"""

import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from vllm import LLM, SamplingParams
from vllm.outputs import RequestOutput


class Expert:
    """
    Represents a single expert in the Mixture of Experts system.
    """
    
    def __init__(
        self,
        name: str,
        model_name: str,
        description: str,
        specialization: List[str],
        temperature: float = 0.2,
        max_tokens: int = 1024
    ):
        """
        Initialize an expert.
        
        Args:
            name: Name of the expert
            model_name: Name of the model to use for this expert
            description: Description of the expert's capabilities
            specialization: List of areas this expert specializes in
            temperature: Temperature for sampling
            max_tokens: Maximum number of tokens to generate
        """
        self.name = name
        self.model_name = model_name
        self.description = description
        self.specialization = specialization
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm = None  # Will be initialized by the MoE
    
    def __str__(self):
        return f"Expert({self.name}, {self.model_name})"
    
    def get_sampling_params(self) -> SamplingParams:
        """Get sampling parameters for this expert."""
        return SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )


class MixtureOfExperts:
    """
    Implements a Mixture of Experts approach using vLLM for efficient routing.
    """
    
    def __init__(self, router_model_name: str = "gpt-3.5-turbo"):
        """
        Initialize the Mixture of Experts.
        
        Args:
            router_model_name: Name of the model to use for routing
        """
        self.router_model_name = router_model_name
        self.experts: Dict[str, Expert] = {}
        self.router_llm = None
        self.expert_llms: Dict[str, LLM] = {}
        self.initialized = False
    
    def add_expert(self, expert: Expert):
        """
        Add an expert to the MoE.
        
        Args:
            expert: Expert to add
        """
        self.experts[expert.name] = expert
    
    def initialize(self):
        """Initialize all LLMs for the router and experts."""
        if self.initialized:
            return
        
        # Initialize router LLM
        self.router_llm = LLM(model=self.router_model_name)
        
        # Group experts by model to avoid loading the same model multiple times
        model_to_experts = {}
        for expert in self.experts.values():
            if expert.model_name not in model_to_experts:
                model_to_experts[expert.model_name] = []
            model_to_experts[expert.model_name].append(expert)
        
        # Initialize LLMs for each unique model
        for model_name, experts in model_to_experts.items():
            llm = LLM(model=model_name)
            for expert in experts:
                expert.llm = llm
                self.expert_llms[expert.name] = llm
        
        self.initialized = True
        print(f"Initialized MoE with router model {self.router_model_name} and {len(self.experts)} experts")
    
    def _route_query(self, query: str) -> str:
        """
        Route a query to the appropriate expert.
        
        Args:
            query: Query to route
            
        Returns:
            Name of the expert to handle the query
        """
        if not self.initialized:
            self.initialize()
        
        # Create a prompt for the router
        expert_descriptions = "\n".join([
            f"{i+1}. {expert.name}: {expert.description} (Specializes in: {', '.join(expert.specialization)})"
            for i, expert in enumerate(self.experts.values())
        ])
        
        prompt = f"""
        You are a routing system for a Mixture of Experts. Your job is to analyze the query and route it to the most appropriate expert.
        
        Available experts:
        {expert_descriptions}
        
        Query: {query}
        
        Based on the query, which expert should handle this request? Respond with just the name of the expert.
        """
        
        # Get routing decision
        sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
        outputs = self.router_llm.generate([prompt], sampling_params)
        expert_name = outputs[0].outputs[0].text.strip()
        
        # Ensure the expert exists
        if expert_name not in self.experts:
            # Find the closest match
            for name in self.experts.keys():
                if name.lower() in expert_name.lower() or expert_name.lower() in name.lower():
                    expert_name = name
                    break
            else:
                # Default to the first expert if no match is found
                expert_name = list(self.experts.keys())[0]
        
        return expert_name
    
    def query(self, query: str, context: Optional[str] = None) -> Dict[str, Any]:
        """
        Query the MoE system.
        
        Args:
            query: Query to process
            context: Optional context to provide
            
        Returns:
            Dictionary with the response and metadata
        """
        if not self.initialized:
            self.initialize()
        
        # Route the query to an expert
        expert_name = self._route_query(query)
        expert = self.experts[expert_name]
        
        # Create a prompt for the expert
        prompt = f"""
        You are {expert.name}, an expert in {', '.join(expert.specialization)}.
        
        {expert.description}
        
        {"Context: " + context if context else ""}
        
        Query: {query}
        
        Please provide a detailed and accurate response based on your expertise.
        """
        
        # Get response from the expert
        sampling_params = expert.get_sampling_params()
        outputs = expert.llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        return {
            "query": query,
            "expert": expert_name,
            "response": response,
            "model": expert.model_name
        }
    
    def batch_query(self, queries: List[str], contexts: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of queries to process
            contexts: Optional list of contexts to provide
            
        Returns:
            List of dictionaries with responses and metadata
        """
        if not self.initialized:
            self.initialize()
        
        if contexts is None:
            contexts = [None] * len(queries)
        
        results = []
        for query, context in zip(queries, contexts):
            results.append(self.query(query, context))
        
        return results
    
    def shutdown(self):
        """Shutdown all LLMs."""
        if self.router_llm:
            del self.router_llm
        
        for llm in set(self.expert_llms.values()):
            del llm
        
        self.expert_llms = {}
        self.initialized = False
        print("Shut down all LLMs")


# Create default experts for ML monitoring
def create_default_experts() -> List[Expert]:
    """
    Create a set of default experts for ML monitoring.
    
    Returns:
        List of default experts
    """
    return [
        Expert(
            name="PerformanceAnalysisExpert",
            model_name="gpt-3.5-turbo",
            description="Specializes in analyzing model performance metrics and identifying issues.",
            specialization=["model performance", "metrics analysis", "performance optimization"]
        ),
        Expert(
            name="DataDriftExpert",
            model_name="gpt-3.5-turbo",
            description="Specializes in detecting and analyzing data drift and distribution shifts.",
            specialization=["data drift", "distribution analysis", "concept shift"]
        ),
        Expert(
            name="ModelOptimizationExpert",
            model_name="gpt-3.5-turbo",
            description="Specializes in recommending optimizations for model architecture and training.",
            specialization=["model architecture", "hyperparameter tuning", "optimization techniques"]
        ),
        Expert(
            name="AlertingExpert",
            model_name="gpt-3.5-turbo",
            description="Specializes in generating appropriate alerts based on monitoring results.",
            specialization=["alert generation", "notification systems", "incident response"]
        ),
        Expert(
            name="ExplanationExpert",
            model_name="gpt-3.5-turbo",
            description="Specializes in explaining model predictions and behavior in human-understandable terms.",
            specialization=["explainable AI", "feature importance", "prediction explanation"]
        )
    ] 