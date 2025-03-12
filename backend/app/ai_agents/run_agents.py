#!/usr/bin/env python
"""
Runner script for AI agents in the ML monitoring system.

This script initializes and runs the AI agents for monitoring ML model performance.
"""

import os
import sys
import argparse
import logging
from typing import List, Dict, Any

from agent_orchestrator import AgentOrchestrator
from monitoring_agents import (
    ModelPerformanceAgent,
    DataDriftAgent,
    ModelOptimizationAgent,
    AlertingAgent
)
from agent_tools import (
    GetModelMetricsTool,
    AnalyzeDataDriftTool,
    GenerateEvidentlyReportTool,
    OptimizeModelTool,
    SendAlertTool,
    QueryKnowledgeBaseTool
)
from crewai import Task, Process

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def setup_orchestrator() -> AgentOrchestrator:
    """
    Set up the agent orchestrator with all necessary agents and tools.
    
    Returns:
        Configured agent orchestrator
    """
    # Initialize the orchestrator
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not found in environment variables. Some functionality may be limited.")
    
    orchestrator = AgentOrchestrator(api_key=api_key)
    
    # Add tools to the orchestrator
    orchestrator.add_tool("model_metrics", GetModelMetricsTool())
    orchestrator.add_tool("data_drift", AnalyzeDataDriftTool())
    orchestrator.add_tool("evidently_report", GenerateEvidentlyReportTool())
    orchestrator.add_tool("optimize_model", OptimizeModelTool())
    orchestrator.add_tool("send_alert", SendAlertTool())
    orchestrator.add_tool("query_knowledge", QueryKnowledgeBaseTool())
    
    # Initialize agents
    performance_agent = ModelPerformanceAgent(orchestrator)
    drift_agent = DataDriftAgent(orchestrator)
    optimization_agent = ModelOptimizationAgent(orchestrator)
    alerting_agent = AlertingAgent(orchestrator)
    
    logger.info("Agent orchestrator set up with all agents and tools")
    return orchestrator


def create_monitoring_tasks(orchestrator: AgentOrchestrator, model_id: str) -> List[Task]:
    """
    Create tasks for monitoring a specific model.
    
    Args:
        orchestrator: The agent orchestrator
        model_id: ID of the model to monitor
        
    Returns:
        List of tasks for the crew
    """
    # Create tasks for each agent
    performance_task = orchestrator.create_task(
        description=f"Analyze the performance metrics of model {model_id} and identify any degradation or anomalies.",
        agent_id="model_performance_agent",
        expected_output="A detailed analysis of model performance with identified issues and recommendations."
    )
    
    drift_task = orchestrator.create_task(
        description=f"Detect and analyze any data drift in the input features for model {model_id}.",
        agent_id="data_drift_agent",
        expected_output="A report on detected data drift, affected features, and potential impact on model performance."
    )
    
    optimization_task = orchestrator.create_task(
        description=f"Based on the performance analysis and data drift report, recommend optimizations for model {model_id}.",
        agent_id="model_optimization_agent",
        expected_output="A list of recommended optimizations with expected improvements and implementation steps."
    )
    
    alerting_task = orchestrator.create_task(
        description=f"Generate appropriate alerts based on the findings from all analyses for model {model_id}.",
        agent_id="alerting_agent",
        expected_output="A set of prioritized alerts with severity levels and recommended actions."
    )
    
    return [performance_task, drift_task, optimization_task, alerting_task]


def run_parallel_monitoring(orchestrator: AgentOrchestrator, model_ids: List[str]):
    """
    Run monitoring tasks for multiple models in parallel.
    
    Args:
        orchestrator: The agent orchestrator
        model_ids: List of model IDs to monitor
    """
    tasks_dict = {}
    for model_id in model_ids:
        tasks_dict[f"performance_{model_id}"] = f"Analyze the performance metrics of model {model_id} and identify any degradation or anomalies."
        tasks_dict[f"drift_{model_id}"] = f"Detect and analyze any data drift in the input features for model {model_id}."
    
    logger.info(f"Running parallel monitoring tasks for models: {', '.join(model_ids)}")
    results = orchestrator.run_parallel_tasks(tasks_dict)
    
    for task_id, result in results.items():
        logger.info(f"Result for task {task_id}: {result[:100]}...")  # Log first 100 chars of each result


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run AI agents for ML model monitoring")
    parser.add_argument("--model-id", type=str, help="ID of the model to monitor")
    parser.add_argument("--parallel", action="store_true", help="Run monitoring in parallel for multiple models")
    parser.add_argument("--process", type=str, choices=["sequential", "hierarchical"], default="sequential",
                        help="Process type for CrewAI (sequential or hierarchical)")
    args = parser.parse_args()
    
    # Set up the orchestrator
    orchestrator = setup_orchestrator()
    
    if args.parallel:
        # Example model IDs - in a real scenario, these would come from a database or config
        model_ids = ["model_20_64_2", "model_50_128_3", "model_100_256_4"]
        run_parallel_monitoring(orchestrator, model_ids)
    else:
        # Monitor a single model using CrewAI
        model_id = args.model_id or "model_20_64_2"  # Default model ID if not provided
        tasks = create_monitoring_tasks(orchestrator, model_id)
        
        # Create and run the crew
        process = Process.hierarchical if args.process == "hierarchical" else Process.sequential
        crew = orchestrator.create_crew(tasks=tasks, process=process)
        
        logger.info(f"Running monitoring crew for model {model_id} with {args.process} process")
        result = orchestrator.run_crew()
        
        logger.info(f"Monitoring completed for model {model_id}")
        logger.info(f"Result summary: {result[:500]}...")  # Log first 500 chars of the result
    
    # Shutdown Ray
    orchestrator.shutdown()


if __name__ == "__main__":
    main() 