"""
Agent Orchestrator for ML Performance Monitoring

This module orchestrates multiple AI agents using CrewAI and Ray for distributed computing.
"""

import os
import ray
from typing import List, Dict, Any, Optional
from crewai import Crew, Agent, Task, Process
from langchain.llms import VLLMOpenAI
from langchain.tools import BaseTool

class AgentOrchestrator:
    """
    Orchestrates multiple AI agents for ML model monitoring and optimization.
    Uses CrewAI for agent collaboration and Ray for distributed computing.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the agent orchestrator.
        
        Args:
            api_key: Optional API key for LLM services
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._initialize_ray()
        self.llm = self._initialize_llm()
        self.agents = {}
        self.tools = {}
        self.crew = None
    
    def _initialize_ray(self):
        """Initialize Ray for distributed computing if not already initialized."""
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            print("Ray initialized for distributed computing")
    
    def _initialize_llm(self):
        """Initialize the LLM using vLLM for efficient inference."""
        return VLLMOpenAI(
            openai_api_key=self.api_key,
            model_name="gpt-3.5-turbo",
            temperature=0.2,
            max_tokens=2000
        )
    
    def add_agent(self, agent_id: str, role: str, goal: str, backstory: str, tools: List[BaseTool] = None):
        """
        Add an agent to the orchestrator.
        
        Args:
            agent_id: Unique identifier for the agent
            role: The role of the agent
            goal: The goal the agent is trying to achieve
            backstory: The backstory of the agent
            tools: Optional list of tools the agent can use
        """
        agent = Agent(
            role=role,
            goal=goal,
            backstory=backstory,
            verbose=True,
            llm=self.llm,
            tools=tools or []
        )
        self.agents[agent_id] = agent
        return agent
    
    def add_tool(self, tool_id: str, tool: BaseTool):
        """
        Add a tool that can be used by agents.
        
        Args:
            tool_id: Unique identifier for the tool
            tool: The tool to add
        """
        self.tools[tool_id] = tool
        return tool
    
    def create_task(self, description: str, agent_id: str, expected_output: str = None):
        """
        Create a task for an agent.
        
        Args:
            description: Description of the task
            agent_id: ID of the agent to assign the task to
            expected_output: Expected output format
        
        Returns:
            The created task
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} not found")
        
        agent = self.agents[agent_id]
        task = Task(
            description=description,
            agent=agent,
            expected_output=expected_output
        )
        return task
    
    def create_crew(self, tasks: List[Task], process: Process = Process.sequential):
        """
        Create a crew with the specified tasks and process.
        
        Args:
            tasks: List of tasks for the crew
            process: Process to use (sequential or hierarchical)
        
        Returns:
            The created crew
        """
        self.crew = Crew(
            agents=list(self.agents.values()),
            tasks=tasks,
            verbose=2,
            process=process
        )
        return self.crew
    
    def run_crew(self):
        """
        Run the crew to execute all tasks.
        
        Returns:
            The result of the crew's execution
        """
        if not self.crew:
            raise ValueError("Crew not created yet. Call create_crew first.")
        
        return self.crew.kickoff()
    
    @ray.remote
    def _run_agent_task(self, agent_id: str, task_description: str):
        """
        Run a task for an agent in a distributed manner using Ray.
        
        Args:
            agent_id: ID of the agent to run the task for
            task_description: Description of the task
        
        Returns:
            The result of the task
        """
        if agent_id not in self.agents:
            raise ValueError(f"Agent with ID {agent_id} not found")
        
        agent = self.agents[agent_id]
        task = Task(description=task_description, agent=agent)
        return task.execute()
    
    def run_parallel_tasks(self, tasks_dict: Dict[str, str]):
        """
        Run multiple tasks in parallel using Ray.
        
        Args:
            tasks_dict: Dictionary mapping agent IDs to task descriptions
        
        Returns:
            Dictionary mapping agent IDs to task results
        """
        futures = {
            agent_id: self._run_agent_task.remote(self, agent_id, task_description)
            for agent_id, task_description in tasks_dict.items()
        }
        
        results = {}
        for agent_id, future in futures.items():
            results[agent_id] = ray.get(future)
        
        return results
    
    def shutdown(self):
        """Shutdown Ray and clean up resources."""
        if ray.is_initialized():
            ray.shutdown()
            print("Ray shutdown complete") 