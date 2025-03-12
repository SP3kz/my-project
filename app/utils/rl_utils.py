import numpy as np
import torch
import gym
from typing import List, Tuple, Dict, Any, Union


class ReplayBuffer:
    """
    A simple replay buffer for reinforcement learning.
    """
    def __init__(self, capacity: int):
        """
        Initialize a replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool) -> None:
        """
        Store a transition in the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode is done
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a batch of transitions from the buffer.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*[self.buffer[i] for i in batch])
        
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)


def create_gym_env(env_name: str) -> gym.Env:
    """
    Create a Gym environment.
    
    Args:
        env_name: Name of the Gym environment
        
    Returns:
        Gym environment
    """
    return gym.make(env_name)


def compute_returns(rewards: List[float], gamma: float = 0.99) -> List[float]:
    """
    Compute discounted returns.
    
    Args:
        rewards: List of rewards
        gamma: Discount factor
        
    Returns:
        List of discounted returns
    """
    returns = []
    R = 0
    
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
        
    return returns 