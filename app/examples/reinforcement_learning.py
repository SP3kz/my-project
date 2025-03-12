import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import gym

# Add the parent directory to the path so we can import from app
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.models import PolicyNetwork
from app.utils import ReplayBuffer, create_gym_env, compute_returns

# Monkey patch numpy.bool8 if it doesn't exist (for compatibility with newer NumPy versions)
if not hasattr(np, 'bool8'):
    np.bool8 = np.bool_


def train_cartpole(episodes=500, gamma=0.99, lr=0.001, render=False):
    """
    Train a policy network on the CartPole environment.
    
    Args:
        episodes: Number of episodes to train for
        gamma: Discount factor
        lr: Learning rate
        render: Whether to render the environment
        
    Returns:
        Trained policy network and episode rewards
    """
    # Create environment
    env = create_gym_env('CartPole-v1')
    
    # Get environment dimensions
    input_size = env.observation_space.shape[0]  # 4 for CartPole
    output_size = env.action_space.n  # 2 for CartPole
    hidden_size = 128
    
    # Create policy network
    policy = PolicyNetwork(input_size, hidden_size, output_size)
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    episode_rewards = []
    
    for episode in range(episodes):
        # Reset environment
        state, _ = env.reset()
        
        # Lists to store episode data
        log_probs = []
        rewards = []
        
        # Episode loop
        done = False
        episode_reward = 0
        
        while not done:
            # Render environment if specified
            if render and episode % 50 == 0:
                env.render()
            
            # Convert state to tensor
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            
            # Get action probabilities
            action_probs = policy(state_tensor)
            
            # Sample action from distribution
            action_distribution = torch.distributions.Categorical(action_probs)
            action = action_distribution.sample()
            
            # Take action in environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated
            
            # Store log probability of action and reward
            log_prob = action_distribution.log_prob(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            
            # Update state
            state = next_state
            episode_reward += reward
        
        # Store episode reward
        episode_rewards.append(episode_reward)
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        returns = torch.FloatTensor(returns)
        
        # Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Compute loss
        policy_loss = []
        for log_prob, R in zip(log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        
        # Print progress
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f'Episode {episode+1}/{episodes}, Average Reward: {avg_reward:.2f}')
    
    # Close environment
    env.close()
    
    return policy, episode_rewards


def plot_rewards(rewards):
    """Plot the episode rewards."""
    plt.figure(figsize=(10, 6))
    plt.plot(rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Episode Rewards')
    
    # Add moving average
    window_size = 10
    moving_avg = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    plt.plot(range(window_size-1, len(rewards)), moving_avg, 'r-', label=f'{window_size}-episode Moving Average')
    
    plt.legend()
    plt.savefig('rl_rewards.png')
    plt.close()


def main():
    # Create directories
    os.makedirs("models", exist_ok=True)
    
    # Train policy on CartPole
    print("Training policy on CartPole...")
    policy, rewards = train_cartpole(episodes=200, gamma=0.99, lr=0.001, render=False)
    
    # Plot rewards
    print("Plotting rewards...")
    plot_rewards(rewards)
    
    # Save policy
    print("Saving policy...")
    torch.save(policy.state_dict(), "models/cartpole_policy.pt")
    print("Policy saved to models/cartpole_policy.pt")
    
    # Test policy
    print("\nTesting policy...")
    env = create_gym_env('CartPole-v1')
    state, _ = env.reset()
    
    done = False
    total_reward = 0
    
    while not done:
        # Convert state to tensor
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        
        # Get action probabilities
        with torch.no_grad():
            action_probs = policy(state_tensor)
        
        # Take best action
        action = torch.argmax(action_probs, dim=1).item()
        
        # Take action in environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Update state and reward
        state = next_state
        total_reward += reward
    
    print(f"Test reward: {total_reward}")
    env.close()


if __name__ == "__main__":
    main() 