import os
import sys
import unittest
import torch
import numpy as np

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.utils import normalize_data, numpy_to_tensor, tensor_to_numpy, split_data
from app.utils import ReplayBuffer, compute_returns


class TestDataUtils(unittest.TestCase):
    """Test cases for data utility functions."""
    
    def test_normalize_data(self):
        """Test data normalization."""
        # Create test data
        data = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9]
        ], dtype=np.float32)
        
        # Normalize data
        normalized = normalize_data(data)
        
        # Check shape
        self.assertEqual(normalized.shape, data.shape)
        
        # Check mean and std
        self.assertTrue(np.allclose(np.mean(normalized, axis=0), np.zeros(3), atol=1e-6))
        self.assertTrue(np.allclose(np.std(normalized, axis=0), np.ones(3), atol=1e-6))
    
    def test_numpy_to_tensor(self):
        """Test numpy to tensor conversion."""
        # Create test data
        data = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=np.float32)
        
        # Convert to tensor
        tensor = numpy_to_tensor(data)
        
        # Check type and shape
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (2, 3))
        
        # Check values
        self.assertTrue(torch.allclose(tensor, torch.tensor(data, dtype=torch.float32)))
    
    def test_tensor_to_numpy(self):
        """Test tensor to numpy conversion."""
        # Create test data
        tensor = torch.tensor([
            [1, 2, 3],
            [4, 5, 6]
        ], dtype=torch.float32)
        
        # Convert to numpy
        array = tensor_to_numpy(tensor)
        
        # Check type and shape
        self.assertIsInstance(array, np.ndarray)
        self.assertEqual(array.shape, (2, 3))
        
        # Check values
        self.assertTrue(np.allclose(array, tensor.numpy()))
    
    def test_split_data(self):
        """Test data splitting."""
        # Create test data
        data = np.random.randn(100, 5)
        labels = np.random.randn(100, 2)
        
        # Split data
        train_data, val_data, test_data, train_labels, val_labels, test_labels = split_data(
            data, labels, test_size=0.2, val_size=0.1
        )
        
        # Check shapes
        self.assertEqual(train_data.shape[0], 70)  # 70% for training
        self.assertEqual(val_data.shape[0], 10)    # 10% for validation
        self.assertEqual(test_data.shape[0], 20)   # 20% for testing
        
        self.assertEqual(train_labels.shape[0], 70)
        self.assertEqual(val_labels.shape[0], 10)
        self.assertEqual(test_labels.shape[0], 20)


class TestRLUtils(unittest.TestCase):
    """Test cases for reinforcement learning utility functions."""
    
    def test_replay_buffer(self):
        """Test replay buffer."""
        # Create replay buffer
        buffer = ReplayBuffer(capacity=100)
        
        # Add transitions
        for i in range(10):
            state = np.array([i, i+1, i+2, i+3])
            action = i % 3
            reward = float(i)
            next_state = np.array([i+1, i+2, i+3, i+4])
            done = (i == 9)
            
            buffer.push(state, action, reward, next_state, done)
        
        # Check buffer size
        self.assertEqual(len(buffer), 10)
        
        # Sample from buffer
        batch_size = 5
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)
        
        # Check shapes
        self.assertEqual(states.shape, (batch_size, 4))
        self.assertEqual(actions.shape, (batch_size,))
        self.assertEqual(rewards.shape, (batch_size,))
        self.assertEqual(next_states.shape, (batch_size, 4))
        self.assertEqual(dones.shape, (batch_size,))
    
    def test_compute_returns(self):
        """Test return computation."""
        # Create test rewards
        rewards = [1, 2, 3, 4, 5]
        gamma = 0.9
        
        # Compute returns
        returns = compute_returns(rewards, gamma)
        
        # Check length
        self.assertEqual(len(returns), len(rewards))
        
        # Check values (manually calculated)
        expected = [
            1 + 0.9 * (2 + 0.9 * (3 + 0.9 * (4 + 0.9 * 5))),
            2 + 0.9 * (3 + 0.9 * (4 + 0.9 * 5)),
            3 + 0.9 * (4 + 0.9 * 5),
            4 + 0.9 * 5,
            5
        ]
        
        self.assertTrue(np.allclose(returns, expected))


if __name__ == "__main__":
    unittest.main() 