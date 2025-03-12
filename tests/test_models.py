import os
import sys
import unittest
import torch
import numpy as np

# Add the parent directory to the path so we can import from app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.models import SimpleNN, PolicyNetwork


class TestSimpleNN(unittest.TestCase):
    """Test cases for the SimpleNN model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 10
        self.hidden_size = 20
        self.output_size = 2
        self.model = SimpleNN(self.input_size, self.hidden_size, self.output_size)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, SimpleNN)
        self.assertEqual(self.model.fc1.in_features, self.input_size)
        self.assertEqual(self.model.fc1.out_features, self.hidden_size)
        self.assertEqual(self.model.fc3.out_features, self.output_size)
        
    def test_forward(self):
        """Test forward pass."""
        batch_size = 5
        x = torch.randn(batch_size, self.input_size)
        output = self.model(x)
        
        self.assertEqual(output.shape, (batch_size, self.output_size))
        
    def test_save_load(self):
        """Test save and load functionality."""
        # Create a temporary file path
        model_path = "temp_model.pt"
        
        # Save the model
        self.model.save(model_path)
        
        # Load the model
        loaded_model = SimpleNN.load(
            model_path, 
            self.input_size, 
            self.hidden_size, 
            self.output_size
        )
        
        # Check that the loaded model has the same parameters
        for param1, param2 in zip(self.model.parameters(), loaded_model.parameters()):
            self.assertTrue(torch.equal(param1, param2))
            
        # Clean up
        if os.path.exists(model_path):
            os.remove(model_path)


class TestPolicyNetwork(unittest.TestCase):
    """Test cases for the PolicyNetwork model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_size = 4
        self.hidden_size = 32
        self.output_size = 2
        self.model = PolicyNetwork(self.input_size, self.hidden_size, self.output_size)
        
    def test_initialization(self):
        """Test model initialization."""
        self.assertIsInstance(self.model, PolicyNetwork)
        self.assertEqual(self.model.fc1.in_features, self.input_size)
        self.assertEqual(self.model.fc1.out_features, self.hidden_size)
        self.assertEqual(self.model.fc3.out_features, self.output_size)
        
    def test_forward(self):
        """Test forward pass."""
        batch_size = 5
        x = torch.randn(batch_size, self.input_size)
        output = self.model(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.output_size))
        
        # Check that output is a valid probability distribution
        self.assertTrue(torch.allclose(output.sum(dim=1), torch.ones(batch_size)))
        self.assertTrue((output >= 0).all())
        self.assertTrue((output <= 1).all())


if __name__ == "__main__":
    unittest.main() 