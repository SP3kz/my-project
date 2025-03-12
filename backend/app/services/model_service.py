import os
import json
import logging
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from app.models.neural_network import SimpleNN
from app.services.dataset_service import DatasetService

logger = logging.getLogger(__name__)

class ModelService:
    def __init__(self):
        self.model = None
        self.model_path = os.path.join(os.getcwd(), 'app', 'models', 'saved_models')
        self.model_info_path = os.path.join(self.model_path, 'model_info.json')
        self.dataset_service = DatasetService()
        
        # Create directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)
        
        # Try to load existing model
        self._load_model()
    
    def _load_model(self):
        """Load the model from disk if it exists"""
        try:
            model_file = os.path.join(self.model_path, 'model.pt')
            if os.path.exists(model_file):
                # Load model info
                with open(self.model_info_path, 'r') as f:
                    model_info = json.load(f)
                
                # Create model with the same architecture
                self.model = SimpleNN(
                    input_size=model_info['input_size'],
                    hidden_size=model_info['hidden_size'],
                    output_size=model_info['output_size']
                )
                
                # Load weights
                self.model.load_state_dict(torch.load(model_file))
                self.model.eval()  # Set to evaluation mode
                logger.info("Model loaded successfully")
            else:
                logger.info("No saved model found")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.model = None
    
    def get_model_info(self):
        """Get information about the currently loaded model"""
        if os.path.exists(self.model_info_path):
            try:
                with open(self.model_info_path, 'r') as f:
                    model_info = json.load(f)
                
                # Add loaded status
                model_info['loaded'] = self.model is not None
                return model_info
            except Exception as e:
                logger.error(f"Error reading model info: {str(e)}")
                return {'loaded': False, 'error': str(e)}
        else:
            return {'loaded': False}
    
    def train_model(self, params):
        """Train a new model with the provided parameters"""
        try:
            # Get dataset
            dataset_id = params['dataset_id']
            dataset = self.dataset_service.get_dataset_by_id(dataset_id)
            
            if not dataset:
                raise ValueError(f"Dataset with ID {dataset_id} not found")
            
            # Prepare data
            X = np.array(dataset['features'])
            y = np.array(dataset['labels'])
            
            # Convert to PyTorch tensors
            X_tensor = torch.FloatTensor(X)
            y_tensor = torch.FloatTensor(y)
            
            # Create dataset and dataloader
            tensor_dataset = TensorDataset(X_tensor, y_tensor)
            dataloader = DataLoader(
                tensor_dataset, 
                batch_size=params['batch_size'], 
                shuffle=True
            )
            
            # Create model
            input_size = X.shape[1]
            hidden_size = params['hidden_size']
            output_size = 1 if len(y.shape) == 1 else y.shape[1]
            
            self.model = SimpleNN(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size
            )
            
            # Define loss function and optimizer
            criterion = nn.MSELoss()
            optimizer = optim.Adam(self.model.parameters(), lr=params['learning_rate'])
            
            # Training loop
            epochs = params['epochs']
            losses = []
            
            for epoch in range(epochs):
                running_loss = 0.0
                for inputs, targets in dataloader:
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                
                # Log epoch loss
                epoch_loss = running_loss / len(dataloader)
                losses.append(epoch_loss)
                logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
            
            # Save model
            self._save_model(input_size, hidden_size, output_size)
            
            return {
                'success': True,
                'message': 'Model trained successfully',
                'epochs': epochs,
                'final_loss': losses[-1],
                'loss_history': losses
            }
        
        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise
    
    def _save_model(self, input_size, hidden_size, output_size):
        """Save the model to disk"""
        try:
            # Save model weights
            model_file = os.path.join(self.model_path, 'model.pt')
            torch.save(self.model.state_dict(), model_file)
            
            # Save model info
            model_info = {
                'input_size': input_size,
                'hidden_size': hidden_size,
                'output_size': output_size,
                'created_at': datetime.now().isoformat(),
                'optimized': False,
                'cached': False
            }
            
            with open(self.model_info_path, 'w') as f:
                json.dump(model_info, f)
            
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise
    
    def predict(self, input_data):
        """Make predictions using the trained model"""
        if self.model is None:
            raise FileNotFoundError("Model not loaded")
        
        try:
            # Convert input to tensor
            input_tensor = torch.FloatTensor(input_data)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Convert to list
            prediction = output.numpy().tolist()
            return prediction
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise
    
    def optimize_model(self):
        """Optimize the model for inference"""
        if self.model is None:
            raise FileNotFoundError("Model not loaded")
        
        try:
            # Set model to evaluation mode
            self.model.eval()
            
            # In a real application, you might use TorchScript, ONNX, or other optimization techniques
            # For this example, we'll just update the model info
            
            # Update model info
            with open(self.model_info_path, 'r') as f:
                model_info = json.load(f)
            
            model_info['optimized'] = True
            model_info['optimized_at'] = datetime.now().isoformat()
            
            with open(self.model_info_path, 'w') as f:
                json.dump(model_info, f)
            
            return {
                'success': True,
                'message': 'Model optimized successfully'
            }
        except Exception as e:
            logger.error(f"Error optimizing model: {str(e)}")
            raise 