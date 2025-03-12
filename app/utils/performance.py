import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List

def optimize_model_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize a PyTorch model for inference.
    
    Args:
        model: The PyTorch model to optimize
        
    Returns:
        Optimized model
    """
    # Set model to evaluation mode
    model.eval()
    
    # Disable gradient calculation
    for param in model.parameters():
        param.requires_grad = False
    
    return model


def batch_predictions(model: torch.nn.Module, 
                     data: Union[np.ndarray, List[List[float]]], 
                     batch_size: int = 32,
                     device: str = 'cpu') -> np.ndarray:
    """
    Make predictions in batches to optimize memory usage.
    
    Args:
        model: The PyTorch model
        data: Input data as numpy array or list of lists
        batch_size: Batch size for predictions
        device: Device to run predictions on ('cpu' or 'cuda')
        
    Returns:
        Predictions as numpy array
    """
    # Convert data to numpy if it's a list
    if isinstance(data, list):
        data = np.array(data, dtype=np.float32)
    
    # Get number of samples
    n_samples = data.shape[0]
    
    # Initialize predictions array
    predictions = []
    
    # Make predictions in batches
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            # Get batch
            batch = data[i:i+batch_size]
            
            # Convert to tensor
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            
            # Make predictions
            batch_predictions = model(batch_tensor)
            
            # Convert to numpy and append to predictions
            predictions.append(batch_predictions.cpu().numpy())
    
    # Concatenate predictions
    return np.vstack(predictions)


def cache_model_weights(model: torch.nn.Module, cache_dir: str = 'models/cache') -> str:
    """
    Cache model weights to disk for faster loading.
    
    Args:
        model: The PyTorch model
        cache_dir: Directory to cache weights in
        
    Returns:
        Path to cached weights
    """
    import os
    import hashlib
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate a hash of the model architecture
    model_str = str(model)
    model_hash = hashlib.md5(model_str.encode()).hexdigest()
    
    # Cache path
    cache_path = os.path.join(cache_dir, f"{model_hash}.pt")
    
    # Save model weights
    torch.save(model.state_dict(), cache_path)
    
    return cache_path


def load_cached_model(model_class: Any, 
                     cache_path: str, 
                     model_args: Dict[str, Any]) -> torch.nn.Module:
    """
    Load a model from cached weights.
    
    Args:
        model_class: The PyTorch model class
        cache_path: Path to cached weights
        model_args: Arguments to pass to model constructor
        
    Returns:
        Loaded model
    """
    # Create model instance
    model = model_class(**model_args)
    
    # Load weights
    model.load_state_dict(torch.load(cache_path))
    
    # Set to evaluation mode
    model.eval()
    
    return model 